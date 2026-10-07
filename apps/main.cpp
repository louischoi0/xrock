#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <format>
#include <iostream>
#include <thread>

#include "analytics/metrics.hpp"
#include "backtest/backtest.hpp"
#include "core/log.hpp"
#include "data/backfill.hpp"
#include "data/collector.hpp"
#include "data/kds/kds_store.hpp"
#include "exchange/registry.hpp"
#include "execution/live.hpp"
#include "ml/dataset.hpp"
#include "ml/metrics.hpp"
#include "ml/model.hpp"
#include "ml/model_strategy.hpp"
#include "ml/window.hpp"
#include "research/research.hpp"

using namespace xrock;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

constexpr const char* kUsage = R"(usage: xrock <command> [-c conf/xrock.conf] [args]
  fetch          L1  download history ([fetch] range) into the [data] backend; resumable
  collect        L1  record live [exchange] market data into the [data] backend
  backtest       L2  run [strategy] over the [backtest] range
  research       L3  parameter sweep + baselines ([research])
  live           L4  trade [strategy] on [exchange]
  report <dir>   L5  metrics of a journal directory
  dataset        L3  export features/labels/folds ([dataset], [splits]) as .npy + meta.json
  model-eval <model.json>  L3  prediction quality (IC, rank IC, buckets) per walk-forward test fold + holdout
  db ping|stop|sql "<stmt>"   the [data.kds] server: health, graceful STOP, one statement
)";

std::filesystem::path out_dir(const Config& cfg) { return cfg.str("analytics.out_dir", "results"); }

std::unique_ptr<Strategy> configured_strategy(const Config& cfg, const Instruments& ins) {
  std::string name = cfg.str("strategy.name");
  return make_strategy(name, cfg.section("strategy." + name), ins);
}

int cmd_collect(const Config& cfg) {
  Instruments ins = Instruments::from_config(cfg);
  auto store = data::make_store(cfg, ins);
  auto feed = exchange::make_feed(cfg, ins);
  data::Collector col(*store);
  feed->start(col);
  auto last_flush = std::chrono::steady_clock::now();
  std::uint64_t last_count = 0;
  while (!g_stop && !feed->done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (std::chrono::steady_clock::now() - last_flush < std::chrono::seconds(10)) continue;
    col.flush();  // bound what a crash can lose
    last_flush = std::chrono::steady_clock::now();
    std::uint64_t n = col.count();
    log::info("collect: {} events (+{})", n, n - last_count);
    last_count = n;
  }
  feed->stop();
  col.flush();
  log::info("collect: {} events stored ({})", col.count(), cfg.str("data.backend", "file"));
  return 0;
}

int cmd_fetch(const Config& cfg) {
  Instruments ins = Instruments::from_config(cfg);
  auto store = data::make_store(cfg, ins);
  auto src = exchange::make_history(cfg, ins);
  std::vector<data::Stream> streams;
  for (const auto& s : cfg.list("fetch.streams")) streams.push_back(data::parse_stream(s));
  auto st = data::backfill(*src, *store, ins, streams, cfg.date("fetch.start"), cfg.date("fetch.end"), g_stop);
  log::info("fetch: {} days stored ({} events), {} already present, {} unavailable", st.fetched, st.events,
            st.skipped, st.missing);
  return 0;
}

int cmd_dataset(const Config& cfg) {
  Instruments ins = Instruments::from_config(cfg);
  auto spec = ml::DatasetSpec::from(cfg, ins);
  auto splits = ml::Splits::from(cfg, spec.start, spec.end, spec.max_horizon());
  auto src = backtest::open_replay(cfg, ins, spec.start, spec.end);
  std::filesystem::path out = cfg.str("dataset.out");
  ml::export_dataset(spec, ins, *src, splits, out);
  std::cout << "dataset " << out.string() << " (" << splits.folds.size() << " folds)\n";
  return 0;
}

int cmd_model_eval(const Config& cfg, const std::string& path) {
  Instruments ins = Instruments::from_config(cfg);
  auto model = ml::load_model(path);
  auto spec = ml::DatasetSpec::from(cfg, ins);
  spec.sym = ins.id(model->info().symbol);
  spec.features = model->info().features;  // the model's contract, not [dataset] features
  if (!model->info().barriers.empty()) {  // the model's barrier contract defines its tb_* labels
    spec.barriers = model->info().barriers;
    spec.barrier_vol = model->info().barrier_vol;
  }
  spec.validate();
  const auto& outs = model->info().outputs;
  const auto labels = spec.label_names();
  std::vector<std::size_t> label_of;  // model output -> label column
  for (const auto& o : outs) {
    auto it = std::find(labels.begin(), labels.end(), o);
    if (it == labels.end()) throw std::runtime_error("model-eval: output " + o + " is not a [dataset] label");
    label_of.push_back(static_cast<std::size_t>(it - labels.begin()));
  }
  auto splits = ml::Splits::from(cfg, spec.start, spec.end, spec.max_horizon());
  std::vector<Timestamp> ts;
  std::vector<std::vector<float>> pred(outs.size()), label(outs.size());
  std::vector<float> y(outs.size());
  std::vector<double> latency_us;
  // Windows over exported (label-valid) samples: consecutive rows only, as python Dataset.windows() builds them.
  ml::SampleWindow seq(model->info().window, spec.features.names.size(), spec.features.sample);
  ml::DatasetBuilder b(spec, [&](const ml::Sample& s) {
    seq.push(s.ts, s.x);
    if (!seq.full()) return;
    auto t0 = std::chrono::steady_clock::now();
    model->predict(seq.data(), y);
    latency_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    ts.push_back(s.ts);
    for (std::size_t o = 0; o < outs.size(); ++o) {
      pred[o].push_back(y[o]);
      label[o].push_back(s.y[label_of[o]]);
    }
  });
  auto src = backtest::open_replay(cfg, ins, spec.start, spec.end);
  while (const MarketEvent* e = src->next()) b.on_event(*e);
  b.finish();
  auto window = [&](std::size_t o, Timestamp from, Timestamp to) {
    std::vector<float> p, l;
    for (std::size_t i = 0; i < ts.size(); ++i)
      if (ts[i] >= from && ts[i] < to) {
        p.push_back(pred[o][i]);
        l.push_back(label[o][i]);
      }
    return ml::to_string(ml::evaluate_predictions(p, l));
  };
  if (!latency_us.empty()) {
    std::sort(latency_us.begin(), latency_us.end());
    std::cout << std::format("{} model, {} predictions: latency p50 {:.1f}us p99 {:.1f}us max {:.1f}us\n",
                             model->info().type, latency_us.size(), latency_us[latency_us.size() / 2],
                             latency_us[latency_us.size() * 99 / 100], latency_us.back());
  }
  for (std::size_t o = 0; o < outs.size(); ++o) {
    std::cout << outs[o] << "\n";
    for (std::size_t i = 0; i < splits.folds.size(); ++i)
      std::cout << std::format("  fold {} test {}..{}  {}\n", i, format_date(splits.folds[i].test_start),
                               format_date(splits.folds[i].test_end - 1), window(o, splits.folds[i].test_start,
                                                                               splits.folds[i].test_end));
    if (splits.holdout_start != kNever)
      std::cout << "  holdout  " << window(o, splits.holdout_start, kNever) << "\n";
  }
  return 0;
}

int cmd_backtest(const Config& cfg) {
  Instruments ins = Instruments::from_config(cfg);
  auto strat = configured_strategy(cfg, ins);
  auto src = backtest::open_replay(cfg, ins);
  auto r = backtest::run(cfg, ins, *strat, *src);
  auto dir = out_dir(cfg) / ("backtest-" + cfg.str("strategy.name"));
  r.journal.write_csv(dir, ins);
  std::cout << analytics::to_string(r.metrics) << analytics::decision_summary(r.journal) << "journal   "
            << dir.string() << "\n";
  return 0;
}

int cmd_research(const Config& cfg) {
  Instruments ins = Instruments::from_config(cfg);
  auto trials = research::run(cfg, ins);
  auto path = out_dir(cfg) / "research.csv";
  research::write_csv(trials, path);
  std::cout << research::format_table(trials) << "csv: " << path.string() << "\n";
  return 0;
}

int cmd_live(const Config& cfg) {
  if (cfg.str("exchange.env", "testnet") == "mainnet" && !cfg.flag("live.allow_mainnet", false))
    throw std::runtime_error("refusing mainnet: set [live] allow_mainnet = true");
  Instruments ins = Instruments::from_config(cfg);
  auto strat = configured_strategy(cfg, ins);
  auto feed = exchange::make_feed(cfg, ins);
  auto gw = exchange::make_gateway(cfg, ins);
  execution::LiveRunner runner(cfg, ins, *feed, *gw, *strat);
  runner.run(g_stop);
  return 0;
}

int cmd_report(const Config& cfg, const std::string& dir) {
  Instruments ins = Instruments::from_config(cfg);
  auto j = analytics::Journal::read_csv(dir, ins);
  std::cout << analytics::to_string(analytics::compute_metrics(j)) << analytics::decision_summary(j);
  return 0;
}

int cmd_db(const Config& cfg, const std::vector<std::string>& args) {
  if (args.empty()) throw std::runtime_error("db: expected ping | stop | sql \"<statement>\"");
  auto opt = data::kds_options(cfg);
  data::kwp::Connection conn(opt.host, opt.port, opt.protocol, 5);
  if (args[0] == "ping") {
    conn.execute("PING");
    std::cout << "kds " << opt.host << ":" << opt.port << " ok\n";
    return 0;
  }
  if (args[0] == "stop") {  // flushes pages and exits; the server closes the socket as its answer
    try {
      conn.execute("STOP");
    } catch (const data::kwp::Error&) {
      throw;
    } catch (const std::runtime_error&) {
    }
    std::cout << "kds " << opt.host << ":" << opt.port << " stopping\n";
    return 0;
  }
  if (args[0] == "sql" && args.size() == 2) {
    auto r = conn.execute(args[1]);
    if (r.fields.empty()) {
      std::cout << r.tag << "\n";
      return 0;
    }
    for (std::size_t c = 0; c < r.fields.size(); ++c) std::cout << (c ? "," : "") << r.fields[c].name;
    std::cout << "\n";
    for (std::size_t i = 0; i < r.rows(); ++i) {
      for (std::size_t c = 0; c < r.fields.size(); ++c) {
        const auto& f = r.fields[c];
        std::cout << (c ? "," : "");
        if (!r.at(i, c).data)
          std::cout << "NULL";
        else if (r.text || f.type_oid == data::kwp::kVarchar || f.type_oid == data::kwp::kChar)
          std::cout << data::kwp::Result::text_of(r.at(i, c));
        else if (f.type_oid == data::kwp::kDecimal)
          std::cout << std::format("{:.{}f}", r.decimal(i, c), f.scale());
        else
          std::cout << r.integer(i, c);
      }
      std::cout << "\n";
    }
    return 0;
  }
  throw std::runtime_error("db: expected ping | stop | sql \"<statement>\"");
}

log::Level parse_level(const std::string& s) {
  if (s == "debug") return log::Level::Debug;
  if (s == "warn") return log::Level::Warn;
  if (s == "error") return log::Level::Error;
  return log::Level::Info;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << kUsage;
    return 2;
  }
  std::string cmd = argv[1], conf = "conf/xrock.conf";
  std::vector<std::string> args;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-c" && i + 1 < argc)
      conf = argv[++i];
    else
      args.push_back(a);
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  ml::register_strategies();
  try {
    Config cfg = Config::load(conf);
    log::threshold = parse_level(cfg.str("general.log_level", "info"));
    if (cmd == "fetch") return cmd_fetch(cfg);
    if (cmd == "collect") return cmd_collect(cfg);
    if (cmd == "backtest") return cmd_backtest(cfg);
    if (cmd == "research") return cmd_research(cfg);
    if (cmd == "live") return cmd_live(cfg);
    if (cmd == "report" && args.size() == 1) return cmd_report(cfg, args[0]);
    if (cmd == "db") return cmd_db(cfg, args);
    if (cmd == "dataset") return cmd_dataset(cfg);
    if (cmd == "model-eval" && args.size() == 1) return cmd_model_eval(cfg, args[0]);
  } catch (const std::exception& e) {
    log::error("{}", e.what());
    return 1;
  }
  std::cerr << kUsage;
  return 2;
}
