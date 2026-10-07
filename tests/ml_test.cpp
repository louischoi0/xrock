#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "backtest/backtest.hpp"
#include "exchange/synthetic.hpp"
#include "ml/dataset.hpp"
#include "ml/metrics.hpp"
#include "ml/model.hpp"
#include "ml/model_strategy.hpp"
#include "ml/window.hpp"
#include <simdjson.h>

using namespace xrock;
namespace fs = std::filesystem;

namespace {

Trade trade(Timestamp ts, double px, double qty = 1, Side s = Side::Buy) {
  return {.ts = ts, .sym = 0, .aggressor = s, .price = px, .qty = qty};
}

Book book(Timestamp ts, double bid, double ask, double bq, double aq) {
  Book b{.ts = ts, .sym = 0};
  b.bids[0] = {bid, bq};
  b.asks[0] = {ask, aq};
  return b;
}

const char* kSynthetic = R"(
[market]
symbols = BTCUSDT
streams = trade, book, funding
[synthetic]
start = 2026-01-01
end = 2026-01-02
trades_per_sec = 0.5
seed = 3
[backtest]
initial_equity = 10000
)";

std::vector<MarketEvent> synthetic_events(const Config& cfg, const Instruments& ins) {
  struct Sink : MarketListener {
    std::vector<MarketEvent> ev;
    void on_market(const MarketEvent& e) override { ev.push_back(e); }
  } sink;
  exchange::SyntheticFeed(cfg, ins).generate(sink);
  std::stable_sort(sink.ev.begin(), sink.ev.end(), [](auto& a, auto& b) { return event_ts(a) < event_ts(b); });
  return sink.ev;
}

}  // namespace

TEST(Features, ValuesAndNoLookahead) {
  ml::FeatureSet f({"ret_2s", "flow_10s", "rate_10s", "spread", "imb_1"}, 10 * kSecond);
  f.on_event(book(1 * kSecond, 99, 101, 3, 1));
  f.on_event(trade(2 * kSecond, 101, 2, Side::Buy));
  f.on_event(book(3 * kSecond, 101, 103, 1, 1));
  f.on_event(trade(4 * kSecond, 101, 1, Side::Sell));
  std::vector<float> x(f.size());
  f.compute(5 * kSecond, x);
  EXPECT_NEAR(x[0], std::log(102.0 / 100.0), 1e-6);  // mid at 5s = 102, mid before 3s = 100
  EXPECT_NEAR(x[1], (2.0 - 1.0) / 3.0, 1e-6);
  EXPECT_NEAR(x[2], 0.2, 1e-6);  // 2 trades / 10s
  EXPECT_NEAR(x[3], 2.0 / 102.0 * 1e4, 1e-3);
  EXPECT_NEAR(x[4], 0.0, 1e-6);
  EXPECT_FALSE(f.ready(5 * kSecond));  // warm-up: 10s windows need 10s of history
  EXPECT_TRUE(f.ready(12 * kSecond));
  EXPECT_FALSE(f.ready(15 * kSecond));  // last event older than max_gap
  EXPECT_THROW(ml::FeatureSet({"nope_1s"}, kSecond), std::runtime_error);
}

TEST(Dataset, LabelsAndSamplingConvention) {
  // mid steps by 1 every second (trade at ts = k s, price 100 + k); sample 1s, horizon 2s.
  ml::DatasetSpec spec{.sym = 0, .features = {{"ret_1s"}, kSecond, 5 * kSecond}, .horizons = {2 * kSecond},
                       .start = 0, .end = 100 * kSecond};
  std::vector<ml::Sample> out;
  ml::DatasetBuilder b(spec, [&](const ml::Sample& s) { out.push_back(s); });
  for (int k = 1; k <= 20; ++k) b.on_event(trade(k * kSecond, 100 + k));
  b.finish();
  ASSERT_FALSE(out.empty());
  for (const auto& s : out) {
    double k = double(s.ts / kSecond);
    // At t = k s the event at k s is not yet seen: mid(t) = 100 + k - 1; label uses mid before t + 2s = 100 + k + 1.
    EXPECT_NEAR(s.x[0], std::log((100 + k - 1) / (100 + k - 2)), 1e-6) << s.ts;
    EXPECT_NEAR(s.y[0], std::log((100 + k + 1) / (100 + k - 1)), 1e-6) << s.ts;
  }
  EXPECT_EQ(out.back().ts, 18 * kSecond);  // later labels would read past the last event
}

TEST(Dataset, FeaturesIgnoreTheFuture) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);
  Timestamp cut = event_ts(ev[ev.size() / 2]);
  auto changed = ev;  // everything from `cut` on is different
  for (auto& e : changed)
    if (auto* t = std::get_if<Trade>(&e); t && t->ts >= cut) t->price *= 1.5;
  ml::DatasetSpec spec{.sym = 0, .features = {{"ret_10s", "vol_60s", "flow_30s", "spread", "imb_5"}, kSecond, 30 * kSecond},
                       .horizons = {10 * kSecond}, .start = 0, .end = kNever};
  auto collect = [&](const std::vector<MarketEvent>& events) {
    std::map<Timestamp, std::vector<float>> x;
    ml::DatasetBuilder b(spec, [&](const ml::Sample& s) { x[s.ts] = s.x; });
    for (const auto& e : events) b.on_event(e);
    b.finish();
    return x;
  };
  auto a = collect(ev), b = collect(changed);
  std::size_t compared = 0;
  for (const auto& [ts, x] : a)
    if (ts <= cut && b.count(ts)) {
      EXPECT_EQ(x, b[ts]) << ts;
      ++compared;
    }
  EXPECT_GT(compared, 1000u);
}

TEST(Splits, WalkForwardWithEmbargoAndHoldout) {
  auto s = ml::Splits::walk_forward(0, 10 * kDay, 3 * kDay, kDay, kDay, kHour, false, 8 * kDay, kMinute);
  ASSERT_EQ(s.folds.size(), 4u);
  for (const auto& f : s.folds) {
    EXPECT_EQ(f.test_start - f.train_end, kHour);
    EXPECT_LE(f.test_end, 8 * kDay);
  }
  EXPECT_EQ(s.folds[1].train_start, kDay);
  EXPECT_THROW(ml::Splits::walk_forward(0, 10 * kDay, 3 * kDay, kDay, kDay, kSecond, false, kNever, kMinute),
               std::runtime_error);  // embargo shorter than the label horizon
  auto e = ml::Splits::walk_forward(0, 10 * kDay, 3 * kDay, kDay, kDay, kHour, true, kNever, kMinute);
  EXPECT_EQ(e.folds.back().train_start, 0);
}

TEST(Npy, HeaderRewrittenWithRowCount) {
  auto p = fs::temp_directory_path() / "xrock_npy_test.npy";
  {
    ml::NpyWriter w(p, "<f4", 3);
    float row[3] = {1, 2, 3};
    for (int i = 0; i < 5; ++i) w.write(row);
  }
  std::ifstream f(p, std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(f)), {});
  EXPECT_EQ(data.substr(0, 6), "\x93NUMPY");
  EXPECT_NE(data.find("'shape': (5, 3)"), std::string::npos);
  EXPECT_EQ(data.size(), 128u + 5 * 3 * 4);
  EXPECT_EQ(data[127], '\n');
}

TEST(Model, LinearRoundTripAndContract) {
  auto p = fs::temp_directory_path() / "xrock_linear_test.json";
  ml::ModelInfo info{.type = "linear", .symbol = "BTCUSDT", .features = {{"ret_1s", "spread"}, kSecond, 5 * kSecond},
                     .outputs = {"fwd_10s"}, .path = {}};
  ml::save_linear_model(p, info, {.mean = {0, 1}, .std = {2, 4}, .bias = {0.5f}, .weights = {{1, -2}}});
  auto m = ml::load_model(p);
  EXPECT_EQ(m->info().features.names, info.features.names);
  EXPECT_EQ(m->info().features.sample, kSecond);
  float x[2] = {4, 9}, y[1];
  m->predict(x, y);
  EXPECT_FLOAT_EQ(y[0], 0.5f + 1 * (4.0f / 2) - 2 * (8.0f / 4));
  std::ofstream(p) << R"({"format":"xrock-model/1","type":"onnx","symbol":"BTCUSDT","features":{"names":["spread"],"sample":"1s","max_gap":"5s"},"outputs":["fwd_10s"],"file":"missing.onnx"})";
  EXPECT_THROW(ml::load_model(p), std::runtime_error);  // no such graph file
}

#ifdef XROCK_WITH_ONNX
// tests/data/linear3x2.onnx was built by python/xrock_ml/onnx_export.py:linear_to_onnx with these parameters.
TEST(Onnx, MatchesLinearReference) {
  fs::path dir = XROCK_TEST_DATA;
  auto onnx = ml::load_model(dir / "linear3x2.json");
  ASSERT_EQ(onnx->info().type, "onnx");
  auto ref_path = fs::temp_directory_path() / "xrock_linear3x2_ref.json";
  ml::ModelInfo info = onnx->info();
  info.type = "linear";
  ml::save_linear_model(ref_path, info,
                        {.mean = {0.5f, -1, 2}, .std = {1, 2, 4}, .bias = {0.1f, -0.2f},
                         .weights = {{1, -2, 0.5f}, {0.25f, 0, -1}}});
  auto ref = ml::load_model(ref_path);
  float x[3], a[2], b[2];
  for (int i = 0; i < 1000; ++i) {
    for (int k = 0; k < 3; ++k) x[k] = std::sin(i * 0.37f + k) * (k + 1) * 3;
    onnx->predict(x, a);
    ref->predict(x, b);
    EXPECT_NEAR(a[0], b[0], 1e-5f);
    EXPECT_NEAR(a[1], b[1], 1e-5f);
  }
}

TEST(Onnx, RefusesContractMismatch) {
  fs::path dir = fs::temp_directory_path() / "xrock_onnx_bad";
  fs::create_directories(dir);
  fs::copy_file(fs::path(XROCK_TEST_DATA) / "linear3x2.onnx", dir / "m.onnx", fs::copy_options::overwrite_existing);
  std::ofstream(dir / "m.json") << R"({"format":"xrock-model/1","type":"onnx","symbol":"BTCUSDT",
    "features":{"names":["ret_1s","flow_10s","spread","funding"],"sample":"1s","max_gap":"5s"},
    "outputs":["fwd_10s","fwd_1m"],"file":"m.onnx"})";
  EXPECT_THROW(ml::load_model(dir / "m.json"), std::runtime_error);  // graph takes 3 features, file declares 4
}
#endif

TEST(Window, ConsecutiveOldestFirstAndResetOnGap) {
  ml::SampleWindow w(3, 2, kSecond);
  auto push = [&](Timestamp t, float v) {
    float x[2] = {v, -v};
    w.push(t * kSecond, x);
  };
  push(1, 1);
  push(2, 2);
  EXPECT_FALSE(w.full());
  push(3, 3);
  ASSERT_TRUE(w.full());
  EXPECT_EQ(std::vector<float>(w.data().begin(), w.data().end()), (std::vector<float>{1, -1, 2, -2, 3, -3}));
  for (int t = 4; t <= 10; ++t) push(t, float(t));  // wraps the ring several times
  EXPECT_EQ(std::vector<float>(w.data().begin(), w.data().end()), (std::vector<float>{8, -8, 9, -9, 10, -10}));
  push(12, 12);  // skipped grid step 11
  EXPECT_FALSE(w.full());
  push(13, 13);
  push(14, 14);
  EXPECT_EQ(std::vector<float>(w.data().begin(), w.data().end()), (std::vector<float>{12, -12, 13, -13, 14, -14}));
  w.reset();
  EXPECT_FALSE(w.full());
}

#ifdef XROCK_WITH_ONNX
// tests/data/seq3x2.onnx: input [1, 3, 2] reshaped oldest-first and fed to the same linear map as below.
TEST(Onnx, SequenceInputMatchesFlattenedLinear) {
  auto onnx = ml::load_model(fs::path(XROCK_TEST_DATA) / "seq3x2.json");
  ASSERT_EQ(onnx->info().window, 3u);
  ml::ModelInfo info = onnx->info();
  info.type = "linear";
  auto ref_path = fs::temp_directory_path() / "xrock_seq_ref.json";
  ml::save_linear_model(ref_path, info,
                        {.mean = {0.1f, -0.2f, 0.3f, 0, 0.5f, -1}, .std = {1, 2, 0.5f, 1, 3, 2}, .bias = {0.05f},
                         .weights = {{1, -1, 0.5f, 2, -0.25f, 3}}});
  auto ref = ml::load_model(ref_path);
  float x[6], a[1], b[1];
  for (int i = 0; i < 1000; ++i) {
    for (int k = 0; k < 6; ++k) x[k] = std::cos(i * 0.11f + k * 0.7f) * (k + 1);
    onnx->predict(x, a);
    ref->predict(x, b);
    EXPECT_NEAR(a[0], b[0], 1e-5f);
  }
}
#endif

TEST(Metrics, PerfectAndInverse) {
  std::vector<float> p{1, 2, 3, 4, 5, 6}, y{-3, -2, -1, 1, 2, 3}, inv{3, 2, 1, -1, -2, -3};
  auto m = ml::evaluate_predictions(p, y, 3);
  EXPECT_NEAR(m.rank_ic, 1, 1e-12);
  EXPECT_GT(m.ic, 0.9);
  EXPECT_EQ(m.hit_rate, 0.5);  // all predictions positive, half the labels negative
  EXPECT_LT(m.bucket_mean.front(), m.bucket_mean.back());
  EXPECT_NEAR(ml::evaluate_predictions(p, inv).rank_ic, -1, 1e-12);
}

// The core guarantee: a model running inside the engine sees exactly the features the dataset export produced.
TEST(Parity, StrategyPredictionsEqualOfflinePredictions) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);

  ml::ModelInfo info{.type = "linear", .symbol = "BTCUSDT",
                     .features = {{"ret_5s", "ret_60s", "vol_60s", "flow_30s", "logvol_30s", "rate_30s", "spread",
                                   "imb_3", "funding"},
                                  kSecond, 20 * kSecond},
                     .outputs = {"fwd_10s", "fwd_60s"}, .path = {}};
  auto path = fs::temp_directory_path() / "xrock_parity_model.json";
  ml::save_linear_model(path, info,
                        {.mean = std::vector<float>(9, 0.f), .std = std::vector<float>(9, 1.f), .bias = {0.f, 1.f},
                         .weights = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {-1, 0.5f, 0, 0, 1, 0, 0.25f, -3, 100}}});
  auto model = ml::load_model(path);

  std::map<Timestamp, std::vector<float>> offline;
  ml::DatasetSpec spec{.sym = 0, .features = info.features, .horizons = {kSecond}, .start = 0, .end = kNever};
  ml::DatasetBuilder b(spec, [&](const ml::Sample& s) {
    std::vector<float> y(2);
    model->predict(s.x, y);
    offline[s.ts] = y;
  });
  for (const auto& e : ev) b.on_event(e);
  b.finish();

  ml::register_strategies();
  Config params = Config::parse("model = " + path.string() + "\ntrade = false\n");
  auto strat = make_strategy("model", params, ins);
  data::VectorSource src(ev);
  auto r = backtest::run(cfg, ins, *strat, src);

  ASSERT_GT(r.journal.predictions.size(), 10000u);
  std::size_t matched = 0;
  for (const auto& p : r.journal.predictions) {
    auto it = offline.find(p.ts);
    if (it == offline.end()) continue;  // offline drops samples whose 1s label runs past the data
    EXPECT_EQ(p.values, it->second) << p.ts;
    ++matched;
  }
  EXPECT_GE(matched + 2, offline.size());
  EXPECT_TRUE(r.journal.fills.empty());  // shadow mode
}

// A windowed model in the engine sees exactly the windows built offline from consecutive exported samples.
TEST(Parity, WindowedModelMatchesOfflineWindows) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);
  constexpr std::size_t T = 5, F = 4;
  ml::ModelInfo info{.type = "linear", .symbol = "BTCUSDT",
                     .features = {{"ret_5s", "flow_30s", "spread", "imb_3"}, kSecond, 20 * kSecond},
                     .outputs = {"fwd_10s"}, .path = {}, .window = T};
  std::vector<float> w(T * F);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i)) * (i % F == 2 ? 1e-3f : 1.f);
  auto path = fs::temp_directory_path() / "xrock_window_model.json";
  ml::save_linear_model(path, info, {.mean = std::vector<float>(T * F, 0.f), .std = std::vector<float>(T * F, 1.f),
                                     .bias = {0.f}, .weights = {w}});
  auto model = ml::load_model(path);

  std::map<Timestamp, float> offline;
  ml::SampleWindow seq(T, F, kSecond);
  ml::DatasetSpec spec{.sym = 0, .features = info.features, .horizons = {kSecond}, .start = 0, .end = kNever};
  ml::DatasetBuilder b(spec, [&](const ml::Sample& s) {
    seq.push(s.ts, s.x);
    if (!seq.full()) return;
    float y;
    model->predict(seq.data(), {&y, 1});
    offline[s.ts] = y;
  });
  for (const auto& e : ev) b.on_event(e);
  b.finish();

  ml::register_strategies();
  auto strat = make_strategy("model", Config::parse("model = " + path.string() + "\n"), ins);
  data::VectorSource src(ev);
  auto r = backtest::run(cfg, ins, *strat, src);
  std::map<Timestamp, float> engine;
  for (const auto& p : r.journal.predictions) engine[p.ts] = p.values[0];

  ASSERT_GT(offline.size(), 10000u);
  std::size_t missing = 0;
  for (const auto& [ts, y] : offline) {
    auto it = engine.find(ts);
    if (it == engine.end()) {
      ++missing;
      continue;
    }
    EXPECT_EQ(it->second, y) << ts;
  }
  EXPECT_EQ(missing, 0u);  // every offline window is an engine window
  EXPECT_GE(engine.size(), offline.size());
}

TEST(VolFeatures, GridReturnsOnKnownPath) {
  // Mid alternates by +-1% each minute (trade just after each boundary): every 1m return is +-0.01.
  ml::FeatureSet f({"rv_1m_5", "ewmavol_1m_3", "garch_1m"}, 2 * kMinute);
  for (int k = 0; k < 60; ++k) f.on_event(trade(k * kMinute + kSecond, 100 * std::exp(k % 2 ? 0.01 : 0.0)));
  std::vector<float> x(3);
  f.compute(60 * kMinute, x);
  EXPECT_NEAR(x[0], 0.01, 1e-6);
  EXPECT_NEAR(x[1], 0.01, 1e-6);
  EXPECT_NEAR(x[2], 0.01, 1e-6);  // constant r^2 is its own variance target
  EXPECT_EQ(ml::vol_interval("ewmavol_1m_3"), kMinute);
  EXPECT_EQ(ml::vol_interval("ret_1m"), 0);
  EXPECT_THROW(ml::FeatureSet({"rv_1m"}, kSecond), std::runtime_error);
}

namespace {
// Mid alternates +-0.1% every second -> rv_1s_5 = 0.001; then whatever `extra` adds.
std::vector<ml::Sample> barrier_samples(const std::vector<Trade>& extra, ml::Barrier b) {
  ml::DatasetSpec spec{.sym = 0, .features = {{"rv_1s_5"}, kSecond, 5 * kSecond}, .horizons = {}, .start = 0,
                       .end = kNever, .barriers = {b}, .barrier_vol = "rv_1s_5"};
  std::vector<ml::Sample> out;
  ml::DatasetBuilder builder(spec, [&](const ml::Sample& s) { out.push_back(s); });
  std::vector<Trade> ev;
  for (int k = 0; k < 60; ++k) ev.push_back(trade(k * kSecond + kMilli, 100 * std::exp(k % 2 ? 0.001 : 0.0)));
  ev.insert(ev.end(), extra.begin(), extra.end());
  std::stable_sort(ev.begin(), ev.end(), [](auto& a, auto& b) { return a.ts < b.ts; });
  for (const auto& t : ev) builder.on_event(t);
  builder.finish();
  return out;
}
}  // namespace

TEST(Barriers, TakeProfitLossCutAndTimeExit) {
  ml::Barrier b{2, 1, 5 * kSecond};
  double sigma_h = 0.001 * std::sqrt(5.0), tp = 2 * sigma_h, sl = sigma_h;
  // A +1% jump at 40.5s: samples in (35.5s, 40.5s] see it inside their 5s horizon.
  auto out = barrier_samples({trade(40 * kSecond + 500 * kMilli, 101.5)}, b);
  ASSERT_FALSE(out.empty());
  std::size_t hit = 0, timed = 0;
  for (const auto& s : out) {
    if (s.ts <= 40 * kSecond) {
      ASSERT_NEAR(s.x[0], 0.001, 1e-6);  // later windows contain the jump
    }
    if (s.ts > 35 * kSecond + 500 * kMilli && s.ts <= 40 * kSecond) {
      EXPECT_NEAR(s.y[0], tp, 1e-6) << s.ts;   // long takes profit
      EXPECT_NEAR(s.y[1], -sl, 1e-6) << s.ts;  // short is stopped out
      ++hit;
    } else if (s.ts <= 35 * kSecond) {
      EXPECT_LT(std::abs(s.y[0]), sl);  // +-0.1% wiggle touches neither barrier: time exit
      EXPECT_NEAR(s.y[0], -s.y[1], 1e-7);
      ++timed;
    }
  }
  EXPECT_EQ(hit, 5u);
  EXPECT_GT(timed, 10u);
}

TEST(Barriers, BothTouchedInOneIntervalIsALoss) {
  ml::Barrier b{1, 1, 5 * kSecond};
  // Within one second: +2% then -2%; both sides see their stop first.
  auto out = barrier_samples({trade(40 * kSecond + 300 * kMilli, 102), trade(40 * kSecond + 600 * kMilli, 98)}, b);
  double sl = 0.001 * std::sqrt(5.0);
  std::size_t checked = 0;
  for (const auto& s : out)
    if (s.ts > 36 * kSecond && s.ts <= 40 * kSecond) {
      EXPECT_NEAR(s.y[0], -sl, 1e-6) << s.ts;
      EXPECT_NEAR(s.y[1], -sl, 1e-6) << s.ts;
      ++checked;
    }
  EXPECT_EQ(checked, 4u);
}

namespace {
fs::path constant_model(const std::string& name, std::vector<std::string> outputs, std::vector<float> bias,
                        std::vector<ml::Barrier> barriers) {
  ml::ModelInfo info{.type = "linear", .symbol = "BTCUSDT",
                     .features = {{"ret_10s", "ewmavol_10s_6"}, kSecond, 20 * kSecond}, .outputs = outputs,
                     .path = {}, .window = 1, .barriers = barriers, .barrier_vol = "ewmavol_10s_6"};
  auto path = fs::temp_directory_path() / name;
  ml::save_linear_model(path, info, {.mean = {0, 0}, .std = {1, 1}, .bias = bias,
                                     .weights = std::vector<std::vector<float>>(outputs.size(), {0, 0})});
  return path;
}

// Round trips from a journal: (entry ts, exit ts, side) in fill order.
std::vector<std::tuple<Timestamp, Timestamp, Side>> round_trips(const analytics::Journal& j) {
  std::vector<std::tuple<Timestamp, Timestamp, Side>> out;
  for (std::size_t i = 0; i + 1 < j.fills.size(); i += 2)
    out.emplace_back(j.fills[i].fill.ts, j.fills[i + 1].fill.ts, j.fills[i].fill.side);
  return out;
}
}  // namespace

TEST(BarrierStrategy, GridModeTakesBestConfigAndAlwaysExits) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);
  // Constant expectations: config 1 short is the best and beats min_edge.
  auto path = constant_model("xrock_grid_model.json", {"tb_0_long", "tb_0_short", "tb_1_long", "tb_1_short"},
                             {0.001f, -0.001f, 0.0f, 0.004f}, {{1, 1, 30 * kSecond}, {2, 1, 2 * kMinute}});
  ml::register_strategies();
  auto strat = make_strategy("barrier", Config::parse("model = " + path.string() +
                                                          "\ntrade = true\nqty = 0.01\nmin_edge = 0.002\n"), ins);
  data::VectorSource src(ev);
  auto r = backtest::run(cfg, ins, *strat, src);
  auto trips = round_trips(r.journal);
  ASSERT_GT(trips.size(), 20u);
  for (const auto& [in, out, side] : trips) {
    EXPECT_EQ(side, Side::Sell);                         // short side chosen
    EXPECT_LE(out - in, 2 * kMinute + 2 * kSecond);      // never held past the config's time barrier (+ latency)
  }
  EXPECT_LE(r.journal.fills.size() % 2, 1u);  // every entry has its exit, except one still open when data ends
}

TEST(BarrierStrategy, DirectModeUsesModelParametersAndMinEdge) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);
  auto path = constant_model("xrock_direct_model.json", {"side", "k_tp", "k_sl", "hold"}, {0.5f, 3, 3, 20}, {});
  ml::register_strategies();
  auto run = [&](const std::string& extra) {
    auto strat = make_strategy("barrier", Config::parse("model = " + path.string() + "\nqty = 0.01\n" + extra), ins);
    data::VectorSource src(ev);
    return backtest::run(cfg, ins, *strat, src);
  };
  auto r = run("trade = true\nmin_edge = 0.1\n");
  auto trips = round_trips(r.journal);
  ASSERT_GT(trips.size(), 50u);
  std::size_t timed = 0;
  for (const auto& [in, out, side] : trips) {
    EXPECT_EQ(side, Side::Buy);
    EXPECT_LE(out - in, 20 * kSecond + 2 * kSecond);
    timed += out - in >= 20 * kSecond;
  }
  EXPECT_GT(timed, 0u);  // wide barriers (3 sigma): some trades leave by the time barrier
  EXPECT_TRUE(run("trade = true\nmin_edge = 0.6\n").journal.fills.empty());  // |side| 0.5 does not beat 0.6
  EXPECT_TRUE(run("trade = false\nmin_edge = 0.1\n").journal.fills.empty());  // shadow
}

// meta.json is the Python side's contract; std::format silently ignores surplus arguments, so check every field.
TEST(Dataset, MetaJsonFieldsWithBarriers) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);
  ml::DatasetSpec spec{.sym = 0, .features = {{"ret_10s", "ewmavol_10s_6"}, kSecond, 20 * kSecond},
                       .horizons = {10 * kSecond}, .start = 0, .end = kNever,
                       .barriers = {{1.5, 1, 30 * kSecond}}, .barrier_vol = "ewmavol_10s_6"};
  auto splits = ml::Splits::walk_forward(event_ts(ev.front()), event_ts(ev.back()), 6 * kHour, 2 * kHour, 2 * kHour,
                                         kMinute, false, kNever, spec.max_horizon());
  auto dir = fs::temp_directory_path() / "xrock_meta_test";
  data::VectorSource src(ev);
  auto n = ml::export_dataset(spec, ins, src, splits, dir);
  simdjson::dom::parser parser;
  simdjson::dom::object meta = parser.load((dir / "meta.json").string());
  EXPECT_EQ(std::string_view(meta["format"]), "xrock-dataset/1");
  EXPECT_EQ(std::uint64_t(meta["samples"]), n);
  EXPECT_EQ(simdjson::dom::array(meta["labels"]).size(), 3u);
  EXPECT_EQ(double(meta["barriers"].at(0)["k_tp"]), 1.5);
  EXPECT_EQ(std::string_view(meta["barrier_vol"]), "ewmavol_10s_6");
  EXPECT_EQ(std::string_view(meta["features"]["sample"]), "1s");
  EXPECT_EQ(simdjson::dom::array(meta["folds"]).size(), splits.folds.size());
  EXPECT_TRUE(meta["holdout_start"].is_null());
}

// Every barrier trade must be explained: enter -> open -> exit, linked to its fills by order id, reasons valid.
TEST(BarrierStrategy, DecisionsExplainEveryOrder) {
  Config cfg = Config::parse(kSynthetic);
  Instruments ins = Instruments::from_config(cfg);
  auto ev = synthetic_events(cfg, ins);
  auto path = constant_model("xrock_grid_model2.json", {"tb_0_long", "tb_0_short"}, {0.003f, 0.0f},
                             {{1.5, 1, 45 * kSecond}});
  ml::register_strategies();
  auto strat = make_strategy("barrier", Config::parse("model = " + path.string() +
                                                          "\ntrade = true\nqty = 0.01\nmin_edge = 0.001\n"), ins);
  data::VectorSource src(ev);
  auto r = backtest::run(cfg, ins, *strat, src);
  const auto& d = r.journal.decisions;
  ASSERT_GT(d.size(), 30u);

  std::map<OrderId, const Fill*> fill_of;
  for (const auto& f : r.journal.fills) fill_of[f.fill.id] = &f.fill;
  auto value = [](const Decision& x, const std::string& k) {
    for (const auto& [n, v] : x.values)
      if (n == k) return v;
    return std::nan("");
  };
  std::size_t trips = 0;
  for (std::size_t i = 0; i + 2 < d.size(); ++i) {
    if (d[i].decision.kind != "enter") continue;
    const auto& enter = d[i].decision;
    const auto& open = d[i + 1].decision;
    const auto& exit = d[i + 2].decision;
    ASSERT_EQ(open.kind, "open");
    ASSERT_EQ(exit.kind, "exit");
    EXPECT_EQ(enter.reason, "tb_0_long");
    EXPECT_EQ(value(enter, "edge"), 0.003f);
    EXPECT_EQ(value(enter, "k_tp"), 1.5);
    EXPECT_EQ(open.order, enter.order);
    ASSERT_TRUE(fill_of.count(enter.order));
    EXPECT_EQ(value(open, "entry"), fill_of[enter.order]->price);
    EXPECT_GT(value(open, "tp"), value(open, "entry"));
    EXPECT_LT(value(open, "sl"), value(open, "entry"));
    ASSERT_TRUE(fill_of.count(exit.order));  // the exit decision produced the closing fill
    double mid = value(exit, "mid");
    if (exit.reason == "take-profit") {
      EXPECT_GE(mid, value(open, "tp"));
    } else if (exit.reason == "loss-cut") {
      EXPECT_LE(mid, value(open, "sl"));
    } else {
      EXPECT_EQ(exit.reason, "time");
      EXPECT_GE(value(exit, "held_s"), 45.0);
    }
    ++trips;
  }
  EXPECT_GT(trips, 10u);

  // decisions.csv round trip
  auto dir = fs::temp_directory_path() / "xrock_decisions_test";
  r.journal.write_csv(dir, ins);
  auto back = analytics::Journal::read_csv(dir, ins);
  ASSERT_EQ(back.decisions.size(), d.size());
  for (std::size_t i = 0; i < d.size(); ++i) {
    EXPECT_EQ(back.decisions[i].ts, d[i].ts);
    EXPECT_EQ(back.decisions[i].decision.kind, d[i].decision.kind);
    EXPECT_EQ(back.decisions[i].decision.reason, d[i].decision.reason);
    EXPECT_EQ(back.decisions[i].decision.order, d[i].decision.order);
    EXPECT_EQ(back.decisions[i].decision.values, d[i].decision.values);
  }
  EXPECT_NE(analytics::decision_summary(back).find("enter"), std::string::npos);
}
