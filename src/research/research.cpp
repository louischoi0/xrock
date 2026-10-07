#include "research/research.hpp"

#include <algorithm>
#include <atomic>
#include <format>
#include <fstream>
#include <thread>

#include "backtest/backtest.hpp"
#include "core/log.hpp"

namespace xrock::research {
namespace {

struct Job {
  std::string strategy;
  Config params;
  std::string label;
};

std::vector<Job> make_jobs(const Config& cfg) {
  std::string name = cfg.str("research.strategy");
  std::vector<Job> jobs{{name, cfg.section("strategy." + name), ""}};
  const Config grid = cfg.section("research.sweep");
  for (const auto& [key, values] : grid.entries()) {
    std::vector<Job> next;
    for (const Job& j : jobs)
      for (const auto& v : split(values, ',')) {
        Job n = j;
        n.params.set(key, v);
        n.label += (n.label.empty() ? "" : " ") + key + "=" + v;
        next.push_back(std::move(n));
      }
    jobs = std::move(next);
  }
  for (const auto& b : cfg.list("research.baselines")) jobs.push_back({b, cfg.section("strategy." + b), "baseline"});
  return jobs;
}

}  // namespace

std::vector<Trial> run(const Config& cfg, const Instruments& ins) {
  auto src = backtest::open_replay(cfg, ins);
  const std::vector<MarketEvent> events = data::load_all(*src);
  const std::vector<Job> jobs = make_jobs(cfg);
  log::info("research: {} events, {} trials", events.size(), jobs.size());

  std::vector<Trial> trials(jobs.size());
  std::atomic<std::size_t> next{0};
  auto worker = [&] {
    for (std::size_t i; (i = next++) < jobs.size();) {
      Trial& t = trials[i];
      t.strategy = jobs[i].strategy;
      t.label = jobs[i].label;
      try {
        auto strat = make_strategy(jobs[i].strategy, jobs[i].params, ins);
        data::VectorSource vs(events);
        t.metrics = backtest::run(cfg, ins, *strat, vs).metrics;
      } catch (const std::exception& e) {
        t.error = e.what();
      }
    }
  };
  {
    std::size_t n = std::min<std::size_t>(jobs.size(), std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::jthread> pool;
    for (std::size_t k = 0; k < n; ++k) pool.emplace_back(worker);
  }
  std::stable_sort(trials.begin(), trials.end(), [](const Trial& a, const Trial& b) {
    if (a.error.empty() != b.error.empty()) return a.error.empty();
    return a.metrics.sharpe > b.metrics.sharpe;
  });
  return trials;
}

std::string format_table(const std::vector<Trial>& trials) {
  std::string out = std::format("{:<12} {:<24} {:>9} {:>8} {:>7} {:>6} {:>6}\n", "strategy", "params", "return",
                                "maxdd", "sharpe", "fills", "win");
  for (const Trial& t : trials) {
    if (!t.error.empty()) {
      out += std::format("{:<12} {:<24} error: {}\n", t.strategy, t.label, t.error);
      continue;
    }
    const auto& m = t.metrics;
    out += std::format("{:<12} {:<24} {:>+8.2f}% {:>7.2f}% {:>7.2f} {:>6} {:>5.1f}%\n", t.strategy, t.label,
                       100 * m.total_return, 100 * m.max_drawdown, m.sharpe, m.fills, 100 * m.win_rate());
  }
  return out;
}

void write_csv(const std::vector<Trial>& trials, const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream f(path);
  f << "strategy,params,return,max_drawdown,sharpe,fills,win_rate,fees,funding,error\n";
  for (const Trial& t : trials) {
    const auto& m = t.metrics;
    f << std::format("{},{},{},{},{},{},{},{},{},{}\n", t.strategy, t.label, m.total_return, m.max_drawdown, m.sharpe,
                     m.fills, m.win_rate(), m.fees, m.funding, t.error);
  }
}

}  // namespace xrock::research
