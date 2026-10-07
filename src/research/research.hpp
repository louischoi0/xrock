#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "analytics/metrics.hpp"
#include "core/config.hpp"
#include "core/instruments.hpp"

namespace xrock::research {

struct Trial {
  std::string strategy;
  std::string label;  // swept params, e.g. "fast=5 slow=30", or "baseline"
  analytics::Metrics metrics;
  std::string error;
};

// Grid over [research] sweep.* for research.strategy, plus each of research.baselines, on one in-memory copy of
// the [backtest] range, run in parallel. Sorted by Sharpe.
// TODO: walk-forward / out-of-sample split; multiple datasets.
std::vector<Trial> run(const Config&, const Instruments&);
std::string format_table(const std::vector<Trial>&);
void write_csv(const std::vector<Trial>&, const std::filesystem::path&);

}  // namespace xrock::research
