#pragma once

#include <string>

#include "analytics/journal.hpp"

namespace xrock::analytics {

struct Metrics {
  double initial_equity = 0, final_equity = 0;
  double total_return = 0;
  double max_drawdown = 0;  // fraction of peak equity
  double sharpe = 0;        // annualized (365d) from equity samples
  double fees = 0, funding = 0, turnover = 0, realized = 0;
  std::size_t fills = 0, closing_fills = 0, winning_fills = 0;
  double win_rate() const { return closing_fills ? double(winning_fills) / closing_fills : 0; }
};

Metrics compute_metrics(const Journal&);
std::string to_string(const Metrics&);
// "decisions  enter 207 | exit: time 173, loss-cut 20, take-profit 14" - counts by kind and reason; "" if none.
std::string decision_summary(const Journal&);

}  // namespace xrock::analytics
