#pragma once

#include <deque>

#include "core/config.hpp"
#include "engine/portfolio.hpp"

namespace xrock::engine {

// [risk]; 0 disables a limit.
struct RiskLimits {
  double max_order_notional = 0;
  double max_position_notional = 0;
  double max_leverage = 0;    // gross notional / equity
  double max_daily_loss = 0;  // from the UTC-day start equity; blocks new risk until the next day
  int max_orders_per_sec = 0;
  static RiskLimits from(const Config&);
};

// Pre-trade checks, shared by backtest and live. Orders that only reduce a position pass the
// notional/leverage/daily-loss checks. TODO: count exposure of resting orders.
class RiskManager {
 public:
  explicit RiskManager(RiskLimits lim) : lim_(lim) {}
  // nullptr if allowed, otherwise a static reason.
  const char* check(const OrderRequest&, double ref_price, const Portfolio&, Timestamp now);
  bool halted() const { return halted_; }

 private:
  RiskLimits lim_;
  std::deque<Timestamp> recent_;  // accepted order times within the last second
  Timestamp day_ = -1;
  double day_start_equity_ = 0;
  bool halted_ = false;
};

}  // namespace xrock::engine
