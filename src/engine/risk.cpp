#include "engine/risk.hpp"

#include <cmath>

namespace xrock::engine {

RiskLimits RiskLimits::from(const Config& c) {
  return {.max_order_notional = c.num("risk.max_order_notional", 0),
          .max_position_notional = c.num("risk.max_position_notional", 0),
          .max_leverage = c.num("risk.max_leverage", 0),
          .max_daily_loss = c.num("risk.max_daily_loss", 0),
          .max_orders_per_sec = static_cast<int>(c.num("risk.max_orders_per_sec", 0))};
}

const char* RiskManager::check(const OrderRequest& r, double ref_price, const Portfolio& pf, Timestamp now) {
  if (Timestamp day = now - now % kDay; day != day_) {
    day_ = day;
    day_start_equity_ = pf.equity();
    halted_ = false;
  }
  if (lim_.max_daily_loss > 0 && day_start_equity_ - pf.equity() >= lim_.max_daily_loss) halted_ = true;

  const Position& p = pf.position(r.sym);
  double next = p.qty + sign(r.side) * r.qty;
  bool reduces = std::abs(next) <= std::abs(p.qty) && next * p.qty >= 0;
  if (!reduces) {
    if (halted_) return "daily loss limit reached";
    if (ref_price <= 0) return "no reference price";
    if (lim_.max_order_notional > 0 && r.qty * ref_price > lim_.max_order_notional) return "order notional limit";
    if (lim_.max_position_notional > 0 && std::abs(next) * ref_price > lim_.max_position_notional)
      return "position notional limit";
    double gross = pf.gross_notional() - p.notional() + std::abs(next) * ref_price;
    if (lim_.max_leverage > 0 && gross > lim_.max_leverage * pf.equity()) return "leverage limit";
  }
  if (lim_.max_orders_per_sec > 0) {
    while (!recent_.empty() && recent_.front() <= now - kSecond) recent_.pop_front();
    if (static_cast<int>(recent_.size()) >= lim_.max_orders_per_sec) return "order rate limit";
    recent_.push_back(now);
  }
  return nullptr;
}

}  // namespace xrock::engine
