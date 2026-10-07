#include <deque>

#include "strategy/strategy.hpp"

namespace xrock {
namespace {

// Tick-driven short-horizon example: when signed taker volume over `window` reaches `threshold`, enter in that
// direction and flatten after `hold` via a timer.
class TradeFlow final : public Strategy {
 public:
  TradeFlow(const Config& p, const Instruments& ins)
      : sym_(ins.id(p.str("symbol"))),
        window_(p.duration("window", 10 * kSecond)),
        hold_(p.duration("hold", 30 * kSecond)),
        threshold_(p.num("threshold")),
        qty_(p.num("qty")) {}

  void on_trade(Context& c, const Trade& t) override {
    if (t.sym != sym_) return;
    window_trades_.push_back(t);
    flow_ += sign(t.aggressor) * t.qty;
    while (window_trades_.front().ts <= t.ts - window_) {
      flow_ -= sign(window_trades_.front().aggressor) * window_trades_.front().qty;
      window_trades_.pop_front();
    }
    if (active_ || std::abs(flow_) < threshold_) return;
    if (c.submit({.sym = sym_, .side = flow_ > 0 ? Side::Buy : Side::Sell, .qty = qty_}) == 0) return;
    active_ = true;
    c.set_timer(t.ts + hold_, 0);
  }

  void on_timer(Context& c, const Timer&) override {
    active_ = false;
    double q = c.position(sym_).qty;
    if (q != 0)
      c.submit({.sym = sym_, .side = q > 0 ? Side::Sell : Side::Buy, .qty = std::abs(q), .reduce_only = true});
  }

 private:
  SymbolId sym_;
  Duration window_, hold_;
  double threshold_, qty_;
  std::deque<Trade> window_trades_;
  double flow_ = 0;
  bool active_ = false;
};

}  // namespace

std::unique_ptr<Strategy> make_trade_flow(const Config& p, const Instruments& ins) {
  return std::make_unique<TradeFlow>(p, ins);
}

}  // namespace xrock
