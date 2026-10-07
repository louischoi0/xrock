#include <deque>
#include <numeric>
#include <stdexcept>

#include "strategy/strategy.hpp"

namespace xrock {
namespace {

// Bar-driven trend follower: long `qty` while the fast SMA of closes is above the slow SMA, short otherwise.
// `interval` from 1s to 1d covers both the short-horizon and the long-horizon setups.
class MaCross final : public Strategy {
 public:
  MaCross(const Config& p, const Instruments& ins)
      : sym_(ins.id(p.str("symbol"))),
        interval_(p.duration("interval", kMinute)),
        fast_(static_cast<std::size_t>(p.num("fast", 10))),
        slow_(static_cast<std::size_t>(p.num("slow", 30))),
        qty_(p.num("qty")) {
    if (fast_ == 0 || fast_ >= slow_) throw std::runtime_error("ma_cross: need 0 < fast < slow");
  }

  void on_start(Context& c) override { c.subscribe_bars(sym_, interval_); }

  void on_bar(Context& c, const Bar& b) override {
    if (b.sym != sym_ || b.interval != interval_) return;
    closes_.push_back(b.close);
    if (closes_.size() > slow_) closes_.pop_front();
    if (closes_.size() < slow_ || working_) return;
    double slow = std::accumulate(closes_.begin(), closes_.end(), 0.0) / slow_;
    double fast = std::accumulate(closes_.end() - static_cast<std::ptrdiff_t>(fast_), closes_.end(), 0.0) / fast_;
    double delta = (fast > slow ? qty_ : -qty_) - c.position(sym_).qty;
    if (std::abs(delta) < c.instruments()[sym_].lot_size / 2) return;
    working_ = c.submit({.sym = sym_, .side = delta > 0 ? Side::Buy : Side::Sell, .qty = std::abs(delta)});
  }

  void on_order(Context&, const OrderUpdate& u) override {
    if (u.id == working_ && is_terminal(u.status)) working_ = 0;
  }

 private:
  SymbolId sym_;
  Duration interval_;
  std::size_t fast_, slow_;
  double qty_;
  std::deque<double> closes_;
  OrderId working_ = 0;  // one order in flight at a time
};

}  // namespace

std::unique_ptr<Strategy> make_ma_cross(const Config& p, const Instruments& ins) {
  return std::make_unique<MaCross>(p, ins);
}

}  // namespace xrock
