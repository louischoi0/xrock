#include "strategy/strategy.hpp"

namespace xrock {
namespace {

// Benchmark baseline: buy `qty` on the first trade and hold.
class BuyHold final : public Strategy {
 public:
  BuyHold(const Config& p, const Instruments& ins) : sym_(ins.id(p.str("symbol"))), qty_(p.num("qty")) {}

  void on_trade(Context& c, const Trade& t) override {
    if (bought_ || t.sym != sym_) return;
    bought_ = c.submit({.sym = sym_, .side = Side::Buy, .qty = qty_}) != 0;
  }

 private:
  SymbolId sym_;
  double qty_;
  bool bought_ = false;
};

}  // namespace

std::unique_ptr<Strategy> make_buy_hold(const Config& p, const Instruments& ins) {
  return std::make_unique<BuyHold>(p, ins);
}

}  // namespace xrock
