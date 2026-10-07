#include "ml/model_strategy.hpp"

#include <algorithm>
#include <cmath>

#include "ml/runner.hpp"
#include "strategy/strategy.hpp"

namespace xrock::ml {
namespace {

class ModelStrategy final : public Strategy {
 public:
  ModelStrategy(const Config& p, const Instruments& ins)
      : run_(p.str("model")),
        sym_(ins.id(p.str("symbol", run_.info().symbol))),
        out_(run_.output_index(p.str("output", run_.info().outputs.front()))),
        trade_(p.flag("trade", false)),
        enter_(p.num("enter", 0)),
        exit_(p.num("exit", 0)),
        qty_(p.num("qty", 0)) {
    if (trade_ && (qty_ <= 0 || enter_ <= 0 || exit_ > enter_))
      throw std::runtime_error("model strategy: trading needs qty > 0 and 0 <= exit <= enter, enter > 0");
  }

  void on_start(Context& c) override { c.set_timer(run_.first_grid_after(c.now()), 0); }
  void on_trade(Context&, const Trade& t) override { feed(t); }
  void on_book(Context&, const Book& b) override { feed(b); }
  void on_funding(Context&, const Funding& f) override { feed(f); }

  void on_timer(Context& c, const Timer& t) override {
    c.set_timer(t.ts + run_.sample(), 0);
    if (!run_.step(t.ts)) return;
    c.log_prediction(sym_, run_.y());
    if (trade_) act(c, run_.y()[out_]);
  }

  void on_order(Context&, const OrderUpdate& u) override {
    if (u.id == working_ && is_terminal(u.status)) working_ = 0;
  }

 private:
  void feed(const MarketEvent& e) {
    if (event_sym(e) == sym_) run_.feed(e);
  }

  void act(Context& c, double p) {
    double pos = c.position(sym_).qty;
    double target = p > enter_ ? qty_ : p < -enter_ ? -qty_ : std::abs(p) < exit_ ? 0 : pos;
    double delta = target - pos;
    if (working_ || std::abs(delta) < c.instruments()[sym_].lot_size / 2) return;
    working_ = c.submit({.sym = sym_, .side = delta > 0 ? Side::Buy : Side::Sell, .qty = std::abs(delta),
                         .reduce_only = std::abs(target) < std::abs(pos) && target * pos >= 0});
    const char* why = p > enter_ ? "above-enter" : p < -enter_ ? "below-enter" : "inside-exit";
    c.log_decision({.sym = sym_, .kind = working_ ? "rebalance" : "rejected", .reason = why, .order = working_,
                    .values = {{"prediction", p}, {"position", pos}, {"target", target}}});
  }

  ModelRunner run_;
  SymbolId sym_;
  std::size_t out_;
  bool trade_;
  double enter_, exit_, qty_;
  OrderId working_ = 0;
};

}  // namespace

void register_barrier_strategy();  // barrier_strategy.cpp

void register_strategies() {
  register_barrier_strategy();
  register_strategy("model", [](const Config& p, const Instruments& ins) -> std::unique_ptr<Strategy> {
    return std::make_unique<ModelStrategy>(p, ins);
  });
}

}  // namespace xrock::ml
