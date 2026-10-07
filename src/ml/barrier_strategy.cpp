#include <array>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "core/log.hpp"
#include "ml/runner.hpp"
#include "strategy/strategy.hpp"

namespace xrock::ml {
namespace {

// "barrier": a model decides the trade parameters; a volatility feature scales them; the strategy executes them.
//
// [strategy.barrier]
//   model    = models/x.json   xrock-model/1; must contain the volatility feature (barrier_vol)
//   symbol   = BTCUSDT         default: the model's symbol
//   trade    = false           false = shadow (predictions.csv only)
//   qty      = 0.01
//   min_edge = 0.0012          enter only if the expected log return beats this (round-trip cost + margin)
//   log_skips = false          also journal a "skip" decision at every flat grid step without an entry (large)
// Two output contracts, chosen by the model's output names:
//   grid    any outputs tb_<i>_long / tb_<i>_short = expected log return of barrier config i (model "barriers") on
//           that side. The strategy takes the best (config, side) if it beats min_edge: buy/sell threshold =
//           min_edge, duration/take-profit/loss-cut = the chosen config.
//   direct  outputs side, k_tp, k_sl, hold (seconds): enter long if side > min_edge, short if side < -min_edge,
//           with take-profit k_tp * sigma_h, loss-cut k_sl * sigma_h, time exit after hold. Clamped to
//           k in [k_min, k_max] (default 0.25 .. 10) and hold in [hold_min, hold_max] (default sample .. 1h).
// sigma_h = vol(t) * sqrt(hold / d) with vol = the model's barrier_vol (or param `vol`), d its grid interval.
// Exits: on every event of the symbol, mid at or beyond the loss-cut (checked first) or the take-profit closes the
// position with a market order; the time barrier closes it at entry + hold. One position at a time.
// Decisions (decisions.csv): enter (reason = chosen output or "direct"; side, k_tp, k_sl, hold_s, sigma, edge),
// open (entry fill: entry, tp, sl, deadline_s), exit (reason time / loss-cut / take-profit; mid, entry, held_s),
// rejected (an entry or exit order refused), skip (optional; best_edge).
class BarrierStrategy final : public Strategy {
 public:
  BarrierStrategy(const Config& p, const Instruments& ins)
      : run_(p.str("model")),
        sym_(ins.id(p.str("symbol", run_.info().symbol))),
        trade_(p.flag("trade", false)),
        qty_(p.num("qty", 0)),
        min_edge_(p.num("min_edge", 0)),
        k_min_(p.num("k_min", 0.25)),
        k_max_(p.num("k_max", 10)),
        hold_min_(p.duration("hold_min", run_.sample())),
        hold_max_(p.duration("hold_max", kHour)),
        log_skips_(p.flag("log_skips", false)) {
    std::string vol = p.str("vol", run_.info().barrier_vol);
    vol_col_ = run_.feature_index(vol);
    vol_d_ = vol_interval(vol);
    if (vol_d_ <= 0) throw std::runtime_error("barrier strategy: vol must be rv_/ewmavol_/garch_ feature");
    const auto& outs = run_.info().outputs;
    // Grid mode as soon as the model has tb_* outputs; other outputs (e.g. fwd_* of a multi-task model) are ignored.
    bool grid = std::any_of(outs.begin(), outs.end(), [](const auto& o) { return o.starts_with("tb_"); });
    if (grid) {
      const auto& bs = run_.info().barriers;
      for (std::size_t i = 0; i < bs.size(); ++i)
        for (int side : {+1, -1}) {
          auto name = std::format("tb_{}_{}", i, side > 0 ? "long" : "short");
          auto it = std::find(outs.begin(), outs.end(), name);
          if (it != outs.end()) grid_.push_back({static_cast<std::size_t>(it - outs.begin()), i, side});
        }
      if (grid_.empty()) throw std::runtime_error("barrier strategy: tb_* outputs without matching model barriers");
    } else {
      direct_ = {run_.output_index("side"), run_.output_index("k_tp"), run_.output_index("k_sl"),
                 run_.output_index("hold")};
    }
    if (trade_ && (qty_ <= 0 || min_edge_ < 0)) throw std::runtime_error("barrier strategy: needs qty > 0, min_edge >= 0");
  }

  void on_start(Context& c) override { c.set_timer(run_.first_grid_after(c.now()), kGridTimer); }
  void on_trade(Context& c, const Trade& t) override { on_event(c, t); }
  void on_book(Context& c, const Book& b) override { on_event(c, b); }
  void on_funding(Context&, const Funding& f) override {
    if (f.sym == sym_) run_.feed(f);
  }

  void on_timer(Context& c, const Timer& t) override {
    if (t.id == kGridTimer) {
      c.set_timer(t.ts + run_.sample(), kGridTimer);
      if (!run_.step(t.ts)) return;
      c.log_prediction(sym_, run_.y());
      if (trade_ && state_ == State::Flat) decide(c);
    } else if (t.id == exit_timer_ && state_ == State::Open) {
      close(c, "time");
    }
  }

  void on_fill(Context& c, const Fill& f) override {
    if (f.id != order_ || state_ != State::Entering) return;
    // Barriers are set from the actual entry price.
    double sigma_h = sigma_ * std::sqrt(static_cast<double>(plan_.hold) / static_cast<double>(vol_d_));
    entry_ = f.price;
    entry_ts_ = f.ts;
    tp_ = f.price * std::exp(plan_.side * plan_.k_tp * sigma_h);
    sl_ = f.price * std::exp(-plan_.side * plan_.k_sl * sigma_h);
    exit_timer_ = kGridTimer + ++trades_;
    c.set_timer(f.ts + plan_.hold, exit_timer_);
    state_ = State::Open;
    record(c, "open", plan_.reason, f.id,
           {{"entry", entry_}, {"tp", tp_}, {"sl", sl_}, {"deadline_s", static_cast<double>(plan_.hold) / kSecond}});
  }

  void on_order(Context& c, const OrderUpdate& u) override {
    if (u.id != order_ || !is_terminal(u.status)) return;
    if (u.status != OrderStatus::Filled)
      record(c, "rejected", std::string(state_ == State::Entering ? "entry:" : "exit:") + to_string(u.status), u.id,
             {});
    if (state_ == State::Entering && u.status != OrderStatus::Filled) state_ = State::Flat;
    if (state_ == State::Exiting) state_ = u.status == OrderStatus::Filled ? State::Flat : State::Open;
    order_ = 0;
  }

 private:
  enum class State { Flat, Entering, Open, Exiting };
  static constexpr std::uint64_t kGridTimer = 0;
  struct GridOutput {
    std::size_t output, barrier;
    int side;
  };
  struct Plan {
    int side = 0;
    double k_tp = 0, k_sl = 0;
    Duration hold = 0;
    std::string reason;  // the output that won, or "direct"
    double edge = 0;     // its value
  };

  void record(Context& c, const char* kind, std::string reason, OrderId order,
              std::vector<std::pair<std::string, double>> values) {
    c.log_decision({.sym = sym_, .kind = kind, .reason = std::move(reason), .order = order, .values = std::move(values)});
  }


  void on_event(Context& c, const MarketEvent& e) {
    if (event_sym(e) != sym_) return;
    run_.feed(e);
    if (state_ != State::Open) return;
    double m = run_.state().mid();
    bool long_side = plan_.side > 0;
    if (long_side ? m <= sl_ : m >= sl_)
      close(c, "loss-cut");
    else if (long_side ? m >= tp_ : m <= tp_)
      close(c, "take-profit");
  }

  void decide(Context& c) {
    auto y = run_.y();
    Plan plan;
    if (!grid_.empty()) {
      const GridOutput* best = nullptr;
      for (const auto& g : grid_)
        if (!best || y[g.output] > y[best->output]) best = &g;
      if (y[best->output] <= min_edge_) {
        if (log_skips_) record(c, "skip", run_.info().outputs[best->output], 0, {{"best_edge", y[best->output]}});
        return;
      }
      const Barrier& b = run_.info().barriers[best->barrier];
      plan = {best->side, b.k_tp, b.k_sl, b.horizon, run_.info().outputs[best->output], y[best->output]};
    } else {
      double side = y[direct_[0]];
      if (std::abs(side) <= min_edge_) {
        if (log_skips_) record(c, "skip", "direct", 0, {{"best_edge", side}});
        return;
      }
      double hold_s = std::clamp(static_cast<double>(y[direct_[3]]), hold_min_ / 1e9, hold_max_ / 1e9);
      plan = {side > 0 ? 1 : -1, std::clamp(static_cast<double>(y[direct_[1]]), k_min_, k_max_),
              std::clamp(static_cast<double>(y[direct_[2]]), k_min_, k_max_), static_cast<Duration>(hold_s * 1e9),
              "direct", side};
    }
    sigma_ = run_.x()[vol_col_];
    if (!(sigma_ > 0)) return;
    std::vector<std::pair<std::string, double>> values{
        {"side", plan.side}, {"k_tp", plan.k_tp}, {"k_sl", plan.k_sl},
        {"hold_s", static_cast<double>(plan.hold) / kSecond}, {"sigma", sigma_}, {"edge", plan.edge}};
    order_ = c.submit({.sym = sym_, .side = plan.side > 0 ? Side::Buy : Side::Sell, .qty = qty_});
    if (order_ == 0) {  // refused pre-trade by risk
      record(c, "rejected", "entry:risk", 0, std::move(values));
      return;
    }
    record(c, "enter", plan.reason, order_, std::move(values));
    plan_ = plan;
    state_ = State::Entering;
  }

  void close(Context& c, const char* why) {
    double pos = c.position(sym_).qty;
    if (pos == 0) {  // closed outside the strategy (e.g. liquidation)
      record(c, "exit", std::string(why) + ":already-flat", 0, {});
      state_ = State::Flat;
      return;
    }
    order_ = c.submit({.sym = sym_, .side = pos > 0 ? Side::Sell : Side::Buy, .qty = std::abs(pos), .reduce_only = true});
    std::vector<std::pair<std::string, double>> values{
        {"mid", run_.state().mid()}, {"entry", entry_},
        {"held_s", static_cast<double>(c.now() - entry_ts_) / kSecond}};
    if (order_ == 0) {
      record(c, "rejected", "exit:risk", 0, std::move(values));
      return;
    }
    record(c, "exit", why, order_, std::move(values));
    state_ = State::Exiting;
  }

  ModelRunner run_;
  SymbolId sym_;
  bool trade_;
  double qty_, min_edge_, k_min_, k_max_;
  Duration hold_min_, hold_max_;
  std::size_t vol_col_ = 0;
  Duration vol_d_ = 0;
  std::vector<GridOutput> grid_;
  std::array<std::size_t, 4> direct_{};
  State state_ = State::Flat;
  Plan plan_;
  double sigma_ = 0, tp_ = 0, sl_ = 0, entry_ = 0;
  Timestamp entry_ts_ = 0;
  bool log_skips_;
  OrderId order_ = 0;
  std::uint64_t trades_ = 0, exit_timer_ = std::numeric_limits<std::uint64_t>::max();
};

}  // namespace

void register_barrier_strategy() {
  register_strategy("barrier", [](const Config& p, const Instruments& ins) -> std::unique_ptr<Strategy> {
    return std::make_unique<BarrierStrategy>(p, ins);
  });
}

}  // namespace xrock::ml
