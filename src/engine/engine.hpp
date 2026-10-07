#pragma once

#include <queue>
#include <unordered_map>
#include <vector>

#include "analytics/journal.hpp"
#include "core/clock.hpp"
#include "core/interfaces.hpp"
#include "data/bar_aggregator.hpp"
#include "engine/portfolio.hpp"
#include "engine/risk.hpp"
#include "strategy/strategy.hpp"

namespace xrock::engine {

// Hosts one Strategy. The same object runs in backtest (SimClock + SimGateway) and live (WallClock + venue
// gateway); only the clock, the gateway and the driver calling on_* / advance() differ.
class Engine final : public Context, public MarketListener, public ExecListener {
 public:
  Engine(const Instruments&, const Clock&, OrderGateway&, RiskManager&, analytics::Journal&, Strategy&,
         double initial_equity, Duration equity_interval);

  void start();
  void stop();
  // Fires timers, closes bars and samples equity due at or before `now`.
  void advance(Timestamp now);
  Timestamp next_deadline() const;  // earliest pending timer / bar close / equity sample
  std::vector<OrderId> open_orders() const;

  void on_market(const MarketEvent&) override;
  void on_order_update(const OrderUpdate&) override;
  void on_fill(const Fill&) override;

  Timestamp now() const override { return clock_.now(); }
  const Instruments& instruments() const override { return ins_; }
  OrderId submit(const OrderRequest&) override;
  void cancel(OrderId) override;
  const Position& position(SymbolId s) const override { return pf_.position(s); }
  double equity() const override { return pf_.equity(); }
  void set_timer(Timestamp at, std::uint64_t id) override { timers_.push({at, id}); }
  void subscribe_bars(SymbolId, Duration interval) override;
  void log_prediction(SymbolId s, std::span<const float> y) override {
    journal_.predictions.push_back({now(), s, {y.begin(), y.end()}});
  }
  void log_decision(const Decision& d) override { journal_.decisions.push_back({now(), d}); }

 private:
  struct Later {
    bool operator()(const Timer& a, const Timer& b) const { return a.ts > b.ts; }
  };

  const Instruments& ins_;
  const Clock& clock_;
  OrderGateway& gw_;
  RiskManager& risk_;
  analytics::Journal& journal_;
  Strategy& strat_;
  Portfolio pf_;
  std::unordered_map<OrderId, Order> open_;
  OrderId next_id_ = 1;
  std::vector<data::BarAggregator> bars_;
  std::priority_queue<Timer, std::vector<Timer>, Later> timers_;
  Duration equity_interval_;
  Timestamp next_equity_ = kNever;
};

}  // namespace xrock::engine
