#pragma once

#include <deque>
#include <vector>

#include "core/clock.hpp"
#include "core/config.hpp"
#include "core/instruments.hpp"
#include "core/interfaces.hpp"
#include "engine/portfolio.hpp"

namespace xrock::backtest {

struct SimParams {
  Duration latency = 50 * kMilli;  // one-way, applied to submit and cancel
  double slippage_bps = 0;         // extra adverse price on market orders
  double leverage = 1;             // isolated leverage, for liquidation
  static SimParams from(const Config&);  // [backtest] latency, slippage_bps; [margin] leverage
};

// Simulated venue: orders arrive after `latency`; market/marketable orders take the recorded touch (taker fee),
// resting limits fill when a trade prints through or the book crosses them (maker fee); isolated-margin liquidation.
// Keeps its own exchange-side Portfolio, like a real venue.
// TODO: partial fills / queue position, mark-price liquidation, cross margin, inverse contracts.
class SimGateway final : public OrderGateway {
 public:
  SimGateway(const Instruments&, const Clock&, SimParams);
  void set_listener(ExecListener& l) override { listener_ = &l; }
  void submit(const Order&) override;
  void cancel(const Order&) override;
  // Delivers arrivals due by the event time, then matches resting orders against the event.
  void on_market(const MarketEvent&);

 private:
  struct Request {
    Timestamp arrive;
    bool cancel;
    Order order;
  };
  struct Quote {
    double bid = 0, ask = 0, last = 0;
  };

  void arrive(const Request&);
  template <class Crosses>
  void match(SymbolId, Timestamp, Crosses);
  void fill(const Order&, Timestamp, double price, bool maker);
  void reject(const Order&, Timestamp, const char* reason);
  void check_liquidation(SymbolId, double mark, Timestamp);

  const Instruments& ins_;
  const Clock& clock_;
  SimParams p_;
  ExecListener* listener_ = nullptr;
  std::deque<Request> inflight_;  // FIFO: latency is constant
  std::vector<Order> resting_;
  std::vector<Quote> quotes_;
  engine::Portfolio pf_;
};

}  // namespace xrock::backtest
