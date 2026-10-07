#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string>

#include "core/config.hpp"
#include "core/events.hpp"
#include "core/instruments.hpp"

namespace xrock {

// What a strategy may see and do. Implemented by engine::Engine, identically in backtest and live.
class Context {
 public:
  virtual ~Context() = default;
  virtual Timestamp now() const = 0;
  virtual const Instruments& instruments() const = 0;
  // qty is floored to the lot size, a limit price rounded to the tick.
  // Returns 0 if rejected pre-trade (journaled, no callback).
  virtual OrderId submit(const OrderRequest&) = 0;
  virtual void cancel(OrderId) = 0;
  virtual const Position& position(SymbolId) const = 0;
  virtual double equity() const = 0;
  virtual void set_timer(Timestamp at, std::uint64_t id) = 0;
  virtual void subscribe_bars(SymbolId, Duration interval) = 0;  // on_bar at each close of a bar with trades
  // Model outputs at now(), journaled (predictions.csv) for shadow-mode and offline/online parity checks.
  virtual void log_prediction(SymbolId, std::span<const float>) {}
  // A decision at now(), journaled (decisions.csv) so every order and exit can be traced back to its reason.
  virtual void log_decision(const Decision&) {}
};

// Event-driven strategy. All callbacks run on one thread, in timestamp order.
class Strategy {
 public:
  virtual ~Strategy() = default;
  virtual void on_start(Context&) {}
  virtual void on_trade(Context&, const Trade&) {}
  virtual void on_book(Context&, const Book&) {}
  virtual void on_bar(Context&, const Bar&) {}
  virtual void on_funding(Context&, const Funding&) {}
  virtual void on_timer(Context&, const Timer&) {}
  virtual void on_order(Context&, const OrderUpdate&) {}
  virtual void on_fill(Context&, const Fill&) {}
  virtual void on_stop(Context&) {}
};

// Strategies by name: built-ins (registry.cpp) plus any added with register_strategy() (e.g. ml's "model").
// params = the [strategy.<name>] section.
using StrategyFactory = std::function<std::unique_ptr<Strategy>(const Config&, const Instruments&)>;
std::unique_ptr<Strategy> make_strategy(const std::string& name, const Config& params, const Instruments&);
void register_strategy(const std::string& name, StrategyFactory);

}  // namespace xrock
