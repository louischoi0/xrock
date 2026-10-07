#pragma once

#include <string>

#include "core/events.hpp"

namespace xrock {

class MarketListener {
 public:
  virtual ~MarketListener() = default;
  virtual void on_market(const MarketEvent&) = 0;
};

class ExecListener {
 public:
  virtual ~ExecListener() = default;
  virtual void on_order_update(const OrderUpdate&) = 0;
  virtual void on_fill(const Fill&) = 0;
};

// Push-based market data. Live feeds call the listener from their own IO threads.
class MarketDataFeed {
 public:
  virtual ~MarketDataFeed() = default;
  virtual void start(MarketListener&) = 0;
  virtual void stop() = 0;
  virtual bool done() const { return false; }  // finite feeds (synthetic) report completion
};

// Pull-based historical download (REST / bulk archives), one UTC day at a time.
class HistoricalSource {
 public:
  virtual ~HistoricalSource() = default;
  // Emits the day's events of one symbol and stream ("trade", "funding", ...) in timestamp order, only after the
  // whole day was downloaded and parsed. Returns false if the venue has no data for that day.
  virtual bool fetch_day(SymbolId, const std::string& stream, Timestamp day, MarketListener&) = 0;
};

// Order entry. Implementations: backtest::SimGateway, exchange::bybit::Gateway.
class OrderGateway {
 public:
  virtual ~OrderGateway() = default;
  virtual void set_listener(ExecListener&) = 0;
  virtual void start() {}
  virtual void stop() {}
  virtual void submit(const Order&) = 0;
  virtual void cancel(const Order&) = 0;
};

}  // namespace xrock
