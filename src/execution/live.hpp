#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <variant>

#include "analytics/journal.hpp"
#include "core/clock.hpp"
#include "engine/engine.hpp"

namespace xrock::execution {

// Venue adapters call on_* from their IO threads; everything is queued and handed to the Engine on one thread, so
// strategy code never runs concurrently and sees the same callback order as in backtest.
class LiveRunner final : public MarketListener, public ExecListener {
 public:
  LiveRunner(const Config&, const Instruments&, MarketDataFeed&, OrderGateway&, Strategy&);
  // Blocks until `stop` is set, then cancels open orders and writes the journal under analytics.out_dir.
  // TODO: wait for cancel acks, optional flatten on exit, sync equity/positions from the venue, stream the journal.
  void run(const std::atomic<bool>& stop);

  void on_market(const MarketEvent& e) override { push(e); }
  void on_order_update(const OrderUpdate& u) override { push(u); }
  void on_fill(const Fill& f) override { push(f); }

 private:
  using Msg = std::variant<MarketEvent, OrderUpdate, Fill>;
  void push(Msg);

  const Instruments& ins_;
  MarketDataFeed& feed_;
  OrderGateway& gw_;
  WallClock clock_;
  engine::RiskManager risk_;
  analytics::Journal journal_;
  engine::Engine engine_;
  Duration tick_;
  std::filesystem::path out_dir_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Msg> queue_;
};

}  // namespace xrock::execution
