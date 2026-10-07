#pragma once

#include <atomic>
#include <thread>

#include "core/config.hpp"
#include "core/instruments.hpp"
#include "core/interfaces.hpp"

namespace xrock::exchange {

// Offline stand-in for a venue: GBM trades, a consistent top-of-book every `book_every` trades and 8-hourly
// funding over [synthetic.start, synthetic.end), emitted as fast as possible. For the demo config and tests.
class SyntheticFeed final : public MarketDataFeed {
 public:
  SyntheticFeed(const Config&, const Instruments&);
  ~SyntheticFeed() override;
  void start(MarketListener&) override;  // runs generate() on a thread
  void stop() override;
  bool done() const override { return done_; }
  void generate(MarketListener&);  // synchronous

 private:
  const Instruments& ins_;
  Timestamp start_, end_;
  double price_, volatility_, trades_per_sec_;
  std::uint64_t book_every_, seed_;
  std::thread thread_;
  std::atomic<bool> stop_{false}, done_{false};
};

}  // namespace xrock::exchange
