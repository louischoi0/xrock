#include "exchange/synthetic.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace xrock::exchange {

SyntheticFeed::SyntheticFeed(const Config& cfg, const Instruments& ins)
    : ins_(ins),
      start_(cfg.date("synthetic.start")),
      end_(cfg.date("synthetic.end")),
      price_(cfg.num("synthetic.price", 60000)),
      volatility_(cfg.num("synthetic.volatility", 0.6)),
      trades_per_sec_(cfg.num("synthetic.trades_per_sec", 1)),
      book_every_(static_cast<std::uint64_t>(std::max(1.0, cfg.num("synthetic.book_every", 5)))),
      seed_(static_cast<std::uint64_t>(cfg.num("synthetic.seed", 42))) {}

SyntheticFeed::~SyntheticFeed() { stop(); }

void SyntheticFeed::start(MarketListener& l) {
  thread_ = std::thread([this, &l] { generate(l); });
}

void SyntheticFeed::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

void SyntheticFeed::generate(MarketListener& l) {
  constexpr Duration kFundingPeriod = 8 * kHour;
  constexpr double kYearSeconds = 365.0 * 86400;
  std::mt19937_64 rng(seed_);
  std::exponential_distribution<double> gap_s(trades_per_sec_), size(1.0);
  std::normal_distribution<double> z;
  std::bernoulli_distribution buy(0.5);

  for (SymbolId s = 0; s < ins_.size() && !stop_; ++s) {
    const Instrument& in = ins_[s];
    double px = price_;
    Timestamp t = start_, funding_at = (start_ / kFundingPeriod + 1) * kFundingPeriod;
    for (std::uint64_t n = 0; !stop_; ++n) {
      double dt = gap_s(rng);
      t += static_cast<Duration>(dt * kSecond);
      for (; funding_at <= t && funding_at < end_; funding_at += kFundingPeriod)
        l.on_market(Funding{.ts = funding_at, .sym = s, .rate = 1e-4 * (1 + z(rng))});
      if (t >= end_) break;

      px *= std::exp(volatility_ * std::sqrt(dt / kYearSeconds) * z(rng));
      double p = round_to(px, in.tick_size);
      Side side = buy(rng) ? Side::Buy : Side::Sell;
      double qty = std::max(in.lot_size, round_to(10 * in.lot_size * size(rng), in.lot_size));
      l.on_market(Trade{.ts = t, .sym = s, .aggressor = side, .price = p, .qty = qty});

      if (n % book_every_ != 0) continue;
      // A buy-aggressor trade printed at the ask, a sell at the bid.
      double ask = side == Side::Buy ? p : p + in.tick_size;
      Book b{.ts = t, .sym = s};
      for (int i = 0; i < kBookDepth; ++i) {
        b.asks[i] = {ask + i * in.tick_size, round_to(5 * in.lot_size * (1 + size(rng)), in.lot_size)};
        b.bids[i] = {ask - (i + 1) * in.tick_size, round_to(5 * in.lot_size * (1 + size(rng)), in.lot_size)};
      }
      l.on_market(b);
    }
  }
  done_ = true;
}

}  // namespace xrock::exchange
