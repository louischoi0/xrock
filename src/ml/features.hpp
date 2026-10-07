#pragma once

#include <deque>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/events.hpp"

namespace xrock::ml {

// What a feature may read besides its own windows. Updated by FeatureSet before features see an event.
struct MarketState {
  Book book{};               // last book (bids[0].price == 0 until one arrives)
  double last_trade = 0;     // last trade price
  double funding_rate = 0;   // last settled funding rate
  Timestamp first_ts = kNever, last_ts = 0;
  std::deque<std::pair<Timestamp, double>> mids;  // (ts, mid) on every mid change, pruned to the longest lookback

  // Book mid when a book was seen, else the last trade. 0 before any data. Labels use the same definition.
  double mid() const { return book.bid() > 0 && book.ask() > 0 ? book.mid() : last_trade; }
  double mid_before(Timestamp t) const;  // last mid with ts < t; 0 if none
};

// One or more feature columns computed online from one symbol's events.
// Contract: compute(now) reads only events with ts < now (the caller feeds events strictly in time order and
// samples at `now` before feeding any event with ts >= now). That is what keeps features free of the future.
class Feature {
 public:
  virtual ~Feature() = default;
  virtual void on_event(const MarketEvent&, const MarketState&) {}
  virtual double compute(Timestamp now, const MarketState&) = 0;  // NaN = not computable yet
  virtual Duration lookback() const { return 0; }
};

// d of a grid-volatility feature name (rv_/ewmavol_/garch_), 0 for any other name.
Duration vol_interval(const std::string& name);

// Feature spec: an ordered list of names. Built-ins (d = duration like 500ms, 10s, 5m; n = book levels):
//   ret_<d>       log(mid_now / mid_{now-d})
//   vol_<d>       realized vol: sqrt(sum of squared trade-to-trade log returns) over the last d
//   flow_<d>      (taker buy qty - taker sell qty) / total qty over the last d (0 when no trades)
//   logvol_<d>    log1p(total traded qty) over the last d
//   rate_<d>      trades per second over the last d
//   spread        (ask - bid) / mid in bps
//   imb_<n>       (bid qty - ask qty) / (bid qty + ask qty) over the top n levels
//   funding       last settled funding rate
// Volatility on a fixed grid of d-returns of the mid (per-interval sigma; scale to a horizon h by sqrt(h / d)):
//   rv_<d>_<n>       sqrt(mean of the last n squared d-returns)
//   ewmavol_<d>_<n>  EWMA of squared d-returns, half-life n intervals
//   garch_<d>        GARCH(1,1) one-step-ahead sigma, alpha 0.05, beta 0.90, variance targeting
// The model file stores this list; the same names always build the same computation, in export and in trading.
class FeatureSet {
 public:
  FeatureSet(std::vector<std::string> names, Duration max_gap);

  void on_event(const MarketEvent&);                // events of one symbol, in time order
  bool ready(Timestamp now) const;                  // warm-up covered and data not older than max_gap
  void compute(Timestamp now, std::span<float> out);  // size() values; NaN where not computable
  const std::vector<std::string>& names() const { return names_; }
  std::size_t size() const { return names_.size(); }
  const MarketState& state() const { return state_; }

 private:
  std::vector<std::string> names_;
  std::vector<std::unique_ptr<Feature>> features_;
  Duration max_gap_, warmup_ = 0;
  MarketState state_;
};

}  // namespace xrock::ml
