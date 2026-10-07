#include "ml/features.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "core/config.hpp"

namespace xrock::ml {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Sum of values over [now - d, now). Values are added in time order and pruned at compute time.
class Window {
 public:
  explicit Window(Duration d) : d_(d) {}
  void add(Timestamp ts, double v) {
    q_.emplace_back(ts, v);
    sum_ += v;
  }
  double sum(Timestamp now) {
    while (!q_.empty() && q_.front().first < now - d_) {
      sum_ -= q_.front().second;
      q_.pop_front();
    }
    if (q_.empty()) sum_ = 0;  // reset drift whenever the window empties
    return sum_;
  }
  std::size_t count(Timestamp now) {
    sum(now);
    return q_.size();
  }
  Duration span() const { return d_; }

 private:
  Duration d_;
  std::deque<std::pair<Timestamp, double>> q_;
  double sum_ = 0;
};

class Ret final : public Feature {
 public:
  explicit Ret(Duration d) : d_(d) {}
  double compute(Timestamp now, const MarketState& s) override {
    double then = s.mid_before(now - d_), m = s.mid();
    return then > 0 && m > 0 ? std::log(m / then) : kNaN;
  }
  Duration lookback() const override { return d_; }

 private:
  Duration d_;
};

class Vol final : public Feature {
 public:
  explicit Vol(Duration d) : w_(d) {}
  void on_event(const MarketEvent& e, const MarketState&) override {
    const Trade* t = std::get_if<Trade>(&e);
    if (!t) return;
    if (prev_ > 0) {
      double r = std::log(t->price / prev_);
      w_.add(t->ts, r * r);
    }
    prev_ = t->price;
  }
  double compute(Timestamp now, const MarketState&) override { return std::sqrt(std::max(0.0, w_.sum(now))); }
  Duration lookback() const override { return w_.span(); }

 private:
  Window w_;
  double prev_ = 0;
};

class Flow final : public Feature {
 public:
  explicit Flow(Duration d) : signed_(d), total_(d) {}
  void on_event(const MarketEvent& e, const MarketState&) override {
    if (const Trade* t = std::get_if<Trade>(&e)) {
      signed_.add(t->ts, sign(t->aggressor) * t->qty);
      total_.add(t->ts, t->qty);
    }
  }
  double compute(Timestamp now, const MarketState&) override {
    double total = total_.sum(now), s = signed_.sum(now);
    return total > 0 ? s / total : 0.0;
  }
  Duration lookback() const override { return total_.span(); }

 private:
  Window signed_, total_;
};

class LogVolume final : public Feature {
 public:
  explicit LogVolume(Duration d) : w_(d) {}
  void on_event(const MarketEvent& e, const MarketState&) override {
    if (const Trade* t = std::get_if<Trade>(&e)) w_.add(t->ts, t->qty);
  }
  double compute(Timestamp now, const MarketState&) override { return std::log1p(std::max(0.0, w_.sum(now))); }
  Duration lookback() const override { return w_.span(); }

 private:
  Window w_;
};

class Rate final : public Feature {
 public:
  explicit Rate(Duration d) : w_(d) {}
  void on_event(const MarketEvent& e, const MarketState&) override {
    if (const Trade* t = std::get_if<Trade>(&e)) w_.add(t->ts, 1);
  }
  double compute(Timestamp now, const MarketState&) override {
    return static_cast<double>(w_.count(now)) * kSecond / static_cast<double>(w_.span());
  }
  Duration lookback() const override { return w_.span(); }

 private:
  Window w_;
};

class Spread final : public Feature {
 public:
  double compute(Timestamp, const MarketState& s) override {
    double bid = s.book.bid(), ask = s.book.ask();
    return bid > 0 && ask > 0 ? (ask - bid) / s.book.mid() * 1e4 : kNaN;
  }
};

class Imbalance final : public Feature {
 public:
  explicit Imbalance(int n) : n_(n) {}
  double compute(Timestamp, const MarketState& s) override {
    if (s.book.bid() <= 0 || s.book.ask() <= 0) return kNaN;
    double b = 0, a = 0;
    for (int i = 0; i < n_; ++i) {
      b += s.book.bids[i].qty;
      a += s.book.asks[i].qty;
    }
    return b + a > 0 ? (b - a) / (b + a) : 0.0;
  }

 private:
  int n_;
};

class FundingRate final : public Feature {
 public:
  double compute(Timestamp, const MarketState& s) override { return s.funding_rate; }
};

// Log mid returns on a fixed grid of interval d: r_k = log(mid(b_k) / mid(b_{k-1})), mid(b) = the mid in force
// just before boundary b. Path-independent: boundaries are processed once, whether reached by an event or by a
// compute(), and each reads only events before it - so export and engine see the same returns.
class GridVol : public Feature {
 public:
  GridVol(Duration d, Duration lookback) : d_(d), lookback_(lookback) {}
  void on_event(const MarketEvent& e, const MarketState& s) override { advance(event_ts(e), s); }
  double compute(Timestamp now, const MarketState& s) override {
    advance(now, s);
    return sigma();
  }
  Duration lookback() const override { return lookback_; }

 protected:
  virtual void add(double r) = 0;
  virtual double sigma() const = 0;  // per-interval standard deviation; NaN until defined

 private:
  void advance(Timestamp now, const MarketState& s) {
    if (next_ == kNever) next_ = (now / d_ + 1) * d_;
    for (; next_ <= now; next_ += d_) {
      double m = s.mid_before(next_);
      if (prev_ > 0 && m > 0) add(std::log(m / prev_));
      if (m > 0) prev_ = m;
    }
  }
  Duration d_, lookback_;
  Timestamp next_ = kNever;
  double prev_ = 0;
};

// rv_<d>_<n>: sqrt(mean of the last n squared d-returns).
class RealizedVol final : public GridVol {
 public:
  RealizedVol(Duration d, int n) : GridVol(d, d * (n + 1)), n_(n) {}

 private:
  void add(double r) override {
    q_.push_back(r * r);
    sum_ += r * r;
    if (static_cast<int>(q_.size()) > n_) {
      sum_ -= q_.front();
      q_.pop_front();
    }
  }
  double sigma() const override {
    return static_cast<int>(q_.size()) == n_ ? std::sqrt(std::max(0.0, sum_) / n_) : kNaN;
  }
  int n_;
  std::deque<double> q_;
  double sum_ = 0;
};

// ewmavol_<d>_<n>: EWMA of squared d-returns with a half-life of n intervals (RiskMetrics-style), from n returns on.
class EwmaVol final : public GridVol {
 public:
  EwmaVol(Duration d, int n) : GridVol(d, d * (n + 1)), n_(n), lambda_(std::exp2(-1.0 / n)) {}

 private:
  void add(double r) override {
    var_ = count_++ == 0 ? r * r : lambda_ * var_ + (1 - lambda_) * r * r;
  }
  double sigma() const override { return count_ >= n_ ? std::sqrt(var_) : kNaN; }
  int n_;
  double lambda_, var_ = 0;
  int count_ = 0;
};

// garch_<d>: GARCH(1,1) one-step-ahead sigma of d-returns with fixed alpha = 0.05, beta = 0.90 and variance
// targeting (omega = (1 - alpha - beta) * running mean of r^2), so no fitting is needed online. Defined after 30
// returns. Fit alpha/beta offline and add parameters to the name if they matter.
class Garch final : public GridVol {
 public:
  explicit Garch(Duration d) : GridVol(d, d * (kMin + 1)) {}

 private:
  static constexpr int kMin = 30;
  static constexpr double kAlpha = 0.05, kBeta = 0.90;
  void add(double r) override {
    ++count_;
    mean_r2_ += (r * r - mean_r2_) / count_;
    double omega = (1 - kAlpha - kBeta) * mean_r2_;
    var_ = count_ == 1 ? r * r : omega + kAlpha * r * r + kBeta * var_;
  }
  double sigma() const override { return count_ >= kMin ? std::sqrt(var_) : kNaN; }
  int count_ = 0;
  double mean_r2_ = 0, var_ = 0;
};

std::unique_ptr<Feature> make_feature(const std::string& name) {
  auto under = name.find('_');
  std::string kind = name.substr(0, under), arg = under == std::string::npos ? "" : name.substr(under + 1);
  if (kind == "rv" || kind == "ewmavol" || kind == "garch") {
    auto parts = split(arg, '_');
    Duration d = parts.empty() ? 0 : parse_duration(parts[0]);
    int n = 0;
    if (parts.size() == 2) std::from_chars(parts[1].data(), parts[1].data() + parts[1].size(), n);
    if (kind == "garch" && parts.size() == 1 && d > 0) return std::make_unique<Garch>(d);
    if (kind != "garch" && parts.size() == 2 && d > 0 && n >= 1)
      return kind == "rv" ? std::unique_ptr<Feature>(std::make_unique<RealizedVol>(d, n))
                          : std::make_unique<EwmaVol>(d, n);
    throw std::runtime_error("feature '" + name + "': expected rv_<d>_<n>, ewmavol_<d>_<n> or garch_<d>");
  }
  auto dur = [&] {
    if (arg.empty()) throw std::runtime_error("feature '" + name + "' needs a duration, e.g. " + kind + "_10s");
    Duration d = parse_duration(arg);
    if (d <= 0) throw std::runtime_error("feature '" + name + "': duration must be > 0");
    return d;
  };
  if (kind == "ret") return std::make_unique<Ret>(dur());
  if (kind == "vol") return std::make_unique<Vol>(dur());
  if (kind == "flow") return std::make_unique<Flow>(dur());
  if (kind == "logvol") return std::make_unique<LogVolume>(dur());
  if (kind == "rate") return std::make_unique<Rate>(dur());
  if (name == "spread") return std::make_unique<Spread>();
  if (name == "funding") return std::make_unique<FundingRate>();
  if (kind == "imb") {
    int n = 0;
    auto [p, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), n);
    if (ec != std::errc{} || p != arg.data() + arg.size() || n < 1 || n > kBookDepth)
      throw std::runtime_error("feature '" + name + "': imb_<levels>, 1.." + std::to_string(kBookDepth));
    return std::make_unique<Imbalance>(n);
  }
  throw std::runtime_error("unknown feature '" + name + "'");
}

}  // namespace

Duration vol_interval(const std::string& name) {
  for (const char* kind : {"rv_", "ewmavol_", "garch_"})
    if (name.starts_with(kind)) return parse_duration(split(name.substr(std::string_view(kind).size()), '_').at(0));
  return 0;
}

double MarketState::mid_before(Timestamp t) const {
  auto it = std::lower_bound(mids.begin(), mids.end(), t, [](const auto& m, Timestamp x) { return m.first < x; });
  return it == mids.begin() ? 0.0 : std::prev(it)->second;
}

FeatureSet::FeatureSet(std::vector<std::string> names, Duration max_gap) : names_(std::move(names)), max_gap_(max_gap) {
  if (names_.empty()) throw std::runtime_error("feature set is empty");
  for (const auto& n : names_) {
    features_.push_back(make_feature(n));
    warmup_ = std::max(warmup_, features_.back()->lookback());
  }
}

void FeatureSet::on_event(const MarketEvent& e) {
  MarketState& s = state_;
  Timestamp ts = event_ts(e);
  s.first_ts = std::min(s.first_ts, ts);
  s.last_ts = ts;
  std::visit(overloaded{[&](const Trade& t) { s.last_trade = t.price; }, [&](const Book& b) { s.book = b; },
                        [&](const Funding& f) { s.funding_rate = f.rate; }},
             e);
  double m = s.mid();
  if (m > 0 && (s.mids.empty() || s.mids.back().second != m)) s.mids.emplace_back(ts, m);
  // Keep one mid at or before the longest lookback so mid_before() still has an answer there.
  while (s.mids.size() > 1 && s.mids[1].first < ts - warmup_) s.mids.pop_front();
  for (auto& f : features_) f->on_event(e, s);
}

bool FeatureSet::ready(Timestamp now) const {
  return state_.first_ts != kNever && now - state_.first_ts > warmup_ && now - state_.last_ts <= max_gap_ &&
         state_.mid() > 0;
}

void FeatureSet::compute(Timestamp now, std::span<float> out) {
  if (out.size() != features_.size()) throw std::invalid_argument("feature output size mismatch");
  for (std::size_t i = 0; i < features_.size(); ++i) out[i] = static_cast<float>(features_[i]->compute(now, state_));
}

}  // namespace xrock::ml
