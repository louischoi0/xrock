#include "ml/splits.hpp"

#include <format>
#include <stdexcept>

namespace xrock::ml {

Splits Splits::walk_forward(Timestamp start, Timestamp end, Duration train, Duration test, Duration step,
                            Duration embargo, bool expanding, Timestamp holdout_start, Duration max_horizon) {
  if (train <= 0 || test <= 0 || step <= 0) throw std::runtime_error("splits: train, test and step must be > 0");
  if (embargo < max_horizon)
    throw std::runtime_error(std::format("splits: embargo ({}s) must cover the longest label horizon ({}s)",
                                         embargo / kSecond, max_horizon / kSecond));
  Splits s;
  s.holdout_start = holdout_start;
  Timestamp limit = std::min(end, holdout_start);
  for (Timestamp from = start;; from += step) {
    Fold f{.train_start = expanding ? start : from, .train_end = from + train};
    f.test_start = f.train_end + embargo;
    f.test_end = f.test_start + test;
    if (f.test_end > limit) break;
    s.folds.push_back(f);
  }
  if (s.folds.empty()) throw std::runtime_error("splits: no fold fits before the holdout; shorten train/test");
  return s;
}

Splits Splits::from(const Config& cfg, Timestamp start, Timestamp end, Duration max_horizon) {
  Config c = cfg.section("splits");
  Duration test = c.duration("test", kDay);
  return walk_forward(start, end, c.duration("train", 7 * kDay), test, c.duration("step", test),
                      c.duration("embargo", max_horizon), c.flag("expanding", false),
                      c.find("holdout_start") ? c.date("holdout_start") : kNever, max_horizon);
}

}  // namespace xrock::ml
