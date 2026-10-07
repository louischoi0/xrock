#pragma once

#include <vector>

#include "core/config.hpp"

namespace xrock::ml {

// One walk-forward fold over sample timestamps: train [train_start, train_end), test [test_start, test_end).
// test_start = train_end + embargo, and embargo >= the longest label horizon, so no training label overlaps the
// test window (purging by construction).
struct Fold {
  Timestamp train_start = 0, train_end = 0, test_start = 0, test_end = 0;
};

// [splits]: train, test, step (default = test), embargo, expanding, holdout_start.
struct Splits {
  std::vector<Fold> folds;
  Timestamp holdout_start = kNever;  // samples at or after it belong to no fold: the final, untouched test

  // Folds tile [start, holdout_start) (or [start, end) without a holdout). Throws if embargo < max_horizon or no
  // fold fits.
  static Splits walk_forward(Timestamp start, Timestamp end, Duration train, Duration test, Duration step,
                             Duration embargo, bool expanding, Timestamp holdout_start, Duration max_horizon);
  static Splits from(const Config&, Timestamp start, Timestamp end, Duration max_horizon);
};

}  // namespace xrock::ml
