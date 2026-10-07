#pragma once

#include <span>
#include <vector>

#include "core/types.hpp"

namespace xrock::ml {

// The last `length` feature vectors on the sampling grid, oldest first: the input of a sequence model ([T, F]).
// A window only ever holds consecutive grid samples: a push that is not exactly one `sample` after the previous one
// (a skipped step - warm-up, stale data, NaN feature) starts it over. Offline windows follow the same rule
// (python/xrock_ml/dataset.py: Dataset.windows), so a window offline is a window here, with the same rows.
//
// Ring buffer with every row written twice (at i and i + length): the window is always one contiguous block,
// so data() needs no copy.
class SampleWindow {
 public:
  SampleWindow(std::size_t length, std::size_t width, Duration sample);
  void push(Timestamp ts, std::span<const float> x);
  void reset() { count_ = 0; }
  bool full() const { return count_ >= length_; }
  std::span<const float> data() const;  // length * width floats, oldest row first; valid while full()
  std::size_t length() const { return length_; }

 private:
  std::size_t length_, width_;
  Duration sample_;
  std::vector<float> buf_;  // 2 * length rows
  std::size_t next_ = 0, count_ = 0;
  Timestamp last_ = 0;
};

}  // namespace xrock::ml
