#include "ml/window.hpp"

#include <algorithm>
#include <stdexcept>

namespace xrock::ml {

SampleWindow::SampleWindow(std::size_t length, std::size_t width, Duration sample)
    : length_(length), width_(width), sample_(sample), buf_(2 * length * width) {
  if (length == 0 || width == 0 || sample <= 0) throw std::invalid_argument("SampleWindow: empty shape");
}

void SampleWindow::push(Timestamp ts, std::span<const float> x) {
  if (x.size() != width_) throw std::invalid_argument("SampleWindow: row width mismatch");
  if (count_ > 0 && ts != last_ + sample_) count_ = 0;  // a gap on the grid: start over
  std::copy(x.begin(), x.end(), buf_.begin() + next_ * width_);
  std::copy(x.begin(), x.end(), buf_.begin() + (next_ + length_) * width_);
  next_ = (next_ + 1) % length_;
  last_ = ts;
  ++count_;
}

std::span<const float> SampleWindow::data() const {
  if (!full()) throw std::logic_error("SampleWindow: not full");
  return {buf_.data() + next_ * width_, length_ * width_};  // rows next_ .. next_ + length - 1 = oldest .. newest
}

}  // namespace xrock::ml
