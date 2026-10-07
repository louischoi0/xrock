#include "ml/runner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xrock::ml {

ModelRunner::ModelRunner(const std::filesystem::path& model_file)
    : model_(load_model(model_file)),
      fs_(info().features.names, info().features.max_gap),
      x_(fs_.size()),
      y_(info().outputs.size()),
      window_(info().window, fs_.size(), info().features.sample) {}

bool ModelRunner::step(Timestamp t) {
  if (!fs_.ready(t)) {
    window_.reset();
    return false;
  }
  fs_.compute(t, x_);
  if (!std::all_of(x_.begin(), x_.end(), [](float v) { return std::isfinite(v); })) {
    window_.reset();
    return false;
  }
  window_.push(t, x_);
  if (!window_.full()) return false;
  model_->predict(window_.data(), y_);
  return true;
}

std::size_t ModelRunner::feature_index(const std::string& name) const {
  const auto& n = info().features.names;
  auto it = std::find(n.begin(), n.end(), name);
  if (it == n.end()) throw std::runtime_error("model has no feature " + name);
  return static_cast<std::size_t>(it - n.begin());
}

std::size_t ModelRunner::output_index(const std::string& name) const {
  const auto& o = info().outputs;
  auto it = std::find(o.begin(), o.end(), name);
  if (it == o.end()) throw std::runtime_error("model has no output " + name);
  return static_cast<std::size_t>(it - o.begin());
}

}  // namespace xrock::ml
