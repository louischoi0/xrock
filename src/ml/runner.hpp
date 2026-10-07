#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "ml/features.hpp"
#include "ml/model.hpp"
#include "ml/window.hpp"

namespace xrock::ml {

// A model with the FeatureSet and SampleWindow its file declares: the sampling path every model strategy shares.
// Feed it the symbol's events; at each grid time step() computes features from events before t, extends the window
// and predicts once the window is full.
class ModelRunner {
 public:
  explicit ModelRunner(const std::filesystem::path& model_file);

  const ModelInfo& info() const { return model_->info(); }
  Duration sample() const { return info().features.sample; }
  Timestamp first_grid_after(Timestamp t) const { return (t / sample() + 1) * sample(); }
  void feed(const MarketEvent& e) { fs_.on_event(e); }
  bool step(Timestamp t);  // false: not ready, NaN feature or window still filling (window then restarts)
  std::span<const float> y() const { return y_; }
  std::span<const float> x() const { return x_; }  // features at the last step
  const MarketState& state() const { return fs_.state(); }
  std::size_t feature_index(const std::string& name) const;  // throws if the model has no such feature
  std::size_t output_index(const std::string& name) const;   // throws if the model has no such output

 private:
  std::unique_ptr<Model> model_;
  FeatureSet fs_;
  std::vector<float> x_, y_;
  SampleWindow window_;
};

}  // namespace xrock::ml
