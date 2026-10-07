#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "ml/dataset.hpp"

namespace xrock::ml {

// Header of a model file (JSON, format "xrock-model/1"). The model carries its feature contract, so whatever runs
// it builds exactly the FeatureSet it was trained on:
//   {"format": "xrock-model/1", "type": "linear" | "onnx", "symbol": "BTCUSDT",
//    "features": {"names": [...], "sample": "1s", "max_gap": "10s"},
//    "outputs": ["fwd_60s", ...],
//    "window": T (optional, default 1): the model reads the last T consecutive samples (ml::SampleWindow)
//    "barriers": [{"k_tp", "k_sl", "horizon"}], "barrier_vol": "<feature>" (optional): the barrier grid behind
//            tb_<i>_long/short outputs, copied from the dataset it was trained on
//    linear: "mean": [T*F], "std": [T*F], "weights": [[T*F] per output], "bias": [outputs]  (window flattened,
//            oldest row first)
//    onnx:   "file": "model.onnx" (relative to the JSON), input float32 [1, F] (T = 1) or [1, T, F]
//            -> output float32 [1, outputs]}
struct ModelInfo {
  std::string type, symbol;
  FeatureSpec features;
  std::vector<std::string> outputs;
  std::filesystem::path path;
  std::size_t window = 1;
  std::vector<Barrier> barriers;
  std::string barrier_vol;
  std::size_t input_size() const { return window * features.names.size(); }
};

class Model {
 public:
  virtual ~Model() = default;
  const ModelInfo& info() const { return info_; }
  // x: info().input_size() values - `window` rows of features (names order), oldest first; y: one per output.
  virtual void predict(std::span<const float> x, std::span<float> y) = 0;

 protected:
  ModelInfo info_;
};

std::unique_ptr<Model> load_model(const std::filesystem::path&);

// y = W ((x - mean) / std) + b. A baseline and the reference implementation of the contract.
struct LinearParams {
  std::vector<float> mean, std, bias;       // mean/std: input_size()
  std::vector<std::vector<float>> weights;  // [outputs][input_size()]
};
void save_linear_model(const std::filesystem::path&, const ModelInfo&, const LinearParams&);

}  // namespace xrock::ml
