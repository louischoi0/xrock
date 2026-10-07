#include <onnxruntime_cxx_api.h>

#include <format>
#include <stdexcept>

#include "ml/model.hpp"

namespace xrock::ml {
namespace {

Ort::Env& env() {
  static Ort::Env e(ORT_LOGGING_LEVEL_ERROR, "xrock");
  return e;
}

// One float32 input [1, F] (window 1) or [1, T, F] (window T; batch may be -1) -> first output [1, O]. Raw features in, so any normalization lives in the
// graph. Single-threaded intra-op: one sample per call, on the engine thread, where latency beats parallelism.
class OnnxModel final : public Model {
 public:
  OnnxModel(ModelInfo info, const std::filesystem::path& file) : session_(nullptr) {
    info_ = std::move(info);
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);
    opt.SetInterOpNumThreads(1);
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
      session_ = Ort::Session(env(), file.c_str(), opt);
    } catch (const Ort::Exception& e) {
      throw std::runtime_error("model " + file.string() + ": " + e.what());
    }
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() < 1)
      throw std::runtime_error("model " + file.string() + ": needs exactly one input and at least one output");
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    output_name_ = session_.GetOutputNameAllocated(0, alloc).get();
    std::int64_t t = static_cast<std::int64_t>(info_.window), f = static_cast<std::int64_t>(info_.features.names.size());
    in_shape_ = info_.window == 1 ? std::vector<std::int64_t>{1, f} : std::vector<std::int64_t>{1, t, f};
    check(file, session_.GetInputTypeInfo(0), in_shape_, "input");
    check(file, session_.GetOutputTypeInfo(0), {1, static_cast<std::int64_t>(info_.outputs.size())}, "output");
  }

  void predict(std::span<const float> x, std::span<float> y) override {
    std::int64_t out_shape[2] = {1, static_cast<std::int64_t>(y.size())};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(x.data()), x.size(), in_shape_.data(),
                                                    in_shape_.size());
    Ort::Value out = Ort::Value::CreateTensor<float>(mem_, y.data(), y.size(), out_shape, 2);  // written in place
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    session_.Run(run_opt_, in_names, &in, 1, out_names, &out, 1);
  }

 private:
  static void check(const std::filesystem::path& file, const Ort::TypeInfo& t, const std::vector<std::int64_t>& want,
                    const char* what) {
    auto shape = t.GetTensorTypeAndShapeInfo();
    auto dims = shape.GetShape();
    bool ok = shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && dims.size() == want.size() &&
              (dims[0] == 1 || dims[0] < 0);
    for (std::size_t i = 1; ok && i < dims.size(); ++i) ok = dims[i] == want[i];
    if (!ok) {
      std::string w;
      for (auto d : want) w += (w.empty() ? "" : ", ") + std::to_string(d);
      throw std::runtime_error(std::format("model {}: {} must be float32 [{}] to match the model file", file.string(),
                                           what, w));
    }
  }

  Ort::Session session_;
  Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::RunOptions run_opt_;
  std::string input_name_, output_name_;
  std::vector<std::int64_t> in_shape_;
};

}  // namespace

std::unique_ptr<Model> make_onnx_model(ModelInfo info, const std::filesystem::path& file) {
  return std::make_unique<OnnxModel>(std::move(info), file);
}

}  // namespace xrock::ml
