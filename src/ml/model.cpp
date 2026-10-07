#include "ml/model.hpp"

#include <format>
#include <fstream>
#include <simdjson.h>
#include <stdexcept>

namespace xrock::ml {
namespace {

using simdjson::dom::array;
using simdjson::dom::element;
using simdjson::dom::object;

std::vector<float> floats(element e) {
  std::vector<float> v;
  for (element x : array(e)) v.push_back(static_cast<float>(double(x)));
  return v;
}

std::vector<std::string> strings(element e) {
  std::vector<std::string> v;
  for (element x : array(e)) v.emplace_back(std::string_view(x));
  return v;
}

std::string join(const std::vector<float>& v) {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += std::format("{}{}", i ? ", " : "", v[i]);
  return s + "]";
}

class LinearModel final : public Model {
 public:
  LinearModel(ModelInfo info, LinearParams p) : p_(std::move(p)) {
    info_ = std::move(info);
    std::size_t f = info_.input_size(), o = info_.outputs.size();
    bool ok = p_.mean.size() == f && p_.std.size() == f && p_.bias.size() == o && p_.weights.size() == o;
    for (const auto& w : p_.weights) ok = ok && w.size() == f;
    for (float s : p_.std) ok = ok && s > 0;
    if (!ok) throw std::runtime_error("model " + info_.path.string() + ": linear parameter shapes do not match");
  }

  void predict(std::span<const float> x, std::span<float> y) override {
    for (std::size_t o = 0; o < y.size(); ++o) {
      double acc = p_.bias[o];
      for (std::size_t i = 0; i < x.size(); ++i) acc += p_.weights[o][i] * ((x[i] - p_.mean[i]) / p_.std[i]);
      y[o] = static_cast<float>(acc);
    }
  }

 private:
  LinearParams p_;
};

}  // namespace

#ifdef XROCK_WITH_ONNX
std::unique_ptr<Model> make_onnx_model(ModelInfo, const std::filesystem::path&);  // onnx_model.cpp
#endif

std::unique_ptr<Model> load_model(const std::filesystem::path& path) {
  simdjson::dom::parser parser;
  object doc;
  try {
    doc = parser.load(path.string());
  } catch (const simdjson::simdjson_error& e) {
    throw std::runtime_error("model " + path.string() + ": " + e.what());
  }
  if (std::string_view(doc["format"]) != "xrock-model/1")
    throw std::runtime_error("model " + path.string() + ": format must be xrock-model/1");
  ModelInfo info;
  info.path = path;
  info.type = std::string_view(doc["type"]);
  info.symbol = std::string_view(doc["symbol"]);
  object fs = doc["features"];
  info.features = {strings(fs["names"]), parse_duration(std::string_view(fs["sample"])),
                   parse_duration(std::string_view(fs["max_gap"]))};
  info.outputs = strings(doc["outputs"]);
  if (std::uint64_t w; doc["window"].get(w) == simdjson::SUCCESS) info.window = static_cast<std::size_t>(w);
  if (info.window == 0) throw std::runtime_error("model " + path.string() + ": window must be >= 1");
  if (array bs; doc["barriers"].get(bs) == simdjson::SUCCESS) {
    for (object b : bs)
      info.barriers.push_back({double(b["k_tp"]), double(b["k_sl"]), parse_duration(std::string_view(b["horizon"]))});
    info.barrier_vol = std::string_view(doc["barrier_vol"]);
  }
  if (info.features.names.empty() || info.outputs.empty())
    throw std::runtime_error("model " + path.string() + ": features and outputs must be non-empty");

  if (info.type == "linear") {
    LinearParams p{floats(doc["mean"]), floats(doc["std"]), floats(doc["bias"]), {}};
    for (element w : array(doc["weights"])) p.weights.push_back(floats(w));
    return std::make_unique<LinearModel>(std::move(info), std::move(p));
  }
  if (info.type == "onnx") {
#ifdef XROCK_WITH_ONNX
    std::filesystem::path file = path.parent_path() / std::string_view(doc["file"]);
    return make_onnx_model(std::move(info), file);
#else
    throw std::runtime_error("model " + path.string() + ": built without ONNX Runtime (XROCK_WITH_ONNX=OFF)");
#endif
  }
  throw std::runtime_error("model " + path.string() + ": unknown type " + info.type);
}

void save_linear_model(const std::filesystem::path& path, const ModelInfo& info, const LinearParams& p) {
  auto names = [](const std::vector<std::string>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += std::format("{}\"{}\"", i ? ", " : "", v[i]);
    return s + "]";
  };
  std::string barriers;
  for (std::size_t i = 0; i < info.barriers.size(); ++i)
    barriers += std::format("{}{{\"k_tp\": {}, \"k_sl\": {}, \"horizon\": \"{}ns\"}}", i ? ", " : "",
                            info.barriers[i].k_tp, info.barriers[i].k_sl, info.barriers[i].horizon);
  std::string weights = "[";
  for (std::size_t o = 0; o < p.weights.size(); ++o) weights += (o ? ", " : "") + join(p.weights[o]);
  weights += "]";
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path.string());
  f << std::format(
      "{{\"format\": \"xrock-model/1\", \"type\": \"linear\", \"symbol\": \"{}\",\n"
      " \"features\": {{\"names\": {}, \"sample\": \"{}ns\", \"max_gap\": \"{}ns\"}},\n"
      " \"outputs\": {},\n \"window\": {},\n \"barriers\": [{}],\n \"barrier_vol\": \"{}\",\n"
      " \"mean\": {},\n \"std\": {},\n \"weights\": {},\n \"bias\": {}}}\n",
      info.symbol, names(info.features.names), info.features.sample, info.features.max_gap, names(info.outputs),
      info.window, barriers, info.barrier_vol, join(p.mean), join(p.std), weights, join(p.bias));
}

}  // namespace xrock::ml
