#include "ml/dataset.hpp"

#include <cmath>
#include <format>
#include <fstream>
#include <stdexcept>

#include "core/log.hpp"

namespace xrock::ml {
namespace {

std::string json_str_list(const std::vector<std::string>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) out += std::format("{}\"{}\"", i ? ", " : "", v[i]);
  return out + "]";
}

std::string duration_str(Duration d) {
  if (d % kDay == 0) return std::format("{}d", d / kDay);
  if (d % kHour == 0) return std::format("{}h", d / kHour);
  if (d % kMinute == 0) return std::format("{}m", d / kMinute);
  if (d % kSecond == 0) return std::format("{}s", d / kSecond);
  return std::format("{}ms", d / kMilli);
}

}  // namespace

std::string label_name(Duration h) { return "fwd_" + duration_str(h); }

std::vector<Barrier> parse_barriers(const std::vector<std::string>& items) {
  std::vector<Barrier> out;
  for (const auto& item : items) {
    auto p = split(item, ':');
    if (p.size() != 3) throw std::runtime_error("barrier '" + item + "': expected k_tp:k_sl:horizon");
    Barrier b{std::stod(p[0]), std::stod(p[1]), parse_duration(p[2])};
    if (b.k_tp <= 0 || b.k_sl <= 0 || b.horizon <= 0) throw std::runtime_error("barrier '" + item + "': must be > 0");
    out.push_back(b);
  }
  return out;
}

std::string format_barrier(const Barrier& b) { return std::format("{}:{}:{}", b.k_tp, b.k_sl, duration_str(b.horizon)); }

std::vector<std::string> DatasetSpec::label_names() const {
  std::vector<std::string> v;
  for (Duration h : horizons) v.push_back(label_name(h));
  for (std::size_t i = 0; i < barriers.size(); ++i) {
    v.push_back(std::format("tb_{}_long", i));
    v.push_back(std::format("tb_{}_short", i));
  }
  return v;
}

Duration DatasetSpec::max_horizon() const {
  Duration m = 0;
  for (Duration h : horizons) m = std::max(m, h);
  for (const auto& b : barriers) m = std::max(m, b.horizon);
  return m;
}

void DatasetSpec::validate() const {
  if (features.names.empty() || (horizons.empty() && barriers.empty()))
    throw std::runtime_error("dataset: features and at least one of horizons / barriers required");
  if (features.sample <= 0) throw std::runtime_error("dataset: sample must be > 0");
  if (barriers.empty()) return;
  auto it = std::find(features.names.begin(), features.names.end(), barrier_vol);
  if (it == features.names.end() || vol_interval(barrier_vol) <= 0)
    throw std::runtime_error("dataset: barrier_vol must be a grid-volatility feature listed in features");
  for (const auto& b : barriers)
    if (b.horizon % features.sample != 0)
      throw std::runtime_error("dataset: barrier horizons must be multiples of sample (" + format_barrier(b) + ")");
}

DatasetSpec DatasetSpec::from(const Config& cfg, const Instruments& ins) {
  Config c = cfg.section("dataset");
  DatasetSpec s;
  s.sym = ins.id(c.str("symbol"));
  s.features = {c.list("features"), c.duration("sample", kSecond), c.duration("max_gap", 10 * kSecond)};
  for (const auto& h : c.list("horizons")) s.horizons.push_back(parse_duration(h));
  s.barriers = parse_barriers(c.list("barriers"));
  s.barrier_vol = c.str("barrier_vol", "");
  s.start = c.date("start");
  s.end = c.date("end");
  s.validate();
  return s;
}

DatasetBuilder::DatasetBuilder(DatasetSpec spec, Sink sink)
    : spec_(std::move(spec)), sink_(std::move(sink)), fs_(spec_.features.names, spec_.features.max_gap) {
  spec_.validate();
  std::size_t col = 0;
  for (Duration h : spec_.horizons) jobs_.push_back({h, std::string::npos, col++});
  for (std::size_t i = 0; i < spec_.barriers.size(); ++i, col += 2) jobs_.push_back({spec_.barriers[i].horizon, i, col});
  n_labels_ = col;
  cursor_.assign(jobs_.size(), 0);
  if (!spec_.barriers.empty()) {
    const auto& n = spec_.features.names;
    vol_col_ = static_cast<std::size_t>(std::find(n.begin(), n.end(), spec_.barrier_vol) - n.begin());
    vol_d_ = vol_interval(spec_.barrier_vol);
  }
}

void DatasetBuilder::on_event(const MarketEvent& e) {
  if (event_sym(e) != spec_.sym) return;
  Timestamp ts = event_ts(e);
  if (next_ == kNever) {
    next_ = (ts / spec_.features.sample + 1) * spec_.features.sample;
    cells_base_ = next_;
  }
  sample_until(ts);   // samples at grid times <= ts see only events before them
  resolve_until(ts);  // labels whose t + h <= ts read the mid in force before this event
  fs_.on_event(e);
  if (double m = fs_.state().mid(); m > 0) {
    if (open_.close <= 0) open_ = {m, m, m};
    open_ = {std::max(open_.high, m), std::min(open_.low, m), m};
  }
}

void DatasetBuilder::sample_until(Timestamp ts) {
  for (; next_ <= ts; next_ += spec_.features.sample) {
    if (!spec_.barriers.empty() && next_ > cells_base_) {  // close the interval ending at next_
      cells_.push_back(open_);
      double m = fs_.state().mid();
      open_ = {m, m, m};
    }
    if (next_ < spec_.start || next_ >= spec_.end) continue;
    if (!fs_.ready(next_)) {
      ++skipped_;
      continue;
    }
    Sample s{next_, std::vector<float>(fs_.size()), std::vector<float>(n_labels_, NAN)};
    fs_.compute(next_, s.x);
    bool finite = true;
    for (float v : s.x) finite = finite && std::isfinite(v);
    if (!finite) {
      ++skipped_;
      continue;
    }
    pending_.push_back(std::move(s));
    pending_mid_.push_back(fs_.state().mid());
  }
}

void DatasetBuilder::barrier_label(std::size_t j, std::size_t i) {
  const Job& job = jobs_[j];
  const Barrier& b = spec_.barriers[job.barrier];
  Sample& s = pending_[i];
  double p0 = pending_mid_[i];
  double sigma_h = s.x[vol_col_] * std::sqrt(static_cast<double>(b.horizon) / static_cast<double>(vol_d_));
  double tp = b.k_tp * sigma_h, sl = b.k_sl * sigma_h;
  std::size_t first = static_cast<std::size_t>((s.ts - cells_base_) / spec_.features.sample);
  std::size_t n = static_cast<std::size_t>(b.horizon / spec_.features.sample);
  double up_tp = p0 * std::exp(tp), dn_sl = p0 * std::exp(-sl);  // long
  double dn_tp = p0 * std::exp(-tp), up_sl = p0 * std::exp(sl);   // short
  float lr = NAN, sr = NAN;
  for (std::size_t k = first; k < first + n && (std::isnan(lr) || std::isnan(sr)); ++k) {
    const Cell& c = cells_[k];
    if (std::isnan(lr)) {
      if (c.low <= dn_sl) lr = static_cast<float>(-sl);
      else if (c.high >= up_tp) lr = static_cast<float>(tp);
    }
    if (std::isnan(sr)) {
      if (c.high >= up_sl) sr = static_cast<float>(-sl);
      else if (c.low <= dn_tp) sr = static_cast<float>(tp);
    }
  }
  double close = cells_[first + n - 1].close;
  if (std::isnan(lr)) lr = static_cast<float>(std::log(close / p0));
  if (std::isnan(sr)) sr = static_cast<float>(-std::log(close / p0));
  s.y[job.column] = lr;
  s.y[job.column + 1] = sr;
}

void DatasetBuilder::resolve_until(Timestamp ts) {
  const MarketState& st = fs_.state();
  for (std::size_t j = 0; j < jobs_.size(); ++j)
    for (; cursor_[j] < pending_.size() && pending_[cursor_[j]].ts + jobs_[j].horizon <= ts; ++cursor_[j]) {
      std::size_t i = cursor_[j];
      Sample& s = pending_[i];
      if (s.ts + jobs_[j].horizon - st.last_ts > spec_.features.max_gap) continue;  // stale: stays NaN, dropped
      if (jobs_[j].barrier == std::string::npos)
        s.y[jobs_[j].column] = static_cast<float>(std::log(st.mid() / pending_mid_[i]));
      else
        barrier_label(j, i);
    }
  std::size_t done = cursor_.empty() ? 0 : *std::min_element(cursor_.begin(), cursor_.end());
  for (std::size_t i = 0; i < done; ++i) {
    Sample& s = pending_.front();
    bool finite = true;
    for (float v : s.y) finite = finite && std::isfinite(v);
    if (finite) {
      sink_(s);
      ++emitted_;
    } else {
      ++skipped_;
    }
    pending_.pop_front();
    pending_mid_.pop_front();
  }
  for (auto& c : cursor_) c -= done;
  // Path cells are needed from the oldest pending sample on.
  Timestamp keep = pending_.empty() ? next_ : pending_.front().ts;
  while (!cells_.empty() && cells_base_ < keep) {
    cells_.pop_front();
    cells_base_ += spec_.features.sample;
  }
}

void DatasetBuilder::finish() {
  skipped_ += pending_.size();
  pending_.clear();
  pending_mid_.clear();
  std::fill(cursor_.begin(), cursor_.end(), 0);
}

NpyWriter::NpyWriter(const std::filesystem::path& p, const char* dtype, std::size_t cols)
    : dtype_(dtype), cols_(cols), itemsize_(dtype_ == "<f4" ? 4 : 8) {
  if (dtype_ != "<f4" && dtype_ != "<i8") throw std::invalid_argument("npy: dtype must be <f4 or <i8");
  f_ = std::fopen(p.c_str(), "wb");
  if (!f_) throw std::runtime_error("npy: cannot write " + p.string());
  header(0);
}

NpyWriter::~NpyWriter() { close(); }

void NpyWriter::header(std::uint64_t rows) {
  constexpr std::size_t kTotal = 128;  // magic + version + len + dict, padded; same width for any row count
  std::string shape = cols_ ? std::format("({}, {})", rows, cols_) : std::format("({},)", rows);
  std::string dict = std::format("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", dtype_, shape);
  dict.resize(kTotal - 10 - 1, ' ');
  dict += '\n';
  std::string h = "\x93NUMPY";
  h += '\x01';
  h += '\x00';
  std::uint16_t len = static_cast<std::uint16_t>(dict.size());
  h.append(reinterpret_cast<const char*>(&len), 2);
  h += dict;
  std::fseek(f_, 0, SEEK_SET);
  std::fwrite(h.data(), 1, h.size(), f_);
  std::fseek(f_, 0, SEEK_END);
}

void NpyWriter::write(const void* row) {
  std::size_t n = (cols_ ? cols_ : 1) * itemsize_;
  if (std::fwrite(row, 1, n, f_) != n) throw std::runtime_error("npy: write failed");
  ++rows_;
}

void NpyWriter::close() {
  if (!f_) return;
  header(rows_);
  std::fclose(f_);
  f_ = nullptr;
}

std::uint64_t export_dataset(const DatasetSpec& spec, const Instruments& ins, data::EventSource& src,
                             const Splits& splits, const std::filesystem::path& dir) {
  std::filesystem::create_directories(dir);
  NpyWriter X(dir / "X.npy", "<f4", spec.features.names.size());
  NpyWriter Y(dir / "Y.npy", "<f4", spec.label_names().size());
  NpyWriter T(dir / "ts.npy", "<i8", 0);
  std::vector<std::uint64_t> fold_train(splits.folds.size()), fold_test(splits.folds.size());
  std::uint64_t holdout = 0;
  DatasetBuilder b(spec, [&](const Sample& s) {
    X.write(s.x.data());
    Y.write(s.y.data());
    T.write(&s.ts);
    for (std::size_t i = 0; i < splits.folds.size(); ++i) {
      const Fold& f = splits.folds[i];
      fold_train[i] += s.ts >= f.train_start && s.ts < f.train_end;
      fold_test[i] += s.ts >= f.test_start && s.ts < f.test_end;
    }
    holdout += s.ts >= splits.holdout_start;
  });
  while (const MarketEvent* e = src.next()) b.on_event(*e);
  b.finish();
  X.close();
  Y.close();
  T.close();

  std::vector<std::string> labels = spec.label_names();
  std::string barriers;
  for (std::size_t i = 0; i < spec.barriers.size(); ++i)
    barriers += std::format(R"({}{{"k_tp": {}, "k_sl": {}, "horizon": "{}"}})", i ? ", " : "",
                            spec.barriers[i].k_tp, spec.barriers[i].k_sl, duration_str(spec.barriers[i].horizon));
  std::string folds;
  for (std::size_t i = 0; i < splits.folds.size(); ++i) {
    const Fold& f = splits.folds[i];
    folds += std::format(
        "{}\n    {{\"train\": [{}, {}], \"test\": [{}, {}], \"train_samples\": {}, \"test_samples\": {}}}",
        i ? "," : "", f.train_start, f.train_end, f.test_start, f.test_end, fold_train[i], fold_test[i]);
  }
  std::ofstream meta(dir / "meta.json");
  meta << std::format(
      "{{\n  \"format\": \"xrock-dataset/1\",\n  \"symbol\": \"{}\",\n  \"start\": {},\n  \"end\": {},\n"
      "  \"features\": {{\"names\": {}, \"sample\": \"{}\", \"max_gap\": \"{}\"}},\n"
      "  \"labels\": {},\n  \"label_definition\": \"log(mid(t + h) / mid(t)); mid = book mid, else last trade; "
      "features read events with ts < t\",\n"
      "  \"barriers\": [{}],\n  \"barrier_vol\": \"{}\",\n"
      "  \"barrier_definition\": \"tb_<i>_long/short: log return entering at mid(t), leaving at the first barrier "
      "(k_tp / k_sl * sigma_h, sigma_h = barrier_vol(t) * sqrt(h / d)) at its level or at mid(t + h); per-interval "
      "high/low, both touched = stop-loss\",\n"
      "  \"samples\": {},\n  \"skipped\": {},\n  \"holdout_start\": {},\n  \"holdout_samples\": {},\n"
      "  \"folds\": [{}\n  ]\n}}\n",
      ins[spec.sym].symbol, spec.start, spec.end, json_str_list(spec.features.names),
      duration_str(spec.features.sample), duration_str(spec.features.max_gap), json_str_list(labels), barriers,
      spec.barrier_vol, b.emitted(),
      b.skipped(), splits.holdout_start == kNever ? "null" : std::to_string(splits.holdout_start), holdout, folds);
  log::info("dataset: {} samples ({} skipped) -> {}", b.emitted(), b.skipped(), dir.string());
  return b.emitted();
}

}  // namespace xrock::ml
