#pragma once

#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/instruments.hpp"
#include "data/replay.hpp"
#include "ml/features.hpp"
#include "ml/splits.hpp"

namespace xrock::ml {

// The feature contract shared by dataset export, model files and ModelStrategy: same names, same sampling grid,
// same staleness rule => same numbers.
struct FeatureSpec {
  std::vector<std::string> names;
  Duration sample = kSecond;      // samples at multiples of `sample` since the epoch
  Duration max_gap = 10 * kSecond;  // no sample when the last event is older than this
};

// Triple barrier around mid(t): take-profit at k_tp * sigma_h, stop-loss at k_sl * sigma_h, time barrier at
// t + horizon, with sigma_h = sigma(barrier_vol at t) * sqrt(horizon / d) (d = the vol feature's interval).
struct Barrier {
  double k_tp = 1, k_sl = 1;
  Duration horizon = kMinute;
};
std::vector<Barrier> parse_barriers(const std::vector<std::string>&);  // "k_tp:k_sl:horizon", e.g. "1.5:1:5m"
std::string format_barrier(const Barrier&);

// [dataset]: symbol, features, sample, max_gap, horizons, barriers, barrier_vol, start, end, out.
struct DatasetSpec {
  SymbolId sym = 0;
  FeatureSpec features;
  std::vector<Duration> horizons;  // labels: fwd_<h> = log(mid(t + h) / mid(t)), mid as in MarketState::mid()
  Timestamp start = 0, end = 0;
  // labels tb_<i>_long / tb_<i>_short: the log return of entering at mid(t) and leaving at the first barrier
  // touched (at the barrier level) or at mid(t + horizon). The path is checked per sample interval on its
  // high/low; an interval touching both barriers counts as the stop-loss (pessimistic).
  std::vector<Barrier> barriers;
  std::string barrier_vol;  // a grid-volatility feature in features.names (rv_/ewmavol_/garch_)
  std::vector<std::string> label_names() const;  // fwd_* in horizons order, then tb_<i>_long, tb_<i>_short
  Duration max_horizon() const;
  void validate() const;
  static DatasetSpec from(const Config&, const Instruments&);
};

struct Sample {
  Timestamp ts = 0;
  std::vector<float> x;  // features, FeatureSpec order
  std::vector<float> y;  // labels, horizons order
};

std::string label_name(Duration h);  // "fwd_60s"

// Turns one symbol's event stream into samples. At every grid time t it samples features from events with ts < t
// (before feeding any event with ts >= t); the label for horizon h is resolved by the mid in force just before
// t + h. Samples that are not ready (warm-up, stale data, NaN feature) or whose labels run past the data are dropped.
class DatasetBuilder {
 public:
  using Sink = std::function<void(const Sample&)>;
  DatasetBuilder(DatasetSpec, Sink);
  void on_event(const MarketEvent&);  // all symbols; others are ignored
  void finish();                      // drops samples whose labels cannot be resolved

  std::uint64_t emitted() const { return emitted_; }
  std::uint64_t skipped() const { return skipped_; }

 private:
  void sample_until(Timestamp ts);
  void resolve_until(Timestamp ts);
  void barrier_label(std::size_t job, std::size_t pending_index);

  struct Job {  // one label resolution: a fwd horizon or a barrier config
    Duration horizon;
    std::size_t barrier;  // index into spec_.barriers, or npos for fwd
    std::size_t column;   // first y column
  };
  struct Cell {  // mid path over one sample interval
    double high = 0, low = 0, close = 0;
  };

  DatasetSpec spec_;
  Sink sink_;
  FeatureSet fs_;
  Timestamp next_ = kNever;
  std::deque<Sample> pending_;
  std::deque<double> pending_mid_;  // mid(t) per pending sample, kept in double for the label ratio
  std::vector<Job> jobs_;
  std::size_t n_labels_ = 0;
  std::vector<std::size_t> cursor_;  // per job: first pending index whose label is unresolved
  std::size_t vol_col_ = 0;          // barrier_vol's column in x
  Duration vol_d_ = 0;
  std::deque<Cell> cells_;           // closed intervals; cells_[i] covers [cells_base_ + i*sample, +sample)
  Timestamp cells_base_ = 0;
  Cell open_{};                      // the interval in progress
  std::uint64_t emitted_ = 0, skipped_ = 0;
};

// NumPy .npy writer for 2-D little-endian arrays of unknown length: the header is rewritten with the final row count
// on close (fixed-width shape field).
class NpyWriter {
 public:
  NpyWriter(const std::filesystem::path&, const char* dtype, std::size_t cols);  // dtype "<f4" or "<i8"
  ~NpyWriter();
  void write(const void* row);  // cols * itemsize bytes
  void close();

 private:
  void header(std::uint64_t rows);
  std::FILE* f_ = nullptr;
  std::string dtype_;
  std::size_t cols_, itemsize_;
  std::uint64_t rows_ = 0;
};

// Writes {dir}/X.npy (float32 [N, F]), Y.npy (float32 [N, L]), ts.npy (int64 [N, 1]) and meta.json
// (format "xrock-dataset/1": feature spec, label names, symbol, range, folds). Returns the sample count.
std::uint64_t export_dataset(const DatasetSpec&, const Instruments&, data::EventSource&, const Splits&,
                             const std::filesystem::path& dir);

}  // namespace xrock::ml
