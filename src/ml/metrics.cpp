#include "ml/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>

namespace xrock::ml {
namespace {

double pearson(std::span<const double> a, std::span<const double> b) {
  double n = static_cast<double>(a.size());
  double ma = std::accumulate(a.begin(), a.end(), 0.0) / n, mb = std::accumulate(b.begin(), b.end(), 0.0) / n;
  double cov = 0, va = 0, vb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    cov += (a[i] - ma) * (b[i] - mb);
    va += (a[i] - ma) * (a[i] - ma);
    vb += (b[i] - mb) * (b[i] - mb);
  }
  return va > 0 && vb > 0 ? cov / std::sqrt(va * vb) : 0.0;
}

std::vector<double> ranks(std::span<const double> v) {  // average ranks for ties
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](auto a, auto b) { return v[a] < v[b]; });
  std::vector<double> r(v.size());
  for (std::size_t i = 0; i < idx.size();) {
    std::size_t j = i;
    while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]]) ++j;
    for (std::size_t k = i; k <= j; ++k) r[idx[k]] = (i + j) / 2.0;
    i = j + 1;
  }
  return r;
}

}  // namespace

PredictionMetrics evaluate_predictions(std::span<const float> pred, std::span<const float> label, int buckets) {
  if (pred.size() != label.size()) throw std::invalid_argument("evaluate_predictions: size mismatch");
  PredictionMetrics m;
  m.n = pred.size();
  if (m.n < 2) return m;
  std::vector<double> p(pred.begin(), pred.end()), y(label.begin(), label.end());
  m.ic = pearson(p, y);
  m.rank_ic = pearson(ranks(p), ranks(y));
  std::size_t agree = 0, counted = 0;
  for (std::size_t i = 0; i < m.n; ++i)
    if (p[i] != 0 && y[i] != 0) {
      ++counted;
      agree += (p[i] > 0) == (y[i] > 0);
    }
  m.hit_rate = counted ? double(agree) / counted : 0;
  std::vector<std::size_t> idx(m.n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](auto a, auto b) { return p[a] < p[b]; });
  for (int b = 0; b < buckets; ++b) {
    std::size_t lo = m.n * b / buckets, hi = m.n * (b + 1) / buckets;
    double s = 0;
    for (std::size_t i = lo; i < hi; ++i) s += y[idx[i]];
    m.bucket_mean.push_back(hi > lo ? s / (hi - lo) : 0);
  }
  return m;
}

std::string to_string(const PredictionMetrics& m) {
  std::string buckets;
  for (double b : m.bucket_mean) buckets += std::format("{}{:+.2f}", buckets.empty() ? "" : " ", b * 1e4);
  return std::format("n={} ic={:+.4f} rank_ic={:+.4f} hit={:.1f}% buckets(bps)=[{}]", m.n, m.ic, m.rank_ic,
                     100 * m.hit_rate, buckets);
}

}  // namespace xrock::ml
