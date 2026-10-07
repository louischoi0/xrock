#pragma once

#include <span>
#include <string>
#include <vector>

namespace xrock::ml {

// Prediction quality before any trading: does the score rank the forward return?
struct PredictionMetrics {
  std::size_t n = 0;
  double ic = 0;        // Pearson correlation(pred, label)
  double rank_ic = 0;   // Spearman correlation
  double hit_rate = 0;  // sign agreement where both are non-zero
  std::vector<double> bucket_mean;  // mean label per prediction quantile, lowest to highest
};

PredictionMetrics evaluate_predictions(std::span<const float> pred, std::span<const float> label, int buckets = 5);
std::string to_string(const PredictionMetrics&);  // one line, label means in bps

}  // namespace xrock::ml
