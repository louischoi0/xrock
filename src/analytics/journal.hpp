#pragma once

#include <filesystem>
#include <vector>

#include "core/events.hpp"
#include "core/instruments.hpp"

namespace xrock::analytics {

struct FillRecord {
  Fill fill;
  double realized = 0;
};

struct FundingRecord {
  Funding funding;
  double payment = 0;
};

struct EquityPoint {
  Timestamp ts = 0;
  double equity = 0;
};

struct PredictionRecord {
  Timestamp ts = 0;
  SymbolId sym = 0;
  std::vector<float> values;  // model outputs, in the model's output order
};

struct DecisionRecord {
  Timestamp ts = 0;
  Decision decision;
};

// Everything one run (backtest or live) did; the input of every analysis.
struct Journal {
  std::vector<Order> orders;
  std::vector<OrderUpdate> updates;
  std::vector<FillRecord> fills;
  std::vector<FundingRecord> funding;
  std::vector<EquityPoint> equity;
  std::vector<PredictionRecord> predictions;
  std::vector<DecisionRecord> decisions;

  // orders.csv, updates.csv, fills.csv, funding.csv, equity.csv, decisions.csv and predictions.csv (when any).
  // decisions.csv: ts,symbol,kind,reason,order_id,values with values = "name=value;name=value".
  void write_csv(const std::filesystem::path& dir, const Instruments&) const;
  // Reads back fills, funding, equity (what compute_metrics needs) and decisions.
  static Journal read_csv(const std::filesystem::path& dir, const Instruments&);
};

}  // namespace xrock::analytics
