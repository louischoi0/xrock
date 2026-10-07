#pragma once

namespace xrock::ml {

// Registers the "model" strategy with make_strategy(). Call once at startup.
//
// [strategy.model]
//   model  = models/x.json   xrock-model/1 file; its feature contract builds the FeatureSet
//   symbol = BTCUSDT         default: the model's symbol
//   output = fwd_60s         which model output drives decisions (default: the first)
//   trade  = false           false = shadow: predict and journal only (predictions.csv)
//   enter  = 0.0005          |prediction| above this -> target +-qty
//   exit   = 0.0001          |prediction| below this -> flat; in between -> keep
//   qty    = 0.01
// Samples on the model's grid (timers at multiples of `sample`), from events with ts < t: the same values the
// dataset export produced for those timestamps. A model with "window": T predicts once the last T grid samples
// are all valid and consecutive (ml::SampleWindow). Each order is journaled as a "rebalance" decision (reason
// above-enter / below-enter / inside-exit; prediction, position, target), or "rejected" if risk refused it.
void register_strategies();

}  // namespace xrock::ml
