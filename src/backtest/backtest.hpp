#pragma once

#include <memory>

#include "analytics/journal.hpp"
#include "analytics/metrics.hpp"
#include "data/replay.hpp"
#include "strategy/strategy.hpp"

namespace xrock::backtest {

struct Result {
  analytics::Journal journal;
  analytics::Metrics metrics;
};

// Replay of [market] streams over [backtest] start..end from the [data] backend.
std::unique_ptr<data::EventSource> open_replay(const Config&, const Instruments&);
// Same, over an explicit range.
std::unique_ptr<data::EventSource> open_replay(const Config&, const Instruments&, Timestamp start, Timestamp end);

// One strategy over one event source. Engine and SimGateway see the same events; timers, bar closes and equity
// samples fire at their own timestamps between events.
Result run(const Config&, const Instruments&, Strategy&, data::EventSource&);

}  // namespace xrock::backtest
