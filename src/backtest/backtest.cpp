#include "backtest/backtest.hpp"

#include <algorithm>

#include "backtest/sim_gateway.hpp"
#include "core/log.hpp"
#include "engine/engine.hpp"

namespace xrock::backtest {

std::unique_ptr<data::EventSource> open_replay(const Config& cfg, const Instruments& ins, Timestamp start,
                                               Timestamp end) {
  std::vector<data::Stream> streams;
  for (const auto& s : cfg.list("market.streams")) streams.push_back(data::parse_stream(s));
  return std::make_unique<data::ReplayFeed>(data::make_store(cfg, ins), ins, streams, start, end);
}

std::unique_ptr<data::EventSource> open_replay(const Config& cfg, const Instruments& ins) {
  return open_replay(cfg, ins, cfg.date("backtest.start"), cfg.date("backtest.end"));
}

Result run(const Config& cfg, const Instruments& ins, Strategy& strat, data::EventSource& src) {
  SimClock clock;
  SimGateway sim(ins, clock, SimParams::from(cfg));
  engine::RiskManager risk(engine::RiskLimits::from(cfg));
  Result r;
  engine::Engine eng(ins, clock, sim, risk, r.journal, strat, cfg.num("backtest.initial_equity", 10000),
                     cfg.duration("analytics.equity_interval", kMinute));
  sim.set_listener(eng);

  const MarketEvent* ev = src.next();
  if (!ev) {
    log::warn("backtest: no market data in range");
    return r;
  }
  clock.set(event_ts(*ev));
  eng.start();
  for (; ev; ev = src.next()) {
    Timestamp ts = event_ts(*ev);
    for (Timestamp d; (d = eng.next_deadline()) <= ts;) {
      clock.set(std::max(d, clock.now()));
      eng.advance(d);
    }
    clock.set(ts);
    sim.on_market(*ev);
    eng.on_market(*ev);
  }
  eng.stop();
  r.metrics = analytics::compute_metrics(r.journal);
  return r;
}

}  // namespace xrock::backtest
