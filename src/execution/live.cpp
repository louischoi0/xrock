#include "execution/live.hpp"

#include <chrono>

#include "core/log.hpp"

namespace xrock::execution {

LiveRunner::LiveRunner(const Config& cfg, const Instruments& ins, MarketDataFeed& feed, OrderGateway& gw,
                       Strategy& strat)
    : ins_(ins),
      feed_(feed),
      gw_(gw),
      risk_(engine::RiskLimits::from(cfg)),
      engine_(ins, clock_, gw, risk_, journal_, strat, cfg.num("live.initial_equity"),
              cfg.duration("analytics.equity_interval", kMinute)),
      tick_(cfg.duration("live.tick", 10 * kMilli)),
      out_dir_(cfg.str("analytics.out_dir", "results")) {}

void LiveRunner::push(Msg m) {
  {
    std::lock_guard lk(mu_);
    queue_.push_back(std::move(m));
  }
  cv_.notify_one();
}

void LiveRunner::run(const std::atomic<bool>& stop) {
  gw_.set_listener(*this);
  gw_.start();
  feed_.start(*this);
  engine_.start();
  log::info("live: running");

  std::deque<Msg> batch;
  while (!stop) {
    {
      std::unique_lock lk(mu_);
      cv_.wait_for(lk, std::chrono::nanoseconds(tick_), [&] { return !queue_.empty(); });
      batch.swap(queue_);
    }
    engine_.advance(clock_.now());
    for (const Msg& m : batch)
      std::visit(overloaded{[&](const MarketEvent& e) { engine_.on_market(e); },
                            [&](const OrderUpdate& u) { engine_.on_order_update(u); },
                            [&](const Fill& f) { engine_.on_fill(f); }},
                 m);
    batch.clear();
  }

  log::info("live: stopping");
  feed_.stop();
  for (OrderId id : engine_.open_orders()) engine_.cancel(id);
  engine_.stop();
  gw_.stop();
  auto dir = out_dir_ / ("live-" + format_date(clock_.now()));
  journal_.write_csv(dir, ins_);
  log::info("live: journal written to {}", dir.string());
}

}  // namespace xrock::execution
