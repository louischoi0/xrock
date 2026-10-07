#include <gtest/gtest.h>

#include "backtest/backtest.hpp"
#include "backtest/sim_gateway.hpp"
#include "exchange/synthetic.hpp"

using namespace xrock;

namespace {

struct Recorder : ExecListener {
  std::vector<Fill> fills;
  std::vector<OrderUpdate> updates;
  void on_order_update(const OrderUpdate& u) override { updates.push_back(u); }
  void on_fill(const Fill& f) override { fills.push_back(f); }
};

Instruments btc() {
  Instruments ins;
  ins.add({.symbol = "BTCUSDT", .tick_size = 0.1, .lot_size = 0.001, .min_qty = 0.001, .maker_fee = 0.0002,
           .taker_fee = 0.0005, .maintenance_margin = 0.005});
  return ins;
}

Order order(OrderId id, Side side, OrderType type, double qty, double price = 0) {
  Order o;
  o.id = id;
  o.side = side;
  o.type = type;
  o.qty = qty;
  o.price = price;
  return o;
}

}  // namespace

TEST(SimGateway, MarketOrderFillsAtTouchAfterLatency) {
  Instruments ins = btc();
  SimClock clock;
  Recorder rec;
  backtest::SimGateway sim(ins, clock, {.latency = 10 * kMilli});
  sim.set_listener(rec);
  Book b{.ts = 0, .sym = 0};
  b.bids[0] = {99, 1};
  b.asks[0] = {101, 1};
  sim.on_market(b);
  sim.submit(order(1, Side::Buy, OrderType::Market, 2));
  sim.on_market(Trade{.ts = 5 * kMilli, .sym = 0, .price = 100});
  EXPECT_TRUE(rec.fills.empty());
  sim.on_market(Trade{.ts = 20 * kMilli, .sym = 0, .price = 100});
  ASSERT_EQ(rec.fills.size(), 1u);
  EXPECT_DOUBLE_EQ(rec.fills[0].price, 101);
  EXPECT_EQ(rec.fills[0].ts, 10 * kMilli);
  EXPECT_FALSE(rec.fills[0].maker);
  EXPECT_DOUBLE_EQ(rec.fills[0].fee, 101 * 2 * 0.0005);
  EXPECT_EQ(rec.updates.back().status, OrderStatus::Filled);
}

TEST(SimGateway, RestingLimitFillsOnlyWhenTradedThrough) {
  Instruments ins = btc();
  SimClock clock;
  Recorder rec;
  backtest::SimGateway sim(ins, clock, {.latency = 0});
  sim.set_listener(rec);
  Book b{.ts = 0, .sym = 0};
  b.bids[0] = {99, 1};
  b.asks[0] = {101, 1};
  sim.on_market(b);
  sim.submit(order(2, Side::Buy, OrderType::Limit, 1, 99.5));
  sim.on_market(Trade{.ts = 1, .sym = 0, .price = 99.5});
  EXPECT_EQ(rec.updates.back().status, OrderStatus::New);
  EXPECT_TRUE(rec.fills.empty());
  sim.on_market(Trade{.ts = 2, .sym = 0, .price = 99.4});
  ASSERT_EQ(rec.fills.size(), 1u);
  EXPECT_DOUBLE_EQ(rec.fills[0].price, 99.5);
  EXPECT_TRUE(rec.fills[0].maker);
}

TEST(SimGateway, IsolatedLiquidation) {
  Instruments ins = btc();
  SimClock clock;
  Recorder rec;
  backtest::SimGateway sim(ins, clock, {.latency = 0, .leverage = 10});
  sim.set_listener(rec);
  sim.on_market(Trade{.ts = 0, .sym = 0, .price = 100});
  sim.submit(order(1, Side::Buy, OrderType::Market, 1));
  sim.on_market(Trade{.ts = 1, .sym = 0, .price = 100});  // fills @100; liq = 100 * (1 - 0.1 + 0.005) = 90.5
  sim.on_market(Trade{.ts = 2, .sym = 0, .price = 91});
  ASSERT_EQ(rec.fills.size(), 1u);
  sim.on_market(Trade{.ts = 3, .sym = 0, .price = 90});
  ASSERT_EQ(rec.fills.size(), 2u);
  EXPECT_TRUE(rec.fills[1].liquidation);
  EXPECT_EQ(rec.fills[1].side, Side::Sell);
  EXPECT_NEAR(rec.fills[1].price, 90.5, 1e-9);
}

TEST(Backtest, MaCrossOnSyntheticData) {
  Config cfg = Config::parse(R"(
[market]
symbols = BTCUSDT
[synthetic]
start = 2026-01-01
end = 2026-01-02
trades_per_sec = 0.2
seed = 7
[strategy.ma_cross]
symbol = BTCUSDT
interval = 5m
fast = 3
slow = 8
qty = 0.01
[backtest]
initial_equity = 10000
)");
  Instruments ins = Instruments::from_config(cfg);
  struct Sink : MarketListener {
    std::vector<MarketEvent> events;
    void on_market(const MarketEvent& e) override { events.push_back(e); }
  } sink;
  exchange::SyntheticFeed(cfg, ins).generate(sink);
  ASSERT_GT(sink.events.size(), 10000u);

  auto strat = make_strategy("ma_cross", cfg.section("strategy.ma_cross"), ins);
  data::VectorSource src(sink.events);
  auto r = backtest::run(cfg, ins, *strat, src);
  EXPECT_GT(r.metrics.fills, 2u);
  EXPECT_GT(r.journal.equity.size(), 1000u);  // 1m samples over a day
  EXPECT_NE(r.metrics.final_equity, r.metrics.initial_equity);
  EXPECT_DOUBLE_EQ(r.metrics.initial_equity, 10000);
}
