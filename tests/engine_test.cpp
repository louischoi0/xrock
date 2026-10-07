#include <gtest/gtest.h>

#include "engine/portfolio.hpp"
#include "engine/risk.hpp"

using namespace xrock;

namespace {
Fill fill(Side s, double px, double q) { return {.sym = 0, .side = s, .price = px, .qty = q}; }
}  // namespace

TEST(Portfolio, OpenAddReduceFlip) {
  engine::Portfolio pf(1000, 1);
  pf.apply_fill(fill(Side::Buy, 100, 1));
  pf.apply_fill(fill(Side::Buy, 110, 1));
  EXPECT_DOUBLE_EQ(pf.position(0).avg_price, 105);
  EXPECT_DOUBLE_EQ(pf.apply_fill(fill(Side::Sell, 120, 1)), 15);
  EXPECT_DOUBLE_EQ(pf.apply_fill(fill(Side::Sell, 100, 3)), -5);  // closes 1 @105, opens short 2 @100
  EXPECT_DOUBLE_EQ(pf.position(0).qty, -2);
  EXPECT_DOUBLE_EQ(pf.position(0).avg_price, 100);
  pf.mark(0, 90);
  EXPECT_DOUBLE_EQ(pf.equity(), 1000 + 15 - 5 + 20);
}

TEST(Portfolio, FundingLongPays) {
  engine::Portfolio pf(1000, 1);
  pf.apply_fill(fill(Side::Buy, 100, 2));
  EXPECT_DOUBLE_EQ(pf.apply_funding({.sym = 0, .rate = 0.001}), 0.2);
  EXPECT_DOUBLE_EQ(pf.cash(), 999.8);
}

TEST(Risk, Limits) {
  engine::RiskManager risk({.max_order_notional = 500, .max_orders_per_sec = 2});
  engine::Portfolio pf(1000, 1);
  OrderRequest r{.sym = 0, .side = Side::Buy, .qty = 1};
  EXPECT_EQ(risk.check(r, 100, pf, 0), nullptr);
  EXPECT_EQ(risk.check(r, 100, pf, 1), nullptr);
  EXPECT_STREQ(risk.check(r, 100, pf, 2), "order rate limit");
  EXPECT_STREQ(risk.check(r, 600, pf, 2 * kSecond), "order notional limit");
}

TEST(Risk, DailyLossHaltsNewRiskButAllowsReducing) {
  engine::RiskManager risk({.max_daily_loss = 50});
  engine::Portfolio pf(1000, 1);
  EXPECT_EQ(risk.check({.sym = 0, .side = Side::Buy, .qty = 1}, 100, pf, 0), nullptr);
  pf.apply_fill(fill(Side::Buy, 100, 1));
  pf.mark(0, 40);  // -60
  EXPECT_STREQ(risk.check({.sym = 0, .side = Side::Buy, .qty = 1}, 40, pf, 1), "daily loss limit reached");
  EXPECT_EQ(risk.check({.sym = 0, .side = Side::Sell, .qty = 1}, 40, pf, 2), nullptr);
}
