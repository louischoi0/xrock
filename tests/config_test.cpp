#include <gtest/gtest.h>

#include <cstdlib>

#include "core/config.hpp"

using namespace xrock;

TEST(Config, SectionsCommentsEnv) {
  setenv("XROCK_TEST_SECRET", "s3cret", 1);
  auto c = Config::parse(R"(
top = 1
[exchange]
name    = bybit   # inline comment
api_key = ${XROCK_TEST_SECRET}
; full-line comment
[strategy.ma_cross]
interval = 5m
fast     = 5
[market]
symbols = BTCUSDT, ETHUSDT
)");
  EXPECT_EQ(c.str("top"), "1");
  EXPECT_EQ(c.str("exchange.name"), "bybit");
  EXPECT_EQ(c.str("exchange.api_key"), "s3cret");
  Config s = c.section("strategy.ma_cross");
  EXPECT_EQ(s.duration("interval", 0), 5 * kMinute);
  EXPECT_EQ(s.num("fast"), 5);
  EXPECT_EQ(c.list("market.symbols"), (std::vector<std::string>{"BTCUSDT", "ETHUSDT"}));
  EXPECT_THROW(s.str("symbol"), std::runtime_error);
}

TEST(Config, DurationAndDate) {
  EXPECT_EQ(parse_duration("250ms"), 250 * kMilli);
  EXPECT_EQ(parse_duration("1.5s"), 1500 * kMilli);
  EXPECT_THROW(parse_duration("5x"), std::runtime_error);
  EXPECT_EQ(parse_date("1970-01-02"), kDay);
  EXPECT_EQ(format_date(parse_date("2026-10-06") + 5 * kHour), "2026-10-06");
  EXPECT_THROW(parse_date("2026-13-01"), std::runtime_error);
}
