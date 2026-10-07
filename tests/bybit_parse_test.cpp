#include <gtest/gtest.h>

#include "exchange/bybit/bybit.hpp"

using namespace xrock;
namespace bybit = xrock::exchange::bybit;

namespace {

struct Sink : MarketListener {
  std::vector<MarketEvent> events;
  void on_market(const MarketEvent& e) override { events.push_back(e); }
};

Instruments two() {
  Instruments ins;
  ins.add({.symbol = "BTCUSDT"});
  ins.add({.symbol = "ETHUSDT"});
  return ins;
}

}  // namespace

TEST(BybitHistory, TradeArchiveByHeaderName) {
  auto trades = bybit::parse_trade_archive(
      "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,RPI\n"
      "1790812800.015,BTCUSDT,Buy,0.042,83573.80,ZeroPlusTick,a,1,0.042,3510.1,0\n"
      "1790812800.1031,BTCUSDT,Sell,0.001,83573.70,MinusTick,b,1,0.001,83.5,0\n",
      1);
  ASSERT_EQ(trades.size(), 2u);
  EXPECT_EQ(trades[0].ts, 1790812800LL * kSecond + 15 * kMilli);
  EXPECT_EQ(trades[1].ts, 1790812800LL * kSecond + 103'100'000);
  EXPECT_EQ(trades[1].aggressor, Side::Sell);
  EXPECT_DOUBLE_EQ(trades[0].price, 83573.8);
  EXPECT_DOUBLE_EQ(trades[0].qty, 0.042);
  EXPECT_EQ(trades[0].sym, 1);
}

TEST(BybitHistory, FundingHistoryOldestFirst) {
  auto f = bybit::parse_funding_history(
      R"({"retCode":0,"retMsg":"OK","result":{"category":"linear","list":[
        {"symbol":"BTCUSDT","fundingRate":"0.00001177","fundingRateTimestamp":"1791244800000"},
        {"symbol":"BTCUSDT","fundingRate":"-0.00003208","fundingRateTimestamp":"1791216000000"}]}})",
      0);
  ASSERT_EQ(f.size(), 2u);
  EXPECT_EQ(f[0].ts, 1791216000000LL * kMilli);
  EXPECT_DOUBLE_EQ(f[0].rate, -0.00003208);
  EXPECT_THROW(bybit::parse_funding_history(R"({"retCode":10001,"retMsg":"bad"})", 0), std::runtime_error);
}

TEST(BybitPublic, Trades) {
  Instruments ins = two();
  Sink sink;
  bybit::PublicStream ps(ins, 0, sink);
  ps.on_message(R"({"success":true,"ret_msg":"","op":"subscribe","conn_id":"x"})");
  ps.on_message(R"({"topic":"publicTrade.ETHUSDT","type":"snapshot","ts":1672304486868,"data":[
    {"T":1672304486865,"s":"ETHUSDT","S":"Sell","v":"0.5","p":"1200.25","L":"PlusTick","i":"x","BT":false}]})");
  ASSERT_EQ(sink.events.size(), 1u);
  auto t = std::get<Trade>(sink.events[0]);
  EXPECT_EQ(t.sym, 1);
  EXPECT_EQ(t.ts, 1672304486865LL * kMilli);
  EXPECT_EQ(t.aggressor, Side::Sell);
  EXPECT_DOUBLE_EQ(t.price, 1200.25);
}

TEST(BybitPublic, OrderbookSnapshotDeltaAndThrottle) {
  Instruments ins = two();
  Sink sink;
  bybit::PublicStream ps(ins, 100 * kMilli, sink);
  ps.on_message(R"({"topic":"orderbook.50.BTCUSDT","type":"delta","ts":999,"data":{"s":"BTCUSDT","b":[["1","1"]],"a":[],"u":5,"seq":1}})");
  EXPECT_TRUE(sink.events.empty());  // delta before snapshot is dropped
  ps.on_message(R"({"topic":"orderbook.50.BTCUSDT","type":"snapshot","ts":1000,"data":{"s":"BTCUSDT",
    "b":[["100.0","1"],["99.5","2"]],"a":[["100.5","3"],["101.0","4"]],"u":1,"seq":1}})");
  ps.on_message(R"({"topic":"orderbook.50.BTCUSDT","type":"delta","ts":1050,"data":{"s":"BTCUSDT",
    "b":[["100.0","0"]],"a":[],"u":2,"seq":2}})");
  ASSERT_EQ(sink.events.size(), 1u);  // second update within book_interval
  ps.on_message(R"({"topic":"orderbook.50.BTCUSDT","type":"delta","ts":1100,"data":{"s":"BTCUSDT",
    "b":[["99.8","5"]],"a":[["100.4","1"]],"u":3,"seq":3}})");
  ASSERT_EQ(sink.events.size(), 2u);
  auto b = std::get<Book>(sink.events[1]);
  EXPECT_EQ(b.ts, 1100 * kMilli);
  EXPECT_DOUBLE_EQ(b.bids[0].price, 99.8);  // 100.0 was deleted
  EXPECT_DOUBLE_EQ(b.bids[1].price, 99.5);
  EXPECT_DOUBLE_EQ(b.asks[0].price, 100.4);
  EXPECT_DOUBLE_EQ(b.asks[1].price, 100.5);
  EXPECT_EQ(b.bids[2].qty, 0);
}

TEST(BybitPublic, FundingSettlesWhenNextFundingTimeRolls) {
  Instruments ins = two();
  Sink sink;
  bybit::PublicStream ps(ins, 0, sink);
  ps.on_message(R"({"topic":"tickers.BTCUSDT","type":"snapshot","ts":1,"data":{"symbol":"BTCUSDT",
    "fundingRate":"0.0001","nextFundingTime":"28800000","markPrice":"100"}})");
  ps.on_message(R"({"topic":"tickers.BTCUSDT","type":"delta","ts":2,"data":{"symbol":"BTCUSDT","fundingRate":"0.0002"}})");
  EXPECT_TRUE(sink.events.empty());
  ps.on_message(R"({"topic":"tickers.BTCUSDT","type":"delta","ts":3,"data":{"symbol":"BTCUSDT",
    "fundingRate":"0.00005","nextFundingTime":"57600000"}})");
  ASSERT_EQ(sink.events.size(), 1u);
  auto f = std::get<Funding>(sink.events[0]);
  EXPECT_EQ(f.ts, 28800000LL * kMilli);
  EXPECT_DOUBLE_EQ(f.rate, 0.0002);
}
