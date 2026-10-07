#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>

#include "data/bar_aggregator.hpp"
#include "data/file_store.hpp"
#include "data/kds/kds_store.hpp"
#include "data/replay.hpp"
#include "data/sqlite_store.hpp"

using namespace xrock;
namespace fs = std::filesystem;

TEST(Bars, AlignedClose) {
  data::BarAggregator agg(0, kMinute);
  agg.add({.ts = 10 * kSecond, .price = 100, .qty = 1});
  agg.add({.ts = 50 * kSecond, .price = 105, .qty = 2});
  EXPECT_FALSE(agg.advance(59 * kSecond));
  EXPECT_EQ(agg.deadline(), kMinute);
  auto b = agg.advance(kMinute);
  ASSERT_TRUE(b);
  EXPECT_EQ(b->ts, kMinute);
  EXPECT_EQ(b->open, 100);
  EXPECT_EQ(b->close, 105);
  EXPECT_EQ(b->volume, 3);
  EXPECT_EQ(b->trades, 2u);
  EXPECT_EQ(agg.deadline(), kNever);
}

// Same contract for every backend: write out of symbol order, read back merged by time within [start, end).
class StoreTest : public ::testing::TestWithParam<std::string> {
 protected:
  std::unique_ptr<data::Store> make() {
    auto root = fs::temp_directory_path() / ("xrock_store_test_" + GetParam());
    if (!opened_) {
      fs::remove_all(root);
      opened_ = true;
    }
    if (GetParam() == "file") return std::make_unique<data::FileStore>(root, ins_);
    if (GetParam() == "kds") {
      // Each run gets fresh relations: KDS has no DROP that reclaims, and a rerun must not see old rows.
      static const std::string prefix = std::format("t{}", std::chrono::system_clock::now().time_since_epoch().count());
      return std::make_unique<data::KdsStore>(
          data::KdsStore::Options{.port = std::atoi(std::getenv("XROCK_KDS_PORT")),
                                  .protocol = std::getenv("XROCK_KDS_TEXT") ? data::kwp::Protocol::Text
                                                                             : data::kwp::Protocol::Kwp,
                                  .table_prefix = prefix,
                                  .venue = "test", .batch_rows = 2},
          ins_);
    }
    return std::make_unique<data::SqliteStore>((root / "test.sqlite").string(), ins_);
  }
  Instruments ins_ = [] {
    Instruments i;
    i.add({.symbol = "AAA"});
    i.add({.symbol = "BBB"});
    return i;
  }();
  bool opened_ = false;
};

TEST_P(StoreTest, WriteThenReplayMerged) {
  if (GetParam() == "kds" && !std::getenv("XROCK_KDS_PORT")) GTEST_SKIP() << "set XROCK_KDS_PORT (and XROCK_KDS_TEXT=1 for a pre-KWP server)";
  {
    auto store = make();
    store->write(Trade{.ts = kSecond, .sym = 0, .price = 1});
    store->write(Trade{.ts = kDay + 1, .sym = 0, .price = 3});  // next UTC day
    store->write(Trade{.ts = 2 * kSecond, .sym = 1, .price = 2});
    Book b{.ts = 3 * kSecond, .sym = 1};
    b.bids[0] = {9, 1};
    store->write(b);
    store->write(Trade{.ts = 3 * kDay, .sym = 1, .price = 4});  // outside range
    store->write(Funding{.ts = 4 * kSecond, .sym = 0, .rate = -0.000123});
    store->flush();
    EXPECT_TRUE(store->has(0, data::Stream::Trade, kDay, 2 * kDay));
    EXPECT_FALSE(store->has(0, data::Stream::Trade, 2 * kDay, 3 * kDay));
    EXPECT_FALSE(store->has(1, data::Stream::Funding, 0, 5 * kDay));
  }
  data::ReplayFeed rf(make(), ins_, {data::Stream::Trade, data::Stream::Book}, 0, 2 * kDay);
  std::vector<Timestamp> ts;
  const MarketEvent* e;
  while ((e = rf.next())) {
    ts.push_back(event_ts(*e));
    if (auto* bk = std::get_if<Book>(e)) {
      EXPECT_EQ(bk->bid(), 9);
    }
  }
  EXPECT_EQ(ts, (std::vector<Timestamp>{kSecond, 2 * kSecond, 3 * kSecond, kDay + 1}));
  data::ReplayFeed funding(make(), ins_, {data::Stream::Funding}, 0, kDay);
  const MarketEvent* f = funding.next();
  ASSERT_TRUE(f);
  EXPECT_DOUBLE_EQ(std::get<Funding>(*f).rate, -0.000123);
}

INSTANTIATE_TEST_SUITE_P(Backends, StoreTest, ::testing::Values("file", "sqlite", "kds"));
