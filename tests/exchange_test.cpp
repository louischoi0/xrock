#include <gtest/gtest.h>

#include "exchange/bybit/bybit.hpp"

using namespace xrock;
namespace bybit = xrock::exchange::bybit;

TEST(Bybit, HmacSha256Rfc4231Case2) {
  EXPECT_EQ(bybit::hmac_sha256_hex("Jefe", "what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}

TEST(Bybit, OrderBodyUsesInstrumentPrecision) {
  Instrument in{.symbol = "BTCUSDT", .tick_size = 0.1, .lot_size = 0.001};
  Order o;
  o.side = Side::Sell;
  o.type = OrderType::Limit;
  o.tif = TimeInForce::PostOnly;
  o.qty = 0.012;
  o.price = 60000.1;
  o.reduce_only = true;
  EXPECT_EQ(bybit::order_create_body(in, o, "s1-7"),
            R"({"category":"linear","symbol":"BTCUSDT","side":"Sell","orderType":"Limit","qty":"0.012",)"
            R"("orderLinkId":"s1-7","price":"60000.1","timeInForce":"PostOnly","reduceOnly":true})");
}

TEST(Bybit, EndpointsPerEnv) {
  bybit::Settings s;
  s.env = "demo";
  EXPECT_EQ(s.rest_url(), "https://api-demo.bybit.com");
  EXPECT_EQ(s.public_ws_url(Category::Linear), "wss://stream.bybit.com/v5/public/linear");
  s.env = "testnet";
  EXPECT_EQ(s.private_ws_url(), "wss://stream-testnet.bybit.com/v5/private");
}
