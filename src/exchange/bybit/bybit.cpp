#include "exchange/bybit/bybit.hpp"

#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <cmath>
#include <format>
#include <stdexcept>

namespace xrock::exchange::bybit {
namespace {

// Decimal places of a tick/lot step: 0.1 -> 1, 0.25 -> 2, 0.0005 -> 4.
int decimals(double step) {
  int d = 0;
  for (double x = step; d < 12 && std::abs(x - std::round(x)) > 1e-9; x *= 10) ++d;
  return d;
}

std::string fixed(double x, double step) { return std::format("{:.{}f}", x, decimals(step)); }

const char* tif_name(TimeInForce t) {
  switch (t) {
    case TimeInForce::GTC: return "GTC";
    case TimeInForce::IOC: return "IOC";
    case TimeInForce::PostOnly: return "PostOnly";
  }
  return "GTC";
}

[[noreturn]] void no_transport(const char* what) {
  throw std::runtime_error(std::string("bybit: ") + what +
                           " is not implemented yet (private WS + order REST)");
}

}  // namespace

Settings Settings::from(const Config& cfg) {
  Settings s;
  s.env = cfg.str("exchange.env", "testnet");
  if (s.env != "mainnet" && s.env != "testnet" && s.env != "demo")
    throw std::runtime_error("exchange.env must be mainnet | testnet | demo");
  s.api_key = cfg.str("exchange.api_key", "");
  s.api_secret = cfg.str("exchange.api_secret", "");
  s.recv_window_ms = static_cast<std::int64_t>(cfg.num("exchange.recv_window", 5000));
  s.book_depth = static_cast<int>(cfg.num("exchange.book_depth", 50));
  s.leverage = cfg.num("margin.leverage", 1);
  s.archive_url = cfg.str("exchange.archive_url", s.archive_url);
  return s;
}

std::string Settings::rest_url() const {
  if (env == "testnet") return "https://api-testnet.bybit.com";
  if (env == "demo") return "https://api-demo.bybit.com";
  return "https://api.bybit.com";
}

// Demo trading has no public streams of its own; it trades against mainnet market data.
std::string Settings::public_ws_url(Category c) const {
  return std::string(env == "testnet" ? "wss://stream-testnet.bybit.com" : "wss://stream.bybit.com") +
         "/v5/public/" + to_string(c);
}

std::string Settings::private_ws_url() const {
  if (env == "testnet") return "wss://stream-testnet.bybit.com/v5/private";
  if (env == "demo") return "wss://stream-demo.bybit.com/v5/private";
  return "wss://stream.bybit.com/v5/private";
}

std::string hmac_sha256_hex(std::string_view key, std::string_view msg) {
  unsigned char mac[EVP_MAX_MD_SIZE];
  unsigned int len = 0;
  HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), reinterpret_cast<const unsigned char*>(msg.data()),
       msg.size(), mac, &len);
  std::string hex;
  for (unsigned i = 0; i < len; ++i) hex += std::format("{:02x}", mac[i]);
  return hex;
}

std::string sign_rest(const Settings& s, std::int64_t ts_ms, std::string_view payload) {
  return hmac_sha256_hex(s.api_secret, std::format("{}{}{}{}", ts_ms, s.api_key, s.recv_window_ms, payload));
}

std::string sign_ws_auth(const Settings& s, std::int64_t expires_ms) {
  return hmac_sha256_hex(s.api_secret, std::format("GET/realtime{}", expires_ms));
}

std::vector<std::string> public_topics(const Settings& s, const std::string& symbol,
                                       const std::vector<std::string>& streams) {
  std::vector<std::string> topics;
  for (const auto& st : streams) {
    if (st == "trade")
      topics.push_back("publicTrade." + symbol);
    else if (st == "book")
      topics.push_back(std::format("orderbook.{}.{}", s.book_depth, symbol));
    else if (st == "funding")
      topics.push_back("tickers." + symbol);
    else
      throw std::runtime_error("bybit: unknown stream " + st);
  }
  return topics;
}

std::string order_create_body(const Instrument& in, const Order& o, std::string_view link_id) {
  std::string b = std::format(
      R"({{"category":"{}","symbol":"{}","side":"{}","orderType":"{}","qty":"{}","orderLinkId":"{}")",
      to_string(in.category), in.symbol, o.side == Side::Buy ? "Buy" : "Sell",
      o.type == OrderType::Market ? "Market" : "Limit", fixed(o.qty, in.lot_size), link_id);
  if (o.type == OrderType::Limit)
    b += std::format(R"(,"price":"{}","timeInForce":"{}")", fixed(o.price, in.tick_size), tif_name(o.tif));
  if (o.reduce_only) b += R"(,"reduceOnly":true)";
  return b + '}';
}

std::string order_cancel_body(const Instrument& in, std::string_view link_id) {
  return std::format(R"({{"category":"{}","symbol":"{}","orderLinkId":"{}"}})", to_string(in.category), in.symbol,
                     link_id);
}

Gateway::Gateway(Settings s, const Instruments& ins) : s_(std::move(s)), ins_(ins) {
  if (s_.api_key.empty() || s_.api_secret.empty())
    throw std::runtime_error("bybit: exchange.api_key / api_secret empty (export BYBIT_API_KEY / BYBIT_API_SECRET)");
}

void Gateway::start() { no_transport("order gateway"); }
void Gateway::submit(const Order&) { no_transport("order gateway"); }
void Gateway::cancel(const Order&) { no_transport("order gateway"); }

}  // namespace xrock::exchange::bybit
