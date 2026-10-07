#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/config.hpp"
#include "core/instruments.hpp"
#include "core/interfaces.hpp"

// Bybit v5 adapter: settings/signing, realtime public feed (Feed), historical download (History) and the order
// gateway (Gateway, transport still TODO).
namespace xrock::exchange::bybit {

struct Settings {
  std::string env = "testnet";  // mainnet | testnet | demo
  std::string api_key, api_secret;
  std::int64_t recv_window_ms = 5000;
  int book_depth = 50;  // orderbook.{1,50,200,1000}
  double leverage = 1;  // [margin] leverage, applied per symbol on start (POST /v5/position/set-leverage)
  std::string archive_url = "https://public.bybit.com";  // daily trade archives (mainnet)

  static Settings from(const Config&);
  std::string rest_url() const;
  std::string public_ws_url(Category) const;
  std::string private_ws_url() const;
};

std::string hmac_sha256_hex(std::string_view key, std::string_view msg);
// REST: HMAC(secret, timestamp + api_key + recv_window + payload); payload = query string (GET) or JSON body (POST).
// Headers: X-BAPI-API-KEY, X-BAPI-TIMESTAMP, X-BAPI-SIGN, X-BAPI-RECV-WINDOW.
std::string sign_rest(const Settings&, std::int64_t ts_ms, std::string_view payload);
// Private WS: {"op":"auth","args":[api_key, expires, HMAC(secret, "GET/realtime" + expires)]}.
std::string sign_ws_auth(const Settings&, std::int64_t expires_ms);

// trade -> publicTrade.SYM, book -> orderbook.N.SYM, funding -> tickers.SYM
std::vector<std::string> public_topics(const Settings&, const std::string& symbol, const std::vector<std::string>& streams);

// POST /v5/order/create and /v5/order/cancel bodies. link_id = orderLinkId, unique per account:
// use "<session>-<OrderId>" so ids survive restarts.
std::string order_create_body(const Instrument&, const Order&, std::string_view link_id);
std::string order_cancel_body(const Instrument&, std::string_view link_id);

// ---- realtime -------------------------------------------------------------------------------------------------

// Turns public WS messages into MarketEvents:
//   publicTrade  -> Trade
//   orderbook    -> local L2 book (snapshot + deltas; size 0 deletes) -> Book, at most once per book_interval
//   tickers      -> Funding{ts = settlement time, rate} when nextFundingTime rolls forward
// Single-threaded: one instance per WS connection.
class PublicStream {
 public:
  PublicStream(const Instruments&, Duration book_interval, MarketListener&);
  void on_message(std::string_view json);  // throws on malformed messages
  void reset();                            // after a reconnect: books are rebuilt from the next snapshot

 private:
  struct L2 {
    std::map<double, double, std::greater<>> bids;
    std::map<double, double> asks;
    Timestamp last_emit = 0;
    bool ready = false;
  };
  struct FundingState {
    double rate = 0;
    Timestamp next = 0;
  };

  const Instruments& ins_;
  Duration book_interval_;
  MarketListener& out_;
  std::unordered_map<std::string, SymbolId> ids_;
  std::vector<L2> books_;
  std::vector<FundingState> funding_;
};

// Realtime collector source: one public WS per instrument category, subscribe on every (re)connect,
// app-level ping every 20s; IXWebSocket reconnects with backoff.
class Feed final : public MarketDataFeed {
 public:
  Feed(Settings, const Instruments&, std::vector<std::string> streams, Duration book_interval);
  ~Feed() override;
  void start(MarketListener&) override;
  void stop() override;

 private:
  struct Connection;
  Settings s_;
  const Instruments& ins_;
  std::vector<std::string> streams_;
  Duration book_interval_;
  std::vector<std::unique_ptr<Connection>> conns_;
  std::thread pinger_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stopping_ = false;
};

// ---- historical -----------------------------------------------------------------------------------------------

// Daily trade archive CSV (timestamp[s.frac],symbol,side,size,price,...), columns located by header name.
std::vector<Trade> parse_trade_archive(std::string_view csv, SymbolId);
// GET /v5/market/funding/history response, returned oldest first.
std::vector<Funding> parse_funding_history(std::string_view json, SymbolId);

// Historical collector source (always mainnet data):
//   trade   -> {archive_url}/trading/{SYM}/{SYM}{YYYY-MM-DD}.csv.gz
//   funding -> GET https://api.bybit.com/v5/market/funding/history
// TODO: book from the orderbook archive (ob500 zip); until then collect books live.
class History final : public HistoricalSource {
 public:
  History(Settings, const Instruments&);
  bool fetch_day(SymbolId, const std::string& stream, Timestamp day, MarketListener&) override;

 private:
  Settings s_;
  const Instruments& ins_;
};

// ---- trading --------------------------------------------------------------------------------------------------

// TODO(transport): orders via REST /v5/order/create|cancel (or the WS trade API, /v5/trade);
// private WS topics: order -> OrderUpdate, execution -> Fill (execType Trade; BustTrade = liquidation;
// Funding is ignored here because the engine books funding from the Funding market event), position / wallet for
// reconciliation.
class Gateway final : public OrderGateway {
 public:
  Gateway(Settings, const Instruments&);
  void set_listener(ExecListener& l) override { listener_ = &l; }
  void start() override;
  void submit(const Order&) override;
  void cancel(const Order&) override;

 private:
  Settings s_;
  const Instruments& ins_;
  ExecListener* listener_ = nullptr;
};

}  // namespace xrock::exchange::bybit
