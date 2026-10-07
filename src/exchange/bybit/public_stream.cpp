#include <ixwebsocket/IXWebSocket.h>

#include <charconv>
#include <format>
#include <simdjson.h>
#include <stdexcept>

#include "core/log.hpp"
#include "exchange/bybit/bybit.hpp"

namespace xrock::exchange::bybit {
namespace {

using simdjson::dom::array;
using simdjson::dom::element;
using simdjson::dom::object;

constexpr Duration kPingEvery = 20 * kSecond;
constexpr std::size_t kTopicsPerSubscribe = 10;

double num(std::string_view s) {
  double x = 0;
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), x);
  if (ec != std::errc{}) throw std::runtime_error("bybit: bad number '" + std::string(s) + "'");
  return x;
}


std::int64_t ms_field(element e) {  // Bybit sends ms both as numbers and as strings
  if (e.is_string()) return static_cast<std::int64_t>(num(std::string_view(e)));
  return std::int64_t(e);
}

void apply_levels(array levels, auto& side) {
  for (element lv : levels) {
    array pq = lv;
    double price = num(pq.at(0)), qty = num(pq.at(1));
    if (qty == 0)
      side.erase(price);
    else
      side[price] = qty;
  }
}

template <class Side>
void top(const Side& side, std::array<Level, kBookDepth>& out) {
  int i = 0;
  for (auto it = side.begin(); it != side.end() && i < kBookDepth; ++it, ++i) out[i] = {it->first, it->second};
}

}  // namespace

PublicStream::PublicStream(const Instruments& ins, Duration book_interval, MarketListener& out)
    : ins_(ins), book_interval_(book_interval), out_(out), books_(ins.size()), funding_(ins.size()) {
  for (SymbolId s = 0; s < ins.size(); ++s) ids_.emplace(ins[s].symbol, s);
}

void PublicStream::reset() {
  for (auto& b : books_) b = L2{};
}

void PublicStream::on_message(std::string_view json) {
  thread_local simdjson::dom::parser parser;
  element doc = parser.parse(json.data(), json.size());
  object msg = doc;

  if (element op; msg["op"].get(op) == simdjson::SUCCESS) {  // subscribe / pong acknowledgements
    if (element ok; msg["success"].get(ok) == simdjson::SUCCESS && !bool(ok))
      log::warn("bybit ws: {} failed: {}", std::string_view(op), std::string_view(msg["ret_msg"]));
    return;
  }
  std::string_view topic = msg["topic"];
  auto dot = topic.rfind('.');
  auto it = ids_.find(std::string(topic.substr(dot + 1)));
  if (it == ids_.end()) return;
  SymbolId sym = it->second;
  std::string_view kind = topic.substr(0, topic.find('.'));
  Timestamp ts = ms_field(msg["ts"]) * kMilli;

  if (kind == "publicTrade") {
    for (element t : array(msg["data"]))
      out_.on_market(Trade{.ts = ms_field(t["T"]) * kMilli, .sym = sym,
                           .aggressor = std::string_view(t["S"]) == "Buy" ? Side::Buy : Side::Sell,
                           .price = num(t["p"]), .qty = num(t["v"])});
  } else if (kind == "orderbook") {
    L2& b = books_[sym];
    object data = msg["data"];
    bool snapshot = std::string_view(msg["type"]) == "snapshot" || std::uint64_t(data["u"]) == 1;
    if (snapshot) {
      b.bids.clear();
      b.asks.clear();
      b.ready = true;
    }
    if (!b.ready) return;  // deltas before the first snapshot
    apply_levels(data["b"], b.bids);
    apply_levels(data["a"], b.asks);
    if (ts - b.last_emit < book_interval_) return;
    b.last_emit = ts;
    Book out{.ts = ts, .sym = sym};
    top(b.bids, out.bids);
    top(b.asks, out.asks);
    out_.on_market(out);
  } else if (kind == "tickers") {
    FundingState& f = funding_[sym];
    object data = msg["data"];
    double rate = f.rate;
    Timestamp next = f.next;
    if (element e; data["fundingRate"].get(e) == simdjson::SUCCESS && !std::string_view(e).empty()) rate = num(std::string_view(e));
    if (element e; data["nextFundingTime"].get(e) == simdjson::SUCCESS) next = ms_field(e) * kMilli;
    // The previous period settled at its nextFundingTime with the last rate quoted for it.
    if (f.next != 0 && next > f.next) out_.on_market(Funding{.ts = f.next, .sym = sym, .rate = f.rate});
    f = {rate, next};
  }
}

struct Feed::Connection {
  ix::WebSocket ws;
  PublicStream stream;
  std::vector<std::string> topics;
  Connection(const Instruments& ins, Duration book_interval, MarketListener& l) : stream(ins, book_interval, l) {}
};

Feed::Feed(Settings s, const Instruments& ins, std::vector<std::string> streams, Duration book_interval)
    : s_(std::move(s)), ins_(ins), streams_(std::move(streams)), book_interval_(book_interval) {}

Feed::~Feed() { stop(); }

void Feed::start(MarketListener& l) {
  std::map<Category, Connection*> by_category;
  for (SymbolId sym = 0; sym < ins_.size(); ++sym) {
    Connection*& c = by_category[ins_[sym].category];
    if (!c) {
      conns_.push_back(std::make_unique<Connection>(ins_, book_interval_, l));
      c = conns_.back().get();
      c->ws.setUrl(s_.public_ws_url(ins_[sym].category));
    }
    for (auto& t : public_topics(s_, ins_[sym].symbol, streams_)) c->topics.push_back(std::move(t));
  }
  for (auto& conn : conns_) {
    Connection* c = conn.get();
    c->ws.setOnMessageCallback([c](const ix::WebSocketMessagePtr& m) {
      switch (m->type) {
        case ix::WebSocketMessageType::Open:
          log::info("bybit ws: connected {}", c->ws.getUrl());
          c->stream.reset();
          for (std::size_t i = 0; i < c->topics.size(); i += kTopicsPerSubscribe) {
            std::string args;
            for (std::size_t j = i; j < std::min(i + kTopicsPerSubscribe, c->topics.size()); ++j)
              args += (args.empty() ? "\"" : ",\"") + c->topics[j] + "\"";
            c->ws.sendText(std::format(R"({{"op":"subscribe","args":[{}]}})", args));
          }
          break;
        case ix::WebSocketMessageType::Message:
          try {
            c->stream.on_message(m->str);
          } catch (const std::exception& e) {
            log::warn("bybit ws: {} in {}", e.what(), m->str.substr(0, 200));
          }
          break;
        case ix::WebSocketMessageType::Close:
          log::warn("bybit ws: closed ({} {})", m->closeInfo.code, m->closeInfo.reason);
          break;
        case ix::WebSocketMessageType::Error:
          log::warn("bybit ws: {} (retry {})", m->errorInfo.reason, m->errorInfo.retries);
          break;
        default:
          break;
      }
    });
    c->ws.start();
  }
  pinger_ = std::thread([this] {
    std::unique_lock lk(mu_);
    while (!cv_.wait_for(lk, std::chrono::nanoseconds(kPingEvery), [this] { return stopping_; }))
      for (auto& c : conns_) c->ws.sendText(R"({"op":"ping"})");
  });
}

void Feed::stop() {
  {
    std::lock_guard lk(mu_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (pinger_.joinable()) pinger_.join();
  for (auto& c : conns_) c->ws.stop();
  conns_.clear();
}

}  // namespace xrock::exchange::bybit
