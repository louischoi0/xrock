#include <algorithm>
#include <charconv>
#include <format>
#include <simdjson.h>
#include <stdexcept>

#include "exchange/bybit/bybit.hpp"
#include "exchange/net.hpp"

namespace xrock::exchange::bybit {
namespace {

constexpr const char* kMainnetRest = "https://api.bybit.com";

// "1790812800.1031" -> ns, exact (no double rounding).
Timestamp parse_seconds(std::string_view s) {
  auto dot = s.find('.');
  std::int64_t sec = 0, frac = 0;
  std::from_chars(s.data(), s.data() + std::min(dot, s.size()), sec);
  Timestamp ns = sec * kSecond;
  if (dot != std::string_view::npos) {
    auto f = s.substr(dot + 1, 9);
    std::from_chars(f.data(), f.data() + f.size(), frac);
    for (auto n = f.size(); n < 9; ++n) frac *= 10;
    ns += frac;
  }
  return ns;
}

double to_double(std::string_view s) {
  double x = 0;
  if (std::from_chars(s.data(), s.data() + s.size(), x).ec != std::errc{})
    throw std::runtime_error("bybit archive: bad number '" + std::string(s) + "'");
  return x;
}

}  // namespace

std::vector<Trade> parse_trade_archive(std::string_view csv, SymbolId sym) {
  auto next_line = [&](std::size_t& pos) {
    auto nl = csv.find('\n', pos);
    if (nl == std::string_view::npos) nl = csv.size();
    auto line = csv.substr(pos, nl - pos);
    pos = nl + 1;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    return line;
  };
  std::size_t pos = 0;
  std::vector<std::string_view> fields;
  auto split_into = [&](std::string_view line) {
    fields.clear();
    for (std::size_t p = 0;;) {
      auto c = line.find(',', p);
      fields.push_back(line.substr(p, c == std::string_view::npos ? std::string_view::npos : c - p));
      if (c == std::string_view::npos) break;
      p = c + 1;
    }
  };

  split_into(next_line(pos));
  auto col = [&](std::string_view name) {
    auto it = std::find(fields.begin(), fields.end(), name);
    if (it == fields.end()) throw std::runtime_error("bybit archive: no column " + std::string(name));
    return static_cast<std::size_t>(it - fields.begin());
  };
  const std::size_t c_ts = col("timestamp"), c_side = col("side"), c_size = col("size"), c_price = col("price");
  const std::size_t need = std::max({c_ts, c_side, c_size, c_price});

  std::vector<Trade> out;
  while (pos < csv.size()) {
    auto line = next_line(pos);
    if (line.empty()) continue;
    split_into(line);
    if (fields.size() <= need) throw std::runtime_error("bybit archive: short row: " + std::string(line));
    out.push_back({.ts = parse_seconds(fields[c_ts]), .sym = sym,
                   .aggressor = fields[c_side] == "Buy" ? Side::Buy : Side::Sell,
                   .price = to_double(fields[c_price]), .qty = to_double(fields[c_size])});
  }
  if (!std::is_sorted(out.begin(), out.end(), [](const Trade& a, const Trade& b) { return a.ts < b.ts; }))
    std::stable_sort(out.begin(), out.end(), [](const Trade& a, const Trade& b) { return a.ts < b.ts; });
  return out;
}

std::vector<Funding> parse_funding_history(std::string_view json, SymbolId sym) {
  simdjson::dom::parser parser;
  simdjson::dom::object doc = parser.parse(json.data(), json.size());
  if (std::int64_t(doc["retCode"]) != 0)
    throw std::runtime_error("bybit funding history: " + std::string(std::string_view(doc["retMsg"])));
  std::vector<Funding> out;
  for (simdjson::dom::element e : simdjson::dom::array(doc["result"]["list"]))
    out.push_back({.ts = static_cast<Timestamp>(to_double(e["fundingRateTimestamp"])) * kMilli, .sym = sym,
                   .rate = to_double(e["fundingRate"])});
  std::sort(out.begin(), out.end(), [](const Funding& a, const Funding& b) { return a.ts < b.ts; });
  return out;
}

History::History(Settings s, const Instruments& ins) : s_(std::move(s)), ins_(ins) {}

bool History::fetch_day(SymbolId sym, const std::string& stream, Timestamp day, MarketListener& out) {
  const Instrument& in = ins_[sym];
  if (stream == "trade") {
    auto r = net::http_get(std::format("{}/trading/{}/{}{}.csv.gz", s_.archive_url, in.symbol, in.symbol,
                                       format_date(day)));
    if (r.status == 404) return false;
    if (r.status != 200) throw std::runtime_error(std::format("bybit archive: HTTP {}", r.status));
    for (const Trade& t : parse_trade_archive(net::gunzip(r.body), sym)) out.on_market(t);
    return true;
  }
  if (stream == "funding") {
    auto r = net::http_get(std::format(
        "{}/v5/market/funding/history?category={}&symbol={}&startTime={}&endTime={}&limit=200", kMainnetRest,
        to_string(in.category), in.symbol, day / kMilli, (day + kDay) / kMilli - 1));
    if (r.status != 200) throw std::runtime_error(std::format("bybit funding history: HTTP {}", r.status));
    auto rows = parse_funding_history(r.body, sym);
    for (const Funding& f : rows) out.on_market(f);
    return !rows.empty();
  }
  throw std::runtime_error("bybit history: stream '" + stream + "' is not downloadable (collect it live)");
}

}  // namespace xrock::exchange::bybit
