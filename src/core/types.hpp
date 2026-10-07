#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace xrock {

using Timestamp = std::int64_t;  // ns since Unix epoch, UTC
using Duration = std::int64_t;   // ns
using SymbolId = std::uint16_t;  // index into Instruments
using OrderId = std::uint64_t;   // client order id assigned by the engine; 0 = exchange-initiated

constexpr Duration kMilli = 1'000'000;
constexpr Duration kSecond = 1'000 * kMilli;
constexpr Duration kMinute = 60 * kSecond;
constexpr Duration kHour = 60 * kMinute;
constexpr Duration kDay = 24 * kHour;
constexpr Timestamp kNever = std::numeric_limits<Timestamp>::max();

enum class Side : std::uint8_t { Buy, Sell };
enum class OrderType : std::uint8_t { Market, Limit };
enum class TimeInForce : std::uint8_t { GTC, IOC, PostOnly };
enum class OrderStatus : std::uint8_t { New, PartiallyFilled, Filled, Canceled, Rejected };
enum class Category : std::uint8_t { Linear, Inverse, Spot };

inline int sign(Side s) { return s == Side::Buy ? 1 : -1; }
inline bool is_terminal(OrderStatus s) {
  return s == OrderStatus::Filled || s == OrderStatus::Canceled || s == OrderStatus::Rejected;
}

inline const char* to_string(Side s) { return s == Side::Buy ? "buy" : "sell"; }
inline const char* to_string(OrderType t) { return t == OrderType::Market ? "market" : "limit"; }
inline const char* to_string(Category c) {
  switch (c) {
    case Category::Linear: return "linear";
    case Category::Inverse: return "inverse";
    case Category::Spot: return "spot";
  }
  return "?";
}
inline const char* to_string(OrderStatus s) {
  switch (s) {
    case OrderStatus::New: return "new";
    case OrderStatus::PartiallyFilled: return "partially_filled";
    case OrderStatus::Filled: return "filled";
    case OrderStatus::Canceled: return "canceled";
    case OrderStatus::Rejected: return "rejected";
  }
  return "?";
}

// Nearest multiple of step / largest multiple of step not above x (float tolerant).
inline double round_to(double x, double step) { return step > 0 ? std::round(x / step) * step : x; }
inline double floor_to(double x, double step) { return step > 0 ? std::floor(x / step + 1e-9) * step : x; }

struct Instrument {
  std::string symbol;  // exchange symbol, e.g. "BTCUSDT"
  Category category = Category::Linear;
  double tick_size = 0.01;
  double lot_size = 0.001;
  double min_qty = 0.001;
  double maker_fee = 0.0002;  // fraction of notional
  double taker_fee = 0.00055;
  double maintenance_margin = 0.005;  // fraction of notional
};

struct OrderRequest {
  SymbolId sym = 0;
  Side side = Side::Buy;
  OrderType type = OrderType::Market;
  TimeInForce tif = TimeInForce::GTC;
  double qty = 0;
  double price = 0;  // Limit only
  bool reduce_only = false;
};

struct Order : OrderRequest {
  OrderId id = 0;
  Timestamp ts = 0;  // submit time
};

struct OrderUpdate {
  Timestamp ts = 0;
  OrderId id = 0;
  SymbolId sym = 0;
  OrderStatus status = OrderStatus::New;
  double filled_qty = 0;    // cumulative
  const char* reason = "";  // static string, set on Rejected
};

// Gateways emit a Fill before the OrderUpdate that reports it.
struct Fill {
  Timestamp ts = 0;
  OrderId id = 0;
  SymbolId sym = 0;
  Side side = Side::Buy;
  double price = 0;
  double qty = 0;
  double fee = 0;  // quote currency, > 0 = paid
  bool maker = false;
  bool liquidation = false;
};

// Why a strategy did something: journaled to decisions.csv next to the orders and fills it explains.
// kind: what happened ("enter", "open", "exit", "skip", "rejected", ...); reason: why ("loss-cut", "tb_3_long", ...);
// order: the order it produced (0 = none); values: the numbers behind it (named, strategy-specific).
struct Decision {
  SymbolId sym = 0;
  std::string kind, reason;
  OrderId order = 0;
  std::vector<std::pair<std::string, double>> values;
};

// Linear (USDT-margined) position. TODO: inverse contracts settle PnL in the base coin.
struct Position {
  double qty = 0;  // signed, > 0 = long
  double avg_price = 0;
  double mark = 0;
  double realized = 0;  // closed PnL before fees
  double fees = 0;
  double funding = 0;  // paid (> 0) or received (< 0)
  double unrealized() const { return qty * (mark - avg_price); }
  double notional() const { return std::abs(qty) * mark; }
};

}  // namespace xrock
