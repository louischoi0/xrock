#pragma once

#include <array>
#include <variant>

#include "core/types.hpp"

namespace xrock {

struct Trade {
  Timestamp ts = 0;
  SymbolId sym = 0;
  Side aggressor = Side::Buy;
  double price = 0;
  double qty = 0;
};

constexpr int kBookDepth = 10;

struct Level {
  double price = 0;
  double qty = 0;  // 0 = empty level
};

// Top kBookDepth levels per side, best first.
struct Book {
  Timestamp ts = 0;
  SymbolId sym = 0;
  std::array<Level, kBookDepth> bids{};
  std::array<Level, kBookDepth> asks{};
  double bid() const { return bids[0].price; }
  double ask() const { return asks[0].price; }
  double mid() const { return bid() > 0 && ask() > 0 ? (bid() + ask()) / 2 : bid() + ask(); }
};

// Funding settlement: each position pays qty * mark * rate (longs pay when rate > 0).
struct Funding {
  Timestamp ts = 0;
  SymbolId sym = 0;
  double rate = 0;
};

// Time bar; ts is the close time, a multiple of interval.
struct Bar {
  Timestamp ts = 0;
  Duration interval = 0;
  SymbolId sym = 0;
  double open = 0, high = 0, low = 0, close = 0, volume = 0;
  std::uint32_t trades = 0;
};

struct Timer {
  Timestamp ts = 0;
  std::uint64_t id = 0;
};

using MarketEvent = std::variant<Trade, Book, Funding>;

inline Timestamp event_ts(const MarketEvent& e) {
  return std::visit([](const auto& x) { return x.ts; }, e);
}
inline SymbolId event_sym(const MarketEvent& e) {
  return std::visit([](const auto& x) { return x.sym; }, e);
}

template <class... F>
struct overloaded : F... {
  using F::operator()...;
};
template <class... F>
overloaded(F...) -> overloaded<F...>;

}  // namespace xrock
