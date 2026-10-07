#pragma once

#include <vector>

#include "core/events.hpp"

namespace xrock::engine {

// Positions and cash from fills, funding and marks. Linear contracts only.
class Portfolio {
 public:
  Portfolio(double initial_cash, std::size_t n_symbols);

  double apply_fill(const Fill&);           // returns the realized PnL of this fill (before fee)
  double apply_funding(const Funding&);     // returns the payment (> 0 = paid)
  void mark(SymbolId, double price);        // ignored if price <= 0

  const Position& position(SymbolId s) const { return pos_.at(s); }
  double cash() const { return cash_; }     // initial + realized - fees - funding
  double equity() const;                    // cash + unrealized
  double gross_notional() const;

 private:
  double cash_;
  std::vector<Position> pos_;
};

}  // namespace xrock::engine
