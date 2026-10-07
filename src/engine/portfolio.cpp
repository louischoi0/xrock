#include "engine/portfolio.hpp"

#include <algorithm>
#include <cmath>

namespace xrock::engine {

Portfolio::Portfolio(double initial_cash, std::size_t n_symbols) : cash_(initial_cash), pos_(n_symbols) {}

double Portfolio::apply_fill(const Fill& f) {
  Position& p = pos_.at(f.sym);
  double dq = sign(f.side) * f.qty;
  double realized = 0;
  if (p.qty == 0 || (p.qty > 0) == (dq > 0)) {
    double next = p.qty + dq;
    p.avg_price = (p.qty * p.avg_price + dq * f.price) / next;
    p.qty = next;
  } else {
    double closed = std::min(std::abs(p.qty), std::abs(dq));
    realized = closed * (f.price - p.avg_price) * (p.qty > 0 ? 1 : -1);
    double next = p.qty + dq;
    if (std::abs(next) < 1e-12) {
      next = 0;
      p.avg_price = 0;
    } else if ((next > 0) != (p.qty > 0)) {
      p.avg_price = f.price;  // flipped: the remainder opened at the fill price
    }
    p.qty = next;
  }
  if (p.mark == 0) p.mark = f.price;
  p.realized += realized;
  p.fees += f.fee;
  cash_ += realized - f.fee;
  return realized;
}

double Portfolio::apply_funding(const Funding& f) {
  Position& p = pos_.at(f.sym);
  double payment = p.qty * p.mark * f.rate;
  p.funding += payment;
  cash_ -= payment;
  return payment;
}

void Portfolio::mark(SymbolId s, double price) {
  if (price > 0) pos_.at(s).mark = price;
}

double Portfolio::equity() const {
  double e = cash_;
  for (const auto& p : pos_) e += p.unrealized();
  return e;
}

double Portfolio::gross_notional() const {
  double n = 0;
  for (const auto& p : pos_) n += p.notional();
  return n;
}

}  // namespace xrock::engine
