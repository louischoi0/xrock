#include "backtest/sim_gateway.hpp"

#include <algorithm>
#include <cmath>

namespace xrock::backtest {

SimParams SimParams::from(const Config& c) {
  return {.latency = c.duration("backtest.latency", 50 * kMilli),
          .slippage_bps = c.num("backtest.slippage_bps", 0),
          .leverage = c.num("margin.leverage", 1)};
}

SimGateway::SimGateway(const Instruments& ins, const Clock& clock, SimParams p)
    : ins_(ins), clock_(clock), p_(p), quotes_(ins.size()), pf_(0, ins.size()) {}

void SimGateway::submit(const Order& o) { inflight_.push_back({clock_.now() + p_.latency, false, o}); }
void SimGateway::cancel(const Order& o) { inflight_.push_back({clock_.now() + p_.latency, true, o}); }

void SimGateway::on_market(const MarketEvent& ev) {
  Timestamp ts = event_ts(ev);
  while (!inflight_.empty() && inflight_.front().arrive <= ts) {
    Request r = inflight_.front();
    inflight_.pop_front();
    arrive(r);
  }
  std::visit(overloaded{[&](const Trade& t) {
                          quotes_[t.sym].last = t.price;
                          pf_.mark(t.sym, t.price);
                          match(t.sym, ts, [&](const Order& o) {
                            return o.side == Side::Buy ? t.price < o.price : t.price > o.price;
                          });
                          check_liquidation(t.sym, t.price, ts);
                        },
                        [&](const Book& b) {
                          quotes_[b.sym].bid = b.bid();
                          quotes_[b.sym].ask = b.ask();
                          match(b.sym, ts, [&](const Order& o) {
                            return o.side == Side::Buy ? b.ask() > 0 && b.ask() <= o.price
                                                       : b.bid() > 0 && b.bid() >= o.price;
                          });
                        },
                        [&](const Funding& f) { pf_.apply_funding(f); }},
             ev);
}

void SimGateway::arrive(const Request& r) {
  Timestamp now = r.arrive;
  if (r.cancel) {
    auto it = std::find_if(resting_.begin(), resting_.end(), [&](const Order& o) { return o.id == r.order.id; });
    if (it == resting_.end()) return;  // already filled or canceled
    resting_.erase(it);
    listener_->on_order_update({.ts = now, .id = r.order.id, .sym = r.order.sym, .status = OrderStatus::Canceled});
    return;
  }

  Order o = r.order;
  const Quote& q = quotes_[o.sym];
  if (o.reduce_only) {
    double pos = pf_.position(o.sym).qty;
    if (pos == 0 || (pos > 0) == (o.side == Side::Buy)) return reject(o, now, "reduce-only would increase position");
    o.qty = std::min(o.qty, std::abs(pos));
  }
  double touch = o.side == Side::Buy ? q.ask : q.bid;  // 0 before the first book
  if (o.type == OrderType::Market) {
    double px = touch > 0 ? touch : q.last;
    if (px <= 0) return reject(o, now, "no market price");
    return fill(o, now, px * (1 + sign(o.side) * p_.slippage_bps * 1e-4), false);
  }
  bool crosses = touch > 0 && (o.side == Side::Buy ? o.price >= touch : o.price <= touch);
  if (crosses) {
    if (o.tif == TimeInForce::PostOnly) return reject(o, now, "post-only would cross");
    return fill(o, now, touch, false);
  }
  if (o.tif == TimeInForce::IOC) {
    listener_->on_order_update({.ts = now, .id = o.id, .sym = o.sym, .status = OrderStatus::Canceled});
    return;
  }
  resting_.push_back(o);
  listener_->on_order_update({.ts = now, .id = o.id, .sym = o.sym, .status = OrderStatus::New});
}

template <class Crosses>
void SimGateway::match(SymbolId sym, Timestamp ts, Crosses crosses) {
  std::vector<Order> hit;
  std::erase_if(resting_, [&](const Order& o) {
    if (o.sym != sym || !crosses(o)) return false;
    hit.push_back(o);
    return true;
  });
  for (const Order& o : hit) fill(o, ts, o.price, true);
}

void SimGateway::fill(const Order& o, Timestamp ts, double price, bool maker) {
  const Instrument& in = ins_[o.sym];
  Fill f{.ts = ts, .id = o.id, .sym = o.sym, .side = o.side, .price = price, .qty = o.qty,
         .fee = price * o.qty * (maker ? in.maker_fee : in.taker_fee), .maker = maker};
  pf_.apply_fill(f);
  listener_->on_fill(f);
  listener_->on_order_update({.ts = ts, .id = o.id, .sym = o.sym, .status = OrderStatus::Filled, .filled_qty = o.qty});
}

void SimGateway::reject(const Order& o, Timestamp ts, const char* reason) {
  listener_->on_order_update({.ts = ts, .id = o.id, .sym = o.sym, .status = OrderStatus::Rejected, .reason = reason});
}

void SimGateway::check_liquidation(SymbolId sym, double mark, Timestamp ts) {
  const Position& p = pf_.position(sym);
  if (p.qty == 0) return;
  double mmr = ins_[sym].maintenance_margin, im = 1 / p_.leverage;
  double liq = p.qty > 0 ? p.avg_price * (1 - im + mmr) : p.avg_price * (1 + im - mmr);
  if (p.qty > 0 ? mark > liq : mark < liq) return;
  double qty = std::abs(p.qty);
  Fill f{.ts = ts, .id = 0, .sym = sym, .side = p.qty > 0 ? Side::Sell : Side::Buy, .price = liq, .qty = qty,
         .fee = liq * qty * ins_[sym].taker_fee, .liquidation = true};
  pf_.apply_fill(f);
  listener_->on_fill(f);
}

}  // namespace xrock::backtest
