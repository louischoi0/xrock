#include "engine/engine.hpp"

#include <algorithm>
#include <stdexcept>

namespace xrock::engine {

Engine::Engine(const Instruments& ins, const Clock& clock, OrderGateway& gw, RiskManager& risk,
               analytics::Journal& journal, Strategy& strat, double initial_equity, Duration equity_interval)
    : ins_(ins),
      clock_(clock),
      gw_(gw),
      risk_(risk),
      journal_(journal),
      strat_(strat),
      pf_(initial_equity, ins.size()),
      equity_interval_(equity_interval) {
  if (equity_interval <= 0) throw std::invalid_argument("analytics.equity_interval must be > 0");
}

void Engine::start() {
  next_equity_ = now();
  strat_.on_start(*this);
}

void Engine::stop() {
  strat_.on_stop(*this);
  if (journal_.equity.empty() || journal_.equity.back().ts < now()) journal_.equity.push_back({now(), pf_.equity()});
}

void Engine::advance(Timestamp now) {
  while (!timers_.empty() && timers_.top().ts <= now) {
    Timer t = timers_.top();
    timers_.pop();
    strat_.on_timer(*this, t);
  }
  for (std::size_t i = 0; i < bars_.size(); ++i)  // index loop: on_bar may subscribe more bars
    if (auto bar = bars_[i].advance(now)) strat_.on_bar(*this, *bar);
  for (; next_equity_ <= now; next_equity_ += equity_interval_) journal_.equity.push_back({next_equity_, pf_.equity()});
}

Timestamp Engine::next_deadline() const {
  Timestamp d = next_equity_;
  if (!timers_.empty()) d = std::min(d, timers_.top().ts);
  for (const auto& b : bars_) d = std::min(d, b.deadline());
  return d;
}

std::vector<OrderId> Engine::open_orders() const {
  std::vector<OrderId> ids;
  for (const auto& [id, o] : open_) ids.push_back(id);
  return ids;
}

void Engine::on_market(const MarketEvent& ev) {
  std::visit(overloaded{[&](const Trade& t) {
                          pf_.mark(t.sym, t.price);
                          for (std::size_t i = 0; i < bars_.size(); ++i) {
                            if (bars_[i].sym() != t.sym) continue;
                            if (auto bar = bars_[i].advance(t.ts)) strat_.on_bar(*this, *bar);
                            bars_[i].add(t);
                          }
                          strat_.on_trade(*this, t);
                        },
                        [&](const Book& b) {
                          pf_.mark(b.sym, b.mid());
                          strat_.on_book(*this, b);
                        },
                        [&](const Funding& f) {
                          journal_.funding.push_back({f, pf_.apply_funding(f)});
                          strat_.on_funding(*this, f);
                        }},
             ev);
}

void Engine::on_order_update(const OrderUpdate& u) {
  journal_.updates.push_back(u);
  if (is_terminal(u.status)) open_.erase(u.id);
  strat_.on_order(*this, u);
}

void Engine::on_fill(const Fill& f) {
  journal_.fills.push_back({f, pf_.apply_fill(f)});
  strat_.on_fill(*this, f);
}

OrderId Engine::submit(const OrderRequest& req) {
  const Instrument& in = ins_[req.sym];
  Order o{req, next_id_++, now()};
  o.qty = floor_to(o.qty, in.lot_size);
  if (o.type == OrderType::Limit) o.price = round_to(o.price, in.tick_size);
  double ref = o.type == OrderType::Limit ? o.price : pf_.position(o.sym).mark;
  const char* why = o.qty < in.min_qty ? "qty below instrument minimum" : risk_.check(o, ref, pf_, now());
  journal_.orders.push_back(o);
  if (why) {
    journal_.updates.push_back({.ts = now(), .id = o.id, .sym = o.sym, .status = OrderStatus::Rejected, .reason = why});
    return 0;
  }
  open_.emplace(o.id, o);
  gw_.submit(o);
  return o.id;
}

void Engine::cancel(OrderId id) {
  if (auto it = open_.find(id); it != open_.end()) gw_.cancel(it->second);
}

void Engine::subscribe_bars(SymbolId sym, Duration interval) {
  for (const auto& b : bars_)
    if (b.sym() == sym && b.interval() == interval) return;
  bars_.emplace_back(sym, interval);
}

}  // namespace xrock::engine
