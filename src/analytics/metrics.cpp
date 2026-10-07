#include "analytics/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>

namespace xrock::analytics {

Metrics compute_metrics(const Journal& j) {
  Metrics m;
  const auto& eq = j.equity;
  if (!eq.empty()) {
    m.initial_equity = eq.front().equity;
    m.final_equity = eq.back().equity;
    if (m.initial_equity > 0) m.total_return = m.final_equity / m.initial_equity - 1;
  }
  double peak = 0;
  for (const auto& p : eq) {
    peak = std::max(peak, p.equity);
    if (peak > 0) m.max_drawdown = std::max(m.max_drawdown, 1 - p.equity / peak);
  }
  if (eq.size() >= 3) {
    std::size_t n = eq.size() - 1;
    double sum = 0, sq = 0;
    for (std::size_t i = 1; i <= n; ++i) {
      double r = eq[i].equity / eq[i - 1].equity - 1;
      sum += r;
      sq += r * r;
    }
    double mean = sum / n, sd = std::sqrt(std::max(0.0, (sq - n * mean * mean) / (n - 1)));
    double dt = double(eq.back().ts - eq.front().ts) / n;
    if (sd > 0 && dt > 0) m.sharpe = mean / sd * std::sqrt(365.0 * kDay / dt);
  }
  for (const auto& [f, realized] : j.fills) {
    ++m.fills;
    m.fees += f.fee;
    m.turnover += f.price * f.qty;
    m.realized += realized;
    if (realized != 0) {
      ++m.closing_fills;
      if (realized > 0) ++m.winning_fills;
    }
  }
  for (const auto& r : j.funding) m.funding += r.payment;
  return m;
}

std::string to_string(const Metrics& m) {
  return std::format(
      "equity    {:.2f} -> {:.2f} ({:+.2f}%)\n"
      "max dd    {:.2f}%\n"
      "sharpe    {:.2f}\n"
      "fills     {} (closing {}, win {:.1f}%)\n"
      "realized  {:.2f}  fees {:.2f}  funding {:.2f}  turnover {:.0f}\n",
      m.initial_equity, m.final_equity, 100 * m.total_return, 100 * m.max_drawdown, m.sharpe, m.fills,
      m.closing_fills, 100 * m.win_rate(), m.realized, m.fees, m.funding, m.turnover);
}

std::string decision_summary(const Journal& j) {
  if (j.decisions.empty()) return "";
  std::map<std::string, std::map<std::string, std::size_t>> by;  // kind -> reason -> count
  for (const auto& [ts, d] : j.decisions) ++by[d.kind][d.reason];
  std::string out = "decisions";
  for (const auto& [kind, reasons] : by) {
    std::size_t total = 0;
    std::string detail;
    for (const auto& [reason, n] : reasons) {
      total += n;
      if (!reason.empty()) detail += std::format("{}{} {}", detail.empty() ? "" : ", ", reason, n);
    }
    out += std::format(" | {} {}{}", kind, total, detail.empty() ? "" : " (" + detail + ")");
  }
  return out + "\n";
}

}  // namespace xrock::analytics
