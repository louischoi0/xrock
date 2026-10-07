#include "analytics/journal.hpp"

#include <format>
#include <fstream>
#include <stdexcept>

#include "core/config.hpp"

namespace xrock::analytics {
namespace {

std::ofstream open_csv(const std::filesystem::path& p, const char* header) {
  std::ofstream f(p);
  if (!f) throw std::runtime_error("journal: cannot write " + p.string());
  f << header << '\n';
  return f;
}

template <class F>
void each_row(const std::filesystem::path& p, F on_row) {
  std::ifstream in(p);
  if (!in) return;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line))
    if (!line.empty()) on_row(split(line, ','));
}

}  // namespace

void Journal::write_csv(const std::filesystem::path& dir, const Instruments& ins) const {
  std::filesystem::create_directories(dir);
  auto f = open_csv(dir / "orders.csv", "ts,id,symbol,side,type,qty,price,reduce_only");
  for (const Order& o : orders)
    f << std::format("{},{},{},{},{},{},{},{}\n", o.ts, o.id, ins[o.sym].symbol, to_string(o.side), to_string(o.type),
                     o.qty, o.price, int(o.reduce_only));
  f = open_csv(dir / "updates.csv", "ts,id,symbol,status,filled_qty,reason");
  for (const OrderUpdate& u : updates)
    f << std::format("{},{},{},{},{},{}\n", u.ts, u.id, ins[u.sym].symbol, to_string(u.status), u.filled_qty, u.reason);
  f = open_csv(dir / "fills.csv", "ts,id,symbol,side,price,qty,fee,maker,liquidation,realized");
  for (const auto& [x, realized] : fills)
    f << std::format("{},{},{},{},{},{},{},{},{},{}\n", x.ts, x.id, ins[x.sym].symbol, to_string(x.side), x.price,
                     x.qty, x.fee, int(x.maker), int(x.liquidation), realized);
  f = open_csv(dir / "funding.csv", "ts,symbol,rate,payment");
  for (const auto& [x, payment] : funding)
    f << std::format("{},{},{},{}\n", x.ts, ins[x.sym].symbol, x.rate, payment);
  f = open_csv(dir / "equity.csv", "ts,equity");
  for (const auto& p : equity) f << std::format("{},{}\n", p.ts, p.equity);
  if (!decisions.empty()) {
    f = open_csv(dir / "decisions.csv", "ts,symbol,kind,reason,order_id,values");
    for (const auto& [ts, d] : decisions) {
      std::string values;
      for (const auto& [k, v] : d.values) values += std::format("{}{}={}", values.empty() ? "" : ";", k, v);
      f << std::format("{},{},{},{},{},{}\n", ts, ins[d.sym].symbol, d.kind, d.reason, d.order, values);
    }
  }
  if (predictions.empty()) return;
  std::string header = "ts,symbol";
  for (std::size_t i = 0; i < predictions.front().values.size(); ++i) header += std::format(",y{}", i);
  f = open_csv(dir / "predictions.csv", header.c_str());
  for (const auto& p : predictions) {
    f << p.ts << ',' << ins[p.sym].symbol;
    for (float v : p.values) f << std::format(",{}", v);
    f << '\n';
  }
}

Journal Journal::read_csv(const std::filesystem::path& dir, const Instruments& ins) {
  Journal j;
  each_row(dir / "fills.csv", [&](const std::vector<std::string>& r) {
    Fill f{.ts = std::stoll(r.at(0)),
           .id = std::stoull(r.at(1)),
           .sym = ins.id(r.at(2)),
           .side = r.at(3) == "buy" ? Side::Buy : Side::Sell,
           .price = std::stod(r.at(4)),
           .qty = std::stod(r.at(5)),
           .fee = std::stod(r.at(6)),
           .maker = r.at(7) == "1",
           .liquidation = r.at(8) == "1"};
    j.fills.push_back({f, std::stod(r.at(9))});
  });
  each_row(dir / "funding.csv", [&](const std::vector<std::string>& r) {
    j.funding.push_back({{.ts = std::stoll(r.at(0)), .sym = ins.id(r.at(1)), .rate = std::stod(r.at(2))},
                         std::stod(r.at(3))});
  });
  each_row(dir / "equity.csv",
           [&](const std::vector<std::string>& r) { j.equity.push_back({std::stoll(r.at(0)), std::stod(r.at(1))}); });
  each_row(dir / "decisions.csv", [&](const std::vector<std::string>& r) {
    // split() drops empty fields, so reason / values may be missing: parse by position on the raw shape.
    Decision d;
    d.sym = ins.id(r.at(1));
    d.kind = r.at(2);
    std::size_t i = 3;
    if (r.size() >= 6 || (r.size() == 5 && r[4].find('=') != std::string::npos)) d.reason = r.at(i++);
    d.order = std::stoull(r.at(i++));
    if (i < r.size())
      for (const auto& kv : split(r[i], ';')) {
        auto eq = kv.find('=');
        d.values.emplace_back(kv.substr(0, eq), std::stod(kv.substr(eq + 1)));
      }
    j.decisions.push_back({std::stoll(r.at(0)), std::move(d)});
  });
  if (j.equity.empty()) throw std::runtime_error("journal: no equity rows in " + dir.string());
  return j;
}

}  // namespace xrock::analytics
