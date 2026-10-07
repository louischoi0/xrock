#include "core/instruments.hpp"

#include <stdexcept>

namespace xrock {
namespace {

Category parse_category(const std::string& s) {
  if (s == "linear") return Category::Linear;
  if (s == "inverse") return Category::Inverse;
  if (s == "spot") return Category::Spot;
  throw std::runtime_error("unknown category: " + s);
}

}  // namespace

Instruments Instruments::from_config(const Config& cfg) {
  Instruments ins;
  for (const auto& sym : cfg.list("market.symbols")) {
    Config c = cfg.section("instrument." + sym);
    Instrument in{.symbol = sym};
    in.category = parse_category(c.str("category", "linear"));
    in.tick_size = c.num("tick_size", in.tick_size);
    in.lot_size = c.num("lot_size", in.lot_size);
    in.min_qty = c.num("min_qty", in.min_qty);
    in.maker_fee = c.num("maker_fee", in.maker_fee);
    in.taker_fee = c.num("taker_fee", in.taker_fee);
    in.maintenance_margin = c.num("maintenance_margin", in.maintenance_margin);
    ins.add(std::move(in));
  }
  if (ins.size() == 0) throw std::runtime_error("config: market.symbols is empty");
  return ins;
}

SymbolId Instruments::add(Instrument in) {
  list_.push_back(std::move(in));
  return static_cast<SymbolId>(list_.size() - 1);
}

SymbolId Instruments::id(std::string_view symbol) const {
  for (std::size_t i = 0; i < list_.size(); ++i)
    if (list_[i].symbol == symbol) return static_cast<SymbolId>(i);
  throw std::runtime_error("unknown symbol: " + std::string(symbol));
}

}  // namespace xrock
