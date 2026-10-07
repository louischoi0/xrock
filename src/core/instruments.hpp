#pragma once

#include <string_view>
#include <vector>

#include "core/config.hpp"
#include "core/types.hpp"

namespace xrock {

// Instrument table; SymbolId is the index. Order follows [market] symbols.
class Instruments {
 public:
  static Instruments from_config(const Config&);  // [market] symbols + [instrument.<SYM>] specs

  SymbolId add(Instrument);
  SymbolId id(std::string_view symbol) const;  // throws if unknown
  const Instrument& operator[](SymbolId s) const { return list_[s]; }
  std::size_t size() const { return list_.size(); }

 private:
  std::vector<Instrument> list_;
};

}  // namespace xrock
