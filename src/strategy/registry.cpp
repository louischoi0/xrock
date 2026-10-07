#include <map>
#include <stdexcept>

#include "strategy/strategy.hpp"

namespace xrock {

// Defined next to each strategy. Add a strategy: new .cpp + one line here + CMakeLists.
std::unique_ptr<Strategy> make_ma_cross(const Config&, const Instruments&);
std::unique_ptr<Strategy> make_trade_flow(const Config&, const Instruments&);
std::unique_ptr<Strategy> make_buy_hold(const Config&, const Instruments&);

namespace {
std::map<std::string, StrategyFactory>& factories() {
  static std::map<std::string, StrategyFactory> m{
      {"ma_cross", make_ma_cross},
      {"trade_flow", make_trade_flow},
      {"buy_hold", make_buy_hold},
  };
  return m;
}
}  // namespace

std::unique_ptr<Strategy> make_strategy(const std::string& name, const Config& params, const Instruments& ins) {
  auto it = factories().find(name);
  if (it == factories().end()) throw std::runtime_error("unknown strategy: " + name);
  return it->second(params, ins);
}

void register_strategy(const std::string& name, StrategyFactory f) { factories()[name] = std::move(f); }

}  // namespace xrock
