#include "exchange/registry.hpp"

#include <stdexcept>

#include "exchange/bybit/bybit.hpp"
#include "exchange/synthetic.hpp"

namespace xrock::exchange {

std::unique_ptr<MarketDataFeed> make_feed(const Config& cfg, const Instruments& ins) {
  std::string name = cfg.str("exchange.name");
  if (name == "bybit")
    return std::make_unique<bybit::Feed>(bybit::Settings::from(cfg), ins, cfg.list("market.streams"),
                                         cfg.duration("market.book_interval", 100 * kMilli));
  if (name == "synthetic") return std::make_unique<SyntheticFeed>(cfg, ins);
  throw std::runtime_error("unknown exchange.name: " + name);
}

std::unique_ptr<OrderGateway> make_gateway(const Config& cfg, const Instruments& ins) {
  std::string name = cfg.str("exchange.name");
  if (name == "bybit") return std::make_unique<bybit::Gateway>(bybit::Settings::from(cfg), ins);
  throw std::runtime_error("exchange.name '" + name + "' has no order gateway");
}

std::unique_ptr<HistoricalSource> make_history(const Config& cfg, const Instruments& ins) {
  std::string name = cfg.str("exchange.name");
  if (name == "bybit") return std::make_unique<bybit::History>(bybit::Settings::from(cfg), ins);
  throw std::runtime_error("exchange.name '" + name + "' has no historical source");
}

}  // namespace xrock::exchange
