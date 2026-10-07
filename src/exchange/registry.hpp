#pragma once

#include <memory>

#include "core/config.hpp"
#include "core/instruments.hpp"
#include "core/interfaces.hpp"

namespace xrock::exchange {

// Venue adapters, selected by [exchange] name.
// Add a venue: implement MarketDataFeed / OrderGateway under exchange/<venue>/ and add a branch in registry.cpp.
std::unique_ptr<MarketDataFeed> make_feed(const Config&, const Instruments&);
std::unique_ptr<OrderGateway> make_gateway(const Config&, const Instruments&);
std::unique_ptr<HistoricalSource> make_history(const Config&, const Instruments&);

}  // namespace xrock::exchange
