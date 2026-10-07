cmake_minimum_required(VERSION 3.20)
project(xrock LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

include(cmake/deps.cmake)
add_compile_options(-Wall -Wextra -Wpedantic)

# One static library per layer. target_link_libraries encodes the allowed dependency direction:
#   engine -> strategy, data, analytics -> core;  backtest -> engine;  research -> backtest;  execution -> engine, exchange
function(xrock_module name)
  cmake_parse_arguments(M "" "" "SOURCES;DEPS" ${ARGN})
  add_library(xrock_${name} STATIC ${M_SOURCES})
  target_include_directories(xrock_${name} PUBLIC ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(xrock_${name} PUBLIC ${M_DEPS})
endfunction()

xrock_module(core
  SOURCES src/core/config.cpp src/core/instruments.cpp
  DEPS Threads::Threads)
xrock_module(analytics
  SOURCES src/analytics/journal.cpp src/analytics/metrics.cpp
  DEPS xrock_core)
xrock_module(exchange
  SOURCES src/exchange/registry.cpp src/exchange/synthetic.cpp src/exchange/net.cpp
          src/exchange/bybit/bybit.cpp src/exchange/bybit/public_stream.cpp src/exchange/bybit/history.cpp
  DEPS xrock_core OpenSSL::Crypto ixwebsocket simdjson ZLIB::ZLIB)
xrock_module(data
  SOURCES src/data/store.cpp src/data/file_store.cpp src/data/sqlite_store.cpp src/data/backfill.cpp
          src/data/kds/kwp.cpp src/data/kds/kds_store.cpp
          src/data/replay.cpp src/data/bar_aggregator.cpp
  DEPS xrock_core SQLite::SQLite3)
xrock_module(strategy
  SOURCES src/strategy/registry.cpp src/strategy/ma_cross.cpp src/strategy/trade_flow.cpp src/strategy/buy_hold.cpp
  DEPS xrock_core)
xrock_module(engine
  SOURCES src/engine/portfolio.cpp src/engine/risk.cpp src/engine/engine.cpp
  DEPS xrock_strategy xrock_data xrock_analytics)
xrock_module(backtest
  SOURCES src/backtest/sim_gateway.cpp src/backtest/backtest.cpp
  DEPS xrock_engine)
xrock_module(research
  SOURCES src/research/research.cpp
  DEPS xrock_backtest)
xrock_module(ml
  SOURCES src/ml/features.cpp src/ml/dataset.cpp src/ml/splits.cpp src/ml/model.cpp src/ml/metrics.cpp
          src/ml/model_strategy.cpp src/ml/window.cpp src/ml/runner.cpp src/ml/barrier_strategy.cpp
  DEPS xrock_strategy xrock_data simdjson)
if(XROCK_WITH_ONNX)
  target_sources(xrock_ml PRIVATE src/ml/onnx_model.cpp)
  target_link_libraries(xrock_ml PUBLIC onnxruntime::onnxruntime)
  target_compile_definitions(xrock_ml PRIVATE XROCK_WITH_ONNX=1)
endif()
xrock_module(execution
  SOURCES src/execution/live.cpp
  DEPS xrock_engine xrock_exchange)

add_executable(xrock apps/main.cpp)
target_link_libraries(xrock PRIVATE xrock_research xrock_execution xrock_ml)

option(XROCK_BUILD_TESTS "Build unit tests (GTest)" ON)
if(XROCK_BUILD_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()
  add_executable(xrock_tests
    tests/config_test.cpp tests/engine_test.cpp tests/data_test.cpp
    tests/backtest_test.cpp tests/exchange_test.cpp tests/bybit_parse_test.cpp tests/ml_test.cpp)
  target_link_libraries(xrock_tests PRIVATE xrock_research xrock_execution xrock_ml simdjson GTest::gtest_main)
  target_compile_options(xrock_tests PRIVATE -Wno-missing-field-initializers)  # designated inits omit members
  target_compile_definitions(xrock_tests PRIVATE XROCK_TEST_DATA="${PROJECT_SOURCE_DIR}/tests/data"
                                                 $<$<BOOL:${XROCK_WITH_ONNX}>:XROCK_WITH_ONNX=1>)
  include(GoogleTest)
  gtest_discover_tests(xrock_tests)
endif()
