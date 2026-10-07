# Third-party dependencies. System packages are preferred; anything missing is built from source, so a fresh
# checkout builds without root. Included before the project's warning flags so they don't apply to vendored code.
include(FetchContent)
cmake_policy(SET CMP0135 NEW)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)  # Bybit signing, TLS for IXWebSocket

find_package(SQLite3 QUIET)  # [data] backend = sqlite
if(NOT SQLite3_FOUND)
  message(STATUS "SQLite3 not found: building the amalgamation")
  FetchContent_Declare(sqlite URL https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip)
  FetchContent_MakeAvailable(sqlite)
  add_library(sqlite3_vendored STATIC ${sqlite_SOURCE_DIR}/sqlite3.c)
  target_include_directories(sqlite3_vendored PUBLIC ${sqlite_SOURCE_DIR})
  target_link_libraries(sqlite3_vendored PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
  add_library(SQLite::SQLite3 ALIAS sqlite3_vendored)
endif()

find_package(ZLIB QUIET)  # history archives are .csv.gz
if(NOT ZLIB_FOUND)
  message(STATUS "zlib headers not found: building zlib")
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(zlib URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
                       EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(zlib)
  target_include_directories(zlibstatic PUBLIC ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
  add_library(ZLIB::ZLIB ALIAS zlibstatic)
endif()

# WebSocket + HTTPS client (OpenSSL). Its own permessage-deflate support is not needed.
set(USE_TLS ON CACHE BOOL "" FORCE)
set(USE_OPEN_SSL ON CACHE BOOL "" FORCE)
set(USE_ZLIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ixwebsocket URL https://github.com/machinezone/IXWebSocket/archive/refs/tags/v11.4.5.tar.gz
                     EXCLUDE_FROM_ALL)
FetchContent_Declare(simdjson URL https://github.com/simdjson/simdjson/archive/refs/tags/v3.10.1.tar.gz
                     EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(ixwebsocket simdjson)

# ONNX Runtime (C++ inference for "onnx" models). Prebuilt CPU release, pinned by hash.
option(XROCK_WITH_ONNX "Link ONNX Runtime for onnx models" ON)
if(XROCK_WITH_ONNX)
  if(NOT TARGET onnxruntime::onnxruntime)
    FetchContent_Declare(onnxruntime_bin
      URL https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz
      URL_HASH SHA256=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd)
    FetchContent_MakeAvailable(onnxruntime_bin)
    # The tarball's own onnxruntimeConfig.cmake points at lib64/, which the archive does not have.
    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION ${onnxruntime_bin_SOURCE_DIR}/lib/libonnxruntime.so.1.30.0
      IMPORTED_SONAME libonnxruntime.so.1
      INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_bin_SOURCE_DIR}/include)
  endif()
endif()
