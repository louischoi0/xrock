#include "exchange/net.hpp"

#include <ixwebsocket/IXHttpClient.h>
#include <zlib.h>

#include <chrono>
#include <stdexcept>
#include <thread>

#include "core/log.hpp"

namespace xrock::exchange::net {

HttpResult http_get(const std::string& url, int attempts) {
  ix::HttpClient client;
  std::string last_error;
  for (int i = 0; i < attempts; ++i) {
    if (i > 0) std::this_thread::sleep_for(std::chrono::seconds(1 << i));
    auto args = client.createRequest(url, ix::HttpClient::kGet);
    args->connectTimeout = 10;
    args->transferTimeout = 300;
    args->followRedirects = true;
    args->compress = false;  // archives are already gzip; JSON is small
    ix::HttpResponsePtr r = client.get(url, args);
    if (r->errorCode == ix::HttpErrorCode::Ok && r->statusCode != 429 && r->statusCode < 500)
      return {r->statusCode, std::move(r->body)};
    last_error = r->errorCode == ix::HttpErrorCode::Ok ? "HTTP " + std::to_string(r->statusCode) : r->errorMsg;
    log::warn("GET {} failed ({}), attempt {}/{}", url, last_error, i + 1, attempts);
  }
  throw std::runtime_error("GET " + url + " failed: " + last_error);
}

std::string gunzip(std::string_view gz) {
  z_stream zs{};
  if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) throw std::runtime_error("gunzip: inflateInit2 failed");
  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(gz.data()));
  zs.avail_in = static_cast<uInt>(gz.size());
  std::string out;
  char buf[1 << 16];
  for (;;) {
    zs.next_out = reinterpret_cast<Bytef*>(buf);
    zs.avail_out = sizeof buf;
    int rc = inflate(&zs, Z_NO_FLUSH);
    out.append(buf, sizeof buf - zs.avail_out);
    if (rc == Z_STREAM_END) {
      if (zs.avail_in == 0) break;
      inflateReset(&zs);  // next gzip member
    } else if (rc != Z_OK) {
      inflateEnd(&zs);
      throw std::runtime_error("gunzip: corrupt or truncated stream");
    }
  }
  inflateEnd(&zs);
  return out;
}

}  // namespace xrock::exchange::net
