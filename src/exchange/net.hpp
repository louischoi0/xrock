#pragma once

#include <string>
#include <string_view>

namespace xrock::exchange::net {

struct HttpResult {
  int status = 0;
  std::string body;
};

// HTTPS GET. Retries transport errors, 429 and 5xx with exponential backoff; other statuses (404, ...) are
// returned to the caller. Throws after the last failed attempt.
HttpResult http_get(const std::string& url, int attempts = 4);

// Decompresses a gzip stream (multi-member allowed).
std::string gunzip(std::string_view gz);

}  // namespace xrock::exchange::net
