#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.hpp"

namespace xrock {

// INI-style config: "[section]" + "key = value", addressed as "section.key".
// "#" or ";" at line start or after whitespace starts a comment.
// "${VAR}" expands from the environment, so API secrets stay out of the file.
class Config {
 public:
  static Config load(const std::string& path);
  static Config parse(std::string_view text);

  std::optional<std::string> find(const std::string& key) const;
  std::string str(const std::string& key) const;  // throws if missing
  std::string str(const std::string& key, const std::string& def) const;
  double num(const std::string& key) const;
  double num(const std::string& key, double def) const;
  bool flag(const std::string& key, bool def) const;
  Duration duration(const std::string& key, Duration def) const;  // "250ms", "5s", "1m", "4h", "1d"
  Timestamp date(const std::string& key) const;                   // "YYYY-MM-DD" -> UTC midnight
  std::vector<std::string> list(const std::string& key) const;    // comma separated; empty if missing

  Config section(const std::string& prefix) const;  // keys under "prefix.", prefix stripped
  void set(const std::string& key, std::string value) { kv_[key] = std::move(value); }
  const std::map<std::string, std::string>& entries() const { return kv_; }

 private:
  std::string full(const std::string& key) const { return prefix_ + key; }  // for error messages

  std::map<std::string, std::string> kv_;
  std::string prefix_;
};

std::vector<std::string> split(std::string_view s, char sep);  // trimmed, empty items dropped
Duration parse_duration(std::string_view);
Timestamp parse_date(std::string_view);
std::string format_date(Timestamp);

}  // namespace xrock
