#include "core/config.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xrock {
namespace {

std::string_view trim(std::string_view s) {
  constexpr const char* kWs = " \t\r\n";
  auto b = s.find_first_not_of(kWs);
  if (b == std::string_view::npos) return {};
  return s.substr(b, s.find_last_not_of(kWs) - b + 1);
}

std::string_view strip_comment(std::string_view line) {
  for (std::size_t i = 0; i < line.size(); ++i)
    if ((line[i] == '#' || line[i] == ';') && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t'))
      return line.substr(0, i);
  return line;
}

std::string expand_env(std::string_view v) {
  std::string out;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v.substr(i, 2) != "${") {
      out += v[i];
      continue;
    }
    auto end = v.find('}', i + 2);
    if (end == std::string_view::npos) throw std::runtime_error("config: unterminated ${ in " + std::string(v));
    std::string name(v.substr(i + 2, end - i - 2));
    if (const char* e = std::getenv(name.c_str())) out += e;
    i = end;
  }
  return out;
}

}  // namespace

Config Config::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("config: cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse(ss.str());
}

Config Config::parse(std::string_view text) {
  Config c;
  std::string section;
  int lineno = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    auto nl = text.find('\n', pos);
    if (nl == std::string_view::npos) nl = text.size();
    auto line = trim(strip_comment(text.substr(pos, nl - pos)));
    pos = nl + 1;
    ++lineno;
    if (line.empty()) continue;
    if (line.front() == '[' && line.back() == ']') {
      section = trim(line.substr(1, line.size() - 2));
      continue;
    }
    auto eq = line.find('=');
    if (eq == std::string_view::npos) throw std::runtime_error(std::format("config: line {}: expected key = value", lineno));
    std::string key(trim(line.substr(0, eq)));
    c.kv_[section.empty() ? key : section + "." + key] = expand_env(trim(line.substr(eq + 1)));
  }
  return c;
}

std::optional<std::string> Config::find(const std::string& key) const {
  auto it = kv_.find(key);
  if (it == kv_.end()) return std::nullopt;
  return it->second;
}

std::string Config::str(const std::string& key) const {
  if (auto v = find(key)) return *v;
  throw std::runtime_error("config: missing key '" + full(key) + "'");
}

std::string Config::str(const std::string& key, const std::string& def) const { return find(key).value_or(def); }

double Config::num(const std::string& key) const {
  std::string v = str(key);
  char* end = nullptr;
  double x = std::strtod(v.c_str(), &end);
  if (v.empty() || *end != '\0') throw std::runtime_error("config: '" + full(key) + "' is not a number: " + v);
  return x;
}

double Config::num(const std::string& key, double def) const { return find(key) ? num(key) : def; }

bool Config::flag(const std::string& key, bool def) const {
  auto v = find(key);
  if (!v) return def;
  if (*v == "true" || *v == "yes" || *v == "on" || *v == "1") return true;
  if (*v == "false" || *v == "no" || *v == "off" || *v == "0") return false;
  throw std::runtime_error("config: '" + full(key) + "' is not a boolean: " + *v);
}

Duration Config::duration(const std::string& key, Duration def) const {
  auto v = find(key);
  return v ? parse_duration(*v) : def;
}

Timestamp Config::date(const std::string& key) const { return parse_date(str(key)); }

std::vector<std::string> Config::list(const std::string& key) const {
  auto v = find(key);
  return v ? split(*v, ',') : std::vector<std::string>{};
}

Config Config::section(const std::string& prefix) const {
  Config c;
  c.prefix_ = prefix_ + prefix + ".";
  std::string p = prefix + ".";
  for (auto it = kv_.lower_bound(p); it != kv_.end() && it->first.starts_with(p); ++it)
    c.kv_[it->first.substr(p.size())] = it->second;
  return c;
}

std::vector<std::string> split(std::string_view s, char sep) {
  std::vector<std::string> out;
  for (std::size_t pos = 0; pos <= s.size();) {
    auto next = s.find(sep, pos);
    if (next == std::string_view::npos) next = s.size();
    if (auto item = trim(s.substr(pos, next - pos)); !item.empty()) out.emplace_back(item);
    pos = next + 1;
  }
  return out;
}

Duration parse_duration(std::string_view s) {
  std::string str(trim(s));
  char* end = nullptr;
  double x = std::strtod(str.c_str(), &end);
  std::string_view unit(end);
  Duration mult = unit == "ns" ? 1
                  : unit == "us" ? 1000
                  : unit == "ms" ? kMilli
                  : unit == "s"  ? kSecond
                  : unit == "m"  ? kMinute
                  : unit == "h"  ? kHour
                  : unit == "d"  ? kDay
                                 : 0;
  if (end == str.c_str() || mult == 0) throw std::runtime_error("bad duration: '" + str + "'");
  return static_cast<Duration>(std::llround(x * static_cast<double>(mult)));
}

Timestamp parse_date(std::string_view s) {
  using namespace std::chrono;
  std::string str(trim(s));
  int y = 0;
  unsigned m = 0, d = 0;
  year_month_day ymd{year{0}, month{0}, day{0}};
  if (std::sscanf(str.c_str(), "%d-%u-%u", &y, &m, &d) == 3) ymd = year_month_day{year{y}, month{m}, day{d}};
  if (!ymd.ok()) throw std::runtime_error("bad date (want YYYY-MM-DD): '" + str + "'");
  return duration_cast<nanoseconds>(sys_days{ymd}.time_since_epoch()).count();
}

std::string format_date(Timestamp ts) {
  using namespace std::chrono;
  year_month_day ymd{floor<days>(sys_time<nanoseconds>{nanoseconds{ts}})};
  return std::format("{:04}-{:02}-{:02}", int(ymd.year()), unsigned(ymd.month()), unsigned(ymd.day()));
}

}  // namespace xrock
