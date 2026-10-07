#pragma once

#include <chrono>
#include <cstdio>
#include <format>
#include <string>

namespace xrock::log {

enum class Level { Debug, Info, Warn, Error };
inline Level threshold = Level::Info;

template <class... A>
void write(Level lv, std::format_string<A...> fmt, A&&... args) {
  if (lv < threshold) return;
  static constexpr const char* kTag[] = {"DBG", "INF", "WRN", "ERR"};
  auto now = std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now());
  std::string line = std::format("{:%F %T} {} ", now, kTag[static_cast<int>(lv)]);
  line += std::format(fmt, std::forward<A>(args)...);
  line += '\n';
  std::fputs(line.c_str(), stderr);
}

template <class... A>
void debug(std::format_string<A...> f, A&&... a) { write(Level::Debug, f, std::forward<A>(a)...); }
template <class... A>
void info(std::format_string<A...> f, A&&... a) { write(Level::Info, f, std::forward<A>(a)...); }
template <class... A>
void warn(std::format_string<A...> f, A&&... a) { write(Level::Warn, f, std::forward<A>(a)...); }
template <class... A>
void error(std::format_string<A...> f, A&&... a) { write(Level::Error, f, std::forward<A>(a)...); }

}  // namespace xrock::log
