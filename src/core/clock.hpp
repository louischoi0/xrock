#pragma once

#include <chrono>

#include "core/types.hpp"

namespace xrock {

class Clock {
 public:
  virtual ~Clock() = default;
  virtual Timestamp now() const = 0;
};

// Backtest time, advanced by the replay loop.
class SimClock final : public Clock {
 public:
  Timestamp now() const override { return t_; }
  void set(Timestamp t) { t_ = t; }

 private:
  Timestamp t_ = 0;
};

class WallClock final : public Clock {
 public:
  Timestamp now() const override {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
  }
};

}  // namespace xrock
