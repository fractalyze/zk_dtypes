/* Copyright 2026 The zk_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_POINT_INDEX_H_
#define ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_POINT_INDEX_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace zk_dtypes {

// Forward declaration.
template <typename F>
struct CirclePoint;

// Log order of the M31 circle group.
constexpr uint32_t kM31CircleLogOrder = 31;

// Integer i that represents the circle point i * CIRCLE_GEN.
// Treated as an additive ring modulo 2³¹.
class CirclePointIndex {
 public:
  constexpr CirclePointIndex() : index_(0) {}
  constexpr explicit CirclePointIndex(size_t index) : index_(index) {}

  constexpr static CirclePointIndex Zero() { return CirclePointIndex(0); }

  constexpr static CirclePointIndex Generator() { return CirclePointIndex(1); }

  // Reduces the index modulo 2³¹.
  constexpr CirclePointIndex Reduce() const {
    return CirclePointIndex(index_ & ((1ULL << kM31CircleLogOrder) - 1));
  }

  // Returns the generator of the subgroup of size 2^log_size.
  constexpr static CirclePointIndex SubgroupGen(uint32_t log_size) {
    assert(log_size <= kM31CircleLogOrder);
    return CirclePointIndex(1ULL << (kM31CircleLogOrder - log_size));
  }

  // Returns half of the index (must be even).
  constexpr CirclePointIndex Half() const {
    assert((index_ & 1) == 0);
    return CirclePointIndex(index_ >> 1);
  }

  constexpr size_t GetIndex() const { return index_; }

  constexpr CirclePointIndex operator+(CirclePointIndex rhs) const {
    return CirclePointIndex(index_ + rhs.index_).Reduce();
  }

  constexpr CirclePointIndex& operator+=(CirclePointIndex rhs) {
    *this = *this + rhs;
    return *this;
  }

  constexpr CirclePointIndex operator-(CirclePointIndex rhs) const {
    return CirclePointIndex(index_ + (1ULL << kM31CircleLogOrder) - rhs.index_)
        .Reduce();
  }

  constexpr CirclePointIndex& operator-=(CirclePointIndex rhs) {
    *this = *this - rhs;
    return *this;
  }

  constexpr CirclePointIndex operator*(size_t scalar) const {
    // Use wrapping multiplication.
    return CirclePointIndex(index_ * scalar).Reduce();
  }

  constexpr CirclePointIndex operator-() const {
    return CirclePointIndex((1ULL << kM31CircleLogOrder) - index_).Reduce();
  }

  constexpr bool operator==(CirclePointIndex rhs) const {
    return index_ == rhs.index_;
  }

  constexpr bool operator!=(CirclePointIndex rhs) const {
    return index_ != rhs.index_;
  }

  constexpr bool operator<(CirclePointIndex rhs) const {
    return index_ < rhs.index_;
  }

  constexpr bool operator<=(CirclePointIndex rhs) const {
    return index_ <= rhs.index_;
  }

  constexpr bool operator>(CirclePointIndex rhs) const {
    return index_ > rhs.index_;
  }

  constexpr bool operator>=(CirclePointIndex rhs) const {
    return index_ >= rhs.index_;
  }

 private:
  size_t index_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_POINT_INDEX_H_
