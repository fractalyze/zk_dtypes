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

#ifndef ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_POINT_H_
#define ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_POINT_H_

#include <cstdint>
#include <string>

#include "zk_dtypes/include/group/group.h"
#include "zk_dtypes/include/str_join.h"

namespace zk_dtypes {

// A point on the complex circle x² + y² = 1. Treated as an additive group.
template <typename F>
struct CirclePoint {
  F x;
  F y;

  constexpr CirclePoint() = default;
  constexpr CirclePoint(const F& x, const F& y) : x(x), y(y) {}

  // Returns the identity element (1, 0).
  constexpr static CirclePoint Zero() { return CirclePoint(F::One(), F::Zero()); }

  // Applies the circle's x-coordinate doubling map: 2x² - 1.
  constexpr static F DoubleX(const F& x) {
    F sx = x.Square();
    return sx + sx - F::One();
  }

  // Returns the log order of a point.
  // All points on the M31 circle have an order of the form 2^k.
  constexpr uint32_t LogOrder() const {
    uint32_t res = 0;
    F cur = x;
    while (cur != F::One()) {
      cur = DoubleX(cur);
      res += 1;
    }
    return res;
  }

  // Doubles the point.
  constexpr CirclePoint Double() const { return *this + *this; }

  // Scalar multiplication using binary method.
  constexpr CirclePoint Mul(uint64_t scalar) const {
    CirclePoint res = Zero();
    CirclePoint cur = *this;
    while (scalar > 0) {
      if (scalar & 1) {
        res = res + cur;
      }
      cur = cur.Double();
      scalar >>= 1;
    }
    return res;
  }

  // Scalar multiplication for 128-bit scalars.
  constexpr CirclePoint Mul128(__uint128_t scalar) const {
    CirclePoint res = Zero();
    CirclePoint cur = *this;
    while (scalar > 0) {
      if (scalar & 1) {
        res = res + cur;
      }
      cur = cur.Double();
      scalar >>= 1;
    }
    return res;
  }

  // Signed scalar multiplication.
  constexpr CirclePoint MulSigned(int64_t off) const {
    if (off > 0) {
      return Mul(static_cast<uint64_t>(off));
    } else {
      return Conjugate().Mul(static_cast<uint64_t>(-off));
    }
  }

  // Repeated doubling.
  constexpr CirclePoint RepeatedDouble(uint32_t n) const {
    CirclePoint res = *this;
    for (uint32_t i = 0; i < n; ++i) {
      res = res.Double();
    }
    return res;
  }

  // Returns the conjugate (x, -y), which is the negation in the group.
  constexpr CirclePoint Conjugate() const { return CirclePoint(x, -y); }

  // Returns the antipode (-x, -y).
  constexpr CirclePoint Antipode() const { return CirclePoint(-x, -y); }

  // Converts to an extension field.
  template <typename EF>
  constexpr CirclePoint<EF> IntoEF() const {
    return CirclePoint<EF>(EF(x), EF(y));
  }

  // Group addition: (x₁, y₁) + (x₂, y₂) = (x₁x₂ - y₁y₂, x₁y₂ + y₁x₂)
  constexpr CirclePoint operator+(const CirclePoint& rhs) const {
    F new_x = x * rhs.x - y * rhs.y;
    F new_y = x * rhs.y + y * rhs.x;
    return CirclePoint(new_x, new_y);
  }

  constexpr CirclePoint& operator+=(const CirclePoint& rhs) {
    *this = *this + rhs;
    return *this;
  }

  // Negation (conjugate).
  constexpr CirclePoint operator-() const { return Conjugate(); }

  // Group subtraction.
  constexpr CirclePoint operator-(const CirclePoint& rhs) const {
    return *this + (-rhs);
  }

  constexpr CirclePoint& operator-=(const CirclePoint& rhs) {
    *this = *this - rhs;
    return *this;
  }

  constexpr bool operator==(const CirclePoint& rhs) const {
    return x == rhs.x && y == rhs.y;
  }

  constexpr bool operator!=(const CirclePoint& rhs) const {
    return !(*this == rhs);
  }

  std::string ToString() const {
    return StrJoin("CirclePoint(", x.ToString(), ", ", y.ToString(), ")");
  }
};

// Register CirclePoint as an additive group.
template <typename F>
struct IsAdditiveGroupImpl<CirclePoint<F>> {
  constexpr static bool value = true;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_POINT_H_
