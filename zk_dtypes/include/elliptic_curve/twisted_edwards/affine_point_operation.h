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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_AFFINE_POINT_OPERATION_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_AFFINE_POINT_OPERATION_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

// CRTP operation base for AffinePoint on a twisted Edwards curve. Mirrors the
// short_weierstrass AffinePointOperation specialization but uses the unified
// (complete) twisted Edwards addition formula
//   x3 = (x1 * y2 + x2 * y1) / (1 + d * x1 * x2 * y1 * y2)
//   y3 = (y1 * y2 - a * x1 * x2) / (1 - d * x1 * x2 * y1 * y2)
// which is valid for ALL pairs of points (including the identity and inverses)
// when the curve has prime-order subgroup with d a non-square — true for
// Ed25519. There is therefore no need for IsZero / equality branching as in
// the short Weierstrass formulas.
template <typename AffinePoint>
class AffinePointOperation<AffinePoint,
                           std::enable_if_t<PointTraits<AffinePoint>::kType ==
                                            CurveType::kTwistedEdwards>> {
 public:
  using ExtendedPoint = typename PointTraits<AffinePoint>::ExtendedPoint;
  using BaseField = typename PointTraits<AffinePoint>::BaseField;

  constexpr ExtendedPoint operator+(const AffinePoint& other) const {
    return AddToExtended(other);
  }

  constexpr ExtendedPoint AddToExtended(const AffinePoint& other) const {
    return DoAddToExtended(other);
  }

  constexpr ExtendedPoint operator+(const ExtendedPoint& other) const {
    return other + static_cast<const AffinePoint&>(*this);
  }

  constexpr ExtendedPoint operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr ExtendedPoint operator-(const ExtendedPoint& other) const {
    return operator+(-other);
  }

  // For twisted Edwards: -P = (-x, y).
  constexpr AffinePoint operator-() const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    std::array<BaseField, 2> p2;
    p2[0] = -p1[0];
    p2[1] = p1[1];
    return static_cast<const AffinePoint&>(*this).FromCoords(p2);
  }

  constexpr ExtendedPoint Double() const { return DoubleToExtended(); }

  constexpr ExtendedPoint DoubleToExtended() const {
    return ToExtended().Double();
  }

  constexpr auto operator==(const AffinePoint& other) const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return cf.And(cf.Equal(p1[0], p2[0]), cf.Equal(p1[1], p2[1]));
  }

  constexpr auto operator!=(const AffinePoint& other) const {
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return cf.Not(operator==(other));
  }

  // Identity element of the twisted Edwards group is (0, 1), not (0, 0).
  constexpr auto IsZero() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    BaseField zero = p[0].CreateConst(0);
    BaseField one = p[0].CreateConst(1);
    return cf.And(cf.Equal(p[0], zero), cf.Equal(p[1], one));
  }

  constexpr auto IsOne() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const BaseField& gx = static_cast<const AffinePoint&>(*this).GetX();
    const BaseField& gy = static_cast<const AffinePoint&>(*this).GetY();
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return cf.And(cf.Equal(p[0], gx), cf.Equal(p[1], gy));
  }

  // Affine -> Extended: (x, y) -> (x, y, 1, x*y).
  constexpr ExtendedPoint ToExtended() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    BaseField one = p[0].CreateConst(1);
    return static_cast<const AffinePoint&>(*this).MaybeConvertToExtended(
        static_cast<const AffinePoint&>(*this).CreateExtendedPoint(
            std::array<BaseField, 4>{p[0], p[1], one, p[0] * p[1]}));
  }

 private:
  // Unified affine addition. Two field inversions; only used for testing /
  // golden checks. Production code should prefer the extended-coordinate path.
  constexpr ExtendedPoint DoAddToExtended(const AffinePoint& other) const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    BaseField a = static_cast<const AffinePoint&>(*this).GetA();
    BaseField d = static_cast<const AffinePoint&>(*this).GetD();

    BaseField x1y2 = x1 * y2;
    BaseField y1x2 = y1 * x2;
    BaseField y1y2 = y1 * y2;
    BaseField x1x2 = x1 * x2;
    BaseField dxxyy = d * x1x2 * y1y2;
    BaseField one = x1.CreateConst(1);
    BaseField num_x = x1y2 + y1x2;
    BaseField num_y = y1y2 - a * x1x2;
    BaseField den_x = (one + dxxyy).Inverse();
    BaseField den_y = (one - dxxyy).Inverse();
    BaseField x3 = num_x * den_x;
    BaseField y3 = num_y * den_y;
    BaseField z3 = one;
    BaseField t3 = x3 * y3;
    return static_cast<const AffinePoint&>(*this).MaybeConvertToExtended(
        static_cast<const AffinePoint&>(*this).CreateExtendedPoint(
            std::array<BaseField, 4>{x3, y3, z3, t3}));
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_AFFINE_POINT_OPERATION_H_
