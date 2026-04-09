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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_EXTENDED_POINT_OPERATION_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_EXTENDED_POINT_OPERATION_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

// Extended-coordinate operations for twisted Edwards curves using
// Hisil-Wong-Carter-Dawson (HWCD-2008) formulas. Extended coordinates
// represent a point as (X : Y : Z : T) with the invariant T = X * Y / Z
// and affine coordinates recovered as x = X / Z, y = Y / Z.
//
// Reference formulas:
//   https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html
//
// The addition formula is unified (works for all input pairs including
// identity and inverses) when the curve has d non-square, which is the case
// for all standard twisted Edwards curves (Ed25519, etc.).
template <typename Point>
class ExtendedPointOperation<
    Point,
    std::enable_if_t<PointTraits<Point>::kType == CurveType::kTwistedEdwards>> {
 public:
  using AffinePoint = typename PointTraits<Point>::AffinePoint;
  using ExtendedPoint = typename PointTraits<Point>::ExtendedPoint;
  using BaseField = typename PointTraits<Point>::BaseField;

  // Unified addition.
  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-unified-2008-hcd
  constexpr ExtendedPoint operator+(const ExtendedPoint& other) const {
    return DoAdd(other);
  }

  // Mixed addition: Affine -> Extended -> add.
  constexpr ExtendedPoint operator+(const AffinePoint& other) const {
    return DoAdd(other.ToExtended());
  }

  constexpr ExtendedPoint operator-(const ExtendedPoint& other) const {
    return operator+(-other);
  }

  constexpr ExtendedPoint operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  // For twisted Edwards: -(X, Y, Z, T) = (-X, Y, Z, -T).
  constexpr ExtendedPoint operator-() const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const ExtendedPoint&>(*this).ToCoords();
    return static_cast<const ExtendedPoint&>(*this).FromCoords(
        {-p1[0], p1[1], p1[2], -p1[3]});
  }

  // Dedicated doubling formula (faster than adding to self).
  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd
  constexpr ExtendedPoint Double() const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const ExtendedPoint&>(*this).ToCoords();
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& z1 = p1[2];
    BaseField a = static_cast<const ExtendedPoint&>(*this).GetA();

    // A = X1²
    BaseField aa = x1.Square();
    // B = Y1²
    BaseField bb = y1.Square();
    // C = 2 * Z1²
    BaseField cc = z1.Square().Double();
    // D = a * A
    BaseField dd = a * aa;
    // E = (X1 + Y1)² - A - B
    BaseField ee = (x1 + y1).Square() - aa - bb;
    // G = D + B
    BaseField gg = dd + bb;
    // F = G - C
    BaseField ff = gg - cc;
    // H = D - B
    BaseField hh = dd - bb;
    // X3 = E * F
    BaseField x3 = ee * ff;
    // Y3 = G * H
    BaseField y3 = gg * hh;
    // T3 = E * H
    BaseField t3 = ee * hh;
    // Z3 = F * G
    BaseField z3 = ff * gg;

    return static_cast<const ExtendedPoint&>(*this).MaybeConvertToExtended(
        static_cast<const ExtendedPoint&>(*this).CreateExtendedPoint(
            {x3, y3, z3, t3}));
  }

  constexpr auto operator==(const ExtendedPoint& other) const {
    // Two projective points are equal iff X1 * Z2 == X2 * Z1 and
    // Y1 * Z2 == Y2 * Z1.
    const std::array<BaseField, 4>& p1 =
        static_cast<const ExtendedPoint&>(*this).ToCoords();
    const std::array<BaseField, 4>& p2 =
        static_cast<const ExtendedPoint&>(other).ToCoords();
    auto cf = static_cast<const ExtendedPoint&>(*this).GetCFOperation();
    return cf.And(cf.Equal(p1[0] * p2[2], p2[0] * p1[2]),
                  cf.Equal(p1[1] * p2[2], p2[1] * p1[2]));
  }

  constexpr auto operator!=(const ExtendedPoint& other) const {
    auto cf = static_cast<const ExtendedPoint&>(*this).GetCFOperation();
    return cf.Not(operator==(other));
  }

  // Identity in extended coords: (0, 1, 1, 0).
  constexpr auto IsZero() const {
    return static_cast<const ExtendedPoint&>(*this) == ExtendedPoint::Zero();
  }

  // Extended -> Affine: x = X / Z, y = Y / Z.
  constexpr AffinePoint ToAffine() const {
    const std::array<BaseField, 4>& p =
        static_cast<const ExtendedPoint&>(*this).ToCoords();
    BaseField z_inv = p[2].Inverse();
    return static_cast<const ExtendedPoint&>(*this).MaybeConvertToAffine(
        static_cast<const ExtendedPoint&>(*this).CreateAffinePoint(
            {p[0] * z_inv, p[1] * z_inv}));
  }

 private:
  // Unified addition.
  // https://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-unified-2008-hcd
  constexpr ExtendedPoint DoAdd(const ExtendedPoint& other) const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const ExtendedPoint&>(*this).ToCoords();
    const std::array<BaseField, 4>& p2 =
        static_cast<const ExtendedPoint&>(other).ToCoords();
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& z1 = p1[2];
    const BaseField& t1 = p1[3];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    const BaseField& z2 = p2[2];
    const BaseField& t2 = p2[3];
    BaseField a = static_cast<const ExtendedPoint&>(*this).GetA();
    BaseField d = static_cast<const ExtendedPoint&>(*this).GetD();

    // A = X1 * X2
    BaseField aa = x1 * x2;
    // B = Y1 * Y2
    BaseField bb = y1 * y2;
    // C = T1 * d * T2
    BaseField cc = t1 * d * t2;
    // D = Z1 * Z2
    BaseField dd = z1 * z2;
    // E = (X1 + Y1) * (X2 + Y2) - A - B
    BaseField ee = (x1 + y1) * (x2 + y2) - aa - bb;
    // F = D - C
    BaseField ff = dd - cc;
    // G = D + C
    BaseField gg = dd + cc;
    // H = B - a * A
    BaseField hh = bb - a * aa;
    // X3 = E * F
    BaseField x3 = ee * ff;
    // Y3 = G * H
    BaseField y3 = gg * hh;
    // T3 = E * H
    BaseField t3 = ee * hh;
    // Z3 = F * G
    BaseField z3 = ff * gg;

    return static_cast<const ExtendedPoint&>(*this).MaybeConvertToExtended(
        static_cast<const ExtendedPoint&>(*this).CreateExtendedPoint(
            {x3, y3, z3, t3}));
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_EXTENDED_POINT_OPERATION_H_
