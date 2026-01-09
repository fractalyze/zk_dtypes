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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_OPERATION_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_OPERATION_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

template <typename JacobianPoint>
class JacobianPointOperation<
    JacobianPoint, std::enable_if_t<PointTraits<JacobianPoint>::kType ==
                                    CurveType::kShortWeierstrass>> {
 public:
  using AffinePoint = typename PointTraits<JacobianPoint>::AffinePoint;
  using PointXyzz = typename PointTraits<JacobianPoint>::PointXyzz;
  using BaseField = typename PointTraits<JacobianPoint>::BaseField;

  constexpr JacobianPoint operator+(const JacobianPoint& other) const {
    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return static_cast<const JacobianPoint&>(*this).MaybeConvertToJacobian(
        cf.If(
            static_cast<const JacobianPoint&>(*this).IsZero(),
            [&other]() { return other; },
            [&]() {
              return cf.If(
                  static_cast<const JacobianPoint&>(other).IsZero(),
                  [this]() { return static_cast<const JacobianPoint&>(*this); },
                  [this, &other]() { return Add(other); });
            }));
  }

  constexpr JacobianPoint& operator+=(const JacobianPoint& other) {
    return static_cast<JacobianPoint&>(*this) = operator+(other);
  }

  constexpr JacobianPoint operator-(const JacobianPoint& other) const {
    return operator+(-other);
  }

  constexpr JacobianPoint& operator-=(const JacobianPoint& other) {
    return static_cast<JacobianPoint&>(*this) = operator-(other);
  }

  constexpr JacobianPoint operator+(const AffinePoint& other) const {
    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return static_cast<const JacobianPoint&>(*this).MaybeConvertToJacobian(
        cf.If(
            static_cast<const JacobianPoint&>(*this).IsZero(),
            [&other]() { return other.ToJacobian(); },
            [&]() {
              return cf.If(
                  static_cast<const AffinePoint&>(other).IsZero(),
                  [this]() { return static_cast<const JacobianPoint&>(*this); },
                  [this, &other]() { return Add(other); });
            }));
  }

  constexpr JacobianPoint& operator+=(const AffinePoint& other) {
    return static_cast<JacobianPoint&>(*this) = operator+(other);
  }

  constexpr JacobianPoint operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr JacobianPoint& operator-=(const AffinePoint& other) {
    return static_cast<JacobianPoint&>(*this) = operator-(other);
  }

  constexpr JacobianPoint operator-() const {
    const std::array<BaseField, 3>& p1 =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    std::array<BaseField, 3> p2;
    p2[0] = p1[0];
    p2[1] = -p1[1];
    p2[2] = p1[2];
    return static_cast<const JacobianPoint&>(*this).FromCoords(p2);
  }

  constexpr JacobianPoint Double() const {
    const std::array<BaseField, 3>& p1 =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    std::array<BaseField, 3> p2;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& z1 = p1[2];
    BaseField& x2 = p2[0];
    BaseField& y2 = p2[1];
    BaseField& z2 = p2[2];
    BaseField a = static_cast<const JacobianPoint&>(*this).GetA();

    BaseField xx = x1.Square();
    BaseField yy = y1.Square();
    BaseField yyyy = yy.Square();

    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return static_cast<const JacobianPoint&>(*this).MaybeConvertToJacobian(
        cf.If(
            static_cast<const JacobianPoint&>(*this).IsAZero(),
            [&]() {
              // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
              // d = 2 * ((x₁ + yy)² - x₁² - yy²)
              // Can be calculated as d = 4 * x₁ * yy, and this is faster.
              BaseField d = (x1 * yy).Double().Double();
              BaseField e = xx.Double() + xx;
              x2 = e.Square() - d.Double();
              y2 = e * (d - x2) - yyyy.Double().Double().Double();
              z2 = (y1 * z1).Double();
              return static_cast<const JacobianPoint&>(*this).FromCoords(p2);
            },
            [&]() {
              // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
              BaseField zz = z1.Square();

              // s = 2 * ((x₁ + yy)² - x₁² - yy²)
              // Can be calculated as s = 4 * x₁ * yy, and this is faster.
              BaseField s = (x1 * yy).Double().Double();
              BaseField m = (xx.Double() + xx) + a * zz.Square();
              x2 = m.Square() - s.Double();
              y2 = m * (s - x2) - yyyy.Double().Double().Double();

              // z₂ = (y₁ + z₁)² - y₁² - z₁²
              // Can be calculated as z₂ = 2 * y₁ * z₁, and this is faster.
              z2 = (y1 * z1).Double();
              return static_cast<const JacobianPoint&>(*this).FromCoords(p2);
            }));
  }

  constexpr auto operator==(const JacobianPoint& other) const {
    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return cf.If(
        static_cast<const JacobianPoint&>(*this).IsZero(),
        [&other]() { return other.IsZero(); },
        [&]() {
          return cf.If(
              static_cast<const JacobianPoint&>(other).IsZero(),
              [&cf]() { return cf.False(); },
              [&]() {
                const std::array<BaseField, 3>& p1 =
                    static_cast<const JacobianPoint&>(*this).ToCoords();
                const std::array<BaseField, 3>& p2 =
                    static_cast<const JacobianPoint&>(other).ToCoords();
                const BaseField& x1 = p1[0];
                const BaseField& y1 = p1[1];
                const BaseField& z1 = p1[2];
                const BaseField& x2 = p2[0];
                const BaseField& y2 = p2[1];
                const BaseField& z2 = p2[2];

                // The points (x₁, y₁, z₁) and (x₂, y₂, z₂)
                // are equal when (x₁ * z₂²) = (x₂ * z₁²)
                // and (y₁ * z₂³) = (y₂ * z₁³).
                const BaseField z1z1 = z1.Square();
                const BaseField z2z2 = z2.Square();

                return cf.And(cf.Equal(x1 * z2z2, x2 * z1z1),
                              cf.Equal(y1 * (z2z2 * z2), y2 * (z1z1 * z1)));
              });
        });
  }

  constexpr auto operator!=(const JacobianPoint& other) const {
    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return cf.Not(operator==(other));
  }

  constexpr auto IsZero() const {
    const std::array<BaseField, 3>& p =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    const BaseField& z = p[2];
    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    BaseField zero = z.CreateConst(0);
    return cf.Equal(z, zero);
  }

  constexpr auto IsOne() const {
    const std::array<BaseField, 3>& p =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& z = p[2];
    const BaseField& gx = static_cast<const JacobianPoint&>(*this).GetX();
    const BaseField& gy = static_cast<const JacobianPoint&>(*this).GetY();
    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    BaseField one = x.CreateConst(1);
    return cf.And(cf.And(cf.Equal(x, gx), cf.Equal(y, gy)), cf.Equal(z, one));
  }

  // The jacobian point X, Y, Z is represented in the affine
  // coordinates as X/Z², Y/Z³.
  constexpr AffinePoint ToAffine() const {
    const std::array<BaseField, 3>& p =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& z = p[2];
    BaseField zero = x.CreateConst(0);

    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return static_cast<const JacobianPoint&>(*this).MaybeConvertToAffine(cf.If(
        static_cast<const JacobianPoint&>(*this).IsZero(),
        [&]() {
          return static_cast<const JacobianPoint&>(*this).CreateAffinePoint(
              std::array<BaseField, 2>{zero, zero});
        },
        [&]() {
          BaseField z_inv = z.Inverse();
          BaseField z_inv_square = z_inv.Square();
          return static_cast<const JacobianPoint&>(*this).CreateAffinePoint(
              std::array<BaseField, 2>{x * z_inv_square,
                                       y * z_inv_square * z_inv});
        }));
  }

  // The jacobian point X, Y, Z is represented in the xyzz
  // coordinates as X, Y, Z², Z³.
  constexpr PointXyzz ToXyzz() const {
    const std::array<BaseField, 3>& p =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& z = p[2];
    BaseField zz = z.Square();
    return static_cast<const JacobianPoint&>(*this).CreatePointXyzz(
        std::array<BaseField, 4>{x, y, zz, zz * z});
  }

 private:
  constexpr JacobianPoint Add(const JacobianPoint& other) const {
    const std::array<BaseField, 3>& p1 =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    const std::array<BaseField, 3>& p2 =
        static_cast<const JacobianPoint&>(other).ToCoords();
    std::array<BaseField, 3> p3;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& z1 = p1[2];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    const BaseField& z2 = p2[2];
    BaseField& x3 = p3[0];
    BaseField& y3 = p3[1];
    BaseField& z3 = p3[2];

    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
    BaseField z1z1 = z1.Square();
    BaseField z2z2 = z2.Square();
    BaseField u1 = x1 * z2z2;
    BaseField u2 = x2 * z1z1;
    BaseField s1 = y1 * z2 * z2z2;
    BaseField s2 = y2 * z1 * z1z1;

    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return static_cast<const JacobianPoint&>(*this).MaybeConvertToJacobian(
        cf.If(
            cf.And(cf.Equal(u1, u2), cf.Equal(s1, s2)),
            [this]() { return Double(); },
            [&]() {
              BaseField h = u2 - u1;
              BaseField i = h.Double().Square();
              BaseField j = -(h * i);
              BaseField r = (s2 - s1).Double();
              BaseField v = u1 * i;
              x3 = r.Square() + j - v.Double();
              y3 = r * (v - x3) + (s1 * j).Double();

              // z₃ = ((z₁ + z₂)² - z₁² - z₂²) * h
              // Can be calculated as z₃ = 2 * z₁ * z₂ * h, and this is faster.
              z3 = (z1 * z2).Double() * h;

              return static_cast<const JacobianPoint&>(*this).FromCoords(p3);
            }));
  }

  constexpr JacobianPoint Add(const AffinePoint& other) const {
    const std::array<BaseField, 3>& p1 =
        static_cast<const JacobianPoint&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    std::array<BaseField, 3> p3;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& z1 = p1[2];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    BaseField& x3 = p3[0];
    BaseField& y3 = p3[1];
    BaseField& z3 = p3[2];

    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
    BaseField z1z1 = z1.Square();
    BaseField u2 = x2 * z1z1;
    BaseField s2 = y2 * z1 * z1z1;

    auto cf = static_cast<const JacobianPoint&>(*this).GetCFOperation();
    return static_cast<const JacobianPoint&>(*this).MaybeConvertToJacobian(
        cf.If(
            cf.And(cf.Equal(x1, u2), cf.Equal(y1, s2)),
            [this]() { return Double(); },
            [&]() {
              BaseField h = u2 - x1;
              BaseField i = h.Square().Double().Double();
              BaseField j = -(h * i);
              BaseField r = (s2 - y1).Double();
              BaseField v = x1 * i;
              x3 = r.Square() + j - v.Double();
              y3 = r * (v - x3) + (y1 * j).Double();

              // z₃ = (z₁ + h)² - z₁² - h²
              // Can be calculated as z₃ = 2 * z₁ * h, and this is faster.
              z3 = (z1 * h).Double();

              return static_cast<const JacobianPoint&>(*this).FromCoords(p3);
            }));
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_OPERATION_H_
