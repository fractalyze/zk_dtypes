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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_OPERATION_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_OPERATION_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

template <typename PointXyzz>
class PointXyzzOperation<PointXyzz,
                         std::enable_if_t<PointTraits<PointXyzz>::kType ==
                                          CurveType::kShortWeierstrass>> {
 public:
  using AffinePoint = typename PointTraits<PointXyzz>::AffinePoint;
  using JacobianPoint = typename PointTraits<PointXyzz>::JacobianPoint;
  using BaseField = typename PointTraits<PointXyzz>::BaseField;

  constexpr PointXyzz operator+(const PointXyzz& other) const {
    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return static_cast<const PointXyzz&>(*this).MaybeConvertToXyzz(cf.If(
        static_cast<const PointXyzz&>(*this).IsZero(),
        [&other]() { return other; },
        [&]() {
          return static_cast<const PointXyzz&>(*this).MaybeConvertToXyzz(cf.If(
              static_cast<const PointXyzz&>(other).IsZero(),
              [this]() { return static_cast<const PointXyzz&>(*this); },
              [this, &other]() { return Add(other); }));
        }));
  }

  constexpr PointXyzz& operator+=(const PointXyzz& other) {
    return static_cast<PointXyzz&>(*this) = operator+(other);
  }

  constexpr PointXyzz operator-(const PointXyzz& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz& operator-=(const PointXyzz& other) {
    return static_cast<PointXyzz&>(*this) = operator-(other);
  }

  constexpr PointXyzz operator+(const AffinePoint& other) const {
    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return static_cast<const PointXyzz&>(*this).MaybeConvertToXyzz(cf.If(
        static_cast<const PointXyzz&>(*this).IsZero(),
        [&other]() { return other.ToXyzz(); },
        [&]() {
          return static_cast<const PointXyzz&>(*this).MaybeConvertToXyzz(cf.If(
              static_cast<const AffinePoint&>(other).IsZero(),
              [this]() { return static_cast<const PointXyzz&>(*this); },
              [this, &other]() { return Add(other); }));
        }));
  }

  constexpr PointXyzz& operator+=(const AffinePoint& other) {
    return static_cast<PointXyzz&>(*this) = operator+(other);
  }

  constexpr PointXyzz operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz& operator-=(const AffinePoint& other) {
    return static_cast<PointXyzz&>(*this) = operator-(other);
  }

  constexpr PointXyzz operator-() const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const PointXyzz&>(*this).ToCoords();
    std::array<BaseField, 4> p2;
    p2[0] = p1[0];
    p2[1] = -p1[1];
    p2[2] = p1[2];
    p2[3] = p1[3];
    return static_cast<const PointXyzz&>(*this).FromCoords(p2);
  }

  constexpr PointXyzz Double() const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const PointXyzz&>(*this).ToCoords();
    std::array<BaseField, 4> p2;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& zz1 = p1[2];
    const BaseField& zzz1 = p1[3];
    BaseField& x2 = p2[0];
    BaseField& y2 = p2[1];
    BaseField& zz2 = p2[2];
    BaseField& zzz2 = p2[3];
    BaseField a = static_cast<const PointXyzz&>(*this).GetA();

    // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
    BaseField u = y1.Double();
    BaseField v = u.Square();
    BaseField w = u * v;
    BaseField s = x1 * v;
    BaseField m = x1.Square();
    m += m.Double();
    m = m + a * zz1.Square();

    x2 = m.Square() - s.Double();
    y2 = m * (s - x2) - w * y1;
    zz2 = v * zz1;
    zzz2 = w * zzz1;
    return static_cast<const PointXyzz&>(*this).FromCoords(p2);
  }

  constexpr auto operator==(const PointXyzz& other) const {
    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return cf.If(
        static_cast<const PointXyzz&>(*this).IsZero(),
        [&other]() { return other.IsZero(); },
        [&]() {
          return cf.If(
              static_cast<const PointXyzz&>(other).IsZero(),
              [&cf]() { return cf.False(); },
              [&]() {
                const std::array<BaseField, 4>& p1 =
                    static_cast<const PointXyzz&>(*this).ToCoords();
                const std::array<BaseField, 4>& p2 =
                    static_cast<const PointXyzz&>(other).ToCoords();
                const BaseField& x1 = p1[0];
                const BaseField& y1 = p1[1];
                const BaseField& zz1 = p1[2];
                const BaseField& zzz1 = p1[3];
                const BaseField& x2 = p2[0];
                const BaseField& y2 = p2[1];
                const BaseField& zz2 = p2[2];
                const BaseField& zzz2 = p2[3];

                // The points (x₁, y₁, zz₁, zzz₁) and (x₂, y₂, zz₂, zzz₂)
                // are equal when (x₁ * zz₂) = (x₂ * zz₁)
                // and (y₁ * zzz₂) = (y₂ * zzz₁).
                return cf.And(cf.Equal(x1 * zz2, x2 * zz1),
                              cf.Equal(y1 * zzz2, y2 * zzz1));
              });
        });
  }

  constexpr auto operator!=(const PointXyzz& other) const {
    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return cf.Not(operator==(other));
  }

  constexpr auto IsZero() const {
    const std::array<BaseField, 4>& p =
        static_cast<const PointXyzz&>(*this).ToCoords();
    const BaseField& zz = p[2];
    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    BaseField zero = zz.CreateConst(0);
    return cf.Equal(zz, zero);
  }

  constexpr auto IsOne() const {
    const std::array<BaseField, 4>& p =
        static_cast<const PointXyzz&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& zz = p[2];
    const BaseField& zzz = p[3];
    const BaseField& gx = static_cast<const PointXyzz&>(*this).GetX();
    const BaseField& gy = static_cast<const PointXyzz&>(*this).GetY();
    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    BaseField one = x.CreateConst(1);
    return cf.And(cf.And(cf.Equal(x, gx), cf.Equal(y, gy)),
                  cf.And(cf.Equal(zz, one), cf.Equal(zzz, one)));
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the affine
  // coordinates as X/ZZ, Y/ZZZ.
  constexpr AffinePoint ToAffine() const {
    const std::array<BaseField, 4>& p =
        static_cast<const PointXyzz&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& zz = p[2];
    const BaseField& zzz = p[3];
    BaseField zero = x.CreateConst(0);

    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return static_cast<const PointXyzz&>(*this).MaybeConvertToAffine(cf.If(
        static_cast<const PointXyzz&>(*this).IsZero(),
        [&]() {
          return static_cast<const PointXyzz&>(*this).CreateAffinePoint(
              std::array<BaseField, 2>{zero, zero});
        },
        [&]() {
          BaseField z_inv_cubic = zzz.Inverse();
          BaseField z_inv_square = (z_inv_cubic * zz).Square();
          return static_cast<const PointXyzz&>(*this).CreateAffinePoint(
              std::array<BaseField, 2>{x * z_inv_square, y * z_inv_cubic});
        }));
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the jacobian
  // coordinates as X, Y, ZZZ/ZZ.
  constexpr JacobianPoint ToJacobian() const {
    const std::array<BaseField, 4>& p =
        static_cast<const PointXyzz&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& zz = p[2];
    const BaseField& zzz = p[3];
    BaseField zero = x.CreateConst(0);

    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return static_cast<const PointXyzz&>(*this).MaybeConvertToJacobian(cf.If(
        static_cast<const PointXyzz&>(*this).IsZero(),
        [&]() {
          return static_cast<const PointXyzz&>(*this).CreateJacobianPoint(
              std::array<BaseField, 3>{zero, zero, zero});
        },
        [&]() {
          BaseField z = zz.Inverse() * zzz;
          return static_cast<const PointXyzz&>(*this).CreateJacobianPoint(
              std::array<BaseField, 3>{x, y, z});
        }));
  }

 private:
  constexpr PointXyzz Add(const PointXyzz& other) const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const PointXyzz&>(*this).ToCoords();
    const std::array<BaseField, 4>& p2 =
        static_cast<const PointXyzz&>(other).ToCoords();
    std::array<BaseField, 4> p3;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& zz1 = p1[2];
    const BaseField& zzz1 = p1[3];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    const BaseField& zz2 = p2[2];
    const BaseField& zzz2 = p2[3];
    BaseField& x3 = p3[0];
    BaseField& y3 = p3[1];
    BaseField& zz3 = p3[2];
    BaseField& zzz3 = p3[3];

    // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
    BaseField u1 = x1 * zz2;
    BaseField s1 = y1 * zzz2;
    BaseField p = x2 * zz1 - u1;
    BaseField r = y2 * zzz1 - s1;

    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return static_cast<const PointXyzz&>(*this).MaybeConvertToXyzz(cf.If(
        cf.And(p.IsZero(), r.IsZero()), [this]() { return Double(); },
        [&]() {
          BaseField pp = p.Square();
          BaseField ppp = p * pp;
          BaseField q = u1 * pp;
          x3 = r.Square() - ppp - q.Double();
          y3 = r * (q - x3) - s1 * ppp;
          zz3 = zz1 * zz2 * pp;
          zzz3 = zzz1 * zzz2 * ppp;

          return static_cast<const PointXyzz&>(*this).FromCoords(p3);
        }));
  }

  constexpr PointXyzz Add(const AffinePoint& other) const {
    const std::array<BaseField, 4>& p1 =
        static_cast<const PointXyzz&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    std::array<BaseField, 4> p3;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& zz1 = p1[2];
    const BaseField& zzz1 = p1[3];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    BaseField& x3 = p3[0];
    BaseField& y3 = p3[1];
    BaseField& zz3 = p3[2];
    BaseField& zzz3 = p3[3];

    // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
    BaseField p = x2 * zz1 - x1;
    BaseField r = y2 * zzz1 - y1;

    auto cf = static_cast<const PointXyzz&>(*this).GetCFOperation();
    return static_cast<const PointXyzz&>(*this).MaybeConvertToXyzz(cf.If(
        cf.And(p.IsZero(), r.IsZero()), [this]() { return Double(); },
        [&]() {
          BaseField pp = p.Square();
          BaseField ppp = p * pp;
          BaseField q = x1 * pp;
          x3 = r.Square() - ppp - q.Double();
          y3 = r * (q - x3) - y1 * ppp;
          zz3 = zz1 * pp;
          zzz3 = zzz1 * ppp;

          return static_cast<const PointXyzz&>(*this).FromCoords(p3);
        }));
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_OPERATION_H_
