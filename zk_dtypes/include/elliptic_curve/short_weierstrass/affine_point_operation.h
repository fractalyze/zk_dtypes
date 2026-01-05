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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_OPERATION_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_OPERATION_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

template <typename AffinePoint>
class AffinePointOperation<AffinePoint,
                           std::enable_if_t<PointTraits<AffinePoint>::kType ==
                                            CurveType::kShortWeierstrass>> {
 public:
  using JacobianPoint = typename PointTraits<AffinePoint>::JacobianPoint;
  using PointXyzz = typename PointTraits<AffinePoint>::PointXyzz;
  using BaseField = typename PointTraits<AffinePoint>::BaseField;

  constexpr JacobianPoint operator+(const AffinePoint& other) const {
    return AddToJacobian(other);
  }

  constexpr JacobianPoint AddToJacobian(const AffinePoint& other) const {
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return static_cast<const AffinePoint&>(*this).MaybeConvertToJacobian(cf.If(
        static_cast<const AffinePoint&>(*this).IsZero(),
        [&other]() { return other.ToJacobian(); },
        [&]() {
          return cf.If(
              static_cast<const AffinePoint&>(other).IsZero(),
              [this]() { return ToJacobian(); },
              [this, &other]() { return DoAddToJacobian(other); });
        }));
  }

  constexpr PointXyzz AddToXyzz(const AffinePoint& other) const {
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return static_cast<const AffinePoint&>(*this).MaybeConvertToXyzz(cf.If(
        static_cast<const AffinePoint&>(*this).IsZero(),
        [&other]() { return other.ToXyzz(); },
        [&]() {
          return cf.If(
              static_cast<const AffinePoint&>(other).IsZero(),
              [this]() { return ToXyzz(); },
              [this, &other]() { return DoAddToXyzz(other); });
        }));
  }

  constexpr JacobianPoint operator+(const JacobianPoint& other) const {
    return other + static_cast<const AffinePoint&>(*this);
  }

  constexpr PointXyzz operator+(const PointXyzz& other) const {
    return other + static_cast<const AffinePoint&>(*this);
  }

  constexpr JacobianPoint operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr JacobianPoint SubToJacobian(const AffinePoint& other) const {
    return AddToJacobian(-other);
  }

  constexpr PointXyzz SubToXyzz(const AffinePoint& other) const {
    return AddToXyzz(-other);
  }

  constexpr JacobianPoint operator-(const JacobianPoint& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz operator-(const PointXyzz& other) const {
    return operator+(-other);
  }

  constexpr AffinePoint operator-() const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    std::array<BaseField, 2> p2;
    p2[0] = p1[0];
    p2[1] = -p1[1];
    return static_cast<const AffinePoint&>(*this).FromCoords(p2);
  }

  constexpr JacobianPoint Double() const { return DoubleToJacobian(); }

  constexpr JacobianPoint DoubleToJacobian() const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    std::array<BaseField, 3> p2;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    BaseField& x2 = p2[0];
    BaseField& y2 = p2[1];
    BaseField& z2 = p2[2];
    BaseField a = static_cast<const AffinePoint&>(*this).GetA();

    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-mdbl-2007-bl
    BaseField xx = x1.Square();
    BaseField yy = y1.Square();
    BaseField yyyy = yy.Square();

    // s = 2 * ((x₁ + yy)² - xx - yyyy)
    // Can be calculated as s = 2 * x₁ * yy, and this is faster.
    BaseField s = (x1 * yy).Double().Double();
    BaseField m = xx.Double() + xx + a;
    BaseField t = m.Square() - s.Double();
    x2 = t;
    y2 = m * (s - t) - yyyy.Double().Double().Double();
    z2 = y1.Double();

    return static_cast<const AffinePoint&>(*this).CreateJacobianPoint(
        std::array<BaseField, 3>{x2, y2, z2});
  }

  constexpr PointXyzz DoubleToXyzz() const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    std::array<BaseField, 4> p2;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    BaseField& x2 = p2[0];
    BaseField& y2 = p2[1];
    BaseField& zz2 = p2[2];
    BaseField& zzz2 = p2[3];
    BaseField a = static_cast<const AffinePoint&>(*this).GetA();

    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
    BaseField u = y1.Double();
    BaseField v = u.Square();
    BaseField w = u * v;
    BaseField s = x1 * v;
    BaseField m = x1.Square();
    m += m.Double();
    m += a;
    x2 = m.Square() - s.Double();
    y2 = m * (s - x2) - w * y1;
    zz2 = v;
    zzz2 = w;
    return static_cast<const AffinePoint&>(*this).CreatePointXyzz(
        std::array<BaseField, 4>{x2, y2, zz2, zzz2});
  }

  constexpr auto operator==(const AffinePoint& other) const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];

    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return cf.And(cf.Equal(x1, x2), cf.Equal(y1, y2));
  }

  constexpr auto operator!=(const AffinePoint& other) const {
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return cf.Not(operator==(other));
  }

  constexpr auto IsZero() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    BaseField zero = x.CreateConst(0);
    return cf.And(cf.Equal(x, zero), cf.Equal(y, zero));
  }

  constexpr auto IsOne() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];
    const BaseField& gx = static_cast<const AffinePoint&>(*this).GetX();
    const BaseField& gy = static_cast<const AffinePoint&>(*this).GetY();
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return cf.And(cf.Equal(x, gx), cf.Equal(y, gy));
  }

  constexpr JacobianPoint ToJacobian() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];

    BaseField one = x.CreateConst(1);
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return static_cast<const AffinePoint&>(*this).MaybeConvertToJacobian(cf.If(
        cf.And(x.IsZero(), y.IsZero()),
        [&]() {
          BaseField zero = x.CreateConst(0);
          return static_cast<const AffinePoint&>(*this).CreateJacobianPoint(
              std::array<BaseField, 3>{one, one, zero});
        },
        [&]() {
          return static_cast<const AffinePoint&>(*this).CreateJacobianPoint(
              std::array<BaseField, 3>{x, y, one});
        }));
  }

  constexpr PointXyzz ToXyzz() const {
    const std::array<BaseField, 2>& p =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const BaseField& x = p[0];
    const BaseField& y = p[1];

    BaseField one = x.CreateConst(1);
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return static_cast<const AffinePoint&>(*this).MaybeConvertToXyzz(cf.If(
        cf.And(x.IsZero(), y.IsZero()),
        [&]() {
          BaseField zero = x.CreateConst(0);
          return static_cast<const AffinePoint&>(*this).CreatePointXyzz(
              std::array<BaseField, 4>{one, one, zero, zero});
        },
        [&]() {
          return static_cast<const AffinePoint&>(*this).CreatePointXyzz(
              std::array<BaseField, 4>{x, y, one, one});
        }));
  }

 private:
  constexpr JacobianPoint DoAddToJacobian(const AffinePoint& other) const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    std::array<BaseField, 3> p3;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    BaseField& x3 = p3[0];
    BaseField& y3 = p3[1];
    BaseField& z3 = p3[2];

    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-mmadd-2007-bl
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return static_cast<const AffinePoint&>(*this).MaybeConvertToJacobian(cf.If(
        cf.And(cf.Equal(x1, x2), cf.Equal(y1, y2)),
        [this]() { return DoubleToJacobian(); },
        [&]() {
          BaseField h = x2 - x1;
          BaseField hh = h.Square();
          BaseField i = hh.Double().Double();
          BaseField j = h * i;
          BaseField r = (y2 - y1).Double();
          BaseField v = x1 * i;
          x3 = r.Square() - j - v.Double();
          y3 = r * (v - x3) - (y1 * j).Double();
          z3 = h.Double();
          return static_cast<const AffinePoint&>(*this).CreateJacobianPoint(
              std::array<BaseField, 3>{x3, y3, z3});
        }));
  }

  constexpr PointXyzz DoAddToXyzz(const AffinePoint& other) const {
    const std::array<BaseField, 2>& p1 =
        static_cast<const AffinePoint&>(*this).ToCoords();
    const std::array<BaseField, 2>& p2 =
        static_cast<const AffinePoint&>(other).ToCoords();
    std::array<BaseField, 4> p3;
    const BaseField& x1 = p1[0];
    const BaseField& y1 = p1[1];
    const BaseField& x2 = p2[0];
    const BaseField& y2 = p2[1];
    BaseField& x3 = p3[0];
    BaseField& y3 = p3[1];
    BaseField& zz3 = p3[2];
    BaseField& zzz3 = p3[3];

    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-mmadd-2008-s
    auto cf = static_cast<const AffinePoint&>(*this).GetCFOperation();
    return static_cast<const AffinePoint&>(*this).MaybeConvertToXyzz(cf.If(
        cf.And(cf.Equal(x1, x2), cf.Equal(y1, y2)),
        [this]() { return DoubleToXyzz(); },
        [&]() {
          BaseField p = x2 - x1;
          BaseField r = y2 - y1;
          BaseField pp = p.Square();
          BaseField ppp = p * pp;
          BaseField q = x1 * pp;
          x3 = r.Square() - ppp - q.Double();
          y3 = r * (q - x3) - y1 * ppp;
          zz3 = pp;
          zzz3 = ppp;
          return static_cast<const AffinePoint&>(*this).CreatePointXyzz(
              std::array<BaseField, 4>{x3, y3, zz3, zzz3});
        }));
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_OPERATION_H_
