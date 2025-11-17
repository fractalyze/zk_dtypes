#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"

namespace zk_dtypes {

#define CLASS    \
  JacobianPoint< \
      Curve, std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

// static
template <typename Curve>
constexpr void CLASS::Add(const JacobianPoint& a, const JacobianPoint& b,
                          JacobianPoint& c) {
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
  // Z1Z1 = Z1²
  BaseField z1z1 = a.z_.Square();

  // Z2Z2 = Z2²
  BaseField z2z2 = b.z_.Square();

  // U1 = X1 * Z2Z2
  BaseField u1 = a.x_ * z2z2;

  // U2 = X2 * Z1Z1
  BaseField u2 = b.x_ * z1z1;

  // S1 = Y1 * Z2 * Z2Z2
  BaseField s1 = a.y_ * b.z_ * z2z2;

  // S2 = Y2 * Z1 * Z1Z1
  BaseField s2 = b.y_ * a.z_ * z1z1;

  if (u1 == u2 && s1 == s2) {
    // The two points are equal, so we Double.
    c = a.Double();
    return;
  }

  // If we're adding -a and a together, c.z_ becomes zero as H becomes zero.

  // H = U2 - U1
  BaseField h = u2 - u1;

  // I = (2 * H)²
  BaseField i = h.Double().Square();

  // J = -H * I
  BaseField j = -(h * i);

  // r = 2 * (S2 - S1)
  BaseField r = (s2 - s1).Double();

  // V = U1 * I
  BaseField v = u1 * i;

  // X3 = r² + J - 2 * V
  c.x_ = r.Square() + j - v.Double();

  // Y3 = r * (V - X3) + 2 * S1 * J
  c.y_ = r * (v - c.x_) + (s1 * j).Double();

  // Z3 = ((Z1 + Z2)² - Z1Z1 - Z2Z2) * H
  // This is equal to Z3 = 2 * Z1 * Z2 * H, and computing it this way is
  // faster.
  c.z_ = (a.z_ * b.z_).Double() * h;
}

// static
template <typename Curve>
constexpr void CLASS::Add(const JacobianPoint& a, const AffinePoint& b,
                          JacobianPoint& c) {
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
  // Z1Z1 = Z1²
  BaseField z1z1 = a.z_.Square();

  // U2 = X2 * Z1Z1
  BaseField u2 = b.x() * z1z1;

  // S2 = Y2 * Z1 * Z1Z1
  BaseField s2 = (b.y() * a.z_) * z1z1;

  if (a.x_ == u2 && a.y_ == s2) {
    c = a.Double();
  } else {
    // If we're adding -a and a together, c.z_ becomes zero as H becomes zero.

    // H = U2 - X1
    BaseField h = u2 - a.x_;

    // I = 4 * H²
    BaseField i = h.Square().Double().Double();

    // J = -H * I
    BaseField j = -(h * i);

    // r = 2 * (S2 - Y1)
    BaseField r = (s2 - a.y_).Double();

    // V = X1 * I
    BaseField v = a.x_ * i;

    // X3 = r² + J - 2 * V
    c.x_ = r.Square() + j - v.Double();

    // Y3 = r * (V - X3) + 2 * Y1 * J
    c.y_ = r * (v - c.x_) + (a.y_ * j).Double();

    // Z3 = 2 * Z1 * H;
    // Can alternatively be computed as (Z1 + H)² - Z1Z1 - HH, but the latter is
    // slower.
    c.z_ = (a.z_ * h).Double();
  }
}

template <typename Curve>
constexpr auto CLASS::Double() const -> JacobianPoint {
  JacobianPoint ret;
  if constexpr (Curve::Config::kA.IsZero()) {
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
    // XX = X1²
    BaseField xx = x_.Square();

    // YY = Y1²
    BaseField yy = y_.Square();

    // YYYY = YY²
    BaseField yyyy = yy.Square();

    // D = 2 * ((X1 + YY)² - XX - YYYY)
    BaseField d = ((x_ + yy).Square() - xx - yyyy).Double();

    // E = 3 * XX
    BaseField e = xx.Double() + xx;

    // Z3 = 2 * Y1 * Z1
    ret.z_ = (y_ * z_).Double();

    // X3 = E² - 2 * D
    ret.x_ = e.Square() - d.Double();

    // Y3 = E * (D - X3) - 8 * YYYY
    ret.y_ = e * (d - ret.x_) - yyyy.Double().Double().Double();
  } else {
    // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
    // XX = X1²
    BaseField xx = x_.Square();

    // YY = Y1²
    BaseField yy = y_.Square();

    // YYYY = YY²
    BaseField yyyy = yy.Square();

    // ZZ = Z1²
    BaseField zz = z_.Square();

    // S = 2 * ((X1 + YY)² - XX - YYYY)
    BaseField s = ((x_ + yy).Square() - xx - yyyy).Double();

    // M = 3 * XX + a * ZZ²
    BaseField m = (xx.Double() + xx) + Curve::Config::kA * zz.Square();

    // T = M² - 2 * S
    // X3 = T
    ret.x_ = m.Square() - s.Double();

    // Z3 = (Y1 + Z1)² - YY - ZZ
    // Can be calculated as Z3 = 2 * Y1 * Z1, and this is faster.
    ret.z_ = (y_ + z_).Double();

    // Y3 = M * (S - X3) - 8 * YYYY
    ret.y_ = m * (s - ret.x_) - yyyy.Double().Double().Double();
  }
  return ret;
}

#undef CLASS

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_IMPL_H_
