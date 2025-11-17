#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"

namespace zk_dtypes {

#define CLASS      \
  PointXyzz<Curve, \
            std::enable_if_t<Curve::kType == CurveType::kShortWeierstrass>>

// static
template <typename Curve>
constexpr void CLASS::Add(const PointXyzz& a, const PointXyzz& b,
                          PointXyzz& c) {
  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
  // U1 = X1 * ZZ2
  BaseField u1 = a.x_ * b.zz_;

  // S1 = Y1 * ZZZ2
  BaseField s1 = a.y_ * b.zzz_;

  // P = X2 * ZZ1 - U1
  BaseField p = b.x_ * a.zz_ - u1;

  // R = Y2 * ZZZ1 - S1
  BaseField r = b.y_ * a.zzz_ - s1;

  if (p.IsZero() && r.IsZero()) {
    c = a.Double();
    return;
  }

  // PP = P²
  BaseField pp = p.Square();

  // PPP = P * PP
  BaseField ppp = p * pp;

  // Q = U1 * PP
  BaseField q = u1 * pp;

  // X3 = R² - PPP - 2 * Q
  c.x_ = r.Square() - ppp - q.Double();

  // Y3 = R * (Q - X3) - S1 * PPP
  c.y_ = r * (q - c.x_) - s1 * ppp;

  // ZZ3 = ZZ1 * ZZ2 * PP
  c.zz_ = a.zz_ * b.zz_ * pp;

  // ZZZ3 = ZZZ1 * ZZZ2 * PPP
  c.zzz_ = a.zzz_ * b.zzz_ * ppp;
}

// static
template <typename Curve>
constexpr void CLASS::Add(const PointXyzz& a, const AffinePoint& b,
                          PointXyzz& c) {
  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s

  // P = X2 * ZZ1 - X1
  BaseField p = b.x() * a.zz_ - a.x_;

  // R = Y2 * ZZZ1 - Y1
  BaseField r = b.y() * a.zzz_ - a.y_;

  if (p.IsZero() && r.IsZero()) {
    c = a.Double();
    return;
  }

  // PP = P²
  BaseField pp = p.Square();

  // PPP = P * PP
  BaseField ppp = p * pp;

  // Q = X1 * PP
  BaseField q = a.x_ * pp;

  // X3 = R² - PPP - 2 * Q
  c.x_ = r.Square() - ppp - q.Double();

  // Y3 = R * (Q - X3) - Y1 * PPP
  c.y_ = r * (q - c.x_) - a.y_ * ppp;

  // ZZ3 = ZZ1 * PP
  c.zz_ = a.zz_ * pp;

  // ZZZ3 = ZZZ1 * PPP
  c.zzz_ = a.zzz_ * ppp;
}

template <typename Curve>
constexpr auto CLASS::Double() const -> PointXyzz {
  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
  // U = 2 * Y1
  BaseField u = y_.Double();

  // V = U²
  BaseField v = u.Square();

  // W = U * V
  BaseField w = u * v;

  // S = X1 * V
  BaseField s = x_ * v;

  // M = 3 * X1² + a * ZZ1²
  BaseField m = x_.Square();
  m += m.Double();
  if constexpr (!Curve::Config::kA.IsZero()) {
    m += Curve::Config::kA * zz_.Square();
  }

  PointXyzz ret;
  // X3 = M² - 2 * S
  ret.x_ = m.Square() - s.Double();

  // Y3 = M * (S - X3) - W * Y1
  ret.y_ = m * (s - ret.x_) - w * y_;

  // ZZ3 = V * ZZ1
  ret.zz_ = v * zz_;

  // ZZZ3 = W * ZZZ1
  ret.zzz_ = w * zzz_;
  return ret;
}

#undef CLASS

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_IMPL_H_
