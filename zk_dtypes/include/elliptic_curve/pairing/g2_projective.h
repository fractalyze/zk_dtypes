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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PROJECTIVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PROJECTIVE_H_

#include <utility>

#include "zk_dtypes/include/elliptic_curve/pairing/ell_coeff.h"
#include "zk_dtypes/include/elliptic_curve/pairing/twist_type.h"

namespace zk_dtypes {

// G2 curve point in homogeneous projective coordinates for Miller loop.
// This class provides efficient point operations that also compute
// line function coefficients needed for pairing computation.
template <typename PairingFriendlyCurveConfig>
class G2Projective {
 public:
  using Config = PairingFriendlyCurveConfig;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;
  using Fp = typename Fp2::BaseField;
  using G2AffinePoint = typename G2Curve::AffinePoint;

  G2Projective() = default;
  G2Projective(const Fp2& x, const Fp2& y, const Fp2& z)
      : x_(x), y_(y), z_(z) {}
  G2Projective(Fp2&& x, Fp2&& y, Fp2&& z)
      : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {}

  const Fp2& x() const { return x_; }
  const Fp2& y() const { return y_; }
  const Fp2& z() const { return z_; }

  // Point addition in projective coordinates.
  // Returns line function coefficients for Miller loop.
  EllCoeff<Fp2> AddInPlace(const G2AffinePoint& q) {
    // Formula for line function when working with
    // homogeneous projective coordinates.
    Fp2 theta = y_ - (q.y() * z_);
    Fp2 lambda = x_ - (q.x() * z_);
    Fp2 c = theta.Square();
    Fp2 d = lambda.Square();
    Fp2 e = lambda * d;
    Fp2 f = z_ * c;
    Fp2 g = x_ * d;
    Fp2 h = e + f - g.Double();
    x_ = lambda * h;
    y_ = theta * (g - h) - (e * y_);
    z_ *= e;
    Fp2 j = theta * q.x() - (lambda * q.y());

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {j, -theta, lambda};
    } else {
      return {lambda, -theta, j};
    }
  }

  // Point doubling in projective coordinates.
  // Returns line function coefficients for Miller loop.
  EllCoeff<Fp2> DoubleInPlace(const Fp& two_inv) {
    // Formula for line function when working with
    // homogeneous projective coordinates.
    Fp2 a = x_ * y_;
    a *= two_inv;
    Fp2 b = y_.Square();
    Fp2 c = z_.Square();
    Fp2 e = G2Curve::Config::kB * (c.Double() + c);
    Fp2 f = e.Double() + e;
    Fp2 g = b + f;
    g *= two_inv;
    Fp2 h = (y_ + z_).Square() - (b + c);
    Fp2 i = e - b;
    Fp2 j = x_.Square();
    Fp2 e_square = e.Square();

    x_ = a * (b - f);
    y_ = g.Square() - (e_square.Double() + e_square);
    z_ = b * h;

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {i, j.Double() + j, -h};
    } else {
      return {-h, j.Double() + j, i};
    }
  }

  G2Projective& NegateInPlace() {
    y_ = -y_;
    return *this;
  }

 private:
  Fp2 x_;
  Fp2 y_;
  Fp2 z_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PROJECTIVE_H_
