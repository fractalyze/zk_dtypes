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

// clang-format off
// G2 curve point in homogeneous projective coordinates (X : Y : Z).
//
// Represents a point (x, y) on G2 as (X : Y : Z) where x = X/Z and y = Y/Z.
// This class provides efficient point operations that simultaneously compute
// line function coefficients needed for pairing computation.
//
// The line function ℓ_{R,Q}(P) represents the line passing through points R
// and Q (or tangent at R for doubling), evaluated at the G1 point P.
// These coefficients are used in the Miller loop to accumulate the pairing.
//
// Reference: https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html
// clang-format on
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

  // clang-format off
  // Mixed addition: R + Q where R is projective, Q is affine.
  // Returns (R + Q, line coefficients ℓ_{R,Q}).
  //
  // The line through R = (X₁ : Y₁ : Z₁) and Q = (x₂, y₂) has slope:
  //   λ = (Y₁ - y₂·Z₁) / (X₁ - x₂·Z₁)
  //
  // Let θ = Y₁ - y₂·Z₁ (numerator of slope)
  //     Λ = X₁ - x₂·Z₁ (denominator of slope)
  //
  // New point R' = R + Q:
  //   X' = Λ · H  where H = Λ² · Z₁ + θ² - 2·Λ²·X₁
  //   Y' = θ · (Λ²·X₁ - H) - Λ³·Y₁
  //   Z' = Λ³ · Z₁
  //
  // Line coefficients encode: ℓ(P) = θ·x₂ - Λ·y₂ + Λ·y_P - θ·x_P
  // clang-format on
  std::pair<G2Projective, EllCoeff<Fp2>> Add(const G2AffinePoint& q) const {
    Fp2 theta = y_ - (q.y() * z_);
    Fp2 lambda = x_ - (q.x() * z_);
    Fp2 c = theta.Square();
    Fp2 d = lambda.Square();
    Fp2 e = lambda * d;
    Fp2 f = z_ * c;
    Fp2 g = x_ * d;
    Fp2 h = e + f - g.Double();
    Fp2 new_x = lambda * h;
    Fp2 new_y = theta * (g - h) - (e * y_);
    Fp2 new_z = z_ * e;
    Fp2 j = theta * q.x() - (lambda * q.y());

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {{std::move(new_x), std::move(new_y), std::move(new_z)},
              {j, -theta, lambda}};
    } else {
      return {{std::move(new_x), std::move(new_y), std::move(new_z)},
              {lambda, -theta, j}};
    }
  }

  // clang-format off
  // Point doubling: 2R where R is projective.
  // Returns (2R, line coefficients ℓ_{R,R} for the tangent line).
  //
  // For curve y² = x³ + b, the tangent at R = (X : Y : Z) has slope:
  //   λ = 3X² / 2Y  (derivative: dy/dx = 3x²/2y)
  //
  // Using projective coordinates with optimizations:
  //   A = X·Y/2
  //   B = Y²
  //   C = Z²
  //   E = b·(C + C + C)  (curve coefficient b times 3C)
  //   F = 3E
  //   G = (B + F)/2
  //   H = (Y + Z)² - B - C  (= 2YZ by binomial expansion)
  //   I = E - B
  //   J = X²
  //
  // New point R' = 2R:
  //   X' = A·(B - F)
  //   Y' = G² - 3E²
  //   Z' = B·H
  //
  // Line coefficients encode the tangent: ℓ(P) = I + 3J·x_P - H·y_P
  // clang-format on
  std::pair<G2Projective, EllCoeff<Fp2>> Double(const Fp& two_inv) const {
    Fp2 a = (x_ * y_) * two_inv;
    Fp2 b = y_.Square();
    Fp2 c = z_.Square();
    Fp2 e = G2Curve::Config::kB * (c.Double() + c);
    Fp2 f = e.Double() + e;
    Fp2 g = (b + f) * two_inv;
    Fp2 h = (y_ + z_).Square() - (b + c);
    Fp2 i = e - b;
    Fp2 j = x_.Square();
    Fp2 e_square = e.Square();

    Fp2 new_x = a * (b - f);
    Fp2 new_y = g.Square() - (e_square.Double() + e_square);
    Fp2 new_z = b * h;

    if constexpr (Config::kTwistType == TwistType::kM) {
      return {{std::move(new_x), std::move(new_y), std::move(new_z)},
              {i, j.Double() + j, -h}};
    } else {
      return {{std::move(new_x), std::move(new_y), std::move(new_z)},
              {-h, j.Double() + j, i}};
    }
  }

  // Point negation: -R = (X : -Y : Z).
  // For elliptic curves, -P reflects the point across the x-axis.
  G2Projective Negate() const { return {x_, -y_, z_}; }

 private:
  Fp2 x_;
  Fp2 y_;
  Fp2 z_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PROJECTIVE_H_
