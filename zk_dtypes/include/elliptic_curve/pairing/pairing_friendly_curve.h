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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_PAIRING_FRIENDLY_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_PAIRING_FRIENDLY_CURVE_H_

#include <vector>

#include "zk_dtypes/include/elliptic_curve/pairing/ell_coeff.h"
#include "zk_dtypes/include/elliptic_curve/pairing/twist_type.h"

namespace zk_dtypes {

// Base class for pairing-friendly curves.
// Provides common infrastructure for Miller loop and final exponentiation.
template <typename _Config>
class PairingFriendlyCurve {
 public:
  using Config = _Config;
  using G1Curve = typename Config::G1Curve;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;
  using Fp12 = typename Config::Fp12;
  using G1AffinePoint = typename G1Curve::AffinePoint;

 protected:
  // Pair holds a G1 point and its corresponding precomputed G2 coefficients.
  class Pair {
   public:
    Pair() = default;
    Pair(const G1AffinePoint* g1, const std::vector<EllCoeff<Fp2>>* ell_coeffs)
        : g1_(g1), ell_coeffs_(ell_coeffs) {}

    const G1AffinePoint& g1() const { return *g1_; }
    const EllCoeff<Fp2>& NextEllCoeff() const { return (*ell_coeffs_)[idx_++]; }

   private:
    const G1AffinePoint* g1_ = nullptr;
    const std::vector<EllCoeff<Fp2>>* ell_coeffs_ = nullptr;
    mutable size_t idx_ = 0;
  };

  // f^x where x is the BN parameter.
  static Fp12 PowByX(const Fp12& f_in) {
    Fp12 f = f_in.CyclotomicPow(Config::kX);
    if constexpr (Config::kXIsNegative) {
      f = f.CyclotomicInverse();
    }
    return f;
  }

  // f^(-x) where x is the BN parameter.
  static Fp12 PowByNegX(const Fp12& f_in) {
    Fp12 f = f_in.CyclotomicPow(Config::kX);
    if constexpr (!Config::kXIsNegative) {
      f = f.CyclotomicInverse();
    }
    return f;
  }

  // Evaluates the line function at point |p|.
  static void Ell(Fp12& f, const EllCoeff<Fp2>& coeffs,
                  const G1AffinePoint& p) {
    if constexpr (Config::kTwistType == TwistType::kM) {
      f = f.MulBy014(coeffs.c0(), coeffs.c1() * p.x(), coeffs.c2() * p.y());
    } else {
      f = f.MulBy034(coeffs.c0() * p.y(), coeffs.c1() * p.x(), coeffs.c2());
    }
  }

  // Creates pairs of (G1 point, precomputed G2 coefficients).
  template <typename G1AffinePointContainer, typename G2PreparedContainer>
  static std::vector<Pair> CreatePairs(const G1AffinePointContainer& a,
                                       const G2PreparedContainer& b) {
    size_t size = std::size(a);
    std::vector<Pair> pairs;
    pairs.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      if (!a[i].IsZero() && !b[i].infinity()) {
        pairs.emplace_back(&a[i], &b[i].ell_coeffs());
      }
    }
    return pairs;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_PAIRING_FRIENDLY_CURVE_H_
