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

// clang-format off
// Base class for pairing-friendly elliptic curves.
//
// Provides common infrastructure for computing bilinear pairings:
//   e: G1 × G2 → GT
//
// The pairing is computed using the Miller loop algorithm, which evaluates
// rational functions (line functions) along a path in the group. The result
// is then raised to a large power (final exponentiation) to ensure the
// pairing maps into the correct subgroup of GT.
//
// Derived classes (e.g., BNCurve) implement curve-specific optimizations
// for the Miller loop and final exponentiation.
// clang-format on
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
  // clang-format off
  // Pairs a G1 point with precomputed G2 line coefficients for Miller loop.
  //
  // During the Miller loop, we need to evaluate line functions passing through
  // points on G2 at the G1 point. These line coefficients are precomputed in
  // G2Prepared to avoid redundant computation when the same G2 point is used
  // in multiple pairings.
  //
  // The idx_ member tracks which coefficient to use next, allowing sequential
  // access during the Miller loop iteration.
  // clang-format on
  class Pair {
   public:
    Pair() = default;
    Pair(const G1AffinePoint* g1, const std::vector<EllCoeff<Fp2>>* ell_coeffs)
        : g1_(g1), ell_coeffs_(ell_coeffs) {}

    const G1AffinePoint& g1() const { return *g1_; }

    // Returns the next line coefficient and advances the index.
    const EllCoeff<Fp2>& NextEllCoeff() const { return (*ell_coeffs_)[idx_++]; }

   private:
    const G1AffinePoint* g1_ = nullptr;
    const std::vector<EllCoeff<Fp2>>* ell_coeffs_ = nullptr;
    mutable size_t idx_ = 0;
  };

  // Computes f^x where x is the curve parameter (e.g., BN parameter).
  // Uses cyclotomic exponentiation for efficiency since f is in the
  // cyclotomic subgroup after the easy part of final exponentiation.
  static Fp12 PowByX(const Fp12& f_in) {
    Fp12 f = f_in.CyclotomicPow(Config::kX);
    if constexpr (Config::kXIsNegative) {
      f = f.CyclotomicInverse();
    }
    return f;
  }

  // Computes f^(-x) where x is the curve parameter.
  static Fp12 PowByNegX(const Fp12& f_in) {
    Fp12 f = f_in.CyclotomicPow(Config::kX);
    if constexpr (!Config::kXIsNegative) {
      f = f.CyclotomicInverse();
    }
    return f;
  }

  // clang-format off
  // Evaluates the line function and multiplies into the accumulator f.
  //
  // The line function ℓ(P) represents the line passing through points on G2
  // (from doubling or addition), evaluated at the G1 point P. The coefficients
  // (c0, c1, c2) encode this line in a sparse form.
  //
  // For M-type twist: ℓ = c0 + c1·x·P.x + c2·y·P.y  (sparse in positions 0,1,4)
  // For D-type twist: ℓ = c0·P.y + c1·x·P.x + c2    (sparse in positions 0,3,4)
  //
  // The sparse structure allows using optimized multiplication (MulBy014/034).
  // clang-format on
  static void Ell(Fp12& f, const EllCoeff<Fp2>& coeffs,
                  const G1AffinePoint& p) {
    if constexpr (Config::kTwistType == TwistType::kM) {
      f = f.MulBy014(coeffs.c0(), coeffs.c1() * p.x(), coeffs.c2() * p.y());
    } else {
      f = f.MulBy034(coeffs.c0() * p.y(), coeffs.c1() * p.x(), coeffs.c2());
    }
  }

  // Creates (G1 point, G2 coefficients) pairs, filtering out identity points.
  // Identity points contribute 1 to the pairing and can be skipped.
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
