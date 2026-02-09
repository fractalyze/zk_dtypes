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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_

#include <functional>
#include <numeric>
#include <vector>

#include "zk_dtypes/include/elliptic_curve/bn/g2_prepared.h"
#include "zk_dtypes/include/elliptic_curve/pairing/pairing_friendly_curve.h"

namespace zk_dtypes {

// clang-format off
// BN curve pairing implementation.
//
// Implements the optimal ate pairing for BN (Barreto-Naehrig) curves.
// The pairing e: G1 × G2 → GT is computed as:
//   e(P, Q) = FinalExponentiation(MillerLoop(P, Q))
//
// References:
// - "High-Speed Software Implementation of the Optimal Ate Pairing over
//   Barreto–Naehrig Curves" by Beuchat et al.
//   https://eprint.iacr.org/2010/354.pdf
// - "Faster hashing to G2" by Fuentes-Castañeda et al.
//   https://eprint.iacr.org/2011/297.pdf
// clang-format on
template <typename Config>
class BNCurve : public PairingFriendlyCurve<Config> {
 public:
  using Base = PairingFriendlyCurve<Config>;
  using Fp12 = typename Config::Fp12;
  using G2Prepared = bn::G2Prepared<Config>;

  // clang-format off
  // Multi-Miller loop for computing multiple pairings simultaneously.
  //
  // Computes f = ∏ᵢ f_{6x+2,Qᵢ}(Pᵢ) where f_{r,Q}(P) is the Miller function.
  // The ate loop count (6x+2) is given in NAF (Non-Adjacent Form) representation
  // in Config::kAteLoopCount for efficient computation.
  //
  // The algorithm processes the NAF bits from MSB to LSB:
  //   1. Square f (doubling step)
  //   2. Multiply by line functions from point doubling
  //   3. If NAF bit is ±1, multiply by line functions from point addition
  //
  // After the main loop, two additional line evaluations are performed
  // for the BN-specific endomorphism corrections using Frobenius.
  // clang-format on
  template <typename G1AffinePointContainer, typename G2PreparedContainer>
  static Fp12 MultiMillerLoop(const G1AffinePointContainer& a,
                              const G2PreparedContainer& b) {
    using Pair = typename Base::Pair;
    std::vector<Pair> pairs = Base::CreatePairs(a, b);

    Fp12 f = Fp12::One();

    // Main Miller loop: iterate through NAF bits of ate loop count (6x+2)
    for (size_t i = std::size(Config::kAteLoopCount) - 1; i >= 1; --i) {
      // Square step (skip on first iteration)
      if (i != std::size(Config::kAteLoopCount) - 1) {
        f = f.Square();
      }

      // Doubling step: multiply f by line functions from point doubling
      for (const Pair& pair : pairs) {
        Base::Ell(f, pair.NextEllCoeff(), pair.g1());
      }

      // Addition step: if NAF bit is ±1, multiply by line functions
      int8_t naf_bit = Config::kAteLoopCount[i - 1];
      if (naf_bit == 1 || naf_bit == -1) {
        for (const Pair& pair : pairs) {
          Base::Ell(f, pair.NextEllCoeff(), pair.g1());
        }
      }
    }

    // Conjugate f if BN parameter x is negative
    if constexpr (Config::kXIsNegative) {
      f = f.CyclotomicInverse();
    }

    // BN-specific: two additional line evaluations for Frobenius corrections
    // These correspond to adding π(Q) and π²(Q) where π is the Frobenius map
    for (const Pair& pair : pairs) {
      Base::Ell(f, pair.NextEllCoeff(), pair.g1());
    }
    for (const Pair& pair : pairs) {
      Base::Ell(f, pair.NextEllCoeff(), pair.g1());
    }

    return f;
  }

  // clang-format off
  // Final exponentiation: computes f^((q^12 - 1) / r)
  //
  // The exponent (q^12 - 1) / r factors as:
  //   (q^12 - 1) / r = (q^6 - 1) · (q^2 + 1) · (q^4 - q^2 + 1) / r
  //                    \_________/ \________/ \________________/
  //                     easy part 1  easy part 2    hard part
  //
  // Easy parts are computed using Frobenius and conjugation.
  // Hard part uses the algorithm from Fuentes-Castañeda et al.
  // "Faster hashing to G2" (https://eprint.iacr.org/2011/297.pdf)
  // clang-format on
  static Fp12 FinalExponentiation(const Fp12& f) {
    // ==================== Easy Part ====================
    // Compute f^((q^6 - 1) · (q^2 + 1))

    // Step 1: f^(q^6 - 1)
    // For Fp12 = Fp6[w]/(w² - v), conjugation is f^(q^6) = conj(f)
    Fp12 f1 = f.CyclotomicInverse();  // f^(q^6) = conj(f)
    Fp12 f2 = f.Inverse();            // f^(-1)
    Fp12 r = f1 * f2;                 // f^(q^6 - 1)

    // Step 2: raise to (q² + 1)
    f2 = r;
    r = r.template FrobeniusMap<2>();  // r^(q²)
    r *= f2;                           // r^(q² + 1)

    // ==================== Hard Part ====================
    // Compute r^((q^4 - q^2 + 1) / r) using Fuentes-Castañeda's algorithm.
    // This is optimized for BN curves where the exponent can be expressed
    // in terms of the BN parameter x.

    // Build intermediate powers of r
    Fp12 y0 = Base::PowByNegX(r);     // r^(-x)
    Fp12 y1 = y0.CyclotomicSquare();  // r^(-2x)
    Fp12 y2 = y1.CyclotomicSquare();  // r^(-4x)
    Fp12 y3 = y2 * y1;                // r^(-6x)
    Fp12 y4 = Base::PowByNegX(y3);    // r^(6x²)
    Fp12 y5 = y4.CyclotomicSquare();  // r^(12x²)
    Fp12 y6 = Base::PowByNegX(y5);    // r^(-12x³)

    // Adjust signs using cyclotomic inverse
    y3 = y3.CyclotomicInverse();  // r^(6x)
    y6 = y6.CyclotomicInverse();  // r^(12x³)

    // Combine powers to build the final result
    Fp12 y7 = y6 * y4;   // r^(12x³ + 6x²)
    Fp12 y8 = y7 * y3;   // r^(12x³ + 6x² + 6x)
    Fp12 y9 = y8 * y1;   // r^(12x³ + 6x² + 4x)
    Fp12 y10 = y8 * y4;  // r^(12x³ + 12x² + 6x)
    Fp12 y11 = y10 * r;  // r^(12x³ + 12x² + 6x + 1)

    // Apply Frobenius maps and combine
    Fp12 y12 = y9.template FrobeniusMap<1>();  // y9^q
    Fp12 y13 = y12 * y11;
    Fp12 y8_frob = y8.template FrobeniusMap<2>();  // y8^(q²)
    Fp12 y14 = y8_frob * y13;

    r = r.CyclotomicInverse();                       // r^(-1)
    Fp12 y15 = (r * y9).template FrobeniusMap<3>();  // (r^(-1) · y9)^(q³)
    Fp12 result = y15 * y14;

    return result;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_
