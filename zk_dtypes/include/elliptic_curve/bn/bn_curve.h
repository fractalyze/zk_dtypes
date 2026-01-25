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

// BN curve pairing implementation.
// Implements Miller loop and final exponentiation for BN curves.
template <typename Config>
class BNCurve : public PairingFriendlyCurve<Config> {
 public:
  using Base = PairingFriendlyCurve<Config>;
  using Fp12 = typename Config::Fp12;
  using G2Prepared = bn::G2Prepared<Config>;

  // Miller loop for multi-pairing computation.
  template <typename G1AffinePointContainer, typename G2PreparedContainer>
  static Fp12 MultiMillerLoop(const G1AffinePointContainer& a,
                              const G2PreparedContainer& b) {
    using Pair = typename Base::Pair;

    std::vector<Pair> pairs = Base::CreatePairs(a, b);

    Fp12 f = Fp12::One();
    for (size_t i = std::size(Config::kAteLoopCount) - 1; i >= 1; --i) {
      if (i != std::size(Config::kAteLoopCount) - 1) {
        f = f.Square();
      }

      for (const Pair& pair : pairs) {
        Base::Ell(f, pair.NextEllCoeff(), pair.g1());
      }

      int8_t bit = Config::kAteLoopCount[i - 1];
      if (bit == 1 || bit == -1) {
        for (const Pair& pair : pairs) {
          Base::Ell(f, pair.NextEllCoeff(), pair.g1());
        }
      }
    }

    if constexpr (Config::kXIsNegative) {
      f = f.CyclotomicInverse();
    }

    for (const Pair& pair : pairs) {
      Base::Ell(f, pair.NextEllCoeff(), pair.g1());
    }

    for (const Pair& pair : pairs) {
      Base::Ell(f, pair.NextEllCoeff(), pair.g1());
    }

    return f;
  }

  // Final exponentiation: f^((q^12 - 1) / r)
  static Fp12 FinalExponentiation(const Fp12& f) {
    // Easy part: f^((q^6 - 1) * (q^2 + 1))
    // f1 = conj(f) = f^(q^6)
    Fp12 f1 = f.CyclotomicInverse();
    // f2 = f^(-1)
    Fp12 f2 = f.Inverse();
    // r = f^(q^6 - 1)
    Fp12 r = f1 * f2;

    // f2 = f^(q^6 - 1)
    f2 = r;
    // r = f^((q^6 - 1)(q^2))
    r = r.template Frobenius<2>();
    // r = f^((q^6 - 1)(q^2 + 1))
    r *= f2;

    // Hard part: uses Laura Fuentes-Castaneda et al. "Faster hashing to G2"
    // y0 = r^(-x)
    Fp12 y0 = Base::PowByNegX(r);
    // y1 = y0^2 = r^(-2x)
    Fp12 y1 = y0.CyclotomicSquare();
    // y2 = y1^2 = r^(-4x)
    Fp12 y2 = y1.CyclotomicSquare();
    // y3 = y2 * y1 = r^(-6x)
    Fp12 y3 = y2 * y1;
    // y4 = y3^(-x) = r^(6x^2)
    Fp12 y4 = Base::PowByNegX(y3);
    // y5 = y4^2 = r^(12x^2)
    Fp12 y5 = y4.CyclotomicSquare();
    // y6 = y5^(-x) = r^(-12x^3)
    Fp12 y6 = Base::PowByNegX(y5);
    // y3 = y3^(-1) = r^(6x)
    y3 = y3.CyclotomicInverse();
    // y6 = y6^(-1) = r^(12x^3)
    y6 = y6.CyclotomicInverse();
    // y7 = y6 * y4 = r^(12x^3 + 6x^2)
    Fp12 y7 = y6 * y4;
    // y8 = y7 * y3 = r^(12x^3 + 6x^2 + 6x)
    Fp12 y8 = y7 * y3;
    // y9 = y8 * y1 = r^(12x^3 + 6x^2 + 4x)
    Fp12 y9 = y8 * y1;
    // y10 = y8 * y4 = r^(12x^3 + 12x^2 + 6x)
    Fp12 y10 = y8 * y4;
    // y11 = y10 * r = r^(12x^3 + 12x^2 + 6x + 1)
    Fp12 y11 = y10 * r;
    // y12 = y9^q = r^(q * (12x^3 + 6x^2 + 4x))
    Fp12 y12 = y9.template Frobenius<1>();
    // y13 = y12 * y11
    Fp12 y13 = y12 * y11;
    // y8 = y8^(q^2)
    Fp12 y8_frob = y8.template Frobenius<2>();
    // y14 = y8 * y13
    Fp12 y14 = y8_frob * y13;
    r = r.CyclotomicInverse();
    // y15 = r^(-1) * y9 = r^(12x^3 + 6x^2 + 4x - 1)
    Fp12 y15 = (r * y9).template Frobenius<3>();
    // y16 = y15 * y14
    Fp12 y16 = y15 * y14;
    return y16;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_
