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

#ifndef ZK_DTYPES_INCLUDE_FIELD_CYCLOTOMIC_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_CYCLOTOMIC_OPERATION_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

namespace internal {

// Detects the extension degree of a type. Returns 0 for non-extension fields.
template <typename T, typename = void>
struct ExtensionDegreeOf : std::integral_constant<size_t, 0> {};

template <typename T>
struct ExtensionDegreeOf<
    T, std::void_t<decltype(ExtensionFieldOperationTraits<T>::kDegree)>>
    : std::integral_constant<size_t,
                             ExtensionFieldOperationTraits<T>::kDegree> {};

}  // namespace internal

// Cyclotomic operations for extension fields.
// These operations are optimized for elements in the cyclotomic subgroup,
// which appear in pairing-based cryptography (e.g., final exponentiation).
//
// For a quadratic extension F_{p²ᵏ} over F_{pᵏ}, the cyclotomic subgroup
// consists of elements x where x^{pᵏ + 1} = 1 (i.e., Norm(x) = 1).
// For such elements, the inverse equals the conjugate: x⁻¹ = conj(x).
template <typename Derived>
class CyclotomicOperation {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;

  // Cyclotomic inverse for quadratic extension fields.
  // For elements in the cyclotomic subgroup, the inverse is the conjugate.
  // This is because x * conj(x) = Norm(x) = 1 for x in the cyclotomic subgroup.
  Derived CyclotomicInverse() const {
    const std::array<BaseField, 2>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    return static_cast<const Derived&>(*this).FromCoeffs({x[0], -x[1]});
  }

  // Cyclotomic square for quadratic extension fields.
  // When the base field is a cubic extension (e.g., Fp12 = Fp6[w]/(w² - v)),
  // uses the Granger-Scott algorithm for ~2x speedup.
  // Otherwise falls back to regular Square().
  //
  // IMPORTANT: Only valid for elements in the cyclotomic subgroup GΦ₆, i.e.,
  // elements x satisfying x^(Φ₆(p²)) = x^(p⁴ - p² + 1) = 1. This is a PROPER
  // subgroup of {x : Norm(x) = 1}. In pairing contexts, elements enter GΦ₆
  // after both steps of the easy part: f^(p⁶ - 1) then f^(p² + 1).
  //
  // Reference: Granger, Scott, "Faster Squaring in the Cyclotomic Subgroup
  // of Sixth Degree Extensions", PKC 2010.
  Derived CyclotomicSquare() const {
    const Derived& self = static_cast<const Derived&>(*this);

    if constexpr (internal::ExtensionDegreeOf<BaseField>::value == 3) {
      // Granger-Scott cyclotomic squaring for degree-6 extensions
      // decomposed as quadratic over cubic (e.g., Fp12 = Fp6[w]/(w² - v)).
      //
      // For x = A + Bw in the cyclotomic subgroup (Norm = A² - B²v = 1):
      //   x² = (2A² - 1) + 2AB·w
      //
      // This decomposes into 3 Fp4 squarings at the Fp2 level,
      // reducing cost from ~12 to ~6 Fp2 multiplications.
      using Fp2 = typename ExtensionFieldOperationTraits<BaseField>::BaseField;

      const std::array<BaseField, 2>& c = self.ToCoeffs();
      const std::array<Fp2, 3>& a = c[0].ToCoeffs();  // A = (a0, a1, a2)
      const std::array<Fp2, 3>& b = c[1].ToCoeffs();  // B = (b0, b1, b2)

      // Cubic non-residue xi: v³ = xi in Fp6 = Fp2[v]/(v³ - xi)
      const Fp2& xi = c[0].NonResidue();

      // Three Fp4 squarings: Sq4(x, y) = (x² + xi·y², 2xy)
      // Pair 1: (a0, b1)
      Fp2 a0_sq = a[0].Square();
      Fp2 b1_sq = b[1].Square();
      Fp2 t0 = a0_sq + xi * b1_sq;
      Fp2 t1 = (a[0] + b[1]).Square() - a0_sq - b1_sq;

      // Pair 2: (b0, a2)
      Fp2 b0_sq = b[0].Square();
      Fp2 a2_sq = a[2].Square();
      Fp2 t2 = b0_sq + xi * a2_sq;
      Fp2 t3 = (b[0] + a[2]).Square() - b0_sq - a2_sq;

      // Pair 3: (a1, b2)
      Fp2 a1_sq = a[1].Square();
      Fp2 b2_sq = b[2].Square();
      Fp2 t4 = a1_sq + xi * b2_sq;
      Fp2 t5 = (a[1] + b[2]).Square() - a1_sq - b2_sq;

      // Reconstruct: "real" part (A') uses 3t - 2a pattern
      Fp2 new_a0 = (t0 - a[0]).Double() + t0;  // 3t0 - 2a0
      Fp2 new_a1 = (t2 - a[1]).Double() + t2;  // 3t2 - 2a1
      Fp2 new_a2 = (t4 - a[2]).Double() + t4;  // 3t4 - 2a2

      // Reconstruct: "imaginary" part (B') uses 3t + 2b pattern
      Fp2 xi_t5 = xi * t5;
      Fp2 new_b0 = (xi_t5 + b[0]).Double() + xi_t5;  // 3·xi·t5 + 2b0
      Fp2 new_b1 = (t1 + b[1]).Double() + t1;        // 3t1 + 2b1
      Fp2 new_b2 = (t3 + b[2]).Double() + t3;        // 3t3 + 2b2

      BaseField new_c0 = c[0].FromCoeffs({new_a0, new_a1, new_a2});
      BaseField new_c1 = c[1].FromCoeffs({new_b0, new_b1, new_b2});

      return self.FromCoeffs({new_c0, new_c1});
    } else {
      return self.Square();
    }
  }

  // Cyclotomic exponentiation using square-and-multiply.
  // Uses CyclotomicSquare for squaring operations, which can be faster
  // than regular squaring for elements in the cyclotomic subgroup.
  template <size_t N>
  Derived CyclotomicPow(const BigInt<N>& exponent) const {
    const Derived& self = static_cast<const Derived&>(*this);
    if (self.IsZero()) return Derived::Zero();

    Derived result = Derived::One();
    bool found_nonzero = false;

    // Process bits from most significant to least significant
    for (size_t i = N; i > 0; --i) {
      uint64_t limb = exponent[i - 1];
      for (int bit = 63; bit >= 0; --bit) {
        if (found_nonzero) {
          result = result.CyclotomicSquare();
        }
        if ((limb >> bit) & 1) {
          if (found_nonzero) {
            result *= self;
          } else {
            result = self;
            found_nonzero = true;
          }
        }
      }
    }
    return result;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_CYCLOTOMIC_OPERATION_H_
