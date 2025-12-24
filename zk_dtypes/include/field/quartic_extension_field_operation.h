/* Copyright 2025 The zk_dtypes Authors.

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

#ifndef ZK_DTYPES_INCLUDE_FIELD_QUARTIC_EXTENSION_FIELD_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_QUARTIC_EXTENSION_FIELD_OPERATION_H_

#include <array>

#include "zk_dtypes/include/field/extension_field_operation.h"
#include "zk_dtypes/include/field/karatsuba_operation.h"
#include "zk_dtypes/include/field/toom_cook_operation.h"
#include "zk_dtypes/include/field/vandermonde_matrix.h"

namespace zk_dtypes {

template <typename Derived>
class QuarticExtensionFieldOperation
    : public ExtensionFieldOperation<Derived>,
      public ToomCookOperation<Derived>,
      public KaratsubaOperation<Derived>,
      public VandermondeMatrix<
          typename ExtensionFieldOperationTraits<Derived>::BaseField, 4> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;

  // Multiplication in Fp4 using Toom-Cook4 or Karatsuba.
  Derived operator*(const Derived& other) const {
    ExtensionFieldMulAlgorithm algorithm =
        static_cast<const Derived&>(*this).GetMulAlgorithm();
    if (algorithm == ExtensionFieldMulAlgorithm::kToomCook) {
      // See https://eprint.iacr.org/2006/471.pdf
      // Devegili OhEig Scott Dahab --- Multiplication and Squaring on
      // Pairing-Friendly Fields.
      return this->ToomCookMultiply(other);
    } else {
      return this->KaratsubaMultiply(other);
    }
  }

  // Square in Fp4 using Toom-Cook4 or Karatsuba.
  Derived Square() const {
    ExtensionFieldMulAlgorithm algorithm =
        static_cast<const Derived&>(*this).GetSquareAlgorithm();
    if (algorithm == ExtensionFieldMulAlgorithm::kCustom) {
      // [Comparison]
      // Custom Algorithm:
      // - square: 5, mul: 4, mul by non-residue: 3, add: 7, sub: 4, double: 4
      // Default Karatsuba:
      // - square: 4, mul: 6, mul by non-residue: 3, add: 12, sub: 0, double: 6
      //
      // Conclusion:
      // The Custom algorithm saves 1 Multiplication, 1 Addition and 2 Double
      // operations.
      //
      // For x = x₀ + x₁·u + x₂·u² + x₃·u³ where u⁴ = ξ:
      //
      // s₀ = x₀²
      // s₁ = x₁²
      // s₂ = x₂²
      // s₃ = x₃²
      // s₄ = (x₀ + x₂)²
      // m₀ = 2 * x₀ * x₁
      // m₁ = 2 * x₂ * x₃
      // m₂ = 2 * x₁ * x₃
      // m₃ = 2 * (x₀ + x₂) * (x₁ + x₃)
      //
      // Result:
      // y₀ = s₀ + ξ * (s₂ + 2x₁x₃)
      // y₁ = m₀ + ξ * m₁
      // y₂ = s₁ + ξ * s₃ + (s₄ - s₀ - s₂)
      // y₃ = m₃ - m₀ - m₁

      std::array<BaseField, 4> x =
          static_cast<const Derived&>(*this).ToBaseField();
      BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

      // 1. Squares (xᵢ²)

      // s₀ = x₀²
      BaseField s0 = x[0].Square();
      // s₁ = x₁²
      BaseField s1 = x[1].Square();
      // s₂ = x₂²
      BaseField s2 = x[2].Square();
      // s₃ = x₃²
      BaseField s3 = x[3].Square();

      // s₄ = (x₀ + x₂)²
      BaseField x0_add_x2 = x[0] + x[2];
      BaseField s4 = x0_add_x2.Square();

      // 2. Products (Cross terms)

      // m₀ = 2 * x₀ * x₁
      BaseField m0 = (x[0] * x[1]).Double();
      // m₁ = 2 * x₂ * x₃
      BaseField m1 = (x[2] * x[3]).Double();
      // m₂ = 2 * x₁ * x₃
      BaseField m2 = (x[1] * x[3]).Double();
      // m₃ = 2 * (x₀ + x₂) * (x₁ + x₃)
      BaseField x1_add_x3 = x[1] + x[3];
      BaseField m3 = (x0_add_x2 * x1_add_x3).Double();

      // 3. Reconstruction (u⁴ = ξ)

      // y₀ = x₀² + ξx₂² + 2ξx₁x₃
      //    = s₀ + ξ(s₂ + 2x₁x₃)
      //    Here we use m₂ directly instead of (s5 - s1 - s3)
      BaseField y0 = s0 + non_residue * (s2 + m2);

      // y₁ = 2x₀x₁ + 2ξx₂x₃
      //    = m₀ + ξm₁
      BaseField y1 = m0 + non_residue * m1;

      // y₂ = x₁² + ξx₃² + 2x₀x₂
      //    = s₁ + ξs₃ + (s₄ - s₀ - s₂)
      //    Note: s₄ - s₀ - s₂ = 2x₀x₂
      BaseField y2 = s1 + non_residue * s3 + s4 - s0 - s2;

      // y₃ = 2x₀x₃ + 2x₁x₂
      //    = 2(x₀+x₂)(x₁+x₃) - 2x₀x₁ - 2x₂x₃
      //    = m₃ - m₀ - m₁
      BaseField y3 = m3 - m0 - m1;

      return static_cast<const Derived&>(*this).FromBaseFields(
          {y0, y1, y2, y3});
    } else if (algorithm == ExtensionFieldMulAlgorithm::kToomCook) {
      return this->ToomCookSquare();
    } else {
      return this->KaratsubaSquare();
    }
  }

 private:
  friend class ToomCookOperation<Derived>;

  static std::array<BaseField, 7> ComputeEvaluations(
      const std::array<BaseField, 4>& x) {
    // 1. Basic Doubles
    BaseField x1_2 = x[1].Double();
    BaseField x2_4 = x[2].Double().Double();
    BaseField x3_2 = x[3].Double();
    BaseField x3_8 = x3_2.Double().Double();

    // Needed for f(3) optimization formula
    BaseField x3_3 = x3_2 + x[3];

    // 2. Intermediates
    BaseField v0 = x[0] + x[2];
    BaseField v1 = x[1] + x[3];
    BaseField v2 = x[0] + x2_4;
    BaseField v3 = x1_2 + x3_8;

    std::array<BaseField, 7> f_x;

    // 3. Evaluations
    // f(0) = x₀
    f_x[0] = x[0];
    // f(1) = v₀ + v₁
    f_x[1] = v0 + v1;
    // f(-1) = v₀ - v₁
    f_x[2] = v0 - v1;
    // f(2) = v₂ + v₃
    f_x[3] = v2 + v3;
    // f(-2) = v₂ - v₃
    f_x[4] = v2 - v3;
    // [Optimized f(3)]
    // f(3) = x₀ + 3x₁ + 9x₂ + 27x₃
    //       = f(1) + 2 * (v₁ + 4 * (x₂ + 3x₃))
    BaseField term = x[2] + x3_3;
    term = term.Double().Double();
    term = term + v1;
    term = term.Double();
    f_x[5] = f_x[1] + term;
    // f(∞) = x₃
    f_x[6] = x[3];
    return f_x;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_QUARTIC_EXTENSION_FIELD_OPERATION_H_
