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

#ifndef ZK_DTYPES_INCLUDE_FIELD_CUBIC_EXTENSION_FIELD_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_CUBIC_EXTENSION_FIELD_OPERATION_H_

#include <array>

#include "zk_dtypes/include/field/extension_field_operation.h"
#include "zk_dtypes/include/field/karatsuba_operation.h"

namespace zk_dtypes {

template <typename Derived>
class CubicExtensionFieldOperation : public ExtensionFieldOperation<Derived>,
                                     public KaratsubaOperation<Derived> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;

  // Multiplication using Karatsuba method.
  Derived operator*(const Derived& other) const {
    return this->KaratsubaMultiply(other);
  }

  Derived Square() const {
    ExtensionFieldMulAlgorithm algorithm =
        static_cast<const Derived&>(*this).GetSquareAlgorithm();
    if (algorithm == ExtensionFieldMulAlgorithm::kCustom) {
      // [Comparison]
      // Custom Algorithm (Chung-Hasan):
      // - square: 3, mul: 2, mul by non-residue: 2, add: 5, sub: 3, double: 2
      // Default Karatsuba:
      // - square: 3, mul: 3, mul by non-residue: 2, add: 3, sub: 0, double: 3
      //
      // Conclusion:
      // The Custom algorithm saves 1 Multiplication at the cost of ~5
      // Add/Sub. Since Mul >> Add in cost, this optimization is generally
      // faster.

      // Square in Fp3 using CH-SQR2 algorithm.
      // See https://eprint.iacr.org/2006/471.pdf
      // Devegili OhEig Scott Dahab --- Multiplication and Squaring on
      // Pairing-Friendly Fields; Section 4 (CH-SQR2)
      //
      // For x = x₀ + x₁·u + x₂·u² where u³ = ξ:
      //
      // s₀ = x₀²
      // s₁ = 2 * x₀ * x₁
      // s₂ = (x₀ - x₁ + x₂)²
      // s₃ = 2 * x₁ * x₂
      // s₄ = x₂²
      //
      // Result:
      // y₀ = s₀ + ξ * s₃
      // y₁ = s₁ + ξ * s₄
      // y₂ = s₁ + s₂ + s₃ - s₀ - s₄

      std::array<BaseField, 3> x =
          static_cast<const Derived&>(*this).ToBaseField();
      BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

      // s₀ = x₀²
      BaseField s0 = x[0].Square();
      // s₁ = 2 * x₀ * x₁
      BaseField s1 = (x[0] * x[1]).Double();
      // s₂ = (x₀ - x₁ + x₂)²
      // Parentheses added for clarity: ((x0 - x1) + x2)^2
      BaseField s2 = (x[0] - x[1] + x[2]).Square();
      // s₃ = 2 * x₁ * x₂
      BaseField s3 = (x[1] * x[2]).Double();
      // s₄ = x₂²
      BaseField s4 = x[2].Square();

      // y₀ = s₀ + ξ * s₃
      BaseField y0 = s0 + non_residue * s3;
      // y₁ = s₁ + ξ * s₄
      BaseField y1 = s1 + non_residue * s4;
      // y₂ = s₁ + s₂ + s₃ - s₀ - s₄
      //    = (x₁² + 2x₀x₂)  <-- Verified term
      BaseField y2 = s1 + s2 + s3 - s0 - s4;

      return static_cast<const Derived&>(*this).FromBaseFields({y0, y1, y2});
    } else {
      return this->KaratsubaSquare();
    }
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_CUBIC_EXTENSION_FIELD_OPERATION_H_
