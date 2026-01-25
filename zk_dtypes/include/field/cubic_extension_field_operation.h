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

      const std::array<BaseField, 3>& x =
          static_cast<const Derived&>(*this).ToCoeffs();
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

      return static_cast<const Derived&>(*this).FromCoeffs({y0, y1, y2});
    } else {
      return this->KaratsubaSquare();
    }
  }

  // Sparse multiplication: (α₀ + α₁x + α₂x²) * β₁x
  // Used in Fp12 pairing operations (MulBy014).
  // Returns α₂β₁q + α₀β₁x + α₁β₁x², where q is the cubic non-residue.
  Derived MulBy1(const BaseField& beta1) const {
    const std::array<BaseField, 3>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // c0 = α₂β₁q
    BaseField c0 = non_residue * (x[2] * beta1);
    // c1 = α₀β₁
    BaseField c1 = x[0] * beta1;
    // c2 = α₁β₁
    BaseField c2 = x[1] * beta1;

    return static_cast<const Derived&>(*this).FromCoeffs({c0, c1, c2});
  }

  // Sparse multiplication: (α₀ + α₁x + α₂x²) * (β₀ + β₁x)
  // Used in Fp12 pairing operations (MulBy014, MulBy034).
  // Returns (α₀β₀ + α₂β₁q) + (α₀β₁ + α₁β₀)x + (α₁β₁ + α₂β₀)x²,
  // where q is the cubic non-residue.
  Derived MulBy01(const BaseField& beta0, const BaseField& beta1) const {
    const std::array<BaseField, 3>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // c0 = α₀β₀ + α₂β₁q
    BaseField c0 = x[0] * beta0 + non_residue * (x[2] * beta1);
    // c1 = α₀β₁ + α₁β₀
    BaseField c1 = x[0] * beta1 + x[1] * beta0;
    // c2 = α₁β₁ + α₂β₀
    BaseField c2 = x[1] * beta1 + x[2] * beta0;

    return static_cast<const Derived&>(*this).FromCoeffs({c0, c1, c2});
  }

  // Returns the multiplicative inverse. Returns Zero() if not invertible.
  Derived Inverse() const {
    // [Comparison]
    // This Algorithm (Matrix Method / Cramer's Rule):
    // - square: 3, mul: 9, inv: 1 (base field ops)
    // Standard Itoh-Tsujii:
    // - square: 3, mul: 12, inv: 1 (approx. 1 Full Fp3-Mul + Dot Product)
    //
    // Conclusion:
    // This algorithm saves ~3 Multiplications. The Matrix Method computes
    // the first column of the inverse directly, avoiding the overhead of
    // a full extension field multiplication required by Itoh-Tsujii.
    const std::array<BaseField, 3>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    BaseField xi = static_cast<const Derived&>(*this)
                       .NonResidue();  // ξ: Irreducible polynomial constant
                                       // [Comparison]

    // Representing an element (x₀ + x₁w + x₂w²) as a 3×3 Matrix:
    //   [ x₀  ξx₂  ξx₁ ]
    //   [ x₁  x₀   ξx₂ ]
    //   [ x₂  x₁   x₀  ]
    // The inverse is (1 / det) * Adjugate(M).

    // 1. Calculate the first column of the Adjugate Matrix (Cofactors)
    // t₀, t₁, t₂ are the cofactors of the first column of the representation
    // matrix.
    // t₀ = x₀² - ξx₁x₂
    BaseField t0 = x[0].Square() - xi * (x[1] * x[2]);
    // t₁ = ξx₂² - x₀x₁
    BaseField t1 = xi * x[2].Square() - x[0] * x[1];
    // t₂ = x₁² - x₀x₂
    BaseField t2 = x[1].Square() - x[0] * x[2];

    // 2. Calculate the Determinant (t₃)
    // Using Laplace expansion along the first column: det = x₀*t₀ + ξx₂*t₁ +
    // ξx₁*t₂
    BaseField t3 = x[0] * t0 + xi * (x[2] * t1 + x[1] * t2);

    // 3. Invert the determinant. Inverse() returns Zero() if not invertible.
    BaseField t3_inv = t3.Inverse();

    // 4. Final Inverse result: (t₀, t₁, t₂) / det
    // Since the field is commutative, we only need the first column of the
    // adjugate.
    std::array<BaseField, 3> y{t0 * t3_inv, t1 * t3_inv, t2 * t3_inv};

    return static_cast<const Derived&>(*this).FromCoeffs(y);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_CUBIC_EXTENSION_FIELD_OPERATION_H_
