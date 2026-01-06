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

#ifndef ZK_DTYPES_INCLUDE_FIELD_KARATSUBA_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_KARATSUBA_OPERATION_H_

#include <array>
#include <cstddef>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// clang-format off
// Generalized Karatsuba Multiplication and Squaring for Extension Fields.
//
// This template class implements the Karatsuba algorithm for extension fields of
// arbitrary degree N. It treats extension field elements as polynomials:
//    x(u) = Σ_{i=0}^{n-1} xᵢ·uⁱ
//    y(u) = Σ_{i=0}^{n-1} yᵢ·uⁱ
// where uⁿ = ξ.
//
// Karatsuba's trick reduces the number of multiplications required for the product
// of two polynomials. For any two terms xᵢ, xⱼ and yᵢ, yⱼ, the cross term
// (xᵢ·yⱼ + xⱼ·yᵢ) is computed as: (xᵢ + xⱼ)(yᵢ + yⱼ) - xᵢ·yᵢ - xⱼ·yⱼ
//
// The operation pipeline consists of:
// 1. Evaluation: Compute diagonal (xᵢ·yᵢ) and cross terms.
// 2. Assembly: Map these terms to the coefficients of the 2n - 2 degree product polynomial.
// 3. Reduction: Reduce the polynomial modulo (uⁿ - ξ).
//
// Complexity: O(n^log2(3)) ≈ O(n^1.58).
//
// References:
// - https://en.wikipedia.org/wiki/Karatsuba_algorithm
// - https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/multiplication/karatsuba-multiplication
// clang-format on
template <typename Derived>
class KaratsubaOperation {
 public:
  // Multiplies this element with another using the Karatsuba method.
  Derived KaratsubaMultiply(const Derived& other) const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    const std::array<BaseField, kDegree>& y =
        static_cast<const Derived&>(other).ToCoeffs();

    return Reduce(AssembleMulPolynomial(ComputeMulTerms(x, y)));
  }

  // Squares this element using the Karatsuba method.
  Derived KaratsubaSquare() const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();

    return Reduce(AssembleSqrPolynomial(ComputeSqrTerms(x)));
  }

 private:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;
  constexpr static size_t kNumCrossTerms = kDegree * (kDegree - 1) / 2;
  constexpr static size_t kNumEvaluation = 2 * kDegree - 1;

  // ----------------------------------------------------------------------
  // Structures for Intermediate Results
  // ----------------------------------------------------------------------

  struct MulEvaluationResult {
    // Diagonals: x₀y₀, x₁y₁, ... (N terms)
    std::array<BaseField, kDegree> diagonal_terms;
    // Cross terms: (x₀+x₁)(y₀+y₁) - x₀y₀ - x₁y₁ , ... (K terms, K = N(N-1)/2)
    std::array<BaseField, kNumCrossTerms> cross_terms;
  };

  struct SqrEvaluationResult {
    // Squares: x₀², x₁², ... (N terms)
    std::array<BaseField, kDegree> square_terms;
    // Products: x₀ x₁, x₀ x₂, ... (K terms, K = N(N-1)/2)
    std::array<BaseField, kNumCrossTerms> product_terms;
  };

  // Computes intermediate terms for multiplication.
  // Leverages the Karatsuba identity to avoid direct computation of xᵢ·yⱼ.
  static MulEvaluationResult ComputeMulTerms(
      const std::array<BaseField, kDegree>& x,
      const std::array<BaseField, kDegree>& y) {
    MulEvaluationResult res;

    // 1. Diagonal Terms (xᵢyᵢ)
    for (size_t i = 0; i < kDegree; ++i) {
      res.diagonal_terms[i] = x[i] * y[i];
    }

    // 2. Cross Terms ((xᵢ + xⱼ)(yᵢ + yⱼ) - xᵢyᵢ - xⱼyⱼ)
    size_t idx = 0;
    for (size_t i = 0; i < kDegree; ++i) {
      for (size_t j = i + 1; j < kDegree; ++j) {
        BaseField sum_x = x[i] + x[j];
        BaseField sum_y = y[i] + y[j];
        res.cross_terms[idx++] =
            sum_x * sum_y - res.diagonal_terms[i] - res.diagonal_terms[j];
      }
    }
    return res;
  }

  // Computes intermediate terms for squaring.
  // Separates squares and cross-products for the (x₀ + x₁·u + ... + xₙ₋₁·uⁿ⁻¹)²
  // expansion.
  static SqrEvaluationResult ComputeSqrTerms(
      const std::array<BaseField, kDegree>& x) {
    SqrEvaluationResult res;

    // 1. Squares (xᵢ²)
    for (size_t i = 0; i < kDegree; ++i) {
      res.square_terms[i] = x[i].Square();
    }

    // 2. Cross Products (xᵢxⱼ)
    size_t idx = 0;
    for (size_t i = 0; i < kDegree; ++i) {
      for (size_t j = i + 1; j < kDegree; ++j) {
        res.product_terms[idx++] = x[i] * x[j];
      }
    }
    return res;
  }

  // ----------------------------------------------------------------------
  // 2. Assembly Step (Construct Polynomial of degree 2N-2)
  // ----------------------------------------------------------------------

  // Maps computed multiplication terms to polynomial coefficients.
  // xᵢ·yᵢ contributes to the coefficient of u²ⁱ.
  // xᵢ·yⱼ contributes to the coefficient of uⁱ⁺ʲ.
  std::array<BaseField, kNumEvaluation> AssembleMulPolynomial(
      const MulEvaluationResult& evals) const {
    BaseField zero = evals.diagonal_terms[0].CreateConst(0);

    std::array<BaseField, kNumEvaluation> c;
    for (size_t i = 0; i < c.size(); ++i) {
      c[i] = zero;
    }

    // Diagonals -> u²ⁱ
    for (size_t i = 0; i < kDegree; ++i) {
      c[2 * i] += evals.diagonal_terms[i];
    }

    // Cross -> uⁱ⁺ʲ
    size_t idx = 0;
    for (size_t i = 0; i < kDegree; ++i) {
      for (size_t j = i + 1; j < kDegree; ++j) {
        c[i + j] += evals.cross_terms[idx++];
      }
    }

    return c;
  }

  // Maps computed squaring terms to polynomial coefficients.
  // xᵢ² contributes to the coefficient of u²ⁱ.
  // xᵢ·xⱼ contributes to the coefficient of uⁱ⁺ʲ.
  std::array<BaseField, kNumEvaluation> AssembleSqrPolynomial(
      const SqrEvaluationResult& evals) const {
    BaseField zero = evals.square_terms[0].CreateConst(0);

    std::array<BaseField, kNumEvaluation> c;
    for (size_t i = 0; i < c.size(); ++i) {
      c[i] = zero;
    }

    // Squares -> u²ⁱ
    for (size_t i = 0; i < kDegree; ++i) {
      c[2 * i] += evals.square_terms[i];
    }

    // Products -> uⁱ⁺ʲ
    size_t idx = 0;
    for (size_t i = 0; i < kDegree; ++i) {
      for (size_t j = i + 1; j < kDegree; ++j) {
        c[i + j] += evals.product_terms[idx++].Double();
      }
    }
    return c;
  }

  // ----------------------------------------------------------------------
  // 3. Reduction Step (Modulo uᴺ - ξ)
  // ----------------------------------------------------------------------

  // Reduces the product polynomial C(u) modulo (uⁿ - ξ).
  // Since uⁿ ≡ ξ, we have uⁱ⁺ⁿ ≡ ξ·uⁱ.
  // The resulting coefficients are zᵢ = cᵢ + ξ·cᵢ₊ₙ.
  Derived Reduce(const std::array<BaseField, kNumEvaluation>& c) const {
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();
    std::array<BaseField, kDegree> ret;
    for (size_t i = 0; i < kDegree - 1; ++i) {
      ret[i] = c[i] + non_residue * c[i + kDegree];
    }
    ret[kDegree - 1] = c[kDegree - 1];
    return static_cast<const Derived&>(*this).FromCoeffs(ret);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_KARATSUBA_OPERATION_H_
