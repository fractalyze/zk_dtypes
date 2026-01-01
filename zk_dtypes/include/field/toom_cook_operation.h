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

#ifndef ZK_DTYPES_INCLUDE_FIELD_TOOM_COOK_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_TOOM_COOK_OPERATION_H_

#include <array>
#include <cstddef>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// clang-format off
// Toom-Cook Multiplication for Extension Fields.
//
// This template class implements the Toom-Cook algorithm, a generalization of
// Karatsuba multiplication that offers lower asymptotic complexity (e.g., Toom-3 is O(n^1.465)).
// It treats extension field elements as polynomials:
//    x(u) = Σ_{i=0}^{n-1} xᵢ·uⁱ
//    y(u) = Σ_{i=0}^{n-1} yᵢ·uⁱ
// where uⁿ = ξ.
//
// The multiplication pipeline follows these 4 steps:
// 1. Evaluation: Evaluate polynomials x(u) and y(u) at k unique points {t₀, t₁, ...}.
// 2. Pointwise Product: Compute zᵢ = x(tᵢ)// y(tᵢ) for each point.
// 3. Interpolation: Use the inverse Vandermonde matrix (V⁻¹) to recover the coefficients
//    of the product polynomial C(u) = x(u)·y(u) of degree 2n - 2.
// 4. Reduction: Apply the field's modulus (uⁿ - ξ) to reduce C(u) back to degree n - 1.
//
// References:
// - https://en.wikipedia.org/wiki/Toom%E2%80%93Cook_multiplication
// - https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/multiplication/toom-cook-multiplication
// clang-format on
template <typename Derived>
class ToomCookOperation {
 public:
  // Multiplies this element with another using the Toom-Cook method.
  Derived ToomCookMultiply(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseFields();

    // Step 1 & 2: Evaluation and Pointwise Multiplication
    auto evaluations_x = Derived::ComputeEvaluations(x);
    auto evaluations_y = Derived::ComputeEvaluations(y);
    decltype(evaluations_x) evaluations_z;
    for (size_t i = 0; i < evaluations_x.size(); ++i) {
      evaluations_z[i] = evaluations_x[i] * evaluations_y[i];
    }

    // Step 3 & 4: Interpolation (Recovery) and Reduction (Modulo uⁿ - ξ)
    return Reduce(Interpolate(evaluations_z));
  }

  // Squares this element using the Toom-Cook method.
  Derived ToomCookSquare() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();

    auto evaluations_x = Derived::ComputeEvaluations(x);
    decltype(evaluations_x) evaluations_y;
    for (size_t i = 0; i < evaluations_x.size(); ++i) {
      evaluations_y[i] = evaluations_x[i].Square();
    }

    return Reduce(Interpolate(evaluations_y));
  }

 private:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;
  constexpr static size_t kNumEvaluations = 2 * kDegree - 1;

  // Polynomial interpolation using inverse Vandermonde matrix.
  //
  // Given N evaluation values v = {f(t₀), ..., f(tₙ₋₁)}, recovers coefficients
  // c by solving the linear system c = V⁻¹ * v.
  //
  // Derived class must implement:
  // - GetVandermondeInverseMatrix(): Returns the precomputed N×N inverse
  //   matrix.
  std::array<BaseField, kNumEvaluations> Interpolate(
      const std::array<BaseField, kNumEvaluations>& v) const {
    const auto& matrix =
        static_cast<const Derived&>(*this).GetVandermondeInverseMatrix();

    std::array<BaseField, kNumEvaluations> c;
    for (size_t i = 0; i < kNumEvaluations; ++i) {
      c[i] = matrix[i][0] * v[0];
      for (size_t j = 1; j < kNumEvaluations; ++j) {
        c[i] += matrix[i][j] * v[j];
      }
    }
    return c;
  }

  // Reduces polynomial C(u) of degree 2n-2 modulo (uⁿ - ξ) to a
  // polynomial of degree n-1.
  //
  // For i < n-1, the coefficient zᵢ = cᵢ + ξ * cᵢ₊ₙ.
  // The highest coefficient zₙ₋₁ = cₙ₋₁.
  Derived Reduce(const std::array<BaseField, kNumEvaluations>& c) const {
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();
    std::array<BaseField, kDegree> ret;
    for (size_t i = 0; i < kDegree - 1; ++i) {
      ret[i] = c[i] + non_residue * c[i + kDegree];
    }
    ret[kDegree - 1] = c[kDegree - 1];
    return static_cast<const Derived&>(*this).FromBaseFields(ret);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_TOOM_COOK_OPERATION_H_
