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

#ifndef ZK_DTYPES_INCLUDE_FIELD_POLYNOMIAL_REDUCTION_H_
#define ZK_DTYPES_INCLUDE_FIELD_POLYNOMIAL_REDUCTION_H_

#include <array>
#include <cstddef>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// Provides polynomial reduction modulo the irreducible polynomial.
// This is shared between KaratsubaOperation and ToomCookOperation.
//
// Reduces a polynomial of degree 2n-2 to degree n-1 modulo the
// irreducible polynomial defining the extension field.

// Reduces the product polynomial C(u) modulo the irreducible polynomial.
//
// For simple form (uⁿ = ξ):
//   Since uⁿ ≡ ξ, we have uⁱ⁺ⁿ ≡ ξ·uⁱ.
//   The resulting coefficients are zᵢ = cᵢ + ξ·cᵢ₊ₙ.
//
// For general form (uⁿ = c₀ + c₁u + ... + cₙ₋₁uⁿ⁻¹):
//   Process from highest degree down, substituting uⁿ with the polynomial.
template <typename Derived, size_t ProductSize>
Derived ReducePolynomial(
    const Derived& self,
    const std::array<typename ExtensionFieldOperationTraits<Derived>::BaseField,
                     ProductSize>& c_in) {
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr size_t kDegree = ExtensionFieldOperationTraits<Derived>::kDegree;

  if (self.HasSimpleNonResidue()) {
    // Optimized path for Xⁿ = ξ (constant non-residue)
    BaseField non_residue = self.NonResidue();
    std::array<BaseField, kDegree> ret;
    for (size_t i = 0; i < kDegree - 1; ++i) {
      ret[i] = c_in[i] + non_residue * c_in[i + kDegree];
    }
    ret[kDegree - 1] = c_in[kDegree - 1];
    return self.FromCoeffs(ret);
  } else {
    // General path for Xⁿ = c₀ + c₁*X + ... + cₙ₋₁*Xⁿ⁻¹
    // Process from highest degree down to n
    std::array<BaseField, ProductSize> c = c_in;
    std::array<BaseField, kDegree> irreducible = self.IrreducibleCoeffs();

    // For each term c[i]*Xⁱ where i >= n, replace Xⁿ with the polynomial
    // and distribute: c[i]*Xⁱ = c[i]*Xⁱ⁻ⁿ*(c₀ + c₁*X + ... + cₙ₋₁*Xⁿ⁻¹)
    for (size_t i = ProductSize - 1; i >= kDegree; --i) {
      if (!c[i].IsZero()) {
        // Xⁱ = Xⁱ⁻ⁿ * Xⁿ ≡ Xⁱ⁻ⁿ * (c₀ + c₁*X + ...)
        // So c[i]*Xⁱ contributes c[i]*cⱼ to coefficient of Xⁱ⁻ⁿ⁺ʲ
        for (size_t j = 0; j < kDegree; ++j) {
          c[i - kDegree + j] += c[i] * irreducible[j];
        }
        c[i] = c[i].CreateConst(0);
      }
    }

    // Extract the reduced coefficients
    std::array<BaseField, kDegree> ret;
    for (size_t i = 0; i < kDegree; ++i) {
      ret[i] = c[i];
    }
    return self.FromCoeffs(ret);
  }
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_POLYNOMIAL_REDUCTION_H_
