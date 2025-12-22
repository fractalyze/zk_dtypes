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

#ifndef ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_H_
#define ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// clang-format off
// Frobenius endomorphism: φᴱ(x) = x^(pᴱ)
//
// For extension field Fₚ[u] / (uⁿ - ξ):
//   φᴱ(aᵢ · uⁱ) = φᴱ(aᵢ) · ξ^(i * (pᴱ - 1) / n) · uⁱ
//
// See:
// https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.2.-optimized-computation-when
// clang-format on

// Apply Frobenius to base field element.
// For prime field (ExtensionDegree() == 1), Frobenius is identity.
template <size_t E, typename T,
          std::enable_if_t<T::ExtensionDegree() == 1>* = nullptr>
[[nodiscard]] T ApplyFrobeniusToBase(const T& x) {
  return x;
}

// For extension field, recursively apply Frobenius.
template <size_t E, typename T,
          std::enable_if_t<(T::ExtensionDegree() > 1)>* = nullptr>
[[nodiscard]] T ApplyFrobeniusToBase(const T& x) {
  return x.template Frobenius<E>();
}

// Core Frobenius logic with coefficients passed as parameter.
// coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) for i = 1, ..., n - 1.
template <size_t E, typename T, typename CoeffsArray>
[[nodiscard]] T ApplyFrobenius(const T& x, const CoeffsArray& coeffs) {
  using BaseField = typename ExtensionFieldOperationTraits<T>::BaseField;
  constexpr size_t kDegree = ExtensionFieldOperationTraits<T>::kDegree;

  std::array<BaseField, kDegree> src = x.ToBaseField();
  std::array<BaseField, kDegree> dst;

  // a₀ · u⁰ → φᴱ(a₀) · u⁰ (coefficient is 1)
  dst[0] = ApplyFrobeniusToBase<E>(src[0]);
  for (size_t i = 1; i < kDegree; ++i) {
    // aᵢ · uⁱ → φᴱ(aᵢ) · ξ^(i * (pᴱ - 1) / n) · uⁱ
    dst[i] = ApplyFrobeniusToBase<E>(src[i]) * coeffs[E - 1][i - 1];
  }

  return x.FromBaseFields(dst);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_H_
