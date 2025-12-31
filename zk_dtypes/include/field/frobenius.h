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

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// clang-format off
// Frobenius endomorphism: φᴱ(x) = x^(pᴱ)
//
// For extension field F[u] / (uⁿ - ξ):
//   φᴱ(aᵢ · uⁱ) = aᵢ · ξ^(i * (pᴱ - 1) / n) · uⁱ
//
// where p is order of F.
//
// See:
// https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.2.-optimized-computation-when
// clang-format on
// Core Frobenius logic with coefficients passed as parameter.
// coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) for i = 1, ..., n - 1.
template <size_t E, typename T, typename CoeffsArray>
[[nodiscard]] T ApplyFrobenius(const T& x, const CoeffsArray& coeffs) {
  using BaseField = typename ExtensionFieldOperationTraits<T>::BaseField;
  constexpr size_t kDegree = ExtensionFieldOperationTraits<T>::kDegree;

  const std::array<BaseField, kDegree>& src = x.ToBaseFields();
  std::array<BaseField, kDegree> dst;

  // a₀ · u⁰ → a₀ · u⁰ (coefficient is 1)
  dst[0] = src[0];
  for (size_t i = 1; i < kDegree; ++i) {
    // aᵢ · uⁱ → aᵢ · ξ^(i * (pᴱ - 1) / n) · uⁱ
    dst[i] = src[i] * coeffs[E - 1][i - 1];
  }

  return x.FromBaseFields(dst);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_H_
