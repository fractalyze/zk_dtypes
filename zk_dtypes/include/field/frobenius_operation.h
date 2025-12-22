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

#ifndef ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_OPERATION_H_

#include "zk_dtypes/include/field/frobenius.h"

namespace zk_dtypes {

// clang-format off
// Frobenius Operation Mixin for Extension Fields.
//
// This mixin provides the Frobenius endomorphism φᴱ(x) = x^(pᴱ) as a member
// function, using instance method to obtain Frobenius coefficients.
//
// Derived class must implement:
//   - GetFrobeniusCoeffs(): returns (n - 1) × (n - 1) array of coefficients
//     where coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n)
//   - ToBaseField(): converts to array of base field elements
//   - FromBaseFields(): constructs from array of base field elements
//
// References:
// - https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion
// clang-format on
template <typename Derived>
class FrobeniusOperation {
 public:
  template <size_t E = 1>
  Derived Frobenius() const {
    const Derived& self = static_cast<const Derived&>(*this);
    return ApplyFrobenius<E>(self, self.GetFrobeniusCoeffs());
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_OPERATION_H_
