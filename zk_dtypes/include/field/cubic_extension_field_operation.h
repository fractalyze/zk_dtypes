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

namespace zk_dtypes {

template <typename Derived>
class CubicExtensionFieldOperation : public ExtensionFieldOperation<Derived> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;

  // Multiplication in Fp3 using Karatsuba method.
  // See https://eprint.iacr.org/2006/471.pdf
  // Devegili OhEig Scott Dahab --- Multiplication and Squaring on
  // Pairing-Friendly Fields; Section 4 (Karatsuba)
  //
  // For x = x₀ + x₁·u + x₂·u² and y = y₀ + y₁·u + y₂·u² where u³ = ξ:
  //
  // v₀ = x₀ * y₀
  // v₁ = x₁ * y₁
  // v₂ = x₂ * y₂
  // v₃ = (x₀ + x₁) * (y₀ + y₁) - v₀ - v₁
  // v₄ = (x₀ + x₂) * (y₀ + y₂) - v₀ - v₂
  // v₅ = (x₁ + x₂) * (y₁ + y₂) - v₁ - v₂
  //
  // Result:
  // z₀ = v₀ + ξ * v₅
  // z₁ = v₃ + ξ * v₂
  // z₂ = v₄ + v₁
  Derived operator*(const Derived& other) const {
    std::array<BaseField, 3> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, 3> y =
        static_cast<const Derived&>(other).ToBaseField();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // v₀ = x₀ * y₀
    BaseField v0 = x[0] * y[0];
    // v₁ = x₁ * y₁
    BaseField v1 = x[1] * y[1];
    // v₂ = x₂ * y₂
    BaseField v2 = x[2] * y[2];

    // v₃ = (x₀ + x₁) * (y₀ + y₁) - v₀ - v₁
    BaseField v3 = (x[0] + x[1]) * (y[0] + y[1]) - v0 - v1;
    // v₄ = (x₀ + x₂) * (y₀ + y₂) - v₀ - v₂
    BaseField v4 = (x[0] + x[2]) * (y[0] + y[2]) - v0 - v2;
    // v₅ = (x₁ + x₂) * (y₁ + y₂) - v₁ - v₂
    BaseField v5 = (x[1] + x[2]) * (y[1] + y[2]) - v1 - v2;

    // z₀ = v₀ + ξ * v₅
    BaseField z0 = v0 + non_residue * v5;
    // z₁ = v₃ + ξ * v₂
    BaseField z1 = v3 + non_residue * v2;
    // z₂ = v₄ + v₁
    BaseField z2 = v4 + v1;

    return static_cast<const Derived&>(*this).FromBaseFields({z0, z1, z2});
  }

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
  Derived Square() const {
    std::array<BaseField, 3> x =
        static_cast<const Derived&>(*this).ToBaseField();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // s₀ = x₀²
    BaseField s0 = x[0].Square();
    // s₁ = 2 * x₀ * x₁
    BaseField s1 = (x[0] * x[1]).Double();
    // s₂ = (x₀ - x₁ + x₂)²
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
    BaseField y2 = s1 + s2 + s3 - s0 - s4;

    return static_cast<const Derived&>(*this).FromBaseFields({y0, y1, y2});
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_CUBIC_EXTENSION_FIELD_OPERATION_H_
