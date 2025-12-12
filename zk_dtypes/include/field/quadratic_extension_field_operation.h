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

#ifndef ZK_DTYPES_INCLUDE_FIELD_QUADRATIC_EXTENSION_FIELD_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_QUADRATIC_EXTENSION_FIELD_OPERATION_H_

#include <array>

#if defined(ZK_DTYPES_USE_ABSL)
#include "absl/status/statusor.h"
#endif
#include "zk_dtypes/include/field/extension_field_operation.h"

namespace zk_dtypes {

template <typename Derived>
class QuadraticExtensionFieldOperation
    : public ExtensionFieldOperation<Derived> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static bool kHasHint =
      ExtensionFieldOperationTraits<Derived>::kHasHint;

  Derived operator*(const Derived& other) const {
    std::array<BaseField, 2> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, 2> y =
        static_cast<const Derived&>(other).ToBaseField();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
    // Karatsuba multiplication;
    // Guide to Pairing-based cryptography, Algorithm 5.16.
    // v₀ = x₀ * y₀
    BaseField v0 = x[0] * y[0];
    // v₁ = x₁ * y₁
    BaseField v1 = x[1] * y[1];

    // z₀ = x₀ * y₀ + q * x₁ * y₁
    BaseField z0 = v0 + non_residue * v1;
    // z₁ = (x₀ + x₁) * (y₀ + y₁) - v₀ - v₁
    // z₁ = x₀ * y₁ + x₁ * y₀
    BaseField z1 = (x[0] + x[1]) * (y[0] + y[1]) - v0 - v1;
    return static_cast<const Derived&>(*this).FromBaseFields({z0, z1});
  }

  Derived Square() const {
    std::array<BaseField, 2> x =
        static_cast<const Derived&>(*this).ToBaseField();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // v₀ = x₀ - x₁
    BaseField v0 = x[0] - x[1];
    // v₁ = x₀ * x₁
    BaseField v1 = x[0] * x[1];
    if constexpr (kHasHint) {
      if constexpr (ExtensionFieldOperationTraits<
                        Derived>::kNonResidueIsMinusOne) {
        // When the non-residue is -1, we save 2 intermediate additions,
        // and use one fewer intermediate variable
        // (x₀² - x[1]², 2 * x₀ * x[1])
        return static_cast<const Derived&>(*this).FromBaseFields(
            {v0 * (x[0] + x[1]), v1.Double()});
      }
    }
    // v₂ = x₀ - x[1] * q
    BaseField v2 = x[0] - x[1] * non_residue;

    // v₃ = (v₀ * v₂)
    //    = (x₀ - x₁) * (x₀ - x₁ * q)
    //    = x₀² - x₀ * x₁ * q - x₀ * x₁ + x₁² * q
    //    = x₀² - (q + 1) * x₀ * x₁ + x₁² * q
    //    = x₀² + x₁² * q - (q + 1) * x₀ * x₁
    BaseField v3 = v0 * v2;
    // clang-format off
    // y₀ = v₃ + (q + 1) * x₀ * x₁
    //    = x₀² + x₁² * q - (q + 1) * x₀ * x₁ + (q + 1) * x₀ * x₁
    //    = x₀² + x₁² * q
    // clang-format on
    // y₁ = 2 * x₀ * x₁
    BaseField y0 = v3 + non_residue * v1 + v1;
    BaseField y1 = v1.Double();
    return static_cast<const Derived&>(*this).FromBaseFields({y0, y1});
  }

#if defined(ZK_DTYPES_USE_ABSL)
  absl::StatusOr<Derived>
#else
  Derived
#endif
  Inverse() const {
    std::array<BaseField, 2> x =
        static_cast<const Derived&>(*this).ToBaseField();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
    // Guide to Pairing-based Cryptography, Algorithm 5.19.
    // v1 = x[1]²
    BaseField v1 = x[1].Square();
    // v0 = x[0]² - q * v1
    BaseField v0 = x[0].Square() - v1 * non_residue;

#if defined(ZK_DTYPES_USE_ABSL)
    absl::StatusOr<BaseField> v0_inv = v0.Inverse();
    if (!v0_inv.ok()) return v0_inv.status();
    std::array<BaseField, 2> y{x[0] * (*v0_inv), -x[1] * (*v0_inv)};
#else
    BaseField v0_inv = v0.Inverse();
    std::array<BaseField, 2> y{x[0] * v0_inv, -x[1] * v0_inv};
#endif
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_QUADRATIC_EXTENSION_FIELD_OPERATION_H_
