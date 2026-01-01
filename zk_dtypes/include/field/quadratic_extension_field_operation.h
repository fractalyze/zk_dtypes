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

#include "absl/status/statusor.h"

#include "zk_dtypes/include/field/extension_field_operation.h"
#include "zk_dtypes/include/field/karatsuba_operation.h"

namespace zk_dtypes {

template <typename Derived>
class QuadraticExtensionFieldOperation
    : public ExtensionFieldOperation<Derived>,
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
      std::array<BaseField, 2> x =
          static_cast<const Derived&>(*this).ToBaseFields();
      // v₀ = x₀ - x₁
      BaseField v0 = x[0] - x[1];
      // v₁ = x₀ * x₁
      BaseField v1 = x[0] * x[1];

      return static_cast<const Derived&>(*this).FromBaseFields(
          {v0 * (x[0] + x[1]), v1.Double()});
    } else if (algorithm == ExtensionFieldMulAlgorithm::kCustom2) {
      // [Comparison]
      // Custom Algorithm:
      // - square: 0, mul: 2, mul by non-residue: 2, add: 1, sub: 2, double: 1
      // Default Karatsuba:
      // - square: 2, mul: 1, mul by non-residue: 1, add: 1, sub: 0, double: 1
      //
      // Conclusion:
      // If 'mul/square' is significantly more expensive than 'mul by
      // non-residue + sub', this algorithm is faster. This holds true for
      // almost all large prime fields assuming ξ is a small constant.
      std::array<BaseField, 2> x =
          static_cast<const Derived&>(*this).ToBaseFields();
      BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

      // v₀ = x₀ - x₁
      BaseField v0 = x[0] - x[1];
      // v₁ = x₀ * x₁
      BaseField v1 = x[0] * x[1];
      // v₂ = x₀ - x₁ * ξ
      BaseField v2 = x[0] - non_residue * x[1];

      // v₃ = v₀ * v₂ = (x₀ - x₁)(x₀ - x₁ξ)
      //    = x₀² - x₀x₁ξ - x₀x₁ + x₁²ξ
      //    = (x₀² + x₁²ξ) - x₀x₁(1 + ξ)
      BaseField v3 = v0 * v2;

      // y₀ = x₀² + x₁²ξ
      //    = v₃ + x₀x₁(1 + ξ)
      //    = v₃ + v₁ + v₁ξ
      BaseField y0 = v3 + v1 + non_residue * v1;

      // y₁ = 2x₀x₁
      BaseField y1 = v1.Double();

      return static_cast<const Derived&>(*this).FromBaseFields({y0, y1});
    } else {
      return this->KaratsubaSquare();
    }
  }

  absl::StatusOr<Derived> Inverse() const {
    std::array<BaseField, 2> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
    // Guide to Pairing-based Cryptography, Algorithm 5.19.
    // v1 = x[1]²
    BaseField v1 = x[1].Square();
    // v0 = x[0]² - q * v1
    BaseField v0 = x[0].Square() - v1 * non_residue;

    absl::StatusOr<BaseField> v0_inv = v0.Inverse();
    if (!v0_inv.ok()) return v0_inv.status();
    std::array<BaseField, 2> y{x[0] * (*v0_inv), -x[1] * (*v0_inv)};
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_QUADRATIC_EXTENSION_FIELD_OPERATION_H_
