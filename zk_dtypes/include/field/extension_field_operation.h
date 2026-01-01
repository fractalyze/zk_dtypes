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

#ifndef ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_

#include <array>
#include <cstddef>
#include <utility>

#include "absl/status/statusor.h"

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"
#include "zk_dtypes/include/field/frobenius_operation.h"

namespace zk_dtypes {

enum class ExtensionFieldMulAlgorithm {
  kCustom,
  kCustom2,
  kKaratsuba,
  kToomCook,
};

template <typename Derived>
class ExtensionFieldOperation : public FrobeniusOperation<Derived> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;

  Derived operator+(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseFields();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] + y[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(z);
  }

  Derived operator-(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseFields();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] - y[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(z);
  }

  Derived operator-() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = -x[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }

  Derived Double() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = x[i].Double();
    }
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }

  // Inverse in extension field using Frobenius endomorphism.
  //
  // Using the norm: Norm(x) = x · φ(x) · ... · φⁿ⁻¹(x) ∈ Fp
  // where φ(x) = xᵖ is the Frobenius endomorphism.
  //
  // From Norm(x) = x · φ(x) · ... · φⁿ⁻¹(x), we derive:
  //   x⁻¹ = φ(x) · ... · φⁿ⁻¹(x) / Norm(x)
  //
  // Since Norm(x) ∈ Fp, we only need Fp inverse (cheaper than Fpⁿ inverse).
  //
  // Note: Child classes may override this with more efficient algorithms.
  absl::StatusOr<Derived> Inverse() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseFields();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // Compute φ¹(x) · φ²(x) · ... · φⁿ⁻¹(x) using precomputed coefficients.
    // Each Frobenius<E> uses coeffs[E - 1] from GetFrobeniusCoeffs().
    // See
    // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.-frobenius-endomorphism
    Derived frob_product =
        ComputeFrobeniusProduct(std::make_index_sequence<kDegree - 1>{});
    std::array<BaseField, kDegree> field_product_comp =
        frob_product.ToBaseFields();

    // Norm(x) = x · φ(x) · ... · φⁿ⁻¹(x) ∈ BaseField
    // Result is [norm, 0, ..., 0] in extension field representation.
    // See
    // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-3.-norm
    BaseField norm = x[1] * field_product_comp[kDegree - 1];
    for (size_t i = 2; i < kDegree; ++i) {
      norm += x[i] * field_product_comp[kDegree - i];
    }
    norm *= non_residue;
    norm += x[0] * field_product_comp[0];

    // BaseField inverse (cheaper than extension field inverse)
    absl::StatusOr<BaseField> norm_inv = norm.Inverse();
    if (!norm_inv.ok()) return norm_inv.status();
    // x⁻¹ = φ(x) · ... · φⁿ⁻¹(x) · norm⁻¹
    return frob_product * (*norm_inv);
  }

 private:
  // Compute φ¹(x) · φ²(x) · ... · φⁿ⁻¹(x) using fold expression.
  template <size_t... Es>
  Derived ComputeFrobeniusProduct(std::index_sequence<Es...>) const {
    return (this->template Frobenius<Es + 1>() * ...);
  }

 public:
  absl::StatusOr<Derived> operator/(const Derived& other) const {
    absl::StatusOr<Derived> inv = other.Inverse();
    if (!inv.ok()) return inv.status();
    return static_cast<const Derived&>(*this) * inv.value();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
