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

#if defined(ZK_DTYPES_USE_ABSL)
#include "absl/status/statusor.h"
#endif
#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"
#include "zk_dtypes/include/field/frobenius.h"

namespace zk_dtypes {

template <typename Derived>
class ExtensionFieldOperation {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;

  Derived operator+(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseField();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] + y[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(z);
  }

  Derived operator-(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseField();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] - y[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(z);
  }

  Derived operator-() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = -x[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }

  Derived Double() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
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
#if defined(ZK_DTYPES_USE_ABSL)
  absl::StatusOr<Derived>
#else
  Derived
#endif
  Inverse() const {
    const Derived& self = static_cast<const Derived&>(*this);

    // Compute φ¹(x) · φ²(x) · ... · φⁿ⁻¹(x) using precomputed coefficients.
    // Each Frobenius<E> uses coeffs[E - 1] from GetFrobeniusCoeffs().
    // See
    // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.-frobenius-endomorphism
    Derived frob_product =
        ComputeFrobeniusProduct(self, std::make_index_sequence<kDegree - 1>{});

    // Norm(x) = x · φ(x) · ... · φⁿ⁻¹(x) ∈ BaseField
    // Result is [norm, 0, ..., 0] in extension field representation.
    // See
    // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-3.-norm
    Derived norm_ext = self * frob_product;
    BaseField norm = norm_ext.ToBaseField()[0];

#if defined(ZK_DTYPES_USE_ABSL)
    // BaseField inverse (cheaper than extension field inverse)
    absl::StatusOr<BaseField> norm_inv = norm.Inverse();
    if (!norm_inv.ok()) return norm_inv.status();
    // x⁻¹ = φ(x) · ... · φⁿ⁻¹(x) · norm⁻¹
    return frob_product * (*norm_inv);
#else
    BaseField norm_inv = norm.Inverse();
    return frob_product * norm_inv;
#endif
  }

 private:
  // Compute φ¹(x) · φ²(x) · ... · φⁿ⁻¹(x) using fold expression.
  // Each Frobenius<E> directly uses precomputed coeffs[E - 1].
  template <size_t... Es>
  static Derived ComputeFrobeniusProduct(const Derived& x,
                                         std::index_sequence<Es...>) {
    return (Frobenius<Es + 1>(x) * ...);
  }

 public:
#if defined(ZK_DTYPES_USE_ABSL)
  absl::StatusOr<Derived> operator/(const Derived& other) const {
    absl::StatusOr<Derived> inv = other.Inverse();
    if (!inv.ok()) return inv.status();
    return static_cast<const Derived&>(*this) * inv.value();
  }
#else
  Derived operator/(const Derived& other) const {
    return static_cast<const Derived&>(*this) * other.Inverse();
  }
#endif
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
