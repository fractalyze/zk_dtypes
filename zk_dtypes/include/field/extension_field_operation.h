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

  // Polynomial interpolation using inverse Vandermonde matrix.
  //
  // Given N evaluation points v = {v₀, v₁, ..., vₙ₋₁}, compute
  // N polynomial coefficients c = {c₀, c₁, ..., cₙ₋₁} using:
  //   c = V⁻¹ * v
  //
  // Derived class must implement:
  //   - GetVandermondeInverseMatrix() returning N×N matrix
  template <size_t N>
  std::array<BaseField, N> Interpolate(
      const std::array<BaseField, N>& v) const {
    const auto& matrix =
        static_cast<const Derived&>(*this).GetVandermondeInverseMatrix();

    std::array<BaseField, N> c;
    for (size_t i = 0; i < N; ++i) {
      c[i] = matrix[i][0] * v[0];
      for (size_t j = 1; j < N; ++j) {
        c[i] += matrix[i][j] * v[j];
      }
    }
    return c;
  }

  // Reduce polynomial coefficients using uⁿ = ξ
  // c[0..2n-2] -> z[0..n-1] where zᵢ = cᵢ + ξ * cᵢ₊ₙ (for i < n-1), zₙ₋₁ = cₙ₋₁
  template <size_t N>
  Derived Reduce(const std::array<BaseField, N>& c) const {
    constexpr size_t kReducedSize = (N + 1) / 2;
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();
    std::array<BaseField, kReducedSize> ret;
    for (size_t i = 0; i < kReducedSize - 1; ++i) {
      ret[i] = c[i] + non_residue * c[i + kReducedSize];
    }
    ret[kReducedSize - 1] = c[kReducedSize - 1];
    return static_cast<const Derived&>(*this).FromBaseFields(ret);
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

    // BaseField inverse (cheaper than extension field inverse)
    absl::StatusOr<BaseField> norm_inv = norm.Inverse();
    if (!norm_inv.ok()) return norm_inv.status();
    // x⁻¹ = φ(x) · ... · φⁿ⁻¹(x) · norm⁻¹
    return frob_product * (*norm_inv);
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
  absl::StatusOr<Derived> operator/(const Derived& other) const {
    absl::StatusOr<Derived> inv = other.Inverse();
    if (!inv.ok()) return inv.status();
    return static_cast<const Derived&>(*this) * inv.value();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
