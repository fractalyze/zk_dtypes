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

#include <array>
#include <cstddef>
#include <utility>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// clang-format off
// Frobenius Operation Mixin for Extension Fields.
//
// This mixin provides the Frobenius endomorphism φᴱ(x) = x^(pᴱ) as a member
// function, using instance method to obtain Frobenius coefficients, where p
// is order of base field.
//
//
// Derived class must implement:
//   - GetFrobeniusCoeffs(): returns (n - 1) × (n - 1) array of coefficients
//     where coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n)
//   - ToCoeffs(): converts to array of base field elements
//   - FromCoeffs(): constructs from array of base field elements
//
// References:
// - https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion
// clang-format on
template <typename Derived>
class FrobeniusOperation {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;

  // clang-format off
  // Base-field Frobenius endomorphism: φᴱ(x) = x^(pᴱ)
  //
  // For extension field F[u] / (uⁿ - ξ):
  //   φᴱ(aᵢ · uⁱ) = aᵢ · ξ^(i * (pᴱ - 1) / n) · uⁱ
  //
  // where p = |F| is the order of the base field.
  // Does NOT recurse into tower levels — each aᵢ is left unchanged.
  // Used for FrobeniusInverse (base-field-linear norm computation).
  //
  // See:
  // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.2.-optimized-computation-when
  // clang-format on
  // Core Frobenius logic with coefficients passed as parameter.
  // coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) for i = 1, ..., n - 1.
  template <size_t E = 1>
  Derived Frobenius() const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    std::array<BaseField, kDegree> y;
    const std::array<std::array<BaseField, kDegree - 1>, kDegree - 1>& coeffs =
        static_cast<const Derived&>(*this).GetFrobeniusCoeffs();

    // a₀ · u⁰ → a₀ · u⁰ (coefficient is 1)
    y[0] = x[0];
    for (size_t i = 1; i < kDegree; ++i) {
      // aᵢ · uⁱ → aᵢ · ξ^(i * (pᴱ - 1) / n) · uⁱ
      y[i] = x[i] * coeffs[E - 1][i - 1];
    }

    return static_cast<const Derived&>(*this).FromCoeffs(y);
  }

  // clang-format off
  // Tower-aware Frobenius map: x ↦ x^(qᴱ)
  //
  // For tower extensions (e.g., Fp12 = Fp6[w] = Fp2[v][w]), computes the
  // q-power Frobenius where q = |BasePrimeField| (the prime).
  //
  // Unlike Frobenius<E> which uses p = |BaseField|, this method:
  //   1. Recursively applies FrobeniusMap<E> to base field components
  //   2. Multiplies by ξ^(i * (qᴱ - 1) / n) using tower Frobenius coefficients
  //
  // Since x^(q^D) = x for any x in the extension field (where D is the total
  // extension degree over the prime field), we reduce E mod D at each level.
  //
  // This is needed for pairing final exponentiation where Frobenius powers
  // 1, 2, 3 are applied to Fp12 elements relative to the prime field.
  // clang-format on
  template <size_t E = 1>
  Derived FrobeniusMap() const {
    // Total extension degree over the prime field.
    constexpr size_t D = kDegree * BaseField::ExtensionDegree();
    // Reduce E mod D: x^(q^D) = x, so FrobeniusMap<E> = FrobeniusMap<E mod D>.
    constexpr size_t EffE = E % D;

    // If EffE == 0, Frobenius is identity.
    if constexpr (EffE == 0) {
      return static_cast<const Derived&>(*this);
    } else {
      const std::array<BaseField, kDegree>& x =
          static_cast<const Derived&>(*this).ToCoeffs();
      std::array<BaseField, kDegree> y;
      const auto& coeffs =
          static_cast<const Derived&>(*this).GetTowerFrobeniusCoeffs();

      if constexpr (BaseField::ExtensionDegree() > 1) {
        // Tower extension: recursively apply FrobeniusMap to base field
        // elements. The recursive call will reduce E mod its own D.
        y[0] = x[0].template FrobeniusMap<E>();
        for (size_t i = 1; i < kDegree; ++i) {
          y[i] = x[i].template FrobeniusMap<E>() * coeffs[EffE - 1][i - 1];
        }
      } else {
        // Base case: BaseField is a prime field, x^(qᴱ) = x for x ∈ Fp
        y[0] = x[0];
        for (size_t i = 1; i < kDegree; ++i) {
          y[i] = x[i] * coeffs[EffE - 1][i - 1];
        }
      }

      return static_cast<const Derived&>(*this).FromCoeffs(y);
    }
  }

  // Inverse in extension field using Frobenius endomorphism.
  //
  // Using the norm: Norm(x) = x · φ(x) · ... · φⁿ⁻¹(x) ∈ BaseField
  // where φ(x) = xᵖ is the Frobenius endomorphism.
  //
  // From Norm(x) = x · φ(x) · ... · φⁿ⁻¹(x), we derive:
  //   x⁻¹ = φ(x) · ... · φⁿ⁻¹(x) / Norm(x)
  //
  // Since Norm(x) ∈ base field, we only need base field inverse (cheaper than
  // extension field inverse).
  //
  // Note: Child classes may override this with more efficient algorithms.
  // Returns the multiplicative inverse. Returns Zero() if not invertible.
  Derived FrobeniusInverse() const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    BaseField non_residue = static_cast<const Derived&>(*this).NonResidue();

    // Compute φ¹(x) · φ²(x) · ... · φⁿ⁻¹(x) using precomputed coefficients.
    // Each Frobenius<E> uses coeffs[E - 1] from GetFrobeniusCoeffs().
    // See
    // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.-frobenius-endomorphism
    Derived frob_product =
        ComputeFrobeniusProduct(std::make_index_sequence<kDegree - 1>{});
    const std::array<BaseField, kDegree>& field_product_comp =
        frob_product.ToCoeffs();

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

    // BaseField inverse (cheaper than extension field inverse).
    // Inverse() returns Zero() if not invertible.
    BaseField norm_inv = norm.Inverse();
    // x⁻¹ = φ(x) · ... · φⁿ⁻¹(x) · norm⁻¹
    return frob_product * norm_inv;
  }

 private:
  // Compute φ¹(x) · φ²(x) · ... · φⁿ⁻¹(x) using fold expression.
  template <size_t... Es>
  Derived ComputeFrobeniusProduct(std::index_sequence<Es...>) const {
    return (this->template Frobenius<Es + 1>() * ...);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_OPERATION_H_
