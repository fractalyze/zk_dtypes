/* Copyright 2026 The zk_dtypes Authors.

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

#ifndef ZK_DTYPES_INCLUDE_FIELD_POLYNOMIAL_EEA_INVERSE_H_
#define ZK_DTYPES_INCLUDE_FIELD_POLYNOMIAL_EEA_INVERSE_H_

#include <algorithm>
#include <array>
#include <cstddef>

#include "absl/log/check.h"

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// Extended Euclidean Algorithm (EEA) for polynomial inverse in extension
// fields.
//
// This class provides a general inverse algorithm that works for any
// irreducible polynomial, not just the simple form Xⁿ = ξ.
//
// For f(X) in F[X]/(m(X)) where m(X) is the irreducible polynomial:
//   gcd(f(X), m(X)) = u(X)*f(X) + v(X)*m(X) = 1
// Since we work modulo m(X), u(X) is the inverse of f(X).
template <typename Derived>
class PolynomialEEAInverse {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;

 private:
  // Finds the degree of a polynomial represented as an array.
  // Returns -1 if the polynomial is zero.
  template <size_t N>
  static int FindDegree(const std::array<BaseField, N>& poly) {
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      if (!poly[i].IsZero()) {
        return i;
      }
    }
    return -1;
  }

 public:
  // Computes the inverse of this element using Extended Euclidean Algorithm.
  // Returns Zero() if not invertible (i.e., element is zero).
  Derived EEAInverse() const {
    const auto& self = static_cast<const Derived&>(*this);
    const std::array<BaseField, kDegree>& f = self.ToCoeffs();

    // Check if zero
    if (std::all_of(f.begin(), f.end(),
                    [](const BaseField& x) { return x.IsZero(); })) {
      return self.FromCoeffs(f);  // Return zero
    }

    // Get the irreducible polynomial coefficients.
    // m(X) = Xⁿ - (c₀ + c₁*X + ... + cₙ₋₁*Xⁿ⁻¹)
    // where kIrreducibleCoeffs = {c₀, c₁, ..., cₙ₋₁}
    std::array<BaseField, kDegree> irreducible = self.IrreducibleCoeffs();

    // We implement EEA for polynomials.
    // r₀ = m(X) (degree n), r₁ = f(X) (degree < n)
    // s₀ = 0, s₁ = 1
    // We maintain: rᵢ = sᵢ * f (mod m)
    // When rᵢ becomes degree 0 (constant), sᵢ is the inverse.

    // Represent polynomials with one extra coefficient for the leading term of
    // m(X).
    constexpr size_t kSize = kDegree + 1;

    // r0 = m(X) = Xⁿ - irreducible[0] - irreducible[1]*X - ...
    // = -c₀ - c₁*X - ... - cₙ₋₁*Xⁿ⁻¹ + Xⁿ
    std::array<BaseField, kSize> r0;
    for (size_t i = 0; i < kDegree; ++i) {
      r0[i] = -irreducible[i];
    }
    r0[kDegree] = f[0].CreateConst(1);  // Leading coefficient of Xⁿ

    // r₁ = f(X)
    std::array<BaseField, kSize> r1;
    for (size_t i = 0; i < kDegree; ++i) {
      r1[i] = f[i];
    }
    r1[kDegree] = f[0].CreateConst(0);

    // s₀ = 0, s₁ = 1
    std::array<BaseField, kSize> s0;
    std::array<BaseField, kSize> s1;
    s1[0] = f[0].CreateConst(1);

    // Extended Euclidean Algorithm
    while (true) {
      int deg_r1 = FindDegree(r1);

      // If r₁ is constant (degree ≤ 0), we're done
      if (deg_r1 <= 0) {
        break;
      }

      int deg_r0 = FindDegree(r0);

      // Perform one step of EEA: r₀ = r₀ - q * r₁
      // where q = (leading coeff of r₀ / leading coeff of r₁) * X^(deg_r0 -
      // deg_r1)
      if (deg_r0 < deg_r1) {
        // Swap r0 <-> r1, s0 <-> s1
        std::swap(r0, r1);
        std::swap(s0, s1);
        continue;
      }

      int deg_diff = deg_r0 - deg_r1;
      BaseField lead_r1_inv = r1[deg_r1].Inverse();
      BaseField q_coeff = r0[deg_r0] * lead_r1_inv;

      // r₀ = r₀ - q_coeff * X^deg_diff * r₁
      for (int i = 0; i <= deg_r1; ++i) {
        r0[i + deg_diff] = r0[i + deg_diff] - q_coeff * r1[i];
      }

      // s₀ = s₀ - q_coeff * X^deg_diff * s₁
      for (size_t i = 0; i + deg_diff < kSize; ++i) {
        s0[i + deg_diff] = s0[i + deg_diff] - q_coeff * s1[i];
      }
    }

    // Now r₁ should be a non-zero constant.
    // s₁ * f ≡ r₁ (mod m)
    // So inverse = s₁ / r₁[0]

    // Find the constant value in r1 (should be r1[0] since deg <= 0)
    BaseField r1_const = r1[0];
    // For a non-zero element f, gcd(f, m) = 1 since m is irreducible.
    // r1_const being zero would indicate a logic error.
    DCHECK(!r1_const.IsZero());
    if (r1_const.IsZero()) {
      // Should not happen for non-zero f, but handle gracefully
      std::array<BaseField, kDegree> zero_result;
      return self.FromCoeffs(zero_result);
    }

    BaseField r1_inv = r1_const.Inverse();
    std::array<BaseField, kDegree> result;
    for (size_t i = 0; i < kDegree; ++i) {
      result[i] = s1[i] * r1_inv;
    }

    return self.FromCoeffs(result);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_POLYNOMIAL_EEA_INVERSE_H_
