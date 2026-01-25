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

#ifndef ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_COEFFS_H_
#define ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_COEFFS_H_

#include <array>
#include <cstddef>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/pow.h"

namespace zk_dtypes {

// Mixin providing precomputed Frobenius coefficients for extension fields.
//
// Provides two sets of coefficients:
//
// 1. Standard Frobenius coefficients (GetFrobeniusCoeffs):
//    coeffs[E - 1][i - 1] = ξ^(i * (qᴱ - 1) / n) where q = |BasePrimeField|.
//    Used for Frobenius (q-power Frobenius across the full tower).
//
// 2. Relative Frobenius coefficients (GetRelativeFrobeniusCoeffs):
//    coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) where p = |BaseField|.
//    Used for FrobeniusInverse (base-field-linear Frobenius).
//
// For Fp3: a = (a₁, a₂, a₃, a₄) where
//   a₁ = ξ^((p - 1) / 3), a₂ = ξ^(2(p - 1) / 3)
//   a₃ = ξ^((p² - 1) / 3), a₄ = ξ^(2(p² - 1) / 3)
// - φ¹(x) = (x₀, x₁ * a₁, x₂ * a₂)
// - φ²(x) = (x₀, x₁ * a₃, x₂ * a₄)
//
// See:
// https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.2.-optimized-computation-when
//
template <typename Config>
class FrobeniusCoeffs {
 public:
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  constexpr static size_t N = Config::kDegreeOverBaseField;

  // Total extension degree over the prime field.
  constexpr static size_t D = N * BaseField::ExtensionDegree();

  // Standard Frobenius coefficients using prime field characteristic q.
  //
  // coeffs[E - 1][i - 1] = ξ^(i * (qᴱ - 1) / n) for E = 1..D-1, i = 1..N-1
  // where q = |BasePrimeField| is the prime field order.
  //
  // These are used for the q-power Frobenius across the full extension
  // tower (Frobenius), which is needed for pairing operations.
  static const std::array<std::array<BaseField, N - 1>, D - 1>&
  GetFrobeniusCoeffs() {
    constexpr size_t kLimbNums = BasePrimeField::kLimbNums * D;
    static const auto coeffs =
        ComputeCoeffs<D - 1, kLimbNums>(BasePrimeField::Order());
    return coeffs;
  }

  // Relative Frobenius coefficients using base field order p.
  //
  // coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) for E = 1..N-1, i = 1..N-1
  // where p = |BaseField| is the base field order.
  //
  // These are used for the p-power Frobenius at a single tower level
  // (RelativeFrobenius), which is needed for FrobeniusInverse.
  static const std::array<std::array<BaseField, N - 1>, N - 1>&
  GetRelativeFrobeniusCoeffs() {
    constexpr size_t kLimbNums = BasePrimeField::kLimbNums * D;
    static const auto coeffs =
        ComputeCoeffs<N - 1, kLimbNums>(BaseField::Order());
    return coeffs;
  }

 private:
  // Computes Frobenius coefficients for a given field order.
  //
  // For E = 1..Rows:
  //   result[E - 1][i - 1] = ξ^(i * (orderᴱ - 1) / n)  for i = 1..N-1
  //
  // kWideLimbNums must be large enough to hold order^Rows without overflow.
  template <size_t Rows, size_t kWideLimbNums, size_t kLimbNums>
  static std::array<std::array<BaseField, N - 1>, Rows> ComputeCoeffs(
      const BigInt<kLimbNums>& field_order) {
    BaseField nr = Config::kNonResidue;

    std::array<std::array<BaseField, N - 1>, Rows> result{};
    BigInt<kWideLimbNums> order(field_order);
    BigInt<kWideLimbNums> order_e = order;
    BigInt<kWideLimbNums> n_big(N);
    for (size_t e = 1; e <= Rows; ++e) {
      auto exp = (order_e - 1) / n_big;
      BaseField nr_exp = zk_dtypes::Pow(nr, exp);

      result[e - 1][0] = nr_exp;
      for (size_t i = 1; i < N - 1; ++i) {
        result[e - 1][i] = result[e - 1][i - 1] * nr_exp;
      }
      order_e = order_e * order;
    }
    return result;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_COEFFS_H_
