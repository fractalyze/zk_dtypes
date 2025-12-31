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
// Returns precomputed Frobenius coefficients for all φᴱ (E = 1, ..., N - 1):
// coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) for i = 1, ..., N - 1.
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

  static const std::array<std::array<BaseField, N - 1>, N - 1>&
  GetFrobeniusCoeffs() {
    static const auto coeffs = []() {
      // Use larger BigInt to avoid overflow when computing pᵉ, where p is order
      // of base field.
      constexpr size_t kLimbNums =
          BasePrimeField::kLimbNums * N * BaseField::ExtensionDegree();
      BigInt<kLimbNums> p = BaseField::Order();
      BaseField nr = Config::kNonResidue;

      std::array<std::array<BaseField, N - 1>, N - 1> result{};
      // p_e = pᵉ, computed iteratively
      BigInt<kLimbNums> p_e = p;
      BigInt<kLimbNums> n_big(N);
      for (size_t e = 1; e < N; ++e) {
        // qₑ = (pᵉ - 1) / n
        auto q_e = ((p_e - 1) / n_big).value();
        BaseField nr_q_e = zk_dtypes::Pow(nr, q_e);

        result[e - 1][0] = nr_q_e;
        for (size_t i = 1; i < N - 1; ++i) {
          result[e - 1][i] = result[e - 1][i - 1] * nr_q_e;
        }
        p_e = p_e * p;
      }
      return result;
    }();
    return coeffs;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FROBENIUS_COEFFS_H_
