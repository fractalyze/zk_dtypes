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

#ifndef ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_SQUARE_ROOT_ALGORITHM9_H_
#define ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_SQUARE_ROOT_ALGORITHM9_H_

#include "absl/status/statusor.h"

#include "zk_dtypes/include/field/finite_field_traits.h"
#include "zk_dtypes/include/field/frobenius.h"

namespace zk_dtypes {

constexpr uint64_t ConstexprPow(uint64_t base, size_t exp) {
  return exp == 0 ? 1 : base * ConstexprPow(base, exp - 1);
}

template <typename F>
constexpr bool IsAlgorithm9SquareRootCompatible() {
  using BasePrimeField = typename FiniteFieldTraits<F>::BasePrimeField;
  constexpr uint64_t p =
      static_cast<uint64_t>(BasePrimeField::Config::kModulus);
  constexpr size_t exponent = F::ExtensionDegree() / 2;
  return ConstexprPow(p, exponent) % 4 == 3;
}

template <typename F>
absl::StatusOr<F> ComputeAlgorithm9SquareRoot(const F& a) {
  // F is quadratic extension field where non-quadratic non-residue i² = -1.
  //
  // Finds x such that x² = a.
  // Assumes the modulus p satisfies p ≡ 3 (mod 4).
  // See: https://eprint.iacr.org/2012/685.pdf (Algorithm 9, page 17)
  // See
  // https://fractalyze.gitbook.io/intro/primitives/modular-arithmetic/modular-square-root/generalized-shanks-algorithm-for-quadratic-extension-field
  // for more details.
  using BasePrimeField = typename FiniteFieldTraits<F>::BasePrimeField;
  static_assert(IsAlgorithm9SquareRootCompatible<F>());
  static_assert(F::ExtensionDegree() % 2 == 0);
  if constexpr (F::ExtensionDegree() != 2) {
    return absl::UnimplementedError(
        "Not implemented for extension degree != 2");
  }
  constexpr auto exponent = (BasePrimeField::Config::kModulus - 3) >> 2;
  F a1 = a.Pow(exponent);
  F alpha = a1.Square() * a;
  F a0 = alpha.template Frobenius<>() * alpha;
  auto neg_one = -F::One();
  if (a0 == neg_one) {
    return absl::NotFoundError("No square root exists");
  }
  F x0 = a1 * a;
  if (alpha == neg_one) {
    return F({-x0[1], x0[0]});
  } else {
    constexpr auto exponent3 = (BasePrimeField::Config::kModulus - 1) >> 1;
    F b = (alpha + 1).Pow(exponent3);
    return b * x0;
  }
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_SQUARE_ROOT_ALGORITHM9_H_
