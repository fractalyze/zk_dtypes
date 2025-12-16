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

#ifndef ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_H_

#include <type_traits>

#include "zk_dtypes/include/always_false.h"
#include "zk_dtypes/include/field/field.h"
#include "zk_dtypes/include/field/finite_field_traits.h"
#include "zk_dtypes/include/field/square_root_algorithms/shanks.h"
#include "zk_dtypes/include/field/square_root_algorithms/square_root_algorithm9.h"
#include "zk_dtypes/include/field/square_root_algorithms/tonelli_shanks.h"

namespace zk_dtypes {

// FiniteField represents a field with a finite number of elements,
// also known as a Galois field. The order of a finite field is always
// a prime or a power of a prime (see Birkhoff and Mac Lane, 1996).
//
// For each prime power, there exists exactly one finite field
// (up to isomorphism), typically denoted GF(pⁿ) or F(pⁿ).
// See: https://mathworld.wolfram.com/FiniteField.html
template <typename F>
class FiniteField {
 public:
  constexpr absl::StatusOr<F> SquareRoot() const {
    using BasePrimeField = typename FiniteFieldTraits<F>::BasePrimeField;

    constexpr uint64_t p =
        static_cast<uint64_t>(BasePrimeField::Config::kModulus);
    if constexpr (F::ExtensionDegree() % 2 == 1) {
      if constexpr (p % 4 == 3) {
        return ComputeShanksSquareRoot(*static_cast<const F*>(this));
      } else {
        static_assert(BasePrimeField::Config::kHasTwoAdicRootOfUnity);
        return ComputeTonelliShanksSquareRoot(*static_cast<const F*>(this));
      }
    } else if constexpr (F::ExtensionDegree() == 2) {
      if (IsModulusToHalfDegreeThreeModFour()) {
        return ComputeAlgorithm9SquareRoot(*static_cast<const F*>(this));
      } else {
        return absl::UnimplementedError(
            "Not implemented for extension degree 2 "
            "and modulus not ≡ 3 (mod 4)");
      }
    } else {
      static_assert(AlwaysFalse<F>,
                    "Not implemented for extension degree > 2 and even");
    }
  }

 private:
  constexpr static bool IsModulusToHalfDegreeThreeModFour() {
    using BasePrimeField = typename FiniteFieldTraits<F>::BasePrimeField;
    constexpr uint64_t p =
        static_cast<uint64_t>(BasePrimeField::Config::kModulus);
    constexpr size_t exponent = F::ExtensionDegree() / 2;
    return ConstexprPow(p, exponent) % 4 == 3;
  }

  constexpr static uint64_t ConstexprPow(uint64_t base, size_t exp) {
    return exp == 0 ? 1 : base * ConstexprPow(base, exp - 1);
  }
};

template <typename T>
struct IsFieldImpl<T, std::enable_if_t<std::is_base_of_v<FiniteField<T>, T>>> {
  constexpr static bool value = true;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_H_
