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

#ifndef ZK_DTYPES_INCLUDE_FIELD_ROOT_OF_UNITY_H_
#define ZK_DTYPES_INCLUDE_FIELD_ROOT_OF_UNITY_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/bits.h"

namespace zk_dtypes {
namespace internal {

struct PrimeFieldFactors {
  uint32_t q_adicity;
  uint64_t q_part;
  uint32_t two_adicity;
  uint64_t two_part;

  constexpr PrimeFieldFactors(uint32_t q_adicity, uint64_t q_part,
                              uint32_t two_adicity, uint64_t two_part)
      : q_adicity(q_adicity),
        q_part(q_part),
        two_adicity(two_adicity),
        two_part(two_part) {}
};

// The integer s such that |n| = |k|ˢ * t for some odd integer t.
constexpr uint32_t ComputeAdicity(uint32_t k, uint64_t n) {
  uint32_t adicity = 0;
  while (n > 1) {
    if (n % k == 0) {
      ++adicity;
      n /= k;
    } else {
      break;
    }
  }
  return adicity;
}

template <typename F>
absl::StatusOr<PrimeFieldFactors> Decompose(uint64_t n) {
  static_assert(F::Config::kHasLargeSubgroupRootOfUnity);

  // Compute the size of our evaluation domain
  uint32_t q = F::Config::kSmallSubgroupBase;
  uint32_t q_adicity = ComputeAdicity(q, n);
  uint64_t q_part = static_cast<uint64_t>(std::pow(q, q_adicity));

  uint32_t two_adicity = ComputeAdicity(2, n);
  uint64_t two_part = static_cast<uint64_t>(std::pow(2, two_adicity));
  if (n != q_part * two_part) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Failed to decompose n = $0 into q-part ($1) * two-part ($2)", n,
        q_part, two_part));
  }

  return PrimeFieldFactors(q_adicity, q_part, two_adicity, two_part);
}

}  // namespace internal

template <typename F>
absl::StatusOr<F> GetRootOfUnity(uint64_t n) {
  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    absl::StatusOr<internal::PrimeFieldFactors> factors =
        internal::Decompose<F>(n);
    if (!factors.ok()) return factors.status();
    if (factors->two_adicity > F::Config::kTwoAdicity ||
        factors->q_adicity > F::Config::kSmallSubgroupAdicity) {
      return absl::InvalidArgumentError(absl::Substitute(
          "GetRootOfUnity error: subgroup factorization exceeds supported "
          "adicity "
          "(two_adicity = $0 max = $1; q_adicity = $2, max = $3)",
          factors->two_adicity, F::Config::kTwoAdicity, factors->q_adicity,
          F::Config::kSmallSubgroupAdicity));
    }

    auto omega = F::FromUnchecked(F::Config::kLargeSubgroupRootOfUnity);
    for (size_t i = factors->q_adicity; i < F::Config::kSmallSubgroupAdicity;
         ++i) {
      omega = omega.Pow(F::Config::kSmallSubgroupBase);
    }

    for (size_t i = factors->two_adicity; i < F::Config::kTwoAdicity; ++i) {
      omega = omega.Square();
    }
    return omega;
  } else if constexpr (F::Config::kHasTwoAdicRootOfUnity) {
    uint32_t log_size_of_group = Log2Ceiling(n);
    uint64_t size = uint64_t{1} << log_size_of_group;

    if (n != size || log_size_of_group > F::Config::kTwoAdicity) {
      return absl::InvalidArgumentError(
          absl::Substitute("GetRootOfUnity error: n ($0) is not a power of "
                           "two or exceeds max adicity ($1)",
                           n, F::Config::kTwoAdicity));
    }

    auto omega = F::FromUnchecked(F::Config::kTwoAdicRootOfUnity);
    for (uint32_t i = log_size_of_group; i < F::Config::kTwoAdicity; ++i) {
      omega = omega.Square();
    }
    return omega;
  } else {
    return absl::NotFoundError("No RootOfUnity");
  }
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_ROOT_OF_UNITY_H_
