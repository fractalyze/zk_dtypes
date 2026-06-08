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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS6_GOLDILOCKS6_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS6_GOLDILOCKS6_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Goldilocks6BaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446743751587004417);

  constexpr static uint32_t kTwoAdicity = 32;

  constexpr static uint32_t kTrace = 4294967221;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Goldilocks6Config : public Goldilocks6BaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = Goldilocks6Config;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(2838577753949000656);
};

struct Goldilocks6MontConfig : public Goldilocks6BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Goldilocks6Config;

  constexpr static uint64_t kRSquared = UINT64_C(1811295082899976);
  constexpr static uint64_t kNPrime = UINT64_C(322122547201);

  constexpr static uint64_t kOne = UINT64_C(322122547199);

  constexpr static uint64_t kTwoAdicRootOfUnity =
      UINT64_C(17183828563792370618);
};

using Goldilocks6 = PrimeField<Goldilocks6Config>;
using Goldilocks6Mont = PrimeField<Goldilocks6MontConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS6_GOLDILOCKS6_H_
