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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS3_GOLDILOCKS3_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS3_GOLDILOCKS3_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Goldilocks3BaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446743880436023297);

  constexpr static uint32_t kTwoAdicity = 32;

  constexpr static uint32_t kTrace = 4294967251;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Goldilocks3Config : public Goldilocks3BaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = Goldilocks3Config;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(8387321423513296549);
};

struct Goldilocks3MontConfig : public Goldilocks3BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Goldilocks3Config;

  constexpr static uint64_t kRSquared = UINT64_C(390992347789336);
  constexpr static uint64_t kNPrime = UINT64_C(193273528321);

  constexpr static uint64_t kOne = UINT64_C(193273528319);

  constexpr static uint64_t kTwoAdicRootOfUnity =
      UINT64_C(15671866519406808041);
};

using Goldilocks3 = PrimeField<Goldilocks3Config>;
using Goldilocks3Mont = PrimeField<Goldilocks3MontConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS3_GOLDILOCKS3_H_
