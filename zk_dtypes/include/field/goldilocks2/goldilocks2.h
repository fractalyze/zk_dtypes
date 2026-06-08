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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS2_GOLDILOCKS2_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS2_GOLDILOCKS2_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Goldilocks2BaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446744056529682433);

  constexpr static uint32_t kTwoAdicity = 34;

  constexpr static uint32_t kTrace = 1073741823;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Goldilocks2Config : public Goldilocks2BaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = Goldilocks2Config;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(9045540773743215239);
};

struct Goldilocks2MontConfig : public Goldilocks2BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Goldilocks2Config;

  constexpr static uint64_t kRSquared = UINT64_C(240518168561);
  constexpr static uint64_t kNPrime = UINT64_C(17179869185);

  constexpr static uint64_t kOne = UINT64_C(17179869183);

  constexpr static uint64_t kTwoAdicRootOfUnity =
      UINT64_C(10094336246325731280);
};

using Goldilocks2 = PrimeField<Goldilocks2Config>;
using Goldilocks2Mont = PrimeField<Goldilocks2MontConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS2_GOLDILOCKS2_H_
