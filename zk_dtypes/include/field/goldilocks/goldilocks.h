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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct GoldilocksBaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446744069414584321);

  constexpr static uint32_t kTwoAdicity = 32;
  constexpr static uint32_t kSmallSubgroupBase = 3;
  constexpr static uint32_t kSmallSubgroupAdicity = 1;

  constexpr static uint32_t kTrace = 4294967295;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = true;
};

struct GoldilocksStdConfig : public GoldilocksBaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = GoldilocksStdConfig;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(1753635133440165772);

  constexpr static uint64_t kLargeSubgroupRootOfUnity =
      UINT64_C(14159254819154955796);
};

struct GoldilocksConfig : public GoldilocksBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = GoldilocksStdConfig;

  constexpr static uint64_t kRSquared = UINT64_C(18446744065119617025);
  constexpr static uint64_t kNPrime = 4294967297;

  constexpr static uint64_t kOne = 4294967295;

  constexpr static uint64_t kTwoAdicRootOfUnity =
      UINT64_C(15733474329512464024);

  constexpr static uint64_t kLargeSubgroupRootOfUnity =
      UINT64_C(3744758565052099247);
};

using Goldilocks = PrimeField<GoldilocksConfig>;
using GoldilocksStd = PrimeField<GoldilocksStdConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS_H_
