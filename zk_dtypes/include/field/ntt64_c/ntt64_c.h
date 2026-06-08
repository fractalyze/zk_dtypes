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

#ifndef ZK_DTYPES_INCLUDE_FIELD_NTT64_C_NTT64_C_H_
#define ZK_DTYPES_INCLUDE_FIELD_NTT64_C_NTT64_C_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Ntt64CBaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446743841781317633);

  constexpr static uint32_t kTwoAdicity = 33;

  constexpr static uint64_t kTrace = 2147483621;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Ntt64CConfig : public Ntt64CBaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = Ntt64CConfig;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(4222427707487412532);
};

struct Ntt64CMontConfig : public Ntt64CBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Ntt64CConfig;

  constexpr static uint64_t kRSquared = UINT64_C(675838873826461);
  constexpr static uint64_t kNPrime = UINT64_C(231928233985);

  constexpr static uint64_t kOne = UINT64_C(231928233983);

  constexpr static uint64_t kTwoAdicRootOfUnity =
      UINT64_C(13184444944827807808);
};

using Ntt64C = PrimeField<Ntt64CConfig>;
using Ntt64CMont = PrimeField<Ntt64CMontConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_NTT64_C_NTT64_C_H_
