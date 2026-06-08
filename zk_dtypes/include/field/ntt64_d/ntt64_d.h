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

#ifndef ZK_DTYPES_INCLUDE_FIELD_NTT64_D_NTT64_D_H_
#define ZK_DTYPES_INCLUDE_FIELD_NTT64_D_NTT64_D_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Ntt64DBaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446743824601448449);

  constexpr static uint32_t kTwoAdicity = 33;

  constexpr static uint64_t kTrace = 2147483619;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Ntt64DConfig : public Ntt64DBaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = Ntt64DConfig;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(8196056441320966731);
};

struct Ntt64DMontConfig : public Ntt64DBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Ntt64DConfig;

  constexpr static uint64_t kRSquared = UINT64_C(837501442847453);
  constexpr static uint64_t kNPrime = UINT64_C(249108103169);

  constexpr static uint64_t kOne = UINT64_C(249108103167);

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(5007139340760443177);
};

using Ntt64D = PrimeField<Ntt64DConfig>;
using Ntt64DMont = PrimeField<Ntt64DMontConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_NTT64_D_NTT64_D_H_
