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

#ifndef ZK_DTYPES_INCLUDE_FIELD_NTT64_F_NTT64_F_H_
#define ZK_DTYPES_INCLUDE_FIELD_NTT64_F_NTT64_F_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Ntt64FBaseConfig {
  constexpr static size_t kStorageBits = 64;
  constexpr static size_t kModulusBits = 64;
  constexpr static uint64_t kModulus = UINT64_C(18446743738702102529);

  constexpr static uint32_t kTwoAdicity = 33;

  constexpr static uint64_t kTrace = 2147483609;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Ntt64FConfig : public Ntt64FBaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = Ntt64FConfig;

  constexpr static uint64_t kOne = 1;

  constexpr static uint64_t kTwoAdicRootOfUnity =
      UINT64_C(17340159251188219253);
};

struct Ntt64FMontConfig : public Ntt64FBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Ntt64FConfig;

  constexpr static uint64_t kRSquared = UINT64_C(2037515305347133);
  constexpr static uint64_t kNPrime = UINT64_C(335007449089);

  constexpr static uint64_t kOne = UINT64_C(335007449087);

  constexpr static uint64_t kTwoAdicRootOfUnity = UINT64_C(4491596143671672011);
};

using Ntt64F = PrimeField<Ntt64FConfig>;
using Ntt64FMont = PrimeField<Ntt64FMontConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_NTT64_F_NTT64_F_H_
