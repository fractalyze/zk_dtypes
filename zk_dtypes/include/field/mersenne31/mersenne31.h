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

#ifndef ZK_DTYPES_INCLUDE_FIELD_MERSENNE31_MERSENNE31_H_
#define ZK_DTYPES_INCLUDE_FIELD_MERSENNE31_MERSENNE31_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct Mersenne31BaseConfig {
  constexpr static size_t kStorageBits = 32;
  constexpr static size_t kModulusBits = 31;
  constexpr static uint32_t kModulus = 2147483647;

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static uint32_t kTrace = 1073741823;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Mersenne31StdConfig : public Mersenne31BaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Mersenne31StdConfig;

  constexpr static uint32_t kOne = 1;

  constexpr static uint32_t kTwoAdicRootOfUnity = 2147483646;

  constexpr static uint32_t SpecialMul(uint32_t a, uint32_t b) {
    uint64_t x = uint64_t{a} * b;

    // x = hi * 2³¹ + lo
    //   = hi * (p + 1) + lo
    //   = hi + lo (mod p)
    uint32_t hi = static_cast<uint32_t>(x >> 31);
    uint32_t lo = static_cast<uint32_t>(x & kModulus);

    uint32_t res = lo + hi;
    if (res >= kModulus) res -= kModulus;
    return res;
  }
};

struct Mersenne31Config : public Mersenne31BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Mersenne31StdConfig;

  constexpr static uint32_t kRSquared = 4;
  constexpr static uint32_t kNPrime = 2147483647;

  constexpr static uint32_t kOne = 2;

  constexpr static uint32_t kTwoAdicRootOfUnity = 2147483645;
};

using Mersenne31 = PrimeField<Mersenne31Config>;
using Mersenne31Std = PrimeField<Mersenne31StdConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_MERSENNE31_MERSENNE31_H_
