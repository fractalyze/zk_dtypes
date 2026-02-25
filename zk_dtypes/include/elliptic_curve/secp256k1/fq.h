/* Copyright 2026 The zk_dtypes Authors.

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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_FQ_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_FQ_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::secp256k1 {

// secp256k1 base field: p = 2²⁵⁶ - 2³² - 977
struct FqBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 256;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(18446744069414583343),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
  };

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(18446744071562067479),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(9223372036854775807),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FqConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FqConfig;

  constexpr static BigInt<4> kOne = 1;

  // p - 1 (the primitive 2nd root of unity, i.e. -1 mod p)
  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(18446744069414583342),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
  };
};

struct FqMontConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FqConfig;

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(8392367050913),
      UINT64_C(1),
      UINT64_C(0),
      UINT64_C(0),
  };
  constexpr static uint64_t kNPrime = UINT64_C(15580212934572586289);

  constexpr static BigInt<4> kOne = {
      UINT64_C(4294968273),
      UINT64_C(0),
      UINT64_C(0),
      UINT64_C(0),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(18446744065119615070),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
  };
};

using Fq = PrimeField<FqConfig>;
using FqMont = PrimeField<FqMontConfig>;

}  // namespace zk_dtypes::secp256k1

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_FQ_H_
