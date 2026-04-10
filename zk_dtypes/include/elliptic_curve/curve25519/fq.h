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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_FQ_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_FQ_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::curve25519 {

// Curve25519 base field: p = 2²⁵⁵ - 19
struct FqBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 255;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(18446744073709551597),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(9223372036854775807),
  };

  // p - 1 = 2² * t
  constexpr static uint32_t kTwoAdicity = 2;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(18446744073709551611),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615),
      UINT64_C(2305843009213693951),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FqConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FqConfig;

  constexpr static BigInt<4> kOne = 1;

  // Primitive 4th root of unity: sqrt(-1) mod p
  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(14190309331451158704),
      UINT64_C(3405592160176694392),
      UINT64_C(3120150775007532967),
      UINT64_C(3135389899092516619),
  };
};

struct FqMontConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FqConfig;

  // R² mod p where R = 2²⁵⁶
  constexpr static BigInt<4> kRSquared = {
      UINT64_C(1444),
      UINT64_C(0),
      UINT64_C(0),
      UINT64_C(0),
  };
  // -p⁻¹ mod 2²⁶⁴
  constexpr static uint64_t kNPrime = UINT64_C(9708812670373448219);

  // R mod p (Montgomery form of 1)
  constexpr static BigInt<4> kOne = {
      UINT64_C(38),
      UINT64_C(0),
      UINT64_C(0),
      UINT64_C(0),
  };

  // sqrt(-1) mod p in Montgomery form
  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(4276176457567034116),
      UINT64_C(285293570747525613),
      UINT64_C(7885265008028943057),
      UINT64_C(8464351723258321832),
  };
};

using Fq = PrimeField<FqConfig>;
using FqMont = PrimeField<FqMontConfig>;

}  // namespace zk_dtypes::curve25519

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_FQ_H_
