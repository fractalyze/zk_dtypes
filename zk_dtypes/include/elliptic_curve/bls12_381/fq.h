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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_FQ_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_FQ_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::bls12_381 {

// bls12-381 base field: 381-bit prime (BigInt<6>)
struct FqBaseConfig {
  constexpr static size_t kStorageBits = 384;
  constexpr static size_t kModulusBits = 381;
  constexpr static BigInt<6> kModulus = {
      UINT64_C(13402431016077863595), UINT64_C(2210141511517208575),
      UINT64_C(7435674573564081700),  UINT64_C(7239337960414712511),
      UINT64_C(5412103778470702295),  UINT64_C(1873798617647539866),
  };

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static BigInt<6> kTrace = {
      UINT64_C(15924587544893707605), UINT64_C(1105070755758604287),
      UINT64_C(12941209323636816658), UINT64_C(12843041017062132063),
      UINT64_C(2706051889235351147),  UINT64_C(936899308823769933),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FqConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FqConfig;

  constexpr static BigInt<6> kOne = 1;

  // p - 1 (the primitive 2nd root of unity, i.e. -1 mod p)
  constexpr static BigInt<6> kTwoAdicRootOfUnity = {
      UINT64_C(13402431016077863594), UINT64_C(2210141511517208575),
      UINT64_C(7435674573564081700),  UINT64_C(7239337960414712511),
      UINT64_C(5412103778470702295),  UINT64_C(1873798617647539866),
  };
};

struct FqMontConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FqConfig;

  constexpr static BigInt<6> kRSquared = {
      UINT64_C(17644856173732828998), UINT64_C(754043588434789617),
      UINT64_C(10224657059481499349), UINT64_C(7488229067341005760),
      UINT64_C(11130996698012816685), UINT64_C(1267921511277847466),
  };
  constexpr static uint64_t kNPrime = UINT64_C(9940570264628428797);

  constexpr static BigInt<6> kOne = {
      UINT64_C(8505329371266088957), UINT64_C(17002214543764226050),
      UINT64_C(6865905132761471162), UINT64_C(8632934651105793861),
      UINT64_C(6631298214892334189), UINT64_C(1582556514881692819),
  };

  constexpr static BigInt<6> kTwoAdicRootOfUnity = {
      UINT64_C(4897101644811774638),  UINT64_C(3654671041462534141),
      UINT64_C(569769440802610537),   UINT64_C(17053147383018470266),
      UINT64_C(17227549637287919721), UINT64_C(291242102765847046),
  };
};

using Fq = PrimeField<FqConfig>;
using FqMont = PrimeField<FqMontConfig>;

}  // namespace zk_dtypes::bls12_381

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_FQ_H_
