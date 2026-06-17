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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PALLAS_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PALLAS_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::pallas {

// Pallas scalar field (= Vesta base field):
// q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// A 255-bit prime with 2-adicity 32. Constants derived (never hand-typed) from
// arkworks ark-pallas (multiplicative generator 5); see docs/development.md
// for the derivation method.
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 255;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(10108024940646105089),
      UINT64_C(2469829653919213789),
      UINT64_C(0),
      UINT64_C(4611686018427387904),
  };

  constexpr static uint32_t kTwoAdicity = 32;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(690362312389225249),
      UINT64_C(575052028),
      UINT64_C(0),
      UINT64_C(1073741824),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kOne = 1;

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(12037607305579515999),
      UINT64_C(11221139188353527881),
      UINT64_C(11411081306099606126),
      UINT64_C(3307517586042601304),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(18200867980676431887),
      UINT64_C(7474641938123724515),
      UINT64_C(9200329640471491984),
      UINT64_C(679271340771891881),
  };
  constexpr static uint64_t kNPrime = UINT64_C(10108024940646105087);

  constexpr static BigInt<4> kOne = {
      UINT64_C(6569413325480787965),
      UINT64_C(11037255111951910247),
      UINT64_C(18446744073709551615),
      UINT64_C(4611686018427387903),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(2414060527980987102),
      UINT64_C(14720393103524889748),
      UINT64_C(12406956448539459298),
      UINT64_C(826967475050360918),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::pallas

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PALLAS_FR_H_
