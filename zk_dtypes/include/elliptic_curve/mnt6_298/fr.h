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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::mnt6_298 {

// MNT6-298 scalar field. r =
// 475922286169261325753349249653048451545124879242694725395555128576210262817955800483758081
// The MNT4/MNT6 cycle: MNT6's Fr equals MNT4's Fq (ark-mnt6-298 defines it as
// `ark_mnt4_298::Fq`). Generator 17, two-adicity 17, small subgroup 7^2.
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 320;
  constexpr static size_t kModulusBits = 298;
  constexpr static BigInt<5> kModulus = {
      UINT64_C(14487189785281953793), UINT64_C(4731562877756902930),
      UINT64_C(14622846468719063274), UINT64_C(11702080941310629006),
      UINT64_C(4110145082483),
  };

  constexpr static uint32_t kTwoAdicity = 17;
  constexpr static uint32_t kSmallSubgroupBase = 7;
  constexpr static uint32_t kSmallSubgroupAdicity = 2;

  constexpr static BigInt<5> kTrace = {
      UINT64_C(507046961542412467),
      UINT64_C(10985722965000136848),
      UINT64_C(3046515236404309641),
      UINT64_C(7654379058973561816),
      UINT64_C(31357918),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = true;
};

struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrConfig;

  constexpr static BigInt<5> kOne = 1;

  constexpr static BigInt<5> kTwoAdicRootOfUnity = {
      UINT64_C(6851414208322399299), UINT64_C(12862850025448111381),
      UINT64_C(5069430316273791837), UINT64_C(2411171371820257181),
      UINT64_C(2286047797525),
  };

  constexpr static BigInt<5> kLargeSubgroupRootOfUnity = {
      UINT64_C(10971634350113958758), UINT64_C(11768941864355092521),
      UINT64_C(6183902089951039463),  UINT64_C(8757983921245469509),
      UINT64_C(3297388348686),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  constexpr static BigInt<5> kRSquared = {
      UINT64_C(28619103704175136),   UINT64_C(11702218449377544339),
      UINT64_C(7403203599591297249), UINT64_C(2248105543421449339),
      UINT64_C(2357678148148),
  };
  constexpr static uint64_t kNPrime = UINT64_C(12714121028002250751);

  constexpr static BigInt<5> kOne = {
      UINT64_C(1784298994435064924),  UINT64_C(16852041090100268533),
      UINT64_C(14258261760832875328), UINT64_C(2961187778261111191),
      UINT64_C(1929014752195),
  };

  constexpr static BigInt<5> kTwoAdicRootOfUnity = {
      UINT64_C(9821480371597472441), UINT64_C(9468346035609379175),
      UINT64_C(9963748368231707135), UINT64_C(14865337659602750405),
      UINT64_C(3984815592673),
  };

  constexpr static BigInt<5> kLargeSubgroupRootOfUnity = {
      UINT64_C(7711798843682337706), UINT64_C(16456007754393011187),
      UINT64_C(7470854640069402569), UINT64_C(10767969225751706229),
      UINT64_C(2250015743691),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::mnt6_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FR_H_
