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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::bn254 {

struct FrBaseConfig {
  constexpr static size_t kModulusBits = 254;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(4891460686036598785),
      UINT64_C(2896914383306846353),
      UINT64_C(13281191951274694749),
      UINT64_C(3486998266802970665),
  };

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(1997599621687373223),
      UINT64_C(6052339484930628067),
      UINT64_C(10108755138030829701),
      UINT64_C(150537098327114917),
  };
  constexpr static uint64_t kNPrime = UINT64_C(14042775128853446655);

  constexpr static uint32_t kTwoAdicity = 28;
  constexpr static uint32_t kSmallSubgroupBase = 3;
  constexpr static uint32_t kSmallSubgroupAdicity = 2;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(11211439779908376895),
      UINT64_C(1735440370612733063),
      UINT64_C(1376415503089949544),
      UINT64_C(12990080814),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = true;
};

struct FrStdConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrStdConfig;

  constexpr static BigInt<4> kOne = 1;

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(11229192882073836016),
      UINT64_C(4624371214017703636),
      UINT64_C(63235024940837564),
      UINT64_C(3043318377369730693),
  };

  constexpr static BigInt<4> kLargeSubgroupRootOfUnity = {
      UINT64_C(10639863269868064110),
      UINT64_C(6020083959115413713),
      UINT64_C(15196548748307230377),
      UINT64_C(1274670453483637722),
  };
};
struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrStdConfig;

  constexpr static BigInt<4> kOne = {
      UINT64_C(12436184717236109307),
      UINT64_C(3962172157175319849),
      UINT64_C(7381016538464732718),
      UINT64_C(1011752739694698287),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(7164790868263648668),
      UINT64_C(11685701338293206998),
      UINT64_C(6216421865291908056),
      UINT64_C(1756667274303109607),
  };

  constexpr static BigInt<4> kLargeSubgroupRootOfUnity = {
      UINT64_C(13572693633482797243),
      UINT64_C(9300120515097308470),
      UINT64_C(4644696547510559629),
      UINT64_C(1568669608320580247),
  };
};

using Fr = PrimeField<FrConfig>;
using FrStd = PrimeField<FrStdConfig>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FR_H_
