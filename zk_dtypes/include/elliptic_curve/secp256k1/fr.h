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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::secp256k1 {

// secp256k1 scalar field: n = group order
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 256;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(13822214165235122497),
      UINT64_C(13451932020343611451),
      UINT64_C(18446744073709551614),
      UINT64_C(18446744073709551615),
  };

  constexpr static uint32_t kTwoAdicity = 6;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(17221564289282791685),
      UINT64_C(18080469759223997056),
      UINT64_C(18446744073709551615),
      UINT64_C(288230376151711743),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kOne = 1;

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(1564147912238879282),
      UINT64_C(13361787288441606650),
      UINT64_C(12095684596446204107),
      UINT64_C(945631314426253740),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(9902555850136342848),
      UINT64_C(8364476168144746616),
      UINT64_C(16616019711348246470),
      UINT64_C(11342065889886772165),
  };
  constexpr static uint64_t kNPrime = UINT64_C(5408259542528602431);

  constexpr static BigInt<4> kOne = {
      UINT64_C(4624529908474429119),
      UINT64_C(4994812053365940164),
      UINT64_C(1),
      UINT64_C(0),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(16727483617216526287),
      UINT64_C(14607548025256143850),
      UINT64_C(15265302390528700431),
      UINT64_C(15433920720005950142),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::secp256k1

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_FR_H_
