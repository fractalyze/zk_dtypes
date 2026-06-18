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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_VESTA_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_VESTA_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::vesta {

// Vesta scalar field (= Pallas base field):
// p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// A 255-bit prime with 2-adicity 32. Constants derived (never hand-typed) from
// arkworks ark-vesta (multiplicative generator 5); see docs/development.md
// for the derivation method.
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 255;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(11037532056220336129),
      UINT64_C(2469829653914515739),
      UINT64_C(0),
      UINT64_C(4611686018427387904),
  };

  constexpr static uint32_t kTwoAdicity = 32;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(670184341500670189),
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
      UINT64_C(13667703228001592111),
      UINT64_C(16875599075175265668),
      UINT64_C(3900434499382671386),
      UINT64_C(3156588888553745370),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(10122100416058490895),
      UINT64_C(15551789045973377255),
      UINT64_C(8617542898466512152),
      UINT64_C(679271340751763220),
  };
  constexpr static uint64_t kNPrime = UINT64_C(11037532056220336127);

  constexpr static BigInt<4> kOne = {
      UINT64_C(3780891978758094845),
      UINT64_C(11037255111966004397),
      UINT64_C(18446744073709551615),
      UINT64_C(4611686018427387903),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(11713220832667294704),
      UINT64_C(10413392179731184095),
      UINT64_C(18133385229535560846),
      UINT64_C(4524191781424318170),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::vesta

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_VESTA_FR_H_
