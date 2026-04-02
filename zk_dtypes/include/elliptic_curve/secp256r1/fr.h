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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256R1_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256R1_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::secp256r1 {

// secp256r1 scalar field: n = group order
// n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 256;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(17562291160714782033),
      UINT64_C(13611842547513532036),
      UINT64_C(18446744073709551615),
      UINT64_C(18446744069414584320),
  };

  constexpr static uint32_t kTwoAdicity = 4;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(5709329215972061781),
      UINT64_C(18144562728322300392),
      UINT64_C(1152921504606846975),
      UINT64_C(1152921504338411520),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kOne = 1;

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(401620792848049666),
      UINT64_C(1533135717938990511),
      UINT64_C(13438876315472772604),
      UINT64_C(18431402614449441170),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(9449762124159643298),
      UINT64_C(5087230966250696614),
      UINT64_C(2901921493521525849),
      UINT64_C(7413256579398063648),
  };
  constexpr static uint64_t kNPrime = UINT64_C(3687945983376704433);

  constexpr static BigInt<4> kOne = {
      UINT64_C(884452912994769583),
      UINT64_C(4834901526196019579),
      UINT64_C(0),
      UINT64_C(4294967295),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(1158956240717909985),
      UINT64_C(3586771055249474833),
      UINT64_C(5945312850030468769),
      UINT64_C(178183135237128168),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::secp256r1

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256R1_FR_H_
