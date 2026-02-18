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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::bls12_381 {

// bls12-381 scalar field: 255-bit order (BigInt<4>)
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 255;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(18446744069414584321),
      UINT64_C(6034159408538082302),
      UINT64_C(3691218898639771653),
      UINT64_C(8353516859464449352),
  };

  constexpr static uint32_t kTwoAdicity = 32;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(18446282274530918399),
      UINT64_C(694073334983140354),
      UINT64_C(2998690675949164552),
      UINT64_C(1944954707),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kOne = 1;

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(1979489610183552799),
      UINT64_C(14123939284078813332),
      UINT64_C(1140938789882484416),
      UINT64_C(149418812792466287),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(14526898881837571181),
      UINT64_C(3129137299524312099),
      UINT64_C(419701826671360399),
      UINT64_C(524908885293268753),
  };
  constexpr static uint64_t kNPrime = UINT64_C(18446744069414584319);

  constexpr static BigInt<4> kOne = {
      UINT64_C(8589934590),
      UINT64_C(6378425256633387010),
      UINT64_C(11064306276430008309),
      UINT64_C(1739710354780652911),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(11289237133041595516),
      UINT64_C(2081200955273736677),
      UINT64_C(967625415375836421),
      UINT64_C(4543825880697944938),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::bls12_381

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_FR_H_
