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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_FR_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_FR_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::curve25519 {

// Ed25519 scalar field (group order):
// ell = 2²⁵² + 27742317777372353535851937790883648493
struct FrBaseConfig {
  constexpr static size_t kStorageBits = 256;
  constexpr static size_t kModulusBits = 253;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(6346243789798364141),
      UINT64_C(1503914060200516822),
      UINT64_C(0),
      UINT64_C(1152921504606846976),
  };

  // ell - 1 = 2² * t
  constexpr static uint32_t kTwoAdicity = 2;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(10809932984304366843),
      UINT64_C(375978515050129205),
      UINT64_C(0),
      UINT64_C(288230376151711744),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FrConfig;

  constexpr static BigInt<4> kOne = 1;

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(13729071593655502804),
      UINT64_C(1076455226544653310),
      UINT64_C(9024489490286232186),
      UINT64_C(669474010940670439),
  };
};

struct FrMontConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FrConfig;

  // R² mod ell
  constexpr static BigInt<4> kRSquared = {
      UINT64_C(11819153939886771969),
      UINT64_C(14991950615390032711),
      UINT64_C(14910419812499177061),
      UINT64_C(259310039853996605),
  };
  // -ell⁻¹ mod 2²⁶⁴
  constexpr static uint64_t kNPrime = UINT64_C(15183074304973897243);

  // R mod ell
  constexpr static BigInt<4> kOne = {
      UINT64_C(15486807595281847581),
      UINT64_C(14334777244411350896),
      UINT64_C(18446744073709551614),
      UINT64_C(1152921504606846975),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(8969215743819189885),
      UINT64_C(5516037659391044808),
      UINT64_C(15508184678381615533),
      UINT64_C(385507852950656554),
  };
};

using Fr = PrimeField<FrConfig>;
using FrMont = PrimeField<FrMontConfig>;

}  // namespace zk_dtypes::curve25519

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_FR_H_
