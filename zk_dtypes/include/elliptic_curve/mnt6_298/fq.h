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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FQ_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FQ_H_

#include "zk_dtypes/include/field/big_prime_field.h"

namespace zk_dtypes::mnt6_298 {

// MNT6-298 base field. p =
// 475922286169261325753349249653048451545124878552823515553267735739164647307408490559963137
// This is the MNT4/MNT6 cycle: MNT6's Fq equals MNT4's Fr (ark-mnt6-298 defines
// it as `ark_mnt4_298::Fr`). Generator 10, two-adicity 34, no small subgroup.
struct FqBaseConfig {
  constexpr static size_t kStorageBits = 320;
  constexpr static size_t kModulusBits = 298;
  constexpr static BigInt<5> kModulus = {
      UINT64_C(13493686787511418881), UINT64_C(18107087372223867603),
      UINT64_C(14622846468717035924), UINT64_C(11702080941310629006),
      UINT64_C(4110145082483),
  };

  constexpr static uint32_t kTwoAdicity = 34;

  constexpr static BigInt<5> kTrace = {
      UINT64_C(16471733895278153001),
      UINT64_C(15509570718765568769),
      UINT64_C(7632498711552832088),
      UINT64_C(4462844154025183527),
      UINT64_C(239),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct FqConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = FqConfig;

  constexpr static BigInt<5> kOne = 1;

  constexpr static BigInt<5> kTwoAdicRootOfUnity = {
      UINT64_C(3610488132116382585),  UINT64_C(326696323574833339),
      UINT64_C(16343429254519383513), UINT64_C(10231842559834597716),
      UINT64_C(1041857165040),
  };
};

struct FqMontConfig : public FqBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = FqConfig;

  constexpr static BigInt<5> kRSquared = {
      UINT64_C(5069492133365307755), UINT64_C(238568745964307313),
      UINT64_C(5457639059775718278), UINT64_C(215892256015403885),
      UINT64_C(1416186078018),
  };
  constexpr static uint64_t kNPrime = UINT64_C(13493686787511418879);

  constexpr static BigInt<5> kOne = {
      UINT64_C(14057839933066544220), UINT64_C(11205174688489019272),
      UINT64_C(14258270859779156058), UINT64_C(2961187778261111191),
      UINT64_C(1929014752195),
  };

  constexpr static BigInt<5> kTwoAdicRootOfUnity = {
      UINT64_C(9334614154892245988), UINT64_C(3090160994209839447),
      UINT64_C(6306351189133647754), UINT64_C(17995603186291776438),
      UINT64_C(455881062568),
  };
};

using Fq = PrimeField<FqConfig>;
using FqMont = PrimeField<FqMontConfig>;

}  // namespace zk_dtypes::mnt6_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FQ_H_
