#ifndef ZK_DTYPES_INCLUDE_FIELD_BABYBEAR_BABYBEAR_H_
#define ZK_DTYPES_INCLUDE_FIELD_BABYBEAR_BABYBEAR_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct BabybearBaseConfig {
  constexpr static size_t kModulusBits = 31;
  constexpr static uint32_t kModulus = 2013265921;

  constexpr static uint32_t kRSquared = 1172168163;
  constexpr static uint32_t kNPrime = 2281701377;

  constexpr static uint32_t kTwoAdicity = 27;

  constexpr static uint32_t kTrace = 15;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct BabybearStdConfig : public BabybearBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = BabybearStdConfig;

  constexpr static uint32_t kOne = 1;

  constexpr static uint32_t kTwoAdicRootOfUnity = 440564289;
};

struct BabybearConfig : public BabybearBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = BabybearStdConfig;

  constexpr static uint32_t kOne = 268435454;

  constexpr static uint32_t kTwoAdicRootOfUnity = 1476048622;
};

using Babybear = PrimeField<BabybearConfig>;
using BabybearStd = PrimeField<BabybearStdConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BABYBEAR_BABYBEAR_H_
