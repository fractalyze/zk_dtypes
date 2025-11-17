#ifndef ZK_DTYPES_INCLUDE_FIELD_KOALABEAR_KOALABEAR_H_
#define ZK_DTYPES_INCLUDE_FIELD_KOALABEAR_KOALABEAR_H_

#include <cstdint>

#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes {

struct KoalabearBaseConfig {
  constexpr static size_t kModulusBits = 31;
  constexpr static uint32_t kModulus = 2130706433;

  constexpr static uint32_t kRSquared = 402124772;
  constexpr static uint32_t kNPrime = 2164260865;

  constexpr static uint32_t kTwoAdicity = 24;

  constexpr static uint32_t kTrace = 127;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct KoalabearStdConfig : public KoalabearBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = KoalabearStdConfig;

  constexpr static uint32_t kOne = 1;

  constexpr static uint32_t kTwoAdicRootOfUnity = 1791270792;
};

struct KoalabearConfig : public KoalabearBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = KoalabearStdConfig;

  constexpr static uint32_t kOne = 33554430;

  constexpr static uint32_t kTwoAdicRootOfUnity = 331895189;
};

using Koalabear = PrimeField<KoalabearConfig>;
using KoalabearStd = PrimeField<KoalabearStdConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_KOALABEAR_KOALABEAR_H_
