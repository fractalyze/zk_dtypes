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

#ifndef ZK_DTYPES_INCLUDE_FIELD_PRIME_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_PRIME_FIELD_H_

#include <cstdint>
#include <ostream>

#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/field/finite_field_traits.h"

namespace zk_dtypes {

template <typename Config>
using UnderlyingType = std::conditional_t<
    Config::kStorageBits <= 32,
    std::conditional_t<
        Config::kStorageBits <= 16,
        std::conditional_t<Config::kStorageBits <= 8, uint8_t, uint16_t>,
        uint32_t>,
    uint64_t>;

template <typename T, typename = void>
struct HasSpecialMulImpl : std::false_type {};

template <typename T>
struct HasSpecialMulImpl<T, std::void_t<decltype(std::declval<T>().SpecialMul(
                                std::declval<UnderlyingType<T>>(),
                                std::declval<UnderlyingType<T>>()))>>
    : std::true_type {};

template <typename T>
constexpr bool HasSpecialMul = HasSpecialMulImpl<T>::value;

template <typename T>
struct IsPrimeFieldImpl {
  constexpr static bool value = false;
};

template <typename Config>
struct IsPrimeFieldImpl<PrimeField<Config>> {
  constexpr static bool value = true;
};

template <typename T>
constexpr bool IsPrimeField = IsPrimeFieldImpl<T>::value;

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeField<Config>& pf) {
  return os << pf.ToString();
}

template <typename T>
struct IsComparableImpl<T, std::enable_if_t<IsPrimeField<T>>> {
  constexpr static bool value = true;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_PRIME_FIELD_H_
