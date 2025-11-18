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

#ifndef ZK_DTYPES_INCLUDE_SCALAR_MUL_H_
#define ZK_DTYPES_INCLUDE_SCALAR_MUL_H_

#include <stddef.h>

#include <type_traits>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/bit_iterator.h"

namespace zk_dtypes {

template <typename T, size_t N>
[[nodiscard]] constexpr T ScalarMul(const T& value, const BigInt<N>& scalar) {
  T ret = T::Zero();
  auto it = BitIteratorBE<BigInt<N>>::begin(&scalar, true);
  auto end = BitIteratorBE<BigInt<N>>::end(&scalar);
  while (it != end) {
    ret = ret.Double();
    if (*it) {
      ret += value;
    }
    ++it;
  }
  return ret;
}

template <typename T, typename U,
          std::enable_if_t<std::is_integral_v<U>>* = nullptr>
[[nodiscard]] constexpr T ScalarMul(const T& value, U scalar) {
  return ScalarMul(value, BigInt<1>(scalar));
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_SCALAR_MUL_H_
