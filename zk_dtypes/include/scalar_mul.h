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
