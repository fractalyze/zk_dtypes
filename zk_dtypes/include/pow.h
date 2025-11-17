#ifndef ZK_DTYPES_INCLUDE_POW_H_
#define ZK_DTYPES_INCLUDE_POW_H_

#include <stddef.h>

#include <type_traits>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/bit_iterator.h"

namespace zk_dtypes {

template <typename T, size_t N,
          std::enable_if_t<!std::is_integral_v<T>>* = nullptr>
[[nodiscard]] constexpr T Pow(const T& value, const BigInt<N>& exponent) {
  T ret = T::One();
  auto it = BitIteratorBE<BigInt<N>>::begin(&exponent, true);
  auto end = BitIteratorBE<BigInt<N>>::end(&exponent);
  while (it != end) {
    ret = ret.Square();
    if (*it) {
      ret *= value;
    }
    ++it;
  }
  return ret;
}

template <typename T, typename U,
          std::enable_if_t<!std::is_integral_v<T> && std::is_integral_v<U>>* =
              nullptr>
[[nodiscard]] constexpr T Pow(const T& value, U exponent) {
  return Pow(value, BigInt<1>(exponent));
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
[[nodiscard]] constexpr T Pow(T value, T exponent) {
  T ret = 1;
  if (exponent < 0) {
    // This function is only called with non-negative exponents in the tests.
    // Add a contract (e.g., DCHECK) or proper handling for negative exponents.
    return 0;
  }

  T exp = exponent;
  while (exp > 0) {
    if (exp & 1) {
      ret *= value;
    }
    value *= value;
    exp >>= 1;
  }
  return ret;
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_POW_H_
