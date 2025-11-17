#ifndef ZK_DTYPES_INCLUDE_BITS_H_
#define ZK_DTYPES_INCLUDE_BITS_H_

#include <type_traits>

#include "absl/numeric/bits.h"

namespace zk_dtypes {

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Floor(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return absl::bit_width(x) - 1;
}

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Ceiling(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return x == 0 ? -1 : absl::bit_width(x - 1);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_BITS_H_
