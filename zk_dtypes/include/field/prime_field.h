#ifndef ZK_DTYPES_INCLUDE_FIELD_PRIME_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_PRIME_FIELD_H_

#include <ostream>

#include "zk_dtypes/include/field/finite_field_traits.h"

namespace zk_dtypes {

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

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_PRIME_FIELD_H_
