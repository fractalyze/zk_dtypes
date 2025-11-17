#ifndef ZK_DTYPES_INCLUDE_RANDOM_H_
#define ZK_DTYPES_INCLUDE_RANDOM_H_

#include "absl/random/random.h"

namespace zk_dtypes {

absl::BitGen& GetAbslBitGen();

template <typename T>
T Uniform() {
  return absl::Uniform<T>(GetAbslBitGen());
}

template <typename T>
T Uniform(T low, T high) {
  return absl::Uniform<T>(GetAbslBitGen(), low, high);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_RANDOM_H_
