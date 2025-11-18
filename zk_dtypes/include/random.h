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
