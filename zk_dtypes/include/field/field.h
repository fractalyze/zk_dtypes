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

#ifndef ZK_DTYPES_INCLUDE_FIELD_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_FIELD_H_

#include <type_traits>

#include "zk_dtypes/include/group/group.h"

namespace zk_dtypes {

template <typename T, typename SFINAE = void>
struct IsFieldImpl {
  constexpr static bool value = false;
};

template <typename T>
constexpr bool IsField = IsFieldImpl<T>::value;

template <typename T>
struct IsAdditiveGroupImpl<T, std::enable_if_t<IsField<T>>> {
  constexpr static bool value = true;
};

template <typename T>
struct IsMultiplicativeGroupImpl<T, std::enable_if_t<IsField<T>>> {
  constexpr static bool value = true;
};

// Forward declaration for BinaryField
template <typename Config, typename SFINAE = void>
class BinaryField;

template <typename T>
struct IsBinaryFieldImpl {
  constexpr static bool value = false;
};

template <typename Config>
struct IsBinaryFieldImpl<BinaryField<Config>> {
  constexpr static bool value = true;
};

template <typename T>
constexpr bool IsBinaryField = IsBinaryFieldImpl<T>::value;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FIELD_H_
