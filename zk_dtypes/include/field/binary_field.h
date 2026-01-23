/* Copyright 2026 The zk_dtypes Authors.

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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_H_

#include <ostream>
#include <type_traits>

#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/field/big_binary_field.h"
#include "zk_dtypes/include/field/small_binary_field.h"

namespace zk_dtypes {

// =============================================================================
// Stream operator
// =============================================================================

template <typename Config>
std::ostream& operator<<(std::ostream& os, const BinaryField<Config>& bf) {
  return os << bf.ToString();
}

// =============================================================================
// Comparable trait specialization for BinaryField
// =============================================================================

template <typename T>
struct IsComparableImpl<T, std::enable_if_t<IsBinaryField<T>>> {
  constexpr static bool value = true;
};

// =============================================================================
// Type aliases for convenience
// =============================================================================

using BinaryFieldT0 = BinaryField<BinaryFieldConfig<0>>;  // GF(2)
using BinaryFieldT1 = BinaryField<BinaryFieldConfig<1>>;  // GF(2²)
using BinaryFieldT2 = BinaryField<BinaryFieldConfig<2>>;  // GF(2⁴)
using BinaryFieldT3 = BinaryField<BinaryFieldConfig<3>>;  // GF(2⁸)
using BinaryFieldT4 = BinaryField<BinaryFieldConfig<4>>;  // GF(2¹⁶)
using BinaryFieldT5 = BinaryField<BinaryFieldConfig<5>>;  // GF(2³²)
using BinaryFieldT6 = BinaryField<BinaryFieldConfig<6>>;  // GF(2⁶⁴)
using BinaryFieldT7 = BinaryField<BinaryFieldConfig<7>>;  // GF(2¹²⁸)

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_H_
