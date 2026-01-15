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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BARRETT_MULTIPLICATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_BARRETT_MULTIPLICATION_H_

#include <type_traits>

#include "zk_dtypes/include/arithmetics.h"

namespace zk_dtypes {

// See
// https://fractalyze.gitbook.io/intro/primitives/modular-arithmetic/modular-reduction/barrett-reduction
template <typename T, typename ExtT = internal::make_promoted_t<T>,
          std::enable_if_t<std::is_integral_v<T> && sizeof(T) <= 4>* = nullptr>
constexpr void BarrettMul(T a, T b, T& c, T n, ExtT m) {
  using ExtExtT = internal::make_promoted_t<ExtT>;

  auto x = ExtT{a} * b;

  ExtT q = (ExtExtT{x} * m) >> (sizeof(ExtT) * 8);
  auto r = static_cast<T>(x - q * n);
  if (r >= n) {
    r -= n;
  }
  c = r;
}

template <typename Config, typename T>
constexpr void BarrettMul(T a, T b, T& c) {
  BarrettMul(a, b, c, Config::kModulus, Config::kMu);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BARRETT_MULTIPLICATION_H_
