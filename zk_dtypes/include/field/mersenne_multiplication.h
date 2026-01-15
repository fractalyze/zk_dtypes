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

#ifndef ZK_DTYPES_INCLUDE_FIELD_MERSENNE_MULTIPLICATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_MERSENNE_MULTIPLICATION_H_

#include <cstddef>

#include "zk_dtypes/include/arithmetics.h"
#include "zk_dtypes/include/bits.h"
#include "zk_dtypes/include/field/modular_operations.h"

namespace zk_dtypes {

template <typename T>
constexpr void MersenneMul(T a, T b, T& c, T modulus,
                           bool has_modulus_spare_bit) {
  using ExtT = internal::make_promoted_t<T>;

  auto x = ExtT{a} * b;

  size_t k = Log2Floor<T>(modulus + 1);

  // x = hi * 2ᵏ + lo
  //   = hi * (p + 1) + lo
  //   = hi + lo (mod p)
  T hi = static_cast<T>(x >> k);
  T lo = static_cast<T>(x & modulus);

  ModAdd<T>(hi, lo, c, modulus, has_modulus_spare_bit);
}

template <typename Config, typename T>
constexpr void MersenneMul(T a, T b, T& c) {
  MersenneMul(a, b, c, Config::kModulus, HasModulusSpareBit<Config>());
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_MERSENNE_MULTIPLICATION_H_
