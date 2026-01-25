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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_MULTIPLICATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_MULTIPLICATION_H_

#include <cstddef>

#include "zk_dtypes/include/arithmetics.h"

namespace zk_dtypes {

// Returns true if modulus has the form p = 2ⁿ - 2ⁿ⸍² + 1.
//
// Examples:
//   - 64-bit: p = 2⁶⁴ - 2³² + 1 = 18446744069414584321 (Goldilocks)
//   - 32-bit: p = 2³² - 2¹⁶ + 1 = 4294901761
//
// Detection: p + ε = 2ⁿ wraps to 0, where ε = 2ⁿ⸍² - 1.
template <typename T>
constexpr bool IsGoldilocksModulus(T modulus) {
  constexpr size_t kHalfBits = sizeof(T) * 4;
  constexpr T kEpsilon = (T{1} << kHalfBits) - 1;
  return modulus + kEpsilon == 0;
}

// Multiplication for primes of the form p = 2ⁿ - 2ⁿ⸍² + 1.
//
// Key property: 2ⁿ ≡ ε (mod p), where ε = 2ⁿ⸍² - 1.
//
// Reduction of a 2n-bit product:
//
//   a × b = hi·2ⁿ + lo
//         ≡ hi·ε + lo (mod p)
//
// Since hi can be up to n bits, we split it: hi = hi_hi·2ⁿ⸍² + hi_lo
//
//   hi·ε = (hi_hi·2ⁿ⸍² + hi_lo)·ε
//        = hi_hi·2ⁿ⸍²·ε + hi_lo·ε
//        = hi_hi·(2ⁿ - 2ⁿ⸍²) + hi_lo·ε      [since 2ⁿ⸍²·ε = 2ⁿ - 2ⁿ⸍²]
//        ≡ hi_hi·(ε - 2ⁿ⸍²) + hi_lo·ε       [since 2ⁿ ≡ ε]
//        = -hi_hi + hi_lo·ε                 [since ε - 2ⁿ⸍² = -1]
//
// Final formula: a × b ≡ lo - hi_hi + hi_lo·ε (mod p)
//
template <typename T>
constexpr void GoldilocksMul(T a, T b, T& c, T modulus) {
  using ExtT = internal::make_promoted_t<T>;

  constexpr size_t kHalfBits = sizeof(T) * 4;
  constexpr size_t kBits = sizeof(T) * 8;
  constexpr T kEpsilon = (T{1} << kHalfBits) - 1;

  ExtT x = ExtT{a} * b;
  T lo = static_cast<T>(x);
  T hi = static_cast<T>(x >> kBits);

  T hi_hi = hi >> kHalfBits;
  T hi_lo = hi & kEpsilon;

  // Step 1: t0 = lo - hi_hi
  // On borrow: wrapped = (lo - hi_hi) + 2ⁿ, but we want (lo - hi_hi) + p.
  // Since p + ε = 2ⁿ, subtract ε: wrapped - ε = (lo - hi_hi) + p.
  T t0 = lo - hi_hi;
  if (lo < hi_hi) {
    t0 -= kEpsilon;
  }

  // Step 2: t1 = hi_lo × ε (fits in n bits)
  T t1 = hi_lo * kEpsilon;

  // Step 3: t2 = t0 + t1 with reduction
  // On overflow: wrapped = (t0 + t1) - 2ⁿ, but we want (t0 + t1) - p.
  // Since 2ⁿ - ε = p, add ε: wrapped + ε = (t0 + t1) - p.
  // Note: t0 + t1 < 2p, so (t0 + t1) - p < p. No further reduction needed.
  T t2 = t0 + t1;
  if (t2 < t0) {
    t2 += kEpsilon;
  } else if (t2 >= modulus) {
    t2 -= modulus;
  }

  c = t2;
}

template <typename Config, typename T>
constexpr void GoldilocksMul(T a, T b, T& c) {
  GoldilocksMul(a, b, c, Config::kModulus);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_MULTIPLICATION_H_
