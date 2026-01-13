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

#ifndef ZK_DTYPES_INCLUDE_FIELD_MONT_MULTIPLICATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_MONT_MULTIPLICATION_H_

#include <cstddef>
#include <limits>
#include <type_traits>

#include "zk_dtypes/include/arithmetics.h"
#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/modular_operations.h"

namespace zk_dtypes {

// Can we use the no-carry optimization for multiplication
// outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
//
// This optimization applies if
// (a) `modulus[biggest_limb_idx] < max(uint64_t) >> 1`, and
// (b) the bits of the modulus are not all 1.
template <size_t N>
constexpr bool CanUseNoCarryMulOptimization(const BigInt<N>& modulus) {
  uint64_t biggest_limb = modulus[N - 1];
  bool top_bit_is_zero = biggest_limb >> 63 == 0;
  bool all_remaining_bits_are_one =
      biggest_limb == std::numeric_limits<uint64_t>::max() >> 1;
  for (size_t i = 0; i < N - 1; ++i) {
    all_remaining_bits_are_one &=
        modulus[i] == std::numeric_limits<uint64_t>::max();
  }
  return top_bit_is_zero && !all_remaining_bits_are_one;
}

template <typename T,
          std::enable_if_t<std::is_integral_v<T> && sizeof(T) <= 64>* = nullptr>
constexpr void MontReduce(T a, T& b, T modulus, T n_prime) {
  using SignedT = std::make_signed_t<T>;
  using SignedExtT = internal::make_promoted_t<SignedT>;

  T m = a * n_prime;
  auto mn = (SignedExtT{m} * modulus) >> (sizeof(T) * 8);
  if (mn == 0) {
    b = 0;
  } else {
    b = modulus - static_cast<T>(mn);
  }
}

template <typename Config, typename T>
constexpr void MontReduce(T a, T& b) {
  MontReduce(a, b, Config::kModulus, Config::kNPrime);
}

template <size_t N>
constexpr void MontReduce(const BigInt<N>& a, BigInt<N>& b,
                          const BigInt<N>& modulus, uint64_t n_prime) {
  b = a;
  for (size_t i = 0; i < N; ++i) {
    uint64_t k = b[i] * n_prime;
    internal::MulResult<uint64_t> result =
        internal::MulAddWithCarry(b[i], k, modulus[0], 0);
    for (size_t j = 1; j < N; ++j) {
      result =
          internal::MulAddWithCarry(b[(j + i) % N], k, modulus[j], result.hi);
      b[(j + i) % N] = result.lo;
    }
    b[i] = result.hi;
  }
}

template <typename Config, size_t N>
constexpr void MontReduce(const BigInt<N>& a, BigInt<N>& b) {
  MontReduce(a, b, Config::kModulus, Config::kNPrime);
}

template <size_t N>
constexpr void MontMulReduce(BigInt<2 * N>& a, BigInt<N>& b,
                             const BigInt<N>& modulus, uint64_t n_prime,
                             bool has_modulus_spare_bit) {
  internal::AddResult<uint64_t> add_result;
  for (size_t i = 0; i < N; ++i) {
    uint64_t tmp = a[i] * n_prime;
    internal::MulResult<uint64_t> mul_result;
    mul_result =
        internal::MulAddWithCarry(a[i], tmp, modulus[0], mul_result.hi);
    for (size_t j = 1; j < N; ++j) {
      mul_result =
          internal::MulAddWithCarry(a[i + j], tmp, modulus[j], mul_result.hi);
      a[i + j] = mul_result.lo;
    }
    add_result =
        internal::AddWithCarry(a[N + i], mul_result.hi, add_result.carry);
    a[N + i] = add_result.value;
  }
  std::copy_n(&a[N], N, &b[0]);
  Reduce(b, modulus, has_modulus_spare_bit);
}

template <typename T,
          std::enable_if_t<std::is_integral_v<T> && sizeof(T) <= 8>* = nullptr>
constexpr void MontMul(T a, T b, T& c, T modulus, T n_prime) {
  using ExtT = internal::make_promoted_t<T>;

  auto t = ExtT{a} * b;
  T t_high = t >> (sizeof(T) * 8);

  T m = static_cast<T>(t) * n_prime;
  T mn_high = (ExtT{m} * modulus) >> (sizeof(T) * 8);

  if (t_high >= mn_high) {
    c = t_high - mn_high;
  } else {
    c = t_high + modulus - mn_high;
  }
}

template <typename Config, typename T>
constexpr void MontMul(T a, T b, T& c) {
  MontMul(a, b, c, Config::kModulus, Config::kNPrime);
}

template <size_t N>
constexpr void FastMontMul(const BigInt<N>& a, const BigInt<N>& b, BigInt<N>& c,
                           const BigInt<N>& modulus, uint64_t n_prime,
                           bool has_modulus_spare_bit) {
  c = {};
  for (size_t i = 0; i < N; ++i) {
    internal::MulResult<uint64_t> result;
    result = internal::MulAddWithCarry(c[0], a[0], b[i]);
    c[0] = result.lo;

    uint64_t k = c[0] * n_prime;
    internal::MulResult<uint64_t> result2;
    result2 = internal::MulAddWithCarry(c[0], k, modulus[0]);

    for (size_t j = 1; j < N; ++j) {
      result = internal::MulAddWithCarry(c[j], a[j], b[i], result.hi);
      c[j] = result.lo;
      result2 = internal::MulAddWithCarry(c[j], k, modulus[j], result2.hi);
      c[j - 1] = result2.lo;
    }
    c[N - 1] = result.hi + result2.hi;
  }
  Reduce(c, modulus, has_modulus_spare_bit);
}

template <size_t N>
constexpr void SlowMontMul(const BigInt<N>& a, const BigInt<N>& b, BigInt<N>& c,
                           const BigInt<N>& modulus, uint64_t n_prime,
                           bool has_modulus_spare_bit) {
  internal::MulResult<BigInt<N>> mul_result = BigInt<N>::Mul(a, b);
  BigInt<2 * N> mul;
  std::copy_n(&mul_result.lo[0], N, &mul[0]);
  std::copy_n(&mul_result.hi[0], N, &mul[N]);
  MontMulReduce(mul, c, modulus, n_prime, has_modulus_spare_bit);
}

template <size_t N>
constexpr void MontMul(const BigInt<N>& a, const BigInt<N>& b, BigInt<N>& c,
                       const BigInt<N>& modulus, uint64_t n_prime,
                       bool has_modulus_spare_bit,
                       bool can_use_no_carry_mul_optimization) {
  if (can_use_no_carry_mul_optimization) {
    FastMontMul(a, b, c, modulus, n_prime, has_modulus_spare_bit);
  } else {
    SlowMontMul(a, b, c, modulus, n_prime, has_modulus_spare_bit);
  }
}

template <typename Config, size_t N>
constexpr void MontMul(const BigInt<N>& a, const BigInt<N>& b, BigInt<N>& c) {
  MontMul(a, b, c, Config::kModulus, Config::kNPrime,
          HasModulusSpareBit<Config>(),
          CanUseNoCarryMulOptimization(Config::kModulus));
}

template <size_t N>
constexpr void MontSquare(const BigInt<N>& a, BigInt<N>& b,
                          const BigInt<N>& modulus, uint64_t n_prime,
                          bool has_modulus_spare_bit) {
  BigInt<2 * N> ret;
  internal::MulResult<uint64_t> mul_result;
  for (size_t i = 0; i < N - 1; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      mul_result =
          internal::MulAddWithCarry(ret[i + j], a[i], a[j], mul_result.hi);
      ret[i + j] = mul_result.lo;
    }
    ret[i + N] = mul_result.hi;
    mul_result.hi = 0;
  }

  ret[2 * N - 1] = ret[2 * N - 2] >> 63;
  for (size_t i = 2; i < 2 * N - 1; ++i) {
    ret[2 * N - i] = (ret[2 * N - i] << 1) | (ret[2 * N - (i + 1)] >> 63);
  }
  ret[1] <<= 1;

  internal::AddResult<uint64_t> add_result;
  for (size_t i = 0; i < N; ++i) {
    mul_result =
        internal::MulAddWithCarry(ret[2 * i], a[i], a[i], mul_result.hi);
    ret[2 * i] = mul_result.lo;
    add_result = internal::AddWithCarry(ret[2 * i + 1], mul_result.hi);
    ret[2 * i + 1] = add_result.value;
    mul_result.hi = add_result.carry;
  }
  MontMulReduce(ret, b, modulus, n_prime, has_modulus_spare_bit);
}

template <typename Config, size_t N>
constexpr void MontSquare(const BigInt<N>& a, BigInt<N>& b) {
  MontSquare(a, b, Config::kModulus, Config::kNPrime,
             HasModulusSpareBit<Config>());
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_MONT_MULTIPLICATION_H_
