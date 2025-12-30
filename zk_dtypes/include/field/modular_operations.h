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

#ifndef ZK_DTYPES_INCLUDE_FIELD_MODULAR_OPERATIONS_H_
#define ZK_DTYPES_INCLUDE_FIELD_MODULAR_OPERATIONS_H_

#include <cstddef>
#include <type_traits>

#include "zk_dtypes/include/arithmetics.h"
#include "zk_dtypes/include/big_int.h"

namespace zk_dtypes {
namespace internal {

template <typename T, typename SFINAE = void>
struct ArgTypeImpl;

template <typename T>
struct ArgTypeImpl<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = T;
};

template <size_t N>
struct ArgTypeImpl<BigInt<N>> {
  using type = const BigInt<N>&;
};

template <typename T>
using ArgType = typename ArgTypeImpl<T>::type;

template <typename T, typename SFINAE = void>
struct MutableArgTypeImpl;

template <typename T>
struct MutableArgTypeImpl<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = T&;
};

template <size_t N>
struct MutableArgTypeImpl<BigInt<N>> {
  using type = BigInt<N>&;
};

template <typename T>
using MutableArgType = typename MutableArgTypeImpl<T>::type;

template <typename T>
AddResult<T> ModAddHelper(ArgType<T> a, ArgType<T> b) {
  if constexpr (std::is_integral_v<T>) {
    return AddWithCarry(a, b, /*carry=*/0);
  } else {
    AddResult<T> ret;
    ret.carry = T::Add(a, b, ret.value);
    return ret;
  }
}

template <typename T>
AddResult<T> ModDoubleHelper(ArgType<T> a) {
  if constexpr (std::is_integral_v<T>) {
    return AddWithCarry(a, a, /*carry=*/0);
  } else {
    AddResult<T> ret;
    ret.carry = T::ShiftLeft(a, ret.value, 1);
    return ret;
  }
}

template <typename T>
T ModSubHelper(ArgType<T> a, ArgType<T> b, ArgType<T> modulus) {
  if constexpr (std::is_integral_v<T>) {
    using ExtT = internal::make_promoted_t<T>;
    return ExtT{a} + modulus - b;
  } else {
    BigInt<T::kLimbNums + 1> ext_a = a;
    BigInt<T::kLimbNums + 1> ext_b = b;
    BigInt<T::kLimbNums + 1> ext_modulus = modulus;
    BigInt<T::kLimbNums + 1> c = ext_a + ext_modulus - ext_b;
    return c.template Truncate<T::kLimbNums>();
  }
}

}  // namespace internal

// Does the modulus have a spare unused bit?
template <typename Config>
constexpr bool HasModulusSpareBit() {
  if constexpr (Config::kStorageBits <= 64) {
    uint64_t biggest_limb = uint64_t{Config::kModulus};
    return biggest_limb >> (Config::kStorageBits - 1) == 0;
  } else {
    uint64_t biggest_limb = Config::kModulus[(Config::kStorageBits / 64) - 1];
    return biggest_limb >> 63 == 0;
  }
}

// This reduces 0 <= value < 2 * modulus to 0 <= value < modulus.
template <typename T>
constexpr void Reduce(T& value, internal::ArgType<T> modulus,
                      bool has_modulus_spare_bit, bool carry = false) {
  bool needs_to_reduce = value >= modulus;
  if (!has_modulus_spare_bit) {
    needs_to_reduce |= carry;
  }
  if (needs_to_reduce) {
    value -= modulus;
  }
}

template <typename Config, typename T>
constexpr void Reduce(T& value, [[maybe_unused]] bool carry = false) {
  Reduce(value, Config::kModulus, HasModulusSpareBit<Config>(), carry);
}

template <typename T>
constexpr void ModAdd(internal::ArgType<T> a, internal::ArgType<T> b,
                      internal::MutableArgType<T> c,
                      internal::ArgType<T> modulus,
                      bool has_modulus_spare_bit) {
  bool carry = false;
  if (has_modulus_spare_bit) {
    c = a + b;
  } else {
    internal::AddResult<T> result = internal::ModAddHelper<T>(a, b);
    carry = result.carry;
    c = result.value;
  }
  Reduce(c, modulus, has_modulus_spare_bit, carry);
}

template <typename Config, typename T>
constexpr void ModAdd(internal::ArgType<T> a, internal::ArgType<T> b,
                      internal::MutableArgType<T> c) {
  ModAdd<T>(a, b, c, Config::kModulus, HasModulusSpareBit<Config>());
}

template <typename T>
constexpr void ModDouble(internal::ArgType<T> a, internal::MutableArgType<T> b,
                         internal::ArgType<T> modulus,
                         bool has_modulus_spare_bit) {
  bool carry = false;
  if (has_modulus_spare_bit) {
    b = a << 1;
  } else {
    internal::AddResult<T> result = internal::ModDoubleHelper<T>(a);
    carry = result.carry;
    b = result.value;
  }
  Reduce(b, modulus, has_modulus_spare_bit, carry);
}

template <typename Config, typename T>
constexpr void ModDouble(internal::ArgType<T> a,
                         internal::MutableArgType<T> b) {
  ModDouble<T>(a, b, Config::kModulus, HasModulusSpareBit<Config>());
}

template <typename T>
constexpr void ModSub(internal::ArgType<T> a, internal::ArgType<T> b,
                      internal::MutableArgType<T> c,
                      internal::ArgType<T> modulus,
                      bool has_modulus_spare_bit) {
  if (b > a) {
    if (has_modulus_spare_bit) {
      c = a + modulus - b;
    } else {
      c = internal::ModSubHelper<T>(a, b, modulus);
    }
  } else {
    c = a - b;
  }
}

template <typename Config, typename T>
constexpr void ModSub(internal::ArgType<T> a, internal::ArgType<T> b,
                      internal::MutableArgType<T> c) {
  ModSub<T>(a, b, c, Config::kModulus, HasModulusSpareBit<Config>());
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_MODULAR_OPERATIONS_H_
