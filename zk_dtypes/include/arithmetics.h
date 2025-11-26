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

#ifndef ZK_DTYPES_INCLUDE_ARITHMETICS_H_
#define ZK_DTYPES_INCLUDE_ARITHMETICS_H_

#include <stdint.h>

namespace zk_dtypes::internal {

template <typename T>
struct AddResult {
  T value{0};
  T carry{0};

  constexpr AddResult() = default;

  constexpr bool operator==(const AddResult& other) const {
    return value == other.value && carry == other.carry;
  }
  constexpr bool operator!=(const AddResult& other) const {
    return !operator==(other);
  }
};

template <typename T>
struct SubResult {
  T value{0};
  T borrow{0};

  constexpr SubResult() = default;

  constexpr bool operator==(const SubResult& other) const {
    return value == other.value && borrow == other.borrow;
  }
  constexpr bool operator!=(const SubResult& other) const {
    return !operator==(other);
  }
};

template <typename T>
struct MulResult {
  T hi{0};
  T lo{0};

  constexpr MulResult() = default;

  constexpr bool operator==(const MulResult& other) const {
    return hi == other.hi && lo == other.lo;
  }
  constexpr bool operator!=(const MulResult& other) const {
    return !operator==(other);
  }
};

template <typename T>
struct DivResult {
  T quotient{0};
  T remainder{0};

  constexpr DivResult() = default;

  constexpr bool operator==(const DivResult& other) const {
    return quotient == other.quotient && remainder == other.remainder;
  }
  constexpr bool operator!=(const DivResult& other) const {
    return !operator==(other);
  }
};

#if !defined(__SIZEOF_INT128__)
#error __uint128_t is not supported
#endif

template <typename T>
class PromotedTypeImpl;

template <>
class PromotedTypeImpl<uint64_t> {
 public:
  using type = __uint128_t;
};

template <>
class PromotedTypeImpl<uint32_t> {
 public:
  using type = uint64_t;
};

template <>
class PromotedTypeImpl<uint16_t> {
 public:
  using type = uint32_t;
};

template <>
class PromotedTypeImpl<uint8_t> {
 public:
  using type = uint16_t;
};

template <>
class PromotedTypeImpl<int64_t> {
 public:
  using type = __int128_t;
};

template <>
class PromotedTypeImpl<int32_t> {
 public:
  using type = int64_t;
};

template <>
class PromotedTypeImpl<int16_t> {
 public:
  using type = int32_t;
};

template <>
class PromotedTypeImpl<int8_t> {
 public:
  using type = int16_t;
};

template <typename T>
using make_promoted_t = typename PromotedTypeImpl<T>::type;

// Calculates a + b + carry.
#define ADD_WITH_CARRY_IMPL(T)                                 \
  constexpr AddResult<T> AddWithCarry(T a, T b, T carry = 0) { \
    using PromotedT = make_promoted_t<T>;                      \
    AddResult<T> result;                                       \
    PromotedT tmp = PromotedT{a} + b + carry;                  \
    result.value = static_cast<T>(tmp);                        \
    result.carry = static_cast<T>(tmp >> (sizeof(T) * 8));     \
    return result;                                             \
  }

ADD_WITH_CARRY_IMPL(uint64_t)
ADD_WITH_CARRY_IMPL(uint32_t)
ADD_WITH_CARRY_IMPL(uint16_t)
ADD_WITH_CARRY_IMPL(uint8_t)

#undef ADD_WITH_CARRY_IMPL

// Calculates a - b - borrow.
#define SUB_WITH_BORROW_IMPL(T)                                          \
  constexpr SubResult<T> SubWithBorrow(T a, T b, T borrow = 0) {         \
    using PromotedT = make_promoted_t<T>;                                \
    SubResult<T> result;                                                 \
    PromotedT tmp = (PromotedT{1} << sizeof(T) * 8) + a - b - borrow;    \
    result.value = static_cast<T>(tmp);                                  \
    result.borrow = static_cast<T>(tmp >> (sizeof(T) * 8)) == 0 ? 1 : 0; \
    return result;                                                       \
  }
SUB_WITH_BORROW_IMPL(uint64_t)
SUB_WITH_BORROW_IMPL(uint32_t)
SUB_WITH_BORROW_IMPL(uint16_t)
SUB_WITH_BORROW_IMPL(uint8_t)

#undef SUB_WITH_BORROW_IMPL

#define MUL_ADD_WITH_CARRY_IMPL(T)                                     \
  constexpr MulResult<T> MulAddWithCarry(T a, T b, T c, T carry = 0) { \
    using PromotedT = make_promoted_t<T>;                              \
    PromotedT tmp = static_cast<PromotedT>(b) * c + a + carry;         \
    MulResult<T> result;                                               \
    result.lo = static_cast<T>(tmp);                                   \
    result.hi = static_cast<T>(tmp >> (sizeof(T) * 8));                \
    return result;                                                     \
  }

// Calculates a + b * c.
MUL_ADD_WITH_CARRY_IMPL(uint64_t)
MUL_ADD_WITH_CARRY_IMPL(uint32_t)
MUL_ADD_WITH_CARRY_IMPL(uint16_t)
MUL_ADD_WITH_CARRY_IMPL(uint8_t)

#undef MUL_ADD_WITH_CARRY_IMPL

}  // namespace zk_dtypes::internal

#endif  // ZK_DTYPES_INCLUDE_ARITHMETICS_H_
