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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_MULTIPLICATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_MULTIPLICATION_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "zk_dtypes/include/big_int.h"

namespace zk_dtypes {

// =============================================================================
// Tower Field Multiplication
// =============================================================================
// Binary tower fields use degree-2 extensions at each level.
// GF(2²ⁿ) is constructed as GF(2ⁿ)[X] / (X² + X + α)
// where α is the primitive element (multiplicative generator) of GF(2ⁿ).
//
// For (a₀ + a₁·X) * (b₀ + b₁·X) in GF(2²ⁿ):
//   = a₀·b₀ + (a₀·b₁ + a₁·b₀)·X + a₁·b₁·X²
// Using X² = X + α:
//   = a₀·b₀ + a₁·b₁·α + (a₀·b₁ + a₁·b₀ + a₁·b₁)·X
//
// With Karatsuba optimization:
//   c₀ = a₀·b₀ + a₁·b₁·α
//   c₁ = (a₀+a₁)·(b₀+b₁) + a₀·b₀  (characteristic 2: subtraction = addition)

// =============================================================================
// Tower Traits - Defines type mappings for each tower level
// =============================================================================

template <size_t TowerLevel>
struct TowerTraits;

template <size_t TowerLevel, typename ValueType>
struct BaseTowerTraits {
  constexpr static size_t kBits = 1 << TowerLevel;
  constexpr static size_t kSubfieldBits = TowerTraits<TowerLevel - 1>::kBits;
  constexpr static ValueType kSubfieldMask = TowerTraits<TowerLevel - 1>::kMask;
  using SubfieldValueType = typename TowerTraits<TowerLevel - 1>::ValueType;
};

template <>
struct TowerTraits<SIZE_MAX> {
  using ValueType = void;
};

template <>
struct TowerTraits<0> : public BaseTowerTraits<0, uint8_t> {
  using ValueType = uint8_t;
  constexpr static ValueType kMask = 0x1;
};

template <>
struct TowerTraits<1> : public BaseTowerTraits<1, uint8_t> {
  using ValueType = uint8_t;
  constexpr static ValueType kMask = 0x3;
};

template <>
struct TowerTraits<2> : public BaseTowerTraits<2, uint8_t> {
  using ValueType = uint8_t;
  constexpr static ValueType kMask = 0xF;
};

template <>
struct TowerTraits<3> : public BaseTowerTraits<3, uint8_t> {
  using ValueType = uint8_t;
  constexpr static ValueType kMask = 0xFF;
};

template <>
struct TowerTraits<4> : public BaseTowerTraits<4, uint16_t> {
  using ValueType = uint16_t;
  constexpr static ValueType kMask = 0xFFFF;
};

template <>
struct TowerTraits<5> : public BaseTowerTraits<5, uint32_t> {
  using ValueType = uint32_t;
  constexpr static ValueType kMask = 0xFFFFFFFF;
};

template <>
struct TowerTraits<6> : public BaseTowerTraits<6, uint64_t> {
  using ValueType = uint64_t;
  static constexpr ValueType kMask = 0xFFFFFFFFFFFFFFFF;
};

template <>
struct TowerTraits<7> : public BaseTowerTraits<7, BigInt<2>> {
  using ValueType = BigInt<2>;
  static constexpr ValueType kMask = BigInt<2>::Max();
};

// =============================================================================
// Tower Polynomial Constants (α for each level)
// =============================================================================
// At each tower level n, we extend GF(2^(2ⁿ⁻¹)) to GF(2^(2ⁿ)) using
// the irreducible polynomial X² + X + αₙ = 0.
// The αₙ value must be chosen such that X² + X + αₙ is irreducible.
// This means αₙ must NOT be in the image of the map a → a² + a.

template <size_t TowerLevel>
struct TowerAlpha;

template <>
struct TowerAlpha<1> {
  // GF(2²): X² + X + 1
  static constexpr uint8_t value = 1;
};

template <>
struct TowerAlpha<2> {
  // GF(2⁴): X² + X + 2
  static constexpr uint8_t value = 2;
};

template <>
struct TowerAlpha<3> {
  // GF(2⁸): X² + X + 8
  static constexpr uint8_t value = 8;
};

template <>
struct TowerAlpha<4> {
  // GF(2¹⁶): X² + X + 128
  static constexpr uint8_t value = 128;
};

template <>
struct TowerAlpha<5> {
  // GF(2³²): X² + X + 0x8000
  static constexpr uint16_t value = 32768;
};

template <>
struct TowerAlpha<6> {
  // GF(2⁶⁴): X² + X + 0x80000000
  static constexpr uint32_t value = 2147483648u;
};

template <>
struct TowerAlpha<7> {
  // GF(2¹²⁸)
  static constexpr uint64_t value = 0x8000000000000000ull;
};

// =============================================================================
// Forward declarations
// =============================================================================

template <size_t TowerLevel,
          typename T = typename TowerTraits<TowerLevel>::ValueType>
constexpr T BinaryMul(T a, T b);

template <size_t TowerLevel,
          typename T = typename TowerTraits<TowerLevel>::ValueType>
constexpr T BinarySquare(T a);

template <size_t TowerLevel,
          typename T = typename TowerTraits<TowerLevel>::ValueType>
constexpr T BinaryMulX(T a);

template <size_t TowerLevel,
          typename T = typename TowerTraits<TowerLevel>::ValueType>
constexpr T BinaryInverse(T a);

// =============================================================================
// Helper Functions for Tower Operations
// =============================================================================

// Combine two subfield values into parent field value
template <size_t TowerLevel>
constexpr auto Combine(typename TowerTraits<TowerLevel>::SubfieldValueType lo,
                       typename TowerTraits<TowerLevel>::SubfieldValueType hi) {
  using T = typename TowerTraits<TowerLevel>::ValueType;
  constexpr size_t shift = TowerTraits<TowerLevel>::kSubfieldBits;
  return static_cast<T>(lo) | (static_cast<T>(hi) << shift);
}

// Extract low subfield value
template <size_t TowerLevel>
constexpr auto ExtractLo(typename TowerTraits<TowerLevel>::ValueType a) {
  using Sub = typename TowerTraits<TowerLevel>::SubfieldValueType;
  return static_cast<Sub>(a & TowerTraits<TowerLevel>::kSubfieldMask);
}

// Extract high subfield value
template <size_t TowerLevel>
constexpr auto ExtractHi(typename TowerTraits<TowerLevel>::ValueType a) {
  using Sub = typename TowerTraits<TowerLevel>::SubfieldValueType;
  constexpr size_t shift = TowerTraits<TowerLevel>::kSubfieldBits;
  return static_cast<Sub>((a >> shift) &
                          TowerTraits<TowerLevel>::kSubfieldMask);
}

// XOR two subfield values (avoids int promotion issues)
template <size_t TowerLevel>
constexpr auto SubXor(typename TowerTraits<TowerLevel>::SubfieldValueType a,
                      typename TowerTraits<TowerLevel>::SubfieldValueType b) {
  using Sub = typename TowerTraits<TowerLevel>::SubfieldValueType;
  return static_cast<Sub>(a ^ b);
}

// =============================================================================
// Binary Operations Implementation - Using Partial Specialization
// =============================================================================

template <size_t TowerLevel, typename Enable = void>
struct BinaryOps;

// -----------------------------------------------------------------------------
// Level 0: GF(2) - Base case
// -----------------------------------------------------------------------------
template <>
struct BinaryOps<0> {
  static constexpr uint8_t Mul(uint8_t a, uint8_t b) { return (a & b) & 0x1; }

  static constexpr uint8_t Square(uint8_t a) { return a & 0x1; }

  static constexpr uint8_t MulX(uint8_t a) {
    return a & 0x1;  // X = 1 in GF(2)
  }

  static constexpr uint8_t Inverse(uint8_t a) {
    return a & 0x1;  // 1⁻¹ = 1 in GF(2)
  }
};

// -----------------------------------------------------------------------------
// Level 1: GF(2²) - Special case with α = 1
// -----------------------------------------------------------------------------
template <>
struct BinaryOps<1> {
  static constexpr uint8_t Mul(uint8_t a, uint8_t b) {
    uint8_t a0 = a & 0x1, a1 = (a >> 1) & 0x1;
    uint8_t b0 = b & 0x1, b1 = (b >> 1) & 0x1;

    uint8_t a0b0 = a0 & b0;
    uint8_t a1b1 = a1 & b1;

    // c₀ = a₀·b₀ + a₁·b₁ (α = 1 in GF(2))
    uint8_t c0 = a0b0 ^ a1b1;
    // c₁ = (a₀ + a₁)·(b₀ + b₁) + a₀·b₀
    uint8_t c1 = ((a0 ^ a1) & (b0 ^ b1)) ^ a0b0;

    return (c0 | (c1 << 1)) & 0x3;
  }

  static constexpr uint8_t Square(uint8_t a) {
    uint8_t a0 = a & 0x1, a1 = (a >> 1) & 0x1;
    // a² = (a₀ + a₁·X)² = a₀ + a₁·X² = a₀ + a₁·(X + 1) = (a₀ + a₁) + a₁·X
    return ((a0 ^ a1) | (a1 << 1)) & 0x3;
  }

  static constexpr uint8_t MulX(uint8_t a) {
    uint8_t a0 = a & 0x1, a1 = (a >> 1) & 0x1;
    // a·X = (a₀ + a₁·X)·X = a₀·X + a₁·X² = a₀·X + a₁·(X + 1) = a₁ + (a₀ + a₁)·X
    return (a1 | ((a0 ^ a1) << 1)) & 0x3;
  }

  static constexpr uint8_t Inverse(uint8_t a) {
    if (a == 0) return 0;
    return Square(a);  // a⁻¹ = a^(2² - 2) = a²
  }
};

// -----------------------------------------------------------------------------
// Levels 2-7: Generic Tower Operations
// -----------------------------------------------------------------------------
template <size_t TowerLevel>
struct BinaryOps<TowerLevel,
                 std::enable_if_t<(TowerLevel >= 2 && TowerLevel <= 7)>> {
  using T = typename TowerTraits<TowerLevel>::ValueType;
  using Sub = typename TowerTraits<TowerLevel>::SubfieldValueType;
  static constexpr size_t kBits = TowerTraits<TowerLevel>::kBits;

  static constexpr T Mul(T a, T b) {
    auto a0 = ExtractLo<TowerLevel>(a), a1 = ExtractHi<TowerLevel>(a);
    auto b0 = ExtractLo<TowerLevel>(b), b1 = ExtractHi<TowerLevel>(b);

    auto a0b0 = BinaryMul<TowerLevel - 1>(a0, b0);
    auto a1b1 = BinaryMul<TowerLevel - 1>(a1, b1);

    // c₀ = a₀·b₀ + α·a₁·b₁
    auto alpha_a1b1 =
        BinaryMul<TowerLevel - 1>(a1b1, TowerAlpha<TowerLevel>::value);
    auto c0 = SubXor<TowerLevel>(a0b0, alpha_a1b1);
    // c1 = (a₀ + a₁)(b₀ + b₁) + a₀·b₀
    auto c1 = SubXor<TowerLevel>(
        BinaryMul<TowerLevel - 1>(SubXor<TowerLevel>(a0, a1),
                                  SubXor<TowerLevel>(b0, b1)),
        a0b0);

    return Combine<TowerLevel>(c0, c1);
  }

  static constexpr T Square(T a) {
    auto a0 = ExtractLo<TowerLevel>(a), a1 = ExtractHi<TowerLevel>(a);
    auto a0_sq = BinarySquare<TowerLevel - 1>(a0);
    auto a1_sq = BinarySquare<TowerLevel - 1>(a1);
    // c₀ = a₀² + α·a1²
    auto alpha_a1_sq =
        BinaryMul<TowerLevel - 1>(a1_sq, TowerAlpha<TowerLevel>::value);
    return Combine<TowerLevel>(SubXor<TowerLevel>(a0_sq, alpha_a1_sq), a1_sq);
  }

  static constexpr T MulX(T a) {
    // a·X = (a₀ + a₁·X)·X = a₀·X + a₁·X² = a₀·X + a₁·(X + α)
    //     = α·a₁ + (a₀ + a₁)·X
    auto a0 = ExtractLo<TowerLevel>(a), a1 = ExtractHi<TowerLevel>(a);
    auto alpha_a1 =
        BinaryMul<TowerLevel - 1>(a1, TowerAlpha<TowerLevel>::value);
    return Combine<TowerLevel>(alpha_a1, SubXor<TowerLevel>(a0, a1));
  }

  static constexpr T Inverse(T a) {
    if (a == 0) return 0;
    // a⁻¹ = a^(2^kBits - 2) using repeated squaring
    T result = BinarySquare<TowerLevel>(a);
    T power = result;
    for (size_t i = 2; i < kBits; ++i) {
      power = BinarySquare<TowerLevel>(power);
      result = BinaryMul<TowerLevel>(result, power);
    }
    return result;
  }
};

// =============================================================================
// Public API - Delegate to BinaryOps
// =============================================================================

template <size_t TowerLevel, typename T>
constexpr T BinaryMul(T a, T b) {
  return BinaryOps<TowerLevel>::Mul(a, b);
}

template <size_t TowerLevel, typename T>
constexpr T BinarySquare(T a) {
  return BinaryOps<TowerLevel>::Square(a);
}

template <size_t TowerLevel, typename T>
constexpr T BinaryMulX(T a) {
  return BinaryOps<TowerLevel>::MulX(a);
}

template <size_t TowerLevel, typename T>
constexpr T BinaryInverse(T a) {
  return BinaryOps<TowerLevel>::Inverse(a);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_MULTIPLICATION_H_
