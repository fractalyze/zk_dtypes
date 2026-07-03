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
// Binary tower fields use degree-2 extensions at each level, following the
// Fan-Paar / Wiedemann tower (Diamond–Posen 2023 §2.3) that Binius' canonical
// BinaryField*b types use. Choosing this tower makes stored bit patterns
// byte-compatible with Binius: the same u8/u16/u32/… value denotes the same
// field element, and products agree (e.g. GF(2⁸): 0x12·0x34 = 0x9B).
//
// Level k (k ≥ 2) is GF(2^{2^{k-1}})[X] / (X² + βₖ₋₁·X + 1), where βₖ₋₁ is the
// generator of the subfield (bit pattern 2^{2^{k-2}}, i.e. the subfield element
// (0,1)). This is NOT a constant-α tower: the linear coefficient is the
// subfield GENERATOR, not a fixed subfield constant. Multiplying a subfield
// element by βₖ₋₁ is exactly the recursive "multiply by generator" — BinaryMulX
// one level down. (At k = 1, β₀ = 1, so X² + X + 1 — the base GF(4) case.)
//
// For (a₀ + a₁·X) * (b₀ + b₁·X), using X² = βₖ₋₁·X + 1:
//   = a₀·b₀ + a₁·b₁ + [(a₀·b₁ + a₁·b₀) + βₖ₋₁·a₁·b₁]·X
// With Karatsuba (characteristic 2, subtraction = addition):
//   c₀ = a₀·b₀ + a₁·b₁
//   c₁ = (a₀+a₁)·(b₀+b₁) + a₀·b₀ + a₁·b₁ + βₖ₋₁·(a₁·b₁)

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

// The tower's per-level linear coefficient βₖ₋₁ is the subfield generator, not
// a stored constant: "multiply by βₖ₋₁" is the recursive BinaryMulX below, so
// no TowerAlpha table is needed.

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

    // X² = βₖ₋₁·X + 1  ⇒  c₀ = a₀·b₀ + a₁·b₁
    auto c0 = SubXor<TowerLevel>(a0b0, a1b1);
    // c₁ = (a₀ + a₁)(b₀ + b₁) + a₀·b₀ + a₁·b₁ + βₖ₋₁·(a₁·b₁), where βₖ₋₁·(·) is
    // multiply-by-generator one level down (BinaryMulX).
    auto cross = BinaryMul<TowerLevel - 1>(SubXor<TowerLevel>(a0, a1),
                                           SubXor<TowerLevel>(b0, b1));
    auto c1 = SubXor<TowerLevel>(
        SubXor<TowerLevel>(SubXor<TowerLevel>(cross, a0b0), a1b1),
        BinaryMulX<TowerLevel - 1>(a1b1));

    return Combine<TowerLevel>(c0, c1);
  }

  static constexpr T Square(T a) {
    auto a0 = ExtractLo<TowerLevel>(a), a1 = ExtractHi<TowerLevel>(a);
    auto a0_sq = BinarySquare<TowerLevel - 1>(a0);
    auto a1_sq = BinarySquare<TowerLevel - 1>(a1);
    // c₀ = a₀² + a₁²,  c₁ = βₖ₋₁·a₁²
    return Combine<TowerLevel>(SubXor<TowerLevel>(a0_sq, a1_sq),
                               BinaryMulX<TowerLevel - 1>(a1_sq));
  }

  static constexpr T MulX(T a) {
    // a·X = (a₀ + a₁·X)·X = a₀·X + a₁·X² = a₀·X + a₁·(βₖ₋₁·X + 1)
    //     = a₁ + (a₀ + βₖ₋₁·a₁)·X
    auto a0 = ExtractLo<TowerLevel>(a), a1 = ExtractHi<TowerLevel>(a);
    return Combine<TowerLevel>(
        a1, SubXor<TowerLevel>(a0, BinaryMulX<TowerLevel - 1>(a1)));
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

// =============================================================================
// GHASH / POLYVAL flat GF(2¹²⁸) multiplication (non-tower)
// =============================================================================
// GF(2¹²⁸) in the GHASH/POLYVAL basis: p(x) = x¹²⁸ + x⁷ + x² + x + 1 (reduction
// constant 0x87), natural (non-bit-reflected) bit order. An element is a
// BigInt<2> whose limb 0 holds x⁰..x⁶³ and limb 1 holds x⁶⁴..x¹²⁷. This is
// isomorphic to the tower GF(2¹²⁸) but a DIFFERENT, bit-incompatible basis: it
// matches the GHASH/POLYVAL representation byte-for-byte. These are the
// reference (portable, constexpr) semantics; a hardware carryless-multiply
// lowering (PCLMULQDQ) lives downstream in the compiler.

// Carryless (GF(2)[x]) 64×64 → 128 product, portable bit-serial. Writes the low
// and high 64 bits of the product to `lo`/`hi`.
constexpr void GhashClmul64(uint64_t a, uint64_t b, uint64_t& lo,
                            uint64_t& hi) {
  lo = 0;
  hi = 0;
  for (size_t i = 0; i < 64; ++i) {
    if ((a >> i) & uint64_t{1}) {
      lo ^= b << i;
      if (i != 0) hi ^= b >> (64 - i);  // i == 0 would be a shift-by-64 (UB)
    }
  }
}

// GF(2¹²⁸) multiply in the GHASH basis: schoolbook carryless 128×128 → 256
// product, then reduction mod x¹²⁸ + x⁷ + x² + x + 1.
constexpr BigInt<2> GhashMul(const BigInt<2>& a, const BigInt<2>& b) {
  uint64_t a_lo = a[0], a_hi = a[1], b_lo = b[0], b_hi = b[1];

  // Schoolbook 128×128 → 256 (unreduced): four 64×64 carryless products.
  uint64_t ll_lo = 0, ll_hi = 0, lh_lo = 0, lh_hi = 0;
  uint64_t hl_lo = 0, hl_hi = 0, hh_lo = 0, hh_hi = 0;
  GhashClmul64(a_lo, b_lo, ll_lo, ll_hi);
  GhashClmul64(a_lo, b_hi, lh_lo, lh_hi);
  GhashClmul64(a_hi, b_lo, hl_lo, hl_hi);
  GhashClmul64(a_hi, b_hi, hh_lo, hh_hi);
  uint64_t cr_lo = lh_lo ^ hl_lo;
  uint64_t cr_hi = lh_hi ^ hl_hi;
  uint64_t r0 = ll_lo;
  uint64_t r1 = ll_hi ^ cr_lo;
  uint64_t r2 = hh_lo ^ cr_hi;
  uint64_t r3 = hh_hi;

  // Fold the high half (r2:r3) down via x¹²⁸ ≡ x⁷ + x² + x + 1, with a 7-bit
  // overflow correction for coefficients pushed past x¹²⁷.
  uint64_t s1_lo = r2 << 1;
  uint64_t s1_hi = (r3 << 1) | (r2 >> 63);
  uint64_t s2_lo = r2 << 2;
  uint64_t s2_hi = (r3 << 2) | (r2 >> 62);
  uint64_t s7_lo = r2 << 7;
  uint64_t s7_hi = (r3 << 7) | (r2 >> 57);
  uint64_t t_lo = r2 ^ s1_lo ^ s2_lo ^ s7_lo;
  uint64_t t_hi = r3 ^ s1_hi ^ s2_hi ^ s7_hi;
  uint64_t ov = (r3 >> 63) ^ (r3 >> 62) ^ (r3 >> 57);
  uint64_t corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
  return BigInt<2>({r0 ^ t_lo ^ corr, r1 ^ t_hi});
}

constexpr BigInt<2> GhashSquare(const BigInt<2>& a) { return GhashMul(a, a); }

// Multiply by the generator x: shift coefficients up by one and reduce the
// x¹²⁸ overflow via x¹²⁸ ≡ x⁷ + x² + x + 1 (= 0x87).
constexpr BigInt<2> GhashMulX(const BigInt<2>& a) {
  uint64_t lo = a[0], hi = a[1];
  uint64_t overflow = hi >> 63;
  uint64_t new_hi = (hi << 1) | (lo >> 63);
  uint64_t new_lo = (lo << 1) ^ (overflow ? uint64_t{0x87} : uint64_t{0});
  return BigInt<2>({new_lo, new_hi});
}

// Inverse via Fermat: a⁻¹ = a^(2¹²⁸ − 2) = a² · a⁴ · a⁸ · … · a^(2¹²⁷).
constexpr BigInt<2> GhashInverse(const BigInt<2>& a) {
  if (a.IsZero()) return BigInt<2>(0);
  BigInt<2> result = GhashSquare(a);
  BigInt<2> power = result;
  for (size_t i = 2; i < 128; ++i) {
    power = GhashSquare(power);
    result = GhashMul(result, power);
  }
  return result;
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_MULTIPLICATION_H_
