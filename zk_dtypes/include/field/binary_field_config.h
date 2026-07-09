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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_CONFIG_H_
#define ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_CONFIG_H_

#include <cstddef>
#include <type_traits>

namespace zk_dtypes {

// =============================================================================
// BinaryField Configs (Tower Field Structure)
// =============================================================================
// Binary tower fields: GF(2) -> GF(2²) -> GF(2⁴) -> ... -> GF(2¹²⁸)
// Each level is a degree-2 extension of the previous level, defined by the
// irreducible polynomial x² + βₖ₋₁·x + 1 where βₖ₋₁ is the generator of the
// subfield (the Fan-Paar/Binius tower — byte-compatible with Binius'
// BinaryField*b). See binary_field_multiplication.h for the arithmetic.

template <size_t TowerLevel>
struct BinaryFieldConfig;

template <size_t TowerLevel>
struct BaseBinaryFieldConfig {
  constexpr static bool kUseMontgomery = false;
  // Tower construction by default; a flat (e.g. GHASH) config overrides this.
  constexpr static bool kIsTower = true;
  constexpr static size_t kTowerLevel = TowerLevel;
  constexpr static size_t kStorageBits = 1 << kTowerLevel;
  constexpr static size_t kModulusBits = kStorageBits + 1;
  using SubfieldConfig = std::conditional_t<kTowerLevel == 0, void,
                                            BinaryFieldConfig<kTowerLevel - 1>>;
};

// GF(2) - Tower Level 0 (base field)
template <>
struct BinaryFieldConfig<0> : public BaseBinaryFieldConfig<0> {
  constexpr static size_t kValueMask = 0x1;
};

// GF(2²) - Tower Level 1
template <>
struct BinaryFieldConfig<1> : public BaseBinaryFieldConfig<1> {
  constexpr static size_t kValueMask = 0x3;
};

// GF(2⁴) - Tower Level 2
template <>
struct BinaryFieldConfig<2> : public BaseBinaryFieldConfig<2> {
  constexpr static size_t kValueMask = 0xF;
};

// GF(2⁸) - Tower Level 3
template <>
struct BinaryFieldConfig<3> : public BaseBinaryFieldConfig<3> {
  constexpr static size_t kValueMask = 0xFF;
};

// GF(2¹⁶) - Tower Level 4
template <>
struct BinaryFieldConfig<4> : public BaseBinaryFieldConfig<4> {
  constexpr static size_t kValueMask = 0xFFFF;
};

// GF(2³²) - Tower Level 5
template <>
struct BinaryFieldConfig<5> : public BaseBinaryFieldConfig<5> {
  constexpr static size_t kValueMask = 0xFFFFFFFF;
};

// GF(2⁶⁴) - Tower Level 6
template <>
struct BinaryFieldConfig<6> : public BaseBinaryFieldConfig<6> {
  constexpr static size_t kValueMask = 0xFFFFFFFFFFFFFFFF;
};

// GF(2¹²⁸) - Tower Level 7
template <>
struct BinaryFieldConfig<7> : public BaseBinaryFieldConfig<7> {};

// GF(2¹²⁸) in the GHASH/POLYVAL basis — a FLAT (non-tower) construction with
// irreducible p(x) = x¹²⁸ + x⁷ + x² + x + 1 (reduction constant 0x87), natural
// (non-bit-reflected) bit order. Reuses the tower-level-7 sizing (128-bit
// storage, BigInt<2>) but flips kIsTower so BinaryField dispatches to the GHASH
// multiply instead of the tower multiply. Isomorphic to BinaryFieldConfig<7>
// but NOT bit-compatible: this basis matches GHASH/POLYVAL byte-for-byte, which
// consumers that hash raw field bytes depend on.
struct GhashFieldConfig : public BaseBinaryFieldConfig<7> {
  constexpr static bool kIsTower = false;
};

// GF(2⁸) in the AES/Rijndael basis — a FLAT (non-tower) construction with
// irreducible p(x) = x⁸ + x⁴ + x³ + x + 1 (reduction constant 0x1B), natural
// bit order. Reuses the tower-level-3 sizing (8-bit storage, uint8, and
// kValueMask = 0xFF inherited from BinaryFieldConfig<3>) but flips kIsTower so
// BinaryField dispatches to the AES multiply instead of the tower multiply.
// Isomorphic to BinaryFieldConfig<3> but NOT bit-compatible: this basis matches
// AES and flock's φ₈ univariate skip byte-for-byte.
struct Gf8AesFieldConfig : public BinaryFieldConfig<3> {
  constexpr static bool kIsTower = false;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_CONFIG_H_
