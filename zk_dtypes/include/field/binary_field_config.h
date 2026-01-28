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
// Each level is a degree-2 extension of the previous level.
// The extension is defined by the irreducible polynomial x^2 + x + alpha,
// where alpha is the multiplicative generator of the subfield.

template <size_t TowerLevel>
struct BinaryFieldConfig;

template <size_t TowerLevel>
struct BaseBinaryFieldConfig {
  constexpr static bool kUseMontgomery = false;
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

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BINARY_FIELD_CONFIG_H_
