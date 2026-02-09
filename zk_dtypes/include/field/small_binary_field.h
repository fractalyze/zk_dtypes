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

#ifndef ZK_DTYPES_INCLUDE_FIELD_SMALL_BINARY_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_SMALL_BINARY_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/binary_field_config.h"
#include "zk_dtypes/include/field/binary_field_multiplication.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zk_dtypes/include/pow.h"
#include "zk_dtypes/include/random.h"

namespace zk_dtypes {

// =============================================================================
// Small Binary Field Implementation (kStorageBits <= 64)
// =============================================================================

template <typename _Config>
class BinaryField<_Config, std::enable_if_t<(_Config::kStorageBits <= 64)>>
    : public FiniteField<BinaryField<_Config>> {
 public:
  using Config = _Config;
  using UnderlyingType = std::conditional_t<
      Config::kStorageBits <= 32,
      std::conditional_t<
          Config::kStorageBits <= 16,
          std::conditional_t<Config::kStorageBits <= 8, uint8_t, uint16_t>,
          uint32_t>,
      uint64_t>;

  constexpr static bool kUseMontgomery =
      false;  // Binary fields don't use Montgomery
  constexpr static size_t kTowerLevel = Config::kTowerLevel;
  constexpr static size_t kStorageBits = Config::kStorageBits;
  constexpr static size_t kLimbNums = 1;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t kBitWidth = 8 * sizeof(UnderlyingType);
  constexpr static size_t kByteWidth = sizeof(UnderlyingType);

  // Subfield type (for tower operations)
  using SubfieldConfig = typename Config::SubfieldConfig;
  using Subfield = std::conditional_t<std::is_void_v<SubfieldConfig>, void,
                                      BinaryField<SubfieldConfig>>;

  constexpr BinaryField() = default;
  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr BinaryField(T value)
      : BinaryField(static_cast<std::make_unsigned_t<T>>(std::abs(value))) {}
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr BinaryField(T value) : value_(value) {
    DCHECK_EQ(value_, static_cast<UnderlyingType>(value) & Config::kValueMask);
  }

  constexpr static uint32_t ExtensionDegree() { return 1 << kTowerLevel; }

  constexpr static auto Order() {
    if constexpr (kTowerLevel <= 5) {
      return BigInt<1>(1) << (1 << kTowerLevel);
    } else {
      return BigInt<2>(1) << (1 << kTowerLevel);
    }
  }

  constexpr static BinaryField Zero() { return BinaryField(); }

  constexpr static BinaryField One() { return FromUnchecked(1); }

  constexpr static BinaryField Min() { return Zero(); }

  constexpr static BinaryField Max() {
    return FromUnchecked(Config::kValueMask);
  }

  constexpr static BinaryField Random() {
    return FromUnchecked(Uniform<UnderlyingType>() & Config::kValueMask);
  }

  constexpr static BinaryField FromUnchecked(UnderlyingType value) {
    BinaryField ret;
    ret.value_ = value;
    return ret;
  }

  constexpr UnderlyingType value() const { return value_; }

  constexpr bool IsZero() const { return value_ == 0; }

  constexpr bool IsOne() const { return value_ == 1; }

  // Addition: XOR in binary fields (characteristic 2)
  constexpr BinaryField operator+(BinaryField other) const {
    return FromUnchecked((value_ ^ other.value_) & Config::kValueMask);
  }

  constexpr BinaryField& operator+=(BinaryField other) {
    return *this = *this + other;
  }

  // Double: 2x = x + x = 0 in characteristic 2
  constexpr BinaryField Double() const { return Zero(); }

  // Subtraction: Same as addition in characteristic 2
  constexpr BinaryField operator-(BinaryField other) const {
    return FromUnchecked((value_ ^ other.value_) & Config::kValueMask);
  }

  constexpr BinaryField& operator-=(BinaryField other) {
    return *this = *this - other;
  }

  // Negation: Identity in characteristic 2 (-a = a)
  constexpr BinaryField operator-() const { return *this; }

  // Multiplication: Tower field multiplication
  constexpr BinaryField operator*(BinaryField other) const {
    return FromUnchecked(
        BinaryMul<kTowerLevel, UnderlyingType>(value_, other.value_));
  }

  constexpr BinaryField& operator*=(BinaryField other) {
    return *this = *this * other;
  }

  constexpr BinaryField Square() const {
    return FromUnchecked(BinarySquare<kTowerLevel, UnderlyingType>(value_));
  }

  // Multiply by X (extension generator)
  constexpr BinaryField MulX() const {
    return FromUnchecked(BinaryMulX<kTowerLevel, UnderlyingType>(value_));
  }

  // Inverse: Returns Zero() if not invertible (i.e., input is zero)
  constexpr BinaryField Inverse() const {
    return FromUnchecked(BinaryInverse<kTowerLevel, UnderlyingType>(value_));
  }

  constexpr BinaryField operator/(BinaryField other) const {
    return *this * other.Inverse();
  }

  constexpr BinaryField& operator/=(BinaryField other) {
    return *this = *this / other;
  }

  template <typename T>
  constexpr BinaryField Pow(T exponent) const {
    if constexpr (std::is_same_v<T, BinaryField>) {
      return zk_dtypes::Pow(*this, BigInt<1>(exponent.value()));
    } else {
      static_assert(std::is_integral_v<T>, "exponent must be an integer");
      return zk_dtypes::Pow(*this, static_cast<uint64_t>(exponent));
    }
  }

  constexpr bool operator==(BinaryField other) const {
    return value_ == other.value_;
  }

  constexpr bool operator!=(BinaryField other) const {
    return value_ != other.value_;
  }

  constexpr bool operator<(BinaryField other) const {
    return value_ < other.value_;
  }

  constexpr bool operator>(BinaryField other) const {
    return value_ > other.value_;
  }

  constexpr bool operator<=(BinaryField other) const {
    return value_ <= other.value_;
  }

  constexpr bool operator>=(BinaryField other) const {
    return value_ >= other.value_;
  }

  // Decompose into two subfield elements (for degree-2 extension)
  template <typename SubConfig = SubfieldConfig,
            std::enable_if_t<!std::is_void_v<SubConfig>>* = nullptr>
  constexpr std::pair<Subfield, Subfield> ToSubfields() const {
    constexpr size_t kSubfieldBits = Subfield::ExtensionDegree();
    constexpr UnderlyingType kSubfieldMask =
        (UnderlyingType{1} << kSubfieldBits) - 1;
    Subfield a0 = Subfield::FromUnchecked(value_ & kSubfieldMask);
    Subfield a1 =
        Subfield::FromUnchecked((value_ >> kSubfieldBits) & kSubfieldMask);
    return {a0, a1};
  }

  // Compose from two subfield elements
  template <typename SubConfig = SubfieldConfig,
            typename Sub = BinaryField<SubConfig>,
            std::enable_if_t<!std::is_void_v<SubConfig>>* = nullptr>
  constexpr static BinaryField FromSubfields(Sub a0, Sub a1) {
    constexpr size_t kSubfieldBits = Sub::ExtensionDegree();
    return FromUnchecked(
        static_cast<UnderlyingType>(a0.value()) |
        (static_cast<UnderlyingType>(a1.value()) << kSubfieldBits));
  }

  std::string ToString() const { return absl::StrCat(value_); }

  std::string ToHexString(bool pad_zero = false) const {
    std::string str = absl::StrCat(absl::Hex(value_));
    if (pad_zero) {
      size_t total_hex_length = (ExtensionDegree() + 3) / 4;  // Round up
      if (str.size() < total_hex_length) {
        return absl::StrCat(std::string(total_hex_length - str.size(), '0'),
                            str);
      }
    }
    return str;
  }

 private:
  UnderlyingType value_ = {};
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_SMALL_BINARY_FIELD_H_
