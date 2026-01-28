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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BIG_BINARY_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_BIG_BINARY_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <string>
#include <type_traits>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/binary_field_config.h"
#include "zk_dtypes/include/field/binary_field_multiplication.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zk_dtypes/include/pow.h"
#include "zk_dtypes/include/random.h"

namespace zk_dtypes {

// =============================================================================
// Big Binary Field Implementation (kStorageBits > 64, i.e., 128-bit)
// =============================================================================
// GF(2¹²⁸) - Tower Level 7

template <typename _Config>
class BinaryField<_Config, std::enable_if_t<(_Config::kStorageBits > 64)>>
    : public FiniteField<BinaryField<_Config>> {
 public:
  constexpr static bool kUseMontgomery = false;
  constexpr static size_t kTowerLevel = _Config::kTowerLevel;
  constexpr static size_t kStorageBits = _Config::kStorageBits;
  constexpr static size_t kLimbNums = (kStorageBits + 63) / 64;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t kBitWidth = kLimbNums * 64;
  constexpr static size_t kByteWidth = kLimbNums * 8;
  static_assert(N == 2);

  using Config = _Config;
  using UnderlyingType = BigInt<N>;

  // Subfield type (GF(2⁶⁴) for GF(2¹²⁸))
  using SubfieldConfig = typename Config::SubfieldConfig;
  using Subfield = BinaryField<SubfieldConfig>;

  constexpr BinaryField() = default;

  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr BinaryField(T value)
      : BinaryField(static_cast<std::make_unsigned_t<T>>(std::abs(value))) {}
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr BinaryField(T value) : BinaryField(BigInt<N>(value)) {}
  template <typename T>
  constexpr BinaryField(std::initializer_list<T> values)
      : BinaryField(BigInt<N>(values)) {}
  constexpr BinaryField(const BigInt<N>& value) : value_(value) {}

  constexpr static uint32_t ExtensionDegree() { return 1 << kTowerLevel; }

  constexpr static BigInt<N + 1> Order() {
    return BigInt<N + 1>(1) << (1 << kTowerLevel);
  }

  constexpr static BinaryField Zero() { return BinaryField(); }

  constexpr static BinaryField One() { return BinaryField::FromUnchecked(1); }

  constexpr static BinaryField Min() { return Zero(); }

  constexpr static BinaryField Max() {
    return BinaryField::FromUnchecked(BigInt<N>::Max());
  }

  constexpr static BinaryField Random() {
    return BinaryField::FromUnchecked(BigInt<N>::Random());
  }

  constexpr static BinaryField FromUnchecked(const BigInt<N>& value) {
    BinaryField ret;
    ret.value_ = value;
    return ret;
  }

  constexpr const BigInt<N>& value() const { return value_; }

  constexpr uint64_t operator[](size_t index) const { return value_[index]; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const { return value_.IsOne(); }

  // Addition: XOR
  constexpr BinaryField operator+(BinaryField other) const {
    return FromUnchecked(value_ ^ other.value_);
  }

  constexpr BinaryField& operator+=(BinaryField other) {
    return *this = *this + other;
  }

  // Double: 0 in characteristic 2
  constexpr BinaryField Double() const { return Zero(); }

  // Subtraction: Same as addition
  constexpr BinaryField operator-(BinaryField other) const {
    return FromUnchecked(value_ ^ other.value_);
  }

  constexpr BinaryField& operator-=(BinaryField other) {
    return *this = *this - other;
  }

  // Negation: Identity
  constexpr BinaryField operator-() const { return *this; }

  // Multiplication: Tower field multiplication
  // GF(2¹²⁸) = GF(2⁶⁴)[x] / (x² + x + α) where α = TowerAlpha<7>::value
  constexpr BinaryField operator*(BinaryField other) const {
    return FromUnchecked(BinaryMul<7>(value_, other.value_));
  }

  constexpr BinaryField& operator*=(BinaryField other) {
    return *this = *this * other;
  }

  constexpr BinaryField Square() const {
    return FromUnchecked(BinarySquare<7>(value_));
  }

  // Multiply by X (extension generator)
  // X² = X + α where α = TowerAlpha<7>::value
  constexpr BinaryField MulX() const {
    return FromUnchecked(BinaryMulX<7>(value_));
  }

  // Inverse using Fermat's Little Theorem: a⁻¹ = a^(2¹²⁸ - 2)
  // a^(2¹²⁸ - 2) = a² · a⁴ · a⁸ · ... · a^(2¹²⁷)
  constexpr BinaryField Inverse() const {
    return FromUnchecked(BinaryInverse<7>(value_));
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
      return zk_dtypes::Pow(*this, exponent.value_);
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

  // Decompose into two GF(2⁶⁴) elements
  constexpr std::pair<Subfield, Subfield> ToSubfields() const {
    return {Subfield::FromUnchecked(value_[0]),
            Subfield::FromUnchecked(value_[1])};
  }

  // Compose from two GF(2⁶⁴) elements
  constexpr static BinaryField FromSubfields(Subfield a0, Subfield a1) {
    return FromUnchecked(BigInt<N>({a0.value(), a1.value()}));
  }

  std::string ToString() const { return value_.ToString(); }

  std::string ToHexString(bool pad_zero = false) const {
    return value_.ToHexString(pad_zero);
  }

 private:
  BigInt<N> value_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BIG_BINARY_FIELD_H_
