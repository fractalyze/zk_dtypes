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

#ifndef ZK_DTYPES_INCLUDE_FIELD_SMALL_PRIME_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_SMALL_PRIME_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest_prod.h"

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/byinverter.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zk_dtypes/include/field/modular_operations.h"
#include "zk_dtypes/include/field/mont_multiplication.h"
#include "zk_dtypes/include/field/prime_field.h"
#include "zk_dtypes/include/intn.h"
#include "zk_dtypes/include/pow.h"
#include "zk_dtypes/include/random.h"

namespace zk_dtypes {

// If Config::kUseMontgomery is true, the operations are performed on montgomery
// domain. Otherwise, the operations are performed on standard domain.
template <typename _Config>
class PrimeField<_Config, std::enable_if_t<(_Config::kStorageBits <= 64)>>
    : public FiniteField<PrimeField<_Config>> {
 public:
  using UnderlyingType = std::conditional_t<
      _Config::kStorageBits <= 32,
      std::conditional_t<
          _Config::kStorageBits <= 16,
          std::conditional_t<_Config::kStorageBits <= 8, uint8_t, uint16_t>,
          uint32_t>,
      uint64_t>;

  constexpr static bool kUseMontgomery = _Config::kUseMontgomery;
  constexpr static size_t kStorageBits = _Config::kStorageBits;
  constexpr static size_t kLimbNums = 1;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t kBitWidth = 8 * sizeof(UnderlyingType);
  constexpr static size_t kByteWidth = sizeof(UnderlyingType);

  static_assert(kStorageBits == kBitWidth,
                "kStorageBits must be equal to kBitWidth");

  using Config = _Config;
  using StdType = PrimeField<typename Config::StdConfig>;

  constexpr PrimeField() = default;
  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr PrimeField(T value) {
    if (value == 0) return;
    if (value == 1) {
      *this = One();
      return;
    }

    if (value > 0) {
      *this = PrimeField(static_cast<UnderlyingType>(value));
    } else {
      *this = -PrimeField(static_cast<UnderlyingType>(-value));
    }
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr PrimeField(T value) : value_(value) {
    DCHECK_LT(value_, Config::kModulus);
    if constexpr (kUseMontgomery) {
      operator*=(PrimeField::FromUnchecked(Config::kRSquared));
    }
  }

  constexpr static uint32_t ExtensionDegree() { return 1; }

  constexpr static BigInt<N> Order() { return Config::kModulus; }

  constexpr static PrimeField Zero() { return PrimeField(); }

  constexpr static PrimeField One() {
    return PrimeField::FromUnchecked(Config::kOne);
  }

  constexpr static PrimeField Min() { return Zero(); }

  constexpr static PrimeField Max() { return PrimeField(-1); }

  constexpr static PrimeField Random() {
    return PrimeField::FromUnchecked(
        Uniform<UnderlyingType>(0, Config::kModulus));
  }

  template <int N, typename UnderlyingTy>
  constexpr static PrimeField FromUnchecked(intN<N, UnderlyingTy> value) {
    if constexpr (std::is_signed_v<UnderlyingTy>) {
      DCHECK_GE(value, 0);
    }
    return PrimeField::FromUnchecked(static_cast<UnderlyingType>(value));
  }

  constexpr static PrimeField FromUnchecked(UnderlyingType value) {
    PrimeField ret = {};
    ret.value_ = value;
    return ret;
  }

  // Convert a decimal string to a PrimeField.
  static absl::StatusOr<PrimeField> FromDecString(std::string_view str) {
    uint64_t ret_value;
    if (!absl::SimpleAtoi(str, &ret_value)) {
      return absl::InvalidArgumentError("failed to convert to uint64_t");
    }
    return PrimeField(ret_value);
  }

  // Convert a hexadecimal string to a PrimeField.
  static absl::StatusOr<PrimeField> FromHexString(std::string_view str) {
    uint64_t ret_value;
    if (!absl::SimpleHexAtoi(str, &ret_value)) {
      return absl::InvalidArgumentError("failed to convert to uint64_t");
    }
    return PrimeField(ret_value);
  }

  constexpr UnderlyingType value() const { return value_; }

  constexpr bool IsZero() const { return value_ == 0; }

  constexpr bool IsOne() const { return value_ == Config::kOne; }

  // See
  // https://github.com/Consensys/gnark-crypto/blob/43897fd/field/generator/internal/templates/element/base.go#L292-L308.
  // Returns true if this element is lexicographically larger than (q-1)/2.
  // This is equivalent to checking if value_ > ((Config::kModulus - 1) / 2).
  constexpr bool LexicographicallyLargest() const {
    constexpr UnderlyingType kHalfModulus = (Config::kModulus - 1) >> 1;
    if constexpr (kUseMontgomery) {
      return MontReduce().value() > kHalfModulus;
    } else {
      return value_ > kHalfModulus;
    }
  }

  constexpr PrimeField operator+(PrimeField other) const {
    PrimeField ret;
    ModAdd<Config, UnderlyingType>(value_, other.value_, ret.value_);
    return ret;
  }

  constexpr PrimeField& operator+=(PrimeField other) {
    return *this = *this + other;
  }

  constexpr PrimeField Double() const {
    PrimeField ret;
    ModDouble<Config, UnderlyingType>(value_, ret.value_);
    return ret;
  }

  constexpr PrimeField operator-(PrimeField other) const {
    PrimeField ret;
    ModSub<Config, UnderlyingType>(value_, other.value_, ret.value_);
    return ret;
  }

  constexpr PrimeField& operator-=(PrimeField other) {
    return *this = *this - other;
  }

  constexpr PrimeField operator-() const {
    if (value_ == 0) return Zero();
    return PrimeField::FromUnchecked(Config::kModulus - value_);
  }

  constexpr PrimeField operator*(PrimeField other) const {
    PrimeField ret;
    if constexpr (kUseMontgomery) {
      zk_dtypes::MontMul(value_, other.value_, ret.value_, Config::kModulus,
                         Config::kNPrime);
    } else {
      VerySlowMul(*this, other, ret);
    }
    return ret;
  }

  constexpr PrimeField& operator*=(PrimeField other) {
    return *this = *this * other;
  }

  constexpr PrimeField Square() const { return *this * *this; }

  template <typename T>
  constexpr PrimeField Pow(T exponent) const {
    if constexpr (std::is_same_v<T, PrimeField>) {
      if constexpr (kUseMontgomery) {
        return zk_dtypes::Pow(*this, BigInt<1>(exponent.MontReduce().value()));
      } else {
        return zk_dtypes::Pow(*this, BigInt<1>(exponent.value()));
      }
    } else {
      static_assert(std::is_integral_v<T>, "exponent must be an integer");
      return zk_dtypes::Pow(*this, static_cast<UnderlyingType>(exponent));
    }
  }

  constexpr PrimeField operator/(PrimeField other) const {
    return operator*(other.Inverse());
  }

  // Returns the multiplicative inverse. Returns Zero() if not invertible.
  constexpr PrimeField Inverse() const {
    BigInt<1> ret;
    if constexpr (kUseMontgomery) {
      constexpr BYInverter<1> inverter =
          BYInverter<1>(Config::kModulus, Config::kRSquared);
      if (!inverter.Invert(BigInt<1>(value_), ret)) {
        return Zero();
      }
    } else {
      constexpr BYInverter<1> inverter =
          BYInverter<1>(Config::kModulus, Config::kOne);
      if (!inverter.Invert(BigInt<1>(value_), ret)) {
        return Zero();
      }
    }
    return PrimeField::FromUnchecked(ret[0]);
  }

  constexpr bool operator==(PrimeField other) const {
    return value_ == other.value_;
  }

  constexpr bool operator!=(PrimeField other) const {
    return !operator==(other);
  }

  constexpr bool operator<(PrimeField other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() < other.MontReduce();
    } else {
      return value_ < other.value_;
    }
  }

  constexpr bool operator>(PrimeField other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() > other.MontReduce();
    } else {
      return value_ > other.value_;
    }
  }

  constexpr bool operator<=(PrimeField other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() <= other.MontReduce();
    } else {
      return value_ <= other.value_;
    }
  }

  constexpr bool operator>=(PrimeField other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() >= other.MontReduce();
    } else {
      return value_ >= other.value_;
    }
  }

  template <typename Config2 = Config,
            std::enable_if_t<Config2::kUseMontgomery>* = nullptr>
  StdType MontReduce() const {
    StdType ret;
    zk_dtypes::MontReduce(value_, ret.value_, Config::kModulus,
                          Config::kNPrime);
    return ret;
  }

  std::string ToString() const {
    if constexpr (kUseMontgomery) {
      return MontReduce().ToString();
    } else {
      return absl::StrCat(value_);
    }
  }

  std::string ToHexString(bool pad_zero = false) const {
    if constexpr (kUseMontgomery) {
      return MontReduce().ToHexString(pad_zero);
    } else {
      std::string str = absl::StrCat(absl::Hex(value_));
      if (pad_zero) {
        size_t total_hex_length = kByteWidth * 2;
        if (str.size() < total_hex_length) {
          return absl::StrCat(std::string(total_hex_length - str.size(), '0'),
                              str);
        }
      }
      return str;
    }
  }

  // ExtensionFieldOperation methods
  constexpr PrimeField CreateConst(int64_t value) const {
    return PrimeField(value);
  }

 private:
  template <typename Config2, typename>
  friend class PrimeField;

  template <typename T>
  FRIEND_TEST(PrimeFieldTypedTest, Operations);

  constexpr static void VerySlowMul(PrimeField a, PrimeField b, PrimeField& c) {
    using PromotedUnderlyingType = internal::make_promoted_t<UnderlyingType>;

    auto mul = PromotedUnderlyingType{a.value_} * b.value_ % Config::kModulus;
    c = PrimeField::FromUnchecked(static_cast<UnderlyingType>(mul));
  }

  UnderlyingType value_ = {};
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_SMALL_PRIME_FIELD_H_
