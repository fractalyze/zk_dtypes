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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BIG_PRIME_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_BIG_PRIME_FIELD_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest_prod.h"

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/byinverter.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zk_dtypes/include/field/modular_operations.h"
#include "zk_dtypes/include/field/mont_multiplication.h"
#include "zk_dtypes/include/field/prime_field.h"
#include "zk_dtypes/include/intn.h"
#include "zk_dtypes/include/pow.h"

namespace zk_dtypes {

// If Config::kUseMontgomery is true, the operations are performed on montgomery
// domain. Otherwise, the operations are performed on standard domain.
template <typename _Config>
class PrimeField<_Config, std::enable_if_t<(_Config::kStorageBits > 64)>>
    : public FiniteField<PrimeField<_Config>> {
 public:
  constexpr static bool kUseMontgomery = _Config::kUseMontgomery;
  constexpr static size_t kStorageBits = _Config::kStorageBits;
  constexpr static size_t kLimbNums = (kStorageBits + 63) / 64;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t kBitWidth = BigInt<N>::kBitWidth;
  constexpr static size_t kByteWidth = BigInt<N>::kByteWidth;

  static_assert(kStorageBits == kBitWidth,
                "kStorageBits must be equal to kBitWidth");

  using Config = _Config;
  using StdType = PrimeField<typename Config::StdConfig>;
  using UnderlyingType = BigInt<N>;

  constexpr PrimeField() = default;
  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr PrimeField(T value) {
    if (value >= 0) {
      *this = PrimeField(BigInt<N>(value));
    } else {
      *this = -PrimeField(BigInt<N>(-value));
    }
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr PrimeField(T value) : PrimeField(BigInt<N>(value)) {}
  template <typename T>
  constexpr PrimeField(std::initializer_list<T> values)
      : PrimeField(BigInt<N>(values)) {}
  constexpr explicit PrimeField(const BigInt<N>& value) : value_(value) {
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
    return PrimeField::FromUnchecked(BigInt<N>::Random(Config::kModulus));
  }

  template <int N, typename UnderlyingTy>
  constexpr static PrimeField FromUnchecked(intN<N, UnderlyingTy> value) {
    if constexpr (std::is_signed_v<UnderlyingTy>) {
      DCHECK_GE(value, 0);
    }
    return PrimeField::FromUnchecked(BigInt<N>(value));
  }

  constexpr static PrimeField FromUnchecked(const BigInt<N>& value) {
    PrimeField ret;
    ret.value_ = value;
    return ret;
  }

  // Convert a decimal string to a PrimeField.
  static absl::StatusOr<PrimeField> FromDecString(std::string_view str) {
    absl::StatusOr<BigInt<N>> ret_value = BigInt<N>::FromDecString(str);
    if (!ret_value.ok()) return ret_value.status();
    return PrimeField(ret_value.value());
  }

  // Convert a hexadecimal string to a PrimeField.
  static absl::StatusOr<PrimeField> FromHexString(std::string_view str) {
    absl::StatusOr<BigInt<N>> ret_value = BigInt<N>::FromHexString(str);
    if (!ret_value.ok()) return ret_value.status();
    return PrimeField(ret_value.value());
  }

  constexpr const BigInt<N>& value() const { return value_; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const { return value_ == Config::kOne; }

  // See
  // https://github.com/Consensys/gnark-crypto/blob/43897fd/field/generator/internal/templates/element/base.go#L292-L308.
  // Returns true if this element is lexicographically larger than (q-1)/2.
  // This is equivalent to checking if value_ > ((Config::kModulus - 1) / 2).
  constexpr bool LexicographicallyLargest() const {
    constexpr BigInt<N> kHalfModulus = (Config::kModulus - 1) >> 1;
    if constexpr (kUseMontgomery) {
      return MontReduce().value() > kHalfModulus;
    } else {
      return value_ > kHalfModulus;
    }
  }

  constexpr PrimeField operator+(const PrimeField& other) const {
    PrimeField ret;
    ModAdd<Config, BigInt<N>>(value_, other.value_, ret.value_);
    return ret;
  }

  constexpr PrimeField& operator+=(const PrimeField& other) {
    return *this = *this + other;
  }

  constexpr PrimeField Double() const {
    PrimeField ret;
    ModDouble<Config, BigInt<N>>(value_, ret.value_);
    return ret;
  }

  constexpr PrimeField operator-(const PrimeField& other) const {
    PrimeField ret;
    ModSub<Config, BigInt<N>>(value_, other.value_, ret.value_);
    return ret;
  }

  constexpr PrimeField& operator-=(const PrimeField& other) {
    return *this = *this - other;
  }

  constexpr PrimeField operator-() const {
    if (IsZero()) return Zero();
    BigInt<N> ret_value = Config::kModulus;
    ret_value -= value_;
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField operator*(const PrimeField& other) const {
    PrimeField ret;
    if constexpr (kUseMontgomery) {
      zk_dtypes::MontMul<Config>(value_, other.value_, ret.value_);
    } else {
      VerySlowMul(*this, other, ret);
    }
    return ret;
  }

  constexpr PrimeField& operator*=(const PrimeField& other) {
    return *this = *this * other;
  }

  constexpr PrimeField Square() const {
    PrimeField ret;
    if constexpr (kUseMontgomery) {
      zk_dtypes::MontSquare<Config>(value_, ret.value_);
    } else {
      VerySlowMul(*this, *this, ret);
    }
    return ret;
  }

  template <typename T>
  constexpr PrimeField Pow(const T& exponent) const {
    if constexpr (std::is_same_v<T, PrimeField>) {
      if constexpr (kUseMontgomery) {
        return zk_dtypes::Pow(*this, exponent.MontReduce().value());
      } else {
        return zk_dtypes::Pow(*this, exponent.value());
      }
    } else {
      return zk_dtypes::Pow(*this, BigInt<N>(exponent));
    }
  }

  constexpr PrimeField operator/(const PrimeField& other) const {
    return operator*(other.Inverse());
  }

  // Returns the multiplicative inverse. Returns Zero() if not invertible.
  constexpr PrimeField Inverse() const {
    PrimeField ret;
    if constexpr (kUseMontgomery) {
      constexpr BYInverter<N> inverter =
          BYInverter<N>(Config::kModulus, Config::kRSquared);
      if (!inverter.Invert(value_, ret.value_)) {
        return Zero();
      }
    } else {
      constexpr BYInverter<N> inverter =
          BYInverter<N>(Config::kModulus, Config::kOne);
      if (!inverter.Invert(value_, ret.value_)) {
        return Zero();
      }
    }
    return ret;
  }

  constexpr uint64_t& operator[](size_t i) {
    DCHECK_LT(i, N);
    return value_[i];
  }
  constexpr const uint64_t& operator[](size_t i) const {
    DCHECK_LT(i, N);
    return value_[i];
  }

  constexpr bool operator==(const PrimeField& other) const {
    return value_ == other.value_;
  }

  constexpr bool operator!=(const PrimeField& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const PrimeField& other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() < other.MontReduce();
    } else {
      return value_ < other.value_;
    }
  }

  constexpr bool operator>(const PrimeField& other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() > other.MontReduce();
    } else {
      return value_ > other.value_;
    }
  }

  constexpr bool operator<=(const PrimeField& other) const {
    if constexpr (kUseMontgomery) {
      return MontReduce() <= other.MontReduce();
    } else {
      return value_ <= other.value_;
    }
  }

  constexpr bool operator>=(const PrimeField& other) const {
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
      return value_.ToString();
    }
  }
  std::string ToHexString(bool pad_zero = false) const {
    if constexpr (kUseMontgomery) {
      return MontReduce().ToHexString(pad_zero);
    } else {
      return value_.ToHexString(pad_zero);
    }
  }

 private:
  template <typename Config2, typename>
  friend class PrimeField;

  template <typename T>
  FRIEND_TEST(PrimeFieldTypedTest, Operations);

  constexpr static void VerySlowMul(const PrimeField& a, const PrimeField& b,
                                    PrimeField& c) {
    BigInt<2 * N> mul;
    auto value = BigInt<N>::Mul(a.value_, b.value_);
    memcpy(&mul[0], &value.lo[0], sizeof(uint64_t) * N);
    memcpy(&mul[N], &value.hi[0], sizeof(uint64_t) * N);
    BigInt<2 * N> modulus = BigInt<2 * N>::Zero();
    memcpy(&modulus[0], &Config::kModulus[0], sizeof(uint64_t) * N);
    BigInt<2 * N> mul_mod = mul % modulus;
    memcpy(&c.value_[0], &mul_mod[0], sizeof(uint64_t) * N);
  }

  BigInt<N> value_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BIG_PRIME_FIELD_H_
