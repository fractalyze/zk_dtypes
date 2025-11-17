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
#include "zk_dtypes/include/field/prime_field.h"
#include "zk_dtypes/include/pow.h"

namespace zk_dtypes {

// If Config::kUseMontgomery is true, the operations are performed on montgomery
// domain. Otherwise, the operations are performed on standard domain.
template <typename _Config>
class PrimeField<_Config, std::enable_if_t<(_Config::kModulusBits > 64)>>
    : public FiniteField<PrimeField<_Config>> {
 public:
  constexpr static bool kUseMontgomery = _Config::kUseMontgomery;
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t kBitWidth = BigInt<N>::kBitWidth;
  constexpr static size_t kByteWidth = BigInt<N>::kByteWidth;

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

  constexpr static PrimeField Zero() { return PrimeField(); }

  constexpr static PrimeField One() {
    return PrimeField::FromUnchecked(Config::kOne);
  }

  constexpr static PrimeField Min() { return Zero(); }

  constexpr static PrimeField Max() {
    return PrimeField::FromUnchecked(Config::kModulus - 1);
  }

  constexpr static PrimeField Random() {
    return PrimeField::FromUnchecked(BigInt<N>::Random(Config::kModulus));
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
    BigInt<N> ret_value;
    bool carry = false;
    if constexpr (HasSpareBit()) {
      ret_value = value_ + other.value_;
    } else {
      carry = BigInt<N>::Add(value_, other.value_, ret_value);
    }
    Clamp(ret_value, carry);
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField& operator+=(const PrimeField& other) {
    return *this = *this + other;
  }

  constexpr PrimeField Double() const {
    BigInt<N> ret_value;
    bool carry = false;
    if constexpr (HasSpareBit()) {
      ret_value = value_ << 1;
    } else {
      carry = BigInt<N>::ShiftLeft(value_, ret_value, 1);
    }
    Clamp(ret_value, carry);
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField operator-(const PrimeField& other) const {
    BigInt<N> ret_value;
    if (other.value_ > value_) {
      static_assert(HasSpareBit());
      ret_value = value_ + Config::kModulus;
      ret_value -= other.value_;
    } else {
      ret_value = value_ - other.value_;
    }
    return PrimeField::FromUnchecked(ret_value);
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
      if constexpr (CanUseNoCarryMulOptimization()) {
        FastMontMul(*this, other, ret);
      } else {
        SlowMontMul(*this, other, ret);
      }
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
      FastMontSquare(*this, ret);
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

  constexpr absl::StatusOr<PrimeField> operator/(
      const PrimeField& other) const {
    absl::StatusOr<PrimeField> inv = other.Inverse();
    if (!inv.ok()) return inv;
    return operator*(inv.value());
  }

  constexpr absl::StatusOr<PrimeField> Inverse() const {
    PrimeField ret;
    if constexpr (kUseMontgomery) {
      constexpr BYInverter<N> inverter =
          BYInverter<N>(Config::kModulus, Config::kRSquared);
      if (!inverter.Invert(value_, ret.value_)) {
        return absl::InvalidArgumentError("division by zero");
      }
    } else {
      constexpr BYInverter<N> inverter =
          BYInverter<N>(Config::kModulus, Config::kOne);
      if (!inverter.Invert(value_, ret.value_)) {
        return absl::InvalidArgumentError("division by zero");
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
    BigInt<N> ret = value_;
    for (size_t i = 0; i < N; ++i) {
      uint64_t k = ret[i] * Config::kNPrime;
      internal::MulResult<uint64_t> result =
          internal::MulAddWithCarry(ret[i], k, Config::kModulus[0], 0);
      for (size_t j = 1; j < N; ++j) {
        result = internal::MulAddWithCarry(ret[(j + i) % N], k,
                                           Config::kModulus[j], result.hi);
        ret[(j + i) % N] = result.lo;
      }
      ret[i] = result.hi;
    }
    return StdType::FromUnchecked(ret);
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
  template <typename T>
  FRIEND_TEST(PrimeFieldTypedTest, Operations);

  // Does the modulus have a spare unused bit?
  //
  // This condition applies if
  // (a) `modulus[biggest_limb_idx] >> 63 == 0`
  constexpr static bool HasSpareBit() {
    uint64_t biggest_limb = Config::kModulus[N - 1];
    return biggest_limb >> 63 == 0;
  }

  // Can we use the no-carry optimization for multiplication
  // outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
  //
  // This optimization applies if
  // (a) `modulus[biggest_limb_idx] < max(uint64_t) >> 1`, and
  // (b) the bits of the modulus are not all 1.
  constexpr static bool CanUseNoCarryMulOptimization() {
    uint64_t biggest_limb = Config::kModulus[N - 1];
    bool top_bit_is_zero = biggest_limb >> 63 == 0;
    bool all_remaining_bits_are_one =
        biggest_limb == std::numeric_limits<uint64_t>::max() >> 1;
    for (size_t i = 0; i < N - 1; ++i) {
      all_remaining_bits_are_one &=
          Config::kModulus[i] == std::numeric_limits<uint64_t>::max();
    }
    return top_bit_is_zero && !all_remaining_bits_are_one;
  }

  constexpr static void Clamp(BigInt<N>& value,
                              [[maybe_unused]] bool carry = false) {
    bool needs_to_clamp = value >= Config::kModulus;
    if constexpr (!HasSpareBit()) {
      needs_to_clamp |= carry;
    }
    if (needs_to_clamp) {
      value -= Config::kModulus;
    }
  }

  constexpr static void FastMontMul(const PrimeField& a, const PrimeField& b,
                                    PrimeField& c) {
    BigInt<N> ret;
    for (size_t i = 0; i < N; ++i) {
      internal::MulResult<uint64_t> result;
      result = internal::MulAddWithCarry(ret[0], a[0], b[i]);
      ret[0] = result.lo;

      uint64_t k = ret[0] * Config::kNPrime;
      internal::MulResult<uint64_t> result2;
      result2 = internal::MulAddWithCarry(ret[0], k, Config::kModulus[0]);

      for (size_t j = 1; j < N; ++j) {
        result = internal::MulAddWithCarry(ret[j], a[j], b[i], result.hi);
        ret[j] = result.lo;
        result2 = internal::MulAddWithCarry(ret[j], k, Config::kModulus[j],
                                            result2.hi);
        ret[j - 1] = result2.lo;
      }
      ret[N - 1] = result.hi + result2.hi;
    }
    Clamp(ret, 0);
    c.value_ = ret;
  }

  constexpr static void SlowMontMul(const PrimeField& a, const PrimeField& b,
                                    PrimeField& c) {
    internal::MulResult<BigInt<N>> mul_result =
        BigInt<N>::Mul(a.value_, b.value_);
    BigInt<2 * N> mul;
    memcpy(&mul[0], &mul_result.lo[0], sizeof(uint64_t) * N);
    memcpy(&mul[N], &mul_result.hi[0], sizeof(uint64_t) * N);
    MontMulReduce(mul, c.value_);
  }

  constexpr static void VerySlowMul(const PrimeField& a, const PrimeField& b,
                                    PrimeField& c) {
    BigInt<2 * N> mul;
    auto value = BigInt<N>::Mul(a.value_, b.value_);
    memcpy(&mul[0], &value.lo[0], sizeof(uint64_t) * N);
    memcpy(&mul[N], &value.hi[0], sizeof(uint64_t) * N);
    BigInt<2 * N> modulus = BigInt<2 * N>::Zero();
    memcpy(&modulus[0], &Config::kModulus[0], sizeof(uint64_t) * N);
    BigInt<2 * N> mul_mod = *(mul % modulus);
    memcpy(&c.value_[0], &mul_mod[0], sizeof(uint64_t) * N);
  }

  constexpr static void FastMontSquare(const PrimeField& a, PrimeField& b) {
    BigInt<2 * N> ret;
    internal::MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N - 1; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        mul_result =
            internal::MulAddWithCarry(ret[i + j], a[i], a[j], mul_result.hi);
        ret[i + j] = mul_result.lo;
      }
      ret[i + N] = mul_result.hi;
      mul_result.hi = 0;
    }

    ret[2 * N - 1] = ret[2 * N - 2] >> 63;
    for (size_t i = 2; i < 2 * N - 1; ++i) {
      ret[2 * N - i] = (ret[2 * N - i] << 1) | (ret[2 * N - (i + 1)] >> 63);
    }
    ret[1] <<= 1;

    internal::AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      mul_result =
          internal::MulAddWithCarry(ret[2 * i], a[i], a[i], mul_result.hi);
      ret[2 * i] = mul_result.lo;
      add_result = internal::AddWithCarry(ret[2 * i + 1], mul_result.hi);
      ret[2 * i + 1] = add_result.value;
      mul_result.hi = add_result.carry;
    }
    MontMulReduce(ret, b.value_);
  }

  constexpr static void MontMulReduce(BigInt<2 * N>& a, BigInt<N>& b) {
    internal::AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      uint64_t tmp = a[i] * Config::kNPrime;
      internal::MulResult<uint64_t> mul_result;
      mul_result = internal::MulAddWithCarry(a[i], tmp, Config::kModulus[0],
                                             mul_result.hi);
      for (size_t j = 1; j < N; ++j) {
        mul_result = internal::MulAddWithCarry(
            a[i + j], tmp, Config::kModulus[j], mul_result.hi);
        a[i + j] = mul_result.lo;
      }
      add_result =
          internal::AddWithCarry(a[N + i], mul_result.hi, add_result.carry);
      a[N + i] = add_result.value;
    }
    memcpy(&b[0], &a[N], sizeof(uint64_t) * N);
    Clamp(b, add_result.carry);
  }

  BigInt<N> value_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BIG_PRIME_FIELD_H_
