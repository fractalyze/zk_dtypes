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

#ifndef ZK_DTYPES_INCLUDE_SIGNED_BIG_INT_H_
#define ZK_DTYPES_INCLUDE_SIGNED_BIG_INT_H_

#include <limits>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/status/statusor.h"

#include "zk_dtypes/include/big_int.h"

namespace zk_dtypes {
namespace internal {

std::string SignedLimbsToString(const uint64_t* limbs, size_t limb_nums);
std::string SignedLimbsToHexString(const uint64_t* limbs, size_t limb_nums,
                                   bool pad_zero);

}  // namespace internal

// SignedBigInt is a signed fixed-size integer type built on top of BigInt.
// Unlike the base BigInt which treats all values as unsigned, SignedBigInt
// interprets values as two's complement signed integers. This affects:
//   - Comparison operators (<, >, <=, >=)
//   - Division and remainder (/, %)
//   - Right shift (>> uses arithmetic shift)
//   - String conversion (ToString, ToHexString)
//   - Min/Max range
//
// Operations that are identical in two's complement (addition, subtraction,
// multiplication, bitwise ops, left shift, equality) are inherited from BigInt.
template <size_t N>
class SignedBigInt : public BigInt<N> {
 public:
  using BigInt<N>::kLimbBitWidth;
  using BigInt<N>::kBitWidth;
  using BigInt<N>::kByteWidth;

  constexpr SignedBigInt() : BigInt<N>(0) {}

  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr SignedBigInt(T value) : BigInt<N>() {
    if (value >= 0) {
      this->limbs_[0] = static_cast<uint64_t>(value);
    } else {
      // Sign-extend: fill all limbs with all-1s, then store the low 64 bits.
      for (size_t i = 0; i < N; ++i) {
        this->limbs_[i] = std::numeric_limits<uint64_t>::max();
      }
      this->limbs_[0] = static_cast<uint64_t>(value);
    }
  }

  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr SignedBigInt(T value) : BigInt<N>() {
    this->limbs_[0] = value;
  }

  constexpr SignedBigInt(const BigInt<N>& v) : BigInt<N>(v) {}  // NOLINT

  // --- Sign predicates ---

  constexpr bool IsNegative() const {
    return (this->limbs_[N - 1] >> (kLimbBitWidth - 1)) != 0;
  }

  constexpr bool IsNonNegative() const { return !IsNegative(); }

  constexpr bool IsStrictlyPositive() const {
    return !IsNegative() && !this->IsZero();
  }

  // --- Signed Min/Max ---

  // Signed minimum: 0x8000...0000 (most negative value).
  constexpr static SignedBigInt Min() {
    SignedBigInt ret;
    ret.limbs_[N - 1] = uint64_t{1} << (kLimbBitWidth - 1);
    return ret;
  }

  // Signed maximum: 0x7FFF...FFFF (most positive value).
  constexpr static SignedBigInt Max() {
    SignedBigInt ret;
    for (size_t i = 0; i < N; ++i) {
      ret.limbs_[i] = std::numeric_limits<uint64_t>::max();
    }
    ret.limbs_[N - 1] = std::numeric_limits<uint64_t>::max() >> 1;
    return ret;
  }

  // --- Signed comparison ---

  constexpr bool operator<(const SignedBigInt& other) const {
    return SCmp(*this, other) < 0;
  }

  constexpr bool operator>(const SignedBigInt& other) const {
    return SCmp(*this, other) > 0;
  }

  constexpr bool operator<=(const SignedBigInt& other) const {
    return SCmp(*this, other) <= 0;
  }

  constexpr bool operator>=(const SignedBigInt& other) const {
    return SCmp(*this, other) >= 0;
  }

  constexpr SignedBigInt operator/(const SignedBigInt& other) const {
    return SignedBigInt(SDiv(*this, other).quotient);
  }

  constexpr SignedBigInt operator%(const SignedBigInt& other) const {
    return SignedBigInt(SDiv(*this, other).remainder);
  }

  // --- Arithmetic right shift (sign-extending) ---

  constexpr SignedBigInt operator>>(uint64_t shift) const {
    SignedBigInt ret;
    AShr(*this, ret, shift);
    return ret;
  }

  constexpr SignedBigInt& operator>>=(uint64_t shift) {
    AShr(*this, *this, shift);
    return *this;
  }

  // --- Return type covariance for inherited operations ---
  // These ensure chained expressions preserve SignedBigInt type.

  constexpr SignedBigInt operator-() const {
    return SignedBigInt(BigInt<N>::operator-());
  }

  constexpr SignedBigInt operator+(const SignedBigInt& other) const {
    return SignedBigInt(BigInt<N>::operator+(other));
  }

  constexpr SignedBigInt& operator+=(const SignedBigInt& other) {
    BigInt<N>::operator+=(other);
    return *this;
  }

  constexpr SignedBigInt operator-(const SignedBigInt& other) const {
    return SignedBigInt(BigInt<N>::operator-(other));
  }

  constexpr SignedBigInt& operator-=(const SignedBigInt& other) {
    BigInt<N>::operator-=(other);
    return *this;
  }

  constexpr SignedBigInt operator*(const SignedBigInt& other) const {
    return SignedBigInt(BigInt<N>::operator*(other));
  }

  constexpr SignedBigInt& operator*=(const SignedBigInt& other) {
    BigInt<N>::operator*=(other);
    return *this;
  }

  constexpr SignedBigInt operator<<(uint64_t shift) const {
    return SignedBigInt(BigInt<N>::operator<<(shift));
  }

  constexpr SignedBigInt& operator<<=(uint64_t shift) {
    BigInt<N>::operator<<=(shift);
    return *this;
  }

  constexpr SignedBigInt operator^(const SignedBigInt& other) const {
    return SignedBigInt(BigInt<N>::operator^(other));
  }

  constexpr SignedBigInt& operator^=(const SignedBigInt& other) {
    BigInt<N>::operator^=(other);
    return *this;
  }

  constexpr SignedBigInt operator&(const SignedBigInt& other) const {
    return SignedBigInt(BigInt<N>::operator&(other));
  }

  constexpr SignedBigInt& operator&=(const SignedBigInt& other) {
    BigInt<N>::operator&=(other);
    return *this;
  }

  constexpr SignedBigInt operator|(const SignedBigInt& other) const {
    return SignedBigInt(BigInt<N>::operator|(other));
  }

  constexpr SignedBigInt& operator|=(const SignedBigInt& other) {
    BigInt<N>::operator|=(other);
    return *this;
  }

  // --- String conversion ---

  std::string ToString() const {
    return internal::SignedLimbsToString(this->limbs(), N);
  }

  std::string ToHexString(bool pad_zero = false) const {
    return internal::SignedLimbsToHexString(this->limbs(), N, pad_zero);
  }

  static absl::StatusOr<SignedBigInt> FromDecString(std::string_view str) {
    bool is_neg = !str.empty() && str[0] == '-';
    if (is_neg) {
      str.remove_prefix(1);
    }
    auto result = BigInt<N>::FromDecString(str);
    if (!result.ok()) return result.status();
    return is_neg ? SignedBigInt(-*result) : SignedBigInt(*result);
  }

  static absl::StatusOr<SignedBigInt> FromHexString(std::string_view str) {
    bool is_neg = !str.empty() && str[0] == '-';
    if (is_neg) {
      str.remove_prefix(1);
    }
    auto result = BigInt<N>::FromHexString(str);
    if (!result.ok()) return result.status();
    return is_neg ? SignedBigInt(-*result) : SignedBigInt(*result);
  }

 private:
  // Returns: -1 if a < b, 0 if a == b, 1 if a > b (signed interpretation).
  constexpr static int SCmp(const SignedBigInt& a, const SignedBigInt& b) {
    bool a_neg = a.IsNegative();
    bool b_neg = b.IsNegative();

    if (a_neg != b_neg) {
      return a_neg ? -1 : 1;
    }
    // Same sign: unsigned limb comparison gives correct ordering in two's
    // complement.
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (a.limbs_[i] < b.limbs_[i]) return -1;
      if (a.limbs_[i] > b.limbs_[i]) return 1;
    }
    return 0;
  }

  // Returns quotient and remainder where:
  //   quotient = trunc(a / b)
  //   remainder sign follows dividend (a).
  constexpr static internal::DivResult<BigInt<N>> SDiv(const SignedBigInt& a,
                                                       const SignedBigInt& b) {
    bool a_neg = a.IsNegative();
    bool b_neg = b.IsNegative();

    BigInt<N> abs_a = a_neg ? -static_cast<const BigInt<N>&>(a)
                            : static_cast<const BigInt<N>&>(a);
    BigInt<N> abs_b = b_neg ? -static_cast<const BigInt<N>&>(b)
                            : static_cast<const BigInt<N>&>(b);

    internal::DivResult<BigInt<N>> result = BigInt<N>::Div(abs_a, abs_b);

    // Quotient is negative when operands have different signs.
    if (a_neg != b_neg && !result.quotient.IsZero()) {
      result.quotient = -result.quotient;
    }
    // Remainder sign follows dividend.
    if (a_neg && !result.remainder.IsZero()) {
      result.remainder = -result.remainder;
    }
    return result;
  }

  // Fills vacated high bits with the sign bit instead of zero.
  constexpr static uint64_t AShr(const SignedBigInt& a, SignedBigInt& b,
                                 uint64_t shift) {
    bool is_neg = a.IsNegative();

    if (shift >= kBitWidth) {
      if (is_neg) {
        for (size_t i = 0; i < N; ++i) {
          b.limbs_[i] = std::numeric_limits<uint64_t>::max();
        }
      } else {
        b = SignedBigInt(0);
      }
      return 0;
    }

    uint64_t borrow = BigInt<N>::ShiftRight(a, b, shift);

    if (is_neg && shift > 0) {
      // Fill the top `shift` bits with 1s.
      size_t fill_start = kBitWidth - shift;
      size_t start_limb = fill_start / kLimbBitWidth;
      size_t start_bit = fill_start % kLimbBitWidth;

      if (start_bit != 0) {
        b.limbs_[start_limb] |= ~((uint64_t{1} << start_bit) - 1);
        ++start_limb;
      }
      for (size_t i = start_limb; i < N; ++i) {
        b.limbs_[i] = std::numeric_limits<uint64_t>::max();
      }
    }

    return borrow;
  }
};

template <size_t N>
std::ostream& operator<<(std::ostream& os, const SignedBigInt<N>& v) {
  return os << v.ToString();
}

template <size_t N>
class BitTraits<SignedBigInt<N>> : public BitTraits<BigInt<N>> {};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_SIGNED_BIG_INT_H_
