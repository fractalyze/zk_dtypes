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

#ifndef ZK_DTYPES_INCLUDE_BIG_INT_H_
#define ZK_DTYPES_INCLUDE_BIG_INT_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <initializer_list>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "zk_dtypes/include/arithmetics.h"
#include "zk_dtypes/include/bit_traits_forward.h"
#include "zk_dtypes/include/random.h"

namespace zk_dtypes {
namespace internal {

absl::Status StringToLimbs(std::string_view str, uint64_t* limbs,
                           size_t limb_nums);
absl::Status HexStringToLimbs(std::string_view str, uint64_t* limbs,
                              size_t limb_nums);

std::string LimbsToString(const uint64_t* limbs, size_t limb_nums);
std::string LimbsToHexString(const uint64_t* limbs, size_t limb_nums,
                             bool pad_zero);

}  // namespace internal

// BigInt is a fixed size array of uint64_t, capable of holding up to N limbs.
template <size_t N>
class BigInt {
 public:
  constexpr static size_t kLimbByteWidth = sizeof(uint64_t);
  constexpr static size_t kLimbBitWidth = kLimbByteWidth * 8;

  constexpr static size_t kLimbNums = N;
  constexpr static size_t kBitWidth = N * kLimbBitWidth;
  constexpr static size_t kByteWidth = N * kLimbByteWidth;

  constexpr BigInt() : BigInt(0) {}
  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr BigInt(T value) : limbs_{0} {
    DCHECK_GE(value, 0);
    limbs_[0] = value;
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr BigInt(T value) : limbs_{0} {
    limbs_[0] = value;
  }
  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr BigInt(std::initializer_list<T> values) : limbs_{0} {
    DCHECK_LE(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < values.size(); ++i, ++it) {
      DCHECK_GE(*it, 0);
      limbs_[i] = *it;
    }
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr BigInt(std::initializer_list<T> values) : limbs_{0} {
    DCHECK_LE(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < values.size(); ++i, ++it) {
      limbs_[i] = *it;
    }
  }
  template <size_t N2>
  constexpr BigInt(const BigInt<N2>& other) : limbs_{0} {
    static_assert(N >= N2,
                  "Destination BigInt size N must be greater than or equal to "
                  "source size N2.");
    std::copy_n(other.limbs_, N2, limbs_);
  }

  // Convert a decimal string to a BigInt.
  static absl::StatusOr<BigInt> FromDecString(std::string_view str) {
    BigInt ret(0);
    absl::Status status = internal::StringToLimbs(str, ret.limbs_, N);
    if (!status.ok()) return status;
    return ret;
  }

  // Convert a hexadecimal string to a BigInt.
  static absl::StatusOr<BigInt> FromHexString(std::string_view str) {
    BigInt ret(0);
    absl::Status status = internal::HexStringToLimbs(str, ret.limbs_, N);
    if (!status.ok()) return status;
    return ret;
  }

  // Constructs a BigInt value from a given array of bits in little-endian
  // order.
  template <size_t BitNums = kBitWidth>
  constexpr static BigInt FromBitsLE(const std::bitset<BitNums>& bits) {
    static_assert(BitNums <= kBitWidth);
    BigInt ret;
    size_t bit_idx = 0;
    size_t limb_idx = 0;
    std::bitset<kLimbBitWidth> limb_bits;
    for (size_t i = 0; i < BitNums; ++i) {
      limb_bits.set(bit_idx++, bits[i]);
      bool set = bit_idx == kLimbBitWidth;
      set |= (i == BitNums - 1);
      if (set) {
        uint64_t limb = absl::bit_cast<uint64_t>(limb_bits.to_ullong());
        ret.limbs_[limb_idx++] = limb;
        limb_bits.reset();
        bit_idx = 0;
      }
    }
    return ret;
  }

  // Constructs a BigInt value from a given array of bits in big-endian order.
  template <size_t BitNums = kBitWidth>
  constexpr static BigInt FromBitsBE(const std::bitset<BitNums>& bits) {
    static_assert(BitNums <= kBitWidth);
    BigInt ret;
    std::bitset<kLimbBitWidth> limb_bits;
    size_t bit_idx = 0;
    size_t limb_idx = 0;
    for (size_t i = BitNums - 1; i != SIZE_MAX; --i) {
      limb_bits.set(bit_idx++, bits[i]);
      bool set = bit_idx == kLimbBitWidth;
      set |= (i == 0);
      if (set) {
        uint64_t limb = absl::bit_cast<uint64_t>(limb_bits.to_ullong());
        ret.limbs_[limb_idx++] = limb;
        limb_bits.reset();
        bit_idx = 0;
      }
    }
    return ret;
  }

  // Constructs a BigInt value from a given byte container interpreted in
  // little-endian order. The method processes each byte of the input, packs
  // them into 64-bit limbs, and then sets these limbs in the resulting BigInt.
  // If the system is big-endian, adjustments are made to ensure correct byte
  // ordering.
  template <typename ByteContainer>
  constexpr static BigInt FromBytesLE(const ByteContainer& bytes) {
    BigInt ret;
    size_t byte_idx = 0;
    size_t limb_idx = 0;
    uint64_t limb = 0;
    for (size_t i = 0; i < std::size(bytes); ++i) {
      reinterpret_cast<uint8_t*>(&limb)[byte_idx++] = bytes[i];
      bool set = byte_idx == kLimbByteWidth;
      set |= (i == std::size(bytes) - 1);
      if (set) {
        ret.limbs_[limb_idx++] = limb;
        limb = 0;
        byte_idx = 0;
      }
    }
    return ret;
  }

  // Constructs a BigInt value from a given byte container interpreted in
  // big-endian order. The method processes each byte of the input, packs them
  // into 64-bit limbs, and then sets these limbs in the resulting BigInt. If
  // the system is little-endian, adjustments are made to ensure correct byte
  // ordering.
  template <typename ByteContainer>
  constexpr static BigInt FromBytesBE(const ByteContainer& bytes) {
    BigInt ret;
    size_t byte_idx = 0;
    size_t limb_idx = 0;
    uint64_t limb = 0;
    for (size_t i = std::size(bytes) - 1; i != SIZE_MAX; --i) {
      reinterpret_cast<uint8_t*>(&limb)[byte_idx++] = bytes[i];
      bool set = byte_idx == kLimbByteWidth;
      set |= (i == 0);
      if (set) {
        ret.limbs_[limb_idx++] = limb;
        limb = 0;
        byte_idx = 0;
      }
    }
    return ret;
  }

  constexpr static BigInt Zero() { return BigInt(0); }

  constexpr static BigInt One() { return BigInt(1); }

  constexpr static BigInt Min() { return Zero(); }

  constexpr static BigInt Max() {
    BigInt ret;
    for (uint64_t& limb : ret.limbs_) {
      limb = std::numeric_limits<uint64_t>::max();
    }
    return ret;
  }

  // Generate a random BigInt between [0, `max`).
  constexpr static BigInt Random(const BigInt& max = Max()) {
    BigInt ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = Uniform<uint64_t>();
    }
    while (ret >= max) {
      ret >>= 1;
    }
    return ret;
  }

  constexpr const uint64_t* limbs() const { return limbs_; }
  constexpr uint64_t* limbs() { return limbs_; }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < N; ++i) {
      if (limbs_[i] != 0) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
    for (size_t i = 1; i < N - 1; ++i) {
      if (limbs_[i] != 0) {
        return false;
      }
    }
    return limbs_[0] == 1 && limbs_[N - 1] == 0;
  }

  constexpr bool IsEven() const { return limbs_[0] % 2 == 0; }
  constexpr bool IsOdd() const { return limbs_[0] % 2 == 1; }

  constexpr explicit operator uint64_t() const { return limbs_[0]; }

  BigInt operator-() const {
    BigInt ret = *this;

    for (size_t i = 0; i < N; ++i) {
      ret[i] = ~ret[i];
    }

    uint64_t carry = 1;
    for (size_t i = 0; i < N; ++i) {
      internal::AddResult<uint64_t> add_result =
          internal::AddWithCarry(ret[i], carry);
      ret[i] = add_result.value;
      carry = add_result.carry;
      if (carry == 0) break;
    }
    return ret;
  }

  constexpr BigInt operator+(const BigInt& other) const {
    BigInt ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr BigInt& operator+=(const BigInt& other) {
    Add(*this, other, *this);
    return *this;
  }

  constexpr BigInt operator-(const BigInt& other) const {
    BigInt ret;
    Sub(*this, other, ret);
    return ret;
  }

  constexpr BigInt& operator-=(const BigInt& other) {
    Sub(*this, other, *this);
    return *this;
  }

  constexpr BigInt operator*(const BigInt& other) const {
    return Mul(*this, other).lo;
  }

  constexpr BigInt& operator*=(const BigInt& other) {
    return *this = Mul(*this, other).lo;
  }

  constexpr BigInt operator<<(uint64_t shift) const {
    BigInt ret;
    ShiftLeft(*this, ret, shift);
    return ret;
  }

  constexpr BigInt& operator<<=(uint64_t shift) {
    ShiftLeft(*this, *this, shift);
    return *this;
  }

  constexpr BigInt operator>>(uint64_t shift) const {
    BigInt ret;
    ShiftRight(*this, ret, shift);
    return ret;
  }

  constexpr BigInt& operator>>=(uint64_t shift) {
    ShiftRight(*this, *this, shift);
    return *this;
  }

  constexpr BigInt operator^(const BigInt& other) const {
    BigInt ret;
    Xor(*this, other, ret);
    return ret;
  }

  constexpr BigInt& operator^=(const BigInt& other) {
    Xor(*this, other, *this);
    return *this;
  }

  constexpr BigInt operator&(const BigInt& other) const {
    BigInt ret;
    And(*this, other, ret);
    return ret;
  }

  constexpr BigInt& operator&=(const BigInt& other) {
    And(*this, other, *this);
    return *this;
  }

  constexpr BigInt operator|(const BigInt& other) const {
    BigInt ret;
    Or(*this, other, ret);
    return ret;
  }

  constexpr BigInt& operator|=(const BigInt& other) {
    Or(*this, other, *this);
    return *this;
  }

  constexpr BigInt operator/(const BigInt& other) const {
    return Div(*this, other).quotient;
  }

  constexpr BigInt operator%(const BigInt& other) const {
    return Div(*this, other).remainder;
  }

  constexpr uint64_t& operator[](size_t i) {
    DCHECK_LT(i, N);
    return limbs_[i];
  }
  constexpr const uint64_t& operator[](size_t i) const {
    DCHECK_LT(i, N);
    return limbs_[i];
  }

  constexpr bool operator==(const BigInt& other) const {
    for (size_t i = 0; i < N; ++i) {
      if (limbs_[i] != other.limbs_[i]) return false;
    }
    return true;
  }

  constexpr bool operator!=(const BigInt& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const BigInt& other) const {
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (limbs_[i] == other.limbs_[i]) continue;
      return limbs_[i] < other.limbs_[i];
    }
    return false;
  }

  constexpr bool operator>(const BigInt& other) const {
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (limbs_[i] == other.limbs_[i]) continue;
      return limbs_[i] > other.limbs_[i];
    }
    return false;
  }

  constexpr bool operator<=(const BigInt& other) const {
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (limbs_[i] == other.limbs_[i]) continue;
      return limbs_[i] < other.limbs_[i];
    }
    return true;
  }

  constexpr bool operator>=(const BigInt& other) const {
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (limbs_[i] == other.limbs_[i]) continue;
      return limbs_[i] > other.limbs_[i];
    }
    return true;
  }

  std::string ToString() const { return internal::LimbsToString(limbs_, N); }
  std::string ToHexString(bool pad_zero = false) const {
    return internal::LimbsToHexString(limbs_, N, pad_zero);
  }

  // Converts the BigInt to a bit array in little-endian.
  template <size_t BitNums = kBitWidth>
  std::bitset<BitNums> ToBitsLE() const {
    std::bitset<BitNums> ret;
    size_t bit_w_idx = 0;
    for (size_t i = 0; i < BitNums; ++i) {
      size_t limb_idx = i / kLimbBitWidth;
      size_t bit_r_idx = i % kLimbBitWidth;
      bool bit = (limbs_[limb_idx] & (uint64_t{1} << bit_r_idx)) >> bit_r_idx;
      ret.set(bit_w_idx++, bit);
    }
    return ret;
  }

  // Converts the BigInt to a bit array in big-endian.
  template <size_t BitNums = kBitWidth>
  std::bitset<BitNums> ToBitsBE() const {
    std::bitset<BitNums> ret;
    size_t bit_w_idx = 0;
    for (size_t i = BitNums - 1; i != SIZE_MAX; --i) {
      size_t limb_idx = i / kLimbBitWidth;
      size_t bit_r_idx = i % kLimbBitWidth;
      bool bit = (limbs_[limb_idx] & (uint64_t{1} << bit_r_idx)) >> bit_r_idx;
      ret.set(bit_w_idx++, bit);
    }
    return ret;
  }

  // Converts the BigInt to a byte array in little-endian order. This method
  // processes the limbs of the BigInt, extracts individual bytes, and sets them
  // in the resulting array.
  std::array<uint8_t, kByteWidth> ToBytesLE() const {
    std::array<uint8_t, kByteWidth> ret;
    auto it = ret.begin();
    for (size_t i = 0; i < kByteWidth; ++i) {
      size_t limb_idx = i / kLimbByteWidth;
      uint64_t limb = limbs_[limb_idx];
      size_t byte_r_idx = i % kLimbByteWidth;
      *(it++) = reinterpret_cast<uint8_t*>(&limb)[byte_r_idx];
    }
    return ret;
  }

  // Converts the BigInt to a byte array in big-endian order. This method
  // processes the limbs of the BigInt, extracts individual bytes, and sets them
  // in the resulting array.
  std::array<uint8_t, kByteWidth> ToBytesBE() const {
    std::array<uint8_t, kByteWidth> ret;
    auto it = ret.begin();
    for (size_t i = kByteWidth - 1; i != SIZE_MAX; --i) {
      size_t limb_idx = i / kLimbByteWidth;
      uint64_t limb = limbs_[limb_idx];
      size_t byte_r_idx = i % kLimbByteWidth;
      *(it++) = reinterpret_cast<uint8_t*>(&limb)[byte_r_idx];
    }
    return ret;
  }

  template <size_t N2>
  BigInt<N2> Truncate() const {
    static_assert(
        N > N2, "Destination BigInt size N2 must be less than source size N.");
    BigInt<N2> ret;
    std::copy_n(limbs_, N2, ret.limbs_);
    return ret;
  }

  constexpr static uint64_t Add(const BigInt& a, const BigInt& b, BigInt& c) {
    internal::AddResult<uint64_t> add_result = {};
    for (size_t i = 0; i < N; ++i) {
      add_result = internal::AddWithCarry(a[i], b[i], add_result.carry);
      c[i] = add_result.value;
    }
    return add_result.carry;
  }

  constexpr static uint64_t Sub(const BigInt& a, const BigInt& b, BigInt& c) {
    internal::SubResult<uint64_t> sub_result = {};
    for (size_t i = 0; i < N; ++i) {
      sub_result = internal::SubWithBorrow(a[i], b[i], sub_result.borrow);
      c[i] = sub_result.value;
    }
    return sub_result.borrow;
  }

  constexpr static internal::MulResult<BigInt> Mul(const BigInt& a,
                                                   const BigInt& b) {
    internal::MulResult<BigInt> ret = {};
    internal::MulResult<uint64_t> mul_result = {};
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        uint64_t& limb = (i + j) >= N ? ret.hi[(i + j) - N] : ret.lo[i + j];
        mul_result = internal::MulAddWithCarry(limb, a[i], b[j], mul_result.hi);
        limb = mul_result.lo;
      }
      ret.hi[i] = mul_result.hi;
      mul_result.hi = 0;
    }
    return ret;
  }

  constexpr static uint64_t ShiftLeft(const BigInt& a, BigInt& b,
                                      uint64_t shift) {
    if (shift == 0) {
      b = a;
      return 0;
    }
    if (shift >= kBitWidth) {
      b = Zero();
      return 0;
    }

    const size_t limb_shift = shift / kLimbBitWidth;
    const size_t bit_shift = shift % kLimbBitWidth;

    if (bit_shift == 0) {
      // Whole limb shift only
      for (size_t i = N - 1; i >= limb_shift; --i) {
        b[i] = a[i - limb_shift];
      }
      for (size_t i = 0; i < limb_shift; ++i) {
        b[i] = 0;
      }
      return 0;
    }

    // Combined limb and bit shift
    uint64_t carry = 0;
    for (size_t i = 0; i < N; ++i) {
      if (i < limb_shift) {
        b[i] = 0;
      } else {
        size_t src_idx = i - limb_shift;
        uint64_t src = a[src_idx];
        b[i] = (src << bit_shift) | carry;
        carry = src >> (kLimbBitWidth - bit_shift);
      }
    }
    return carry;
  }

  constexpr static uint64_t ShiftRight(const BigInt& a, BigInt& b,
                                       uint64_t shift) {
    if (shift == 0) {
      b = a;
      return 0;
    }
    if (shift >= kBitWidth) {
      b = Zero();
      return 0;
    }

    const size_t limb_shift = shift / kLimbBitWidth;
    const size_t bit_shift = shift % kLimbBitWidth;

    if (bit_shift == 0) {
      // Whole limb shift only
      for (size_t i = 0; i < N - limb_shift; ++i) {
        b[i] = a[i + limb_shift];
      }
      for (size_t i = N - limb_shift; i < N; ++i) {
        b[i] = 0;
      }
      return 0;
    }

    // Combined limb and bit shift
    uint64_t borrow = 0;
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (i + limb_shift >= N) {
        b[i] = 0;
      } else {
        size_t src_idx = i + limb_shift;
        uint64_t src = a[src_idx];
        b[i] = (src >> bit_shift) | borrow;
        borrow = src << (kLimbBitWidth - bit_shift);
      }
    }
    return borrow;
  }

  constexpr static void Xor(const BigInt& a, const BigInt& b, BigInt& c) {
    for (size_t i = 0; i < N; ++i) {
      c[i] = a[i] ^ b[i];
    }
  }

  constexpr static void And(const BigInt& a, const BigInt& b, BigInt& c) {
    for (size_t i = 0; i < N; ++i) {
      c[i] = a[i] & b[i];
    }
  }

  constexpr static void Or(const BigInt& a, const BigInt& b, BigInt& c) {
    for (size_t i = 0; i < N; ++i) {
      c[i] = a[i] | b[i];
    }
  }

  // Multi-precision division using Knuth's Algorithm D.
  //
  // Divides a by b, returning both quotient and remainder.
  // Uses __uint128_t for 2-digit estimation and multiply-subtract.
  // Complexity: O(m * n) in 64-bit operations where m, n are the number of
  // significant limbs in a and b respectively. This is ~10-15x faster than
  // the previous bit-by-bit approach for typical 256-bit / 256-bit division.
  constexpr static internal::DivResult<BigInt> Div(const BigInt<N>& a,
                                                   const BigInt<N>& b) {
    bool is_zero = b.IsZero();
    DCHECK(!is_zero);
    internal::DivResult<BigInt> ret;
    if (is_zero) return ret;

    // Find significant limbs.
    size_t m = N;
    while (m > 0 && a.limbs_[m - 1] == 0) --m;
    if (m == 0) return ret;  // a == 0.

    size_t n = N;
    while (n > 1 && b.limbs_[n - 1] == 0) --n;

    // If dividend < divisor, remainder = dividend.
    if (m < n || (m == n && a < b)) {
      ret.remainder = a;
      return ret;
    }

    // Single-limb divisor: simple long division with __uint128_t.
    if (n == 1) {
      uint64_t rem = 0;
      for (size_t i = m; i-- > 0;) {
        __uint128_t cur = (static_cast<__uint128_t>(rem) << 64) | a.limbs_[i];
        ret.quotient.limbs_[i] = static_cast<uint64_t>(cur / b.limbs_[0]);
        rem = static_cast<uint64_t>(cur % b.limbs_[0]);
      }
      ret.remainder.limbs_[0] = rem;
      return ret;
    }

    // Knuth Algorithm D for multi-limb divisor.
    // D1. Normalize: shift so MSB of b[n-1] is set.
    int s = __builtin_clzll(b.limbs_[n - 1]);

    // Normalized copies: un has m+1 limbs, vn has n limbs.
    uint64_t un[N + 1] = {};
    uint64_t vn[N] = {};

    if (s > 0) {
      for (size_t i = n - 1; i > 0; --i)
        vn[i] = (b.limbs_[i] << s) | (b.limbs_[i - 1] >> (64 - s));
      vn[0] = b.limbs_[0] << s;

      un[m] = a.limbs_[m - 1] >> (64 - s);
      for (size_t i = m - 1; i > 0; --i)
        un[i] = (a.limbs_[i] << s) | (a.limbs_[i - 1] >> (64 - s));
      un[0] = a.limbs_[0] << s;
    } else {
      for (size_t i = 0; i < n; ++i) vn[i] = b.limbs_[i];
      for (size_t i = 0; i < m; ++i) un[i] = a.limbs_[i];
      un[m] = 0;
    }

    // D2-D7. Main loop: compute one quotient digit per iteration.
    for (size_t j1 = m - n + 1; j1 > 0; --j1) {
      size_t j = j1 - 1;

      // D3. Estimate quotient digit q̂.
      __uint128_t num =
          (static_cast<__uint128_t>(un[j + n]) << 64) | un[j + n - 1];
      uint64_t qhat;

      if (un[j + n] >= vn[n - 1]) {
        qhat = UINT64_MAX;
      } else {
        qhat = static_cast<uint64_t>(num / vn[n - 1]);
        uint64_t rhat = static_cast<uint64_t>(num % vn[n - 1]);

        // Refine q̂ using second divisor digit.
        while (static_cast<__uint128_t>(qhat) * vn[n - 2] >
               ((static_cast<__uint128_t>(rhat) << 64) | un[j + n - 2])) {
          --qhat;
          uint64_t old_rhat = rhat;
          rhat += vn[n - 1];
          if (rhat < old_rhat) break;  // Overflow → rhat >= 2⁶⁴, done.
        }
      }

      // D4. Multiply and subtract: un[j..j+n] -= q̂ * vn[0..n-1].
      uint64_t mul_carry = 0;
      int64_t sub_borrow = 0;
      for (size_t i = 0; i < n; ++i) {
        __uint128_t prod = static_cast<__uint128_t>(qhat) * vn[i] + mul_carry;
        mul_carry = static_cast<uint64_t>(prod >> 64);
        uint64_t prod_lo = static_cast<uint64_t>(prod);

        __int128_t diff =
            static_cast<__int128_t>(un[j + i]) - prod_lo + sub_borrow;
        un[j + i] = static_cast<uint64_t>(diff);
        sub_borrow = static_cast<int64_t>(diff >> 64);
      }
      __int128_t diff =
          static_cast<__int128_t>(un[j + n]) - mul_carry + sub_borrow;
      un[j + n] = static_cast<uint64_t>(diff);

      // D5/D6. If result is negative, add back and decrement q̂.
      if (static_cast<int64_t>(diff >> 64) < 0) {
        --qhat;
        __uint128_t carry = 0;
        for (size_t i = 0; i < n; ++i) {
          __uint128_t sum = static_cast<__uint128_t>(un[j + i]) + vn[i] + carry;
          un[j + i] = static_cast<uint64_t>(sum);
          carry = sum >> 64;
        }
        un[j + n] += static_cast<uint64_t>(carry);
      }

      if (j < N) ret.quotient.limbs_[j] = qhat;
    }

    // D8. Unnormalize remainder.
    if (s > 0) {
      for (size_t i = 0; i < n - 1; ++i)
        ret.remainder.limbs_[i] = (un[i] >> s) | (un[i + 1] << (64 - s));
      ret.remainder.limbs_[n - 1] = un[n - 1] >> s;
    } else {
      for (size_t i = 0; i < n; ++i) ret.remainder.limbs_[i] = un[i];
    }

    return ret;
  }

 private:
  template <size_t N2>
  friend class BigInt;

  uint64_t limbs_[N];
};

template <size_t N>
std::ostream& operator<<(std::ostream& os, const BigInt<N>& big_int) {
  return os << big_int.ToString();
}

template <size_t N>
class BitTraits<BigInt<N>> {
 public:
  constexpr static size_t GetNumBits(const BigInt<N>& _) { return N * 64; }

  constexpr static bool TestBit(const BigInt<N>& bigint, size_t index) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return false;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = uint64_t{1} << bit_index;
    return (bigint[limb_index] & bit_index_value) == bit_index_value;
  }

  constexpr static void SetBit(BigInt<N>& bigint, size_t index,
                               bool bit_value) {
    size_t limb_index = index >> 6;
    if (limb_index >= N) return;
    size_t bit_index = index & 63;
    uint64_t bit_index_value = uint64_t{1} << bit_index;
    if (bit_value) {
      bigint[limb_index] |= bit_index_value;
    } else {
      bigint[limb_index] &= ~bit_index_value;
    }
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_BIG_INT_H_
