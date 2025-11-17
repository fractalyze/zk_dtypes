#ifndef ZK_DTYPES_INCLUDE_BIG_INT_H_
#define ZK_DTYPES_INCLUDE_BIG_INT_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <bitset>
#include <initializer_list>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/base/casts.h"
#include "absl/base/optimization.h"
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

  constexpr absl::StatusOr<BigInt> operator/(const BigInt& other) const {
    absl::StatusOr<internal::DivResult<BigInt>> div_result = Div(*this, other);
    if (!div_result.ok()) return div_result.status();
    return div_result.value().quotient;
  }

  constexpr absl::StatusOr<BigInt> operator%(const BigInt& other) const {
    absl::StatusOr<internal::DivResult<BigInt>> div_result = Div(*this, other);
    if (!div_result.ok()) return div_result.status();
    return div_result.value().remainder;
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

  constexpr static uint64_t Add(const BigInt& a, const BigInt& b, BigInt& c) {
    internal::AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      add_result = internal::AddWithCarry(a[i], b[i], add_result.carry);
      c[i] = add_result.value;
    }
    return add_result.carry;
  }

  constexpr static uint64_t Sub(const BigInt& a, const BigInt& b, BigInt& c) {
    internal::SubResult<uint64_t> sub_result;
    for (size_t i = 0; i < N; ++i) {
      sub_result = internal::SubWithBorrow(a[i], b[i], sub_result.borrow);
      c[i] = sub_result.value;
    }
    return sub_result.borrow;
  }

  constexpr static internal::MulResult<BigInt> Mul(const BigInt& a,
                                                   const BigInt& b) {
    internal::MulResult<BigInt> ret;
    internal::MulResult<uint64_t> mul_result;
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
    CHECK_LT(shift, 64);
    uint64_t carry = 0;
    for (size_t i = 0; i < N; ++i) {
      uint64_t temp = a[i] >> (64 - shift);
      b[i] = a[i] << shift;
      b[i] |= carry;
      carry = temp;
    }
    return carry;
  }

  constexpr static uint64_t ShiftRight(const BigInt& a, BigInt& b,
                                       uint64_t shift) {
    CHECK_LT(shift, 64);
    uint64_t borrow = 0;
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      uint64_t temp = a[i] << (64 - shift);
      b[i] = a[i] >> shift;
      b[i] |= borrow;
      borrow = temp;
    }
    return borrow;
  }

  constexpr static absl::StatusOr<internal::DivResult<BigInt>> Div(
      const BigInt<N>& a, const BigInt<N>& b) {
    if (b.IsZero()) return absl::InvalidArgumentError("Division by zero");

    // Stupid slow base-2 long division taken from
    // https://en.wikipedia.org/wiki/Division_algorithm
    internal::DivResult<BigInt> ret;
    size_t bits = BitTraits<BigInt>::GetNumBits(a);
    uint64_t& smallest_bit = ret.remainder[0];
    for (size_t i = bits - 1; i != SIZE_MAX; --i) {
      uint64_t carry = ShiftLeft(ret.remainder, ret.remainder, 1);
      smallest_bit |= BitTraits<BigInt>::TestBit(a, i);
      if (ret.remainder >= b || carry) {
        uint64_t borrow = Sub(ret.remainder, b, ret.remainder);
        if (ABSL_PREDICT_FALSE(borrow != carry))
          return absl::InternalError("Division error: borrow/carry mismatch");
        BitTraits<BigInt>::SetBit(ret.quotient, i, 1);
      }
    }
    return ret;
  }

 private:
  uint64_t limbs_[N];
};

template <size_t N>
std::ostream& operator<<(std::ostream& os, const BigInt<N>& big_int) {
  return os << big_int.ToHexString(true);
}

template <size_t N>
class BitTraits<BigInt<N>> {
 public:
  constexpr static bool kIsDynamic = false;

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
