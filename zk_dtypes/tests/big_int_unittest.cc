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

#include "zk_dtypes/include/big_int.h"

#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zk_dtypes {
namespace {

using ::absl_testing::StatusIs;

TEST(BigIntTest, Zero) {
  BigInt<2> big_int = BigInt<2>::Zero();
  EXPECT_TRUE(big_int.IsZero());
  EXPECT_FALSE(big_int.IsOne());
}

TEST(BigIntTest, One) {
  BigInt<2> big_int = BigInt<2>::One();
  EXPECT_FALSE(big_int.IsZero());
  EXPECT_TRUE(big_int.IsOne());
}

TEST(BigIntTest, DecString) {
  // 1 << 65
  absl::StatusOr<BigInt<2>> big_int =
      BigInt<2>::FromDecString("36893488147419103232");
  ASSERT_TRUE(big_int.ok());
  EXPECT_EQ(big_int->ToString(), "36893488147419103232");

  // Invalid input
  EXPECT_THAT(BigInt<2>::FromDecString("x"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 1 << 128
  EXPECT_THAT(
      BigInt<2>::FromDecString("340282366920938463463374607431768211456"),
      StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(BigIntTest, HexString) {
  {
    // 1 << 65
    absl::StatusOr<BigInt<2>> big_int =
        BigInt<2>::FromHexString("20000000000000000");
    ASSERT_TRUE(big_int.ok());
    EXPECT_EQ(big_int->ToHexString(), "0x20000000000000000");
  }
  {
    // 1 << 65
    absl::StatusOr<BigInt<2>> big_int =
        BigInt<2>::FromHexString("0x20000000000000000");
    ASSERT_TRUE(big_int.ok());
    EXPECT_EQ(big_int->ToHexString(), "0x20000000000000000");
  }

  // Invalid input
  EXPECT_THAT(BigInt<2>::FromDecString("g"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 1 << 128
  EXPECT_THAT(BigInt<2>::FromHexString("0x100000000000000000000000000000000"),
              StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(BigIntTest, Comparison) {
  // 1 << 65
  BigInt<2> big_int = *BigInt<2>::FromHexString("20000000000000000");
  BigInt<2> big_int2 = *BigInt<2>::FromHexString("20000000000000001");
  EXPECT_TRUE(big_int == big_int);
  EXPECT_TRUE(big_int != big_int2);
  EXPECT_TRUE(big_int < big_int2);
  EXPECT_TRUE(big_int <= big_int2);
  EXPECT_TRUE(big_int2 > big_int);
  EXPECT_TRUE(big_int2 >= big_int);
}

TEST(BigIntTest, Operations) {
  BigInt<2> a =
      *BigInt<2>::FromDecString("123456789012345678909876543211235312");
  BigInt<2> b =
      *BigInt<2>::FromDecString("734581237591230158128731489729873983");

  EXPECT_EQ(a + b,
            *BigInt<2>::FromDecString("858038026603575837038608032941109295"));
  EXPECT_EQ(a << 1,
            *BigInt<2>::FromDecString("246913578024691357819753086422470624"));
  EXPECT_EQ(a >> 1,
            *BigInt<2>::FromDecString("61728394506172839454938271605617656"));
  EXPECT_EQ(a - b, *BigInt<2>::FromDecString(
                       "339671242472359578984155752485249572785"));
  EXPECT_EQ(b - a,
            *BigInt<2>::FromDecString("611124448578884479218854946518638671"));
  EXPECT_EQ(a * b, *BigInt<2>::FromDecString(
                       "335394729415762779748307316131549975568"));
  BigInt<2> divisor(123456789);
  EXPECT_EQ(a / divisor,
            *BigInt<2>::FromDecString("1000000000100000000080000000"));
  EXPECT_EQ(a % divisor, BigInt<2>(91235312));
  EXPECT_EQ(
      -a, *BigInt<2>::FromDecString("340158910131926117784464730888556976144"));
}

TEST(BigIntTest, MultiLimbDivision) {
  // 4-limb / 2-limb divisor.
  {
    BigInt<4> a = *BigInt<4>::FromHexString(
        "FFFFFFFFFFFFFFFF0000000000000001"
        "AAAAAAAAAAAAAAAA5555555555555555");
    BigInt<4> b = *BigInt<4>::FromHexString("DEADBEEFCAFEBABE1234567890ABCDEF");
    EXPECT_EQ(a / b,
              *BigInt<4>::FromHexString("1264EB564B347462ADE70B59B51BD2005"));
    EXPECT_EQ(a % b,
              *BigInt<4>::FromHexString("7093DD4880733E6F0787BD305FC96FAA"));
  }
  // 4-limb / 4-limb, near-equal (secp256k1 p / n).
  {
    BigInt<4> a = *BigInt<4>::FromHexString(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    BigInt<4> b = *BigInt<4>::FromHexString(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    EXPECT_EQ(a / b, BigInt<4>::One());
    EXPECT_EQ(a % b,
              *BigInt<4>::FromHexString("14551231950B75FC4402DA1722FC9BAEE"));
  }
  // Dividend < divisor → quotient = 0, remainder = dividend.
  {
    BigInt<4> a = *BigInt<4>::FromHexString(
        "0000000000000001FFFFFFFFFFFFFFFF"
        "0000000000000000AAAAAAAAAAAAAAAA");
    BigInt<4> b = *BigInt<4>::FromHexString(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    EXPECT_EQ(a / b, BigInt<4>::Zero());
    EXPECT_EQ(a % b, a);
  }
  // Equal values → quotient = 1, remainder = 0.
  {
    BigInt<4> a = *BigInt<4>::FromHexString(
        "DEADBEEFCAFEBABE1234567890ABCDEF0011223344556677AABBCCDDEEFF0011");
    EXPECT_EQ(a / a, BigInt<4>::One());
    EXPECT_EQ(a % a, BigInt<4>::Zero());
  }
  // 4-limb / 3-limb divisor.
  {
    BigInt<4> a = *BigInt<4>::FromHexString(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000");
    BigInt<4> b = *BigInt<4>::FromHexString(
        "DEADBEEFCAFEBABEABCDEF01234567891111111111111111");
    EXPECT_EQ(a / b, *BigInt<4>::FromHexString("1264EB564B347462B"));
    EXPECT_EQ(a % b, *BigInt<4>::FromHexString(
                         "323957D04BDC3C54F167141D028CEE1B57E31D28D8C07C25"));
  }
}

TEST(BigIntTest, ShiftLeftExtended) {
  // Test with BigInt<2> (128-bit)
  BigInt<2> a = *BigInt<2>::FromHexString("123456789ABCDEF0FEDCBA9876543210");

  // Shift by 0 - should be identity
  EXPECT_EQ(a << 0, a);

  // Shift by less than 64
  EXPECT_EQ(a << 4,
            *BigInt<2>::FromHexString("23456789ABCDEF0FEDCBA98765432100"));

  // Shift by exactly 64 (whole limb shift)
  BigInt<2> b({0x1234567890ABCDEFull, 0ull});
  EXPECT_EQ(b << 64, BigInt<2>({0ull, 0x1234567890ABCDEFull}));

  // Shift by 65 (one limb + 1 bit)
  EXPECT_EQ(b << 65, BigInt<2>({0ull, 0x2468ACF121579BDEull}));

  // Shift by 70 (one limb + 6 bits)
  BigInt<2> c({0xFFFFFFFFFFFFFFFFull, 0ull});
  EXPECT_EQ(c << 70, BigInt<2>({0ull, 0xFFFFFFFFFFFFFFC0ull}));

  // Shift by >= kBitWidth (128) should result in zero
  EXPECT_EQ(a << 128, BigInt<2>::Zero());

  // Test with BigInt<4> (256-bit)
  BigInt<4> d({0x1111111111111111ull, 0x2222222222222222ull,
               0x3333333333333333ull, 0x4444444444444444ull});

  // Shift by 128 (two whole limbs)
  EXPECT_EQ(d << 128, BigInt<4>({0ull, 0ull, 0x1111111111111111ull,
                                 0x2222222222222222ull}));

  // Shift by 130 (two limbs + 2 bits)
  EXPECT_EQ(d << 130, BigInt<4>({0ull, 0ull, 0x4444444444444444ull,
                                 0x8888888888888888ull}));
}

TEST(BigIntTest, ShiftRightExtended) {
  // Test with BigInt<2> (128-bit)
  BigInt<2> a = *BigInt<2>::FromHexString("123456789ABCDEF0FEDCBA9876543210");

  // Shift by 0 - should be identity
  EXPECT_EQ(a >> 0, a);

  // Shift by less than 64
  EXPECT_EQ(a >> 4,
            *BigInt<2>::FromHexString("0123456789ABCDEF0FEDCBA987654321"));

  // Shift by exactly 64 (whole limb shift)
  BigInt<2> b({0ull, 0x1234567890ABCDEFull});
  EXPECT_EQ(b >> 64, BigInt<2>({0x1234567890ABCDEFull, 0ull}));

  // Shift by 65 (one limb + 1 bit)
  // 0x1234567890ABCDEF >> 1 = 0x091A2B3C4855E6F7
  EXPECT_EQ(b >> 65, BigInt<2>({0x091A2B3C4855E6F7ull, 0ull}));

  // Shift by 70 (one limb + 6 bits)
  BigInt<2> c({0ull, 0xFFFFFFFFFFFFFFFFull});
  EXPECT_EQ(c >> 70, BigInt<2>({0x03FFFFFFFFFFFFFFull, 0ull}));

  // Shift by >= kBitWidth (128) should result in zero
  EXPECT_EQ(a >> 128, BigInt<2>::Zero());

  // Test with BigInt<4> (256-bit)
  BigInt<4> d({0x1111111111111111ull, 0x2222222222222222ull,
               0x3333333333333333ull, 0x4444444444444444ull});

  // Shift by 128 (two whole limbs)
  EXPECT_EQ(d >> 128, BigInt<4>({0x3333333333333333ull, 0x4444444444444444ull,
                                 0ull, 0ull}));

  // Shift by 130 (two limbs + 2 bits)
  EXPECT_EQ(d >> 130, BigInt<4>({0x0CCCCCCCCCCCCCCCull, 0x1111111111111111ull,
                                 0ull, 0ull}));
}

TEST(BigIntTest, ShiftRoundTrip) {
  // Verify that shift left followed by shift right restores original
  // Use a small value that won't overflow when shifted left by 70 on 128-bit
  BigInt<2> a({0x0000000012345678ull, 0ull});

  // Small shift round trip
  EXPECT_EQ((a << 10) >> 10, a);

  // Large shift round trip (shift by 64)
  EXPECT_EQ((a << 64) >> 64, a);

  // Shift by 70 - should preserve value since it fits in 128 bits
  EXPECT_EQ((a << 70) >> 70, a);
}

TEST(BigIntTest, BitsLEConversion) {
  // clang-format off
  std::bitset<255> input("011101111110011110110101010100110010011011110111011101000111010111110011000100011000011100111011011100111101100101100111001101011010000011111110000010011110011110001011111101111001100001100000111010000101111101010010101011110101110101011101011001100110000");
  BigInt<4> big_int = BigInt<4>::FromBitsLE(input);
  ASSERT_EQ(big_int, *BigInt<4>::FromDecString("27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  // clang-format on
  EXPECT_EQ(big_int.ToBitsLE<255>(), input);
}

TEST(BigIntTest, BitsBEConversion) {
  // clang-format off
  std::bitset<255> input("0000110011001101011101010111010111101010100101011111010000101110000011000011001111011111101000111100111100100000111111100000101101011001110011010011011110011101101110011100001100010001100111110101110001011101110111101100100110010101010110111100111111011100110000");
  BigInt<4> big_int = BigInt<4>::FromBitsBE(input);
  ASSERT_EQ(big_int, *BigInt<4>::FromDecString("27117311055620256798560880810000042840428971800021819916023577129547249660720"));
  // clang-format on
  EXPECT_EQ(big_int.ToBitsBE<255>(), input);
}

TEST(BigIntTest, Truncate) {
  BigInt<4> input = BigInt<4>::Random();
  BigInt<2> truncated = input.Truncate<2>();
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(truncated[i], input[i]);
  }
}

namespace {

template <typename Container>
class BigIntConversionTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    // clang-format off
    expected_ = *BigInt<4>::FromDecString("27117311055620256798560880810000042840428971800021819916023577129547249660720");
    // clang-format on
  }

 protected:
  static BigInt<4> expected_;

  static constexpr uint8_t kInputLE[32] = {
      48,  179, 174, 174, 87,  169, 47,  116, 48,  204, 251,
      197, 243, 4,   127, 208, 154, 179, 236, 185, 157, 195,
      136, 249, 58,  186, 123, 147, 169, 218, 243, 59};

  static constexpr uint8_t kInputBE[32] = {
      59,  243, 218, 169, 147, 123, 186, 58,  249, 136, 195,
      157, 185, 236, 179, 154, 208, 127, 4,   243, 197, 251,
      204, 48,  116, 47,  169, 87,  174, 174, 179, 48};
};

template <typename Container>
BigInt<4> BigIntConversionTest<Container>::expected_;
template <typename Container>
constexpr uint8_t BigIntConversionTest<Container>::kInputLE[32];
template <typename Container>
constexpr uint8_t BigIntConversionTest<Container>::kInputBE[32];

}  // namespace

using ContainerTypes =
    testing::Types<std::vector<uint8_t>, std::array<uint8_t, 32>,
                   absl::InlinedVector<uint8_t, 32>, absl::Span<const uint8_t>>;
TYPED_TEST_SUITE(BigIntConversionTest, ContainerTypes);

TYPED_TEST(BigIntConversionTest, BytesLEConversion) {
  using Container = TypeParam;

  Container expected_input;

  if constexpr (std::is_same_v<Container, std::vector<uint8_t>> ||
                std::is_same_v<Container, absl::InlinedVector<uint8_t, 32>>) {
    expected_input =
        Container(std::begin(this->kInputLE), std::end(this->kInputLE));
  } else if constexpr (std::is_same_v<Container, std::array<uint8_t, 32>>) {
    std::copy(std::begin(this->kInputLE), std::end(this->kInputLE),
              expected_input.begin());
  } else if constexpr (std::is_same_v<Container, absl::Span<const uint8_t>>) {
    expected_input = Container(this->kInputLE, sizeof(this->kInputLE));
  }

  BigInt<4> actual = BigInt<4>::FromBytesLE(expected_input);
  ASSERT_EQ(actual, this->expected_);

  std::array<uint8_t, 32> actual_input = actual.ToBytesLE();
  EXPECT_TRUE(std::equal(actual_input.begin(), actual_input.end(),
                         expected_input.begin()));
}

TYPED_TEST(BigIntConversionTest, BytesBEConversion) {
  using Container = TypeParam;

  Container expected_input;

  if constexpr (std::is_same_v<Container, std::vector<uint8_t>> ||
                std::is_same_v<Container, absl::InlinedVector<uint8_t, 32>>) {
    expected_input =
        Container(std::begin(this->kInputBE), std::end(this->kInputBE));
  } else if constexpr (std::is_same_v<Container, std::array<uint8_t, 32>>) {
    std::copy(std::begin(this->kInputBE), std::end(this->kInputBE),
              expected_input.begin());
  } else if constexpr (std::is_same_v<Container, absl::Span<const uint8_t>>) {
    expected_input = Container(this->kInputBE, sizeof(this->kInputBE));
  }

  BigInt<4> actual = BigInt<4>::FromBytesBE(expected_input);
  ASSERT_EQ(actual, this->expected_);

  std::array<uint8_t, 32> actual_input = actual.ToBytesBE();
  EXPECT_TRUE(std::equal(actual_input.begin(), actual_input.end(),
                         expected_input.begin()));
}

}  // namespace
}  // namespace zk_dtypes
