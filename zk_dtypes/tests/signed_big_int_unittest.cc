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

#include "zk_dtypes/include/signed_big_int.h"

#include "absl/status/status_matchers.h"
#include "gtest/gtest.h"

namespace zk_dtypes {
namespace {

// --- Constructor tests ---

TEST(SignedBigIntTest, DefaultIsZero) {
  SignedBigInt<2> v;
  EXPECT_TRUE(v.IsZero());
  EXPECT_FALSE(v.IsNegative());
}

TEST(SignedBigIntTest, PositiveValue) {
  SignedBigInt<2> v(42);
  EXPECT_EQ(v.ToString(), "42");
  EXPECT_FALSE(v.IsNegative());
  EXPECT_TRUE(v.IsStrictlyPositive());
}

TEST(SignedBigIntTest, NegativeValue) {
  SignedBigInt<2> v(-1);
  EXPECT_TRUE(v.IsNegative());
  EXPECT_EQ(v.ToString(), "-1");

  // All limbs should be all-1s for -1 in two's complement.
  EXPECT_EQ(v[0], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(v[1], std::numeric_limits<uint64_t>::max());
}

TEST(SignedBigIntTest, NegativeValueSmall) {
  SignedBigInt<2> v(-5);
  EXPECT_TRUE(v.IsNegative());
  EXPECT_EQ(v.ToString(), "-5");
}

TEST(SignedBigIntTest, FromBigInt) {
  BigInt<2> big(100);
  SignedBigInt<2> v(big);
  EXPECT_EQ(v.ToString(), "100");
  EXPECT_FALSE(v.IsNegative());
}

// --- Sign predicates ---

TEST(SignedBigIntTest, SignPredicatesPositive) {
  SignedBigInt<2> v(10);
  EXPECT_FALSE(v.IsNegative());
  EXPECT_TRUE(v.IsNonNegative());
  EXPECT_TRUE(v.IsStrictlyPositive());
}

TEST(SignedBigIntTest, SignPredicatesZero) {
  SignedBigInt<2> v(0);
  EXPECT_FALSE(v.IsNegative());
  EXPECT_TRUE(v.IsNonNegative());
  EXPECT_FALSE(v.IsStrictlyPositive());
}

TEST(SignedBigIntTest, SignPredicatesNegative) {
  SignedBigInt<2> v(-3);
  EXPECT_TRUE(v.IsNegative());
  EXPECT_FALSE(v.IsNonNegative());
  EXPECT_FALSE(v.IsStrictlyPositive());
}

// --- Min / Max ---

TEST(SignedBigIntTest, Min) {
  auto min = SignedBigInt<2>::Min();
  EXPECT_TRUE(min.IsNegative());
  // MSB set, all other bits clear.
  EXPECT_EQ(min[0], 0);
  EXPECT_EQ(min[1], uint64_t{1} << 63);
}

TEST(SignedBigIntTest, Max) {
  auto max = SignedBigInt<2>::Max();
  EXPECT_FALSE(max.IsNegative());
  // All bits set except MSB.
  EXPECT_EQ(max[0], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(max[1], std::numeric_limits<uint64_t>::max() >> 1);
}

TEST(SignedBigIntTest, MinLessThanMax) {
  EXPECT_TRUE(SignedBigInt<2>::Min() < SignedBigInt<2>::Max());
}

// --- Signed comparison ---

TEST(SignedBigIntTest, CompareNegativeVsPositive) {
  SignedBigInt<2> neg(-1);
  SignedBigInt<2> pos(1);
  EXPECT_TRUE(neg < pos);
  EXPECT_TRUE(pos > neg);
  EXPECT_FALSE(neg > pos);
  EXPECT_FALSE(pos < neg);
}

TEST(SignedBigIntTest, CompareNegativeVsZero) {
  SignedBigInt<2> neg(-1);
  SignedBigInt<2> zero(0);
  EXPECT_TRUE(neg < zero);
  EXPECT_TRUE(zero > neg);
}

TEST(SignedBigIntTest, CompareTwoNegatives) {
  SignedBigInt<2> a(-2);
  SignedBigInt<2> b(-1);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);
}

TEST(SignedBigIntTest, CompareEqual) {
  SignedBigInt<2> a(-5);
  SignedBigInt<2> b(-5);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
}

TEST(SignedBigIntTest, CompareTwoPositives) {
  SignedBigInt<2> a(10);
  SignedBigInt<2> b(20);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b >= a);
}

// --- Signed division ---

TEST(SignedBigIntTest, DivPositivePositive) {
  SignedBigInt<2> a(7);
  SignedBigInt<2> b(2);
  EXPECT_EQ((a / b).ToString(), "3");
  EXPECT_EQ((a % b).ToString(), "1");
}

TEST(SignedBigIntTest, DivNegativePositive) {
  SignedBigInt<2> a(-7);
  SignedBigInt<2> b(2);
  // Truncation toward zero: (-7) / 2 = -3
  EXPECT_EQ((a / b).ToString(), "-3");
  // Remainder sign follows dividend: (-7) % 2 = -1
  EXPECT_EQ((a % b).ToString(), "-1");
}

TEST(SignedBigIntTest, DivPositiveNegative) {
  SignedBigInt<2> a(7);
  SignedBigInt<2> b(-2);
  EXPECT_EQ((a / b).ToString(), "-3");
  EXPECT_EQ((a % b).ToString(), "1");
}

TEST(SignedBigIntTest, DivNegativeNegative) {
  SignedBigInt<2> a(-7);
  SignedBigInt<2> b(-2);
  EXPECT_EQ((a / b).ToString(), "3");
  EXPECT_EQ((a % b).ToString(), "-1");
}

TEST(SignedBigIntTest, DivExact) {
  SignedBigInt<2> a(-10);
  SignedBigInt<2> b(5);
  EXPECT_EQ((a / b).ToString(), "-2");
  EXPECT_EQ((a % b).ToString(), "0");
}

// --- Arithmetic right shift ---

TEST(SignedBigIntTest, AShrPositive) {
  SignedBigInt<2> v(4);
  EXPECT_EQ((v >> 1).ToString(), "2");
  EXPECT_EQ((v >> 2).ToString(), "1");
  EXPECT_EQ((v >> 3).ToString(), "0");
}

TEST(SignedBigIntTest, AShrNegative) {
  SignedBigInt<2> v(-4);
  // Arithmetic shift: (-4) >> 1 = -2
  EXPECT_EQ((v >> 1).ToString(), "-2");
  // (-4) >> 2 = -1
  EXPECT_EQ((v >> 2).ToString(), "-1");
}

TEST(SignedBigIntTest, AShrNegativeOne) {
  SignedBigInt<2> v(-1);
  // -1 >> any = -1
  EXPECT_EQ((v >> 1).ToString(), "-1");
  EXPECT_EQ((v >> 64).ToString(), "-1");
}

TEST(SignedBigIntTest, AShrByZero) {
  SignedBigInt<2> v(-42);
  EXPECT_EQ((v >> 0).ToString(), "-42");
}

TEST(SignedBigIntTest, AShrByBitWidth) {
  SignedBigInt<2> pos(100);
  SignedBigInt<2> neg(-100);
  // Shift by kBitWidth or more: positive → 0, negative → -1.
  EXPECT_EQ((pos >> 128).ToString(), "0");
  EXPECT_EQ((neg >> 128).ToString(), "-1");
}

TEST(SignedBigIntTest, AShrAssignment) {
  SignedBigInt<2> v(-8);
  v >>= 2;
  EXPECT_EQ(v.ToString(), "-2");
}

// --- ToString ---

TEST(SignedBigIntTest, ToStringPositive) {
  SignedBigInt<2> v(12345);
  EXPECT_EQ(v.ToString(), "12345");
}

TEST(SignedBigIntTest, ToStringNegative) {
  SignedBigInt<2> v(-12345);
  EXPECT_EQ(v.ToString(), "-12345");
}

TEST(SignedBigIntTest, ToStringZero) {
  SignedBigInt<2> v(0);
  EXPECT_EQ(v.ToString(), "0");
}

TEST(SignedBigIntTest, ToHexStringNegative) {
  SignedBigInt<2> v(-255);
  EXPECT_EQ(v.ToHexString(), "-0xff");
}

// --- FromDecString ---

TEST(SignedBigIntTest, FromDecStringPositive) {
  auto result = SignedBigInt<2>::FromDecString("123");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->ToString(), "123");
}

TEST(SignedBigIntTest, FromDecStringNegative) {
  auto result = SignedBigInt<2>::FromDecString("-123");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->ToString(), "-123");
  EXPECT_TRUE(result->IsNegative());
}

TEST(SignedBigIntTest, FromHexStringNegative) {
  auto result = SignedBigInt<2>::FromHexString("-0xff");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->ToString(), "-255");
  EXPECT_TRUE(result->IsNegative());
}

TEST(SignedBigIntTest, FromDecStringRoundTrip) {
  auto result = SignedBigInt<2>::FromDecString("-9999999999999999999");
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->ToString(), "-9999999999999999999");
}

// --- Inherited arithmetic operations ---

TEST(SignedBigIntTest, Addition) {
  SignedBigInt<2> a(3);
  SignedBigInt<2> b(-5);
  auto result = a + b;
  EXPECT_EQ(result.ToString(), "-2");
}

TEST(SignedBigIntTest, Subtraction) {
  SignedBigInt<2> a(-3);
  SignedBigInt<2> b(4);
  auto result = a - b;
  EXPECT_EQ(result.ToString(), "-7");
}

TEST(SignedBigIntTest, Multiplication) {
  SignedBigInt<2> a(-3);
  SignedBigInt<2> b(4);
  auto result = a * b;
  EXPECT_EQ(result.ToString(), "-12");
}

TEST(SignedBigIntTest, Negation) {
  SignedBigInt<2> v(42);
  EXPECT_EQ((-v).ToString(), "-42");
  EXPECT_EQ((-(-v)).ToString(), "42");
}

TEST(SignedBigIntTest, NegationNegative) {
  SignedBigInt<2> v(-7);
  EXPECT_EQ((-v).ToString(), "7");
}

// --- Return type covariance ---

TEST(SignedBigIntTest, ChainedOpsReturnSignedBigInt) {
  SignedBigInt<2> a(-10);
  SignedBigInt<2> b(3);
  // (a + b) >> 1 should use arithmetic shift.
  auto result = (a + b) >> 1;
  // (-10 + 3) = -7, (-7) >> 1 = -4 (arithmetic shift, truncation toward -inf)
  EXPECT_EQ(result.ToString(), "-4");
}

TEST(SignedBigIntTest, LeftShift) {
  SignedBigInt<2> v(1);
  auto result = v << 3;
  EXPECT_EQ(result.ToString(), "8");
}

TEST(SignedBigIntTest, BitwiseAnd) {
  SignedBigInt<2> a(-1);
  SignedBigInt<2> b(0xFF);
  auto result = a & b;
  EXPECT_EQ(result.ToString(), "255");
}

// --- Stream output ---

TEST(SignedBigIntTest, StreamOutput) {
  SignedBigInt<2> v(-42);
  std::ostringstream oss;
  oss << v;
  EXPECT_EQ(oss.str(), "-42");
}

// --- 4-limb (256-bit) tests ---

TEST(SignedBigIntTest, FourLimbNegative) {
  SignedBigInt<4> v(-1);
  EXPECT_TRUE(v.IsNegative());
  EXPECT_EQ(v.ToString(), "-1");
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(v[i], std::numeric_limits<uint64_t>::max());
  }
}

TEST(SignedBigIntTest, FourLimbComparison) {
  SignedBigInt<4> a(-100);
  SignedBigInt<4> b(100);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
}

TEST(SignedBigIntTest, FourLimbDivision) {
  SignedBigInt<4> a(-15);
  SignedBigInt<4> b(4);
  EXPECT_EQ((a / b).ToString(), "-3");
  EXPECT_EQ((a % b).ToString(), "-3");
}

}  // namespace
}  // namespace zk_dtypes
