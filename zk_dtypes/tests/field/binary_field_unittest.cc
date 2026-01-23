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

#include "zk_dtypes/include/field/binary_field.h"

#include "gtest/gtest.h"

#include "zk_dtypes/include/field/big_binary_field.h"
#include "zk_dtypes/include/field/prime_field.h"
#include "zk_dtypes/include/field/small_binary_field.h"
#include "zk_dtypes/include/random.h"

namespace zk_dtypes {
namespace {

template <typename F>
uint64_t GetLo(const F& f) {
  if constexpr (F::kStorageBits > 64) {
    return f[0];
  } else {
    return static_cast<uint64_t>(f.value());
  }
}

template <typename F>
uint64_t GetHi(const F& f) {
  if constexpr (F::kStorageBits > 64) {
    return f[1];
  } else {
    return 0;
  }
}

template <typename T>
class BinaryFieldTypedTest : public testing::Test {};

using BinaryFieldTypes =
    testing::Types<BinaryFieldT0, BinaryFieldT1, BinaryFieldT2, BinaryFieldT3,
                   BinaryFieldT4, BinaryFieldT5, BinaryFieldT6, BinaryFieldT7>;

TYPED_TEST_SUITE(BinaryFieldTypedTest, BinaryFieldTypes);

TYPED_TEST(BinaryFieldTypedTest, Traits) {
  using F = TypeParam;

  static_assert(IsComparable<F>);
  static_assert(IsField<F>);
  static_assert(IsBinaryField<F>);
  static_assert(!IsPrimeField<F>);
}

TYPED_TEST(BinaryFieldTypedTest, Zero) {
  using F = TypeParam;
  EXPECT_TRUE(F::Zero().IsZero());
  EXPECT_FALSE(F::One().IsZero());
}

TYPED_TEST(BinaryFieldTypedTest, One) {
  using F = TypeParam;
  EXPECT_TRUE(F::One().IsOne());
  EXPECT_FALSE(F::Zero().IsOne());
}

TYPED_TEST(BinaryFieldTypedTest, Min) {
  using F = TypeParam;
  EXPECT_EQ(F::Min(), F::Zero());
}

TYPED_TEST(BinaryFieldTypedTest, Max) {
  using F = TypeParam;
  if constexpr (F::kStorageBits <= 64) {
    EXPECT_EQ(F::Max(), F(F::Config::kValueMask));
  } else {
    EXPECT_EQ(F::Max(), F(BigInt<2>::Max()));
  }
}

TYPED_TEST(BinaryFieldTypedTest, AdditionIsXOR) {
  using F = TypeParam;

  F a = F::Random();
  F b = F::Random();

  F sum = a + b;
  EXPECT_EQ(GetLo(sum), GetLo(a) ^ GetLo(b));
  EXPECT_EQ(GetHi(sum), GetHi(a) ^ GetHi(b));
}

TYPED_TEST(BinaryFieldTypedTest, SubtractionIsXOR) {
  using F = TypeParam;

  F a = F::Random();
  F b = F::Random();

  EXPECT_EQ(a - b, a + b);

  F diff = a - b;
  EXPECT_EQ(GetLo(diff), GetLo(a) ^ GetLo(b));
  EXPECT_EQ(GetHi(diff), GetHi(a) ^ GetHi(b));
}

TYPED_TEST(BinaryFieldTypedTest, NegationIsIdentity) {
  using F = TypeParam;

  F a = F::Random();

  EXPECT_EQ(-a, a);
}

TYPED_TEST(BinaryFieldTypedTest, DoubleIsZero) {
  using F = TypeParam;

  F a = F::Random();

  EXPECT_EQ(a.Double(), F::Zero());
  EXPECT_EQ(a.Double(), a + a);
}

TYPED_TEST(BinaryFieldTypedTest, Square) {
  using F = TypeParam;

  F a = F::Random();

  EXPECT_EQ(a.Square(), a * a);
}

TYPED_TEST(BinaryFieldTypedTest, Inverse) {
  using F = TypeParam;

  F a = F::Random();
  while (a.IsZero()) {
    a = F::Random();
  }
  F a_inv = a.Inverse();
  EXPECT_TRUE((a * a_inv).IsOne());

  EXPECT_TRUE(F::Zero().Inverse().IsZero());
}

TYPED_TEST(BinaryFieldTypedTest, SubfieldDecomposition) {
  using F = TypeParam;

  // Skip for GF(2) which has no subfield
  if constexpr (F::kTowerLevel > 0) {
    F a = F::Random();
    auto [a0, a1] = a.ToSubfields();

    F reconstructed = F::FromSubfields(a0, a1);
    EXPECT_EQ(a, reconstructed);
  }
}

}  // namespace
}  // namespace zk_dtypes
