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

#include "zk_dtypes/include/field/prime_field.h"

#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"
#include "zk_dtypes/include/random.h"

namespace zk_dtypes {

template <typename T>
class PrimeFieldTypedTest : public testing::Test {};

using PrimeFieldTypes = testing::Types<
    // clang-format off
    // 8-bit prime fields
    test::Fr,
    test::FrStd,
    // 32-bit prime fields
    Babybear,
    BabybearStd,
    Koalabear,
    Mersenne31,
    // 64-bit prime fields
    Goldilocks,
    // 256-bit prime fields
    bn254::Fq,
    bn254::FqStd,
    bn254::Fr
    // clang-format on
    >;
TYPED_TEST_SUITE(PrimeFieldTypedTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldTypedTest, Zero) {
  using F = TypeParam;
  EXPECT_TRUE(F::Zero().IsZero());
  EXPECT_FALSE(F::One().IsZero());
}

TYPED_TEST(PrimeFieldTypedTest, One) {
  using F = TypeParam;

  EXPECT_TRUE(F::One().IsOne());
  EXPECT_FALSE(F::Zero().IsOne());
}

TYPED_TEST(PrimeFieldTypedTest, Operations) {
  using F = TypeParam;
  using UnderlyingType = typename F::UnderlyingType;

  UnderlyingType a_value, b_value;
  if constexpr (F::Config::kModulusBits <= 64) {
    a_value = Uniform(UnderlyingType{0}, F::Config::kModulus);
    b_value = Uniform(UnderlyingType{0}, F::Config::kModulus);
  } else {
    a_value = UnderlyingType::Random(F::Config::kModulus);
    b_value = UnderlyingType::Random(F::Config::kModulus);
  }

  F a = F(a_value);
  F b = F(b_value);

  EXPECT_EQ(a > b, a_value > b_value);
  EXPECT_EQ(a < b, a_value < b_value);
  EXPECT_EQ(a == b, a_value == b_value);
  if constexpr (F::HasSpareBit()) {
    if constexpr (F::Config::kModulusBits <= 64) {
      EXPECT_EQ(a + b, F((a_value + b_value) % F::Config::kModulus));
      EXPECT_EQ(a.Double(), F((a_value + a_value) % F::Config::kModulus));
      if (a >= b) {
        EXPECT_EQ(a - b, F((a_value - b_value) % F::Config::kModulus));
      } else {
        EXPECT_EQ(a - b, F((a_value + F::Config::kModulus - b_value) %
                           F::Config::kModulus));
      }
    } else {
      EXPECT_EQ(a + b, F(*((a_value + b_value) % F::Config::kModulus)));
      EXPECT_EQ(a.Double(), F(*((a_value + a_value) % F::Config::kModulus)));
      if (a >= b) {
        EXPECT_EQ(a - b, F(*((a_value - b_value) % F::Config::kModulus)));
      } else {
        EXPECT_EQ(a - b, F(*((a_value + F::Config::kModulus - b_value) %
                             F::Config::kModulus)));
      }
    }
  }

  if constexpr (F::kUseMontgomery) {
    using StdF = typename F::StdType;
    StdF a_std = a.MontReduce();
    StdF b_std = b.MontReduce();
    StdF mul;
    StdF::VerySlowMul(a_std, b_std, mul);
    EXPECT_EQ(a * b, F(mul.value()));

    StdF square;
    StdF::VerySlowMul(a_std, a_std, square);
    EXPECT_EQ(a.Square(), F(square.value()));
  } else {
    GTEST_SKIP()
        << "Skipping test because mul operation already uses VerySlowMul()";
  }
}

TYPED_TEST(PrimeFieldTypedTest, SquareRoot) {
  using F = TypeParam;

  F a = F::Random();
  F a2 = a.Square();
  absl::StatusOr<F> sqrt = a2.SquareRoot();
  ASSERT_TRUE(sqrt.ok());
  EXPECT_TRUE(a == (*sqrt) || a == -(*sqrt));
}

TYPED_TEST(PrimeFieldTypedTest, Inverse) {
  using F = TypeParam;

  F a = F::Random();
  while (a.IsZero()) {
    a = F::Random();
  }
  absl::StatusOr<F> a_inv = a.Inverse();
  ASSERT_TRUE(a_inv.ok());
  EXPECT_TRUE((a * (*a_inv)).IsOne());
}

}  // namespace zk_dtypes
