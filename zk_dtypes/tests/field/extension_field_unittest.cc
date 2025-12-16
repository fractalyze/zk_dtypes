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

#include "gtest/gtest.h"

#include "zk_dtypes/include/all_types.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"

namespace zk_dtypes {
namespace {

template <typename T>
class ExtensionFieldTypedTest : public testing::Test {};

using ExtensionFieldTypes = testing::Types<
#define EXTENSION_FIELD_TYPE(ActualType, ...) ActualType,
    ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(EXTENSION_FIELD_TYPE)
#undef EXTENSION_FIELD_TYPE
        test::Fq2,
    test::Fq2Std>;

TYPED_TEST_SUITE(ExtensionFieldTypedTest, ExtensionFieldTypes);

TYPED_TEST(ExtensionFieldTypedTest, Zero) {
  using ExtF = TypeParam;

  EXPECT_TRUE(ExtF::Zero().IsZero());
  EXPECT_FALSE(ExtF::One().IsZero());
}

TYPED_TEST(ExtensionFieldTypedTest, One) {
  using ExtF = TypeParam;

  EXPECT_TRUE(ExtF::One().IsOne());
  EXPECT_FALSE(ExtF::Zero().IsOne());
}

TYPED_TEST(ExtensionFieldTypedTest, Add) {
  using ExtF = TypeParam;
  constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

  ExtF a = ExtF::Random();
  ExtF b = ExtF::Random();
  ExtF c = a + b;

  // (a + b)ᵢ = aᵢ + bᵢ
  for (size_t i = 0; i < kDegree; ++i) {
    EXPECT_EQ(c[i], a[i] + b[i]);
  }
}

TYPED_TEST(ExtensionFieldTypedTest, Sub) {
  using ExtF = TypeParam;
  constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

  ExtF a = ExtF::Random();
  ExtF b = ExtF::Random();
  ExtF c = a - b;

  // (a - b)ᵢ = aᵢ - bᵢ
  for (size_t i = 0; i < kDegree; ++i) {
    EXPECT_EQ(c[i], a[i] - b[i]);
  }
}

TYPED_TEST(ExtensionFieldTypedTest, Double) {
  using ExtF = TypeParam;

  ExtF a = ExtF::Random();

  // a + a = a.Double()
  EXPECT_EQ(a + a, a.Double());
}

TYPED_TEST(ExtensionFieldTypedTest, Square) {
  using ExtF = TypeParam;

  ExtF a = ExtF::Random();

  // a * a = a.Square()
  EXPECT_EQ(a * a, a.Square());
}

TYPED_TEST(ExtensionFieldTypedTest, Mul) {
  using ExtF = TypeParam;
  using BaseF = typename ExtF::BaseField;
  constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

  ExtF a = ExtF::Random();
  ExtF b = ExtF::Random();
  ExtF c = a * b;

  // Schoolbook multiplication in Fp[u] / (uⁿ - ξ)
  // c = Σᵢ Σⱼ aᵢbⱼu^(i+j) mod (uⁿ - ξ)
  // When i + j >= n, u^(i+j) = u^(i+j-n) · ξ
  BaseF non_residue = ExtF::Config::kNonResidue;
  ExtF expected;
  for (size_t i = 0; i < kDegree; ++i) {
    for (size_t j = 0; j < kDegree; ++j) {
      size_t idx = i + j;
      if (idx < kDegree) {
        expected[idx] += a[i] * b[j];
      } else {
        // u^(i+j) = ξ · u^(i+j-n)
        expected[idx - kDegree] += a[i] * b[j] * non_residue;
      }
    }
  }

  EXPECT_EQ(c, expected);
}

TYPED_TEST(ExtensionFieldTypedTest, SquareRoot) {
  using ExtF = TypeParam;
  if constexpr (std::is_same_v<ExtF, Goldilocks3> ||
                std::is_same_v<ExtF, Goldilocks3Std>) {
    GTEST_SKIP() << "Skipping test because Goldilocks3 has large trace value.";
  }

  ExtF a = ExtF::Random();
  ExtF a2 = a.Square();
  absl::StatusOr<ExtF> sqrt = a2.SquareRoot();
  ASSERT_TRUE(sqrt.ok());
  EXPECT_TRUE(a == (*sqrt) || a == -(*sqrt));
}

TYPED_TEST(ExtensionFieldTypedTest, Inverse) {
  using ExtF = TypeParam;

  ExtF a = ExtF::Random();
  while (a.IsZero()) {
    a = ExtF::Random();
  }
  absl::StatusOr<ExtF> a_inverse = a.Inverse();
  ASSERT_TRUE(a_inverse.ok());
  EXPECT_TRUE((a * (*a_inverse)).IsOne());
}

TYPED_TEST(ExtensionFieldTypedTest, MontReduce) {
  using ExtF = TypeParam;
  if constexpr (!ExtF::kUseMontgomery) {
    GTEST_SKIP() << "Skipping test for non-Montgomery types.";
  } else {
    using ExtFStd = typename ExtF::StdType;
    constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

    ExtF a = ExtF::Random();
    ExtFStd reduced = a.MontReduce();

    for (size_t i = 0; i < kDegree; ++i) {
      EXPECT_EQ(reduced[i], a[i].MontReduce());
    }
  }
}

}  // namespace
}  // namespace zk_dtypes
