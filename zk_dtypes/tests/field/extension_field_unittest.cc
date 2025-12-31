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

#include <algorithm>

#include "gtest/gtest.h"

#include "zk_dtypes/include/all_types.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
// TODO(chokobole33): Remove this header after we include this field to
// ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST.
#include "zk_dtypes/include/field/mersenne31/mersenne314.h"

namespace zk_dtypes {
namespace {

struct AutoReset {
  AutoReset(std::optional<ExtensionFieldMulAlgorithm>& ptr,
            ExtensionFieldMulAlgorithm new_value)
      : ptr(ptr), old_value(ptr) {
    ptr = new_value;
  }
  ~AutoReset() { ptr = old_value; }

  std::optional<ExtensionFieldMulAlgorithm>& ptr;
  std::optional<ExtensionFieldMulAlgorithm> old_value;
};

template <typename T>
class ExtensionFieldTypedTest : public testing::Test {};

using ExtensionFieldTypes = testing::Types<
#define EXTENSION_FIELD_TYPE(ActualType, ...) ActualType,
    ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(EXTENSION_FIELD_TYPE)
#undef EXTENSION_FIELD_TYPE
        Mersenne314,
    Mersenne314Std, test::Fq2, test::Fq2Std>;

TYPED_TEST_SUITE(ExtensionFieldTypedTest, ExtensionFieldTypes);

TYPED_TEST(ExtensionFieldTypedTest, Zero) {
  using ExtF = TypeParam;

  EXPECT_TRUE(ExtF(0).IsZero());
  EXPECT_TRUE(ExtF::Zero().IsZero());
  EXPECT_FALSE(ExtF::One().IsZero());
}

TYPED_TEST(ExtensionFieldTypedTest, One) {
  using ExtF = TypeParam;

  EXPECT_TRUE(ExtF(1).IsOne());
  EXPECT_TRUE(ExtF::One().IsOne());
  EXPECT_FALSE(ExtF::Zero().IsOne());
}

TYPED_TEST(ExtensionFieldTypedTest, ConstructFromUnsignedInteger) {
  using ExtF = TypeParam;
  using BaseField = typename ExtF::BaseField;
  constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

  ExtF a(uint32_t{3});

  // The first coefficient should be the value, others should be zero.
  EXPECT_EQ(a[0], BaseField(3));
  for (size_t i = 1; i < kDegree; ++i) {
    EXPECT_TRUE(a[i].IsZero());
  }
}

TYPED_TEST(ExtensionFieldTypedTest, ConstructFromBaseField) {
  using ExtF = TypeParam;
  using BaseField = typename ExtF::BaseField;
  constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

  ExtF a(BaseField(3));
  EXPECT_EQ(a[0], BaseField(3));
  for (size_t i = 1; i < kDegree; ++i) {
    EXPECT_TRUE(a[i].IsZero());
  }
}

TYPED_TEST(ExtensionFieldTypedTest, ConstructFromBasePrimeField) {
  using ExtF = TypeParam;
  using BaseField = typename ExtF::BaseField;
  using BasePrimeField = typename ExtF::BasePrimeField;

  if constexpr (std::is_same_v<BaseField, BasePrimeField>) {
    GTEST_SKIP()
        << "Skipping test because BaseField and BasePrimeField are the same.";
  } else {
    ExtF a(BasePrimeField(3));
    EXPECT_EQ(a[0], BaseField(3));
    absl::Span<const BasePrimeField> base_prime_fields = a.AsBasePrimeFields();
    EXPECT_EQ(base_prime_fields[0], BasePrimeField(3));
    for (size_t i = 1; i < ExtF::ExtensionDegree(); ++i) {
      EXPECT_TRUE(base_prime_fields[i].IsZero());
    }
  }
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

  std::vector<ExtensionFieldMulAlgorithm> algorithms;
  if constexpr (ExtF::Config::kDegreeOverBaseField == 2) {
    using BaseField = typename ExtF::BaseField;
    algorithms = {
        ExtensionFieldMulAlgorithm::kCustom2,
        ExtensionFieldMulAlgorithm::kKaratsuba,
    };
    if (ExtF::Config::kNonResidue == BaseField(-1)) {
      algorithms.push_back(ExtensionFieldMulAlgorithm::kCustom);
    }
  } else if constexpr (ExtF::Config::kDegreeOverBaseField == 3) {
    algorithms = {
        ExtensionFieldMulAlgorithm::kCustom,
        ExtensionFieldMulAlgorithm::kKaratsuba,
    };
  } else {
    algorithms = {
        ExtensionFieldMulAlgorithm::kCustom,
        ExtensionFieldMulAlgorithm::kKaratsuba,
        ExtensionFieldMulAlgorithm::kToomCook,
    };
  }
  for (auto algorithm : algorithms) {
    SCOPED_TRACE(
        absl::Substitute("algorithm: $0", static_cast<int>(algorithm)));
    AutoReset reset(ExtF::square_algorithm_, algorithm);
    EXPECT_EQ(a * a, a.Square());
  }
}

template <typename ExtF>
ExtF SchoolbookMul(const ExtF& a, const ExtF& b) {
  using F = typename ExtF::BaseField;
  constexpr size_t kDegree = ExtF::Config::kDegreeOverBaseField;

  // Schoolbook multiplication in Fp[u] / (uⁿ - ξ)
  // c = Σᵢ Σⱼ aᵢbⱼu^(i+j) mod (uⁿ - ξ)
  // When i + j >= n, u^(i+j) = u^(i+j-n) · ξ
  F non_residue = ExtF::Config::kNonResidue;
  ExtF ret;
  for (size_t i = 0; i < kDegree; ++i) {
    for (size_t j = 0; j < kDegree; ++j) {
      size_t idx = i + j;
      if (idx < kDegree) {
        ret[idx] += a[i] * b[j];
      } else {
        // u^(i+j) = ξ · u^(i+j-n)
        ret[idx - kDegree] += a[i] * b[j] * non_residue;
      }
    }
  }
  return ret;
}

TYPED_TEST(ExtensionFieldTypedTest, Mul) {
  using ExtF = TypeParam;

  ExtF a = ExtF::Random();
  ExtF b = ExtF::Random();
  ExtF c = a * b;

  std::vector<ExtensionFieldMulAlgorithm> algorithms;
  if constexpr (ExtF::Config::kDegreeOverBaseField == 4) {
    algorithms = {
        ExtensionFieldMulAlgorithm::kKaratsuba,
        ExtensionFieldMulAlgorithm::kToomCook,
    };
  } else {
    algorithms = {
        ExtensionFieldMulAlgorithm::kKaratsuba,
    };
  }
  for (auto algorithm : algorithms) {
    SCOPED_TRACE(
        absl::Substitute("algorithm: $0", static_cast<int>(algorithm)));
    AutoReset reset(ExtF::mul_algorithm_, algorithm);
    EXPECT_EQ(c, SchoolbookMul(a, b));
  }
}

TYPED_TEST(ExtensionFieldTypedTest, SquareRoot) {
  using ExtF = TypeParam;
  // clang-format off
  if constexpr (std::is_same_v<ExtF, Babybear4> ||
                std::is_same_v<ExtF, Babybear4Std> ||
                std::is_same_v<ExtF, Koalabear4> ||
                std::is_same_v<ExtF, Koalabear4Std> ||
                std::is_same_v<ExtF, Mersenne314> ||
                std::is_same_v<ExtF, Mersenne314Std>) {
    GTEST_SKIP() << "SquareRoot is not implemented for quartic extension "
                    "fields.";
  } else if constexpr (std::is_same_v<ExtF, Goldilocks3> ||  // NOLINT(readability/braces)
                       std::is_same_v<ExtF, Goldilocks3Std>) {
    GTEST_SKIP() << "Skipping because Goldilocks3 has large trace value.";
  } else {  // NOLINT(readability/braces)
    // clang-format on
    ExtF a = ExtF::Random();
    ExtF a2 = a.Square();
    absl::StatusOr<ExtF> sqrt = a2.SquareRoot();
    ASSERT_TRUE(sqrt.ok());
    EXPECT_TRUE(a == (*sqrt) || a == -(*sqrt));
  }
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

TYPED_TEST(ExtensionFieldTypedTest, AsBasePrimeFields) {
  using ExtF = TypeParam;
  using BasePrimeField = typename ExtF::BasePrimeField;

  ExtF a = ExtF::Random();
  absl::Span<BasePrimeField> span = a.AsBasePrimeFields();

  // Test that AsBasePrimeFields() returns the correct size.
  EXPECT_EQ(span.size(), ExtF::ExtensionDegree());

  // Test that span points to actual internal data.
  EXPECT_TRUE(std::equal(a.begin(), a.end(), span.begin()));

  const ExtF const_a = ExtF::Random();
  absl::Span<const BasePrimeField> const_span = const_a.AsBasePrimeFields();

  // Test that AsBasePrimeFields() returns the correct size.
  EXPECT_EQ(const_span.size(), ExtF::ExtensionDegree());

  // Test that span points to actual internal data.
  EXPECT_TRUE(std::equal(const_a.begin(), const_a.end(), const_span.begin()));
}

// Define Fq4 as a quadratic extension over Fq2.
// u² - (2 + 1i) = 0, where (2 + 1i) is in Fq2.
REGISTER_EXTENSION_FIELD_WITH_TOWER(Fq4, test::Fq2, test::Fq, 2, {2, 1});

TEST(ExtensionFieldTest, BasePrimeFieldIteratorWithTower) {
  using Fq = test::Fq;

  Fq4 val = Fq4::Random();
  std::vector<Fq> elements;
  for (const auto& prime_field_element : val) {
    elements.push_back(prime_field_element);
  }

  ASSERT_EQ(elements.size(), Fq4::ExtensionDegree());

  // Check that elements are correct
  auto values = val.values();              // std::array<Fq2, 2>
  auto values0_vals = values[0].values();  // std::array<Fq, 2>
  auto values1_vals = values[1].values();  // std::array<Fq, 2>
  EXPECT_EQ(elements[0], values0_vals[0]);
  EXPECT_EQ(elements[1], values0_vals[1]);
  EXPECT_EQ(elements[2], values1_vals[0]);
  EXPECT_EQ(elements[3], values1_vals[1]);
}

}  // namespace
}  // namespace zk_dtypes
