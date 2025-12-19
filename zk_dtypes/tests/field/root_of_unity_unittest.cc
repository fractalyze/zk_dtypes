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

#include "zk_dtypes/include/field/root_of_unity.h"

#include "gtest/gtest.h"

#include "zk_dtypes/include/all_types.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"

namespace zk_dtypes {
namespace {

using PrimeFieldTypes = testing::Types<
#define PRIME_FIELD_TYPE(ActualType, ...) ActualType,
    ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST(PRIME_FIELD_TYPE)
#undef PRIME_FIELD_TYPE
        test::Fr,
    test::FrStd>;

namespace {

template <typename PrimeField>
class RootOfUnityTest : public testing::Test {};

}  // namespace

TYPED_TEST_SUITE(RootOfUnityTest, PrimeFieldTypes);

TYPED_TEST(RootOfUnityTest, Decompose) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (uint32_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (uint32_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        uint64_t n = (uint64_t{1} << i) *
                     std::pow(uint64_t{F::Config::kSmallSubgroupBase}, j);

        ASSERT_TRUE(internal::Decompose<F>(n).ok());
      }
    }
  } else {
    GTEST_SKIP() << "No LargeSubgroupRootOfUnity";
  }
}

TYPED_TEST(RootOfUnityTest, TwoAdicRootOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasTwoAdicRootOfUnity) {
    F n = F(2).Pow(F::Config::kTwoAdicity);
    ASSERT_TRUE(
        F::FromUnchecked(F::Config::kTwoAdicRootOfUnity).Pow(n).IsOne());
  } else {
    GTEST_SKIP() << "No TwoAdicRootOfUnity";
  }
}

TYPED_TEST(RootOfUnityTest, LargeSubgroupOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    F n =
        F(2).Pow(F::Config::kTwoAdicity) *
        F(F::Config::kSmallSubgroupBase).Pow(F::Config::kSmallSubgroupAdicity);
    ASSERT_TRUE(
        F::FromUnchecked(F::Config::kLargeSubgroupRootOfUnity).Pow(n).IsOne());
  } else {
    GTEST_SKIP() << "No LargeSubgroupRootOfUnity";
  }
}

TYPED_TEST(RootOfUnityTest, GetRootOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (uint32_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (uint32_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        uint64_t n = (uint64_t{1} << i) *
                     std::pow(uint64_t{F::Config::kSmallSubgroupBase}, j);
        absl::StatusOr<F> root = GetRootOfUnity<F>(n);
        ASSERT_TRUE(root.ok());
        ASSERT_TRUE(root->Pow(n).IsOne());
      }
    }
  } else if constexpr (F::Config::kHasTwoAdicRootOfUnity) {
    for (uint32_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      uint64_t n = uint64_t{1} << i;
      absl::StatusOr<F> root = GetRootOfUnity<F>(n);
      ASSERT_TRUE(root.ok());
      ASSERT_TRUE(root->Pow(n).IsOne());
    }
  } else {
    GTEST_SKIP() << "No RootOfUnity";
  }
}

}  // namespace
}  // namespace zk_dtypes
