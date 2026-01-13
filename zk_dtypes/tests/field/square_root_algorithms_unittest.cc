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

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zk_dtypes/include/field/square_root_algorithms/shanks.h"
#include "zk_dtypes/include/field/square_root_algorithms/square_root_algorithm9.h"
#include "zk_dtypes/include/field/square_root_algorithms/tonelli_shanks.h"

namespace zk_dtypes::test {
namespace {

TEST(SquareRootAlgorithms, ComputeShanksSquareRoot) {
  for (size_t i = 0; i < 7; ++i) {
    Fr a(i);
    absl::StatusOr<Fr> sqrt = ComputeShanksSquareRoot(a);
    if (i == 0 || i == 1 || i == 2 || i == 4) {
      ASSERT_TRUE(sqrt.ok());
      EXPECT_EQ(sqrt->Square(), a);
    } else {
      ASSERT_FALSE(sqrt.ok());
    }
  }
}

TEST(SquareRootAlgorithms, ComputeTonelliShanksSquareRoot) {
  for (size_t i = 0; i < 7; ++i) {
    Fr a(i);
    absl::StatusOr<Fr> sqrt = ComputeTonelliShanksSquareRoot(a);
    if (i == 0 || i == 1 || i == 2 || i == 4) {
      ASSERT_TRUE(sqrt.ok());
      EXPECT_EQ(sqrt->Square(), a);
    } else {
      ASSERT_FALSE(sqrt.ok());
    }
  }
}

TEST(SquareRootAlgorithms, ComputeAlgorithm9SquareRoot) {
  for (size_t i = 0; i < 49; ++i) {
    FqX2 a({i % 7, i / 7});
    absl::StatusOr<FqX2> sqrt = ComputeAlgorithm9SquareRoot(a);
    if (i == 0 || i == 1 || i == 2 || i == 3 || i == 4 || i == 5 || i == 6 ||
        i == 7 || i == 8 || i == 13 || i == 14 || i == 16 || i == 19 ||
        i == 21 || i == 24 || i == 25 || i == 28 || i == 31 || i == 32 ||
        i == 35 || i == 37 || i == 40 || i == 42 || i == 43 || i == 48) {
      ASSERT_TRUE(sqrt.ok());
      EXPECT_EQ(sqrt->Square(), a);
    } else {
      ASSERT_FALSE(sqrt.ok());
    }
  }
}

}  // namespace
}  // namespace zk_dtypes::test
