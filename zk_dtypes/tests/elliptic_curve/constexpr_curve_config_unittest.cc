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

/// Compile-time and runtime verification of elliptic curve config constants.
///
/// static_assert: ensures all kA/kB/kX/kY are truly constexpr.
/// TYPED_TEST: verifies Montgomery values match standard-form counterparts
///             and generator points lie on the curve (y² = x³ + ax + b).
///
/// Regression test for https://github.com/fractalyze/zk_dtypes/issues/111.

#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/bls12_381/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/elliptic_curve/secp256k1/g1.h"
#include "zk_dtypes/include/elliptic_curve/secp256r1/fq.h"
#include "zk_dtypes/include/elliptic_curve/secp256r1/g1.h"

namespace zk_dtypes {
namespace {

// =========================================================================
// Compile-time constexpr verification.
// =========================================================================

template <typename Config>
constexpr bool VerifyConstexpr() {
  [[maybe_unused]] auto a = Config::kA;
  [[maybe_unused]] auto b = Config::kB;
  [[maybe_unused]] auto x = Config::kX;
  [[maybe_unused]] auto y = Config::kY;
  return true;
}

// Montgomery configs.
static_assert(VerifyConstexpr<bn254::G1SwCurveMontConfig>());
static_assert(VerifyConstexpr<bn254::G2SwCurveMontConfig>());
static_assert(VerifyConstexpr<secp256k1::G1SwCurveMontConfig>());
static_assert(VerifyConstexpr<secp256r1::G1SwCurveMontConfig>());
static_assert(VerifyConstexpr<bls12_381::G1SwCurveMontConfig>());

// =========================================================================
// Runtime correctness via TYPED_TEST on MontConfig.
// =========================================================================

using MontConfigs =
    ::testing::Types<bn254::G1SwCurveMontConfig, bn254::G2SwCurveMontConfig,
                     secp256k1::G1SwCurveMontConfig,
                     secp256r1::G1SwCurveMontConfig,
                     bls12_381::G1SwCurveMontConfig>;

template <typename T>
class CurveConfigTest : public ::testing::Test {};

TYPED_TEST_SUITE(CurveConfigTest, MontConfigs);

TYPED_TEST(CurveConfigTest, GeneratorOnCurve) {
  using Mont = TypeParam;
  using F = typename Mont::BaseField;

  F x = Mont::kX;
  F y = Mont::kY;
  F a = Mont::kA;
  F b = Mont::kB;

  // y² = x³ + ax + b
  EXPECT_EQ(y * y, x.Square() * x + a * x + b);
}

TYPED_TEST(CurveConfigTest, MontReduceMatchesStandard) {
  using Mont = TypeParam;
  using Std = typename Mont::StdConfig;

  EXPECT_EQ(Mont::kA.MontReduce(), Std::kA) << "kA MontReduce mismatch";
  EXPECT_EQ(Mont::kB.MontReduce(), Std::kB) << "kB MontReduce mismatch";
  EXPECT_EQ(Mont::kX.MontReduce(), Std::kX) << "kX MontReduce mismatch";
  EXPECT_EQ(Mont::kY.MontReduce(), Std::kY) << "kY MontReduce mismatch";
}

}  // namespace
}  // namespace zk_dtypes
