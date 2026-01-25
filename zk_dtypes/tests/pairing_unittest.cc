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

#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/bn/bn254/bn254_curve.h"

namespace zk_dtypes::bn254 {
namespace {

class PairingTest : public ::testing::Test {
 protected:
  void SetUp() override { BN254CurveConfig::Init(); }
};

// Test that extension fields are correctly set up.
TEST_F(PairingTest, ExtensionFieldSetup) {
  // Test Fq2
  FqX2 a = FqX2::One();
  FqX2 b = FqX2::One();
  EXPECT_EQ(a * b, FqX2::One());

  // Test Fq6
  FqX6 c = FqX6::One();
  FqX6 d = FqX6::One();
  EXPECT_EQ(c * d, FqX6::One());

  // Test Fq12
  FqX12 e = FqX12::One();
  FqX12 f = FqX12::One();
  EXPECT_EQ(e * f, FqX12::One());
}

// Test that sparse multiplications work.
TEST_F(PairingTest, SparseMul) {
  FqX6 a = FqX6::Random();

  // Test MulBy01
  FqX2 beta0 = FqX2::Random();
  FqX2 beta1 = FqX2::Random();

  FqX6 result = a.MulBy01(beta0, beta1);

  // Verify by computing the full multiplication
  FqX6 b = FqX6({beta0, beta1, FqX2::Zero()});
  FqX6 expected = a * b;
  EXPECT_EQ(result, expected);
}

// Test cyclotomic operations.
TEST_F(PairingTest, CyclotomicOps) {
  FqX12 a = FqX12::Random();

  // Test cyclotomic inverse (conjugation)
  FqX12 a_inv = a.CyclotomicInverse();
  // For elements NOT in the cyclotomic subgroup, a * conj(a) != 1
  // But the operation should still be well-defined.

  // Test cyclotomic square
  FqX12 a_sq = a.CyclotomicSquare();
  EXPECT_EQ(a_sq, a.Square());
}

// Test Frobenius map.
TEST_F(PairingTest, FrobeniusMap) {
  FqX2 a = FqX2::Random();

  // Frobenius^1 applied twice on Fq2 should be identity (since degree is 2)
  FqX2 a_frob = a.template Frobenius<1>();
  FqX2 a_frob2 = a_frob.template Frobenius<1>();
  EXPECT_EQ(a_frob2, a);
}

// Test TwoInv.
TEST_F(PairingTest, TwoInv) {
  Fq two = Fq(2);
  Fq two_inv = Fq::TwoInv();
  EXPECT_EQ(two * two_inv, Fq::One());
}

// Test G2Prepared precomputation.
TEST_F(PairingTest, G2Prepared) {
  using G2Prepared = BN254Curve::G2Prepared;
  using G2AffinePoint = G2Curve::AffinePoint;

  // Create a G2 point from the generator
  G2AffinePoint g2_gen = G2AffinePoint::Generator();

  // Create prepared G2
  G2Prepared g2_prep = G2Prepared::From(g2_gen);

  // Should not be infinity
  EXPECT_FALSE(g2_prep.infinity());

  // Should have precomputed coefficients
  EXPECT_GT(g2_prep.ell_coeffs().size(), 0u);
}

}  // namespace
}  // namespace zk_dtypes::bn254
