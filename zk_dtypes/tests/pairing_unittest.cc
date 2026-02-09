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

// Test full pairing: e(P, Q) * e(-P, Q) == 1.
TEST_F(PairingTest, PairingCancellation) {
  using G1Affine = G1Curve::AffinePoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1 = G1Affine::Generator();
  G1Affine neg_g1 = -g1;
  G2Affine g2 = G2Affine::Generator();

  std::vector<G1Affine> g1_pts = {g1, neg_g1};
  std::vector<G2Prep> g2_pts = {G2Prep::From(g2), G2Prep::From(g2)};

  auto f = BN254Curve::MultiMillerLoop(g1_pts, g2_pts);
  auto result = BN254Curve::FinalExponentiation(f);
  EXPECT_TRUE(result.IsOne());
}

// Test that FrobeniusMap<1> applied 12 times on Fp12 is identity.
TEST_F(PairingTest, FrobeniusMapCycle) {
  FqX12 a = FqX12::Random();
  FqX12 b = a;
  for (int i = 0; i < 12; ++i) {
    b = b.template FrobeniusMap<1>();
  }
  EXPECT_EQ(a, b) << "FrobeniusMap<1>^12 should be identity on Fp12";
}

// Test FrobeniusMap on Fp2: φ(x) = x^q is conjugation for BN254.
TEST_F(PairingTest, FrobeniusMapFp2) {
  FqX2 a = FqX2::Random();
  // For Fp2, FrobeniusMap<1> should be x^q = conjugation
  FqX2 frob = a.template FrobeniusMap<1>();
  // Applying twice should give identity
  FqX2 frob2 = frob.template FrobeniusMap<1>();
  EXPECT_EQ(a, frob2) << "FrobeniusMap<1>^2 should be identity on Fp2";
}

// Test FrobeniusMap on Fp6: φ^6 should be identity.
TEST_F(PairingTest, FrobeniusMapFp6) {
  FqX6 a = FqX6::Random();
  FqX6 b = a;
  for (int i = 0; i < 6; ++i) {
    b = b.template FrobeniusMap<1>();
  }
  EXPECT_EQ(a, b) << "FrobeniusMap<1>^6 should be identity on Fp6";
}

// Verify FrobeniusMap<2> = FrobeniusMap<1> applied twice on Fp12.
TEST_F(PairingTest, FrobeniusMapConsistencyFp12) {
  FqX12 a = FqX12::Random();

  // FrobeniusMap<2> directly
  FqX12 frob2_direct = a.template FrobeniusMap<2>();

  // FrobeniusMap<1> applied twice
  FqX12 frob2_iter = a.template FrobeniusMap<1>();
  frob2_iter = frob2_iter.template FrobeniusMap<1>();

  EXPECT_EQ(frob2_direct, frob2_iter)
      << "FrobeniusMap<2> should equal FrobeniusMap<1> applied twice";

  // FrobeniusMap<3> directly
  FqX12 frob3_direct = a.template FrobeniusMap<3>();

  // FrobeniusMap<1> applied three times
  FqX12 frob3_iter = a.template FrobeniusMap<1>();
  frob3_iter = frob3_iter.template FrobeniusMap<1>();
  frob3_iter = frob3_iter.template FrobeniusMap<1>();

  EXPECT_EQ(frob3_direct, frob3_iter)
      << "FrobeniusMap<3> should equal FrobeniusMap<1> applied three times";
}

// Test MulBy034 sparse multiplication against full multiplication.
TEST_F(PairingTest, SparseMulBy034) {
  FqX12 a = FqX12::Random();
  FqX2 beta0 = FqX2::Random();
  FqX2 beta3 = FqX2::Random();
  FqX2 beta4 = FqX2::Random();

  FqX12 result = a.MulBy034(beta0, beta3, beta4);

  // Verify by computing full multiplication.
  // Sparse element: Fp12 = Fp6[w], c₀ = (β₀, 0, 0) ∈ Fp6, c₁ = (β₃, β₄, 0) ∈
  // Fp6
  FqX6 c0 = FqX6({beta0, FqX2::Zero(), FqX2::Zero()});
  FqX6 c1 = FqX6({beta3, beta4, FqX2::Zero()});
  FqX12 b = FqX12({c0, c1});
  FqX12 expected = a * b;
  EXPECT_EQ(result, expected);
}

// Test EC arithmetic: 3G1 + 5G1 = 8G1.
TEST_F(PairingTest, ECArithmetic) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;

  G1Jacobian gen = G1Jacobian::Generator();
  G1Jacobian g1_3 = gen * Fr(3);
  G1Jacobian g1_5 = gen * Fr(5);
  G1Jacobian g1_8 = gen * Fr(8);

  // 3·G1 + 5·G1 = 8·G1
  G1Jacobian sum = g1_3 + g1_5;
  EXPECT_EQ(sum.ToAffine(), g1_8.ToAffine());

  // -(8·G1) + 8·G1 = O
  G1Jacobian zero = g1_8 + (-g1_8);
  EXPECT_TRUE(zero.IsZero());
}

// Non-generator cancellation: e(3G1, G2) · e(-3G1, G2) == 1.
TEST_F(PairingTest, NonGeneratorCancellation) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1_3 = (G1Jacobian::Generator() * Fr(3)).ToAffine();
  G1Affine neg_g1_3 = -g1_3;
  G2Affine g2 = G2Affine::Generator();

  std::vector<G1Affine> g1_pts = {g1_3, neg_g1_3};
  std::vector<G2Prep> g2_pts = {G2Prep::From(g2), G2Prep::From(g2)};

  auto f = BN254Curve::MultiMillerLoop(g1_pts, g2_pts);
  auto result = BN254Curve::FinalExponentiation(f);
  EXPECT_TRUE(result.IsOne());
}

// 3-pair bilinearity using generators: e(G1, G2) · e(G1, G2) · e(-2G1, G2)
// == 1.
TEST_F(PairingTest, ThreePairGeneratorBilinearity) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1 = G1Affine::Generator();
  G1Affine neg_g1_2 = -(G1Jacobian::Generator() * Fr(2)).ToAffine();
  G2Affine g2 = G2Affine::Generator();

  std::vector<G1Affine> g1_pts = {g1, g1, neg_g1_2};
  std::vector<G2Prep> g2_pts = {G2Prep::From(g2), G2Prep::From(g2),
                                G2Prep::From(g2)};

  auto f = BN254Curve::MultiMillerLoop(g1_pts, g2_pts);
  auto result = BN254Curve::FinalExponentiation(f);
  EXPECT_TRUE(result.IsOne());
}

// 2-pair bilinearity: e(2G1, G2) · e(-2G1, G2) == 1.
TEST_F(PairingTest, TwoPairScalarCancellation) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1_2 = (G1Jacobian::Generator() * Fr(2)).ToAffine();
  G1Affine neg_g1_2 = -g1_2;
  G2Affine g2 = G2Affine::Generator();

  std::vector<G1Affine> g1_pts = {g1_2, neg_g1_2};
  std::vector<G2Prep> g2_pts = {G2Prep::From(g2), G2Prep::From(g2)};

  auto f = BN254Curve::MultiMillerLoop(g1_pts, g2_pts);
  auto result = BN254Curve::FinalExponentiation(f);
  EXPECT_TRUE(result.IsOne());
}

// Single-pairing consistency: e(G1, G2) should not be 1.
TEST_F(PairingTest, SinglePairingNonTrivial) {
  using G1Affine = G1Curve::AffinePoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1 = G1Affine::Generator();
  G2Affine g2 = G2Affine::Generator();

  std::vector<G1Affine> g1_pts = {g1};
  std::vector<G2Prep> g2_pts = {G2Prep::From(g2)};

  auto f = BN254Curve::MultiMillerLoop(g1_pts, g2_pts);
  auto result = BN254Curve::FinalExponentiation(f);
  EXPECT_FALSE(result.IsOne()) << "e(G1, G2) should not be 1 for generators";
}

// Single-pairing bilinearity: e(2G1, G2) == e(G1, G2)².
TEST_F(PairingTest, SinglePairingBilinearity) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1 = G1Affine::Generator();
  G1Affine g1_2 = (G1Jacobian::Generator() * Fr(2)).ToAffine();
  G2Affine g2 = G2Affine::Generator();

  // e(G1, G2)
  std::vector<G1Affine> g1_pts_1 = {g1};
  std::vector<G2Prep> g2_pts_1 = {G2Prep::From(g2)};
  auto f1 = BN254Curve::MultiMillerLoop(g1_pts_1, g2_pts_1);
  auto e1 = BN254Curve::FinalExponentiation(f1);

  // e(2G1, G2)
  std::vector<G1Affine> g1_pts_2 = {g1_2};
  std::vector<G2Prep> g2_pts_2 = {G2Prep::From(g2)};
  auto f2 = BN254Curve::MultiMillerLoop(g1_pts_2, g2_pts_2);
  auto e2 = BN254Curve::FinalExponentiation(f2);

  // e(G1, G2)² should equal e(2G1, G2)
  FqX12 e1_sq = e1 * e1;
  EXPECT_EQ(e1_sq, e2) << "Bilinearity: e(2G1, G2) should equal e(G1, G2)²";
}

// Verify FqX6 non-residue ξ = (9, 1) ∈ Fp2 and ξ * (3, 5) = (22, 48).
// Regression test for brace-initialization bug where {{9,1}} was parsed as
// a single Fq element BigInt({9,1}) = 2⁶⁴ + 9 instead of {Fq(9), Fq(1)}.
TEST_F(PairingTest, Fp6NonResidueCorrectness) {
  FqX2 xi = FqX6::One().NonResidue();
  auto xi_s = xi.MontReduce();
  EXPECT_EQ(xi_s.ToCoeffs()[0], Fq(9).MontReduce()) << "ξ[0] should be 9";
  EXPECT_EQ(xi_s.ToCoeffs()[1], Fq(1).MontReduce()) << "ξ[1] should be 1";

  // ξ * (3 + 5u) = (9 + u)(3 + 5u) = 27 + 45u + 3u + 5u² = 22 + 48u
  FqX2 b = FqX2({Fq(3), Fq(5)});
  auto r = (xi * b).MontReduce();
  EXPECT_EQ(r.ToCoeffs()[0], Fq(22).MontReduce());
  EXPECT_EQ(r.ToCoeffs()[1], Fq(48).MontReduce());
}

// Verify Square() == self * self for Fp12.
TEST_F(PairingTest, Fp12SquareConsistency) {
  FqX12 a = FqX12::Random();
  FqX12 sq = a.Square();
  FqX12 mul = a * a;
  EXPECT_EQ(sq, mul) << "Fp12 Square() should equal self * self";
}

// Test bilinearity: e(3G1, G2) * e(5G1, G2) * e(-8G1, G2) == 1.
TEST_F(PairingTest, Bilinearity) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;
  using G2Affine = G2Curve::AffinePoint;
  using G2Prep = BN254Curve::G2Prepared;

  G1Affine g1_3 = (G1Jacobian::Generator() * Fr(3)).ToAffine();
  G1Affine g1_5 = (G1Jacobian::Generator() * Fr(5)).ToAffine();
  G1Affine neg_g1_8 = -(G1Jacobian::Generator() * Fr(8)).ToAffine();
  G2Affine g2 = G2Affine::Generator();

  std::vector<G1Affine> g1_pts = {g1_3, g1_5, neg_g1_8};
  std::vector<G2Prep> g2_pts = {G2Prep::From(g2), G2Prep::From(g2),
                                G2Prep::From(g2)};

  auto f = BN254Curve::MultiMillerLoop(g1_pts, g2_pts);
  auto result = BN254Curve::FinalExponentiation(f);
  EXPECT_TRUE(result.IsOne());
}

}  // namespace
}  // namespace zk_dtypes::bn254
