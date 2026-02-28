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
#include "zk_dtypes/include/pow.h"

namespace zk_dtypes::bn254 {
namespace {

TEST(PairingTest, SparseMulBy01) {
  FqX6 a = FqX6::Random();

  FqX2 beta0 = FqX2::Random();
  FqX2 beta1 = FqX2::Random();

  FqX6 result = a.MulBy01(beta0, beta1);

  FqX6 b = FqX6({beta0, beta1, FqX2::Zero()});
  FqX6 expected = a * b;
  EXPECT_EQ(result, expected);
}

TEST(PairingTest, SparseMulBy034) {
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

// Test cyclotomic operations.
TEST(PairingTest, CyclotomicOps) {
  FqX12 a = FqX12::Random();

  // Test cyclotomic inverse: involution (double conjugation = identity)
  FqX12 a_inv = a.CyclotomicInverse();
  EXPECT_EQ(a_inv.CyclotomicInverse(), a);

  // Test cyclotomic inverse on Norm=1 element.
  // f1 = conj(a) * a⁻¹ = a^(p⁶-1) has Norm=1.
  FqX12 f1 = a.CyclotomicInverse() * a.Inverse();
  EXPECT_EQ(f1 * f1.CyclotomicInverse(), FqX12::One());

  // Construct f2 in GΦ₆: f2 = f1^(p²+1).
  // After f1 = a^(p⁶-1) (Norm=1), applying f1^(p²+1) gives an element
  // in the cyclotomic subgroup GΦ₆ = {x : x^(p⁴-p²+1) = 1}.
  BigInt<4> q(Fq::Order());
  FqX12 f1_pow_q = zk_dtypes::Pow(f1, q);
  FqX12 f1_pow_q2 = zk_dtypes::Pow(f1_pow_q, q);
  FqX12 f2 = f1_pow_q2 * f1;

  EXPECT_NE(f2, FqX12::One());
  EXPECT_EQ(f2 * f2.CyclotomicInverse(), FqX12::One());

  // Test cyclotomic square on GΦ₆ element.
  EXPECT_EQ(f2.CyclotomicSquare(), f2.Square());
}

TEST(PairingTest, Frobenius) {
  FqX2 a = FqX2::Random();

  // Frobenius^1 applied twice on Fq2 should be identity (since degree is 2)
  FqX2 a_frob = a.template Frobenius<1>();
  FqX2 a_frob2 = a_frob.template Frobenius<1>();
  EXPECT_EQ(a_frob2, a);

  // Check FqX2 Frobenius coefficient: should be -1 since u^((q - 1) / 2) = -1.
  {
    const auto& coeffs = FqX2::One().GetRelativeFrobeniusCoeffs();
    EXPECT_EQ(coeffs[0][0], -Fq::One());
  }

  // Check FqX12 relative Frobenius coefficient: should be -1 since
  // v^((q⁶ - 1) / 2) = -1.
  {
    const auto& coeffs = FqX12::One().GetRelativeFrobeniusCoeffs();
    EXPECT_EQ(coeffs[0][0], -FqX6::One());
  }

  // Verify v³ = ξ = (9 + u) in FqX6.
  {
    FqX6 v = FqX12::One().NonResidue();
    FqX2 xi({Fq(9), Fq(1)});
    EXPECT_EQ(v * v * v, FqX6({xi, FqX2::Zero(), FqX2::Zero()}));
  }
}

TEST(PairingTest, TwoInv) {
  Fq two = Fq(2);
  Fq two_inv = Fq::TwoInv();
  EXPECT_EQ(two * two_inv, Fq::One());
}

TEST(PairingTest, FusedMultiMillerLoop) {
  using G1Affine = G1Curve::AffinePoint;
  using G1Jacobian = G1Curve::JacobianPoint;
  using G2Affine = G2Curve::AffinePoint;

  Fr a = Fr::Random();
  Fr b = Fr::Random();

  G1Affine g1_a = (G1Jacobian::Generator() * a).ToAffine();
  G1Affine g1_b = (G1Jacobian::Generator() * b).ToAffine();
  G1Affine neg_g1_a = -(G1Jacobian::Generator() * a).ToAffine();
  G2Affine g2 = G2Affine::Generator();

  // 2 pairs: e(a*G1, G2) * e(-(a)*G1, G2) = 1
  {
    std::array<G1Affine, 2> g1_pts = {g1_a, neg_g1_a};
    std::array<G2Affine, 2> g2_pts = {g2, g2};
    auto f = BN254Curve::FusedMultiMillerLoop<2>(g1_pts, g2_pts);
    auto result = BN254Curve::FinalExponentiation(f);
    EXPECT_TRUE(result.IsOne());
  }

  // 4 pairs: e(a*G1, G2) * e(b*G1, G2) * e(-(a+b)*G1, G2) * e(0, G2) = 1
  // Use e(a*G1, G2) * e(b*G1, G2) * e(-a*G1, G2) * e(-b*G1, G2) = 1
  {
    G1Affine neg_g1_b = -(G1Jacobian::Generator() * b).ToAffine();
    std::array<G1Affine, 4> g1_pts = {g1_a, g1_b, neg_g1_a, neg_g1_b};
    std::array<G2Affine, 4> g2_pts = {g2, g2, g2, g2};
    auto f = BN254Curve::FusedMultiMillerLoop<4>(g1_pts, g2_pts);
    auto result = BN254Curve::FinalExponentiation(f);
    EXPECT_TRUE(result.IsOne());
  }
}

}  // namespace
}  // namespace zk_dtypes::bn254
