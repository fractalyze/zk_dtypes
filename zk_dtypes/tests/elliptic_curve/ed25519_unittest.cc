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

#include "zk_dtypes/include/elliptic_curve/curve25519/ed25519/g1.h"

namespace zk_dtypes::ed25519 {
namespace {

using Ap = G1AffinePoint;
using Ep = G1ExtendedPoint;
using ApMont = G1AffinePointMont;
using EpMont = G1ExtendedPointMont;

TEST(Ed25519Test, IdentityIsOnCurve) {
  Ap id = Ap::Zero();
  EXPECT_EQ(id.x(), Fq(0));
  EXPECT_EQ(id.y(), Fq(1));
  EXPECT_TRUE(id.IsZero());
}

TEST(Ed25519Test, GeneratorIsOnCurve) {
  Fq gx = G1TeCurveConfig::kX;
  Fq gy = G1TeCurveConfig::kY;
  Fq a = G1TeCurveConfig::kA;
  Fq d = G1TeCurveConfig::kD;
  Fq lhs = a * gx.Square() + gy.Square();
  Fq rhs = Fq::One() + d * gx.Square() * gy.Square();
  EXPECT_EQ(lhs, rhs);
}

TEST(Ed25519Test, GeneratorMontIsOnCurve) {
  FqMont gx = G1TeCurveMontConfig::kX;
  FqMont gy = G1TeCurveMontConfig::kY;
  FqMont a = G1TeCurveMontConfig::kA;
  FqMont d = G1TeCurveMontConfig::kD;
  FqMont lhs = a * gx.Square() + gy.Square();
  FqMont rhs = FqMont::One() + d * gx.Square() * gy.Square();
  EXPECT_EQ(lhs, rhs);
}

TEST(Ed25519Test, IdentityAddition) {
  Ep gen = Ep::Generator();
  Ep zero = Ep::Zero();
  EXPECT_EQ((gen + zero).ToAffine(), gen.ToAffine());
  EXPECT_EQ((zero + gen).ToAffine(), gen.ToAffine());
}

TEST(Ed25519Test, InverseProperty) {
  Ep gen = Ep::Generator();
  Ep neg_gen = -gen;
  Ep sum = gen + neg_gen;
  EXPECT_TRUE(sum.IsZero());
}

TEST(Ed25519Test, DoubleEqualsAddSelf) {
  Ep gen = Ep::Generator();
  Ep g2_add = gen + gen;
  Ep g2_dbl = gen.Double();
  EXPECT_EQ(g2_add, g2_dbl);
}

TEST(Ed25519Test, AdditionAssociativity) {
  Ep g = Ep::Generator();
  Ep g2 = g.Double();
  Ep g3 = g2 + g;
  EXPECT_EQ(g3, g + g2);
}

TEST(Ed25519Test, MontNonMontConsistency) {
  Ep g = Ep::Generator();
  Ep g2 = g.Double();
  Ap g2_aff = g2.ToAffine();

  EpMont gm = EpMont::Generator();
  EpMont g2m = gm.Double();
  ApMont g2m_aff = g2m.ToAffine();

  EXPECT_EQ(g2_aff.x(), g2m_aff.MontReduce().x());
  EXPECT_EQ(g2_aff.y(), g2m_aff.MontReduce().y());
}

TEST(Ed25519Test, FourGEqualsDoubleTwice) {
  Ep g = Ep::Generator();
  Ep g4_via_double = g.Double().Double();
  Ep g4_via_add = g + g + g + g;
  EXPECT_EQ(g4_via_double, g4_via_add);
}

}  // namespace
}  // namespace zk_dtypes::ed25519
