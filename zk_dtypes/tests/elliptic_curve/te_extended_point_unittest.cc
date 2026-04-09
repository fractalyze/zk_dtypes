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

#include "zk_dtypes/include/elliptic_curve/twisted_edwards/test/te_curve_config.h"

namespace zk_dtypes::test {
namespace {

// Test curve: -x² + y² = 1 + 2*x²*y² over F₁₃
// Identity (extended): (0, 1, 1, 0). Generator: (2, 4, 1, 8). Order: 16.

TEST(TeExtendedPointTest, Traits) {
  static_assert(!IsComparable<TeExtendedPoint>);
  static_assert(IsAdditiveGroup<TeExtendedPoint>);
  static_assert(IsEcPoint<TeExtendedPoint>);
  static_assert(IsExtendedPoint<TeExtendedPoint>);
  static_assert(!IsAffinePoint<TeExtendedPoint>);
}

TEST(TeExtendedPointTest, Zero) {
  TeExtendedPoint zero = TeExtendedPoint::Zero();
  EXPECT_TRUE(zero.IsZero());
  EXPECT_EQ(zero.x(), TeTestFq(0));
  EXPECT_EQ(zero.y(), TeTestFq(1));
  EXPECT_EQ(zero.z(), TeTestFq(1));
  EXPECT_EQ(zero.t(), TeTestFq(0));
}

TEST(TeExtendedPointTest, Generator) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  EXPECT_EQ(gen.x(), TeTestFq(2));
  EXPECT_EQ(gen.y(), TeTestFq(4));
  EXPECT_EQ(gen.z(), TeTestFq(1));
  EXPECT_EQ(gen.t(), TeTestFq(8));  // 2 * 4 = 8
}

TEST(TeExtendedPointTest, Negate) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint neg_gen = -gen;
  // -(X, Y, Z, T) = (-X, Y, Z, -T)
  EXPECT_EQ(neg_gen.x(), TeTestFq(11));  // -2 mod 13
  EXPECT_EQ(neg_gen.y(), TeTestFq(4));
  EXPECT_EQ(neg_gen.z(), TeTestFq(1));
  EXPECT_EQ(neg_gen.t(), TeTestFq(5));  // -8 mod 13
}

TEST(TeExtendedPointTest, Addition) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint g2 = gen + gen;
  EXPECT_EQ(g2.ToAffine(), TeAffinePoint(10, 11));  // 2*G

  TeExtendedPoint g3 = g2 + gen;
  EXPECT_EQ(g3.ToAffine(), TeAffinePoint(6, 10));  // 3*G

  TeExtendedPoint g4 = g3 + gen;
  EXPECT_EQ(g4.ToAffine(), TeAffinePoint(8, 0));  // 4*G
}

TEST(TeExtendedPointTest, Double) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint g2 = gen.Double();
  EXPECT_EQ(g2.ToAffine(), TeAffinePoint(10, 11));

  TeExtendedPoint g4 = g2.Double();
  EXPECT_EQ(g4.ToAffine(), TeAffinePoint(8, 0));  // 4*G

  TeExtendedPoint g8 = g4.Double();
  EXPECT_EQ(g8.ToAffine(), TeAffinePoint(0, 12));  // 8*G = (0, -1)
}

TEST(TeExtendedPointTest, IdentityProperties) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint zero = TeExtendedPoint::Zero();

  // G + 0 = G
  EXPECT_EQ((gen + zero).ToAffine(), gen.ToAffine());

  // 0 + G = G
  EXPECT_EQ((zero + gen).ToAffine(), gen.ToAffine());

  // G + (-G) = 0
  TeExtendedPoint sum = gen + (-gen);
  EXPECT_TRUE(sum.IsZero());
}

TEST(TeExtendedPointTest, Subtraction) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint g3 = gen + gen + gen;
  TeExtendedPoint g2 = gen.Double();

  TeExtendedPoint result = g3 - g2;
  EXPECT_EQ(result.ToAffine(), gen.ToAffine());  // 3G - 2G = G
}

TEST(TeExtendedPointTest, Equality) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint gen2 = TeExtendedPoint::Generator();
  EXPECT_EQ(gen, gen2);

  // Same point in different projective coordinates.
  TeExtendedPoint g2a = gen + gen;
  TeExtendedPoint g2b = gen.Double();
  EXPECT_EQ(g2a, g2b);
}

TEST(TeExtendedPointTest, GroupOrderIs16) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeExtendedPoint current = gen;
  for (int i = 1; i < 16; ++i) {
    EXPECT_FALSE(current.IsZero()) << "Order divides " << i;
    current = current + gen;
  }
  EXPECT_TRUE(current.IsZero()) << "16*G should be identity";
}

TEST(TeExtendedPointTest, ToAffineRoundTrip) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeAffinePoint affine = gen.ToAffine();
  TeExtendedPoint back = affine.ToExtended();
  EXPECT_EQ(gen, back);
}

TEST(TeExtendedPointTest, MixedAddition) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  TeAffinePoint gen_aff = TeAffinePoint::Generator();

  // Extended + Affine
  TeExtendedPoint g2_via_mixed = gen + gen_aff;
  TeExtendedPoint g2_via_ext = gen + gen;
  EXPECT_EQ(g2_via_mixed, g2_via_ext);
}

TEST(TeExtendedPointTest, BatchToAffine) {
  TeExtendedPoint gen = TeExtendedPoint::Generator();
  std::vector<TeExtendedPoint> extended_points;
  TeExtendedPoint current = TeExtendedPoint::Zero();
  for (int i = 0; i < 4; ++i) {
    extended_points.push_back(current);
    current = current + gen;
  }

  std::vector<TeAffinePoint> affine_points;
  ASSERT_TRUE(
      TeExtendedPoint::BatchToAffine(extended_points, &affine_points).ok());
  ASSERT_EQ(affine_points.size(), 4);
  EXPECT_EQ(affine_points[0], TeAffinePoint(0, 1));    // 0*G = identity
  EXPECT_EQ(affine_points[1], TeAffinePoint(2, 4));    // 1*G
  EXPECT_EQ(affine_points[2], TeAffinePoint(10, 11));  // 2*G
  EXPECT_EQ(affine_points[3], TeAffinePoint(6, 10));   // 3*G
}

}  // namespace
}  // namespace zk_dtypes::test
