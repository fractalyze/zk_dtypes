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

// Test curve: -x² + y² = 1 + 2*x²*y² over F_13
// Identity: (0, 1). Generator: (2, 4). Group order: 16.

TEST(TeAffinePointTest, Traits) {
  static_assert(!IsComparable<TeAffinePoint>);
  static_assert(IsAdditiveGroup<TeAffinePoint>);
  static_assert(IsEcPoint<TeAffinePoint>);
  static_assert(IsAffinePoint<TeAffinePoint>);
  static_assert(!IsExtendedPoint<TeAffinePoint>);
}

TEST(TeAffinePointTest, Zero) {
  // Identity on twisted Edwards is (0, 1), not (0, 0).
  TeAffinePoint zero = TeAffinePoint::Zero();
  EXPECT_TRUE(zero.IsZero());
  EXPECT_EQ(zero.x(), TeTestFq(0));
  EXPECT_EQ(zero.y(), TeTestFq(1));
}

TEST(TeAffinePointTest, Generator) {
  TeAffinePoint gen = TeAffinePoint::Generator();
  EXPECT_EQ(gen.x(), TeTestFq(2));
  EXPECT_EQ(gen.y(), TeTestFq(4));
  EXPECT_TRUE(gen.IsOne());
}

TEST(TeAffinePointTest, Equality) {
  TeAffinePoint p(2, 4);
  TeAffinePoint p2(10, 11);
  EXPECT_EQ(p, p);
  EXPECT_NE(p, p2);
}

TEST(TeAffinePointTest, Negate) {
  // For twisted Edwards: -(x, y) = (-x, y).
  TeAffinePoint gen(2, 4);
  TeAffinePoint neg_gen = -gen;
  EXPECT_EQ(neg_gen, TeAffinePoint(11, 4));  // -2 mod 13 = 11

  // Identity negation: -(0, 1) = (0, 1).
  EXPECT_EQ(-TeAffinePoint::Zero(), TeAffinePoint::Zero());
}

TEST(TeAffinePointTest, AffineAddition) {
  TeAffinePoint g(2, 4);
  TeExtendedPoint g2_ext = g + g;
  TeAffinePoint g2 = g2_ext.ToAffine();
  EXPECT_EQ(g2, TeAffinePoint(10, 11));  // 2*G

  TeExtendedPoint g3_ext = g + g2_ext;
  TeAffinePoint g3 = g3_ext.ToAffine();
  EXPECT_EQ(g3, TeAffinePoint(6, 10));  // 3*G
}

TEST(TeAffinePointTest, AffineSubtraction) {
  TeAffinePoint g(2, 4);
  TeAffinePoint g2(10, 11);
  TeAffinePoint g3(6, 10);
  TeExtendedPoint result = g3 - g2;
  EXPECT_EQ(result.ToAffine(), g);  // 3G - 2G = G
}

TEST(TeAffinePointTest, IdentityAddition) {
  TeAffinePoint g(2, 4);
  TeAffinePoint zero = TeAffinePoint::Zero();
  // G + 0 should give G (via extended coords).
  TeExtendedPoint result = g + zero;
  EXPECT_EQ(result.ToAffine(), g);
}

TEST(TeAffinePointTest, ToExtended) {
  TeAffinePoint p(2, 4);
  TeExtendedPoint ext = p.ToExtended();
  EXPECT_EQ(ext.x(), TeTestFq(2));
  EXPECT_EQ(ext.y(), TeTestFq(4));
  EXPECT_EQ(ext.z(), TeTestFq(1));
  EXPECT_EQ(ext.t(), TeTestFq(2 * 4 % 13));  // T = x*y = 8
}

TEST(TeAffinePointTest, Double) {
  TeAffinePoint g(2, 4);
  TeExtendedPoint g2 = g.Double();
  EXPECT_EQ(g2.ToAffine(), TeAffinePoint(10, 11));  // 2*G
}

}  // namespace
}  // namespace zk_dtypes::test
