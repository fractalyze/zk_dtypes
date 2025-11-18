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

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"

namespace zk_dtypes::test {
namespace {

TEST(JacobianPointTest, Zero) {
  EXPECT_TRUE(JacobianPoint::Zero().IsZero());
  EXPECT_FALSE(JacobianPoint(1, 2, 1).IsZero());
}

TEST(JacobianPointTest, One) {
  auto generator = JacobianPoint::Generator();
  EXPECT_EQ(generator, JacobianPoint(JacobianPoint::Curve::Config::kX,
                                     JacobianPoint::Curve::Config::kY, 1));
  EXPECT_EQ(JacobianPoint::Generator(), JacobianPoint::One());
  EXPECT_TRUE(generator.IsOne());
}

TEST(JacobianPointTest, EqualityOperations) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    JacobianPoint p(1, 2, 0);
    JacobianPoint p2(3, 4, 0);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    JacobianPoint p(1, 2, 1);
    JacobianPoint p2(3, 4, 0);
    EXPECT_NE(p, p2);
    EXPECT_NE(p2, p);
  }

  {
    SCOPED_TRACE("other");
    JacobianPoint p(1, 2, 3);
    JacobianPoint p2(1, 2, 3);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }
}

TEST(JacobianPointTest, GroupOperations) {
  JacobianPoint p(5, 5, 1);
  JacobianPoint p2(3, 2, 1);
  JacobianPoint p3(3, 5, 1);
  JacobianPoint p4(6, 5, 1);
  absl::StatusOr<AffinePoint> ap = p.ToAffine();
  ASSERT_TRUE(ap.ok());
  absl::StatusOr<AffinePoint> ap2 = p2.ToAffine();
  ASSERT_TRUE(ap2.ok());

  EXPECT_EQ(p + p2, p3);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p + p, p4);
  EXPECT_EQ(p - p4, -p);

  {
    JacobianPoint p_tmp = p;
    p_tmp += p2;
    EXPECT_EQ(p_tmp, p3);
    p_tmp -= p2;
    EXPECT_EQ(p_tmp, p);
  }

  EXPECT_EQ(p + (*ap2), p3);
  EXPECT_EQ(p + (*ap), p4);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p - p4, -p);

  EXPECT_EQ(p.Double(), p4);

  EXPECT_EQ(-p, JacobianPoint(5, 2, 1));

  EXPECT_EQ(p * 2, p4);
  EXPECT_EQ(Fr(2) * p, p4);
  EXPECT_EQ(p *= 2, p4);
}

TEST(JacobianPointTest, CyclicScalarMul) {
  std::vector<AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    absl::StatusOr<AffinePoint> ap =
        (Fr(i) * JacobianPoint::Generator()).ToAffine();
    ASSERT_TRUE(ap.ok());
    points.push_back(*ap);
  }

  EXPECT_THAT(points,
              testing::UnorderedElementsAreArray(std::vector<AffinePoint>{
                  AffinePoint(0, 0),
                  AffinePoint(3, 2),
                  AffinePoint(5, 2),
                  AffinePoint(6, 2),
                  AffinePoint(3, 5),
                  AffinePoint(5, 5),
                  AffinePoint(6, 5),
              }));
}

TEST(JacobianPointTest, ToAffine) {
  JacobianPoint p(1, 2, 0);
  JacobianPoint p2(1, 2, 1);
  JacobianPoint p3(1, 2, 3);
  absl::StatusOr<AffinePoint> ap = p.ToAffine();
  ASSERT_TRUE(ap.ok());
  absl::StatusOr<AffinePoint> ap2 = p2.ToAffine();
  ASSERT_TRUE(ap2.ok());
  absl::StatusOr<AffinePoint> ap3 = p3.ToAffine();
  ASSERT_TRUE(ap3.ok());
  EXPECT_EQ(*ap, AffinePoint::Zero());
  EXPECT_EQ(*ap2, AffinePoint(1, 2));
  EXPECT_EQ(*ap3, AffinePoint(4, 5));
}

TEST(JacobianPointTest, ToXyzz) {
  auto p = JacobianPoint::Random();
  EXPECT_EQ(p.ToXyzz(), p.ToAffine()->ToXyzz());
}

TEST(JacobianPointTest, BatchToAffine) {
  std::vector<JacobianPoint> jacobian_points = {
      JacobianPoint(1, 2, 0),
      JacobianPoint(1, 2, 1),
      JacobianPoint(1, 2, 3),
  };

  absl::Span<AffinePoint> affine_points_span;
  ASSERT_FALSE(
      JacobianPoint::BatchToAffine(jacobian_points, &affine_points_span).ok());

  std::vector<AffinePoint> affine_points;
  ASSERT_TRUE(
      JacobianPoint::BatchToAffine(jacobian_points, &affine_points).ok());

  std::vector<AffinePoint> expected_affine_points = {
      AffinePoint::Zero(), AffinePoint(1, 2), AffinePoint(4, 5)};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST(JacobianPointTest, MontReduce) {
  JacobianPoint p(3, 2, 1);
  JacobianPoint::StdType reduced = p.MontReduce();

  EXPECT_EQ(reduced.x(), Fq(3).MontReduce());
  EXPECT_EQ(reduced.y(), Fq(2).MontReduce());
  EXPECT_EQ(reduced.z(), Fq(1).MontReduce());
}

}  // namespace
}  // namespace zk_dtypes::test
