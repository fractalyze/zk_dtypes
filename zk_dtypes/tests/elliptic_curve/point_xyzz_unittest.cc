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

TEST(PointXyzzTest, Traits) {
  static_assert(!IsComparable<PointXyzz>);
  static_assert(IsAdditiveGroup<PointXyzz>);
  static_assert(IsEcPoint<PointXyzz>);
}

TEST(PointXyzzTest, Zero) {
  EXPECT_TRUE(PointXyzz(0).IsZero());
  EXPECT_TRUE(PointXyzz::Zero().IsZero());
  EXPECT_TRUE(PointXyzz(1, 2, 0, 0).IsZero());
  EXPECT_FALSE(PointXyzz(1, 2, 1, 0).IsZero());
  EXPECT_TRUE(PointXyzz(1, 2, 0, 1).IsZero());
}

TEST(PointXyzzTest, One) {
  auto generator = PointXyzz::Generator();
  EXPECT_EQ(generator, PointXyzz(PointXyzz::Curve::Config::kX,
                                 PointXyzz::Curve::Config::kY, 1, 1));
  EXPECT_EQ(PointXyzz::Generator(), PointXyzz::One());
  EXPECT_TRUE(generator.IsOne());
  EXPECT_TRUE(PointXyzz(1).IsOne());
  EXPECT_TRUE(PointXyzz::One().IsOne());
}

TEST(PointXyzzTest, EqualityOperations) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    PointXyzz p(1, 2, 0, 0);
    PointXyzz p2(3, 4, 0, 0);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    PointXyzz p(1, 2, 1, 0);
    PointXyzz p2(3, 4, 0, 0);
    EXPECT_NE(p, p2);
    EXPECT_NE(p2, p);
  }

  {
    SCOPED_TRACE("other");
    PointXyzz p(1, 2, 2, 6);
    PointXyzz p2(1, 2, 2, 6);
    EXPECT_EQ(p, p2);
    EXPECT_EQ(p2, p);
  }
}

TEST(PointXyzzTest, GroupOperations) {
  PointXyzz p(5, 5, 1, 1);
  PointXyzz p2(3, 2, 1, 1);
  PointXyzz p3(3, 5, 1, 1);
  PointXyzz p4(6, 5, 1, 1);
  AffinePoint ap = p.ToAffine();
  AffinePoint ap2 = p2.ToAffine();

  EXPECT_EQ(p + p2, p3);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p + p, p4);
  EXPECT_EQ(p - p4, -p);

  {
    PointXyzz p_tmp = p;
    p_tmp += p2;
    EXPECT_EQ(p_tmp, p3);
    p_tmp -= p2;
    EXPECT_EQ(p_tmp, p);
  }

  EXPECT_EQ(p + ap2, p3);
  EXPECT_EQ(p + ap, p4);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p - p4, -p);

  EXPECT_EQ(p.Double(), p4);

  EXPECT_EQ(-p, PointXyzz(5, 2, 1, 1));

  EXPECT_EQ(p * 2, p4);
  EXPECT_EQ(Fr(2) * p, p4);
  EXPECT_EQ(p *= 2, p4);
}

TEST(PointXyzzTest, CyclicScalarMul) {
  std::vector<AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    AffinePoint ap = (Fr(i) * PointXyzz::Generator()).ToAffine();
    points.push_back(ap);
  }

  EXPECT_THAT(points,
              testing::UnorderedElementsAreArray(std::vector<AffinePoint>{
                  AffinePoint(0, 0), AffinePoint(3, 2), AffinePoint(5, 2),
                  AffinePoint(6, 2), AffinePoint(3, 5), AffinePoint(5, 5),
                  AffinePoint(6, 5)}));
}

TEST(PointXyzzTest, ToAffine) {
  PointXyzz p(1, 2, 0, 0);
  PointXyzz p2(1, 2, 1, 1);
  PointXyzz p3(1, 2, 2, 6);
  AffinePoint ap = p.ToAffine();
  AffinePoint ap2 = p2.ToAffine();
  AffinePoint ap3 = p3.ToAffine();
  EXPECT_EQ(ap, AffinePoint::Zero());
  EXPECT_EQ(ap2, AffinePoint(1, 2));
  EXPECT_EQ(ap3, AffinePoint(4, 5));
}

TEST(PointXyzzTest, ToJacobian) {
  auto p = PointXyzz::Random();
  EXPECT_EQ(p.ToJacobian(), p.ToAffine().ToJacobian());
}

TEST(PointXyzzTest, BatchToAffine) {
  std::vector<PointXyzz> point_xyzzs = {
      PointXyzz(1, 2, 0, 0),
      PointXyzz(1, 2, 1, 1),
      PointXyzz(1, 2, 2, 6),
  };

  absl::Span<AffinePoint> affine_points_span;
  ASSERT_FALSE(PointXyzz::BatchToAffine(point_xyzzs, &affine_points_span).ok());

  std::vector<AffinePoint> affine_points;
  ASSERT_TRUE(PointXyzz::BatchToAffine(point_xyzzs, &affine_points).ok());

  std::vector<AffinePoint> expected_affine_points = {
      AffinePoint::Zero(), AffinePoint(1, 2), AffinePoint(4, 5)};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST(PointXyzzTest, MontReduce) {
  PointXyzzMont p(3, 2, 1, 1);
  PointXyzzMont::StdType reduced = p.MontReduce();

  EXPECT_EQ(reduced.x(), FqMont(3).MontReduce());
  EXPECT_EQ(reduced.y(), FqMont(2).MontReduce());
  EXPECT_EQ(reduced.zz(), FqMont(1).MontReduce());
  EXPECT_EQ(reduced.zzz(), FqMont(1).MontReduce());
}

}  // namespace
}  // namespace zk_dtypes::test
