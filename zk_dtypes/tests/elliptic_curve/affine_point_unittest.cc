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

TEST(AffinePointTest, Zero) {
  EXPECT_TRUE(AffinePoint(0).IsZero());
  EXPECT_TRUE(AffinePoint::Zero().IsZero());
  EXPECT_FALSE(AffinePoint(1, 2).IsZero());
}

TEST(AffinePointTest, One) {
  auto generator = AffinePoint::Generator();
  EXPECT_EQ(generator, AffinePoint(AffinePoint::Curve::Config::kX,
                                   AffinePoint::Curve::Config::kY));
  EXPECT_EQ(AffinePoint::Generator(), AffinePoint::One());
  EXPECT_TRUE(generator.IsOne());
  EXPECT_TRUE(AffinePoint(1).IsOne());
  EXPECT_TRUE(AffinePoint::One().IsOne());
}

TEST(AffinePointTest, EqualityOperations) {
  AffinePoint p(1, 2);
  AffinePoint p2(3, 4);
  EXPECT_EQ(p, p);
  EXPECT_NE(p, p2);
}

TEST(AffinePointTest, GroupOperations) {
  AffinePoint ap(5, 5);
  AffinePoint ap2(3, 2);
  AffinePoint ap3(3, 5);
  AffinePoint ap4(6, 5);
  JacobianPoint jp = ap.ToJacobian();
  JacobianPoint jp2 = ap2.ToJacobian();
  JacobianPoint jp3 = ap3.ToJacobian();
  JacobianPoint jp4 = ap4.ToJacobian();
  PointXyzz xp = ap.ToXyzz();
  PointXyzz xp2 = ap2.ToXyzz();
  PointXyzz xp3 = ap3.ToXyzz();
  PointXyzz xp4 = ap4.ToXyzz();

  EXPECT_EQ(ap + ap2, jp3);
  EXPECT_EQ(ap + ap, jp4);
  EXPECT_EQ(ap3 - ap2, jp);
  EXPECT_EQ(ap4 - ap, jp);

  EXPECT_EQ(ap + jp2, jp3);
  EXPECT_EQ(ap + jp, jp4);
  EXPECT_EQ(ap - jp3, -jp2);
  EXPECT_EQ(ap - jp4, -jp);

  EXPECT_EQ(xp + xp2, xp3);
  EXPECT_EQ(xp + xp, xp4);
  EXPECT_EQ(xp - xp3, -xp2);
  EXPECT_EQ(xp - xp4, -xp);

  EXPECT_EQ(-ap, AffinePoint(5, 2));

  EXPECT_EQ(ap * 2, jp4);
  EXPECT_EQ(Fr(2) * ap, jp4);
}

TEST(AffinePointTest, CyclicScalarMul) {
  std::vector<AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    absl::StatusOr<AffinePoint> ap =
        (Fr(i) * AffinePoint::Generator()).ToAffine();
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

TEST(AffinePointTest, ToJacobian) {
  EXPECT_EQ(AffinePoint::Zero().ToJacobian(), JacobianPoint::Zero());
  AffinePoint p(3, 2);
  EXPECT_EQ(p.ToJacobian(), JacobianPoint(3, 2, 1));
}

TEST(AffinePointTest, ToPointXyzz) {
  EXPECT_EQ(AffinePoint::Zero().ToXyzz(), PointXyzz::Zero());
  AffinePoint p(3, 2);
  EXPECT_EQ(p.ToXyzz(), PointXyzz(3, 2, 1, 1));
}

TEST(AffinePointTest, MontReduce) {
  AffinePoint p(3, 2);
  AffinePoint::StdType reduced = p.MontReduce();

  EXPECT_EQ(reduced.x(), Fq(3).MontReduce());
  EXPECT_EQ(reduced.y(), Fq(2).MontReduce());
}

}  // namespace
}  // namespace zk_dtypes::test
