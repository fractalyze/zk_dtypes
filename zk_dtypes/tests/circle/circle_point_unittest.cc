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

#include "zk_dtypes/include/circle/circle_point.h"

#include "gtest/gtest.h"
#include "zk_dtypes/include/circle/circle_point_index.h"
#include "zk_dtypes/include/circle/m31_circle.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {
namespace {

TEST(CirclePointTest, Zero) {
  auto zero = CirclePoint<Mersenne31>::Zero();
  EXPECT_EQ(zero.x, Mersenne31::One());
  EXPECT_EQ(zero.y, Mersenne31::Zero());
}

TEST(CirclePointTest, GeneratorIsOnCircle) {
  auto gen = M31CircleGen();
  // Verify x² + y² = 1
  auto x_sq = gen.x.Square();
  auto y_sq = gen.y.Square();
  EXPECT_EQ(x_sq + y_sq, Mersenne31::One());
}

TEST(CirclePointTest, Addition) {
  auto gen = M31CircleGen();
  auto zero = CirclePoint<Mersenne31>::Zero();

  // Adding zero should return the same point.
  auto p = gen + zero;
  EXPECT_EQ(p, gen);

  // Adding a point to itself (doubling).
  auto p2 = gen + gen;
  EXPECT_EQ(p2, gen.Double());
}

TEST(CirclePointTest, Negation) {
  auto gen = M31CircleGen();
  auto neg_gen = -gen;

  // Negation should flip y-coordinate.
  EXPECT_EQ(neg_gen.x, gen.x);
  EXPECT_EQ(neg_gen.y, -gen.y);

  // Adding a point to its negation should give zero.
  auto zero = gen + neg_gen;
  EXPECT_EQ(zero, CirclePoint<Mersenne31>::Zero());
}

TEST(CirclePointTest, DoubleX) {
  auto gen = M31CircleGen();
  auto doubled = gen.Double();

  // DoubleX(x) should equal the x-coordinate of the doubled point.
  auto double_x = CirclePoint<Mersenne31>::DoubleX(gen.x);
  EXPECT_EQ(double_x, doubled.x);
}

TEST(CirclePointTest, ScalarMultiplication) {
  auto gen = M31CircleGen();
  auto zero = CirclePoint<Mersenne31>::Zero();

  // Multiplying by 0 should give zero.
  EXPECT_EQ(gen.Mul(0), zero);

  // Multiplying by 1 should give the same point.
  EXPECT_EQ(gen.Mul(1), gen);

  // Multiplying by 2 should give the doubled point.
  EXPECT_EQ(gen.Mul(2), gen.Double());

  // Multiplying by 3 should give gen + gen + gen.
  EXPECT_EQ(gen.Mul(3), gen + gen + gen);
}

TEST(CirclePointTest, RepeatedDouble) {
  auto gen = M31CircleGen();

  // Repeated doubling 0 times should give the same point.
  EXPECT_EQ(gen.RepeatedDouble(0), gen);

  // Repeated doubling 1 time should give the doubled point.
  EXPECT_EQ(gen.RepeatedDouble(1), gen.Double());

  // Repeated doubling n times should be equivalent to multiplying by 2^n.
  EXPECT_EQ(gen.RepeatedDouble(3), gen.Mul(8));
}

TEST(CirclePointTest, GeneratorOrder) {
  auto gen = M31CircleGen();

  // Adding the generator to itself 2^30 times should NOT yield the identity.
  auto circle_point = gen.RepeatedDouble(30);
  EXPECT_NE(circle_point, CirclePoint<Mersenne31>::Zero());

  // Adding the generator to itself 2^31 times should yield the identity.
  circle_point = gen.RepeatedDouble(31);
  EXPECT_EQ(circle_point, CirclePoint<Mersenne31>::Zero());
}

TEST(CirclePointTest, LogOrder) {
  auto gen = M31CircleGen();
  EXPECT_EQ(gen.LogOrder(), kM31CircleLogOrder);

  // The identity has order 2^0 = 1.
  EXPECT_EQ(CirclePoint<Mersenne31>::Zero().LogOrder(), 0u);

  // The doubled generator has order 2^30.
  EXPECT_EQ(gen.Double().LogOrder(), 30u);
}

TEST(CirclePointTest, Conjugate) {
  auto gen = M31CircleGen();
  auto conj = gen.Conjugate();

  EXPECT_EQ(conj.x, gen.x);
  EXPECT_EQ(conj.y, -gen.y);
  EXPECT_EQ(conj, -gen);
}

TEST(CirclePointTest, Antipode) {
  auto gen = M31CircleGen();
  auto anti = gen.Antipode();

  EXPECT_EQ(anti.x, -gen.x);
  EXPECT_EQ(anti.y, -gen.y);
}

}  // namespace
}  // namespace zk_dtypes
