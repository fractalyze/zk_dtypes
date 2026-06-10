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

#include "zk_dtypes/include/field/goldilocks/goldilocksx3.h"

#include "gtest/gtest.h"

namespace zk_dtypes {
namespace {

// The byte-match vectors below were generated from pil2-proofman v0.18.0's
// `fields` crate (`CubicExtensionField<Goldilocks>` — the exact arithmetic
// the ZisK prover runs) on the fixed inputs in `A` and `B`:
// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/fields/src/extended_field.rs
constexpr uint64_t A[3] = {UINT64_C(0xFFFFFFFF00000000),
                           UINT64_C(12345678901234567), UINT64_C(3)};
constexpr uint64_t B[3] = {UINT64_C(987654321), UINT64_C(0xFFFFFFFEFFFFFFFF),
                           UINT64_C(0xDEADBEEFCAFEBABE)};

GoldilocksX3 FromU64(const uint64_t (&v)[3]) {
  return GoldilocksX3({Goldilocks(v[0]), Goldilocks(v[1]), Goldilocks(v[2])});
}

GoldilocksX3Mont FromU64Mont(const uint64_t (&v)[3]) {
  return GoldilocksX3Mont(
      {GoldilocksMont(v[0]), GoldilocksMont(v[1]), GoldilocksMont(v[2])});
}

TEST(GoldilocksX3Test, ModulusIsXCubedMinusXMinusOne) {
  // u³ must reduce to 1 + u — the pil2 `Goldilocks3` convention, and the
  // single fact that distinguishes this field from GoldilocksX3 (u³ = 7).
  GoldilocksX3 u({Goldilocks(0), Goldilocks(1), Goldilocks(0)});
  GoldilocksX3 expected({Goldilocks(1), Goldilocks(1), Goldilocks(0)});
  EXPECT_EQ(u * u * u, expected);
}

TEST(GoldilocksX3Test, MulMatchesPil2) {
  constexpr uint64_t kExpected[3] = {UINT64_C(16583703172221849613),
                                     UINT64_C(11556631869400146760),
                                     UINT64_C(13619946544752105600)};
  EXPECT_EQ(FromU64(A) * FromU64(B), FromU64(kExpected));
  EXPECT_EQ((FromU64Mont(A) * FromU64Mont(B)).MontReduce(), FromU64(kExpected));
}

TEST(GoldilocksX3Test, SquareMatchesPil2) {
  constexpr uint64_t kExpectedA[3] = {UINT64_C(74074073407407403),
                                      UINT64_C(49382715604938277),
                                      UINT64_C(18132399027456170824)};
  constexpr uint64_t kExpectedB[3] = {UINT64_C(10579673397435916141),
                                      UINT64_C(13434714062957531168),
                                      UINT64_C(9427914128801059483)};
  EXPECT_EQ(FromU64(A).Square(), FromU64(kExpectedA));
  EXPECT_EQ(FromU64(B).Square(), FromU64(kExpectedB));
  EXPECT_EQ(FromU64Mont(A).Square().MontReduce(), FromU64(kExpectedA));
  EXPECT_EQ(FromU64Mont(B).Square().MontReduce(), FromU64(kExpectedB));
}

TEST(GoldilocksX3Test, InverseMatchesPil2) {
  constexpr uint64_t kExpectedA[3] = {UINT64_C(7422903702349599745),
                                      UINT64_C(4764441002090916551),
                                      UINT64_C(17439373525960107448)};
  constexpr uint64_t kExpectedB[3] = {UINT64_C(17210005891561599323),
                                      UINT64_C(11708350379886342939),
                                      UINT64_C(2232599005831579362)};
  EXPECT_EQ(FromU64(A).Inverse(), FromU64(kExpectedA));
  EXPECT_EQ(FromU64(B).Inverse(), FromU64(kExpectedB));
  EXPECT_EQ(FromU64Mont(A).Inverse().MontReduce(), FromU64(kExpectedA));
  EXPECT_EQ(FromU64Mont(B).Inverse().MontReduce(), FromU64(kExpectedB));
}

TEST(GoldilocksX3Test, FieldProperties) {
  for (int i = 0; i < 100; ++i) {
    GoldilocksX3 a = GoldilocksX3::Random();
    GoldilocksX3 b = GoldilocksX3::Random();
    GoldilocksX3 c = GoldilocksX3::Random();
    EXPECT_EQ(a * b, b * a);
    EXPECT_EQ((a * b) * c, a * (b * c));
    EXPECT_EQ(a * (b + c), a * b + a * c);
    EXPECT_EQ(a.Square(), a * a);
    if (!a.IsZero()) {
      EXPECT_TRUE((a * a.Inverse()).IsOne());
    }
  }
}

TEST(GoldilocksX3Test, InverseOfZeroIsZero) {
  EXPECT_TRUE(GoldilocksX3::Zero().Inverse().IsZero());
  EXPECT_TRUE(GoldilocksX3Mont::Zero().Inverse().IsZero());
}

TEST(GoldilocksX3Test, MontStdConsistency) {
  for (int i = 0; i < 100; ++i) {
    GoldilocksX3Mont a = GoldilocksX3Mont::Random();
    GoldilocksX3Mont b = GoldilocksX3Mont::Random();
    EXPECT_EQ((a * b).MontReduce(), a.MontReduce() * b.MontReduce());
    EXPECT_EQ(a.Square().MontReduce(), a.MontReduce().Square());
    EXPECT_EQ(a.Inverse().MontReduce(), a.MontReduce().Inverse());
  }
}

}  // namespace
}  // namespace zk_dtypes
