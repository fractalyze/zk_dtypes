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

#include "zk_dtypes/include/field/goldilocks/goldilocks3_pil.h"

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

Goldilocks3Pil FromU64(const uint64_t (&v)[3]) {
  return Goldilocks3Pil(
      {Goldilocks(v[0]), Goldilocks(v[1]), Goldilocks(v[2])});
}

Goldilocks3PilMont FromU64Mont(const uint64_t (&v)[3]) {
  return Goldilocks3PilMont(
      {GoldilocksMont(v[0]), GoldilocksMont(v[1]), GoldilocksMont(v[2])});
}

TEST(Goldilocks3PilTest, ModulusIsXCubedMinusXMinusOne) {
  // u³ must reduce to 1 + u — the pil2 `Goldilocks3` convention, and the
  // single fact that distinguishes this field from GoldilocksX3 (u³ = 7).
  Goldilocks3Pil u({Goldilocks(0), Goldilocks(1), Goldilocks(0)});
  Goldilocks3Pil expected({Goldilocks(1), Goldilocks(1), Goldilocks(0)});
  EXPECT_EQ(u * u * u, expected);
}

TEST(Goldilocks3PilTest, MulMatchesPil2) {
  constexpr uint64_t kExpected[3] = {UINT64_C(16583703172221849613),
                                     UINT64_C(11556631869400146760),
                                     UINT64_C(13619946544752105600)};
  EXPECT_EQ(FromU64(A) * FromU64(B), FromU64(kExpected));
  EXPECT_EQ((FromU64Mont(A) * FromU64Mont(B)).MontReduce(),
            FromU64(kExpected));
}

TEST(Goldilocks3PilTest, SquareMatchesPil2) {
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

TEST(Goldilocks3PilTest, InverseMatchesPil2) {
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

TEST(Goldilocks3PilTest, FieldProperties) {
  for (int i = 0; i < 100; ++i) {
    Goldilocks3Pil a = Goldilocks3Pil::Random();
    Goldilocks3Pil b = Goldilocks3Pil::Random();
    Goldilocks3Pil c = Goldilocks3Pil::Random();
    EXPECT_EQ(a * b, b * a);
    EXPECT_EQ((a * b) * c, a * (b * c));
    EXPECT_EQ(a * (b + c), a * b + a * c);
    EXPECT_EQ(a.Square(), a * a);
    if (!a.IsZero()) {
      EXPECT_TRUE((a * a.Inverse()).IsOne());
    }
  }
}

TEST(Goldilocks3PilTest, InverseOfZeroIsZero) {
  EXPECT_TRUE(Goldilocks3Pil::Zero().Inverse().IsZero());
  EXPECT_TRUE(Goldilocks3PilMont::Zero().Inverse().IsZero());
}

TEST(Goldilocks3PilTest, MontStdConsistency) {
  for (int i = 0; i < 100; ++i) {
    Goldilocks3PilMont a = Goldilocks3PilMont::Random();
    Goldilocks3PilMont b = Goldilocks3PilMont::Random();
    EXPECT_EQ((a * b).MontReduce(), a.MontReduce() * b.MontReduce());
    EXPECT_EQ(a.Square().MontReduce(), a.MontReduce().Square());
    EXPECT_EQ(a.Inverse().MontReduce(), a.MontReduce().Inverse());
  }
}

}  // namespace
}  // namespace zk_dtypes
