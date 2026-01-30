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

#include "zk_dtypes/include/circle/coset.h"

#include <vector>

#include "gtest/gtest.h"
#include "zk_dtypes/include/circle/circle_point.h"
#include "zk_dtypes/include/circle/circle_point_index.h"
#include "zk_dtypes/include/circle/m31_circle.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {
namespace {

TEST(CosetTest, Subgroup) {
  constexpr uint32_t log_size = 3;
  auto coset = Coset::Subgroup(log_size);

  EXPECT_EQ(coset.LogSize(), log_size);
  EXPECT_EQ(coset.Size(), 1ULL << log_size);
  EXPECT_EQ(coset.initial_index, CirclePointIndex::Zero());
}

TEST(CosetTest, Odds) {
  constexpr uint32_t log_size = 3;
  auto coset = Coset::Odds(log_size);

  EXPECT_EQ(coset.LogSize(), log_size);
  EXPECT_EQ(coset.Size(), 1ULL << log_size);
  EXPECT_EQ(coset.initial_index, CirclePointIndex::SubgroupGen(log_size + 1));
}

TEST(CosetTest, HalfOdds) {
  constexpr uint32_t log_size = 3;
  auto coset = Coset::HalfOdds(log_size);

  EXPECT_EQ(coset.LogSize(), log_size);
  EXPECT_EQ(coset.Size(), 1ULL << log_size);
  EXPECT_EQ(coset.initial_index, CirclePointIndex::SubgroupGen(log_size + 2));
}

TEST(CosetTest, Iterator) {
  auto coset = Coset(CirclePointIndex(1), 3);

  std::vector<CirclePointIndex> expected_indices = {
      CirclePointIndex(1),
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 1,
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 2,
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 3,
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 4,
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 5,
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 6,
      CirclePointIndex(1) + CirclePointIndex::SubgroupGen(3) * 7,
  };

  std::vector<CirclePointIndex> actual_indices;
  for (auto it = coset.IndicesBegin(); it != coset.IndicesEnd(); ++it) {
    actual_indices.push_back(*it);
  }
  EXPECT_EQ(actual_indices, expected_indices);

  std::vector<CirclePoint<Mersenne31>> actual_points;
  for (const auto& point : coset) {
    actual_points.push_back(point);
  }
  EXPECT_EQ(actual_points.size(), expected_indices.size());
  for (size_t i = 0; i < expected_indices.size(); ++i) {
    EXPECT_EQ(actual_points[i], ToPoint(expected_indices[i]));
  }
}

TEST(CosetTest, Double) {
  auto coset = Coset(CirclePointIndex(1), 3);
  auto doubled = coset.Double();

  EXPECT_EQ(doubled.LogSize(), coset.LogSize() - 1);
  EXPECT_EQ(doubled.Size(), coset.Size() / 2);
  EXPECT_EQ(doubled.initial, coset.initial.Double());
}

TEST(CosetTest, RepeatedDouble) {
  auto coset = Coset(CirclePointIndex(1), 5);
  auto doubled = coset.RepeatedDouble(3);

  EXPECT_EQ(doubled.LogSize(), 2u);
  EXPECT_EQ(doubled.Size(), 4u);
}

TEST(CosetTest, Conjugate) {
  auto coset = Coset(CirclePointIndex(1), 3);
  auto conj = coset.Conjugate();

  EXPECT_EQ(conj.initial_index, -coset.initial_index);
  EXPECT_EQ(conj.step_size, -coset.step_size);
  EXPECT_EQ(conj.LogSize(), coset.LogSize());
}

TEST(CosetTest, Shift) {
  auto coset = Coset(CirclePointIndex(1), 3);
  auto shifted = coset.Shift(CirclePointIndex(10));

  EXPECT_EQ(shifted.initial_index, CirclePointIndex(1) + CirclePointIndex(10));
  EXPECT_EQ(shifted.step_size, coset.step_size);
  EXPECT_EQ(shifted.LogSize(), coset.LogSize());
}

TEST(CosetTest, IndexAt) {
  auto coset = Coset(CirclePointIndex(1), 3);

  EXPECT_EQ(coset.IndexAt(0), coset.initial_index);
  EXPECT_EQ(coset.IndexAt(1), coset.initial_index + coset.step_size);
  EXPECT_EQ(coset.IndexAt(2), coset.initial_index + coset.step_size * 2);
}

TEST(CosetTest, At) {
  auto coset = Coset(CirclePointIndex(1), 3);

  EXPECT_EQ(coset.At(0), ToPoint(coset.IndexAt(0)));
  EXPECT_EQ(coset.At(1), ToPoint(coset.IndexAt(1)));
}

}  // namespace
}  // namespace zk_dtypes
