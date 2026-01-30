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

#include "zk_dtypes/include/circle/circle_domain.h"

#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "zk_dtypes/include/circle/circle_point.h"
#include "zk_dtypes/include/circle/circle_point_index.h"
#include "zk_dtypes/include/circle/coset.h"
#include "zk_dtypes/include/circle/m31_circle.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {
namespace {

TEST(CanonicCosetTest, Creation) {
  constexpr uint32_t log_size = 8;
  CanonicCoset canonic_coset(log_size);

  EXPECT_EQ(canonic_coset.LogSize(), log_size);
  EXPECT_EQ(canonic_coset.Size(), 1ULL << log_size);
}

TEST(CanonicCosetTest, HalfCoset) {
  constexpr uint32_t log_size = 8;
  CanonicCoset canonic_coset(log_size);
  auto half_coset = canonic_coset.HalfCoset();

  EXPECT_EQ(half_coset.LogSize(), log_size - 1);
  EXPECT_EQ(half_coset.Size(), canonic_coset.Size() / 2);
}

TEST(CanonicCosetTest, CosetIsHalfCosetWithConjugate) {
  constexpr uint32_t log_size = 8;
  CanonicCoset canonic_coset(log_size);

  // Collect coset points.
  std::set<uint64_t> coset_point_hashes;
  for (size_t i = 0; i < canonic_coset.Size(); ++i) {
    auto point = canonic_coset.At(i);
    uint64_t hash =
        static_cast<uint64_t>(point.x.value()) << 32 | point.y.value();
    coset_point_hashes.insert(hash);
  }

  // Collect half coset points.
  std::set<uint64_t> half_coset_point_hashes;
  auto half_coset = canonic_coset.HalfCoset();
  for (const auto& point : half_coset) {
    uint64_t hash =
        static_cast<uint64_t>(point.x.value()) << 32 | point.y.value();
    half_coset_point_hashes.insert(hash);
  }

  // Collect half coset conjugate points.
  std::set<uint64_t> half_coset_conj_hashes;
  auto half_coset_conj = half_coset.Conjugate();
  for (const auto& point : half_coset_conj) {
    uint64_t hash =
        static_cast<uint64_t>(point.x.value()) << 32 | point.y.value();
    half_coset_conj_hashes.insert(hash);
  }

  // Half coset and its conjugate should be disjoint.
  std::set<uint64_t> intersection;
  std::set_intersection(half_coset_point_hashes.begin(),
                        half_coset_point_hashes.end(),
                        half_coset_conj_hashes.begin(),
                        half_coset_conj_hashes.end(),
                        std::inserter(intersection, intersection.begin()));
  EXPECT_TRUE(intersection.empty());

  // Union of half coset and its conjugate should be the coset.
  std::set<uint64_t> union_set;
  std::set_union(half_coset_point_hashes.begin(), half_coset_point_hashes.end(),
                 half_coset_conj_hashes.begin(), half_coset_conj_hashes.end(),
                 std::inserter(union_set, union_set.begin()));
  EXPECT_EQ(union_set, coset_point_hashes);
}

TEST(CircleDomainTest, IsCanonic) {
  // Canonic domain should pass the check.
  CanonicCoset canonic_coset(8);
  auto domain = canonic_coset.GetCircleDomain();
  EXPECT_TRUE(domain.IsCanonic());

  // Non-canonic domain should fail the check.
  auto half_coset = Coset(CirclePointIndex::Generator(), 4);
  auto not_canonic_domain = CircleDomain(half_coset);
  EXPECT_FALSE(not_canonic_domain.IsCanonic());
}

TEST(CircleDomainTest, Iterator) {
  auto domain = CircleDomain(Coset(CirclePointIndex::Generator(), 2));

  std::vector<CirclePoint<Mersenne31>> points;
  for (const auto& point : domain) {
    points.push_back(point);
  }

  EXPECT_EQ(points.size(), domain.Size());

  // First half should be from half_coset.
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(
        points[i],
        ToPoint(CirclePointIndex::Generator() +
                CirclePointIndex::SubgroupGen(2) * static_cast<size_t>(i)));
  }

  // Second half should be from negated half_coset indices.
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(points[i + 4],
              ToPoint(-(CirclePointIndex::Generator() +
                        CirclePointIndex::SubgroupGen(2) * i)));
  }
}

TEST(CircleDomainTest, AtCircleDomain) {
  CanonicCoset canonic_coset(7);
  auto domain = canonic_coset.GetCircleDomain();
  size_t half_domain_size = domain.Size() / 2;

  for (size_t i = 0; i < half_domain_size; ++i) {
    EXPECT_EQ(domain.IndexAt(i), -domain.IndexAt(i + half_domain_size));
    EXPECT_EQ(domain.At(i), domain.At(i + half_domain_size).Conjugate());
  }
}

TEST(CircleDomainTest, Split) {
  CanonicCoset canonic_coset(5);
  auto domain = canonic_coset.GetCircleDomain();
  auto [subdomain, shifts] = domain.Split(2);

  EXPECT_EQ(shifts.size(), 4u);

  // Collect domain points.
  std::vector<CirclePoint<Mersenne31>> domain_points;
  for (const auto& point : domain) {
    domain_points.push_back(point);
  }

  // Collect points from each shifted subdomain.
  std::vector<std::vector<CirclePoint<Mersenne31>>> points_for_each_domain;
  for (const auto& shift : shifts) {
    std::vector<CirclePoint<Mersenne31>> subpoints;
    auto shifted = subdomain.Shift(shift);
    for (const auto& point : shifted) {
      subpoints.push_back(point);
    }
    points_for_each_domain.push_back(subpoints);
  }

  // Interleave the points from each subdomain.
  std::vector<CirclePoint<Mersenne31>> extended_points;
  for (size_t point_ind = 0; point_ind < (1ULL << 3); ++point_ind) {
    for (size_t shift_ind = 0; shift_ind < (1ULL << 2); ++shift_ind) {
      extended_points.push_back(points_for_each_domain[shift_ind][point_ind]);
    }
  }

  EXPECT_EQ(domain_points, extended_points);
}

}  // namespace
}  // namespace zk_dtypes
