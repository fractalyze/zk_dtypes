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

#ifndef ZK_DTYPES_INCLUDE_CIRCLE_COSET_H_
#define ZK_DTYPES_INCLUDE_CIRCLE_COSET_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "zk_dtypes/include/circle/circle_point.h"
#include "zk_dtypes/include/circle/circle_point_index.h"
#include "zk_dtypes/include/circle/m31_circle.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {

// Generic iterator for cosets.
template <typename T>
class CosetIterator {
 public:
  T cur;
  T step;
  size_t remaining;

  constexpr CosetIterator(T cur, T step, size_t remaining)
      : cur(cur), step(step), remaining(remaining) {}

  constexpr T operator*() const { return cur; }

  constexpr CosetIterator& operator++() {
    cur = cur + step;
    --remaining;
    return *this;
  }

  constexpr bool operator!=(const CosetIterator& other) const {
    return remaining != other.remaining;
  }

  // Sentinel for range-based for loop.
  constexpr bool operator==(const CosetIterator& other) const {
    return remaining == other.remaining;
  }
};

// Represents the coset: initial + <step>.
class Coset {
 public:
  CirclePointIndex initial_index;
  CirclePoint<Mersenne31> initial;
  CirclePointIndex step_size;
  CirclePoint<Mersenne31> step;
  uint32_t log_size;

  Coset() = default;

  Coset(CirclePointIndex initial_index, uint32_t log_size)
      : initial_index(initial_index),
        initial(ToPoint(initial_index)),
        step_size(CirclePointIndex::SubgroupGen(log_size)),
        step(ToPoint(step_size)),
        log_size(log_size) {
    assert(log_size <= kM31CircleLogOrder);
  }

  // Creates a coset of the form <G_n>.
  // For example, for n=8, we get the point indices [0,1,2,3,4,5,6,7].
  static Coset Subgroup(uint32_t log_size) {
    return Coset(CirclePointIndex::Zero(), log_size);
  }

  // Creates a coset of the form G_{2n} + <G_n>.
  // For example, let n = 8 and denote G_16 = x, <G_8> = <2x>.
  // The point indices are [x, 3x, 5x, 7x, 9x, 11x, 13x, 15x].
  static Coset Odds(uint32_t log_size) {
    return Coset(CirclePointIndex::SubgroupGen(log_size + 1), log_size);
  }

  // Creates a coset of the form G_{4n} + <G_n>. Its conjugate is 3 * G_{4n} + <G_n>.
  // For example, let n = 8 and denote G_32 = x, <G_8> = <4x>.
  // The point indices are [x, 5x, 9x, 13x, 17x, 21x, 25x, 29x].
  // Conjugate coset indices are [3x, 7x, 11x, 15x, 19x, 23x, 27x, 31x].
  // Note: This coset union with its conjugate coset is the odds(log_size + 1) coset.
  static Coset HalfOdds(uint32_t log_size) {
    return Coset(CirclePointIndex::SubgroupGen(log_size + 2), log_size);
  }

  // Returns the size of the coset.
  constexpr size_t Size() const { return 1ULL << log_size; }

  // Returns the log size of the coset.
  constexpr uint32_t LogSize() const { return log_size; }

  // Returns the initial point.
  constexpr CirclePoint<Mersenne31> Initial() const { return initial; }

  // Returns the index at position i.
  CirclePointIndex IndexAt(size_t index) const {
    return initial_index + step_size * index;
  }

  // Returns the point at position i.
  CirclePoint<Mersenne31> At(size_t index) const {
    return ToPoint(IndexAt(index));
  }

  // Returns a new coset comprising of all points in current coset doubled.
  Coset Double() const {
    assert(log_size > 0);
    CirclePointIndex new_initial_index = initial_index * 2;
    CirclePointIndex new_step_size = step_size * 2;
    Coset result;
    result.initial_index = new_initial_index;
    result.initial = initial.Double();
    result.step_size = new_step_size;
    result.step = step.Double();
    result.log_size = log_size - 1;
    return result;
  }

  // Repeated doubling.
  Coset RepeatedDouble(uint32_t n_doubles) const {
    Coset result = *this;
    for (uint32_t i = 0; i < n_doubles; ++i) {
      result = result.Double();
    }
    return result;
  }

  // Checks if this coset is the result of doubling another coset.
  bool IsDoublingOf(const Coset& other) const {
    return log_size <= other.log_size &&
           *this == other.RepeatedDouble(other.log_size - log_size);
  }

  // Shifts the coset by the given amount.
  Coset Shift(CirclePointIndex shift_size) const {
    CirclePointIndex new_initial_index = initial_index + shift_size;
    Coset result = *this;
    result.initial_index = new_initial_index;
    result.initial = ToPoint(new_initial_index);
    return result;
  }

  // Creates the conjugate coset: -initial - <step>.
  Coset Conjugate() const {
    CirclePointIndex new_initial_index = -initial_index;
    CirclePointIndex new_step_size = -step_size;
    Coset result;
    result.initial_index = new_initial_index;
    result.initial = ToPoint(new_initial_index);
    result.step_size = new_step_size;
    result.step = ToPoint(new_step_size);
    result.log_size = log_size;
    return result;
  }

  // Iterator over points.
  CosetIterator<CirclePoint<Mersenne31>> begin() const {
    return CosetIterator<CirclePoint<Mersenne31>>(initial, step, Size());
  }

  CosetIterator<CirclePoint<Mersenne31>> end() const {
    return CosetIterator<CirclePoint<Mersenne31>>(initial, step, 0);
  }

  // Iterator over indices.
  CosetIterator<CirclePointIndex> IndicesBegin() const {
    return CosetIterator<CirclePointIndex>(initial_index, step_size, Size());
  }

  CosetIterator<CirclePointIndex> IndicesEnd() const {
    return CosetIterator<CirclePointIndex>(initial_index, step_size, 0);
  }

  bool operator==(const Coset& other) const {
    return initial_index == other.initial_index && step_size == other.step_size &&
           log_size == other.log_size;
  }

  bool operator!=(const Coset& other) const { return !(*this == other); }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CIRCLE_COSET_H_
