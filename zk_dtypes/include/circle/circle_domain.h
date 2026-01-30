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

#ifndef ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_DOMAIN_H_
#define ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_DOMAIN_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "zk_dtypes/include/circle/circle_point.h"
#include "zk_dtypes/include/circle/circle_point_index.h"
#include "zk_dtypes/include/circle/coset.h"
#include "zk_dtypes/include/circle/m31_circle.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {

// Maximum log size for a circle domain (one less than M31_CIRCLE_LOG_ORDER).
constexpr uint32_t kMaxCircleDomainLogSize = kM31CircleLogOrder - 1;

// Minimum log size for a circle domain.
constexpr uint32_t kMinCircleDomainLogSize = 1;

// A valid domain for circle polynomial interpolation and evaluation.
// Valid domains are a disjoint union of two conjugate cosets: +-C + <G_n>.
// The ordering defined on this domain is C + iG_n, and then -C - iG_n.
class CircleDomain {
 public:
  Coset half_coset;

  CircleDomain() = default;

  explicit CircleDomain(const Coset& half_coset) : half_coset(half_coset) {}

  // Returns the size of the domain.
  constexpr size_t Size() const { return 1ULL << LogSize(); }

  // Returns the log size of the domain.
  constexpr uint32_t LogSize() const { return half_coset.LogSize() + 1; }

  // Returns the i-th domain element.
  CirclePoint<Mersenne31> At(size_t i) const { return ToPoint(IndexAt(i)); }

  // Returns the CirclePointIndex of the i-th domain element.
  CirclePointIndex IndexAt(size_t i) const {
    if (i < half_coset.Size()) {
      return half_coset.IndexAt(i);
    } else {
      return -half_coset.IndexAt(i - half_coset.Size());
    }
  }

  // Returns true if the domain is canonic.
  // Canonic domains are domains with elements that are the entire set of points defined by
  // G_{2n} + <G_n> where G_n and G_{2n} are obtained by repeatedly doubling M31_CIRCLE_GEN.
  bool IsCanonic() const {
    return half_coset.initial_index * 4 == half_coset.step_size;
  }

  // Splits a circle domain into a smaller CircleDomain, shifted by offsets.
  std::pair<CircleDomain, std::vector<CirclePointIndex>> Split(
      uint32_t log_parts) const {
    assert(log_parts <= half_coset.LogSize());
    CircleDomain subdomain(
        Coset(half_coset.initial_index, half_coset.LogSize() - log_parts));
    std::vector<CirclePointIndex> shifts;
    shifts.reserve(1ULL << log_parts);
    for (size_t i = 0; i < (1ULL << log_parts); ++i) {
      shifts.push_back(half_coset.step_size * i);
    }
    return {subdomain, shifts};
  }

  // Shifts the domain by the given amount.
  CircleDomain Shift(CirclePointIndex shift) const {
    return CircleDomain(half_coset.Shift(shift));
  }

  bool operator==(const CircleDomain& other) const {
    return half_coset == other.half_coset;
  }

  bool operator!=(const CircleDomain& other) const { return !(*this == other); }

  // Iterator that chains half_coset and its conjugate.
  class Iterator {
   public:
    Iterator(const Coset& half_coset, size_t index)
        : half_coset_(&half_coset), index_(index) {}

    CirclePoint<Mersenne31> operator*() const {
      if (index_ < half_coset_->Size()) {
        return half_coset_->At(index_);
      } else {
        return half_coset_->Conjugate().At(index_ - half_coset_->Size());
      }
    }

    Iterator& operator++() {
      ++index_;
      return *this;
    }

    bool operator!=(const Iterator& other) const {
      return index_ != other.index_;
    }

   private:
    const Coset* half_coset_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(half_coset, 0); }

  Iterator end() const { return Iterator(half_coset, Size()); }

  // Index iterator that chains half_coset indices and their negations.
  class IndexIterator {
   public:
    IndexIterator(const Coset& half_coset, size_t index)
        : half_coset_(&half_coset), index_(index) {}

    CirclePointIndex operator*() const {
      if (index_ < half_coset_->Size()) {
        return half_coset_->IndexAt(index_);
      } else {
        return -half_coset_->IndexAt(index_ - half_coset_->Size());
      }
    }

    IndexIterator& operator++() {
      ++index_;
      return *this;
    }

    bool operator!=(const IndexIterator& other) const {
      return index_ != other.index_;
    }

   private:
    const Coset* half_coset_;
    size_t index_;
  };

  IndexIterator IndicesBegin() const { return IndexIterator(half_coset, 0); }

  IndexIterator IndicesEnd() const {
    return IndexIterator(half_coset, Size());
  }
};

// A coset of the form G_{2n} + <G_n>, where G_n is the generator of the subgroup of order n.
// The ordering on this coset is G_{2n} + i * G_n.
// These cosets can be used as a CircleDomain, and be interpolated on.
// Note that this changes the ordering on the coset to be like CircleDomain,
// which is G_{2n} + i * G_{n/2} and then -G_{2n} - i * G_{n/2}.
class CanonicCoset {
 public:
  Coset coset;

  CanonicCoset() = default;

  explicit CanonicCoset(uint32_t log_size) : coset(Coset::Odds(log_size)) {
    assert(log_size > 0);
  }

  // Gets the full coset represented G_{2n} + <G_n>.
  const Coset& GetCoset() const { return coset; }

  // Gets half of the coset (its conjugate complements to the whole coset), G_{2n} + <G_{n/2}>.
  Coset HalfCoset() const { return Coset::HalfOdds(LogSize() - 1); }

  // Gets the CircleDomain representing the same point set (in another order).
  CircleDomain GetCircleDomain() const { return CircleDomain(HalfCoset()); }

  // Returns the log size of the coset.
  constexpr uint32_t LogSize() const { return coset.LogSize(); }

  // Returns the size of the coset.
  constexpr size_t Size() const { return coset.Size(); }

  CirclePointIndex InitialIndex() const { return coset.initial_index; }

  CirclePointIndex StepSize() const { return coset.step_size; }

  CirclePoint<Mersenne31> Step() const { return coset.step; }

  CirclePointIndex IndexAt(size_t index) const { return coset.IndexAt(index); }

  CirclePoint<Mersenne31> At(size_t i) const { return coset.At(i); }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CIRCLE_CIRCLE_DOMAIN_H_
