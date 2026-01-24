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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_ELL_COEFF_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_ELL_COEFF_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

namespace zk_dtypes {

// Coefficients for line function evaluation in Miller loop.
// Stores three field elements (c0, c1, c2) representing the line
// passing through points during pairing computation.
template <typename F>
class EllCoeff {
 public:
  EllCoeff() = default;
  EllCoeff(const F& c0, const F& c1, const F& c2) : c0_(c0), c1_(c1), c2_(c2) {}
  EllCoeff(F&& c0, F&& c1, F&& c2)
      : c0_(std::move(c0)), c1_(std::move(c1)), c2_(std::move(c2)) {}

  const F& c0() const { return c0_; }
  const F& c1() const { return c1_; }
  const F& c2() const { return c2_; }

  std::string ToString() const {
    return absl::Substitute("{c0: $0, c1: $1, c2: $2}", c0_.ToString(),
                            c1_.ToString(), c2_.ToString());
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("{c0: $0, c1: $1, c2: $2}",
                            c0_.ToHexString(pad_zero), c1_.ToHexString(pad_zero),
                            c2_.ToHexString(pad_zero));
  }

 private:
  F c0_;
  F c1_;
  F c2_;
};

template <typename F>
using EllCoeffs = std::vector<EllCoeff<F>>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_ELL_COEFF_H_
