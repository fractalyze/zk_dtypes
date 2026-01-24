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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PREPARED_BASE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PREPARED_BASE_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "zk_dtypes/include/elliptic_curve/pairing/ell_coeff.h"
#include "zk_dtypes/include/str_join.h"

namespace zk_dtypes {

// Base class for prepared G2 points used in pairing computation.
// Stores precomputed line coefficients to optimize the Miller loop.
template <typename PairingFriendlyCurveConfig>
class G2PreparedBase {
 public:
  using Config = PairingFriendlyCurveConfig;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;

  G2PreparedBase() = default;
  explicit G2PreparedBase(const EllCoeffs<Fp2>& ell_coeffs)
      : ell_coeffs_(ell_coeffs), infinity_(false) {}
  explicit G2PreparedBase(EllCoeffs<Fp2>&& ell_coeffs)
      : ell_coeffs_(std::move(ell_coeffs)), infinity_(false) {}

  const EllCoeffs<Fp2>& ell_coeffs() const { return ell_coeffs_; }
  bool infinity() const { return infinity_; }

  std::string ToString() const {
    return absl::Substitute(
        "{ell_coeffs: [$0], infinity: $1}",
        StrJoin(ell_coeffs_, ", ",
                [](const auto& c) { return c.ToString(); }),
        infinity_);
  }

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute(
        "{ell_coeffs: [$0], infinity: $1}",
        StrJoin(ell_coeffs_, ", ",
                [pad_zero](const auto& c) { return c.ToHexString(pad_zero); }),
        infinity_);
  }

 protected:
  // Stores the coefficients of the line evaluations as calculated in
  // https://eprint.iacr.org/2013/722.pdf
  EllCoeffs<Fp2> ell_coeffs_;
  bool infinity_ = true;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_G2_PREPARED_BASE_H_
