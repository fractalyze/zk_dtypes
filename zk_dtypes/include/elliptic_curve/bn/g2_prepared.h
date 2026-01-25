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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_G2_PREPARED_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_G2_PREPARED_H_

#include <utility>

#include "zk_dtypes/include/elliptic_curve/pairing/g2_prepared_base.h"
#include "zk_dtypes/include/elliptic_curve/pairing/g2_projective.h"

namespace zk_dtypes::bn {

// G2Prepared for BN curves.
// Precomputes line function coefficients for efficient Miller loop evaluation.
template <typename BNCurveConfig>
class G2Prepared : public G2PreparedBase<BNCurveConfig> {
 public:
  using Config = BNCurveConfig;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;
  using Fp = typename Fp2::BaseField;
  using G2AffinePoint = typename G2Curve::AffinePoint;

  G2Prepared() = default;
  explicit G2Prepared(const EllCoeffs<Fp2>& ell_coeffs)
      : G2PreparedBase<BNCurveConfig>(ell_coeffs) {}
  explicit G2Prepared(EllCoeffs<Fp2>&& ell_coeffs)
      : G2PreparedBase<BNCurveConfig>(std::move(ell_coeffs)) {}

  // Precompute line function coefficients from an affine G2 point.
  static G2Prepared From(const G2AffinePoint& q) {
    if (q.IsZero()) {
      return {};
    }

    G2Projective<Config> r(q.x(), q.y(), Fp2::One());

    EllCoeffs<Fp2> ell_coeffs;
    size_t size = std::size(Config::kAteLoopCount);
    ell_coeffs.reserve(size + size * 2 / 3);

    G2AffinePoint neg_q = -q;
    Fp two_inv = Fp::TwoInv();

    // Skip the first bit (most significant)
    for (size_t i = size - 2; i != SIZE_MAX; --i) {
      auto [new_r, coeff] = r.Double(two_inv);
      r = std::move(new_r);
      ell_coeffs.push_back(std::move(coeff));

      switch (Config::kAteLoopCount[i]) {
        case 1: {
          auto [add_r, add_coeff] = r.Add(q);
          r = std::move(add_r);
          ell_coeffs.push_back(std::move(add_coeff));
          break;
        }
        case -1: {
          auto [add_r, add_coeff] = r.Add(neg_q);
          r = std::move(add_r);
          ell_coeffs.push_back(std::move(add_coeff));
          break;
        }
        default:
          continue;
      }
    }

    G2AffinePoint q1 = MulByCharacteristic(q);
    G2AffinePoint q2 = -MulByCharacteristic(q1);

    if constexpr (Config::kXIsNegative) {
      r = r.Negate();
    }

    auto [r1, coeff1] = r.Add(q1);
    r = std::move(r1);
    ell_coeffs.push_back(std::move(coeff1));

    auto [r2, coeff2] = r.Add(q2);
    ell_coeffs.push_back(std::move(coeff2));

    return G2Prepared(std::move(ell_coeffs));
  }

 private:
  // Multiply point by the Frobenius characteristic.
  static G2AffinePoint MulByCharacteristic(const G2AffinePoint& r) {
    Fp2 x = r.x().template Frobenius<1>() * Config::kTwistMulByQX;
    Fp2 y = r.y().template Frobenius<1>() * Config::kTwistMulByQY;
    return G2AffinePoint(std::move(x), std::move(y));
  }
};

}  // namespace zk_dtypes::bn

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_G2_PREPARED_H_
