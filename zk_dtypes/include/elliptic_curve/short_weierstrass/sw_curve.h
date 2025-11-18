/* Copyright 2025 The zk_dtypes Authors.

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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_SW_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_SW_CURVE_H_

#include "absl/status/statusor.h"

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"

namespace zk_dtypes {

// Curve for Short Weierstrass model.
// See https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html for more details.
// This config represents y² = x³ + a * x + b, where a and b are constants.
template <typename _Config>
class SwCurve {
 public:
  using Config = _Config;

  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePoint = zk_dtypes::AffinePoint<SwCurve<Config>>;
  using JacobianPoint = zk_dtypes::JacobianPoint<SwCurve<Config>>;
  using PointXyzz = zk_dtypes::PointXyzz<SwCurve<Config>>;

  constexpr static bool kUseMontgomery = Config::kUseMontgomery;
  constexpr static CurveType kType = CurveType::kShortWeierstrass;

  // Attempts to construct an affine point given an x-coordinate.
  constexpr static absl::StatusOr<AffinePoint> GetPointFromX(
      const BaseField& x) {
    absl::StatusOr<BaseField> y = GetYFromX(x);
    if (!y.ok()) return y.status();
    return AffinePoint(x, *y);
  }

  // Returns the y-coordinate corresponding to the given x-coordinate if x lies
  // on the curve.
  constexpr static absl::StatusOr<BaseField> GetYFromX(const BaseField& x) {
    BaseField right = x.Square() * x + Config::kB;
    if constexpr (!Config::kA.IsZero()) {
      right += Config::kA * x;
    }
    return right.SquareRoot();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_SW_CURVE_H_
