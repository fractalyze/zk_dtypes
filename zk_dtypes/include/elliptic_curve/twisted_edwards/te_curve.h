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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_TE_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_TE_CURVE_H_

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"

namespace zk_dtypes {

// Curve for Twisted Edwards model.
// See https://www.hyperelliptic.org/EFD/g1p/auto-twisted.html for details.
// This config represents a * x² + y² = 1 + d * x² * y², where a and d are
// non-zero constants. The standard untwisted Edwards form is recovered when
// a = 1, while Ed25519 uses a = -1.
template <typename _Config>
class TwistedEdwardsCurve {
 public:
  using Config = _Config;

  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePoint = zk_dtypes::AffinePoint<TwistedEdwardsCurve<Config>>;
  using ExtendedPoint = zk_dtypes::ExtendedPoint<TwistedEdwardsCurve<Config>>;

  constexpr static bool kUseMontgomery = Config::kUseMontgomery;
  constexpr static CurveType kType = CurveType::kTwistedEdwards;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_TE_CURVE_H_
