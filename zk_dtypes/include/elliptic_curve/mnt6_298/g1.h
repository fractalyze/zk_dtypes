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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_G1_H_

#include "zk_dtypes/include/elliptic_curve/mnt6_298/fq.h"
#include "zk_dtypes/include/elliptic_curve/mnt6_298/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::mnt6_298 {

// MNT6-298 G1: y² = x³ + 11x + b over Fq (a = 11).
// b, generator from ark-mnt6-298.
class G1SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1SwCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  constexpr static BaseField kA = 11;
  constexpr static BaseField kB = {
      UINT64_C(15827219621888807554), UINT64_C(2355843104595895421),
      UINT64_C(4425953106273765812),  UINT64_C(15925905401844974669),
      UINT64_C(921479880133),
  };
  constexpr static BaseField kX = {
      UINT64_C(12532321609140159613), UINT64_C(16191784749896139965),
      UINT64_C(7056391821740972887),  UINT64_C(3200319740236116666),
      UINT64_C(2907674911997),
  };
  constexpr static BaseField kY = {
      UINT64_C(9958370768819400744),  UINT64_C(6733419960382232352),
      UINT64_C(18260350450562793410), UINT64_C(15793986406456999802),
      UINT64_C(3476889421302),
  };
};

class G1SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G1SwCurveConfig;
  using BaseField = FqMont;
  using ScalarField = FrMont;

  constexpr static BaseField kA = BaseField::FromUnchecked({
      UINT64_C(13380829031336685551),
      UINT64_C(14274740638550322365),
      UINT64_C(9939770819147330555),
      UINT64_C(10956149001738181307),
      UINT64_C(668436861728),
  });
  constexpr static BaseField kB = BaseField::FromUnchecked({
      UINT64_C(762457536267711291),
      UINT64_C(1017480769655388902),
      UINT64_C(12579245979485372705),
      UINT64_C(11028678579857772437),
      UINT64_C(1017891033839),
  });
  constexpr static BaseField kX = BaseField::FromUnchecked({
      UINT64_C(1902266591782772004),
      UINT64_C(13966178682311351161),
      UINT64_C(15710654711326981618),
      UINT64_C(8125800953529631193),
      UINT64_C(576925825192),
  });
  constexpr static BaseField kY = BaseField::FromUnchecked({
      UINT64_C(8851191548877995954),
      UINT64_C(9862165831291588720),
      UINT64_C(4895755467660702521),
      UINT64_C(11132806403512354364),
      UINT64_C(3384698221971),
  });
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::mnt6_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_G1_H_
