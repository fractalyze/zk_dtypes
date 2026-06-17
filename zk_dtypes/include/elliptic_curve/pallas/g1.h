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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PALLAS_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PALLAS_G1_H_

#include "zk_dtypes/include/elliptic_curve/pallas/fq.h"
#include "zk_dtypes/include/elliptic_curve/pallas/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::pallas {

// Pallas: y² = x³ + 5, generator (-1, 2). Constants derived from arkworks
// ark-pallas (kX = p - 1); see docs/development.md for the derivation method.
class G1SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1SwCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 5;
  constexpr static BaseField kX = {UINT64_C(11037532056220336128),
                                   UINT64_C(2469829653914515739), UINT64_C(0),
                                   UINT64_C(4611686018427387904)};
  constexpr static BaseField kY = {UINT64_C(2), UINT64_C(0), UINT64_C(0),
                                   UINT64_C(0)};
};

class G1SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G1SwCurveConfig;
  using BaseField = FqMont;
  using ScalarField = FrMont;

  constexpr static BaseField kA = BaseField::FromUnchecked(
      {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)});
  constexpr static BaseField kB = BaseField::FromUnchecked(
      {UINT64_C(11647819816328232941), UINT64_C(8413468796752855795),
       UINT64_C(18446744073709551613), UINT64_C(4611686018427387903)});
  constexpr static BaseField kX = BaseField::FromUnchecked(
      {UINT64_C(7256640077462241284), UINT64_C(9879318615658062958),
       UINT64_C(0), UINT64_C(0)});
  constexpr static BaseField kY = BaseField::FromUnchecked(
      {UINT64_C(14970995975005405177), UINT64_C(1157936496307941438),
       UINT64_C(18446744073709551615), UINT64_C(4611686018427387903)});
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::pallas

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PALLAS_G1_H_
