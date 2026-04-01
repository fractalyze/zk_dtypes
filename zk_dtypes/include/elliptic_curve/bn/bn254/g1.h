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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G1_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::bn254 {

// BN254 G1: y² = x³ + 3, generator (1, 2).
class G1SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1SwCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 3;
  constexpr static BaseField kX = 1;
  constexpr static BaseField kY = 2;
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
      {UINT64_C(8797723225643362519), UINT64_C(2263834496217719225),
       UINT64_C(3696305541684646532), UINT64_C(3035258219084094862)});
  constexpr static BaseField kX = BaseField::FromUnchecked(
      {UINT64_C(15230403791020821917), UINT64_C(754611498739239741),
       UINT64_C(7381016538464732716), UINT64_C(1011752739694698287)});
  constexpr static BaseField kY = BaseField::FromUnchecked(
      {UINT64_C(12014063508332092218), UINT64_C(1509222997478479483),
       UINT64_C(14762033076929465432), UINT64_C(2023505479389396574)});
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G1_H_
