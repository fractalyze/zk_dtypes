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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_G1_H_

#include "zk_dtypes/include/elliptic_curve/bls12_381/fq.h"
#include "zk_dtypes/include/elliptic_curve/bls12_381/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::bls12_381 {

// BLS12-381 G1: y² = x³ + 4
class G1SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1SwCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 4;
  constexpr static BaseField kX = {
      UINT64_C(18103045581585958587), UINT64_C(7806400890582735599),
      UINT64_C(11623291730934869080), UINT64_C(14080658508445169925),
      UINT64_C(2780237799254240271),  UINT64_C(1725392847304644500)};
  constexpr static BaseField kY = {
      UINT64_C(912580534683953121),   UINT64_C(15005087156090211044),
      UINT64_C(61670280795567085),    UINT64_C(18227722000993880822),
      UINT64_C(11573741888802228964), UINT64_C(627113611842199793)};
};

class G1SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G1SwCurveConfig;
  using BaseField = FqMont;
  using ScalarField = FrMont;

  constexpr static BaseField kA =
      BaseField::FromUnchecked({UINT64_C(0), UINT64_C(0), UINT64_C(0),
                                UINT64_C(0), UINT64_C(0), UINT64_C(0)});
  constexpr static BaseField kB = BaseField::FromUnchecked(
      {UINT64_C(12260768510540316659), UINT64_C(6038201419376623626),
       UINT64_C(5156596810353639551), UINT64_C(12813724723179037911),
       UINT64_C(10288881524157229871), UINT64_C(708830206584151678)});
  constexpr static BaseField kX = BaseField::FromUnchecked(
      {UINT64_C(6679831729115696150), UINT64_C(8653662730902241269),
       UINT64_C(1535610680227111361), UINT64_C(17342916647841752903),
       UINT64_C(17135755455211762752), UINT64_C(1297449291367578485)});
  constexpr static BaseField kY = BaseField::FromUnchecked(
      {UINT64_C(13451288730302620273), UINT64_C(10097742279870053774),
       UINT64_C(15949884091978425806), UINT64_C(5885175747529691540),
       UINT64_C(1016841820992199104), UINT64_C(845620083434234474)});
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::bls12_381

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BLS12_381_G1_H_
