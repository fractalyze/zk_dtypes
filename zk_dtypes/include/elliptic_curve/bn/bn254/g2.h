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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G2_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G2_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fqx2.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::bn254 {

// BN254 G2: y² = x³ + b' over FqX2 (degree-2 extension of Fq).
class G2SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G2SwCurveConfig;
  using BaseField = FqX2;
  using ScalarField = Fr;

  constexpr static BaseField kA = {{0, 0}};
  constexpr static BaseField kB = {{
                                       UINT64_C(3632125457679333605),
                                       UINT64_C(13093307605518643107),
                                       UINT64_C(9348936922344483523),
                                       UINT64_C(3104278944836790958),
                                   },
                                   {
                                       UINT64_C(16474938222586303954),
                                       UINT64_C(12056031220135172178),
                                       UINT64_C(14784384838321896948),
                                       UINT64_C(42524369107353300),
                                   }};
  constexpr static BaseField kX = {{
                                       UINT64_C(5106727233969649389),
                                       UINT64_C(7440829307424791261),
                                       UINT64_C(4785637993704342649),
                                       UINT64_C(1729627375292849782),
                                   },
                                   {
                                       UINT64_C(10945020018377822914),
                                       UINT64_C(17413811393473931026),
                                       UINT64_C(8241798111626485029),
                                       UINT64_C(1841571559660931130),
                                   }};
  constexpr static BaseField kY = {{
                                       UINT64_C(5541340697920699818),
                                       UINT64_C(16416156555105522555),
                                       UINT64_C(5380518976772849807),
                                       UINT64_C(1353435754470862315),
                                   },
                                   {
                                       UINT64_C(6173549831154472795),
                                       UINT64_C(13567992399387660019),
                                       UINT64_C(17050234209342075797),
                                       UINT64_C(650358724130500725),
                                   }};
};

class G2SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G2SwCurveConfig;
  using BaseField = FqX2Mont;
  using ScalarField = FrMont;

  constexpr static BaseField kA = {FqMont::FromUnchecked(
      {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)})};
  constexpr static BaseField kB = {
      FqMont::FromUnchecked(
          {UINT64_C(4321547867055981224), UINT64_C(147241268046680925),
           UINT64_C(2789960110459671136), UINT64_C(2671978398120978541)}),
      FqMont::FromUnchecked(
          {UINT64_C(4100506350182530919), UINT64_C(7345568344173317438),
           UINT64_C(15513160039642431658), UINT64_C(90557763186888013)})};
  constexpr static BaseField kX = {
      FqMont::FromUnchecked(
          {UINT64_C(10269251484633538598), UINT64_C(15918845024527909234),
           UINT64_C(18138289588161026783), UINT64_C(1825990028691918907)}),
      FqMont::FromUnchecked(
          {UINT64_C(12660871435976991040), UINT64_C(6936631231174072516),
           UINT64_C(714191060563144582), UINT64_C(1512910971262892907)})};
  constexpr static BaseField kY = {
      FqMont::FromUnchecked(
          {UINT64_C(7034053747528165878), UINT64_C(18338607757778656120),
           UINT64_C(18419188534790028798), UINT64_C(2953656481336934918)}),
      FqMont::FromUnchecked(
          {UINT64_C(7208393106848765678), UINT64_C(15877432936589245627),
           UINT64_C(6195041853444001910), UINT64_C(983087530859390082)})};
};

using G2Curve = SwCurve<G2SwCurveConfig>;
using G2CurveMont = SwCurve<G2SwCurveMontConfig>;
using G2AffinePoint = AffinePoint<G2Curve>;
using G2AffinePointMont = AffinePoint<G2CurveMont>;
using G2JacobianPoint = JacobianPoint<G2Curve>;
using G2JacobianPointMont = JacobianPoint<G2CurveMont>;
using G2PointXyzz = PointXyzz<G2Curve>;
using G2PointXyzzMont = PointXyzz<G2CurveMont>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G2_H_
