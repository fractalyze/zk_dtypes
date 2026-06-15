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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_G2_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_G2_H_

#include "zk_dtypes/include/elliptic_curve/mnt6_298/fqx3.h"
#include "zk_dtypes/include/elliptic_curve/mnt6_298/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::mnt6_298 {

// MNT6-298 G2 over FqX3 (the cubic twist): y² = x³ + a'x + b' with
// a' = (0, 0, 11) = (0, 0, COEFF_A) and b' = (5·b, 0, 0).
// Values (twist coeffs, generator) from ark-mnt6-298. Each FqX3 coordinate is
// (c0, c1, c2) with c0 + c1·u + c2·u² and u³ = 5.
class G2SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G2SwCurveConfig;
  using BaseField = FqX3;
  using ScalarField = Fr;

  constexpr static BaseField kA = {
      {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)},
      {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)},
      {UINT64_C(11), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)}};
  constexpr static BaseField kB = {
      {UINT64_C(10302179100803964041), UINT64_C(12118872224465161121),
       UINT64_C(7506919062651793135), UINT64_C(12587213846785589491),
       UINT64_C(497254318185)},
      {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)},
      {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)}};
  constexpr static BaseField kX = {
      {UINT64_C(4948741468954847251), UINT64_C(12513040782488949478),
       UINT64_C(201442015971770421), UINT64_C(6255028380047225929),
       UINT64_C(3639768817963)},
      {UINT64_C(12227656358602506490), UINT64_C(4683400904595509419),
       UINT64_C(12997221507588427923), UINT64_C(5975930421125778101),
       UINT64_C(890155174826)},
      {UINT64_C(82616830145402830), UINT64_C(17517601131757822667),
       UINT64_C(13006404289022529628), UINT64_C(16367843824136356655),
       UINT64_C(1235224038928)}};
  constexpr static BaseField kY = {
      {UINT64_C(8815844766680796305), UINT64_C(9442157791365292401),
       UINT64_C(6775643458208415428), UINT64_C(14289185687982283841),
       UINT64_C(4012999503932)},
      {UINT64_C(3434657911289699438), UINT64_C(3299744831849133120),
       UINT64_C(4979489892731608504), UINT64_C(15786531497538547650),
       UINT64_C(869169113061)},
      {UINT64_C(8912474332398151671), UINT64_C(13506681642645607842),
       UINT64_C(11104537585227651130), UINT64_C(200873007283111778),
       UINT64_C(1062420207747)}};
};

class G2SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G2SwCurveConfig;
  using BaseField = FqX3Mont;
  using ScalarField = FrMont;

  constexpr static BaseField kA = {
      FqMont::FromUnchecked(
          {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)}),
      FqMont::FromUnchecked(
          {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)}),
      FqMont::FromUnchecked(
          {UINT64_C(13380829031336685551), UINT64_C(14274740638550322365),
           UINT64_C(9939770819147330555), UINT64_C(10956149001738181307),
           UINT64_C(668436861728)})};
  constexpr static BaseField kB = {
      FqMont::FromUnchecked(
          {UINT64_C(8765344967536689190), UINT64_C(5427060549762628522),
           UINT64_C(11379895281290724368), UINT64_C(6547823810559129949),
           UINT64_C(979310086714)}),
      FqMont::FromUnchecked(
          {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)}),
      FqMont::FromUnchecked(
          {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)})};
  constexpr static BaseField kX = {
      FqMont::FromUnchecked(
          {UINT64_C(1570088295198957223), UINT64_C(11388243547153545593),
           UINT64_C(16638094191055501418), UINT64_C(5096397944772901895),
           UINT64_C(2489020867565)}),
      FqMont::FromUnchecked(
          {UINT64_C(3174542784159244592), UINT64_C(2855666475624403019),
           UINT64_C(8007133152924412546), UINT64_C(682346899378287834),
           UINT64_C(3034590943330)}),
      FqMont::FromUnchecked(
          {UINT64_C(17574722413557572081), UINT64_C(456974755699165048),
           UINT64_C(3254619369127186675), UINT64_C(8279179948042967436),
           UINT64_C(1102598794141)})};
  constexpr static BaseField kY = {
      FqMont::FromUnchecked(
          {UINT64_C(12812139972553768031), UINT64_C(11537818555351683770),
           UINT64_C(260815648150822609), UINT64_C(10680555275562460751),
           UINT64_C(1291865091856)}),
      FqMont::FromUnchecked(
          {UINT64_C(11624524503831104487), UINT64_C(2433606431341056774),
           UINT64_C(11186207532538858274), UINT64_C(4792775529631887891),
           UINT64_C(909584932857)}),
      FqMont::FromUnchecked(
          {UINT64_C(6991090668950567135), UINT64_C(5882326072780932593),
           UINT64_C(11371546271046018568), UINT64_C(7882928134413989632),
           UINT64_C(3108546647457)})};
};

using G2Curve = SwCurve<G2SwCurveConfig>;
using G2CurveMont = SwCurve<G2SwCurveMontConfig>;
using G2AffinePoint = AffinePoint<G2Curve>;
using G2AffinePointMont = AffinePoint<G2CurveMont>;
using G2JacobianPoint = JacobianPoint<G2Curve>;
using G2JacobianPointMont = JacobianPoint<G2CurveMont>;
using G2PointXyzz = PointXyzz<G2Curve>;
using G2PointXyzzMont = PointXyzz<G2CurveMont>;

}  // namespace zk_dtypes::mnt6_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_G2_H_
