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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_G2_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_G2_H_

#include "zk_dtypes/include/elliptic_curve/mnt4_298/fqx2.h"
#include "zk_dtypes/include/elliptic_curve/mnt4_298/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::mnt4_298 {

// MNT4-298 G2 over FqX2 (the quadratic twist): y² = x³ + a'x + b' with
// a' = (34, 0) = (COEFF_A · non_residue, 0) and b' = (0, b·non_residue).
// Values (twist coeffs, generator) from ark-mnt4-298.
class G2SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G2SwCurveConfig;
  using BaseField = FqX2;
  using ScalarField = Fr;

  constexpr static BaseField kA = {{
                                       UINT64_C(34),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                   },
                                   {
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                   }};
  constexpr static BaseField kB = {{
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                       UINT64_C(0),
                                   },
                                   {
                                       UINT64_C(7654834016275860758),
                                       UINT64_C(6639627127254871117),
                                       UINT64_C(2219482899261599850),
                                       UINT64_C(16636899235165397998),
                                       UINT64_C(581843102222),
                                   }};
  constexpr static BaseField kX = {{
                                       UINT64_C(13436775221361748388),
                                       UINT64_C(3220442077098796845),
                                       UINT64_C(10376841120081384103),
                                       UINT64_C(7351407308195180280),
                                       UINT64_C(3785879753157),
                                   },
                                   {
                                       UINT64_C(10848013905752533385),
                                       UINT64_C(229731644904075564),
                                       UINT64_C(7601096639288137571),
                                       UINT64_C(15714851233040728495),
                                       UINT64_C(324900896626),
                                   }};
  constexpr static BaseField kY = {{
                                       UINT64_C(978726687638789922),
                                       UINT64_C(17048641329042926333),
                                       UINT64_C(15212698069489290080),
                                       UINT64_C(12310704625412826629),
                                       UINT64_C(323315774463),
                                   },
                                   {
                                       UINT64_C(9582359988289986801),
                                       UINT64_C(4274312890841155524),
                                       UINT64_C(14663043291447860476),
                                       UINT64_C(11890381798806765824),
                                       UINT64_C(3667102669929),
                                   }};
};

class G2SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G2SwCurveConfig;
  using BaseField = FqX2Mont;
  using ScalarField = FrMont;

  constexpr static BaseField kA = {
      FqMont::FromUnchecked(
          {UINT64_C(9379015694948865065), UINT64_C(3933863906897692531),
           UINT64_C(7183785805598089445), UINT64_C(17382890709766103498),
           UINT64_C(3934325337380)}),
      FqMont::FromUnchecked(
          {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)})};
  constexpr static BaseField kB = {
      FqMont::FromUnchecked(
          {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)}),
      FqMont::FromUnchecked(
          {UINT64_C(9511110677122940475), UINT64_C(13403516020116973437),
           UINT64_C(1464701424831086967), UINT64_C(4646785117660390394),
           UINT64_C(1747881737068)})};
  constexpr static BaseField kX = {
      FqMont::FromUnchecked(
          {UINT64_C(5356671649366391794), UINT64_C(2684151262065976452),
           UINT64_C(4683110650642896126), UINT64_C(10421299515941681582),
           UINT64_C(1618695480960)}),
      FqMont::FromUnchecked(
          {UINT64_C(133394645290266480), UINT64_C(15395232932057272770),
           UINT64_C(18271324022738539173), UINT64_C(9095178119640120034),
           UINT64_C(2303787573609)})};
  constexpr static BaseField kY = {
      FqMont::FromUnchecked(
          {UINT64_C(16920448081812496532), UINT64_C(15580160192086626100),
           UINT64_C(3974467672100342742), UINT64_C(8216505962266760277),
           UINT64_C(2643162835232)}),
      FqMont::FromUnchecked(
          {UINT64_C(73816197493558356), UINT64_C(8663991890578965996),
           UINT64_C(11575903875707445958), UINT64_C(17953546933481201011),
           UINT64_C(2167465829200)})};
};

using G2Curve = SwCurve<G2SwCurveConfig>;
using G2CurveMont = SwCurve<G2SwCurveMontConfig>;
using G2AffinePoint = AffinePoint<G2Curve>;
using G2AffinePointMont = AffinePoint<G2CurveMont>;
using G2JacobianPoint = JacobianPoint<G2Curve>;
using G2JacobianPointMont = JacobianPoint<G2CurveMont>;
using G2PointXyzz = PointXyzz<G2Curve>;
using G2PointXyzzMont = PointXyzz<G2CurveMont>;

}  // namespace zk_dtypes::mnt4_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_G2_H_
