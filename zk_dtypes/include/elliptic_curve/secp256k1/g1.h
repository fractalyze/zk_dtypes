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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_G1_H_

#include "zk_dtypes/include/elliptic_curve/secp256k1/fq.h"
#include "zk_dtypes/include/elliptic_curve/secp256k1/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::secp256k1 {

// secp256k1: y² = x³ + 7
class G1SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1SwCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 7;
  constexpr static BaseField kX = {
      UINT64_C(6481385041966929816), UINT64_C(188021827762530521),
      UINT64_C(6170039885052185351), UINT64_C(8772561819708210092)};
  constexpr static BaseField kY = {
      UINT64_C(11261198710074299576), UINT64_C(18237243440184513561),
      UINT64_C(6747795201694173352), UINT64_C(5204712524664259685)};
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
      {UINT64_C(30064777911), UINT64_C(0), UINT64_C(0), UINT64_C(0)});
  constexpr static BaseField kX = BaseField::FromUnchecked(
      {UINT64_C(15507633332195041431), UINT64_C(2530505477788034779),
       UINT64_C(10925531211367256732), UINT64_C(11061375339145502536)});
  constexpr static BaseField kY = BaseField::FromUnchecked(
      {UINT64_C(12780836216951778274), UINT64_C(10231155108014310989),
       UINT64_C(8121878653926228278), UINT64_C(14933801261141951190)});
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::secp256k1

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256K1_G1_H_
