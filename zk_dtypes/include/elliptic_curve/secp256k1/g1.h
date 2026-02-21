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

// y² = x³ + 7 (a = 0, b = 7)
template <typename BaseField>
class G1SwCurveBaseConfig {
 public:
  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 7;

  // Generator point (standard form, auto-converted to Montgomery if needed).
  constexpr static BaseField kX = {
      UINT64_C(6481385041966929816),
      UINT64_C(188021827762530521),
      UINT64_C(6170039885052185351),
      UINT64_C(8772561819708210092),
  };
  constexpr static BaseField kY = {
      UINT64_C(11261198710074299576),
      UINT64_C(18237243440184513561),
      UINT64_C(6747795201694173352),
      UINT64_C(5204712524664259685),
  };
};

class G1SwCurveConfig : public G1SwCurveBaseConfig<Fq> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = G1SwCurveConfig;

  using BaseField = Fq;
  using ScalarField = Fr;
};

class G1SwCurveMontConfig : public G1SwCurveBaseConfig<FqMont> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = G1SwCurveConfig;

  using BaseField = FqMont;
  using ScalarField = FrMont;
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
