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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256R1_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256R1_G1_H_

#include "zk_dtypes/include/elliptic_curve/secp256r1/fq.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::secp256r1 {

// secp256r1 (P-256): y² = x³ + ax + b
// a = -3, b =
// 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
template <typename BaseField>
class G1SwCurveBaseConfig {
 public:
  constexpr static BaseField kA = {
      UINT64_C(18446744073709551612),
      UINT64_C(4294967295),
      UINT64_C(0),
      UINT64_C(18446744069414584321),
  };
  constexpr static BaseField kB = {
      UINT64_C(4309448131093880907),
      UINT64_C(7285987128567378166),
      UINT64_C(12964664127075681980),
      UINT64_C(6540974713487397863),
  };

  // Generator point (standard form).
  constexpr static BaseField kX = {
      UINT64_C(17627433388654248598),
      UINT64_C(8575836109218198432),
      UINT64_C(17923454489921339634),
      UINT64_C(7716867327612699207),
  };
  constexpr static BaseField kY = {
      UINT64_C(14678990851816772085),
      UINT64_C(3156516839386865358),
      UINT64_C(10297457778147434006),
      UINT64_C(5756518291402817435),
  };
};

class G1SwCurveConfig : public G1SwCurveBaseConfig<Fq> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = G1SwCurveConfig;

  using BaseField = Fq;
};

class G1SwCurveMontConfig : public G1SwCurveBaseConfig<FqMont> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = G1SwCurveConfig;

  using BaseField = FqMont;
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::secp256r1

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SECP256R1_G1_H_
