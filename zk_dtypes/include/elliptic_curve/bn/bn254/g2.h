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

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq2.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::bn254 {

template <typename BaseField>
class G2SwCurveBaseConfig {
 public:
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

class G2SwCurveStdConfig : public G2SwCurveBaseConfig<Fq2Std> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = G2SwCurveStdConfig;

  using BaseField = Fq2Std;
  using ScalarField = FrStd;
};

class G2SwCurveConfig : public G2SwCurveBaseConfig<Fq2> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = G2SwCurveStdConfig;

  using BaseField = Fq2;
  using ScalarField = Fr;
};

using G2Curve = SwCurve<G2SwCurveConfig>;
using G2CurveStd = SwCurve<G2SwCurveStdConfig>;
using G2AffinePoint = AffinePoint<G2Curve>;
using G2AffinePointStd = AffinePoint<G2CurveStd>;
using G2JacobianPoint = JacobianPoint<G2Curve>;
using G2JacobianPointStd = JacobianPoint<G2CurveStd>;
using G2PointXyzz = PointXyzz<G2Curve>;
using G2PointXyzzStd = PointXyzz<G2CurveStd>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G2_H_
