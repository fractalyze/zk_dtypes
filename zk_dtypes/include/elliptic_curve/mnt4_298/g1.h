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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_G1_H_

#include "zk_dtypes/include/elliptic_curve/mnt4_298/fq.h"
#include "zk_dtypes/include/elliptic_curve/mnt4_298/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::mnt4_298 {

// MNT4-298 G1: y² = x³ + 2x + b over Fq (a = 2, unlike BN curves).
// b, generator (G1_GENERATOR_X, G1_GENERATOR_Y) from ark-mnt4-298.
class G1SwCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1SwCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  constexpr static BaseField kA = 2;
  constexpr static BaseField kB = {
      UINT64_C(6722483314896932821),  UINT64_C(8905885093438036561),
      UINT64_C(14118172000221005916), UINT64_C(1538083334790521679),
      UINT64_C(3660824667028),
  };
  constexpr static BaseField kX = {
      UINT64_C(11679722493174787910), UINT64_C(6972706457774541022),
      UINT64_C(12603440906699527824), UINT64_C(13440185221295782005),
      UINT64_C(524735709857),
  };
  constexpr static BaseField kY = {
      UINT64_C(11293419172838804146), UINT64_C(7455922144122925538),
      UINT64_C(9930459777607426771),  UINT64_C(9396531370401983534),
      UINT64_C(3141258207692),
  };
};

class G1SwCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G1SwCurveConfig;
  using BaseField = FqMont;
  using ScalarField = FrMont;

  constexpr static BaseField kA = BaseField::FromUnchecked({
      UINT64_C(3568597988870129848),
      UINT64_C(15257338106490985450),
      UINT64_C(10069779447956199041),
      UINT64_C(5922375556522222383),
      UINT64_C(3858029504390),
  });
  constexpr static BaseField kB = BaseField::FromUnchecked({
      UINT64_C(7842808090366692145),
      UINT64_C(288200302308193399),
      UINT64_C(4162060950790347941),
      UINT64_C(5488589108190218591),
      UINT64_C(1553456013645),
  });
  constexpr static BaseField kX = BaseField::FromUnchecked({
      UINT64_C(6046301378120906932),
      UINT64_C(15105298306031900263),
      UINT64_C(15757949605695610691),
      UINT64_C(6113949277267426050),
      UINT64_C(3063081829217),
  });
  constexpr static BaseField kY = BaseField::FromUnchecked({
      UINT64_C(8798367863963590781),
      UINT64_C(9770379341721339603),
      UINT64_C(17697354471293810920),
      UINT64_C(15252694996423733496),
      UINT64_C(3845520398052),
  });
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveMont = SwCurve<G1SwCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointMont = JacobianPoint<G1CurveMont>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzMont = PointXyzz<G1CurveMont>;

}  // namespace zk_dtypes::mnt4_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_G1_H_
