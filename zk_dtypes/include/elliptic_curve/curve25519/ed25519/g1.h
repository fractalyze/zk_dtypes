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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_ED25519_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_ED25519_G1_H_

#include "zk_dtypes/include/elliptic_curve/curve25519/fq.h"
#include "zk_dtypes/include/elliptic_curve/curve25519/fr.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/extended_point.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/te_curve.h"

namespace zk_dtypes::ed25519 {

using curve25519::Fq;
using curve25519::FqMont;
using curve25519::Fr;
using curve25519::FrMont;

// Ed25519: -x² + y² = 1 + d * x² * y² over GF(2²⁵⁵ - 19)
// See RFC 8032 §5.1 for the curve parameters and generator.
class G1TeCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = G1TeCurveConfig;
  using BaseField = Fq;
  using ScalarField = Fr;

  // a = -1 mod p
  constexpr static BaseField kA = {
      UINT64_C(18446744073709551596), UINT64_C(18446744073709551615),
      UINT64_C(18446744073709551615), UINT64_C(9223372036854775807)};

  // d = -121665/121666 mod p
  constexpr static BaseField kD = {
      UINT64_C(8496970652267935907), UINT64_C(31536524315187371),
      UINT64_C(10144147576115030168), UINT64_C(5909686906226998899)};

  // Generator from RFC 8032 §5.1
  constexpr static BaseField kX = {
      UINT64_C(14507833142362363162), UINT64_C(7578651490590762930),
      UINT64_C(13881468655802702940), UINT64_C(2407515759118799870)};
  constexpr static BaseField kY = {
      UINT64_C(7378697629483820632), UINT64_C(7378697629483820646),
      UINT64_C(7378697629483820646), UINT64_C(7378697629483820646)};
};

class G1TeCurveMontConfig {
 public:
  constexpr static bool kUseMontgomery = true;
  using StdConfig = G1TeCurveConfig;
  using BaseField = FqMont;
  using ScalarField = FrMont;

  // a = -1 mod p in Montgomery form
  constexpr static BaseField kA = BaseField::FromUnchecked(
      {UINT64_C(18446744073709551559), UINT64_C(18446744073709551615),
       UINT64_C(18446744073709551615), UINT64_C(9223372036854775807)});

  // d in Montgomery form
  constexpr static BaseField kD = BaseField::FromUnchecked(
      {UINT64_C(9290235533119187450), UINT64_C(1198387923977120115),
       UINT64_C(16542726418180114064), UINT64_C(3207173552111338790)});

  // Generator in Montgomery form
  constexpr static BaseField kX = BaseField::FromUnchecked(
      {UINT64_C(16342081272192803463), UINT64_C(11287595536805717129),
       UINT64_C(10986974856635266487), UINT64_C(8475250514821412816)});
  constexpr static BaseField kY = BaseField::FromUnchecked(
      {UINT64_C(3689348814741910346), UINT64_C(3689348814741910323),
       UINT64_C(3689348814741910323), UINT64_C(3689348814741910323)});
};

using G1Curve = TwistedEdwardsCurve<G1TeCurveConfig>;
using G1CurveMont = TwistedEdwardsCurve<G1TeCurveMontConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointMont = AffinePoint<G1CurveMont>;
using G1ExtendedPoint = ExtendedPoint<G1Curve>;
using G1ExtendedPointMont = ExtendedPoint<G1CurveMont>;

}  // namespace zk_dtypes::ed25519

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_CURVE25519_ED25519_G1_H_
