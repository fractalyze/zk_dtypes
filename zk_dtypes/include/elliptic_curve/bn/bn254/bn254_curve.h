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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_BN254_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_BN254_CURVE_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fqx12.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn_curve.h"
#include "zk_dtypes/include/elliptic_curve/pairing/twist_type.h"

namespace zk_dtypes::bn254 {

// BN254 curve configuration for pairing.
class BN254CurveConfig {
 public:
  using G1Curve = zk_dtypes::bn254::G1Curve;
  using G2Curve = zk_dtypes::bn254::G2Curve;
  using Fp = zk_dtypes::bn254::Fq;
  using Fp2 = zk_dtypes::bn254::FqX2;
  using Fp6 = zk_dtypes::bn254::FqX6;
  using Fp12 = zk_dtypes::bn254::FqX12;

  // BN parameter x (also called u in some literature).
  // For BN254: x = 4965661367192848881
  constexpr static BigInt<1> kX = BigInt<1>({UINT64_C(4965661367192848881)});
  constexpr static bool kXIsNegative = false;

  // Ate loop count: NAF representation of 6x + 2.
  // 6x + 2 = 29793968203157093290 for BN254.
  constexpr static int8_t kAteLoopCount[] = {
      0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, 1,
      0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1,
      1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, 1, 1,
  };

  // Twist type for BN254 is D-type (divisive).
  constexpr static TwistType kTwistType = TwistType::kD;

  // Frobenius endomorphism coefficients for twist.
  // kTwistMulByQX = ω where ω is a primitive 6th root of unity in Fp2.
  static Fp2 kTwistMulByQX;
  static Fp2 kTwistMulByQY;

  static void Init() {
    // TwistMulByQX in Montgomery form
    // = (21575463638280843010398324269430826099269044274347216827212613867836435027261,
    //    10307601595873709700152284273816112264069230130616436755625194854815875713954)
    kTwistMulByQX = Fp2({
        {UINT64_C(8314163329781907090), UINT64_C(11942187234498585702),
         UINT64_C(5393139320892109860), UINT64_C(1288095885241975189)},
        {UINT64_C(14901948865669067533), UINT64_C(12485201766733748343),
         UINT64_C(15150072231498546846), UINT64_C(3315918954353925999)},
    });

    // TwistMulByQY in Montgomery form
    // = (2821565182194536844548159561693502659359617185244120367078079554186484126554,
    //    3505843767911556378687030309984248845540243509899259641013678093033130930403)
    kTwistMulByQY = Fp2({
        {UINT64_C(12147909053109254563), UINT64_C(14017466716612724529),
         UINT64_C(14920291267698255063), UINT64_C(2287871709057722285)},
        {UINT64_C(6044016917766013442), UINT64_C(8850240673049509562),
         UINT64_C(14441037729567679453), UINT64_C(572619090230498490)},
    });
  }
};

BN254CurveConfig::Fp2 BN254CurveConfig::kTwistMulByQX;
BN254CurveConfig::Fp2 BN254CurveConfig::kTwistMulByQY;

// BN254 pairing curve.
using BN254Curve = BNCurve<BN254CurveConfig>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_BN254_CURVE_H_
