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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_CURVE_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"

namespace zk_dtypes {

class CurveStd {
 public:
  using G1Curve = bn254::G1CurveStd;
  using G2Curve = bn254::G2CurveStd;

  using StdConfig = CurveStd;
};

class Curve {
 public:
  using G1Curve = bn254::G1Curve;
  using G2Curve = bn254::G2Curve;

  using StdCurve = CurveStd;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_CURVE_H_
