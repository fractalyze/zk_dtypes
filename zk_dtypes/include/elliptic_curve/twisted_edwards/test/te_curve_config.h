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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_TEST_TE_CURVE_CONFIG_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_TEST_TE_CURVE_CONFIG_H_

#include "zk_dtypes/include/elliptic_curve/twisted_edwards/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/extended_point.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/te_curve.h"
#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes::test {

// Small field F_13 for testing twisted Edwards curves.
struct TeTestFqBaseConfig {
  constexpr static size_t kStorageBits = 8;
  constexpr static size_t kModulusBits = 4;
  constexpr static uint8_t kModulus = 13;

  constexpr static uint32_t kTwoAdicity = 2;
  constexpr static uint8_t kTrace = 3;  // (13-1)/4 = 3

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct TeTestFqConfig : public TeTestFqBaseConfig {
  constexpr static bool kUseMontgomery = false;
  constexpr static bool kUseBarrett = false;

  using StdConfig = TeTestFqConfig;

  constexpr static uint8_t kOne = 1;
  constexpr static uint8_t kTwoAdicRootOfUnity = 5;  // 5^2 = 25 = -1 mod 13
};

using TeTestFq = PrimeField<TeTestFqConfig>;

// Twisted Edwards curve over F_13: -x^2 + y^2 = 1 + 2*x^2*y^2
// Has 16 points (cofactor 16 / prime subgroup). Generator: (2, 4).
//
// Scalar multiples of G = (2, 4):
//   0*G = (0, 1)    [identity]
//   1*G = (2, 4)
//   2*G = (10, 11)
//   3*G = (6, 10)
//   4*G = (8, 0)
//   5*G = (6, 3)
//   6*G = (10, 2)
//   7*G = (2, 9)
//   8*G = (0, 12)   [-identity = (0, -1)]
//   ...
//  15*G = (11, 4)
class TeTestCurveConfig {
 public:
  constexpr static bool kUseMontgomery = false;
  using StdConfig = TeTestCurveConfig;
  using BaseField = TeTestFq;
  using ScalarField = TeTestFq;

  constexpr static BaseField kA = 12;  // -1 mod 13
  constexpr static BaseField kD = 2;   // non-square mod 13
  constexpr static BaseField kX = 2;   // Generator x
  constexpr static BaseField kY = 4;   // Generator y
};

using TeTestCurve = TwistedEdwardsCurve<TeTestCurveConfig>;
using TeAffinePoint = AffinePoint<TeTestCurve>;
using TeExtendedPoint = ExtendedPoint<TeTestCurve>;

}  // namespace zk_dtypes::test

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_TEST_TE_CURVE_CONFIG_H_
