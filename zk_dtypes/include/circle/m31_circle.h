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

#ifndef ZK_DTYPES_INCLUDE_CIRCLE_M31_CIRCLE_H_
#define ZK_DTYPES_INCLUDE_CIRCLE_M31_CIRCLE_H_

#include "zk_dtypes/include/circle/circle_point.h"
#include "zk_dtypes/include/circle/circle_point_index.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {

// Generator of the circle group over M31.
// Has order 2³¹.
// Verification: 2² + 1268011823² ≡ 1 (mod 2³¹ - 1)
inline CirclePoint<Mersenne31> M31CircleGen() {
  return CirclePoint<Mersenne31>(Mersenne31::FromUnchecked(2),
                                 Mersenne31::FromUnchecked(1268011823));
}

// Converts a CirclePointIndex to an actual CirclePoint over M31.
inline CirclePoint<Mersenne31> ToPoint(CirclePointIndex index) {
  return M31CircleGen().Mul(index.GetIndex());
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CIRCLE_M31_CIRCLE_H_
