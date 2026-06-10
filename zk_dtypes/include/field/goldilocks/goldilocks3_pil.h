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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS3_PIL_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS3_PIL_H_

#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

namespace zk_dtypes {

// pil2-stark's Goldilocks cubic extension (`Goldilocks3`):
// Goldilocks[u] / (u³ - u - 1), elements [c₀, c₁, c₂] = c₀ + c₁·u + c₂·u².
// A trinomial modulus, so it registers through the monic-modulus path
// (u³ ≡ 1 + u → m = 1, 1, 0); the existing GoldilocksX3 (u³ - 7) is a
// DIFFERENT field representation kept untouched for its holders — pil2
// byte-match is only possible on this one.
// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/goldilocks/src/goldilocks_cubic_extension.hpp
REGISTER_EXTENSION_FIELD_WITH_MONT_MODULUS(Goldilocks3Pil, Goldilocks, 3, 1, 1,
                                           0);

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS3_PIL_H_
