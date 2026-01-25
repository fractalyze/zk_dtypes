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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQX6_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQX6_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fqx2.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes::bn254 {

// Fq6 = Fq2[v] / (v³ - (9 + u))
// where u is the non-residue of Fq2 (u² = -1)
// Non-residue for cubic extension: 9 + u = {9, 1} in Fq2
REGISTER_EXTENSION_FIELD_WITH_TOWER(FqX6, FqX2, Fq, 3, {{9, 1}});

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQX6_H_
