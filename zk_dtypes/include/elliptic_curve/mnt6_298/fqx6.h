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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FQX6_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FQX6_H_

#include "zk_dtypes/include/elliptic_curve/mnt6_298/fqx3.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes::mnt6_298 {

// Fq6 = Fq3[v] / (v^2 - u), where u is the non-residue of Fq3 (u^3 = 5).
// Non-residue for the quadratic tower is the Fq3 element (0, 1, 0) = u,
// matching ark-mnt6-298 (Fp6 = Fp6_2over3, NONRESIDUE = Fq3::new(0, 1, 0)). GT
// of the MNT6 pairing lives here.
REGISTER_EXTENSION_FIELD_TOWER_WITH_MONT(FqX6, FqX3, Fq, 2, {0, 1, 0});

}  // namespace zk_dtypes::mnt6_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT6_298_FQX6_H_
