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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_FQX4_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_FQX4_H_

#include "zk_dtypes/include/elliptic_curve/mnt4_298/fqx2.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes::mnt4_298 {

// Fq4 = Fq2[v] / (v^2 - u), where u is the non-residue of Fq2 (u^2 = 17).
// Non-residue for the quadratic tower is the Fq2 element (0, 1) = u, matching
// ark-mnt4-298 (Fq4 NONRESIDUE = Fq2::new(Fq::ZERO, Fq::ONE)). GT of the MNT4
// pairing lives here.
REGISTER_EXTENSION_FIELD_TOWER_WITH_MONT(FqX4, FqX2, Fq, 2, {0, 1});

}  // namespace zk_dtypes::mnt4_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_FQX4_H_
