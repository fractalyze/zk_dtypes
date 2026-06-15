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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_FQX2_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_FQX2_H_

#include "zk_dtypes/include/elliptic_curve/mnt4_298/fq.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes::mnt4_298 {

// Fq2 = Fq[u] / (u^2 - 17). The quadratic non-residue 17 matches ark-mnt4-298
// (and libff mnt4_init).
REGISTER_EXTENSION_FIELD_WITH_MONT(FqX2, Fq, 2, 17);

}  // namespace zk_dtypes::mnt4_298

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_MNT4_298_FQX2_H_
