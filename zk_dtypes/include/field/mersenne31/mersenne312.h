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

#ifndef ZK_DTYPES_INCLUDE_FIELD_MERSENNE31_MERSENNE312_H_
#define ZK_DTYPES_INCLUDE_FIELD_MERSENNE31_MERSENNE312_H_

#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {

// Quadratic extension field over Mersenne31:
// Mersenne31² = Mersenne31[u] / (u² + 1)
// W = -1 is a quadratic non-residue in Mersenne31 field.
REGISTER_EXTENSION_FIELD(Mersenne312, Mersenne31, 2, -1);

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_MERSENNE31_MERSENNE312_H_
