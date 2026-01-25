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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQX12_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQX12_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fqx6.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes::bn254 {

// Fq12 = Fq6[w] / (w² - v)
// where v is the generator of Fq6 over Fq2
// Non-residue for quadratic extension: v = {0, 1, 0} in Fq6
REGISTER_EXTENSION_FIELD_WITH_TOWER(FqX12, FqX6, Fq, 2, {{0}, {1}, {0}});

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQX12_H_
