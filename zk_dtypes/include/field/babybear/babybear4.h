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

#ifndef ZK_DTYPES_INCLUDE_FIELD_BABYBEAR_BABYBEAR4_H_
#define ZK_DTYPES_INCLUDE_FIELD_BABYBEAR_BABYBEAR4_H_

#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes {

// Quartic extension field over Babybear: Babybear⁴ = Babybear[u] / (u⁴ - 11)
// W = 11 is a quartic non-residue in Babybear field.
REGISTER_EXTENSION_FIELD(Babybear4, Babybear, 4, 11);

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_BABYBEAR_BABYBEAR4_H_
