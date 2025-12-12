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

#ifndef ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS3_H_
#define ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS3_H_

#include <stdint.h>

#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

namespace zk_dtypes {

// Cubic extension field over Goldilocks: Goldilocks³ = Goldilocks[u] / (u³ - 7)
// W = 7 is a cubic non-residue in Goldilocks field.
template <typename BaseField>
class Goldilocks3BaseConfig {
 public:
  constexpr static uint32_t kDegreeOverBaseField = 3;
  constexpr static BaseField kNonResidue = 7;
};

class Goldilocks3StdConfig : public Goldilocks3BaseConfig<GoldilocksStd> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Goldilocks3StdConfig;

  using BaseField = GoldilocksStd;
  using BasePrimeField = GoldilocksStd;
};

class Goldilocks3Config : public Goldilocks3BaseConfig<Goldilocks> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Goldilocks3StdConfig;

  using BaseField = Goldilocks;
  using BasePrimeField = Goldilocks;
};

using Goldilocks3 = ExtensionField<Goldilocks3Config>;
using Goldilocks3Std = ExtensionField<Goldilocks3StdConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_GOLDILOCKS_GOLDILOCKS3_H_
