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

#ifndef ZK_DTYPES_INCLUDE_FIELD_KOALABEAR_KOALABEAR4_H_
#define ZK_DTYPES_INCLUDE_FIELD_KOALABEAR_KOALABEAR4_H_

#include <stdint.h>

#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"

namespace zk_dtypes {

// Quartic extension field over Koalabear: Koalabear⁴ = Koalabear[u] / (u⁴ - 3)
// W = 3 is a quartic non-residue in Koalabear field.
template <typename BaseField>
class Koalabear4BaseConfig {
 public:
  constexpr static uint32_t kDegreeOverBaseField = 4;
  constexpr static BaseField kNonResidue = 3;
};

class Koalabear4StdConfig : public Koalabear4BaseConfig<KoalabearStd> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Koalabear4StdConfig;

  using BaseField = KoalabearStd;
  using BasePrimeField = KoalabearStd;
};

class Koalabear4Config : public Koalabear4BaseConfig<Koalabear> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Koalabear4StdConfig;

  using BaseField = Koalabear;
  using BasePrimeField = Koalabear;
};

using Koalabear4 = ExtensionField<Koalabear4Config>;
using Koalabear4Std = ExtensionField<Koalabear4StdConfig>;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_KOALABEAR_KOALABEAR4_H_
