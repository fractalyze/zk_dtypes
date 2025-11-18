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

#ifndef ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_TRAITS_H_
#define ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_TRAITS_H_

namespace zk_dtypes {

template <typename Config, typename SFINAE = void>
class PrimeField;

template <typename Config>
class ExtensionField;

template <typename T>
class FiniteFieldTraits;

template <typename Config>
class FiniteFieldTraits<PrimeField<Config>> {
 public:
  using BasePrimeField = PrimeField<Config>;
};

template <typename Config>
class FiniteFieldTraits<ExtensionField<Config>> {
 public:
  using BasePrimeField = typename Config::BasePrimeField;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_TRAITS_H_
