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

#ifndef ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_

#include <array>
#include <cstddef>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

template <typename Derived>
class ExtensionFieldOperation {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;

  Derived operator+(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseField();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] + y[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(z);
  }

  Derived operator-(const Derived& other) const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y =
        static_cast<const Derived&>(other).ToBaseField();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] - y[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(z);
  }

  Derived operator-() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = -x[i];
    }
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }

  Derived Double() const {
    std::array<BaseField, kDegree> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = x[i].Double();
    }
    return static_cast<const Derived&>(*this).FromBaseFields(y);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
