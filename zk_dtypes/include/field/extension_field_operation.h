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
#include <type_traits>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"
#include "zk_dtypes/include/field/frobenius_operation.h"

namespace zk_dtypes {

enum class ExtensionFieldMulAlgorithm {
  kCustom,
  kCustom2,
  kKaratsuba,
  kToomCook,
};

template <typename Derived>
class ExtensionFieldOperation : public FrobeniusOperation<Derived> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;
  constexpr static size_t kDegree =
      ExtensionFieldOperationTraits<Derived>::kDegree;

  constexpr Derived operator+(const Derived& other) const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    const std::array<BaseField, kDegree>& y =
        static_cast<const Derived&>(other).ToCoeffs();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] + y[i];
    }
    return static_cast<const Derived&>(*this).FromCoeffs(z);
  }

  constexpr Derived operator-(const Derived& other) const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    const std::array<BaseField, kDegree>& y =
        static_cast<const Derived&>(other).ToCoeffs();
    std::array<BaseField, kDegree> z;
    for (size_t i = 0; i < kDegree; ++i) {
      z[i] = x[i] - y[i];
    }
    return static_cast<const Derived&>(*this).FromCoeffs(z);
  }

  constexpr Derived operator-() const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = -x[i];
    }
    return static_cast<const Derived&>(*this).FromCoeffs(y);
  }

  constexpr Derived Double() const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = x[i].Double();
    }
    return static_cast<const Derived&>(*this).FromCoeffs(y);
  }

  // Scalar multiplication: ExtensionField * BaseField
  constexpr Derived operator*(const BaseField& scalar) const {
    const std::array<BaseField, kDegree>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    std::array<BaseField, kDegree> y;
    for (size_t i = 0; i < kDegree; ++i) {
      y[i] = x[i] * scalar;
    }
    return static_cast<const Derived&>(*this).FromCoeffs(y);
  }

  // BaseField arithmetic: only the constant coefficient is affected.
  // SFINAE ensures these only match exact BaseField arguments, avoiding
  // ambiguity with integer literals that can implicitly convert to both
  // BaseField and ExtensionField (Derived).
  template <typename B = BaseField,
            std::enable_if_t<std::is_same_v<B, BaseField>>* = nullptr>
  constexpr Derived operator+(const B& other) const {
    std::array<BaseField, kDegree> z =
        static_cast<const Derived&>(*this).ToCoeffs();
    z[0] += other;
    return static_cast<const Derived&>(*this).FromCoeffs(z);
  }

  template <typename B = BaseField,
            std::enable_if_t<std::is_same_v<B, BaseField>>* = nullptr>
  constexpr Derived operator-(const B& other) const {
    std::array<BaseField, kDegree> z =
        static_cast<const Derived&>(*this).ToCoeffs();
    z[0] -= other;
    return static_cast<const Derived&>(*this).FromCoeffs(z);
  }

  template <typename B = BaseField,
            std::enable_if_t<std::is_same_v<B, BaseField>>* = nullptr>
  constexpr Derived operator/(const B& scalar) const {
    return static_cast<const Derived&>(*this) * scalar.Inverse();
  }

  constexpr Derived operator/(const Derived& other) const {
    return static_cast<const Derived&>(*this) * other.Inverse();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_OPERATION_H_
