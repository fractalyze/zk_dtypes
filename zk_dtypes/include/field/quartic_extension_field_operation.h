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

#ifndef ZK_DTYPES_INCLUDE_FIELD_QUARTIC_EXTENSION_FIELD_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_QUARTIC_EXTENSION_FIELD_OPERATION_H_

#include <array>

#include "zk_dtypes/include/field/extension_field_operation.h"

namespace zk_dtypes {

// Quartic extension field operations using Toom-Cook4 algorithm.
//
// See https://eprint.iacr.org/2006/471.pdf
// Devegili OhEig Scott Dahab --- Multiplication and Squaring on
// Pairing-Friendly Fields.
//
// For Fp4 = Fp[u] / (u⁴ - ξ), where x = x₀ + x₁u + x₂u² + x₃u³.
template <typename Derived>
class QuarticExtensionFieldOperation : public ExtensionFieldOperation<Derived> {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;

  // Returns the 7x7 inverse Vandermonde matrix V⁻¹ for Toom-Cook4
  // interpolation.
  //
  // For evaluation points t = {0, 1, -1, 2, -2, 3, ∞}, V⁻¹ satisfies c = V⁻¹ *
  // v where:
  //   v = {v₀, v₁, v₂, v₃, v₄, v₅, v₆} are evaluation results
  //   c = {c₀, c₁, c₂, c₃, c₄, c₅, c₆} are polynomial coefficients
  const std::array<std::array<BaseField, 7>, 7>& GetVandermondeInverseMatrix()
      const {
    static const auto matrix = ComputeVandermondeInverseMatrix();
    return matrix;
  }

  // Multiplication in Fp4 using Toom-Cook4.
  Derived operator*(const Derived& other) const {
    std::array<BaseField, 4> x =
        static_cast<const Derived&>(*this).ToBaseField();
    std::array<BaseField, 4> y =
        static_cast<const Derived&>(other).ToBaseField();

    std::array<BaseField, 7> evaluations_x = ComputeEvaluations(x);
    std::array<BaseField, 7> evaluations_y = ComputeEvaluations(y);
    std::array<BaseField, 7> evaluations_z;
    for (size_t i = 0; i < 7; ++i) {
      evaluations_z[i] = evaluations_x[i] * evaluations_y[i];
    }

    // Interpolation and reduction
    return this->Reduce(this->Interpolate(evaluations_z));
  }

  // Square in Fp4 using Toom-Cook4.
  Derived Square() const {
    std::array<BaseField, 4> x =
        static_cast<const Derived&>(*this).ToBaseField();

    std::array<BaseField, 7> evaluations_x = ComputeEvaluations(x);
    std::array<BaseField, 7> evaluations_y;
    for (size_t i = 0; i < 7; ++i) {
      evaluations_y[i] = evaluations_x[i].Square();
    }

    // Interpolation and reduction
    return this->Reduce(this->Interpolate(evaluations_y));
  }

 private:
  static std::array<BaseField, 7> ComputeEvaluations(
      const std::array<BaseField, 4>& x) {
    // 1. Basic Doubles
    BaseField x1_2 = x[1].Double();
    BaseField x2_4 = x[2].Double().Double();
    BaseField x3_2 = x[3].Double();
    BaseField x3_8 = x3_2.Double().Double();

    // Needed for f(3) optimization formula
    BaseField x3_3 = x3_2 + x[3];

    // 2. Intermediates
    BaseField v0 = x[0] + x[2];
    BaseField v1 = x[1] + x[3];
    BaseField v2 = x[0] + x2_4;
    BaseField v3 = x1_2 + x3_8;

    std::array<BaseField, 7> f_x;

    // 3. Evaluations
    // f(0) = x₀
    f_x[0] = x[0];
    // f(1) = v₀ + v₁
    f_x[1] = v0 + v1;
    // f(-1) = v₀ - v₁
    f_x[2] = v0 - v1;
    // f(2) = v₂ + v₃
    f_x[3] = v2 + v3;
    // f(-2) = v₂ - v₃
    f_x[4] = v2 - v3;
    // [Optimized f(3)]
    // f(3) = x₀ + 3x₁ + 9x₂ + 27x₃
    //       = f(1) + 2 * (v₁ + 4 * (x₂ + 3x₃))
    BaseField term = x[2] + x3_3;
    term = term.Double().Double();
    term = term + v1;
    term = term.Double();
    f_x[5] = f_x[1] + term;
    // f(∞) = x₃
    f_x[6] = x[3];
    return f_x;
  }

  // Computes the 7x7 inverse Vandermonde matrix for Toom-Cook4 interpolation.
  static std::array<std::array<BaseField, 7>, 7>
  ComputeVandermondeInverseMatrix() {
#define C(x) BaseField(x)
#define C2(x, y) (*(BaseField(x) / BaseField(y)))

    // clang-format off
    // V⁻¹ matrix for Toom-Cook4 interpolation
    // Row i gives coefficients for cᵢ in terms of v₀...v₆
    return {{  // NOLINT(readability/braces)
      // c₀ = 1*v₀ + 0*v₁ + 0*v₂ + 0*v₃ + 0*v₄ + 0*v₅ + 0*v₆
      {C(1), C(0), C(0), C(0), C(0), C(0), C(0)},
      // c₁ = -(1/3)v₀ + 1*v₁ - (1/2)v₂ - (1/4)v₃ + (1/20)v₄ + (1/30)v₅ - 12*v₆
      {C2(-1, 3), C(1), C2(-1, 2), C2(-1, 4), C2(1, 20), C2(1, 30), C(-12)},
      // c₂ = -(5/4)v₀ + (2/3)v₁ + (2/3)v₂ - (1/24)v₃ - (1/24)v₄ + 0*v₅ + 4*v₆
      {C2(-5, 4), C2(2, 3), C2(2, 3), C2(-1, 24), C2(-1, 24), C(0), C(4)},
      // c₃ = (5/12)v₀ - (7/12)v₁ - (1/24)v₂ + (7/24)v₃ - (1/24)v₄ - (1/24)v₅ + 15*v₆
      {C2(5, 12), C2(-7, 12), C2(-1, 24), C2(7, 24), C2(-1, 24), C2(-1, 24), C(15)},
      // c₄ = (1/4)v₀ - (1/6)v₁ - (1/6)v₂ + (1/24)v₃ + (1/24)v₄ + 0*v₅ - 5*v₆
      {C2(1, 4), C2(-1, 6), C2(-1, 6), C2(1, 24), C2(1, 24), C(0), C(-5)},
      // c₅ = -(1/12)v₀ + (1/12)v₁ + (1/24)v₂ - (1/24)v₃ - (1/120)v₄ + (1/120)v₅ - 3*v₆
      {C2(-1, 12), C2(1, 12), C2(1, 24), C2(-1, 24), C2(-1, 120), C2(1, 120), C(-3)},
      // c₆ = 0*v₀ + 0*v₁ + 0*v₂ + 0*v₃ + 0*v₄ + 0*v₅ + 1*v₆
      {C(0), C(0), C(0), C(0), C(0), C(0), C(1)},
    }};
    // clang-format on

#undef C
#undef C2
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_QUARTIC_EXTENSION_FIELD_OPERATION_H_
