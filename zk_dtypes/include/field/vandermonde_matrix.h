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

#ifndef ZK_DTYPES_INCLUDE_FIELD_VANDERMONDE_MATRIX_H_
#define ZK_DTYPES_INCLUDE_FIELD_VANDERMONDE_MATRIX_H_

#include <array>
#include <cstddef>
#include <type_traits>

namespace zk_dtypes {

// Mixin providing Vandermonde inverse matrix for Toom-Cook multiplication.
//
// Uses CRTP pattern: Derived class must implement CreateConstBaseField(int).
// Template parameter N is the extension degree (e.g., N=4 for Toom-Cook4).
// Returns the (2N - 1) x (2N - 1) inverse Vandermonde matrix V⁻¹.
//
// For N=4, evaluation points t = {0, 1, -1, 2, -2, 3, ∞}, V⁻¹ satisfies
// c = V⁻¹ * v where:
//   v = {v₀, v₁, v₂, v₃, v₄, v₅, v₆} are evaluation results
//   c = {c₀, c₁, c₂, c₃, c₄, c₅, c₆} are polynomial coefficients
template <typename Derived, size_t N>
class VandermondeMatrix {
 public:
  static constexpr size_t kNumEvaluation = 2 * N - 1;

 protected:
  // Creates a constant base field element from an integer.
  // Derived class must implement CreateConstBaseField(int).
  auto C(int x) const {
    return static_cast<const Derived&>(*this).CreateConstBaseField(x);
  }

  // Creates a rational constant (x / y) in the base field.
  auto C2(int x, int y) const { return *(C(x) / C(y)); }

 public:
  template <size_t N2 = N, std::enable_if_t<N2 == 4>* = nullptr>
  const auto& GetVandermondeInverseMatrix() const {
    using F = decltype(C(0));
    static const auto matrix = [this]() {
      // clang-format off
      // V⁻¹ matrix for Toom-Cook4 interpolation
      // Row i gives coefficients for cᵢ in terms of v₀...v₆
      return std::array<std::array<F, kNumEvaluation>, kNumEvaluation>{{  // NOLINT(readability/braces)
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
    }();
    return matrix;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_VANDERMONDE_MATRIX_H_
