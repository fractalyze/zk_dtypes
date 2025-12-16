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

#ifndef ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/always_false.h"
#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/cubic_extension_field_operation.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"
#include "zk_dtypes/include/field/quartic_extension_field_operation.h"
#include "zk_dtypes/include/pow.h"
#include "zk_dtypes/include/str_join.h"

namespace zk_dtypes {

// Forward declaration
template <typename _Config>
class ExtensionField;

// Selects the appropriate extension field operation based on degree.
template <typename Config, size_t Degree>
struct ExtensionFieldOperationSelector {
  static_assert(
      AlwaysFalse<Config>,
      "Unsupported extension degree. Only 2, 3, and 4 are supported.");
};

template <typename Config>
struct ExtensionFieldOperationSelector<Config, 2> {
  using Type = QuadraticExtensionFieldOperation<ExtensionField<Config>>;
};

template <typename Config>
struct ExtensionFieldOperationSelector<Config, 3> {
  using Type = CubicExtensionFieldOperation<ExtensionField<Config>>;
};

template <typename Config>
struct ExtensionFieldOperationSelector<Config, 4> {
  using Type = QuarticExtensionFieldOperation<ExtensionField<Config>>;
};

template <typename _Config>
class ExtensionField : public FiniteField<ExtensionField<_Config>>,
                       public ExtensionFieldOperationSelector<
                           _Config, _Config::kDegreeOverBaseField>::Type {
 public:
  using Config = _Config;
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using StdType = ExtensionField<typename Config::StdConfig>;

  constexpr static bool kUseMontgomery = Config::kUseMontgomery;
  constexpr static uint32_t N = Config::kDegreeOverBaseField;
  constexpr static size_t kBitWidth = N * BaseField::kBitWidth;
  constexpr static size_t kByteWidth = N * BaseField::kByteWidth;

  constexpr ExtensionField() {
    for (size_t i = 0; i < N; ++i) {
      values_[i] = BaseField::Zero();
    }
  }

  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr ExtensionField(T value) {
    if (value >= 0) {
      *this = ExtensionField({value});
    } else {
      *this = -ExtensionField({-value});
    }
  }

  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr ExtensionField(T value) : ExtensionField({BigInt<N>(value)}) {}

  constexpr ExtensionField(const BaseField& value) { values_[0] = value; }

  constexpr ExtensionField(std::initializer_list<BaseField> values) {
    DCHECK_LE(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < values.size(); ++i, ++it) {
      values_[i] = *it;
    }
    for (size_t i = values.size(); i < N; ++i) {
      values_[i] = BaseField::Zero();
    }
  }
  constexpr ExtensionField(const std::array<BaseField, N>& values)
      : values_(values) {}

  constexpr static uint32_t ExtensionDegree() {
    return N * BaseField::ExtensionDegree();
  }

  constexpr static ExtensionField Zero() { return ExtensionField(); }

  constexpr static ExtensionField One() {
    return ExtensionField({BaseField::One()});
  }

  constexpr static ExtensionField Random() {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(ret.values_); ++i) {
      ret[i] = BaseField::Random();
    }
    return ret;
  }

  // Returns precomputed Frobenius coefficients for all φᴱ (E = 1, ..., N - 1):
  // coeffs[E - 1][i - 1] = ξ^(i * (pᴱ - 1) / n) for i = 1, ..., N - 1.
  //
  // For Fp3: a = (a₁, a₂, a₃, a₄) where
  //   a₁ = ξ^((p - 1) / 3), a₂ = ξ^(2(p - 1) / 3)
  //   a₃ = ξ^((p² - 1) / 3), a₄ = ξ^(2(p² - 1) / 3)
  // - φ¹(x) = (x₀, x₁ * a₁, x₂ * a₂)
  // - φ²(x) = (x₀, x₁ * a₃, x₂ * a₄)
  //
  // See:
  // https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.2.-optimized-computation-when
  static const std::array<std::array<BaseField, N - 1>, N - 1>&
  GetFrobeniusCoeffs() {
    static const auto coeffs = []() {
      // Use larger BigInt to avoid overflow when computing pᵉ.
      constexpr size_t kLimbNums = BasePrimeField::kLimbNums * N;
      BigInt<kLimbNums> p(BasePrimeField::Config::kModulus);
      BaseField nr = Config::kNonResidue;

      std::array<std::array<BaseField, N - 1>, N - 1> result{};
      // p_e = pᵉ, computed iteratively
      BigInt<kLimbNums> p_e = p;
      BigInt<kLimbNums> n_big(N);
      for (size_t e = 1; e < N; ++e) {
        // qₑ = (pᵉ - 1) / n
        auto q_e = ((p_e - 1) / n_big).value();
        BaseField nr_q_e = zk_dtypes::Pow(nr, q_e);

        result[e - 1][0] = nr_q_e;
        for (size_t i = 1; i < N - 1; ++i) {
          result[e - 1][i] = result[e - 1][i - 1] * nr_q_e;
        }
        p_e = p_e * p;
      }
      return result;
    }();
    return coeffs;
  }

  constexpr const std::array<BaseField, N>& values() const { return values_; }
  constexpr std::array<BaseField, N>& values() { return values_; }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < std::size(values_); ++i) {
      if (!values_[i].IsZero()) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
    for (size_t i = 1; i < std::size(values_); ++i) {
      if (!values_[i].IsZero()) return false;
    }
    return values_[0].IsOne();
  }

  // See
  // https://github.com/Consensys/gnark-crypto/blob/43897fd/field/generator/internal/templates/extensions/e2.go.tmpl#L29-L37
  constexpr bool LexicographicallyLargest() const {
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (!values_[i].IsZero()) {
        return values_[i].LexicographicallyLargest();
      }
    }
    return false;
  }

  constexpr ExtensionField& operator+=(const ExtensionField& other) {
    for (size_t i = 0; i < std::size(values_); ++i) {
      values_[i] += other[i];
    }
    return *this;
  }

  constexpr ExtensionField& operator-=(const ExtensionField& other) {
    for (size_t i = 0; i < std::size(values_); ++i) {
      values_[i] -= other[i];
    }
    return *this;
  }

  using ExtensionFieldOperationBase = typename ExtensionFieldOperationSelector<
      _Config, _Config::kDegreeOverBaseField>::Type;
  using ExtensionFieldOperationBase::operator*;

  constexpr ExtensionField operator*(const BaseField& other) const {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = values_[i] * other;
    }
    return ret;
  }

  constexpr ExtensionField& operator*=(const ExtensionField& other) {
    return *this = *this * other;
  }

  constexpr ExtensionField& operator*=(const BaseField& other) {
    return *this = *this * other;
  }

  template <size_t N>
  constexpr ExtensionField Pow(const BigInt<N>& exponent) const {
    return zk_dtypes::Pow(*this, exponent);
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  constexpr ExtensionField Pow(T exponent) const {
    return zk_dtypes::Pow(*this, BigInt<1>(exponent));
  }

  constexpr BaseField& operator[](size_t i) {
    DCHECK_LT(i, N);
    return values_[i];
  }
  constexpr const BaseField& operator[](size_t i) const {
    DCHECK_LT(i, N);
    return values_[i];
  }

  constexpr bool operator==(const ExtensionField& other) const {
    for (size_t i = 0; i < std::size(values_); ++i) {
      if (values_[i] != other[i]) return false;
    }
    return true;
  }
  constexpr bool operator!=(const ExtensionField& other) const {
    return !operator==(other);
  }

  template <typename Config2 = Config,
            std::enable_if_t<Config2::kUseMontgomery>* = nullptr>
  StdType MontReduce() const {
    StdType ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = values_[i].MontReduce();
    }
    return ret;
  }

  std::string ToString() const {
    return StrJoin(values_, [](std::ostream& os, const BaseField& value) {
      os << value.ToString();
    });
  }
  std::string ToHexString(bool pad_zero = false) const {
    return StrJoin(values_,
                   [pad_zero](std::ostream& os, const BaseField& value) {
                     os << value.ToHexString(pad_zero);
                   });
  }

  // ExtensionFieldOperation methods
  std::array<BaseField, N> ToBaseField() const { return values_; }
  ExtensionField FromBaseFields(const std::array<BaseField, N>& values) const {
    return ExtensionField(values);
  }
  size_t DegreeOverBasePrimeField() const {
    return N * BaseField::ExtensionDegree();
  }
  const BaseField& NonResidue() const { return Config::kNonResidue; }

 private:
  std::array<BaseField, N> values_;
};

template <typename Config>
class ExtensionFieldOperationTraits<ExtensionField<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  constexpr static size_t kDegree = Config::kDegreeOverBaseField;

  constexpr static bool kHasHint = true;
  constexpr static bool kNonResidueIsMinusOne =
      Config::kNonResidue == BaseField(-1);
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const ExtensionField<Config>& ef) {
  return os << ef.ToString();
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_H_
