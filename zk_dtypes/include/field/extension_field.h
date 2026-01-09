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
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/types/span.h"

#include "zk_dtypes/include/always_false.h"
#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/field/cubic_extension_field_operation.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zk_dtypes/include/field/frobenius_coeffs.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"
#include "zk_dtypes/include/field/quartic_extension_field_operation.h"
#include "zk_dtypes/include/pow.h"
#include "zk_dtypes/include/str_join.h"

#define REGISTER_EXTENSION_FIELD_CONFIGS(Name, BaseFieldIn, BasePrimeFieldIn, \
                                         Degree, ...)                         \
  template <typename BaseField>                                               \
  class Name##BaseConfig {                                                    \
   public:                                                                    \
    constexpr static uint32_t kDegreeOverBaseField = Degree;                  \
    constexpr static BaseField kNonResidue = __VA_ARGS__;                     \
  };                                                                          \
                                                                              \
  class Name##StdConfig : public Name##BaseConfig<BaseFieldIn##Std> {         \
   public:                                                                    \
    constexpr static bool kUseMontgomery = false;                             \
    using StdConfig = Name##StdConfig;                                        \
    using BaseField = BaseFieldIn##Std;                                       \
    using BasePrimeField = BasePrimeFieldIn##Std;                             \
  };                                                                          \
                                                                              \
  class Name##Config : public Name##BaseConfig<BaseFieldIn> {                 \
   public:                                                                    \
    constexpr static bool kUseMontgomery = true;                              \
    using StdConfig = Name##StdConfig;                                        \
    using BaseField = BaseFieldIn;                                            \
    using BasePrimeField = BasePrimeFieldIn;                                  \
  };                                                                          \
                                                                              \
  using Name = ExtensionField<Name##Config>;                                  \
  using Name##Std = ExtensionField<Name##StdConfig>

#define REGISTER_EXTENSION_FIELD(Name, BaseFieldIn, Degree, NonResidue)    \
  REGISTER_EXTENSION_FIELD_CONFIGS(Name, BaseFieldIn, BaseFieldIn, Degree, \
                                   NonResidue)

#define REGISTER_EXTENSION_FIELD_WITH_TOWER(Name, BaseFieldIn,             \
                                            BasePrimeFieldIn, Degree, ...) \
  REGISTER_EXTENSION_FIELD_CONFIGS(Name, BaseFieldIn, BasePrimeFieldIn,    \
                                   Degree, __VA_ARGS__)

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
                           _Config, _Config::kDegreeOverBaseField>::Type,
                       public FrobeniusCoeffs<_Config> {
 public:
  using Config = _Config;
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;
  using StdType = ExtensionField<typename Config::StdConfig>;

  constexpr static bool kUseMontgomery = Config::kUseMontgomery;
  constexpr static uint32_t N = Config::kDegreeOverBaseField;
  constexpr static size_t kBitWidth = N * BaseField::kBitWidth;
  constexpr static size_t kByteWidth = N * BaseField::kByteWidth;

  static std::optional<ExtensionFieldMulAlgorithm> mul_algorithm_;
  static std::optional<ExtensionFieldMulAlgorithm> square_algorithm_;

  constexpr ExtensionField() {
    for (size_t i = 0; i < N; ++i) {
      values_[i] = BaseField::Zero();
    }
  }

  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr ExtensionField(T value) {
    if (value == 0) return;
    if (value == 1) {
      *this = One();
      return;
    }

    if (value > 0) {
      *this = ExtensionField({value});
    } else {
      *this = -ExtensionField({-value});
    }
  }

  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr ExtensionField(T value) : ExtensionField({BaseField(value)}) {}

  constexpr ExtensionField(const BaseField& value) { values_[0] = value; }

  template <typename Config2 = Config,
            std::enable_if_t<
                !std::is_same_v<typename Config2::kBaseField,
                                typename Config2::kBasePrimeField>>* = nullptr>
  constexpr ExtensionField(const BasePrimeField& value) {
    AsBasePrimeFields()[0] = value;
  }

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

  constexpr static auto Order() {
    constexpr size_t kLimbNums = BasePrimeField::kLimbNums * ExtensionDegree();
    BigInt<kLimbNums> order = BasePrimeField::Order();
    for (size_t i = 1; i < ExtensionDegree(); ++i) {
      order *= BasePrimeField::Order();
    }
    return order;
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

  constexpr const std::array<BaseField, N>& values() const { return values_; }
  constexpr std::array<BaseField, N>& values() { return values_; }

  constexpr absl::Span<const BasePrimeField> AsBasePrimeFields() const {
    return absl::Span<const BasePrimeField>(
        reinterpret_cast<const BasePrimeField*>(values_.data()),
        ExtensionDegree());
  }

  constexpr absl::Span<BasePrimeField> AsBasePrimeFields() {
    return absl::Span<BasePrimeField>(
        reinterpret_cast<BasePrimeField*>(values_.data()), ExtensionDegree());
  }

  // Iterator for BasePrimeField elements that handles tower structures
  template <bool IsConst>
  class BasePrimeFieldIteratorImpl {
   private:
    constexpr static bool kBaseFieldIsExtensionField =
        BaseField::ExtensionDegree() > 1;

    // Helper to get nested iterator type only if BaseField is ExtensionField
    template <bool IsConstIter, typename Field, typename = void>
    struct NestedIteratorTypeHelper {
      using type = std::nullptr_t;
    };

    template <typename C>
    struct NestedIteratorTypeHelper<false, ExtensionField<C>, void> {
      using type = typename ExtensionField<C>::BasePrimeFieldIterator;
    };

    template <typename C>
    struct NestedIteratorTypeHelper<true, ExtensionField<C>, void> {
      using type = typename ExtensionField<C>::ConstBasePrimeFieldIterator;
    };

   public:
    using FieldType =
        std::conditional_t<IsConst, const ExtensionField, ExtensionField>;
    using BaseFieldType =
        std::conditional_t<IsConst, const BaseField, BaseField>;
    using BasePrimeFieldType =
        std::conditional_t<IsConst, const BasePrimeField, BasePrimeField>;
    using NestedIteratorType =
        typename NestedIteratorTypeHelper<IsConst, BaseField>::type;

    using iterator_category = std::forward_iterator_tag;
    using value_type = BasePrimeFieldType;
    using difference_type = std::ptrdiff_t;
    using pointer = BasePrimeFieldType*;
    using reference = BasePrimeFieldType&;

    BasePrimeFieldIteratorImpl() : BasePrimeFieldIteratorImpl(nullptr) {}
    BasePrimeFieldIteratorImpl(FieldType* field, size_t base_field_idx = 0)
        : field_(field), base_field_idx_(base_field_idx) {
      if (field_ && base_field_idx_ < N) {
        AdvanceToNextBasePrimeField();
      }
    }

    BasePrimeFieldIteratorImpl(const BasePrimeFieldIteratorImpl&) = default;
    BasePrimeFieldIteratorImpl& operator=(const BasePrimeFieldIteratorImpl&) =
        default;

    reference operator*() const {
      DCHECK(field_);
      DCHECK_LT(base_field_idx_, N);
      if constexpr (kBaseFieldIsExtensionField) {
        DCHECK(nested_iterator_.has_value());
        return **nested_iterator_;
      } else {
        return static_cast<reference>(field_->values_[base_field_idx_]);
      }
    }

    pointer operator->() const { return &(**this); }

    BasePrimeFieldIteratorImpl& operator++() {
      if (!field_ || base_field_idx_ >= N) {
        return *this;
      }

      if constexpr (kBaseFieldIsExtensionField) {
        // For a valid, non-end iterator, nested_iterator_ should always have a
        // value.
        DCHECK(nested_iterator_.has_value());
        ++(*nested_iterator_);
        if (*nested_iterator_ == nested_end_) {
          // Finished iterating through the current BaseField, move to the next.
          ++base_field_idx_;
          AdvanceToNextBasePrimeField();
        }
      } else {
        // For prime fields, just move to the next element.
        ++base_field_idx_;
      }
      return *this;
    }

    BasePrimeFieldIteratorImpl operator++(int) {
      BasePrimeFieldIteratorImpl tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const BasePrimeFieldIteratorImpl& other) const {
      if (field_ != other.field_ || base_field_idx_ != other.base_field_idx_) {
        return false;
      }
      if (base_field_idx_ >= N) {
        return true;  // Both are end iterators.
      }
      if constexpr (kBaseFieldIsExtensionField) {
        return nested_iterator_ == other.nested_iterator_;
      }
      return true;  // Not a nested iterator.
    }

    bool operator!=(const BasePrimeFieldIteratorImpl& other) const {
      return !(*this == other);
    }

   private:
    void AdvanceToNextBasePrimeField() {
      if (!field_ || base_field_idx_ >= N) {
        return;
      }

      if constexpr (kBaseFieldIsExtensionField) {
        // BaseField is also an ExtensionField, use its iterator
        BaseFieldType& base_field = field_->values_[base_field_idx_];
        nested_iterator_ = base_field.begin();
        nested_end_ = base_field.end();
      }
      // If BaseField is PrimeField, we're already pointing to it
    }

    FieldType* field_;
    size_t base_field_idx_;
    [[no_unique_address]] std::conditional_t<kBaseFieldIsExtensionField,
                                             std::optional<NestedIteratorType>,
                                             std::nullptr_t> nested_iterator_;
    [[no_unique_address]] std::conditional_t<kBaseFieldIsExtensionField,
                                             NestedIteratorType, std::nullptr_t>
        nested_end_;
  };

  using BasePrimeFieldIterator = BasePrimeFieldIteratorImpl<false>;
  using ConstBasePrimeFieldIterator = BasePrimeFieldIteratorImpl<true>;

  BasePrimeFieldIterator begin() { return BasePrimeFieldIterator(this, 0); }

  BasePrimeFieldIterator end() { return BasePrimeFieldIterator(this, N); }

  ConstBasePrimeFieldIterator begin() const {
    return ConstBasePrimeFieldIterator(this, 0);
  }

  ConstBasePrimeFieldIterator end() const {
    return ConstBasePrimeFieldIterator(this, N);
  }

  ConstBasePrimeFieldIterator cbegin() const {
    return ConstBasePrimeFieldIterator(this, 0);
  }

  ConstBasePrimeFieldIterator cend() const {
    return ConstBasePrimeFieldIterator(this, N);
  }

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
  constexpr StdType MontReduce() const {
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
  constexpr const std::array<BaseField, N>& ToCoeffs() const { return values_; }
  constexpr ExtensionField FromCoeffs(
      const std::array<BaseField, N>& values) const {
    return ExtensionField(values);
  }
  constexpr ExtensionField CreateConst(int64_t value) const {
    return ExtensionField(value);
  }
  constexpr BaseField CreateConstBaseField(int64_t value) const {
    return BaseField(value);
  }
  constexpr const BaseField& NonResidue() const { return Config::kNonResidue; }
  ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    if (mul_algorithm_.has_value()) {
      return mul_algorithm_.value();
    }
    if constexpr (Config::kDegreeOverBaseField == 4) {
      // TODO(chokobole): Choose the best algorithm for quartic extensions.
      return ExtensionFieldMulAlgorithm::kToomCook;
    } else {
      return ExtensionFieldMulAlgorithm::kKaratsuba;
    }
  }
  ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    if (square_algorithm_.has_value()) {
      return square_algorithm_.value();
    }
    if constexpr (Config::kDegreeOverBaseField == 2) {
      static absl::once_flag once;
      static bool is_non_residue_minus_one = false;
      absl::call_once(once, [&]() {
        is_non_residue_minus_one = Config::kNonResidue == BaseField(-1);
      });

      if (is_non_residue_minus_one) {
        return ExtensionFieldMulAlgorithm::kCustom;
      }

      // NOTE(chokobole): This heuristic determines if the custom squaring
      // algorithm outperforms the Karatsuba algorithm. The custom algorithm is
      // selected when n² > 2n + C, where 'n' is the total number of limbs
      // (BaseFieldLimbs * ExtensionDegree) and 'C' represents the cost of
      // multiplication by a non-residue. This model assumes a multiplication
      // cost of O(n²) and an addition cost of O(n).
      if (Config::BasePrimeField::kLimbNums * ExtensionDegree() >= 2) {
        return ExtensionFieldMulAlgorithm::kCustom2;
      } else {
        return ExtensionFieldMulAlgorithm::kKaratsuba;
      }
    } else if constexpr (Config::kDegreeOverBaseField == 3) {
      // NOTE(chokobole): This heuristic determines if the custom squaring
      // algorithm outperforms the Karatsuba algorithm for cubic extensions.
      // The custom algorithm is selected when n² > 4n, where 'n' is the total
      // number of limbs (BaseFieldLimbs * ExtensionDegree). This model
      // assumes a multiplication cost of O(n²) and an addition cost of O(n).
      if (Config::BasePrimeField::kLimbNums * ExtensionDegree() >= 4) {
        return ExtensionFieldMulAlgorithm::kCustom;
      } else {
        return ExtensionFieldMulAlgorithm::kKaratsuba;
      }
    } else {
      // TODO(chokobole): Choose the best algorithm for quartic extensions.
      return ExtensionFieldMulAlgorithm::kToomCook;
    }
  }

 private:
  std::array<BaseField, N> values_;
};

template <typename Config>
std::optional<ExtensionFieldMulAlgorithm>
    ExtensionField<Config>::mul_algorithm_ = std::nullopt;

template <typename Config>
std::optional<ExtensionFieldMulAlgorithm>
    ExtensionField<Config>::square_algorithm_ = std::nullopt;

template <typename Config>
class ExtensionFieldOperationTraits<ExtensionField<Config>> {
 public:
  using BaseField = typename Config::BaseField;
  constexpr static size_t kDegree = Config::kDegreeOverBaseField;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const ExtensionField<Config>& ef) {
  return os << ef.ToString();
}

template <typename T>
struct IsExtensionFieldImpl {
  static constexpr bool value = false;
};

template <typename Config>
struct IsExtensionFieldImpl<ExtensionField<Config>> {
  static constexpr bool value = true;
};

template <typename T>
constexpr bool IsExtensionField = IsExtensionFieldImpl<T>::value;

template <typename T>
struct IsComparableImpl<T, std::enable_if_t<IsExtensionField<T>>> {
  constexpr static bool value = false;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_EXTENSION_FIELD_H_
