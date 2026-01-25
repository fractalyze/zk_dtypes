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

#ifndef ZK_DTYPES_INCLUDE_FIELD_CYCLOTOMIC_OPERATION_H_
#define ZK_DTYPES_INCLUDE_FIELD_CYCLOTOMIC_OPERATION_H_

#include <array>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace zk_dtypes {

// Cyclotomic operations for extension fields.
// These operations are optimized for elements in the cyclotomic subgroup,
// which appear in pairing-based cryptography (e.g., final exponentiation).
//
// For a quadratic extension F_{p^2k} over F_{p^k}, the cyclotomic subgroup
// consists of elements x where x^{p^k + 1} = 1 (i.e., Norm(x) = 1).
// For such elements, the inverse equals the conjugate: x^{-1} = conj(x).
template <typename Derived>
class CyclotomicOperation {
 public:
  using BaseField = typename ExtensionFieldOperationTraits<Derived>::BaseField;

  // Cyclotomic inverse for quadratic extension fields.
  // For elements in the cyclotomic subgroup, the inverse is the conjugate.
  // This is because x * conj(x) = Norm(x) = 1 for x in the cyclotomic subgroup.
  Derived CyclotomicInverse() const {
    const std::array<BaseField, 2>& x =
        static_cast<const Derived&>(*this).ToCoeffs();
    return static_cast<const Derived&>(*this).FromCoeffs({x[0], -x[1]});
  }

  // Cyclotomic square for quadratic extension fields.
  // Default implementation uses regular square. Derived classes can override
  // with optimized algorithms (e.g., Granger-Scott for Fp12).
  Derived CyclotomicSquare() const {
    return static_cast<const Derived&>(*this).Square();
  }

  // Cyclotomic exponentiation using square-and-multiply.
  // Uses CyclotomicSquare for squaring operations, which can be faster
  // than regular squaring for elements in the cyclotomic subgroup.
  template <size_t N>
  Derived CyclotomicPow(const BigInt<N>& exponent) const {
    const Derived& self = static_cast<const Derived&>(*this);
    if (self.IsZero()) return Derived::Zero();

    Derived result = Derived::One();
    bool found_nonzero = false;

    // Process bits from most significant to least significant
    for (size_t i = N; i > 0; --i) {
      uint64_t limb = exponent[i - 1];
      for (int bit = 63; bit >= 0; --bit) {
        if (found_nonzero) {
          result = result.CyclotomicSquare();
        }
        if ((limb >> bit) & 1) {
          if (found_nonzero) {
            result *= self;
          } else {
            result = self;
            found_nonzero = true;
          }
        }
      }
    }
    return result;
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_CYCLOTOMIC_OPERATION_H_
