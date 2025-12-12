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

#ifndef ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_TONELLI_SHANKS_H_
#define ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_TONELLI_SHANKS_H_

#include "absl/status/statusor.h"

#include "zk_dtypes/include/field/finite_field_traits.h"

namespace zk_dtypes {

template <typename F>
constexpr absl::StatusOr<F> ComputeTonelliShanksSquareRoot(const F& a) {
  // Finds x such that x² = a.
  // The modulus p is assumed to have the form p = 2ˢ * T + 1,
  // where s is the 2-adicity and T is the "trace".
  // See: https://eprint.iacr.org/2012/685.pdf (Algorithm 5, page 12)
  using BasePrimeField = typename FiniteFieldTraits<F>::BasePrimeField;
  static_assert(BasePrimeField::Config::kHasTwoAdicRootOfUnity);

  if (a.IsZero()) {
    return F::Zero();
  }

  // If a has a square root (i.e., a is a quadratic residue),
  // then by Euler's criterion:
  //   a^((p - 1) / 2) = 1
  // See: https://en.wikipedia.org/wiki/Euler%27s_criterion
  //
  // Since:
  //   (p - 1) / 2 = 2ˢ⁻¹ * T
  //   a^((p - 1)/2) = (aᵀ)^(2ˢ⁻¹) = 1
  // aᵀ is therefore 1 or the 2ˢ⁻¹-th root of unity.
  //
  // If aᵀ == 1, then:
  //   aᵀ * a = (a^((T + 1) / 2))^2
  // So the square root of a is a^((T + 1) / 2).

  constexpr auto exponent = (BasePrimeField::Config::kTrace - 1) >> 1;
  F w = a.Pow(exponent);
  // x = aw = a^((T + 1) / 2)
  F x = a * w;
  // b = xw = aᵀ
  F b = x * w;

  if (!b.IsOne()) {
    // Otherwise, we find (x, b) such that:
    //   1) x² = a * b
    //   2) b is a 2ᵏ⁻¹-th root of unity
    // We repeat until b = 1.

    // z = cᵀ, where c is a known quadratic non-residue.
    // Then:
    //   z^(2ˢ⁻¹) = c^(2ˢ⁻¹ * T) = c^((2ˢ * T) / 2) = c^((p - 1) / 2) = -1
    // (since v = s)
    // TODO(chokobole): construct extension field from a base field.
    F z = BasePrimeField::FromUnchecked(
        BasePrimeField::Config::kTwoAdicRootOfUnity);
    // v = s
    size_t v = size_t{BasePrimeField::Config::kTwoAdicity};
    do {
      size_t k = 0;

      // Find the smallest integer k ≥ 0 such that b^(2ᵏ) = 1.
      F b2k = b;
      while (!b2k.IsOne()) {
        b2k = b2k.Square();
        ++k;
      }

      if (k == size_t{BasePrimeField::Config::kTwoAdicity}) {
        // If k == s, then no square root exists because:
        //   a^(2ˢ * T) = xᵖ⁻¹ = 1
        // implies a is a quadratic non-residue.
        return absl::NotFoundError("No square root exists");
      }

      // clang-format off
      // Update step:
      //   w = z^(2ᵛ⁻ᵏ⁻¹)
      //
      // We then update:
      //   x ← x * w
      //   b ← b * w²
      //   z ← w²
      //   v ← k
      //
      // Correctness:
      //   (x')² = (xw)² = x² * w² = a * b * w² = a * b'
      //   (b')^(2ᵏ⁻¹) = b^(2ᵏ⁻¹) * (w²)^(2ᵏ⁻¹)
      //
      // Why does this work?
      //   a) b^(2ᵏ⁻¹) = -1 because b^(2ᵏ) = 1 and Halving Lemma shows b^(2ᵏ⁻¹) ≠ 1.
      //   b) (w²)^(2ᵏ⁻¹) = -1 because:
      //      ((z^(2ᵛ⁻ᵏ⁻¹))^2)^(2ᵏ⁻¹) = (z^(2ᵛ⁻ᵏ))^(2ᵏ⁻¹) = z^(2ᵛ⁻¹) = -1
      //
      // Therefore, b' remains a 2ᵏ⁻¹-th root of unity.
      // clang-format on
      size_t j = v - k;
      w = z;
      for (size_t i = 1; i < j; ++i) {
        w = w.Square();
      }

      z = w.Square();
      b *= z;
      x *= w;
      v = k;
    } while (!b.IsOne());
  }

  // If we exit the loop with b == 1, then x² = a.
  if (x.Square() != a) {
    return absl::NotFoundError("No square root exists");
  }
  return x;
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_TONELLI_SHANKS_H_
