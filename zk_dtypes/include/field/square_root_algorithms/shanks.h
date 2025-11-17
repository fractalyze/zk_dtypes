#ifndef ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_SHANKS_H_
#define ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_SHANKS_H_

#include "absl/status/statusor.h"

#include "zk_dtypes/include/field/finite_field_traits.h"

namespace zk_dtypes {

template <typename F>
constexpr absl::StatusOr<F> ComputeShanksSquareRoot(const F& a) {
  // Finds x such that x² = a.
  // Assumes the modulus p satisfies p ≡ 3 (mod 4).
  // See: https://eprint.iacr.org/2012/685.pdf (Algorithm 2, page 9)
  // clang-format off
  // a² = b
  // a⁴ = b²
  //    = b^(p+1) (since b^(p-1) = 1, See https://en.wikipedia.org/wiki/Fermat%27s_little_theorem)
  // a  = b^((p + 1) / 4)
  // clang-format on
  using BasePrimeField = typename FiniteFieldTraits<F>::BasePrimeField;
  static_assert(static_cast<uint64_t>(BasePrimeField::Config::kModulus) % 4 ==
                3);
  constexpr auto exponent = (BasePrimeField::Config::kModulus + 1) >> 2;
  F x = a.Pow(exponent);
  if (x.Square() != a) {
    return absl::NotFoundError("No square root exists");
  }
  return x;
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_SQUARE_ROOT_ALGORITHMS_SHANKS_H_
