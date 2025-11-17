#ifndef ZK_DTYPES_INCLUDE_BATCH_INVERSE_H_
#define ZK_DTYPES_INCLUDE_BATCH_INVERSE_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/template_util.h"

namespace zk_dtypes {

// Batch inverse: [a₁, a₂, ..., aₙ] -> [c * a₁⁻¹, c * a₂⁻¹, ... , c * aₙ⁻¹]
template <typename T, typename R, typename value_type = typename T::value_type>
absl::Status BatchInverse(const T& inputs, R* outputs,
                          const value_type& c = value_type::One()) {
  if constexpr (internal::has_resize_v<R>) {
    outputs->resize(std::size(inputs));
  } else {
    if (std::size(inputs) != std::size(*outputs)) {
      return absl::InvalidArgumentError(
          absl::Substitute("size do not match $0 vs $1", std::size(inputs),
                           std::size(*outputs)));
    }
  }

  // First pass: compute [a₁, a₁ * a₂, ..., a₁ * a₂ * ... * aₙ]
  std::vector<value_type> productions;
  productions.reserve(std::size(inputs) + 1);
  productions.push_back(value_type::One());
  value_type product = value_type::One();
  for (const value_type& input : inputs) {
    if (ABSL_PREDICT_TRUE(!input.IsZero())) {
      product *= input;
      productions.push_back(product);
    }
  }

  // Invert product.
  // (a₁ * a₂ * ... *  aₙ)⁻¹
  absl::StatusOr<value_type> product_inv_or = product.Inverse();
  if (!product_inv_or.ok()) return product_inv_or.status();
  value_type product_inv = *product_inv_or;

  // Multiply product_inv by c, so all inverses will be scaled by c.
  // c * (a₁ * a₂ * ... *  aₙ)⁻¹
  if (ABSL_PREDICT_FALSE(!c.IsOne())) product_inv *= c;

  // Second pass: iterate backwards to compute inverses.
  //              [c * a₁⁻¹, c * a₂,⁻¹ ..., c * aₙ⁻¹]
  auto prod_it = productions.rbegin();
  ++prod_it;
  auto output_it = outputs->rbegin();
  for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
    const value_type& input = *it;
    if (ABSL_PREDICT_TRUE(!input.IsZero())) {
      // c * (a₁ * a₂ * ... *  aᵢ)⁻¹ * aᵢ = c * (a₁ * a₂ * ... *  aᵢ₋₁)⁻¹
      value_type new_product_inv = product_inv * input;
      // v = c * (a₁ * a₂ * ... *  aᵢ)⁻¹ * (a₁ * a₂ * ... aᵢ₋₁) = c * aᵢ⁻¹
      *(output_it++) = product_inv * (*(prod_it++));
      product_inv = new_product_inv;
    } else {
      *(output_it++) = value_type::Zero();
    }
  }
  return absl::OkStatus();
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_BATCH_INVERSE_H_
