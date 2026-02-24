/* Copyright 2026 The zk_dtypes Authors.

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

#include "zk_dtypes/include/signed_big_int.h"

#include <vector>

#include "absl/numeric/int128.h"

namespace zk_dtypes::internal {

namespace {

// Two's complement negation to get absolute value.
std::vector<uint64_t> NegateLimbs(const uint64_t* limbs, size_t limb_nums) {
  std::vector<uint64_t> abs_limbs(limb_nums);
  for (size_t i = 0; i < limb_nums; ++i) {
    abs_limbs[i] = ~limbs[i];
  }
  uint64_t carry = 1;
  for (size_t i = 0; i < limb_nums; ++i) {
    absl::uint128 sum = absl::uint128{abs_limbs[i]} + carry;
    abs_limbs[i] = absl::Uint128Low64(sum);
    carry = static_cast<uint64_t>(absl::Uint128High64(sum));
    if (carry == 0) break;
  }
  return abs_limbs;
}

}  // namespace

std::string SignedLimbsToString(const uint64_t* limbs, size_t limb_nums) {
  DCHECK(limbs);
  DCHECK_GT(limb_nums, 0);

  bool is_neg = (limbs[limb_nums - 1] >> 63) != 0;
  if (!is_neg) {
    return LimbsToString(limbs, limb_nums);
  }

  std::vector<uint64_t> abs_limbs = NegateLimbs(limbs, limb_nums);
  return "-" + LimbsToString(abs_limbs.data(), limb_nums);
}

std::string SignedLimbsToHexString(const uint64_t* limbs, size_t limb_nums,
                                   bool pad_zero) {
  DCHECK(limbs);
  DCHECK_GT(limb_nums, 0);

  bool is_neg = (limbs[limb_nums - 1] >> 63) != 0;
  if (!is_neg) {
    return LimbsToHexString(limbs, limb_nums, pad_zero);
  }

  std::vector<uint64_t> abs_limbs = NegateLimbs(limbs, limb_nums);
  return "-" + LimbsToHexString(abs_limbs.data(), limb_nums, pad_zero);
}

}  // namespace zk_dtypes::internal
