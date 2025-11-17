#include "zk_dtypes/include/big_int.h"

#include <string.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

namespace zk_dtypes::internal {

namespace {

template <size_t Base>
absl::Status DoStringToLimbs(std::string_view str, uint64_t* limbs,
                             size_t limb_nums) {
  static_assert(Base == 10 || Base == 16, "Only Base 10 and 16 supported.");

  DCHECK(limbs);
  DCHECK_GT(limb_nums, 0);

  // Clear limbs
  std::memset(limbs, 0, sizeof(uint64_t) * limb_nums);

  // Skip optional prefix
  if constexpr (Base == 16) {
    if (str.size() >= 2 && str[0] == '0' && (str[1] == 'x' || str[1] == 'X')) {
      str.remove_prefix(2);
    }
  }

  for (char c : str) {
    uint8_t digit = 0;
    if constexpr (Base == 10) {
      if ('0' <= c && c <= '9')
        digit = c - '0';
      else
        return absl::InvalidArgumentError("Invalid character in input string.");
    } else {
      if ('0' <= c && c <= '9')
        digit = c - '0';
      else if (Base == 16 && 'a' <= c && c <= 'f')
        digit = c - 'a' + 10;
      else if (Base == 16 && 'A' <= c && c <= 'F')
        digit = c - 'A' + 10;
      else
        return absl::InvalidArgumentError("Invalid character in input string.");
    }

    // Multiply limbs by Base
    uint64_t carry = digit;
    for (size_t i = 0; i < limb_nums; ++i) {
      absl::uint128 product = absl::uint128{limbs[i]} * Base + carry;
      limbs[i] = absl::Uint128Low64(product);
      carry = absl::Uint128High64(product);
    }

    if (carry != 0) {
      return absl::OutOfRangeError("Value too large for limb buffer.");
    }
  }

  return absl::OkStatus();
}

template <size_t Base>
std::string DoLimbsToString(const uint64_t* limbs, size_t limb_nums,
                            bool pad_zero) {
  static_assert(Base == 10 || Base == 16, "Only Base 10 and 16 supported.");

  DCHECK(limbs);
  DCHECK_GT(limb_nums, 0);

  if constexpr (Base == 16) {
    std::ostringstream oss;
    bool leading = true;

    for (size_t i = limb_nums - 1; i != SIZE_MAX; --i) {
      uint64_t limb = limbs[i];
      if (leading) {
        if (limb == 0) continue;
        oss << std::hex << limb;
        leading = false;
      } else {
        oss << std::setw(16) << std::setfill('0') << std::hex << limb;
      }
    }

    return oss.str().empty() ? "0" : oss.str();
  } else {
    std::vector<uint64_t> temp(limbs, limbs + limb_nums);
    std::string result;

    auto is_zero = [](const std::vector<uint64_t>& v) {
      return std::all_of(v.begin(), v.end(), [](uint64_t x) { return x == 0; });
    };

    while (!is_zero(temp)) {
      uint64_t rem = 0;
      for (size_t i = limb_nums - 1; i != SIZE_MAX; --i) {
        absl::uint128 cur = (absl::uint128{rem} << 64) + temp[i];
        temp[i] = absl::Uint128Low64(cur / 10);
        rem = absl::Uint128Low64(cur % 10);
      }
      result.push_back('0' + rem);
    }

    if (result.empty())
      result = "0";
    else
      std::reverse(result.begin(), result.end());
    return result;
  }
}

std::string MaybePrepend0x(std::string_view str) {
  constexpr const char* k0x = "0x";

  if (absl::StartsWith(str, k0x)) {
    return std::string(str);
  }
  return absl::StrCat(k0x, str);
}

}  // namespace

absl::Status StringToLimbs(std::string_view str, uint64_t* limbs,
                           size_t limb_nums) {
  return DoStringToLimbs<10>(str, limbs, limb_nums);
}

absl::Status HexStringToLimbs(std::string_view str, uint64_t* limbs,
                              size_t limb_nums) {
  return DoStringToLimbs<16>(str, limbs, limb_nums);
}

std::string LimbsToString(const uint64_t* limbs, size_t limb_nums) {
  return DoLimbsToString<10>(limbs, limb_nums, false);
}

std::string LimbsToHexString(const uint64_t* limbs, size_t limb_nums,
                             bool pad_zero) {
  return MaybePrepend0x(DoLimbsToString<16>(limbs, limb_nums, pad_zero));
}

}  // namespace zk_dtypes::internal
