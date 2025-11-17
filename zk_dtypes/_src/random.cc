#include "zk_dtypes/include/random.h"

namespace zk_dtypes {

absl::BitGen& GetAbslBitGen() {
  static absl::BitGen bitgen;
  return bitgen;
}

}  // namespace zk_dtypes
