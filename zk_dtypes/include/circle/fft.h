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

#ifndef ZK_DTYPES_INCLUDE_CIRCLE_FFT_H_
#define ZK_DTYPES_INCLUDE_CIRCLE_FFT_H_

#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {

// Forward butterfly operation.
// v0' = v0 + v1 * twid
// v1' = v0 - v1 * twid
template <typename F>
void Butterfly(F& v0, F& v1, const Mersenne31& twid) {
  F tmp = v1 * twid;
  v1 = v0 - tmp;
  v0 += tmp;
}

// Inverse butterfly operation.
// v0' = v0 + v1
// v1' = (v0 - v1) * itwid
template <typename F>
void IButterfly(F& v0, F& v1, const Mersenne31& itwid) {
  F tmp = v0;
  v0 = tmp + v1;
  v1 = (tmp - v1) * itwid;
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CIRCLE_FFT_H_
