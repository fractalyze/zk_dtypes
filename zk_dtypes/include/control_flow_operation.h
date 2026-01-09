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

#ifndef ZK_DTYPES_INCLUDE_CONTROL_FLOW_OPERATION_H_
#define ZK_DTYPES_INCLUDE_CONTROL_FLOW_OPERATION_H_

#include "zk_dtypes/include/control_flow_operation_forward.h"

namespace zk_dtypes {

template <>
class ControlFlowOperation<bool> {
 public:
  template <typename F1, typename F2>
  constexpr static auto If(bool condition, F1&& then, F2&& otherwise)
      -> decltype(then()) {
    if (condition) {
      return then();
    } else {
      return otherwise();
    }
  }

  template <typename R>
  constexpr static bool Equal(const R& a, const R& b) {
    return a == b;
  }
  template <typename R>
  constexpr static bool NotEqual(const R& a, const R& b) {
    return a != b;
  }
  constexpr static bool And(bool a, bool b) { return a && b; }
  constexpr static bool Or(bool a, bool b) { return a || b; }
  constexpr static bool Not(bool a) { return !a; }

  constexpr static bool True() { return true; }
  constexpr static bool False() { return false; }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_CONTROL_FLOW_OPERATION_H_
