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

#ifndef ZK_DTYPES_INCLUDE_STR_JOIN_H_
#define ZK_DTYPES_INCLUDE_STR_JOIN_H_

#include <sstream>
#include <string>
#include <vector>

namespace zk_dtypes {

template <typename Container, typename Callback>
std::string StrJoin(const Container& container, Callback&& callback,
                    std::string_view delim = ",", std::string_view prefix = "[",
                    std::string_view suffix = "]") {
  size_t size = std::size(container);

  if (size == 0) return "[]";

  std::stringstream ss;
  ss << prefix;
  for (size_t i = 0; i < size - 1; ++i) {
    callback(ss, container[i]);
    ss << delim;
  }
  callback(ss, container[size - 1]);
  ss << suffix;
  return ss.str();
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_STR_JOIN_H_
