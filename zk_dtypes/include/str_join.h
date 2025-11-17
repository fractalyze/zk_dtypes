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
