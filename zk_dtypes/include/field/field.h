#ifndef ZK_DTYPES_INCLUDE_FIELD_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_FIELD_H_

namespace zk_dtypes {

template <typename T, typename SFINAE = void>
struct IsFieldImpl {
  constexpr static bool value = false;
};

template <typename T>
constexpr bool IsField = IsFieldImpl<T>::value;

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FIELD_H_
