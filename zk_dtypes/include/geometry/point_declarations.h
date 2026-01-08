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

#ifndef ZK_DTYPES_INCLUDE_GEOMETRY_POINT_DECLARATIONS_H_
#define ZK_DTYPES_INCLUDE_GEOMETRY_POINT_DECLARATIONS_H_

#include <ostream>

#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/group/group.h"

namespace zk_dtypes {

template <typename Curve, typename SFINAE = void>
class AffinePoint;

template <typename Curve, typename SFINAE = void>
class JacobianPoint;

template <typename Curve, typename SFINAE = void>
class PointXyzz;

template <typename T>
struct IsAffinePointImpl {
  constexpr static bool value = false;
};

template <typename Curve>
struct IsAffinePointImpl<AffinePoint<Curve>> {
  constexpr static bool value = true;
};

template <typename T>
constexpr bool IsAffinePoint = IsAffinePointImpl<T>::value;

template <typename T>
struct IsJacobianPointImpl {
  constexpr static bool value = false;
};

template <typename Curve>
struct IsJacobianPointImpl<JacobianPoint<Curve>> {
  constexpr static bool value = true;
};

template <typename T>
constexpr bool IsJacobianPoint = IsJacobianPointImpl<T>::value;

template <typename T>
struct IsPointXyzzImpl {
  constexpr static bool value = false;
};

template <typename Curve>
struct IsPointXyzzImpl<PointXyzz<Curve>> {
  constexpr static bool value = true;
};

template <typename T>
constexpr bool IsPointXyzz = IsPointXyzzImpl<T>::value;

template <typename T>
constexpr bool IsEcPoint =
    IsAffinePoint<T> || IsJacobianPoint<T> || IsPointXyzz<T>;

template <typename T>
struct IsAdditiveGroupImpl<T, std::enable_if_t<IsEcPoint<T>>> {
  constexpr static bool value = true;
};

template <typename T>
struct IsComparableImpl<T, std::enable_if_t<IsEcPoint<T>>> {
  constexpr static bool value = false;
};

template <typename T>
struct AddResult {
  using Type = T;
};

template <typename Curve>
struct AddResult<AffinePoint<Curve>> {
  using Type = JacobianPoint<Curve>;
};

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const AffinePoint<Curve>& point) {
  return point * v;
}

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const JacobianPoint<Curve>& point) {
  return point * v;
}

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const PointXyzz<Curve>& point) {
  return point * v;
}

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const AffinePoint<Curve>& point) {
  return os << point.ToString();
}

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const JacobianPoint<Curve>& point) {
  return os << point.ToString();
}

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const PointXyzz<Curve>& point) {
  return os << point.ToString();
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_GEOMETRY_POINT_DECLARATIONS_H_
