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

#ifndef ZK_DTYPES_INCLUDE_GEOMETRY_POINT_TRAITS_H_
#define ZK_DTYPES_INCLUDE_GEOMETRY_POINT_TRAITS_H_

#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"

namespace zk_dtypes {

template <typename Curve>
class PointTraits;

template <typename _Curve>
class PointTraits<AffinePoint<_Curve>> {
 public:
  using Curve = _Curve;

  constexpr static CurveType kType = Curve::kType;
  constexpr static size_t kNumCoords = 2;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using JacobianPoint = zk_dtypes::JacobianPoint<Curve>;
  using PointXyzz = zk_dtypes::PointXyzz<Curve>;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
};

template <typename _Curve>
class PointTraits<JacobianPoint<_Curve>> {
 public:
  using Curve = _Curve;

  constexpr static CurveType kType = Curve::kType;
  constexpr static size_t kNumCoords = 3;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using JacobianPoint = zk_dtypes::JacobianPoint<Curve>;
  using PointXyzz = zk_dtypes::PointXyzz<Curve>;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
};

template <typename _Curve>
class PointTraits<PointXyzz<_Curve>> {
 public:
  using Curve = _Curve;

  constexpr static CurveType kType = Curve::kType;
  constexpr static size_t kNumCoords = 4;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using JacobianPoint = zk_dtypes::JacobianPoint<Curve>;
  using PointXyzz = zk_dtypes::PointXyzz<Curve>;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_GEOMETRY_POINT_TRAITS_H_
