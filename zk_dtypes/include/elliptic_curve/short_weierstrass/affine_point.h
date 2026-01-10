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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point_operation.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_base.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"

namespace zk_dtypes {

template <typename _Curve>
class AffinePoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final : public PointBase<AffinePoint<_Curve>>,
            public AffinePointOperation<AffinePoint<_Curve>> {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType = AffinePoint<SwCurve<typename Curve::Config::StdConfig>>;

  using JacobianPoint = zk_dtypes::JacobianPoint<Curve>;
  using PointXyzz = zk_dtypes::PointXyzz<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 2;

  constexpr AffinePoint() : AffinePoint(BaseField::Zero(), BaseField::Zero()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr AffinePoint(T value) : AffinePoint(ScalarField(value)) {}
  constexpr AffinePoint(ScalarField value) {
    AffinePoint point = (AffinePoint::Generator() * value).ToAffine();
    this->coords_ = point.coords_;
  }
  constexpr AffinePoint(const std::array<BaseField, 2>& coords)
      : PointBase<AffinePoint<_Curve>>(coords) {}
  constexpr AffinePoint(const BaseField& x, const BaseField& y)
      : PointBase<AffinePoint<_Curve>>({x, y}) {}

  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint One() { return Generator(); }

  constexpr static AffinePoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY};
  }

  constexpr static absl::StatusOr<AffinePoint> CreateFromX(const BaseField& x) {
    return Curve::GetPointFromX(x);
  }

  constexpr static AffinePoint Random() {
    return JacobianPoint::Random().ToAffine();
  }

  constexpr const BaseField& x() const { return this->coords_[0]; }
  constexpr const BaseField& y() const { return this->coords_[1]; }

  constexpr JacobianPoint operator*(const ScalarField& v) const {
    if constexpr (kUseMontgomery) {
      return ScalarMul(this->ToJacobian(), v.MontReduce().value());
    } else {
      return ScalarMul(this->ToJacobian(), v.value());
    }
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x().MontReduce(), y().MontReduce()};
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_H_
