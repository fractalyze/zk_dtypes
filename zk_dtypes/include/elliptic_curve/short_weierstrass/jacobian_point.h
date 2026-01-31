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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_

#include <array>
#include <string>
#include <type_traits>

#include "absl/strings/substitute.h"

#include "zk_dtypes/include/batch_inverse.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point_operation.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_base.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"
#include "zk_dtypes/include/template_util.h"

namespace zk_dtypes {

template <typename _Curve>
class JacobianPoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final : public PointBase<JacobianPoint<_Curve>>,
            public JacobianPointOperation<JacobianPoint<_Curve>> {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType = JacobianPoint<SwCurve<typename Curve::Config::StdConfig>>;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using PointXyzz = zk_dtypes::PointXyzz<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kByteWidth = BaseField::kByteWidth * 3;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 3;

  constexpr JacobianPoint()
      : JacobianPoint(BaseField::One(), BaseField::One(), BaseField::Zero()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr JacobianPoint(T value) : JacobianPoint(ScalarField(value)) {}
  constexpr JacobianPoint(ScalarField value) {
    JacobianPoint point = JacobianPoint::Generator() * value;
    this->coords_ = point.coords_;
  }
  constexpr JacobianPoint(const std::array<BaseField, 3>& coords)
      : PointBase<JacobianPoint<_Curve>>(coords) {}
  constexpr JacobianPoint(const BaseField& x, const BaseField& y,
                          const BaseField& z)
      : PointBase<JacobianPoint<_Curve>>({x, y, z}) {}

  constexpr static JacobianPoint Zero() { return JacobianPoint(); }

  constexpr static JacobianPoint One() { return Generator(); }

  constexpr static JacobianPoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY, BaseField::One()};
  }

  constexpr static JacobianPoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return this->coords_[0]; }
  constexpr const BaseField& y() const { return this->coords_[1]; }
  constexpr const BaseField& z() const { return this->coords_[2]; }

  constexpr JacobianPoint operator*(const ScalarField& v) const {
    if constexpr (kUseMontgomery) {
      return ScalarMul(*this, v.MontReduce().value());
    } else {
      return ScalarMul(*this, v.value());
    }
  }

  constexpr JacobianPoint& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

  // The jacobian point X, Y, Z is represented in the xyzz
  // coordinates as X, Y, Z², Z³.
  constexpr PointXyzz ToXyzz() const {
    BaseField zz = z().Square();
    return {x(), y(), zz, zz * z()};
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x().MontReduce(), y().MontReduce(), z().MontReduce()};
  }

  template <typename JacobianContainer, typename AffineContainer>
  static absl::Status BatchToAffine(const JacobianContainer& jacobian_points,
                                    AffineContainer* affine_points) {
    if constexpr (internal::has_resize_v<AffineContainer>) {
      affine_points->resize(std::size(jacobian_points));
    } else {
      if (std::size(jacobian_points) != std::size(*affine_points)) {
        return absl::InvalidArgumentError(absl::Substitute(
            "size do not match $0 vs $1", std::size(jacobian_points),
            std::size(*affine_points)));
      }
    }
    std::vector<BaseField> z_inverses;
    z_inverses.reserve(std::size(jacobian_points));
    for (const JacobianPoint& point : jacobian_points) {
      z_inverses.push_back(point.z());
    }
    absl::Status status = BatchInverse(z_inverses, &z_inverses);
    if (!status.ok()) return status;
    for (size_t i = 0; i < std::size(*affine_points); ++i) {
      const BaseField& z_inv = z_inverses[i];
      if (z_inv.IsZero()) {
        (*affine_points)[i] = AffinePoint::Zero();
      } else if (z_inv.IsOne()) {
        (*affine_points)[i] = {jacobian_points[i].x(), jacobian_points[i].y()};
      } else {
        BaseField z_inv_square = z_inv.Square();
        (*affine_points)[i] = {jacobian_points[i].x() * z_inv_square,
                               jacobian_points[i].y() * z_inv_square * z_inv};
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
