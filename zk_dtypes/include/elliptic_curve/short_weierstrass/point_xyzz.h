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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_H_

#include <array>
#include <string>
#include <type_traits>

#include "absl/strings/substitute.h"

#include "zk_dtypes/include/batch_inverse.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_base.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz_operation.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"
#include "zk_dtypes/include/template_util.h"

namespace zk_dtypes {

template <typename _Curve>
class PointXyzz<_Curve,
                std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final : public PointBase<PointXyzz<_Curve>>,
            public PointXyzzOperation<PointXyzz<_Curve>> {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType = PointXyzz<SwCurve<typename Curve::Config::StdConfig>>;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using JacobianPoint = zk_dtypes::JacobianPoint<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kByteWidth = BaseField::kByteWidth * 4;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 4;

  constexpr PointXyzz()
      : PointXyzz(BaseField::One(), BaseField::One(), BaseField::Zero(),
                  BaseField::Zero()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr PointXyzz(T value) : PointXyzz(ScalarField(value)) {}
  constexpr PointXyzz(ScalarField value) {
    PointXyzz point = PointXyzz::Generator() * value;
    this->coords_ = point.coords_;
  }
  constexpr PointXyzz(const std::array<BaseField, 4>& coords)
      : PointBase<PointXyzz<_Curve>>(coords) {}
  constexpr PointXyzz(const BaseField& x, const BaseField& y,
                      const BaseField& zz, const BaseField& zzz)
      : PointBase<PointXyzz<_Curve>>({x, y, zz, zzz}) {}

  constexpr static PointXyzz Zero() { return PointXyzz(); }

  constexpr static PointXyzz One() { return Generator(); }

  constexpr static PointXyzz Generator() {
    return {Curve::Config::kX, Curve::Config::kY, BaseField::One(),
            BaseField::One()};
  }

  constexpr static PointXyzz Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return this->coords_[0]; }
  constexpr const BaseField& y() const { return this->coords_[1]; }
  constexpr const BaseField& zz() const { return this->coords_[2]; }
  constexpr const BaseField& zzz() const { return this->coords_[3]; }

  constexpr PointXyzz operator*(const ScalarField& v) const {
    if constexpr (kUseMontgomery) {
      return ScalarMul(*this, v.MontReduce().value());
    } else {
      return ScalarMul(*this, v.value());
    }
  }

  constexpr PointXyzz& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x().MontReduce(), y().MontReduce(), zz().MontReduce(),
            zzz().MontReduce()};
  }

  template <typename XyzzContainer, typename AffineContainer>
  static absl::Status BatchToAffine(const XyzzContainer& point_xyzzs,
                                    AffineContainer* affine_points) {
    if constexpr (internal::has_resize_v<AffineContainer>) {
      affine_points->resize(std::size(point_xyzzs));
    } else {
      if (std::size(point_xyzzs) != std::size(*affine_points)) {
        return absl::InvalidArgumentError(absl::Substitute(
            "size do not match $0 vs $1", std::size(point_xyzzs),
            std::size(*affine_points)));
      }
    }
    std::vector<BaseField> zzz_inverses;
    zzz_inverses.reserve(std::size(point_xyzzs));
    for (const PointXyzz& point : point_xyzzs) {
      zzz_inverses.push_back(point.zzz());
    }
    absl::Status status = BatchInverse(zzz_inverses, &zzz_inverses);
    if (!status.ok()) return status;
    for (size_t i = 0; i < std::size(*affine_points); ++i) {
      const PointXyzz& point_xyzz = point_xyzzs[i];
      if (point_xyzz.zz().IsZero()) {
        (*affine_points)[i] = AffinePoint::Zero();
      } else if (point_xyzz.zz().IsOne()) {
        (*affine_points)[i] = {point_xyzz.x(), point_xyzz.y()};
      } else {
        const BaseField& z_inv_cubic = zzz_inverses[i];
        BaseField z_inv_square = (z_inv_cubic * point_xyzz.zz()).Square();
        (*affine_points)[i] = {point_xyzz.x() * z_inv_square,
                               point_xyzz.y() * z_inv_cubic};
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_H_
