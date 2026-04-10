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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_EXTENDED_POINT_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_EXTENDED_POINT_H_

#include <array>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/batch_inverse.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/extended_point_operation.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/point_base.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/te_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"
#include "zk_dtypes/include/template_util.h"

namespace zk_dtypes {

template <typename _Curve>
class ExtendedPoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kTwistedEdwards>>
    final : public TePointBase<ExtendedPoint<_Curve>>,
            public ExtendedPointOperation<ExtendedPoint<_Curve>> {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType =
      ExtendedPoint<TwistedEdwardsCurve<typename Curve::Config::StdConfig>>;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kByteWidth = BaseField::kByteWidth * 4;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 4;

  // Identity element for twisted Edwards in extended coords: (0, 1, 1, 0).
  constexpr ExtendedPoint()
      : ExtendedPoint(BaseField::Zero(), BaseField::One(), BaseField::One(),
                      BaseField::Zero()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr ExtendedPoint(T value) : ExtendedPoint(ScalarField(value)) {}
  constexpr ExtendedPoint(ScalarField value) {
    ExtendedPoint point = ExtendedPoint::Generator() * value;
    this->coords_ = point.coords_;
  }
  constexpr ExtendedPoint(const std::array<BaseField, 4>& coords)
      : TePointBase<ExtendedPoint<_Curve>>(coords) {}
  constexpr ExtendedPoint(const BaseField& x, const BaseField& y,
                          const BaseField& z, const BaseField& t)
      : TePointBase<ExtendedPoint<_Curve>>({x, y, z, t}) {}

  // Identity: (0, 1, 1, 0).
  constexpr static ExtendedPoint Zero() { return ExtendedPoint(); }

  constexpr static ExtendedPoint One() { return Generator(); }

  constexpr static ExtendedPoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY, BaseField::One(),
            Curve::Config::kX * Curve::Config::kY};
  }

  constexpr static ExtendedPoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return this->coords_[0]; }
  constexpr const BaseField& y() const { return this->coords_[1]; }
  constexpr const BaseField& z() const { return this->coords_[2]; }
  constexpr const BaseField& t() const { return this->coords_[3]; }

  constexpr ExtendedPoint operator*(const ScalarField& v) const {
    if constexpr (kUseMontgomery) {
      return ScalarMul(*this, v.MontReduce().value());
    } else {
      return ScalarMul(*this, v.value());
    }
  }

  constexpr ExtendedPoint& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x().MontReduce(), y().MontReduce(), z().MontReduce(),
            t().MontReduce()};
  }

  template <typename ExtendedContainer, typename AffineContainer>
  static absl::Status BatchToAffine(const ExtendedContainer& extended_points,
                                    AffineContainer* affine_points) {
    if constexpr (internal::has_resize_v<AffineContainer>) {
      affine_points->resize(std::size(extended_points));
    } else {
      if (std::size(extended_points) != std::size(*affine_points)) {
        return absl::InvalidArgumentError(absl::Substitute(
            "sizes do not match $0 vs $1", std::size(extended_points),
            std::size(*affine_points)));
      }
    }
    std::vector<BaseField> z_inverses;
    z_inverses.reserve(std::size(extended_points));
    for (const ExtendedPoint& point : extended_points) {
      z_inverses.push_back(point.z());
    }
    absl::Status status = BatchInverse(z_inverses, &z_inverses);
    if (!status.ok()) return status;
    for (size_t i = 0; i < std::size(*affine_points); ++i) {
      const BaseField& z_inv = z_inverses[i];
      (*affine_points)[i] = {extended_points[i].x() * z_inv,
                             extended_points[i].y() * z_inv};
    }
    return absl::OkStatus();
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_EXTENDED_POINT_H_
