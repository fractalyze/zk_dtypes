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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_AFFINE_POINT_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_AFFINE_POINT_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/elliptic_curve/twisted_edwards/affine_point_operation.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/point_base.h"
#include "zk_dtypes/include/elliptic_curve/twisted_edwards/te_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"

namespace zk_dtypes {

template <typename _Curve>
class AffinePoint<_Curve,
                  std::enable_if_t<_Curve::kType == CurveType::kTwistedEdwards>>
    final : public TePointBase<AffinePoint<_Curve>>,
            public AffinePointOperation<AffinePoint<_Curve>> {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType =
      AffinePoint<TwistedEdwardsCurve<typename Curve::Config::StdConfig>>;

  using ExtendedPoint = zk_dtypes::ExtendedPoint<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kByteWidth = BaseField::kByteWidth * 2;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 2;

  // Identity element for twisted Edwards: (0, 1).
  constexpr AffinePoint() : AffinePoint(BaseField::Zero(), BaseField::One()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr AffinePoint(T value) : AffinePoint(ScalarField(value)) {}
  constexpr AffinePoint(ScalarField value) {
    AffinePoint point =
        (AffinePoint::Generator().ToExtended() * value).ToAffine();
    this->coords_ = point.coords_;
  }
  constexpr AffinePoint(const std::array<BaseField, 2>& coords)
      : TePointBase<AffinePoint<_Curve>>(coords) {}
  constexpr AffinePoint(const BaseField& x, const BaseField& y)
      : TePointBase<AffinePoint<_Curve>>({x, y}) {}

  // Identity element for twisted Edwards: (0, 1).
  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint One() { return Generator(); }

  constexpr static AffinePoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY};
  }

  constexpr static AffinePoint Random() {
    return ExtendedPoint::Random().ToAffine();
  }

  constexpr const BaseField& x() const { return this->coords_[0]; }
  constexpr const BaseField& y() const { return this->coords_[1]; }

  constexpr ExtendedPoint operator*(const ScalarField& v) const {
    if constexpr (kUseMontgomery) {
      return ScalarMul(this->ToExtended(), v.MontReduce().value());
    } else {
      return ScalarMul(this->ToExtended(), v.value());
    }
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x().MontReduce(), y().MontReduce()};
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_AFFINE_POINT_H_
