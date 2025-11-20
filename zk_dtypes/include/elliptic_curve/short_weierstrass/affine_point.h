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

#include <string>
#include <type_traits>

#include "absl/log/log.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"

namespace zk_dtypes {

template <typename _Curve>
class AffinePoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final {
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
    AffinePoint point = *(AffinePoint::Generator() * value).ToAffine();
    x_ = point.x_;
    y_ = point.y_;
  }
  constexpr AffinePoint(const BaseField& x, const BaseField& y)
      : x_(x), y_(y) {}

  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint One() { return Generator(); }

  constexpr static AffinePoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY};
  }

  constexpr static absl::StatusOr<AffinePoint> CreateFromX(const BaseField& x) {
    return Curve::GetPointFromX(x);
  }

  constexpr static AffinePoint Random() {
    return *JacobianPoint::Random().ToAffine();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }

  constexpr bool IsZero() const { return x_.IsZero() && y_.IsZero(); }
  constexpr bool IsOne() const {
    return x_ == Curve::Config::kX && y_ == Curve::Config::kY;
  }

  constexpr bool operator==(const AffinePoint& other) const {
    return x_ == other.x_ && y_ == other.y_;
  }

  constexpr bool operator!=(const AffinePoint& other) const {
    return !operator==(other);
  }

  constexpr JacobianPoint operator+(const AffinePoint& other) const {
    return ToJacobian() + other;
  }
  constexpr JacobianPoint operator+(const JacobianPoint& other) const {
    return other + *this;
  }
  constexpr PointXyzz operator+(const PointXyzz& other) const {
    return other + *this;
  }
  constexpr AffinePoint& operator+=(const AffinePoint& other) {
    LOG(FATAL) << "Invalid call to operator+=; this exists only to allow "
                  "compilation with reduction. See in_process_communicator.cc "
                  "for details";
    return *this;
  }
  constexpr JacobianPoint operator-(const AffinePoint& other) const {
    return ToJacobian() - other;
  }
  constexpr JacobianPoint operator-(const JacobianPoint& other) const {
    return -(other - *this);
  }
  constexpr PointXyzz operator-(const PointXyzz& other) const {
    return -(other - *this);
  }
  constexpr AffinePoint operator-() const { return {x_, -y_}; }

  constexpr JacobianPoint operator*(const ScalarField& v) const {
    if constexpr (kUseMontgomery) {
      return ScalarMul(ToJacobian(), v.MontReduce().value());
    } else {
      return ScalarMul(ToJacobian(), v.value());
    }
  }

  constexpr JacobianPoint ToJacobian() const {
    if (IsZero()) return JacobianPoint::Zero();
    return {x_, y_, BaseField::One()};
  }

  constexpr PointXyzz ToXyzz() const {
    if (IsZero()) return PointXyzz::Zero();
    return {x_, y_, BaseField::One(), BaseField::One()};
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x_.MontReduce(), y_.MontReduce()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }
  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero));
  }

 private:
  BaseField x_;
  BaseField y_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_H_
