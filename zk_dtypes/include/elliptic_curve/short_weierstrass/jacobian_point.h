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

#include <string>
#include <type_traits>

#include "absl/strings/substitute.h"

#include "zk_dtypes/include/batch_inverse.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"
#include "zk_dtypes/include/template_util.h"

namespace zk_dtypes {

template <typename _Curve>
class JacobianPoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType = JacobianPoint<SwCurve<typename Curve::Config::StdConfig>>;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using PointXyzz = zk_dtypes::PointXyzz<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 3;

  constexpr JacobianPoint()
      : JacobianPoint(BaseField::One(), BaseField::One(), BaseField::Zero()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr JacobianPoint(T value) : JacobianPoint(ScalarField(value)) {}
  constexpr JacobianPoint(ScalarField value) {
    JacobianPoint point = JacobianPoint::Generator() * value;
    x_ = point.x_;
    y_ = point.y_;
    z_ = point.z_;
  }
  constexpr JacobianPoint(const BaseField& x, const BaseField& y,
                          const BaseField& z)
      : x_(x), y_(y), z_(z) {}

  constexpr static JacobianPoint Zero() { return JacobianPoint(); }

  constexpr static JacobianPoint One() { return Generator(); }

  constexpr static JacobianPoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY, BaseField::One()};
  }

  constexpr static JacobianPoint Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr const BaseField& z() const { return z_; }

  constexpr bool IsZero() const { return z_.IsZero(); }
  constexpr bool IsOne() const {
    return x_ == Curve::Config::kX && y_ == Curve::Config::kY && z_.IsOne();
  }

  constexpr bool operator==(const JacobianPoint& other) const {
    if (IsZero()) {
      return other.IsZero();
    }

    if (other.IsZero()) {
      return false;
    }

    // The points (X, Y, Z) and (X', Y', Z')
    // are equal when (X * Z'²) = (X' * Z²)
    // and (Y * Z'³) = (Y' * Z³).
    const BaseField z1z1 = z_.Square();
    const BaseField z2z2 = other.z_.Square();

    if (x_ * z2z2 != other.x_ * z1z1) {
      return false;
    } else {
      return y_ * (z2z2 * other.z_) == other.y_ * (z1z1 * z_);
    }
  }

  constexpr bool operator!=(const JacobianPoint& other) const {
    return !operator==(other);
  }

  constexpr JacobianPoint operator+(const JacobianPoint& other) const {
    if (IsZero()) {
      return other;
    }

    if (other.IsZero()) {
      return *this;
    }

    JacobianPoint ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr JacobianPoint& operator+=(const JacobianPoint& other) {
    if (IsZero()) {
      return *this = other;
    }

    if (other.IsZero()) {
      return *this;
    }

    Add(*this, other, *this);
    return *this;
  }

  constexpr JacobianPoint operator+(const AffinePoint& other) const {
    if (IsZero()) {
      return other.ToJacobian();
    }

    if (other.IsZero()) {
      return *this;
    }

    JacobianPoint ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr JacobianPoint& operator+=(const AffinePoint& other) {
    if (IsZero()) {
      return *this = other.ToJacobian();
    }

    if (other.IsZero()) {
      return *this;
    }

    Add(*this, other, *this);
    return *this;
  }

  constexpr JacobianPoint Double() const;

  constexpr JacobianPoint operator-(const JacobianPoint& other) const {
    return operator+(-other);
  }

  constexpr JacobianPoint& operator-=(const JacobianPoint& other) {
    return *this = operator-(other);
  }

  constexpr JacobianPoint operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr JacobianPoint& operator-=(const AffinePoint& other) {
    return *this = operator-(other);
  }

  constexpr JacobianPoint operator-() const { return {x_, -y_, z_}; }

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

  // The jacobian point X, Y, Z is represented in the affine
  // coordinates as X/Z², Y/Z³.
  constexpr AffinePoint ToAffine() const {
    if (IsZero()) {
      return AffinePoint::Zero();
    } else if (z_.IsOne()) {
      return AffinePoint(x_, y_);
    } else {
      BaseField z_inv = z_.Inverse();
      BaseField z_inv_square = z_inv.Square();
      return AffinePoint(x_ * z_inv_square, y_ * z_inv_square * z_inv);
    }
  }

  // The jacobian point X, Y, Z is represented in the xyzz
  // coordinates as X, Y, Z², Z³.
  constexpr PointXyzz ToXyzz() const {
    BaseField zz = z_.Square();
    return {x_, y_, zz, zz * z_};
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x_.MontReduce(), y_.MontReduce(), z_.MontReduce()};
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
      z_inverses.push_back(point.z_);
    }
    absl::Status status = BatchInverse(z_inverses, &z_inverses);
    if (!status.ok()) return status;
    for (size_t i = 0; i < std::size(*affine_points); ++i) {
      const BaseField& z_inv = z_inverses[i];
      if (z_inv.IsZero()) {
        (*affine_points)[i] = AffinePoint::Zero();
      } else if (z_inv.IsOne()) {
        (*affine_points)[i] = {jacobian_points[i].x_, jacobian_points[i].y_};
      } else {
        BaseField z_inv_square = z_inv.Square();
        (*affine_points)[i] = {jacobian_points[i].x_ * z_inv_square,
                               jacobian_points[i].y_ * z_inv_square * z_inv};
      }
    }
    return absl::OkStatus();
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", x_.ToString(), y_.ToString(),
                            z_.ToString());
  }
  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero), z_.ToHexString(pad_zero));
  }

 private:
  constexpr static void Add(const JacobianPoint& a, const JacobianPoint& b,
                            JacobianPoint& c);
  constexpr static void Add(const JacobianPoint& a, const AffinePoint& b,
                            JacobianPoint& c);

  BaseField x_;
  BaseField y_;
  BaseField z_;
};

}  // namespace zk_dtypes

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point_impl.h"

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_H_
