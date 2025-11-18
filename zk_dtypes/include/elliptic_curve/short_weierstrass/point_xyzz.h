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

#include <string>
#include <type_traits>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"

#include "zk_dtypes/include/batch_inverse.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_declarations.h"
#include "zk_dtypes/include/scalar_mul.h"
#include "zk_dtypes/include/template_util.h"

namespace zk_dtypes {

template <typename _Curve>
class PointXyzz<_Curve,
                std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;
  using StdType = PointXyzz<SwCurve<typename Curve::Config::StdConfig>>;

  using AffinePoint = zk_dtypes::AffinePoint<Curve>;
  using JacobianPoint = zk_dtypes::JacobianPoint<Curve>;

  constexpr static bool kUseMontgomery = Curve::kUseMontgomery;
  constexpr static size_t kBitWidth = BaseField::kBitWidth * 4;

  constexpr PointXyzz()
      : PointXyzz(BaseField::One(), BaseField::One(), BaseField::Zero(),
                  BaseField::Zero()) {}
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr PointXyzz(T value) : PointXyzz(ScalarField(value)) {}
  constexpr PointXyzz(ScalarField value) {
    PointXyzz point = PointXyzz::Generator() * value;
    x_ = point.x_;
    y_ = point.y_;
    zz_ = point.zz_;
    zzz_ = point.zzz_;
  }
  constexpr PointXyzz(const BaseField& x, const BaseField& y,
                      const BaseField& zz, const BaseField& zzz)
      : x_(x), y_(y), zz_(zz), zzz_(zzz) {}

  constexpr static PointXyzz Zero() { return PointXyzz(); }

  constexpr static PointXyzz One() { return Generator(); }

  constexpr static PointXyzz Generator() {
    return {Curve::Config::kX, Curve::Config::kY, BaseField::One(),
            BaseField::One()};
  }

  constexpr static PointXyzz Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr const BaseField& zz() const { return zz_; }
  constexpr const BaseField& zzz() const { return zzz_; }

  constexpr bool IsZero() const { return zz_.IsZero(); }
  constexpr bool IsOne() const {
    return x_ == Curve::Config::kX && y_ == Curve::Config::kY && zz_.IsOne() &&
           zzz_.IsOne();
  }

  constexpr bool operator==(const PointXyzz& other) const {
    if (IsZero()) {
      return other.IsZero();
    }

    if (other.IsZero()) {
      return false;
    }

    // The points (X, Y, ZZ, ZZZ) and (X', Y', ZZ', ZZZ')
    // are equal when (X * ZZ') = (X' * ZZ)
    // and (Y * Z'³) = (Y' * Z³).
    if (x_ * other.zz_ != other.x_ * zz_) {
      return false;
    } else {
      return y_ * other.zzz_ == other.y_ * zzz_;
    }
  }

  constexpr bool operator!=(const PointXyzz& other) const {
    return !operator==(other);
  }

  constexpr PointXyzz operator+(const PointXyzz& other) const {
    if (IsZero()) {
      return other;
    }

    if (other.IsZero()) {
      return *this;
    }

    PointXyzz ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr PointXyzz& operator+=(const PointXyzz& other) {
    if (IsZero()) {
      return *this = other;
    }

    if (other.IsZero()) {
      return *this;
    }

    Add(*this, other, *this);
    return *this;
  }

  constexpr PointXyzz operator+(const AffinePoint& other) const {
    if (IsZero()) {
      return other.ToXyzz();
    }

    if (other.IsZero()) {
      return *this;
    }

    PointXyzz ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr PointXyzz& operator+=(const AffinePoint& other) {
    if (IsZero()) {
      return *this = other.ToXyzz();
    }

    if (other.IsZero()) {
      return *this;
    }

    Add(*this, other, *this);
    return *this;
  }

  constexpr PointXyzz Double() const;

  constexpr PointXyzz operator-(const PointXyzz& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz& operator-=(const PointXyzz& other) {
    return *this = operator-(other);
  }

  constexpr PointXyzz operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz& operator-=(const AffinePoint& other) {
    return *this = operator-(other);
  }

  constexpr PointXyzz operator-() const { return {x_, -y_, zz_, zzz_}; }

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

  // The xyzz point X, Y, ZZ, ZZZ is represented in the affine
  // coordinates as X/ZZ, Y/ZZZ.
  constexpr absl::StatusOr<AffinePoint> ToAffine() const {
    if (IsZero()) {
      return AffinePoint::Zero();
    } else if (zz_.IsOne()) {
      return AffinePoint(x_, y_);
    } else {
      absl::StatusOr<BaseField> z_inv_cubic = zzz_.Inverse();
      if (!z_inv_cubic.ok()) return z_inv_cubic.status();
      BaseField z_inv_square = ((*z_inv_cubic) * zz_).Square();
      return AffinePoint(x_ * z_inv_square, y_ * (*z_inv_cubic));
    }
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the jacobian
  // coordinates as X, Y, ZZZ/ZZ.
  constexpr JacobianPoint ToJacobian() const {
    if (IsZero()) {
      return JacobianPoint::Zero();
    } else if (zz_.IsOne()) {
      return {x_, y_, BaseField::One()};
    } else {
      BaseField z = (*zz_.Inverse()) * zzz_;
      return {x_, y_, z};
    }
  }

  template <typename Curve2 = Curve,
            std::enable_if_t<Curve2::kUseMontgomery>* = nullptr>
  constexpr StdType MontReduce() const {
    return {x_.MontReduce(), y_.MontReduce(), zz_.MontReduce(),
            zzz_.MontReduce()};
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
      zzz_inverses.push_back(point.zzz_);
    }
    absl::Status status = BatchInverse(zzz_inverses, &zzz_inverses);
    if (!status.ok()) return status;
    for (size_t i = 0; i < std::size(*affine_points); ++i) {
      const PointXyzz& point_xyzz = point_xyzzs[i];
      if (point_xyzz.zz_.IsZero()) {
        (*affine_points)[i] = AffinePoint::Zero();
      } else if (point_xyzz.zz_.IsOne()) {
        (*affine_points)[i] = {point_xyzz.x_, point_xyzz.y_};
      } else {
        const BaseField& z_inv_cubic = zzz_inverses[i];
        BaseField z_inv_square = (z_inv_cubic * point_xyzz.zz_).Square();
        (*affine_points)[i] = {point_xyzz.x_ * z_inv_square,
                               point_xyzz.y_ * z_inv_cubic};
      }
    }
    return absl::OkStatus();
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToString(), y_.ToString(),
                            zz_.ToString(), zzz_.ToString());
  }
  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero), zz_.ToHexString(pad_zero),
                            zzz_.ToHexString(pad_zero));
  }

 private:
  constexpr static void Add(const PointXyzz& a, const PointXyzz& b,
                            PointXyzz& c);
  constexpr static void Add(const PointXyzz& a, const AffinePoint& b,
                            PointXyzz& c);

  BaseField x_;
  BaseField y_;
  BaseField zz_;
  BaseField zzz_;
};

}  // namespace zk_dtypes

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz_impl.h"

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_POINT_XYZZ_H_
