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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_POINT_BASE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_POINT_BASE_H_

#include <array>
#include <string>
#include <utility>

#include "zk_dtypes/include/control_flow_operation.h"
#include "zk_dtypes/include/geometry/point_traits.h"
#include "zk_dtypes/include/str_join.h"

namespace zk_dtypes {

// CRTP base for twisted Edwards point types. Mirrors
// short_weierstrass/point_base.h but exposes only the AffinePoint /
// ExtendedPoint hooks because the twisted Edwards model has no Jacobian or
// XYZZ representations.
template <typename Derived>
class TePointBase {
 public:
  constexpr static size_t N = PointTraits<Derived>::kNumCoords;

  using Curve = typename PointTraits<Derived>::Curve;
  using AffinePoint = typename PointTraits<Derived>::AffinePoint;
  using ExtendedPoint = typename PointTraits<Derived>::ExtendedPoint;
  using BaseField = typename Curve::BaseField;

  constexpr TePointBase() = default;
  constexpr TePointBase(const std::array<BaseField, N>& coords)
      : coords_(coords) {}

  constexpr const BaseField& operator[](size_t i) const { return coords_[i]; }

  std::string ToString() const {
    return StrJoin(
        coords_,
        [](std::ostream& os, const BaseField& value) {
          os << value.ToString();
        },
        ", ", "(", ")");
  }
  std::string ToHexString(bool pad_zero = false) const {
    return StrJoin(
        coords_,
        [pad_zero](std::ostream& os, const BaseField& value) {
          os << value.ToHexString(pad_zero);
        },
        ", ", "(", ")");
  }

  constexpr Derived FromCoords(const std::array<BaseField, N>& coords) const {
    return Derived(coords);
  }
  constexpr const std::array<BaseField, N>& ToCoords() const { return coords_; }

  // Curve constants `a` and `d` for the twisted Edwards equation
  // a * x² + y² = 1 + d * x² * y², plus the generator coordinates Gx/Gy.
  constexpr const BaseField& GetA() const { return Curve::Config::kA; }
  constexpr const BaseField& GetD() const { return Curve::Config::kD; }
  constexpr const BaseField& GetX() const { return Curve::Config::kX; }
  constexpr const BaseField& GetY() const { return Curve::Config::kY; }
  constexpr ControlFlowOperation<bool> GetCFOperation() const { return {}; }

  constexpr AffinePoint CreateAffinePoint(
      const std::array<BaseField, 2>& coords) const {
    return AffinePoint(coords);
  }
  constexpr ExtendedPoint CreateExtendedPoint(
      const std::array<BaseField, 4>& coords) const {
    return ExtendedPoint(coords);
  }
  constexpr AffinePoint&& MaybeConvertToAffine(AffinePoint&& point) const {
    return std::move(point);
  }
  constexpr ExtendedPoint&& MaybeConvertToExtended(
      ExtendedPoint&& point) const {
    return std::move(point);
  }

 protected:
  std::array<BaseField, N> coords_;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_TWISTED_EDWARDS_POINT_BASE_H_
