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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_PAIRING_TRAITS_FORWARD_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_PAIRING_TRAITS_FORWARD_H_

#include <type_traits>

namespace zk_dtypes {

// Forward declaration of PairingTraits. Specializations provide type aliases
// (Fp, Fp2, Fp12, G1AffinePoint, G2AffinePoint) that the pairing
// algorithm templates use. For code generation, a specialization maps these
// to IR builder types instead of concrete field types.
template <typename Derived>
struct PairingTraits;

// Type resolution helper for pairing templates.
// When Derived = void (default), resolves types from Config.
// When Derived is provided, resolves types from PairingTraits<Derived>.
template <typename Config, typename Derived, typename = void>
struct PairingTypes {
  using G1Curve = typename Config::G1Curve;
  using G2Curve = typename Config::G2Curve;
  using Fp2 = typename G2Curve::BaseField;
  using Fp = typename Fp2::BaseField;
  using Fp12 = typename Config::Fp12;
  using G1AffinePoint = typename G1Curve::AffinePoint;
  using G2AffinePoint = typename G2Curve::AffinePoint;
  using BoolType = bool;
};

template <typename Config, typename Derived>
struct PairingTypes<Config, Derived,
                    std::enable_if_t<!std::is_void_v<Derived>>> {
  using Fp2 = typename PairingTraits<Derived>::Fp2;
  using Fp = typename PairingTraits<Derived>::Fp;
  using Fp12 = typename PairingTraits<Derived>::Fp12;
  using G1AffinePoint = typename PairingTraits<Derived>::G1AffinePoint;
  using G2AffinePoint = typename PairingTraits<Derived>::G2AffinePoint;
  using BoolType = typename PairingTraits<Derived>::BoolType;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_PAIRING_TRAITS_FORWARD_H_
