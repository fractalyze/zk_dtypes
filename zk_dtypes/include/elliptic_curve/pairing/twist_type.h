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

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_TWIST_TYPE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_TWIST_TYPE_H_

namespace zk_dtypes {

// clang-format off
// Twist type for pairing-friendly curves.
//
// A twist is an isomorphic curve E'(Fp²) that allows G2 points to be
// represented over a smaller field (Fp² instead of Fp¹²). The isomorphism
// ψ: E'(Fp²) → E(Fp¹²) maps points from the twist to the full curve.
//
// For BN/BLS curves with equation y² = x³ + b, the sextic twist is:
//   M-twist: y² = x³ + b/ξ   where ξ is a non-residue in Fp²
//   D-twist: y² = x³ + b·ξ
//
// The twist type affects:
//   1. How line coefficients are positioned in Fp12 (sparse structure)
//   2. Whether we use MulBy014 (M-twist) or MulBy034 (D-twist)
//
// BN254 typically uses D-twist, while BLS12-381 uses M-twist.
// clang-format on
enum class TwistType {
  kM,  // M-twist (multiplicative)
  kD,  // D-twist (divisive)
};

inline const char* TwistTypeToString(TwistType type) {
  switch (type) {
    case TwistType::kM:
      return "M";
    case TwistType::kD:
      return "D";
  }
  return "";
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_PAIRING_TWIST_TYPE_H_
