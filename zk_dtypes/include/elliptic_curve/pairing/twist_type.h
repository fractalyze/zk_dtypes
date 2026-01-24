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

// Twist type for pairing-friendly curves.
// - M-twist: y² = x³ + b/ξ (where ξ is a non-residue)
// - D-twist: y² = x³ + b·ξ
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
