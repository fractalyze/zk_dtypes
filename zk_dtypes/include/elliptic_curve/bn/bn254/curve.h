#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_CURVE_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"

namespace zk_dtypes {

class CurveStd {
 public:
  using G1Curve = bn254::G1CurveStd;
  using G2Curve = bn254::G2CurveStd;

  using StdConfig = CurveStd;
};

class Curve {
 public:
  using G1Curve = bn254::G1Curve;
  using G2Curve = bn254::G2Curve;

  using StdCurve = CurveStd;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_CURVE_H_
