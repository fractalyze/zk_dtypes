#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G1_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G1_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"

namespace zk_dtypes::bn254 {

template <typename BaseField>
class G1SwCurveBaseConfig {
 public:
  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 3;
  constexpr static BaseField kX = 1;
  constexpr static BaseField kY = 2;
};

class G1SwCurveStdConfig : public G1SwCurveBaseConfig<FqStd> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = G1SwCurveStdConfig;

  using BaseField = FqStd;
  using ScalarField = FrStd;
};

class G1SwCurveConfig : public G1SwCurveBaseConfig<Fq> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = G1SwCurveStdConfig;

  using BaseField = Fq;
  using ScalarField = Fr;
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1CurveStd = SwCurve<G1SwCurveStdConfig>;
using G1AffinePoint = AffinePoint<G1Curve>;
using G1AffinePointStd = AffinePoint<G1CurveStd>;
using G1JacobianPoint = JacobianPoint<G1Curve>;
using G1JacobianPointStd = JacobianPoint<G1CurveStd>;
using G1PointXyzz = PointXyzz<G1Curve>;
using G1PointXyzzStd = PointXyzz<G1CurveStd>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_G1_H_
