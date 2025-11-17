#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/small_prime_field.h"

namespace zk_dtypes::test {

struct PrimeFieldBaseConfig {
 public:
  constexpr static size_t kModulusBits = 4;
  constexpr static uint8_t kModulus = 7;

  constexpr static uint8_t kRSquared = UINT64_C(2);
  constexpr static uint64_t kNPrime = UINT64_C(183);

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static uint8_t kTrace = 3;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct PrimeFieldStdConfig : public PrimeFieldBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = PrimeFieldStdConfig;

  constexpr static uint8_t kOne = 1;

  constexpr static uint8_t kTwoAdicRootOfUnity = 6;
};

struct PrimeFieldConfig : public PrimeFieldBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = PrimeFieldStdConfig;

  constexpr static uint8_t kOne = 4;

  constexpr static uint8_t kTwoAdicRootOfUnity = 3;
};

using Fq = PrimeField<PrimeFieldConfig>;
using FqStd = PrimeField<PrimeFieldStdConfig>;
using Fr = PrimeField<PrimeFieldConfig>;
using FrStd = PrimeField<PrimeFieldStdConfig>;

struct Fq2BaseConfig {
  using BaseField = Fq;
  using BasePrimeField = Fq;

  constexpr static uint32_t kDegreeOverBaseField = 2;
  constexpr static BaseField kNonResidue = -1;
};

struct Fq2StdConfig : public Fq2BaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Fq2StdConfig;
};

struct Fq2Config : public Fq2BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Fq2StdConfig;
};

using Fq2 = ExtensionField<Fq2Config>;
using Fq2Std = ExtensionField<Fq2StdConfig>;

template <typename _BaseField, typename _ScalarField>
class SwCurveBaseConfig {
 public:
  using BaseField = _BaseField;
  using ScalarField = _ScalarField;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 5;
  constexpr static BaseField kX = 5;
  constexpr static BaseField kY = 5;
};

class SwCurveStdConfig : public SwCurveBaseConfig<FqStd, FrStd> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = SwCurveStdConfig;
};

class SwCurveConfig : public SwCurveBaseConfig<Fq, Fr> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = SwCurveStdConfig;
};

using G1Curve = SwCurve<SwCurveConfig>;
using G1CurveStd = SwCurve<SwCurveStdConfig>;
using AffinePoint = zk_dtypes::AffinePoint<G1Curve>;
using AffinePointStd = zk_dtypes::AffinePoint<G1CurveStd>;
using JacobianPoint = zk_dtypes::JacobianPoint<G1Curve>;
using JacobianPointStd = zk_dtypes::JacobianPoint<G1CurveStd>;
using PointXyzz = zk_dtypes::PointXyzz<G1Curve>;
using PointXyzzStd = zk_dtypes::PointXyzz<G1CurveStd>;

}  // namespace zk_dtypes::test

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
