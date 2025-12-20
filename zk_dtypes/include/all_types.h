#ifndef ZK_DTYPES_INCLUDE_ALL_TYPES_H_
#define ZK_DTYPES_INCLUDE_ALL_TYPES_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/babybear/babybear4.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks3.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"
#include "zk_dtypes/include/field/koalabear/koalabear4.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

// clang-format off
#define WITH_STD(V, ActualType, UpperCamelCaseName, UpperSnakeCaseName, LowerSnakeCaseName) \
V(ActualType, UpperCamelCaseName, UpperSnakeCaseName, LowerSnakeCaseName)                   \
V(ActualType##Std, UpperCamelCaseName##Std, UpperSnakeCaseName##_STD, LowerSnakeCaseName##_std)

//===----------------------------------------------------------------------===//
// PrimeField Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(V)                        \
WITH_STD(V, ::zk_dtypes::Babybear, Babybear, BABYBEAR, babybear)         \
WITH_STD(V, ::zk_dtypes::Mersenne31, Mersenne31, MERSENNE31, mersenne31) \
WITH_STD(V, ::zk_dtypes::Goldilocks, Goldilocks, GOLDILOCKS, goldilocks) \
WITH_STD(V, ::zk_dtypes::Koalabear, Koalabear, KOALABEAR, koalabear)     \
WITH_STD(V, ::zk_dtypes::bn254::Fr, Bn254Sf, BN254_SF, bn254_sf)

#define ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(V)      \
WITH_STD(V, ::zk_dtypes::bn254::Fq, Bn254Bf, BN254_BF, bn254_bf)

//===----------------------------------------------------------------------===//
// ExtendedField Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(V)                              \
WITH_STD(V, ::zk_dtypes::Babybear4, Babybear4, BABYBEAR4, babybear4)         \
WITH_STD(V, ::zk_dtypes::Koalabear4, Koalabear4, KOALABEAR4, koalabear4)     \
WITH_STD(V, ::zk_dtypes::Goldilocks3, Goldilocks3, GOLDILOCKS3, goldilocks3)

#define ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(V)      \
WITH_STD(V, ::zk_dtypes::bn254::Fq2, Bn254Bf2, BN254_BF2, bn254_bf2)

//===----------------------------------------------------------------------===//
// Field Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(V)   \
ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(V)

#define ZK_DTYPES_ALL_FIELD_TYPE_LIST(V) \
ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST(V)   \
ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(V)

//===----------------------------------------------------------------------===//
// ScalarField Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_SCALAR_FIELD_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::Fr, Bn254Sf, BN254_SF, bn254_sf)

//===----------------------------------------------------------------------===//
// AffinePoint Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_R1_AFFINE_POINT_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::G1AffinePoint, Bn254G1Affine, BN254_G1_AFFINE, bn254_g1_affine)

#define ZK_DTYPES_PUBLIC_R2_AFFINE_POINT_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::G2AffinePoint, Bn254G2Affine, BN254_G2_AFFINE, bn254_g2_affine)

#define ZK_DTYPES_PUBLIC_AFFINE_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_AFFINE_POINT_TYPE_LIST(V)      \
ZK_DTYPES_PUBLIC_R2_AFFINE_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R1_AFFINE_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_AFFINE_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R2_AFFINE_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R2_AFFINE_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_AFFINE_POINT_TYPE_LIST(V) \
ZK_DTYPES_ALL_R1_AFFINE_POINT_TYPE_LIST(V)      \
ZK_DTYPES_ALL_R2_AFFINE_POINT_TYPE_LIST(V)

//===----------------------------------------------------------------------===//
// JacobianPoint Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_R1_JACOBIAN_POINT_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::G1JacobianPoint, Bn254G1Jacobian, BN254_G1_JACOBIAN, bn254_g1_jacobian)

#define ZK_DTYPES_PUBLIC_R2_JACOBIAN_POINT_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::G2JacobianPoint, Bn254G2Jacobian, BN254_G2_JACOBIAN, bn254_g2_jacobian)

#define ZK_DTYPES_PUBLIC_JACOBIAN_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_JACOBIAN_POINT_TYPE_LIST(V)      \
ZK_DTYPES_PUBLIC_R2_JACOBIAN_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R1_JACOBIAN_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_JACOBIAN_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R2_JACOBIAN_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R2_JACOBIAN_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_JACOBIAN_POINT_TYPE_LIST(V) \
ZK_DTYPES_ALL_R1_JACOBIAN_POINT_TYPE_LIST(V)      \
ZK_DTYPES_ALL_R2_JACOBIAN_POINT_TYPE_LIST(V)

//===----------------------------------------------------------------------===//
// PointXyzz Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_R1_XYZZ_POINT_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::G1PointXyzz, Bn254G1Xyzz, BN254_G1_XYZZ, bn254_g1_xyzz)

#define ZK_DTYPES_PUBLIC_R2_XYZZ_POINT_TYPE_LIST(V) \
WITH_STD(V, ::zk_dtypes::bn254::G2PointXyzz, Bn254G2Xyzz, BN254_G2_XYZZ, bn254_g2_xyzz)

#define ZK_DTYPES_PUBLIC_XYZZ_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_XYZZ_POINT_TYPE_LIST(V)      \
ZK_DTYPES_PUBLIC_R2_XYZZ_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R1_XYZZ_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_XYZZ_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R2_XYZZ_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R2_XYZZ_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_XYZZ_POINT_TYPE_LIST(V) \
ZK_DTYPES_ALL_R1_XYZZ_POINT_TYPE_LIST(V)      \
ZK_DTYPES_ALL_R2_XYZZ_POINT_TYPE_LIST(V)

//===----------------------------------------------------------------------===//
// Elliptic Curve Point Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_R1_EC_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_AFFINE_POINT_TYPE_LIST(V)     \
ZK_DTYPES_PUBLIC_R1_JACOBIAN_POINT_TYPE_LIST(V)   \
ZK_DTYPES_PUBLIC_R1_XYZZ_POINT_TYPE_LIST(V)

#define ZK_DTYPES_PUBLIC_R2_EC_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R2_AFFINE_POINT_TYPE_LIST(V)     \
ZK_DTYPES_PUBLIC_R2_JACOBIAN_POINT_TYPE_LIST(V)   \
ZK_DTYPES_PUBLIC_R2_XYZZ_POINT_TYPE_LIST(V)

#define ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_EC_POINT_TYPE_LIST(V)      \
ZK_DTYPES_PUBLIC_R2_EC_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R1_EC_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R1_EC_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_R2_EC_POINT_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_R2_EC_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_EC_POINT_TYPE_LIST(V) \
ZK_DTYPES_ALL_R1_EC_POINT_TYPE_LIST(V)      \
ZK_DTYPES_ALL_R2_EC_POINT_TYPE_LIST(V)

//===----------------------------------------------------------------------===//
// All Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_FIELD_TYPE_LIST(V)   \
ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(V)

#define ZK_DTYPES_ALL_TYPE_LIST(V) \
ZK_DTYPES_ALL_FIELD_TYPE_LIST(V)   \
ZK_DTYPES_ALL_EC_POINT_TYPE_LIST(V)
// clang-format on

#endif  // ZK_DTYPES_INCLUDE_ALL_TYPES_H_
