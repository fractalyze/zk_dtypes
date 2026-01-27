#ifndef ZK_DTYPES_INCLUDE_ALL_TYPES_H_
#define ZK_DTYPES_INCLUDE_ALL_TYPES_H_

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/babybear/babybearx4.h"
#include "zk_dtypes/include/field/binary_field.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zk_dtypes/include/field/goldilocks/goldilocksx3.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"
#include "zk_dtypes/include/field/koalabear/koalabearx4.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31x2.h"

// clang-format off
#define WITH_MONT(V, ActualType, UpperCamelCaseName, UpperSnakeCaseName, LowerSnakeCaseName) \
V(ActualType, UpperCamelCaseName, UpperSnakeCaseName, LowerSnakeCaseName)                    \
V(ActualType##Std, UpperCamelCaseName##Std, UpperSnakeCaseName##_STD, LowerSnakeCaseName##_std)

//===----------------------------------------------------------------------===//
// BinaryField Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_BINARY_FIELD_TYPE_LIST(V)                             \
V(::zk_dtypes::BinaryFieldT0, BinaryFieldT0, BINARY_FIELD_T0, binary_field_t0) \
V(::zk_dtypes::BinaryFieldT1, BinaryFieldT1, BINARY_FIELD_T1, binary_field_t1) \
V(::zk_dtypes::BinaryFieldT2, BinaryFieldT2, BINARY_FIELD_T2, binary_field_t2) \
V(::zk_dtypes::BinaryFieldT3, BinaryFieldT3, BINARY_FIELD_T3, binary_field_t3) \
V(::zk_dtypes::BinaryFieldT4, BinaryFieldT4, BINARY_FIELD_T4, binary_field_t4) \
V(::zk_dtypes::BinaryFieldT5, BinaryFieldT5, BINARY_FIELD_T5, binary_field_t5) \
V(::zk_dtypes::BinaryFieldT6, BinaryFieldT6, BINARY_FIELD_T6, binary_field_t6) \
V(::zk_dtypes::BinaryFieldT7, BinaryFieldT7, BINARY_FIELD_T7, binary_field_t7)

#define ZK_DTYPES_ALL_BINARY_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_BINARY_FIELD_TYPE_LIST(V)      \

//===----------------------------------------------------------------------===//
// PrimeField Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(V)                         \
WITH_MONT(V, ::zk_dtypes::Babybear, Babybear, BABYBEAR, babybear)         \
V(::zk_dtypes::Mersenne31, Mersenne31, MERSENNE31, mersenne31)            \
WITH_MONT(V, ::zk_dtypes::Goldilocks, Goldilocks, GOLDILOCKS, goldilocks) \
WITH_MONT(V, ::zk_dtypes::Koalabear, Koalabear, KOALABEAR, koalabear)     \
WITH_MONT(V, ::zk_dtypes::bn254::Fr, Bn254Sf, BN254_SF, bn254_sf)

#define ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(V)      \
WITH_MONT(V, ::zk_dtypes::bn254::Fq, Bn254Bf, BN254_BF, bn254_bf)

//===----------------------------------------------------------------------===//
// ExtendedField Types
//===----------------------------------------------------------------------===//

// TODO(chokobole): Add Mersenne31X2X2.
#define ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(V)                               \
WITH_MONT(V, ::zk_dtypes::BabybearX4, BabybearX4, BABYBEARX4, babybearx4)     \
WITH_MONT(V, ::zk_dtypes::KoalabearX4, KoalabearX4, KOALABEARX4, koalabearx4) \
V(::zk_dtypes::Mersenne31X2, Mersenne31X2, MERSENNE31X2, mersenne31x2)        \
WITH_MONT(V, ::zk_dtypes::GoldilocksX3, GoldilocksX3, GOLDILOCKSX3, goldilocksx3)

#define ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(V)      \
WITH_MONT(V, ::zk_dtypes::bn254::FqX2, Bn254BfX2, BN254_BFX2, bn254_bfx2)

//===----------------------------------------------------------------------===//
// Field Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_FIELD_TYPE_LIST(V) \
ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(V)   \
ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(V)     \
ZK_DTYPES_PUBLIC_BINARY_FIELD_TYPE_LIST(V)

#define ZK_DTYPES_ALL_FIELD_TYPE_LIST(V) \
ZK_DTYPES_ALL_PRIME_FIELD_TYPE_LIST(V)   \
ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(V)     \
ZK_DTYPES_ALL_BINARY_FIELD_TYPE_LIST(V)

//===----------------------------------------------------------------------===//
// ScalarField Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_SCALAR_FIELD_TYPE_LIST(V) \
WITH_MONT(V, ::zk_dtypes::bn254::Fr, Bn254Sf, BN254_SF, bn254_sf)

//===----------------------------------------------------------------------===//
// AffinePoint Types
//===----------------------------------------------------------------------===//

#define ZK_DTYPES_PUBLIC_R1_AFFINE_POINT_TYPE_LIST(V) \
WITH_MONT(V, ::zk_dtypes::bn254::G1AffinePoint, Bn254G1Affine, BN254_G1_AFFINE, bn254_g1_affine)

#define ZK_DTYPES_PUBLIC_R2_AFFINE_POINT_TYPE_LIST(V) \
WITH_MONT(V, ::zk_dtypes::bn254::G2AffinePoint, Bn254G2Affine, BN254_G2_AFFINE, bn254_g2_affine)

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
WITH_MONT(V, ::zk_dtypes::bn254::G1JacobianPoint, Bn254G1Jacobian, BN254_G1_JACOBIAN, bn254_g1_jacobian)

#define ZK_DTYPES_PUBLIC_R2_JACOBIAN_POINT_TYPE_LIST(V) \
WITH_MONT(V, ::zk_dtypes::bn254::G2JacobianPoint, Bn254G2Jacobian, BN254_G2_JACOBIAN, bn254_g2_jacobian)

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
WITH_MONT(V, ::zk_dtypes::bn254::G1PointXyzz, Bn254G1Xyzz, BN254_G1_XYZZ, bn254_g1_xyzz)

#define ZK_DTYPES_PUBLIC_R2_XYZZ_POINT_TYPE_LIST(V) \
WITH_MONT(V, ::zk_dtypes::bn254::G2PointXyzz, Bn254G2Xyzz, BN254_G2_XYZZ, bn254_g2_xyzz)

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
