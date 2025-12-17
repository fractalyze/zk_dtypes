/* Copyright 2017 The ml_dtypes Authors.
Copyright 2025 The zk_dtypes Authors.

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

// Enable cmath defines on Windows
#define _USE_MATH_DEFINES

#include <cstdint>

// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "zk_dtypes/_src/ec_point_numpy.h"
#include "zk_dtypes/_src/field_numpy.h"
#include "zk_dtypes/_src/intn_numpy.h"
#include "zk_dtypes/include/all_types.h"
#include "zk_dtypes/include/intn.h"

namespace zk_dtypes {

template <>
struct TypeDescriptor<int2> : IntNTypeDescriptor<int2> {
  typedef int2 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int2";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.int2";
  static constexpr const char* kTpDoc = "int2 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'c';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<uint2> : IntNTypeDescriptor<uint2> {
  typedef uint2 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "uint2";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.uint2";
  static constexpr const char* kTpDoc = "uint2 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'C';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<int4> : IntNTypeDescriptor<int4> {
  typedef int4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "int4";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.int4";
  static constexpr const char* kTpDoc = "int4 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'a';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<uint4> : IntNTypeDescriptor<uint4> {
  typedef uint4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = true;
  static constexpr const char* kTypeName = "uint4";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.uint4";
  static constexpr const char* kTpDoc = "uint4 integer values";
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'A';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Babybear> : FieldTypeDescriptor<Babybear> {
  typedef Babybear T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "babybear";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.babybear";
  static constexpr const char* kTpDoc =
      "babybear field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'b';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<BabybearStd> : FieldTypeDescriptor<BabybearStd> {
  typedef BabybearStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "babybear_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.babybear_std";
  static constexpr const char* kTpDoc =
      "babybear field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'B';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Goldilocks> : FieldTypeDescriptor<Goldilocks> {
  typedef Goldilocks T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "goldilocks";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.goldilocks";
  static constexpr const char* kTpDoc =
      "goldilocks field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'g';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<GoldilocksStd> : FieldTypeDescriptor<GoldilocksStd> {
  typedef GoldilocksStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "goldilocks_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.goldilocks_std";
  static constexpr const char* kTpDoc =
      "goldilocks field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'G';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Koalabear> : FieldTypeDescriptor<Koalabear> {
  typedef Koalabear T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "koalabear";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.koalabear";
  static constexpr const char* kTpDoc =
      "koalabear field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'k';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<KoalabearStd> : FieldTypeDescriptor<KoalabearStd> {
  typedef KoalabearStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "koalabear_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.koalabear_std";
  static constexpr const char* kTpDoc =
      "koalabear field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'K';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Mersenne31> : FieldTypeDescriptor<Mersenne31> {
  typedef Mersenne31 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "mersenne31";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.mersenne31";
  static constexpr const char* kTpDoc =
      "mersenne31 field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'm';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Mersenne31Std> : FieldTypeDescriptor<Mersenne31Std> {
  typedef Mersenne31Std T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "mersenne31_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.mersenne31_std";
  static constexpr const char* kTpDoc =
      "mersenne31 field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'M';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::Fr> : FieldTypeDescriptor<bn254::Fr> {
  typedef bn254::Fr T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_sf";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.bn254_sf";
  static constexpr const char* kTpDoc =
      "bn254 scalar field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'b';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::FrStd> : FieldTypeDescriptor<bn254::FrStd> {
  typedef bn254::FrStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_sf_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.bn254_sf_std";
  static constexpr const char* kTpDoc =
      "bn254 scalar field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'B';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Babybear4> : FieldTypeDescriptor<Babybear4> {
  typedef Babybear4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "babybear4";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.babybear4";
  static constexpr const char* kTpDoc =
      "babybear quartic extension field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'd';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Babybear4Std> : FieldTypeDescriptor<Babybear4Std> {
  typedef Babybear4Std T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "babybear4_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.babybear4_std";
  static constexpr const char* kTpDoc =
      "babybear quartic extension field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'D';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Koalabear4> : FieldTypeDescriptor<Koalabear4> {
  typedef Koalabear4 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "koalabear4";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.koalabear4";
  static constexpr const char* kTpDoc =
      "koalabear quartic extension field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'e';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Koalabear4Std> : FieldTypeDescriptor<Koalabear4Std> {
  typedef Koalabear4Std T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "koalabear4_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.koalabear4_std";
  static constexpr const char* kTpDoc =
      "koalabear quartic extension field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'E';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Goldilocks3> : FieldTypeDescriptor<Goldilocks3> {
  typedef Goldilocks3 T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "goldilocks3";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.goldilocks3";
  static constexpr const char* kTpDoc =
      "goldilocks cubic extension field values on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 't';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<Goldilocks3Std> : FieldTypeDescriptor<Goldilocks3Std> {
  typedef Goldilocks3Std T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "goldilocks3_std";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.goldilocks3_std";
  static constexpr const char* kTpDoc =
      "goldilocks cubic extension field values on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'T';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G1AffinePoint>
    : EcPointTypeDescriptor<bn254::G1AffinePoint> {
  typedef bn254::G1AffinePoint T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g1_affine";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.bn254_g1_affine";
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve affine point on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'a';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G1AffinePointStd>
    : EcPointTypeDescriptor<bn254::G1AffinePointStd> {
  typedef bn254::G1AffinePointStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g1_affine_std";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g1_affine_std";
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve affine point on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'A';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G1JacobianPoint>
    : EcPointTypeDescriptor<bn254::G1JacobianPoint> {
  typedef bn254::G1JacobianPoint T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g1_jacobian";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g1_jacobian";
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve jacobian point on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'j';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G1JacobianPointStd>
    : EcPointTypeDescriptor<bn254::G1JacobianPointStd> {
  typedef bn254::G1JacobianPointStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g1_jacobian_std";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g1_jacobian_std";
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve jacobian point on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'J';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G1PointXyzz>
    : EcPointTypeDescriptor<bn254::G1PointXyzz> {
  typedef bn254::G1PointXyzz T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g1_xyzz";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.bn254_g1_xyzz";
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve xyzz point on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'x';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G1PointXyzzStd>
    : EcPointTypeDescriptor<bn254::G1PointXyzzStd> {
  typedef bn254::G1PointXyzzStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g1_xyzz_std";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g1_xyzz_std";
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve xyzz point on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'X';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G2AffinePoint>
    : EcPointTypeDescriptor<bn254::G2AffinePoint> {
  typedef bn254::G2AffinePoint T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g2_affine";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.bn254_g2_affine";
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve affine point on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'a';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G2AffinePointStd>
    : EcPointTypeDescriptor<bn254::G2AffinePointStd> {
  typedef bn254::G2AffinePointStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g2_affine_std";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g2_affine_std";
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve affine point on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'A';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G2JacobianPoint>
    : EcPointTypeDescriptor<bn254::G2JacobianPoint> {
  typedef bn254::G2JacobianPoint T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g2_jacobian";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g2_jacobian";
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve jacobian point on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'j';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G2JacobianPointStd>
    : EcPointTypeDescriptor<bn254::G2JacobianPointStd> {
  typedef bn254::G2JacobianPointStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g2_jacobian_std";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g2_jacobian_std";
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve jacobian point on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'J';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G2PointXyzz>
    : EcPointTypeDescriptor<bn254::G2PointXyzz> {
  typedef bn254::G2PointXyzz T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g2_xyzz";
  static constexpr const char* kQualifiedTypeName = "zk_dtypes.bn254_g2_xyzz";
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve xyzz point on montgomery domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'x';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<bn254::G2PointXyzzStd>
    : EcPointTypeDescriptor<bn254::G2PointXyzzStd> {
  typedef bn254::G2PointXyzzStd T;
  static constexpr bool is_floating = false;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "bn254_g2_xyzz_std";
  static constexpr const char* kQualifiedTypeName =
      "zk_dtypes.bn254_g2_xyzz_std";
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve xyzz point on standard domain";
  static constexpr char kNpyDescrKind = 'V';
  static constexpr char kNpyDescrType = 'X';
  static constexpr char kNpyDescrByteorder = '=';
};

namespace {

// Performs a NumPy array cast from type 'From' to 'To' via `Via`.
template <typename From, typename To, typename Via>
void PyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
            void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(static_cast<Via>(from[i]));
  }
}

template <typename Type1, typename Type2, typename Via>
bool RegisterOneWayCustomCast() {
  int nptype1 = TypeDescriptor<Type1>::npy_type;
  int nptype2 = TypeDescriptor<Type2>::npy_type;
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, PyCast<Type1, Type2, Via>) <
      0) {
    return false;
  }
  return true;
}

// Initialize type attribute in the module object.
template <typename T>
bool InitModuleType(PyObject* obj, const char* name) {
  return PyObject_SetAttrString(
             obj, name,
             reinterpret_cast<PyObject*>(TypeDescriptor<T>::type_ptr)) >= 0;
}

template <typename... Types>
bool RegisterIntNDtypes(PyObject* numpy) {
  return (RegisterIntNDtype<Types>(numpy) && ...);
}

}  // namespace

// Initializes the module.
bool Initialize() {
  zk_dtypes::ImportNumpy();
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  if (!RegisterIntNDtypes<
          // clang-format off
          int2,
          uint2,
          int4,
          uint4
          // clang-format on
          >(numpy.get())) {
    return false;
  }

#define REGISTER_FIELD_DTYPES(ActualType, ...)        \
  if (!RegisterFieldDtype<ActualType>(numpy.get())) { \
    return false;                                     \
  }
  ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(REGISTER_FIELD_DTYPES)
#undef REGISTER_FIELD_DTYPES

#define REGISTER_EXT_FIELD_DTYPES(ActualType, ...)    \
  if (!RegisterFieldDtype<ActualType>(numpy.get())) { \
    return false;                                     \
  }
  ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(REGISTER_EXT_FIELD_DTYPES)
#undef REGISTER_EXT_FIELD_DTYPES

#define REGISTER_EC_POINT_DTYPES(ActualType, ...)       \
  if (!RegisterEcPointDtype<ActualType>(numpy.get())) { \
    return false;                                       \
  }
  ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(REGISTER_EC_POINT_DTYPES)
#undef REGISTER_EC_POINT_DTYPES

  // CAUTION: RegisterEcPointCast must be executed before
  // RegisterEcPointMultiplyUFunc and RegisterEcPointAddOrSubUFunc.
  // Failure to adhere to this order will result in the NumPy
  // RuntimeWarning because the cast will be considered impossible after
  // dependent UFuncs have been used.
  bool success = RegisterOneWayCustomCast<int2, int4, int8_t>();
  success &= RegisterOneWayCustomCast<uint2, uint4, uint8_t>();
#define REGISTER_EC_POINT_CASTS(ActualType, ...) \
  if (!RegisterEcPointCast<ActualType>()) {      \
    return false;                                \
  }
  ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(REGISTER_EC_POINT_CASTS)
#undef REGISTER_EC_POINT_CASTS

  // NOTE: Elliptic curve operations requires every elliptic curve point type
  // to be registered.
#define REGISTER_EC_POINT_MULTIPLY_UFUNCS(ActualType, ...)      \
  if (!RegisterEcPointMultiplyUFunc<ActualType>(numpy.get())) { \
    return false;                                               \
  }
  ZK_DTYPES_SCALAR_FIELD_TYPE_LIST(REGISTER_EC_POINT_MULTIPLY_UFUNCS)
#undef REGISTER_EC_POINT_MULTIPLY_UFUNCS

#define REGISTER_EC_POINT_ADD_OR_SUB_UFUNCS(ActualType, ...)    \
  if (!RegisterEcPointAddOrSubUFunc<ActualType>(numpy.get())) { \
    return false;                                               \
  }
  ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(REGISTER_EC_POINT_ADD_OR_SUB_UFUNCS)
#undef REGISTER_EC_POINT_ADD_OR_SUB_UFUNCS

  return success;
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_zk_dtypes_ext",
};

// TODO(phawkins): PyMODINIT_FUNC handles visibility correctly in Python 3.9+.
// Just use PyMODINIT_FUNC after dropping Python 3.8 support.
#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

extern "C" EXPORT_SYMBOL PyObject* PyInit__zk_dtypes_ext() {
  Safe_PyObjectPtr m = make_safe(PyModule_Create(&module_def));
  if (!m) {
    return nullptr;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load _zk_dtypes_ext module.");
    }
    return nullptr;
  }

  if (!InitModuleType<int2>(m.get(), "int2") ||
      !InitModuleType<int4>(m.get(), "int4") ||
      !InitModuleType<uint2>(m.get(), "uint2") ||
      !InitModuleType<uint4>(m.get(), "uint4")) {
    return nullptr;
  }

#define INIT_MODULE_TYPE(ActualType, UpperCamelCaseName, UpperSnakeCaseName, \
                         LowerSnakeCaseName)                                 \
  if (!InitModuleType<ActualType>(m.get(), #LowerSnakeCaseName)) {           \
    return nullptr;                                                          \
  }
  ZK_DTYPES_PUBLIC_TYPE_LIST(INIT_MODULE_TYPE)
  ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(INIT_MODULE_TYPE)
#undef INIT_MODULE_TYPE

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m.get(), Py_MOD_GIL_NOT_USED);
#endif

  return m.release();
}
}  // namespace zk_dtypes
