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

template <typename T>
struct TypeDescriptorBase;

//===----------------------------------------------------------------------===//
// PrimeField TypeDescriptorBase
//===----------------------------------------------------------------------===//

template <>
struct TypeDescriptorBase<Babybear> : FieldTypeDescriptor<Babybear> {
  static constexpr const char* kTpDoc =
      "babybear field values on standard domain";
  static constexpr char kNpyDescrType = 'B';
};

template <>
struct TypeDescriptorBase<BabybearMont> : FieldTypeDescriptor<BabybearMont> {
  static constexpr const char* kTpDoc =
      "babybear field values on montgomery domain";
  static constexpr char kNpyDescrType = 'b';
};

template <>
struct TypeDescriptorBase<Goldilocks> : FieldTypeDescriptor<Goldilocks> {
  static constexpr const char* kTpDoc =
      "goldilocks field values on standard domain";
  static constexpr char kNpyDescrType = 'G';
};

template <>
struct TypeDescriptorBase<GoldilocksMont>
    : FieldTypeDescriptor<GoldilocksMont> {
  static constexpr const char* kTpDoc =
      "goldilocks field values on montgomery domain";
  static constexpr char kNpyDescrType = 'g';
};

template <>
struct TypeDescriptorBase<Koalabear> : FieldTypeDescriptor<Koalabear> {
  static constexpr const char* kTpDoc =
      "koalabear field values on standard domain";
  static constexpr char kNpyDescrType = 'K';
};

template <>
struct TypeDescriptorBase<KoalabearMont> : FieldTypeDescriptor<KoalabearMont> {
  static constexpr const char* kTpDoc =
      "koalabear field values on montgomery domain";
  static constexpr char kNpyDescrType = 'k';
};

template <>
struct TypeDescriptorBase<Mersenne31> : FieldTypeDescriptor<Mersenne31> {
  static constexpr const char* kTpDoc =
      "mersenne31 field values on standard domain";
  static constexpr char kNpyDescrType = 'm';
};

template <>
struct TypeDescriptorBase<bn254::Fr> : FieldTypeDescriptor<bn254::Fr> {
  static constexpr const char* kTpDoc =
      "bn254 scalar field values on standard domain";
  static constexpr char kNpyDescrType = 'B';
};

template <>
struct TypeDescriptorBase<bn254::FrMont> : FieldTypeDescriptor<bn254::FrMont> {
  static constexpr const char* kTpDoc =
      "bn254 scalar field values on montgomery domain";
  static constexpr char kNpyDescrType = 'b';
};

//===----------------------------------------------------------------------===//
// ExtendedField TypeDescriptorBase
//===----------------------------------------------------------------------===//

template <>
struct TypeDescriptorBase<BabybearX4> : FieldTypeDescriptor<BabybearX4> {
  static constexpr const char* kTpDoc =
      "babybear quartic extension field values on standard domain";
  static constexpr char kNpyDescrType = 'D';
};

template <>
struct TypeDescriptorBase<BabybearX4Mont>
    : FieldTypeDescriptor<BabybearX4Mont> {
  static constexpr const char* kTpDoc =
      "babybear quartic extension field values on montgomery domain";
  static constexpr char kNpyDescrType = 'd';
};

template <>
struct TypeDescriptorBase<KoalabearX4> : FieldTypeDescriptor<KoalabearX4> {
  static constexpr const char* kTpDoc =
      "koalabear quartic extension field values on standard domain";
  static constexpr char kNpyDescrType = 'E';
};

template <>
struct TypeDescriptorBase<KoalabearX4Mont>
    : FieldTypeDescriptor<KoalabearX4Mont> {
  static constexpr const char* kTpDoc =
      "koalabear quartic extension field values on montgomery domain";
  static constexpr char kNpyDescrType = 'e';
};

template <>
struct TypeDescriptorBase<GoldilocksX3> : FieldTypeDescriptor<GoldilocksX3> {
  static constexpr const char* kTpDoc =
      "goldilocks cubic extension field values on standard domain";
  static constexpr char kNpyDescrType = 'T';
};

template <>
struct TypeDescriptorBase<GoldilocksX3Mont>
    : FieldTypeDescriptor<GoldilocksX3Mont> {
  static constexpr const char* kTpDoc =
      "goldilocks cubic extension field values on montgomery domain";
  static constexpr char kNpyDescrType = 't';
};

template <>
struct TypeDescriptorBase<Mersenne31X2> : FieldTypeDescriptor<Mersenne31X2> {
  static constexpr const char* kTpDoc =
      "mersenne31x2 quadratic extension field values on standard domain";
  static constexpr char kNpyDescrType = 'q';
};

template <>
struct TypeDescriptorBase<BinaryFieldT0> : FieldTypeDescriptor<BinaryFieldT0> {
  static constexpr const char* kTpDoc = "GF(2) binary field values";
  static constexpr char kNpyDescrType = '1';
};

template <>
struct TypeDescriptorBase<BinaryFieldT1> : FieldTypeDescriptor<BinaryFieldT1> {
  static constexpr const char* kTpDoc = "GF(2²) binary field values";
  static constexpr char kNpyDescrType = '2';
};

template <>
struct TypeDescriptorBase<BinaryFieldT2> : FieldTypeDescriptor<BinaryFieldT2> {
  static constexpr const char* kTpDoc = "GF(2⁴) binary field values";
  static constexpr char kNpyDescrType = '4';
};

template <>
struct TypeDescriptorBase<BinaryFieldT3> : FieldTypeDescriptor<BinaryFieldT3> {
  static constexpr const char* kTpDoc = "GF(2⁸) binary field values";
  static constexpr char kNpyDescrType = '8';
};

template <>
struct TypeDescriptorBase<BinaryFieldT4> : FieldTypeDescriptor<BinaryFieldT4> {
  static constexpr const char* kTpDoc = "GF(2¹⁶) binary field values";
  static constexpr char kNpyDescrType = 's';
};

template <>
struct TypeDescriptorBase<BinaryFieldT5> : FieldTypeDescriptor<BinaryFieldT5> {
  static constexpr const char* kTpDoc = "GF(2³²) binary field values";
  static constexpr char kNpyDescrType = 'l';
};

template <>
struct TypeDescriptorBase<BinaryFieldT6> : FieldTypeDescriptor<BinaryFieldT6> {
  static constexpr const char* kTpDoc = "GF(2⁶⁴) binary field values";
  static constexpr char kNpyDescrType = 'L';
};

template <>
struct TypeDescriptorBase<BinaryFieldT7> : FieldTypeDescriptor<BinaryFieldT7> {
  static constexpr const char* kTpDoc = "GF(2¹²⁸) binary field values";
  static constexpr char kNpyDescrType = 'w';
};

//===----------------------------------------------------------------------===//
// EcPoint TypeDescriptorBase
//===----------------------------------------------------------------------===//

template <>
struct TypeDescriptorBase<bn254::G1AffinePoint>
    : EcPointTypeDescriptor<bn254::G1AffinePoint> {
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve affine point on standard domain";
  static constexpr char kNpyDescrType = 'A';
};

template <>
struct TypeDescriptorBase<bn254::G1AffinePointMont>
    : EcPointTypeDescriptor<bn254::G1AffinePointMont> {
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve affine point on montgomery domain";
  static constexpr char kNpyDescrType = 'a';
};

template <>
struct TypeDescriptorBase<bn254::G1JacobianPoint>
    : EcPointTypeDescriptor<bn254::G1JacobianPoint> {
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve jacobian point on standard domain";
  static constexpr char kNpyDescrType = 'J';
};

template <>
struct TypeDescriptorBase<bn254::G1JacobianPointMont>
    : EcPointTypeDescriptor<bn254::G1JacobianPointMont> {
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve jacobian point on montgomery domain";
  static constexpr char kNpyDescrType = 'j';
};

template <>
struct TypeDescriptorBase<bn254::G1PointXyzz>
    : EcPointTypeDescriptor<bn254::G1PointXyzz> {
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve xyzz point on standard domain";
  static constexpr char kNpyDescrType = 'X';
};

template <>
struct TypeDescriptorBase<bn254::G1PointXyzzMont>
    : EcPointTypeDescriptor<bn254::G1PointXyzzMont> {
  static constexpr const char* kTpDoc =
      "bn254 G1 elliptic curve xyzz point on montgomery domain";
  static constexpr char kNpyDescrType = 'x';
};

template <>
struct TypeDescriptorBase<bn254::G2AffinePoint>
    : EcPointTypeDescriptor<bn254::G2AffinePoint> {
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve affine point on standard domain";
  static constexpr char kNpyDescrType = 'A';
};

template <>
struct TypeDescriptorBase<bn254::G2AffinePointMont>
    : EcPointTypeDescriptor<bn254::G2AffinePointMont> {
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve affine point on montgomery domain";
  static constexpr char kNpyDescrType = 'a';
};

template <>
struct TypeDescriptorBase<bn254::G2JacobianPoint>
    : EcPointTypeDescriptor<bn254::G2JacobianPoint> {
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve jacobian point on standard domain";
  static constexpr char kNpyDescrType = 'J';
};

template <>
struct TypeDescriptorBase<bn254::G2JacobianPointMont>
    : EcPointTypeDescriptor<bn254::G2JacobianPointMont> {
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve jacobian point on montgomery domain";
  static constexpr char kNpyDescrType = 'j';
};

template <>
struct TypeDescriptorBase<bn254::G2PointXyzz>
    : EcPointTypeDescriptor<bn254::G2PointXyzz> {
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve xyzz point on standard domain";
  static constexpr char kNpyDescrType = 'X';
};

template <>
struct TypeDescriptorBase<bn254::G2PointXyzzMont>
    : EcPointTypeDescriptor<bn254::G2PointXyzzMont> {
  static constexpr const char* kTpDoc =
      "bn254 G2 elliptic curve xyzz point on montgomery domain";
  static constexpr char kNpyDescrType = 'x';
};

#define REGISTER_TYPE_DESCRIPTOR(ActualType, UpperCamelCaseName,         \
                                 UpperSnakeCaseName, LowerSnakeCaseName) \
  template <>                                                            \
  struct TypeDescriptor<ActualType> : TypeDescriptorBase<ActualType> {   \
    typedef ActualType T;                                                \
    static constexpr bool is_floating = false;                           \
    static constexpr bool is_integral = false;                           \
    static constexpr const char* kTypeName = #LowerSnakeCaseName;        \
    static constexpr const char* kQualifiedTypeName =                    \
        "zk_dtypes." #LowerSnakeCaseName;                                \
    static constexpr char kNpyDescrKind = 'V';                           \
    static constexpr char kNpyDescrByteorder = '=';                      \
  };

ZK_DTYPES_PUBLIC_TYPE_LIST(REGISTER_TYPE_DESCRIPTOR)
#undef REGISTER_TYPE_DESCRIPTOR

}  // namespace zk_dtypes

// Pairing support: close namespace to include bn254_curve.h (which opens its
// own zk_dtypes namespace), then reopen for the rest of dtypes.cc.
#include <vector>

#include "zk_dtypes/include/elliptic_curve/bn/bn254/bn254_curve.h"

namespace zk_dtypes {

// pairing_check(g1_points, g2_points) -> bool
//
// Computes: e(g1[0], g2[0]) · ... · e(g1[n-1], g2[n-1]) == 1
PyObject* PyBn254PairingCheck(PyObject* /*self*/, PyObject* args) {
  PyObject* g1_obj;
  PyObject* g2_obj;

  if (!PyArg_ParseTuple(args, "OO", &g1_obj, &g2_obj)) {
    return nullptr;
  }

  int g1_dtype = TypeDescriptor<bn254::G1AffinePoint>::Dtype();
  int g2_dtype = TypeDescriptor<bn254::G2AffinePoint>::Dtype();

  PyArrayObject* g1_arr = reinterpret_cast<PyArrayObject*>(
      PyArray_FromAny(g1_obj, PyArray_DescrFromType(g1_dtype), 1, 1,
                      NPY_ARRAY_C_CONTIGUOUS, nullptr));
  if (!g1_arr) {
    PyErr_SetString(PyExc_TypeError,
                    "g1_points must be a 1-D array of bn254_g1_affine");
    return nullptr;
  }
  Safe_PyObjectPtr g1_guard(reinterpret_cast<PyObject*>(g1_arr));

  PyArrayObject* g2_arr = reinterpret_cast<PyArrayObject*>(
      PyArray_FromAny(g2_obj, PyArray_DescrFromType(g2_dtype), 1, 1,
                      NPY_ARRAY_C_CONTIGUOUS, nullptr));
  if (!g2_arr) {
    PyErr_SetString(PyExc_TypeError,
                    "g2_points must be a 1-D array of bn254_g2_affine");
    return nullptr;
  }
  Safe_PyObjectPtr g2_guard(reinterpret_cast<PyObject*>(g2_arr));

  npy_intp n = PyArray_SIZE(g1_arr);
  if (n != PyArray_SIZE(g2_arr)) {
    PyErr_SetString(PyExc_ValueError,
                    "g1_points and g2_points must have the same length");
    return nullptr;
  }
  if (n == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "g1_points and g2_points must not be empty");
    return nullptr;
  }

  const auto* g1_data =
      static_cast<const bn254::G1AffinePoint*>(PyArray_DATA(g1_arr));
  const auto* g2_data =
      static_cast<const bn254::G2AffinePoint*>(PyArray_DATA(g2_arr));

  bn254::BN254CurveConfig::Init();

  using G2Prepared = bn254::BN254Curve::G2Prepared;
  std::vector<G2Prepared> g2_prepared;
  g2_prepared.reserve(n);
  for (npy_intp i = 0; i < n; ++i) {
    g2_prepared.push_back(G2Prepared::From(g2_data[i]));
  }

  std::vector<bn254::G1AffinePoint> g1_vec(g1_data, g1_data + n);
  auto f = bn254::BN254Curve::MultiMillerLoop(g1_vec, g2_prepared);
  auto result = bn254::BN254Curve::FinalExponentiation(f);

  if (result.IsOne()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyMethodDef kPairingMethods[] = {
    {"pairing_check", PyBn254PairingCheck, METH_VARARGS,
     "BN254 multi-pairing check.\n\n"
     "pairing_check(g1_points, g2_points) -> bool\n\n"
     "Checks: e(g1[0], g2[0]) * ... * e(g1[n-1], g2[n-1]) == 1\n\n"
     "Args:\n"
     "  g1_points: 1-D array of bn254_g1_affine, shape (n,)\n"
     "  g2_points: 1-D array of bn254_g2_affine, shape (n,)\n\n"
     "Returns:\n"
     "  True if product of pairings equals identity in GT."},
    {nullptr, nullptr, 0, nullptr},
};

bool RegisterPairingMethods(PyObject* module) {
  for (PyMethodDef* def = kPairingMethods; def->ml_name != nullptr; ++def) {
    Safe_PyObjectPtr func(PyCFunction_NewEx(def, nullptr, nullptr));
    if (!func) {
      return false;
    }
    if (PyModule_AddObject(module, def->ml_name, func.release()) < 0) {
      return false;
    }
  }
  return true;
}

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
  ZK_DTYPES_PUBLIC_FIELD_TYPE_LIST(REGISTER_FIELD_DTYPES)
#undef REGISTER_FIELD_DTYPES

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
#undef INIT_MODULE_TYPE

  if (!RegisterPairingMethods(m.get())) {
    return nullptr;
  }

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m.get(), Py_MOD_GIL_NOT_USED);
#endif

  return m.release();
}
}  // namespace zk_dtypes
