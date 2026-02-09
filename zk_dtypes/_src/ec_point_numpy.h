/* Copyright 2025 The zk_dtypes Authors.

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

#ifndef ZK_DTYPES__SRC_EC_POINT_NUMPY_H_
#define ZK_DTYPES__SRC_EC_POINT_NUMPY_H_

#include <array>

// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "numpy/ndarraytypes.h"

#include "zk_dtypes/_src/field_numpy.h"
#include "zk_dtypes/_src/ufuncs.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/geometry/point_declarations.h"

namespace zk_dtypes {

constexpr char kEcPointOutOfRange[] =
    "out of range value cannot be converted to elliptic curve point";

template <typename T>
struct EcPointTypeDescriptor {
  static int Dtype() { return npy_type; }

  // Registered numpy type ID. Global variable populated by the registration
  // code. Protected by the GIL.
  static int npy_type;

  // Pointer to the python type object we are using. This is either a pointer
  // to type, if we choose to register it, or to the python type
  // registered by another system into NumPy.
  static PyObject* type_ptr;

  static PyType_Spec type_spec;
  static PyType_Slot type_slots[];

  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
  static PyArray_Descr* npy_descr;

  static PyGetSetDef getset[];
  static PyMethodDef methods[];
};

template <typename T>
int EcPointTypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* EcPointTypeDescriptor<T>::type_ptr = nullptr;
template <typename T>
PyArray_DescrProto EcPointTypeDescriptor<T>::npy_descr_proto;
template <typename T>
PyArray_Descr* EcPointTypeDescriptor<T>::npy_descr = nullptr;

// Representation of a Python field object.
template <typename T>
struct PyEcPoint {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyEcPoint.
template <typename T>
bool PyEcPoint_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
}

// Extracts the value of a PyEcPoint object.
template <typename T>
T PyEcPoint_Value_Unchecked(PyObject* object) {
  return reinterpret_cast<PyEcPoint<T>*>(object)->value;
}

template <typename T>
bool PyEcPoint_Value(PyObject* arg, T* output) {
  if (PyEcPoint_Check<T>(arg)) {
    *output = PyEcPoint_Value_Unchecked<T>(arg);
    return true;
  }
  return false;
}

// Constructs a PyEcPoint object from PyEcPoint<T>::T.
template <typename T>
Safe_PyObjectPtr PyEcPoint_FromValue(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyEcPoint<T>* p = reinterpret_cast<PyEcPoint<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a field value. Returns true on success,
// returns false and reports a Python error on failure.
template <typename T>
bool CastToEcPoint(PyObject* arg, T* output) {
  using BaseField = typename T::BaseField;
  using ScalarField = typename T::ScalarField;

  if (PyTuple_Check(arg)) {
    Py_ssize_t size = PyTuple_Size(arg);
    if constexpr (IsAffinePoint<T>) {
      if (size != 2) {
        PyErr_Format(PyExc_TypeError,
                     "ec point takes exactly two arguments, got %d", size);
        return false;
      }
    } else if constexpr (IsJacobianPoint<T>) {
      if (size != 3) {
        PyErr_Format(PyExc_TypeError,
                     "ec point takes exactly three arguments, got %d", size);
        return false;
      }
    } else if constexpr (IsPointXyzz<T>) {
      if (size != 4) {
        PyErr_Format(PyExc_TypeError,
                     "ec point takes exactly four arguments, got %d", size);
        return false;
      }
    }
    PyObject* arg0 = PyTuple_GetItem(arg, 0);
    PyObject* arg1 = PyTuple_GetItem(arg, 1);
    BaseField x, y;
    if (!CastToField(arg0, &x)) {
      PyErr_Format(PyExc_TypeError, "expected base field, got %s",
                   Py_TYPE(arg0)->tp_name);
      return false;
    }
    if (!CastToField(arg1, &y)) {
      PyErr_Format(PyExc_TypeError, "expected base field, got %s",
                   Py_TYPE(arg1)->tp_name);
      return false;
    }
    if constexpr (IsAffinePoint<T>) {
      *output = T(x, y);
      return true;
    } else if constexpr (IsJacobianPoint<T>) {
      PyObject* arg2 = PyTuple_GetItem(arg, 2);
      BaseField z;
      if (!CastToField(arg2, &z)) {
        PyErr_Format(PyExc_TypeError, "expected base field, got %s",
                     Py_TYPE(arg2)->tp_name);
        return false;
      }
      *output = T(x, y, z);
      return true;
    } else if constexpr (IsPointXyzz<T>) {
      PyObject* arg2 = PyTuple_GetItem(arg, 2);
      PyObject* arg3 = PyTuple_GetItem(arg, 3);
      BaseField zz, zzz;
      if (!CastToField(arg2, &zz)) {
        PyErr_Format(PyExc_TypeError, "expected base field, got %s",
                     Py_TYPE(arg2)->tp_name);
        return false;
      }
      if (!CastToField(arg3, &zzz)) {
        PyErr_Format(PyExc_TypeError, "expected base field, got %s",
                     Py_TYPE(arg3)->tp_name);
        return false;
      }
      *output = T(x, y, zz, zzz);
      return true;
    }
  } else {
    ScalarField scalar;
    if (CastToField(arg, &scalar)) {
      if constexpr (IsAffinePoint<T>) {
        if (scalar.IsZero()) {
          *output = T::Zero();
          return true;
        }
        *output = (T::Generator() * scalar).ToAffine();
      } else {
        *output = T::Generator() * scalar;
      }
      return true;
    }
  }
  return false;
}

// Constructs a new PyEcPoint.
template <typename T>
PyObject* PyEcPoint_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "constructor takes exactly one argument, got %d", size);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  T value;
  if (PyEcPoint_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToEcPoint(arg, &value)) {
    return PyEcPoint_FromValue(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      return PyArray_Cast(arr, TypeDescriptor<T>::Dtype());
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    // Parse integer from string, then cast to T.
    PyObject* f = PyLong_FromUnicodeObject(arg, /*base=*/0);
    if (PyErr_Occurred()) {
      return nullptr;
    }
    if (CastToEcPoint(f, &value)) {
      return PyEcPoint_FromValue(value).release();
    }
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

template <typename T, bool kIsAdd>
PyObject* PyEcPoint_nb_add_or_sub(PyObject* a, PyObject* b) {
  T x;
  if (!PyEcPoint_Value(a, &x)) {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for %s",
                 TypeDescriptor<T>::kTypeName,
                 kIsAdd ? "addition" : "subtraction");
    return nullptr;
  }
  T ay;
  if (PyEcPoint_Value(b, &ay)) {
    if constexpr (kIsAdd) {
      return PyEcPoint_FromValue(x + ay).release();
    } else {
      return PyEcPoint_FromValue(x - ay).release();
    }
  }
  if constexpr (IsAffinePoint<T>) {
    using JacobianPoint = typename T::JacobianPoint;
    using PointXyzz = typename T::PointXyzz;

    JacobianPoint jy;
    if (PyEcPoint_Value(b, &jy)) {
      if constexpr (kIsAdd) {
        return PyEcPoint_FromValue(x + jy).release();
      } else {
        return PyEcPoint_FromValue(x - jy).release();
      }
    }
    PointXyzz xy;
    if (PyEcPoint_Value(b, &xy)) {
      if constexpr (kIsAdd) {
        return PyEcPoint_FromValue(x + xy).release();
      } else {
        return PyEcPoint_FromValue(x - xy).release();
      }
    }
  } else {
    using AffinePoint = typename T::AffinePoint;

    AffinePoint ay;
    if (PyEcPoint_Value(b, &ay)) {
      if constexpr (kIsAdd) {
        return PyEcPoint_FromValue(x + ay).release();
      } else {
        return PyEcPoint_FromValue(x - ay).release();
      }
    }
  }

  PyErr_Format(PyExc_TypeError, "invalid argument for %s",
               kIsAdd ? "addition" : "subtraction");
  return nullptr;
}

template <typename T>
PyObject* PyEcPoint_nb_multiply(PyObject* a, PyObject* b) {
  using ScalarField = typename T::ScalarField;

  T x;
  ScalarField y;
  if (PyEcPoint_Value(a, &x) && PyField_Value(b, &y)) {
    return PyEcPoint_FromValue(x * y).release();
  } else {
    PyErr_SetString(PyExc_TypeError, "invalid argument for multiplication");
    return nullptr;
  }
}

template <typename T>
PyObject* PyEcPoint_nb_negative(PyObject* a) {
  T x;
  if (PyEcPoint_Value(a, &x)) {
    return PyEcPoint_FromValue(-x).release();
  } else {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for negation",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
}

// Implementation of repr() for PyEcPoint.
template <typename T>
PyObject* PyEcPoint_Repr(PyObject* self) {
  T x = PyEcPoint_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

// Implementation of str() for PyEcPoint.
template <typename T>
PyObject* PyEcPoint_Str(PyObject* self) {
  T x = PyEcPoint_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

template <typename T>
uint64_t PyField_Hash_Impl(T x);

// Hash function for PyEcPoint.
template <typename T>
Py_hash_t PyEcPoint_Hash(PyObject* self) {
  T x = PyEcPoint_Value_Unchecked<T>(self);

  uint64_t hash = PyField_Hash_Impl(x.x());
  if constexpr (IsAffinePoint<T>) {
    hash ^= PyField_Hash_Impl(x.y());
  } else if constexpr (IsJacobianPoint<T>) {
    hash ^= PyField_Hash_Impl(x.y());
    hash ^= PyField_Hash_Impl(x.z());
  } else if constexpr (IsPointXyzz<T>) {
    hash ^= PyField_Hash_Impl(x.y());
    hash ^= PyField_Hash_Impl(x.zz());
    hash ^= PyField_Hash_Impl(x.zzz());
  }

  // Hash functions must not return -1.
  return hash == -1 ? static_cast<Py_hash_t>(-2) : static_cast<Py_hash_t>(hash);
}

// Comparisons on PyEcPoints.
template <typename T>
PyObject* PyEcPoint_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!PyEcPoint_Value(a, &x) || !PyEcPoint_Value(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "Invalid op type");
      return nullptr;
  }
  PyArrayScalar_RETURN_BOOL_FROM_LONG(result);
}

template <typename... Args>
PyObject* MakeRawTuple(Args&&... args) {
  constexpr size_t N = sizeof...(Args);
  PyObject* tuple = PyTuple_New(N);
  if (!tuple) return nullptr;

  std::array<PyObject*, N> items = {
      FieldToRawPyObject(std::forward<Args>(args))...};

  for (size_t i = 0; i < N; ++i) {
    if (!items[i]) {
      // Only decref items not yet set into tuple (j > i).
      // Items 0..i-1 are already owned by tuple and will be cleaned up by
      // Py_DECREF(tuple). Item i is null, so skip it.
      // Use Py_XDECREF (not Py_DECREF) because items[j] may be null if
      // FieldToRawPyObject failed during pack expansion.
      for (size_t j = i + 1; j < N; ++j) {
        Py_XDECREF(items[j]);
      }
      Py_DECREF(tuple);
      return nullptr;
    }
    // PyTuple_SET_ITEM steals a reference
    PyTuple_SET_ITEM(tuple, i, items[i]);
  }
  return tuple;
}

template <typename T>
PyObject* PyEcPoint_GetRaw(PyObject* self, void* closure) {
  T p = PyEcPoint_Value_Unchecked<T>(self);

  if constexpr (IsAffinePoint<T>) {
    return MakeRawTuple(p.x(), p.y());
  } else if constexpr (IsJacobianPoint<T>) {
    return MakeRawTuple(p.x(), p.y(), p.z());
  } else if constexpr (IsPointXyzz<T>) {
    return MakeRawTuple(p.x(), p.y(), p.zz(), p.zzz());
  }
  return nullptr;
}

// Helper to get point type name for error messages
template <typename T>
constexpr const char* EcPointTypeName() {
  if constexpr (IsAffinePoint<T>)
    return "affine";
  else if constexpr (IsJacobianPoint<T>)
    return "jacobian";
  else if constexpr (IsPointXyzz<T>)
    return "xyzz";
  return "unknown";
}

// Helper to parse coordinates from tuple using fold expression
template <typename BaseField, size_t... Is>
bool ParseCoordinates(PyObject* tuple,
                      std::array<BaseField, sizeof...(Is)>& coords,
                      std::index_sequence<Is...>) {
  return (... &&
          ParseRawFieldFromPyObject(PyTuple_GetItem(tuple, Is), &coords[Is]));
}

// Helper to construct EC point from coordinates array
template <typename T, typename BaseField, size_t N>
T ConstructEcPoint(const std::array<BaseField, N>& c) {
  if constexpr (IsAffinePoint<T>)
    return T(c[0], c[1]);
  else if constexpr (IsJacobianPoint<T>)
    return T(c[0], c[1], c[2]);
  else if constexpr (IsPointXyzz<T>)
    return T(c[0], c[1], c[2], c[3]);
}

// Helper to construct EC point from raw coordinate values
template <typename T>
bool CastToEcPointFromRaw(PyObject* arg, T* output) {
  using BaseField = typename T::BaseField;
  constexpr size_t kNumCoords = PointTraits<T>::kNumCoords;

  if (!PyTuple_Check(arg)) {
    PyErr_Format(PyExc_TypeError, "expected tuple, got %s",
                 Py_TYPE(arg)->tp_name);
    return false;
  }

  Py_ssize_t size = PyTuple_Size(arg);
  if (size != static_cast<Py_ssize_t>(kNumCoords)) {
    PyErr_Format(PyExc_TypeError,
                 "from_raw() for %s point takes %zu elements, got %zd",
                 EcPointTypeName<T>(), kNumCoords, size);
    return false;
  }

  std::array<BaseField, kNumCoords> coords;
  if (!ParseCoordinates(arg, coords, std::make_index_sequence<kNumCoords>{})) {
    return false;
  }

  *output = ConstructEcPoint<T>(coords);
  return true;
}

// Implementation of 'from_raw' classmethod for PyEcPoint.
template <typename T>
PyObject* PyEcPoint_FromRaw(PyObject* cls, PyObject* args) {
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "from_raw() takes exactly one argument, got %d", size);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  T value;
  if (!CastToEcPointFromRaw(arg, &value)) {
    return nullptr;
  }
  return PyEcPoint_FromValue(value).release();
}

template <typename T>
PyGetSetDef EcPointTypeDescriptor<T>::getset[] = {
    {"raw", PyEcPoint_GetRaw<T>, nullptr,
     "Raw internal representation of coordinates", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <typename T>
PyMethodDef EcPointTypeDescriptor<T>::methods[] = {
    {"from_raw", PyEcPoint_FromRaw<T>, METH_VARARGS | METH_CLASS,
     "Construct from raw internal coordinate representation"},
    {nullptr, nullptr, 0, nullptr}};

template <typename T>
PyType_Slot EcPointTypeDescriptor<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyEcPoint_tp_new<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyEcPoint_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyEcPoint_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyEcPoint_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyEcPoint_RichCompare<T>)},
    {Py_tp_getset, reinterpret_cast<void*>(EcPointTypeDescriptor<T>::getset)},
    {Py_tp_methods, reinterpret_cast<void*>(EcPointTypeDescriptor<T>::methods)},
    {Py_nb_add, reinterpret_cast<void*>(PyEcPoint_nb_add_or_sub<T, true>)},
    {Py_nb_subtract,
     reinterpret_cast<void*>(PyEcPoint_nb_add_or_sub<T, false>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyEcPoint_nb_multiply<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyEcPoint_nb_negative<T>)},
    {0, nullptr},
};

template <typename T>
PyType_Spec EcPointTypeDescriptor<T>::type_spec = {
    .name = TypeDescriptor<T>::kQualifiedTypeName,
    .basicsize = static_cast<int>(sizeof(PyEcPoint<T>)),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = EcPointTypeDescriptor<T>::type_slots,
};

// Numpy support
template <typename T>
PyArray_ArrFuncs EcPointTypeDescriptor<T>::arr_funcs;

template <typename T>
PyArray_DescrProto GetEcPointDescrProto() {
  return {
      PyObject_HEAD_INIT(nullptr).typeobj = nullptr,  // Filled in later
      .kind = TypeDescriptor<T>::kNpyDescrKind,
      .type = TypeDescriptor<T>::kNpyDescrType,
      .byteorder = TypeDescriptor<T>::kNpyDescrByteorder,
      .flags = NPY_USE_SETITEM,
      .type_num = 0,
      .elsize = sizeof(T),
      .alignment = alignof(T),
      .subarray = nullptr,
      .fields = nullptr,
      .names = nullptr,
      .f = &EcPointTypeDescriptor<T>::arr_funcs,
      .metadata = nullptr,
      .c_metadata = nullptr,
      .hash = -1,  // -1 means "not computed yet".
  };
}

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyEcPoint_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyEcPoint_FromValue(x).release();
}

template <typename T>
int NPyEcPoint_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (PyEcPoint_Check<T>(item)) {
    x = PyEcPoint_Value_Unchecked<T>(item);
  } else if (!CastToEcPoint<T>(item, &x)) {
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

template <typename T>
void NPyEcPoint_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                          npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);

  if (src) {
    if (dstride == sizeof(T) && sstride == sizeof(T)) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (npy_intp i = 0; i < n; i++) {
        memcpy(dst + dstride * i, src + sstride * i, sizeof(T));
      }
    }
  }
  // Note: No byte swapping needed for 8-bit integer types
}

template <typename T>
void NPyEcPoint_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (src) {
    memcpy(dst, src, sizeof(T));
  }
  // Note: No byte swapping needed for 8-bit integer types
}

template <typename T>
npy_bool NPyEcPoint_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return !x.IsZero();
}

template <typename T>
int NPyEcPoint_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  if constexpr (IsAffinePoint<T>) {
    using R = typename T::JacobianPoint;

    std::vector<R> tmp_buffers;
    tmp_buffers.resize(length);
    T* const buffer = reinterpret_cast<T*>(buffer_raw);
    R* tmp_buffer = const_cast<R*>(tmp_buffers.data());
    const T start(buffer[0]);
    const R delta = static_cast<T>(buffer[1]) - start;
    tmp_buffers[0] = start.ToJacobian();
    tmp_buffers[1] = static_cast<T>(buffer[1]).ToJacobian();
    for (npy_intp i = 2; i < length; ++i) {
      tmp_buffer[i] = tmp_buffer[i - 1] + delta;
    }
    absl::Span<T> output(buffer, length);
    absl::Status status =
        R::BatchToAffine(absl::Span<const R>(tmp_buffer, length), &output);
    if (!status.ok()) return -1;
  } else {
    T* const buffer = reinterpret_cast<T*>(buffer_raw);
    const T start(buffer[0]);
    const T delta = static_cast<T>(buffer[1]) - start;
    for (npy_intp i = 2; i < length; ++i) {
      buffer[i] = buffer[i - 1] + delta;
    }
  }
  return 0;
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyEcPoint_PointCast(void* from_void, void* to_void, npy_intp n,
                          void* fromarr, void* toarr) {
  using SrcType = typename TypeDescriptor<From>::T;
  using DstType = typename TypeDescriptor<To>::T;

  absl::Span<const SrcType> from(reinterpret_cast<const SrcType*>(from_void),
                                 n);
  absl::Span<DstType> to(reinterpret_cast<DstType*>(to_void), n);

  if constexpr (IsAffinePoint<DstType>) {
    absl::Status status = SrcType::BatchToAffine(from, &to);
    CHECK(status.ok());
  } else {
    for (npy_intp i = 0; i < n; ++i) {
      if constexpr (IsJacobianPoint<DstType>) {
        to[i] = from[i].ToJacobian();
      } else {
        to[i] = from[i].ToXyzz();
      }
    }
  }
}

// Cast from integer type to EC point type.
// 0 → identity (Zero), non-zero n → n * Generator.
template <typename IntType, typename EcType>
void NPyEcPoint_IntegerCast(void* from_void, void* to_void, npy_intp n,
                            void* /*fromarr*/, void* /*toarr*/) {
  using IntT = typename TypeDescriptor<IntType>::T;
  const auto* from = reinterpret_cast<const IntT*>(from_void);
  auto* to = reinterpret_cast<EcType*>(to_void);
  using ScalarField = typename EcType::ScalarField;
  for (npy_intp i = 0; i < n; ++i) {
    auto val = static_cast<uint64_t>(from[i]);
    if (val == 0) {
      to[i] = EcType::Zero();
    } else {
      ScalarField scalar(val);
      if constexpr (IsAffinePoint<EcType>) {
        to[i] = (EcType::Generator() * scalar).ToAffine();
      } else {
        to[i] = EcType::Generator() * scalar;
      }
    }
  }
}

// Registers a cast between 'T' and type 'OtherT'. 'numpy_type'
// is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterEcPointCast_Impl(
    int numpy_type = EcPointTypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, EcPointTypeDescriptor<T>::Dtype(),
                               NPyEcPoint_PointCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(EcPointTypeDescriptor<T>::npy_descr, numpy_type,
                               NPyEcPoint_PointCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterEcPointCast() {
  if constexpr (IsAffinePoint<T>) {
    using J = typename T::JacobianPoint;
    using X = typename T::PointXyzz;

    if (!RegisterEcPointCast_Impl<T, J>()) {
      return false;
    }
    if (!RegisterEcPointCast_Impl<T, X>()) {
      return false;
    }

    // Safe casts from T to J and X
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr,
                                EcPointTypeDescriptor<J>::Dtype(),
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr,
                                EcPointTypeDescriptor<X>::Dtype(),
                                NPY_NOSCALAR) < 0) {
      return false;
    }

    // Safe casts from J and X to T
    if (PyArray_RegisterCanCast(TypeDescriptor<J>::npy_descr,
                                EcPointTypeDescriptor<T>::Dtype(),
                                NPY_NOSCALAR) < 0) {
      return false;
    }
    if (PyArray_RegisterCanCast(TypeDescriptor<X>::npy_descr,
                                EcPointTypeDescriptor<T>::Dtype(),
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  } else {
    using A = typename T::AffinePoint;

    if (!RegisterEcPointCast_Impl<T, A>()) {
      return false;
    }

    // Safe casts from T to A
    if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr,
                                EcPointTypeDescriptor<A>::Dtype(),
                                NPY_NOSCALAR) < 0) {
      return false;
    }

    // Safe casts from A to T
    if (PyArray_RegisterCanCast(TypeDescriptor<A>::npy_descr,
                                EcPointTypeDescriptor<T>::Dtype(),
                                NPY_NOSCALAR) < 0) {
      return false;
    }
  }
  return true;
}

// Registers a one-way cast from integer type 'IntType' to EC point type 'T'.
template <typename T, typename IntType>
bool RegisterEcPointIntegerCast_Impl(
    int numpy_type = TypeDescriptor<IntType>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, EcPointTypeDescriptor<T>::Dtype(),
                               NPyEcPoint_IntegerCast<IntType, T>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterEcPointIntegerCasts() {
  return RegisterEcPointIntegerCast_Impl<T, bool>(NPY_BOOL) &&
         RegisterEcPointIntegerCast_Impl<T, unsigned char>(NPY_UBYTE) &&
         RegisterEcPointIntegerCast_Impl<T, unsigned short>(NPY_USHORT) &&
         RegisterEcPointIntegerCast_Impl<T, unsigned int>(NPY_UINT) &&
         RegisterEcPointIntegerCast_Impl<T, unsigned long>(NPY_ULONG) &&
         RegisterEcPointIntegerCast_Impl<T, unsigned long long>(
             NPY_ULONGLONG) &&
         RegisterEcPointIntegerCast_Impl<T, signed char>(NPY_BYTE) &&
         RegisterEcPointIntegerCast_Impl<T, short>(NPY_SHORT) &&
         RegisterEcPointIntegerCast_Impl<T, int>(NPY_INT) &&
         RegisterEcPointIntegerCast_Impl<T, long>(NPY_LONG) &&
         RegisterEcPointIntegerCast_Impl<T, long long>(NPY_LONGLONG);
}

template <typename T>
bool RegisterEcPointUFuncs(PyObject* numpy) {
  // NOTE: Register ufuncs that operate solely on T types. If the operation
  // involves other types, register them lazily. Examples of lazy registration
  // include RegisterEcPointMultiplyUFunc and RegisterEcPointAddOrSubUFunc.

  bool ok =
      RegisterUFunc<UFunc<ufuncs::Negative<T>, T, T>, T>(numpy, "negative") &&
      // Comparison functions
      RegisterUFunc<UFunc<ufuncs::Eq<T>, bool, T, T>, T>(numpy, "equal") &&
      RegisterUFunc<UFunc<ufuncs::Ne<T>, bool, T, T>, T>(numpy, "not_equal");

  return ok;
}

template <typename ScalarField>
bool RegisterEcPointMul_Impl(PyObject* numpy) {
  return true;
}

template <typename ScalarField, typename T, typename... Args>
bool RegisterEcPointMul_Impl(PyObject* numpy) {
  using R = typename AddResult<T>::Type;

  bool ok =
      RegisterUFunc<UFunc<ufuncs::Multiply<ScalarField, T>, R, ScalarField, T>,
                    ScalarField>(numpy, "multiply") &&
      RegisterUFunc<UFunc<ufuncs::Multiply<T, ScalarField>, R, T, ScalarField>,
                    T>(numpy, "multiply") &&
      RegisterEcPointMul_Impl<ScalarField, Args...>(numpy);

  return ok;
}

template <typename ScalarField>
bool RegisterEcPointMultiplyUFunc(PyObject* numpy) {
  if constexpr (std::is_same_v<ScalarField, bn254::Fr>) {
    return RegisterEcPointMul_Impl<bn254::Fr, bn254::G1AffinePoint,
                                   bn254::G1JacobianPoint, bn254::G1PointXyzz,
                                   bn254::G2AffinePoint, bn254::G2JacobianPoint,
                                   bn254::G2PointXyzz>(numpy);
  } else if constexpr (std::is_same_v<ScalarField, bn254::FrMont>) {
    return RegisterEcPointMul_Impl<
        bn254::FrMont, bn254::G1AffinePointMont, bn254::G1JacobianPointMont,
        bn254::G1PointXyzzMont, bn254::G2AffinePointMont,
        bn254::G2JacobianPointMont, bn254::G2PointXyzzMont>(numpy);
  } else {
    return false;
  }
}

template <typename T, template <typename, typename> class Functor>
bool RegisterEcPointAddOrSubUFunc_Impl(PyObject* numpy, const char* name) {
  if constexpr (IsAffinePoint<T>) {
    using A = T;
    using J = typename T::JacobianPoint;
    using X = typename T::PointXyzz;

    return RegisterUFunc<UFunc<Functor<A, A>, J, A, A>, A>(numpy, name) &&
           RegisterUFunc<UFunc<Functor<A, J>, J, A, J>, A>(numpy, name) &&
           RegisterUFunc<UFunc<Functor<A, X>, X, A, X>, A>(numpy, name);
  } else {
    using A = typename T::AffinePoint;

    return RegisterUFunc<UFunc<Functor<T, T>, T, T, T>, T>(numpy, name) &&
           RegisterUFunc<UFunc<Functor<T, A>, T, T, A>, T>(numpy, name);
  }

  return false;
}

template <typename T>
bool RegisterEcPointAddOrSubUFunc(PyObject* numpy) {
  return RegisterEcPointAddOrSubUFunc_Impl<T, ufuncs::Add>(numpy, "add") &&
         RegisterEcPointAddOrSubUFunc_Impl<T, ufuncs::Subtract>(numpy,
                                                                "subtract");
}

template <typename T>
bool RegisterEcPointDtype(PyObject* numpy) {
  // bases must be a tuple for Python 3.9 and earlier. Change to just pass
  // the base type directly when dropping Python 3.9 support.
  // TODO(jakevdp): it would be better to inherit from PyNumberArrType or
  // PyIntegerArrType, but this breaks some assumptions made by NumPy,
  // because dtype.kind='V' is then interpreted as a 'void' type in some
  // contexts.
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type = PyType_FromSpecWithBases(
      &EcPointTypeDescriptor<T>::type_spec, bases.get());
  if (!type) {
    return false;
  }
  TypeDescriptor<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("zk_dtypes"));
  if (!module) {
    return false;
  }
  if (PyObject_SetAttrString(TypeDescriptor<T>::type_ptr, "__module__",
                             module.get()) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_ArrFuncs& arr_funcs = EcPointTypeDescriptor<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyEcPoint_GetItem<T>;
  arr_funcs.setitem = NPyEcPoint_SetItem<T>;
  arr_funcs.copyswapn = NPyEcPoint_CopySwapN<T>;
  arr_funcs.copyswap = NPyEcPoint_CopySwap<T>;
  arr_funcs.nonzero = NPyEcPoint_NonZero<T>;
  arr_funcs.fill = NPyEcPoint_Fill<T>;

  // This is messy, but that's because the NumPy 2.0 API transition is messy.
  // Before 2.0, NumPy assumes we'll keep the descriptor passed in to
  // RegisterDataType alive, because it stores its pointer.
  // After 2.0, the proto and descriptor types diverge, and NumPy allocates
  // and manages the lifetime of the descriptor itself.
  PyArray_DescrProto& descr_proto = EcPointTypeDescriptor<T>::npy_descr_proto;
  descr_proto = GetEcPointDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  TypeDescriptor<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (TypeDescriptor<T>::npy_type < 0) {
    return false;
  }
  // TODO(phawkins): We intentionally leak the pointer to the descriptor.
  // Implement a better module destructor to handle this.
  EcPointTypeDescriptor<T>::npy_descr =
      PyArray_DescrFromType(TypeDescriptor<T>::npy_type);

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype(type_name)` work.
  if (PyDict_SetItemString(typeDict_obj.get(), TypeDescriptor<T>::kTypeName,
                           TypeDescriptor<T>::type_ptr) < 0) {
    return false;
  }

  // Support dtype(type_name)
  if (PyObject_SetAttrString(TypeDescriptor<T>::type_ptr, "dtype",
                             reinterpret_cast<PyObject*>(
                                 EcPointTypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterEcPointIntegerCasts<T>() && RegisterEcPointUFuncs<T>(numpy);
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_EC_POINT_NUMPY_H_
