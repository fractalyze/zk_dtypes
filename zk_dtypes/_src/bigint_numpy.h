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

#ifndef ZK_DTYPES__SRC_BIGINT_NUMPY_H_
#define ZK_DTYPES__SRC_BIGINT_NUMPY_H_

#include <string.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

#include "zk_dtypes/_src/common.h"
#include "zk_dtypes/_src/ufuncs.h"
#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/signed_big_int.h"

#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace zk_dtypes {

// Type trait to detect SignedBigInt.
template <typename T>
struct IsSignedBigIntImpl {
  static constexpr bool value = false;
};

template <size_t N>
struct IsSignedBigIntImpl<SignedBigInt<N>> {
  static constexpr bool value = true;
};

template <typename T>
constexpr bool IsSignedBigIntType = IsSignedBigIntImpl<T>::value;

// Limb count for BigInt-derived types.
template <typename T>
struct BigIntLimbCount;

template <size_t N>
struct BigIntLimbCount<BigInt<N>> {
  static constexpr size_t value = N;
};

template <size_t N>
struct BigIntLimbCount<SignedBigInt<N>> {
  static constexpr size_t value = N;
};

//===----------------------------------------------------------------------===//
// BigIntNTypeDescriptor
//===----------------------------------------------------------------------===//

template <typename T>
struct BigIntNTypeDescriptor {
  static int Dtype() { return npy_type; }

  static int npy_type;
  static PyObject* type_ptr;
  static PyType_Spec type_spec;
  static PyType_Slot type_slots[];
  static PyArray_ArrFuncs arr_funcs;
  static PyArray_DescrProto npy_descr_proto;
  static PyArray_Descr* npy_descr;
};

template <typename T>
int BigIntNTypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* BigIntNTypeDescriptor<T>::type_ptr = nullptr;
template <typename T>
PyArray_DescrProto BigIntNTypeDescriptor<T>::npy_descr_proto;
template <typename T>
PyArray_Descr* BigIntNTypeDescriptor<T>::npy_descr = nullptr;

//===----------------------------------------------------------------------===//
// Python object wrapper
//===----------------------------------------------------------------------===//

template <typename T>
struct PyBigIntN {
  PyObject_HEAD;
  T value;
};

template <typename T>
bool PyBigIntN_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
}

template <typename T>
T PyBigIntN_Value_Unchecked(PyObject* object) {
  return reinterpret_cast<PyBigIntN<T>*>(object)->value;
}

template <typename T>
bool PyBigIntN_Value(PyObject* arg, T* output) {
  if (PyBigIntN_Check<T>(arg)) {
    *output = PyBigIntN_Value_Unchecked<T>(arg);
    return true;
  }
  return false;
}

template <typename T>
Safe_PyObjectPtr PyBigIntN_FromValue(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyBigIntN<T>* p = reinterpret_cast<PyBigIntN<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

//===----------------------------------------------------------------------===//
// BigInt ↔ Python int conversion
//===----------------------------------------------------------------------===//

// Convert BigInt/SignedBigInt to Python int.
template <typename T>
PyObject* BigIntToPyLong(const T& value) {
  constexpr size_t N = BigIntLimbCount<T>::value;
  constexpr size_t kByteLen = N * 8;

  if constexpr (IsSignedBigIntType<T>) {
    // For signed types, check sign and negate if needed.
    if (value.IsNegative()) {
      auto abs_val = -static_cast<const BigInt<N>&>(value);
      auto bytes = abs_val.ToBytesLE();
      PyObject* pos = _PyLong_FromByteArray(bytes.data(), kByteLen,
                                            /*little_endian=*/true,
                                            /*is_signed=*/false);
      if (!pos) return nullptr;
      PyObject* result = PyNumber_Negative(pos);
      Py_DECREF(pos);
      return result;
    }
  }

  auto bytes = value.ToBytesLE();
  return _PyLong_FromByteArray(bytes.data(), kByteLen,
                               /*little_endian=*/true,
                               /*is_signed=*/false);
}

// Convert Python int to BigInt/SignedBigInt. Returns true on success.
template <typename T>
bool PyLongToBigInt(PyObject* obj, T* output) {
  if (!PyLong_Check(obj)) return false;

  constexpr size_t N = BigIntLimbCount<T>::value;
  constexpr size_t kByteLen = N * 8;
  constexpr size_t kBitLen = N * 64;

  int sign = _PyLong_Sign(obj);
  if (sign == 0) {
    *output = T(0);
    return true;
  }

  size_t bits = _PyLong_NumBits(obj);
  if (bits == static_cast<size_t>(-1)) return false;

  if constexpr (IsSignedBigIntType<T>) {
    // Positive: max is 2^(kBitLen-1) - 1, needs at most kBitLen-1 bits.
    // Negative: min is -2^(kBitLen-1), abs needs exactly kBitLen bits.
    // Values like -(2^(kBitLen-1) + 1) also need kBitLen bits but are
    // out of range. We catch that after byte conversion by verifying the
    // negated result is actually negative (two's complement).
    size_t max_bits = (sign < 0) ? kBitLen : kBitLen - 1;
    if (bits > max_bits) {
      PyErr_SetString(PyExc_OverflowError,
                      "value out of range for signed big integer type");
      return false;
    }
  } else {
    if (sign < 0) {
      PyErr_SetString(PyExc_OverflowError,
                      "negative value for unsigned big integer type");
      return false;
    }
    if (bits > kBitLen) {
      PyErr_SetString(PyExc_OverflowError,
                      "value out of range for unsigned big integer type");
      return false;
    }
  }

  // For negative signed values, convert the absolute value then negate.
  PyObject* to_convert = obj;
  Safe_PyObjectPtr abs_obj;
  if (sign < 0) {
    abs_obj = make_safe(PyNumber_Negative(obj));
    if (!abs_obj) return false;
    to_convert = abs_obj.get();
  }

  std::array<uint8_t, kByteLen> bytes = {};
#if PY_VERSION_HEX >= 0x030D0000
  int ret = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(to_convert),
                                bytes.data(), kByteLen,
                                /*little_endian=*/true,
                                /*is_signed=*/false,
                                /*with_exceptions=*/true);
#else
  int ret = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(to_convert),
                                bytes.data(), kByteLen,
                                /*little_endian=*/true,
                                /*is_signed=*/false);
#endif
  if (ret == -1) return false;

  BigInt<N> result =
      BigInt<N>::FromBytesLE(absl::Span<uint8_t>(bytes.data(), kByteLen));

  if (sign < 0) {
    T negated(-result);
    if constexpr (IsSignedBigIntType<T>) {
      // Verify the negation didn't wrap: the result must be negative
      // (or zero, which can't happen since sign < 0 implies non-zero).
      // e.g. -(2^255 + 1) wraps to a positive value in 256-bit two's
      // complement.
      if (negated.IsNonNegative()) {
        PyErr_SetString(PyExc_OverflowError,
                        "value out of range for signed big integer type");
        return false;
      }
    }
    *output = negated;
  } else {
    *output = T(result);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Python type methods
//===----------------------------------------------------------------------===//

template <typename T>
PyObject* PyBigIntN_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
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
  if (PyBigIntN_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (PyLong_Check(arg)) {
    if (!PyLongToBigInt(arg, &value)) return nullptr;
    return PyBigIntN_FromValue(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != TypeDescriptor<T>::Dtype()) {
      return PyArray_Cast(arr, TypeDescriptor<T>::Dtype());
    } else {
      Py_INCREF(arg);
      return arg;
    }
  } else if (PyUnicode_Check(arg) || PyBytes_Check(arg)) {
    PyObject* f = PyLong_FromUnicodeObject(arg, /*base=*/0);
    if (PyErr_Occurred()) return nullptr;
    if (PyLongToBigInt(f, &value)) {
      Py_DECREF(f);
      return PyBigIntN_FromValue(value).release();
    }
    Py_DECREF(f);
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

template <typename T>
PyObject* PyBigIntN_nb_int(PyObject* self) {
  T x = PyBigIntN_Value_Unchecked<T>(self);
  return BigIntToPyLong(x);
}

template <typename T>
PyObject* PyBigIntN_nb_negative(PyObject* self) {
  T x = PyBigIntN_Value_Unchecked<T>(self);
  return PyBigIntN_FromValue(-x).release();
}

template <typename T>
PyObject* PyBigIntN_nb_positive(PyObject* self) {
  T x = PyBigIntN_Value_Unchecked<T>(self);
  return PyBigIntN_FromValue(x).release();
}

template <typename T>
PyObject* PyBigIntN_nb_add(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    return PyBigIntN_FromValue(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

template <typename T>
PyObject* PyBigIntN_nb_subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    return PyBigIntN_FromValue(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

template <typename T>
PyObject* PyBigIntN_nb_multiply(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    return PyBigIntN_FromValue(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

template <typename T>
PyObject* PyBigIntN_nb_remainder(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    if (y == T(0)) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    return PyBigIntN_FromValue(x % y).release();
  }
  return PyArray_Type.tp_as_number->nb_remainder(a, b);
}

template <typename T>
PyObject* PyBigIntN_nb_floor_divide(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    if (y == T(0)) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    return PyBigIntN_FromValue(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_floor_divide(a, b);
}

template <typename T>
PyObject* PyBigIntN_nb_and(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    return PyBigIntN_FromValue(x & y).release();
  }
  Py_RETURN_NOTIMPLEMENTED;
}

template <typename T>
PyObject* PyBigIntN_nb_or(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    return PyBigIntN_FromValue(x | y).release();
  }
  Py_RETURN_NOTIMPLEMENTED;
}

template <typename T>
PyObject* PyBigIntN_nb_xor(PyObject* a, PyObject* b) {
  T x, y;
  if (PyBigIntN_Value(a, &x) && PyBigIntN_Value(b, &y)) {
    return PyBigIntN_FromValue(x ^ y).release();
  }
  Py_RETURN_NOTIMPLEMENTED;
}

template <typename T>
PyObject* PyBigIntN_nb_lshift(PyObject* a, PyObject* b) {
  T x;
  if (!PyBigIntN_Value(a, &x)) Py_RETURN_NOTIMPLEMENTED;

  long shift = PyLong_AsLong(b);
  if (shift == -1 && PyErr_Occurred()) return nullptr;
  if (shift < 0) {
    PyErr_SetString(PyExc_ValueError, "negative shift count");
    return nullptr;
  }
  return PyBigIntN_FromValue(x << static_cast<uint64_t>(shift)).release();
}

template <typename T>
PyObject* PyBigIntN_nb_rshift(PyObject* a, PyObject* b) {
  T x;
  if (!PyBigIntN_Value(a, &x)) Py_RETURN_NOTIMPLEMENTED;

  long shift = PyLong_AsLong(b);
  if (shift == -1 && PyErr_Occurred()) return nullptr;
  if (shift < 0) {
    PyErr_SetString(PyExc_ValueError, "negative shift count");
    return nullptr;
  }
  return PyBigIntN_FromValue(x >> static_cast<uint64_t>(shift)).release();
}

template <typename T>
PyObject* PyBigIntN_Repr(PyObject* self) {
  T x = PyBigIntN_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

template <typename T>
PyObject* PyBigIntN_Str(PyObject* self) {
  T x = PyBigIntN_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

template <typename T>
Py_hash_t PyBigIntN_Hash(PyObject* self) {
  T x = PyBigIntN_Value_Unchecked<T>(self);
  // Combine limbs into a hash using FNV-like mixing.
  constexpr size_t N = BigIntLimbCount<T>::value;
  Py_hash_t h = 0;
  for (size_t i = 0; i < N; ++i) {
    h ^=
        static_cast<Py_hash_t>(x.limbs()[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
  }
  return h == -1 ? -2 : h;
}

template <typename T>
PyObject* PyBigIntN_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!PyBigIntN_Value(a, &x) || !PyBigIntN_Value(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "Invalid op type");
      return nullptr;
  }
  PyArrayScalar_RETURN_BOOL_FROM_LONG(result);
}

//===----------------------------------------------------------------------===//
// Type slots
//===----------------------------------------------------------------------===//

template <typename T>
PyType_Slot BigIntNTypeDescriptor<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyBigIntN_tp_new<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyBigIntN_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyBigIntN_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyBigIntN_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyBigIntN_RichCompare<T>)},
    {Py_nb_add, reinterpret_cast<void*>(PyBigIntN_nb_add<T>)},
    {Py_nb_subtract, reinterpret_cast<void*>(PyBigIntN_nb_subtract<T>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyBigIntN_nb_multiply<T>)},
    {Py_nb_remainder, reinterpret_cast<void*>(PyBigIntN_nb_remainder<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyBigIntN_nb_negative<T>)},
    {Py_nb_positive, reinterpret_cast<void*>(PyBigIntN_nb_positive<T>)},
    {Py_nb_int, reinterpret_cast<void*>(PyBigIntN_nb_int<T>)},
    {Py_nb_floor_divide, reinterpret_cast<void*>(PyBigIntN_nb_floor_divide<T>)},
    {Py_nb_and, reinterpret_cast<void*>(PyBigIntN_nb_and<T>)},
    {Py_nb_or, reinterpret_cast<void*>(PyBigIntN_nb_or<T>)},
    {Py_nb_xor, reinterpret_cast<void*>(PyBigIntN_nb_xor<T>)},
    {Py_nb_lshift, reinterpret_cast<void*>(PyBigIntN_nb_lshift<T>)},
    {Py_nb_rshift, reinterpret_cast<void*>(PyBigIntN_nb_rshift<T>)},
    {0, nullptr},
};

template <typename T>
PyType_Spec BigIntNTypeDescriptor<T>::type_spec = {
    .name = TypeDescriptor<T>::kQualifiedTypeName,
    .basicsize = static_cast<int>(sizeof(PyBigIntN<T>)),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = BigIntNTypeDescriptor<T>::type_slots,
};

//===----------------------------------------------------------------------===//
// NumPy array functions
//===----------------------------------------------------------------------===//

template <typename T>
PyArray_ArrFuncs BigIntNTypeDescriptor<T>::arr_funcs;

template <typename T>
PyArray_DescrProto GetBigIntNDescrProto() {
  return {
      PyObject_HEAD_INIT(nullptr).typeobj = nullptr,
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
      .f = &BigIntNTypeDescriptor<T>::arr_funcs,
      .metadata = nullptr,
      .c_metadata = nullptr,
      .hash = -1,
  };
}

template <typename T>
PyObject* NPyBigIntN_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return BigIntToPyLong(x);
}

template <typename T>
int NPyBigIntN_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (PyBigIntN_Check<T>(item)) {
    x = PyBigIntN_Value_Unchecked<T>(item);
  } else if (PyLong_Check(item)) {
    if (!PyLongToBigInt(item, &x)) return -1;
  } else {
    PyErr_Format(PyExc_TypeError, "expected integer, got %s",
                 Py_TYPE(item)->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

template <typename T>
void NPyBigIntN_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
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
}

template <typename T>
void NPyBigIntN_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (src) {
    memcpy(dst, src, sizeof(T));
  }
}

template <typename T>
npy_bool NPyBigIntN_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return !x.IsZero();
}

template <typename T>
int NPyBigIntN_CompareFunc(const void* v1, const void* v2, void* arr) {
  T b1 = *reinterpret_cast<const T*>(v1);
  T b2 = *reinterpret_cast<const T*>(v2);
  if (b1 < b2) return -1;
  if (b1 > b2) return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// Casts between BigIntN and standard numpy integer types
//===----------------------------------------------------------------------===//

// Cast from standard numpy type to BigIntN.
template <typename From, typename To>
void NPyBigIntN_CastFromStd(void* from_void, void* to_void, npy_intp n,
                            void* fromarr, void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    if constexpr (std::is_signed_v<From>) {
      if constexpr (IsSignedBigIntType<To>) {
        to[i] = To(static_cast<int64_t>(from[i]));
      } else {
        to[i] = To(static_cast<uint64_t>(from[i]));
      }
    } else {
      to[i] = To(static_cast<uint64_t>(from[i]));
    }
  }
}

// Cast from BigIntN to standard numpy type.
template <typename From, typename To>
void NPyBigIntN_CastToStd(void* from_void, void* to_void, npy_intp n,
                          void* fromarr, void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(static_cast<uint64_t>(from[i]));
  }
}

template <typename T>
bool RegisterBigIntNCasts() {
  // Register casts from standard types to BigIntN.
  auto register_from = [](int numpy_type, auto cast_fn) -> bool {
    PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
    return PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                                    cast_fn) >= 0;
  };

  // Register casts from BigIntN to standard types.
  auto register_to = [](int numpy_type, auto cast_fn) -> bool {
    return PyArray_RegisterCastFunc(BigIntNTypeDescriptor<T>::npy_descr,
                                    numpy_type, cast_fn) >= 0;
  };

  // From standard integers → BigIntN
  if (!register_from(NPY_INT8, NPyBigIntN_CastFromStd<int8_t, T>)) return false;
  if (!register_from(NPY_INT16, NPyBigIntN_CastFromStd<int16_t, T>))
    return false;
  if (!register_from(NPY_INT32, NPyBigIntN_CastFromStd<int32_t, T>))
    return false;
  if (!register_from(NPY_INT64, NPyBigIntN_CastFromStd<int64_t, T>))
    return false;
  if (!register_from(NPY_UINT8, NPyBigIntN_CastFromStd<uint8_t, T>))
    return false;
  if (!register_from(NPY_UINT16, NPyBigIntN_CastFromStd<uint16_t, T>))
    return false;
  if (!register_from(NPY_UINT32, NPyBigIntN_CastFromStd<uint32_t, T>))
    return false;
  if (!register_from(NPY_UINT64, NPyBigIntN_CastFromStd<uint64_t, T>))
    return false;

  // From BigIntN → standard integers
  if (!register_to(NPY_INT64, NPyBigIntN_CastToStd<T, int64_t>)) return false;
  if (!register_to(NPY_UINT64, NPyBigIntN_CastToStd<T, uint64_t>)) return false;

  // Safe casts to BigIntN from small types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL),
                              TypeDescriptor<T>::Dtype(), NPY_NOSCALAR) < 0) {
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// UFuncs
//===----------------------------------------------------------------------===//

template <typename T>
bool RegisterBigIntNUFuncs(PyObject* numpy) {
  bool ok = RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(numpy, "add") &&
            RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(numpy,
                                                                  "subtract") &&
            RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(numpy,
                                                                  "multiply") &&
            RegisterUFunc<UFunc<ufuncs::FloorDivide<T>, T, T, T>, T>(
                numpy, "floor_divide") &&
            RegisterUFunc<UFunc<ufuncs::Remainder<T>, T, T, T>, T>(
                numpy, "remainder") &&
            RegisterUFunc<UFunc<ufuncs::BitwiseAnd<T>, T, T, T>, T>(
                numpy, "bitwise_and") &&
            RegisterUFunc<UFunc<ufuncs::BitwiseOr<T>, T, T, T>, T>(
                numpy, "bitwise_or") &&
            RegisterUFunc<UFunc<ufuncs::BitwiseXor<T>, T, T, T>, T>(
                numpy, "bitwise_xor") &&
            RegisterUFunc<UFunc<ufuncs::LeftShift<T>, T, T, T>, T>(
                numpy, "left_shift") &&
            RegisterUFunc<UFunc<ufuncs::RightShift<T>, T, T, T>, T>(
                numpy, "right_shift");
  return ok;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

template <typename T>
bool RegisterBigIntNDtype(PyObject* numpy) {
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type = PyType_FromSpecWithBases(
      &BigIntNTypeDescriptor<T>::type_spec, bases.get());
  if (!type) return false;
  TypeDescriptor<T>::type_ptr = type;

  Safe_PyObjectPtr module = make_safe(PyUnicode_FromString("zk_dtypes"));
  if (!module) return false;
  if (PyObject_SetAttrString(TypeDescriptor<T>::type_ptr, "__module__",
                             module.get()) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_ArrFuncs& arr_funcs = BigIntNTypeDescriptor<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyBigIntN_GetItem<T>;
  arr_funcs.setitem = NPyBigIntN_SetItem<T>;
  arr_funcs.copyswapn = NPyBigIntN_CopySwapN<T>;
  arr_funcs.copyswap = NPyBigIntN_CopySwap<T>;
  arr_funcs.nonzero = NPyBigIntN_NonZero<T>;
  arr_funcs.compare = NPyBigIntN_CompareFunc<T>;

  PyArray_DescrProto& descr_proto = BigIntNTypeDescriptor<T>::npy_descr_proto;
  descr_proto = GetBigIntNDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  TypeDescriptor<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (TypeDescriptor<T>::npy_type < 0) return false;

  BigIntNTypeDescriptor<T>::npy_descr =
      PyArray_DescrFromType(TypeDescriptor<T>::npy_type);

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy, "sctypeDict"));
  if (!typeDict_obj) return false;
  if (PyDict_SetItemString(typeDict_obj.get(), TypeDescriptor<T>::kTypeName,
                           TypeDescriptor<T>::type_ptr) < 0) {
    return false;
  }

  if (PyObject_SetAttrString(TypeDescriptor<T>::type_ptr, "dtype",
                             reinterpret_cast<PyObject*>(
                                 BigIntNTypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  return RegisterBigIntNCasts<T>() && RegisterBigIntNUFuncs<T>(numpy);
}

template <typename... Types>
bool RegisterBigIntNDtypes(PyObject* numpy) {
  return (RegisterBigIntNDtype<Types>(numpy) && ...);
}

}  // namespace zk_dtypes

#if NPY_ABI_VERSION < 0x02000000
#undef PyArray_DescrProto
#endif

#endif  // ZK_DTYPES__SRC_BIGINT_NUMPY_H_
