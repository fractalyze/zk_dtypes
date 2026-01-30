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

#ifndef ZK_DTYPES__SRC_FIELD_NUMPY_H_
#define ZK_DTYPES__SRC_FIELD_NUMPY_H_

#include <array>
#include <type_traits>

// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

#include "absl/base/optimization.h"
#include "numpy/ndarraytypes.h"

#include "zk_dtypes/_src/int_caster.h"
#include "zk_dtypes/_src/ufuncs.h"
#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/field/prime_field.h"

namespace zk_dtypes {

template <typename T>
struct FieldTypeDescriptor {
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
int FieldTypeDescriptor<T>::npy_type = NPY_NOTYPE;
template <typename T>
PyObject* FieldTypeDescriptor<T>::type_ptr = nullptr;
template <typename T>
PyArray_DescrProto FieldTypeDescriptor<T>::npy_descr_proto;
template <typename T>
PyArray_Descr* FieldTypeDescriptor<T>::npy_descr = nullptr;

// Representation of a Python field object.
template <typename T>
struct PyField {
  PyObject_HEAD;  // Python object header
  T value;
};

// Returns true if 'object' is a PyField.
template <typename T>
bool PyField_Check(PyObject* object) {
  return PyObject_IsInstance(object, TypeDescriptor<T>::type_ptr);
}

// Extracts the value of a PyField object.
template <typename T>
T PyField_Value_Unchecked(PyObject* object) {
  return reinterpret_cast<PyField<T>*>(object)->value;
}

template <typename T>
bool PyField_Value(PyObject* arg, T* output) {
  if (PyField_Check<T>(arg)) {
    *output = PyField_Value_Unchecked<T>(arg);
    return true;
  }
  return false;
}

// Constructs a PyField object from PyField<T>::T.
template <typename T>
Safe_PyObjectPtr PyField_FromValue(T x) {
  PyTypeObject* type =
      reinterpret_cast<PyTypeObject*>(TypeDescriptor<T>::type_ptr);
  Safe_PyObjectPtr ref = make_safe(type->tp_alloc(type, 0));
  PyField<T>* p = reinterpret_cast<PyField<T>*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Traits for FieldIntCaster - provides type-specific overflow checking logic.
template <typename T, typename Enable = void>
struct FieldIntCasterTraits;

template <typename T>
struct FieldIntCasterTraits<T, std::enable_if_t<IsPrimeField<T>>> {
  static constexpr size_t kStorageBits = T::Config::kStorageBits;
  static constexpr size_t N = (kStorageBits + 63) / 64;
  static constexpr const char* kOverflowError =
      "out of range value cannot be converted to field";

  static bool IsSmallOverflow(int64_t v) {
    if (v < 0) {
      return static_cast<uint64_t>(-v) >= T::Config::kModulus;
    }
    return static_cast<uint64_t>(v) >= T::Config::kModulus;
  }

  static bool IsSmallOverflow(uint64_t v) { return v >= T::Config::kModulus; }

  static bool IsBigOverflow(const BigInt<N>& value) {
    return value >= T::Config::kModulus;
  }

  static T CastBigInt(const BigInt<N>& value, int sign) {
    if (sign == -1) {
      return -T(value);
    }
    return T(value);
  }
};

template <typename T>
struct FieldIntCasterTraits<T, std::enable_if_t<IsBinaryField<T>>> {
  static constexpr size_t kStorageBits = T::kStorageBits;
  static constexpr size_t N = T::kLimbNums;
  static constexpr const char* kOverflowError =
      "out of range value cannot be converted to binary field";

  static bool IsSmallOverflow(int64_t v) {
    uint64_t mask = T::Config::kValueMask;
    if (v < 0) {
      return static_cast<uint64_t>(-v) > mask;
    }
    return static_cast<uint64_t>(v) > mask;
  }

  static bool IsSmallOverflow(uint64_t v) { return v > T::Config::kValueMask; }

  static bool IsBigOverflow(const BigInt<N>& /*value*/) {
    // 128-bit binary field uses all 128 bits, no overflow possible
    return false;
  }

  static T CastBigInt(const BigInt<N>& value, int /*sign*/) {
    // Binary fields ignore sign
    return T(value);
  }
};

// Unified int caster for prime fields and binary fields.
template <typename T>
class FieldIntCaster
    : public std::conditional_t<(FieldIntCasterTraits<T>::kStorageBits <= 64),
                                IntCaster, BigIntCaster<FieldIntCaster<T>>> {
  using Traits = FieldIntCasterTraits<T>;

 public:
  static constexpr size_t N = Traits::N;

  T value() const { return value_; }

  // IntCaster methods.
  bool Cast(int64_t v) override {
    value_ = T(v);
    return true;
  }

  bool Cast(uint64_t v) override {
    value_ = T(v);
    return true;
  }

  bool IsOverflow(int64_t v) const override {
    if constexpr (Traits::kStorageBits <= 64) {
      return Traits::IsSmallOverflow(v);
    }
    return false;
  }

  bool IsOverflow(uint64_t v) const override {
    if constexpr (Traits::kStorageBits <= 64) {
      return Traits::IsSmallOverflow(v);
    }
    return false;
  }

  void SetOverflowError() const override {
    PyErr_SetString(PyExc_OverflowError, Traits::kOverflowError);
  }

  // BigIntCaster methods.
  bool Cast(const BigInt<N>& value, int sign) {
    if constexpr (Traits::kStorageBits > 64) {
      value_ = Traits::CastBigInt(value, sign);
    } else {
      ABSL_UNREACHABLE();
    }
    return true;
  }

  bool IsOverflow(const BigInt<N>& value) const {
    if constexpr (Traits::kStorageBits > 64) {
      return Traits::IsBigOverflow(value);
    } else {
      ABSL_UNREACHABLE();
    }
    return false;
  }

 private:
  T value_;
};

// Converts a Python object to a field value (prime or binary).
// Returns true on success, returns false and reports a Python error on failure.
template <typename T>
bool CastToSmallField(PyObject* arg, T* output) {
  FieldIntCaster<T> caster;
  if constexpr (FieldIntCasterTraits<T>::kStorageBits <= 64) {
    if (!caster.CastInt(arg)) return false;
  } else {
    if (!caster.CastBigInt(arg)) return false;
  }
  *output = caster.value();
  return true;
}

template <typename T>
bool CastToField(PyObject* arg, T* output) {
  // Prime fields and binary fields use unified FieldIntCaster
  if constexpr (IsBinaryField<T> || T::ExtensionDegree() == 1) {
    return CastToSmallField<T>(arg, output);
  } else {
    using BaseField = typename T::BaseField;
    // Use F::N (degree over immediate base field) for tower extension support
    constexpr size_t kDegree = T::N;

    if (PyTuple_Check(arg)) {
      Py_ssize_t size = PyTuple_Size(arg);
      if (size != static_cast<Py_ssize_t>(kDegree)) {
        PyErr_Format(PyExc_TypeError,
                     "expected %zu elements for extension field, got %zd",
                     kDegree, size);
        return false;
      }

      std::array<BaseField, kDegree> values;
      for (size_t i = 0; i < kDegree; ++i) {
        PyObject* arg_i = PyTuple_GetItem(arg, i);  // borrowed ref
        if (!arg_i) {
          return false;
        }
        BaseField base_field;
        if (!CastToField(arg_i, &base_field)) {
          return false;
        }
        values[i] = base_field;
      }
      *output = T(values);
      return true;
    } else {
      BaseField base_field;
      if (!CastToField(arg, &base_field)) {
        return false;
      }
      *output = T(base_field);
      return true;
    }
  }
}

// Constructs a new PyField.
template <typename T>
PyObject* PyField_tp_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
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
  if (PyField_Check<T>(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToField(arg, &value)) {
    return PyField_FromValue(value).release();
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
    if (CastToField(f, &value)) {
      return PyField_FromValue(value).release();
    }
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               Py_TYPE(arg)->tp_name);
  return nullptr;
}

template <typename T>
PyObject* PyField_nb_add(PyObject* a, PyObject* b) {
  T x, y;
  if (PyField_Value(a, &x) && PyField_Value(b, &y)) {
    return PyField_FromValue(x + y).release();
  } else {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for addition",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
}

template <typename T>
PyObject* PyField_nb_subtract(PyObject* a, PyObject* b) {
  T x, y;
  if (PyField_Value(a, &x) && PyField_Value(b, &y)) {
    return PyField_FromValue(x - y).release();
  } else {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for subtraction",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
}

template <typename T>
Safe_PyObjectPtr PyEcPoint_FromValue(T x);

template <typename T>
PyObject* PyField_nb_ec_multiply(T x, PyObject* b) {
  PyErr_SetString(PyExc_TypeError, "invalid argument for multiplication");
  return nullptr;
}

template <typename T, typename U, typename... Args>
PyObject* PyField_nb_ec_multiply(T x, PyObject* b) {
  U y;
  if (PyEcPoint_Value(b, &y)) {
    return PyEcPoint_FromValue(x * y).release();
  }
  return PyField_nb_ec_multiply<T, Args...>(x, b);
}

template <typename T>
PyObject* PyField_nb_multiply(PyObject* a, PyObject* b) {
  T x;
  if (!PyField_Value(a, &x)) {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for multiplication",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
  T y;
  if (PyField_Value(b, &y)) {
    return PyField_FromValue(x * y).release();
  } else {
    if constexpr (std::is_same_v<T, bn254::Fr>) {
      return PyField_nb_ec_multiply<bn254::Fr, bn254::G1AffinePoint,
                                    bn254::G1JacobianPoint, bn254::G1PointXyzz,
                                    bn254::G2AffinePoint,
                                    bn254::G2JacobianPoint, bn254::G2PointXyzz>(
          x, b);
    } else if constexpr (std::is_same_v<T, bn254::FrMont>) {
      return PyField_nb_ec_multiply<
          bn254::FrMont, bn254::G1AffinePointMont, bn254::G1JacobianPointMont,
          bn254::G1PointXyzzMont, bn254::G2AffinePointMont,
          bn254::G2JacobianPointMont, bn254::G2PointXyzzMont>(x, b);
    }
  }

  PyErr_SetString(PyExc_TypeError, "invalid argument for multiplication");
  return nullptr;
}

template <typename T>
PyObject* PyField_nb_true_divide(PyObject* a, PyObject* b) {
  T x, y;
  if (PyField_Value(a, &x) && PyField_Value(b, &y)) {
    if (y.IsZero()) {
      PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
      return nullptr;
    }
    return PyField_FromValue(x / y).release();
  } else {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for division",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
}

template <typename T>
PyObject* PyField_nb_int(PyObject* self) {
  if constexpr (T::ExtensionDegree() > 1 && !IsBinaryField<T>) {
    PyErr_SetString(PyExc_TypeError, "cannot convert extension field to int");
    return nullptr;
  } else {
    T x = PyField_Value_Unchecked<T>(self);
    if constexpr (T::Config::kStorageBits <= 64) {
      if constexpr (T::kUseMontgomery) {
        return PyLong_FromUnsignedLongLong(
            static_cast<unsigned long>(x.MontReduce().value()));
      } else {
        return PyLong_FromUnsignedLongLong(
            static_cast<unsigned long>(x.value()));
      }
    } else {
      std::array<uint8_t, T::kByteWidth> bytes;
      if constexpr (T::kUseMontgomery) {
        bytes = x.MontReduce().value().ToBytesLE();
      } else {
        bytes = x.value().ToBytesLE();
      }
      return _PyLong_FromByteArray(bytes.data(), bytes.size(),
                                   /*little_endian=*/true,
                                   /*is_signed=*/false);
    }
  }
}

template <typename T>
PyObject* PyField_nb_negative(PyObject* a) {
  T x;
  if (PyField_Value(a, &x)) {
    return PyField_FromValue(-x).release();
  } else {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for negation",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
}

class Uint32Caster : public IntCaster {
 public:
  Uint32Caster() = default;

  bool negative() const { return negative_; }
  uint32_t abs_value() const { return abs_value_; }

  // IntCaster methods.
  bool Cast(int64_t v) override {
    if (v < 0) {
      negative_ = true;
      abs_value_ = static_cast<uint32_t>(-v);
    } else {
      abs_value_ = static_cast<uint32_t>(v);
      negative_ = false;
    }
    return true;
  }

  bool Cast(uint64_t v) override {
    abs_value_ = static_cast<uint32_t>(v);
    negative_ = false;
    return true;
  }

  bool IsOverflow(int64_t v) const override {
    if (v < 0) {
      return -v >= static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    } else {
      return v >= static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    }
  }

  bool IsOverflow(uint64_t v) const override {
    return v >= uint64_t{std::numeric_limits<uint32_t>::max()};
  }

  void SetOverflowError() const override {
    PyErr_SetString(PyExc_OverflowError, "value out of range for uint32_t");
  }

 private:
  bool negative_ = false;
  uint32_t abs_value_;
};

template <typename T>
PyObject* PyField_nb_power(PyObject* a, PyObject* b) {
  Uint32Caster caster;
  if (!caster.CastInt(b)) {
    PyErr_SetString(PyExc_TypeError,
                    "expected long long as exponent for power");
    return nullptr;
  }

  T x;
  if (PyField_Value(a, &x)) {
    if (caster.negative()) {
      if (x.IsZero()) {
        PyErr_SetString(PyExc_ZeroDivisionError,
                        "inverse of zero encountered in power");
        return nullptr;
      }
      return PyField_FromValue(x.Inverse().Pow(caster.abs_value())).release();
    } else {
      return PyField_FromValue(x.Pow(caster.abs_value())).release();
    }
  } else {
    PyErr_Format(PyExc_TypeError, "expected %s as argument for power",
                 TypeDescriptor<T>::kTypeName);
    return nullptr;
  }
}

// Implementation of repr() for PyField.
template <typename T>
PyObject* PyField_Repr(PyObject* self) {
  T x = PyField_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

// Implementation of str() for PyField.
template <typename T>
PyObject* PyField_Str(PyObject* self) {
  T x = PyField_Value_Unchecked<T>(self);
  std::string s = x.ToString();
  return PyUnicode_FromString(s.c_str());
}

template <typename T>
uint64_t PyField_Hash_Impl(T x) {
  if constexpr (IsBinaryField<T>) {
    // Binary fields: hash based on the underlying value
    if constexpr (T::kStorageBits <= 64) {
      return static_cast<uint64_t>(x.value());
    } else {
      // For 128-bit binary fields
      uint64_t hash = x[0];
      for (size_t i = 1; i < T::kLimbNums; ++i) {
        hash ^= x[i];
      }
      return hash;
    }
  } else if constexpr (T::ExtensionDegree() == 1) {
    uint64_t hash = static_cast<uint64_t>(x.value());
    if constexpr (T::Config::kStorageBits > 64) {
      for (size_t i = 1; i < T::kLimbNums; ++i) {
        hash ^= x[i];
      }
    }
    return hash;
  } else {
    uint64_t hash = PyField_Hash_Impl(x[0]);
    for (size_t i = 1; i < T::ExtensionDegree(); ++i) {
      hash ^= PyField_Hash_Impl(x[i]);
    }
    return hash;
  }
}

// Hash function for PyField.
template <typename T>
Py_hash_t PyField_Hash(PyObject* self) {
  T x = PyField_Value_Unchecked<T>(self);
  uint64_t hash = PyField_Hash_Impl(x);
  // Hash functions must not return -1.
  return hash == -1 ? static_cast<Py_hash_t>(-2) : static_cast<Py_hash_t>(hash);
}

// Comparisons on PyFields.
template <typename T>
PyObject* PyField_RichCompare(PyObject* a, PyObject* b, int op) {
  T x, y;
  if (!PyField_Value(a, &x) || !PyField_Value(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_LT:
      if constexpr (IsComparable<T>) {
        result = x < y;
      } else {
        PyErr_SetString(
            PyExc_TypeError,
            "ordering comparison not supported for extension field");
        return nullptr;
      }
      break;
    case Py_LE:
      if constexpr (IsComparable<T>) {
        result = x <= y;
      } else {
        PyErr_SetString(
            PyExc_TypeError,
            "ordering comparison not supported for extension field");
        return nullptr;
      }
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      if constexpr (IsComparable<T>) {
        result = x > y;
      } else {
        PyErr_SetString(
            PyExc_TypeError,
            "ordering comparison not supported for extension field");
        return nullptr;
      }
      break;
    case Py_GE:
      if constexpr (IsComparable<T>) {
        result = x >= y;
      } else {
        PyErr_SetString(
            PyExc_TypeError,
            "ordering comparison not supported for extension field");
        return nullptr;
      }
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "Invalid op type");
      return nullptr;
  }
  PyArrayScalar_RETURN_BOOL_FROM_LONG(result);
}

// Helper to convert a field element to its raw Python representation.
// Works for any field type (prime or extension) without requiring
// TypeDescriptor.
template <typename F>
PyObject* FieldToRawPyObject(const F& f) {
  if constexpr (F::ExtensionDegree() > 1 && !IsBinaryField<F>) {
    // Extension field: return tuple of raw values
    // Use F::N (degree over immediate base field) for tower extension support
    constexpr size_t kDegree = F::N;
    PyObject* tuple = PyTuple_New(kDegree);
    if (!tuple) return nullptr;
    for (size_t i = 0; i < kDegree; ++i) {
      PyObject* item = FieldToRawPyObject(f[i]);
      if (!item) {
        Py_DECREF(tuple);
        return nullptr;
      }
      PyTuple_SET_ITEM(tuple, i, item);
    }
    return tuple;
  } else {
    // Prime or binary field: return raw value as int
    if constexpr (F::Config::kStorageBits <= 64) {
      return PyLong_FromUnsignedLongLong(
          static_cast<unsigned long long>(f.value()));
    } else {
      std::array<uint8_t, F::kByteWidth> bytes = f.value().ToBytesLE();
      return _PyLong_FromByteArray(bytes.data(), bytes.size(),
                                   /*little_endian=*/true,
                                   /*is_signed=*/false);
    }
  }
}

// Implementation of 'raw' property getter for PyField.
// Returns the raw internal representation without Montgomery reduction.
template <typename T>
PyObject* PyField_GetRaw(PyObject* self, void* closure) {
  T x = PyField_Value_Unchecked<T>(self);
  return FieldToRawPyObject(x);
}

// Helper to parse a raw field value from Python object.
// Works for any field type (prime or extension) without requiring
// TypeDescriptor.
template <typename F>
bool ParseRawFieldFromPyObject(PyObject* arg, F* output) {
  if constexpr (F::ExtensionDegree() > 1 && !IsBinaryField<F>) {
    // Extension field: expect a tuple
    // Use F::N (degree over immediate base field) for tower extension support
    constexpr size_t kDegree = F::N;
    using BaseField = typename F::BaseField;

    if (PyTuple_Check(arg)) {
      Py_ssize_t size = PyTuple_Size(arg);
      if (size != static_cast<Py_ssize_t>(kDegree)) {
        PyErr_Format(PyExc_TypeError,
                     "expected %zu elements for extension field, got %zd",
                     kDegree, size);
        return false;
      }

      std::array<BaseField, kDegree> values;
      for (size_t i = 0; i < kDegree; ++i) {
        if (!ParseRawFieldFromPyObject(PyTuple_GetItem(arg, i), &values[i])) {
          return false;
        }
      }
      *output = F(values);
      return true;
    } else {
      // Single value: treat as first coefficient (for field types only)
      BaseField base_field;
      if (!ParseRawFieldFromPyObject(arg, &base_field)) {
        return false;
      }
      std::array<BaseField, kDegree> values = {};
      values[0] = base_field;
      *output = F(values);
      return true;
    }
  } else {
    // Prime or binary field: parse as integer and use FromUnchecked
    if (!PyLong_Check(arg)) {
      PyErr_Format(PyExc_TypeError, "expected int, got %s",
                   Py_TYPE(arg)->tp_name);
      return false;
    }

    if (_PyLong_Sign(arg) < 0) {
      PyErr_SetString(PyExc_ValueError,
                      "raw value for field construction cannot be negative");
      return false;
    }

    if constexpr (F::Config::kStorageBits <= 64) {
      unsigned long long val = PyLong_AsUnsignedLongLong(arg);
      if (PyErr_Occurred()) {
        // Error is set by PyLong_AsUnsignedLongLong on overflow.
        return false;
      }
      *output = F::FromUnchecked(static_cast<typename F::UnderlyingType>(val));
      return true;
    } else {
      constexpr size_t N = F::N;
      if (_PyLong_NumBits(arg) > F::Config::kStorageBits) {
        PyErr_SetString(PyExc_OverflowError,
                        "value too large for raw field construction");
        return false;
      }
      std::array<uint8_t, N * 8> bytes = {};
#if PY_VERSION_HEX >= 0x030D0000  // Python 3.13 or later
      int ret = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(arg),
                                    bytes.data(), bytes.size(),
                                    /*little_endian=*/true,
                                    /*is_signed=*/false,
                                    /*with_exceptions=*/true);
#else
      int ret = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(arg),
                                    bytes.data(), bytes.size(),
                                    /*little_endian=*/true,
                                    /*is_signed=*/false);
#endif
      if (ret == -1) {
        // Error already set by _PyLong_AsByteArray.
        return false;
      }
      BigInt<N> value = BigInt<N>::FromBytesLE(bytes);
      *output = F::FromUnchecked(value);
      return true;
    }
  }
}

// Implementation of 'from_raw' classmethod for PyField.
// Constructs a field element from raw internal representation.
template <typename T>
PyObject* PyField_FromRaw(PyObject* cls, PyObject* args) {
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_Format(PyExc_TypeError,
                 "from_raw() takes exactly one argument, got %d", size);
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  T value;
  if (!ParseRawFieldFromPyObject(arg, &value)) {
    if (!PyErr_Occurred()) {
      PyErr_Format(PyExc_TypeError, "expected number or tuple, got %s",
                   Py_TYPE(arg)->tp_name);
    }
    return nullptr;
  }
  return PyField_FromValue(value).release();
}

template <typename T>
PyGetSetDef FieldTypeDescriptor<T>::getset[] = {
    {"raw", PyField_GetRaw<T>, nullptr, "Raw internal representation", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <typename T>
PyMethodDef FieldTypeDescriptor<T>::methods[] = {
    {"from_raw", PyField_FromRaw<T>, METH_VARARGS | METH_CLASS,
     "Construct from raw internal representation"},
    {nullptr, nullptr, 0, nullptr}};

template <typename T>
PyType_Slot FieldTypeDescriptor<T>::type_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyField_tp_new<T>)},
    {Py_tp_repr, reinterpret_cast<void*>(PyField_Repr<T>)},
    {Py_tp_hash, reinterpret_cast<void*>(PyField_Hash<T>)},
    {Py_tp_str, reinterpret_cast<void*>(PyField_Str<T>)},
    {Py_tp_doc,
     reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
    {Py_tp_richcompare, reinterpret_cast<void*>(PyField_RichCompare<T>)},
    {Py_tp_getset, reinterpret_cast<void*>(FieldTypeDescriptor<T>::getset)},
    {Py_tp_methods, reinterpret_cast<void*>(FieldTypeDescriptor<T>::methods)},
    {Py_nb_add, reinterpret_cast<void*>(PyField_nb_add<T>)},
    {Py_nb_subtract, reinterpret_cast<void*>(PyField_nb_subtract<T>)},
    {Py_nb_multiply, reinterpret_cast<void*>(PyField_nb_multiply<T>)},
    {Py_nb_true_divide, reinterpret_cast<void*>(PyField_nb_true_divide<T>)},
    {Py_nb_int, reinterpret_cast<void*>(PyField_nb_int<T>)},
    {Py_nb_negative, reinterpret_cast<void*>(PyField_nb_negative<T>)},
    {Py_nb_power, reinterpret_cast<void*>(PyField_nb_power<T>)},
    {0, nullptr},
};

template <typename T>
PyType_Spec FieldTypeDescriptor<T>::type_spec = {
    .name = TypeDescriptor<T>::kQualifiedTypeName,
    .basicsize = static_cast<int>(sizeof(PyField<T>)),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = FieldTypeDescriptor<T>::type_slots,
};

// Numpy support
template <typename T>
PyArray_ArrFuncs FieldTypeDescriptor<T>::arr_funcs;

template <typename T>
PyArray_DescrProto GetFieldDescrProto() {
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
      .f = &FieldTypeDescriptor<T>::arr_funcs,
      .metadata = nullptr,
      .c_metadata = nullptr,
      .hash = -1,  // -1 means "not computed yet".
  };
}

// Implementations of NumPy array methods.

template <typename T>
PyObject* NPyField_GetItem(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(T));
  return PyField_FromValue(x).release();
}

template <typename T>
int NPyField_SetItem(PyObject* item, void* data, void* arr) {
  T x;
  if (PyField_Check<T>(item)) {
    x = PyField_Value_Unchecked<T>(item);
  } else if (!CastToField(item, &x)) {
    return -1;
  }
  memcpy(data, &x, sizeof(T));
  return 0;
}

template <typename T>
void NPyField_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
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
void NPyField_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (src) {
    memcpy(dst, src, sizeof(T));
  }
  // Note: No byte swapping needed for 8-bit integer types
}

template <typename T>
npy_bool NPyField_NonZero(void* data, void* arr) {
  T x;
  memcpy(&x, data, sizeof(x));
  return !x.IsZero();
}

template <typename T>
int NPyField_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  T* const buffer = reinterpret_cast<T*>(buffer_raw);
  const T start(buffer[0]);
  const T delta = static_cast<T>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = buffer[i - 1] + delta;
  }
  return 0;
}

template <typename T>
void NPyField_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                      void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  T acc = 0;
  for (npy_intp i = 0; i < n; ++i) {
    T* const b1 = reinterpret_cast<T*>(c1);
    T* const b2 = reinterpret_cast<T*>(c2);
    acc += *b1 * *b2;
    c1 += is1;
    c2 += is2;
  }
  T* out = reinterpret_cast<T*>(op);
  *out = acc;
}

template <typename T>
int NPyField_CompareFunc(const void* v1, const void* v2, void* arr) {
  T b1 = *reinterpret_cast<const T*>(v1);
  T b2 = *reinterpret_cast<const T*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

template <typename T>
int NPyField_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind, void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  // Start with a max_val of T::Zero(), this results in the first iteration
  // preferring bdata[0].
  T max_val = T::Zero();
  for (npy_intp i = 0; i < n; ++i) {
    if (bdata[i] > max_val) {
      max_val = bdata[i];
      *max_ind = i;
    }
  }
  return 0;
}

template <typename T>
int NPyField_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind, void* arr) {
  const T* bdata = reinterpret_cast<const T*>(data);
  T min_val = T::Max();
  // Start with a min_val of T::Max(), this results in the first iteration
  // preferring bdata[0].
  for (npy_intp i = 0; i < n; ++i) {
    if (bdata[i] < min_val) {
      min_val = bdata[i];
      *min_ind = i;
    }
  }
  return 0;
}

template <typename T>
uint64_t NPyField_CastToInt(T value) {
  if constexpr (IsPrimeField<T> || IsBinaryField<T>) {
    if constexpr (T::Config::kStorageBits <= 64) {
      if constexpr (T::Config::kUseMontgomery) {
        return value.MontReduce().value();
      } else {
        return value.value();
      }
    } else {
      BigInt<T::N> v;
      if constexpr (T::Config::kUseMontgomery) {
        v = value.MontReduce().value();
      } else {
        v = value.value();
      }
      for (size_t i = 1; i < T::N; ++i) {
        if (v[i] != 0) {
          PyErr_SetString(PyExc_OverflowError,
                          "Cannot cast field value to 64-bit integer without "
                          "loss of precision");
          return -1;
        }
      }
      return v[0];
    }
  } else {
    static_assert(std::numeric_limits<T>::is_integer);
    return static_cast<uint64_t>(value);
  }
}

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyField_IntegerCast(void* from_void, void* to_void, npy_intp n,
                          void* fromarr, void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<typename TypeDescriptor<To>::T>(
        static_cast<To>(NPyField_CastToInt(from[i])));
    if (PyErr_Occurred()) {
      return;
    }
  }
}

// Registers a cast between 'T' and type 'OtherT'. 'numpy_type'
// is the NumPy type corresponding to 'OtherT'.
template <typename T, typename OtherT>
bool RegisterFieldCast(int numpy_type = TypeDescriptor<OtherT>::Dtype()) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, TypeDescriptor<T>::Dtype(),
                               NPyField_IntegerCast<OtherT, T>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(FieldTypeDescriptor<T>::npy_descr, numpy_type,
                               NPyField_IntegerCast<T, OtherT>) < 0) {
    return false;
  }
  return true;
}

template <typename T>
bool RegisterFieldCasts() {
  if (!RegisterFieldCast<T, bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterFieldCast<T, unsigned char>(NPY_UBYTE)) {
    return false;
  }
  if (!RegisterFieldCast<T, unsigned short>(NPY_USHORT)) {
    return false;
  }
  if (!RegisterFieldCast<T, unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterFieldCast<T, unsigned long>(NPY_ULONG)) {
    return false;
  }
  if (!RegisterFieldCast<T, unsigned long long>(NPY_ULONGLONG)) {
    return false;
  }
  if (!RegisterFieldCast<T, signed char>(NPY_BYTE)) {
    return false;
  }
  if (!RegisterFieldCast<T, short>(NPY_SHORT)) {
    return false;
  }
  if (!RegisterFieldCast<T, int>(NPY_INT)) {
    return false;
  }
  if (!RegisterFieldCast<T, long>(NPY_LONG)) {
    return false;
  }
  if (!RegisterFieldCast<T, long long>(NPY_LONGLONG)) {
    return false;
  }

  constexpr int kUnsignedTypes[] = {NPY_UINT8, NPY_UINT16, NPY_UINT32,
                                    NPY_UINT64};
  constexpr int kBits[] = {8, 16, 32, 64};

  // Safe casts from T to other types
  for (int i = 0; i < std::size(kBits); ++i) {
    if (T::Config::kStorageBits <= kBits[i]) {
      if (PyArray_RegisterCanCast(TypeDescriptor<T>::npy_descr,
                                  kUnsignedTypes[i], NPY_NOSCALAR) < 0) {
        return false;
      }
    }
  }

  // Safe casts to T from other types
  for (int i = 0; i < std::size(kBits); ++i) {
    if (kBits[i] <= T::Config::kStorageBits) {
      if (PyArray_RegisterCanCast(PyArray_DescrFromType(kUnsignedTypes[i]),
                                  TypeDescriptor<T>::Dtype(),
                                  NPY_NOSCALAR) < 0) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool RegisterFieldUFuncs(PyObject* numpy) {
  bool ok =
      RegisterUFunc<UFunc<ufuncs::Add<T>, T, T, T>, T>(numpy, "add") &&
      RegisterUFunc<UFunc<ufuncs::Subtract<T>, T, T, T>, T>(numpy,
                                                            "subtract") &&
      RegisterUFunc<UFunc<ufuncs::Multiply<T>, T, T, T>, T>(numpy,
                                                            "multiply") &&
      RegisterUFunc<UFunc<ufuncs::TrueDivide<T>, T, T, T>, T>(numpy,
                                                              "true_divide") &&
      RegisterUFunc<UFunc<ufuncs::Negative<T>, T, T>, T>(numpy, "negative") &&
      RegisterUFunc<UFunc<ufuncs::Power<T>, T, T, uint32_t>, T>(numpy,
                                                                "power") &&

      // Comparison functions
      RegisterUFunc<UFunc<ufuncs::Eq<T>, bool, T, T>, T>(numpy, "equal") &&
      RegisterUFunc<UFunc<ufuncs::Ne<T>, bool, T, T>, T>(numpy, "not_equal");

  if constexpr (IsComparable<T>) {
    ok = ok &&
         RegisterUFunc<UFunc<ufuncs::Lt<T>, bool, T, T>, T>(numpy, "less") &&
         RegisterUFunc<UFunc<ufuncs::Gt<T>, bool, T, T>, T>(numpy, "greater") &&
         RegisterUFunc<UFunc<ufuncs::Le<T>, bool, T, T>, T>(numpy,
                                                            "less_equal") &&
         RegisterUFunc<UFunc<ufuncs::Ge<T>, bool, T, T>, T>(numpy,
                                                            "greater_equal") &&
         RegisterUFunc<UFunc<ufuncs::Maximum<T>, T, T, T>, T>(numpy,
                                                              "maximum") &&
         RegisterUFunc<UFunc<ufuncs::Minimum<T>, T, T, T>, T>(numpy, "minimum");
  }

  return ok;
}

template <typename T>
bool RegisterFieldDtype(PyObject* numpy) {
  // bases must be a tuple for Python 3.9 and earlier. Change to just pass
  // the base type directly when dropping Python 3.9 support.
  // TODO(jakevdp): it would be better to inherit from PyNumberArrType or
  // PyIntegerArrType, but this breaks some assumptions made by NumPy,
  // because dtype.kind='V' is then interpreted as a 'void' type in some
  // contexts.
  Safe_PyObjectPtr bases(
      PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type)));
  PyObject* type =
      PyType_FromSpecWithBases(&FieldTypeDescriptor<T>::type_spec, bases.get());
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
  PyArray_ArrFuncs& arr_funcs = FieldTypeDescriptor<T>::arr_funcs;
  PyArray_InitArrFuncs(&arr_funcs);
  arr_funcs.getitem = NPyField_GetItem<T>;
  arr_funcs.setitem = NPyField_SetItem<T>;
  arr_funcs.copyswapn = NPyField_CopySwapN<T>;
  arr_funcs.copyswap = NPyField_CopySwap<T>;
  arr_funcs.nonzero = NPyField_NonZero<T>;
  arr_funcs.fill = NPyField_Fill<T>;
  arr_funcs.dotfunc = NPyField_DotFunc<T>;
  // Ordering functions are supported for prime fields and binary fields
  if constexpr (IsComparable<T>) {
    arr_funcs.compare = NPyField_CompareFunc<T>;
    arr_funcs.argmax = NPyField_ArgMaxFunc<T>;
    arr_funcs.argmin = NPyField_ArgMinFunc<T>;
  }

  // This is messy, but that's because the NumPy 2.0 API transition is messy.
  // Before 2.0, NumPy assumes we'll keep the descriptor passed in to
  // RegisterDataType alive, because it stores its pointer.
  // After 2.0, the proto and descriptor types diverge, and NumPy allocates
  // and manages the lifetime of the descriptor itself.
  PyArray_DescrProto& descr_proto = FieldTypeDescriptor<T>::npy_descr_proto;
  descr_proto = GetFieldDescrProto<T>();
  Py_SET_TYPE(&descr_proto, &PyArrayDescr_Type);
  descr_proto.typeobj = reinterpret_cast<PyTypeObject*>(type);

  TypeDescriptor<T>::npy_type = PyArray_RegisterDataType(&descr_proto);
  if (TypeDescriptor<T>::npy_type < 0) {
    return false;
  }
  // TODO(phawkins): We intentionally leak the pointer to the descriptor.
  // Implement a better module destructor to handle this.
  FieldTypeDescriptor<T>::npy_descr =
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
  if (PyObject_SetAttrString(
          TypeDescriptor<T>::type_ptr, "dtype",
          reinterpret_cast<PyObject*>(FieldTypeDescriptor<T>::npy_descr)) < 0) {
    return false;
  }

  // Extension fields don't support integer casts (but binary fields do)
  if constexpr (T::ExtensionDegree() == 1 || IsBinaryField<T>) {
    return RegisterFieldCasts<T>() && RegisterFieldUFuncs<T>(numpy);
  } else {
    return RegisterFieldUFuncs<T>(numpy);
  }
}

}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_FIELD_NUMPY_H_
