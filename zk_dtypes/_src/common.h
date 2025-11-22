/* Copyright 2022 The ml_dtypes Authors.
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

#ifndef ZK_DTYPES__SRC_COMMON_H_
#define ZK_DTYPES__SRC_COMMON_H_

// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

#include <Python.h>

namespace zk_dtypes {

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
inline Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
};

template <>
struct TypeDescriptor<unsigned char> {
  typedef unsigned char T;
  static int Dtype() { return NPY_UBYTE; }
};

template <>
struct TypeDescriptor<unsigned short> {
  typedef unsigned short T;
  static int Dtype() { return NPY_USHORT; }
};

// We register "int", "long", and "long long" types for portability across
// Linux, where "int" and "long" are the same type, and Windows, where "long"
// and "longlong" are the same type.
template <>
struct TypeDescriptor<unsigned int> {
  typedef unsigned int T;
  static int Dtype() { return NPY_UINT; }
};

template <>
struct TypeDescriptor<unsigned long> {
  typedef unsigned long T;
  static int Dtype() { return NPY_ULONG; }
};

template <>
struct TypeDescriptor<unsigned long long> {
  typedef unsigned long long T;
  static int Dtype() { return NPY_ULONGLONG; }
};

template <>
struct TypeDescriptor<signed char> {
  typedef signed char T;
  static int Dtype() { return NPY_BYTE; }
};

template <>
struct TypeDescriptor<short> {
  typedef short T;
  static int Dtype() { return NPY_SHORT; }
};

template <>
struct TypeDescriptor<int> {
  typedef int T;
  static int Dtype() { return NPY_INT; }
};

template <>
struct TypeDescriptor<long> {
  typedef long T;
  static int Dtype() { return NPY_LONG; }
};

template <>
struct TypeDescriptor<long long> {
  typedef long long T;
  static int Dtype() { return NPY_LONGLONG; }
};

template <>
struct TypeDescriptor<bool> {
  typedef unsigned char T;
  static int Dtype() { return NPY_BOOL; }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_COMMON_H_
