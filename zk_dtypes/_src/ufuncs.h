/* Copyright 2022 The ml_dtypes Authors
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

#ifndef ZK_DTYPES__SRC_UFUNCS_H_
#define ZK_DTYPES__SRC_UFUNCS_H_

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

#include "zk_dtypes/_src/common.h"

// Some versions of MSVC define a "copysign" macro which wreaks havoc.
#if defined(_MSC_VER) && defined(copysign)
#undef copysign
#endif

namespace zk_dtypes {

template <typename Functor, typename OutType, typename... InTypes>
struct UFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InTypes>::Dtype()...,
            TypeDescriptor<OutType>::Dtype()};
  }
  static constexpr int kInputArity = sizeof...(InTypes);

  template <std::size_t... Is>
  static void CallImpl(std::index_sequence<Is...>, char** args,
                       const npy_intp* dimensions, const npy_intp* steps,
                       void* data) {
    std::array<const char*, kInputArity> inputs = {args[Is]...};
    char* o = args[kInputArity];
    for (npy_intp k = 0; k < *dimensions; k++) {
      *reinterpret_cast<OutType*>(o) =
          Functor()(*reinterpret_cast<const InTypes*>(inputs[Is])...);
      ([&]() { inputs[Is] += steps[Is]; }(), ...);
      o += steps[kInputArity];
    }
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    return CallImpl(std::index_sequence_for<InTypes...>(), args, dimensions,
                    steps, data);
  }
};

template <typename UFuncT, typename CustomT>
bool RegisterUFunc(PyObject* numpy, const char* name) {
  std::vector<int> types = UFuncT::Types();
  PyUFuncGenericFunction fn =
      reinterpret_cast<PyUFuncGenericFunction>(UFuncT::Call);
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %lu", name,
                 ufunc->nargs, types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, TypeDescriptor<CustomT>::Dtype(), fn,
                                  const_cast<int*>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

template <typename T>
struct Add {
  T operator()(T a, T b) { return a + b; }
};
template <typename T>
struct Subtract {
  T operator()(T a, T b) { return a - b; }
};
template <typename T>
struct Multiply {
  T operator()(T a, T b) { return a * b; }
};
template <typename T>
struct TrueDivide {
  T operator()(T a, T b) { return a / b; }
};

template <typename T>
struct FloorDivide {
  template <typename U = T,
            std::enable_if_t<TypeDescriptor<U>::is_integral, bool> = true>
  T operator()(T x, T y) {
    if (y == T(0)) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "divide by zero encountered in floor_divide", 1);
      return T(0);
    }
    T v = x / y;
    if (((x > 0) != (y > 0)) && x % y != 0) {
      v = v - T(1);
    }
    return v;
  }
};

template <typename T>
struct Remainder {
  template <typename U = T,
            std::enable_if_t<TypeDescriptor<U>::is_integral, bool> = true>
  T operator()(T x, T y) {
    if (y == 0) {
      PyErr_WarnEx(PyExc_RuntimeWarning,
                   "divide by zero encountered in remainder", 1);
      return T(0);
    }
    T v = x % y;
    if (v != 0 && ((v < 0) != (y < 0))) {
      v = v + y;
    }
    return v;
  }
};

}  // namespace ufuncs
}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_UFUNCS_H_
