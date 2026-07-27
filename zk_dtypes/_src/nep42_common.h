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

#ifndef ZK_DTYPES__SRC_NEP42_COMMON_H_
#define ZK_DTYPES__SRC_NEP42_COMMON_H_

// NEP-42 registration ceremony shared by the parametric dtypes. Every ufunc
// loop registration is the same dance — fetch the ufunc, fill a
// PyArrayMethod_Spec with one resolve + one strided-loop slot, add, drop the
// ref — differing only in the names, arity, dtype row, and the two function
// pointers. Registering a loop for a new parametric dtype should cost one
// call here, not another copy of the dance.

// clang-format off
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include <Python.h>
#include "zk_dtypes/_src/numpy.h"
#include "numpy/dtype_api.h"
// clang-format on

namespace zk_dtypes {
namespace nep42 {

// Registers `loop` (with `resolve`) on numpy's `ufunc_name` for the given
// dtype row (nin inputs + 1 output). Returns false with a Python error set on
// failure.
inline bool AddUfuncLoop(PyObject* numpy, const char* ufunc_name,
                         const char* spec_name, int nin,
                         PyArray_DTypeMeta** dtypes, void* resolve,
                         void* loop) {
  PyObject* ufunc = PyObject_GetAttrString(numpy, ufunc_name);
  if (ufunc == nullptr) {
    return false;
  }
  PyType_Slot slots[] = {
      {NPY_METH_resolve_descriptors, resolve},
      {NPY_METH_strided_loop, loop},
      {0, nullptr},
  };
  PyArrayMethod_Spec spec = {};
  spec.name = spec_name;
  spec.nin = nin;
  spec.nout = 1;
  spec.casting = NPY_NO_CASTING;
  spec.flags = NPY_METH_REQUIRES_PYAPI;
  spec.dtypes = dtypes;
  spec.slots = slots;
  int rc = PyUFunc_AddLoopFromSpec(ufunc, &spec);
  Py_DECREF(ufunc);
  return rc >= 0;
}

}  // namespace nep42
}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_NEP42_COMMON_H_
