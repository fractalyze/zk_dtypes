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

#ifndef ZK_DTYPES__SRC_FIELD_DTYPE_H_
#define ZK_DTYPES__SRC_FIELD_DTYPE_H_

#include <Python.h>

namespace zk_dtypes {

// Registers the parametric `FieldDType` (NEP-42) and installs the
// `field_descr(modulus, degree, non_residue, base_width_bits, is_montgomery
// [, r_mod_p, rinv_mod_p])` factory on `module`. One DType class serves every
// finite field — prime (degree 1) and binomial extension (degree k over the
// prime, X^k = non_residue) alike; the parameters live on each descriptor
// instance, so a user-defined field needs no new C++ type. Returns false (with
// a Python error set) on failure. Must run after numpy is imported.
bool RegisterFieldDType(PyObject* numpy, PyObject* module);

// Seam for EC scalar multiplication (the scalar is a prime FieldDType element).
// `FieldDTypeMetaObject` returns the FieldDType metaclass (a
// `PyArray_DTypeMeta*` as `PyObject*`) for registering a mixed ufunc loop
// against. `PrimeFieldValue` returns the canonical integer value of a degree-1
// (prime) FieldDType element at `data`, given its descriptor (a
// `PyArray_Descr*` as `PyObject*`), or NULL with an error set if the descriptor
// is not a prime FieldDType.
PyObject* FieldDTypeMetaObject();
PyObject* PrimeFieldValue(PyObject* descr, const char* data);

}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_FIELD_DTYPE_H_
