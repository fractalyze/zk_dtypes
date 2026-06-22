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

#ifndef ZK_DTYPES__SRC_EC_POINT_DTYPE_H_
#define ZK_DTYPES__SRC_EC_POINT_DTYPE_H_

#include <Python.h>

namespace zk_dtypes {

// Registers the parametric `EcPointDType` (NEP-42) and installs the
// `ec_point_descr(modulus, base_width_bits, is_montgomery[, r, rinv])` factory
// on `module`. This first cut covers short-Weierstrass G1 points (a=0) in the
// Jacobian representation over a prime coordinate field, with the group law
// (add / subtract / negate) registered on numpy. The curve, coordinate
// representation, and field live on each descriptor instance. Returns false
// (with a Python error set) on failure. Must run after numpy is imported.
bool RegisterEcPointDType(PyObject* numpy, PyObject* module);

}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_EC_POINT_DTYPE_H_
