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

#ifndef ZK_DTYPES__SRC_PYFIELD_OPS_H_
#define ZK_DTYPES__SRC_PYFIELD_OPS_H_

// CPython-int prime-field primitives shared by the parametric field and EC
// point dtypes: canonical modular arithmetic on Python ints, and the
// canonical<->stored codec for one base-field slot (little-endian bytes,
// optionally Montgomery-scaled through R / R^-1). Both dtypes describe their
// base field with the same five parameters, so the codec takes them as one
// value instead of each file re-implementing the scaling dance.

#include <Python.h>

namespace zk_dtypes {
namespace pyfield {

// _PyLong_AsByteArray grew a `with_exceptions` parameter in Python 3.13;
// every AsByteArray call in the parametric dtypes routes through here so the
// version split lives once.
inline int LongAsBytesLE(PyObject* v, unsigned char* bytes, size_t n,
                         bool is_signed = false) {
#if PY_VERSION_HEX >= 0x030D0000  // Python 3.13 or later
  return _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(v), bytes, n,
                             /*little_endian=*/1, is_signed ? 1 : 0,
                             /*with_exceptions=*/1);
#else
  return _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(v), bytes, n,
                             /*little_endian=*/1, is_signed ? 1 : 0);
#endif
}

// (a + b) mod p on canonical Python ints.
inline PyObject* ModAdd(PyObject* p, PyObject* a, PyObject* b) {
  PyObject* s = PyNumber_Add(a, b);
  if (s == nullptr) return nullptr;
  PyObject* r = PyNumber_Remainder(s, p);
  Py_DECREF(s);
  return r;
}

// (a - b) mod p (CPython remainder is nonnegative for positive modulus).
inline PyObject* ModSub(PyObject* p, PyObject* a, PyObject* b) {
  PyObject* d = PyNumber_Subtract(a, b);
  if (d == nullptr) return nullptr;
  PyObject* r = PyNumber_Remainder(d, p);
  Py_DECREF(d);
  return r;
}

// (a * b) mod p.
inline PyObject* ModMul(PyObject* p, PyObject* a, PyObject* b) {
  PyObject* m = PyNumber_Multiply(a, b);
  if (m == nullptr) return nullptr;
  PyObject* r = PyNumber_Remainder(m, p);
  Py_DECREF(m);
  return r;
}

// (a * k) mod p for a small machine integer k.
inline PyObject* ModMulInt(PyObject* p, PyObject* a, long k) {
  PyObject* kk = PyLong_FromLong(k);
  if (kk == nullptr) return nullptr;
  PyObject* r = ModMul(p, a, kk);
  Py_DECREF(kk);
  return r;
}

// x^-1 mod p via Fermat (x^(p-2) mod p); p must be prime.
inline PyObject* ModInv(PyObject* p, PyObject* x) {
  PyObject* two = PyLong_FromLong(2);
  PyObject* pm2 = two ? PyNumber_Subtract(p, two) : nullptr;
  Py_XDECREF(two);
  if (pm2 == nullptr) return nullptr;
  PyObject* r = PyNumber_Power(x, pm2, p);  // 3-arg pow == modular exponent
  Py_DECREF(pm2);
  return r;
}

// One base-field slot's storage parameters.
struct BaseCodec {
  PyObject* modulus;
  PyObject* r_mod_p;     // Montgomery only; else NULL
  PyObject* rinv_mod_p;  // Montgomery only; else NULL
  int width_bytes;
  bool is_montgomery;
};

// Reads the coefficient stored at `slot` and returns its canonical value.
inline PyObject* Decode(const BaseCodec& c, const char* slot) {
  PyObject* stored = _PyLong_FromByteArray(
      reinterpret_cast<const unsigned char*>(slot), c.width_bytes,
      /*little_endian=*/1, /*is_signed=*/0);
  if (stored == nullptr || !c.is_montgomery) {
    return stored;
  }
  PyObject* scaled = PyNumber_Multiply(stored, c.rinv_mod_p);
  Py_DECREF(stored);
  if (scaled == nullptr) {
    return nullptr;
  }
  PyObject* canonical = PyNumber_Remainder(scaled, c.modulus);
  Py_DECREF(scaled);
  return canonical;
}

// Writes canonical `value` into the coefficient slot; -1 on error.
inline int Encode(const BaseCodec& c, char* slot, PyObject* value) {
  PyObject* rem = PyNumber_Remainder(value, c.modulus);
  if (rem == nullptr) {
    return -1;
  }
  if (c.is_montgomery) {
    PyObject* scaled = PyNumber_Multiply(rem, c.r_mod_p);
    Py_DECREF(rem);
    if (scaled == nullptr) {
      return -1;
    }
    rem = PyNumber_Remainder(scaled, c.modulus);
    Py_DECREF(scaled);
    if (rem == nullptr) {
      return -1;
    }
  }
  int rc =
      LongAsBytesLE(rem, reinterpret_cast<unsigned char*>(slot), c.width_bytes);
  Py_DECREF(rem);
  return rc < 0 ? -1 : 0;
}

}  // namespace pyfield
}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_PYFIELD_OPS_H_
