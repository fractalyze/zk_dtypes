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

// Parametric elliptic-curve point numpy DType (NEP-42). An EC point group is a
// Module, not a Field: the ops are point add / subtract / negate (and, later,
// scalar multiplication) — there is no point*point. This first cut covers
// short-Weierstrass G1 (a=0) Jacobian points over a prime coordinate field.
//
// A point is `num_coords` coordinate-field elements (X, Y, Z for Jacobian),
// stored exactly as the coordinate field stores them (canonical or per-coord
// Montgomery, little-endian). The add/double formulas (EFD add-2007-bl /
// dbl-2009-l) are R-linear, so we decode coordinates to canonical, run the
// group law with prime-field arithmetic, and re-encode — byte-identical to the
// legacy Montgomery formulas because the resulting representative is the same.

// numpy.h must precede every other numpy header (it sets the API symbol) and
// NPY_TARGET_VERSION must precede numpyconfig.h; the associated header pulls in
// only <Python.h>, so it can lead. Keep this order — do not let clang-format
// sort it.
// clang-format off
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "zk_dtypes/_src/ec_point_dtype.h"

#include <cstdint>
#include <cstring>

#include "zk_dtypes/_src/numpy.h"
#include "zk_dtypes/_src/field_dtype.h"
#include "numpy/dtype_api.h"
#include "numpy/ndarraytypes.h"
// clang-format on

namespace zk_dtypes {
namespace {

constexpr int kMaxCoords = 4;

struct EcPointDescr {
  PyArray_Descr base;
  PyObject* modulus;  // owned: coordinate prime field modulus
  PyObject* r_mod_p;  // owned (Montgomery): R = 2^base_width mod p; else NULL
  PyObject* rinv_mod_p;  // owned (Montgomery): R^-1 mod p; else NULL
  uint8_t base_width_bytes;
  uint8_t num_coords;  // 3 for Jacobian (X, Y, Z)
  uint8_t is_montgomery;
};

PyArray_DTypeMeta EcPointDType = {};
PyTypeObject EcPointScalar_Type = {};

EcPointDescr* AsEc(PyArray_Descr* d) {
  return reinterpret_cast<EcPointDescr*>(d);
}

// --- coordinate (prime field) encode / decode ---------------------------

PyObject* DecodeCoord(EcPointDescr* d, const char* slot) {
  PyObject* stored = _PyLong_FromByteArray(
      reinterpret_cast<const unsigned char*>(slot), d->base_width_bytes, 1, 0);
  if (stored == nullptr || !d->is_montgomery) {
    return stored;
  }
  PyObject* scaled = PyNumber_Multiply(stored, d->rinv_mod_p);
  Py_DECREF(stored);
  if (scaled == nullptr) {
    return nullptr;
  }
  PyObject* canonical = PyNumber_Remainder(scaled, d->modulus);
  Py_DECREF(scaled);
  return canonical;
}

int EncodeCoord(EcPointDescr* d, char* slot, PyObject* value) {
  PyObject* rem = PyNumber_Remainder(value, d->modulus);
  if (rem == nullptr) {
    return -1;
  }
  if (d->is_montgomery) {
    PyObject* scaled = PyNumber_Multiply(rem, d->r_mod_p);
    Py_DECREF(rem);
    if (scaled == nullptr) {
      return -1;
    }
    rem = PyNumber_Remainder(scaled, d->modulus);
    Py_DECREF(scaled);
    if (rem == nullptr) {
      return -1;
    }
  }
  int rc = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(rem),
                               reinterpret_cast<unsigned char*>(slot),
                               d->base_width_bytes, 1, 0);
  Py_DECREF(rem);
  return rc < 0 ? -1 : 0;
}

int DecodePoint(EcPointDescr* d, const char* ptr, PyObject** out) {
  for (int i = 0; i < d->num_coords; ++i) {
    out[i] = DecodeCoord(d, ptr + i * d->base_width_bytes);
    if (out[i] == nullptr) {
      for (int j = 0; j < i; ++j) Py_DECREF(out[j]);
      return -1;
    }
  }
  return 0;
}

int EncodePoint(EcPointDescr* d, char* ptr, PyObject* const* coords) {
  for (int i = 0; i < d->num_coords; ++i) {
    if (EncodeCoord(d, ptr + i * d->base_width_bytes, coords[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

// --- prime-field helpers on canonical Python ints -----------------------

PyObject* FMod(PyObject* p, PyObject* x) { return PyNumber_Remainder(x, p); }

PyObject* FAdd(PyObject* p, PyObject* a, PyObject* b) {
  PyObject* s = PyNumber_Add(a, b);
  if (!s) return nullptr;
  PyObject* r = PyNumber_Remainder(s, p);
  Py_DECREF(s);
  return r;
}

PyObject* FSub(PyObject* p, PyObject* a, PyObject* b) {
  PyObject* s = PyNumber_Subtract(a, b);
  if (!s) return nullptr;
  PyObject* r = PyNumber_Remainder(s, p);  // Python % is non-negative
  Py_DECREF(s);
  return r;
}

PyObject* FMul(PyObject* p, PyObject* a, PyObject* b) {
  PyObject* s = PyNumber_Multiply(a, b);
  if (!s) return nullptr;
  PyObject* r = PyNumber_Remainder(s, p);
  Py_DECREF(s);
  return r;
}

PyObject* FMulInt(PyObject* p, PyObject* a, long k) {
  PyObject* kk = PyLong_FromLong(k);
  if (!kk) return nullptr;
  PyObject* r = FMul(p, a, kk);
  Py_DECREF(kk);
  return r;
}

// Field inverse via Fermat: x^(p-2) mod p (p prime). x must be nonzero.
PyObject* FInv(PyObject* p, PyObject* x) {
  PyObject* two = PyLong_FromLong(2);
  PyObject* pm2 = two ? PyNumber_Subtract(p, two) : nullptr;
  Py_XDECREF(two);
  if (pm2 == nullptr) return nullptr;
  PyObject* r = PyNumber_Power(x, pm2, p);  // 3-arg pow == modular exponent
  Py_DECREF(pm2);
  return r;
}

bool IsZeroCoord(PyObject* x) { return PyObject_IsTrue(x) == 0; }

// Jacobian doubling, a == 0 (EFD dbl-2009-l). in/out: 3 canonical coords.
int JacDouble(PyObject* p, PyObject* const* in, PyObject** out) {
  PyObject* X = in[0];
  PyObject* Y = in[1];
  PyObject* Z = in[2];
  PyObject* xx = FMul(p, X, X);
  PyObject* yy = FMul(p, Y, Y);
  PyObject* yyyy = yy ? FMul(p, yy, yy) : nullptr;
  PyObject* xyy = (xx && yy) ? FMul(p, X, yy) : nullptr;
  PyObject* d = xyy ? FMulInt(p, xyy, 4) : nullptr;  // d = 4*X*yy
  PyObject* e = xx ? FMulInt(p, xx, 3) : nullptr;    // e = 3*xx
  PyObject* ee = e ? FMul(p, e, e) : nullptr;
  PyObject* twod = d ? FMulInt(p, d, 2) : nullptr;
  PyObject* X2 = (ee && twod) ? FSub(p, ee, twod) : nullptr;  // e^2 - 2d
  PyObject* dmx = (d && X2) ? FSub(p, d, X2) : nullptr;
  PyObject* edmx = (e && dmx) ? FMul(p, e, dmx) : nullptr;
  PyObject* eightyyyy = yyyy ? FMulInt(p, yyyy, 8) : nullptr;
  PyObject* Y2 = (edmx && eightyyyy) ? FSub(p, edmx, eightyyyy) : nullptr;
  PyObject* yz = FMul(p, Y, Z);
  PyObject* Z2 = yz ? FMulInt(p, yz, 2) : nullptr;
  Py_XDECREF(xx);
  Py_XDECREF(yy);
  Py_XDECREF(yyyy);
  Py_XDECREF(xyy);
  Py_XDECREF(d);
  Py_XDECREF(e);
  Py_XDECREF(ee);
  Py_XDECREF(twod);
  Py_XDECREF(dmx);
  Py_XDECREF(edmx);
  Py_XDECREF(eightyyyy);
  Py_XDECREF(yz);
  if (!X2 || !Y2 || !Z2) {
    Py_XDECREF(X2);
    Py_XDECREF(Y2);
    Py_XDECREF(Z2);
    return -1;
  }
  out[0] = X2;
  out[1] = Y2;
  out[2] = Z2;
  return 0;
}

void CopyPoint(PyObject* const* in, PyObject** out, int n) {
  for (int i = 0; i < n; ++i) {
    Py_INCREF(in[i]);
    out[i] = in[i];
  }
}

// Jacobian addition (EFD add-2007-bl), a == 0. in1/in2/out: 3 canonical coords.
int JacAdd(PyObject* p, PyObject* const* P, PyObject* const* Q,
           PyObject** out) {
  if (IsZeroCoord(P[2])) {  // P is infinity
    CopyPoint(Q, out, 3);
    return 0;
  }
  if (IsZeroCoord(Q[2])) {  // Q is infinity
    CopyPoint(P, out, 3);
    return 0;
  }
  PyObject *X1 = P[0], *Y1 = P[1], *Z1 = P[2];
  PyObject *X2 = Q[0], *Y2 = Q[1], *Z2 = Q[2];
  PyObject* z1z1 = FMul(p, Z1, Z1);
  PyObject* z2z2 = FMul(p, Z2, Z2);
  PyObject* u1 = z2z2 ? FMul(p, X1, z2z2) : nullptr;
  PyObject* u2 = z1z1 ? FMul(p, X2, z1z1) : nullptr;
  PyObject* yz2 = z2z2 ? FMul(p, Y1, Z2) : nullptr;
  PyObject* s1 = yz2 ? FMul(p, yz2, z2z2) : nullptr;
  PyObject* yz1 = z1z1 ? FMul(p, Y2, Z1) : nullptr;
  PyObject* s2 = yz1 ? FMul(p, yz1, z1z1) : nullptr;
  int rc = -1;
  if (!u1 || !u2 || !s1 || !s2) goto cleanup;
  if (PyObject_RichCompareBool(u1, u2, Py_EQ) == 1 &&
      PyObject_RichCompareBool(s1, s2, Py_EQ) == 1) {
    rc = JacDouble(p, P, out);
    goto cleanup;
  }
  {
    PyObject* h = FSub(p, u2, u1);
    PyObject* twoh = h ? FMulInt(p, h, 2) : nullptr;
    PyObject* i = twoh ? FMul(p, twoh, twoh) : nullptr;  // (2h)^2
    PyObject* hi = (h && i) ? FMul(p, h, i) : nullptr;
    PyObject* j =
        hi ? FSub(p, p, hi) : nullptr;  // j = -(h*i) = p - h*i (then mod)
    PyObject* sdiff = FSub(p, s2, s1);
    PyObject* r = sdiff ? FMulInt(p, sdiff, 2) : nullptr;  // 2(s2-s1)
    PyObject* v = i ? FMul(p, u1, i) : nullptr;
    PyObject* rr = r ? FMul(p, r, r) : nullptr;
    PyObject* twov = v ? FMulInt(p, v, 2) : nullptr;
    PyObject* rrj = (rr && j) ? FAdd(p, rr, j) : nullptr;
    PyObject* X3 =
        (rrj && twov) ? FSub(p, rrj, twov) : nullptr;  // r^2 + j - 2v
    PyObject* vmx = (v && X3) ? FSub(p, v, X3) : nullptr;
    PyObject* rvmx = (r && vmx) ? FMul(p, r, vmx) : nullptr;
    PyObject* s1j = (s1 && j) ? FMul(p, s1, j) : nullptr;
    PyObject* twos1j = s1j ? FMulInt(p, s1j, 2) : nullptr;
    PyObject* Y3 =
        (rvmx && twos1j) ? FAdd(p, rvmx, twos1j) : nullptr;  // r(v-X3) + 2*s1*j
    PyObject* z1z2 = FMul(p, Z1, Z2);
    PyObject* z1z2h = (z1z2 && h) ? FMul(p, z1z2, h) : nullptr;
    PyObject* Z3 = z1z2h ? FMulInt(p, z1z2h, 2) : nullptr;  // 2*Z1*Z2*h
    Py_XDECREF(h);
    Py_XDECREF(twoh);
    Py_XDECREF(i);
    Py_XDECREF(hi);
    Py_XDECREF(sdiff);
    Py_XDECREF(r);
    Py_XDECREF(v);
    Py_XDECREF(rr);
    Py_XDECREF(twov);
    Py_XDECREF(rrj);
    Py_XDECREF(vmx);
    Py_XDECREF(rvmx);
    Py_XDECREF(s1j);
    Py_XDECREF(twos1j);
    Py_XDECREF(z1z2);
    Py_XDECREF(z1z2h);
    if (!X3 || !Y3 || !Z3 || !j) {
      Py_XDECREF(X3);
      Py_XDECREF(Y3);
      Py_XDECREF(Z3);
      Py_XDECREF(j);
      goto cleanup;
    }
    Py_DECREF(j);
    out[0] = X3;
    out[1] = Y3;
    out[2] = Z3;
    rc = 0;
  }
cleanup:
  Py_XDECREF(z1z1);
  Py_XDECREF(z2z2);
  Py_XDECREF(u1);
  Py_XDECREF(u2);
  Py_XDECREF(yz2);
  Py_XDECREF(s1);
  Py_XDECREF(yz1);
  Py_XDECREF(s2);
  return rc;
}

// Negate: flip Y only.
int JacNegate(PyObject* p, PyObject* const* in, PyObject** out) {
  PyObject* negY = FSub(p, p, in[1]);  // -Y = p - Y (then mod)
  if (negY == nullptr) {
    return -1;
  }
  PyObject* y = FMod(p, negY);
  Py_DECREF(negY);
  if (y == nullptr) {
    return -1;
  }
  Py_INCREF(in[0]);
  out[0] = in[0];
  out[1] = y;
  Py_INCREF(in[2]);
  out[2] = in[2];
  return 0;
}

// Group equality (cross-representative): a Jacobian point has many byte
// encodings for one group element. Returns 1 if P == Q as group elements,
// 0 if not, -1 on error. Both infinity -> equal; one infinity -> not equal;
// else x1*z2^2 == x2*z1^2 and y1*z2^3 == y2*z1^3.
int JacEqual(PyObject* p, PyObject* const* P, PyObject* const* Q) {
  bool pz = IsZeroCoord(P[2]);
  bool qz = IsZeroCoord(Q[2]);
  if (pz || qz) {
    return (pz && qz) ? 1 : 0;
  }
  int result = -1;
  PyObject* z1s = FMul(p, P[2], P[2]);  // z1^2
  PyObject* z2s = FMul(p, Q[2], Q[2]);  // z2^2
  PyObject* lx = z2s ? FMul(p, P[0], z2s) : nullptr;
  PyObject* rx = z1s ? FMul(p, Q[0], z1s) : nullptr;
  PyObject* z1c = z1s ? FMul(p, z1s, P[2]) : nullptr;  // z1^3
  PyObject* z2c = z2s ? FMul(p, z2s, Q[2]) : nullptr;  // z2^3
  PyObject* ly = z2c ? FMul(p, P[1], z2c) : nullptr;
  PyObject* ry = z1c ? FMul(p, Q[1], z1c) : nullptr;
  if (lx && rx && ly && ry) {
    int xe = PyObject_RichCompareBool(lx, rx, Py_EQ);
    int ye = PyObject_RichCompareBool(ly, ry, Py_EQ);
    if (xe >= 0 && ye >= 0) {
      result = (xe == 1 && ye == 1) ? 1 : 0;
    }
  }
  Py_XDECREF(z1s);
  Py_XDECREF(z2s);
  Py_XDECREF(lx);
  Py_XDECREF(rx);
  Py_XDECREF(z1c);
  Py_XDECREF(z2c);
  Py_XDECREF(ly);
  Py_XDECREF(ry);
  return result;
}

void MovePoint(PyObject** dst, PyObject* const* src, int n) {
  for (int i = 0; i < n; ++i) {
    Py_DECREF(dst[i]);
    dst[i] = src[i];
  }
}

// MSB-first double-and-add: ret = scalar * point (canonical Jacobian coords).
// The scalar is a canonical integer (Montgomery already decoded by the caller),
// matching the legacy curve operator* which de-Montgomery's the scalar first.
int JacScalarMul(PyObject* p, PyObject* scalar, PyObject* const* point,
                 PyObject** out) {
  PyObject* ret[3] = {PyLong_FromLong(1), PyLong_FromLong(1),
                      PyLong_FromLong(0)};  // Zero = (1, 1, 0)
  if (!ret[0] || !ret[1] || !ret[2]) {
    for (int j = 0; j < 3; ++j) Py_XDECREF(ret[j]);
    return -1;
  }
  size_t nbits = _PyLong_NumBits(scalar);
  size_t nbytes = (nbits + 7) / 8;
  unsigned char buf[64] = {0};
  if (nbytes > sizeof(buf)) {
    for (int j = 0; j < 3; ++j) Py_DECREF(ret[j]);
    PyErr_SetString(PyExc_OverflowError, "EC scalar too large");
    return -1;
  }
  if (nbytes > 0) {
    _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(scalar), buf, nbytes, 1,
                        0);
  }
  for (Py_ssize_t i = static_cast<Py_ssize_t>(nbits) - 1; i >= 0; --i) {
    PyObject* tmp[3];
    if (JacDouble(p, ret, tmp) < 0) {
      for (int j = 0; j < 3; ++j) Py_DECREF(ret[j]);
      return -1;
    }
    MovePoint(ret, tmp, 3);
    if ((buf[i >> 3] >> (i & 7)) & 1) {
      if (JacAdd(p, ret, point, tmp) < 0) {
        for (int j = 0; j < 3; ++j) Py_DECREF(ret[j]);
        return -1;
      }
      MovePoint(ret, tmp, 3);
    }
  }
  out[0] = ret[0];
  out[1] = ret[1];
  out[2] = ret[2];
  return 0;
}

// --- descriptor lifecycle ------------------------------------------------

PyArray_Descr* MakeDescr(PyObject* modulus, int base_width_bytes,
                         int num_coords, int is_montgomery, PyObject* r,
                         PyObject* rinv) {
  auto* d = reinterpret_cast<EcPointDescr*>(PyArrayDescr_Type.tp_new(
      reinterpret_cast<PyTypeObject*>(&EcPointDType), nullptr, nullptr));
  if (d == nullptr) {
    return nullptr;
  }
  Py_INCREF(modulus);
  d->modulus = modulus;
  Py_XINCREF(r);
  d->r_mod_p = r;
  Py_XINCREF(rinv);
  d->rinv_mod_p = rinv;
  d->base_width_bytes = static_cast<uint8_t>(base_width_bytes);
  d->num_coords = static_cast<uint8_t>(num_coords);
  d->is_montgomery = static_cast<uint8_t>(is_montgomery ? 1 : 0);
  PyArray_Descr* base = &d->base;
  base->kind = 'V';
  base->type = 'j';
  base->byteorder = '=';
  base->flags = NPY_USE_GETITEM | NPY_USE_SETITEM;
  base->elsize = base_width_bytes * num_coords;
  base->alignment = base_width_bytes <= 8 ? base_width_bytes : 8;
  return base;
}

void Descr_dealloc(PyObject* self) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  Py_XDECREF(d->modulus);
  Py_XDECREF(d->r_mod_p);
  Py_XDECREF(d->rinv_mod_p);
  PyArrayDescr_Type.tp_dealloc(self);
}

PyObject* DType_new(PyTypeObject* /*cls*/, PyObject* /*args*/,
                    PyObject* /*kwds*/) {
  PyErr_SetString(
      PyExc_TypeError,
      "construct an EC point dtype via zk_dtypes.ec_point(...), not "
      "EcPointDType(...) directly");
  return nullptr;
}

PyObject* Descr_repr(PyObject* self) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  return PyUnicode_FromFormat(
      "EcPointDType(modulus=%R, coords=%d, base_width=%d, mont=%d)", d->modulus,
      static_cast<int>(d->num_coords), d->base_width_bytes * 8,
      static_cast<int>(d->is_montgomery));
}

// --- NEP-42 DType slots --------------------------------------------------

PyArray_Descr* DefaultDescr(PyArray_DTypeMeta* /*cls*/) {
  PyObject* two = PyLong_FromLong(2);
  if (two == nullptr) {
    return nullptr;
  }
  PyArray_Descr* d = MakeDescr(two, 4, 3, 0, nullptr, nullptr);
  Py_DECREF(two);
  return d;
}

PyArray_DTypeMeta* CommonDType(PyArray_DTypeMeta* a, PyArray_DTypeMeta* b) {
  if (a == b) {
    Py_INCREF(a);
    return a;
  }
  Py_INCREF(Py_NotImplemented);
  return reinterpret_cast<PyArray_DTypeMeta*>(Py_NotImplemented);
}

bool SameCurve(EcPointDescr* a, EcPointDescr* b) {
  return a->base_width_bytes == b->base_width_bytes &&
         a->num_coords == b->num_coords &&
         a->is_montgomery == b->is_montgomery &&
         PyObject_RichCompareBool(a->modulus, b->modulus, Py_EQ) == 1;
}

PyArray_Descr* CommonInstance(PyArray_Descr* a, PyArray_Descr* b) {
  if (SameCurve(AsEc(a), AsEc(b))) {
    Py_INCREF(a);
    return a;
  }
  PyErr_SetString(PyExc_TypeError, "cannot combine points of different curves");
  return nullptr;
}

PyArray_Descr* EnsureCanonical(PyArray_Descr* self) {
  Py_INCREF(self);
  return self;
}

PyArray_Descr* DiscoverDescrFromPyobject(PyArray_DTypeMeta* /*cls*/,
                                         PyObject* /*obj*/) {
  PyErr_SetString(PyExc_TypeError,
                  "cannot infer an EC point dtype from a scalar; pass an "
                  "explicit dtype=zk_dtypes.ec_point(...)");
  return nullptr;
}

// setitem accepts a length-num_coords sequence of coordinate integers.
int SetItem(PyArray_Descr* descr, PyObject* obj, char* dataptr) {
  EcPointDescr* d = AsEc(descr);
  PyObject* seq = PySequence_Fast(obj, "EC point needs a coordinate sequence");
  if (seq == nullptr) {
    return -1;
  }
  if (PySequence_Fast_GET_SIZE(seq) != d->num_coords) {
    PyErr_Format(PyExc_ValueError, "EC point needs %d coordinates, got %zd",
                 d->num_coords, PySequence_Fast_GET_SIZE(seq));
    Py_DECREF(seq);
    return -1;
  }
  PyObject* coords[kMaxCoords] = {nullptr};
  int rc = -1;
  for (int i = 0; i < d->num_coords; ++i) {
    coords[i] = PyNumber_Index(PySequence_Fast_GET_ITEM(seq, i));
    if (coords[i] == nullptr) {
      goto done;
    }
  }
  rc = EncodePoint(d, dataptr, coords);
done:
  Py_DECREF(seq);
  for (int i = 0; i < d->num_coords; ++i) Py_XDECREF(coords[i]);
  return rc;
}

PyObject* GetItem(PyArray_Descr* descr, char* dataptr) {
  EcPointDescr* d = AsEc(descr);
  PyObject* coords[kMaxCoords] = {nullptr};
  if (DecodePoint(d, dataptr, coords) < 0) {
    return nullptr;
  }
  PyObject* tuple = PyTuple_New(d->num_coords);
  if (tuple == nullptr) {
    for (int i = 0; i < d->num_coords; ++i) Py_DECREF(coords[i]);
    return nullptr;
  }
  for (int i = 0; i < d->num_coords; ++i) {
    PyTuple_SET_ITEM(tuple, i, coords[i]);
  }
  return tuple;
}

// Same curve and coordinate field, ignoring the representation (coord count).
bool SameCurveField(EcPointDescr* a, EcPointDescr* b) {
  return a->base_width_bytes == b->base_width_bytes &&
         a->is_montgomery == b->is_montgomery &&
         PyObject_RichCompareBool(a->modulus, b->modulus, Py_EQ) == 1;
}

PyObject* One() { return PyLong_FromLong(1); }
PyObject* ZeroL() { return PyLong_FromLong(0); }

// Converts a point between coordinate representations (canonical coords
// in/out). num_coords: affine 2, Jacobian 3, xyzz 4. Jacobian<->xyzz keep the
// projective coordinates (legacy direct formulas); the rest go through affine
// (needs a field inverse), matching the legacy registered casts byte-for-byte.
int ConvertRep(PyObject* p, int fn, int tn, PyObject* const* in,
               PyObject** out) {
  if (fn == tn) {
    CopyPoint(in, out, fn);
    return 0;
  }
  if (fn == 3 && tn == 4) {  // jac (X,Y,Z) -> xyzz (X,Y,Z^2,Z^3)
    PyObject* z2 = FMul(p, in[2], in[2]);
    PyObject* z3 = z2 ? FMul(p, z2, in[2]) : nullptr;
    if (!z2 || !z3) {
      Py_XDECREF(z2);
      Py_XDECREF(z3);
      return -1;
    }
    Py_INCREF(in[0]);
    out[0] = in[0];
    Py_INCREF(in[1]);
    out[1] = in[1];
    out[2] = z2;
    out[3] = z3;
    return 0;
  }
  if (fn == 4 && tn == 3) {  // xyzz (X,Y,ZZ,ZZZ) -> jac (X,Y,ZZZ/ZZ)
    if (IsZeroCoord(in[2])) {
      out[0] = One();
      out[1] = One();
      out[2] = ZeroL();
      if (!out[0] || !out[1] || !out[2]) {
        Py_XDECREF(out[0]);
        Py_XDECREF(out[1]);
        Py_XDECREF(out[2]);
        return -1;
      }
      return 0;
    }
    PyObject* zzinv = FInv(p, in[2]);
    PyObject* z = zzinv ? FMul(p, in[3], zzinv) : nullptr;
    Py_XDECREF(zzinv);
    if (!z) return -1;
    Py_INCREF(in[0]);
    out[0] = in[0];
    Py_INCREF(in[1]);
    out[1] = in[1];
    out[2] = z;
    return 0;
  }
  // Remaining pairs route through affine (x, y).
  PyObject* ax = nullptr;
  PyObject* ay = nullptr;
  bool inf = false;
  if (fn == 2) {
    inf = IsZeroCoord(in[0]) && IsZeroCoord(in[1]);
    Py_INCREF(in[0]);
    ax = in[0];
    Py_INCREF(in[1]);
    ay = in[1];
  } else if (fn == 3) {  // jac -> affine
    if (IsZeroCoord(in[2])) {
      inf = true;
    } else {
      PyObject* zi = FInv(p, in[2]);
      PyObject* z2 = zi ? FMul(p, zi, zi) : nullptr;
      PyObject* z3 = z2 ? FMul(p, z2, zi) : nullptr;
      ax = z2 ? FMul(p, in[0], z2) : nullptr;
      ay = z3 ? FMul(p, in[1], z3) : nullptr;
      Py_XDECREF(zi);
      Py_XDECREF(z2);
      Py_XDECREF(z3);
    }
  } else {  // xyzz -> affine
    if (IsZeroCoord(in[2])) {
      inf = true;
    } else {
      PyObject* zzi = FInv(p, in[2]);
      PyObject* zzzi = FInv(p, in[3]);
      ax = zzi ? FMul(p, in[0], zzi) : nullptr;
      ay = zzzi ? FMul(p, in[1], zzzi) : nullptr;
      Py_XDECREF(zzi);
      Py_XDECREF(zzzi);
    }
  }
  if (!inf && (!ax || !ay)) {
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    return -1;
  }
  int rc = 0;
  if (inf) {
    Py_XDECREF(ax);
    Py_XDECREF(ay);
    if (tn == 2) {
      out[0] = ZeroL();
      out[1] = ZeroL();
    } else {
      out[0] = One();
      out[1] = One();
      out[2] = ZeroL();
      if (tn == 4) out[3] = ZeroL();
    }
    for (int i = 0; i < tn; ++i) {
      if (!out[i]) rc = -1;
    }
  } else {
    out[0] = ax;
    out[1] = ay;
    if (tn >= 3) out[2] = One();
    if (tn == 4) out[3] = One();
    for (int i = 2; i < tn; ++i) {
      if (!out[i]) rc = -1;
    }
  }
  if (rc < 0) {
    for (int i = 0; i < tn; ++i) Py_XDECREF(out[i]);
  }
  return rc;
}

// --- within-DType cast (copy + representation conversion) ----------------

NPY_CASTING CastResolve(struct PyArrayMethodObject_tag* /*method*/,
                        PyArray_DTypeMeta* const* /*dtypes*/,
                        PyArray_Descr* const* given, PyArray_Descr** loop,
                        npy_intp* view_offset) {
  PyArray_Descr* from = given[0];
  Py_INCREF(from);
  loop[0] = from;
  PyArray_Descr* to = given[1];
  if (to == nullptr) {
    Py_INCREF(from);
    loop[1] = from;
    *view_offset = 0;
    return NPY_NO_CASTING;
  }
  Py_INCREF(to);
  loop[1] = to;
  if (SameCurve(AsEc(from), AsEc(to))) {
    *view_offset = 0;
    return NPY_NO_CASTING;
  }
  return NPY_UNSAFE_CASTING;  // same field, different representation
}

int CastLoop(PyArrayMethod_Context* context, char* const* data,
             const npy_intp* dimensions, const npy_intp* strides,
             NpyAuxData* /*aux*/) {
  EcPointDescr* from = AsEc(context->descriptors[0]);
  EcPointDescr* to = AsEc(context->descriptors[1]);
  npy_intp n = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  if (from->num_coords == to->num_coords) {
    npy_intp elsize = context->descriptors[0]->elsize;
    for (npy_intp i = 0; i < n; ++i) {
      std::memcpy(out, in, elsize);
      in += strides[0];
      out += strides[1];
    }
    return 0;
  }
  PyObject* p = from->modulus;
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* src[kMaxCoords];
    PyObject* dst[kMaxCoords];
    if (DecodePoint(from, in, src) < 0) return -1;
    int rc = ConvertRep(p, from->num_coords, to->num_coords, src, dst);
    for (int j = 0; j < from->num_coords; ++j) Py_DECREF(src[j]);
    if (rc < 0) return -1;
    int erc = EncodePoint(to, out, dst);
    for (int j = 0; j < to->num_coords; ++j) Py_DECREF(dst[j]);
    if (erc < 0) return -1;
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

// --- factory -------------------------------------------------------------

PyObject* MakeEcPointDescrPy(PyObject* /*self*/, PyObject* args) {
  PyObject* modulus_obj;
  int base_width_bits;
  int num_coords;
  int is_montgomery;
  PyObject* r_obj = nullptr;
  PyObject* rinv_obj = nullptr;
  if (!PyArg_ParseTuple(args, "Oiii|OO", &modulus_obj, &base_width_bits,
                        &num_coords, &is_montgomery, &r_obj, &rinv_obj)) {
    return nullptr;
  }
  if (base_width_bits != 32 && base_width_bits != 64 &&
      base_width_bits != 128 && base_width_bits != 256) {
    PyErr_SetString(PyExc_ValueError,
                    "base_width_bits must be one of 32, 64, 128, 256");
    return nullptr;
  }
  if (num_coords < 2 || num_coords > kMaxCoords) {
    PyErr_Format(PyExc_ValueError, "num_coords must be in [2, %d]", kMaxCoords);
    return nullptr;
  }
  if (is_montgomery && (r_obj == nullptr || rinv_obj == nullptr)) {
    PyErr_SetString(PyExc_ValueError,
                    "Montgomery storage requires r_mod_p and rinv_mod_p");
    return nullptr;
  }
  PyObject* modulus = PyNumber_Index(modulus_obj);
  if (modulus == nullptr) {
    return nullptr;
  }
  PyObject* r = nullptr;
  PyObject* rinv = nullptr;
  if (is_montgomery) {
    r = PyNumber_Index(r_obj);
    rinv = (r == nullptr) ? nullptr : PyNumber_Index(rinv_obj);
    if (r == nullptr || rinv == nullptr) {
      Py_DECREF(modulus);
      Py_XDECREF(r);
      return nullptr;
    }
  }
  PyArray_Descr* d = MakeDescr(modulus, base_width_bits / 8, num_coords,
                               is_montgomery, r, rinv);
  Py_DECREF(modulus);
  Py_XDECREF(r);
  Py_XDECREF(rinv);
  return reinterpret_cast<PyObject*>(d);
}

PyMethodDef kModuleMethods[] = {
    {"ec_point_descr", MakeEcPointDescrPy, METH_VARARGS,
     "ec_point_descr(modulus, base_width_bits, num_coords, is_montgomery"
     "[, r_mod_p, rinv_mod_p]) -> dtype\n\n"
     "Build a parametric G1 Jacobian EC-point descriptor."},
    {nullptr, nullptr, 0, nullptr},
};

// --- group-law ufunc loops ----------------------------------------------

NPY_CASTING BinResolve(struct PyArrayMethodObject_tag* /*method*/,
                       PyArray_DTypeMeta* const* /*dtypes*/,
                       PyArray_Descr* const* given, PyArray_Descr** loop,
                       npy_intp* view_offset) {
  if (!SameCurve(AsEc(given[0]), AsEc(given[1]))) {
    PyErr_SetString(PyExc_TypeError, "point op requires the same curve");
    return static_cast<NPY_CASTING>(-1);
  }
  if (AsEc(given[0])->num_coords != 3) {
    PyErr_SetString(PyExc_TypeError,
                    "EC arithmetic requires the Jacobian representation; cast "
                    "affine/xyzz points to Jacobian first");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  Py_INCREF(given[1]);
  loop[1] = given[1];
  PyArray_Descr* out = given[2] == nullptr ? given[0] : given[2];
  Py_INCREF(out);
  loop[2] = out;
  *view_offset = NPY_MIN_INTP;
  return NPY_NO_CASTING;
}

NPY_CASTING UnaryResolve(struct PyArrayMethodObject_tag* /*method*/,
                         PyArray_DTypeMeta* const* /*dtypes*/,
                         PyArray_Descr* const* given, PyArray_Descr** loop,
                         npy_intp* view_offset) {
  Py_INCREF(given[0]);
  loop[0] = given[0];
  PyArray_Descr* out = given[1] == nullptr ? given[0] : given[1];
  Py_INCREF(out);
  loop[1] = out;
  *view_offset = NPY_MIN_INTP;
  return NPY_NO_CASTING;
}

enum class BinOp { kAdd, kSub };

template <BinOp op>
int BinLoop(PyArrayMethod_Context* context, char* const* data,
            const npy_intp* dimensions, const npy_intp* strides,
            NpyAuxData* /*aux*/) {
  EcPointDescr* d = AsEc(context->descriptors[0]);
  PyObject* p = d->modulus;
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* P[kMaxCoords];
    PyObject* Q[kMaxCoords];
    PyObject* R[kMaxCoords];
    if (DecodePoint(d, a, P) < 0) return -1;
    if (DecodePoint(d, b, Q) < 0) {
      for (int j = 0; j < d->num_coords; ++j) Py_DECREF(P[j]);
      return -1;
    }
    int rc;
    if (op == BinOp::kSub) {
      PyObject* negQ[kMaxCoords];
      rc = JacNegate(p, Q, negQ);
      if (rc == 0) {
        rc = JacAdd(p, P, negQ, R);
        for (int j = 0; j < d->num_coords; ++j) Py_DECREF(negQ[j]);
      }
    } else {
      rc = JacAdd(p, P, Q, R);
    }
    for (int j = 0; j < d->num_coords; ++j) {
      Py_DECREF(P[j]);
      Py_DECREF(Q[j]);
    }
    if (rc < 0) return -1;
    int erc = EncodePoint(d, o, R);
    for (int j = 0; j < d->num_coords; ++j) Py_DECREF(R[j]);
    if (erc < 0) return -1;
    a += strides[0];
    b += strides[1];
    o += strides[2];
  }
  return 0;
}

int NegLoop(PyArrayMethod_Context* context, char* const* data,
            const npy_intp* dimensions, const npy_intp* strides,
            NpyAuxData* /*aux*/) {
  EcPointDescr* d = AsEc(context->descriptors[0]);
  PyObject* p = d->modulus;
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* o = data[1];
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* P[kMaxCoords];
    PyObject* R[kMaxCoords];
    if (DecodePoint(d, a, P) < 0) return -1;
    int rc = JacNegate(p, P, R);
    for (int j = 0; j < d->num_coords; ++j) Py_DECREF(P[j]);
    if (rc < 0) return -1;
    int erc = EncodePoint(d, o, R);
    for (int j = 0; j < d->num_coords; ++j) Py_DECREF(R[j]);
    if (erc < 0) return -1;
    a += strides[0];
    o += strides[1];
  }
  return 0;
}

bool AddBinLoop(PyObject* numpy, const char* name,
                PyArrayMethod_StridedLoop* loop) {
  PyObject* ufunc = PyObject_GetAttrString(numpy, name);
  if (ufunc == nullptr) return false;
  PyArray_DTypeMeta* dtypes[3] = {&EcPointDType, &EcPointDType, &EcPointDType};
  PyType_Slot slots[] = {
      {NPY_METH_resolve_descriptors, reinterpret_cast<void*>(BinResolve)},
      {NPY_METH_strided_loop, reinterpret_cast<void*>(loop)},
      {0, nullptr},
  };
  PyArrayMethod_Spec spec = {};
  spec.name = "ec_point_binop";
  spec.nin = 2;
  spec.nout = 1;
  spec.casting = NPY_NO_CASTING;
  spec.flags = NPY_METH_REQUIRES_PYAPI;
  spec.dtypes = dtypes;
  spec.slots = slots;
  int rc = PyUFunc_AddLoopFromSpec(ufunc, &spec);
  Py_DECREF(ufunc);
  return rc >= 0;
}

bool AddNegLoop(PyObject* numpy) {
  PyObject* ufunc = PyObject_GetAttrString(numpy, "negative");
  if (ufunc == nullptr) return false;
  PyArray_DTypeMeta* dtypes[2] = {&EcPointDType, &EcPointDType};
  PyType_Slot slots[] = {
      {NPY_METH_resolve_descriptors, reinterpret_cast<void*>(UnaryResolve)},
      {NPY_METH_strided_loop, reinterpret_cast<void*>(NegLoop)},
      {0, nullptr},
  };
  PyArrayMethod_Spec spec = {};
  spec.name = "ec_point_negate";
  spec.nin = 1;
  spec.nout = 1;
  spec.casting = NPY_NO_CASTING;
  spec.flags = NPY_METH_REQUIRES_PYAPI;
  spec.dtypes = dtypes;
  spec.slots = slots;
  int rc = PyUFunc_AddLoopFromSpec(ufunc, &spec);
  Py_DECREF(ufunc);
  return rc >= 0;
}

// Comparison: (point, point) -> bool. Inputs share a curve; output is bool.
NPY_CASTING CmpResolve(struct PyArrayMethodObject_tag* /*method*/,
                       PyArray_DTypeMeta* const* /*dtypes*/,
                       PyArray_Descr* const* given, PyArray_Descr** loop,
                       npy_intp* view_offset) {
  if (!SameCurve(AsEc(given[0]), AsEc(given[1]))) {
    PyErr_SetString(PyExc_TypeError,
                    "point comparison requires the same curve");
    return static_cast<NPY_CASTING>(-1);
  }
  if (AsEc(given[0])->num_coords != 3) {
    PyErr_SetString(PyExc_TypeError,
                    "EC comparison requires the Jacobian representation; cast "
                    "affine/xyzz points to Jacobian first");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  Py_INCREF(given[1]);
  loop[1] = given[1];
  loop[2] = given[2] != nullptr ? given[2] : PyArray_DescrFromType(NPY_BOOL);
  if (loop[2] == nullptr) {
    return static_cast<NPY_CASTING>(-1);
  }
  if (given[2] != nullptr) {
    Py_INCREF(loop[2]);
  }
  *view_offset = NPY_MIN_INTP;
  return NPY_NO_CASTING;
}

template <bool negate>
int CmpLoop(PyArrayMethod_Context* context, char* const* data,
            const npy_intp* dimensions, const npy_intp* strides,
            NpyAuxData* /*aux*/) {
  EcPointDescr* d = AsEc(context->descriptors[0]);
  PyObject* p = d->modulus;
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* P[kMaxCoords];
    PyObject* Q[kMaxCoords];
    if (DecodePoint(d, a, P) < 0) return -1;
    if (DecodePoint(d, b, Q) < 0) {
      for (int j = 0; j < d->num_coords; ++j) Py_DECREF(P[j]);
      return -1;
    }
    int eq = JacEqual(p, P, Q);
    for (int j = 0; j < d->num_coords; ++j) {
      Py_DECREF(P[j]);
      Py_DECREF(Q[j]);
    }
    if (eq < 0) return -1;
    *reinterpret_cast<npy_bool*>(o) = (negate ? !eq : eq) ? 1 : 0;
    a += strides[0];
    b += strides[1];
    o += strides[2];
  }
  return 0;
}

bool AddCmpLoop(PyObject* numpy, const char* name,
                PyArrayMethod_StridedLoop* loop) {
  PyObject* ufunc = PyObject_GetAttrString(numpy, name);
  if (ufunc == nullptr) return false;
  PyArray_DTypeMeta* booldt =
      reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(PyArray_DescrFromType(
          NPY_BOOL)));  // borrowed: builtin bool descr/DType are immortal
  PyArray_DTypeMeta* dtypes[3] = {&EcPointDType, &EcPointDType, booldt};
  PyType_Slot slots[] = {
      {NPY_METH_resolve_descriptors, reinterpret_cast<void*>(CmpResolve)},
      {NPY_METH_strided_loop, reinterpret_cast<void*>(loop)},
      {0, nullptr},
  };
  PyArrayMethod_Spec spec = {};
  spec.name = "ec_point_compare";
  spec.nin = 2;
  spec.nout = 1;
  spec.casting = NPY_NO_CASTING;
  spec.flags = NPY_METH_REQUIRES_PYAPI;
  spec.dtypes = dtypes;
  spec.slots = slots;
  int rc = PyUFunc_AddLoopFromSpec(ufunc, &spec);
  Py_DECREF(ufunc);
  return rc >= 0;
}

// Scalar multiplication: np.multiply(scalar_field, point) and the reverse. The
// output is the point's Ec dtype; the scalar field is the other operand.
NPY_CASTING ScalarMulResolve(struct PyArrayMethodObject_tag* /*method*/,
                             PyArray_DTypeMeta* const* /*dtypes*/,
                             PyArray_Descr* const* given, PyArray_Descr** loop,
                             npy_intp* view_offset) {
  PyArray_Descr* point =
      Py_TYPE(given[0]) == reinterpret_cast<PyTypeObject*>(&EcPointDType)
          ? given[0]
          : given[1];
  if (AsEc(point)->num_coords != 3) {
    PyErr_SetString(PyExc_TypeError,
                    "EC scalar multiplication requires the Jacobian "
                    "representation; cast the point to Jacobian first");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  Py_INCREF(given[1]);
  loop[1] = given[1];
  loop[2] = given[2] != nullptr ? given[2] : point;
  Py_INCREF(loop[2]);
  *view_offset = NPY_MIN_INTP;
  return NPY_NO_CASTING;
}

template <bool scalar_first>
int ScalarMulLoop(PyArrayMethod_Context* context, char* const* data,
                  const npy_intp* dimensions, const npy_intp* strides,
                  NpyAuxData* /*aux*/) {
  PyArray_Descr* scalar_descr = context->descriptors[scalar_first ? 0 : 1];
  EcPointDescr* d = AsEc(context->descriptors[scalar_first ? 1 : 0]);
  EcPointDescr* od = AsEc(context->descriptors[2]);
  PyObject* p = d->modulus;
  npy_intp n = dimensions[0];
  char* s = data[scalar_first ? 0 : 1];
  char* pt = data[scalar_first ? 1 : 0];
  char* o = data[2];
  npy_intp s_stride = strides[scalar_first ? 0 : 1];
  npy_intp pt_stride = strides[scalar_first ? 1 : 0];
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* scalar =
        PrimeFieldValue(reinterpret_cast<PyObject*>(scalar_descr), s);
    if (scalar == nullptr) return -1;
    PyObject* P[kMaxCoords];
    if (DecodePoint(d, pt, P) < 0) {
      Py_DECREF(scalar);
      return -1;
    }
    PyObject* R[3];
    int rc = JacScalarMul(p, scalar, P, R);
    Py_DECREF(scalar);
    for (int j = 0; j < d->num_coords; ++j) Py_DECREF(P[j]);
    if (rc < 0) return -1;
    int erc = EncodePoint(od, o, R);
    for (int j = 0; j < 3; ++j) Py_DECREF(R[j]);
    if (erc < 0) return -1;
    s += s_stride;
    pt += pt_stride;
    o += strides[2];
  }
  return 0;
}

bool AddScalarMulLoops(PyObject* numpy) {
  PyObject* ufunc = PyObject_GetAttrString(numpy, "multiply");
  if (ufunc == nullptr) return false;
  PyArray_DTypeMeta* field =
      reinterpret_cast<PyArray_DTypeMeta*>(FieldDTypeMetaObject());
  bool ok = true;
  for (int order = 0; order < 2 && ok; ++order) {
    PyArray_DTypeMeta* dtypes[3];
    PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors,
         reinterpret_cast<void*>(ScalarMulResolve)},
        {NPY_METH_strided_loop,
         reinterpret_cast<void*>(order == 0 ? ScalarMulLoop<true>
                                            : ScalarMulLoop<false>)},
        {0, nullptr},
    };
    if (order == 0) {  // scalar * point
      dtypes[0] = field;
      dtypes[1] = &EcPointDType;
    } else {  // point * scalar
      dtypes[0] = &EcPointDType;
      dtypes[1] = field;
    }
    dtypes[2] = &EcPointDType;
    PyArrayMethod_Spec spec = {};
    spec.name = "ec_point_scalar_mul";
    spec.nin = 2;
    spec.nout = 1;
    spec.casting = NPY_NO_CASTING;
    spec.flags = NPY_METH_REQUIRES_PYAPI;
    spec.dtypes = dtypes;
    spec.slots = slots;
    ok = PyUFunc_AddLoopFromSpec(ufunc, &spec) >= 0;
  }
  Py_DECREF(ufunc);
  return ok;
}

}  // namespace

bool RegisterEcPointDType(PyObject* /*numpy*/, PyObject* module) {
  EcPointScalar_Type.tp_name = "zk_dtypes._zk_dtypes_ext.EcPointScalar";
  EcPointScalar_Type.tp_basicsize = 0;
  EcPointScalar_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  EcPointScalar_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&EcPointScalar_Type) < 0) {
    return false;
  }

  PyTypeObject* type = reinterpret_cast<PyTypeObject*>(&EcPointDType);
  Py_SET_TYPE(reinterpret_cast<PyObject*>(&EcPointDType),
              &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(reinterpret_cast<PyObject*>(&EcPointDType), 1);
  type->tp_name = "zk_dtypes._zk_dtypes_ext.EcPointDType";
  type->tp_basicsize = sizeof(EcPointDescr);
  type->tp_flags = Py_TPFLAGS_DEFAULT;
  type->tp_base = &PyArrayDescr_Type;
  type->tp_dealloc = Descr_dealloc;
  type->tp_repr = Descr_repr;
  type->tp_str = Descr_repr;
  type->tp_new = DType_new;
  if (PyType_Ready(type) < 0) {
    return false;
  }

  PyArray_DTypeMeta* cast_dtypes[2] = {nullptr, nullptr};
  PyType_Slot cast_slots[] = {
      {NPY_METH_resolve_descriptors, reinterpret_cast<void*>(CastResolve)},
      {NPY_METH_strided_loop, reinterpret_cast<void*>(CastLoop)},
      {NPY_METH_unaligned_strided_loop, reinterpret_cast<void*>(CastLoop)},
      {0, nullptr},
  };
  PyArrayMethod_Spec copy_cast = {};
  copy_cast.name = "ec_point_copy";
  copy_cast.nin = 1;
  copy_cast.nout = 1;
  copy_cast.casting = NPY_UNSAFE_CASTING;
  copy_cast.flags = NPY_METH_SUPPORTS_UNALIGNED;
  copy_cast.dtypes = cast_dtypes;
  copy_cast.slots = cast_slots;
  PyArrayMethod_Spec* casts[] = {&copy_cast, nullptr};

  PyType_Slot dtype_slots[] = {
      {NPY_DT_default_descr, reinterpret_cast<void*>(DefaultDescr)},
      {NPY_DT_common_dtype, reinterpret_cast<void*>(CommonDType)},
      {NPY_DT_common_instance, reinterpret_cast<void*>(CommonInstance)},
      {NPY_DT_ensure_canonical, reinterpret_cast<void*>(EnsureCanonical)},
      {NPY_DT_discover_descr_from_pyobject,
       reinterpret_cast<void*>(DiscoverDescrFromPyobject)},
      {NPY_DT_setitem, reinterpret_cast<void*>(SetItem)},
      {NPY_DT_getitem, reinterpret_cast<void*>(GetItem)},
      {0, nullptr},
  };

  PyArrayDTypeMeta_Spec spec = {};
  spec.typeobj = &EcPointScalar_Type;
  spec.flags = NPY_DT_PARAMETRIC;
  spec.casts = casts;
  spec.slots = dtype_slots;
  spec.baseclass = nullptr;
  if (PyArrayInitDTypeMeta_FromSpec(&EcPointDType, &spec) < 0) {
    return false;
  }
  EcPointDType.singleton = PyArray_GetDefaultDescr(&EcPointDType);
  if (EcPointDType.singleton == nullptr) {
    return false;
  }

  if (PyModule_AddObject(module, "EcPointDType",
                         reinterpret_cast<PyObject*>(&EcPointDType)) < 0) {
    return false;
  }
  Py_INCREF(reinterpret_cast<PyObject*>(&EcPointDType));

  PyObject* fn = PyCFunction_New(&kModuleMethods[0], nullptr);
  if (fn == nullptr) {
    return false;
  }
  if (PyModule_AddObject(module, "ec_point_descr", fn) < 0) {
    Py_DECREF(fn);
    return false;
  }

  if (_import_umath() < 0) {
    return false;
  }
  PyObject* numpy = PyImport_ImportModule("numpy");
  if (numpy == nullptr) {
    return false;
  }
  bool ok = AddBinLoop(numpy, "add", BinLoop<BinOp::kAdd>) &&
            AddBinLoop(numpy, "subtract", BinLoop<BinOp::kSub>) &&
            AddNegLoop(numpy) && AddCmpLoop(numpy, "equal", CmpLoop<false>) &&
            AddCmpLoop(numpy, "not_equal", CmpLoop<true>) &&
            AddScalarMulLoops(numpy);
  Py_DECREF(numpy);
  return ok;
}

}  // namespace zk_dtypes
