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

#include "zk_dtypes/_src/field_dtype.h"
#include "zk_dtypes/_src/field_modarith.h"
#include "zk_dtypes/_src/numpy.h"
#include "numpy/dtype_api.h"
#include "numpy/ndarraytypes.h"
// clang-format on

namespace zk_dtypes {
namespace {

constexpr int kMaxCoords = 4;

struct EcPointDescr {
  PyArray_Descr base;
  PyObject* modulus;  // owned: base prime field modulus
  PyObject* r_mod_p;  // owned (Montgomery): R = 2^base_width mod p; else NULL
  PyObject* rinv_mod_p;  // owned (Montgomery): R^-1 mod p; else NULL
  // Coordinate field: prime (G1, degree 1) or Fp2 (G2, degree 2, u^2 = nr).
  PyObject* non_residue;  // owned (degree 2): the Fp2 non-residue; else NULL
  uint8_t coord_degree;   // 1 = G1 (Fq), 2 = G2 (Fp2)
  uint8_t base_width_bytes;
  uint8_t num_coords;  // 2 affine, 3 Jacobian, 4 xyzz
  uint8_t is_montgomery;
};

PyArray_DTypeMeta EcPointDType = {};
PyTypeObject EcPointScalar_Type = {};

EcPointDescr* AsEc(PyArray_Descr* d) {
  return reinterpret_cast<EcPointDescr*>(d);
}

// --- coordinate encode / decode -----------------------------------------
// A coordinate is one base-field element (G1) or `coord_degree` of them packed
// low-to-high (G2 Fp2 = c0,c1). Its decoded value is a canonical int (degree 1)
// or a tuple of canonical ints (degree > 1).

PyObject* DecodeBase(EcPointDescr* d, const char* slot) {
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

int EncodeBase(EcPointDescr* d, char* slot, PyObject* value) {
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

PyObject* DecodeCoord(EcPointDescr* d, const char* slot) {
  if (d->coord_degree == 1) {
    return DecodeBase(d, slot);
  }
  PyObject* tuple = PyTuple_New(d->coord_degree);
  if (tuple == nullptr) {
    return nullptr;
  }
  for (int k = 0; k < d->coord_degree; ++k) {
    PyObject* c = DecodeBase(d, slot + k * d->base_width_bytes);
    if (c == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, k, c);
  }
  return tuple;
}

int EncodeCoord(EcPointDescr* d, char* slot, PyObject* coord) {
  if (d->coord_degree == 1) {
    return EncodeBase(d, slot, coord);
  }
  for (int k = 0; k < d->coord_degree; ++k) {
    if (EncodeBase(d, slot + k * d->base_width_bytes,
                   PyTuple_GET_ITEM(coord, k)) < 0) {
      return -1;
    }
  }
  return 0;
}

int DecodePoint(EcPointDescr* d, const char* ptr, PyObject** out) {
  int stride = d->coord_degree * d->base_width_bytes;
  for (int i = 0; i < d->num_coords; ++i) {
    out[i] = DecodeCoord(d, ptr + i * stride);
    if (out[i] == nullptr) {
      for (int j = 0; j < i; ++j) Py_DECREF(out[j]);
      return -1;
    }
  }
  return 0;
}

int EncodePoint(EcPointDescr* d, char* ptr, PyObject* const* coords) {
  int stride = d->coord_degree * d->base_width_bytes;
  for (int i = 0; i < d->num_coords; ++i) {
    if (EncodeCoord(d, ptr + i * stride, coords[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

// --- prime-field helpers on canonical Python ints -----------------------

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

// --- coordinate-field ops (prime Fq for G1; Fp2 for G2, u^2 = non_residue) --
// A coordinate value is a canonical Python int (degree 1) or a 2-tuple of
// canonical ints (a0, a1) for Fp2. The C* helpers dispatch on coord_degree and
// build on the base-field F* helpers, so the group-law formulas are written
// once and work over either coordinate field.

PyObject* MakeFp2(PyObject* c0, PyObject* c1) {  // steals c0, c1
  if (!c0 || !c1) {
    Py_XDECREF(c0);
    Py_XDECREF(c1);
    return nullptr;
  }
  PyObject* t = PyTuple_New(2);
  if (!t) {
    Py_DECREF(c0);
    Py_DECREF(c1);
    return nullptr;
  }
  PyTuple_SET_ITEM(t, 0, c0);
  PyTuple_SET_ITEM(t, 1, c1);
  return t;
}

PyObject* CZero(EcPointDescr* d) {
  if (d->coord_degree == 1) return PyLong_FromLong(0);
  return MakeFp2(PyLong_FromLong(0), PyLong_FromLong(0));
}

PyObject* COne(EcPointDescr* d) {
  if (d->coord_degree == 1) return PyLong_FromLong(1);
  return MakeFp2(PyLong_FromLong(1), PyLong_FromLong(0));
}

bool CIsZero(EcPointDescr* d, PyObject* a) {
  if (d->coord_degree == 1) return IsZeroCoord(a);
  return IsZeroCoord(PyTuple_GET_ITEM(a, 0)) &&
         IsZeroCoord(PyTuple_GET_ITEM(a, 1));
}

PyObject* CAdd(EcPointDescr* d, PyObject* a, PyObject* b) {
  if (d->coord_degree == 1) return FAdd(d->modulus, a, b);
  return MakeFp2(
      FAdd(d->modulus, PyTuple_GET_ITEM(a, 0), PyTuple_GET_ITEM(b, 0)),
      FAdd(d->modulus, PyTuple_GET_ITEM(a, 1), PyTuple_GET_ITEM(b, 1)));
}

PyObject* CSub(EcPointDescr* d, PyObject* a, PyObject* b) {
  if (d->coord_degree == 1) return FSub(d->modulus, a, b);
  return MakeFp2(
      FSub(d->modulus, PyTuple_GET_ITEM(a, 0), PyTuple_GET_ITEM(b, 0)),
      FSub(d->modulus, PyTuple_GET_ITEM(a, 1), PyTuple_GET_ITEM(b, 1)));
}

PyObject* CNeg(EcPointDescr* d, PyObject* a) {
  PyObject* m = d->modulus;
  if (d->coord_degree == 1) return FSub(m, m, a);  // (p - a) mod p
  return MakeFp2(FSub(m, m, PyTuple_GET_ITEM(a, 0)),
                 FSub(m, m, PyTuple_GET_ITEM(a, 1)));
}

PyObject* CMulInt(EcPointDescr* d, PyObject* a, long k) {
  if (d->coord_degree == 1) return FMulInt(d->modulus, a, k);
  return MakeFp2(FMulInt(d->modulus, PyTuple_GET_ITEM(a, 0), k),
                 FMulInt(d->modulus, PyTuple_GET_ITEM(a, 1), k));
}

PyObject* CMul(EcPointDescr* d, PyObject* a, PyObject* b) {
  PyObject* m = d->modulus;
  if (d->coord_degree == 1) return FMul(m, a, b);
  PyObject* a0 = PyTuple_GET_ITEM(a, 0);
  PyObject* a1 = PyTuple_GET_ITEM(a, 1);
  PyObject* b0 = PyTuple_GET_ITEM(b, 0);
  PyObject* b1 = PyTuple_GET_ITEM(b, 1);
  // (a0 + a1 u)(b0 + b1 u) = (a0 b0 + nr a1 b1) + (a0 b1 + a1 b0) u
  PyObject* a0b0 = FMul(m, a0, b0);
  PyObject* a1b1 = FMul(m, a1, b1);
  PyObject* nra1b1 = a1b1 ? FMul(m, d->non_residue, a1b1) : nullptr;
  PyObject* c0 = (a0b0 && nra1b1) ? FAdd(m, a0b0, nra1b1) : nullptr;
  PyObject* a0b1 = FMul(m, a0, b1);
  PyObject* a1b0 = FMul(m, a1, b0);
  PyObject* c1 = (a0b1 && a1b0) ? FAdd(m, a0b1, a1b0) : nullptr;
  Py_XDECREF(a0b0);
  Py_XDECREF(a1b1);
  Py_XDECREF(nra1b1);
  Py_XDECREF(a0b1);
  Py_XDECREF(a1b0);
  return MakeFp2(c0, c1);
}

// Coordinate-field inverse (used by representation casts).
PyObject* CInv(EcPointDescr* d, PyObject* a) {
  PyObject* m = d->modulus;
  if (d->coord_degree == 1) return FInv(m, a);
  PyObject* a0 = PyTuple_GET_ITEM(a, 0);
  PyObject* a1 = PyTuple_GET_ITEM(a, 1);
  // norm = a0^2 - nr a1^2 ; a^-1 = (a0 - a1 u) / norm
  PyObject* a0sq = FMul(m, a0, a0);
  PyObject* a1sq = FMul(m, a1, a1);
  PyObject* nra1sq = a1sq ? FMul(m, d->non_residue, a1sq) : nullptr;
  PyObject* norm = (a0sq && nra1sq) ? FSub(m, a0sq, nra1sq) : nullptr;
  PyObject* ninv = norm ? FInv(m, norm) : nullptr;
  PyObject* c0 = ninv ? FMul(m, a0, ninv) : nullptr;
  PyObject* na1 = FSub(m, m, a1);
  PyObject* c1 = (ninv && na1) ? FMul(m, na1, ninv) : nullptr;
  Py_XDECREF(a0sq);
  Py_XDECREF(a1sq);
  Py_XDECREF(nra1sq);
  Py_XDECREF(norm);
  Py_XDECREF(ninv);
  Py_XDECREF(na1);
  return MakeFp2(c0, c1);
}

// Jacobian doubling, a == 0 (EFD dbl-2009-l). in/out: 3 canonical coords.
int JacDouble(EcPointDescr* ec, PyObject* const* in, PyObject** out) {
  PyObject* X = in[0];
  PyObject* Y = in[1];
  PyObject* Z = in[2];
  PyObject* xx = CMul(ec, X, X);
  PyObject* yy = CMul(ec, Y, Y);
  PyObject* yyyy = yy ? CMul(ec, yy, yy) : nullptr;
  PyObject* xyy = (xx && yy) ? CMul(ec, X, yy) : nullptr;
  PyObject* d = xyy ? CMulInt(ec, xyy, 4) : nullptr;  // d = 4*X*yy
  PyObject* e = xx ? CMulInt(ec, xx, 3) : nullptr;    // e = 3*xx
  PyObject* ee = e ? CMul(ec, e, e) : nullptr;
  PyObject* twod = d ? CMulInt(ec, d, 2) : nullptr;
  PyObject* X2 = (ee && twod) ? CSub(ec, ee, twod) : nullptr;  // e^2 - 2d
  PyObject* dmx = (d && X2) ? CSub(ec, d, X2) : nullptr;
  PyObject* edmx = (e && dmx) ? CMul(ec, e, dmx) : nullptr;
  PyObject* eightyyyy = yyyy ? CMulInt(ec, yyyy, 8) : nullptr;
  PyObject* Y2 = (edmx && eightyyyy) ? CSub(ec, edmx, eightyyyy) : nullptr;
  PyObject* yz = CMul(ec, Y, Z);
  PyObject* Z2 = yz ? CMulInt(ec, yz, 2) : nullptr;
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
int JacAdd(EcPointDescr* ec, PyObject* const* P, PyObject* const* Q,
           PyObject** out) {
  if (CIsZero(ec, P[2])) {  // P is infinity
    CopyPoint(Q, out, 3);
    return 0;
  }
  if (CIsZero(ec, Q[2])) {  // Q is infinity
    CopyPoint(P, out, 3);
    return 0;
  }
  PyObject *X1 = P[0], *Y1 = P[1], *Z1 = P[2];
  PyObject *X2 = Q[0], *Y2 = Q[1], *Z2 = Q[2];
  PyObject* z1z1 = CMul(ec, Z1, Z1);
  PyObject* z2z2 = CMul(ec, Z2, Z2);
  PyObject* u1 = z2z2 ? CMul(ec, X1, z2z2) : nullptr;
  PyObject* u2 = z1z1 ? CMul(ec, X2, z1z1) : nullptr;
  PyObject* yz2 = z2z2 ? CMul(ec, Y1, Z2) : nullptr;
  PyObject* s1 = yz2 ? CMul(ec, yz2, z2z2) : nullptr;
  PyObject* yz1 = z1z1 ? CMul(ec, Y2, Z1) : nullptr;
  PyObject* s2 = yz1 ? CMul(ec, yz1, z1z1) : nullptr;
  int rc = -1;
  if (!u1 || !u2 || !s1 || !s2) goto cleanup;
  if (PyObject_RichCompareBool(u1, u2, Py_EQ) == 1 &&
      PyObject_RichCompareBool(s1, s2, Py_EQ) == 1) {
    rc = JacDouble(ec, P, out);
    goto cleanup;
  }
  {
    PyObject* h = CSub(ec, u2, u1);
    PyObject* twoh = h ? CMulInt(ec, h, 2) : nullptr;
    PyObject* i = twoh ? CMul(ec, twoh, twoh) : nullptr;  // (2h)^2
    PyObject* hi = (h && i) ? CMul(ec, h, i) : nullptr;
    PyObject* j = hi ? CNeg(ec, hi) : nullptr;  // j = -(h*i)
    PyObject* sdiff = CSub(ec, s2, s1);
    PyObject* r = sdiff ? CMulInt(ec, sdiff, 2) : nullptr;  // 2(s2-s1)
    PyObject* v = i ? CMul(ec, u1, i) : nullptr;
    PyObject* rr = r ? CMul(ec, r, r) : nullptr;
    PyObject* twov = v ? CMulInt(ec, v, 2) : nullptr;
    PyObject* rrj = (rr && j) ? CAdd(ec, rr, j) : nullptr;
    PyObject* X3 =
        (rrj && twov) ? CSub(ec, rrj, twov) : nullptr;  // r^2 + j - 2v
    PyObject* vmx = (v && X3) ? CSub(ec, v, X3) : nullptr;
    PyObject* rvmx = (r && vmx) ? CMul(ec, r, vmx) : nullptr;
    PyObject* s1j = (s1 && j) ? CMul(ec, s1, j) : nullptr;
    PyObject* twos1j = s1j ? CMulInt(ec, s1j, 2) : nullptr;
    PyObject* Y3 =
        (rvmx && twos1j) ? CAdd(ec, rvmx, twos1j) : nullptr;  // r(v-X3)+2*s1*j
    PyObject* z1z2 = CMul(ec, Z1, Z2);
    PyObject* z1z2h = (z1z2 && h) ? CMul(ec, z1z2, h) : nullptr;
    PyObject* Z3 = z1z2h ? CMulInt(ec, z1z2h, 2) : nullptr;  // 2*Z1*Z2*h
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
// Negate flips Y (coordinate 1) and copies the rest — rep-safe for affine /
// Jacobian / xyzz alike (Y is coordinate 1 in every representation).
int JacNegate(EcPointDescr* ec, PyObject* const* in, PyObject** out) {
  PyObject* negY = CNeg(ec, in[1]);
  if (negY == nullptr) {
    return -1;
  }
  for (int i = 0; i < ec->num_coords; ++i) {
    if (i == 1) {
      out[i] = negY;
    } else {
      Py_INCREF(in[i]);
      out[i] = in[i];
    }
  }
  return 0;
}

// Group equality (cross-representative): a Jacobian point has many byte
// encodings for one group element. Returns 1 if P == Q as group elements,
// 0 if not, -1 on error. Both infinity -> equal; one infinity -> not equal;
// else x1*z2^2 == x2*z1^2 and y1*z2^3 == y2*z1^3.
int JacEqual(EcPointDescr* ec, PyObject* const* P, PyObject* const* Q) {
  bool pz = CIsZero(ec, P[2]);
  bool qz = CIsZero(ec, Q[2]);
  if (pz || qz) {
    return (pz && qz) ? 1 : 0;
  }
  int result = -1;
  PyObject* z1s = CMul(ec, P[2], P[2]);
  PyObject* z2s = CMul(ec, Q[2], Q[2]);
  PyObject* lx = z2s ? CMul(ec, P[0], z2s) : nullptr;
  PyObject* rx = z1s ? CMul(ec, Q[0], z1s) : nullptr;
  PyObject* z1c = z1s ? CMul(ec, z1s, P[2]) : nullptr;
  PyObject* z2c = z2s ? CMul(ec, z2s, Q[2]) : nullptr;
  PyObject* ly = z2c ? CMul(ec, P[1], z2c) : nullptr;
  PyObject* ry = z1c ? CMul(ec, Q[1], z1c) : nullptr;
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

// Decodes a canonical scalar int into `buf` (little-endian, zero-padded) and
// returns its bit length, or -1 (with an exception set) if it needs more than
// buf_size bytes.
Py_ssize_t ScalarToBytesLE(PyObject* scalar, unsigned char* buf,
                           size_t buf_size) {
  size_t nbits = _PyLong_NumBits(scalar);
  size_t nbytes = (nbits + 7) / 8;
  if (nbytes > buf_size) {
    PyErr_SetString(PyExc_OverflowError, "EC scalar too large");
    return -1;
  }
  std::memset(buf, 0, buf_size);
  if (nbytes > 0) {
    _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(scalar), buf, nbytes, 1,
                        0);
  }
  return static_cast<Py_ssize_t>(nbits);
}

// MSB-first double-and-add: ret = scalar * point (canonical Jacobian coords).
// The scalar is a canonical integer (Montgomery already decoded by the caller),
// matching the legacy curve operator* which de-Montgomery's the scalar first.
int JacScalarMul(EcPointDescr* ec, PyObject* scalar, PyObject* const* point,
                 PyObject** out) {
  PyObject* ret[3] = {COne(ec), COne(ec), CZero(ec)};  // Zero = (1, 1, 0)
  if (!ret[0] || !ret[1] || !ret[2]) {
    for (int j = 0; j < 3; ++j) Py_XDECREF(ret[j]);
    return -1;
  }
  unsigned char buf[64];
  Py_ssize_t nbits = ScalarToBytesLE(scalar, buf, sizeof(buf));
  if (nbits < 0) {
    for (int j = 0; j < 3; ++j) Py_DECREF(ret[j]);
    return -1;
  }
  for (Py_ssize_t i = nbits - 1; i >= 0; --i) {
    PyObject* tmp[3];
    if (JacDouble(ec, ret, tmp) < 0) {
      for (int j = 0; j < 3; ++j) Py_DECREF(ret[j]);
      return -1;
    }
    MovePoint(ret, tmp, 3);
    if ((buf[i >> 3] >> (i & 7)) & 1) {
      if (JacAdd(ec, ret, point, tmp) < 0) {
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

// --- native fixed-width group law ----------------------------------------
// The Python-int group law above is correct but allocates a CPython int per
// coordinate operation. For Montgomery-stored points over a coordinate field
// whose base width the native kernels handle, run the identical EFD formulas in
// Montgomery space directly on the stored bytes (mont(x)*mont(y)*R^-1 =
// mont(x*y); add/sub are linear; X^... = non_residue folds with mont(nr)), so
// no decode/encode and no Python ints — byte-identical to the path above.
//
// These are byte-space twins of the PyObject Jac* functions: EcDouble<->
// JacDouble, EcAdd<->JacAdd, EcNegate<->JacNegate, EcEqual<->JacEqual,
// EcScalarMul<->JacScalarMul, CoordField::Mul (Fp2) <-> CMul. The two MUST stay
// in lockstep — the Jac* versions are the canonical spec; a formula change must
// be made in both (the byte-identity tests only catch divergence on the cases
// they exercise).

constexpr int kCoordBytes = 64;  // max coordinate: Fp2 over 256-bit base

// One coordinate field (Fq for G1, Fp2 for G2) over a Montgomery base field.
struct CoordField {
  modarith::PrimeField fq;
  int degree = 1;                   // 1 = Fq, 2 = Fp2
  int wb = 0;                       // base-field width in bytes
  int cb = 0;                       // coordinate width = degree * wb
  bool ok = false;                  // native path usable for this descriptor
  unsigned char one_le[32] = {0};   // mont(1) = R mod p
  unsigned char mont_nr[32] = {0};  // mont(non_residue), Fp2 only

  static CoordField Make(EcPointDescr* d) {
    CoordField cf;
    if (!d->is_montgomery) return cf;  // native group law is Montgomery-only
    cf.wb = d->base_width_bytes;
    cf.degree = d->coord_degree;
    cf.cb = cf.degree * cf.wb;
    if (cf.degree < 1 || cf.degree > 2) return cf;
    unsigned char mod_le[32];
    if (_PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(d->modulus), mod_le,
                            cf.wb, 1, 0) < 0) {
      PyErr_Clear();
      return cf;
    }
    cf.fq = modarith::PrimeField::Make(mod_le, cf.wb, /*is_mont=*/true);
    if (!cf.fq.native) return cf;
    // The native EC temporaries are fixed kCoordBytes stack buffers (= 2*32,
    // the Fp2-over-256-bit max). degree<=2 and a native base width<=32 keep cb
    // in range; bail to the Python path otherwise rather than overflow.
    if (cf.cb > kCoordBytes) return cf;
    if (_PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(d->r_mod_p),
                            cf.one_le, cf.wb, 1, 0) < 0) {
      PyErr_Clear();
      return cf;
    }
    if (cf.degree == 2) {
      // mont(nr) = nr * R mod p; computed once here in Python, then native.
      PyObject* scaled = PyNumber_Multiply(d->non_residue, d->r_mod_p);
      PyObject* mn = scaled ? PyNumber_Remainder(scaled, d->modulus) : nullptr;
      Py_XDECREF(scaled);
      if (mn == nullptr ||
          _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(mn), cf.mont_nr,
                              cf.wb, 1, 0) < 0) {
        Py_XDECREF(mn);
        PyErr_Clear();
        return cf;
      }
      Py_DECREF(mn);
    }
    cf.ok = true;
    return cf;
  }

  bool IsZero(const unsigned char* a) const {
    for (int i = 0; i < cb; ++i) {
      if (a[i] != 0) return false;
    }
    return true;
  }
  bool Equal(const unsigned char* a, const unsigned char* b) const {
    return std::memcmp(a, b, cb) == 0;
  }
  void SetZero(unsigned char* o) const { std::memset(o, 0, cb); }
  void SetOne(unsigned char* o) const {  // (mont(1), 0)
    std::memset(o, 0, cb);
    std::memcpy(o, one_le, wb);
  }
  void Add(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    for (int k = 0; k < degree; ++k) {
      fq.Add(a + k * wb, b + k * wb, o + k * wb);
    }
  }
  void Sub(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    for (int k = 0; k < degree; ++k) {
      fq.Sub(a + k * wb, b + k * wb, o + k * wb);
    }
  }
  void Neg(const unsigned char* a, unsigned char* o) const {  // p - a per coeff
    unsigned char zero[32] = {0};
    for (int k = 0; k < degree; ++k) {
      fq.Sub(zero, a + k * wb, o + k * wb);
    }
  }
  void Mul(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    if (degree == 1) {
      fq.Mul(a, b, o);
      return;
    }
    // Fp2: (a0+a1 u)(b0+b1 u) = (a0 b0 + nr a1 b1) + (a0 b1 + a1 b0) u.
    const unsigned char* a0 = a;
    const unsigned char* a1 = a + wb;
    const unsigned char* b0 = b;
    const unsigned char* b1 = b + wb;
    unsigned char a0b0[32], a1b1[32], nra1b1[32], a0b1[32], a1b0[32];
    unsigned char c0[32], c1[32];
    fq.Mul(a0, b0, a0b0);
    fq.Mul(a1, b1, a1b1);
    fq.Mul(mont_nr, a1b1, nra1b1);
    fq.Add(a0b0, nra1b1, c0);
    fq.Mul(a0, b1, a0b1);
    fq.Mul(a1, b0, a1b0);
    fq.Add(a0b1, a1b0, c1);
    std::memcpy(o, c0, wb);
    std::memcpy(o + wb, c1, wb);
  }
  void MulInt(const unsigned char* a, int k, unsigned char* o) const {
    std::memcpy(o, a, cb);  // 1*a
    for (int j = 1; j < k; ++j) Add(o, a, o);
  }
};

// Jacobian doubling (EFD dbl-2009-l, a == 0); 3 coords. out may alias in.
void EcDouble(const CoordField& cf, const unsigned char* in,
              unsigned char* out) {
  const int cb = cf.cb;
  const unsigned char* X = in;
  const unsigned char* Y = in + cb;
  const unsigned char* Z = in + 2 * cb;
  unsigned char xx[kCoordBytes], yy[kCoordBytes], yyyy[kCoordBytes];
  unsigned char xyy[kCoordBytes], dd[kCoordBytes], e[kCoordBytes];
  unsigned char ee[kCoordBytes], twod[kCoordBytes], X2[kCoordBytes];
  unsigned char dmx[kCoordBytes], edmx[kCoordBytes], eightyyyy[kCoordBytes];
  unsigned char Y2[kCoordBytes], yz[kCoordBytes], Z2[kCoordBytes];
  cf.Mul(X, X, xx);
  cf.Mul(Y, Y, yy);
  cf.Mul(yy, yy, yyyy);
  cf.Mul(X, yy, xyy);
  cf.MulInt(xyy, 4, dd);
  cf.MulInt(xx, 3, e);
  cf.Mul(e, e, ee);
  cf.MulInt(dd, 2, twod);
  cf.Sub(ee, twod, X2);
  cf.Sub(dd, X2, dmx);
  cf.Mul(e, dmx, edmx);
  cf.MulInt(yyyy, 8, eightyyyy);
  cf.Sub(edmx, eightyyyy, Y2);
  cf.Mul(Y, Z, yz);
  cf.MulInt(yz, 2, Z2);
  std::memcpy(out, X2, cb);
  std::memcpy(out + cb, Y2, cb);
  std::memcpy(out + 2 * cb, Z2, cb);
}

// Jacobian addition (EFD add-2007-bl, a == 0); 3 coords. `out` may alias an
// input: every read of P/Q completes into stack temporaries before the result
// is written, and the infinity copies below skip a no-op self-copy.
void EcAdd(const CoordField& cf, const unsigned char* P, const unsigned char* Q,
           unsigned char* out) {
  const int cb = cf.cb;
  const unsigned char* Z1 = P + 2 * cb;
  const unsigned char* Z2 = Q + 2 * cb;
  if (cf.IsZero(Z1)) {
    if (out != Q) std::memcpy(out, Q, 3 * cb);
    return;
  }
  if (cf.IsZero(Z2)) {
    if (out != P) std::memcpy(out, P, 3 * cb);
    return;
  }
  const unsigned char* X1 = P;
  const unsigned char* Y1 = P + cb;
  const unsigned char* X2 = Q;
  const unsigned char* Y2 = Q + cb;
  unsigned char z1z1[kCoordBytes], z2z2[kCoordBytes], u1[kCoordBytes];
  unsigned char u2[kCoordBytes], yz2[kCoordBytes], s1[kCoordBytes];
  unsigned char yz1[kCoordBytes], s2[kCoordBytes];
  cf.Mul(Z1, Z1, z1z1);
  cf.Mul(Z2, Z2, z2z2);
  cf.Mul(X1, z2z2, u1);
  cf.Mul(X2, z1z1, u2);
  cf.Mul(Y1, Z2, yz2);
  cf.Mul(yz2, z2z2, s1);
  cf.Mul(Y2, Z1, yz1);
  cf.Mul(yz1, z1z1, s2);
  if (cf.Equal(u1, u2) && cf.Equal(s1, s2)) {  // P == Q
    EcDouble(cf, P, out);
    return;
  }
  unsigned char h[kCoordBytes], twoh[kCoordBytes], ii[kCoordBytes];
  unsigned char hi[kCoordBytes], j[kCoordBytes], sdiff[kCoordBytes];
  unsigned char r[kCoordBytes], v[kCoordBytes], rr[kCoordBytes];
  unsigned char twov[kCoordBytes], rrj[kCoordBytes], X3[kCoordBytes];
  unsigned char vmx[kCoordBytes], rvmx[kCoordBytes], s1j[kCoordBytes];
  unsigned char twos1j[kCoordBytes], Y3[kCoordBytes], z1z2[kCoordBytes];
  unsigned char z1z2h[kCoordBytes], Z3[kCoordBytes];
  cf.Sub(u2, u1, h);
  cf.MulInt(h, 2, twoh);
  cf.Mul(twoh, twoh, ii);
  cf.Mul(h, ii, hi);
  cf.Neg(hi, j);
  cf.Sub(s2, s1, sdiff);
  cf.MulInt(sdiff, 2, r);
  cf.Mul(u1, ii, v);
  cf.Mul(r, r, rr);
  cf.MulInt(v, 2, twov);
  cf.Add(rr, j, rrj);
  cf.Sub(rrj, twov, X3);  // r^2 + j - 2v
  cf.Sub(v, X3, vmx);
  cf.Mul(r, vmx, rvmx);
  cf.Mul(s1, j, s1j);
  cf.MulInt(s1j, 2, twos1j);
  cf.Add(rvmx, twos1j, Y3);  // r(v - X3) + 2 s1 j
  cf.Mul(Z1, Z2, z1z2);
  cf.Mul(z1z2, h, z1z2h);
  cf.MulInt(z1z2h, 2, Z3);  // 2 Z1 Z2 h
  std::memcpy(out, X3, cb);
  std::memcpy(out + cb, Y3, cb);
  std::memcpy(out + 2 * cb, Z3, cb);
}

void EcNegate(const CoordField& cf, const unsigned char* in, unsigned char* out,
              int num_coords) {
  const int cb = cf.cb;
  for (int i = 0; i < num_coords; ++i) {
    if (i == 1) {
      cf.Neg(in + cb, out + cb);
    } else {
      std::memcpy(out + i * cb, in + i * cb, cb);
    }
  }
}

// Cross-representative group equality of two Jacobian points; 1 / 0.
int EcEqual(const CoordField& cf, const unsigned char* P,
            const unsigned char* Q) {
  const int cb = cf.cb;
  bool pz = cf.IsZero(P + 2 * cb);
  bool qz = cf.IsZero(Q + 2 * cb);
  if (pz || qz) return (pz && qz) ? 1 : 0;
  unsigned char z1s[kCoordBytes], z2s[kCoordBytes], lx[kCoordBytes];
  unsigned char rx[kCoordBytes], z1c[kCoordBytes], z2c[kCoordBytes];
  unsigned char ly[kCoordBytes], ry[kCoordBytes];
  cf.Mul(P + 2 * cb, P + 2 * cb, z1s);
  cf.Mul(Q + 2 * cb, Q + 2 * cb, z2s);
  cf.Mul(P, z2s, lx);
  cf.Mul(Q, z1s, rx);
  cf.Mul(z1s, P + 2 * cb, z1c);
  cf.Mul(z2s, Q + 2 * cb, z2c);
  cf.Mul(P + cb, z2c, ly);
  cf.Mul(Q + cb, z1c, ry);
  return (cf.Equal(lx, rx) && cf.Equal(ly, ry)) ? 1 : 0;
}

// ret = scalar * point, MSB-first double-and-add; scalar as little-endian
// bytes.
void EcScalarMul(const CoordField& cf, const unsigned char* point,
                 const unsigned char* sbytes, int nbits, unsigned char* out) {
  const int cb = cf.cb;
  unsigned char ret[3 * kCoordBytes];
  cf.SetOne(ret);  // Jacobian zero = (1, 1, 0)
  cf.SetOne(ret + cb);
  cf.SetZero(ret + 2 * cb);
  // EcDouble/EcAdd read all of their inputs before writing the result, so they
  // accept out == in1 in place; no scratch point is needed.
  for (int i = nbits - 1; i >= 0; --i) {
    EcDouble(cf, ret, ret);
    if ((sbytes[i >> 3] >> (i & 7)) & 1) {
      EcAdd(cf, ret, point, ret);
    }
  }
  std::memcpy(out, ret, 3 * cb);
}

// --- descriptor lifecycle ------------------------------------------------

PyArray_Descr* MakeDescr(PyObject* modulus, int base_width_bytes,
                         int num_coords, int is_montgomery, PyObject* r,
                         PyObject* rinv, int coord_degree,
                         PyObject* non_residue) {
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
  Py_XINCREF(non_residue);
  d->non_residue = non_residue;
  d->coord_degree = static_cast<uint8_t>(coord_degree);
  d->base_width_bytes = static_cast<uint8_t>(base_width_bytes);
  d->num_coords = static_cast<uint8_t>(num_coords);
  d->is_montgomery = static_cast<uint8_t>(is_montgomery ? 1 : 0);
  PyArray_Descr* base = &d->base;
  base->kind = 'V';
  base->type = 'j';
  base->byteorder = '=';
  base->flags = NPY_USE_GETITEM | NPY_USE_SETITEM;
  base->elsize = base_width_bytes * coord_degree * num_coords;
  base->alignment = base_width_bytes <= 8 ? base_width_bytes : 8;
  return base;
}

void Descr_dealloc(PyObject* self) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  Py_XDECREF(d->modulus);
  Py_XDECREF(d->r_mod_p);
  Py_XDECREF(d->rinv_mod_p);
  Py_XDECREF(d->non_residue);
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
  PyArray_Descr* d = MakeDescr(two, 4, 3, 0, nullptr, nullptr, 1, nullptr);
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
  if (a->base_width_bytes != b->base_width_bytes ||
      a->num_coords != b->num_coords || a->coord_degree != b->coord_degree ||
      a->is_montgomery != b->is_montgomery ||
      PyObject_RichCompareBool(a->modulus, b->modulus, Py_EQ) != 1) {
    return false;
  }
  // Fp2 (G2) is parameterized by the non-residue; distinct non-residues are
  // distinct fields even at the same prime.
  if (a->coord_degree == 2) {
    return PyObject_RichCompareBool(a->non_residue, b->non_residue, Py_EQ) == 1;
  }
  return true;
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

// Converts a point between coordinate representations (canonical coords
// in/out). num_coords: affine 2, Jacobian 3, xyzz 4. Jacobian<->xyzz keep the
// projective coordinates (legacy direct formulas); the rest go through affine
// (needs a field inverse), matching the legacy registered casts byte-for-byte.
int ConvertRep(EcPointDescr* ec, int fn, int tn, PyObject* const* in,
               PyObject** out) {
  if (fn == tn) {
    CopyPoint(in, out, fn);
    return 0;
  }
  if (fn == 3 && tn == 4) {  // jac (X,Y,Z) -> xyzz (X,Y,Z^2,Z^3)
    PyObject* z2 = CMul(ec, in[2], in[2]);
    PyObject* z3 = z2 ? CMul(ec, z2, in[2]) : nullptr;
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
    if (CIsZero(ec, in[2])) {
      out[0] = COne(ec);
      out[1] = COne(ec);
      out[2] = CZero(ec);
      if (!out[0] || !out[1] || !out[2]) {
        Py_XDECREF(out[0]);
        Py_XDECREF(out[1]);
        Py_XDECREF(out[2]);
        return -1;
      }
      return 0;
    }
    PyObject* zzinv = CInv(ec, in[2]);
    PyObject* z = zzinv ? CMul(ec, in[3], zzinv) : nullptr;
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
    inf = CIsZero(ec, in[0]) && CIsZero(ec, in[1]);
    Py_INCREF(in[0]);
    ax = in[0];
    Py_INCREF(in[1]);
    ay = in[1];
  } else if (fn == 3) {  // jac -> affine
    if (CIsZero(ec, in[2])) {
      inf = true;
    } else {
      PyObject* zi = CInv(ec, in[2]);
      PyObject* z2 = zi ? CMul(ec, zi, zi) : nullptr;
      PyObject* z3 = z2 ? CMul(ec, z2, zi) : nullptr;
      ax = z2 ? CMul(ec, in[0], z2) : nullptr;
      ay = z3 ? CMul(ec, in[1], z3) : nullptr;
      Py_XDECREF(zi);
      Py_XDECREF(z2);
      Py_XDECREF(z3);
    }
  } else {  // xyzz -> affine
    if (CIsZero(ec, in[2])) {
      inf = true;
    } else {
      PyObject* zzi = CInv(ec, in[2]);
      PyObject* zzzi = CInv(ec, in[3]);
      ax = zzi ? CMul(ec, in[0], zzi) : nullptr;
      ay = zzzi ? CMul(ec, in[1], zzzi) : nullptr;
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
      out[0] = CZero(ec);
      out[1] = CZero(ec);
    } else {
      out[0] = COne(ec);
      out[1] = COne(ec);
      out[2] = CZero(ec);
      if (tn == 4) out[3] = CZero(ec);
    }
    for (int i = 0; i < tn; ++i) {
      if (!out[i]) rc = -1;
    }
  } else {
    out[0] = ax;
    out[1] = ay;
    if (tn >= 3) out[2] = COne(ec);
    if (tn == 4) out[3] = COne(ec);
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
  // Only a byte-identical field is a raw copy; anything else (a coordinate-rep
  // change OR a Montgomery<->canonical re-encoding at the same shape) must go
  // through decode/encode below.
  if (SameCurve(from, to)) {
    npy_intp elsize = context->descriptors[0]->elsize;
    for (npy_intp i = 0; i < n; ++i) {
      std::memcpy(out, in, elsize);
      in += strides[0];
      out += strides[1];
    }
    return 0;
  }
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* src[kMaxCoords];
    PyObject* dst[kMaxCoords];
    if (DecodePoint(from, in, src) < 0) return -1;
    int rc = ConvertRep(from, from->num_coords, to->num_coords, src, dst);
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
  int coord_degree = 1;
  PyObject* nr_obj = nullptr;
  if (!PyArg_ParseTuple(args, "Oiii|OOiO", &modulus_obj, &base_width_bits,
                        &num_coords, &is_montgomery, &r_obj, &rinv_obj,
                        &coord_degree, &nr_obj)) {
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
  if (coord_degree < 1 || coord_degree > 2) {
    PyErr_SetString(PyExc_ValueError, "coord_degree must be 1 (G1) or 2 (G2)");
    return nullptr;
  }
  if (coord_degree == 2 && nr_obj == nullptr) {
    PyErr_SetString(PyExc_ValueError,
                    "coord_degree 2 (Fp2) requires a non_residue");
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
  PyObject* nr = nullptr;
  if (coord_degree == 2) {
    nr = PyNumber_Index(nr_obj);
    if (nr == nullptr) {
      Py_DECREF(modulus);
      Py_XDECREF(r);
      Py_XDECREF(rinv);
      return nullptr;
    }
  }
  PyArray_Descr* d = MakeDescr(modulus, base_width_bits / 8, num_coords,
                               is_montgomery, r, rinv, coord_degree, nr);
  Py_DECREF(modulus);
  Py_XDECREF(r);
  Py_XDECREF(rinv);
  Py_XDECREF(nr);
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
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  CoordField cf = CoordField::Make(d);
  if (cf.ok) {  // num_coords == 3 guaranteed by BinResolve
    unsigned char neg_q[3 * kCoordBytes];
    for (npy_intp i = 0; i < n; ++i) {
      const auto* ua = reinterpret_cast<const unsigned char*>(a);
      const auto* ub = reinterpret_cast<const unsigned char*>(b);
      auto* uo = reinterpret_cast<unsigned char*>(o);
      if (op == BinOp::kSub) {
        EcNegate(cf, ub, neg_q, 3);
        EcAdd(cf, ua, neg_q, uo);
      } else {
        EcAdd(cf, ua, ub, uo);
      }
      a += strides[0];
      b += strides[1];
      o += strides[2];
    }
    return 0;
  }
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
      rc = JacNegate(d, Q, negQ);
      if (rc == 0) {
        rc = JacAdd(d, P, negQ, R);
        for (int j = 0; j < d->num_coords; ++j) Py_DECREF(negQ[j]);
      }
    } else {
      rc = JacAdd(d, P, Q, R);
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
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* o = data[1];
  CoordField cf = CoordField::Make(d);
  if (cf.ok) {
    for (npy_intp i = 0; i < n; ++i) {
      EcNegate(cf, reinterpret_cast<const unsigned char*>(a),
               reinterpret_cast<unsigned char*>(o), d->num_coords);
      a += strides[0];
      o += strides[1];
    }
    return 0;
  }
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* P[kMaxCoords];
    PyObject* R[kMaxCoords];
    if (DecodePoint(d, a, P) < 0) return -1;
    int rc = JacNegate(d, P, R);
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
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  CoordField cf = CoordField::Make(d);
  if (cf.ok) {  // num_coords == 3 guaranteed by CmpResolve
    for (npy_intp i = 0; i < n; ++i) {
      int eq = EcEqual(cf, reinterpret_cast<const unsigned char*>(a),
                       reinterpret_cast<const unsigned char*>(b));
      *reinterpret_cast<npy_bool*>(o) = (negate ? !eq : eq) ? 1 : 0;
      a += strides[0];
      b += strides[1];
      o += strides[2];
    }
    return 0;
  }
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* P[kMaxCoords];
    PyObject* Q[kMaxCoords];
    if (DecodePoint(d, a, P) < 0) return -1;
    if (DecodePoint(d, b, Q) < 0) {
      for (int j = 0; j < d->num_coords; ++j) Py_DECREF(P[j]);
      return -1;
    }
    int eq = JacEqual(d, P, Q);
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
  // PyArray_DescrFromType returns a new reference; we only need its DType meta
  // (Py_TYPE, borrowed and immortal), so drop the descr ref.
  PyArray_Descr* bool_descr = PyArray_DescrFromType(NPY_BOOL);
  if (bool_descr == nullptr) {
    Py_DECREF(ufunc);
    return false;
  }
  PyArray_DTypeMeta* booldt =
      reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(bool_descr));
  Py_DECREF(bool_descr);
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
  // The output is a Jacobian point on the same curve. A user-supplied out= of a
  // different curve/representation would be written with the input point's
  // width (the loop sizes its store from the input) — reject it instead of an
  // out-of-bounds / wrong-width write.
  PyArray_Descr* out = given[2] != nullptr ? given[2] : point;
  if (given[2] != nullptr && !SameCurve(AsEc(out), AsEc(point))) {
    PyErr_SetString(
        PyExc_TypeError,
        "EC scalar multiplication output must be the same curve and "
        "representation as the point");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  Py_INCREF(given[1]);
  loop[1] = given[1];
  Py_INCREF(out);
  loop[2] = out;
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
  npy_intp n = dimensions[0];
  char* s = data[scalar_first ? 0 : 1];
  char* pt = data[scalar_first ? 1 : 0];
  char* o = data[2];
  npy_intp s_stride = strides[scalar_first ? 0 : 1];
  npy_intp pt_stride = strides[scalar_first ? 1 : 0];
  CoordField cf = CoordField::Make(d);
  if (cf.ok && d->num_coords == 3) {
    for (npy_intp i = 0; i < n; ++i) {
      PyObject* scalar =
          PrimeFieldValue(reinterpret_cast<PyObject*>(scalar_descr), s);
      if (scalar == nullptr) return -1;
      unsigned char buf[64];
      Py_ssize_t nbits = ScalarToBytesLE(scalar, buf, sizeof(buf));
      Py_DECREF(scalar);
      if (nbits < 0) return -1;
      EcScalarMul(cf, reinterpret_cast<const unsigned char*>(pt), buf,
                  static_cast<int>(nbits), reinterpret_cast<unsigned char*>(o));
      s += s_stride;
      pt += pt_stride;
      o += strides[2];
    }
    return 0;
  }
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
    int rc = JacScalarMul(d, scalar, P, R);
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
