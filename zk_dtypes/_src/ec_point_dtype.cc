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
// Module, not a Field: the ops are point add / subtract / negate and scalar
// multiplication — there is no point*point. Covers short-Weierstrass (a=0)
// Jacobian points over a prime (G1) or Fp2 (G2) coordinate field.
//
// A point is `num_coords` coordinate-field elements (X, Y, Z for Jacobian),
// stored exactly as the coordinate field stores them (canonical or per-coord
// Montgomery, little-endian). The group-law formulas live once, in
// ec_group_law.h, and are instantiated here for three execution tiers that
// differ only in their coordinate ops (see `Tier`): CPython ints (any config;
// decode to canonical, compute, re-encode — the EFD formulas are R-linear, so
// this is byte-identical to computing in Montgomery space), native Montgomery
// kernels on the stored bytes, and 256-bit typed limbs. The tier is chosen
// once per descriptor at creation.

// numpy.h must precede every other numpy header (it sets the API symbol) and
// NPY_TARGET_VERSION must precede numpyconfig.h; the associated header pulls in
// only <Python.h>, so it can lead. Keep this order — do not let clang-format
// sort it.
// clang-format off
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "zk_dtypes/_src/ec_point_dtype.h"

#include <cstdint>
#include <cstring>

#include "zk_dtypes/_src/ec_group_law.h"
#include "zk_dtypes/_src/field_dtype.h"
#include "zk_dtypes/include/field/runtime_field.h"
#include "zk_dtypes/_src/nep42_common.h"
#include "zk_dtypes/_src/pyfield_ops.h"
#include "zk_dtypes/_src/numpy.h"
#include "numpy/dtype_api.h"
#include "numpy/ndarraytypes.h"
// clang-format on

namespace zk_dtypes {
namespace {

constexpr int kMaxCoords = 4;

struct CoordField;

// Which group-law arithmetic serves a descriptor. Decided once at descriptor
// creation from the coordinate field's width and storage; the ufunc loops
// only switch on it.
enum class Tier : uint8_t {
  kPy,        // CPython-int arithmetic; any width, storage, and degree
  kByte,      // native Montgomery kernels on the stored bytes
  kTyped256,  // typed-limb BigInt<4> / Fp2<4> locals (256-bit base field)
};

struct EcPointDescr {
  PyArray_Descr base;
  PyObject* modulus;  // owned: base prime field modulus
  PyObject* r_mod_p;  // owned (Montgomery): R = 2^base_width mod p; else NULL
  PyObject* rinv_mod_p;  // owned (Montgomery): R^-1 mod p; else NULL
  // Coordinate field: prime (G1, degree 1) or Fp2 (G2, degree 2, u^2 = nr).
  PyObject* non_residue;  // owned (degree 2): the Fp2 non-residue; else NULL
  // Optional generator, as a tuple of `num_coords` canonical coordinate
  // values. Present only when the factory was given one; it is what makes
  // `arr[i] = n` (n*G) and int -> point casts meaningful.
  PyObject* generator;
  CoordField* native;    // owned: native-kernel constants; may have ok=false
  uint8_t coord_degree;  // 1 = G1 (Fq), 2 = G2 (Fp2)
  uint8_t base_width_bytes;
  uint8_t num_coords;  // 2 affine, 3 Jacobian, 4 xyzz
  uint8_t is_montgomery;
  Tier tier;
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

pyfield::BaseCodec Codec(EcPointDescr* d) {
  return {d->modulus, d->r_mod_p, d->rinv_mod_p, d->base_width_bytes,
          d->is_montgomery != 0};
}

PyObject* DecodeBase(EcPointDescr* d, const char* slot) {
  return pyfield::Decode(Codec(d), slot);
}

int EncodeBase(EcPointDescr* d, char* slot, PyObject* value) {
  return pyfield::Encode(Codec(d), slot, value);
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
  if (d->coord_degree == 1) return pyfield::ModAdd(d->modulus, a, b);
  return MakeFp2(pyfield::ModAdd(d->modulus, PyTuple_GET_ITEM(a, 0),
                                 PyTuple_GET_ITEM(b, 0)),
                 pyfield::ModAdd(d->modulus, PyTuple_GET_ITEM(a, 1),
                                 PyTuple_GET_ITEM(b, 1)));
}

PyObject* CSub(EcPointDescr* d, PyObject* a, PyObject* b) {
  if (d->coord_degree == 1) return pyfield::ModSub(d->modulus, a, b);
  return MakeFp2(pyfield::ModSub(d->modulus, PyTuple_GET_ITEM(a, 0),
                                 PyTuple_GET_ITEM(b, 0)),
                 pyfield::ModSub(d->modulus, PyTuple_GET_ITEM(a, 1),
                                 PyTuple_GET_ITEM(b, 1)));
}

PyObject* CNeg(EcPointDescr* d, PyObject* a) {
  PyObject* m = d->modulus;
  if (d->coord_degree == 1) return pyfield::ModSub(m, m, a);  // (p - a) mod p
  return MakeFp2(pyfield::ModSub(m, m, PyTuple_GET_ITEM(a, 0)),
                 pyfield::ModSub(m, m, PyTuple_GET_ITEM(a, 1)));
}

PyObject* CMulInt(EcPointDescr* d, PyObject* a, long k) {
  if (d->coord_degree == 1) return pyfield::ModMulInt(d->modulus, a, k);
  return MakeFp2(pyfield::ModMulInt(d->modulus, PyTuple_GET_ITEM(a, 0), k),
                 pyfield::ModMulInt(d->modulus, PyTuple_GET_ITEM(a, 1), k));
}

PyObject* CMul(EcPointDescr* d, PyObject* a, PyObject* b) {
  PyObject* m = d->modulus;
  if (d->coord_degree == 1) return pyfield::ModMul(m, a, b);
  PyObject* a0 = PyTuple_GET_ITEM(a, 0);
  PyObject* a1 = PyTuple_GET_ITEM(a, 1);
  PyObject* b0 = PyTuple_GET_ITEM(b, 0);
  PyObject* b1 = PyTuple_GET_ITEM(b, 1);
  // (a0 + a1 u)(b0 + b1 u) = (a0 b0 + nr a1 b1) + (a0 b1 + a1 b0) u
  PyObject* a0b0 = pyfield::ModMul(m, a0, b0);
  PyObject* a1b1 = pyfield::ModMul(m, a1, b1);
  PyObject* nra1b1 = a1b1 ? pyfield::ModMul(m, d->non_residue, a1b1) : nullptr;
  PyObject* c0 = (a0b0 && nra1b1) ? pyfield::ModAdd(m, a0b0, nra1b1) : nullptr;
  PyObject* a0b1 = pyfield::ModMul(m, a0, b1);
  PyObject* a1b0 = pyfield::ModMul(m, a1, b0);
  PyObject* c1 = (a0b1 && a1b0) ? pyfield::ModAdd(m, a0b1, a1b0) : nullptr;
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
  if (d->coord_degree == 1) return pyfield::ModInv(m, a);
  PyObject* a0 = PyTuple_GET_ITEM(a, 0);
  PyObject* a1 = PyTuple_GET_ITEM(a, 1);
  // norm = a0^2 - nr a1^2 ; a^-1 = (a0 - a1 u) / norm
  PyObject* a0sq = pyfield::ModMul(m, a0, a0);
  PyObject* a1sq = pyfield::ModMul(m, a1, a1);
  PyObject* nra1sq = a1sq ? pyfield::ModMul(m, d->non_residue, a1sq) : nullptr;
  PyObject* norm =
      (a0sq && nra1sq) ? pyfield::ModSub(m, a0sq, nra1sq) : nullptr;
  PyObject* ninv = norm ? pyfield::ModInv(m, norm) : nullptr;
  PyObject* c0 = ninv ? pyfield::ModMul(m, a0, ninv) : nullptr;
  PyObject* na1 = pyfield::ModSub(m, m, a1);
  PyObject* c1 = (ninv && na1) ? pyfield::ModMul(m, na1, ninv) : nullptr;
  Py_XDECREF(a0sq);
  Py_XDECREF(a1sq);
  Py_XDECREF(nra1sq);
  Py_XDECREF(norm);
  Py_XDECREF(ninv);
  Py_XDECREF(na1);
  return MakeFp2(c0, c1);
}

// --- CPython-int instantiation of the shared group law -------------------
// `PyRef` is the value-semantic coordinate (copy = INCREF) and `PyCoordOps`
// adapts the C* helpers to the ec_law Ops concept. A failing CPython call
// (allocation, comparison) poisons the ops: the failing helper has already set
// the exception, every later operation no-ops on the null value, and the Jac*
// wrappers translate the poisoned flag into their usual -1.

class PyRef {
 public:
  PyRef() = default;
  explicit PyRef(PyObject* p) : p_(p) {}  // steals
  PyRef(const PyRef& o) : p_(o.p_) { Py_XINCREF(p_); }
  PyRef(PyRef&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  PyRef& operator=(const PyRef& o) {
    if (this != &o) {
      Py_XINCREF(o.p_);
      Py_XDECREF(p_);
      p_ = o.p_;
    }
    return *this;
  }
  PyRef& operator=(PyRef&& o) noexcept {
    if (this != &o) {
      Py_XDECREF(p_);
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ~PyRef() { Py_XDECREF(p_); }
  PyObject* get() const { return p_; }
  PyObject* release() {
    PyObject* r = p_;
    p_ = nullptr;
    return r;
  }
  static PyRef Borrow(PyObject* p) {
    Py_XINCREF(p);
    return PyRef(p);
  }

 private:
  PyObject* p_ = nullptr;
};

struct PyCoordOps {
  EcPointDescr* d;
  mutable bool failed = false;
  PyRef Wrap(PyObject* r) const {
    if (r == nullptr) failed = true;
    return PyRef(r);
  }
  bool Bad(const PyRef& a) const { return failed || a.get() == nullptr; }
  PyRef One() const { return Wrap(COne(d)); }
  PyRef Zero() const { return Wrap(CZero(d)); }
  PyRef Add(const PyRef& a, const PyRef& b) const {
    if (Bad(a) || Bad(b)) {
      failed = true;
      return PyRef();
    }
    return Wrap(CAdd(d, a.get(), b.get()));
  }
  PyRef Sub(const PyRef& a, const PyRef& b) const {
    if (Bad(a) || Bad(b)) {
      failed = true;
      return PyRef();
    }
    return Wrap(CSub(d, a.get(), b.get()));
  }
  PyRef Mul(const PyRef& a, const PyRef& b) const {
    if (Bad(a) || Bad(b)) {
      failed = true;
      return PyRef();
    }
    return Wrap(CMul(d, a.get(), b.get()));
  }
  PyRef Neg(const PyRef& a) const {
    if (Bad(a)) {
      failed = true;
      return PyRef();
    }
    return Wrap(CNeg(d, a.get()));
  }
  PyRef MulInt(const PyRef& a, int k) const {
    if (Bad(a)) {
      failed = true;
      return PyRef();
    }
    return Wrap(CMulInt(d, a.get(), k));
  }
  bool IsZero(const PyRef& a) const {
    if (Bad(a)) return false;
    return CIsZero(d, a.get());
  }
  bool Equal(const PyRef& a, const PyRef& b) const {
    if (Bad(a) || Bad(b)) return false;
    int r = PyObject_RichCompareBool(a.get(), b.get(), Py_EQ);
    if (r < 0) {
      failed = true;
      return false;
    }
    return r == 1;
  }
};

// Jacobian addition; in1/in2/out: 3 canonical coords. -1 on CPython failure.
int JacAdd(EcPointDescr* ec, PyObject* const* P, PyObject* const* Q,
           PyObject** out) {
  PyCoordOps f{ec};
  PyRef Pr[3] = {PyRef::Borrow(P[0]), PyRef::Borrow(P[1]), PyRef::Borrow(P[2])};
  PyRef Qr[3] = {PyRef::Borrow(Q[0]), PyRef::Borrow(Q[1]), PyRef::Borrow(Q[2])};
  PyRef R[3];
  ec_law::EcAddT<PyRef, PyCoordOps>(f, Pr, Qr, R);
  if (f.failed) return -1;
  for (int i = 0; i < 3; ++i) out[i] = R[i].release();
  return 0;
}

void CopyPoint(PyObject* const* in, PyObject** out, int n) {
  for (int i = 0; i < n; ++i) {
    Py_INCREF(in[i]);
    out[i] = in[i];
  }
}

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

// Group equality (cross-representative): 1 equal, 0 not, -1 on error.
int JacEqual(EcPointDescr* ec, PyObject* const* P, PyObject* const* Q) {
  PyCoordOps f{ec};
  PyRef Pr[3] = {PyRef::Borrow(P[0]), PyRef::Borrow(P[1]), PyRef::Borrow(P[2])};
  PyRef Qr[3] = {PyRef::Borrow(Q[0]), PyRef::Borrow(Q[1]), PyRef::Borrow(Q[2])};
  int r = ec_law::EcEqualT<PyRef, PyCoordOps>(f, Pr, Qr);
  return f.failed ? -1 : r;
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
    pyfield::LongAsBytesLE(scalar, buf, nbytes);
  }
  return static_cast<Py_ssize_t>(nbits);
}

// ret = scalar * point (canonical Jacobian coords). The scalar is a canonical
// integer (Montgomery already decoded by the caller), matching the legacy
// curve operator* which de-Montgomery's the scalar first.
int JacScalarMul(EcPointDescr* ec, PyObject* scalar, PyObject* const* point,
                 PyObject** out) {
  unsigned char buf[64];
  Py_ssize_t nbits = ScalarToBytesLE(scalar, buf, sizeof(buf));
  if (nbits < 0) return -1;
  PyCoordOps f{ec};
  PyRef P[3] = {PyRef::Borrow(point[0]), PyRef::Borrow(point[1]),
                PyRef::Borrow(point[2])};
  PyRef R[3];
  ec_law::EcScalarMulT<PyRef, PyCoordOps>(f, P, buf, static_cast<int>(nbits),
                                          R);
  if (f.failed) return -1;
  for (int i = 0; i < 3; ++i) out[i] = R[i].release();
  return 0;
}

// --- native fixed-width coordinate field ---------------------------------
// The CPython-int ops above allocate an int per coordinate operation. For
// Montgomery-stored points over a coordinate field whose base width the native
// kernels handle, run the same shared formulas in Montgomery space directly on
// the stored bytes (mont(x)*mont(y)*R^-1 = mont(x*y); add/sub are linear;
// non_residue folds with mont(nr)), so no decode/encode and no Python ints —
// byte-identical to the CPython tier because both instantiate the one
// ec_law formula body.

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
    if (pyfield::LongAsBytesLE(d->modulus, mod_le, cf.wb) < 0) {
      PyErr_Clear();
      return cf;
    }
    cf.fq = modarith::PrimeField::Make(mod_le, cf.wb, /*is_mont=*/true);
    if (!cf.fq.native) return cf;
    // The native EC temporaries are fixed kCoordBytes stack buffers (= 2*32,
    // the Fp2-over-256-bit max). degree<=2 and a native base width<=32 keep cb
    // in range; bail to the Python path otherwise rather than overflow.
    if (cf.cb > kCoordBytes) return cf;
    if (pyfield::LongAsBytesLE(d->r_mod_p, cf.one_le, cf.wb) < 0) {
      PyErr_Clear();
      return cf;
    }
    if (cf.degree == 2) {
      // mont(nr) = nr * R mod p; computed once here in Python, then native.
      PyObject* scaled = PyNumber_Multiply(d->non_residue, d->r_mod_p);
      PyObject* mn = scaled ? PyNumber_Remainder(scaled, d->modulus) : nullptr;
      Py_XDECREF(scaled);
      if (mn == nullptr || pyfield::LongAsBytesLE(mn, cf.mont_nr, cf.wb) < 0) {
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

// ec_law instantiation over the stored bytes: `ByteCoord` is one coordinate
// (fixed max-width buffer; only the descriptor's cb bytes are meaningful) and
// `ByteOps` adapts CoordField's kernels to the Ops concept.
struct ByteCoord {
  unsigned char b[kCoordBytes] = {};
};

struct ByteOps {
  const CoordField* cf;
  int coord_bytes() const { return cf->cb; }
  void Load(ByteCoord& c, const unsigned char* p) const {
    std::memcpy(c.b, p, cf->cb);
  }
  void Store(unsigned char* p, const ByteCoord& c) const {
    std::memcpy(p, c.b, cf->cb);
  }
  ByteCoord One() const {
    ByteCoord o;
    cf->SetOne(o.b);
    return o;
  }
  ByteCoord Zero() const {
    ByteCoord o;
    cf->SetZero(o.b);
    return o;
  }
  ByteCoord Add(const ByteCoord& a, const ByteCoord& b) const {
    ByteCoord o;
    cf->Add(a.b, b.b, o.b);
    return o;
  }
  ByteCoord Sub(const ByteCoord& a, const ByteCoord& b) const {
    ByteCoord o;
    cf->Sub(a.b, b.b, o.b);
    return o;
  }
  ByteCoord Neg(const ByteCoord& a) const {
    ByteCoord o;
    cf->Neg(a.b, o.b);
    return o;
  }
  ByteCoord Mul(const ByteCoord& a, const ByteCoord& b) const {
    ByteCoord o;
    cf->Mul(a.b, b.b, o.b);
    return o;
  }
  ByteCoord MulInt(const ByteCoord& a, int k) const {
    ByteCoord o;
    cf->MulInt(a.b, k, o.b);
    return o;
  }
  bool IsZero(const ByteCoord& a) const { return cf->IsZero(a.b); }
  bool Equal(const ByteCoord& a, const ByteCoord& b) const {
    return cf->Equal(a.b, b.b);
  }
};

// --- typed-limb coordinate field (256-bit) --------------------------------
// ByteOps round-trips every coordinate through memory per field operation. For
// the dominant case — a 256-bit Montgomery coordinate field (bn254 /
// bls12-381 / secp256k1 Fq, G1 and G2) — instantiate the same formulas on
// BigInt<4> values held in locals: load each coordinate from the stored bytes
// once, compute, store once. Same MontMul<4>/ModAdd/ModSub kernels as
// CoordField, so the result is byte-identical.

template <size_t N>
struct FqOpsBig {
  BigInt<N> p;
  BigInt<N> one_mont;  // mont(1) = R mod p
  uint64_t nprime = 0;
  bool spare = false;
  bool no_carry = false;
  BigInt<N> One() const { return one_mont; }
  BigInt<N> Zero() const { return BigInt<N>(0); }
  BigInt<N> Mul(const BigInt<N>& a, const BigInt<N>& b) const {
    BigInt<N> c;
    MontMul<N>(a, b, c, p, nprime, spare, no_carry);
    return c;
  }
  BigInt<N> Add(const BigInt<N>& a, const BigInt<N>& b) const {
    BigInt<N> c;
    ModAdd<BigInt<N>>(a, b, c, p, spare);
    return c;
  }
  BigInt<N> Sub(const BigInt<N>& a, const BigInt<N>& b) const {
    BigInt<N> c;
    ModSub<BigInt<N>>(a, b, c, p, spare);
    return c;
  }
  BigInt<N> Neg(const BigInt<N>& a) const { return Sub(BigInt<N>(0), a); }
  BigInt<N> MulInt(const BigInt<N>& a, int k) const {
    BigInt<N> c = a;
    for (int j = 1; j < k; ++j) c = Add(c, a);
    return c;
  }
  bool IsZero(const BigInt<N>& a) const { return a == BigInt<N>(0); }
  bool Equal(const BigInt<N>& a, const BigInt<N>& b) const { return a == b; }
  int coord_bytes() const { return static_cast<int>(N * 8); }
  void Load(BigInt<N>& c, const unsigned char* p) const {
    std::memcpy(&c[0], p, N * 8);
  }
  void Store(unsigned char* p, const BigInt<N>& c) const {
    std::memcpy(p, &c[0], N * 8);
  }
};

template <size_t N>
struct Fp2 {
  BigInt<N> c0, c1;
};

template <size_t N>
struct Fp2OpsBig {
  FqOpsBig<N> fq;
  BigInt<N> mont_nr;
  Fp2<N> One() const { return {fq.One(), BigInt<N>(0)}; }
  Fp2<N> Zero() const { return {BigInt<N>(0), BigInt<N>(0)}; }
  Fp2<N> Add(const Fp2<N>& a, const Fp2<N>& b) const {
    return {fq.Add(a.c0, b.c0), fq.Add(a.c1, b.c1)};
  }
  Fp2<N> Sub(const Fp2<N>& a, const Fp2<N>& b) const {
    return {fq.Sub(a.c0, b.c0), fq.Sub(a.c1, b.c1)};
  }
  Fp2<N> Neg(const Fp2<N>& a) const { return {fq.Neg(a.c0), fq.Neg(a.c1)}; }
  Fp2<N> MulInt(const Fp2<N>& a, int k) const {
    return {fq.MulInt(a.c0, k), fq.MulInt(a.c1, k)};
  }
  Fp2<N> Mul(const Fp2<N>& a, const Fp2<N>& b) const {
    // (a0 + a1 u)(b0 + b1 u) = (a0 b0 + nr a1 b1) + (a0 b1 + a1 b0) u.
    BigInt<N> a0b0 = fq.Mul(a.c0, b.c0);
    BigInt<N> a1b1 = fq.Mul(a.c1, b.c1);
    BigInt<N> c0 = fq.Add(a0b0, fq.Mul(mont_nr, a1b1));
    BigInt<N> c1 = fq.Add(fq.Mul(a.c0, b.c1), fq.Mul(a.c1, b.c0));
    return {c0, c1};
  }
  bool IsZero(const Fp2<N>& a) const {
    return fq.IsZero(a.c0) && fq.IsZero(a.c1);
  }
  bool Equal(const Fp2<N>& a, const Fp2<N>& b) const {
    return fq.Equal(a.c0, b.c0) && fq.Equal(a.c1, b.c1);
  }
  int coord_bytes() const { return static_cast<int>(2 * N * 8); }
  void Load(Fp2<N>& c, const unsigned char* p) const {
    fq.Load(c.c0, p);
    fq.Load(c.c1, p + N * 8);
  }
  void Store(unsigned char* p, const Fp2<N>& c) const {
    fq.Store(p, c.c0);
    fq.Store(p + N * 8, c.c1);
  }
};

// Build the typed 256-bit ops from the already-extracted CoordField constants.
FqOpsBig<4> MakeFqOps(const CoordField& cf) {
  FqOpsBig<4> f;
  f.p = cf.fq.p256;
  std::memcpy(&f.one_mont[0], cf.one_le, 4 * 8);
  f.nprime = cf.fq.nprime_neg;
  f.spare = cf.fq.spare;
  f.no_carry = cf.fq.no_carry;
  return f;
}
Fp2OpsBig<4> MakeFp2Ops(const CoordField& cf) {
  Fp2OpsBig<4> f;
  f.fq = MakeFqOps(cf);
  std::memcpy(&f.mont_nr[0], cf.mont_nr, 4 * 8);
  return f;
}

// Strided loop bodies over any ec_law Ops with Load/Store (ByteOps and the
// typed 256-bit ops alike).
template <typename C, typename Ops>
void LoadPt(const Ops& f, C out[3], const char* p) {
  const auto* u = reinterpret_cast<const unsigned char*>(p);
  for (int i = 0; i < 3; ++i) f.Load(out[i], u + i * f.coord_bytes());
}
template <typename C, typename Ops>
void StorePt(const Ops& f, char* p, const C in[3]) {
  auto* u = reinterpret_cast<unsigned char*>(p);
  for (int i = 0; i < 3; ++i) f.Store(u + i * f.coord_bytes(), in[i]);
}

template <typename C, typename Ops>
void RunBinT(const Ops& f, bool sub, char* a, char* b, char* o, npy_intp n,
             const npy_intp* st) {
  for (npy_intp i = 0; i < n; ++i) {
    C P[3], Q[3], R[3];
    LoadPt<C>(f, P, a);
    LoadPt<C>(f, Q, b);
    if (sub) {
      C nq[3] = {Q[0], f.Neg(Q[1]), Q[2]};
      ec_law::EcAddT<C, Ops>(f, P, nq, R);
    } else {
      ec_law::EcAddT<C, Ops>(f, P, Q, R);
    }
    StorePt<C>(f, o, R);
    a += st[0];
    b += st[1];
    o += st[2];
  }
}
template <typename C, typename Ops>
void RunNegT(const Ops& f, char* a, char* o, npy_intp n, const npy_intp* st,
             int num_coords) {
  for (npy_intp i = 0; i < n; ++i) {
    for (int k = 0; k < num_coords; ++k) {
      C c;
      f.Load(c,
             reinterpret_cast<const unsigned char*>(a) + k * f.coord_bytes());
      if (k == 1) c = f.Neg(c);
      f.Store(reinterpret_cast<unsigned char*>(o) + k * f.coord_bytes(), c);
    }
    a += st[0];
    o += st[1];
  }
}
template <typename C, typename Ops>
void RunCmpT(const Ops& f, bool negate, char* a, char* b, char* o, npy_intp n,
             const npy_intp* st) {
  for (npy_intp i = 0; i < n; ++i) {
    C P[3], Q[3];
    LoadPt<C>(f, P, a);
    LoadPt<C>(f, Q, b);
    int eq = ec_law::EcEqualT<C, Ops>(f, P, Q);
    *reinterpret_cast<npy_bool*>(o) = (negate ? !eq : eq) ? 1 : 0;
    a += st[0];
    b += st[1];
    o += st[2];
  }
}
template <typename C, typename Ops>
int RunScalarT(const Ops& f, PyArray_Descr* scalar_descr, char* s, char* pt,
               char* o, npy_intp n, npy_intp s_stride, npy_intp pt_stride,
               npy_intp o_stride) {
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* scalar =
        PrimeFieldValue(reinterpret_cast<PyObject*>(scalar_descr), s);
    if (scalar == nullptr) return -1;
    unsigned char buf[64];
    Py_ssize_t nbits = ScalarToBytesLE(scalar, buf, sizeof(buf));
    Py_DECREF(scalar);
    if (nbits < 0) return -1;
    C point[3], ret[3];
    LoadPt<C>(f, point, pt);
    ec_law::EcScalarMulT<C, Ops>(f, point, buf, static_cast<int>(nbits), ret);
    StorePt<C>(f, o, ret);
    s += s_stride;
    pt += pt_stride;
    o += o_stride;
  }
  return 0;
}

// --- descriptor lifecycle ------------------------------------------------

npy_bool NonZero(void* data, void* arr);

PyArray_Descr* MakeDescr(PyObject* modulus, int base_width_bytes,
                         int num_coords, int is_montgomery, PyObject* r,
                         PyObject* rinv, int coord_degree,
                         PyObject* non_residue, PyObject* generator) {
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
  Py_XINCREF(generator);
  d->generator = generator;
  d->coord_degree = static_cast<uint8_t>(coord_degree);
  d->base_width_bytes = static_cast<uint8_t>(base_width_bytes);
  d->num_coords = static_cast<uint8_t>(num_coords);
  d->is_montgomery = static_cast<uint8_t>(is_montgomery ? 1 : 0);
  // Group-law tier: decided once here; the ufunc loops only switch on it.
  d->native = new CoordField(CoordField::Make(d));
  d->tier = !d->native->ok        ? Tier::kPy
            : d->native->wb == 32 ? Tier::kTyped256
                                  : Tier::kByte;
  PyArray_Descr* base = &d->base;
  base->kind = 'V';
  base->type = 'j';
  base->byteorder = '=';
  base->flags = NPY_USE_GETITEM | NPY_USE_SETITEM;
  base->elsize = base_width_bytes * coord_degree * num_coords;
  base->alignment = base_width_bytes <= 8 ? base_width_bytes : 8;
  // See field_dtype.cc: the ArrFuncs nonzero entry is set per descriptor
  // because the DType-spec slot id is not stable across supported numpy
  // versions.
  PyDataType_GetArrFuncs(base)->nonzero = NonZero;
  return base;
}

void Descr_dealloc(PyObject* self) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  Py_XDECREF(d->modulus);
  Py_XDECREF(d->r_mod_p);
  Py_XDECREF(d->rinv_mod_p);
  Py_XDECREF(d->non_residue);
  Py_XDECREF(d->generator);
  delete d->native;
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

PyObject* Get_modulus(PyObject* self, void* /*closure*/) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  Py_INCREF(d->modulus);
  return d->modulus;
}

PyObject* Get_num_coords(PyObject* self, void* /*closure*/) {
  return PyLong_FromLong(
      AsEc(reinterpret_cast<PyArray_Descr*>(self))->num_coords);
}

PyObject* Get_coord_degree(PyObject* self, void* /*closure*/) {
  return PyLong_FromLong(
      AsEc(reinterpret_cast<PyArray_Descr*>(self))->coord_degree);
}

PyObject* Get_non_residue(PyObject* self, void* /*closure*/) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  if (d->non_residue == nullptr) Py_RETURN_NONE;
  Py_INCREF(d->non_residue);
  return d->non_residue;
}

PyObject* Get_base_width_bits(PyObject* self, void* /*closure*/) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  return PyLong_FromLong(d->base_width_bytes * 8);
}

PyObject* Get_is_montgomery(PyObject* self, void* /*closure*/) {
  return PyBool_FromLong(
      AsEc(reinterpret_cast<PyArray_Descr*>(self))->is_montgomery);
}

PyObject* Get_generator(PyObject* self, void* /*closure*/) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  if (d->generator == nullptr) Py_RETURN_NONE;
  Py_INCREF(d->generator);
  return d->generator;
}

PyGetSetDef Descr_getset[] = {
    {"modulus", Get_modulus, nullptr, "base field modulus", nullptr},
    {"generator", Get_generator, nullptr,
     "generator coordinates, or None when the dtype carries no generator",
     nullptr},
    {"num_coords", Get_num_coords, nullptr, "2 affine, 3 Jacobian, 4 xyzz",
     nullptr},
    {"coord_degree", Get_coord_degree, nullptr,
     "1 = prime coordinate field (G1), 2 = Fp2 (G2)", nullptr},
    {"non_residue", Get_non_residue, nullptr, "Fp2 non-residue (None for G1)",
     nullptr},
    {"base_width_bits", Get_base_width_bits, nullptr,
     "base-field storage width in bits", nullptr},
    {"is_montgomery", Get_is_montgomery, nullptr,
     "True when coordinates are Montgomery-encoded", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyObject* Descr_reduce(PyObject* self, PyObject* /*unused*/) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  PyObject* module = PyImport_ImportModule("zk_dtypes._zk_dtypes_ext");
  if (module == nullptr) return nullptr;
  PyObject* factory = PyObject_GetAttrString(module, "ec_point_descr");
  Py_DECREF(module);
  if (factory == nullptr) return nullptr;
  PyObject* args = Py_BuildValue(
      "(OiiiOOiOO)", d->modulus, d->base_width_bytes * 8,
      static_cast<int>(d->num_coords), static_cast<int>(d->is_montgomery),
      d->r_mod_p ? d->r_mod_p : Py_None,
      d->rinv_mod_p ? d->rinv_mod_p : Py_None,
      static_cast<int>(d->coord_degree),
      d->non_residue ? d->non_residue : Py_None,
      d->generator ? d->generator : Py_None);
  if (args == nullptr) {
    Py_DECREF(factory);
    return nullptr;
  }
  return Py_BuildValue("(NN)", factory, args);
}

PyMethodDef Descr_methods[] = {
    {"__reduce__", Descr_reduce, METH_NOARGS,
     "Pickle support: rebuild through the descriptor factory."},
    {nullptr, nullptr, 0, nullptr},
};

PyObject* Descr_repr(PyObject* self) {
  EcPointDescr* d = AsEc(reinterpret_cast<PyArray_Descr*>(self));
  if (d->coord_degree == 2) {
    return PyUnicode_FromFormat(
        "EcPointDType(modulus=%R, coords=%d, base_width=%d, mont=%d, "
        "degree=2, non_residue=%R)",
        d->modulus, static_cast<int>(d->num_coords), d->base_width_bytes * 8,
        static_cast<int>(d->is_montgomery), d->non_residue);
  }
  return PyUnicode_FromFormat(
      "EcPointDType(modulus=%R, coords=%d, base_width=%d, mont=%d, degree=1)",
      d->modulus, static_cast<int>(d->num_coords), d->base_width_bytes * 8,
      static_cast<int>(d->is_montgomery));
}

// --- NEP-42 DType slots --------------------------------------------------

PyArray_Descr* DefaultDescr(PyArray_DTypeMeta* /*cls*/) {
  PyObject* two = PyLong_FromLong(2);
  if (two == nullptr) {
    return nullptr;
  }
  PyArray_Descr* d =
      MakeDescr(two, 4, 3, 0, nullptr, nullptr, 1, nullptr, nullptr);
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

// A point is "zero" iff it is the group identity. Jacobian/xyzz encode the
// identity with a zero (third) projective coordinate; affine cannot encode
// it, so every affine element is nonzero. numpy calls this for np.nonzero /
// truthiness; leaving the slot NULL segfaults.
npy_bool NonZero(void* data, void* arr) {
  EcPointDescr* d = AsEc(PyArray_DESCR(reinterpret_cast<PyArrayObject*>(arr)));
  if (d->num_coords < 3) return 1;
  const int cb = d->base_width_bytes * d->coord_degree;
  const auto* z = static_cast<const unsigned char*>(data) + 2 * cb;
  for (int i = 0; i < cb; ++i) {
    if (z[i]) return 1;
  }
  return 0;
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

// Normalizes one setitem coordinate into the form EncodeCoord expects, the
// mirror of what GetItem/DecodeCoord produce: a canonical int for a degree-1
// (Fq) coordinate, or a `coord_degree`-tuple of ints for an Fp2 coordinate
// (G2). Returns a new reference, or NULL with an exception set.
PyObject* CoordFromPy(EcPointDescr* d, PyObject* item) {
  if (d->coord_degree == 1) {
    return PyNumber_Index(item);
  }
  PyObject* seq = PySequence_Fast(item, "Fp2 coordinate needs (c0, c1)");
  if (seq == nullptr) {
    return nullptr;
  }
  if (PySequence_Fast_GET_SIZE(seq) != d->coord_degree) {
    PyErr_Format(PyExc_ValueError,
                 "Fp2 coordinate needs %d components, got %zd", d->coord_degree,
                 PySequence_Fast_GET_SIZE(seq));
    Py_DECREF(seq);
    return nullptr;
  }
  PyObject* tuple = PyTuple_New(d->coord_degree);
  if (tuple == nullptr) {
    Py_DECREF(seq);
    return nullptr;
  }
  for (int k = 0; k < d->coord_degree; ++k) {
    PyObject* c = PyNumber_Index(PySequence_Fast_GET_ITEM(seq, k));
    if (c == nullptr) {
      Py_DECREF(seq);
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, k, c);
  }
  Py_DECREF(seq);
  return tuple;
}

// setitem accepts a length-num_coords sequence of coordinates; each coordinate
// is an int (G1) or an (c0, c1) pair (G2 Fp2), matching getitem's output.
int SetItem(PyArray_Descr* descr, PyObject* obj, char* dataptr) {
  EcPointDescr* d = AsEc(descr);
  if (PyIndex_Check(obj)) {  // n -> n*G, the legacy dtypes' int construction
    if (d->generator == nullptr) {
      PyErr_SetString(PyExc_TypeError,
                      "this point dtype carries no generator; build it with "
                      "ec_point_descr(..., generator=(x, y, z)) to construct "
                      "points from integers");
      return -1;
    }
    if (d->num_coords != 3) {
      PyErr_SetString(PyExc_TypeError,
                      "integer construction requires the Jacobian "
                      "representation");
      return -1;
    }
    PyObject* scalar = PyNumber_Index(obj);
    if (scalar == nullptr) return -1;
    PyObject* gen[kMaxCoords] = {nullptr};
    int rc = -1;
    for (int i = 0; i < d->num_coords; ++i) {
      gen[i] = CoordFromPy(d, PyTuple_GET_ITEM(d->generator, i));
      if (gen[i] == nullptr) goto int_done;
    }
    {
      PyObject* out[3];
      if (JacScalarMul(d, scalar, gen, out) < 0) goto int_done;
      rc = EncodePoint(d, dataptr, out);
      for (int i = 0; i < 3; ++i) Py_DECREF(out[i]);
    }
  int_done:
    Py_DECREF(scalar);
    for (int i = 0; i < d->num_coords; ++i) Py_XDECREF(gen[i]);
    return rc;
  }
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
    coords[i] = CoordFromPy(d, PySequence_Fast_GET_ITEM(seq, i));
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
  // Only representation (num_coords) and storage form (Montgomery/canonical)
  // may differ; a cast across coordinate fields would silently decode under
  // one field and re-encode under another.
  EcPointDescr* f = AsEc(from);
  EcPointDescr* t = AsEc(to);
  if (f->base_width_bytes != t->base_width_bytes ||
      f->coord_degree != t->coord_degree ||
      PyObject_RichCompareBool(f->modulus, t->modulus, Py_EQ) != 1 ||
      (f->coord_degree == 2 &&
       PyObject_RichCompareBool(f->non_residue, t->non_residue, Py_EQ) != 1)) {
    PyErr_SetString(PyExc_TypeError,
                    "cannot cast between points of different curves");
    return static_cast<NPY_CASTING>(-1);
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
  PyObject* gen_obj = nullptr;
  if (!PyArg_ParseTuple(args, "Oiii|OOiOO", &modulus_obj, &base_width_bits,
                        &num_coords, &is_montgomery, &r_obj, &rinv_obj,
                        &coord_degree, &nr_obj, &gen_obj)) {
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
  // Primality stays the caller's contract (a Miller-Rabin gate belongs in a
  // Python-level factory), but an even or tiny modulus breaks the Montgomery
  // inverse outright — reject the cases that can never be a coordinate field.
  {
    int is_odd = -1;
    PyObject* one = PyLong_FromLong(1);
    if (one != nullptr) {
      PyObject* low = PyNumber_And(modulus_obj, one);
      if (low != nullptr) {
        is_odd = PyObject_IsTrue(low);
        Py_DECREF(low);
      }
      Py_DECREF(one);
    }
    if (is_odd < 0) return nullptr;
    if (is_odd == 0) {
      PyErr_SetString(PyExc_ValueError,
                      "modulus must be an odd prime (coordinate field)");
      return nullptr;
    }
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
  PyObject* gen = nullptr;
  if (gen_obj != nullptr && gen_obj != Py_None) {
    gen = PySequence_Tuple(gen_obj);
    if (gen == nullptr || PyTuple_GET_SIZE(gen) != num_coords) {
      if (gen != nullptr) {
        PyErr_Format(PyExc_ValueError,
                     "generator needs %d coordinates, got %zd", num_coords,
                     PyTuple_GET_SIZE(gen));
      }
      Py_DECREF(modulus);
      Py_XDECREF(r);
      Py_XDECREF(rinv);
      Py_XDECREF(nr);
      Py_XDECREF(gen);
      return nullptr;
    }
  }
  PyArray_Descr* d = MakeDescr(modulus, base_width_bits / 8, num_coords,
                               is_montgomery, r, rinv, coord_degree, nr, gen);
  Py_DECREF(modulus);
  Py_XDECREF(r);
  Py_XDECREF(rinv);
  Py_XDECREF(nr);
  Py_XDECREF(gen);
  return reinterpret_cast<PyObject*>(d);
}

PyMethodDef kModuleMethods[] = {
    {"ec_point_descr", MakeEcPointDescrPy, METH_VARARGS,
     "ec_point_descr(modulus, base_width_bits, num_coords, is_montgomery"
     "[, r_mod_p, rinv_mod_p, coord_degree, non_residue, generator])"
     " -> dtype\n\n"
     "Build a parametric short-Weierstrass (a=0) EC-point descriptor over a\n"
     "prime (G1) or Fp2 (G2) coordinate field; num_coords selects the\n"
     "representation (2 affine, 3 Jacobian, 4 xyzz)."},
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
  // An explicit out= must be the same curve and representation: the loop
  // sizes every write from descriptors[0].
  if (given[2] != nullptr && !SameCurve(AsEc(given[0]), AsEc(given[2]))) {
    PyErr_SetString(PyExc_TypeError, "point op output requires the same curve");
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
  // Same-curve requirement as BinResolve, for an explicit out=.
  if (given[1] != nullptr && !SameCurve(AsEc(given[0]), AsEc(given[1]))) {
    PyErr_SetString(PyExc_TypeError, "point op output requires the same curve");
    return static_cast<NPY_CASTING>(-1);
  }
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
  const CoordField& cf = *d->native;
  if (d->tier == Tier::kTyped256) {
    if (cf.degree == 1) {
      RunBinT<BigInt<4>>(MakeFqOps(cf), op == BinOp::kSub, a, b, o, n, strides);
    } else {
      RunBinT<Fp2<4>>(MakeFp2Ops(cf), op == BinOp::kSub, a, b, o, n, strides);
    }
    return 0;
  }
  if (d->tier == Tier::kByte) {  // num_coords == 3 guaranteed by BinResolve
    RunBinT<ByteCoord>(ByteOps{&cf}, op == BinOp::kSub, a, b, o, n, strides);
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
  const CoordField& cf = *d->native;
  if (d->tier == Tier::kTyped256) {  // negate flips Y per coord
    if (cf.degree == 1) {
      RunNegT<BigInt<4>>(MakeFqOps(cf), a, o, n, strides, d->num_coords);
    } else {
      RunNegT<Fp2<4>>(MakeFp2Ops(cf), a, o, n, strides, d->num_coords);
    }
    return 0;
  }
  if (d->tier == Tier::kByte) {
    RunNegT<ByteCoord>(ByteOps{&cf}, a, o, n, strides, d->num_coords);
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
  PyArray_Descr* out_descr =
      given[2] != nullptr ? given[2] : PyArray_DescrFromType(NPY_BOOL);
  if (out_descr == nullptr) {
    return static_cast<NPY_CASTING>(-1);
  }
  if (given[2] != nullptr) {
    Py_INCREF(out_descr);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  Py_INCREF(given[1]);
  loop[1] = given[1];
  loop[2] = out_descr;
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
  const CoordField& cf = *d->native;
  if (d->tier == Tier::kTyped256) {
    if (cf.degree == 1) {
      RunCmpT<BigInt<4>>(MakeFqOps(cf), negate, a, b, o, n, strides);
    } else {
      RunCmpT<Fp2<4>>(MakeFp2Ops(cf), negate, a, b, o, n, strides);
    }
    return 0;
  }
  if (d->tier == Tier::kByte) {  // num_coords == 3 guaranteed by CmpResolve
    RunCmpT<ByteCoord>(ByteOps{&cf}, negate, a, b, o, n, strides);
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
  // PyArray_DescrFromType returns a new reference; we only need its DType meta
  // (Py_TYPE, borrowed and immortal), so drop the descr ref.
  PyArray_Descr* bool_descr = PyArray_DescrFromType(NPY_BOOL);
  if (bool_descr == nullptr) {
    return false;
  }
  PyArray_DTypeMeta* booldt =
      reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(bool_descr));
  Py_DECREF(bool_descr);
  PyArray_DTypeMeta* dtypes[3] = {&EcPointDType, &EcPointDType, booldt};
  return nep42::AddUfuncLoop(numpy, name, "ec_point_compare", 2, dtypes,
                             reinterpret_cast<void*>(CmpResolve),
                             reinterpret_cast<void*>(loop));
}

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
  const CoordField& cf = *d->native;
  if (d->tier == Tier::kTyped256 && d->num_coords == 3) {
    if (cf.degree == 1) {
      return RunScalarT<BigInt<4>>(MakeFqOps(cf), scalar_descr, s, pt, o, n,
                                   s_stride, pt_stride, strides[2]);
    }
    return RunScalarT<Fp2<4>>(MakeFp2Ops(cf), scalar_descr, s, pt, o, n,
                              s_stride, pt_stride, strides[2]);
  }
  if (d->tier == Tier::kByte && d->num_coords == 3) {
    return RunScalarT<ByteCoord>(ByteOps{&cf}, scalar_descr, s, pt, o, n,
                                 s_stride, pt_stride, strides[2]);
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
  PyArray_DTypeMeta* field =
      reinterpret_cast<PyArray_DTypeMeta*>(FieldDTypeMetaObject());
  PyArray_DTypeMeta* sf[3] = {field, &EcPointDType, &EcPointDType};
  PyArray_DTypeMeta* fs[3] = {&EcPointDType, field, &EcPointDType};
  return nep42::AddUfuncLoop(numpy, "multiply", "ec_point_scalar_mul", 2, sf,
                             reinterpret_cast<void*>(ScalarMulResolve),
                             reinterpret_cast<void*>(ScalarMulLoop<true>)) &&
         nep42::AddUfuncLoop(numpy, "multiply", "ec_point_scalar_mul", 2, fs,
                             reinterpret_cast<void*>(ScalarMulResolve),
                             reinterpret_cast<void*>(ScalarMulLoop<false>));
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
  type->tp_getset = Descr_getset;
  type->tp_methods = Descr_methods;
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
  // The cast loop runs CPython API (decode/re-encode through Python
  // ints), so numpy must keep the GIL held around it.
  copy_cast.flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(
      NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI);
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
  PyArray_DTypeMeta* ec3[3] = {&EcPointDType, &EcPointDType, &EcPointDType};
  PyArray_DTypeMeta* ec2[2] = {&EcPointDType, &EcPointDType};
  bool ok =
      nep42::AddUfuncLoop(numpy, "add", "ec_point_binop", 2, ec3,
                          reinterpret_cast<void*>(BinResolve),
                          reinterpret_cast<void*>(BinLoop<BinOp::kAdd>)) &&
      nep42::AddUfuncLoop(numpy, "subtract", "ec_point_binop", 2, ec3,
                          reinterpret_cast<void*>(BinResolve),
                          reinterpret_cast<void*>(BinLoop<BinOp::kSub>)) &&
      nep42::AddUfuncLoop(numpy, "negative", "ec_point_negate", 1, ec2,
                          reinterpret_cast<void*>(UnaryResolve),
                          reinterpret_cast<void*>(NegLoop)) &&
      AddCmpLoop(numpy, "equal", CmpLoop<false>) &&
      AddCmpLoop(numpy, "not_equal", CmpLoop<true>) && AddScalarMulLoops(numpy);
  Py_DECREF(numpy);
  return ok;
}

}  // namespace zk_dtypes
