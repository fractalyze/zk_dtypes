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

// Parametric finite-field numpy DType (NEP-42). One DType class serves every
// field — prime (degree 1) and binomial extension Fp[X]/(X^k - non_residue)
// (degree k) — with modulus / degree / non_residue / storage form carried on
// each descriptor *instance*, so a user-defined field needs no new C++ type.
// This is the host counterpart to the parametric `algebraic*<...>` element type
// in xla_fork; the compiler stack below is already modulus-generic.
//
// Storage matches the legacy named types byte-for-byte: an element is `degree`
// base-field coefficients, constant term first, each coefficient stored at
// base width little-endian and (for Montgomery storage) encoded as `c*R mod p`
// with R = 2^base_width. Arithmetic is host Python-C-API bignum (correct and
// width-generic; host eager arithmetic is not the perf path — the device is).

// numpy.h must precede every other numpy header (it sets the API symbol) and
// NPY_TARGET_VERSION must precede numpyconfig.h; the associated header pulls in
// only <Python.h>, so it can lead. Keep this order — do not let clang-format
// sort it.
// clang-format off
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "zk_dtypes/_src/field_dtype.h"

#include <cstdint>
#include <cstring>

#include "zk_dtypes/_src/field_modarith.h"
#include "zk_dtypes/_src/numpy.h"
#include "numpy/dtype_api.h"
#include "numpy/ndarraytypes.h"
// clang-format on

namespace zk_dtypes {
namespace {

// Extension degree is tiny in practice (<= 4 for the shipped families); cap the
// stack coefficient buffers generously.
constexpr int kMaxDegree = 16;

// A field is one of two structurally-distinct families behind one DType. Both
// are Fields (+ - * /); they share no arithmetic, so the loops dispatch on
// kind.
enum FieldKind : uint8_t {
  kOddField = 0,     // prime or binomial extension over an odd prime p
  kBinaryTower = 1,  // GF(2^(2^level)) tower, characteristic 2
};

struct FieldDescr {
  PyArray_Descr base;
  // Odd-field parameters (kBinaryTower leaves these NULL):
  PyObject* modulus;  // owned: base prime p
  PyObject*
      non_residue;    // owned (degree > 1): X^degree = non_residue; else NULL
  PyObject* r_mod_p;  // owned (Montgomery): R = 2^base_width mod p; else NULL
  PyObject* rinv_mod_p;  // owned (Montgomery): R^-1 mod p; else NULL
  // Binary-tower parameters (kOddField leaves value_mask NULL):
  PyObject* value_mask;  // owned (binary): 2^(2^level) - 1
  uint8_t kind;
  uint8_t
      base_width_bytes;  // odd: per-coefficient width; binary: storage width
  uint8_t degree;        // odd: extension degree (1 = prime); binary: 1
  uint8_t tower_level;   // binary tower level
  uint8_t is_montgomery;
};

PyArray_DTypeMeta FieldDType = {};
PyTypeObject FieldScalar_Type = {};

FieldDescr* AsField(PyArray_Descr* d) {
  return reinterpret_cast<FieldDescr*>(d);
}

// --- binary tower helpers (characteristic 2) ----------------------------

// 2^bits - 1 as a Python int.
PyObject* Mask(int bits) {
  PyObject* one = PyLong_FromLong(1);
  PyObject* shift = PyLong_FromLong(bits);
  PyObject* shifted = (one && shift) ? PyNumber_Lshift(one, shift) : nullptr;
  Py_XDECREF(one);
  Py_XDECREF(shift);
  if (shifted == nullptr) {
    return nullptr;
  }
  PyObject* minus_one = PyLong_FromLong(1);
  PyObject* mask = minus_one ? PyNumber_Subtract(shifted, minus_one) : nullptr;
  Py_DECREF(shifted);
  Py_XDECREF(minus_one);
  return mask;
}

// Recursive Karatsuba tower multiply in GF(2^(2^level)) =
// GF(2^(2^(level-1)))[X]/(X^2 + X + alpha_level), alpha = 1 << (2^(level-1)-1).
PyObject* TowerMul(int level, PyObject* a, PyObject* b) {
  if (level == 0) {
    PyObject* ab = PyNumber_And(a, b);
    if (ab == nullptr) {
      return nullptr;
    }
    PyObject* one = PyLong_FromLong(1);
    PyObject* r = one ? PyNumber_And(ab, one) : nullptr;
    Py_DECREF(ab);
    Py_XDECREF(one);
    return r;
  }
  const int sb = 1 << (level - 1);
  PyObject* submask = Mask(sb);
  PyObject* shift = PyLong_FromLong(sb);
  PyObject* one = PyLong_FromLong(1);
  PyObject* abits = PyLong_FromLong(sb - 1);
  PyObject* alpha = (one && abits) ? PyNumber_Lshift(one, abits) : nullptr;
  Py_XDECREF(one);
  Py_XDECREF(abits);
  PyObject *a0 = nullptr, *a1 = nullptr, *b0 = nullptr, *b1 = nullptr;
  PyObject *a0b0 = nullptr, *a1b1 = nullptr, *a1b1a = nullptr;
  PyObject *axor = nullptr, *bxor = nullptr, *mid = nullptr;
  PyObject *c0 = nullptr, *c1 = nullptr, *c1sh = nullptr, *result = nullptr;
  if (!submask || !shift || !alpha) {
    goto done;
  }
  a0 = PyNumber_And(a, submask);
  b0 = PyNumber_And(b, submask);
  {
    PyObject* ah = PyNumber_Rshift(a, shift);
    a1 = ah ? PyNumber_And(ah, submask) : nullptr;
    Py_XDECREF(ah);
    PyObject* bh = PyNumber_Rshift(b, shift);
    b1 = bh ? PyNumber_And(bh, submask) : nullptr;
    Py_XDECREF(bh);
  }
  if (!a0 || !a1 || !b0 || !b1) {
    goto done;
  }
  a0b0 = TowerMul(level - 1, a0, b0);
  a1b1 = TowerMul(level - 1, a1, b1);
  if (!a0b0 || !a1b1) {
    goto done;
  }
  a1b1a = TowerMul(level - 1, a1b1, alpha);
  if (!a1b1a) {
    goto done;
  }
  c0 = PyNumber_Xor(a0b0, a1b1a);
  axor = PyNumber_Xor(a0, a1);
  bxor = PyNumber_Xor(b0, b1);
  if (!c0 || !axor || !bxor) {
    goto done;
  }
  mid = TowerMul(level - 1, axor, bxor);
  if (!mid) {
    goto done;
  }
  c1 = PyNumber_Xor(mid, a0b0);
  if (!c1) {
    goto done;
  }
  c1sh = PyNumber_Lshift(c1, shift);
  if (!c1sh) {
    goto done;
  }
  result = PyNumber_Or(c0, c1sh);
done:
  Py_XDECREF(submask);
  Py_XDECREF(shift);
  Py_XDECREF(alpha);
  Py_XDECREF(a0);
  Py_XDECREF(a1);
  Py_XDECREF(b0);
  Py_XDECREF(b1);
  Py_XDECREF(a0b0);
  Py_XDECREF(a1b1);
  Py_XDECREF(a1b1a);
  Py_XDECREF(axor);
  Py_XDECREF(bxor);
  Py_XDECREF(mid);
  Py_XDECREF(c0);
  Py_XDECREF(c1);
  Py_XDECREF(c1sh);
  return result;
}

// --- per-coefficient encode / decode (canonical or Montgomery) ----------

// Reads one base-field coefficient at `slot` and returns its canonical value.
PyObject* DecodeCoeff(FieldDescr* d, const char* slot) {
  PyObject* stored = _PyLong_FromByteArray(
      reinterpret_cast<const unsigned char*>(slot), d->base_width_bytes,
      /*little_endian=*/1, /*is_signed=*/0);
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

// Writes canonical value `value` into the base-field coefficient at `slot`.
int EncodeCoeff(FieldDescr* d, char* slot, PyObject* value) {
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
                               d->base_width_bytes, /*little_endian=*/1,
                               /*is_signed=*/0);
  Py_DECREF(rem);
  return rc < 0 ? -1 : 0;
}

// Fills `out[0..degree-1]` with new canonical-value references. On failure sets
// a Python error, clears any it set, and returns -1.
int DecodeElement(FieldDescr* d, const char* ptr, PyObject** out) {
  for (int i = 0; i < d->degree; ++i) {
    out[i] = DecodeCoeff(d, ptr + i * d->base_width_bytes);
    if (out[i] == nullptr) {
      for (int j = 0; j < i; ++j) {
        Py_DECREF(out[j]);
      }
      return -1;
    }
  }
  return 0;
}

int EncodeElement(FieldDescr* d, char* ptr, PyObject* const* coeffs) {
  for (int i = 0; i < d->degree; ++i) {
    if (EncodeCoeff(d, ptr + i * d->base_width_bytes, coeffs[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

// --- descriptor lifecycle ------------------------------------------------

PyArray_Descr* MakeDescr(PyObject* modulus, PyObject* non_residue, int degree,
                         int base_width_bytes, int is_montgomery, PyObject* r,
                         PyObject* rinv) {
  auto* d = reinterpret_cast<FieldDescr*>(PyArrayDescr_Type.tp_new(
      reinterpret_cast<PyTypeObject*>(&FieldDType), nullptr, nullptr));
  if (d == nullptr) {
    return nullptr;
  }
  Py_INCREF(modulus);
  d->modulus = modulus;
  Py_XINCREF(non_residue);
  d->non_residue = non_residue;
  Py_XINCREF(r);
  d->r_mod_p = r;
  Py_XINCREF(rinv);
  d->rinv_mod_p = rinv;
  d->value_mask = nullptr;
  d->kind = kOddField;
  d->base_width_bytes = static_cast<uint8_t>(base_width_bytes);
  d->degree = static_cast<uint8_t>(degree);
  d->tower_level = 0;
  d->is_montgomery = static_cast<uint8_t>(is_montgomery ? 1 : 0);
  PyArray_Descr* base = &d->base;
  base->kind = 'V';
  base->type = 'F';
  base->byteorder = '=';
  // Route scalar access (arr[i]) through the ArrFuncs getitem/setitem rather
  // than the copyswap path, which is NULL for this minimal scalar type.
  base->flags = NPY_USE_GETITEM | NPY_USE_SETITEM;
  base->elsize = base_width_bytes * degree;
  base->alignment = base_width_bytes <= 8 ? base_width_bytes : 8;
  return base;
}

// Binary tower GF(2^(2^level)). All odd-field params stay NULL.
PyArray_Descr* MakeBinaryDescr(int tower_level, int width_bytes) {
  auto* d = reinterpret_cast<FieldDescr*>(PyArrayDescr_Type.tp_new(
      reinterpret_cast<PyTypeObject*>(&FieldDType), nullptr, nullptr));
  if (d == nullptr) {
    return nullptr;
  }
  d->value_mask = Mask(1 << tower_level);  // 2^(2^level) - 1
  if (d->value_mask == nullptr) {
    Py_DECREF(d);
    return nullptr;
  }
  d->modulus = nullptr;
  d->non_residue = nullptr;
  d->r_mod_p = nullptr;
  d->rinv_mod_p = nullptr;
  d->kind = kBinaryTower;
  d->base_width_bytes = static_cast<uint8_t>(width_bytes);
  d->degree = 1;
  d->tower_level = static_cast<uint8_t>(tower_level);
  d->is_montgomery = 0;
  PyArray_Descr* base = &d->base;
  base->kind = 'V';
  base->type = 'B';
  base->byteorder = '=';
  // Route scalar access (arr[i]) through the ArrFuncs getitem/setitem rather
  // than the copyswap path, which is NULL for this minimal scalar type.
  base->flags = NPY_USE_GETITEM | NPY_USE_SETITEM;
  base->elsize = width_bytes;
  base->alignment = width_bytes <= 8 ? width_bytes : 8;
  return base;
}

void Descr_dealloc(PyObject* self) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  Py_XDECREF(d->modulus);
  Py_XDECREF(d->non_residue);
  Py_XDECREF(d->r_mod_p);
  Py_XDECREF(d->rinv_mod_p);
  Py_XDECREF(d->value_mask);
  PyArrayDescr_Type.tp_dealloc(self);
}

PyObject* DType_new(PyTypeObject* /*cls*/, PyObject* /*args*/,
                    PyObject* /*kwds*/) {
  PyErr_SetString(
      PyExc_TypeError,
      "construct a field via zk_dtypes.prime_field(p) / "
      "zk_dtypes.extension_field(...), not FieldDType(...) directly");
  return nullptr;
}

PyObject* Descr_repr(PyObject* self) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  if (d->kind == kBinaryTower) {
    return PyUnicode_FromFormat("FieldDType(binary_tower_level=%d, bits=%d)",
                                static_cast<int>(d->tower_level),
                                1 << d->tower_level);
  }
  if (d->degree == 1) {
    return PyUnicode_FromFormat("FieldDType(modulus=%R, width=%d, mont=%d)",
                                d->modulus, d->base_width_bytes * 8,
                                static_cast<int>(d->is_montgomery));
  }
  return PyUnicode_FromFormat(
      "FieldDType(modulus=%R, degree=%d, non_residue=%R, base_width=%d, "
      "mont=%d)",
      d->modulus, static_cast<int>(d->degree), d->non_residue,
      d->base_width_bytes * 8, static_cast<int>(d->is_montgomery));
}

// --- NEP-42 DType slots --------------------------------------------------

PyArray_Descr* DefaultDescr(PyArray_DTypeMeta* /*cls*/) {
  PyObject* two = PyLong_FromLong(2);
  if (two == nullptr) {
    return nullptr;
  }
  PyArray_Descr* d =
      MakeDescr(two, nullptr, 1, 4, /*is_montgomery=*/0, nullptr, nullptr);
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

// Two field descriptors are the same field iff every parameter matches.
bool SameField(FieldDescr* a, FieldDescr* b) {
  if (a->kind != b->kind) {
    return false;
  }
  if (a->kind == kBinaryTower) {
    return a->tower_level == b->tower_level;
  }
  if (a->base_width_bytes != b->base_width_bytes || a->degree != b->degree ||
      a->is_montgomery != b->is_montgomery) {
    return false;
  }
  if (PyObject_RichCompareBool(a->modulus, b->modulus, Py_EQ) != 1) {
    return false;
  }
  if (a->degree == 1) {
    return true;
  }
  return PyObject_RichCompareBool(a->non_residue, b->non_residue, Py_EQ) == 1;
}

PyArray_Descr* CommonInstance(PyArray_Descr* a, PyArray_Descr* b) {
  if (SameField(AsField(a), AsField(b))) {
    Py_INCREF(a);
    return a;
  }
  PyErr_SetString(PyExc_TypeError,
                  "cannot combine field arrays of different fields");
  return nullptr;
}

PyArray_Descr* EnsureCanonical(PyArray_Descr* self) {
  Py_INCREF(self);
  return self;
}

PyArray_Descr* DiscoverDescrFromPyobject(PyArray_DTypeMeta* /*cls*/,
                                         PyObject* /*obj*/) {
  PyErr_SetString(PyExc_TypeError,
                  "cannot infer a field from a scalar; pass an explicit "
                  "dtype=zk_dtypes.prime_field(p) / extension_field(...)");
  return nullptr;
}

// setitem accepts a Python int (constant-term embedding) or a length-`degree`
// sequence of coefficients (constant term first).
int SetItem(PyArray_Descr* descr, PyObject* obj, char* dataptr) {
  FieldDescr* d = AsField(descr);
  if (d->kind == kBinaryTower) {
    PyObject* idx = PyNumber_Index(obj);
    if (idx == nullptr) {
      return -1;
    }
    PyObject* masked = PyNumber_And(idx, d->value_mask);
    Py_DECREF(idx);
    if (masked == nullptr) {
      return -1;
    }
    int brc = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(masked),
                                  reinterpret_cast<unsigned char*>(dataptr),
                                  d->base_width_bytes, 1, 0);
    Py_DECREF(masked);
    return brc < 0 ? -1 : 0;
  }
  PyObject* coeffs[kMaxDegree] = {nullptr};
  int rc = -1;
  if (PyIndex_Check(obj)) {
    coeffs[0] = PyNumber_Index(obj);
    if (coeffs[0] == nullptr) {
      goto done;
    }
    for (int i = 1; i < d->degree; ++i) {
      coeffs[i] = PyLong_FromLong(0);
      if (coeffs[i] == nullptr) {
        goto done;
      }
    }
  } else {
    PyObject* seq = PySequence_Fast(
        obj, "field element must be an int or a sequence of coefficients");
    if (seq == nullptr) {
      goto done;
    }
    if (PySequence_Fast_GET_SIZE(seq) != d->degree) {
      Py_DECREF(seq);
      PyErr_Format(PyExc_ValueError,
                   "field element needs %d coefficients, got %zd", d->degree,
                   PySequence_Fast_GET_SIZE(seq));
      goto done;
    }
    for (int i = 0; i < d->degree; ++i) {
      coeffs[i] = PyNumber_Index(PySequence_Fast_GET_ITEM(seq, i));
      if (coeffs[i] == nullptr) {
        Py_DECREF(seq);
        goto done;
      }
    }
    Py_DECREF(seq);
  }
  rc = EncodeElement(d, dataptr, coeffs);
done:
  for (int i = 0; i < d->degree; ++i) {
    Py_XDECREF(coeffs[i]);
  }
  return rc;
}

// getitem returns a Python int for a prime field, or a tuple of canonical
// coefficients (constant term first) for an extension field.
PyObject* GetItem(PyArray_Descr* descr, char* dataptr) {
  FieldDescr* d = AsField(descr);
  if (d->kind == kBinaryTower) {
    return _PyLong_FromByteArray(reinterpret_cast<unsigned char*>(dataptr),
                                 d->base_width_bytes, 1, 0);
  }
  PyObject* coeffs[kMaxDegree] = {nullptr};
  if (DecodeElement(d, dataptr, coeffs) < 0) {
    return nullptr;
  }
  if (d->degree == 1) {
    return coeffs[0];
  }
  PyObject* tuple = PyTuple_New(d->degree);
  if (tuple == nullptr) {
    for (int i = 0; i < d->degree; ++i) {
      Py_DECREF(coeffs[i]);
    }
    return nullptr;
  }
  for (int i = 0; i < d->degree; ++i) {
    PyTuple_SET_ITEM(tuple, i, coeffs[i]);  // steals the reference
  }
  return tuple;
}

// --- within-DType copy cast (numpy requires at least a self-copy) --------

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
  if (SameField(AsField(from), AsField(to))) {
    *view_offset = 0;
    return NPY_NO_CASTING;
  }
  return NPY_UNSAFE_CASTING;
}

int CastLoop(PyArrayMethod_Context* context, char* const* data,
             const npy_intp* dimensions, const npy_intp* strides,
             NpyAuxData* /*aux*/) {
  npy_intp n = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  npy_intp elsize = context->descriptors[0]->elsize;
  for (npy_intp i = 0; i < n; ++i) {
    std::memcpy(out, in, elsize);
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

// --- factory -------------------------------------------------------------

// field_descr(modulus, degree, non_residue, base_width_bits, is_montgomery
//             [, r_mod_p, rinv_mod_p]) -> dtype
// non_residue is ignored for degree 1. Montgomery storage passes R and R^-1
// (mod p, R = 2^base_width) computed in Python.
PyObject* MakeFieldDescrPy(PyObject* /*self*/, PyObject* args) {
  PyObject* modulus_obj;
  int degree;
  PyObject* non_residue_obj;
  int base_width_bits;
  int is_montgomery;
  PyObject* r_obj = nullptr;
  PyObject* rinv_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OiOii|OO", &modulus_obj, &degree,
                        &non_residue_obj, &base_width_bits, &is_montgomery,
                        &r_obj, &rinv_obj)) {
    return nullptr;
  }
  if (base_width_bits != 32 && base_width_bits != 64 &&
      base_width_bits != 128 && base_width_bits != 256) {
    PyErr_SetString(PyExc_ValueError,
                    "base_width_bits must be one of 32, 64, 128, 256");
    return nullptr;
  }
  if (degree < 1 || degree > kMaxDegree) {
    PyErr_Format(PyExc_ValueError, "degree must be in [1, %d]", kMaxDegree);
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
  PyObject* non_residue = nullptr;
  if (degree > 1) {
    non_residue = PyNumber_Index(non_residue_obj);
    if (non_residue == nullptr) {
      Py_DECREF(modulus);
      return nullptr;
    }
  }
  PyObject* r = nullptr;
  PyObject* rinv = nullptr;
  if (is_montgomery) {
    r = PyNumber_Index(r_obj);
    rinv = (r == nullptr) ? nullptr : PyNumber_Index(rinv_obj);
    if (r == nullptr || rinv == nullptr) {
      Py_DECREF(modulus);
      Py_XDECREF(non_residue);
      Py_XDECREF(r);
      return nullptr;
    }
  }
  PyArray_Descr* d = MakeDescr(modulus, non_residue, degree,
                               base_width_bits / 8, is_montgomery, r, rinv);
  Py_DECREF(modulus);
  Py_XDECREF(non_residue);
  Py_XDECREF(r);
  Py_XDECREF(rinv);
  return reinterpret_cast<PyObject*>(d);
}

PyObject* MakeBinaryFieldDescrPy(PyObject* /*self*/, PyObject* args) {
  int tower_level;
  if (!PyArg_ParseTuple(args, "i", &tower_level)) {
    return nullptr;
  }
  if (tower_level < 0 || tower_level > 12) {
    PyErr_SetString(PyExc_ValueError, "tower_level must be in [0, 12]");
    return nullptr;
  }
  int m = 1 << tower_level;             // field bit width 2^level
  int width_bytes = m < 8 ? 1 : m / 8;  // small levels occupy one byte
  return reinterpret_cast<PyObject*>(MakeBinaryDescr(tower_level, width_bytes));
}

PyMethodDef kModuleMethods[] = {
    {"field_descr", MakeFieldDescrPy, METH_VARARGS,
     "field_descr(modulus, degree, non_residue, base_width_bits, is_montgomery"
     "[, r_mod_p, rinv_mod_p]) -> dtype\n\n"
     "Build a parametric field descriptor (prime or binomial extension)."},
    {"binary_field_descr", MakeBinaryFieldDescrPy, METH_VARARGS,
     "binary_field_descr(tower_level) -> dtype\n\n"
     "Build a parametric binary tower field GF(2^(2^level)) descriptor."},
    {nullptr, nullptr, 0, nullptr},
};

// --- arithmetic ufunc loops (host eager add / sub / mul) -----------------

enum class Op { kAdd, kSub, kMul };

NPY_CASTING ArithResolve(struct PyArrayMethodObject_tag* /*method*/,
                         PyArray_DTypeMeta* const* /*dtypes*/,
                         PyArray_Descr* const* given, PyArray_Descr** loop,
                         npy_intp* view_offset) {
  if (!SameField(AsField(given[0]), AsField(given[1]))) {
    PyErr_SetString(PyExc_TypeError,
                    "field operation requires identical fields");
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

// Computes out[] = a[] op b[] in the field (canonical coefficients in/out).
// For mul, multiplies the degree-(k-1) polynomials and reduces X^k = nr.
int ComputeFieldOp(FieldDescr* d, Op op, PyObject* const* a, PyObject* const* b,
                   PyObject** out) {
  const int k = d->degree;
  if (op != Op::kMul) {
    for (int i = 0; i < k; ++i) {
      PyObject* r = op == Op::kAdd ? PyNumber_Add(a[i], b[i])
                                   : PyNumber_Subtract(a[i], b[i]);
      if (r == nullptr) {
        for (int j = 0; j < i; ++j) Py_DECREF(out[j]);
        return -1;
      }
      out[i] = PyNumber_Remainder(r, d->modulus);
      Py_DECREF(r);
      if (out[i] == nullptr) {
        for (int j = 0; j < i; ++j) Py_DECREF(out[j]);
        return -1;
      }
    }
    return 0;
  }
  // Polynomial product, then fold X^k = non_residue.
  PyObject* prod[2 * kMaxDegree] = {nullptr};
  int rc = -1;
  for (int i = 0; i < 2 * k - 1; ++i) {
    prod[i] = PyLong_FromLong(0);
    if (prod[i] == nullptr) goto cleanup;
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      PyObject* term = PyNumber_Multiply(a[i], b[j]);
      if (term == nullptr) goto cleanup;
      PyObject* sum = PyNumber_Add(prod[i + j], term);
      Py_DECREF(term);
      if (sum == nullptr) goto cleanup;
      Py_SETREF(prod[i + j], sum);
    }
  }
  for (int i = 2 * k - 2; i >= k; --i) {
    PyObject* scaled = PyNumber_Multiply(d->non_residue, prod[i]);
    if (scaled == nullptr) goto cleanup;
    PyObject* sum = PyNumber_Add(prod[i - k], scaled);
    Py_DECREF(scaled);
    if (sum == nullptr) goto cleanup;
    Py_SETREF(prod[i - k], sum);
  }
  for (int i = 0; i < k; ++i) {
    out[i] = PyNumber_Remainder(prod[i], d->modulus);
    if (out[i] == nullptr) {
      for (int j = 0; j < i; ++j) Py_DECREF(out[j]);
      goto cleanup;
    }
  }
  rc = 0;
cleanup:
  for (int i = 0; i < 2 * k - 1; ++i) {
    Py_XDECREF(prod[i]);
  }
  return rc;
}

// Advances the three element cursors by their strides.
inline void Advance(char*& a, char*& b, char*& o, const npy_intp* strides) {
  a += strides[0];
  b += strides[1];
  o += strides[2];
}

template <Op op>
int ArithLoop(PyArrayMethod_Context* context, char* const* data,
              const npy_intp* dimensions, const npy_intp* strides,
              NpyAuxData* /*aux*/) {
  FieldDescr* d = AsField(context->descriptors[0]);
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  const int wb = d->base_width_bytes;

  if (d->kind == kBinaryTower) {
    if (op != Op::kMul) {  // characteristic 2: add == sub == XOR
      for (npy_intp i = 0; i < n; ++i) {
        for (int k = 0; k < wb; ++k) o[k] = static_cast<char>(a[k] ^ b[k]);
        Advance(a, b, o, strides);
      }
      return 0;
    }
    if (d->tower_level <= 7) {  // native recursive Karatsuba tower multiply
      for (npy_intp i = 0; i < n; ++i) {
        modarith::BinaryTowerMul(d->tower_level, wb,
                                 reinterpret_cast<const unsigned char*>(a),
                                 reinterpret_cast<const unsigned char*>(b),
                                 reinterpret_cast<unsigned char*>(o));
        Advance(a, b, o, strides);
      }
      return 0;
    }
    for (npy_intp i = 0; i < n; ++i) {  // wide tower: Python-int Karatsuba
      PyObject* av =
          _PyLong_FromByteArray(reinterpret_cast<unsigned char*>(a), wb, 1, 0);
      PyObject* bv =
          _PyLong_FromByteArray(reinterpret_cast<unsigned char*>(b), wb, 1, 0);
      if (!av || !bv) {
        Py_XDECREF(av);
        Py_XDECREF(bv);
        return -1;
      }
      PyObject* rv = TowerMul(d->tower_level, av, bv);
      Py_DECREF(av);
      Py_DECREF(bv);
      if (!rv) return -1;
      int rc =
          _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(rv),
                              reinterpret_cast<unsigned char*>(o), wb, 1, 0);
      Py_DECREF(rv);
      if (rc < 0) return -1;
      Advance(a, b, o, strides);
    }
    return 0;
  }

  // Odd field: native fixed-width path where supported, Python-int otherwise.
  unsigned char mod_le[32];
  if (_PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(d->modulus), mod_le,
                          wb, 1, 0) >= 0) {
    modarith::PrimeField pf =
        modarith::PrimeField::Make(mod_le, wb, d->is_montgomery);
    if (d->degree == 1 && pf.native) {
      for (npy_intp i = 0; i < n; ++i) {
        const auto* ua = reinterpret_cast<const unsigned char*>(a);
        const auto* ub = reinterpret_cast<const unsigned char*>(b);
        auto* uo = reinterpret_cast<unsigned char*>(o);
        if (op == Op::kAdd) {
          pf.Add(ua, ub, uo);
        } else if (op == Op::kSub) {
          pf.Sub(ua, ub, uo);
        } else {
          pf.Mul(ua, ub, uo);
        }
        Advance(a, b, o, strides);
      }
      return 0;
    }
    const int k = d->degree;
    if (k > 1 && op != Op::kMul && pf.native) {  // extension add/sub: linear
      for (npy_intp i = 0; i < n; ++i) {
        for (int c = 0; c < k; ++c) {
          const auto* ua = reinterpret_cast<const unsigned char*>(a) + c * wb;
          const auto* ub = reinterpret_cast<const unsigned char*>(b) + c * wb;
          auto* uo = reinterpret_cast<unsigned char*>(o) + c * wb;
          if (op == Op::kAdd) {
            pf.Add(ua, ub, uo);
          } else {
            pf.Sub(ua, ub, uo);
          }
        }
        Advance(a, b, o, strides);
      }
      return 0;
    }
    if (k > 1 && op == Op::kMul && pf.ext_native) {  // binomial polynomial mul
      unsigned char nr_le[8] = {0};
      if (_PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(d->non_residue),
                              nr_le, wb, 1, 0) >= 0) {
        for (npy_intp i = 0; i < n; ++i) {
          // Coefficients stay in their storage form; the product/accumulate
          // (montmul/modadd) and the X^k = non_residue fold preserve it, so the
          // output is byte-identical to the decode/compute/encode path.
          unsigned char prod[2 * kMaxDegree][8] = {{0}};
          for (int ii = 0; ii < k; ++ii) {
            for (int jj = 0; jj < k; ++jj) {
              unsigned char term[8];
              pf.Mul(reinterpret_cast<const unsigned char*>(a) + ii * wb,
                     reinterpret_cast<const unsigned char*>(b) + jj * wb, term);
              pf.Add(prod[ii + jj], term, prod[ii + jj]);
            }
          }
          for (int ii = 2 * k - 2; ii >= k; --ii) {
            unsigned char scaled[8];
            pf.CanonMulBytes(nr_le, prod[ii], scaled);
            pf.Add(prod[ii - k], scaled, prod[ii - k]);
          }
          for (int c = 0; c < k; ++c) {
            std::memcpy(reinterpret_cast<unsigned char*>(o) + c * wb, prod[c],
                        wb);
          }
          Advance(a, b, o, strides);
        }
        return 0;
      }
      PyErr_Clear();
    }
  } else {
    PyErr_Clear();
  }

  for (npy_intp i = 0; i < n; ++i) {  // generic Python-int fallback
    PyObject* av[kMaxDegree];
    PyObject* bv[kMaxDegree];
    PyObject* ov[kMaxDegree];
    if (DecodeElement(d, a, av) < 0) {
      return -1;
    }
    if (DecodeElement(d, b, bv) < 0) {
      for (int j = 0; j < d->degree; ++j) Py_DECREF(av[j]);
      return -1;
    }
    int rc = ComputeFieldOp(d, op, av, bv, ov);
    for (int j = 0; j < d->degree; ++j) {
      Py_DECREF(av[j]);
      Py_DECREF(bv[j]);
    }
    if (rc < 0) {
      return -1;
    }
    int erc = EncodeElement(d, o, ov);
    for (int j = 0; j < d->degree; ++j) Py_DECREF(ov[j]);
    if (erc < 0) {
      return -1;
    }
    Advance(a, b, o, strides);
  }
  return 0;
}

template <Op op>
bool AddArithLoop(PyObject* numpy, const char* ufunc_name) {
  PyObject* ufunc = PyObject_GetAttrString(numpy, ufunc_name);
  if (ufunc == nullptr) {
    return false;
  }
  PyArray_DTypeMeta* dtypes[3] = {&FieldDType, &FieldDType, &FieldDType};
  PyType_Slot slots[] = {
      {NPY_METH_resolve_descriptors, reinterpret_cast<void*>(ArithResolve)},
      {NPY_METH_strided_loop, reinterpret_cast<void*>(ArithLoop<op>)},
      {0, nullptr},
  };
  PyArrayMethod_Spec spec = {};
  spec.name = "field_arith";
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

}  // namespace

PyObject* FieldDTypeMetaObject() {
  return reinterpret_cast<PyObject*>(&FieldDType);
}

PyObject* PrimeFieldValue(PyObject* descr, const char* data) {
  FieldDescr* f = AsField(reinterpret_cast<PyArray_Descr*>(descr));
  if (f->kind != kOddField || f->degree != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "EC scalar must be a prime (degree-1) field element");
    return nullptr;
  }
  return DecodeCoeff(f, data);
}

bool RegisterFieldDType(PyObject* /*numpy*/, PyObject* module) {
  FieldScalar_Type.tp_name = "zk_dtypes._zk_dtypes_ext.FieldScalar";
  FieldScalar_Type.tp_basicsize = 0;
  FieldScalar_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  FieldScalar_Type.tp_base = &PyGenericArrType_Type;
  if (PyType_Ready(&FieldScalar_Type) < 0) {
    return false;
  }

  PyTypeObject* type = reinterpret_cast<PyTypeObject*>(&FieldDType);
  Py_SET_TYPE(reinterpret_cast<PyObject*>(&FieldDType), &PyArrayDTypeMeta_Type);
  Py_SET_REFCNT(reinterpret_cast<PyObject*>(&FieldDType), 1);
  type->tp_name = "zk_dtypes._zk_dtypes_ext.FieldDType";
  type->tp_basicsize = sizeof(FieldDescr);
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
  copy_cast.name = "field_copy";
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
  spec.typeobj = &FieldScalar_Type;
  spec.flags = NPY_DT_PARAMETRIC;
  spec.casts = casts;
  spec.slots = dtype_slots;
  spec.baseclass = nullptr;
  if (PyArrayInitDTypeMeta_FromSpec(&FieldDType, &spec) < 0) {
    return false;
  }
  FieldDType.singleton = PyArray_GetDefaultDescr(&FieldDType);
  if (FieldDType.singleton == nullptr) {
    return false;
  }

  if (PyModule_AddObject(module, "FieldDType",
                         reinterpret_cast<PyObject*>(&FieldDType)) < 0) {
    return false;
  }
  Py_INCREF(reinterpret_cast<PyObject*>(&FieldDType));

  PyObject* fn = PyCFunction_New(&kModuleMethods[0], nullptr);
  if (fn == nullptr) {
    return false;
  }
  if (PyModule_AddObject(module, "field_descr", fn) < 0) {
    Py_DECREF(fn);
    return false;
  }
  PyObject* bfn = PyCFunction_New(&kModuleMethods[1], nullptr);
  if (bfn == nullptr) {
    return false;
  }
  if (PyModule_AddObject(module, "binary_field_descr", bfn) < 0) {
    Py_DECREF(bfn);
    return false;
  }

  if (_import_umath() < 0) {
    return false;
  }
  PyObject* numpy = PyImport_ImportModule("numpy");
  if (numpy == nullptr) {
    return false;
  }
  bool ok = AddArithLoop<Op::kAdd>(numpy, "add") &&
            AddArithLoop<Op::kSub>(numpy, "subtract") &&
            AddArithLoop<Op::kMul>(numpy, "multiply");
  Py_DECREF(numpy);
  return ok;
}

}  // namespace zk_dtypes
