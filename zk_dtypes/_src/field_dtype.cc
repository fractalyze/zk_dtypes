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
#include <limits>
#include <type_traits>
#include <vector>
#include <cstring>

#include "zk_dtypes/_src/field_modarith.h"
#include "zk_dtypes/_src/nep42_common.h"
#include "zk_dtypes/_src/numpy.h"
#include "zk_dtypes/_src/pyfield_ops.h"
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
  // odd: per-coefficient width; binary: storage width. uint16 so a high binary
  // tower (level 11/12 -> 256/512 bytes) does not truncate to 0.
  uint16_t base_width_bytes;
  uint8_t degree;       // odd: extension degree (1 = prime); binary: 1
  uint8_t tower_level;  // binary tower level
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
  PyObject* one_again = PyLong_FromLong(1);
  PyObject* mask = one_again ? PyNumber_Subtract(shifted, one_again) : nullptr;
  Py_DECREF(shifted);
  Py_XDECREF(one_again);
  return mask;
}

// Canonical Fan-Paar tower multiply, GF(2^(2^level)) =
// GF(2^(2^(level-1)))[X]/(X^2 + beta*X + 1) — the same construction as
// BinaryOps<k>::Mul (include/field/binary_field_multiplication.h), so levels
// 8-12 (which have no native kernel) extend the exact tower the native levels
// 0-7 build: embedding a level-k element in level k+1 (high half zero)
// multiplies identically.

PyObject* TowerXor(PyObject* a, PyObject* b) { return PyNumber_Xor(a, b); }

// Multiply by the generator X_level. At level 0, X = 1 in GF(2): identity.
PyObject* TowerMulX(int level, PyObject* a) {
  if (level == 0) {
    Py_INCREF(a);
    return a;
  }
  const int sb = 1 << (level - 1);
  PyObject* submask = Mask(sb);
  PyObject* shift = PyLong_FromLong(sb);
  PyObject *a0 = nullptr, *a1 = nullptr, *mx = nullptr, *hi = nullptr;
  PyObject *hish = nullptr, *result = nullptr;
  if (!submask || !shift) goto done;
  a0 = PyNumber_And(a, submask);
  {
    PyObject* ah = PyNumber_Rshift(a, shift);
    a1 = ah ? PyNumber_And(ah, submask) : nullptr;
    Py_XDECREF(ah);
  }
  if (!a0 || !a1) goto done;
  // a*X = a1 + (a0 + beta*a1)*X, beta*(.) = TowerMulX one level down.
  mx = TowerMulX(level - 1, a1);
  hi = mx ? TowerXor(a0, mx) : nullptr;
  hish = hi ? PyNumber_Lshift(hi, shift) : nullptr;
  result = hish ? PyNumber_Or(a1, hish) : nullptr;
done:
  Py_XDECREF(submask);
  Py_XDECREF(shift);
  Py_XDECREF(a0);
  Py_XDECREF(a1);
  Py_XDECREF(mx);
  Py_XDECREF(hi);
  Py_XDECREF(hish);
  return result;
}

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
  PyObject *a0 = nullptr, *a1 = nullptr, *b0 = nullptr, *b1 = nullptr;
  PyObject *a0b0 = nullptr, *a1b1 = nullptr, *ax = nullptr, *bx = nullptr;
  PyObject *cross = nullptr, *c0 = nullptr, *mx = nullptr;
  PyObject *c1a = nullptr, *c1b = nullptr, *c1 = nullptr;
  PyObject *c1sh = nullptr, *result = nullptr;
  if (!submask || !shift) goto done;
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
  if (!a0 || !a1 || !b0 || !b1) goto done;
  a0b0 = TowerMul(level - 1, a0, b0);
  a1b1 = TowerMul(level - 1, a1, b1);
  ax = TowerXor(a0, a1);
  bx = TowerXor(b0, b1);
  if (!a0b0 || !a1b1 || !ax || !bx) goto done;
  cross = TowerMul(level - 1, ax, bx);
  if (!cross) goto done;
  // X^2 = beta*X + 1  =>  c0 = a0*b0 + a1*b1,
  // c1 = cross + a0*b0 + a1*b1 + beta*(a1*b1).
  c0 = TowerXor(a0b0, a1b1);
  mx = TowerMulX(level - 1, a1b1);
  c1a = (c0 && mx) ? TowerXor(cross, a0b0) : nullptr;
  c1b = c1a ? TowerXor(c1a, a1b1) : nullptr;
  c1 = c1b ? TowerXor(c1b, mx) : nullptr;
  c1sh = c1 ? PyNumber_Lshift(c1, shift) : nullptr;
  result = c1sh ? PyNumber_Or(c0, c1sh) : nullptr;
done:
  Py_XDECREF(submask);
  Py_XDECREF(shift);
  Py_XDECREF(a0);
  Py_XDECREF(a1);
  Py_XDECREF(b0);
  Py_XDECREF(b1);
  Py_XDECREF(a0b0);
  Py_XDECREF(a1b1);
  Py_XDECREF(ax);
  Py_XDECREF(bx);
  Py_XDECREF(cross);
  Py_XDECREF(c0);
  Py_XDECREF(mx);
  Py_XDECREF(c1a);
  Py_XDECREF(c1b);
  Py_XDECREF(c1);
  Py_XDECREF(c1sh);
  return result;
}

// --- per-coefficient encode / decode (canonical or Montgomery) ----------

pyfield::BaseCodec Codec(FieldDescr* d) {
  return {d->modulus, d->r_mod_p, d->rinv_mod_p, d->base_width_bytes,
          d->is_montgomery != 0};
}

// Reads one base-field coefficient at `slot` and returns its canonical value.
PyObject* DecodeCoeff(FieldDescr* d, const char* slot) {
  return pyfield::Decode(Codec(d), slot);
}

// Writes canonical value `value` into the base-field coefficient at `slot`.
int EncodeCoeff(FieldDescr* d, char* slot, PyObject* value) {
  return pyfield::Encode(Codec(d), slot, value);
}

// One binary-tower element as a canonical Python int, and back (masked).
PyObject* DecodeBinary(FieldDescr* d, const char* ptr) {
  return _PyLong_FromByteArray(reinterpret_cast<const unsigned char*>(ptr),
                               d->base_width_bytes, 1, 0);
}

int EncodeBinary(FieldDescr* d, char* ptr, PyObject* value) {
  PyObject* masked = PyNumber_And(value, d->value_mask);
  if (masked == nullptr) return -1;
  int rc = pyfield::LongAsBytesLE(masked, reinterpret_cast<unsigned char*>(ptr),
                                  d->base_width_bytes);
  Py_DECREF(masked);
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

npy_bool NonZero(void* data, void* arr);

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
  d->base_width_bytes = static_cast<uint16_t>(base_width_bytes);
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
  // np.nonzero and truthiness call the legacy ArrFuncs entry; without it numpy
  // dereferences a null function pointer. Set it per descriptor through the
  // accessor rather than the DType-spec slot, whose numeric id moved between
  // supported numpy versions.
  PyDataType_GetArrFuncs(base)->nonzero = NonZero;
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
  d->base_width_bytes = static_cast<uint16_t>(width_bytes);
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
  PyDataType_GetArrFuncs(base)->nonzero = NonZero;
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

// Descriptor introspection: the parameters a consumer needs to reconstruct or
// reason about the field. Kept read-only — a descriptor is immutable once the
// factory mints it.
PyObject* Get_modulus(PyObject* self, void* /*closure*/) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  if (d->modulus == nullptr) Py_RETURN_NONE;
  Py_INCREF(d->modulus);
  return d->modulus;
}

PyObject* Get_degree(PyObject* self, void* /*closure*/) {
  return PyLong_FromLong(
      AsField(reinterpret_cast<PyArray_Descr*>(self))->degree);
}

PyObject* Get_non_residue(PyObject* self, void* /*closure*/) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  if (d->non_residue == nullptr) Py_RETURN_NONE;
  Py_INCREF(d->non_residue);
  return d->non_residue;
}

PyObject* Get_base_width_bits(PyObject* self, void* /*closure*/) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  return PyLong_FromLong(d->base_width_bytes * 8);
}

PyObject* Get_is_montgomery(PyObject* self, void* /*closure*/) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  return PyBool_FromLong(d->is_montgomery);
}

PyObject* Get_tower_level(PyObject* self, void* /*closure*/) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  if (d->kind != kBinaryTower) Py_RETURN_NONE;
  return PyLong_FromLong(d->tower_level);
}

PyGetSetDef Descr_getset[] = {
    {"modulus", Get_modulus, nullptr, "base prime modulus (None for binary)",
     nullptr},
    {"degree", Get_degree, nullptr, "extension degree (1 for prime)", nullptr},
    {"non_residue", Get_non_residue, nullptr,
     "X^degree = non_residue (None for degree 1 / binary)", nullptr},
    {"base_width_bits", Get_base_width_bits, nullptr,
     "per-coefficient storage width in bits", nullptr},
    {"is_montgomery", Get_is_montgomery, nullptr,
     "True when coefficients are Montgomery-encoded", nullptr},
    {"tower_level", Get_tower_level, nullptr,
     "binary tower level (None for odd fields)", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

// numpy's default descriptor pickling refuses custom DTypes, so route through
// the module-level factory that minted this descriptor.
PyObject* Descr_reduce(PyObject* self, PyObject* /*unused*/) {
  FieldDescr* d = AsField(reinterpret_cast<PyArray_Descr*>(self));
  PyObject* module = PyImport_ImportModule("zk_dtypes._zk_dtypes_ext");
  if (module == nullptr) return nullptr;
  const char* fname =
      d->kind == kBinaryTower ? "binary_field_descr" : "field_descr";
  PyObject* factory = PyObject_GetAttrString(module, fname);
  Py_DECREF(module);
  if (factory == nullptr) return nullptr;
  PyObject* args = nullptr;
  if (d->kind == kBinaryTower) {
    args = Py_BuildValue("(i)", static_cast<int>(d->tower_level));
  } else if (d->is_montgomery) {
    args = Py_BuildValue("(OiOiiOO)", d->modulus, static_cast<int>(d->degree),
                         d->non_residue ? d->non_residue : Py_None,
                         d->base_width_bytes * 8, 1, d->r_mod_p, d->rinv_mod_p);
  } else {
    args = Py_BuildValue("(OiOii)", d->modulus, static_cast<int>(d->degree),
                         d->non_residue ? d->non_residue : Py_None,
                         d->base_width_bytes * 8, 0);
  }
  if (args == nullptr) {
    Py_DECREF(factory);
    return nullptr;
  }
  PyObject* result = Py_BuildValue("(NN)", factory, args);
  return result;
}

PyMethodDef Descr_methods[] = {
    {"__reduce__", Descr_reduce, METH_NOARGS,
     "Pickle support: rebuild through the descriptor factory."},
    {nullptr, nullptr, 0, nullptr},
};

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
  // A field absorbs integers (including numpy's weak Python-int DType), so
  // `a + 1` and `np.add(a, np.int64(1))` land in the field via the int cast.
  PyArray_DTypeMeta* other = a == &FieldDType ? b : a;
  PyArray_DTypeMeta* const kIntDTypes[] = {
      &PyArray_PyLongDType, &PyArray_Int8DType,   &PyArray_UInt8DType,
      &PyArray_Int16DType,  &PyArray_UInt16DType, &PyArray_Int32DType,
      &PyArray_UInt32DType, &PyArray_Int64DType,  &PyArray_UInt64DType,
  };
  for (PyArray_DTypeMeta* intdt : kIntDTypes) {
    if (other == intdt) {
      Py_INCREF(&FieldDType);
      return &FieldDType;
    }
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

// --- rich scalar type -----------------------------------------------------
// `arr[i]` returns one of these instead of a bare int/tuple: it carries the
// descriptor alongside the canonical value, so it can print its field, do
// field arithmetic with operators, and expose the stored (possibly Montgomery)
// representation as `.raw`. The legacy per-family scalars get the field from
// their C++ type; a parametric scalar has to hold the descriptor instead.

struct FieldScalarObject {
  PyObject_HEAD PyArray_Descr* descr;  // owned: the field this value belongs to
  PyObject* value;  // owned: canonical int, or tuple of coefficients
};

PyObject* MakeScalar(PyArray_Descr* descr, PyObject* value);  // steals `value`

bool IsFieldScalar(PyObject* obj) { return Py_TYPE(obj) == &FieldScalar_Type; }

FieldScalarObject* AsScalar(PyObject* obj) {
  return reinterpret_cast<FieldScalarObject*>(obj);
}

// Coefficient array from a scalar's value (borrowed refs into `value`).
int ScalarCoeffs(FieldScalarObject* s, PyObject** out) {
  FieldDescr* d = AsField(s->descr);
  if (d->kind == kBinaryTower || d->degree == 1) {
    out[0] = s->value;
    return 1;
  }
  for (int i = 0; i < d->degree; ++i) {
    out[i] = PyTuple_GET_ITEM(s->value, i);
  }
  return d->degree;
}

// setitem accepts a Python int (constant-term embedding) or a length-`degree`
// sequence of coefficients (constant term first).
int SetItem(PyArray_Descr* descr, PyObject* obj, char* dataptr) {
  FieldDescr* d = AsField(descr);
  if (IsFieldScalar(obj)) {  // assigning an element of the same field
    FieldScalarObject* s = AsScalar(obj);
    if (!SameField(d, AsField(s->descr))) {
      PyErr_SetString(PyExc_TypeError,
                      "cannot assign an element of a different field");
      return -1;
    }
    if (d->kind == kBinaryTower) {
      return EncodeBinary(d, dataptr, s->value);
    }
    PyObject* coeffs[kMaxDegree];
    ScalarCoeffs(s, coeffs);
    return EncodeElement(d, dataptr, coeffs);
  }
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
    int brc = pyfield::LongAsBytesLE(
        masked, reinterpret_cast<unsigned char*>(dataptr), d->base_width_bytes);
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
    return MakeScalar(descr, DecodeBinary(d, dataptr));
  }
  PyObject* coeffs[kMaxDegree] = {nullptr};
  if (DecodeElement(d, dataptr, coeffs) < 0) {
    return nullptr;
  }
  if (d->degree == 1) {
    return MakeScalar(descr, coeffs[0]);
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
  return MakeScalar(descr, tuple);
}

// --- within-DType copy cast (numpy requires at least a self-copy) --------

// Same base field and storage form, source degree 1, target degree > 1: the
// base element embeds as the constant coefficient of the extension.
bool IsBaseEmbedding(FieldDescr* from, FieldDescr* to) {
  return from->kind == kOddField && to->kind == kOddField &&
         from->degree == 1 && to->degree > 1 &&
         from->base_width_bytes == to->base_width_bytes &&
         from->is_montgomery == to->is_montgomery &&
         PyObject_RichCompareBool(from->modulus, to->modulus, Py_EQ) == 1;
}

// Same odd field except for the storage form (Montgomery vs canonical) — the
// only non-identity cast we re-encode. Anything else (different modulus/degree/
// width, binary level, or kind) is a meaningless cast and is rejected rather
// than raw-copied (which would corrupt values and, across widths, write out of
// bounds).
bool CastCompatible(FieldDescr* a, FieldDescr* b) {
  if (a->kind != kOddField || b->kind != kOddField) return false;
  if (IsBaseEmbedding(a, b)) return true;  // base -> extension embed
  if (a->base_width_bytes != b->base_width_bytes || a->degree != b->degree) {
    return false;
  }
  if (PyObject_RichCompareBool(a->modulus, b->modulus, Py_EQ) != 1)
    return false;
  if (a->degree > 1) {
    return PyObject_RichCompareBool(a->non_residue, b->non_residue, Py_EQ) == 1;
  }
  return true;
}

NPY_CASTING CastResolve(struct PyArrayMethodObject_tag* /*method*/,
                        PyArray_DTypeMeta* const* /*dtypes*/,
                        PyArray_Descr* const* given, PyArray_Descr** loop,
                        npy_intp* view_offset) {
  PyArray_Descr* from = given[0];
  PyArray_Descr* to = given[1];
  if (to != nullptr && !SameField(AsField(from), AsField(to)) &&
      !CastCompatible(AsField(from), AsField(to))) {
    PyErr_SetString(PyExc_TypeError,
                    "cannot cast between different parametric fields");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(from);
  loop[0] = from;
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
  // Both remaining in-family casts preserve the value: base -> extension
  // embeds as the constant coefficient, Montgomery <-> canonical re-encodes
  // the same element.
  return IsBaseEmbedding(AsField(from), AsField(to)) ? NPY_SAFE_CASTING
                                                     : NPY_SAME_KIND_CASTING;
}

int CastLoop(PyArrayMethod_Context* context, char* const* data,
             const npy_intp* dimensions, const npy_intp* strides,
             NpyAuxData* /*aux*/) {
  FieldDescr* from = AsField(context->descriptors[0]);
  FieldDescr* to = AsField(context->descriptors[1]);
  npy_intp n = dimensions[0];
  char* in = data[0];
  char* out = data[1];
  if (SameField(from, to)) {  // byte-identical: raw copy
    npy_intp elsize = context->descriptors[0]->elsize;
    for (npy_intp i = 0; i < n; ++i) {
      std::memcpy(out, in, elsize);
      in += strides[0];
      out += strides[1];
    }
    return 0;
  }
  if (IsBaseEmbedding(from, to)) {  // c -> (c, 0, ..., 0)
    const int wb = to->base_width_bytes;
    for (npy_intp i = 0; i < n; ++i) {
      std::memcpy(out, in, wb);  // same storage form: constant term verbatim
      std::memset(out + wb, 0, static_cast<size_t>(wb) * (to->degree - 1));
      in += strides[0];
      out += strides[1];
    }
    return 0;
  }
  // Montgomery <-> canonical of the same field: decode then re-encode.
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* coeffs[kMaxDegree];
    if (DecodeElement(from, in, coeffs) < 0) return -1;
    int rc = EncodeElement(to, out, coeffs);
    for (int j = 0; j < from->degree; ++j) Py_DECREF(coeffs[j]);
    if (rc < 0) return -1;
    in += strides[0];
    out += strides[1];
  }
  return 0;
}

// --- int <-> field casts --------------------------------------------------
// Legacy dtypes register both directions; the parametric family mirrors them
// for the standard integer widths. int -> field reduces mod p into the
// constant coefficient (higher extension coefficients zero), matching setitem;
// field -> int yields the canonical value and overflows loudly when it does
// not fit the target width.

NPY_CASTING IntToFieldResolve(struct PyArrayMethodObject_tag* /*method*/,
                              PyArray_DTypeMeta* const* /*dtypes*/,
                              PyArray_Descr* const* given, PyArray_Descr** loop,
                              npy_intp* view_offset) {
  if (given[1] == nullptr) {  // no target instance: cannot invent a field
    PyErr_SetString(PyExc_TypeError,
                    "casting an integer array to a field needs an explicit "
                    "field dtype");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  Py_INCREF(given[1]);
  loop[1] = given[1];
  *view_offset = NPY_MIN_INTP;
  return NPY_SAME_KIND_CASTING;  // in-range preserved, else reduced mod p
}

template <typename T>
int IntToFieldLoop(PyArrayMethod_Context* context, char* const* data,
                   const npy_intp* dimensions, const npy_intp* strides,
                   NpyAuxData* /*aux*/) {
  PyArray_Descr* dst = context->descriptors[1];
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* o = data[1];
  for (npy_intp i = 0; i < n; ++i) {
    T raw;
    std::memcpy(&raw, a, sizeof(raw));
    PyObject* value;
    if constexpr (std::is_signed_v<T>) {
      value = PyLong_FromLongLong(static_cast<long long>(raw));
    } else {
      value = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(raw));
    }
    if (value == nullptr) return -1;
    int rc = SetItem(dst, value, o);
    Py_DECREF(value);
    if (rc < 0) return -1;
    a += strides[0];
    o += strides[1];
  }
  return 0;
}

NPY_CASTING FieldToIntResolve(struct PyArrayMethodObject_tag* /*method*/,
                              PyArray_DTypeMeta* const* dtypes,
                              PyArray_Descr* const* given, PyArray_Descr** loop,
                              npy_intp* view_offset) {
  FieldDescr* d = AsField(given[0]);
  if (d->kind != kBinaryTower && d->degree != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "only a degree-1 field element casts to an integer");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  if (given[1] != nullptr) {
    Py_INCREF(given[1]);
    loop[1] = given[1];
  } else {
    loop[1] = PyArray_GetDefaultDescr(dtypes[1]);
    if (loop[1] == nullptr) {
      Py_DECREF(loop[0]);
      return static_cast<NPY_CASTING>(-1);
    }
  }
  *view_offset = NPY_MIN_INTP;
  return NPY_UNSAFE_CASTING;
}

template <typename T>
int FieldToIntLoop(PyArrayMethod_Context* context, char* const* data,
                   const npy_intp* dimensions, const npy_intp* strides,
                   NpyAuxData* /*aux*/) {
  FieldDescr* d = AsField(context->descriptors[0]);
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* o = data[1];
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* value =
        d->kind == kBinaryTower ? DecodeBinary(d, a) : DecodeCoeff(d, a);
    if (value == nullptr) return -1;
    T raw;
    if constexpr (std::is_signed_v<T>) {
      long long v = PyLong_AsLongLong(value);
      Py_DECREF(value);
      if (v == -1 && PyErr_Occurred()) return -1;
      if (v < static_cast<long long>(std::numeric_limits<T>::min()) ||
          v > static_cast<long long>(std::numeric_limits<T>::max())) {
        PyErr_SetString(PyExc_OverflowError,
                        "field value does not fit the target integer width");
        return -1;
      }
      raw = static_cast<T>(v);
    } else {
      unsigned long long v = PyLong_AsUnsignedLongLong(value);
      Py_DECREF(value);
      if (v == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        return -1;
      }
      if (v > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        PyErr_SetString(PyExc_OverflowError,
                        "field value does not fit the target integer width");
        return -1;
      }
      raw = static_cast<T>(v);
    }
    std::memcpy(o, &raw, sizeof(raw));
    a += strides[0];
    o += strides[1];
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
    // A base-field operand embeds into its extension (legacy dtypes promote
    // BaseField op ExtField to the extension); anything else is a real
    // mismatch.
    FieldDescr* f0 = AsField(given[0]);
    FieldDescr* f1 = AsField(given[1]);
    PyArray_Descr* ext = nullptr;
    if (IsBaseEmbedding(f0, f1)) {
      ext = given[1];
    } else if (IsBaseEmbedding(f1, f0)) {
      ext = given[0];
    }
    if (ext == nullptr) {
      PyErr_SetString(PyExc_TypeError,
                      "field operation requires identical fields");
      return static_cast<NPY_CASTING>(-1);
    }
    if (given[2] != nullptr && !SameField(AsField(ext), AsField(given[2]))) {
      PyErr_SetString(PyExc_TypeError,
                      "field operation output requires the same field");
      return static_cast<NPY_CASTING>(-1);
    }
    Py_INCREF(ext);
    loop[0] = ext;
    Py_INCREF(ext);
    loop[1] = ext;
    PyArray_Descr* mixed_out = given[2] == nullptr ? ext : given[2];
    Py_INCREF(mixed_out);
    loop[2] = mixed_out;
    *view_offset = NPY_MIN_INTP;
    return NPY_SAFE_CASTING;  // embedding preserves the value
  }
  // An explicit out= must be the same field: the loop sizes every write from
  // descriptors[0], so a narrower out would be overrun and a different field
  // would silently mislabel the result.
  if (given[2] != nullptr && !SameField(AsField(given[0]), AsField(given[2]))) {
    PyErr_SetString(PyExc_TypeError,
                    "field operation output requires the same field");
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

// Weak-operand variant: `a + 1` hands numpy a Python int whose DType is
// neither ours nor a concrete integer. Resolving both operands to the field
// descriptor makes numpy insert the registered int->field cast, so the loops
// still see two field operands.
// Returns whichever operand descriptor is the field one (either side may be
// the Python-int operand, e.g. `a + 1` vs `1 + a`).
PyArray_Descr* FieldOperand(PyArray_Descr* const* given) {
  return Py_TYPE(given[0]) == reinterpret_cast<PyTypeObject*>(&FieldDType)
             ? given[0]
             : given[1];
}

NPY_CASTING WeakArithResolve(struct PyArrayMethodObject_tag* /*method*/,
                             PyArray_DTypeMeta* const* /*dtypes*/,
                             PyArray_Descr* const* given, PyArray_Descr** loop,
                             npy_intp* view_offset) {
  PyArray_Descr* field = FieldOperand(given);
  if (given[2] != nullptr && !SameField(AsField(field), AsField(given[2]))) {
    PyErr_SetString(PyExc_TypeError,
                    "field operation output requires the same field");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(field);
  loop[0] = field;
  Py_INCREF(field);
  loop[1] = field;  // the integer operand casts into the field
  PyArray_Descr* out = given[2] == nullptr ? field : given[2];
  Py_INCREF(out);
  loop[2] = out;
  *view_offset = NPY_MIN_INTP;
  // same-kind, not unsafe: numpy's default ufunc casting rule rejects unsafe,
  // and a Python-int operand entering its field is the weak-scalar case numpy
  // itself treats as same-kind (the value is reduced mod p, as setitem does).
  return NPY_SAME_KIND_CASTING;
}

NPY_CASTING WeakCmpResolve(struct PyArrayMethodObject_tag* /*method*/,
                           PyArray_DTypeMeta* const* /*dtypes*/,
                           PyArray_Descr* const* given, PyArray_Descr** loop,
                           npy_intp* view_offset) {
  PyArray_Descr* field = FieldOperand(given);
  PyArray_Descr* out_descr =
      given[2] != nullptr ? given[2] : PyArray_DescrFromType(NPY_BOOL);
  if (out_descr == nullptr) {
    return static_cast<NPY_CASTING>(-1);
  }
  if (given[2] != nullptr) {
    Py_INCREF(out_descr);
  }
  Py_INCREF(field);
  loop[0] = field;
  Py_INCREF(field);
  loop[1] = field;
  loop[2] = out_descr;
  *view_offset = NPY_MIN_INTP;
  return NPY_SAME_KIND_CASTING;
}

// Computes out[] = a[] op b[] in the field (canonical coefficients in/out).
// For mul, multiplies the degree-(k-1) polynomials and reduces X^k = nr.
int ComputeFieldOp(FieldDescr* d, Op op, PyObject* const* a, PyObject* const* b,
                   PyObject** out) {
  const int k = d->degree;
  if (op != Op::kMul) {
    for (int i = 0; i < k; ++i) {
      out[i] = op == Op::kAdd ? pyfield::ModAdd(d->modulus, a[i], b[i])
                              : pyfield::ModSub(d->modulus, a[i], b[i]);
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

// True when all three operands are contiguous (stride == element size) and
// pointer-aligned to T, so the element bytes can be read as a flat T[] and the
// inner loop auto-vectorizes the same way the legacy compile-time loops do.
template <typename T>
inline bool FlatAligned(char* a, char* b, char* o, const npy_intp* strides,
                        npy_intp elsize) {
  return strides[0] == elsize && strides[1] == elsize && strides[2] == elsize &&
         reinterpret_cast<uintptr_t>(a) % sizeof(T) == 0 &&
         reinterpret_cast<uintptr_t>(b) % sizeof(T) == 0 &&
         reinterpret_cast<uintptr_t>(o) % sizeof(T) == 0;
}

// Monomorphic modular add/sub over `degree` coefficients of a single-word field
// (T = uint32_t / uint64_t). Free of the per-element width switch; the
// contiguous case runs as one flat T[] loop (degree folds into the count) so
// the compiler vectorizes it. Prime (degree 1) and extension share it.
template <typename T, bool kSub>
void TypedAddSub(char* a, char* b, char* o, npy_intp n, const npy_intp* strides,
                 int degree, int wb, T modulus, bool spare) {
  npy_intp elsize = static_cast<npy_intp>(degree) * wb;
  if (FlatAligned<T>(a, b, o, strides, elsize)) {
    npy_intp m = n * degree;  // every coefficient is contiguous
    const T* ap = reinterpret_cast<const T*>(a);
    const T* bp = reinterpret_cast<const T*>(b);
    T* op = reinterpret_cast<T*>(o);
    if (spare) {
      // a, b < p < 2^(w-1), so a+b does not overflow: the conditional reduce is
      // a branchless min(x, x-+p), which auto-vectorizes (legacy gets the same
      // from its compile-time modulus). Byte-identical to ModAdd/ModSub.
      for (npy_intp i = 0; i < m; ++i) {
        if (kSub) {
          T d = ap[i] - bp[i];
          T dp = d + modulus;
          op[i] = dp < d ? dp : d;
        } else {
          T s = ap[i] + bp[i];
          T sm = s - modulus;
          op[i] = sm < s ? sm : s;
        }
      }
      return;
    }
    for (npy_intp i = 0; i < m; ++i) {  // no spare bit: carry-aware reduce
      if (kSub) {
        ModSub<T>(ap[i], bp[i], op[i], modulus, spare);
      } else {
        ModAdd<T>(ap[i], bp[i], op[i], modulus, spare);
      }
    }
    return;
  }
  for (npy_intp i = 0; i < n; ++i) {
    for (int c = 0; c < degree; ++c) {
      T x, y, r;
      std::memcpy(&x, a + c * wb, sizeof(T));
      std::memcpy(&y, b + c * wb, sizeof(T));
      if (kSub) {
        ModSub<T>(x, y, r, modulus, spare);
      } else {
        ModAdd<T>(x, y, r, modulus, spare);
      }
      std::memcpy(o + c * wb, &r, sizeof(T));
    }
    Advance(a, b, o, strides);
  }
}

// Monomorphic multiply for a degree-1 single-word prime field: Montgomery
// (kMont) on the stored representatives, or canonical a*b mod p. Contiguous
// case is a flat vectorizable loop.
template <typename T, bool kMont>
void TypedMul(char* a, char* b, char* o, npy_intp n, const npy_intp* strides,
              T modulus, T mont_nprime) {
  if (FlatAligned<T>(a, b, o, strides, sizeof(T))) {
    const T* ap = reinterpret_cast<const T*>(a);
    const T* bp = reinterpret_cast<const T*>(b);
    T* op = reinterpret_cast<T*>(o);
    for (npy_intp i = 0; i < n; ++i) {
      if (kMont) {
        MontMul<T>(ap[i], bp[i], op[i], modulus, mont_nprime);
      } else {
        op[i] = static_cast<T>(
            (static_cast<internal::make_promoted_t<T>>(ap[i]) * bp[i]) %
            modulus);
      }
    }
    return;
  }
  for (npy_intp i = 0; i < n; ++i) {
    T x, y, r;
    std::memcpy(&x, a, sizeof(T));
    std::memcpy(&y, b, sizeof(T));
    if (kMont) {
      MontMul<T>(x, y, r, modulus, mont_nprime);
    } else {
      r = static_cast<T>((static_cast<internal::make_promoted_t<T>>(x) * y) %
                         modulus);
    }
    std::memcpy(o, &r, sizeof(T));
    a += strides[0];
    b += strides[1];
    o += strides[2];
  }
}

// Native binomial-extension (degree k) multiply over a single-word base field
// (T = uint32_t / uint64_t), in Montgomery space: coefficient products via
// MontMul<T>, the X^k = non_residue fold via a canonical scalar multiply, all
// typed (no per-coefficient width switch). nr is the canonical non-residue;
// mont_nprime is the single-word +p^-1. Byte-identical to the byte EF-mul path.
template <typename T>
void EfMulTyped(char* a, char* b, char* o, npy_intp n, const npy_intp* strides,
                int k, int wb, T modulus, T mont_nprime, bool spare, T nr) {
  for (npy_intp i = 0; i < n; ++i) {
    T av[kMaxDegree], bv[kMaxDegree], prod[2 * kMaxDegree];
    for (int c = 0; c < 2 * k - 1; ++c) prod[c] = 0;
    for (int c = 0; c < k; ++c) {
      std::memcpy(&av[c], a + c * wb, sizeof(T));
      std::memcpy(&bv[c], b + c * wb, sizeof(T));
    }
    for (int ii = 0; ii < k; ++ii) {
      for (int jj = 0; jj < k; ++jj) {
        T t, s;
        MontMul<T>(av[ii], bv[jj], t, modulus, mont_nprime);
        ModAdd<T>(prod[ii + jj], t, s, modulus, spare);
        prod[ii + jj] = s;
      }
    }
    for (int ii = 2 * k - 2; ii >= k; --ii) {  // fold X^k = non_residue
      T sc = static_cast<T>(
          (static_cast<internal::make_promoted_t<T>>(nr) * prod[ii]) % modulus);
      T s;
      ModAdd<T>(prod[ii - k], sc, s, modulus, spare);
      prod[ii - k] = s;
    }
    for (int c = 0; c < k; ++c) std::memcpy(o + c * wb, &prod[c], sizeof(T));
    Advance(a, b, o, strides);
  }
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
      if (strides[0] == wb && strides[1] == wb && strides[2] == wb) {
        npy_intp total = n * wb;  // contiguous: one flat XOR sweep, vectorizes
        for (npy_intp i = 0; i < total; ++i) {
          o[i] = static_cast<char>(a[i] ^ b[i]);
        }
      } else {
        for (npy_intp i = 0; i < n; ++i) {
          for (int k = 0; k < wb; ++k) o[k] = static_cast<char>(a[k] ^ b[k]);
          Advance(a, b, o, strides);
        }
      }
      return 0;
    }
    if (d->tower_level <= 7) {  // native recursive Karatsuba tower multiply
      for (npy_intp i = 0; i < n; ++i) {
        modarith::BinaryTowerMul(d->tower_level,
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
          pyfield::LongAsBytesLE(rv, reinterpret_cast<unsigned char*>(o), wb);
      Py_DECREF(rv);
      if (rc < 0) return -1;
      Advance(a, b, o, strides);
    }
    return 0;
  }

  // Odd field: native fixed-width path where supported, Python-int otherwise.
  unsigned char mod_le[32];
  if (pyfield::LongAsBytesLE(d->modulus, mod_le, wb) >= 0) {
    modarith::PrimeField pf =
        modarith::PrimeField::Make(mod_le, wb, d->is_montgomery);
    const int k = d->degree;
    // Add/sub: a monomorphic typed loop at single-word widths
    // (auto-vectorizes), BigInt per-element at 128/256-bit. Prime (k == 1) and
    // extension share it.
    if (op != Op::kMul && pf.native) {
      if (wb == 4) {
        TypedAddSub<uint32_t, op == Op::kSub>(a, b, o, n, strides, k, wb,
                                              pf.p32, pf.spare);
        return 0;
      }
      if (wb == 8) {
        TypedAddSub<uint64_t, op == Op::kSub>(a, b, o, n, strides, k, wb,
                                              pf.p64, pf.spare);
        return 0;
      }
      for (npy_intp i = 0; i < n; ++i) {
        for (int c = 0; c < k; ++c) {
          const auto* ua = reinterpret_cast<const unsigned char*>(a) + c * wb;
          const auto* ub = reinterpret_cast<const unsigned char*>(b) + c * wb;
          auto* uo = reinterpret_cast<unsigned char*>(o) + c * wb;
          if (op == Op::kSub) {
            pf.Sub(ua, ub, uo);
          } else {
            pf.Add(ua, ub, uo);
          }
        }
        Advance(a, b, o, strides);
      }
      return 0;
    }
    if (k == 1 && pf.native) {  // prime multiply
      if (wb == 4) {
        if (pf.is_mont) {
          TypedMul<uint32_t, true>(a, b, o, n, strides, pf.p32,
                                   static_cast<uint32_t>(pf.inv));
        } else {
          TypedMul<uint32_t, false>(a, b, o, n, strides, pf.p32, 0);
        }
        return 0;
      }
      if (wb == 8) {
        if (pf.is_mont) {
          TypedMul<uint64_t, true>(a, b, o, n, strides, pf.p64, pf.inv);
        } else {
          TypedMul<uint64_t, false>(a, b, o, n, strides, pf.p64, 0);
        }
        return 0;
      }
      for (npy_intp i = 0; i < n; ++i) {  // 128/256-bit
        pf.Mul(reinterpret_cast<const unsigned char*>(a),
               reinterpret_cast<const unsigned char*>(b),
               reinterpret_cast<unsigned char*>(o));
        Advance(a, b, o, strides);
      }
      return 0;
    }
    if (k > 1 && op == Op::kMul && pf.ext_native && pf.is_mont) {
      // Binomial polynomial mul. ext_native ⟹ base width 4 or 8
      // (PrimeField::Make); dispatch the typed Montgomery-space schoolbook
      // (coefficients stay in storage form, so the result is byte-identical to
      // the decode/compute/encode path). EfMulTyped runs the Montgomery kernel,
      // so it is Montgomery-only — a canonical extension falls through to the
      // Python-int path below (which decodes to canonical and is storage-safe).
      if (wb == 4) {
        uint32_t nr = 0;
        if (pyfield::LongAsBytesLE(d->non_residue,
                                   reinterpret_cast<unsigned char*>(&nr),
                                   4) >= 0) {
          EfMulTyped<uint32_t>(a, b, o, n, strides, k, wb, pf.p32,
                               static_cast<uint32_t>(pf.inv), pf.spare, nr);
          return 0;
        }
      } else {  // wb == 8
        uint64_t nr = 0;
        if (pyfield::LongAsBytesLE(d->non_residue,
                                   reinterpret_cast<unsigned char*>(&nr),
                                   8) >= 0) {
          EfMulTyped<uint64_t>(a, b, o, n, strides, k, wb, pf.p64, pf.inv,
                               pf.spare, nr);
          return 0;
        }
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

// --- inverse / negative / divide / power (host CPython path) --------------

bool CoeffsAllZero(FieldDescr* d, PyObject* const* coeffs) {
  for (int i = 0; i < d->degree; ++i) {
    int t = PyObject_IsTrue(coeffs[i]);
    if (t != 0) return false;  // nonzero or error: treat as nonzero
  }
  return true;
}

void DecCoeffs(FieldDescr* d, PyObject** coeffs) {
  for (int i = 0; i < d->degree; ++i) Py_XDECREF(coeffs[i]);
}

// out = a^e via square-and-multiply, e a canonical non-negative Python int.
// Multiplication is ComputeFieldOp(kMul), so this serves prime and extension
// elements alike; the caller owns the coefficient arrays.
int FieldPowCoeffs(FieldDescr* d, PyObject* const* a, PyObject* e,
                   PyObject** out) {
  unsigned char bits[520] = {0};  // 16 coeffs x 256-bit base, ceil(bits/8)
  size_t nbits = _PyLong_NumBits(e);
  if (nbits > sizeof(bits) * 8) {
    PyErr_SetString(PyExc_OverflowError, "field exponent too large");
    return -1;
  }
  if (pyfield::LongAsBytesLE(e, bits, (nbits + 7) / 8) < 0) return -1;
  PyObject* acc[kMaxDegree] = {nullptr};
  acc[0] = PyLong_FromLong(1);  // one
  if (acc[0] == nullptr) return -1;
  for (int i = 1; i < d->degree; ++i) {
    acc[i] = PyLong_FromLong(0);
    if (acc[i] == nullptr) {
      for (int j = 0; j < i; ++j) Py_DECREF(acc[j]);
      return -1;
    }
  }
  for (Py_ssize_t i = static_cast<Py_ssize_t>(nbits) - 1; i >= 0; --i) {
    PyObject* sq[kMaxDegree];
    if (ComputeFieldOp(d, Op::kMul, acc, acc, sq) < 0) {
      DecCoeffs(d, acc);
      return -1;
    }
    for (int j = 0; j < d->degree; ++j) Py_SETREF(acc[j], sq[j]);
    if ((bits[i >> 3] >> (i & 7)) & 1) {
      PyObject* m[kMaxDegree];
      if (ComputeFieldOp(d, Op::kMul, acc, a, m) < 0) {
        DecCoeffs(d, acc);
        return -1;
      }
      for (int j = 0; j < d->degree; ++j) Py_SETREF(acc[j], m[j]);
    }
  }
  for (int i = 0; i < d->degree; ++i) out[i] = acc[i];
  return 0;
}

// out = a^-1. Fermat: a^(q-2) with q = p^degree (odd fields). Zero input sets
// ZeroDivisionError, matching the legacy scalar semantics.
int FieldInvCoeffs(FieldDescr* d, PyObject* const* a, PyObject** out) {
  if (CoeffsAllZero(d, a)) {
    PyErr_SetString(PyExc_ZeroDivisionError, "division by zero field element");
    return -1;
  }
  if (d->degree == 1) {
    out[0] = pyfield::ModInv(d->modulus, a[0]);
    return out[0] == nullptr ? -1 : 0;
  }
  PyObject* deg = PyLong_FromLong(d->degree);
  PyObject* q = deg ? PyNumber_Power(d->modulus, deg, Py_None) : nullptr;
  Py_XDECREF(deg);
  PyObject* two = q ? PyLong_FromLong(2) : nullptr;
  PyObject* e = two ? PyNumber_Subtract(q, two) : nullptr;
  Py_XDECREF(q);
  Py_XDECREF(two);
  if (e == nullptr) return -1;
  int rc = FieldPowCoeffs(d, a, e, out);
  Py_DECREF(e);
  return rc;
}

// Binary tower inverse: a^(2^n - 2) with n = 2^level, via TowerMul.
PyObject* TowerInv(FieldDescr* d, PyObject* a) {
  int t = PyObject_IsTrue(a);
  if (t < 0) return nullptr;
  if (t == 0) {
    PyErr_SetString(PyExc_ZeroDivisionError, "division by zero field element");
    return nullptr;
  }
  PyObject* one = PyLong_FromLong(1);
  PyObject* e = one ? PyNumber_Subtract(d->value_mask, one) : nullptr;
  Py_XDECREF(one);  // e = 2^n - 2 (mask = 2^n - 1)
  if (e == nullptr) return nullptr;
  size_t nbits = _PyLong_NumBits(e);
  unsigned char bits[520] = {0};
  if (nbits > sizeof(bits) * 8 ||
      pyfield::LongAsBytesLE(e, bits, (nbits + 7) / 8) < 0) {
    Py_DECREF(e);
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_OverflowError, "tower exponent too large");
    }
    return nullptr;
  }
  Py_DECREF(e);
  PyObject* acc = PyLong_FromLong(1);
  for (Py_ssize_t i = static_cast<Py_ssize_t>(nbits) - 1;
       acc != nullptr && i >= 0; --i) {
    PyObject* sq = TowerMul(d->tower_level, acc, acc);
    Py_SETREF(acc, sq);
    if (acc != nullptr && ((bits[i >> 3] >> (i & 7)) & 1)) {
      PyObject* m = TowerMul(d->tower_level, acc, a);
      Py_SETREF(acc, m);
    }
  }
  return acc;
}

// Zero encodes as all-zero bytes in every storage form (canonical 0,
// mont(0) = 0, binary 0), so element-nonzero is byte-nonzero. numpy calls
// this for np.nonzero / truthiness; leaving the slot NULL segfaults.
npy_bool NonZero(void* data, void* arr) {
  PyArray_Descr* descr = PyArray_DESCR(reinterpret_cast<PyArrayObject*>(arr));
  const auto* pbytes = static_cast<const unsigned char*>(data);
  for (npy_intp i = 0; i < descr->elsize; ++i) {
    if (pbytes[i]) return 1;
  }
  return 0;
}

NPY_CASTING FieldUnaryResolve(struct PyArrayMethodObject_tag* /*method*/,
                              PyArray_DTypeMeta* const* /*dtypes*/,
                              PyArray_Descr* const* given, PyArray_Descr** loop,
                              npy_intp* view_offset) {
  if (given[1] != nullptr && !SameField(AsField(given[0]), AsField(given[1]))) {
    PyErr_SetString(PyExc_TypeError,
                    "field operation output requires the same field");
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

int NegLoop(PyArrayMethod_Context* context, char* const* data,
            const npy_intp* dimensions, const npy_intp* strides,
            NpyAuxData* /*aux*/) {
  FieldDescr* d = AsField(context->descriptors[0]);
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* o = data[1];
  if (d->kind == kBinaryTower) {  // characteristic 2: -x == x
    for (npy_intp i = 0; i < n; ++i) {
      std::memcpy(o, a, d->base.elsize);
      a += strides[0];
      o += strides[1];
    }
    return 0;
  }
  for (npy_intp i = 0; i < n; ++i) {
    PyObject* in[kMaxDegree];
    if (DecodeElement(d, a, in) < 0) return -1;
    PyObject* neg[kMaxDegree];
    int rc = 0;
    for (int k = 0; k < d->degree && rc == 0; ++k) {
      neg[k] = pyfield::ModSub(d->modulus, d->modulus, in[k]);  // (p - c) mod p
      if (neg[k] == nullptr) {
        for (int j = 0; j < k; ++j) Py_DECREF(neg[j]);
        rc = -1;
      }
    }
    DecCoeffs(d, in);
    if (rc < 0) return -1;
    int erc = EncodeElement(d, o, neg);
    DecCoeffs(d, neg);
    if (erc < 0) return -1;
    a += strides[0];
    o += strides[1];
  }
  return 0;
}

int DivLoop(PyArrayMethod_Context* context, char* const* data,
            const npy_intp* dimensions, const npy_intp* strides,
            NpyAuxData* /*aux*/) {
  FieldDescr* d = AsField(context->descriptors[0]);
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  for (npy_intp i = 0; i < n; ++i) {
    if (d->kind == kBinaryTower) {
      PyObject* av = DecodeBinary(d, a);
      PyObject* bv = av != nullptr ? DecodeBinary(d, b) : nullptr;
      PyObject* binv = bv != nullptr ? TowerInv(d, bv) : nullptr;
      PyObject* r =
          binv != nullptr ? TowerMul(d->tower_level, av, binv) : nullptr;
      Py_XDECREF(av);
      Py_XDECREF(bv);
      Py_XDECREF(binv);
      if (r == nullptr) return -1;
      int erc = EncodeBinary(d, o, r);
      Py_DECREF(r);
      if (erc < 0) return -1;
    } else {
      PyObject* av[kMaxDegree];
      PyObject* bv[kMaxDegree];
      PyObject* binv[kMaxDegree];
      PyObject* r[kMaxDegree];
      if (DecodeElement(d, a, av) < 0) return -1;
      if (DecodeElement(d, b, bv) < 0) {
        DecCoeffs(d, av);
        return -1;
      }
      int rc = FieldInvCoeffs(d, bv, binv);
      DecCoeffs(d, bv);
      if (rc == 0) {
        rc = ComputeFieldOp(d, Op::kMul, av, binv, r);
        DecCoeffs(d, binv);
      }
      DecCoeffs(d, av);
      if (rc < 0) return -1;
      int erc = EncodeElement(d, o, r);
      DecCoeffs(d, r);
      if (erc < 0) return -1;
    }
    Advance(a, b, o, strides);
  }
  return 0;
}

// Power: (field, int64) -> field. Negative exponents invert first, so
// x ** -1 is the field inverse and 0 ** -1 raises ZeroDivisionError.
NPY_CASTING PowResolve(struct PyArrayMethodObject_tag* /*method*/,
                       PyArray_DTypeMeta* const* /*dtypes*/,
                       PyArray_Descr* const* given, PyArray_Descr** loop,
                       npy_intp* view_offset) {
  if (given[2] != nullptr && !SameField(AsField(given[0]), AsField(given[2]))) {
    PyErr_SetString(PyExc_TypeError,
                    "field operation output requires the same field");
    return static_cast<NPY_CASTING>(-1);
  }
  Py_INCREF(given[0]);
  loop[0] = given[0];
  PyArray_Descr* exp_descr = PyArray_DescrFromType(NPY_INT64);
  if (exp_descr == nullptr) {
    Py_DECREF(loop[0]);
    return static_cast<NPY_CASTING>(-1);
  }
  loop[1] = exp_descr;
  PyArray_Descr* out = given[2] == nullptr ? given[0] : given[2];
  Py_INCREF(out);
  loop[2] = out;
  *view_offset = NPY_MIN_INTP;
  return NPY_NO_CASTING;
}

int PowLoop(PyArrayMethod_Context* context, char* const* data,
            const npy_intp* dimensions, const npy_intp* strides,
            NpyAuxData* /*aux*/) {
  FieldDescr* d = AsField(context->descriptors[0]);
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  for (npy_intp i = 0; i < n; ++i) {
    int64_t e64;
    std::memcpy(&e64, b, sizeof(e64));
    PyObject* e = PyLong_FromLongLong(e64 < 0 ? -e64 : e64);
    if (e == nullptr) return -1;
    int rc = 0;
    if (d->kind == kBinaryTower) {
      PyObject* av = DecodeBinary(d, a);
      PyObject* base = av;
      if (av != nullptr && e64 < 0) {
        base = TowerInv(d, av);
        Py_DECREF(av);
      }
      PyObject* acc = nullptr;
      if (base != nullptr) {
        acc = PyLong_FromLong(1);
        size_t nbits = _PyLong_NumBits(e);
        unsigned char bits[520] = {0};
        if (acc != nullptr && nbits <= sizeof(bits) * 8 &&
            pyfield::LongAsBytesLE(e, bits, (nbits + 7) / 8) == 0) {
          for (Py_ssize_t k = static_cast<Py_ssize_t>(nbits) - 1;
               acc != nullptr && k >= 0; --k) {
            Py_SETREF(acc, TowerMul(d->tower_level, acc, acc));
            if (acc != nullptr && ((bits[k >> 3] >> (k & 7)) & 1)) {
              Py_SETREF(acc, TowerMul(d->tower_level, acc, base));
            }
          }
        } else {
          Py_CLEAR(acc);
        }
        Py_DECREF(base);
      }
      rc = acc == nullptr ? -1 : EncodeBinary(d, o, acc);
      Py_XDECREF(acc);
    } else {
      PyObject* av[kMaxDegree];
      PyObject* r[kMaxDegree];
      if (DecodeElement(d, a, av) < 0) {
        Py_DECREF(e);
        return -1;
      }
      if (e64 < 0) {
        PyObject* inv[kMaxDegree];
        rc = FieldInvCoeffs(d, av, inv);
        DecCoeffs(d, av);
        if (rc == 0) {
          rc = FieldPowCoeffs(d, inv, e, r);
          DecCoeffs(d, inv);
        }
      } else {
        rc = FieldPowCoeffs(d, av, e, r);
        DecCoeffs(d, av);
      }
      if (rc == 0) {
        rc = EncodeElement(d, o, r);
        DecCoeffs(d, r);
      }
    }
    Py_DECREF(e);
    if (rc < 0) return -1;
    Advance(a, b, o, strides);
  }
  return 0;
}

// --- scalar behavior ------------------------------------------------------

void Scalar_dealloc(PyObject* self) {
  FieldScalarObject* s = AsScalar(self);
  Py_XDECREF(s->descr);
  Py_XDECREF(s->value);
  Py_TYPE(self)->tp_free(self);
}

PyObject* MakeScalar(PyArray_Descr* descr, PyObject* value) {
  if (value == nullptr) return nullptr;
  auto* s = reinterpret_cast<FieldScalarObject*>(
      FieldScalar_Type.tp_alloc(&FieldScalar_Type, 0));
  if (s == nullptr) {
    Py_DECREF(value);
    return nullptr;
  }
  Py_INCREF(descr);
  s->descr = descr;
  s->value = value;  // stolen
  return reinterpret_cast<PyObject*>(s);
}

PyObject* Scalar_repr(PyObject* self) {
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  if (d->kind == kBinaryTower) {
    return PyUnicode_FromFormat("FieldScalar(%R, tower_level=%d)", s->value,
                                static_cast<int>(d->tower_level));
  }
  return PyUnicode_FromFormat("FieldScalar(%R, modulus=%R)", s->value,
                              d->modulus);
}

PyObject* Scalar_str(PyObject* self) {
  return PyObject_Str(AsScalar(self)->value);
}

Py_hash_t Scalar_hash(PyObject* self) {
  return PyObject_Hash(AsScalar(self)->value);
}

// Coerces `obj` to coefficients of field `d`: another scalar of the same
// field, or an integer (constant-term embedding). Returns the coefficient
// count, or -1 with a TypeError for anything else. `storage` owns the new
// references it fills in.
int CoerceOperand(FieldDescr* d, PyArray_Descr* descr, PyObject* obj,
                  PyObject** storage, PyObject** out) {
  if (IsFieldScalar(obj)) {
    FieldScalarObject* s = AsScalar(obj);
    if (!SameField(d, AsField(s->descr))) {
      PyErr_SetString(PyExc_TypeError,
                      "field operation requires identical fields");
      return -1;
    }
    return ScalarCoeffs(s, out);
  }
  if (PyIndex_Check(obj)) {
    PyObject* idx = PyNumber_Index(obj);
    if (idx == nullptr) return -1;
    if (d->kind == kBinaryTower) {
      PyObject* masked = PyNumber_And(idx, d->value_mask);
      Py_DECREF(idx);
      if (masked == nullptr) return -1;
      storage[0] = masked;
      out[0] = masked;
      return 1;
    }
    PyObject* red = PyNumber_Remainder(idx, d->modulus);
    Py_DECREF(idx);
    if (red == nullptr) return -1;
    storage[0] = red;
    out[0] = red;
    for (int i = 1; i < d->degree; ++i) {
      storage[i] = PyLong_FromLong(0);
      if (storage[i] == nullptr) return -1;
      out[i] = storage[i];
    }
    return d->degree;
  }
  PyErr_SetString(PyExc_TypeError,
                  "field arithmetic needs a field scalar of the same field or "
                  "an integer");
  return -1;
}

// Packs coefficients back into a scalar value (int or tuple), stealing them.
PyObject* ScalarValueFromCoeffs(FieldDescr* d, PyObject** coeffs) {
  if (d->kind == kBinaryTower || d->degree == 1) {
    return coeffs[0];
  }
  PyObject* tuple = PyTuple_New(d->degree);
  if (tuple == nullptr) {
    DecCoeffs(d, coeffs);
    return nullptr;
  }
  for (int i = 0; i < d->degree; ++i) PyTuple_SET_ITEM(tuple, i, coeffs[i]);
  return tuple;
}

enum class ScalarOp { kAdd, kSub, kMul, kDiv };

PyObject* ScalarBinOp(PyObject* a, PyObject* b, ScalarOp op) {
  PyObject* self = IsFieldScalar(a) ? a : b;
  if (!IsFieldScalar(self)) Py_RETURN_NOTIMPLEMENTED;
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  PyObject* store_l[kMaxDegree] = {nullptr};
  PyObject* store_r[kMaxDegree] = {nullptr};
  PyObject* lhs[kMaxDegree];
  PyObject* rhs[kMaxDegree];
  PyObject* result = nullptr;
  if (CoerceOperand(d, s->descr, a, store_l, lhs) < 0 ||
      CoerceOperand(d, s->descr, b, store_r, rhs) < 0) {
    goto done;
  }
  if (d->kind == kBinaryTower) {
    if (op == ScalarOp::kAdd || op == ScalarOp::kSub) {
      result = MakeScalar(s->descr, PyNumber_Xor(lhs[0], rhs[0]));
    } else if (op == ScalarOp::kMul) {
      result = MakeScalar(s->descr, TowerMul(d->tower_level, lhs[0], rhs[0]));
    } else {
      PyObject* inv = TowerInv(d, rhs[0]);
      result =
          inv == nullptr
              ? nullptr
              : MakeScalar(s->descr, TowerMul(d->tower_level, lhs[0], inv));
      Py_XDECREF(inv);
    }
    goto done;
  }
  {
    PyObject* out[kMaxDegree];
    int rc;
    if (op == ScalarOp::kDiv) {
      PyObject* inv[kMaxDegree];
      rc = FieldInvCoeffs(d, rhs, inv);
      if (rc == 0) {
        rc = ComputeFieldOp(d, Op::kMul, lhs, inv, out);
        DecCoeffs(d, inv);
      }
    } else {
      Op fop = op == ScalarOp::kAdd   ? Op::kAdd
               : op == ScalarOp::kSub ? Op::kSub
                                      : Op::kMul;
      rc = ComputeFieldOp(d, fop, lhs, rhs, out);
    }
    if (rc == 0) result = MakeScalar(s->descr, ScalarValueFromCoeffs(d, out));
  }
done:
  for (int i = 0; i < kMaxDegree; ++i) {
    Py_XDECREF(store_l[i]);
    Py_XDECREF(store_r[i]);
  }
  return result;
}

PyObject* Scalar_add(PyObject* a, PyObject* b) {
  return ScalarBinOp(a, b, ScalarOp::kAdd);
}
PyObject* Scalar_sub(PyObject* a, PyObject* b) {
  return ScalarBinOp(a, b, ScalarOp::kSub);
}
PyObject* Scalar_mul(PyObject* a, PyObject* b) {
  return ScalarBinOp(a, b, ScalarOp::kMul);
}
PyObject* Scalar_div(PyObject* a, PyObject* b) {
  return ScalarBinOp(a, b, ScalarOp::kDiv);
}

PyObject* Scalar_negative(PyObject* self) {
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  if (d->kind == kBinaryTower) {  // characteristic 2
    Py_INCREF(s->value);
    return MakeScalar(s->descr, s->value);
  }
  PyObject* in[kMaxDegree];
  int k = ScalarCoeffs(s, in);
  PyObject* out[kMaxDegree];
  for (int i = 0; i < k; ++i) {
    out[i] = pyfield::ModSub(d->modulus, d->modulus, in[i]);
    if (out[i] == nullptr) {
      for (int j = 0; j < i; ++j) Py_DECREF(out[j]);
      return nullptr;
    }
  }
  return MakeScalar(s->descr, ScalarValueFromCoeffs(d, out));
}

PyObject* Scalar_power(PyObject* self, PyObject* exp, PyObject* mod) {
  if (mod != Py_None || !IsFieldScalar(self)) Py_RETURN_NOTIMPLEMENTED;
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  PyObject* e = PyNumber_Index(exp);
  if (e == nullptr) return nullptr;
  int negative = PyObject_RichCompareBool(e, PyLong_FromLong(0), Py_LT) == 1;
  if (negative) {
    PyObject* abs_e = PyNumber_Absolute(e);
    Py_SETREF(e, abs_e);
    if (e == nullptr) return nullptr;
  }
  PyObject* base[kMaxDegree];
  int k = ScalarCoeffs(s, base);
  PyObject* result = nullptr;
  if (d->kind == kBinaryTower) {
    PyObject* b0 = base[0];
    PyObject* inv = negative ? TowerInv(d, b0) : nullptr;
    if (!negative || inv != nullptr) {
      PyObject* acc = PyLong_FromLong(1);
      PyObject* cur = negative ? inv : b0;
      size_t nbits = _PyLong_NumBits(e);
      unsigned char bits[520] = {0};
      if (acc != nullptr && nbits <= sizeof(bits) * 8 &&
          pyfield::LongAsBytesLE(e, bits, (nbits + 7) / 8) == 0) {
        for (Py_ssize_t i = static_cast<Py_ssize_t>(nbits) - 1;
             acc != nullptr && i >= 0; --i) {
          Py_SETREF(acc, TowerMul(d->tower_level, acc, acc));
          if (acc != nullptr && ((bits[i >> 3] >> (i & 7)) & 1)) {
            Py_SETREF(acc, TowerMul(d->tower_level, acc, cur));
          }
        }
      } else {
        Py_CLEAR(acc);
      }
      if (acc != nullptr) result = MakeScalar(s->descr, acc);
    }
    Py_XDECREF(inv);
  } else {
    PyObject* out[kMaxDegree];
    int rc;
    if (negative) {
      PyObject* inv[kMaxDegree];
      rc = FieldInvCoeffs(d, base, inv);
      if (rc == 0) {
        rc = FieldPowCoeffs(d, inv, e, out);
        DecCoeffs(d, inv);
      }
    } else {
      rc = FieldPowCoeffs(d, base, e, out);
    }
    if (rc == 0) result = MakeScalar(s->descr, ScalarValueFromCoeffs(d, out));
  }
  (void)k;
  Py_DECREF(e);
  return result;
}

PyObject* Scalar_int(PyObject* self) {
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  if (d->kind != kBinaryTower && d->degree > 1) {
    PyErr_SetString(PyExc_TypeError,
                    "an extension-field element has no single integer value; "
                    "index its coefficients");
    return nullptr;
  }
  Py_INCREF(s->value);
  return s->value;
}

PyObject* Scalar_richcompare(PyObject* a, PyObject* b, int op) {
  if (op != Py_EQ && op != Py_NE) Py_RETURN_NOTIMPLEMENTED;
  PyObject* self = IsFieldScalar(a) ? a : b;
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  PyObject* store_l[kMaxDegree] = {nullptr};
  PyObject* store_r[kMaxDegree] = {nullptr};
  PyObject* lhs[kMaxDegree];
  PyObject* rhs[kMaxDegree];
  PyObject* result = nullptr;
  if (CoerceOperand(d, s->descr, a, store_l, lhs) < 0 ||
      CoerceOperand(d, s->descr, b, store_r, rhs) < 0) {
    PyErr_Clear();
    result = Py_NotImplemented;
    Py_INCREF(result);
  } else {
    int k = d->kind == kBinaryTower ? 1 : d->degree;
    bool eq = true;
    for (int i = 0; i < k && eq; ++i) {
      int cmp = PyObject_RichCompareBool(lhs[i], rhs[i], Py_EQ);
      if (cmp < 0) {
        eq = false;
        result = nullptr;
      } else {
        eq = cmp == 1;
      }
    }
    if (result == nullptr && !PyErr_Occurred()) {
      result = PyBool_FromLong(op == Py_EQ ? eq : !eq);
    }
  }
  for (int i = 0; i < kMaxDegree; ++i) {
    Py_XDECREF(store_l[i]);
    Py_XDECREF(store_r[i]);
  }
  return result;
}

// Extension elements index their coefficients (constant term first); a
// degree-1 or binary element has length 1.
Py_ssize_t Scalar_length(PyObject* self) {
  FieldDescr* d = AsField(AsScalar(self)->descr);
  return d->kind == kBinaryTower ? 1 : d->degree;
}

PyObject* Scalar_item(PyObject* self, Py_ssize_t i) {
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  Py_ssize_t n = d->kind == kBinaryTower ? 1 : d->degree;
  if (i < 0 || i >= n) {
    PyErr_SetString(PyExc_IndexError, "field coefficient index out of range");
    return nullptr;
  }
  PyObject* coeffs[kMaxDegree];
  ScalarCoeffs(s, coeffs);
  Py_INCREF(coeffs[i]);
  return coeffs[i];
}

// `.raw` is the stored representation (Montgomery-encoded when the field is),
// mirroring the legacy scalars' raw accessor.
PyObject* Scalar_get_raw(PyObject* self, void* /*closure*/) {
  FieldScalarObject* s = AsScalar(self);
  FieldDescr* d = AsField(s->descr);
  const int elsize = s->descr->elsize;
  std::vector<char> buf(static_cast<size_t>(elsize), 0);
  if (d->kind == kBinaryTower) {
    if (EncodeBinary(d, buf.data(), s->value) < 0) return nullptr;
  } else {
    PyObject* coeffs[kMaxDegree];
    ScalarCoeffs(s, coeffs);
    if (EncodeElement(d, buf.data(), coeffs) < 0) return nullptr;
  }
  if (d->kind != kBinaryTower && d->degree > 1) {
    PyObject* tuple = PyTuple_New(d->degree);
    if (tuple == nullptr) return nullptr;
    for (int i = 0; i < d->degree; ++i) {
      PyObject* limb =
          _PyLong_FromByteArray(reinterpret_cast<unsigned char*>(buf.data()) +
                                    i * d->base_width_bytes,
                                d->base_width_bytes, 1, 0);
      if (limb == nullptr) {
        Py_DECREF(tuple);
        return nullptr;
      }
      PyTuple_SET_ITEM(tuple, i, limb);
    }
    return tuple;
  }
  return _PyLong_FromByteArray(reinterpret_cast<unsigned char*>(buf.data()),
                               d->base_width_bytes, 1, 0);
}

PyObject* Scalar_get_dtype(PyObject* self, void* /*closure*/) {
  PyObject* descr = reinterpret_cast<PyObject*>(AsScalar(self)->descr);
  Py_INCREF(descr);
  return descr;
}

PyObject* Scalar_method_item(PyObject* self, PyObject* /*unused*/) {
  FieldScalarObject* s = AsScalar(self);
  Py_INCREF(s->value);
  return s->value;  // canonical int, or tuple of coefficients
}

PyGetSetDef Scalar_getset[] = {
    {"raw", Scalar_get_raw, nullptr,
     "stored representation (Montgomery-encoded when the field is)", nullptr},
    {"dtype", Scalar_get_dtype, nullptr, "the field dtype of this element",
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef Scalar_methods[] = {
    {"item", Scalar_method_item, METH_NOARGS,
     "canonical Python value: an int, or a tuple of coefficients"},
    {nullptr, nullptr, 0, nullptr},
};

PyNumberMethods Scalar_as_number = {};
PySequenceMethods Scalar_as_sequence = {};
// Must be installed even though it is empty: the base type
// (numpy.generic) supplies an mp_subscript that reads a numpy scalar's inline
// data buffer, which this object does not have — inheriting it turns `x[i]`
// into a wild read. An owned, zeroed mapping table sends indexing to
// sq_item instead.
PyMappingMethods Scalar_as_mapping = {};

// Own the subscript slot rather than leaving it NULL: the base type
// (numpy.generic) supplies one that reads a numpy scalar's inline data buffer,
// which this object does not have.
PyObject* Scalar_subscript(PyObject* self, PyObject* key) {
  Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
  if (i == -1 && PyErr_Occurred()) return nullptr;
  if (i < 0) i += Scalar_length(self);
  return Scalar_item(self, i);
}

// Comparison: (field, field) -> bool. Same-descriptor encodings are injective
// (canonical residues are unique, Montgomery scaling is a bijection, binary
// values are masked), so byte equality is field equality.
NPY_CASTING CmpResolve(struct PyArrayMethodObject_tag* /*method*/,
                       PyArray_DTypeMeta* const* /*dtypes*/,
                       PyArray_Descr* const* given, PyArray_Descr** loop,
                       npy_intp* view_offset) {
  if (!SameField(AsField(given[0]), AsField(given[1]))) {
    PyErr_SetString(PyExc_TypeError,
                    "field comparison requires identical fields");
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
  npy_intp elsize = context->descriptors[0]->elsize;
  npy_intp n = dimensions[0];
  char* a = data[0];
  char* b = data[1];
  char* o = data[2];
  for (npy_intp i = 0; i < n; ++i) {
    bool eq = std::memcmp(a, b, elsize) == 0;
    *reinterpret_cast<npy_bool*>(o) = (negate ? !eq : eq) ? 1 : 0;
    a += strides[0];
    b += strides[1];
    o += strides[2];
  }
  return 0;
}

bool AddCmpLoop(PyObject* numpy, const char* name,
                PyArrayMethod_StridedLoop* loop) {
  PyArray_Descr* bool_descr = PyArray_DescrFromType(NPY_BOOL);
  if (bool_descr == nullptr) {
    return false;
  }
  PyArray_DTypeMeta* booldt =
      reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(bool_descr));
  Py_DECREF(bool_descr);
  PyArray_DTypeMeta* dtypes[3] = {&FieldDType, &FieldDType, booldt};
  if (!nep42::AddUfuncLoop(numpy, name, "field_compare", 2, dtypes,
                           reinterpret_cast<void*>(CmpResolve),
                           reinterpret_cast<void*>(loop))) {
    return false;
  }
  PyArray_DTypeMeta* const int_dts[] = {
      &PyArray_PyLongDType, &PyArray_Int8DType,   &PyArray_UInt8DType,
      &PyArray_Int16DType,  &PyArray_UInt16DType, &PyArray_Int32DType,
      &PyArray_UInt32DType, &PyArray_Int64DType,  &PyArray_UInt64DType,
  };
  for (PyArray_DTypeMeta* intdt : int_dts) {
    PyArray_DTypeMeta* rweak[3] = {&FieldDType, intdt, booldt};
    PyArray_DTypeMeta* lweak[3] = {intdt, &FieldDType, booldt};
    if (!nep42::AddUfuncLoop(numpy, name, "field_compare", 2, rweak,
                             reinterpret_cast<void*>(WeakCmpResolve),
                             reinterpret_cast<void*>(loop)) ||
        !nep42::AddUfuncLoop(numpy, name, "field_compare", 2, lweak,
                             reinterpret_cast<void*>(WeakCmpResolve),
                             reinterpret_cast<void*>(loop))) {
      return false;
    }
  }
  return true;
}

bool AddArithLoopFn(PyObject* numpy, const char* ufunc_name,
                    const char* spec_name, void* loop) {
  PyArray_DTypeMeta* dtypes[3] = {&FieldDType, &FieldDType, &FieldDType};
  if (!nep42::AddUfuncLoop(numpy, ufunc_name, spec_name, 2, dtypes,
                           reinterpret_cast<void*>(ArithResolve), loop)) {
    return false;
  }
  // Integer operand on either side: numpy's weak Python-int DType plus the
  // concrete widths (a python int and an int64 array reach the ufunc as
  // different DTypes, and neither is promoted into ours without a loop).
  PyArray_DTypeMeta* const int_dts[] = {
      &PyArray_PyLongDType, &PyArray_Int8DType,   &PyArray_UInt8DType,
      &PyArray_Int16DType,  &PyArray_UInt16DType, &PyArray_Int32DType,
      &PyArray_UInt32DType, &PyArray_Int64DType,  &PyArray_UInt64DType,
  };
  for (PyArray_DTypeMeta* intdt : int_dts) {
    PyArray_DTypeMeta* rweak[3] = {&FieldDType, intdt, &FieldDType};
    PyArray_DTypeMeta* lweak[3] = {intdt, &FieldDType, &FieldDType};
    if (!nep42::AddUfuncLoop(numpy, ufunc_name, spec_name, 2, rweak,
                             reinterpret_cast<void*>(WeakArithResolve), loop) ||
        !nep42::AddUfuncLoop(numpy, ufunc_name, spec_name, 2, lweak,
                             reinterpret_cast<void*>(WeakArithResolve), loop)) {
      return false;
    }
  }
  return true;
}

template <Op op>
bool AddArithLoop(PyObject* numpy, const char* ufunc_name) {
  return AddArithLoopFn(numpy, ufunc_name, "field_arith",
                        reinterpret_cast<void*>(ArithLoop<op>));
}

bool AddNegLoop(PyObject* numpy) {
  PyArray_DTypeMeta* dtypes[2] = {&FieldDType, &FieldDType};
  return nep42::AddUfuncLoop(numpy, "negative", "field_negate", 1, dtypes,
                             reinterpret_cast<void*>(FieldUnaryResolve),
                             reinterpret_cast<void*>(NegLoop));
}

bool AddPowLoop(PyObject* numpy) {
  PyArray_Descr* i64 = PyArray_DescrFromType(NPY_INT64);
  if (i64 == nullptr) return false;
  PyArray_DTypeMeta* i64dt = reinterpret_cast<PyArray_DTypeMeta*>(Py_TYPE(i64));
  Py_DECREF(i64);
  PyArray_DTypeMeta* dtypes[3] = {&FieldDType, i64dt, &FieldDType};
  if (!nep42::AddUfuncLoop(numpy, "power", "field_power", 2, dtypes,
                           reinterpret_cast<void*>(PowResolve),
                           reinterpret_cast<void*>(PowLoop))) {
    return false;
  }
  // `a ** 3` hands numpy a weak Python int, whose DType is neither int64 nor
  // ours; a second loop over PyLongDType makes the natural spelling work
  // (PowResolve pins the exponent operand to int64 either way).
  PyArray_DTypeMeta* weak[3] = {&FieldDType, &PyArray_PyLongDType, &FieldDType};
  return nep42::AddUfuncLoop(numpy, "power", "field_power", 2, weak,
                             reinterpret_cast<void*>(PowResolve),
                             reinterpret_cast<void*>(PowLoop));
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
  Scalar_as_number.nb_add = Scalar_add;
  Scalar_as_number.nb_subtract = Scalar_sub;
  Scalar_as_number.nb_multiply = Scalar_mul;
  Scalar_as_number.nb_true_divide = Scalar_div;
  Scalar_as_number.nb_negative = Scalar_negative;
  Scalar_as_number.nb_power = Scalar_power;
  Scalar_as_number.nb_int = Scalar_int;
  Scalar_as_number.nb_index = Scalar_int;
  Scalar_as_sequence.sq_length = Scalar_length;
  Scalar_as_sequence.sq_item = Scalar_item;
  FieldScalar_Type.tp_name = "zk_dtypes._zk_dtypes_ext.FieldScalar";
  FieldScalar_Type.tp_basicsize = sizeof(FieldScalarObject);
  FieldScalar_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  FieldScalar_Type.tp_base = &PyGenericArrType_Type;
  FieldScalar_Type.tp_dealloc = Scalar_dealloc;
  FieldScalar_Type.tp_repr = Scalar_repr;
  FieldScalar_Type.tp_str = Scalar_str;
  FieldScalar_Type.tp_hash = Scalar_hash;
  FieldScalar_Type.tp_richcompare = Scalar_richcompare;
  FieldScalar_Type.tp_as_number = &Scalar_as_number;
  FieldScalar_Type.tp_as_sequence = &Scalar_as_sequence;
  Scalar_as_mapping.mp_subscript = Scalar_subscript;
  FieldScalar_Type.tp_as_mapping = &Scalar_as_mapping;
  FieldScalar_Type.tp_getset = Scalar_getset;
  FieldScalar_Type.tp_methods = Scalar_methods;
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
  copy_cast.name = "field_copy";
  copy_cast.nin = 1;
  copy_cast.nout = 1;
  // Neither in-family cast loses information: Montgomery <-> canonical is the
  // same field element in another storage form, and base -> extension embeds
  // it as the constant coefficient. Rating this same-kind (not unsafe) is what
  // lets mixed base/extension arithmetic promote under numpy's default rule.
  copy_cast.casting = NPY_SAME_KIND_CASTING;
  // The cast loop runs CPython API (decode/re-encode through Python
  // ints), so numpy must keep the GIL held around it.
  copy_cast.flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(
      NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI);
  copy_cast.dtypes = cast_dtypes;
  copy_cast.slots = cast_slots;
  // int <-> field casts for the standard widths; the same resolver/loop pair
  // per width, so the table is generated rather than hand-repeated.
  struct IntCast {
    PyArray_DTypeMeta* dt;
    PyArrayMethod_StridedLoop* to_field;
    PyArrayMethod_StridedLoop* from_field;
  };
  static const IntCast kIntCasts[] = {
      {&PyArray_Int8DType, IntToFieldLoop<int8_t>, FieldToIntLoop<int8_t>},
      {&PyArray_UInt8DType, IntToFieldLoop<uint8_t>, FieldToIntLoop<uint8_t>},
      {&PyArray_Int16DType, IntToFieldLoop<int16_t>, FieldToIntLoop<int16_t>},
      {&PyArray_UInt16DType, IntToFieldLoop<uint16_t>,
       FieldToIntLoop<uint16_t>},
      {&PyArray_Int32DType, IntToFieldLoop<int32_t>, FieldToIntLoop<int32_t>},
      {&PyArray_UInt32DType, IntToFieldLoop<uint32_t>,
       FieldToIntLoop<uint32_t>},
      {&PyArray_Int64DType, IntToFieldLoop<int64_t>, FieldToIntLoop<int64_t>},
      {&PyArray_UInt64DType, IntToFieldLoop<uint64_t>,
       FieldToIntLoop<uint64_t>},
  };
  constexpr size_t kNumIntCasts = sizeof(kIntCasts) / sizeof(kIntCasts[0]);
  static PyArrayMethod_Spec int_specs[2 * kNumIntCasts];
  static PyArray_DTypeMeta* int_dtypes[2 * kNumIntCasts][2];
  static PyType_Slot int_slots[2 * kNumIntCasts][3];
  PyArrayMethod_Spec* casts[2 * kNumIntCasts + 2] = {&copy_cast, nullptr};
  size_t next = 1;
  for (size_t i = 0; i < kNumIntCasts; ++i) {
    for (int dir = 0; dir < 2; ++dir) {  // 0: int -> field, 1: field -> int
      size_t k = 2 * i + dir;
      int_dtypes[k][0] = dir == 0 ? kIntCasts[i].dt : &FieldDType;
      int_dtypes[k][1] = dir == 0 ? &FieldDType : kIntCasts[i].dt;
      int_slots[k][0] = {NPY_METH_resolve_descriptors,
                         reinterpret_cast<void*>(dir == 0 ? IntToFieldResolve
                                                          : FieldToIntResolve)};
      int_slots[k][1] = {
          NPY_METH_strided_loop,
          reinterpret_cast<void*>(dir == 0 ? kIntCasts[i].to_field
                                           : kIntCasts[i].from_field)};
      int_slots[k][2] = {0, nullptr};
      int_specs[k] = {};
      int_specs[k].name = dir == 0 ? "int_to_field" : "field_to_int";
      int_specs[k].nin = 1;
      int_specs[k].nout = 1;
      // int -> field is same-kind (in-range values are preserved, others
      // reduce mod p, exactly as numpy's int64 -> int32 narrows); the reverse
      // stays unsafe so a field never silently degrades to an integer in a
      // ufunc — explicit .astype(int) still works.
      int_specs[k].casting =
          dir == 0 ? NPY_SAME_KIND_CASTING : NPY_UNSAFE_CASTING;
      int_specs[k].flags = NPY_METH_REQUIRES_PYAPI;
      int_specs[k].dtypes = int_dtypes[k];
      int_specs[k].slots = int_slots[k];
      casts[next++] = &int_specs[k];
    }
  }
  casts[next] = nullptr;

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
            AddArithLoop<Op::kMul>(numpy, "multiply") &&
            AddCmpLoop(numpy, "equal", CmpLoop<false>) &&
            AddCmpLoop(numpy, "not_equal", CmpLoop<true>) &&
            AddArithLoopFn(numpy, "divide", "field_divide",
                           reinterpret_cast<void*>(DivLoop)) &&
            AddNegLoop(numpy) && AddPowLoop(numpy);
  Py_DECREF(numpy);
  return ok;
}

}  // namespace zk_dtypes
