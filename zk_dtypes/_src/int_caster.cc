// Must be included first
// clang-format off
#include "zk_dtypes/_src/numpy.h"
// clang-format on

#include "zk_dtypes/_src/int_caster.h"

namespace zk_dtypes {

bool IntCaster::CastInt(PyObject* obj) {
  if (PyLong_Check(obj)) {
    int sign = _PyLong_Sign(obj);
    if (sign == -1) {
      static_assert(sizeof(int64_t) >= sizeof(long long));
      int64_t v = PyLong_AsLongLong(obj);
      if (IsOverflow(v)) {
        SetOverflowError();
        return false;
      }
      return Cast(v);
    } else {
      static_assert(sizeof(uint64_t) >= sizeof(unsigned long long));
      uint64_t v = PyLong_AsUnsignedLongLong(obj);
      if (IsOverflow(v)) {
        SetOverflowError();
        return false;
      }
      return Cast(v);
    }
  }
  return CastNumpyInt(obj);
}

bool IntCaster::CastNumpyInt(PyObject* obj) {
  if (!PyArray_IsScalar(obj, Integer)) return false;

  PyArray_Descr* descr = PyArray_DescrFromScalar(obj);

  if (descr->type_num == NPY_INT8 || descr->type_num == NPY_INT16 ||
      descr->type_num == NPY_INT32 || descr->type_num == NPY_INT64) {
    int64_t v;
    PyArray_CastScalarToCtype(obj, &v, PyArray_DescrFromType(NPY_INT64));

    if (IsOverflow(v)) {
      SetOverflowError();
      return false;
    }
    return Cast(v);
  } else if (descr->type_num == NPY_UINT8 || descr->type_num == NPY_UINT16 ||
             descr->type_num == NPY_UINT32 || descr->type_num == NPY_UINT64) {
    uint64_t v;
    PyArray_CastScalarToCtype(obj, &v, PyArray_DescrFromType(NPY_UINT64));

    if (IsOverflow(v)) {
      SetOverflowError();
      return false;
    }
    return Cast(v);
  }
  return false;
}

}  // namespace zk_dtypes
