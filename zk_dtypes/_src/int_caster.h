#ifndef ZK_DTYPES__SRC_INT_CASTER_H_
#define ZK_DTYPES__SRC_INT_CASTER_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include <Python.h>

#include "zk_dtypes/include/big_int.h"

namespace zk_dtypes {

class IntCaster {
 public:
  IntCaster() = default;
  virtual ~IntCaster() = default;

  bool CastInt(PyObject* obj);
  bool CastNumpyInt(PyObject* obj);

 private:
  virtual bool Cast(int64_t v) = 0;
  virtual bool Cast(uint64_t v) = 0;
  virtual bool IsOverflow(int64_t v) const { return false; }
  virtual bool IsOverflow(uint64_t v) const { return false; }
  virtual void SetOverflowError() const = 0;
};

template <typename T>
class BigIntCaster : public IntCaster {
 public:
  static constexpr size_t N = T::N;

  BigIntCaster() = default;

  bool CastBigInt(PyObject* obj) {
    if (PyLong_Check(obj)) {
      int sign = _PyLong_Sign(obj);

      size_t bits = _PyLong_NumBits(obj);
      if (bits == static_cast<size_t>(-1)) {
        // Overflow error already set by _PyLong_NumBits.
        return false;
      }

      constexpr size_t kBitLen = N * 64;
      if (bits > kBitLen) {
        static_cast<T*>(this)->SetOverflowError();
        return false;
      }

      constexpr size_t kByteLen = N * 8;
      std::array<uint8_t, N * 8> bytes;
      int ret = _PyLong_AsByteArray(reinterpret_cast<PyLongObject*>(obj),
                                    bytes.data(), kByteLen,
                                    /*little_endian=*/true,
                                    /*is_signed=*/sign == -1);
      if (ret == -1) {
        // Error already set by _PyLong_AsByteArray.
        return false;
      }

      auto value =
          BigInt<N>::FromBytesLE(absl::Span<uint8_t>(bytes.data(), kByteLen));
      if (sign == -1) {
        value = -value;
      }

      if (static_cast<T*>(this)->IsOverflow(value)) {
        DCHECK(PyErr_Occurred());
        return false;
      }

      return static_cast<T*>(this)->Cast(value, sign);
    }
    return static_cast<IntCaster*>(this)->CastNumpyInt(obj);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_INT_CASTER_H_
