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

#ifndef ZK_DTYPES__SRC_FIELD_MODARITH_H_
#define ZK_DTYPES__SRC_FIELD_MODARITH_H_

// Native fixed-width modular arithmetic for the parametric numpy DTypes, with
// the modulus supplied at runtime (the loops in field_dtype.cc / ec_point_dtype
// .cc can't bake it at compile time). It reuses zk_dtypes' own Montgomery and
// modular-add/sub kernels (field/mont_multiplication.h,
// field/modular_operations .h) — the same code the legacy compile-time field
// configs use — so a native result is byte-identical to the legacy dtype by
// construction. The only piece not provided by those headers is n' = -p^-1 mod
// 2^w, computed here by Newton-Hensel lifting.
//
// Storage matches the legacy types: a base-field element is `width_bytes`
// little-endian, Montgomery-encoded (c*R mod p, R = 2^(width_bytes*8)) for the
// Montgomery form. On a little-endian host the byte buffer is the limb buffer,
// so load/store are plain memcpy.

#include <cstdint>
#include <cstring>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/field/binary_field_multiplication.h"
#include "zk_dtypes/include/field/modular_operations.h"
#include "zk_dtypes/include/field/mont_multiplication.h"

namespace zk_dtypes {
namespace modarith {

// p^-1 mod 2^64 from the low limb of p (p odd). Newton-Hensel doubles the
// number of correct low bits each step: 3 -> 6 -> 12 -> 24 -> 48 -> 96 (>= 64).
// The two Montgomery kernels disagree on the sign of n': the multi-limb
// FastMontMul/MontReduce add k*p and need n' = -p^-1 mod 2^64, while the
// single-word MontMul<T> subtracts and needs n' = +p^-1 mod 2^w.
inline uint64_t ComputeInverse(uint64_t p_low) {
  uint64_t inv = p_low;  // correct mod 2^3 for odd p_low
  for (int i = 0; i < 6; ++i) inv *= 2 - p_low * inv;
  return inv;
}

// A prime base field of `width_bytes` (4/8/16/32) with a runtime modulus.
// `native` is false when the (width, storage-form) combination is not handled
// here and the caller must fall back to the generic Python-int path.
struct PrimeField {
  int width_bytes = 0;
  bool is_mont = false;
  bool native = false;      // prime add/sub/mul fully supported natively
  bool ext_native = false;  // usable as an extension-field coefficient field
  bool spare = false;       // modulus has a spare high bit in its storage width
  bool no_carry = false;    // gnark no-carry Montgomery optimization applies
  uint64_t nprime_neg = 0;  // -p^-1 mod 2^64 (multi-limb MontMul<N>)
  uint64_t inv = 0;         // +p^-1 mod 2^64 (single-word MontMul<uint32/64>)
  uint32_t p32 = 0;
  uint64_t p64 = 0;
  BigInt<2> p128{};
  BigInt<4> p256{};

  static PrimeField Make(const unsigned char* mod_le, int width_bytes,
                         bool is_mont) {
    PrimeField f;
    f.width_bytes = width_bytes;
    f.is_mont = is_mont;
    uint64_t low = 0;
    std::memcpy(&low, mod_le, width_bytes < 8 ? width_bytes : 8);
    f.inv = ComputeInverse(low);
    f.nprime_neg = uint64_t{0} - f.inv;
    switch (width_bytes) {
      case 4:
        f.p32 = static_cast<uint32_t>(low);
        f.spare = (f.p32 >> 31) == 0;
        // The single-word uint32 Montgomery kernel needs the spare bit to avoid
        // overflow; without it, defer to the generic path.
        f.native = f.spare;
        f.ext_native = f.spare;
        break;
      case 8:
        f.p64 = low;
        f.spare = (f.p64 >> 63) == 0;
        f.no_carry = CanUseNoCarryMulOptimization(BigInt<1>(f.p64));
        f.native = true;
        f.ext_native = true;
        break;
      case 16:
        std::memcpy(&f.p128[0], mod_le, 16);
        f.spare = (f.p128[1] >> 63) == 0;
        f.no_carry = CanUseNoCarryMulOptimization(f.p128);
        f.native = true;
        f.ext_native = false;  // extension coeff mul over 128-bit base: defer
        break;
      case 32:
        std::memcpy(&f.p256[0], mod_le, 32);
        f.spare = (f.p256[3] >> 63) == 0;
        f.no_carry = CanUseNoCarryMulOptimization(f.p256);
        f.native = true;
        f.ext_native = false;
        break;
      default:
        f.native = false;
    }
    // Canonical (non-Montgomery) multiply is only implemented natively for the
    // single-word widths (a 128/256-bit reduce-by-division is deferred).
    if (!is_mont && width_bytes > 8) {
      f.native = false;
      f.ext_native = false;
    }
    return f;
  }

  void Add(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    switch (width_bytes) {
      case 4: {
        uint32_t x, y, r;
        std::memcpy(&x, a, 4);
        std::memcpy(&y, b, 4);
        ModAdd<uint32_t>(x, y, r, p32, spare);
        std::memcpy(o, &r, 4);
        break;
      }
      case 8: {
        uint64_t x, y, r;
        std::memcpy(&x, a, 8);
        std::memcpy(&y, b, 8);
        ModAdd<uint64_t>(x, y, r, p64, spare);
        std::memcpy(o, &r, 8);
        break;
      }
      case 16: {
        BigInt<2> x, y, r;
        std::memcpy(&x[0], a, 16);
        std::memcpy(&y[0], b, 16);
        ModAdd<BigInt<2>>(x, y, r, p128, spare);
        std::memcpy(o, &r[0], 16);
        break;
      }
      case 32: {
        BigInt<4> x, y, r;
        std::memcpy(&x[0], a, 32);
        std::memcpy(&y[0], b, 32);
        ModAdd<BigInt<4>>(x, y, r, p256, spare);
        std::memcpy(o, &r[0], 32);
        break;
      }
    }
  }

  void Sub(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    switch (width_bytes) {
      case 4: {
        uint32_t x, y, r;
        std::memcpy(&x, a, 4);
        std::memcpy(&y, b, 4);
        ModSub<uint32_t>(x, y, r, p32, spare);
        std::memcpy(o, &r, 4);
        break;
      }
      case 8: {
        uint64_t x, y, r;
        std::memcpy(&x, a, 8);
        std::memcpy(&y, b, 8);
        ModSub<uint64_t>(x, y, r, p64, spare);
        std::memcpy(o, &r, 8);
        break;
      }
      case 16: {
        BigInt<2> x, y, r;
        std::memcpy(&x[0], a, 16);
        std::memcpy(&y[0], b, 16);
        ModSub<BigInt<2>>(x, y, r, p128, spare);
        std::memcpy(o, &r[0], 16);
        break;
      }
      case 32: {
        BigInt<4> x, y, r;
        std::memcpy(&x[0], a, 32);
        std::memcpy(&y[0], b, 32);
        ModSub<BigInt<4>>(x, y, r, p256, spare);
        std::memcpy(o, &r[0], 32);
        break;
      }
    }
  }

  // Montgomery multiply on stored (Montgomery) representatives:
  // mont(x)*mont(y)*R^-1 = mont(x*y).
  void MontMulBytes(const unsigned char* a, const unsigned char* b,
                    unsigned char* o) const {
    switch (width_bytes) {
      case 4: {
        uint32_t x, y, r;
        std::memcpy(&x, a, 4);
        std::memcpy(&y, b, 4);
        MontMul<uint32_t>(x, y, r, p32, static_cast<uint32_t>(inv));
        std::memcpy(o, &r, 4);
        break;
      }
      case 8: {
        uint64_t x, y, r;
        std::memcpy(&x, a, 8);
        std::memcpy(&y, b, 8);
        MontMul<uint64_t>(x, y, r, p64, inv);
        std::memcpy(o, &r, 8);
        break;
      }
      case 16: {
        BigInt<2> x, y, r;
        std::memcpy(&x[0], a, 16);
        std::memcpy(&y[0], b, 16);
        MontMul<2>(x, y, r, p128, nprime_neg, spare, no_carry);
        std::memcpy(o, &r[0], 16);
        break;
      }
      case 32: {
        BigInt<4> x, y, r;
        std::memcpy(&x[0], a, 32);
        std::memcpy(&y[0], b, 32);
        MontMul<4>(x, y, r, p256, nprime_neg, spare, no_carry);
        std::memcpy(o, &r[0], 32);
        break;
      }
    }
  }

  // Canonical multiply r = a*b mod p on plain residues (width 4/8 only).
  void CanonMulBytes(const unsigned char* a, const unsigned char* b,
                     unsigned char* o) const {
    if (width_bytes == 4) {
      uint32_t x, y;
      std::memcpy(&x, a, 4);
      std::memcpy(&y, b, 4);
      uint32_t r = static_cast<uint32_t>((static_cast<uint64_t>(x) * y) % p32);
      std::memcpy(o, &r, 4);
    } else {  // width_bytes == 8
      uint64_t x, y;
      std::memcpy(&x, a, 8);
      std::memcpy(&y, b, 8);
      uint64_t r =
          static_cast<uint64_t>((static_cast<unsigned __int128>(x) * y) % p64);
      std::memcpy(o, &r, 8);
    }
  }

  // Storage-form-appropriate multiply for a degree-1 prime field.
  void Mul(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    if (is_mont) {
      MontMulBytes(a, b, o);
    } else {
      CanonMulBytes(a, b, o);
    }
  }
};

// Binary tower GF(2^(2^level)) multiply, levels 0..7 (1..128 bits). Returns
// false for higher levels (caller falls back). Inputs/outputs are width_bytes
// little-endian; the legacy binary_field_t* dtypes use the same BinaryMul.
inline bool BinaryTowerMul(int level, int width_bytes, const unsigned char* a,
                           const unsigned char* b, unsigned char* o) {
  switch (level) {
    case 0: {
      uint8_t x = 0, y = 0;
      std::memcpy(&x, a, 1);
      std::memcpy(&y, b, 1);
      uint8_t r = BinaryMul<0, uint8_t>(x, y);
      std::memcpy(o, &r, 1);
      return true;
    }
    case 1: {
      uint8_t x = 0, y = 0;
      std::memcpy(&x, a, 1);
      std::memcpy(&y, b, 1);
      uint8_t r = BinaryMul<1, uint8_t>(x, y);
      std::memcpy(o, &r, 1);
      return true;
    }
    case 2: {
      uint8_t x = 0, y = 0;
      std::memcpy(&x, a, 1);
      std::memcpy(&y, b, 1);
      uint8_t r = BinaryMul<2, uint8_t>(x, y);
      std::memcpy(o, &r, 1);
      return true;
    }
    case 3: {
      uint8_t x = 0, y = 0;
      std::memcpy(&x, a, 1);
      std::memcpy(&y, b, 1);
      uint8_t r = BinaryMul<3, uint8_t>(x, y);
      std::memcpy(o, &r, 1);
      return true;
    }
    case 4: {
      uint16_t x = 0, y = 0;
      std::memcpy(&x, a, 2);
      std::memcpy(&y, b, 2);
      uint16_t r = BinaryMul<4, uint16_t>(x, y);
      std::memcpy(o, &r, 2);
      return true;
    }
    case 5: {
      uint32_t x = 0, y = 0;
      std::memcpy(&x, a, 4);
      std::memcpy(&y, b, 4);
      uint32_t r = BinaryMul<5, uint32_t>(x, y);
      std::memcpy(o, &r, 4);
      return true;
    }
    case 6: {
      uint64_t x = 0, y = 0;
      std::memcpy(&x, a, 8);
      std::memcpy(&y, b, 8);
      uint64_t r = BinaryMul<6, uint64_t>(x, y);
      std::memcpy(o, &r, 8);
      return true;
    }
    case 7: {
      // Level 7 (128-bit) stores as BigInt<2>, matching the legacy
      // binary_field_t7 ValueType (TowerTraits<7>).
      BigInt<2> x, y;
      std::memcpy(&x[0], a, 16);
      std::memcpy(&y[0], b, 16);
      BigInt<2> r = BinaryMul<7, BigInt<2>>(x, y);
      std::memcpy(o, &r[0], 16);
      return true;
    }
    default:
      (void)width_bytes;
      return false;
  }
}

}  // namespace modarith
}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_FIELD_MODARITH_H_
