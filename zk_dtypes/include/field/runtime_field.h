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

#ifndef ZK_DTYPES_INCLUDE_FIELD_RUNTIME_FIELD_H_
#define ZK_DTYPES_INCLUDE_FIELD_RUNTIME_FIELD_H_

// Native fixed-width modular arithmetic with the modulus supplied at runtime,
// for callers that cannot bake it in at compile time: the parametric numpy
// DTypes (the loops in field_dtype.cc / ec_point_dtype.cc), and compilers
// materializing constants for a field chosen by a user rather than curated
// here.
//
// It reuses zk_dtypes' own Montgomery and modular-add/sub kernels
// (field/mont_multiplication.h, field/modular_operations.h) — the same code the
// compile-time field configs use — so a runtime result is byte-identical to the
// corresponding curated type by construction, not by testing. That equivalence
// is the point: a consumer can serve an arbitrary modulus without its curated
// results changing. The only piece not provided by those headers is
// n' = -p^-1 mod 2^w, computed here by Newton-Hensel lifting.
//
// Storage matches the legacy types: a base-field element is `width_bytes`
// little-endian, Montgomery-encoded (c*R mod p, R = 2^(width_bytes*8)) for the
// Montgomery form. On a little-endian host the byte buffer is the limb buffer,
// so load/store are plain memcpy.

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/bits.h"
#include "zk_dtypes/include/field/binary_field_multiplication.h"
#include "zk_dtypes/include/field/modular_operations.h"
#include "zk_dtypes/include/field/mont_multiplication.h"

namespace zk_dtypes {
namespace runtime_field {

// p^-1 mod 2^64 from the low limb of p (p odd). Newton-Hensel doubles the
// number of correct low bits each step: 3 -> 6 -> 12 -> 24 -> 48 -> 96 (>= 64).
// The two Montgomery kernels disagree on the sign of n': the multi-limb
// FastMontMul/MontReduce add k*p and need n' = -p^-1 mod 2^64, while the
// single-word MontMul<T> subtracts and needs n' = +p^-1 mod 2^w.
inline uint64_t ComputeInverse(uint64_t p_low) {
  uint64_t inv = p_low;  // correct mod 2^3 for odd p_low
  for (int i = 0; i < 5; ++i) inv *= 2 - p_low * inv;  // 3->6->12->24->48->96
  return inv;
}

// A prime base field of `width_bytes` (4/8/16/32) with a runtime modulus.
//
// `native` is false when this (width, storage-form) pair has no kernel here.
// Add/Sub/Mul must not be called in that case — they have no default branch and
// would leave the output untouched — so the caller has to supply its own
// arbitrary-precision path.
struct RuntimeField {
  // Widest modulus a field may carry, and hence the size of every buffer these
  // operations read or write. Public because a caller has no other way to size
  // one, and every consumer otherwise hardcodes the same 32.
  static constexpr int kMaxWidthBytes = 32;

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
  // Modulus, little-endian, exactly width_bytes. Kept so the exponent
  // (p-1)/2^k and the adicity can be derived without the caller re-supplying
  // it.
  unsigned char p_le[kMaxWidthBytes] = {};

  static RuntimeField Make(const unsigned char* mod_le, int width_bytes,
                           bool is_mont) {
    RuntimeField f;
    // Checked before anything reads mod_le: the widths below are the only ones
    // with kernels, and a caller passing a user-chosen modulus can hand over
    // any width at all.
    if (width_bytes != 4 && width_bytes != 8 && width_bytes != 16 &&
        width_bytes != 32) {
      return f;  // width_bytes 0, native false — unusable, by construction
    }
    f.width_bytes = width_bytes;
    f.is_mont = is_mont;
    std::memcpy(f.p_le, mod_le, width_bytes);
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
        // Same constraint as width 4: the single-word Montgomery kernel needs
        // the spare bit. Canonical multiply reduces through __int128 division
        // and does not.
#if defined(__SIZEOF_INT128__)
        f.native = !is_mont || f.spare;
#else
        f.native = is_mont && f.spare;
#endif
        f.ext_native = f.spare;
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
    }
    // Canonical (non-Montgomery) multiply is only implemented natively for the
    // single-word widths (a 128/256-bit reduce-by-division is deferred).
    if (!is_mont && width_bytes > 8) {
      f.native = false;
      f.ext_native = false;
    }
    return f;
  }

  // ModAdd and ModSub share one width switch; kSub picks the operation (the
  // dead branch folds away at compile time).
  template <bool kSub>
  void AddSub(const unsigned char* a, const unsigned char* b,
              unsigned char* o) const {
    switch (width_bytes) {
      case 4: {
        uint32_t x, y, r;
        std::memcpy(&x, a, 4);
        std::memcpy(&y, b, 4);
        if (kSub) {
          ModSub<uint32_t>(x, y, r, p32, spare);
        } else {
          ModAdd<uint32_t>(x, y, r, p32, spare);
        }
        std::memcpy(o, &r, 4);
        break;
      }
      case 8: {
        uint64_t x, y, r;
        std::memcpy(&x, a, 8);
        std::memcpy(&y, b, 8);
        if (kSub) {
          ModSub<uint64_t>(x, y, r, p64, spare);
        } else {
          ModAdd<uint64_t>(x, y, r, p64, spare);
        }
        std::memcpy(o, &r, 8);
        break;
      }
      case 16: {
        BigInt<2> x, y, r;
        std::memcpy(&x[0], a, 16);
        std::memcpy(&y[0], b, 16);
        if (kSub) {
          ModSub<BigInt<2>>(x, y, r, p128, spare);
        } else {
          ModAdd<BigInt<2>>(x, y, r, p128, spare);
        }
        std::memcpy(o, &r[0], 16);
        break;
      }
      case 32: {
        BigInt<4> x, y, r;
        std::memcpy(&x[0], a, 32);
        std::memcpy(&y[0], b, 32);
        if (kSub) {
          ModSub<BigInt<4>>(x, y, r, p256, spare);
        } else {
          ModAdd<BigInt<4>>(x, y, r, p256, spare);
        }
        std::memcpy(o, &r[0], 32);
        break;
      }
    }
  }

  void Add(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    AddSub<false>(a, b, o);
  }
  void Sub(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    AddSub<true>(a, b, o);
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
#if defined(__SIZEOF_INT128__)
      uint64_t x, y;
      std::memcpy(&x, a, 8);
      std::memcpy(&y, b, 8);
      uint64_t r =
          static_cast<uint64_t>((static_cast<unsigned __int128>(x) * y) % p64);
      std::memcpy(o, &r, 8);
#else
      // Unreachable: Make() only marks width-8 canonical native when __int128
      // exists.
      std::abort();
#endif
    }
  }

  // Storage-form-appropriate multiply for a degree-1 prime field.
  void Mul(const unsigned char* a, const unsigned char* b,
           unsigned char* o) const {
    if (is_mont) {
      MontMulBytes(a, b, o);
    } else {
      // Make() only marks canonical storage native up to single-word widths;
      // CanonMulBytes reads exactly 4 or 8 bytes.
      assert(width_bytes <= 8);
      CanonMulBytes(a, b, o);
    }
  }

  // --- values a caller needs to compute with, not just combine -------------
  //
  // The four above take stored representatives and hand back stored
  // representatives, which is all a numpy ufunc loop needs. Materializing a
  // constant needs more: an identity to start from, a way in and out of the
  // storage encoding, and exponentiation.

  // Multiplicative identity in storage form: canonical 1, or R mod p when
  // Montgomery-encoded.
  void One(unsigned char* o) const { SetSmall(o, 1); }

  // Canonical residue -> stored. Identity for canonical storage; c*R mod p for
  // Montgomery, which is c doubled 8*width_bytes times — R is 2^(8*width_bytes)
  // and doubling is the ModAdd the stored values already use, so no separate
  // reduction path can disagree with it.
  void Encode(const unsigned char* c, unsigned char* o) const {
    std::memcpy(o, c, width_bytes);
    if (!is_mont) return;
    for (int i = 0; i < width_bytes * 8; ++i) Add(o, o, o);
  }

  // Stored -> canonical residue. MontMul(x, 1) = x*1*R^-1 undoes the encoding
  // without needing R^-1 as a separate value.
  void Decode(const unsigned char* x, unsigned char* o) const {
    if (!is_mont) {
      std::memcpy(o, x, width_bytes);
      return;
    }
    unsigned char one_le[kMaxWidthBytes];
    std::memset(one_le, 0, width_bytes);
    one_le[0] = 1;
    MontMulBytes(x, one_le, o);
  }

  // o = base^exp on stored representatives, exp little-endian. Square and
  // multiply, most-significant bit first.
  void Pow(const unsigned char* base, const unsigned char* exp_le,
           int exp_bytes, unsigned char* o) const {
    unsigned char acc[kMaxWidthBytes];
    unsigned char b[kMaxWidthBytes];
    One(acc);
    // Copied because `base` may alias `o` — RootOfUnity relies on that.
    std::memcpy(b, base, width_bytes);
    // Whole leading zero bytes are skipped; the up-to-seven leading zero bits
    // inside the top byte only square the identity, so they cost nothing but a
    // multiply and buy a loop without a "have we started" flag in it.
    int top = exp_bytes - 1;
    while (top >= 0 && exp_le[top] == 0) --top;
    for (int byte = top; byte >= 0; --byte) {
      for (int bit = 7; bit >= 0; --bit) {
        Mul(acc, acc, acc);
        if ((exp_le[byte] >> bit) & 1) Mul(acc, b, acc);
      }
    }
    std::memcpy(o, acc, width_bytes);
  }

  void Pow(const unsigned char* base, uint64_t exp, unsigned char* o) const {
    unsigned char e[8];
    for (int i = 0; i < 8; ++i)
      e[i] = static_cast<unsigned char>(exp >> (8 * i));
    Pow(base, e, 8, o);
  }

  // Number of factors of two in p-1, i.e. the largest k with a 2^k-th root of
  // unity. p is odd, so p-1 only clears the low bit of the low limb.
  int TwoAdicity() const {
    int adicity = 0;
    for (int byte = 0; byte < width_bytes; ++byte) {
      unsigned char v = p_le[byte];
      if (byte == 0) v = static_cast<unsigned char>(v - 1);  // p odd
      if (v != 0) {
        while ((v & 1) == 0) {
          ++adicity;
          v = static_cast<unsigned char>(v >> 1);
        }
        return adicity;
      }
      adicity += 8;
    }
    return adicity;
  }

  // A primitive n-th root of unity for n a power of two dividing
  // 2^TwoAdicity(), written to `o` in storage form. False when no such root
  // exists.
  //
  // `generator` pins the subgroup generator g and yields g^((p-1)/n) — the form
  // a caller uses to reproduce a specific root. 0 selects the smallest
  // quadratic non-residue instead, which is deterministic but is NOT in general
  // the root a curated config stores: those hold a chosen constant, and any of
  // the φ(n) primitive n-th roots is as valid. A caller that needs a particular
  // one must pin the generator.
  bool RootOfUnity(uint64_t n, uint64_t generator, unsigned char* o) const {
    if (!IsPowerOf2(n)) return false;
    const int log_n = static_cast<int>(Log2Ceiling(n));
    if (log_n > TwoAdicity()) return false;

    unsigned char g[kMaxWidthBytes];
    if (generator == 0) {
      if (!FindNonResidue(g)) return false;
    } else {
      SetSmall(g, generator);
    }
    unsigned char exp[kMaxWidthBytes];
    PMinusOneShiftedRight(exp, log_n);
    Pow(g, exp, width_bytes, o);
    // A pinned generator is the caller's assertion, not a fact: g = 1 raises to
    // 1 for every n, and any g that is a square gives a root of smaller order.
    // Both would return a plausible value that silently corrupts every
    // butterfly downstream, so the order is checked rather than assumed. For a
    // found non-residue the check is implied, and going through it anyway is
    // what makes "both paths carry the same guarantee" a fact about the code
    // rather than an argument about it.
    return HasOrder(o, n);
  }

  // True when x has order exactly n. For n a power of two, x^(n/2) decides it:
  // every proper divisor of n divides n/2, so an order below n shows up there,
  // and squaring that value is the rest of the test.
  bool HasOrder(const unsigned char* x, uint64_t n) const {
    unsigned char one_s[kMaxWidthBytes], acc[kMaxWidthBytes];
    One(one_s);
    if (n == 1) {
      return std::memcmp(x, one_s, width_bytes) == 0;
    }
    Pow(x, n / 2, acc);
    if (std::memcmp(acc, one_s, width_bytes) == 0) return false;
    Mul(acc, acc, acc);  // x^n
    return std::memcmp(acc, one_s, width_bytes) == 0;
  }

 private:
  // (p-1) >> bits, little-endian. p is odd, so p-1 only clears the low bit.
  void PMinusOneShiftedRight(unsigned char* o, int bits) const {
    std::memcpy(o, p_le, width_bytes);
    o[0] = static_cast<unsigned char>(o[0] - 1);
    ShiftRight(o, bits);
  }

  // The smallest quadratic non-residue, in storage form. g is one iff
  // g^((p-1)/2) == -1, and that is exactly what makes g^((p-1)/2^k) have order
  // 2^k. Small candidates suffice — non-residues have density 1/2 — and the
  // bound only stops a search that cannot terminate because the modulus is not
  // prime.
  bool FindNonResidue(unsigned char* g) const {
    unsigned char half[kMaxWidthBytes], probe[kMaxWidthBytes];
    unsigned char minus_one[kMaxWidthBytes], one_s[kMaxWidthBytes];
    PMinusOneShiftedRight(half, 1);
    One(one_s);
    std::memset(minus_one, 0, width_bytes);
    Sub(minus_one, one_s, minus_one);  // p-1 in storage form
    for (uint64_t cand = 2; cand < 4096; ++cand) {
      SetSmall(g, cand);
      Pow(g, half, width_bytes, probe);
      if (std::memcmp(probe, minus_one, width_bytes) == 0) return true;
    }
    return false;
  }

  // o = the storage-form element for the small canonical value v.
  void SetSmall(unsigned char* o, uint64_t v) const {
    // Reduced first: the kernels require inputs below the modulus, and a caller
    // is free to hand over a generator that is not (p itself, say). Above 8
    // bytes the modulus exceeds every uint64, so v is already reduced.
    if (width_bytes == 4) {
      v %= p32;
    } else if (width_bytes == 8) {
      v %= p64;
    }
    unsigned char c[kMaxWidthBytes];
    std::memset(c, 0, width_bytes);
    for (int i = 0; i < 8 && i < width_bytes; ++i) {
      c[i] = static_cast<unsigned char>(v >> (8 * i));
    }
    Encode(c, o);
  }

  // In-place logical right shift of a little-endian buffer.
  void ShiftRight(unsigned char* v, int bits) const {
    while (bits >= 8) {
      std::memmove(v, v + 1, width_bytes - 1);
      v[width_bytes - 1] = 0;
      bits -= 8;
    }
    if (bits == 0) return;
    for (int i = 0; i < width_bytes; ++i) {
      unsigned char hi =
          (i + 1 < width_bytes)
              ? static_cast<unsigned char>(v[i + 1] << (8 - bits))
              : 0;
      v[i] = static_cast<unsigned char>((v[i] >> bits) | hi);
    }
  }
};

// Binary tower GF(2^(2^level)) multiply, levels 0..7 (1..128 bits). Returns
// false for higher levels (caller falls back). The byte width is implied by the
// level; inputs/outputs are little-endian; the legacy binary_field_t* dtypes
// use the same BinaryMul.
inline bool BinaryTowerMul(int level, const unsigned char* a,
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
      return false;
  }
}

}  // namespace runtime_field
}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_RUNTIME_FIELD_H_
