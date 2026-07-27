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

#ifndef ZK_DTYPES__SRC_EC_GROUP_LAW_H_
#define ZK_DTYPES__SRC_EC_GROUP_LAW_H_

// Short-Weierstrass (a = 0) Jacobian group law, written once over an abstract
// coordinate field and instantiated per execution tier. `C` is a
// value-semantic coordinate (copyable; copies may be reference bumps, as in
// the CPython-int tier) and `Ops` provides:
//
//   C    One() / Zero()
//   C    Add(a, b) / Sub(a, b) / Neg(a) / Mul(a, b) / MulInt(a, k)
//   bool IsZero(a)
//   bool Equal(a, b)
//
// An Ops whose operations can fail (the CPython tier) reports failure out of
// band (a poisoned flag checked by its caller) and must make every operation
// accept the values produced after a failure; the formulas themselves have no
// error path. Each formula reads all of its inputs before writing `out`, so
// `out` may alias an input.
//
// Formulas: EFD dbl-2009-l and add-2007-bl. Every tier goes through these
// exact bodies, so cross-tier byte-identity holds by construction — there is
// no second copy of a formula to keep in lockstep.

namespace zk_dtypes {
namespace ec_law {

// Jacobian doubling (EFD dbl-2009-l, a == 0). out may alias in.
template <typename C, typename Ops>
void EcDoubleT(const Ops& f, const C in[3], C out[3]) {
  const C& X = in[0];
  const C& Y = in[1];
  const C& Z = in[2];
  C xx = f.Mul(X, X);
  C yy = f.Mul(Y, Y);
  C yyyy = f.Mul(yy, yy);
  C dd = f.MulInt(f.Mul(X, yy), 4);
  C e = f.MulInt(xx, 3);
  C X2 = f.Sub(f.Mul(e, e), f.MulInt(dd, 2));
  C Y2 = f.Sub(f.Mul(e, f.Sub(dd, X2)), f.MulInt(yyyy, 8));
  C Z2 = f.MulInt(f.Mul(Y, Z), 2);
  out[0] = X2;
  out[1] = Y2;
  out[2] = Z2;
}

// Jacobian addition (EFD add-2007-bl, a == 0). out may alias an input.
template <typename C, typename Ops>
void EcAddT(const Ops& f, const C P[3], const C Q[3], C out[3]) {
  if (f.IsZero(P[2])) {
    out[0] = Q[0];
    out[1] = Q[1];
    out[2] = Q[2];
    return;
  }
  if (f.IsZero(Q[2])) {
    out[0] = P[0];
    out[1] = P[1];
    out[2] = P[2];
    return;
  }
  C z1z1 = f.Mul(P[2], P[2]);
  C z2z2 = f.Mul(Q[2], Q[2]);
  C u1 = f.Mul(P[0], z2z2);
  C u2 = f.Mul(Q[0], z1z1);
  C s1 = f.Mul(f.Mul(P[1], Q[2]), z2z2);
  C s2 = f.Mul(f.Mul(Q[1], P[2]), z1z1);
  if (f.Equal(u1, u2) && f.Equal(s1, s2)) {  // P == Q
    EcDoubleT<C, Ops>(f, P, out);
    return;
  }
  C h = f.Sub(u2, u1);
  C ii = f.Mul(f.MulInt(h, 2), f.MulInt(h, 2));
  C j = f.Neg(f.Mul(h, ii));
  C r = f.MulInt(f.Sub(s2, s1), 2);
  C v = f.Mul(u1, ii);
  C X3 = f.Sub(f.Add(f.Mul(r, r), j), f.MulInt(v, 2));  // r^2 + j - 2v
  C Y3 = f.Add(f.Mul(r, f.Sub(v, X3)),
               f.MulInt(f.Mul(s1, j), 2));          // r(v - X3) + 2 s1 j
  C Z3 = f.MulInt(f.Mul(f.Mul(P[2], Q[2]), h), 2);  // 2 Z1 Z2 h
  out[0] = X3;
  out[1] = Y3;
  out[2] = Z3;
}

// Cross-representative group equality of two Jacobian points; 1 / 0.
template <typename C, typename Ops>
int EcEqualT(const Ops& f, const C P[3], const C Q[3]) {
  bool pz = f.IsZero(P[2]);
  bool qz = f.IsZero(Q[2]);
  if (pz || qz) return (pz && qz) ? 1 : 0;
  C z1s = f.Mul(P[2], P[2]);
  C z2s = f.Mul(Q[2], Q[2]);
  bool xe = f.Equal(f.Mul(P[0], z2s), f.Mul(Q[0], z1s));
  bool ye =
      f.Equal(f.Mul(P[1], f.Mul(z2s, Q[2])), f.Mul(Q[1], f.Mul(z1s, P[2])));
  return (xe && ye) ? 1 : 0;
}

// ret = scalar * point, MSB-first double-and-add; scalar as little-endian
// bytes. ret must not alias point (ret is the running accumulator).
// NOT constant-time: branches on scalar bits and on nbits — fine for the
// public scalars numpy arrays carry, unsuitable for secret scalars.
template <typename C, typename Ops>
void EcScalarMulT(const Ops& f, const C point[3], const unsigned char* sbytes,
                  int nbits, C ret[3]) {
  ret[0] = f.One();  // Jacobian zero = (1, 1, 0)
  ret[1] = f.One();
  ret[2] = f.Zero();
  for (int i = nbits - 1; i >= 0; --i) {
    EcDoubleT<C, Ops>(f, ret, ret);
    if ((sbytes[i >> 3] >> (i & 7)) & 1) EcAddT<C, Ops>(f, ret, point, ret);
  }
}

}  // namespace ec_law
}  // namespace zk_dtypes

#endif  // ZK_DTYPES__SRC_EC_GROUP_LAW_H_
