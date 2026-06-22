# Copyright 2025 The zk_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for zk_dtypes.prime_field runtime field resolution."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import zk_dtypes
from zk_dtypes._field_factory import _is_probable_prime
from zk_dtypes._field_factory import _storage_width

# Modulus / storage / expected curated scalar type.
_CURATED_CASES = (
    (2013265921, "mont", zk_dtypes.babybear),
    (2130706433, "mont", zk_dtypes.koalabear),
    (18446744069414584321, "mont", zk_dtypes.goldilocks),
    (2147483647, "std", zk_dtypes.mersenne31),
)


class PrimeFieldFactoryTest(parameterized.TestCase):

  @parameterized.parameters(*_CURATED_CASES)
  def test_curated_resolves_to_existing_type(self, modulus, storage, expected):
    # A curated prime keeps its legacy dtype so the non-parametric stack still
    # recognizes it.
    self.assertEqual(
        zk_dtypes.prime_field(modulus, storage), np.dtype(expected)
    )

  def test_storage_aliases_agree(self):
    self.assertEqual(
        zk_dtypes.prime_field(2130706433, "mont"),
        zk_dtypes.prime_field(2130706433, "montgomery"),
    )
    self.assertEqual(
        zk_dtypes.prime_field(2147483647, "std"),
        zk_dtypes.prime_field(2147483647, "canonical"),
    )

  def test_bn254_sf_curated(self):
    info = zk_dtypes.pfinfo(zk_dtypes.bn254_sf)
    self.assertEqual(
        zk_dtypes.prime_field(info.modulus, "mont"),
        np.dtype(zk_dtypes.bn254_sf),
    )

  def test_composite_modulus_rejected(self):
    # 2^31 (even) and a Carmichael number — the latter passes Fermat but must
    # fail Miller-Rabin.
    with self.assertRaisesRegex(ValueError, "not prime"):
      zk_dtypes.prime_field(2**31, "mont")
    with self.assertRaisesRegex(ValueError, "not prime"):
      zk_dtypes.prime_field(561, "mont")  # 3 * 11 * 17, Carmichael

  def test_unknown_storage_rejected(self):
    with self.assertRaisesRegex(ValueError, "storage must be"):
      zk_dtypes.prime_field(2130706433, "redc")

  def test_non_int_modulus_rejected(self):
    with self.assertRaisesRegex(TypeError, "must be an int"):
      zk_dtypes.prime_field(2130706433.0, "mont")
    with self.assertRaisesRegex(TypeError, "must be an int"):
      zk_dtypes.prime_field(True, "mont")

  def test_novel_prime_canonical_round_trips(self):
    # A genuine prime that is not a curated family: 10**9 + 7.
    novel = 10**9 + 7
    self.assertTrue(_is_probable_prime(novel))
    dt = zk_dtypes.prime_field(novel, "canonical")
    self.assertEqual(dt.itemsize, 4)
    self.assertEqual(dt.kind, "V")
    vals = [1, 2, novel - 1, novel, novel + 5, 999999999]
    arr = np.array(vals, dtype=dt)
    self.assertEqual([int(x) for x in arr.tolist()], [v % novel for v in vals])
    # Negative inputs canonicalize to non-negative residues.
    self.assertEqual(
        [int(x) for x in np.array([-1, -5], dtype=dt).tolist()],
        [(-1) % novel, (-5) % novel],
    )

  def test_novel_prime_wide_canonical_round_trips(self):
    # A 127-bit Mersenne prime exercises the 128-bit (16-byte) storage path.
    novel = 2**127 - 1
    self.assertTrue(_is_probable_prime(novel))
    dt = zk_dtypes.prime_field(novel, "canonical")
    self.assertEqual(dt.itemsize, 16)
    vals = [2**130 + 7, 5, novel - 1]
    arr = np.array(vals, dtype=dt)
    self.assertEqual([int(x) for x in arr.tolist()], [v % novel for v in vals])

  def test_novel_prime_montgomery_round_trips(self):
    novel = 10**9 + 7
    dt = zk_dtypes.prime_field(novel, "mont")  # 'mont' is the default
    vals = [0, 1, 2, novel - 1, 123456789]
    arr = np.array(vals, dtype=dt)
    # getitem decodes Montgomery back to the canonical value.
    self.assertEqual([int(x) for x in arr.tolist()], [v % novel for v in vals])

  def test_montgomery_raw_bytes_match_device_encoding(self):
    # Stored bytes are `a*R mod p` with R = 2^width, matching prime_ir's device
    # encoding (so host-built buffers are byte-compatible with the device).
    novel = 10**9 + 7
    r = (1 << 32) % novel
    dt = zk_dtypes.prime_field(novel, "mont")
    raw = int(np.array([7], dtype=dt).view(np.uint32)[0])
    self.assertEqual(raw, (7 * r) % novel)
    # Canonical storage keeps the residue verbatim.
    dc = zk_dtypes.prime_field(novel, "canonical")
    self.assertEqual(int(np.array([7], dtype=dc).view(np.uint32)[0]), 7)

  def test_oversize_modulus_rejected(self):
    # A prime wider than the widest (256-bit) storage class. 2^521 - 1 is a
    # Mersenne prime.
    with self.assertRaisesRegex(ValueError, "widest field storage"):
      zk_dtypes.prime_field(2**521 - 1, "mont")

  @parameterized.parameters(
      (31, 32), (32, 32), (33, 64), (64, 64), (65, 128), (254, 256), (256, 256)
  )
  def test_storage_width_roundup(self, bits, expected):
    self.assertEqual(_storage_width(bits), expected)

  @parameterized.parameters("canonical", "mont")
  def test_novel_prime_host_arithmetic(self, storage):
    p = 10**9 + 7
    dt = zk_dtypes.prime_field(p, storage)
    av = [5, 999999999, p - 1, 123456, 0]
    bv = [7, 999999999, 2, 654321, p - 1]
    a = np.array(av, dtype=dt)
    b = np.array(bv, dtype=dt)
    self.assertEqual(
        [int(x) for x in (a + b).tolist()],
        [(x + y) % p for x, y in zip(av, bv)],
    )
    self.assertEqual(
        [int(x) for x in (a - b).tolist()],
        [(x - y) % p for x, y in zip(av, bv)],
    )
    self.assertEqual(
        [int(x) for x in (a * b).tolist()],
        [(x * y) % p for x, y in zip(av, bv)],
    )

  def test_arithmetic_across_distinct_fields_errors(self):
    a = np.array([1], dtype=zk_dtypes.prime_field(10**9 + 7))
    b = np.array([1], dtype=zk_dtypes.prime_field(2147483647))
    with self.assertRaises(TypeError):
      np.add(a, b)

  def _encode_ef(self, base_mod, base_width_bytes, is_mont, coeffs):
    r = (1 << (base_width_bytes * 8)) % base_mod
    out = b""
    for c in coeffs:
      v = c % base_mod
      if is_mont:
        v = v * r % base_mod
      out += v.to_bytes(base_width_bytes, "little")
    return np.frombuffer(out, np.uint8)

  # Legacy family / parametric equivalent / base modulus / base width / mont.
  _EF_BYTE_MATCH = (
      ("babybearx4", 2013265921, 4, 11, "mont", 4, True),
      ("koalabearx4", 2130706433, 4, 3, "mont", 4, True),
      ("mersenne31x2", 2147483647, 2, 2147483646, "canonical", 4, False),
  )

  @parameterized.parameters(*_EF_BYTE_MATCH)
  def test_extension_field_byte_matches_legacy(
      self, legacy_name, base_mod, degree, nr, storage, bw, is_mont
  ):
    legacy = getattr(zk_dtypes, legacy_name)
    param = zk_dtypes.extension_field(base_mod, degree, nr, storage)
    self.assertEqual(np.dtype(param).itemsize, np.dtype(legacy).itemsize)
    ca = [(i * 7 + 5) % base_mod for i in range(degree)]
    cb = [(i * 11 + 3) % base_mod for i in range(degree)]
    pa, pb = np.zeros(1, dtype=param), np.zeros(1, dtype=param)
    la, lb = np.zeros(1, dtype=legacy), np.zeros(1, dtype=legacy)
    for arr in (pa, la):
      arr.view(np.uint8)[:] = self._encode_ef(base_mod, bw, is_mont, ca)
    for arr in (pb, lb):
      arr.view(np.uint8)[:] = self._encode_ef(base_mod, bw, is_mont, cb)
    for op in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y):
      np.testing.assert_array_equal(
          op(pa, pb).view(np.uint8), op(la, lb).view(np.uint8)
      )

  def test_novel_extension_field_arithmetic(self):
    # A novel cubic extension Fp[X]/(X^3 - 5) over a novel prime, checked against
    # a pure-Python binomial reference.
    p, deg, nr = 10**9 + 7, 3, 5
    dt = zk_dtypes.extension_field(p, deg, nr, "mont")
    self.assertEqual(np.dtype(dt).itemsize, 12)

    def ref_mul(a, b):
      prod = [0] * (2 * deg - 1)
      for i in range(deg):
        for j in range(deg):
          prod[i + j] = (prod[i + j] + a[i] * b[j]) % p
      for i in range(2 * deg - 2, deg - 1, -1):
        prod[i - deg] = (prod[i - deg] + nr * prod[i]) % p
      return [prod[i] % p for i in range(deg)]

    ca, cb = [3, 5, 7], [11, 13, 17]
    pa, pb = np.zeros(1, dtype=dt), np.zeros(1, dtype=dt)
    pa.view(np.uint8)[:] = self._encode_ef(p, 4, True, ca)
    pb.view(np.uint8)[:] = self._encode_ef(p, 4, True, cb)
    got = (pa * pb).view(np.uint32)
    want = self._encode_ef(p, 4, True, ref_mul(ca, cb)).view(np.uint32)
    np.testing.assert_array_equal(got, want)

  def test_curated_extension_resolves_to_legacy(self):
    self.assertEqual(
        zk_dtypes.extension_field(2013265921, 4, 11, "mont"),
        np.dtype(zk_dtypes.babybearx4),
    )

  def test_extension_field_degree_one_rejected(self):
    with self.assertRaisesRegex(ValueError, "degree must be >= 2"):
      zk_dtypes.extension_field(10**9 + 7, 1, 0, "mont")

  @parameterized.parameters(0, 1, 2, 3, 4, 5, 6, 7)
  def test_binary_field_byte_matches_legacy(self, level):
    legacy = getattr(zk_dtypes, f"binary_field_t{level}")
    param = np.dtype(zk_dtypes._zk_dtypes_ext.binary_field_descr(level))
    wb = np.dtype(legacy).itemsize
    self.assertEqual(np.dtype(param).itemsize, wb)
    m = 1 << level
    rng = np.random.default_rng(level)
    n = 64
    raw_a = rng.integers(0, 256, size=(n, wb), dtype=np.uint8)
    raw_b = rng.integers(0, 256, size=(n, wb), dtype=np.uint8)
    pa, pb = np.zeros(n, dtype=param), np.zeros(n, dtype=param)
    la, lb = np.zeros(n, dtype=legacy), np.zeros(n, dtype=legacy)
    for arr in (pa, la):
      arr.view(np.uint8).reshape(n, wb)[:] = raw_a
    for arr in (pb, lb):
      arr.view(np.uint8).reshape(n, wb)[:] = raw_b
    if m < 8:  # small levels: high byte bits are masked off
      for arr in (pa, la, pb, lb):
        arr.view(np.uint8)[:] &= (1 << m) - 1
    np.testing.assert_array_equal(
        (pa + pb).view(np.uint8), (la + lb).view(np.uint8)
    )
    np.testing.assert_array_equal(
        (pa * pb).view(np.uint8), (la * lb).view(np.uint8)
    )

  def test_binary_field_curated_resolves_to_legacy(self):
    for level in range(8):
      self.assertEqual(
          zk_dtypes.binary_field(level),
          np.dtype(getattr(zk_dtypes, f"binary_field_t{level}")),
      )

  def test_binary_field_negative_level_rejected(self):
    with self.assertRaisesRegex(ValueError, "level must be >= 0"):
      zk_dtypes.binary_field(-1)

  _BN254_FQ = 21888242871839275222246405745257275088696311157297823662689037894645226208583

  def test_ec_g1_jacobian_group_law_byte_matches_legacy(self):
    legacy = zk_dtypes.bn254_g1_jacobian
    p = self._BN254_FQ
    info = zk_dtypes.ecinfo(np.dtype(legacy))
    if getattr(info, "is_montgomery", True):
      r = (1 << 256) % p
      param = np.dtype(
          zk_dtypes._zk_dtypes_ext.ec_point_descr(
              p, 256, 3, 1, r, pow(r, -1, p)
          )
      )
    else:
      param = np.dtype(zk_dtypes._zk_dtypes_ext.ec_point_descr(p, 256, 3, 0))
    self.assertEqual(np.dtype(param).itemsize, np.dtype(legacy).itemsize)

    def to_param(legacy_arr):
      out = np.zeros(len(legacy_arr), dtype=param)
      out.view(np.uint8)[:] = legacy_arr.view(np.uint8)
      return out

    # n*G are valid curve points; identical input representatives let byte
    # equality test that the EFD formulas reproduce the exact legacy output.
    for a, b in [(1, 2), (3, 5), (7, 11), (2, 2), (10, 10), (123, 456), (0, 5)]:
      la, lb = np.array([a], dtype=legacy), np.array([b], dtype=legacy)
      pa, pb = to_param(la), to_param(lb)
      np.testing.assert_array_equal(
          (pa + pb).view(np.uint8), (la + lb).view(np.uint8)
      )
      np.testing.assert_array_equal(
          (pa - pb).view(np.uint8), (la - lb).view(np.uint8)
      )
      np.testing.assert_array_equal((-pa).view(np.uint8), (-la).view(np.uint8))

  def _ec_g1_jacobian_param(self):
    p = self._BN254_FQ
    info = zk_dtypes.ecinfo(np.dtype(zk_dtypes.bn254_g1_jacobian))
    if getattr(info, "is_montgomery", True):
      r = (1 << 256) % p
      return np.dtype(
          zk_dtypes._zk_dtypes_ext.ec_point_descr(
              p, 256, 3, 1, r, pow(r, -1, p)
          )
      )
    return np.dtype(zk_dtypes._zk_dtypes_ext.ec_point_descr(p, 256, 3, 0))

  def test_ec_group_equality_cross_representative(self):
    legacy = zk_dtypes.bn254_g1_jacobian
    param = self._ec_g1_jacobian_param()

    def pt(n):
      out = np.zeros(1, dtype=param)
      out.view(np.uint8)[:] = np.array([n], dtype=legacy).view(np.uint8)
      return out

    p1, p2, p3, p5 = pt(1), pt(2), pt(3), pt(5)
    self.assertTrue(bool((p3 == p3)[0]))
    self.assertFalse(bool((p3 == p5)[0]))
    self.assertTrue(bool((p3 != p5)[0]))
    # 5G and 2G+3G are the same group element with different Jacobian
    # representatives (byte-different); == must compare by group element.
    s = p2 + p3
    self.assertFalse(np.array_equal(p5.view(np.uint8), s.view(np.uint8)))
    self.assertTrue(bool((p5 == s)[0]))
    self.assertTrue(bool((p3 == (p1 + p2))[0]))

  def test_scalar_subscript_returns_element(self):
    # arr[i] (integer subscript to a scalar) must not segfault and returns the
    # element via getitem.
    p = 10**9 + 7
    a = np.array([6, 7, 8], dtype=zk_dtypes.prime_field(p, "canonical"))
    self.assertEqual(int(a[0]), 6)
    self.assertEqual(int(a[2]), 8)
    ef = zk_dtypes.extension_field(p, 3, 5, "canonical")
    e = np.zeros(1, dtype=ef)
    e.view(np.uint32).reshape(1, 3)[0] = [10, 20, 30]
    self.assertEqual(tuple(int(c) for c in e[0]), (10, 20, 30))
    bf = np.dtype(zk_dtypes._zk_dtypes_ext.binary_field_descr(5))
    b = np.array([0x12345], dtype=bf)
    self.assertEqual(int(b[0]), 0x12345)

  def test_miller_rabin_known_primes(self):
    for p in (
        2,
        3,
        5,
        2147483647,
        2013265921,
        2130706433,
        18446744069414584321,
    ):
      self.assertTrue(_is_probable_prime(p), p)
    for n in (0, 1, 4, 9, 561, 1105, 2147483646):
      self.assertFalse(_is_probable_prime(n), n)


if __name__ == "__main__":
  absltest.main()
