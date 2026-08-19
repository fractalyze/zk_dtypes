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
from multi_thread_utils import multi_threaded
import pickle
import random

import numpy as np
import zk_dtypes
import warnings

from zk_dtypes._field_factory import _GOLDILOCKS_MODULUS
from zk_dtypes._field_factory import _is_probable_prime
from zk_dtypes._field_factory import _storage_width
from zk_dtypes._pfinfo import pfinfo

# Modulus / storage / expected curated scalar type.
_CURATED_CASES = (
    (2013265921, "mont", zk_dtypes.babybear_mont),
    (2130706433, "mont", zk_dtypes.koalabear_mont),
    (18446744069414584321, "mont", zk_dtypes.goldilocks_mont),
    (2147483647, "std", zk_dtypes.mersenne31),
)


@multi_threaded(num_workers=3)
class PrimeFieldFactoryTest(parameterized.TestCase):

  @parameterized.parameters(*_CURATED_CASES)
  def test_curated_resolves_to_existing_type(self, modulus, storage, expected):
    # A curated prime keeps its legacy dtype so the non-parametric stack still
    # recognizes it. Storage-efficiency advisories are orthogonal to which
    # dtype comes back, so they do not bear on this assertion.
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", DeprecationWarning)
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
        np.dtype(zk_dtypes.bn254_sf_mont),
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
        [(x + y) % p for x, y in zip(av, bv, strict=True)],
    )
    self.assertEqual(
        [int(x) for x in (a - b).tolist()],
        [(x - y) % p for x, y in zip(av, bv, strict=True)],
    )
    self.assertEqual(
        [int(x) for x in (a * b).tolist()],
        [(x * y) % p for x, y in zip(av, bv, strict=True)],
    )

  @parameterized.parameters("canonical", "mont")
  def test_no_spare_bit_64bit_prime_arithmetic(self, storage):
    # 2^64 - 59 has bit 63 set: no spare bit at its storage width, so the
    # single-word Montgomery kernel must not be dispatched (its overflow
    # avoidance needs the spare bit) and multiply falls back correctly.
    p = 2**64 - 59
    dt = zk_dtypes.prime_field(p, storage)
    av = [5, p - 1, 2**63, 123456789012345678, 0]
    bv = [7, p - 1, 2**63 + 1, 987654321098765432, p - 1]
    a = np.array(av, dtype=dt)
    b = np.array(bv, dtype=dt)
    self.assertEqual(
        [int(x) for x in (a * b).tolist()],
        [(x * y) % p for x, y in zip(av, bv, strict=True)],
    )
    self.assertEqual(
        [int(x) for x in (a + b).tolist()],
        [(x + y) % p for x, y in zip(av, bv, strict=True)],
    )

  def test_large_array_storage_cast_survives(self):
    # Casting releases the GIL past numpy's 500-element threshold unless the
    # cast method declares NPY_METH_REQUIRES_PYAPI; the re-encode path runs
    # CPython API, so an undeclared cast segfaults on arrays this size.
    p = 10**9 + 7
    n = 600
    vals = list(range(1, n + 1))
    a = np.array(vals, dtype=zk_dtypes.prime_field(p, "mont"))
    c = a.astype(zk_dtypes.prime_field(p, "canonical"))
    self.assertEqual([int(x) for x in c.tolist()], vals)

  def test_out_across_distinct_fields_errors(self):
    # An explicit out= of a different field must be rejected: the loop sizes
    # writes from the input descriptor, so a narrower out would be overrun.
    a = np.array([1], dtype=zk_dtypes.prime_field(10**9 + 7))
    b = np.array([2], dtype=zk_dtypes.prime_field(10**9 + 7))
    out = np.zeros(1, dtype=zk_dtypes.prime_field(2147483647))
    with self.assertRaises(TypeError):
      np.add(a, b, out=out)
    wide = np.array([1], dtype=zk_dtypes.prime_field(2**127 - 1))
    wout = np.zeros(1, dtype=zk_dtypes.prime_field(10**9 + 7))
    with self.assertRaises(TypeError):
      np.add(wide, wide, out=wout)

  def test_zero_non_residue_rejected(self):
    # X^k = 0 is a ring with zero divisors, not a field.
    with self.assertRaisesRegex(ValueError, "non_residue"):
      zk_dtypes.extension_field(2013265921, 2, 0)
    with self.assertRaisesRegex(ValueError, "non_residue"):
      zk_dtypes.extension_field(2013265921, 2, 2013265921)

  @parameterized.parameters("canonical", "mont")
  def test_negative_divide_power(self, storage):
    p = 10**9 + 7
    dt = zk_dtypes.prime_field(p, storage)
    vals = [1, 2, p - 1, 123456789, 7]
    a = np.array(vals, dtype=dt)
    self.assertEqual([int(x) for x in (-a).tolist()], [(-v) % p for v in vals])
    b = np.array([3, 7, 2, 999, p - 2], dtype=dt)
    got = (a / b).tolist()
    for g, (x, y) in zip(got, zip(vals, [3, 7, 2, 999, p - 2]), strict=True):
      self.assertEqual((int(g) * y) % p, x % p)
    with self.assertRaises(ZeroDivisionError):
      a / np.array([1, 0, 1, 1, 1], dtype=dt)
    self.assertEqual(
        [int(x) for x in (a**3).tolist()], [pow(v, 3, p) for v in vals]
    )
    # Negative exponent is the inverse chain; 0 ** -1 raises.
    self.assertEqual(
        [int(x) for x in (a**-1).tolist()], [pow(v, -1, p) for v in vals]
    )
    with self.assertRaises(ZeroDivisionError):
      np.array([0], dtype=dt) ** -1

  def test_extension_divide_roundtrips(self):
    # Novel base/non-residue so this is the parametric descriptor, not the
    # curated babybearx4 legacy dtype.
    dt = zk_dtypes.extension_field(10**9 + 7, 4, 5)
    rng = np.random.default_rng(0)
    a = np.zeros(4, dtype=dt)
    b = np.zeros(4, dtype=dt)
    for i in range(4):
      a[i] = tuple(int(x) for x in rng.integers(1, 10**9 + 7, size=4))
      b[i] = tuple(int(x) for x in rng.integers(1, 10**9 + 7, size=4))
    q = a / b
    np.testing.assert_array_equal((q * b).view(np.uint8), a.view(np.uint8))

  def test_binary_divide_roundtrips(self):
    # Level 8 has no legacy dtype, so binary_field(8) mints the parametric
    # descriptor whose TowerInv/TowerMul path this exercises (levels <= 7
    # resolve to the curated legacy dtype and its own kernels).
    dt = zk_dtypes.binary_field(8)
    a = np.array([1, 0xAB, 0x1234, (1 << 200) | 0xFFFF], dtype=dt)
    b = np.array([3, 0x11, 0xBEEF, 2], dtype=dt)
    q = a / b
    np.testing.assert_array_equal((q * b).view(np.uint8), a.view(np.uint8))
    with self.assertRaises(ZeroDivisionError):
      a / np.array([1, 0, 1, 1], dtype=dt)

  def test_descriptor_attributes(self):
    p = 10**9 + 7
    dt = zk_dtypes.prime_field(p, "canonical")
    self.assertEqual(dt.modulus, p)
    self.assertEqual(dt.degree, 1)
    self.assertIsNone(dt.non_residue)
    self.assertEqual(dt.base_width_bits, 32)
    self.assertFalse(dt.is_montgomery)
    self.assertIsNone(dt.tower_level)
    ext = zk_dtypes.extension_field(10**9 + 7, 4, 5)
    self.assertEqual((ext.degree, ext.non_residue), (4, 5))
    self.assertTrue(ext.is_montgomery)
    bf = zk_dtypes.binary_field(9)
    self.assertEqual(bf.tower_level, 9)
    self.assertIsNone(bf.modulus)
    ec = self._ec_g1_jacobian_param()
    self.assertEqual(ec.num_coords, 3)
    self.assertEqual(ec.coord_degree, 1)
    self.assertEqual(ec.base_width_bits, 256)

  def test_pickle_round_trips(self):
    for dt in (
        zk_dtypes.prime_field(10**9 + 7),
        zk_dtypes.prime_field(10**9 + 7, "canonical"),
        zk_dtypes.extension_field(10**9 + 7, 4, 5),
        zk_dtypes.binary_field(9),
        self._ec_g1_jacobian_param(),
    ):
      restored = pickle.loads(pickle.dumps(dt))
      self.assertEqual(restored, dt)
      # Arrays of the restored dtype interoperate with the original.
      a = np.zeros(2, dtype=dt)
      b = np.zeros(2, dtype=restored)
      self.assertEqual((a + b).dtype, dt)

  def test_pfinfo_on_parametric(self):
    p = 10**9 + 7
    info = zk_dtypes.pfinfo(zk_dtypes.prime_field(p, "canonical"))
    self.assertEqual(info.modulus, p)
    self.assertEqual(info.storage_bits, 32)
    self.assertEqual(info.modulus_bits, p.bit_length())
    self.assertFalse(info.is_montgomery)
    self.assertEqual(info.two_adicity, 1)

  @parameterized.parameters("canonical", "mont")
  def test_int_operand_and_casts(self, storage):
    p = 10**9 + 7
    dt = zk_dtypes.prime_field(p, storage)
    a = np.array([1, 2, p - 1], dtype=dt)
    # Python int, numpy scalar, and numpy array operands, either side.
    self.assertEqual([int(x) for x in (a + 1).tolist()], [2, 3, 0])
    self.assertEqual([int(x) for x in (1 + a).tolist()], [2, 3, 0])
    self.assertEqual([int(x) for x in (a * 2).tolist()], [2, 4, p - 2])
    self.assertEqual([int(x) for x in (a + np.int64(1)).tolist()], [2, 3, 0])
    self.assertEqual(
        [int(x) for x in (np.array([1, 1, 1], dtype=np.int32) + a).tolist()],
        [2, 3, 0],
    )
    self.assertEqual((a == 1).tolist(), [True, False, False])
    # Casts both directions, mirroring the legacy uint64 round-trip.
    small = np.array([0, 1, 2, 100, 1000], dtype=dt)
    self.assertEqual(
        [int(x) for x in small.astype(np.uint64).astype(dt).tolist()],
        [0, 1, 2, 100, 1000],
    )
    self.assertEqual(
        [int(x) for x in np.array([5, p + 3], dtype=np.int64).astype(dt)],
        [5, 3],
    )
    # Negative ints canonicalize like setitem does.
    self.assertEqual(
        [int(x) for x in np.array([-1], dtype=np.int64).astype(dt)],
        [p - 1],
    )
    # A field wider than the target integer overflows loudly.
    wide = np.array([2**200], dtype=zk_dtypes.prime_field(2**127 - 1))
    with self.assertRaises(OverflowError):
      wide.astype(np.uint64)

  def test_extension_int_operand_embeds_constant_term(self):
    dt = zk_dtypes.extension_field(10**9 + 7, 4, 5)
    e = np.zeros(1, dtype=dt)
    e[0] = (1, 2, 3, 4)
    self.assertEqual(tuple(int(c) for c in (e + 1).tolist()[0]), (2, 2, 3, 4))

  def test_mixed_base_extension_arithmetic(self):
    p = 10**9 + 7
    base = zk_dtypes.prime_field(p)
    ext = zk_dtypes.extension_field(p, 4, 5)
    b = np.array([42], dtype=base)
    e = np.zeros(1, dtype=ext)
    e[0] = (1, 2, 3, 4)
    # A base element embeds as the constant coefficient, either operand side.
    self.assertEqual(tuple(int(c) for c in (e + b).tolist()[0]), (43, 2, 3, 4))
    self.assertEqual(tuple(int(c) for c in (b + e).tolist()[0]), (43, 2, 3, 4))
    self.assertEqual(
        tuple(int(c) for c in (e * b).tolist()[0]), (42, 84, 126, 168)
    )
    self.assertEqual(
        tuple(int(c) for c in (e - b).tolist()[0]), ((1 - 42) % p, 2, 3, 4)
    )
    # The embedding is also available as an explicit safe cast.
    self.assertEqual(
        tuple(int(c) for c in b.astype(ext).tolist()[0]), (42, 0, 0, 0)
    )
    self.assertTrue(np.can_cast(base, ext))
    # A different base field still does not mix.
    other = np.array([1], dtype=zk_dtypes.prime_field(2147483647))
    with self.assertRaises(TypeError):
      np.add(other, e)

  def test_ec_int_construction_matches_legacy(self):
    legacy = zk_dtypes.bn254_g1_jacobian_mont
    p = self._BN254_FQ
    r = (1 << 256) % p
    plain = self._ec_g1_jacobian_param()
    # Read G's stored coordinates out of a legacy 1*G array, then mint a
    # descriptor that carries it as the generator.
    tmp = np.zeros(1, dtype=plain)
    tmp.view(np.uint8)[:] = np.array([1], dtype=legacy).view(np.uint8)
    gen = tuple(int(c) for c in tmp[0])
    withgen = np.dtype(
        zk_dtypes._zk_dtypes_ext.ec_point_descr(
            p, 256, 3, 1, r, pow(r, -1, p), 1, None, gen
        )
    )
    self.assertEqual(withgen.generator, gen)
    ns = [1, 2, 3, 7, 12345]
    arr = np.zeros(len(ns), dtype=withgen)
    for i, n in enumerate(ns):
      arr[i] = n
    np.testing.assert_array_equal(
        arr.view(np.uint8), np.array(ns, dtype=legacy).view(np.uint8)
    )
    # Without a generator, integer construction is refused rather than wrong.
    with self.assertRaisesRegex(TypeError, "no generator"):
      np.zeros(1, dtype=plain)[0] = 5

  @parameterized.parameters("canonical", "mont")
  def test_scalar_type_surface(self, storage):
    p = 10**9 + 7
    dt = zk_dtypes.prime_field(p, storage)
    a = np.array([3, 5], dtype=dt)
    x, y = a[0], a[1]
    # Operators, both operand orders, ints coerced as constant terms.
    self.assertEqual(int(x + y), 8)
    self.assertEqual(int(x - y), (3 - 5) % p)
    self.assertEqual(int(x * y), 15)
    self.assertEqual(int((x / y) * y), 3)
    self.assertEqual(int(x + 1), 4)
    self.assertEqual(int(1 + x), 4)
    self.assertEqual(int(-x), p - 3)
    self.assertEqual(int(x**3), 27)
    self.assertEqual(int(x**-1), pow(3, -1, p))
    with self.assertRaises(ZeroDivisionError):
      x / np.array([0], dtype=dt)[0]
    # Equality, hashing, and the value/raw/dtype accessors.
    self.assertTrue(x == 3)
    self.assertFalse(x == y)
    self.assertEqual(hash(x), hash(a[0]))
    self.assertEqual(x.item(), 3)
    self.assertEqual(x.dtype, dt)
    expected_raw = 3 * (1 << 32) % p if storage == "mont" else 3
    self.assertEqual(x.raw, expected_raw)
    # A scalar assigns back into an array of its field; a foreign field does
    # not.
    out = np.zeros(1, dtype=dt)
    out[0] = x
    self.assertEqual(int(out[0]), 3)
    with self.assertRaises(TypeError):
      np.zeros(1, dtype=zk_dtypes.prime_field(2147483647))[0] = x
    # str is the bare value; repr names the field.
    self.assertEqual(str(x), "3")
    self.assertIn(str(p), repr(x))

  def test_extension_scalar_coefficients(self):
    p = 10**9 + 7
    dt = zk_dtypes.extension_field(p, 4, 5)
    e = np.zeros(2, dtype=dt)
    e[0] = (1, 2, 3, 4)
    e[1] = (5, 6, 7, 8)
    u, v = e[0], e[1]
    self.assertEqual(len(u), 4)
    self.assertEqual([int(c) for c in u], [1, 2, 3, 4])
    self.assertEqual([int(c) for c in (u + v)], [6, 8, 10, 12])
    self.assertTrue((u / v) * v == u)
    self.assertEqual(u.item(), (1, 2, 3, 4))
    self.assertEqual(len(u.raw), 4)
    with self.assertRaises(TypeError):
      int(u)  # no single integer value
    with self.assertRaises(IndexError):
      u[4]

  def test_binary_scalar_surface(self):
    dt = zk_dtypes.binary_field(9)  # parametric: no legacy dtype at level 9
    z = np.array([0xAB], dtype=dt)[0]
    self.assertEqual(int(z), 0xAB)
    self.assertEqual(int(z + z), 0)  # characteristic 2
    self.assertEqual(int(-z), 0xAB)
    self.assertEqual(int((z / z)), 1)
    self.assertEqual(int(z**2), int(z * z))
    self.assertIn("tower_level=9", repr(z))

  def test_nonzero_and_truthiness(self):
    p = 10**9 + 7
    for storage in ("canonical", "mont"):
      a = np.array([0, 3, 0, p - 1], dtype=zk_dtypes.prime_field(p, storage))
      np.testing.assert_array_equal(np.nonzero(a)[0], [1, 3])
    bf = np.array([0, 1, 0, 5], dtype=zk_dtypes.binary_field(3))
    np.testing.assert_array_equal(np.nonzero(bf)[0], [1, 3])

  def test_field_equality_ufunc(self):
    p = 10**9 + 7
    for storage in ("canonical", "mont"):
      dt = zk_dtypes.prime_field(p, storage)
      a = np.array([1, 2, 3, 0], dtype=dt)
      b = np.array([1, 5, 3, 0], dtype=dt)
      np.testing.assert_array_equal(a == b, [True, False, True, True])
      np.testing.assert_array_equal(a != b, [False, True, False, False])
    # Separately-minted descriptors of the same field compare fine; a
    # different field is rejected rather than silently unequal.
    x = np.array([7], dtype=zk_dtypes.prime_field(p))
    y = np.array([7], dtype=zk_dtypes.prime_field(p))
    self.assertTrue(bool((x == y)[0]))
    with self.assertRaises(TypeError):
      np.equal(x, np.array([7], dtype=zk_dtypes.prime_field(2147483647)))

  def test_dtype_identity_and_hash(self):
    p = 10**9 + 7
    a = zk_dtypes.prime_field(p)
    b = zk_dtypes.prime_field(p)
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))
    self.assertNotEqual(a, zk_dtypes.prime_field(2147483647))
    self.assertNotEqual(a, zk_dtypes.prime_field(p, "canonical"))

  def test_dtype_repr_smoke(self):
    p = 10**9 + 7
    self.assertIn(str(p), repr(zk_dtypes.prime_field(p)))
    self.assertIn("mont=1", repr(zk_dtypes.prime_field(p)))

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

  # Legacy dtype name / base modulus / degree / non-residue / storage /
  # base width in bytes / is_montgomery.
  _EF_BYTE_MATCH = (
      ("babybearx4_mont", 2013265921, 4, 11, "mont", 4, True),
      ("koalabearx4_mont", 2130706433, 4, 3, "mont", 4, True),
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
        np.dtype(zk_dtypes.babybearx4_mont),
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
    legacy = zk_dtypes.bn254_g1_jacobian_mont
    param = self._ec_g1_jacobian_param()
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
    info = zk_dtypes.ecinfo(np.dtype(zk_dtypes.bn254_g1_jacobian_mont))
    if getattr(info, "is_montgomery", True):
      r = (1 << 256) % p
      return np.dtype(
          zk_dtypes._zk_dtypes_ext.ec_point_descr(
              p, 256, 3, 1, r, pow(r, -1, p)
          )
      )
    return np.dtype(zk_dtypes._zk_dtypes_ext.ec_point_descr(p, 256, 3, 0))

  def test_ec_identity_and_nonzero(self):
    legacy = zk_dtypes.bn254_g1_jacobian_mont
    param = self._ec_g1_jacobian_param()

    def pt(n):
      out = np.zeros(1, dtype=param)
      out.view(np.uint8)[:] = np.array([n], dtype=legacy).view(np.uint8)
      return out

    identity = np.zeros(1, dtype=param)  # all-zero bytes: Z = 0
    p3 = pt(3)
    # P + (-P) reduces to a representative of the identity (Z = 0), which the
    # cross-representative equality recognizes.
    self.assertTrue(bool(((p3 + (-p3)) == identity)[0]))
    # Adding the identity is a no-op up to group equality.
    self.assertTrue(bool(((p3 + identity) == p3)[0]))
    # np.nonzero counts non-identity points only.
    arr = np.zeros(3, dtype=param)
    arr[1:2] = p3
    np.testing.assert_array_equal(np.nonzero(arr)[0], [1])

  def test_ec_group_equality_cross_representative(self):
    legacy = zk_dtypes.bn254_g1_jacobian_mont
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

  _BN254_FR = 21888242871839275222246405745257275088548364400416034343698204186575808495617

  def _param_fr_scalar(self):
    # Parametric Fr scalar dtype (built directly: the Fr modulus is a curated
    # family that prime_field would resolve to the legacy dtype).
    return _param_prime(self._BN254_FR, 256, True)

  def test_ec_scalar_multiplication_byte_matches_legacy(self):
    legacy = zk_dtypes.bn254_g1_jacobian_mont
    param = self._ec_g1_jacobian_param()
    scalar_dt = self._param_fr_scalar()

    def pt(n):
      out = np.zeros(1, dtype=param)
      out.view(np.uint8)[:] = np.array([n], dtype=legacy).view(np.uint8)
      return out

    g = pt(1)
    for s in (2, 3, 7, 100, 12345):
      res = np.array([s], dtype=scalar_dt) * g
      np.testing.assert_array_equal(res.view(np.uint8), pt(s).view(np.uint8))
      # point * scalar (reverse operand order)
      res2 = g * np.array([s], dtype=scalar_dt)
      np.testing.assert_array_equal(res2.view(np.uint8), pt(s).view(np.uint8))

  def _ec_param(self, num_coords):
    p = self._BN254_FQ
    r = (1 << 256) % p
    return np.dtype(
        zk_dtypes._zk_dtypes_ext.ec_point_descr(
            p, 256, num_coords, 1, r, pow(r, -1, p)
        )
    )

  def test_ec_coordinate_rep_casts(self):
    legacy = {
        2: zk_dtypes.bn254_g1_affine_mont,
        3: zk_dtypes.bn254_g1_jacobian_mont,
        4: zk_dtypes.bn254_g1_xyzz_mont,
    }
    param = {nc: self._ec_param(nc) for nc in (2, 3, 4)}

    def to_param(arr, nc):
      out = np.zeros(len(arr), dtype=param[nc])
      out.view(np.uint8)[:] = arr.view(np.uint8)
      return out

    # The four affine<->projective directions the legacy backend registers;
    # byte-match its cast.
    for frm, to in [(3, 2), (4, 2), (2, 3), (2, 4)]:
      lsrc = np.array([5], dtype=legacy[frm])
      ldst = lsrc.astype(legacy[to])
      pdst = to_param(lsrc, frm).astype(param[to])
      np.testing.assert_array_equal(pdst.view(np.uint8), ldst.view(np.uint8))

    # Jacobian<->xyzz (legacy registers neither direction): verify correctness
    # by round-trip group equality and a shared affine projection.
    j = to_param(np.array([5], dtype=legacy[3]), 3)
    xy = j.astype(param[4])
    self.assertTrue(bool((xy.astype(param[3]) == j)[0]))
    np.testing.assert_array_equal(
        xy.astype(param[2]).view(np.uint8), j.astype(param[2]).view(np.uint8)
    )

  def test_ec_g2_jacobian_byte_matches_legacy(self):
    # G2 points are over Fp2 = Fq[u]/(u^2 + 1); the same Jacobian formulas run
    # over the degree-2 coordinate field.
    q = self._BN254_FQ
    r = (1 << 256) % q
    g2 = np.dtype(
        zk_dtypes._zk_dtypes_ext.ec_point_descr(
            q, 256, 3, 1, r, pow(r, -1, q), 2, q - 1
        )
    )
    legacy = zk_dtypes.bn254_g2_jacobian_mont
    self.assertEqual(np.dtype(g2).itemsize, np.dtype(legacy).itemsize)

    def pt(n):
      out = np.zeros(1, dtype=g2)
      out.view(np.uint8)[:] = np.array([n], dtype=legacy).view(np.uint8)
      return out

    for a, b in [(1, 2), (3, 5), (2, 2), (123, 456)]:
      la, lb = np.array([a], dtype=legacy), np.array([b], dtype=legacy)
      np.testing.assert_array_equal(
          (pt(a) + pt(b)).view(np.uint8), (la + lb).view(np.uint8)
      )
      np.testing.assert_array_equal(
          (pt(a) - pt(b)).view(np.uint8), (la - lb).view(np.uint8)
      )
      np.testing.assert_array_equal(
          (-pt(a)).view(np.uint8), (-la).view(np.uint8)
      )

    scalar = self._param_fr_scalar()
    g = pt(1)
    for s in (2, 3, 7, 100):
      np.testing.assert_array_equal(
          (np.array([s], dtype=scalar) * g).view(np.uint8), pt(s).view(np.uint8)
      )
    self.assertTrue(bool((pt(5) == (pt(2) + pt(3)))[0]))

  def test_ec_non_jacobian_arithmetic_rejected(self):
    # Group-law arithmetic is defined on the Jacobian representation; affine /
    # xyzz must be cast to Jacobian first (rather than misreading coordinates).
    aff = self._ec_param(2)
    a = np.zeros(2, dtype=aff)
    with self.assertRaisesRegex(TypeError, "Jacobian"):
      a + a  # noqa: B015
    with self.assertRaisesRegex(TypeError, "Jacobian"):
      np.equal(a, a)
    scalar = np.array(
        [3],
        dtype=np.dtype(
            zk_dtypes._zk_dtypes_ext.field_descr(
                self._BN254_FQ,
                1,
                0,
                256,
                1,
                (1 << 256) % self._BN254_FQ,
                pow((1 << 256) % self._BN254_FQ, -1, self._BN254_FQ),
            )
        ),
    )
    with self.assertRaisesRegex(TypeError, "Jacobian"):
      scalar * a

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

  @parameterized.parameters(
      "pallas_sf",
      "pallas_sf_mont",
      "vesta_sf",
      "vesta_sf_mont",
      "curve25519_bf",
      "curve25519_bf_mont",
      "curve25519_sf",
      "curve25519_sf_mont",
      "secp256k1_bf",
      "secp256k1_bf_mont",
      "secp256k1_sf",
      "secp256k1_sf_mont",
      "secp256r1_bf",
      "secp256r1_bf_mont",
      "secp256r1_sf",
      "secp256r1_sf_mont",
  )
  def test_curated_family_added_after_factory_resolves_to_legacy(self, name):
    # The curated maps are built by probing every registered dtype, so a family
    # added to the legacy stack after this factory (pallas/vesta, the classical
    # signature curves) resolves to its legacy dtype instead of minting a
    # duplicate parametric one.
    legacy = np.dtype(getattr(zk_dtypes, name))
    info = pfinfo(legacy)
    storage = "mont" if info.is_montgomery else "canonical"
    self.assertEqual(zk_dtypes.prime_field(info.modulus, storage), legacy)

  @parameterized.parameters(
      # (dtype name, the modulus its standard states)
      ("curve25519_bf", 2**255 - 19),  # RFC 7748
      (
          "curve25519_sf",  # ed25519 subgroup order L, RFC 8032 §5.1
          2**252 + 27742317777372353535851937790883648493,
      ),
      (
          "secp256k1_bf",  # SEC 2 §2.4.1
          0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
      ),
      (
          "secp256k1_sf",
          0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
      ),
      (
          "secp256r1_bf",  # SEC 2 §2.4.2 (NIST P-256)
          0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF,
      ),
      (
          "secp256r1_sf",
          0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551,
      ),
  )
  def test_classical_curve_fields_carry_their_standards_moduli(
      self, name, modulus
  ):
    # Both storage forms report the standard's modulus, and the arithmetic
    # actually reduces by it: (modulus - 1)² ≡ 1.
    for suffix in ("", "_mont"):
      dt = np.dtype(getattr(zk_dtypes, name + suffix))
      self.assertEqual(pfinfo(dt).modulus, modulus, name + suffix)
      top = np.array([modulus - 1], dtype=dt)
      self.assertTrue(
          (top * top == np.array([1], dtype=dt)).all(), name + suffix
      )

  def test_goldilocks_montgomery_deprecation_warning(self):
    # Montgomery goldilocks is inefficient (Solinas reduction); the factory
    # warns and steers to canonical. Other fields and canonical goldilocks
    # stay quiet.
    with self.assertWarnsRegex(DeprecationWarning, "Goldilocks"):
      zk_dtypes.prime_field(_GOLDILOCKS_MODULUS, "mont")
    with self.assertWarnsRegex(DeprecationWarning, "Goldilocks"):
      zk_dtypes.extension_field(_GOLDILOCKS_MODULUS, 2, 7, "mont")
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      zk_dtypes.prime_field(_GOLDILOCKS_MODULUS, "canonical")
      zk_dtypes.prime_field(2013265921, "mont")  # babybear stays fine in mont

  def test_novel_canonical_extension_multiply(self):
    # A canonical (non-Montgomery) extension multiply must not run the
    # Montgomery kernel; check the product against a plain-int reference.
    # Montgomery extensions are covered by the byte-match-legacy tests.
    p = 2**61 - 1  # single-word canonical base
    nr = 5
    ef = zk_dtypes.extension_field(p, 3, nr, "canonical")
    a = np.zeros(2, dtype=ef)
    v = a.view(np.uint64).reshape(2, 3)
    v[0] = [1, 2, 3]
    v[1] = [4, 5, 6]
    got = tuple(int(c) for c in (a[0:1] * a[1:2])[0])

    def ref(x, y):
      c = [0] * 5
      for i in range(3):
        for j in range(3):
          c[i + j] = (c[i + j] + x[i] * y[j]) % p
      for i in (4, 3):
        c[i - 3] = (c[i - 3] + nr * c[i]) % p
      return tuple(c[:3])

    self.assertEqual(got, ref([1, 2, 3], [4, 5, 6]))

  def test_ec_setitem_getitem_roundtrip_g1_and_g2(self):
    # setitem must accept the coordinate shape getitem returns: a bare int per
    # Fq coordinate (G1), or a (c0, c1) pair per Fp2 coordinate (G2). Regression:
    # G2 setitem forced PyNumber_Index on each coordinate, rejecting the tuple.
    q = self._BN254_FQ
    r = (1 << 256) % q
    g1 = self._ec_param(3)  # Jacobian G1, Montgomery
    a = np.zeros(1, dtype=g1)
    a[0] = (11, 22, 33)
    self.assertEqual(tuple(int(c) for c in a[0]), (11, 22, 33))

    g2 = np.dtype(
        zk_dtypes._zk_dtypes_ext.ec_point_descr(
            q, 256, 3, 1, r, pow(r, -1, q), 2, q - 1
        )
    )
    b = np.zeros(1, dtype=g2)
    b[0] = ((11, 12), (22, 23), (1, 0))
    self.assertEqual(
        tuple(tuple(int(x) for x in coord) for coord in b[0]),
        ((11, 12), (22, 23), (1, 0)),
    )
    # A wrong-width Fp2 coordinate is rejected.
    with self.assertRaisesRegex(ValueError, "Fp2 coordinate needs"):
      b[0] = ((1, 2, 3), (1, 0), (1, 0))

  def test_mont_canonical_astype_reencodes(self):
    # astype between the Montgomery and canonical forms of one field must
    # re-encode the value, not raw-copy the bytes.
    p = 10**9 + 7
    mont = zk_dtypes.prime_field(p, "mont")
    canon = zk_dtypes.prime_field(p, "canonical")
    vals = [7, 123456789, p - 1]
    a = np.array(vals, dtype=canon)
    b = a.astype(mont)
    self.assertEqual([int(x) for x in b.tolist()], vals)
    r = (1 << 32) % p
    self.assertEqual(b.view(np.uint32).tolist(), [(v * r) % p for v in vals])
    self.assertEqual([int(x) for x in b.astype(canon).tolist()], vals)

  def test_cast_between_distinct_fields_rejected(self):
    # A cast between genuinely different fields is meaningless (and across widths
    # would write out of bounds) — it must raise, not silently copy bytes.
    a = np.array([1], dtype=zk_dtypes.prime_field(10**9 + 7))
    with self.assertRaises(TypeError):
      a.astype(zk_dtypes.prime_field(2130706433))

  def test_high_binary_tower_round_trips(self):
    # Levels 11/12 (256/512-byte storage) must not truncate base_width_bytes to
    # 0 (would silently zero every element).
    for level, width in ((11, 256), (12, 512)):
      bf = np.dtype(zk_dtypes._zk_dtypes_ext.binary_field_descr(level))
      self.assertEqual(np.dtype(bf).itemsize, width)
      v = 0x123456789ABCDEF
      a = np.array([v], dtype=bf)
      self.assertEqual(int(a[0]), v)
      b = np.array([0xFF], dtype=bf)
      self.assertEqual(int((a + b)[0]), v ^ 0xFF)

  def test_scalar_mul_output_curve_mismatch_rejected(self):
    # out= of a different representation than the Jacobian point must be rejected
    # (the native scalar-mul sizes its write from the input point's width).
    jac = self._ec_param(3)
    affine = self._ec_param(2)
    scalar_dt = self._param_fr_scalar()
    s = np.array([3], dtype=scalar_dt)
    g = np.zeros(1, dtype=jac)
    out = np.zeros(1, dtype=affine)
    with self.assertRaises((TypeError, ValueError)):
      np.multiply(s, g, out=out)

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


# --- systematic parametric-kernel coverage --------------------------------
# These tests build the dtype directly through field_descr / binary_field_descr,
# bypassing the factory's curated resolution, so the *parametric* kernels run
# even for production moduli (goldilocks, bn254, ...) — the curated path resolves
# those to legacy dtypes and never exercises this code. Every result is checked
# against an independent pure-Python reference, not against the legacy dtype, so
# a bug shared by both the kernel and the legacy path could not hide (and a
# canonical-only or width-specific kernel bug is caught regardless of storage).

_EXT_MOD = zk_dtypes._zk_dtypes_ext


def _r_mod_p(width_bits, p):
  return (1 << width_bits) % p


def _param_prime(p, width_bits, is_mont):
  if is_mont:
    r = _r_mod_p(width_bits, p)
    return np.dtype(
        _EXT_MOD.field_descr(p, 1, 0, width_bits, 1, r, pow(r, -1, p))
    )
  return np.dtype(_EXT_MOD.field_descr(p, 1, 0, width_bits, 0))


def _param_ext(p, degree, nr, width_bits, is_mont):
  if is_mont:
    r = _r_mod_p(width_bits, p)
    return np.dtype(
        _EXT_MOD.field_descr(p, degree, nr, width_bits, 1, r, pow(r, -1, p))
    )
  return np.dtype(_EXT_MOD.field_descr(p, degree, nr, width_bits, 0))


def _encode_elems(elems, p, width_bits, is_mont):
  """Packs coefficient-lists (one per array element) into stored little-endian
  bytes, applying the Montgomery factor R = 2^width_bits — an encoder written
  independently of the C++ setitem so a shared bug cannot mask a wrong kernel.
  """
  wb = width_bits // 8
  r = _r_mod_p(width_bits, p) if is_mont else 1
  out = bytearray()
  for coeffs in elems:
    for c in coeffs:
      v = (c % p) * r % p if is_mont else c % p
      out += (v).to_bytes(wb, "little")
  return np.frombuffer(bytes(out), np.uint8).copy()


def _decode_elem(raw_u8, p, width_bits, is_mont, degree):
  wb = width_bits // 8
  rinv = pow(_r_mod_p(width_bits, p), -1, p) if is_mont else None
  out = []
  for k in range(degree):
    v = int.from_bytes(bytes(raw_u8[k * wb : (k + 1) * wb]), "little")
    out.append(v * rinv % p if is_mont else v % p)
  return out


def _ref_prime(op, a, b, p):
  return {"add": (a + b) % p, "sub": (a - b) % p, "mul": a * b % p}[op]


def _ref_ext(op, ca, cb, p, degree, nr):
  if op != "mul":
    s = 1 if op == "add" else -1
    return [(ca[i] + s * cb[i]) % p for i in range(degree)]
  prod = [0] * (2 * degree - 1)
  for i in range(degree):
    for j in range(degree):
      prod[i + j] = (prod[i + j] + ca[i] * cb[j]) % p
  for i in range(2 * degree - 2, degree - 1, -1):
    prod[i - degree] = (prod[i - degree] + nr * prod[i]) % p
  return [prod[i] % p for i in range(degree)]


_OPS = {"add": np.add, "sub": np.subtract, "mul": np.multiply}

# (label, modulus, storage width bits) spanning every native width and both the
# single-word and BigInt kernels; curated production moduli plus novel ones.
_PRIME_MATRIX = (
    ("babybear", 2013265921, 32),
    ("mersenne31", 2147483647, 32),
    ("novel30", 10**9 + 7, 32),
    ("goldilocks", 2**64 - 2**32 + 1, 64),
    ("mersenne61", 2**61 - 1, 64),
    ("mersenne127", 2**127 - 1, 128),
    (
        "bn254_fr",
        21888242871839275222246405745257275088548364400416034343698204186575808495617,
        256,
    ),
    (
        "pallas",
        0x40000000000000000000000000000000224698FC0994A8DD8C46EB2100000001,
        256,
    ),
)

# (label, base modulus, degree, non_residue, base width bits). The non-residue
# need not be irreducible — the ring Fp[X]/(X^d - nr) multiply is well-defined
# either way, which is what the kernel computes.
_EXT_MATRIX = (
    ("babybear_d4", 2013265921, 4, 11, 32),
    ("koalabear_d4", 2130706433, 4, 3, 32),
    ("mersenne31_d2", 2147483647, 2, 5, 32),
    ("goldilocks_d2", 2**64 - 2**32 + 1, 2, 7, 64),
    ("goldilocks_d3", 2**64 - 2**32 + 1, 3, 7, 64),
    ("mersenne61_d3", 2**61 - 1, 3, 5, 64),
)


@multi_threaded(num_workers=3)
class ParametricFieldMatrixTest(parameterized.TestCase):

  @parameterized.parameters(
      *(
          (lbl, p, w, mont, op)
          for (lbl, p, w) in _PRIME_MATRIX
          for mont in (False, True)
          for op in ("add", "sub", "mul")
      )
  )
  def test_prime_kernel(self, label, p, width_bits, is_mont, op):
    dt = _param_prime(p, width_bits, is_mont)
    # Fixed edge values plus seeded random vectors: the edges pin boundary
    # behavior, the random draws catch carry bugs off the edge set.
    rng = random.Random(f"{label}-{op}-{is_mont}")
    av = [0, 1, 2, p - 1, p // 2, 7, p - 3, p - 1]
    bv = [0, p - 1, 3, p - 1, p // 3, 7, 5, 1]
    av += [rng.randrange(p) for _ in range(24)]
    bv += [rng.randrange(p) for _ in range(24)]
    a = np.zeros(len(av), dtype=dt)
    b = np.zeros(len(bv), dtype=dt)
    a.view(np.uint8)[:] = _encode_elems(
        [[v] for v in av], p, width_bits, is_mont
    )
    b.view(np.uint8)[:] = _encode_elems(
        [[v] for v in bv], p, width_bits, is_mont
    )
    got = _OPS[op](a, b).view(np.uint8).reshape(len(av), width_bits // 8)
    for i in range(len(av)):
      dec = _decode_elem(got[i], p, width_bits, is_mont, 1)[0]
      self.assertEqual(
          dec,
          _ref_prime(op, av[i], bv[i], p),
          f"{label} {op} mont={is_mont} i={i}",
      )

  @parameterized.parameters(
      *(
          (lbl, p, d, nr, w, mont, op)
          for (lbl, p, d, nr, w) in _EXT_MATRIX
          for mont in (False, True)
          for op in ("add", "sub", "mul")
      )
  )
  def test_extension_kernel(
      self, label, p, degree, nr, width_bits, is_mont, op
  ):
    dt = _param_ext(p, degree, nr, width_bits, is_mont)
    ea = [[(i * 7 + 5 + j * 3) % p for j in range(degree)] for i in range(4)]
    eb = [[(i * 11 + 2 + j * 9) % p for j in range(degree)] for i in range(4)]
    a = np.zeros(len(ea), dtype=dt)
    b = np.zeros(len(eb), dtype=dt)
    a.view(np.uint8)[:] = _encode_elems(ea, p, width_bits, is_mont)
    b.view(np.uint8)[:] = _encode_elems(eb, p, width_bits, is_mont)
    got = (
        _OPS[op](a, b).view(np.uint8).reshape(len(ea), degree * width_bits // 8)
    )
    for i in range(len(ea)):
      dec = _decode_elem(got[i], p, width_bits, is_mont, degree)
      self.assertEqual(
          dec,
          _ref_ext(op, ea[i], eb[i], p, degree, nr),
          f"{label} {op} mont={is_mont} i={i}",
      )

  @parameterized.parameters(*range(0, 13))
  def test_binary_tower_kernel(self, level):
    bf = np.dtype(_EXT_MOD.binary_field_descr(level))
    wb = bf.itemsize
    mask = (1 << (1 << level)) - 1

    def arr(vals):
      out = np.zeros(len(vals), dtype=bf)
      out.view(np.uint8).reshape(len(vals), wb)[:] = [
          list((v & mask).to_bytes(wb, "little")) for v in vals
      ]
      return out

    xs = [0, 1, mask, (mask ^ (mask >> 1)) & mask, 0xA5, mask // 3, 0xDEADBEEF]
    ys = [0, mask, 2, 1, 0x5A, mask // 7, 0x1234567]
    a, b, c = arr(xs), arr(ys), arr(ys[::-1])
    one = arr([1] * len(xs))

    # Addition is XOR (characteristic 2).
    for i, (x, y) in enumerate(zip(xs, ys)):
      self.assertEqual(
          int((a + b)[i]), (x & mask) ^ (y & mask), f"L{level} xor {i}"
      )

    if level <= 7:
      # Authoritative cross-check: an independent C++ kernel (the legacy tower
      # dtype) multiplies the same bytes to the same result.
      legacy = np.dtype(getattr(zk_dtypes, f"binary_field_t{level}"))
      la, lb = np.zeros(len(xs), dtype=legacy), np.zeros(len(ys), dtype=legacy)
      la.view(np.uint8)[:] = a.view(np.uint8)
      lb.view(np.uint8)[:] = b.view(np.uint8)
      np.testing.assert_array_equal(
          (a * b).view(np.uint8), (la * lb).view(np.uint8)
      )

    # Basis-independent field axioms — the only reference available for levels
    # > 7 (no legacy dtype), and enough to catch a wrong Karatsuba recombination
    # or reduction: identity, commutativity, distributivity over XOR.
    np.testing.assert_array_equal((a * one).view(np.uint8), a.view(np.uint8))
    np.testing.assert_array_equal(
        (a * b).view(np.uint8), (b * a).view(np.uint8)
    )
    np.testing.assert_array_equal(
        (a * (b + c)).view(np.uint8), ((a * b) + (a * c)).view(np.uint8)
    )
    np.testing.assert_array_equal(
        ((a * b) * c).view(np.uint8), (a * (b * c)).view(np.uint8)
    )

  def test_binary_tower_level8_extends_level7(self):
    # GF(2^128) sits inside GF(2^256) as the low half; multiplying embedded
    # level-7 elements at level 8 must reproduce the level-7 (canonical
    # Fan-Paar) product bit-for-bit, pinning TowerMul to the same tower the
    # native kernels build.
    b7 = np.dtype(zk_dtypes.binary_field(7))
    b8 = np.dtype(zk_dtypes.binary_field(8))

    def arr(dt, wb, vals):
      out = np.zeros(len(vals), dtype=dt)
      out.view(np.uint8).reshape(len(vals), wb)[:] = [
          list(v.to_bytes(wb, "little")) for v in vals
      ]
      return out

    xs = [1, 2, 1 << 64, (1 << 128) - 1, 0xDEADBEEFCAFEBABE]
    ys = [1, 3, (1 << 64) | 1, 0xA5A5A5A5A5A5A5A5, (1 << 127) | 7]
    a7, b7v = arr(b7, 16, xs), arr(b7, 16, ys)
    a8, b8v = arr(b8, 32, xs), arr(b8, 32, ys)
    prod7 = (a7 * b7v).view(np.uint8).reshape(len(xs), 16)
    prod8 = (a8 * b8v).view(np.uint8).reshape(len(xs), 32)
    np.testing.assert_array_equal(prod8[:, :16], prod7)
    np.testing.assert_array_equal(
        prod8[:, 16:], np.zeros((len(xs), 16), np.uint8)
    )


if __name__ == "__main__":
  absltest.main()
