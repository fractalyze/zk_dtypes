# Copyright 2026 The zk_dtypes Authors.
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

"""Test cases for binary field types (GF(2^n) tower fields)."""

import contextlib
import copy
import operator
import pickle
import random
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes
from multi_thread_utils import multi_threaded
import numpy as np

binary_field_t0 = zk_dtypes.binary_field_t0
binary_field_t1 = zk_dtypes.binary_field_t1
binary_field_t2 = zk_dtypes.binary_field_t2
binary_field_t3 = zk_dtypes.binary_field_t3
binary_field_t4 = zk_dtypes.binary_field_t4
binary_field_t5 = zk_dtypes.binary_field_t5
binary_field_t6 = zk_dtypes.binary_field_t6
binary_field_t7 = zk_dtypes.binary_field_t7
binary_field_ghash = zk_dtypes.binary_field_ghash
binary_field_gf8_aes = zk_dtypes.binary_field_gf8_aes

BINARY_FIELD_TYPES = [
    binary_field_t0,
    binary_field_t1,
    binary_field_t2,
    binary_field_t3,
    binary_field_t4,
    binary_field_t5,
    binary_field_t6,
    binary_field_t7,
    binary_field_ghash,
    binary_field_gf8_aes,
]

# Small binary fields (fit in 64 bits) for tests that need int conversion
SMALL_BINARY_FIELD_TYPES = [
    binary_field_t0,
    binary_field_t1,
    binary_field_t2,
    binary_field_t3,
    binary_field_t4,
    binary_field_t5,
    binary_field_t6,
    binary_field_gf8_aes,
]

# Value masks for each binary field type
VALUE_MASKS = {
    binary_field_t0: (1 << 1) - 1,
    binary_field_t1: (1 << 2) - 1,
    binary_field_t2: (1 << 4) - 1,
    binary_field_t3: (1 << 8) - 1,
    binary_field_t4: (1 << 16) - 1,
    binary_field_t5: (1 << 32) - 1,
    binary_field_t6: (1 << 64) - 1,
    binary_field_t7: (1 << 128) - 1,
    binary_field_ghash: (1 << 128) - 1,
    binary_field_gf8_aes: (1 << 8) - 1,
}

# Test values for each binary field type (within valid range)
VALUES = {
    binary_field_t0: [0, 1],
    binary_field_t1: [0, 1, 2, 3],
    binary_field_t2: [0, 1, 7, 15],
    binary_field_t3: random.sample(range(0, 256), 4),
    binary_field_t4: random.sample(range(0, 65536), 4),
    binary_field_t5: random.sample(range(0, 2**16), 4),
    binary_field_t6: random.sample(range(0, 2**16), 4),
    binary_field_t7: random.sample(range(0, 2**16), 4),
    binary_field_ghash: random.sample(range(0, 2**16), 4),
    binary_field_gf8_aes: random.sample(range(0, 256), 4),
}


# Reference GF(2^128) multiply in the flat GHASH/POLYVAL basis
# (p(x) = x^128 + x^7 + x^2 + x + 1), independent of the C++ implementation:
# a schoolbook carryless product then bit-by-bit reduction. Bit i of each 128-bit
# int is the coefficient of x^i. Pins binary_field_ghash to the exact basis that
# GHASH/POLYVAL consumers hash raw field bytes in.
_GHASH_MASK = (1 << 128) - 1
_GHASH_REDUCE = (1 << 7) | (1 << 2) | (1 << 1) | 1  # x^7 + x^2 + x + 1 = 0x87


def _ghash_ref_mul(a: int, b: int) -> int:
  prod = 0
  for i in range(128):
    if (a >> i) & 1:
      prod ^= b << i
  for i in range(255, 127, -1):
    if (prod >> i) & 1:
      prod ^= (1 << i) | (_GHASH_REDUCE << (i - 128))
  return prod & _GHASH_MASK


# Reference GF(2^8) multiply in the flat AES/Rijndael basis
# (p(x) = x^8 + x^4 + x^3 + x + 1), independent of the C++ implementation:
# a schoolbook carryless product then bit-by-bit reduction. Bit i of each 8-bit
# int is the coefficient of x^i. Pins binary_field_gf8_aes to the AES basis that
# flock's phi8 univariate skip depends on.
_AES_MASK = (1 << 8) - 1
_AES_REDUCE = (1 << 4) | (1 << 3) | (1 << 1) | 1  # x^4 + x^3 + x + 1 = 0x1B


def _aes_ref_mul(a: int, b: int) -> int:
  prod = 0
  for i in range(8):
    if (a >> i) & 1:
      prod ^= b << i
  for i in range(15, 7, -1):
    if (prod >> i) & 1:
      prod ^= (1 << i) | (_AES_REDUCE << (i - 8))
  return prod & _AES_MASK


# Reference multiply for the Fan-Paar / Binius tower, independent of the C++
# implementation. Level k (k >= 1) is GF(2^{2^{k-1}})[X] / (X^2 + beta_{k-1}*X
# + 1), where beta_{k-1} is the subfield generator; multiplying by it is the
# recursive "multiply by generator" _tower_mulgen. Pins BinaryFieldT* to the
# same tower Binius' BinaryField*b types use (byte-compatible).
def _tower_mulgen(x: int, level: int) -> int:
  if level == 0:
    return x & 1  # generator of GF(2) is 1
  h = 1 << (level - 1)
  mask = (1 << h) - 1
  a0, a1 = x & mask, (x >> h) & mask
  return a1 | ((a0 ^ _tower_mulgen(a1, level - 1)) << h)


def _tower_ref_mul(a: int, b: int, level: int) -> int:
  if level == 0:
    return a & b & 1
  h = 1 << (level - 1)
  mask = (1 << h) - 1
  a0, a1 = a & mask, (a >> h) & mask
  b0, b1 = b & mask, (b >> h) & mask
  a0b0 = _tower_ref_mul(a0, b0, level - 1)
  a1b1 = _tower_ref_mul(a1, b1, level - 1)
  c0 = a0b0 ^ a1b1
  cross = _tower_ref_mul(a0 ^ a1, b0 ^ b1, level - 1)
  c1 = cross ^ a0b0 ^ a1b1 ^ _tower_mulgen(a1b1, level - 1)
  return c0 | (c1 << h)


# Tower level per type; the flat GHASH field is not a tower and is excluded.
TOWER_LEVELS = {
    binary_field_t0: 0,
    binary_field_t1: 1,
    binary_field_t2: 2,
    binary_field_t3: 3,
    binary_field_t4: 4,
    binary_field_t5: 5,
    binary_field_t6: 6,
    binary_field_t7: 7,
}

# Known-answer products from Binius' canonical BinaryField{8,16,32}b (real
# reference outputs, per issue #147). Pin byte-compatibility with Binius, which
# the self-consistent oracle above cannot establish on its own.
BINIUS_CANONICAL_KAT = {
    binary_field_t3: [
        (0x12, 0x34, 0x9B),
        (0x2D, 0x2D, 0xCC),
        (0x80, 0x80, 0x57),
    ],
    binary_field_t4: [(0x1234, 0x5678, 0x54FE), (0x8000, 0x8000, 0xA557)],
    binary_field_t5: [(0x12345678, 0x9ABCDEF0, 0x9F77A270)],
}


@contextlib.contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


# Tests for the Python scalar type
@multi_threaded(num_workers=3)
class ScalarTest(parameterized.TestCase):

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testModuleName(self, scalar_type):
    self.assertEqual(scalar_type.__module__, "zk_dtypes")

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testPickleable(self, scalar_type):
    x = np.array(VALUES[scalar_type], dtype=scalar_type)
    serialized = pickle.dumps(x)
    x_out = pickle.loads(serialized)
    self.assertEqual(x_out.dtype, x.dtype)
    self.assertTrue((x_out == x).all())

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testRoundTripToPythonScalar(self, scalar_type):
    for v in VALUES[scalar_type]:
      self.assertEqual(v, int(scalar_type(v)))
      self.assertEqual(scalar_type(v), scalar_type(int(scalar_type(v))))

  @parameterized.product(scalar_type=SMALL_BINARY_FIELD_TYPES)
  def testRoundTripNumpyTypes(self, scalar_type):
    for dtype in [np.uint64]:
      for f in VALUES[scalar_type]:
        self.assertEqual(dtype(f), dtype(scalar_type(dtype(f))))
        self.assertEqual(int(dtype(f)), int(scalar_type(dtype(f))))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testStr(self, scalar_type):
    for value in VALUES[scalar_type]:
      self.assertEqual(str(value), str(scalar_type(value)))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testRepr(self, scalar_type):
    for value in VALUES[scalar_type]:
      self.assertEqual(str(value), repr(scalar_type(value)))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testItem(self, scalar_type):
    self.assertIsInstance(scalar_type(1).item(), scalar_type)
    self.assertEqual(scalar_type(1).item(), scalar_type(1))

  @parameterized.product(
      scalar_type=BINARY_FIELD_TYPES,
      op=[
          operator.le,
          operator.lt,
          operator.eq,
          operator.ne,
          operator.ge,
          operator.gt,
      ],
  )
  def testComparison(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        result = op(scalar_type(v), scalar_type(w))
        self.assertEqual(op(v, w), result)
        self.assertIsInstance(result, np.bool_)

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testNegation(self, scalar_type):
    """In characteristic 2, negation is identity: -x = x."""
    for v in VALUES[scalar_type]:
      out = -scalar_type(v)
      self.assertIsInstance(out, scalar_type)
      self.assertEqual(scalar_type(v), out)

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testAddition(self, scalar_type):
    """In characteristic 2, addition is XOR."""
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        out = scalar_type(v) + scalar_type(w)
        self.assertIsInstance(out, scalar_type)
        self.assertEqual(scalar_type(v ^ w), out, msg=(v, w))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testSubtraction(self, scalar_type):
    """In characteristic 2, subtraction is also XOR (same as addition)."""
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        out = scalar_type(v) - scalar_type(w)
        self.assertIsInstance(out, scalar_type)
        self.assertEqual(scalar_type(v ^ w), out, msg=(v, w))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testDoubleIsZero(self, scalar_type):
    """In characteristic 2, x + x = 0."""
    for v in VALUES[scalar_type]:
      out = scalar_type(v) + scalar_type(v)
      self.assertEqual(scalar_type(0), out)

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testMultiplication(self, scalar_type):
    """Tower field multiplication (NOT integer multiplication)."""
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        out = scalar_type(v) * scalar_type(w)
        self.assertIsInstance(out, scalar_type)
        # Verify multiplicative identity
        self.assertEqual(scalar_type(v) * scalar_type(1), scalar_type(v))
        self.assertEqual(scalar_type(1) * scalar_type(w), scalar_type(w))
        # Verify zero
        self.assertEqual(scalar_type(v) * scalar_type(0), scalar_type(0))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testDivision(self, scalar_type):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        if w == 0:
          with self.assertRaises(ZeroDivisionError):
            scalar_type(v) / scalar_type(w)
        else:
          out = scalar_type(v) / scalar_type(w)
          self.assertIsInstance(out, scalar_type)
          # Verify: (v / w) * w = v
          self.assertEqual(out * scalar_type(w), scalar_type(v), msg=(v, w))

  @parameterized.product(
      scalar_type=BINARY_FIELD_TYPES,
      op=[operator.add, operator.sub, operator.mul, operator.truediv],
  )
  def testPyIntCoercion(self, scalar_type, op):
    """Binary ops accept a Python int on either side."""
    mask = VALUE_MASKS[scalar_type]
    int_values = [w for w in [1, 3, 7] if w <= mask]
    for v in VALUES[scalar_type]:
      if op is operator.truediv and v == 0:
        continue
      x = scalar_type(v)
      for w in int_values:
        out_r = op(x, w)
        self.assertIsInstance(out_r, scalar_type)
        self.assertEqual(out_r, op(x, scalar_type(w)), msg=(v, w, "right"))
        out_l = op(w, x)
        self.assertIsInstance(out_l, scalar_type)
        self.assertEqual(out_l, op(scalar_type(w), x), msg=(v, w, "left"))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testInverse(self, scalar_type):
    """Test multiplicative inverse: x * x^(-1) = 1."""
    for v in VALUES[scalar_type]:
      if v == 0:
        with self.assertRaises(ZeroDivisionError):
          scalar_type(v) ** -1
      else:
        inv = scalar_type(v) ** -1
        self.assertIsInstance(inv, scalar_type)
        self.assertEqual(scalar_type(v) * inv, scalar_type(1))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testPower(self, scalar_type):
    for v in VALUES[scalar_type]:
      # Test positive exponents
      out = scalar_type(v) ** 3
      self.assertIsInstance(out, scalar_type)
      self.assertEqual(scalar_type(v) * scalar_type(v) * scalar_type(v), out)

      # Test x^0 = 1
      self.assertEqual(scalar_type(v) ** 0, scalar_type(1))

      # Test x^1 = x
      self.assertEqual(scalar_type(v) ** 1, scalar_type(v))

  CAST_DTYPES = [
      np.int8,
      np.int16,
      np.int32,
      np.int64,
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
  ]

  @parameterized.product(a=[binary_field_t3], b=CAST_DTYPES + [binary_field_t3])
  def test8BitCanCast(self, a, b):
    allowed_casts = [
        (binary_field_t3, binary_field_t3),
        (binary_field_t3, np.uint8),
        (binary_field_t3, np.uint16),
        (binary_field_t3, np.uint32),
        (binary_field_t3, np.uint64),
    ]
    self.assertEqual(
        ((a, b) in allowed_casts), np.can_cast(a, b, casting="safe")
    )

  @parameterized.product(a=[binary_field_t6], b=CAST_DTYPES + [binary_field_t6])
  def test64BitCanCast(self, a, b):
    allowed_casts = [
        (binary_field_t6, binary_field_t6),
        (binary_field_t6, np.uint64),
    ]
    self.assertEqual(
        ((a, b) in allowed_casts), np.can_cast(a, b, casting="safe")
    )

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testIssubdtype(self, scalar_type):
    self.assertTrue(np.issubdtype(scalar_type, np.generic))
    self.assertTrue(np.issubdtype(np.dtype(scalar_type), np.generic))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testCastToDtype(self, scalar_type):
    name = scalar_type.__name__
    dt = np.dtype(scalar_type)
    self.assertIs(dt.type, scalar_type)
    self.assertEqual(dt.name, name)
    self.assertEqual(repr(dt), f"dtype({name})")


# Tests for numpy arrays
@multi_threaded(num_workers=3)
class ArrayTest(parameterized.TestCase):

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testDtype(self, scalar_type):
    self.assertEqual(scalar_type, np.dtype(scalar_type))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testHash(self, scalar_type):
    h = hash(np.dtype(scalar_type))
    self.assertEqual(h, hash(np.dtype(scalar_type.dtype)))
    self.assertEqual(h, hash(np.dtype(scalar_type.__name__)))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testDeepCopyDoesNotAlterHash(self, scalar_type):
    dtype = np.dtype(scalar_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testArray(self, scalar_type):
    values = (
        VALUES[scalar_type][:3]
        if len(VALUES[scalar_type]) >= 3
        else VALUES[scalar_type]
    )
    x = np.array([values], dtype=scalar_type)
    self.assertEqual(scalar_type, x.dtype)
    self.assertTrue((x == x).all())

  @parameterized.product(
      scalar_type=BINARY_FIELD_TYPES,
      ufunc=[np.nonzero, np.argmax, np.argmin],
  )
  def testUnaryPredicateUfunc(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type], dtype=np.uint64)
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = ufunc(y)
    x_result = ufunc(x)
    np.testing.assert_array_equal(x_result, y_result)

  @parameterized.product(
      scalar_type=BINARY_FIELD_TYPES,
      ufunc=[
          np.less,
          np.less_equal,
          np.greater,
          np.greater_equal,
          np.equal,
          np.not_equal,
      ],
  )
  def testPredicateUfuncs(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type], dtype=np.uint64)
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    np.testing.assert_array_equal(
        ufunc(x[:, None], x[None, :]),
        ufunc(y[:, None], y[None, :]),
    )

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testNegativeUfunc(self, scalar_type):
    """In characteristic 2, negation is identity."""
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = np.negative(y)
    self.assertTrue((y == y_result).all())

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testAddUfunc(self, scalar_type):
    """In characteristic 2, x + x = 0."""
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = np.add(y, y)
    zeros = np.zeros(len(VALUES[scalar_type]), dtype=scalar_type)
    self.assertTrue((y_result == zeros).all())

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testSubtractUfunc(self, scalar_type):
    """In characteristic 2, x - x = 0."""
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = np.subtract(y, y)
    zeros = np.zeros(len(VALUES[scalar_type]), dtype=scalar_type)
    self.assertTrue((y_result == zeros).all())

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testMultiplyUfunc(self, scalar_type):
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    one = np.array([1], dtype=scalar_type)
    # y * 1 = y
    y_result = np.multiply(y, np.broadcast_to(one, y.shape))
    self.assertTrue((y == y_result).all())

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  @ignore_warning(category=RuntimeWarning, message="divide by zero encountered")
  def testDivideUfunc(self, scalar_type):
    y = np.array([v for v in VALUES[scalar_type] if v != 0], dtype=scalar_type)
    if len(y) == 0:
      self.skipTest("No non-zero values to test")
    # y / y = 1
    y_result = np.divide(y, y)
    ones = np.ones(len(y), dtype=scalar_type)
    self.assertTrue((y_result == ones).all())

  @parameterized.product(scalar_type=BINARY_FIELD_TYPES)
  def testPowerUfunc(self, scalar_type):
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = y**3
    for i in range(len(y_result)):
      x = scalar_type(VALUES[scalar_type][i])
      self.assertEqual(x**3, y_result[i])

  @parameterized.product(scalar_type=SMALL_BINARY_FIELD_TYPES)
  def testArrayCastToInt(self, scalar_type):
    """Test casting binary field arrays to integer arrays."""
    values = VALUES[scalar_type]
    y = np.array(values, dtype=scalar_type)
    y_int = y.astype(np.uint64)
    np.testing.assert_array_equal(y_int, np.array(values, dtype=np.uint64))

  @parameterized.product(scalar_type=SMALL_BINARY_FIELD_TYPES)
  def testArrayCastFromInt(self, scalar_type):
    """Test casting integer arrays to binary field arrays."""
    values = VALUES[scalar_type]
    x = np.array(values, dtype=np.uint64)
    y = x.astype(scalar_type)
    for i, v in enumerate(values):
      self.assertEqual(int(y[i]), v)


@multi_threaded(num_workers=3)
class TowerBasisTest(parameterized.TestCase):
  """BinaryFieldT* must realize the Fan-Paar / Binius tower exactly."""

  @parameterized.product(scalar_type=list(TOWER_LEVELS))
  def testMultiplyMatchesTowerOracle(self, scalar_type):
    level = TOWER_LEVELS[scalar_type]
    bits = 1 << level
    rng = random.Random(0xB1)
    samples = [0, 1, VALUE_MASKS[scalar_type]] + [
        rng.getrandbits(bits) for _ in range(64)
    ]
    for a in samples:
      for b in samples:
        self.assertEqual(
            int(scalar_type(a) * scalar_type(b)),
            _tower_ref_mul(a, b, level),
            msg=(scalar_type.__name__, hex(a), hex(b)),
        )

  @parameterized.product(scalar_type=list(BINIUS_CANONICAL_KAT))
  def testMatchesBiniusCanonicalVectors(self, scalar_type):
    for a, b, want in BINIUS_CANONICAL_KAT[scalar_type]:
      self.assertEqual(
          int(scalar_type(a) * scalar_type(b)),
          want,
          msg=(scalar_type.__name__, hex(a), hex(b)),
      )


@multi_threaded(num_workers=3)
class GhashBasisTest(parameterized.TestCase):
  """binary_field_ghash must realize the flat GHASH/POLYVAL basis exactly."""

  GF = binary_field_ghash
  EDGE = [
      0,
      1,
      2,  # x
      1 << 63,
      1 << 64,  # x^64
      1 << 127,  # x^127
      (1 << 128) - 1,
      0x0123456789ABCDEFFEDCBA9876543210,
  ]

  def _samples(self, n=400):
    rng = random.Random(20260702)
    return self.EDGE + [rng.getrandbits(128) for _ in range(n)]

  def testFullWidthRoundTrip(self):
    for a in self._samples(50):
      self.assertEqual(int(self.GF(a)), a, msg=hex(a))

  def testMultiplyMatchesReference(self):
    rng = random.Random(7)
    for a in self._samples():
      b = rng.getrandbits(128)
      self.assertEqual(
          int(self.GF(a) * self.GF(b)),
          _ghash_ref_mul(a, b),
          msg=(hex(a), hex(b)),
      )

  def testSquareMatchesReference(self):
    for a in self._samples():
      self.assertEqual(
          int(self.GF(a) * self.GF(a)), _ghash_ref_mul(a, a), msg=hex(a)
      )

  def testInverseIsMultiplicative(self):
    for a in self._samples(100):
      if a == 0:
        continue
      self.assertEqual(int(self.GF(a) * (self.GF(a) ** -1)), 1, msg=hex(a))

  def testByteLayoutIsLittleEndianLoHi(self):
    for a in self.EDGE:
      arr = np.array([a], dtype=self.GF)
      lo = a & ((1 << 64) - 1)
      hi = (a >> 64) & ((1 << 64) - 1)
      self.assertEqual(
          arr.tobytes(),
          lo.to_bytes(8, "little") + hi.to_bytes(8, "little"),
          msg=hex(a),
      )


@multi_threaded(num_workers=3)
class Gf8AesBasisTest(parameterized.TestCase):
  """binary_field_gf8_aes must realize the flat AES/Rijndael basis exactly."""

  GF = binary_field_gf8_aes
  EDGE = [0, 1, 2, 0x10, 0x80, 0x53, 0xCA, 0xFF]

  def _samples(self, n=256):
    rng = random.Random(20260709)
    return self.EDGE + [rng.getrandbits(8) for _ in range(n)]

  def testFullWidthRoundTrip(self):
    for a in self._samples(50):
      self.assertEqual(int(self.GF(a)), a, msg=hex(a))

  def testMultiplyMatchesReference(self):
    rng = random.Random(7)
    for a in self._samples():
      b = rng.getrandbits(8)
      self.assertEqual(
          int(self.GF(a) * self.GF(b)),
          _aes_ref_mul(a, b),
          msg=(hex(a), hex(b)),
      )

  def testSquareMatchesReference(self):
    for a in self._samples():
      self.assertEqual(
          int(self.GF(a) * self.GF(a)), _aes_ref_mul(a, a), msg=hex(a)
      )

  def testInverseIsMultiplicative(self):
    for a in self._samples(100):
      if a == 0:
        continue
      self.assertEqual(int(self.GF(a) * (self.GF(a) ** -1)), 1, msg=hex(a))

  def testByteLayoutIsSingleByte(self):
    for a in self.EDGE:
      arr = np.array([a], dtype=self.GF)
      self.assertEqual(arr.tobytes(), a.to_bytes(1, "little"), msg=hex(a))

  def testMatchesFipsExample(self):
    # FIPS-197 §4.2 worked example: {57} · {83} = {c1}.
    self.assertEqual(int(self.GF(0x57) * self.GF(0x83)), 0xC1)


if __name__ == "__main__":
  absltest.main()
