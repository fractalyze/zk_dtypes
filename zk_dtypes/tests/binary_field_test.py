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

BINARY_FIELD_TYPES = [
    binary_field_t0,
    binary_field_t1,
    binary_field_t2,
    binary_field_t3,
    binary_field_t4,
    binary_field_t5,
    binary_field_t6,
    binary_field_t7,
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


if __name__ == "__main__":
  absltest.main()
