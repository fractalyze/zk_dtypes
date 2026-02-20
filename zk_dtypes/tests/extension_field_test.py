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

"""Test cases for extension field types."""

import copy
import operator
import pickle
import random

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes
from zk_dtypes import efinfo
from multi_thread_utils import multi_threaded
import numpy as np

babybearx4 = zk_dtypes.babybearx4
babybearx4_mont = zk_dtypes.babybearx4_mont
goldilocksx3 = zk_dtypes.goldilocksx3
goldilocksx3_mont = zk_dtypes.goldilocksx3_mont
koalabearx4 = zk_dtypes.koalabearx4
koalabearx4_mont = zk_dtypes.koalabearx4_mont
mersenne31x2 = zk_dtypes.mersenne31x2

EXT_FIELD_TYPES = [
    babybearx4,
    babybearx4_mont,
    goldilocksx3,
    goldilocksx3_mont,
    koalabearx4,
    koalabearx4_mont,
    mersenne31x2,
]


def make_values(degree):
  """Create 4 random extension field elements as tuples."""
  return [tuple(random.sample(range(-100, 100), degree)) for _ in range(4)]


VALUES = {
    babybearx4: make_values(4),
    babybearx4_mont: make_values(4),
    goldilocksx3: make_values(3),
    goldilocksx3_mont: make_values(3),
    koalabearx4: make_values(4),
    koalabearx4_mont: make_values(4),
    mersenne31x2: make_values(2),
}


def make_array(scalar_type):
  """Create a numpy array from VALUES by first converting to scalars."""
  return np.array([scalar_type(v) for v in VALUES[scalar_type]])


@multi_threaded(num_workers=3)
class ExtensionFieldScalarTest(parameterized.TestCase):

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testModuleName(self, scalar_type):
    self.assertEqual(scalar_type.__module__, "zk_dtypes")

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromTuple(self, scalar_type):
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      self.assertIsInstance(x, scalar_type)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromInt(self, scalar_type):
    # Integer creates extension field with base field embedding
    x = scalar_type(42)
    self.assertIsInstance(x, scalar_type)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testWrongLengthRaisesError(self, scalar_type):
    for v in VALUES[scalar_type]:
      wrong_elem = v + (99,)  # one extra element
      with self.assertRaises(TypeError):
        scalar_type(wrong_elem)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testStr(self, scalar_type):
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      s = str(x)
      self.assertIsInstance(s, str)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testRepr(self, scalar_type):
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      r = repr(x)
      self.assertIsInstance(r, str)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testItem(self, scalar_type):
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      self.assertIsInstance(x.item(), scalar_type)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testIntConversionRaisesError(self, scalar_type):
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      with self.assertRaises(TypeError):
        int(x)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      op=[operator.eq, operator.ne],
  )
  def testComparison(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        result = op(scalar_type(v), scalar_type(w))
        self.assertIsInstance(result, np.bool_)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      op=[operator.lt, operator.le, operator.gt, operator.ge],
  )
  def testOrderingRaisesError(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        with self.assertRaises(TypeError):
          op(scalar_type(v), scalar_type(w))

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      op=[operator.neg],
  )
  def testUnop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      out = op(scalar_type(v))
      self.assertIsInstance(out, scalar_type)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      op=[operator.add, operator.sub, operator.mul],
  )
  def testBinop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        out = op(scalar_type(v), scalar_type(w))
        self.assertIsInstance(out, scalar_type)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testDivop(self, scalar_type):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        out = scalar_type(v) / scalar_type(w)
        self.assertIsInstance(out, scalar_type)
        # (v / w) * w == v
        self.assertEqual(out * scalar_type(w), scalar_type(v))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testDivByZeroRaisesError(self, scalar_type):
    for v in VALUES[scalar_type]:
      with self.assertRaises(ZeroDivisionError):
        _ = scalar_type(v) / scalar_type(0)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testPowerOp(self, scalar_type):
    for v in VALUES[scalar_type]:
      out = scalar_type(v) ** 5
      self.assertIsInstance(out, scalar_type)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testCastToDtype(self, scalar_type):
    name = scalar_type.__name__
    dt = np.dtype(scalar_type)
    self.assertIs(dt.type, scalar_type)
    self.assertEqual(dt.name, name)
    self.assertEqual(repr(dt), f"dtype({name})")


@multi_threaded(num_workers=3)
class ExtensionFieldArrayTest(parameterized.TestCase):

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testDtype(self, scalar_type):
    self.assertEqual(scalar_type, np.dtype(scalar_type))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testHash(self, scalar_type):
    h = hash(np.dtype(scalar_type))
    self.assertEqual(h, hash(np.dtype(scalar_type.dtype)))
    self.assertEqual(h, hash(np.dtype(scalar_type.__name__)))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testDeepCopyDoesNotAlterHash(self, scalar_type):
    dtype = np.dtype(scalar_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testArray(self, scalar_type):
    x = make_array(scalar_type)
    self.assertEqual(scalar_type, x.dtype)
    self.assertEqual(x.shape, (4,))
    self.assertTrue((x == x).all())  # pylint: disable=comparison-with-itself

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def test2DArray(self, scalar_type):
    row = make_array(scalar_type)
    x = np.array([row, row])
    self.assertEqual(scalar_type, x.dtype)
    self.assertEqual(x.shape, (2, 4))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testPickleable(self, scalar_type):
    x = make_array(scalar_type)
    serialized = pickle.dumps(x)
    x_out = pickle.loads(serialized)
    self.assertEqual(x_out.dtype, x.dtype)
    self.assertTrue((x_out == x).all())

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testIssubdtype(self, scalar_type):
    self.assertTrue(np.issubdtype(scalar_type, np.generic))
    self.assertTrue(np.issubdtype(np.dtype(scalar_type), np.generic))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testNonzeroUfunc(self, scalar_type):
    x = make_array(scalar_type)
    result = np.nonzero(x)
    self.assertIsInstance(result, tuple)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      ufunc=[np.equal, np.not_equal],
  )
  def testPredicateUfuncs(self, scalar_type, ufunc):
    x = make_array(scalar_type)
    y = make_array(scalar_type)
    result = ufunc(x, y)
    self.assertEqual(result.dtype, np.bool_)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      ufunc=[np.less, np.less_equal, np.greater, np.greater_equal],
  )
  def testOrderingUfuncsRaiseError(self, scalar_type, ufunc):
    x = make_array(scalar_type)
    y = make_array(scalar_type)
    with self.assertRaises(TypeError):
      ufunc(x, y)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      ufunc=[np.negative],
  )
  def testUnaryUfuncs(self, scalar_type, ufunc):
    x = make_array(scalar_type)
    result = ufunc(x)
    self.assertEqual(scalar_type, result.dtype)

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      ufunc=[np.add, np.subtract, np.multiply, np.divide],
  )
  def testBinaryUfuncs(self, scalar_type, ufunc):
    x = make_array(scalar_type)
    y = make_array(scalar_type)
    z = ufunc(x, y)
    self.assertEqual(scalar_type, z.dtype)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testPowerUfunc(self, scalar_type):
    x = make_array(scalar_type)
    z = x**5
    self.assertEqual(scalar_type, z.dtype)


# Tests for raw/from_raw Montgomery conversion helpers for extension fields
@multi_threaded(num_workers=3)
class ExtensionFieldRawConversionTest(parameterized.TestCase):

  # Standard extension field types (non-Montgomery)
  STD_EXT_FIELD_TYPES = [
      babybearx4,
      koalabearx4,
      goldilocksx3,
      mersenne31x2,
  ]

  # Montgomery extension field types
  MONT_EXT_FIELD_TYPES = [
      babybearx4_mont,
      koalabearx4_mont,
      goldilocksx3_mont,
  ]

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testRawPropertyExists(self, scalar_type):
    """Test that raw property is accessible and returns a tuple."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      self.assertIsInstance(raw, tuple)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testRawPropertyLength(self, scalar_type):
    """Test that raw property returns tuple of correct length."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      expected_len = len(v)  # Extension degree
      self.assertEqual(len(raw), expected_len)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawExists(self, scalar_type):
    """Test that from_raw classmethod is accessible."""
    self.assertTrue(hasattr(scalar_type, "from_raw"))
    self.assertTrue(callable(scalar_type.from_raw))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testRawFromRawRoundTrip(self, scalar_type):
    """Test that from_raw(x.raw) == x for extension fields."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      y = scalar_type.from_raw(raw)
      self.assertEqual(x, y, msg=f"Round trip failed for {v}")

  @parameterized.product(scalar_type=STD_EXT_FIELD_TYPES)
  def testRawTupleContainsIntsForStandardTypes(self, scalar_type):
    """For standard types, raw tuple should contain integer values."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      for r in raw:
        self.assertIsInstance(r, int)

  @parameterized.product(scalar_type=MONT_EXT_FIELD_TYPES)
  def testRawTupleContainsIntsForMontgomeryTypes(self, scalar_type):
    """For Montgomery types, raw tuple should contain integer values."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      for r in raw:
        self.assertIsInstance(r, int)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawPreservesValue(self, scalar_type):
    """Test that from_raw stores the value directly."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      y = scalar_type.from_raw(raw)
      self.assertEqual(x, y)
      self.assertEqual(x.raw, y.raw)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawWithZeroTuple(self, scalar_type):
    """Test from_raw with zero tuple."""
    # Get the degree from VALUES
    degree = len(VALUES[scalar_type][0])
    zero_tuple = tuple(0 for _ in range(degree))
    x = scalar_type.from_raw(zero_tuple)
    self.assertEqual(x, scalar_type(0))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawWrongTupleLengthRaisesError(self, scalar_type):
    """Test that from_raw raises error with wrong tuple length."""
    degree = len(VALUES[scalar_type][0])
    # Use wrong length (degree + 1)
    wrong_tuple = tuple(0 for _ in range(degree + 1))
    with self.assertRaises(TypeError):
      scalar_type.from_raw(wrong_tuple)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawNegativeValueRaisesError(self, scalar_type):
    """Test that from_raw raises error with negative raw value."""
    degree = len(VALUES[scalar_type][0])
    # Tuple with one negative value
    negative_tuple = tuple(-1 if i == 0 else 0 for i in range(degree))
    with self.assertRaises(ValueError):
      scalar_type.from_raw(negative_tuple)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawWrongTypeRaisesError(self, scalar_type):
    """Test that from_raw raises error with wrong element type."""
    # String instead of int/tuple
    with self.assertRaises(TypeError):
      scalar_type.from_raw("invalid")
    # List instead of tuple (should still work - converted to tuple internally)
    # Float instead of int
    degree = len(VALUES[scalar_type][0])
    float_tuple = tuple(1.5 if i == 0 else 0 for i in range(degree))
    with self.assertRaises(TypeError):
      scalar_type.from_raw(float_tuple)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawSingleValueTreatedAsFirstCoefficient(self, scalar_type):
    """Test that from_raw with single value sets first coefficient only."""
    # When passing a single int, it should be treated as the first coefficient
    x = scalar_type.from_raw(42)
    # Create the expected value: (42, 0, 0, ...) tuple
    degree = len(VALUES[scalar_type][0])
    expected_tuple = tuple(42 if i == 0 else 0 for i in range(degree))
    y = scalar_type.from_raw(expected_tuple)
    self.assertEqual(x, y)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawSingleValueZero(self, scalar_type):
    """Test that from_raw(0) creates zero element."""
    x = scalar_type.from_raw(0)
    self.assertEqual(x, scalar_type(0))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawNoArgumentsRaisesError(self, scalar_type):
    """Test that from_raw() with no arguments raises error."""
    with self.assertRaises(TypeError):
      scalar_type.from_raw()

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testFromRawTooManyArgumentsRaisesError(self, scalar_type):
    """Test that from_raw() with multiple arguments raises error."""
    with self.assertRaises(TypeError):
      scalar_type.from_raw((0, 0), (1, 1))


@multi_threaded(num_workers=3)
class ExtensionFieldIntegerCastTest(parameterized.TestCase):

  INT_DTYPES = [
      np.bool_,
      np.int8,
      np.int16,
      np.int32,
      np.int64,
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
  ]

  @parameterized.product(
      scalar_type=EXT_FIELD_TYPES,
      int_dtype=INT_DTYPES,
  )
  def testIntegerToExtFieldCast(self, scalar_type, int_dtype):
    """Integer arrays can be cast to extension field arrays."""
    arr = np.array([1, 2, 3], dtype=int_dtype)
    result = arr.astype(scalar_type)
    self.assertEqual(result.dtype, scalar_type)
    # Each element should equal the scalar constructor from the casted value
    for i in range(len(arr)):
      self.assertEqual(result[i], scalar_type(int(arr[i])))

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testExtFieldToIntegerCastRaisesError(self, scalar_type):
    """Extension field arrays cannot be cast to integer arrays."""
    arr = make_array(scalar_type)
    with self.assertRaises(TypeError):
      arr.astype(np.int64)


class EfinfoTest(parameterized.TestCase):

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testDegreeOverPrimeEqualsDegreeForDirectExtensions(self, scalar_type):
    """All current types are direct extensions, so degree_over_prime == degree."""
    info = efinfo(scalar_type)
    self.assertEqual(info.degree_over_prime, info.degree)

  @parameterized.product(scalar_type=EXT_FIELD_TYPES)
  def testStorageBitsEqualsDegreeTimesPrimeFieldBits(self, scalar_type):
    """storage_bits should be degree_over_prime * base prime field element bits."""
    info = efinfo(scalar_type)
    from zk_dtypes import pfinfo

    base_dtype = info.base_field_dtype
    while True:
      try:
        base_info = efinfo(base_dtype)
        base_dtype = base_info.base_field_dtype
      except ValueError:
        break
    prime_bits = pfinfo(base_dtype).storage_bits
    self.assertEqual(info.storage_bits, info.degree_over_prime * prime_bits)


if __name__ == "__main__":
  absltest.main()
