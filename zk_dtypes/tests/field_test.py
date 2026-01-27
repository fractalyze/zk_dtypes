# Copyright 2022 The ml_dtypes Authors.
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

"""Test cases for field types."""

# pylint: disable=g-complex-comprehension

import contextlib
import copy
import operator
import pickle
import random
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes
from zk_dtypes._pfinfo import pfinfo
from multi_thread_utils import multi_threaded
import numpy as np

babybear = zk_dtypes.babybear
babybear_std = zk_dtypes.babybear_std
goldilocks = zk_dtypes.goldilocks
goldilocks_std = zk_dtypes.goldilocks_std
koalabear = zk_dtypes.koalabear
koalabear_std = zk_dtypes.koalabear_std
mersenne31 = zk_dtypes.mersenne31
bn254_sf = zk_dtypes.bn254_sf
bn254_sf_std = zk_dtypes.bn254_sf_std

FIELD_TYPES = [
    babybear,
    babybear_std,
    goldilocks,
    goldilocks_std,
    koalabear,
    koalabear_std,
    mersenne31,
    bn254_sf,
    bn254_sf_std,
]

# Expected 2-adicity for each field type
TWO_ADICITY = {
    babybear: 27,
    babybear_std: 27,
    goldilocks: 32,
    goldilocks_std: 32,
    koalabear: 24,
    koalabear_std: 24,
    mersenne31: 1,
    bn254_sf: 28,
    bn254_sf_std: 28,
}

BABYBEAR_MODULUS = 2**31 - 2**27 + 1
GOLDILOCKS_MODULUS = 2**64 - 2**32 + 1
KOALABEAR_MODULUS = 2**31 - 2**24 + 1
MERSENNE31_MODULUS = 2**31 - 1
BN254_SF_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617

VALUES = {
    babybear: random.sample(range(-100, 100), 4),
    babybear_std: random.sample(range(-100, 100), 4),
    goldilocks: random.sample(range(-100, 100), 4),
    goldilocks_std: random.sample(range(-100, 100), 4),
    koalabear: random.sample(range(-100, 100), 4),
    koalabear_std: random.sample(range(-100, 100), 4),
    mersenne31: random.sample(range(-100, 100), 4),
    bn254_sf: random.sample(range(-100, 100), 4),
    bn254_sf_std: random.sample(range(-100, 100), 4),
}


@contextlib.contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


# Normalize [-P, P] to [0, P)
def normalize(scalar_type, value):
  if value < 0:
    value += pfinfo(scalar_type).modulus
  return value


# Tests for the Python scalar type
@multi_threaded(num_workers=3)
class ScalarTest(parameterized.TestCase):

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testModuleName(self, scalar_type):
    self.assertEqual(scalar_type.__module__, "zk_dtypes")

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testTwoAdicityKnownValues(self, scalar_type):
    """two_adicity from pfinfo matches known values for supported fields."""
    self.assertEqual(
        pfinfo(scalar_type).two_adicity,
        TWO_ADICITY[scalar_type],
    )

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testPickleable(self, scalar_type):
    # https://github.com/jax-ml/jax/discussions/8505
    x = np.arange(10, dtype=scalar_type)
    serialized = pickle.dumps(x)
    x_out = pickle.loads(serialized)
    self.assertEqual(x_out.dtype, x.dtype)
    np.testing.assert_array_equal(x_out.astype(int), x.astype(int))

  @parameterized.product(
      scalar_type=FIELD_TYPES,
      python_scalar=[int],
  )
  def testRoundTripToPythonScalar(self, scalar_type, python_scalar):
    for v in VALUES[scalar_type]:
      self.assertEqual(
          python_scalar(normalize(scalar_type, v)),
          python_scalar(scalar_type(v)),
      )
      self.assertEqual(
          scalar_type(v), scalar_type(python_scalar(scalar_type(v)))
      )

  @parameterized.product(scalar_type=[babybear])
  def testRoundTripNumpyTypes(self, scalar_type):
    for dtype in [np.uint64]:
      for f in VALUES[scalar_type]:
        f = normalize(scalar_type, f)
        self.assertEqual(dtype(f), dtype(scalar_type(dtype(f))))
        self.assertEqual(int(dtype(f)), int(scalar_type(dtype(f))))
        self.assertEqual(dtype(f), dtype(scalar_type(np.array(f, dtype))))

      np.testing.assert_equal(
          dtype(np.array(VALUES[scalar_type], scalar_type)),
          np.array(
              [normalize(scalar_type, v) for v in VALUES[scalar_type]], dtype
          ),
      )

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testStr(self, scalar_type):
    for value in VALUES[scalar_type]:
      self.assertEqual(
          str(normalize(scalar_type, value)), str(scalar_type(value))
      )

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testRepr(self, scalar_type):
    for value in VALUES[scalar_type]:
      self.assertEqual(
          str(normalize(scalar_type, value)), str(scalar_type(value))
      )

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testItem(self, scalar_type):
    self.assertIsInstance(scalar_type(1).item(), scalar_type)
    self.assertEqual(scalar_type(1).item(), scalar_type(1))

  @parameterized.product(
      scalar_type=FIELD_TYPES,
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
        self.assertEqual(
            op(normalize(scalar_type, v), normalize(scalar_type, w)), result
        )
        self.assertIsInstance(result, np.bool_)

  @parameterized.product(
      scalar_type=FIELD_TYPES,
      op=[
          operator.neg,
      ],
  )
  def testUnop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      out = op(scalar_type(v))
      self.assertIsInstance(out, scalar_type)
      self.assertEqual(scalar_type(op(v)), out, msg=v)

  @parameterized.product(
      scalar_type=FIELD_TYPES,
      op=[
          operator.add,
          operator.sub,
          operator.mul,
      ],
  )
  def testBinop(self, scalar_type, op):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        out = op(scalar_type(v), scalar_type(w))
        self.assertIsInstance(out, scalar_type)
        # NOTE(chokobole): The sampled values are small, so it's safe to
        # create a scalar from the result of the operation
        self.assertEqual(scalar_type(op(v, w)), out, msg=(v, w))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testDivop(self, scalar_type):
    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        if w == 0:
          with self.assertRaises(ZeroDivisionError):
            scalar_type(v) / scalar_type(w)
        else:
          out = scalar_type(v) / scalar_type(w)
          self.assertIsInstance(out, scalar_type)
          self.assertEqual(out * scalar_type(w), scalar_type(v), msg=(v, w))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testPowerOp(self, scalar_type):
    if pfinfo(scalar_type).modulus > 2**64:
      self.skipTest("Modulus is too large to support")

    for v in VALUES[scalar_type]:
      for w in VALUES[scalar_type]:
        if v == 0 and w < 0:
          with self.assertRaises(ZeroDivisionError):
            scalar_type(v) ** w
        else:
          out = scalar_type(v) ** w
          self.assertIsInstance(out, scalar_type)
          self.assertEqual(
              scalar_type(pow(v, w, pfinfo(scalar_type).modulus)),
              out,
              msg=(v, w),
          )

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

  @parameterized.product(a=[babybear], b=CAST_DTYPES + [babybear])
  def test32BitCanCast(self, a, b):
    allowed_casts = [
        (babybear, babybear),
        (babybear, np.uint32),
        (babybear, np.uint64),
    ]
    self.assertEqual(
        ((a, b) in allowed_casts), np.can_cast(a, b, casting="safe")
    )

  @parameterized.product(a=[goldilocks], b=CAST_DTYPES + [goldilocks])
  def test64BitCanCast(self, a, b):
    allowed_casts = [
        (goldilocks, goldilocks),
        (goldilocks, np.uint64),
    ]
    self.assertEqual(
        ((a, b) in allowed_casts), np.can_cast(a, b, casting="safe")
    )

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testIssubdtype(self, scalar_type):
    # In the future, we may want to make these more specific (e.g. use
    # np.number or np.integer instead of np.generic) by changing the
    # base in RegisterIntNDtype.
    self.assertTrue(np.issubdtype(scalar_type, np.generic))
    self.assertTrue(np.issubdtype(np.dtype(scalar_type), np.generic))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testCastToDtype(self, scalar_type):
    name = scalar_type.__name__
    dt = np.dtype(scalar_type)
    self.assertIs(dt.type, scalar_type)
    self.assertEqual(dt.name, name)
    self.assertEqual(repr(dt), f"dtype({name})")


# Tests for the Python scalar type
@multi_threaded(num_workers=3)
class ArrayTest(parameterized.TestCase):

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testDtype(self, scalar_type):
    self.assertEqual(scalar_type, np.dtype(scalar_type))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testHash(self, scalar_type):
    h = hash(np.dtype(scalar_type))
    self.assertEqual(h, hash(np.dtype(scalar_type.dtype)))
    self.assertEqual(h, hash(np.dtype(scalar_type.__name__)))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testDeepCopyDoesNotAlterHash(self, scalar_type):
    # For context, see https://github.com/jax-ml/jax/issues/4651. If the hash
    # value of the type descriptor is not initialized correctly, a deep copy
    # can change the type hash.
    dtype = np.dtype(scalar_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testArray(self, scalar_type):
    x = np.array([[1, 2, 3]], dtype=scalar_type)
    self.assertEqual("[[1 2 3]]", str(x))
    self.assertEqual(scalar_type, x.dtype)
    self.assertTrue((x == x).all())  # pylint: disable=comparison-with-itself

  @parameterized.product(
      scalar_type=FIELD_TYPES,
      ufunc=[np.nonzero, np.argmax, np.argmin],
  )
  def testUnaryPredicateUfunc(self, scalar_type, ufunc):
    if pfinfo(scalar_type).modulus > 2**64:
      self.skipTest("Modulus is too large to support")

    x = np.array(
        [normalize(scalar_type, x) for x in VALUES[scalar_type]],
        dtype=np.uint64,
    )
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    # Compute `ufunc(y)` first so we don't get lucky by reusing memory
    # initialized by `ufunc(x)`.
    y_result = ufunc(y)
    x_result = ufunc(x)
    np.testing.assert_array_equal(x_result, y_result)

  @parameterized.product(
      scalar_type=FIELD_TYPES,
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
    if pfinfo(scalar_type).modulus > 2**64:
      self.skipTest("Modulus is too large to support")

    x = np.array(
        [normalize(scalar_type, x) for x in VALUES[scalar_type]],
        dtype=np.uint64,
    )
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    np.testing.assert_array_equal(
        ufunc(x[:, None], x[None, :]),
        ufunc(y[:, None], y[None, :]),
    )

  @parameterized.product(
      scalar_type=FIELD_TYPES,
      ufunc=[
          np.negative,
      ],
  )
  def testUnaryUfuncs(self, scalar_type, ufunc):
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = ufunc(y)
    for i in range(len(y_result)):
      x = scalar_type(VALUES[scalar_type][i])
      self.assertEqual(ufunc(x), y_result[i])

  @parameterized.product(
      scalar_type=FIELD_TYPES,
      ufunc=[
          np.add,
          np.subtract,
          np.multiply,
          np.divide,
      ],
  )
  @ignore_warning(category=RuntimeWarning, message="divide by zero encountered")
  def testBinaryUfuncs(self, scalar_type, ufunc):
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = ufunc(y, y)
    for i in range(len(y_result)):
      x = scalar_type(VALUES[scalar_type][i])
      self.assertEqual(ufunc(x, x), y_result[i])

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testPowerUfunc(self, scalar_type):
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    y_result = y**5
    for i in range(len(y_result)):
      x = scalar_type(VALUES[scalar_type][i])
      self.assertEqual(x**5, y_result[i])


# Tests for raw/from_raw Montgomery conversion helpers
@multi_threaded(num_workers=3)
class RawConversionTest(parameterized.TestCase):

  # Montgomery types (without _std suffix, uses Montgomery multiplication)
  MONT_FIELD_TYPES = [
      babybear,
      goldilocks,
      koalabear,
      mersenne31,
      bn254_sf,
  ]

  # Standard types (non-Montgomery, with _std suffix)
  STD_FIELD_TYPES = [
      babybear_std,
      goldilocks_std,
      koalabear_std,
      mersenne31_std,
      bn254_sf_std,
  ]

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testRawPropertyExists(self, scalar_type):
    """Test that raw property is accessible."""
    x = scalar_type(42)
    raw = x.raw
    self.assertIsInstance(raw, int)

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawExists(self, scalar_type):
    """Test that from_raw classmethod is accessible."""
    self.assertTrue(hasattr(scalar_type, "from_raw"))
    self.assertTrue(callable(scalar_type.from_raw))

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testRawFromRawRoundTrip(self, scalar_type):
    """Test that from_raw(x.raw) == x for all field types."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      raw = x.raw
      y = scalar_type.from_raw(raw)
      self.assertEqual(x, y, msg=f"Round trip failed for {v}")

  @parameterized.product(scalar_type=STD_FIELD_TYPES)
  def testRawEqualsIntForStandardTypes(self, scalar_type):
    """For non-Montgomery types, raw should equal int()."""
    for v in VALUES[scalar_type]:
      x = scalar_type(v)
      self.assertEqual(x.raw, int(x))

  @parameterized.product(scalar_type=MONT_FIELD_TYPES)
  def testRawDiffersFromIntForMontgomeryTypes(self, scalar_type):
    """For Montgomery types, raw should differ from int() (except 0)."""
    # Use a non-trivial value that's definitely not 0 or a special case
    x = scalar_type(42)
    raw = x.raw
    int_val = int(x)
    # For Montgomery representation: raw = value * R, int = value
    # They should differ unless R = 1 (which is not the case for Montgomery)
    self.assertNotEqual(
        raw, int_val, msg="raw should differ from int() for Montgomery types"
    )

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawPreservesValue(self, scalar_type):
    """Test that from_raw stores the value directly without transformation."""
    # Create a field element
    x = scalar_type(7)
    raw = x.raw

    # Create another field element from the same raw value
    y = scalar_type.from_raw(raw)

    # They should be equal
    self.assertEqual(x, y)

    # The raw values should also be identical
    self.assertEqual(x.raw, y.raw)

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawNegativeValueRaisesError(self, scalar_type):
    """Test that from_raw raises error with negative raw value."""
    with self.assertRaises(ValueError):
      scalar_type.from_raw(-1)

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawWrongTypeRaisesError(self, scalar_type):
    """Test that from_raw raises error with wrong type."""
    with self.assertRaises(TypeError):
      scalar_type.from_raw("invalid")
    with self.assertRaises(TypeError):
      scalar_type.from_raw(1.5)
    with self.assertRaises(TypeError):
      scalar_type.from_raw([1, 2, 3])

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawZero(self, scalar_type):
    """Test that from_raw(0) creates zero element."""
    x = scalar_type.from_raw(0)
    self.assertEqual(x, scalar_type(0))
    self.assertEqual(x.raw, 0)

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawNoArgumentsRaisesError(self, scalar_type):
    """Test that from_raw() with no arguments raises error."""
    with self.assertRaises(TypeError):
      scalar_type.from_raw()

  @parameterized.product(scalar_type=FIELD_TYPES)
  def testFromRawTooManyArgumentsRaisesError(self, scalar_type):
    """Test that from_raw() with multiple arguments raises error."""
    with self.assertRaises(TypeError):
      scalar_type.from_raw(1, 2)


if __name__ == "__main__":
  absltest.main()
