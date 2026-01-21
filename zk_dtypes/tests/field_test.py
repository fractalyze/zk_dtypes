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
mersenne31_std = zk_dtypes.mersenne31_std
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
    mersenne31_std,
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
    mersenne31_std: 1,
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
    mersenne31_std: random.sample(range(-100, 100), 4),
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


if __name__ == "__main__":
  absltest.main()
