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

"""Test cases for BigInt numpy dtype wrappers (int128/uint128/int256/uint256)."""

import copy
import operator
import pickle

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes
from multi_thread_utils import multi_threaded
import numpy as np

int128 = zk_dtypes.int128
uint128 = zk_dtypes.uint128
int256 = zk_dtypes.int256
uint256 = zk_dtypes.uint256

BIGINT_TYPES = [int128, uint128, int256, uint256]

SIGNED_TYPES = [int128, int256]
UNSIGNED_TYPES = [uint128, uint256]

# BN254 base field modulus — a real-world 256-bit value.
BN254_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Small test values that fit in all types.
SMALL_VALUES = [0, 1, 2, 42, 127, 255, 1000, 2**16, 2**32 - 1, 2**63]

# Per-type test values including edge cases.
VALUES = {
    int128: [0, 1, -1, 42, -42, 2**63, -(2**63), 2**127 - 1, -(2**127)],
    uint128: [0, 1, 42, 255, 2**64, 2**127, 2**128 - 1],
    int256: [0, 1, -1, 42, -42, 2**127, -(2**127), 2**255 - 1, -(2**255)],
    uint256: [0, 1, 42, 2**64, 2**128, 2**200, BN254_P, 2**256 - 1],
}


@multi_threaded(num_workers=3)
class ScalarTest(parameterized.TestCase):

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testModuleName(self, scalar_type):
    self.assertEqual(scalar_type.__module__, "zk_dtypes")

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testRoundTripToPythonInt(self, scalar_type):
    for v in VALUES[scalar_type]:
      scalar = scalar_type(v)
      self.assertEqual(v, int(scalar), msg=f"v={v}")

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testStr(self, scalar_type):
    for v in VALUES[scalar_type]:
      self.assertEqual(str(v), str(scalar_type(v)), msg=f"v={v}")

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testRepr(self, scalar_type):
    for v in VALUES[scalar_type]:
      self.assertEqual(str(v), repr(scalar_type(v)), msg=f"v={v}")

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testHash(self, scalar_type):
    for v in VALUES[scalar_type]:
      # Just check it doesn't crash and is consistent.
      h1 = hash(scalar_type(v))
      h2 = hash(scalar_type(v))
      self.assertEqual(h1, h2, msg=f"v={v}")

  @parameterized.product(
      scalar_type=BIGINT_TYPES,
      op=[
          operator.eq,
          operator.ne,
          operator.lt,
          operator.le,
          operator.gt,
          operator.ge,
      ],
  )
  def testComparison(self, scalar_type, op):
    vals = VALUES[scalar_type][:4]  # Use a subset for O(n²) test.
    for v in vals:
      for w in vals:
        result = op(scalar_type(v), scalar_type(w))
        self.assertEqual(
            op(v, w), result, msg=f"op={op.__name__}, v={v}, w={w}"
        )

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testNeg(self, scalar_type):
    for v in VALUES[scalar_type]:
      out = -scalar_type(v)
      self.assertIsInstance(out, scalar_type)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testAdd(self, scalar_type):
    a = scalar_type(100)
    b = scalar_type(42)
    self.assertEqual(int(a + b), 142)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testSub(self, scalar_type):
    a = scalar_type(100)
    b = scalar_type(42)
    self.assertEqual(int(a - b), 58)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testMul(self, scalar_type):
    a = scalar_type(100)
    b = scalar_type(42)
    self.assertEqual(int(a * b), 4200)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testFloorDiv(self, scalar_type):
    a = scalar_type(100)
    b = scalar_type(42)
    self.assertEqual(int(a // b), 2)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testMod(self, scalar_type):
    a = scalar_type(100)
    b = scalar_type(42)
    self.assertEqual(int(a % b), 16)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testDivByZero(self, scalar_type):
    a = scalar_type(42)
    b = scalar_type(0)
    with self.assertRaises(ZeroDivisionError):
      _ = a // b
    with self.assertRaises(ZeroDivisionError):
      _ = a % b

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testBitwiseAnd(self, scalar_type):
    a = scalar_type(0xFF)
    b = scalar_type(0x0F)
    self.assertEqual(int(a & b), 0x0F)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testBitwiseOr(self, scalar_type):
    a = scalar_type(0xF0)
    b = scalar_type(0x0F)
    self.assertEqual(int(a | b), 0xFF)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testBitwiseXor(self, scalar_type):
    a = scalar_type(0xFF)
    b = scalar_type(0x0F)
    self.assertEqual(int(a ^ b), 0xF0)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testLeftShift(self, scalar_type):
    a = scalar_type(1)
    self.assertEqual(int(a << 10), 1024)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testRightShift(self, scalar_type):
    a = scalar_type(1024)
    self.assertEqual(int(a >> 10), 1)

  def testUint256Overflow(self):
    with self.assertRaises(OverflowError):
      uint256(2**256)

  def testUint256NegativeOverflow(self):
    with self.assertRaises(OverflowError):
      uint256(-1)

  def testInt256Overflow(self):
    with self.assertRaises(OverflowError):
      int256(2**255)

  def testUint128Overflow(self):
    with self.assertRaises(OverflowError):
      uint128(2**128)

  def testInt128Overflow(self):
    with self.assertRaises(OverflowError):
      int128(2**127)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testConstructFromString(self, scalar_type):
    s = scalar_type("42")
    self.assertEqual(int(s), 42)

  def testBN254ModulusRoundtrip(self):
    p = uint256(BN254_P)
    self.assertEqual(int(p), BN254_P)

  def testLargeValueRoundtrip(self):
    val = 2**200 + 12345
    self.assertEqual(int(uint256(val)), val)
    self.assertEqual(int(int256(val)), val)

  def testInt128NegativeRoundtrip(self):
    for v in [-(2**127), -(2**100), -1, -42]:
      self.assertEqual(int(int128(v)), v)

  def testInt256NegativeRoundtrip(self):
    for v in [-(2**255), -(2**200), -1, -42]:
      self.assertEqual(int(int256(v)), v)


@multi_threaded(num_workers=3)
class ArrayTest(parameterized.TestCase):

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testDtype(self, scalar_type):
    self.assertEqual(scalar_type, np.dtype(scalar_type))

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testDtypeSize(self, scalar_type):
    dt = np.dtype(scalar_type)
    if scalar_type in (int128, uint128):
      self.assertEqual(dt.itemsize, 16)
    else:
      self.assertEqual(dt.itemsize, 32)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArrayCreation(self, scalar_type):
    x = np.array([1, 2, 3], dtype=scalar_type)
    self.assertEqual(scalar_type, x.dtype.type)
    self.assertEqual(x.shape, (3,))

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArrayGetItem(self, scalar_type):
    x = np.array([10, 20, 30], dtype=scalar_type)
    self.assertEqual(int(x[0]), 10)
    self.assertEqual(int(x[1]), 20)
    self.assertEqual(int(x[2]), 30)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArraySetItem(self, scalar_type):
    x = np.zeros(3, dtype=scalar_type)
    x[1] = 42
    self.assertEqual(int(x[1]), 42)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArrayNonZero(self, scalar_type):
    x = np.array([0, 1, 0, 42], dtype=scalar_type)
    (indices,) = np.nonzero(x)
    np.testing.assert_array_equal(indices, [1, 3])

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArrayEquality(self, scalar_type):
    x = np.array([1, 2, 3], dtype=scalar_type)
    y = np.array([1, 2, 3], dtype=scalar_type)
    for i in range(len(x)):
      self.assertEqual(int(x[i]), int(y[i]))

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArrayCopySwap(self, scalar_type):
    x = np.array([1, 2, 3], dtype=scalar_type)
    y = x.copy()
    np.testing.assert_array_equal(
        np.array([int(v) for v in x]),
        np.array([int(v) for v in y]),
    )

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testArrayRoundtripViaInt64(self, scalar_type):
    vals = [0, 1, 42, 255, 2**16]
    x = np.array(vals, dtype=scalar_type)
    y = x.astype(np.int64)
    np.testing.assert_array_equal(y, vals)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testCastFromInt64(self, scalar_type):
    vals = [0, 1, 42, 255, 2**16]
    x = np.array(vals, dtype=np.int64)
    y = x.astype(scalar_type)
    for i, v in enumerate(vals):
      self.assertEqual(int(y[i]), v)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testPickle(self, scalar_type):
    x = np.array([1, 42, 100], dtype=scalar_type)
    data = pickle.dumps(x)
    y = pickle.loads(data)
    self.assertEqual(x.dtype, y.dtype)
    for i in range(len(x)):
      self.assertEqual(int(x[i]), int(y[i]))

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testDeepCopyDoesNotAlterHash(self, scalar_type):
    dtype = np.dtype(scalar_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  def testUint256LargeArrayRoundtrip(self):
    """Test that large 256-bit values survive array round-trip."""
    vals = [0, 2**64, 2**128, 2**200, BN254_P, 2**256 - 1]
    x = np.array(vals, dtype=uint256)
    for i, v in enumerate(vals):
      self.assertEqual(int(x[i]), v, msg=f"index={i}, expected={v}")

  def testInt256NegativeArray(self):
    vals = [-1, -42, -(2**127), 0, 1, 2**127]
    x = np.array(vals, dtype=int256)
    for i, v in enumerate(vals):
      self.assertEqual(int(x[i]), v, msg=f"index={i}, expected={v}")

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testUfuncAdd(self, scalar_type):
    x = np.array([10, 20, 30], dtype=scalar_type)
    y = np.array([1, 2, 3], dtype=scalar_type)
    z = np.add(x, y)
    self.assertEqual(z.dtype.type, scalar_type)
    for i, expected in enumerate([11, 22, 33]):
      self.assertEqual(int(z[i]), expected)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testUfuncSubtract(self, scalar_type):
    x = np.array([10, 20, 30], dtype=scalar_type)
    y = np.array([1, 2, 3], dtype=scalar_type)
    z = np.subtract(x, y)
    self.assertEqual(z.dtype.type, scalar_type)
    for i, expected in enumerate([9, 18, 27]):
      self.assertEqual(int(z[i]), expected)

  @parameterized.product(scalar_type=BIGINT_TYPES)
  def testUfuncMultiply(self, scalar_type):
    x = np.array([10, 20, 30], dtype=scalar_type)
    y = np.array([1, 2, 3], dtype=scalar_type)
    z = np.multiply(x, y)
    self.assertEqual(z.dtype.type, scalar_type)
    for i, expected in enumerate([10, 40, 90]):
      self.assertEqual(int(z[i]), expected)


@multi_threaded(num_workers=3)
class IInfoTest(parameterized.TestCase):

  def testIinfoInt128(self):
    info = zk_dtypes.iinfo(int128)
    self.assertEqual(info.bits, 128)
    self.assertEqual(info.kind, "i")
    self.assertEqual(info.min, -(2**127))
    self.assertEqual(info.max, 2**127 - 1)

  def testIinfoUint128(self):
    info = zk_dtypes.iinfo(uint128)
    self.assertEqual(info.bits, 128)
    self.assertEqual(info.kind, "u")
    self.assertEqual(info.min, 0)
    self.assertEqual(info.max, 2**128 - 1)

  def testIinfoInt256(self):
    info = zk_dtypes.iinfo(int256)
    self.assertEqual(info.bits, 256)
    self.assertEqual(info.kind, "i")
    self.assertEqual(info.min, -(2**255))
    self.assertEqual(info.max, 2**255 - 1)

  def testIinfoUint256(self):
    info = zk_dtypes.iinfo(uint256)
    self.assertEqual(info.bits, 256)
    self.assertEqual(info.kind, "u")
    self.assertEqual(info.min, 0)
    self.assertEqual(info.max, 2**256 - 1)


if __name__ == "__main__":
  absltest.main()
