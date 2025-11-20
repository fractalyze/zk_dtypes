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

"""Test cases for elliptic curve point types."""

# pylint: disable=g-complex-comprehension

import copy

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes
from multi_thread_utils import multi_threaded
import numpy as np

bn254_sf = zk_dtypes.bn254_sf
bn254_sf_std = zk_dtypes.bn254_sf_std
bn254_g1_affine = zk_dtypes.bn254_g1_affine
bn254_g1_affine_std = zk_dtypes.bn254_g1_affine_std
bn254_g1_jacobian = zk_dtypes.bn254_g1_jacobian
bn254_g1_jacobian_std = zk_dtypes.bn254_g1_jacobian_std
bn254_g1_xyzz = zk_dtypes.bn254_g1_xyzz
bn254_g1_xyzz_std = zk_dtypes.bn254_g1_xyzz_std
bn254_g2_affine = zk_dtypes.bn254_g2_affine
bn254_g2_affine_std = zk_dtypes.bn254_g2_affine_std
bn254_g2_jacobian = zk_dtypes.bn254_g2_jacobian
bn254_g2_jacobian_std = zk_dtypes.bn254_g2_jacobian_std
bn254_g2_xyzz = zk_dtypes.bn254_g2_xyzz
bn254_g2_xyzz_std = zk_dtypes.bn254_g2_xyzz_std

EC_MONT_POINT_TYPES = [
    bn254_g1_affine,
    bn254_g1_jacobian,
    bn254_g1_xyzz,
    bn254_g2_affine,
    bn254_g2_jacobian,
    bn254_g2_xyzz,
]

EC_STD_POINT_TYPES = [
    bn254_g1_affine_std,
    bn254_g1_jacobian_std,
    bn254_g1_xyzz_std,
    bn254_g2_affine_std,
    bn254_g2_jacobian_std,
    bn254_g2_xyzz_std,
]

EC_POINT_TYPES = EC_MONT_POINT_TYPES + EC_STD_POINT_TYPES

VALUES = {
    bn254_g1_affine: [bn254_g1_affine(3), bn254_g1_affine(4)],
    bn254_g1_affine_std: [bn254_g1_affine_std(3), bn254_g1_affine_std(4)],
    bn254_g1_jacobian: [bn254_g1_jacobian(3), bn254_g1_jacobian(4)],
    bn254_g1_jacobian_std: [bn254_g1_jacobian_std(3), bn254_g1_jacobian_std(4)],
    bn254_g1_xyzz: [bn254_g1_xyzz(3), bn254_g1_xyzz(4)],
    bn254_g1_xyzz_std: [bn254_g1_xyzz_std(3), bn254_g1_xyzz_std(4)],
    bn254_g2_affine: [bn254_g2_affine(3), bn254_g2_affine(4)],
    bn254_g2_affine_std: [bn254_g2_affine_std(3), bn254_g2_affine_std(4)],
    bn254_g2_jacobian: [bn254_g2_jacobian(3), bn254_g2_jacobian(4)],
    bn254_g2_jacobian_std: [bn254_g2_jacobian_std(3), bn254_g2_jacobian_std(4)],
    bn254_g2_xyzz: [bn254_g2_xyzz(3), bn254_g2_xyzz(4)],
    bn254_g2_xyzz_std: [bn254_g2_xyzz_std(3), bn254_g2_xyzz_std(4)],
}

ADD_OP_RESULT_TYPES = {
    bn254_g1_affine: bn254_g1_jacobian,
    bn254_g1_affine_std: bn254_g1_jacobian_std,
    bn254_g1_jacobian: bn254_g1_jacobian,
    bn254_g1_jacobian_std: bn254_g1_jacobian_std,
    bn254_g1_xyzz: bn254_g1_xyzz,
    bn254_g1_xyzz_std: bn254_g1_xyzz_std,
    bn254_g2_affine: bn254_g2_jacobian,
    bn254_g2_affine_std: bn254_g2_jacobian_std,
    bn254_g2_jacobian: bn254_g2_jacobian,
    bn254_g2_jacobian_std: bn254_g2_jacobian_std,
    bn254_g2_xyzz: bn254_g2_xyzz,
    bn254_g2_xyzz_std: bn254_g2_xyzz_std,
}

SCALAR_FIELD_TYPES = {
    bn254_g1_affine: bn254_sf,
    bn254_g1_affine_std: bn254_sf_std,
    bn254_g1_jacobian: bn254_sf,
    bn254_g1_jacobian_std: bn254_sf_std,
    bn254_g1_xyzz: bn254_sf,
    bn254_g1_xyzz_std: bn254_sf_std,
    bn254_g2_affine: bn254_sf,
    bn254_g2_affine_std: bn254_sf_std,
    bn254_g2_jacobian: bn254_sf,
    bn254_g2_jacobian_std: bn254_sf_std,
    bn254_g2_xyzz: bn254_sf,
    bn254_g2_xyzz_std: bn254_sf_std,
}


# Tests for the Python scalar type
@multi_threaded(num_workers=3)
class ScalarTest(parameterized.TestCase):

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testModuleName(self, scalar_type):
    self.assertEqual(scalar_type.__module__, "zk_dtypes")

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testItem(self, scalar_type):
    self.assertIsInstance(scalar_type(1).item(), scalar_type)
    self.assertEqual(scalar_type(1).item(), scalar_type(1))

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testComparison(self, scalar_type):
    self.assertTrue(VALUES[scalar_type][0] != VALUES[scalar_type][1])
    self.assertTrue(VALUES[scalar_type][0] == VALUES[scalar_type][0])

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testAddop(self, scalar_type):
    out = VALUES[scalar_type][0] + VALUES[scalar_type][1]
    self.assertIsInstance(out, ADD_OP_RESULT_TYPES[scalar_type])
    self.assertEqual(out, scalar_type(7))

  @parameterized.product(
      param=[
          (bn254_g1_affine, bn254_g1_jacobian),
          (bn254_g1_affine, bn254_g1_xyzz),
          (bn254_g1_affine_std, bn254_g1_jacobian_std),
          (bn254_g1_affine_std, bn254_g1_xyzz_std),
          (bn254_g2_affine, bn254_g2_jacobian),
          (bn254_g2_affine, bn254_g2_xyzz),
          (bn254_g2_affine_std, bn254_g2_jacobian_std),
          (bn254_g2_affine_std, bn254_g2_xyzz_std),
      ]
  )
  def testAffineMixedAddop(self, param):
    a, b = param
    out = VALUES[a][0] + VALUES[b][1]
    self.assertIsInstance(out, b)
    self.assertEqual(out, b(7))

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testNegop(self, scalar_type):
    out = -VALUES[scalar_type][0]
    self.assertIsInstance(out, scalar_type)
    self.assertEqual(VALUES[scalar_type][0] + out, scalar_type(0))

  @parameterized.product(scalar_type=EC_MONT_POINT_TYPES)
  def testMulop(self, scalar_type):
    out = VALUES[scalar_type][0] * SCALAR_FIELD_TYPES[scalar_type](4)
    self.assertIsInstance(out, ADD_OP_RESULT_TYPES[scalar_type])
    self.assertEqual(out, scalar_type(12))

  @parameterized.product(a=EC_POINT_TYPES, b=EC_POINT_TYPES)
  def testCanCast(self, a, b):
    allowed_casts = [
        (bn254_g1_affine, bn254_g1_affine),
        (bn254_g1_affine, bn254_g1_jacobian),
        (bn254_g1_affine, bn254_g1_xyzz),
        (bn254_g1_affine_std, bn254_g1_affine_std),
        (bn254_g1_affine_std, bn254_g1_jacobian_std),
        (bn254_g1_affine_std, bn254_g1_xyzz_std),
        (bn254_g1_jacobian, bn254_g1_affine),
        (bn254_g1_jacobian, bn254_g1_jacobian),
        (bn254_g1_jacobian_std, bn254_g1_affine_std),
        (bn254_g1_jacobian_std, bn254_g1_jacobian_std),
        (bn254_g1_xyzz, bn254_g1_affine),
        (bn254_g1_xyzz, bn254_g1_xyzz),
        (bn254_g1_xyzz_std, bn254_g1_affine_std),
        (bn254_g1_xyzz_std, bn254_g1_xyzz_std),
        (bn254_g2_affine, bn254_g2_affine),
        (bn254_g2_affine, bn254_g2_jacobian),
        (bn254_g2_affine, bn254_g2_xyzz),
        (bn254_g2_affine_std, bn254_g2_affine_std),
        (bn254_g2_affine_std, bn254_g2_jacobian_std),
        (bn254_g2_affine_std, bn254_g2_xyzz_std),
        (bn254_g2_jacobian, bn254_g2_affine),
        (bn254_g2_jacobian, bn254_g2_jacobian),
        (bn254_g2_jacobian_std, bn254_g2_affine_std),
        (bn254_g2_jacobian_std, bn254_g2_jacobian_std),
        (bn254_g2_xyzz, bn254_g2_affine),
        (bn254_g2_xyzz, bn254_g2_xyzz),
        (bn254_g2_xyzz_std, bn254_g2_affine_std),
        (bn254_g2_xyzz_std, bn254_g2_xyzz_std),
    ]
    self.assertEqual(
        ((a, b) in allowed_casts), np.can_cast(a, b, casting="safe")
    )

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testIssubdtype(self, scalar_type):
    # In the future, we may want to make these more specific (e.g. use
    # np.number or np.integer instead of np.generic) by changing the
    # base in RegisterIntNDtype.
    self.assertTrue(np.issubdtype(scalar_type, np.generic))
    self.assertTrue(np.issubdtype(np.dtype(scalar_type), np.generic))

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testCastToDtype(self, scalar_type):
    name = scalar_type.__name__
    dt = np.dtype(scalar_type)
    self.assertIs(dt.type, scalar_type)
    self.assertEqual(dt.name, name)
    self.assertEqual(repr(dt), f"dtype({name})")


# Tests for the Python scalar type
@multi_threaded(num_workers=3)
class ArrayTest(parameterized.TestCase):

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testDtype(self, scalar_type):
    self.assertEqual(scalar_type, np.dtype(scalar_type))

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testHash(self, scalar_type):
    h = hash(np.dtype(scalar_type))
    self.assertEqual(h, hash(np.dtype(scalar_type.dtype)))
    self.assertEqual(h, hash(np.dtype(scalar_type.__name__)))

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testDeepCopyDoesNotAlterHash(self, scalar_type):
    # For context, see https://github.com/jax-ml/jax/issues/4651. If the hash
    # value of the type descriptor is not initialized correctly, a deep copy
    # can change the type hash.
    dtype = np.dtype(scalar_type)
    h = hash(dtype)
    _ = copy.deepcopy(dtype)
    self.assertEqual(h, hash(dtype))

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testArray(self, scalar_type):
    x = np.array([[1, 2, 3]], dtype=scalar_type)
    self.assertEqual(scalar_type, x.dtype)
    self.assertTrue((x == x).all())  # pylint: disable=comparison-with-itself

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testNegUfunc(self, scalar_type):
    x = np.array(VALUES[scalar_type], dtype=scalar_type)
    y = -x
    for i in range(len(x)):
      self.assertEqual(x[i] + y[i], scalar_type(0))

  @parameterized.product(
      scalar_type=EC_POINT_TYPES,
      ufunc=[
          np.add,
          np.subtract,
      ],
  )
  def testBinaryUfuncs(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type], dtype=scalar_type)
    y = ufunc(x, x)
    for i in range(len(x)):
      self.assertEqual(ufunc(x[i], x[i]), y[i])

  @parameterized.product(scalar_type=EC_POINT_TYPES)
  def testMulUfunc(self, scalar_type):
    x = np.array(VALUES[scalar_type], dtype=scalar_type)
    y = x * SCALAR_FIELD_TYPES[scalar_type](4)
    for i in range(len(x)):
      self.assertEqual(x[i] * SCALAR_FIELD_TYPES[scalar_type](4), y[i])


if __name__ == "__main__":
  absltest.main()
