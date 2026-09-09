"""The Montgomery Goldilocks dtypes warn on access and steer to canonical."""

import warnings

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import zk_dtypes
from zk_dtypes import _zk_dtypes_ext


class DeprecatedDtypesTest(parameterized.TestCase):

  @parameterized.parameters(
      ("goldilocks_mont", "goldilocks"),
      ("goldilocksx3_mont", "goldilocksx3"),
  )
  def test_montgomery_goldilocks_warns_and_still_resolves(self, name, canon):
    with self.assertWarnsRegex(DeprecationWarning, canon):
      dtype = getattr(zk_dtypes, name)
    # Still the registered dtype, so existing data and interop keep working.
    self.assertEqual(np.dtype(dtype), np.dtype(getattr(_zk_dtypes_ext, name)))
    self.assertNotIn(name, zk_dtypes.__all__)

  def test_canonical_goldilocks_is_quiet(self):
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      np.dtype(zk_dtypes.goldilocks)
      np.dtype(zk_dtypes.goldilocksx3)

  def test_unknown_attribute_still_raises(self):
    with self.assertRaises(AttributeError):
      zk_dtypes.no_such_dtype  # pylint: disable=pointless-statement


if __name__ == "__main__":
  absltest.main()
