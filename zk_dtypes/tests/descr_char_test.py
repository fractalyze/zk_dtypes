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
"""Guards the kNpyDescrType chars of every registered dtype.

A custom dtype whose ``.char`` shadows a builtin numpy typecode makes
``np.dtype(arr.dtype.char)`` round-trips silently resolve to the BUILTIN
(``np.dtype('f')`` stays float32), i.e. a wrong type with no error. New
dtypes must pick chars outside ``np.typecodes['All']``; the frozen set below
grandfathers the pre-existing offenders — shrink it, never grow it.
"""

from absl.testing import absltest
import numpy as np

import zk_dtypes

# name -> shadowed builtin char. Frozen as of binary_field_gf8_aes ('8', the
# first dtype gated by this test); each entry predates the test.
_GRANDFATHERED_BUILTIN_CHARS = {
    "babybear": "B",
    "babybear_mont": "b",
    "babybearx4": "D",
    "babybearx4_mont": "d",
    "binary_field_ghash": "h",
    "binary_field_t5": "l",
    "binary_field_t6": "L",
    "bn254_sf": "B",
    "bn254_sf_mont": "b",
    "goldilocks": "G",
    "goldilocks_mont": "g",
    "int128": "n",
    "koalabearx4_mont": "e",
    "mersenne31": "m",
    "mersenne31x2": "q",
    "pallas_sf": "P",
    "pallas_sf_mont": "p",
    "uint128": "N",
}


def _all_scalar_types():
  for name in zk_dtypes.__all__:
    t = getattr(zk_dtypes, name)
    if isinstance(t, type) and issubclass(t, np.generic):
      yield name, t


class DescrCharTest(absltest.TestCase):

  def testNewDtypesDoNotShadowBuiltinTypecodes(self):
    builtin = set(np.typecodes["All"])
    offenders = {
        name: np.dtype(t).char
        for name, t in _all_scalar_types()
        if np.dtype(t).char in builtin
    }
    self.assertEqual(
        offenders,
        _GRANDFATHERED_BUILTIN_CHARS,
        "a dtype's kNpyDescrType shadows a builtin numpy typecode (or a "
        "grandfathered one changed): pick a char outside "
        "np.typecodes['All']; only remove entries from the frozen set",
    )

  def testBuiltinCharLookupUnaffectedByRegistration(self):
    # Registering the zk dtypes must never change what a builtin typecode
    # resolves to.
    for c in np.typecodes["All"]:
      self.assertLess(
          np.dtype(c).num,
          256,
          f"np.dtype({c!r}) resolves to a "
          "user type after zk_dtypes registration",
      )


if __name__ == "__main__":
  absltest.main()
