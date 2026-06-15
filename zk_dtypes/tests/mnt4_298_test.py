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

"""Tests for the MNT4-298 numpy dtypes registered in zk_dtypes.

Reference constants are the standard-form values from ark-mnt4-298.
"""

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes

import numpy as np

_FR_MODULUS = 475922286169261325753349249653048451545124878552823515553267735739164647307408490559963137
_G1_A = 2
_G1_B = 423894536526684178289416011533888240029318103673896002803341544124054745019340795360841685
_G1_GX = 60760244141852568949126569781626075788424196370144486719385562369396875346601926534016838
_G1_GY = 363732850702582978263902770815145784459747722357071843971107674179038674942891694705904306
_G2_A = [34, 0]
_G2_B = [
    0,
    67372828414711144619833451280373307321534573815811166723479321465776723059456513877937430,
]
_G2_NON_RESIDUE = 17
_G2_GX = [
    438374926219350099854919100077809681842783509163790991847867546339851681564223481322252708,
    37620953615500480110935514360923278605464476459712393277679280819942849043649216370485641,
]
_G2_GY = [
    37437409008528968268352521034936931842973546441370663118543015118291998305624025037512482,
    424621479598893882672393190337420680597584695892317197646113820787463109735345923009077489,
]

# MNT4-298 base field Fq modulus and Montgomery radix R = 2^320, used to check
# that ecinfo's Montgomery-form coefficients equal value·R mod p.
_FQ_MODULUS = 475922286169261325753349249653048451545124879242694725395555128576210262817955800483758081
_R = 1 << 320


class Mnt4298DtypeTest(parameterized.TestCase):

  @parameterized.parameters(
      "mnt4_298_sf",
      "mnt4_298_sf_mont",
      "mnt4_298_g1_affine",
      "mnt4_298_g1_affine_mont",
      "mnt4_298_g1_jacobian",
      "mnt4_298_g1_jacobian_mont",
      "mnt4_298_g1_xyzz",
      "mnt4_298_g1_xyzz_mont",
      "mnt4_298_g2_affine",
      "mnt4_298_g2_affine_mont",
      "mnt4_298_g2_jacobian",
      "mnt4_298_g2_jacobian_mont",
      "mnt4_298_g2_xyzz",
      "mnt4_298_g2_xyzz_mont",
  )
  def test_dtype_registered_and_allocatable(self, name):
    dt = np.dtype(getattr(zk_dtypes, name))
    arr = np.zeros(4, dtype=dt)
    self.assertEqual(arr.dtype, dt)
    self.assertLen(arr, 4)

  def test_pfinfo_scalar_field(self):
    for t in (zk_dtypes.mnt4_298_sf, zk_dtypes.mnt4_298_sf_mont):
      info = zk_dtypes.pfinfo(t)
      self.assertEqual(info.modulus, _FR_MODULUS)
      self.assertEqual(info.storage_bits, 320)
      self.assertEqual(info.modulus_bits, 298)
      self.assertEqual(info.two_adicity, 34)
    self.assertFalse(zk_dtypes.pfinfo(zk_dtypes.mnt4_298_sf).is_montgomery)
    self.assertTrue(zk_dtypes.pfinfo(zk_dtypes.mnt4_298_sf_mont).is_montgomery)

  @parameterized.parameters(
      ("mnt4_298_g1_affine", "affine", 640),
      ("mnt4_298_g1_jacobian", "jacobian", 960),
      ("mnt4_298_g1_xyzz", "xyzz", 1280),
  )
  def test_g1_ecinfo(self, name, point_repr, bits):
    info = zk_dtypes.ecinfo(getattr(zk_dtypes, name))
    self.assertEqual(info.curve_group, "g1")
    self.assertEqual(info.point_repr, point_repr)
    self.assertEqual(info.storage_bits, bits)
    self.assertFalse(info.is_montgomery)
    self.assertEqual(info.a, _G1_A)
    self.assertEqual(info.b, _G1_B)
    self.assertEqual(info.gx, _G1_GX)
    self.assertEqual(info.gy, _G1_GY)
    self.assertIsNone(info.non_residue)

  @parameterized.parameters(
      ("mnt4_298_g2_affine", "affine", 1280),
      ("mnt4_298_g2_jacobian", "jacobian", 1920),
      ("mnt4_298_g2_xyzz", "xyzz", 2560),
  )
  def test_g2_ecinfo(self, name, point_repr, bits):
    info = zk_dtypes.ecinfo(getattr(zk_dtypes, name))
    self.assertEqual(info.curve_group, "g2")
    self.assertEqual(info.point_repr, point_repr)
    self.assertEqual(info.storage_bits, bits)
    self.assertEqual(info.a, _G2_A)
    self.assertEqual(info.b, _G2_B)
    self.assertEqual(info.non_residue, _G2_NON_RESIDUE)

  def _mont(self, value):
    return (value * _R) % _FQ_MODULUS

  def test_g1_montgomery_coefficients(self):
    # ecinfo's Montgomery constants must equal value·R mod p. MNT4 G1 a=2 is
    # non-zero, so this genuinely exercises the Montgomery path (bn254 a=0 would
    # pass trivially). Also the only check of _ecinfo.py's mont decimals, which
    # are transcribed independently of the C++ limb constants.
    info = zk_dtypes.ecinfo(zk_dtypes.mnt4_298_g1_affine_mont)
    self.assertTrue(info.is_montgomery)
    self.assertEqual(info.a, self._mont(_G1_A))
    self.assertEqual(info.b, self._mont(_G1_B))
    self.assertEqual(info.gx, self._mont(_G1_GX))
    self.assertEqual(info.gy, self._mont(_G1_GY))

  def test_g2_montgomery_coefficients(self):
    info = zk_dtypes.ecinfo(zk_dtypes.mnt4_298_g2_affine_mont)
    self.assertTrue(info.is_montgomery)
    self.assertEqual(info.a, [self._mont(_G2_A[0]), self._mont(_G2_A[1])])
    self.assertEqual(info.b, [self._mont(_G2_B[0]), self._mont(_G2_B[1])])
    self.assertEqual(info.gx, [self._mont(_G2_GX[0]), self._mont(_G2_GX[1])])
    self.assertEqual(info.gy, [self._mont(_G2_GY[0]), self._mont(_G2_GY[1])])
    self.assertEqual(info.non_residue, self._mont(_G2_NON_RESIDUE))


if __name__ == "__main__":
  absltest.main()
