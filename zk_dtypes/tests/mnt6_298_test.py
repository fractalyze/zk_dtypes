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

"""Tests for the MNT6-298 numpy dtypes registered in zk_dtypes.

Reference constants are the standard-form values from ark-mnt6-298. MNT6 G2 is
defined over Fp3 (the cubic twist), so its coordinates are 3-element lists and
its point storage is 3x wider than G1.
"""

from absl.testing import absltest
from absl.testing import parameterized
import zk_dtypes

import numpy as np

# MNT6-298 is the MNT4 cycle: its scalar field is MNT4's Fq, its base field is
# MNT4's Fr.
_FR_MODULUS = 475922286169261325753349249653048451545124879242694725395555128576210262817955800483758081
_FQ_MODULUS = 475922286169261325753349249653048451545124878552823515553267735739164647307408490559963137
_R = 1 << 320

_G1_A = 11
_G1_B = 106700080510851735677967319632585352256454251201367587890185989362936000262606668469523074
_G1_GX = 336685752883082228109289846353937104185698209371404178342968838739115829740084426881123453
_G1_GY = 402596290139780989709332707716568920777622032073762749862342374583908837063963736098549800
_G2_A = [0, 0, 11]
_G2_B = [
    57578116384997352636487348509878309737146377454014423897662211075515354005624851787652233,
    0,
    0,
]
_G2_GX = [
    421456435772811846256826561593908322288509115489119907560382401870203318738334702321297427,
    103072927438548502463527009961344915021167584706439945404959058962657261178393635706405114,
    143029172143731852627002926324735183809768363301149009204849580478324784395590388826052558,
]
_G2_GY = [
    464673596668689463130099227575639512541218133445388869383893594087634649237515554342751377,
    100642907501977375184575075967118071807821117960152743335603284583254620685343989304941678,
    123019855502969896026940545715841181300275180157288044663051565390506010149881373807142903,
]
_G2_NON_RESIDUE = 5


class Mnt6298DtypeTest(parameterized.TestCase):

  @parameterized.parameters(
      "mnt6_298_sf",
      "mnt6_298_sf_mont",
      "mnt6_298_g1_affine",
      "mnt6_298_g1_affine_mont",
      "mnt6_298_g1_jacobian",
      "mnt6_298_g1_jacobian_mont",
      "mnt6_298_g1_xyzz",
      "mnt6_298_g1_xyzz_mont",
      "mnt6_298_g2_affine",
      "mnt6_298_g2_affine_mont",
      "mnt6_298_g2_jacobian",
      "mnt6_298_g2_jacobian_mont",
      "mnt6_298_g2_xyzz",
      "mnt6_298_g2_xyzz_mont",
  )
  def test_dtype_registered_and_allocatable(self, name):
    dt = np.dtype(getattr(zk_dtypes, name))
    arr = np.zeros(4, dtype=dt)
    self.assertEqual(arr.dtype, dt)
    self.assertLen(arr, 4)

  def test_pfinfo_scalar_field(self):
    for t in (zk_dtypes.mnt6_298_sf, zk_dtypes.mnt6_298_sf_mont):
      info = zk_dtypes.pfinfo(t)
      self.assertEqual(info.modulus, _FR_MODULUS)
      self.assertEqual(info.storage_bits, 320)
      self.assertEqual(info.modulus_bits, 298)
      self.assertEqual(info.two_adicity, 17)
    self.assertFalse(zk_dtypes.pfinfo(zk_dtypes.mnt6_298_sf).is_montgomery)
    self.assertTrue(zk_dtypes.pfinfo(zk_dtypes.mnt6_298_sf_mont).is_montgomery)

  @parameterized.parameters(
      ("mnt6_298_g1_affine", "affine", 640),
      ("mnt6_298_g1_jacobian", "jacobian", 960),
      ("mnt6_298_g1_xyzz", "xyzz", 1280),
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
      # G2 over Fp3: storage is num_coords * 3 * 320 bits.
      ("mnt6_298_g2_affine", "affine", 1920),
      ("mnt6_298_g2_jacobian", "jacobian", 2880),
      ("mnt6_298_g2_xyzz", "xyzz", 3840),
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
    info = zk_dtypes.ecinfo(zk_dtypes.mnt6_298_g1_affine_mont)
    self.assertTrue(info.is_montgomery)
    self.assertEqual(info.a, self._mont(_G1_A))
    self.assertEqual(info.b, self._mont(_G1_B))
    self.assertEqual(info.gx, self._mont(_G1_GX))
    self.assertEqual(info.gy, self._mont(_G1_GY))

  def test_g2_montgomery_coefficients(self):
    info = zk_dtypes.ecinfo(zk_dtypes.mnt6_298_g2_affine_mont)
    self.assertTrue(info.is_montgomery)
    self.assertEqual(info.a, [self._mont(c) for c in _G2_A])
    self.assertEqual(info.b, [self._mont(c) for c in _G2_B])
    self.assertEqual(info.gx, [self._mont(c) for c in _G2_GX])
    self.assertEqual(info.gy, [self._mont(c) for c in _G2_GY])
    self.assertEqual(info.non_residue, self._mont(_G2_NON_RESIDUE))


if __name__ == "__main__":
  absltest.main()
