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

"""Overload of numpy.ecinfo to handle EC point dtypes defined in zk_dtypes."""

from zk_dtypes._zk_dtypes_ext import bn254_sf
from zk_dtypes._zk_dtypes_ext import bn254_sf_mont
from zk_dtypes._zk_dtypes_ext import bn254_g1_affine
from zk_dtypes._zk_dtypes_ext import bn254_g1_affine_mont
from zk_dtypes._zk_dtypes_ext import bn254_g1_jacobian
from zk_dtypes._zk_dtypes_ext import bn254_g1_jacobian_mont
from zk_dtypes._zk_dtypes_ext import bn254_g1_xyzz
from zk_dtypes._zk_dtypes_ext import bn254_g1_xyzz_mont
from zk_dtypes._zk_dtypes_ext import bn254_g2_affine
from zk_dtypes._zk_dtypes_ext import bn254_g2_affine_mont
from zk_dtypes._zk_dtypes_ext import bn254_g2_jacobian
from zk_dtypes._zk_dtypes_ext import bn254_g2_jacobian_mont
from zk_dtypes._zk_dtypes_ext import bn254_g2_xyzz
from zk_dtypes._zk_dtypes_ext import bn254_g2_xyzz_mont

import numpy as np

_bn254_sf_dtype = np.dtype(bn254_sf)
_bn254_sf_mont_dtype = np.dtype(bn254_sf_mont)
_bn254_g1_affine_dtype = np.dtype(bn254_g1_affine)
_bn254_g1_affine_mont_dtype = np.dtype(bn254_g1_affine_mont)
_bn254_g1_jacobian_dtype = np.dtype(bn254_g1_jacobian)
_bn254_g1_jacobian_mont_dtype = np.dtype(bn254_g1_jacobian_mont)
_bn254_g1_xyzz_dtype = np.dtype(bn254_g1_xyzz)
_bn254_g1_xyzz_mont_dtype = np.dtype(bn254_g1_xyzz_mont)
_bn254_g2_affine_dtype = np.dtype(bn254_g2_affine)
_bn254_g2_affine_mont_dtype = np.dtype(bn254_g2_affine_mont)
_bn254_g2_jacobian_dtype = np.dtype(bn254_g2_jacobian)
_bn254_g2_jacobian_mont_dtype = np.dtype(bn254_g2_jacobian_mont)
_bn254_g2_xyzz_dtype = np.dtype(bn254_g2_xyzz)
_bn254_g2_xyzz_mont_dtype = np.dtype(bn254_g2_xyzz_mont)

_BN254_A = 0
_BN254_B = 3
_BN254_G1_GX = 1
_BN254_G1_GY = 2

# G1 Montgomery form: each value * R mod Fq
_BN254_G1_B_MONT = 19052624634359457937016868847204597229365286637454337178037183604060995791063
_BN254_G1_GX_MONT = (
    6350874878119819312338956282401532409788428879151445726012394534686998597021
)
_BN254_G1_GY_MONT = 12701749756239638624677912564803064819576857758302891452024789069373997194042

# G2 standard form: Fp2 elements [c₀, c₁] where val = c₀ + c₁ * u
_BN254_G2_B = [
    19485874751759354771024239261021720505790618469301721065564631296452457478373,
    266929791119991161246907387137283842545076965332900288569378510910307636690,
]
_BN254_G2_GX = [
    10857046999023057135944570762232829481370756359578518086990519993285655852781,
    11559732032986387107991004021392285783925812861821192530917403151452391805634,
]
_BN254_G2_GY = [
    8495653923123431417604973247489272438418190587263600148770280649306958101930,
    4082367875863433681332203403145435568316851327593401208105741076214120093531,
]
_BN254_G2_NON_RESIDUE = 21888242871839275222246405745257275088696311157297823662689037894645226208582

# G2 Montgomery form: each Fp² component * R mod Fq
_BN254_G2_B_MONT = [
    16772280239760917788496391897731603718812008455956943122563801666366297604776,
    568440292453150825972223760836185707764922522371208948902804025364325400423,
]
_BN254_G2_GX_MONT = [
    11461925177900819176832270005713103520318409907105193817603008068482420711462,
    9496696083199853777875401760424613833161720860855390556979200160215841136960,
]
_BN254_G2_GY_MONT = [
    18540402224736191443939503902445128293982106376239432540843647066670759668214,
    6170940445994484564222204938066213705353407449799250191249554538140978927342,
]
_BN254_G2_NON_RESIDUE_MONT = 15537367993719455909907449462855742678907882278146377936676643359958227611562


class ecinfo:  # pylint: disable=invalid-name,missing-class-docstring
  base_field_dtype: np.dtype
  storage_bits: int
  point_repr: str  # 'affine', 'jacobian', or 'xyzz'
  curve_group: str  # 'g1' or 'g2'
  is_montgomery: bool
  a: int | list[int]  # curve coefficient a in y² = x³ + ax + b
  b: int | list[int]  # curve coefficient b
  gx: int | list[int]  # generator x-coordinate
  gy: int | list[int]  # generator y-coordinate
  non_residue: int | None  # Fp² non-residue (None for G1, int for G2)
  dtype: np.dtype

  def __init__(self, ec_type):
    ec_type = np.dtype(ec_type)
    self.dtype = ec_type
    self.a = _BN254_A

    # G1 types (base field is bn254_sf)
    if ec_type == _bn254_g1_affine_dtype:
      self.base_field_dtype = _bn254_sf_dtype
      self.storage_bits = 512  # 2 × 256 bits (x, y)
      self.point_repr = 'affine'
      self.curve_group = 'g1'
      self.is_montgomery = False
    elif ec_type == _bn254_g1_affine_mont_dtype:
      self.base_field_dtype = _bn254_sf_mont_dtype
      self.storage_bits = 512
      self.point_repr = 'affine'
      self.curve_group = 'g1'
      self.is_montgomery = True
    elif ec_type == _bn254_g1_jacobian_dtype:
      self.base_field_dtype = _bn254_sf_dtype
      self.storage_bits = 768  # 3 × 256 bits (x, y, z)
      self.point_repr = 'jacobian'
      self.curve_group = 'g1'
      self.is_montgomery = False
    elif ec_type == _bn254_g1_jacobian_mont_dtype:
      self.base_field_dtype = _bn254_sf_mont_dtype
      self.storage_bits = 768
      self.point_repr = 'jacobian'
      self.curve_group = 'g1'
      self.is_montgomery = True
    elif ec_type == _bn254_g1_xyzz_dtype:
      self.base_field_dtype = _bn254_sf_dtype
      self.storage_bits = 1024  # 4 × 256 bits (x, y, zz, zzz)
      self.point_repr = 'xyzz'
      self.curve_group = 'g1'
      self.is_montgomery = False
    elif ec_type == _bn254_g1_xyzz_mont_dtype:
      self.base_field_dtype = _bn254_sf_mont_dtype
      self.storage_bits = 1024
      self.point_repr = 'xyzz'
      self.curve_group = 'g1'
      self.is_montgomery = True
    # G2 types (base field is Fp2 extension field, but we use bn254_sf for now)
    # NOTE: G2 points use extension field Fp2 as their base field.
    # For MLIR lowering, this will require creating an ExtensionFieldType.
    elif ec_type == _bn254_g2_affine_dtype:
      self.base_field_dtype = _bn254_sf_dtype  # Will be Fp2 in MLIR
      self.storage_bits = 1024  # 2 × 2 × 256 bits (Fp2 has 2 elements)
      self.point_repr = 'affine'
      self.curve_group = 'g2'
      self.is_montgomery = False
    elif ec_type == _bn254_g2_affine_mont_dtype:
      self.base_field_dtype = _bn254_sf_mont_dtype
      self.storage_bits = 1024
      self.point_repr = 'affine'
      self.curve_group = 'g2'
      self.is_montgomery = True
    elif ec_type == _bn254_g2_jacobian_dtype:
      self.base_field_dtype = _bn254_sf_dtype
      self.storage_bits = 1536  # 3 × 2 × 256 bits
      self.point_repr = 'jacobian'
      self.curve_group = 'g2'
      self.is_montgomery = False
    elif ec_type == _bn254_g2_jacobian_mont_dtype:
      self.base_field_dtype = _bn254_sf_mont_dtype
      self.storage_bits = 1536
      self.point_repr = 'jacobian'
      self.curve_group = 'g2'
      self.is_montgomery = True
    elif ec_type == _bn254_g2_xyzz_dtype:
      self.base_field_dtype = _bn254_sf_dtype
      self.storage_bits = 2048  # 4 × 2 × 256 bits
      self.point_repr = 'xyzz'
      self.curve_group = 'g2'
      self.is_montgomery = False
    elif ec_type == _bn254_g2_xyzz_mont_dtype:
      self.base_field_dtype = _bn254_sf_mont_dtype
      self.storage_bits = 2048
      self.point_repr = 'xyzz'
      self.curve_group = 'g2'
      self.is_montgomery = True
    else:
      raise ValueError(f'Unknown elliptic curve point type: {ec_type}')

    # Set curve parameters based on group and Montgomery form
    if self.curve_group == 'g1':
      if self.is_montgomery:
        self.b = _BN254_G1_B_MONT
        self.gx = _BN254_G1_GX_MONT
        self.gy = _BN254_G1_GY_MONT
      else:
        self.b = _BN254_B
        self.gx = _BN254_G1_GX
        self.gy = _BN254_G1_GY
      self.non_residue = None
    else:
      self.a = [0, 0]
      if self.is_montgomery:
        self.b = _BN254_G2_B_MONT
        self.gx = _BN254_G2_GX_MONT
        self.gy = _BN254_G2_GY_MONT
        self.non_residue = _BN254_G2_NON_RESIDUE_MONT
      else:
        self.b = _BN254_G2_B
        self.gx = _BN254_G2_GX
        self.gy = _BN254_G2_GY
        self.non_residue = _BN254_G2_NON_RESIDUE

  def __repr__(self):
    return f'ecinfo(curve_group={self.curve_group}, point_repr={self.point_repr}, dtype={self.dtype})'

  def __str__(self):
    return repr(self)
