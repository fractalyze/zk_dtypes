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

_BN254_PARAM = 4965661367192848881
_BN254_BASE_FIELD_MODULUS = (
    36 * _BN254_PARAM**4
    + 36 * _BN254_PARAM**3
    + 24 * _BN254_PARAM**2
    + 6 * _BN254_PARAM
    + 1
)

_BN254_A = 0
_BN254_B = 3

_BN254_G1_GX = 1
_BN254_G1_GY = 2


class ecinfo:  # pylint: disable=invalid-name,missing-class-docstring
  base_field_dtype: np.dtype
  storage_bits: int
  point_repr: str  # 'affine', 'jacobian', or 'xyzz'
  curve_group: str  # 'g1' or 'g2'
  is_montgomery: bool
  a: int  # curve coefficient a in y² = x³ + ax + b
  b: int  # curve coefficient b
  gx: int  # generator x-coordinate
  gy: int  # generator y-coordinate
  dtype: np.dtype

  def __init__(self, ec_type):
    ec_type = np.dtype(ec_type)
    self.dtype = ec_type
    # BN254 curve parameters are the same for all representations
    self.a = _BN254_A
    self.b = _BN254_B
    self.gx = _BN254_G1_GX
    self.gy = _BN254_G1_GY

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

  def __repr__(self):
    return f'ecinfo(curve_group={self.curve_group}, point_repr={self.point_repr}, dtype={self.dtype})'

  def __str__(self):
    return repr(self)
