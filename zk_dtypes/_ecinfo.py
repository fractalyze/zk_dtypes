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


# Curve parameters keyed by (curve, group, is_montgomery). Each entry holds the
# curve coefficients and generator for that group/representation. Adding a new
# curve is a matter of adding its g1/g2 std+mont entries here.
_CURVE_PARAMS = {
    ('bn254', 'g1', False): dict(
        a=_BN254_A,
        b=_BN254_B,
        gx=_BN254_G1_GX,
        gy=_BN254_G1_GY,
        non_residue=None,
    ),
    ('bn254', 'g1', True): dict(
        a=_BN254_A,
        b=_BN254_G1_B_MONT,
        gx=_BN254_G1_GX_MONT,
        gy=_BN254_G1_GY_MONT,
        non_residue=None,
    ),
    ('bn254', 'g2', False): dict(
        a=[0, 0],
        b=_BN254_G2_B,
        gx=_BN254_G2_GX,
        gy=_BN254_G2_GY,
        non_residue=_BN254_G2_NON_RESIDUE,
    ),
    ('bn254', 'g2', True): dict(
        a=[0, 0],
        b=_BN254_G2_B_MONT,
        gx=_BN254_G2_GX_MONT,
        gy=_BN254_G2_GY_MONT,
        non_residue=_BN254_G2_NON_RESIDUE_MONT,
    ),
}

# Point layout shared by every curve: (group, repr, num_coords, ext_degree).
# storage_bits = num_coords * ext_degree * field_bits. G1 lives in the base
# field (ext_degree 1); G2 lives in the Fp² extension field (ext_degree 2).
_EC_LAYOUT = [
    ('g1', 'affine', 2, 1),
    ('g1', 'jacobian', 3, 1),
    ('g1', 'xyzz', 4, 1),
    ('g2', 'affine', 2, 2),
    ('g2', 'jacobian', 3, 2),
    ('g2', 'xyzz', 4, 2),
]


def _build_meta(
    curve, field_bits, scalar_std_dtype, scalar_mont_dtype, dtype_table
):
  """Builds the dtype -> point metadata map for one curve.

  Args:
    curve: Curve name, e.g. 'bn254'.
    field_bits: Base field width in bits (e.g. 256 for bn254).
    scalar_std_dtype: np.dtype of the standard-form scalar/base field.
    scalar_mont_dtype: np.dtype of the Montgomery-form scalar/base field.
    dtype_table: Map of (group, repr, is_montgomery) -> point scalar type.

  Returns:
    Map of np.dtype(point) -> (curve, group, repr, is_montgomery,
    storage_bits, base_field_dtype).
  """
  meta = {}
  for group, repr_, num_coords, ext_degree in _EC_LAYOUT:
    storage_bits = num_coords * ext_degree * field_bits
    for is_mont in (False, True):
      point_dtype = dtype_table[(group, repr_, is_mont)]
      base_field_dtype = scalar_mont_dtype if is_mont else scalar_std_dtype
      meta[np.dtype(point_dtype)] = (
          curve,
          group,
          repr_,
          is_mont,
          storage_bits,
          base_field_dtype,
      )
  return meta


# Curve-list-driven dtype metadata. Each curve contributes one _build_meta call.
_EC_DTYPE_META = _build_meta(
    'bn254',
    256,
    _bn254_sf_dtype,
    _bn254_sf_mont_dtype,
    {
        ('g1', 'affine', False): bn254_g1_affine,
        ('g1', 'affine', True): bn254_g1_affine_mont,
        ('g1', 'jacobian', False): bn254_g1_jacobian,
        ('g1', 'jacobian', True): bn254_g1_jacobian_mont,
        ('g1', 'xyzz', False): bn254_g1_xyzz,
        ('g1', 'xyzz', True): bn254_g1_xyzz_mont,
        ('g2', 'affine', False): bn254_g2_affine,
        ('g2', 'affine', True): bn254_g2_affine_mont,
        ('g2', 'jacobian', False): bn254_g2_jacobian,
        ('g2', 'jacobian', True): bn254_g2_jacobian_mont,
        ('g2', 'xyzz', False): bn254_g2_xyzz,
        ('g2', 'xyzz', True): bn254_g2_xyzz_mont,
    },
)


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

    meta = _EC_DTYPE_META.get(ec_type)
    if meta is None:
      raise ValueError(f'Unknown elliptic curve point type: {ec_type}')
    curve, group, repr_, is_mont, storage_bits, base_field_dtype = meta
    self.curve_group = group
    self.point_repr = repr_
    self.is_montgomery = is_mont
    self.storage_bits = storage_bits
    self.base_field_dtype = base_field_dtype

    params = _CURVE_PARAMS[(curve, group, is_mont)]
    self.a = params['a']
    self.b = params['b']
    self.gx = params['gx']
    self.gy = params['gy']
    self.non_residue = params['non_residue']

  def __repr__(self):
    return f'ecinfo(curve_group={self.curve_group}, point_repr={self.point_repr}, dtype={self.dtype})'

  def __str__(self):
    return repr(self)
