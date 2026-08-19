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

"""Overload of numpy.pfinfo to handle dtypes defined in zk_dtypes."""

from zk_dtypes._zk_dtypes_ext import babybear
from zk_dtypes._zk_dtypes_ext import babybear_mont
from zk_dtypes._zk_dtypes_ext import goldilocks
from zk_dtypes._zk_dtypes_ext import goldilocks_mont
from zk_dtypes._zk_dtypes_ext import koalabear
from zk_dtypes._zk_dtypes_ext import koalabear_mont
from zk_dtypes._zk_dtypes_ext import mersenne31
from zk_dtypes._zk_dtypes_ext import bn254_sf
from zk_dtypes._zk_dtypes_ext import bn254_sf_mont
from zk_dtypes._zk_dtypes_ext import curve25519_bf
from zk_dtypes._zk_dtypes_ext import curve25519_bf_mont
from zk_dtypes._zk_dtypes_ext import curve25519_sf
from zk_dtypes._zk_dtypes_ext import curve25519_sf_mont
from zk_dtypes._zk_dtypes_ext import pallas_sf
from zk_dtypes._zk_dtypes_ext import pallas_sf_mont
from zk_dtypes._zk_dtypes_ext import secp256k1_bf
from zk_dtypes._zk_dtypes_ext import secp256k1_bf_mont
from zk_dtypes._zk_dtypes_ext import secp256k1_sf
from zk_dtypes._zk_dtypes_ext import secp256k1_sf_mont
from zk_dtypes._zk_dtypes_ext import secp256r1_bf
from zk_dtypes._zk_dtypes_ext import secp256r1_bf_mont
from zk_dtypes._zk_dtypes_ext import secp256r1_sf
from zk_dtypes._zk_dtypes_ext import secp256r1_sf_mont
from zk_dtypes._zk_dtypes_ext import vesta_sf
from zk_dtypes._zk_dtypes_ext import vesta_sf_mont

import numpy as np

_babybear_dtype = np.dtype(babybear)
_babybear_mont_dtype = np.dtype(babybear_mont)
_goldilocks_dtype = np.dtype(goldilocks)
_goldilocks_mont_dtype = np.dtype(goldilocks_mont)
_koalabear_dtype = np.dtype(koalabear)
_koalabear_mont_dtype = np.dtype(koalabear_mont)
_mersenne31_dtype = np.dtype(mersenne31)
_bn254_sf_dtype = np.dtype(bn254_sf)
_bn254_sf_mont_dtype = np.dtype(bn254_sf_mont)
_curve25519_bf_dtype = np.dtype(curve25519_bf)
_curve25519_bf_mont_dtype = np.dtype(curve25519_bf_mont)
_curve25519_sf_dtype = np.dtype(curve25519_sf)
_curve25519_sf_mont_dtype = np.dtype(curve25519_sf_mont)
_pallas_sf_dtype = np.dtype(pallas_sf)
_pallas_sf_mont_dtype = np.dtype(pallas_sf_mont)
_secp256k1_bf_dtype = np.dtype(secp256k1_bf)
_secp256k1_bf_mont_dtype = np.dtype(secp256k1_bf_mont)
_secp256k1_sf_dtype = np.dtype(secp256k1_sf)
_secp256k1_sf_mont_dtype = np.dtype(secp256k1_sf_mont)
_secp256r1_bf_dtype = np.dtype(secp256r1_bf)
_secp256r1_bf_mont_dtype = np.dtype(secp256r1_bf_mont)
_secp256r1_sf_dtype = np.dtype(secp256r1_sf)
_secp256r1_sf_mont_dtype = np.dtype(secp256r1_sf_mont)
_vesta_sf_dtype = np.dtype(vesta_sf)
_vesta_sf_mont_dtype = np.dtype(vesta_sf_mont)


_BN254_PARAM = 4965661367192848881

# Pasta scalar field moduli (arkworks ark-pallas / ark-vesta). The Pallas
# scalar field equals the Vesta base field (q) and vice versa (p).
_PALLAS_SF_MODULUS = (
    0x40000000000000000000000000000000224698FC0994A8DD8C46EB2100000001
)
_VESTA_SF_MODULUS = (
    0x40000000000000000000000000000000224698FC094CF91B992D30ED00000001
)

# curve25519 (RFC 7748): base field 2²⁵⁵ - 19; scalar field is the ed25519
# prime-order subgroup size L (RFC 8032 §5.1).
_CURVE25519_BF_MODULUS = 2**255 - 19
_CURVE25519_SF_MODULUS = 2**252 + 27742317777372353535851937790883648493

# secp256k1 (SEC 2 §2.4.1): base field p and group order n.
_SECP256K1_BF_MODULUS = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
)
_SECP256K1_SF_MODULUS = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)

# secp256r1, NIST P-256 (SEC 2 §2.4.2): base field p and group order n.
_SECP256R1_BF_MODULUS = (
    0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
)
_SECP256R1_SF_MODULUS = (
    0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
)


def _get_bn_scalar_field_modulus(x):
  return 36 * x**4 + 36 * x**3 + 18 * x**2 + 6 * x + 1


class pfinfo:  # pylint: disable=invalid-name,missing-class-docstring
  storage_bits: int
  modulus_bits: int
  modulus: int
  is_montgomery: bool
  dtype: np.dtype
  two_adicity: int

  def __init__(self, pf_type):
    if pf_type == _babybear_dtype or pf_type == _babybear_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 32
      self.modulus_bits = 31
      self.modulus = 2**31 - 2**27 + 1
      self.is_montgomery = pf_type == _babybear_mont_dtype
    elif pf_type == _goldilocks_dtype or pf_type == _goldilocks_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 64
      self.modulus_bits = 64
      self.modulus = 2**64 - 2**32 + 1
      self.is_montgomery = pf_type == _goldilocks_mont_dtype
    elif pf_type == _koalabear_dtype or pf_type == _koalabear_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 32
      self.modulus_bits = 31
      self.modulus = 2**31 - 2**24 + 1
      self.is_montgomery = pf_type == _koalabear_mont_dtype
    elif pf_type == _mersenne31_dtype:
      self.dtype = pf_type
      self.storage_bits = 32
      self.modulus_bits = 31
      self.modulus = 2**31 - 1
      self.is_montgomery = False
    elif pf_type == _bn254_sf_dtype or pf_type == _bn254_sf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 254
      self.modulus = _get_bn_scalar_field_modulus(_BN254_PARAM)
      self.is_montgomery = pf_type == _bn254_sf_mont_dtype
    elif pf_type == _pallas_sf_dtype or pf_type == _pallas_sf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 255
      self.modulus = _PALLAS_SF_MODULUS
      self.is_montgomery = pf_type == _pallas_sf_mont_dtype
    elif pf_type == _vesta_sf_dtype or pf_type == _vesta_sf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 255
      self.modulus = _VESTA_SF_MODULUS
      self.is_montgomery = pf_type == _vesta_sf_mont_dtype
    elif (
        pf_type == _curve25519_bf_dtype or pf_type == _curve25519_bf_mont_dtype
    ):
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 255
      self.modulus = _CURVE25519_BF_MODULUS
      self.is_montgomery = pf_type == _curve25519_bf_mont_dtype
    elif (
        pf_type == _curve25519_sf_dtype or pf_type == _curve25519_sf_mont_dtype
    ):
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 253
      self.modulus = _CURVE25519_SF_MODULUS
      self.is_montgomery = pf_type == _curve25519_sf_mont_dtype
    elif pf_type == _secp256k1_bf_dtype or pf_type == _secp256k1_bf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 256
      self.modulus = _SECP256K1_BF_MODULUS
      self.is_montgomery = pf_type == _secp256k1_bf_mont_dtype
    elif pf_type == _secp256k1_sf_dtype or pf_type == _secp256k1_sf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 256
      self.modulus = _SECP256K1_SF_MODULUS
      self.is_montgomery = pf_type == _secp256k1_sf_mont_dtype
    elif pf_type == _secp256r1_bf_dtype or pf_type == _secp256r1_bf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 256
      self.modulus = _SECP256R1_BF_MODULUS
      self.is_montgomery = pf_type == _secp256r1_bf_mont_dtype
    elif pf_type == _secp256r1_sf_dtype or pf_type == _secp256r1_sf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 256
      self.modulus = _SECP256R1_SF_MODULUS
      self.is_montgomery = pf_type == _secp256r1_sf_mont_dtype
    elif getattr(pf_type, "modulus", None) is not None and (
        getattr(pf_type, "degree", None) == 1
    ):
      # Parametric FieldDType descriptor: read the parameters it carries.
      self.dtype = pf_type
      self.storage_bits = pf_type.base_width_bits
      self.modulus = pf_type.modulus
      self.modulus_bits = self.modulus.bit_length()
      self.is_montgomery = pf_type.is_montgomery
    else:
      raise ValueError(f"Unknown prime field type: {pf_type}")
    # Calculate the 2-adicity of `modulus - 1`, which is the number of
    # trailing zeros in its binary representation.
    m1 = self.modulus - 1
    self.two_adicity = (m1 & -m1).bit_length() - 1

  def __repr__(self):
    return f"pfinfo(modulus_bits={self.modulus_bits}, dtype={self.dtype})"

  def __str__(self):
    return repr(self)
