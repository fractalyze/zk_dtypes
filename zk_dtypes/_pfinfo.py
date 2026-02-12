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


_BN254_PARAM = 4965661367192848881


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
      self.two_adic_primitive_root = 440564289  # 31¹⁵ mod p
    elif pf_type == _goldilocks_dtype or pf_type == _goldilocks_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 64
      self.modulus_bits = 64
      self.modulus = 2**64 - 2**32 + 1
      self.is_montgomery = pf_type == _goldilocks_mont_dtype
      self.two_adic_primitive_root = (
          1753635133440165772  # 7^((p - 1) / 2³²) mod p
      )
    elif pf_type == _koalabear_dtype or pf_type == _koalabear_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 32
      self.modulus_bits = 31
      self.modulus = 2**31 - 2**24 + 1
      self.is_montgomery = pf_type == _koalabear_mont_dtype
      self.two_adic_primitive_root = 1791270792  # 3¹²⁷ mod p
    elif pf_type == _mersenne31_dtype:
      self.dtype = pf_type
      self.storage_bits = 32
      self.modulus_bits = 31
      self.modulus = 2**31 - 1
      self.is_montgomery = False
      self.two_adic_primitive_root = 2**31 - 2  # p - 1 (two_adicity = 1)
    elif pf_type == _bn254_sf_dtype or pf_type == _bn254_sf_mont_dtype:
      self.dtype = pf_type
      self.storage_bits = 256
      self.modulus_bits = 254
      self.modulus = _get_bn_scalar_field_modulus(_BN254_PARAM)
      self.is_montgomery = pf_type == _bn254_sf_mont_dtype
      self.two_adic_primitive_root = pow(
          5, (self.modulus - 1) >> 28, self.modulus
      )
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
