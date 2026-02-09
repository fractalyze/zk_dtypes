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

"""Overload of numpy.efinfo to handle extension field dtypes defined in zk_dtypes."""

from zk_dtypes._zk_dtypes_ext import babybear
from zk_dtypes._zk_dtypes_ext import babybear_mont
from zk_dtypes._zk_dtypes_ext import babybearx4
from zk_dtypes._zk_dtypes_ext import babybearx4_mont
from zk_dtypes._zk_dtypes_ext import goldilocks
from zk_dtypes._zk_dtypes_ext import goldilocks_mont
from zk_dtypes._zk_dtypes_ext import goldilocksx3
from zk_dtypes._zk_dtypes_ext import goldilocksx3_mont
from zk_dtypes._zk_dtypes_ext import koalabear
from zk_dtypes._zk_dtypes_ext import koalabear_mont
from zk_dtypes._zk_dtypes_ext import koalabearx4
from zk_dtypes._zk_dtypes_ext import koalabearx4_mont
from zk_dtypes._zk_dtypes_ext import mersenne31
from zk_dtypes._zk_dtypes_ext import mersenne31x2

import numpy as np

_babybear_dtype = np.dtype(babybear)
_babybear_mont_dtype = np.dtype(babybear_mont)
_babybearx4_dtype = np.dtype(babybearx4)
_babybearx4_mont_dtype = np.dtype(babybearx4_mont)
_goldilocks_dtype = np.dtype(goldilocks)
_goldilocks_mont_dtype = np.dtype(goldilocks_mont)
_goldilocksx3_dtype = np.dtype(goldilocksx3)
_goldilocksx3_mont_dtype = np.dtype(goldilocksx3_mont)
_koalabear_dtype = np.dtype(koalabear)
_koalabear_mont_dtype = np.dtype(koalabear_mont)
_koalabearx4_dtype = np.dtype(koalabearx4)
_koalabearx4_mont_dtype = np.dtype(koalabearx4_mont)
_mersenne31_dtype = np.dtype(mersenne31)
_mersenne31x2_dtype = np.dtype(mersenne31x2)

# Mersenne31 modulus: 2^(31) - 1
_MERSENNE31_MODULUS = 2**31 - 1


class efinfo:  # pylint: disable=invalid-name,missing-class-docstring
  base_field_dtype: np.dtype
  degree: int
  non_residue: int
  storage_bits: int
  is_montgomery: bool
  dtype: np.dtype

  def __init__(self, ef_type):
    ef_type = np.dtype(ef_type)
    self.dtype = ef_type

    if ef_type == _babybearx4_dtype or ef_type == _babybearx4_mont_dtype:
      self.base_field_dtype = (
          _babybear_mont_dtype
          if ef_type == _babybearx4_mont_dtype
          else _babybear_dtype
      )
      self.degree = 4
      self.non_residue = 11
      self.storage_bits = 32
      self.is_montgomery = ef_type == _babybearx4_mont_dtype
    elif ef_type == _goldilocksx3_dtype or ef_type == _goldilocksx3_mont_dtype:
      self.base_field_dtype = (
          _goldilocks_mont_dtype
          if ef_type == _goldilocksx3_mont_dtype
          else _goldilocks_dtype
      )
      self.degree = 3
      self.non_residue = 7
      self.storage_bits = 64
      self.is_montgomery = ef_type == _goldilocksx3_mont_dtype
    elif ef_type == _koalabearx4_dtype or ef_type == _koalabearx4_mont_dtype:
      self.base_field_dtype = (
          _koalabear_mont_dtype
          if ef_type == _koalabearx4_mont_dtype
          else _koalabear_dtype
      )
      self.degree = 4
      self.non_residue = 3
      self.storage_bits = 32
      self.is_montgomery = ef_type == _koalabearx4_mont_dtype
    elif ef_type == _mersenne31x2_dtype:
      self.base_field_dtype = _mersenne31_dtype
      self.degree = 2
      self.non_residue = _MERSENNE31_MODULUS - 1  # p - 1 (i.e. -1 mod p)
      self.storage_bits = 32
      self.is_montgomery = False
    else:
      raise ValueError(f"Unknown extension field type: {ef_type}")

  def __repr__(self):
    return f"efinfo(degree={self.degree}, dtype={self.dtype})"

  def __str__(self):
    return repr(self)
