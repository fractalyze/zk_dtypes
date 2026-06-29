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

"""Runtime resolution of a field to a numpy dtype.

`prime_field(p, storage)` and `extension_field(p, degree, non_residue, storage)`
map a field to the numpy dtype used to materialize its buffers. A curated family
(babybear, koalabear, goldilocks, mersenne31, bn254_sf and their extensions)
resolves to its pre-built legacy dtype — those keep their enum identity so the
non-parametric stack still recognizes them. A novel field is registered on the
parametric ``FieldDType`` (one numpy DType serving every field).

The (base) modulus is validated with a Miller-Rabin test: neither the compiler
nor the verifier can check primality, and a composite modulus silently produces
garbage (every field operation assumes Z/pZ is a field, not just a ring).
"""

import numpy as np

from zk_dtypes import _zk_dtypes_ext as _ext
from zk_dtypes._efinfo import efinfo
from zk_dtypes._pfinfo import pfinfo
from zk_dtypes._zk_dtypes_ext import binary_field_t0
from zk_dtypes._zk_dtypes_ext import binary_field_t1
from zk_dtypes._zk_dtypes_ext import binary_field_t2
from zk_dtypes._zk_dtypes_ext import binary_field_t3
from zk_dtypes._zk_dtypes_ext import binary_field_t4
from zk_dtypes._zk_dtypes_ext import binary_field_t5
from zk_dtypes._zk_dtypes_ext import binary_field_t6
from zk_dtypes._zk_dtypes_ext import binary_field_t7
from zk_dtypes._zk_dtypes_ext import babybear
from zk_dtypes._zk_dtypes_ext import babybear_mont
from zk_dtypes._zk_dtypes_ext import babybearx4
from zk_dtypes._zk_dtypes_ext import babybearx4_mont
from zk_dtypes._zk_dtypes_ext import bn254_sf
from zk_dtypes._zk_dtypes_ext import bn254_sf_mont
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

# Storage width classes the parametric stack materializes (base) fields into;
# modulus bit length rounds up to the smallest that fits (xla_fork FIELD32/64/
# 128/256).
_WIDTH_CLASSES = (32, 64, 128, 256)

_STORAGE_ALIASES = {
    "mont": True,
    "montgomery": True,
    "std": False,
    "canonical": False,
}

# First 12 primes as Miller-Rabin bases: deterministic for n < 3.3e24, and a
# strong probabilistic test (error < 4^-12) for crypto-scale moduli above it.
_MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_probable_prime(n: int) -> bool:
  if n < 2:
    return False
  for base in _MILLER_RABIN_BASES:
    if n % base == 0:
      return n == base
  d, r = n - 1, 0
  while d % 2 == 0:
    d //= 2
    r += 1
  for a in _MILLER_RABIN_BASES:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
      continue
    for _ in range(r - 1):
      x = x * x % n
      if x == n - 1:
        break
    else:
      return False
  return True


def _storage_width(modulus_bits: int) -> int:
  for width in _WIDTH_CLASSES:
    if modulus_bits <= width:
      return width
  raise ValueError(
      f"modulus needs {modulus_bits} bits; the widest field storage class is "
      f"{_WIDTH_CLASSES[-1]} bits"
  )


def _field_descr(
    modulus: int,
    degree: int,
    non_residue: int,
    base_width: int,
    is_montgomery: bool,
) -> np.dtype:
  """Builds a parametric FieldDType dtype, computing Montgomery constants."""
  if not is_montgomery:
    return np.dtype(
        _ext.field_descr(modulus, degree, non_residue, base_width, 0)
    )
  # Per-coefficient Montgomery: bytes are `c*R mod p` with R = 2^base_width
  # (prime_ir's device encoding). Decode multiplies by R^-1.
  r_mod_p = (1 << base_width) % modulus
  rinv_mod_p = pow(r_mod_p, -1, modulus)
  return np.dtype(
      _ext.field_descr(
          modulus, degree, non_residue, base_width, 1, r_mod_p, rinv_mod_p
      )
  )


def _curated_prime_map() -> dict[tuple[int, bool], type]:
  out: dict[tuple[int, bool], type] = {}
  # Both storage forms are curated where the legacy stack registers them
  # (Mersenne uses canonical reduction, so it has no Montgomery variant). The
  # (modulus, is_montgomery) key keeps the two forms distinct.
  for family in (
      babybear,
      babybear_mont,
      koalabear,
      koalabear_mont,
      goldilocks,
      goldilocks_mont,
      mersenne31,
      bn254_sf,
      bn254_sf_mont,
  ):
    info = pfinfo(np.dtype(family))
    out[(info.modulus, info.is_montgomery)] = family
  return out


def _curated_ext_map() -> dict[tuple[int, int, int, bool], type]:
  out: dict[tuple[int, int, int, bool], type] = {}
  for family in (
      babybearx4,
      babybearx4_mont,
      koalabearx4,
      koalabearx4_mont,
      goldilocksx3,
      goldilocksx3_mont,
      mersenne31x2,
  ):
    info = efinfo(np.dtype(family))
    base_modulus = pfinfo(info.base_field_dtype).modulus
    out[(base_modulus, info.degree, info.non_residue, info.is_montgomery)] = (
        family
    )
  return out


_CURATED_PRIME = _curated_prime_map()
_CURATED_EXT = _curated_ext_map()
_CURATED_BINARY = {
    0: binary_field_t0,
    1: binary_field_t1,
    2: binary_field_t2,
    3: binary_field_t3,
    4: binary_field_t4,
    5: binary_field_t5,
    6: binary_field_t6,
    7: binary_field_t7,
}


def prime_field(modulus: int, storage: str = "mont") -> np.dtype:
  """Resolves a prime modulus + storage form to a field dtype.

  Args:
    modulus: the field characteristic p. Validated prime (Miller-Rabin).
    storage: 'mont'/'montgomery' for Montgomery-form buffers, 'std'/'canonical'
      for canonical residues.

  Returns:
    The numpy dtype for Z/pZ in the requested storage form.

  Raises:
    TypeError: modulus is not an int.
    ValueError: modulus is composite, storage is unrecognized, or the modulus
      exceeds the widest storage class.
  """
  if isinstance(modulus, bool) or not isinstance(modulus, int):
    raise TypeError(f"modulus must be an int, got {type(modulus).__name__}")
  if storage not in _STORAGE_ALIASES:
    raise ValueError(
        f"storage must be one of {sorted(_STORAGE_ALIASES)}, got {storage!r}"
    )
  is_montgomery = _STORAGE_ALIASES[storage]
  if not _is_probable_prime(modulus):
    raise ValueError(
        f"modulus {modulus} is not prime; Z/{modulus}Z is a ring, not a field"
    )
  width = _storage_width(modulus.bit_length())
  curated = _CURATED_PRIME.get((modulus, is_montgomery))
  if curated is not None:
    return np.dtype(curated)
  return _field_descr(modulus, 1, 0, width, is_montgomery)


def extension_field(
    base_modulus: int, degree: int, non_residue: int, storage: str = "mont"
) -> np.dtype:
  """Resolves a binomial extension field Fp[X]/(X^degree - non_residue).

  Args:
    base_modulus: the base prime p. Validated prime (Miller-Rabin).
    degree: extension degree >= 2 (use ``prime_field`` for degree 1).
    non_residue: the constant of the binomial modulus; X^degree = non_residue.
      Must be a non-residue for the field to be a field (the caller is trusted
      on irreducibility — it cannot be checked cheaply).
    storage: per-coefficient storage form, as in ``prime_field``.

  Returns:
    The numpy dtype for the extension field. Elements are ``degree`` base-field
    coefficients, constant term first.

  Raises:
    TypeError: base_modulus is not an int.
    ValueError: base is composite, degree < 2, storage unrecognized, or the base
      exceeds the widest storage class.
  """
  if isinstance(base_modulus, bool) or not isinstance(base_modulus, int):
    raise TypeError(
        f"base_modulus must be an int, got {type(base_modulus).__name__}"
    )
  if storage not in _STORAGE_ALIASES:
    raise ValueError(
        f"storage must be one of {sorted(_STORAGE_ALIASES)}, got {storage!r}"
    )
  if degree < 2:
    raise ValueError(
        f"degree must be >= 2 (use prime_field for 1), got {degree}"
    )
  if not _is_probable_prime(base_modulus):
    raise ValueError(
        f"base_modulus {base_modulus} is not prime; the base must be a field"
    )
  is_montgomery = _STORAGE_ALIASES[storage]
  non_residue %= base_modulus
  base_width = _storage_width(base_modulus.bit_length())
  curated = _CURATED_EXT.get((base_modulus, degree, non_residue, is_montgomery))
  if curated is not None:
    return np.dtype(curated)
  return _field_descr(
      base_modulus, degree, non_residue, base_width, is_montgomery
  )


def binary_field(level: int) -> np.dtype:
  """Resolves a binary tower field GF(2^(2^level)) to a numpy dtype.

  Levels 0..7 are curated families (binary_field_t0..t7) and resolve to their
  legacy dtype; higher levels mint a parametric ``BinaryFieldDType``. Addition is
  XOR; multiplication is the recursive Karatsuba tower product.

  Args:
    level: tower level >= 0; the field is GF(2^(2^level)).

  Returns:
    The numpy dtype for the binary tower field.

  Raises:
    TypeError: level is not an int.
    ValueError: level is negative.
  """
  if isinstance(level, bool) or not isinstance(level, int):
    raise TypeError(f"level must be an int, got {type(level).__name__}")
  if level < 0:
    raise ValueError(f"level must be >= 0, got {level}")
  curated = _CURATED_BINARY.get(level)
  if curated is not None:
    return np.dtype(curated)
  return np.dtype(_ext.binary_field_descr(level))
