# Copyright 2022 The ml_dtypes Authors.
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

__version__ = "0.0.3"
__all__ = [
    "__version__",
    "iinfo",
    "pfinfo",
    "int2",
    "int4",
    "uint2",
    "uint4",
    # Small prime field types
    "babybear",
    "babybear_mont",
    "goldilocks",
    "goldilocks_mont",
    "koalabear",
    "koalabear_mont",
    "mersenne31",
    # Big prime field types
    "bn254_sf",
    "bn254_sf_mont",
    # Extension field types
    "babybearx4",
    "babybearx4_mont",
    "goldilocksx3",
    "goldilocksx3_mont",
    "koalabearx4",
    "koalabearx4_mont",
    "mersenne31x2",
    # Binary field types (GF(2ⁿ) tower fields)
    "binary_field_t0",
    "binary_field_t1",
    "binary_field_t2",
    "binary_field_t3",
    "binary_field_t4",
    "binary_field_t5",
    "binary_field_t6",
    "binary_field_t7",
    # Elliptic curve types
    "bn254_g1_affine",
    "bn254_g1_affine_mont",
    "bn254_g1_jacobian",
    "bn254_g1_jacobian_mont",
    "bn254_g1_xyzz",
    "bn254_g1_xyzz_mont",
    "bn254_g2_affine",
    "bn254_g2_affine_mont",
    "bn254_g2_jacobian",
    "bn254_g2_jacobian_mont",
    "bn254_g2_xyzz",
    "bn254_g2_xyzz_mont",
]

from typing import Type

from zk_dtypes._iinfo import iinfo
from zk_dtypes._pfinfo import pfinfo
from zk_dtypes._zk_dtypes_ext import int2
from zk_dtypes._zk_dtypes_ext import int4
from zk_dtypes._zk_dtypes_ext import uint2
from zk_dtypes._zk_dtypes_ext import uint4
from zk_dtypes._zk_dtypes_ext import babybear
from zk_dtypes._zk_dtypes_ext import babybear_mont
from zk_dtypes._zk_dtypes_ext import goldilocks
from zk_dtypes._zk_dtypes_ext import goldilocks_mont
from zk_dtypes._zk_dtypes_ext import koalabear
from zk_dtypes._zk_dtypes_ext import koalabear_mont
from zk_dtypes._zk_dtypes_ext import mersenne31
from zk_dtypes._zk_dtypes_ext import bn254_sf
from zk_dtypes._zk_dtypes_ext import bn254_sf_mont
from zk_dtypes._zk_dtypes_ext import babybearx4
from zk_dtypes._zk_dtypes_ext import babybearx4_mont
from zk_dtypes._zk_dtypes_ext import goldilocksx3
from zk_dtypes._zk_dtypes_ext import goldilocksx3_mont
from zk_dtypes._zk_dtypes_ext import koalabearx4
from zk_dtypes._zk_dtypes_ext import koalabearx4_mont
from zk_dtypes._zk_dtypes_ext import mersenne31x2
from zk_dtypes._zk_dtypes_ext import binary_field_t0
from zk_dtypes._zk_dtypes_ext import binary_field_t1
from zk_dtypes._zk_dtypes_ext import binary_field_t2
from zk_dtypes._zk_dtypes_ext import binary_field_t3
from zk_dtypes._zk_dtypes_ext import binary_field_t4
from zk_dtypes._zk_dtypes_ext import binary_field_t5
from zk_dtypes._zk_dtypes_ext import binary_field_t6
from zk_dtypes._zk_dtypes_ext import binary_field_t7
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

int2: Type[np.generic]
int4: Type[np.generic]
uint2: Type[np.generic]
uint4: Type[np.generic]
babybear: Type[np.generic]
babybear_mont: Type[np.generic]
goldilocks: Type[np.generic]
goldilocks_mont: Type[np.generic]
koalabear: Type[np.generic]
koalabear_mont: Type[np.generic]
mersenne31: Type[np.generic]
bn254_sf: Type[np.generic]
bn254_sf_mont: Type[np.generic]
babybearx4: Type[np.generic]
babybearx4_mont: Type[np.generic]
goldilocksx3: Type[np.generic]
goldilocksx3_mont: Type[np.generic]
koalabearx4: Type[np.generic]
koalabearx4_mont: Type[np.generic]
mersenne31x2: Type[np.generic]
binary_field_t0: Type[np.generic]
binary_field_t1: Type[np.generic]
binary_field_t2: Type[np.generic]
binary_field_t3: Type[np.generic]
binary_field_t4: Type[np.generic]
binary_field_t5: Type[np.generic]
binary_field_t6: Type[np.generic]
binary_field_t7: Type[np.generic]
bn254_g1_affine: Type[np.generic]
bn254_g1_affine_mont: Type[np.generic]
bn254_g1_jacobian: Type[np.generic]
bn254_g1_jacobian_mont: Type[np.generic]
bn254_g1_xyzz: Type[np.generic]
bn254_g1_xyzz_mont: Type[np.generic]
bn254_g2_affine: Type[np.generic]
bn254_g2_affine_mont: Type[np.generic]
bn254_g2_jacobian: Type[np.generic]
bn254_g2_jacobian_mont: Type[np.generic]
bn254_g2_xyzz: Type[np.generic]
bn254_g2_xyzz_mont: Type[np.generic]

del np, Type
