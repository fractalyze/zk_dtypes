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

__version__ = "0.0.1"
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
    "babybear_std",
    "goldilocks",
    "goldilocks_std",
    "koalabear",
    "koalabear_std",
    "mersenne31",
    "mersenne31_std",
    # Big prime field types
    "bn254_sf",
    "bn254_sf_std",
    # Elliptic curve types
    "bn254_g1_affine",
    "bn254_g1_affine_std",
    "bn254_g1_jacobian",
    "bn254_g1_jacobian_std",
    "bn254_g1_xyzz",
    "bn254_g1_xyzz_std",
    "bn254_g2_affine",
    "bn254_g2_affine_std",
    "bn254_g2_jacobian",
    "bn254_g2_jacobian_std",
    "bn254_g2_xyzz",
    "bn254_g2_xyzz_std",
]

from typing import Type

from zk_dtypes._iinfo import iinfo
from zk_dtypes._pfinfo import pfinfo
from zk_dtypes._zk_dtypes_ext import int2
from zk_dtypes._zk_dtypes_ext import int4
from zk_dtypes._zk_dtypes_ext import uint2
from zk_dtypes._zk_dtypes_ext import uint4
from zk_dtypes._zk_dtypes_ext import babybear
from zk_dtypes._zk_dtypes_ext import babybear_std
from zk_dtypes._zk_dtypes_ext import goldilocks
from zk_dtypes._zk_dtypes_ext import goldilocks_std
from zk_dtypes._zk_dtypes_ext import koalabear
from zk_dtypes._zk_dtypes_ext import koalabear_std
from zk_dtypes._zk_dtypes_ext import mersenne31
from zk_dtypes._zk_dtypes_ext import mersenne31_std
from zk_dtypes._zk_dtypes_ext import bn254_sf
from zk_dtypes._zk_dtypes_ext import bn254_sf_std
from zk_dtypes._zk_dtypes_ext import bn254_g1_affine
from zk_dtypes._zk_dtypes_ext import bn254_g1_affine_std
from zk_dtypes._zk_dtypes_ext import bn254_g1_jacobian
from zk_dtypes._zk_dtypes_ext import bn254_g1_jacobian_std
from zk_dtypes._zk_dtypes_ext import bn254_g1_xyzz
from zk_dtypes._zk_dtypes_ext import bn254_g1_xyzz_std
from zk_dtypes._zk_dtypes_ext import bn254_g2_affine
from zk_dtypes._zk_dtypes_ext import bn254_g2_affine_std
from zk_dtypes._zk_dtypes_ext import bn254_g2_jacobian
from zk_dtypes._zk_dtypes_ext import bn254_g2_jacobian_std
from zk_dtypes._zk_dtypes_ext import bn254_g2_xyzz
from zk_dtypes._zk_dtypes_ext import bn254_g2_xyzz_std

import numpy as np

int2: Type[np.generic]
int4: Type[np.generic]
uint2: Type[np.generic]
uint4: Type[np.generic]
babybear: Type[np.generic]
babybear_std: Type[np.generic]
goldilocks: Type[np.generic]
goldilocks_std: Type[np.generic]
koalabear: Type[np.generic]
koalabear_std: Type[np.generic]
mersenne31: Type[np.generic]
mersenne31_std: Type[np.generic]
bn254_sf: Type[np.generic]
bn254_sf_std: Type[np.generic]
bn254_g1_affine: Type[np.generic]
bn254_g1_affine_std: Type[np.generic]
bn254_g1_jacobian: Type[np.generic]
bn254_g1_jacobian_std: Type[np.generic]
bn254_g1_xyzz: Type[np.generic]
bn254_g1_xyzz_std: Type[np.generic]
bn254_g2_affine: Type[np.generic]
bn254_g2_affine_std: Type[np.generic]
bn254_g2_jacobian: Type[np.generic]
bn254_g2_jacobian_std: Type[np.generic]
bn254_g2_xyzz: Type[np.generic]
bn254_g2_xyzz_std: Type[np.generic]

del np, Type
