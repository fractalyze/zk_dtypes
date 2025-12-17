# zk_dtypes

[![CI](https://github.com/fractalyze/zk_dtypes/actions/workflows/ci.yml/badge.svg)](https://github.com/fractalyze/zk_dtypes/actions/workflows/ci.yml)

`zk_dtypes` is a stand-alone implementation of several NumPy dtype extensions
used in Zero Knowledge libraries inspired by
[ml_dtypes](https://github.com/jax-ml/ml_dtypes), including:

- Narrow integer:

  - `int2`
  - `int4`
  - `uint2`
  - `uint4`

- Field:

  - `babybear`
  - `babybear_std`
  - `goldilocks`
  - `goldilocks_std`
  - `koalabear`
  - `koalabear_std`
  - `mersenne31`
  - `mersenne31_std`

- Extension Field:

  - `babybear4`
  - `babybear4_std`
  - `goldilocks3`
  - `goldilocks3_std`
  - `koalabear4`
  - `koalabear4_std`

- Elliptic curve:

  - `bn254_sf`
  - `bn254_sf_std`
  - `bn254_g1_affine`
  - `bn254_g1_affine_std`
  - `bn254_g1_jacobian`
  - `bn254_g1_jacobian_std`
  - `bn254_g1_xyzz`
  - `bn254_g1_xyzz_std`
  - `bn254_g2_affine`
  - `bn254_g2_affine_std`
  - `bn254_g2_jacobian`
  - `bn254_g2_jacobian_std`
  - `bn254_g2_xyzz`
  - `bn254_g2_xyzz_std`

## Prerequisite

1. Follow the [bazel installation guide](https://bazel.build/install).

## Build instructions

1. Clone the zk_dtypes repo

   ```sh
   git clone https://github.com/fractalyze/zk_dtypes
   ```

1. Build zk_dtypes

   ```sh
   bazel build //...
   ```

1. Test zk_dtypes

   ```sh
   bazel test //...
   ```

## Installation

The `zk_dtypes` package is tested with Python versions 3.11-3.13, and can be
installed with the following command:

```shell
pip install zk_dtypes
```

To test your installation, you can run the following:

```shell
pip install absl-py pytest
pytest zk_dtypes/tests
```

To build from source, clone the repository and run:

```shell
pip install .
```

## Example Usage

```python
>>> from zk_dtypes import babybear
>>> import numpy as np
>>> a = np.array([-1, -3, 2**30, 7], dtype=babybear)
>>> b = np.array([5, 2, 4, 10], dtype=babybear)
>>> a + b
array([4, 2013265920, 1073741828, 17], dtype=babybear)
```

Importing `zk_dtypes` also registers the data types with numpy, so that they may
be referred to by their string name:

```python
>>> np.dtype('babybear')
dtype(babybear)
```

## License

The `zk_dtypes` source code is a modified derivative of the `ml_dtypes` project
and inherits the original Apache 2.0 License (see [LICENSE](/LICENSE)). All
subsequent modifications comply with and are released under the same license.

### Pre-compiled Wheels Dependencies

Note that pre-compiled wheels utilize the following dependencies:

- The [EIGEN](https://eigen.tuxfamily.org/) project, licensed under the MPL 2.0
  license (see [LICENSE.eigen](/LICENSE.eigen)).
- [Chromium](https://github.com/chromium/chromium/), licensed under the Free-BSD
  3-Clause license (see [LICENSE.chromium](/LICENSE.chromium)).
