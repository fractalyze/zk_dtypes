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

The `zk_dtypes` package is tested with Python versions 3.9-3.12, and can be
installed with the following command:

```shell
pip install zk_dtypes
```

To test your installation, you can run the following:

```shell
pip install absl-py pytest
pytest --pyargs zk_dtypes/tests
```

To build from source, clone the repository and run:

```shell
pip install .
```

## Example Usage

```python
>>> from zk_dtypes import uint4
>>> import numpy as np
>>> np.zeros(4, dtype=uint4)
array([0, 0, 0, 0], dtype=uint4)
```

Importing `zk_dtypes` also registers the data types with numpy, so that they may
be referred to by their string name:

```python
>>> np.dtype('uint4')
dtype(uint4)
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
