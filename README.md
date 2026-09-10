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

- Prime Field:

  - `babybear`
  - `babybear_mont`
  - `goldilocks`
  - `goldilocks_mont`
  - `koalabear`
  - `koalabear_mont`
  - `mersenne31`

- Extension Field:

  - `babybearx4`
  - `babybearx4_mont`
  - `goldilocksx3`
  - `goldilocksx3_mont`
  - `koalabearx4`
  - `koalabearx4_mont`
  - `mersenne31x2`

- Binary field:

  - `binary_field_t0`
  - `binary_field_t1`
  - `binary_field_t2`
  - `binary_field_t3`
  - `binary_field_t4`
  - `binary_field_t5`
  - `binary_field_t6`
  - `binary_field_t7`

- Elliptic curve:

  - `bn254_sf`
  - `bn254_sf_mont`
  - `bn254_g1_affine`
  - `bn254_g1_affine_mont`
  - `bn254_g1_jacobian`
  - `bn254_g1_jacobian_mont`
  - `bn254_g1_xyzz`
  - `bn254_g1_xyzz_mont`
  - `bn254_g2_affine`
  - `bn254_g2_affine_mont`
  - `bn254_g2_jacobian`
  - `bn254_g2_jacobian_mont`
  - `bn254_g2_xyzz`
  - `bn254_g2_xyzz_mont`

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

### Depending on zk_dtypes from another Bazel repo

`WORKSPACE.bazel` is the default and stays the supported path for Bazel 7
consumers. `MODULE.bazel` declares the same dependency set for bzlmod consumers,
and `--config=bzlmod` builds this repo through it:

```sh
bazel test --config=bzlmod //...
```

A consumer picks the repo up with `bazel_dep` plus an override:

```py
bazel_dep(name = "zk_dtypes", version = "0.0.17")
git_override(
    module_name = "zk_dtypes",
    remote = "https://github.com/fractalyze/zk_dtypes",
    commit = "<commit>",
)
```

Where the registry cannot carry a WORKSPACE pin, `MODULE.bazel` records the
version it resolves to instead, at the dependency.

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

### Installation from source

To build and install the package from source, run:

```shell
pip install .
```

#### Installation from prebuilt binary

Use `USE_BAZEL_OUTPUT=1` for a faster installation that uses pre-built Bazel
artifacts. This is the recommended path for development.

- **On Linux / macOS**

  ```shell
  # Build the shared library (.so)
  bazel build //zk_dtypes:_zk_dtypes_ext.so

  # Install using the Bazel output
  USE_BAZEL_OUTPUT=1 pip install .
  ```

- **On Windows**

  ```powershell
  # Build the Python extension (.pyd)
  bazel build //zk_dtypes:_zk_dtypes_ext.pyd

  # Install using the Bazel output (PowerShell syntax)
  $env:USE_BAZEL_OUTPUT=1; pip install .
  ```

## Example Usage

```python
>>> from zk_dtypes import babybear_mont
>>> import numpy as np
>>> a = np.array([-1, -3, 2**30, 7], dtype=babybear_mont)
>>> b = np.array([5, 2, 4, 10], dtype=babybear_mont)
>>> a + b
array([4, 2013265920, 1073741828, 17], dtype=babybear_mont)
```

Importing `zk_dtypes` also registers the data types with numpy, so that they may
be referred to by their string name:

```python
>>> np.dtype('babybear_mont')
dtype(babybear_mont)
```

See [examples/zk_dtypes_examples.ipynb](/examples/zk_dtypes_examples.ipynb) for
more examples.

## Benchmarks

Benchmarks are disabled by default (though CI verifies their validity). To
execute a specific benchmark (e.g., `field_mul_benchmark`), run the following
command manually:

```shell
bazel run -c opt //zk_dtypes:field_mul_benchmark
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
