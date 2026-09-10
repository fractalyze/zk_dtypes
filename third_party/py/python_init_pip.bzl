# Copyright The OpenXLA Authors.
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

"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")
load(
    "//third_party/py:python_init_toolchains.bzl",
    "get_toolchain_name_per_python_version",
)

def python_init_pip():
    # Kept byte-identical to //third_party/py/numpy:numpy_headers.BUILD, which
    # MODULE.bazel reads directly. `package_annotation` takes only a string and
    # MODULE.bazel cannot `load()` a shared constant, so the text is repeated
    # here and //third_party/py/numpy:numpy_headers_sync_test guards the pair.
    numpy_annotations = {
        "numpy": package_annotation(
            additive_build_content = """\
# numpy ships its C headers inside the wheel but exposes no cc_library for
# them, and the include root moved from `core` to `_core` in numpy 2, so both
# are declared and the wrapper depends on whichever glob matched.
cc_library(
    name = "numpy_headers_2",
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
    strip_include_prefix = "site-packages/numpy/_core/include/",
)

cc_library(
    name = "numpy_headers_1",
    hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
    strip_include_prefix = "site-packages/numpy/core/include/",
)

cc_library(
    name = "numpy_headers",
    deps = [
        ":numpy_headers_1",
        ":numpy_headers_2",
    ],
)
""",
        ),
    }

    pip_parse(
        name = "pypi",
        annotations = numpy_annotations,
        python_interpreter_target = "@{}_host//:python".format(
            get_toolchain_name_per_python_version("python"),
        ),
        extra_hub_aliases = {
            "numpy": ["numpy_headers"],
        },
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
    )
