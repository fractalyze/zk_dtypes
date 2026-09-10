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

"""Dependencies that ship no Bazel module of their own.

Pinned once here for both dependency paths: `zk_dtypes_deps()` on the WORKSPACE
build, and the `non_module_deps` module extension on the bzlmod build. A
dependency the registry does carry belongs in MODULE.bazel as a `bazel_dep`
instead, so that consumers resolve one shared version of it with us.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_EIGEN_COMMIT = "4c38131a16803130b66266a912029504f2cf23cd"

def zk_dtypes_non_module_deps():
    """Declares the repositories that have no Bazel module."""
    http_archive(
        name = "eigen_archive",
        build_file = Label("//third_party/eigen3:eigen_archive.BUILD"),
        sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
        strip_prefix = "eigen-{commit}".format(commit = _EIGEN_COMMIT),
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = _EIGEN_COMMIT)],
    )

def _non_module_deps_impl(_module_ctx):
    zk_dtypes_non_module_deps()

non_module_deps = module_extension(implementation = _non_module_deps_impl)
