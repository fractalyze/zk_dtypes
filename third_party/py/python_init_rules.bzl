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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def python_init_rules(extra_patches = []):
    """Defines (doesn't setup) the rules_python repository.

    Args:
      extra_patches: list of labels. Additional patches to apply after the default
        set of patches.
    """

    http_archive(
        name = "rules_cc",
        sha256 = "4b12149a041ddfb8306a8fd0e904e39d673552ce82e4296e96fac9cbf0780e59",
        strip_prefix = "rules_cc-0.1.0",
        urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.1.0.tar.gz"],
        patches = [Label("//third_party/py:rules_cc_protobuf.patch")],
        patch_args = ["-p1"],
    )

    http_archive(
        name = "com_google_protobuf",
        sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
        strip_prefix = "protobuf-6.31.1",
        urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"],
        patches = [Label("@zk_dtypes//third_party/protobuf:protobuf.patch")],
        patch_args = ["-p1"],
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@protobuf_pip_deps": "@pypi",
        },
    )

    http_archive(
        name = "rules_python",
        sha256 = "fa7dd2c6b7d63b3585028dd8a90a6cf9db83c33b250959c2ee7b583a6c130e12",
        strip_prefix = "rules_python-1.6.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/1.6.0/rules_python-1.6.0.tar.gz",
        patch_args = ["-p1"],
        patches = [
            Label("//third_party/py:rules_python_pip_version.patch"),
            Label("//third_party/py:rules_python_freethreaded.patch"),
            Label("//third_party/py:rules_python_versions.patch"),
        ] + extra_patches,
    )
