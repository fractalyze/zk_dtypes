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

"""zk_dtypes dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

def zk_dtypes_deps():
    """zk_dtypes dependencies."""
    protobuf_deps()

    http_archive(
        name = "com_google_absl",
        sha256 = "9b2b72d4e8367c0b843fa2bcfa2b08debbe3cee34f7aaa27de55a6cbb3e843db",
        strip_prefix = "abseil-cpp-20250814.0",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.tar.gz"],
        patches = ["@zk_dtypes//third_party/absl:endian.patch"],
        patch_args = ["-p1"],
        repo_mapping = {
            "@googletest": "@com_google_googletest",
        },
    )

    http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        urls = ["https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"],
    )

    http_archive(
        name = "com_google_googletest",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"],
    )

    EIGEN_COMMIT = "4c38131a16803130b66266a912029504f2cf23cd"
    http_archive(
        name = "eigen_archive",
        build_file = "@zk_dtypes//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)],
    )
