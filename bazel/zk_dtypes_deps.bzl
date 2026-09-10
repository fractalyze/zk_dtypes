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
load("//bazel:non_module_deps.bzl", "zk_dtypes_non_module_deps")

def zk_dtypes_deps():
    """zk_dtypes dependencies."""
    protobuf_deps()

    http_archive(
        name = "com_google_absl",
        sha256 = "9b2b72d4e8367c0b843fa2bcfa2b08debbe3cee34f7aaa27de55a6cbb3e843db",
        strip_prefix = "abseil-cpp-20250814.0",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.0.tar.gz"],
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

    BM_COMMIT = "f7547e29ccaed7b64ef4f7495ecfff1c9f6f3d03"
    BM_SHA256 = "552ca3d4d1af4beeb1907980f7096315aa24150d6baf5ac1e5ad90f04846c670"
    http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        urls = ["https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT)],
    )

    zk_dtypes_non_module_deps()
