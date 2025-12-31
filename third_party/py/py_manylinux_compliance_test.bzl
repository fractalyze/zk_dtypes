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

""" Macros for manylinux compliance verification test. """

load("@rules_python//python:py_test.bzl", "py_test")

def verify_manylinux_compliance_test(
        name,
        wheel,
        aarch64_compliance_tag,
        x86_64_compliance_tag,
        ppc64le_compliance_tag,
        test_tags = []):
    py_test(
        name = name,
        srcs = [Label("//third_party/py:manylinux_compliance_test.py")],
        data = [
            wheel,
        ],
        deps = ["@pypi_auditwheel//:pkg"],
        args = [
            "--wheel-path=$(location {})".format(wheel),
            "--aarch64-compliance-tag={}".format(aarch64_compliance_tag),
            "--x86_64-compliance-tag={}".format(x86_64_compliance_tag),
            "--ppc64le-compliance-tag={}".format(ppc64le_compliance_tag),
        ],
        main = "manylinux_compliance_test.py",
        tags = ["manual"] + test_tags,
    )
