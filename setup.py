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

"""Setuptool-based build for zk_dtypes."""

import os
import platform
import shutil
from setuptools import setup
from distutils.command.build import build

if platform.system() == "Windows":
  FILE = "_zk_dtypes_ext.pyd"
else:
  FILE = "_zk_dtypes_ext.so"

BAZEL_OUTPUT_FILE = f"bazel-bin/zk_dtypes/{FILE}"
TARGET_FILE_IN_PACKAGE = f"zk_dtypes/{FILE}"


class CustomBuildCommand(build):

  def run(self):
    if not os.path.exists(BAZEL_OUTPUT_FILE):
      raise FileNotFoundError(
          f"Bazel C Extension output not found: {BAZEL_OUTPUT_FILE}."
      )

    build.run(self)

    dst_path = os.path.join(self.build_lib, "zk_dtypes", FILE)
    shutil.copyfile(BAZEL_OUTPUT_FILE, dst_path)


setup(
    name="zk_dtypes",
    package_data={
        "zk_dtypes": [FILE],
    },
    cmdclass={
        "build": CustomBuildCommand,
    },
    zip_safe=False,
)
