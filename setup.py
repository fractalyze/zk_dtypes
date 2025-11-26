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
import subprocess
from setuptools import Extension
from setuptools import setup

if platform.system() == "Windows":
  FILE = "_zk_dtypes_ext.pyd"
  COMPILE_ARGS = [
      "/std:c++17",
      "/DEIGEN_MPL2_ONLY",
      "/EHsc",
      "/bigobj",
  ]
else:
  FILE = "_zk_dtypes_ext.so"
  COMPILE_ARGS = [
      "-std=c++17",
      "-DEIGEN_MPL2_ONLY",
      "-fvisibility=hidden",
  ]


def get_sources():
  target = f"//zk_dtypes:{FILE}"
  result = subprocess.run(
      [
          "bazel",
          "query",
          f"kind('source file', deps({target}))",
          "--output=location",
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True,
      text=True,
  )
  sources = []
  for line in result.stdout.splitlines():
    if ":" in line:
      path = line.split(":")[0]
      if os.path.isfile(path):
        if path.endswith((".c", ".cc", ".cpp")):
          if path.endswith(
              ("def_parser/def_parser.cc", "def_parser/def_parser_main.cc")
          ):
            continue
          sources.append(path)
  return sources


output_base = subprocess.run(
    ["bazel", "info", "output_base"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=True,
    text=True,
).stdout.strip()

setup(
    name="zk_dtypes",
    ext_modules=[
        Extension(
            "zk_dtypes._zk_dtypes_ext",
            get_sources(),
            include_dirs=[
                ".",
                f"{output_base}/external/bazel_tools",
                f"{output_base}/external/com_google_absl",
                f"{output_base}/external/com_google_googletest/googletest/include",
                f"{output_base}/external/eigen_archive",
                f"{output_base}/external/pypi_numpy/site-packages/numpy/_core/include",
            ],
            extra_compile_args=COMPILE_ARGS,
        )
    ],
)
