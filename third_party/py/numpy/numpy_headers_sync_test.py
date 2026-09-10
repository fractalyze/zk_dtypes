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
"""Guards the two copies of the numpy `additive_build_content` snippet.

The bzlmod build reads `numpy_headers.BUILD` directly; the WORKSPACE build has
to inline the same text, because `package_annotation` accepts only a string and
MODULE.bazel cannot `load()` a shared constant. This test fails when the two
stop matching.
"""

import pathlib

from absl.testing import absltest

_CANONICAL = pathlib.Path("third_party/py/numpy/numpy_headers.BUILD")
_WORKSPACE_COPY = pathlib.Path("third_party/py/python_init_pip.bzl")

_OPEN = 'additive_build_content = """\\\n'
_CLOSE = '""",\n'


def _inlined_snippet(source: str) -> str:
  start = source.index(_OPEN) + len(_OPEN)
  return source[start : source.index(_CLOSE, start)]


class NumpyHeadersSyncTest(absltest.TestCase):

  def test_workspace_copy_matches_canonical_file(self):
    canonical = _CANONICAL.read_text()
    inlined = _inlined_snippet(_WORKSPACE_COPY.read_text())
    self.assertEqual(
        canonical,
        inlined,
        f"{_WORKSPACE_COPY} has drifted from {_CANONICAL}. Copy the file's "
        "contents into the additive_build_content string verbatim.",
    )


if __name__ == "__main__":
  absltest.main()
