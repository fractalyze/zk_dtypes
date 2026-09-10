/* Copyright 2026 The zk_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Asserts that zk_dtypes resolves, compiles and links as a non-root Bazel
// module. The arithmetic is deliberately trivial: zk_dtypes' own suite covers
// behaviour, and what is under test here is dependency resolution.
//
// Compiling a field pulls in big_int.h and, through it, bit_iterator.h — so
// this also covers @com_google_absl//absl/base:config, the dependency whose
// patched predecessor could not have survived a non-root resolution.

#include "gtest/gtest.h"

#include "zk_dtypes/include/field/goldilocks/goldilocks.h"

namespace {

using F = zk_dtypes::Goldilocks;

TEST(ConsumeZkDtypesTest, FieldArithmeticLinks) {
  EXPECT_EQ(F(2) + F(3), F(5));
  EXPECT_EQ(F(3) * F(4), F(12));
  EXPECT_EQ(F(5) - F(5), F(0));
}

}  // namespace
