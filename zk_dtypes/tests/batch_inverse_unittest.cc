/* Copyright 2025 The zk_dtypes Authors.

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

#include "zk_dtypes/include/batch_inverse.h"

#include <vector>

#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"

namespace zk_dtypes {
namespace {

using Fr = test::Fr;

class BatchInverseTest : public testing::Test {
 public:
  void SetUp() override {
    for (size_t i = 0; i < 10; ++i) {
      // NOTE: BatchInverse supports zero values.
      auto value = Fr::Random();
      inputs_.push_back(value);
      if (value.IsZero()) {
        answers_.push_back(Fr::Zero());
      } else {
        answers_.push_back(*value.Inverse());
      }
    }
  }

 protected:
  std::vector<Fr> inputs_;
  std::vector<Fr> answers_;
};

TEST_F(BatchInverseTest, SizeMismatchError) {
  absl::Span<Fr> outputs_span;
  ASSERT_FALSE(BatchInverse(inputs_, &outputs_span).ok());
}

TEST_F(BatchInverseTest, OutOfPlace) {
  std::vector<Fr> outputs;
  ASSERT_TRUE(BatchInverse(inputs_, &outputs).ok());
  EXPECT_EQ(outputs, answers_);
}

TEST_F(BatchInverseTest, InPlace) {
  ASSERT_TRUE(BatchInverse(inputs_, &inputs_).ok());
  EXPECT_EQ(inputs_, answers_);
}

TEST_F(BatchInverseTest, WithCoeff) {
  ASSERT_TRUE(BatchInverse(inputs_, &inputs_, Fr(2)).ok());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    EXPECT_EQ(inputs_[i], answers_[i].Double());
  }
}

}  // namespace
}  // namespace zk_dtypes
