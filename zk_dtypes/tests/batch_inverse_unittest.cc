#include "zk_dtypes/include/batch_inverse.h"

#include <vector>

#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"

namespace zk_dtypes::bn254 {
namespace {

class BatchInverseTest : public testing::Test {
 public:
  void SetUp() override {
    inputs_.push_back(*Fr::FromHexString(
        "0xb94db59332f8a619901d39188315c421beafb516eb8a3ab56ceed7df960ede2"));
    inputs_.push_back(*Fr::FromHexString(
        "0xecec51689891ef3f7ff39040036fd0e282687d392abe8f011589f1c755500a2"));
    answers_.push_back(*inputs_[0].Inverse());
    answers_.push_back(*inputs_[1].Inverse());
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

TEST_F(BatchInverseTest, WithZero) {
  inputs_.push_back(Fr::Zero());
  ASSERT_TRUE(BatchInverse(inputs_, &inputs_).ok());
  answers_.push_back(Fr::Zero());
  EXPECT_EQ(inputs_, answers_);
}

TEST_F(BatchInverseTest, WithCoeff) {
  ASSERT_TRUE(BatchInverse(inputs_, &inputs_, Fr(2)).ok());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    EXPECT_EQ(inputs_[i], answers_[i].Double());
  }
}

}  // namespace
}  // namespace zk_dtypes::bn254
