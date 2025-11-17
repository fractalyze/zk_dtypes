#include "gtest/gtest.h"

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq2.h"

namespace zk_dtypes {
namespace {

template <typename T>
class Fq2TypedTest : public testing::Test {};

using Fq2Types = testing::Types<bn254::Fq2, bn254::Fq2Std>;
TYPED_TEST_SUITE(Fq2TypedTest, Fq2Types);

TYPED_TEST(Fq2TypedTest, Operations) {
  using ExtF = TypeParam;
  using BaseF = typename ExtF::BaseField;

  // clang-format off
  ExtF a = {
      *BaseF::FromHexString(
          "0xb94db59332f8a619901d39188315c421beafb516eb8a3ab56ceed7df960ede2"),
      *BaseF::FromHexString(
          "0xecec51689891ef3f7ff39040036fd0e282687d392abe8f011589f1c755500a2"),
  };
  ExtF b = {
      *BaseF::FromHexString(
          "0x2fd5bb90b6043853a651f3e930e7c549f448e5562cbc823bbe2a84a67830e1d"),
      *BaseF::FromHexString(
          "0x284f3562b7cda2c9f329ac7019d4778a9bfb820f4639687e4573eadfed505d97"),
  };
  EXPECT_EQ(a + b, ExtF({
      *BaseF::FromHexString(
          "0xe9237123e8fcde6d366f2d01b3fd896bb2f89a6d1846bcf12b195c860e3fbff"),
      *BaseF::FromHexString(
          "0x6b9ac066025219432d89fbd988a1c3b2ca09f51707386e11aabfde58a2860f2"),
  }));
  EXPECT_EQ(a.Double(), ExtF({
      *BaseF::FromHexString(
          "0x1729b6b2665f14c33203a7231062b88437d5f6a2dd714756ad9ddafbf2c1dbc4"),
      *BaseF::FromHexString(
          "0x1d9d8a2d13123de7effe7208006dfa1c504d0fa72557d1e022b13e38eaaa0144"),
  }));
  EXPECT_EQ(a - b, ExtF({
      *BaseF::FromHexString(
          "0x8977fa027cf46dc5e9cb452f522dfed7ca66cfc0becdb879aec453391dddfc5"),
      *BaseF::FromHexString(
          "0x16e3de26b2ed1c53bd25d24a67e3dde123ac7055b4e44aff080540536081a052"),
  }));
  EXPECT_EQ(a * b, ExtF({
      *BaseF::FromHexString(
          "0x25908f85559394401b8ced51130b426c319a22feb86b4b36e2b5aa6de4ab5ebf"),
      *BaseF::FromHexString(
          "0x283abe3026369b6da48074048233145a1c88dce48308e952c948eff8e8f0bc51"),
  }));
  // clang-format on
}

template <typename T>
class ExtensionFieldTypedTest : public testing::Test {};

using ExtensionFieldTypes = testing::Types<bn254::Fq2>;
TYPED_TEST_SUITE(ExtensionFieldTypedTest, ExtensionFieldTypes);

TYPED_TEST(ExtensionFieldTypedTest, Zero) {
  using ExtF = TypeParam;

  EXPECT_TRUE(ExtF::Zero().IsZero());
  EXPECT_FALSE(ExtF::One().IsZero());
}

TYPED_TEST(ExtensionFieldTypedTest, One) {
  using ExtF = TypeParam;

  EXPECT_TRUE(ExtF::One().IsOne());
  EXPECT_FALSE(ExtF::Zero().IsOne());
}

TYPED_TEST(ExtensionFieldTypedTest, SquareRoot) {
  using ExtF = TypeParam;

  ExtF a = ExtF::Random();
  ExtF a2 = a.Square();
  absl::StatusOr<ExtF> sqrt = a2.SquareRoot();
  ASSERT_TRUE(sqrt.ok());
  EXPECT_TRUE(a == (*sqrt) || a == -(*sqrt));
}

TYPED_TEST(ExtensionFieldTypedTest, Inverse) {
  using ExtF = TypeParam;

  ExtF a = ExtF::Random();
  while (a.IsZero()) {
    a = ExtF::Random();
  }
  absl::StatusOr<ExtF> a_inverse = a.Inverse();
  ASSERT_TRUE(a_inverse.ok());
  EXPECT_TRUE((a * (*a_inverse)).IsOne());
}

TYPED_TEST(ExtensionFieldTypedTest, MontReduce) {
  using ExtF = TypeParam;
  using ExtFStd = typename ExtF::StdType;
  using BaseF = typename ExtF::BaseField;

  ExtF a = {BaseF(1), BaseF(2)};
  ExtFStd reduced = a.MontReduce();

  EXPECT_EQ(reduced[0], BaseF(1).MontReduce());
  EXPECT_EQ(reduced[1], BaseF(2).MontReduce());
}

}  // namespace
}  // namespace zk_dtypes
