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

#include "zk_dtypes/include/field/runtime_field.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"

namespace zk_dtypes {
namespace {

using modarith::PrimeField;

// The runtime field is only worth having if it agrees with the compile-time
// configs bit for bit — that is what lets a consumer serve an arbitrary modulus
// without changing any curated result. These tests compare against the curated
// types rather than against a reimplementation of modular arithmetic.
template <typename F>
PrimeField RuntimeOf() {
  const auto modulus = F::Config::kModulus;
  unsigned char le[32] = {};
  constexpr int kWidth = sizeof(typename F::UnderlyingType);
  std::memcpy(le, &modulus, kWidth <= 8 ? kWidth : 8);
  return PrimeField::Make(le, kWidth, F::Config::kUseMontgomery);
}

// Stored bytes of a curated element, which is what the runtime path produces.
template <typename F>
std::vector<unsigned char> StoredBytes(const F& v) {
  std::vector<unsigned char> out(sizeof(typename F::UnderlyingType));
  const auto raw = v.value();
  std::memcpy(out.data(), &raw, out.size());
  return out;
}

TEST(RuntimeFieldTest, MulMatchesTheCuratedConfig) {
  const PrimeField rt = RuntimeOf<BabybearMont>();
  ASSERT_TRUE(rt.native);
  for (uint64_t a = 1; a < 40; ++a) {
    for (uint64_t b = 1; b < 40; ++b) {
      const BabybearMont want = BabybearMont(a) * BabybearMont(b);
      unsigned char ea[4], eb[4], got[4];
      const auto ba = StoredBytes(BabybearMont(a));
      const auto bb = StoredBytes(BabybearMont(b));
      std::memcpy(ea, ba.data(), 4);
      std::memcpy(eb, bb.data(), 4);
      rt.Mul(ea, eb, got);
      EXPECT_EQ(0, std::memcmp(got, StoredBytes(want).data(), 4))
          << "a=" << a << " b=" << b;
    }
  }
}

TEST(RuntimeFieldTest, EncodeDecodeRoundTripsThroughTheStorageForm) {
  for (const PrimeField rt :
       {RuntimeOf<BabybearMont>(), RuntimeOf<KoalabearMont>()}) {
    ASSERT_TRUE(rt.native);
    for (uint64_t v = 0; v < 64; ++v) {
      unsigned char canon[4] = {}, stored[4], back[4];
      std::memcpy(canon, &v, 4);
      rt.Encode(canon, stored);
      rt.Decode(stored, back);
      EXPECT_EQ(0, std::memcmp(canon, back, 4)) << "v=" << v;
    }
  }
}

// Encode has to agree with what the curated type stores, or a constant built
// this way would be a different element than the same constant built there.
TEST(RuntimeFieldTest, EncodeMatchesTheCuratedStorage) {
  const PrimeField rt = RuntimeOf<BabybearMont>();
  for (uint64_t v = 0; v < 64; ++v) {
    unsigned char canon[4] = {}, stored[4];
    std::memcpy(canon, &v, 4);
    rt.Encode(canon, stored);
    EXPECT_EQ(0, std::memcmp(stored, StoredBytes(BabybearMont(v)).data(), 4))
        << "v=" << v;
  }
}

TEST(RuntimeFieldTest, OneIsTheMultiplicativeIdentity) {
  for (const PrimeField rt :
       {RuntimeOf<BabybearMont>(), RuntimeOf<KoalabearMont>()}) {
    unsigned char one[4], x[4], got[4];
    rt.One(one);
    for (uint64_t v = 1; v < 32; ++v) {
      unsigned char canon[4] = {};
      std::memcpy(canon, &v, 4);
      rt.Encode(canon, x);
      rt.Mul(x, one, got);
      EXPECT_EQ(0, std::memcmp(got, x, 4)) << "v=" << v;
    }
  }
}

TEST(RuntimeFieldTest, PowMatchesTheCuratedConfig) {
  const PrimeField rt = RuntimeOf<BabybearMont>();
  for (uint64_t base = 2; base < 12; ++base) {
    for (uint64_t e = 0; e < 20; ++e) {
      unsigned char b[4], got[4];
      std::memcpy(b, StoredBytes(BabybearMont(base)).data(), 4);
      rt.Pow(b, e, got);
      const BabybearMont want = BabybearMont(base).Pow(e);
      EXPECT_EQ(0, std::memcmp(got, StoredBytes(want).data(), 4))
          << "base=" << base << " e=" << e;
    }
  }
}

TEST(RuntimeFieldTest, TwoAdicityMatchesTheCuratedConfig) {
  EXPECT_EQ(RuntimeOf<BabybearMont>().TwoAdicity(),
            static_cast<int>(BabybearMont::Config::kTwoAdicity));
  EXPECT_EQ(RuntimeOf<KoalabearMont>().TwoAdicity(),
            static_cast<int>(KoalabearMont::Config::kTwoAdicity));
}

// ML-KEM's 3329 is the reason a runtime field is needed at all: 2-adicity 8
// means it has no 512th root of unity, so a length-256 negacyclic transform is
// arithmetically impossible over it while a length-128 one is fine. No curated
// config covers this modulus.
TEST(RuntimeFieldTest, MlKemTwoAdicityBoundsTheTransformLength) {
  unsigned char le[32] = {};
  const uint32_t q = 3329;
  std::memcpy(le, &q, 4);
  const PrimeField rt = PrimeField::Make(le, 4, /*is_mont=*/true);
  ASSERT_TRUE(rt.native);
  EXPECT_EQ(rt.TwoAdicity(), 8);

  unsigned char root[4];
  EXPECT_TRUE(rt.RootOfUnity(256, /*generator=*/0, root));   // length-128 nega
  EXPECT_FALSE(rt.RootOfUnity(512, /*generator=*/0, root));  // length-256 nega
}

// ML-DSA's 8380417 is the other modulus driving this work, and unlike ML-KEM's
// it has room to spare: 2-adicity 13 covers a length-256 negacyclic transform
// (which needs a 512th root) with seven layers left over. No curated config
// covers it either.
TEST(RuntimeFieldTest, MlDsaSupportsTheFullTransformLength) {
  unsigned char le[32] = {};
  const uint32_t q = 8380417;
  std::memcpy(le, &q, 4);
  const PrimeField rt = PrimeField::Make(le, 4, /*is_mont=*/true);
  ASSERT_TRUE(rt.native);
  EXPECT_EQ(rt.TwoAdicity(), 13);

  unsigned char root[4];
  EXPECT_TRUE(rt.RootOfUnity(512, /*generator=*/0, root));  // length-256 nega
  EXPECT_TRUE(rt.RootOfUnity(8192, /*generator=*/0, root));
  EXPECT_FALSE(rt.RootOfUnity(16384, /*generator=*/0, root));

  // The round trip the two motivating moduli have to survive.
  for (uint64_t v : {1, 2, 1234, 8380416}) {
    unsigned char canon[4] = {}, stored[4], back[4], sq[4];
    const uint32_t v32 = static_cast<uint32_t>(v);
    std::memcpy(canon, &v32, 4);
    rt.Encode(canon, stored);
    rt.Mul(stored, stored, sq);
    rt.Decode(sq, back);
    uint32_t got = 0;
    std::memcpy(&got, back, 4);
    EXPECT_EQ(got, static_cast<uint32_t>((v * v) % q)) << "v=" << v;
  }
}

// A pinned generator is the caller's claim, not a checked fact, so RootOfUnity
// has to reject the ones that produce a plausible but non-primitive value —
// they would silently corrupt every butterfly built on the result.
TEST(RuntimeFieldTest, PinnedGeneratorThatIsNotPrimitiveIsRejected) {
  const PrimeField rt = RuntimeOf<BabybearMont>();
  unsigned char root[4];
  // 1^k == 1 for every k, so it "succeeds" at producing a value of order 1.
  EXPECT_FALSE(rt.RootOfUnity(1024, /*generator=*/1, root));
  // A square has no odd part left to reach order 2^k: 3 is a residue here.
  EXPECT_FALSE(rt.RootOfUnity(1024, /*generator=*/3, root));
  // Congruent to zero mod p, which reduces to 0 rather than running out of
  // range through the kernels.
  EXPECT_FALSE(rt.RootOfUnity(1024, /*generator=*/2013265921, root));
  // n == 1 is the one case where the identity is the right answer.
  EXPECT_TRUE(rt.RootOfUnity(1, /*generator=*/1, root));
}

// A root found by search is a *valid* primitive n-th root, not the specific one
// a curated config stores — any of the phi(n) primitive roots is as correct.
// What must hold is the defining property.
TEST(RuntimeFieldTest, SearchedRootHasExactlyOrderN) {
  for (const PrimeField rt :
       {RuntimeOf<BabybearMont>(), RuntimeOf<KoalabearMont>()}) {
    for (uint64_t n : {2, 4, 8, 256, 1024}) {
      unsigned char root[4], acc[4], one[4];
      ASSERT_TRUE(rt.RootOfUnity(n, /*generator=*/0, root)) << "n=" << n;
      rt.One(one);
      rt.Pow(root, n, acc);
      EXPECT_EQ(0, std::memcmp(acc, one, 4)) << "root^n != 1, n=" << n;
      if (n > 1) {
        rt.Pow(root, n / 2, acc);
        EXPECT_NE(0, std::memcmp(acc, one, 4))
            << "root^(n/2) == 1, so order < n, n=" << n;
      }
    }
  }
}

// Pinning the generator is the reproducible form: the same g and n must give
// the same root every time, which is what a caller needing a *particular* root
// uses instead of the search.
TEST(RuntimeFieldTest, PinnedGeneratorIsDeterministic) {
  const PrimeField rt = RuntimeOf<BabybearMont>();
  unsigned char a[4], b[4];
  ASSERT_TRUE(rt.RootOfUnity(1024, /*generator=*/31, a));
  ASSERT_TRUE(rt.RootOfUnity(1024, /*generator=*/31, b));
  EXPECT_EQ(0, std::memcmp(a, b, 4));

  unsigned char one[4], acc[4];
  rt.One(one);
  rt.Pow(a, uint64_t{1024}, acc);
  EXPECT_EQ(0, std::memcmp(acc, one, 4));
  // a^1024 == 1 alone would also hold for the identity, so the order has to be
  // pinned from below as well.
  rt.Pow(a, uint64_t{512}, acc);
  EXPECT_NE(0, std::memcmp(acc, one, 4));
}

}  // namespace
}  // namespace zk_dtypes
