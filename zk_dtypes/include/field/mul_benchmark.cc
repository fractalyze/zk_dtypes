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

#include "benchmark/benchmark.h"

#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {
namespace {

struct Mersenne31MontConfig : public Mersenne31BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Mersenne31Config;

  constexpr static uint32_t kRSquared = 4;
  constexpr static uint32_t kNPrime = 2147483647;

  constexpr static uint32_t kOne = 2;
};

using Mersenne31Mont = PrimeField<Mersenne31MontConfig>;

template <typename T, size_t N>
void BM_Mul(benchmark::State& state) {
  for (auto _ : state) {
    T prod(3);
    for (int i = 0; i < N; ++i) {
      prod = prod * prod;
    }
    if constexpr (T::kUseMontgomery) {
      auto prod_std = prod.MontReduce();
      benchmark::DoNotOptimize(prod_std);
    } else {
      benchmark::DoNotOptimize(prod);
    }
  }
}

BENCHMARK_TEMPLATE(BM_Mul, Mersenne31Mont, 10);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31Mont, 100);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31Mont, 1000);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31, 10);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31, 100);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31, 1000);
BENCHMARK_TEMPLATE(BM_Mul, Babybear, 10);
BENCHMARK_TEMPLATE(BM_Mul, Babybear, 100);
BENCHMARK_TEMPLATE(BM_Mul, Babybear, 1000);
BENCHMARK_TEMPLATE(BM_Mul, BabybearStd, 10);
BENCHMARK_TEMPLATE(BM_Mul, BabybearStd, 100);
BENCHMARK_TEMPLATE(BM_Mul, BabybearStd, 1000);
BENCHMARK_TEMPLATE(BM_Mul, Goldilocks, 10);
BENCHMARK_TEMPLATE(BM_Mul, Goldilocks, 100);
BENCHMARK_TEMPLATE(BM_Mul, Goldilocks, 1000);
BENCHMARK_TEMPLATE(BM_Mul, GoldilocksStd, 10);
BENCHMARK_TEMPLATE(BM_Mul, GoldilocksStd, 100);
BENCHMARK_TEMPLATE(BM_Mul, GoldilocksStd, 1000);

}  // namespace
}  // namespace zk_dtypes

// clang-format off
// -----------------------------------------------------------------------------
// 2026-01-24T09:27:54+00:00
// Run on (32 X 5570.73 MHz CPU s)
// CPU Caches:
  // L1 Data 48 KiB (x16)
  // L1 Instruction 32 KiB (x16)
  // L2 Unified 1024 KiB (x16)
  // L3 Unified 98304 KiB (x2)
// Load Average: 0.18, 0.38, 1.22
// ----------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations
// ----------------------------------------------------------------------
// BM_Mul<Mersenne31Mont, 10>        10.7 ns         10.7 ns     65491568
// BM_Mul<Mersenne31Mont, 100>        168 ns          168 ns      4168746
// BM_Mul<Mersenne31Mont, 1000>      1743 ns         1742 ns       402034
// BM_Mul<Mersenne31, 10>            6.65 ns         6.65 ns    105071208
// BM_Mul<Mersenne31, 100>            131 ns          130 ns      5347122
// BM_Mul<Mersenne31, 1000>          1394 ns         1393 ns       503957
// BM_Mul<Babybear, 10>              11.0 ns         11.0 ns     63561717
// BM_Mul<Babybear, 100>              197 ns          197 ns      3552128
// BM_Mul<Babybear, 1000>            2085 ns         2085 ns       335703
// BM_Mul<BabybearStd, 10>           10.7 ns         10.7 ns     65445989
// BM_Mul<BabybearStd, 100>           224 ns          224 ns      3122550
// BM_Mul<BabybearStd, 1000>         2430 ns         2430 ns       288461
// BM_Mul<Goldilocks, 10>            9.88 ns         9.88 ns     70806018
// BM_Mul<Goldilocks, 100>            180 ns          180 ns      3893941
// BM_Mul<Goldilocks, 1000>          1910 ns         1910 ns       366742
// BM_Mul<GoldilocksStd, 10>         10.3 ns         10.3 ns     68124030
// BM_Mul<GoldilocksStd, 100>         172 ns          172 ns      4057025
// BM_Mul<GoldilocksStd, 1000>       1815 ns         1815 ns       382557
// clang-format on
