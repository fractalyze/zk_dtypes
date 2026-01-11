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
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

namespace zk_dtypes {
namespace {

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

BENCHMARK_TEMPLATE(BM_Mul, Mersenne31, 10);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31, 100);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31, 1000);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31Std, 10);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31Std, 100);
BENCHMARK_TEMPLATE(BM_Mul, Mersenne31Std, 1000);
BENCHMARK_TEMPLATE(BM_Mul, Babybear, 10);
BENCHMARK_TEMPLATE(BM_Mul, Babybear, 100);
BENCHMARK_TEMPLATE(BM_Mul, Babybear, 1000);
BENCHMARK_TEMPLATE(BM_Mul, BabybearStd, 10);
BENCHMARK_TEMPLATE(BM_Mul, BabybearStd, 100);
BENCHMARK_TEMPLATE(BM_Mul, BabybearStd, 1000);

}  // namespace
}  // namespace zk_dtypes

// clang-format off
// -----------------------------------------------------------------------------
// 2026-01-11T10:37:24+00:00
// Run on (32 X 5564.26 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 0.09, 0.19, 0.17
// ----------------------------------------------------------------------
// Benchmark                            Time             CPU   Iterations
// ----------------------------------------------------------------------
// BM_Mul<Mersenne31, 10>            10.6 ns         10.6 ns     65667785
// BM_Mul<Mersenne31, 100>            168 ns          168 ns      4174377
// BM_Mul<Mersenne31, 1000>          1742 ns         1741 ns       401768
// BM_Mul<Mersenne31Std, 10>         6.64 ns         6.64 ns    105335115
// BM_Mul<Mersenne31Std, 100>         131 ns          131 ns      5261470
// BM_Mul<Mersenne31Std, 1000>       1389 ns         1389 ns       503874
// BM_Mul<Babybear, 10>              11.0 ns         11.0 ns     63461560
// BM_Mul<Babybear, 100>              197 ns          197 ns      3548845
// BM_Mul<Babybear, 1000>            2088 ns         2087 ns       335691
// BM_Mul<BabybearStd, 10>           10.7 ns         10.7 ns     65468757
// BM_Mul<BabybearStd, 100>           224 ns          224 ns      3123421
// BM_Mul<BabybearStd, 1000>         2431 ns         2431 ns       287980
// clang-format on
