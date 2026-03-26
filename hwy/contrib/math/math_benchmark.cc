// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_benchmark.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/math/fast_math-inl.h"
#include "hwy/nanobenchmark.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

#include <stdio.h>

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Helper to safely convert float/double results to benchmark output
// without triggering SIGILL on Infinity or compile errors on size mismatch.
template <class Val>
hwy::FuncOutput SafeCast(Val v) {
  using BitsT = hwy::MakeUnsigned<Val>;
  auto bits = hwy::BitCastScalar<BitsT>(v);
  return static_cast<hwy::FuncOutput>(bits);
}

// Macro to define benchmark body
#define DEFINE_MATH_BENCH(NAME, FUNC, MAP_EXPR)                             \
  template <class D>                                                        \
  void Bench##NAME(D d) {                                                   \
    using T = hn::TFromD<D>;                                                \
    printf("Benchmarking " #NAME " for %s:\n",                              \
           hwy::TypeName(T(), hn::Lanes(d)).c_str());                       \
    auto func = [d](const hwy::FuncInput in) -> hwy::FuncOutput {           \
      const double val = MAP_EXPR;                                          \
      const auto v = hn::Set(d, static_cast<T>(val));                       \
      const auto res = FUNC;                                                \
      return SafeCast(hn::GetLane(res));                                    \
    };                                                                      \
    const size_t kNumInputs = 16;                                           \
    hwy::FuncInput inputs[kNumInputs];                                      \
    for (size_t i = 0; i < kNumInputs; ++i) inputs[i] = i;                  \
    hwy::Result results[kNumInputs];                                        \
    hwy::Params params = hwy::DefaultBenchmarkParams();                     \
    const size_t num_results =                                              \
        hwy::MeasureClosure(func, inputs, kNumInputs, results, params);     \
    double sum_ticks = 0;                                                   \
    for (size_t i = 0; i < num_results; ++i) sum_ticks += results[i].ticks; \
    if (num_results > 0) {                                                  \
      printf("  Avg ticks: %f\n", sum_ticks / num_results);                 \
    }                                                                       \
  }

// EXP
DEFINE_MATH_BENCH(CallFastExp, hn::CallFastExp(d, v),
                  -10.0 + static_cast<double>(in) * (20.0 / 15.0))
DEFINE_MATH_BENCH(CallExp, hn::CallExp(d, v),
                  -10.0 + static_cast<double>(in) * (20.0 / 15.0))

struct RunBenchmarks {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    BenchCallFastExp(d);
    BenchCallExp(d);
  }
};

HWY_NOINLINE void RunAllBenchmarks() {
  hn::ForFloat3264Types(hn::ForPartialVectors<RunBenchmarks>());
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(RunAllBenchmarks);
}  // namespace
}  // namespace hwy

int main(int argc, char** argv) {
  HWY_DYNAMIC_DISPATCH(hwy::RunAllBenchmarks)();
  return 0;
}
#endif  // HWY_ONCE
