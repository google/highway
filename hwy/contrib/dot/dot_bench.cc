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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hwy/aligned_allocator.h"
#include "hwy/nanobenchmark.h"
#include "hwy/per_target.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/dot/dot_bench.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

constexpr size_t kCounts[] = {256, 1024, 16 * 1024, 256 * 1024, 1024 * 1024};

HWY_INLINE float RandomFloat(RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  return static_cast<float>(bits - 512) * (1.0f / 64.0f);
}

template <class T>
HWY_INLINE T RandomValue(RandomState& rng) {
  return ConvertScalarTo<T>(RandomFloat(rng));
}

template <>
HWY_INLINE int16_t RandomValue<int16_t>(RandomState& rng) {
  const int32_t bits = static_cast<int32_t>(Random32(&rng)) & 1023;
  return static_cast<int16_t>(bits - 512);
}

template <class T>
HWY_INLINE FuncOutput ToOutput(T value) {
  if constexpr (IsFloat<T>()) {
    using BitsT = MakeUnsigned<T>;
    const BitsT bits = BitCastScalar<BitsT>(value);
    return static_cast<FuncOutput>(bits);
  } else {
    return static_cast<FuncOutput>(value);
  }
}

template <class TA, class TB>
void FillArrays(TA* HWY_RESTRICT a, TB* HWY_RESTRICT b, size_t count,
                RandomState& rng) {
  for (size_t i = 0; i < count; ++i) {
    a[i] = RandomValue<TA>(rng);
    b[i] = RandomValue<TB>(rng);
  }
}

template <class D, class TA, class TB>
HWY_NOINLINE void BenchDotType(const char* label, D d) {
  using ResultT = decltype(Dot::Compute<0>(d, (const TA*)nullptr,
                                          (const TB*)nullptr, 0));

  const size_t max_count = kCounts[sizeof(kCounts) / sizeof(kCounts[0]) - 1];
  auto a = AllocateAligned<TA>(max_count);
  auto b = AllocateAligned<TB>(max_count);
  HWY_ASSERT(a && b);

  RandomState rng(static_cast<uint64_t>(Unpredictable1()) *
                  0x9E3779B97F4A7C15ULL);
  FillArrays(a.get(), b.get(), max_count, rng);

  const size_t N = Lanes(d);
  (void)N;
  printf("Dot %s (%s):\n", label, TypeName(TA(), Lanes(d)).c_str());

  Params params = DefaultBenchmarkParams();
  params.verbose = false;
  Result results[1];
  FuncInput input = Unpredictable1();

  for (size_t count : kCounts) {
    const double bytes =
        static_cast<double>(count) * (sizeof(TA) + sizeof(TB));

    const size_t num_results_generic = MeasureClosure(
        [&](FuncInput) {
          const ResultT sum = Dot::Compute<0>(d, a.get(), b.get(), count);
          return ToOutput(sum);
        },
        &input, 1, results, params);

    double gbps_generic = 0.0;
    double mad_generic = 0.0;
    if (num_results_generic == 1) {
      const double seconds =
          results[0].ticks / platform::InvariantTicksPerSecond();
      gbps_generic = bytes / seconds * 1E-9;
      mad_generic = results[0].variability * 100.0;
    }

    const bool is_multiple = (count % N) == 0;
    double gbps_opt = 0.0;
    double mad_opt = 0.0;
    if (is_multiple) {
      const size_t num_results_opt = MeasureClosure(
          [&](FuncInput) {
            const ResultT sum = Dot::Compute<Dot::kMultipleOfVector>(
                d, a.get(), b.get(), count);
            return ToOutput(sum);
          },
          &input, 1, results, params);

      if (num_results_opt == 1) {
        const double seconds =
            results[0].ticks / platform::InvariantTicksPerSecond();
        gbps_opt = bytes / seconds * 1E-9;
        mad_opt = results[0].variability * 100.0;
      }
    }

    if (num_results_generic != 1) {
      HWY_WARN("Measurement failed for %s count=%zu", label, count);
      continue;
    }

    if (is_multiple && gbps_opt > 0.0) {
      printf(
          "  %8zu elems: gen %7.2f GB/s (MAD %4.2f%%) | mult %7.2f GB/s (MAD "
          "%4.2f%%)\n",
          count, gbps_generic, mad_generic, gbps_opt, mad_opt);
    } else {
      printf("  %8zu elems: gen %7.2f GB/s (MAD %4.2f%%) | mult N/A\n", count,
             gbps_generic, mad_generic);
    }
  }
}

struct BenchDotSameType {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    BenchDotType<D, T, T>("same", d);
  }
};

struct BenchDotF32BF16 {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    BenchDotType<D, float, bfloat16_t>("f32xbf16", d);
  }
};

struct BenchDotBF16 {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    BenchDotType<D, bfloat16_t, bfloat16_t>("bf16xbf16", d);
  }
};

struct BenchDotI16 {
  template <class T, class D>
  HWY_NOINLINE void operator()(T, D d) {
    BenchDotType<D, int16_t, int16_t>("i16xi16", d);
  }
};

HWY_NOINLINE void BenchAllDot() {
  ForFloat3264Types(ForPartialVectors<BenchDotSameType>());
}

HWY_NOINLINE void BenchAllDotF32BF16() {
  ForPartialVectors<BenchDotF32BF16>()(float());
}

HWY_NOINLINE void BenchAllDotBF16() {
  ForShrinkableVectors<BenchDotBF16>()(bfloat16_t());
}

HWY_NOINLINE void BenchAllDotI16() {
  ForShrinkableVectors<BenchDotI16>()(int16_t());
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(DotBench);
HWY_EXPORT_AND_TEST_BEST_P(DotBench, BenchAllDot);
HWY_EXPORT_AND_TEST_BEST_P(DotBench, BenchAllDotF32BF16);
HWY_EXPORT_AND_TEST_BEST_P(DotBench, BenchAllDotBF16);
HWY_EXPORT_AND_TEST_BEST_P(DotBench, BenchAllDotI16);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
