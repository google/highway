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

#include <stdint.h>
#include <stdio.h>

#include <vector>
#include <algorithm>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/algo/algo_bench.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// After foreach_target
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/tests/test_util-inl.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
constexpr bool kBenchUnique = true;
constexpr bool kBenchAllUnique = true;

// copied from sort
enum class BenchmarkModes {
  kDefault,
  k1M,
  k10K,
  kAllSmall,
  kSmallPow2,
  kSmallPow2Between,  // includes padding
  kPow4,
  kPow10
};
std::vector<size_t> SizesToBenchmark(BenchmarkModes mode) {
  std::vector<size_t> sizes;
  switch (mode) {
    case BenchmarkModes::kDefault:
      sizes.push_back(100);
      sizes.push_back(100 * 1000);
      break;
    case BenchmarkModes::k1M:
      sizes.push_back(1000 * 1000);
      break;
    case BenchmarkModes::k10K:
      sizes.push_back(10 * 1000);
      break;

    case BenchmarkModes::kAllSmall:
      sizes.reserve(128);
      for (size_t i = 1; i <= 128; ++i) {
        sizes.push_back(i);
      }
      break;
    case BenchmarkModes::kSmallPow2:
      for (size_t size = 2; size <= 128; size *= 2) {
        sizes.push_back(size);
      }
      break;
    case BenchmarkModes::kSmallPow2Between:
      for (size_t size = 2; size <= 128; size *= 2) {
        sizes.push_back(3 * size / 2);
      }
      break;
    case BenchmarkModes::kPow4:
      for (size_t size = 4; size <= 256 * 1024; size *= 4) {
        sizes.push_back(size);
      }
      break;
    case BenchmarkModes::kPow10:
      for (size_t size = 10; size <= 100 * 1000; size *= 10) {
        sizes.push_back(size);
      }
      break;
  }
  return sizes;
}
double SummarizeMeasurements(std::vector<double>& ns) {
  std::sort(ns.begin(), ns.end());
  double sum = 0;
  int count = 0;
  const size_t num = ns.size();
  for (size_t i = num / 4; i < num / 2; ++i) {
    sum += ns[i];
    count += 1;
  }
  return sum / count;
}

constexpr size_t kElemsPerInnerRep = 5000000;
constexpr size_t kInnerRepsMax = 1000;
constexpr size_t kOuterReps = 50;

// in a single benchmark run, take the mean of inner_reps = min(kInnerRepsMax, max(kElemsPerInnerRep / size, 3))
// and such runs: kOuterReps

template <class Gen>
struct Ctx {
  Gen gen;
  RandomState rng;
  size_t size; // every algo function has a size field
  size_t inner_reps;
};

// A generator is a functor, all kParams variants are run. Its fields are:
// kName = the name in the output string
// PrintParam = the parameter output format
// kParams = the parameter list
// operator() = generates input. It receives Ctx, the pointer and p - parameter

// A measured function is a functor. Its fields are:
// kName = the name in the output string
// operator() = runs the function. Generates data via ctx.gen and measures the total
// time of inner_reps runs. Receives: Ctx, the array of pointers and p - generator parameter

// Benchmarks are run via BenchAllTypes(Func{}, Gen{}, size)

template <class T, class Func, class Gen>
void Bench(Func func, Gen gen, size_t size) {
  RandomState rng(static_cast<uint64_t>(Unpredictable1() * 42));

  const size_t inner_reps = HWY_MIN(kInnerRepsMax, HWY_MAX(kElemsPerInnerRep / size, size_t{3}));

  std::vector<AlignedFreeUniquePtr<T[]>> ptrs(inner_reps);
  Ctx<Gen> ctx{gen, rng, size, inner_reps};

  const ScalableTag<T> d;
  const size_t N = Lanes(d);

  for (size_t i = 0; i < inner_reps; ++i) {
    ptrs[i] = AllocateAligned<T>(size + N);
  }

  std::vector<T*> aligned(inner_reps);
  for (size_t i = 0; i < ctx.inner_reps; ++i) {
    aligned[i] = ptrs[i].get();
  }
  for (size_t p : gen.kParams) {
    std::vector<double> ns;
    for (size_t rep = 0; rep < kOuterReps; ++rep) {
      ns.push_back(func(ctx, aligned, p));
    }
    printf("%s: %9s: %12s", TargetName(DispatchedTarget()), func.kName, gen.kName);
    Gen::PrintParam(p);
    printf(":%4s %7zu", hwy::TypeName(T(), 1).c_str(), size);
    double time = SummarizeMeasurements(ns) / static_cast<double>(ctx.inner_reps);
    printf(" %10.1f ns, %10f GB/s\n", time, static_cast<double>(ctx.gen.ElemsProcessed(size, p)) * sizeof(T) / time);
  }
}

template <class Func, class Gen>
HWY_NOINLINE void BenchAllTypes(Func func, Gen gen, size_t size) {
  Bench<uint8_t>(func, gen, size);
  Bench<int16_t>(func, gen, size);
  Bench<int32_t>(func, gen, size);
  Bench<int64_t>(func, gen, size);
}

// p = probability in % that two adjacent elements are equal
struct RandomRuns01 {
  static constexpr const char* kName = "random-runs";
  static void PrintParam(size_t p) {
    printf("%14zu%%", p);
  }
  static constexpr const size_t kParams[] = {0, 10, 20, 50, 80, 90, 100};
  template <class T>
  void operator()(Ctx<RandomRuns01>& ctx, T* HWY_RESTRICT aligned, size_t p) const {
    T prev = 0;

    for (size_t i = 0; i < ctx.size; ++i) {
      aligned[i] = static_cast<T>(prev = (ctx.rng() % 100 >= p) ? static_cast<T>(1 - prev) : prev);
    }
  }
  size_t ElemsProcessed(size_t n, size_t /*p*/) const {
    return n;
  }
};
// p = period with which blocks of 1 and 0 alternate
struct FixedRuns01 {
  static constexpr const char* kName = "fixed-runs";
  static void PrintParam(size_t p) {
    printf("   period =%4zu", p); // at all padding = 15, same as RandomRuns01
  }
  static constexpr const size_t kParams[] = {1, 31, 63, 127, 255, 511};
  template <class T>
  void operator()(Ctx<FixedRuns01>& ctx, T* HWY_RESTRICT aligned, size_t p) const {
    T prev = 0;

    for (size_t i = 0; i < ctx.size; ++i) {
      if (i % p == p - 1) prev = static_cast<T>(1 - prev);
      aligned[i] = prev;
    }
  }
  size_t ElemsProcessed(size_t n, size_t /*p*/) const {
    return n;
  }
};

// the first p% of the array alternates 1 and 0, and then the same element
struct AltPrefixThenConstTail {
  static void PrintParam(size_t p) {
    printf("%14zu%%", p);
  }

  static constexpr const char* kName = "alt-prefix";
  static constexpr size_t kParams[] = {25, 50, 75, 100};
  template <class T>
  void operator()(Ctx<AltPrefixThenConstTail>& ctx, T* HWY_RESTRICT aligned, size_t p) const {
    size_t idx = ctx.size * p / 100;
    T prev = 0;
    for (size_t i = 0; i < ctx.size; ++i) {
      prev = (i < idx) ? static_cast<T>(1 - prev) : prev;
      aligned[i] = prev;
    }
  }
  size_t ElemsProcessed(size_t n, size_t p) const {
    return n * p / 100;
  }
};

struct BenchUnique {
  static constexpr const char* kName = "unique";
  template <class Gen, class T>
  double operator()(Ctx<Gen>& ctx, std::vector<T*>& aligned, size_t p) const {
    const ScalableTag<T> d;
    for (size_t i = 0; i < ctx.inner_reps; ++i) {
      ctx.gen(ctx, aligned[i], p);
    }
    const Timestamp t0;
    size_t total = 0;
    for (size_t i = 0; i < ctx.inner_reps; ++i) {
      total += hwy::HWY_NAMESPACE::Unique(d, aligned[i], ctx.size);
    }
    double res = SecondsSince(t0);
    PreventElision(total);
    return res * 1e9;
  }
};

struct BenchAllUnique {
  static constexpr const char* kName = "AllUnique";

  template <class Gen, class T>
  double operator()(Ctx<Gen>& ctx, std::vector<T*>& aligned, size_t p) const {
    const ScalableTag<T> d;
    for (size_t i = 0; i < ctx.inner_reps; ++i) {
      ctx.gen(ctx, aligned[i], p);
    }
    const Timestamp t0;
    size_t total = 0;
    for (size_t i = 0; i < ctx.inner_reps; ++i) {
      total += hwy::HWY_NAMESPACE::AllUnique(d, aligned[i], ctx.size);
    }
    double res = SecondsSince(t0);
    PreventElision(total);
    return res * 1e9;
  }
};

HWY_NOINLINE void BenchAll() {
  for (size_t size : SizesToBenchmark(BenchmarkModes::kSmallPow2)) {
    if (kBenchUnique) {
      BenchAllTypes(BenchUnique{}, RandomRuns01{}, size);
      BenchAllTypes(BenchUnique{}, FixedRuns01{}, size);
    }
    if (kBenchAllUnique) {
      BenchAllTypes(BenchAllUnique{}, AltPrefixThenConstTail{}, size);
    }
  }
}
}  // namespace 
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(AlgoBench);
HWY_EXPORT_AND_TEST_P(AlgoBench, BenchAll);
HWY_AFTER_TEST();
}  // namespace hwy

HWY_TEST_MAIN();
#endif  // HWY_ONCE
