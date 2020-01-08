// Copyright 2019 Google LLC
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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/benchmark.cc"

// First time this file is included: include headers as usual.
#include <stddef.h>
#include <stdio.h>

#include <cmath>
#include <memory>
#include <new>
#include <numeric>  // iota

#include "hwy/nanobenchmark.h"
#include "hwy/runtime_dispatch.h"

struct RunBenchmarks {
  // Benchmark relies on this being one, AND the compiler not knowing it.
  HWY_DECLARE(void, (int unpredictable1))
};

int main(int argc, char** /*argv*/) {
  // No buffering (helps if running remotely)
  setvbuf(stdin, nullptr, _IONBF, 0);

  const hwy::TargetBitfield targets;  // supported by this CPU
  // No need to return the bitfield of targets - benchmark already prints each.
  // != 999 is an expression that evaluates to 1, but unknown to the compiler.
  (void)targets.Foreach(RunBenchmarks(), argc != 999);
}

#endif  // HWY_TARGET_INCLUDE
// Generates RunBenchmarks implementations for all targets.
#include "hwy/foreach_target.h"

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

class TwoArray {
 public:
  // Passed to ctor as a value NOT known to the compiler. Must be a multiple of
  // the vector lane count * 8.
  static size_t NumItems() { return 3456; }

  explicit TwoArray(const size_t num_items)
      : a_(new (std::align_val_t(kMaxVectorSize)) float[num_items * 2]),
        b_(a_.get() + num_items) {
    const float init = num_items / NumItems();  // 1, but compiler doesn't know
    std::iota(a_.get(), a_.get() + num_items, init);
    std::iota(b_, b_ + num_items, init);
  }

 protected:
  std::unique_ptr<float[]> a_;
  float* b_;
};

// Calls operator() of the given closure (lambda function).
template <class Closure>
static FuncOutput CallClosure(const Closure* f, const FuncInput input) {
  return (*f)(input);
}

// Same as Measure, except "closure" is typically a lambda function of
// FuncInput -> FuncOutput with a capture list.
template <class Closure>
static inline size_t MeasureClosure(const Closure& closure,
                                    const FuncInput* inputs,
                                    const size_t num_inputs, Result* results,
                                    const Params& p = Params()) {
  return Measure(reinterpret_cast<Func>(&CallClosure<Closure>),
                 reinterpret_cast<const uint8_t*>(&closure), inputs, num_inputs,
                 results, p);
}

// Measures durations, verifies results, prints timings.
template <class Benchmark>
void RunBenchmark(const char* caption, const int unpredictable1) {
  printf("%10s: ", caption);
  const size_t kNumInputs = 1;
  const uint64_t inputs[kNumInputs] = {Benchmark::NumItems() * unpredictable1};
  Result results[kNumInputs];

  const size_t num_items = Benchmark::NumItems() * unpredictable1;
  Benchmark benchmark(num_items);

  Params p;
  p.verbose = false;
  p.max_evals = 7;
  p.target_rel_mad = 0.002;
  const size_t num_results = MeasureClosure(
      [&benchmark](const uint64_t input) { return benchmark(input); }, inputs,
      kNumInputs, results, p);
  if (num_results != kNumInputs) {
    fprintf(stderr, "MeasureClosure failed.\n");
  }

  benchmark.Verify(num_items);

  for (size_t i = 0; i < num_results; ++i) {
    const double cycles_per_item = results[i].ticks / results[i].input;
    const double mad = results[i].variability * cycles_per_item;
    printf("%6zu: %6.3f (+/- %5.3f)\n", results[i].input, cycles_per_item, mad);
  }
}

HWY_ATTR void Intro() {
  HWY_ALIGN const float in[16] = {1, 2, 3, 4, 5, 6};
  HWY_ALIGN float out[16];
  HWY_FULL(float) d;  // largest possible vector
  for (size_t i = 0; i < 16; i += d.N) {
    const auto vec = Load(d, in + i);  // aligned!
    auto result = vec * vec;
    result += result;  // can update if not const
    Store(result, d, out + i);
  }
  printf("F(x)->2*x^2, F(%.0f) = %.1f\n", in[2], out[2]);
}

// BEGINNER: dot product
// 0.4 cyc/float = bronze, 0.25 = silver, 0.15 = gold!
class BenchmarkDot : public TwoArray {
 public:
  explicit BenchmarkDot(size_t num_items) : TwoArray(num_items), dot_{-1.0f} {}

  HWY_ATTR uint64_t operator()(const size_t num_items) {
    HWY_FULL(float) d;
    using V = decltype(Zero(d));
    constexpr int unroll = 8;
    // Compiler doesn't make independent sum* accumulators, so unroll manually.
    // Some older compilers might not be able to fit the 8 arrays in registers,
    // so manual unrolling can be helpfull if you run into this issue.
    // 2 FMA ports * 4 cycle latency = 8x unrolled.
    V sum[unroll];
    for (int i = 0; i < unroll; ++i) {
      sum[i] = Zero(d);
    }
    const float* const HWY_RESTRICT pa = &a_[0];
    const float* const HWY_RESTRICT pb = b_;
    for (size_t i = 0; i < num_items; i += unroll * d.N) {
      for (int j = 0; j < unroll; ++j) {
        const auto a = Load(d, pa + i + j * d.N);
        const auto b = Load(d, pb + i + j * d.N);
        sum[j] = MulAdd(a, b, sum[j]);
      }
    }
    // Reduction tree: sum of all accumulators by pairs into sum[0], then the
    // lanes.
    for (int power = 1; power < unroll; power *= 2) {
      for (int i = 0; i < unroll; i += 2 * power) {
        sum[i] += sum[i + power];
      }
    }
    return dot_ = GetLane(ext::SumOfLanes(sum[0]));
  }
  void Verify(size_t num_items) {
    if (dot_ == -1.0f) {
      fprintf(stderr, "Dot: must call Verify after benchmark");
      abort();
    }

    const float expected =
        std::inner_product(a_.get(), a_.get() + num_items, b_, 0.0f);
    const float rel_err = std::abs(expected - dot_) / expected;
    if (rel_err > 1.1E-6f) {
      fprintf(stderr, "Dot: expected %e actual %e (%e)\n", expected, dot_,
              rel_err);
      abort();
    }
  }

 private:
  float dot_;  // for Verify
};

// INTERMEDIATE: delta coding
// 1.0 cycles/float = bronze, 0.7 = silver, 0.4 = gold!
struct BenchmarkDelta : public TwoArray {
  explicit BenchmarkDelta(size_t num_items) : TwoArray(num_items) {}

  HWY_ATTR uint64_t operator()(const size_t num_items) const {
#if HWY_BITS == 0
    b_[0] = a_[0];
    for (size_t i = 1; i < num_items; ++i) {
      b_[i] = a_[i] - a_[i - 1];
    }
#elif HWY_BITS == 128
    // Slightly better than unaligned loads
    const HWY_CAPPED(float, 4) df;
    const HWY_CAPPED(int32_t, 4) di;
    size_t i;
    b_[0] = a_[0];
    for (i = 1; i < df.N; ++i) {
      b_[i] = a_[i] - a_[i - 1];
    }
    auto prev = Load(df, &a_[0]);
    for (; i < num_items; i += df.N) {
      const auto a = Load(df, &a_[i]);
      constexpr int kBytes = (df.N - 1) * sizeof(float);
      const auto shifted = BitCast(df, CombineShiftRightBytes<kBytes>(
                                           BitCast(di, a), BitCast(di, prev)));
      prev = a;
      Store(a - shifted, df, &b_[i]);
    }
#else
    // Larger vectors are split into 128-bit blocks, easiest to use the
    // unaligned load support to shift between them.
    const HWY_FULL(float) df;
    size_t i;
    b_[0] = a_[0];
    for (i = 1; i < df.N; ++i) {
      b_[i] = a_[i] - a_[i - 1];
    }
    for (; i < num_items; i += df.N) {
      const auto a = Load(df, &a_[i]);
      const auto shifted = LoadU(df, &a_[i - 1]);
      Store(a - shifted, df, &b_[i]);
    }
#endif
    return b_[num_items - 1];
  }

  void Verify(size_t num_items) {
    for (size_t i = 0; i < num_items; ++i) {
      const float expected = (i == 0) ? a_[0] : a_[i] - a_[i - 1];
      const float err = std::abs(expected - b_[i]);
      if (err > 1E-6f) {
        fprintf(stderr, "Delta: expected %e, actual %e\n", expected, b_[i]);
      }
    }
  }
};

void RunBenchmarks(int unpredictable1) {
  Intro();
  printf("------------------------ %d-bit vectors\n", HWY_BITS);
  RunBenchmark<BenchmarkDot>("dot", unpredictable1);
  RunBenchmark<BenchmarkDelta>("delta", unpredictable1);
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

void RunBenchmarks::HWY_FUNC(int unpredictable1) {
  hwy::HWY_NAMESPACE::RunBenchmarks(unpredictable1);
}
