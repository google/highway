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
//
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "hwy/contrib/thread_pool/index_range.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/examples/stream_triad.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/timer.h"

/*
Highway SIMD Tutorial: Stream triad

This example demonstrates how to combine multithreading and
simd vectorization.

1) https://www.cs.virginia.edu/stream/
2) https://github.com/RRZE-HPC/likwid

*/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

using DF = hn::ScalableTag<float>;
const DF df;
using VF = hn::Vec<DF>;

void StreamTriadScalar(float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                       const float* HWY_RESTRICT c, const float q,
                       const size_t count) {
  for (size_t i = 0; i < count; ++i) {
    a[i] = b[i] + q * c[i];
  }
  return;
}

void StreamTriadSIMD(float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                     const float* HWY_RESTRICT c, const float q,
                     const size_t count) {
  const size_t NF = hn::Lanes(df);
  size_t i = 0;
  const VF qv = hn::Set(df, q);
  // Unroll by 4 to mask latency
  HWY_DASSERT(count >= 4 * NF);
  for (; i <= count - 4 * NF; i += 4 * NF) {
    hn::Store(MulAdd(hn::Load(df, c + i), qv, hn::Load(df, b + i)), df, a + i);
    hn::Store(MulAdd(hn::Load(df, c + i + NF), qv, hn::Load(df, b + i + NF)),
              df, a + i + NF);
    hn::Store(
        MulAdd(hn::Load(df, c + i + 2 * NF), qv, hn::Load(df, b + i + 2 * NF)),
        df, a + i + 2 * NF);
    hn::Store(
        MulAdd(hn::Load(df, c + i + 3 * NF), qv, hn::Load(df, b + i + 3 * NF)),
        df, a + i + 3 * NF);
  }
  for (; i <= count - NF; i += NF) {
    hn::Store(MulAdd(hn::Load(df, c + i), qv, hn::Load(df, b + i)), df, a + i);
  }

  // Use LoadN for remainder, sets 0 for values beyond remainder
  size_t remainder = count - i;
  HWY_DASSERT(remainder < NF);
  if (remainder > 0) {
    hn::StoreN(MulAdd(hn::LoadN(df, c + i, remainder), qv,
                      hn::LoadN(df, b + i, remainder)),
               df, a + i, remainder);
  }
  return;
}

IndexRangePartition StaticSIMDPartition(size_t num_items, size_t workers,
                                        size_t NF) {
  size_t size = hwy::RoundUpTo(hwy::DivCeil(num_items, workers), NF);
  return IndexRangePartition(IndexRange(0, num_items),
                             HWY_MIN(size, num_items));
}

void StreamTriadSIMDThreads(float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                            const float* HWY_RESTRICT c, const float q,
                            const size_t count) {
  const size_t NF = hn::Lanes(df);
  ThreadPool pool(ThreadPool::MaxThreads());
  size_t workers = pool.NumWorkers();
  IndexRangePartition partition = StaticSIMDPartition(count, workers, NF);
  pool.Run(0, workers, [=](uint64_t task, HWY_MAYBE_UNUSED size_t thread) {
    const VF qv = hn::Set(df, q);
    // Use work chunks that are multiples of the number of lane
    const IndexRange range = partition.Range(static_cast<size_t>(task));
    const size_t low = range.begin();
    const size_t high = range.end();
    size_t i = low;
    // Unroll to hide latency
    for (; i <= high - 4 * NF; i += 4 * NF) {
      hn::Store(MulAdd(hn::Load(df, c + i), qv, hn::Load(df, b + i)), df,
                a + i);
      hn::Store(MulAdd(hn::Load(df, c + i + NF), qv, hn::Load(df, b + i + NF)),
                df, a + i + NF);
      hn::Store(MulAdd(hn::Load(df, c + i + 2 * NF), qv,
                       hn::Load(df, b + i + 2 * NF)),
                df, a + i + 2 * NF);
      hn::Store(MulAdd(hn::Load(df, c + i + 3 * NF), qv,
                       hn::Load(df, b + i + 3 * NF)),
                df, a + i + 3 * NF);
    }
    for (; i <= high - NF; i += NF) {
      hn::Store(MulAdd(hn::Load(df, c + i), qv, hn::Load(df, b + i)), df,
                a + i);
    }
    // Use LoadN for remainder, sets 0 for values beyond remainder
    size_t remainder = high - i;
    HWY_DASSERT(remainder < NF);
    if (remainder > 0) {
      hn::StoreN(MulAdd(hn::LoadN(df, c + i, remainder), qv,
                        hn::LoadN(df, b + i, remainder)),
                 df, a + i, remainder);
    }
  });

  return;
}

IndexRangePartition StaticScalarPartition(size_t num_items, size_t workers) {
  size_t size = hwy::DivCeil(num_items, workers);
  return IndexRangePartition(IndexRange(0, num_items),
                             HWY_MIN(size, num_items));
}

void StreamTriadScalarThreads(float* HWY_RESTRICT a,
                              const float* HWY_RESTRICT b,
                              const float* HWY_RESTRICT c, const float q,
                              const size_t count) {
  ThreadPool pool(ThreadPool::MaxThreads());
  size_t workers = pool.NumWorkers();
  IndexRangePartition partition = StaticScalarPartition(count, workers);
  pool.Run(0, workers, [=](uint64_t task, HWY_MAYBE_UNUSED size_t thread) {
    // Divide up work between the tasks
    const IndexRange range = partition.Range(static_cast<size_t>(task));
    size_t low = range.begin();
    size_t high = range.end();
    for (size_t i = low; i < high; ++i) {
      a[i] = b[i] + q * c[i];
    }
  });

  return;
}

bool TriadValidate(const float* HWY_RESTRICT ref,
                   const float* HWY_RESTRICT check, const size_t count,
                   const float threshold) {
  const size_t NF = hn::Lanes(df);
  size_t i = 0;
  VF sumdiff = hn::Zero(df);
  bool ret = false;
  if (count >= NF) {
    for (; i <= count - NF; i += NF) {
      sumdiff = hn::Add(
          sumdiff, hn::AbsDiff(hn::Load(df, ref + i), hn::Load(df, check + i)));
    }
  }

  // Use LoadN for remainder, sets 0 for values beyond remainder
  size_t remainder = count - i;
  HWY_DASSERT(remainder < NF);
  if (remainder > 0) {
    sumdiff =
        hn::Add(sumdiff, hn::AbsDiff(hn::LoadN(df, ref + i, remainder),
                                     hn::LoadN(df, check + i, remainder)));
  }
  if (hn::ReduceSum(df, sumdiff) < threshold) {
    ret = true;
  }
  return ret;
}
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(StreamTriadScalar);
HWY_EXPORT(StreamTriadScalarThreads);
HWY_EXPORT(StreamTriadSIMD);
HWY_EXPORT(StreamTriadSIMDThreads);
HWY_EXPORT(TriadValidate);

void PrintResults(const bool check_validated, const bool validated,
                  const std::string test_type, const double t_0,
                  const double t_1, const size_t count) {
  const double dt = 1000.0 * (t_1 - t_0);
  const double bandwidth =
      3.0 * sizeof(float) * static_cast<double>(count) / (1000.0 * dt);
  if (check_validated) {
    if (validated) {
      std::cout << test_type << " validated" << std::endl;
    } else {
      std::cout << test_type << " validation failed" << std::endl;
    }
  }
  std::cout << test_type << " execution time: " << dt << " ms" << std::endl;
  std::cout << test_type << " bandwidth: " << bandwidth << " MB/s" << std::endl;
}

int Run() {
  const size_t count = 20000025;
  const float threshold = 0.00001f;
  AlignedVector<float> a_scalar(count);
  AlignedVector<float> a_simd(count);
  AlignedVector<float> a_scalar_threads(count);
  AlignedVector<float> a_simd_threads(count);
  AlignedVector<float> b(count);
  AlignedVector<float> c(count);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-10.5f, 10.5f);
  for (size_t i = 0; i < count; ++i) {
    b[i] = dis(gen);
    c[i] = dis(gen);
  }
  int ret = 1;
  const float q = 3.142f;
  // Record start time
  const double t_scalar_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(StreamTriadScalar)(a_scalar.data(), b.data(), c.data(),
                                          q, count);
  // Record end time and print execution time and dot product
  const double t_scalar_1 = hwy::platform::Now();
  PrintResults(false, false, "Scalar", t_scalar_0, t_scalar_1, count);
  // Record start time
  const double t_simd_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(StreamTriadSIMD)(a_simd.data(), b.data(), c.data(), q,
                                        count);
  // Record stop timer and print time
  const double t_simd_1 = hwy::platform::Now();
  const bool simd_validate = HWY_DYNAMIC_DISPATCH(TriadValidate)(
      a_scalar.data(), a_simd.data(), count, threshold);
  PrintResults(true, simd_validate, "SIMD", t_simd_0, t_simd_1, count);
  // Record start time
  const double t_scalar_threads_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(StreamTriadScalar)(a_scalar_threads.data(), b.data(),
                                          c.data(), q, count);
  // Record end time and print execution time and dot product
  const double t_scalar_threads_1 = hwy::platform::Now();
  const bool scalar_threads_validate = HWY_DYNAMIC_DISPATCH(TriadValidate)(
      a_scalar.data(), a_scalar_threads.data(), count, threshold);
  PrintResults(true, scalar_threads_validate, "Threaded scalar",
               t_scalar_threads_0, t_scalar_threads_1, count);
  // Record start time
  const double t_simd_threads_0 = hwy::platform::Now();
  HWY_DYNAMIC_DISPATCH(StreamTriadSIMDThreads)(a_simd_threads.data(), b.data(),
                                               c.data(), q, count);
  // Record stop timer and print time
  const double t_simd_threads_1 = hwy::platform::Now();
  const bool simd_threads_validate = HWY_DYNAMIC_DISPATCH(TriadValidate)(
      a_scalar.data(), a_simd_threads.data(), count, threshold);
  PrintResults(true, simd_threads_validate, "Threaded SIMD", t_simd_threads_0,
               t_simd_threads_1, count);
  if (simd_validate && scalar_threads_validate && simd_threads_validate) {
    ret = 0;
  }

  return ret;
}
}  // namespace hwy

int main() { return hwy::Run(); }
#endif  // HWY_ONCE
