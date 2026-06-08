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

// If set, we also benchmark absl::flat_hash_set.
#include "hwy/detect_compiler_arch.h"
#define HWY_HAVE_ABSL 0

#include <stdint.h>
#include <stdio.h>

#include <vector>

#if HWY_HAVE_ABSL
#include "third_party/absl/container/flat_hash_set.h"
#endif

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/nanobenchmark.h"
#include "hwy/per_target.h"  // VectorBytes
#include "hwy/robust_statistics.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/phast_bench.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/phast-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if (HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128) || HWY_IDE

// Increase when running manually; this is to keep tests fast. Must be a power
// of two of at least NumWorkers * 2 * VectorBytes()/4 to enable wraparound.
HWY_INLINE_VAR constexpr size_t kNumKeys =
    HWY_IS_DEBUG_BUILD ? 16 * 1024 : 128 * 1024;

static ThreadPool MakePool() {
  static Topology topology;
  if (topology.packages.empty()) return ThreadPool(ThreadPool::MaxThreads());
  // Minus one because these are in addition to the main thread.
  return ThreadPool(topology.packages[0].cores.size() - 1);
}

HWY_NOINLINE AlignedVector<uint32_t> GenerateKeys(size_t num_keys) {
  // Round up to two vectors so we do not have to handle remainders here.
  num_keys = RoundUpTo(num_keys, 2 * VectorBytes() / sizeof(uint32_t));
  // Must be distinct, hence do not use FillRandom().
  AlignedVector<uint32_t> keys;
  keys.reserve(num_keys);
  AesCtrEngine engine(/*deterministic=*/true);
  Triple32 permutation(engine, Unpredictable1());
  for (size_t i = 0; i < num_keys; ++i) {
    keys.push_back(permutation(i));
  }
  return keys;
}

HWY_NOINLINE Phast MakePhast(const AlignedVector<uint32_t>& keys,
                             ThreadPool& pool) {
  const size_t num_keys = keys.size();
  const uint32_t slice_length = num_keys > 4 * 1024 * 1024 ? 8192
                                : num_keys > 256 * 1024    ? 4096
                                : num_keys >= 10 * 1000    ? 512
                                                           : 256;
  const uint32_t headroom_percent = num_keys > 4 * 1024 * 1024 ? 6
                                    : num_keys > 256 * 1024    ? 2
                                                               : 5;
  PhastConfig config(num_keys, 2, slice_length, headroom_percent);
  return BuildPhast(keys.data(), config, pool);
}

HWY_NOINLINE void TestLatency() {
  const AlignedVector<uint32_t> keys = GenerateKeys(1000);
  ThreadPool pool = MakePool();
  const Phast phast = MakePhast(keys, pool);
  if (phast.IsEmpty()) {
    HWY_WARN("Phast build failed, skipping latency test.\n");
    return;
  }

  FuncInput input = Unpredictable1();
  Params params = DefaultBenchmarkParams();
  params.verbose = false;
  Result results[1];

  const size_t num_results = MeasureClosure(
      [&phast](FuncInput func_input) {
        return phast(static_cast<uint32_t>(func_input));
      },
      &input, 1, results, params);
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    printf("Query latency: %6.2f ns; measurement MAD=%4.2f%%\n", ns,
           results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

HWY_NOINLINE void TestBW() {
  const ScalableTag<uint8_t> du8;
  using VU8 = Vec<decltype(du8)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du8);
  const VU8 k1 = Set(du8, static_cast<uint8_t>(Unpredictable1()));

  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const size_t kBytesPerWorker = kNumKeys * sizeof(uint32_t);
  const size_t num_bytes = pool.NumWorkers() * kBytesPerWorker;
  // Large array, avoid AlignedVector because it zero-initializes on 1 thread.
  AlignedFreeUniquePtr<uint8_t[]> bytes = AllocateAligned<uint8_t>(num_bytes);
  pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t /*worker*/) {
    FillBytes(&bytes[task_idx * kBytesPerWorker],
              static_cast<uint8_t>(Unpredictable1()), kBytesPerWorker);
  });

  // Using nanobenchmark is too slow because it involves multiple iterations.
  constexpr size_t kNumReps = AdjustedReps(20);
  std::vector<double> elapsed_times;
  elapsed_times.reserve(kNumReps);
  for (size_t rep = 0; rep < kNumReps; ++rep) {
    const double t0 = platform::Now();
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t /*worker*/) {
      uint8_t* my_bytes = &bytes[task_idx * kBytesPerWorker];
      for (size_t i = 0; i < kBytesPerWorker; i += 2 * N) {
        const VU8 v0 = Load(du8, my_bytes + i);
        const VU8 v1 = Load(du8, my_bytes + i + N);
        Store(Add(v0, k1), du8, my_bytes + i);
        Store(Add(v1, k1), du8, my_bytes + i + N);
      }
    });
    uint32_t result = bytes[Unpredictable1()];
    PreventElision(result);
    elapsed_times.push_back(platform::Now() - t0);
  }
  const double elapsed =
      robust_statistics::Median(elapsed_times.data(), elapsed_times.size());
  printf("MemBW: %7.2f ms = %4.1f GB/s\n", elapsed * 1E3,
         num_bytes / elapsed * 1E-9);
}

// Benchmarks PHAST used as a hash table, with an extra Gather at the returned
// index to verify set membership.
HWY_NOINLINE void TestThroughput() {
  const AlignedVector<uint32_t> keys = GenerateKeys(kNumKeys);
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const Phast phast = MakePhast(keys, pool);
  if (phast.IsEmpty()) {
    HWY_WARN("Phast build failed, skipping throughput test.\n");
    return;
  }

  const ScalableTag<uint32_t> du32;
  const RebindToSigned<decltype(du32)> di32;
  using VU32 = Vec<decltype(du32)>;
  using MU32 = Mask<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
  HWY_DASSERT(kNumKeys % (2 * N) == 0);  // See GenerateKeys().

  // Scatter keys to the verification slots. Could also sort K32V32 and copy.
  AlignedVector<uint32_t> key_verify(phast.Config().NumSlots());
  for (size_t i = 0; i < kNumKeys; i += 2 * N) {
    const VU32 keys0 = Load(du32, &keys[i]);
    const VU32 keys1 = Load(du32, &keys[i + N]);
    VU32 idx0, idx1;
    phast.Query2(du32, keys0, keys1, idx0, idx1);
    ScatterIndex(keys0, du32, key_verify.data(), BitCast(di32, idx0));
    ScatterIndex(keys1, du32, key_verify.data(), BitCast(di32, idx1));
  }

  FuncInput input = Unpredictable1();
  Result results[1];
  Params params = DefaultBenchmarkParams();
  params.min_samples_per_eval = 2;
  params.max_evals = 3;
  params.verbose = false;

  // Each worker starts at a different offset in the keys to avoid unrealistic
  // cache behavior, without requiring separate per-worker allocations.
  const size_t num_workers_pow2 = 1u << hwy::CeilLog2(pool.NumWorkers());
  const size_t keys_per_chunk = kNumKeys / (num_workers_pow2 * 2 * N);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  const size_t num_results = MeasureClosure(
      [&](FuncInput func_input) {
        pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
          MU32 eq0 = SetMask(du32, true);
          MU32 eq1 = SetMask(du32, true);
          for (size_t i = 0; i < kNumKeys; i += 2 * N) {
            const size_t wrapped_i = (worker * keys_per_chunk + i) % kNumKeys;
            const VU32 keys0 = Load(du32, &keys[wrapped_i]);
            const VU32 keys1 = Load(du32, &keys[wrapped_i + N]);
            VU32 idx0, idx1;
            phast.Query2(du32, keys0, keys1, idx0, idx1);
            eq0 = MaskedEq(
                eq0, keys0,
                GatherIndex(du32, key_verify.data(), BitCast(di32, idx0)));
            eq1 = MaskedEq(
                eq1, keys1,
                GatherIndex(du32, key_verify.data(), BitCast(di32, idx1)));
          }
          per_worker[worker * HWY_ALIGNMENT] = AllTrue(du32, And(eq0, eq1));
        });
        return per_worker[Unpredictable1() * HWY_ALIGNMENT];
      },
      &input, 1, results, params);
  for (size_t i = 0; i < pool.NumWorkers(); ++i) {
    HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
  }
  printf("\n");
  if (num_results == 1) {
    const double ns =
        results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
    const size_t bytes = kNumKeys * sizeof(uint32_t) * pool.NumWorkers();
    printf(
        "Batch verify throughput: %7.2f ns = %4.1f GB/s; measurement "
        "MAD=%4.2f%%\n",
        ns, static_cast<double>(bytes) / ns, results[0].variability * 100.0);
  } else {
    HWY_WARN("Measurement failed.");
  }
}

// Compare with absl::flat_hash_set - just set membership.
HWY_NOINLINE void TestAbslThroughput() {
  if constexpr (HWY_HAVE_ABSL) {
    const AlignedVector<uint32_t> keys = GenerateKeys(kNumKeys);
    absl::flat_hash_set<uint32_t> set(keys.begin(), keys.end());

    FuncInput input = Unpredictable1();
    Result results[1];
    Params params = DefaultBenchmarkParams();
    params.min_samples_per_eval = 2;
    params.max_evals = 3;
    params.verbose = false;

    ThreadPool pool = MakePool();
    pool.SetWaitMode(PoolWaitMode::kSpin);

    // Each worker starts at a different offset in the keys to avoid unrealistic
    // cache behavior, without requiring separate per-worker allocations.
    const size_t num_workers_pow2 = 1u << hwy::CeilLog2(pool.NumWorkers());
    const size_t N = VectorBytes() / sizeof(uint32_t);
    const size_t keys_per_chunk = kNumKeys / (num_workers_pow2 * 2 * N);

    AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
    const size_t num_results = MeasureClosure(
        [&](FuncInput func_input) {
          pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
            bool all_found = true;
            for (size_t i = 0; i < kNumKeys; ++i) {
              all_found &=
                  set.contains(keys[(i + worker * keys_per_chunk) % kNumKeys]);
            }
            per_worker[worker * HWY_ALIGNMENT] = all_found;
          });
          return per_worker[Unpredictable1() * HWY_ALIGNMENT];
        },
        &input, 1, results, params);
    for (size_t i = 0; i < pool.NumWorkers(); ++i) {
      HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
    }
    if (num_results == 1) {
      const double ns =
          results[0].ticks / platform::InvariantTicksPerSecond() * 1E9;
      const size_t bytes = kNumKeys * sizeof(uint32_t) * pool.NumWorkers();
      printf(
          "Batch absl verify throughput: %7.2f ns = %4.1f GB/s; measurement "
          "MAD=%4.2f%%\n",
          ns, static_cast<double>(bytes) / ns, results[0].variability * 100.0);
    } else {
      HWY_WARN("Measurement failed.");
    }
  } else {
    HWY_WARN("absl::flat_hash_set not available, skipping test.");
  }
}

#else   // HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128
void TestLatency() {}
void TestBW() {}
void TestThroughput() {}
void TestAbslThroughput() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(PhastBench);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestLatency);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestBW);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestThroughput);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestAbslThroughput);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
