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

// Strong decrease during automated tests to keep them fast, while ensuring
// this is a multiple of 2 * N.
HWY_INLINE_VAR constexpr size_t kNumKeys =
    AdjustedReps(AdjustedReps(1024)) * 1024;

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
  return FillRandomDistinct<uint32_t>(num_keys, Unpredictable1());
}

struct MeasureResult {
  double ns;
  double mad_percent;
};

template <class Func>
MeasureResult Measure(const Func& func) {
  FuncInput input = Unpredictable1();
  Params params = DefaultBenchmarkParams();
  params.min_samples_per_eval = 2;
  params.max_evals = 4;
  params.verbose = false;
  Result results[1];

  const size_t num_results = MeasureClosure(func, &input, 1, results, params);
  if (num_results == 1) {
    return {results[0].ticks / platform::InvariantTicksPerSecond() * 1E9,
            results[0].variability * 100.0};
  } else {
    HWY_WARN("Measurement failed.");
    return MeasureResult{};
  }
}

HWY_NOINLINE void TestLatency() {
  const AlignedVector<uint32_t> keys = GenerateKeys(1000);
  ThreadPool pool = MakePool();
  const Phast phast = MakePhast(keys.data(), keys.size(), pool);

  MeasureResult result = Measure([&phast](FuncInput func_input) {
    return phast(static_cast<uint32_t>(func_input));
  });
  printf("Query latency: %6.2f ns; measurement MAD=%4.2f%%\n", result.ns,
         result.mad_percent);
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

// Benchmarks PHAST + u16 verification, which saves memory while still only
// using native u32 gathers.
HWY_NOINLINE void TestThroughput() {
  // Too slow under MSAN/TSAN.
#if !HWY_IS_MSAN && !HWY_IS_TSAN
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const AlignedVector<uint32_t> keys = GenerateKeys(kNumKeys);
  const Phast phast = MakePhast(keys.data(), keys.size(), pool);
  const Triple32 hash(phast.Data().config.hash_key);

  const ScalableTag<uint32_t> du32;
  const RebindToSigned<decltype(du32)> di32;
  using VU32 = Vec<decltype(du32)>;
  using MU32 = Mask<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
  HWY_ASSERT(kNumKeys % (2 * N) == 0);  // See GenerateKeys().

  // Store u16 fingerprints at PHAST positions. The fps array is indexed as
  // u16, but we gather as u32 (pairs of adjacent fps).
  const size_t num_slots = phast.Data().NumSlots();
  // +1 to ensure the last odd-indexed u16 has a valid u32 to gather from.
  AlignedVector<uint32_t> fp_u32(DivCeil(num_slots + 1, size_t{2}));
  const VU32 k1 = Set(du32, 1);

  // Populate fp_u32: non-vectorized due to scatter conflicts: updates to the
  // same u32 word from different lanes would be lost. This is anyway a
  // build-time operation.
  for (size_t i = 0; i < kNumKeys; i += 2 * N) {
    VU32 h0 = Load(du32, &keys[i]);
    VU32 h1 = Load(du32, &keys[i + N]);
    hash.TwoVec(du32, h0, h1);
    VU32 idx0, idx1;
    phast.PosFromHash(du32, h0, h1, idx0, idx1);
    HWY_ALIGN uint32_t fp_buf[4 * MaxLanes(du32)];
    Store(ShiftRight<16>(h0), du32, fp_buf);
    Store(ShiftRight<16>(h1), du32, fp_buf + 1 * N);
    Store(idx0, du32, fp_buf + 2 * N);
    Store(idx1, du32, fp_buf + 3 * N);
    for (size_t j = 0; j < 2 * N; ++j) {
      const uint32_t idx = fp_buf[2 * N + j];
      const uint32_t fp = fp_buf[j];
      const uint32_t word_idx = idx >> 1;
      const uint32_t shift = (idx & 1) * 16;
      fp_u32[word_idx] |= (fp << shift);
    }
  }

  // Each worker starts at a different offset in the keys.
  const size_t keys_per_chunk =
      RoundDownTo(kNumKeys / pool.NumWorkers(), 2 * N);

  const VU32 kFPMask = Set(du32, 0xFFFF);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  MeasureResult result = Measure([&](FuncInput func_input) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
      MU32 eq0 = SetMask(du32, true);
      MU32 eq1 = SetMask(du32, true);
      for (size_t i = 0; i < kNumKeys; i += 2 * N) {
        // Faster to wrap than have two loops (likely due to code size).
        const size_t wrapped_i = (worker * keys_per_chunk + i) % kNumKeys;
        VU32 hash0 = Load(du32, &keys[wrapped_i]);
        VU32 hash1 = Load(du32, &keys[wrapped_i + N]);

        // Hash keys - for expected fingerprints (upper 16 bits) and PHAST.
        hash.TwoVec(du32, hash0, hash1);
        const VU32 expected0 = ShiftRight<16>(hash0);
        const VU32 expected1 = ShiftRight<16>(hash1);
        VU32 idx0, idx1;
        phast.PosFromHash(du32, hash0, hash1, idx0, idx1);

        // Gather u32 words containing our u16 fp (one u32 gather per N).
        const VU32 word_idx0 = ShiftRight<1>(idx0);
        const VU32 word_idx1 = ShiftRight<1>(idx1);
        const VU32 word0 = MaskedGatherIndex(eq0, du32, fp_u32.data(),
                                             BitCast(di32, word_idx0));
        const VU32 word1 = MaskedGatherIndex(eq1, du32, fp_u32.data(),
                                             BitCast(di32, word_idx1));

        // Extract correct u16: if pos is odd, shift right by 16.
        const MU32 is_odd0 = Ne(And(idx0, k1), Zero(du32));
        const MU32 is_odd1 = Ne(And(idx1, k1), Zero(du32));
        const VU32 shifted0 = MaskedShiftRightOr<16>(word0, is_odd0, word0);
        const VU32 shifted1 = MaskedShiftRightOr<16>(word1, is_odd1, word1);
        const VU32 our_fp0 = And(shifted0, kFPMask);
        const VU32 our_fp1 = And(shifted1, kFPMask);
        eq0 = MaskedEq(eq0, expected0, our_fp0);
        eq1 = MaskedEq(eq1, expected1, our_fp1);
      }
      per_worker[worker * HWY_ALIGNMENT] = AllTrue(du32, And(eq0, eq1));
    });
    return per_worker[Unpredictable1() * HWY_ALIGNMENT];
  });
  for (size_t i = 0; i < pool.NumWorkers(); ++i) {
    HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
  }
  const size_t bytes = kNumKeys * sizeof(uint32_t) * pool.NumWorkers();
  printf(
      "Batch verify throughput: %4zuK keys = %4.1f GB/s; "
      "measurement MAD=%4.2f%%\n",
      kNumKeys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent);
#endif  // !HWY_IS_MSAN && !HWY_IS_TSAN
}

// Compare with absl::flat_hash_set - just set membership.
HWY_NOINLINE void TestAbslThroughput() {
  // Too slow under MSAN/TSAN.
#if HWY_HAVE_ABSL && !HWY_IS_MSAN && !HWY_IS_TSAN
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const AlignedVector<uint32_t> keys = GenerateKeys(kNumKeys);
  absl::flat_hash_set<uint32_t> set(keys.begin(), keys.end());

  // Each worker starts at a different offset in the keys to avoid unrealistic
  // cache behavior, without requiring separate per-worker allocations.
  const size_t N = VectorBytes() / sizeof(uint32_t);
  const size_t keys_per_chunk =
      RoundDownTo(kNumKeys / pool.NumWorkers(), 2 * N);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  MeasureResult result =
      Measure([&](FuncInput func_input) {
        pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
          bool all_found = true;
          const size_t offset = worker * keys_per_chunk;
          // First loop from the per-worker offset to end.
          for (size_t i = offset; i < kNumKeys; ++i) {
            all_found &= set.contains(keys[i]);
          }
          // Second loop from the beginning to the per-worker offset.
          for (size_t i = 0; i < offset; ++i) {
            all_found &= set.contains(keys[i]);
          }
          per_worker[worker * HWY_ALIGNMENT] = all_found;
        });
        return per_worker[Unpredictable1() * HWY_ALIGNMENT];
      });
  for (size_t i = 0; i < pool.NumWorkers(); ++i) {
    HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
  }
  const size_t bytes = kNumKeys * sizeof(uint32_t) * pool.NumWorkers();
  printf(
      "Batch absl verify throughput: %4zuK keys = %4.1f GB/s; "
      "measurement MAD=%4.2f%%\n",
      kNumKeys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent);
#else
  HWY_WARN("absl::flat_hash_set not available, skipping test.");
#endif  // HWY_HAVE_ABSL
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
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestThroughput);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestAbslThroughput);
// Measure last so they reflect the current (less-boosted) CPU clock rate.
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestBW);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestLatency);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
