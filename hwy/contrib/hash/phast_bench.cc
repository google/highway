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
#define HWY_HAVE_TCMALLOC 0

#include <stdint.h>
#include <stdio.h>

#include <vector>

#if HWY_HAVE_ABSL
#include "third_party/absl/container/flat_hash_set.h"
#endif
#if HWY_HAVE_TCMALLOC
#include // Placeholder for tcmalloc, do not remove
#endif

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/thread_pool/thread_pool.h"
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
#include "hwy/contrib/hash/cuckoo-inl.h"
#include "hwy/contrib/hash/cuckoo2x2-inl.h"
#include "hwy/contrib/hash/phast-inl.h"
#include "hwy/contrib/hash/shardmul-inl.h"
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

// Set to true to benchmark across a range of key counts.
HWY_INLINE_VAR constexpr bool kSweepThroughput = false;
HWY_INLINE_VAR constexpr bool kSweepLatency = false;

static ThreadPool MakePool() {
  return ThreadPool(ThreadPool::NumThreadsFromCores());
}

HWY_NOINLINE AlignedVector<uint32_t> GenerateKeys(size_t num_keys) {
  // Round up to two vectors so we do not have to handle remainders here.
  num_keys = RoundUpTo(num_keys, 2 * VectorBytes() / sizeof(uint32_t));
  // Must be distinct, hence do not use FillRandom().
  return FillRandomDistinct<uint32_t>(num_keys, Unpredictable1());
}

HWY_NOINLINE AlignedVector<uint64_t> GenerateKeys64(size_t num_keys) {
  // Round up to two vectors so we do not have to handle remainders here.
  num_keys = RoundUpTo(num_keys, 2 * VectorBytes() / sizeof(uint64_t));
  // Must be distinct, hence do not use FillRandom().
  return FillRandomDistinct<uint64_t>(num_keys, Unpredictable1());
}

size_t AllocatedBefore() {
#if HWY_HAVE_TCMALLOC
  return tcmalloc::MallocExtension::GetNumericProperty(
             "generic.current_allocated_bytes")
      .value_or(0);
#else
  return 0;
#endif  // HWY_HAVE_TCMALLOC
}

size_t AllocatedBytes(size_t before, size_t guessed) {
#if HWY_HAVE_TCMALLOC
  const size_t after = tcmalloc::MallocExtension::GetNumericProperty(
                           "generic.current_allocated_bytes")
                           .value_or(0);
  size_t allocated_bytes = after - before;
  if (allocated_bytes == 0) {
    return guessed;
    HWY_WARN("tcmalloc tracking returned 0, estimating from capacity.");
  }
  return allocated_bytes;
#else
  return guessed;
#endif  // HWY_HAVE_TCMALLOC
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

// NOTE: unlike the others, this does not verify the keys because it is intended
// to be used alongside PHAST. In real usage, the original u64 should be stored
// at the slot returned by PHAST for the u32 output of ShardMul.
HWY_NOINLINE void TestShardMulThroughput(size_t num_keys) {
  // Too slow under MSAN/TSAN.
  if constexpr (HWY_IS_MSAN || HWY_IS_TSAN) return;
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const ScalableTag<uint64_t> du64;
  const RepartitionToNarrow<decltype(du64)> du32;
  using VU32 = Vec<decltype(du32)>;
  using VU64 = Vec<decltype(du64)>;
  HWY_LANES_CONSTEXPR size_t NU32 = Lanes(du32);
  HWY_LANES_CONSTEXPR size_t NU64 = Lanes(du64);
  HWY_ASSERT(num_keys % NU32 == 0);  // See GenerateKeys64().

  const AlignedVector<uint64_t> keys = GenerateKeys64(num_keys);
  num_keys = keys.size();  // May have been rounded up.

  const size_t before = AllocatedBefore();
  const ShardMul shardmul = MakeShardMul(Span(keys), pool);
  const size_t allocated_bytes = AllocatedBytes(before, 0);

  // Each worker starts at a different offset in the keys.
  const size_t keys_per_chunk = RoundDownTo(num_keys / pool.NumWorkers(), NU32);

  AlignedVector<uint32_t> per_worker(pool.NumWorkers() * num_keys);
  MeasureResult result = Measure([&](FuncInput func_input) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
      for (size_t i = 0; i < num_keys; i += 2 * NU32) {
        // Faster to wrap than have two loops (likely due to code size).
        const size_t wrapped_i = (worker * keys_per_chunk + i) % num_keys;
        const VU64 key0 = Load(du64, &keys[wrapped_i]);
        const VU64 key1 = Load(du64, &keys[wrapped_i + NU64]);

        // Hash keys for PHAST position lookup.
        const VU32 out = shardmul.TwoVec(du32, key0, key1);
        Store(out, du32, &per_worker[worker * num_keys + i]);
      }
      PreventElision(per_worker[worker * num_keys + Unpredictable1()]);
    });
    return per_worker[Unpredictable1()];
  });
  const size_t bytes = num_keys * sizeof(uint64_t) * pool.NumWorkers();
  printf(
      "Batch ShardMul u64 reduce throughput: %4zuKi keys = %4.1f GB/s; "
      "measurement MAD=%5.2f%%, allocated %5zu KiB\n",
      num_keys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent, allocated_bytes / 1024);
}

// Benchmarks PHAST + u32 key verification. Stores the full key at each PHAST
// position, guaranteeing zero false positives (at 4 bytes/key payload).
HWY_NOINLINE void TestPhastThroughput(size_t num_keys) {
  // Too slow under MSAN/TSAN.
  if constexpr (HWY_IS_MSAN || HWY_IS_TSAN) return;
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const ScalableTag<uint32_t> du32;
  const RebindToSigned<decltype(du32)> di32;
  using VU32 = Vec<decltype(du32)>;
  using MU32 = Mask<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
  HWY_ASSERT(num_keys % (2 * N) == 0);  // See GenerateKeys().

  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  num_keys = keys.size();  // May have been rounded up.

  const size_t before = AllocatedBefore();
  constexpr size_t kPayloadBytes = sizeof(uint32_t);
  const Phast phast = MakePhast(Span(keys), kPayloadBytes, pool);
  const Triple32 hash(phast.Data().config.hash_key);

  // Store the full key at each PHAST position for verification. The complex
  // PHAST index assignment function does not allow the same u16 fingerprint
  // trick that we use in Cuckoo2x2.
  const size_t num_slots = phast.Data().NumSlots();
  AlignedVector<uint32_t> stored_keys(num_slots);

  const size_t allocated_bytes =
      AllocatedBytes(before, phast.Data().AllocatedBytes(kPayloadBytes));

  // Populate stored_keys via scatter. No conflicts because PHAST is a perfect
  // hash: all member keys have unique indices.
  for (size_t i = 0; i < num_keys; i += 2 * N) {
    VU32 h0 = Load(du32, &keys[i]);
    VU32 h1 = Load(du32, &keys[i + N]);
    const VU32 key0 = h0;  // Save original keys before hashing.
    const VU32 key1 = h1;
    hash.TwoVec(du32, h0, h1);
    VU32 idx0, idx1;
    phast.PosFromHash(du32, h0, h1, idx0, idx1);
    ScatterIndex(key0, du32, stored_keys.data(), BitCast(di32, idx0));
    ScatterIndex(key1, du32, stored_keys.data(), BitCast(di32, idx1));
  }

  // Each worker starts at a different offset in the keys.
  const size_t keys_per_chunk =
      RoundDownTo(num_keys / pool.NumWorkers(), 2 * N);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  MeasureResult result = Measure([&](FuncInput func_input) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
      MU32 eq0 = SetMask(du32, true);
      MU32 eq1 = SetMask(du32, true);
      for (size_t i = 0; i < num_keys; i += 2 * N) {
        // Faster to wrap than have two loops (likely due to code size).
        const size_t wrapped_i = (worker * keys_per_chunk + i) % num_keys;
        const VU32 key0 = Load(du32, &keys[wrapped_i]);
        const VU32 key1 = Load(du32, &keys[wrapped_i + N]);
        VU32 hash0 = key0;
        VU32 hash1 = key1;

        // Hash keys for PHAST position lookup.
        hash.TwoVec(du32, hash0, hash1);
        VU32 idx0, idx1;
        phast.PosFromHash(du32, hash0, hash1, idx0, idx1);

        // Gather stored keys and compare directly.
        const VU32 stored0 = MaskedGatherIndex(eq0, du32, stored_keys.data(),
                                               BitCast(di32, idx0));
        const VU32 stored1 = MaskedGatherIndex(eq1, du32, stored_keys.data(),
                                               BitCast(di32, idx1));
        eq0 = MaskedEq(eq0, key0, stored0);
        eq1 = MaskedEq(eq1, key1, stored1);
      }
      per_worker[worker * HWY_ALIGNMENT] = AllTrue(du32, And(eq0, eq1));
    });
    return per_worker[Unpredictable1() * HWY_ALIGNMENT];
  });
  for (size_t i = 0; i < pool.NumWorkers(); ++i) {
    HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
  }
  const size_t bytes = num_keys * sizeof(uint32_t) * pool.NumWorkers();
  printf(
      "Batch PHAST u32 verify throughput: %4zuKi keys = %4.1f GB/s; "
      "measurement MAD=%5.2f%%, allocated %5zu KiB\n",
      num_keys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent, allocated_bytes / 1024);
}

// Benchmarks Cuckoo2x2 used as a set membership test, checking returned masks.
HWY_NOINLINE void TestCuckoo2x2Throughput(size_t num_keys) {
  // Too slow under MSAN/TSAN.
  if constexpr (HWY_IS_MSAN || HWY_IS_TSAN) return;
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  num_keys = keys.size();  // May have been rounded up.

  const size_t before = AllocatedBefore();
  const Cuckoo2x2 set = MakeCuckoo2x2(Span(keys), pool);
  const size_t allocated_bytes =
      AllocatedBytes(before, set.Data().AllocatedBytes());

  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  using MU32 = Mask<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
  HWY_ASSERT(num_keys % N == 0);  // See GenerateKeys().

  // Each worker starts at a different offset in the keys to avoid unrealistic
  // cache behavior, without requiring separate per-worker allocations.
  const size_t keys_per_chunk = RoundDownTo(num_keys / pool.NumWorkers(), N);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  MeasureResult result = Measure([&](FuncInput func_input) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
      MU32 any_missing = SetMask(du32, false);
      for (size_t i = 0; i < num_keys; i += N) {
        // Faster to wrap than have two loops (likely due to code size).
        const size_t wrapped_i = (worker * keys_per_chunk + i) % num_keys;
        const VU32 vkeys = Load(du32, &keys[wrapped_i]);
        any_missing = Or(any_missing, set(du32, vkeys));
      }
      per_worker[worker * HWY_ALIGNMENT] = AllFalse(du32, any_missing);
    });
    return per_worker[Unpredictable1() * HWY_ALIGNMENT];
  });
  for (size_t i = 0; i < pool.NumWorkers(); ++i) {
    HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
  }
  const size_t bytes = num_keys * sizeof(uint32_t) * pool.NumWorkers();
  printf(
      "Batch Cuckoo2x2 verify throughput: %4zuKi keys = %4.1f GB/s; "
      "measurement MAD=%5.2f%%, allocated %5zu KiB\n",
      num_keys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent, allocated_bytes / 1024);
}

// Compare with absl::flat_hash_set - just set membership.
HWY_NOINLINE void TestAbslThroughput(size_t num_keys) {
  // Too slow under MSAN/TSAN.
  if constexpr (HWY_IS_MSAN || HWY_IS_TSAN) return;

#if HWY_HAVE_ABSL
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  num_keys = keys.size();  // May have been rounded up.

  const size_t before = AllocatedBefore();

  absl::flat_hash_set<uint32_t> set(keys.begin(), keys.end());
  // Capacity + 1 control byte per slot + group sentinel.
  const size_t absl_allocated_bytes =
      set.capacity() * (sizeof(uint32_t) + 1) + 16;
  const size_t allocated_bytes = AllocatedBytes(before, absl_allocated_bytes);

  // Each worker starts at a different offset in the keys to avoid unrealistic
  // cache behavior, without requiring separate per-worker allocations.
  const size_t N = VectorBytes() / sizeof(uint32_t);
  const size_t keys_per_chunk =
      RoundDownTo(num_keys / pool.NumWorkers(), 2 * N);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  MeasureResult result = Measure([&](FuncInput func_input) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
      bool all_found = true;
      const size_t offset = worker * keys_per_chunk;
      // First loop from the per-worker offset to end.
      for (size_t i = offset; i < num_keys; ++i) {
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
  const size_t bytes = num_keys * sizeof(uint32_t) * pool.NumWorkers();
  printf(
      "Batch absl::set verify throughput: %4zuKi keys = %4.1f GB/s; "
      "measurement MAD=%5.2f%%, allocated %5zu KiB\n",
      num_keys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent, allocated_bytes / 1024);
#else
  (void)num_keys;
  HWY_WARN("absl::flat_hash_set not available, skipping test.");
#endif  // HWY_HAVE_ABSL
}

HWY_NOINLINE void TestShardMulLatency(size_t num_keys) {
  const AlignedVector<uint64_t> keys = GenerateKeys64(num_keys);
  ThreadPool pool = MakePool();
  const ShardMul shard_mul = MakeShardMul(Span(keys), pool);

  MeasureResult result = Measure(
      [&shard_mul](FuncInput func_input) { return shard_mul(func_input); });
  printf(
      "Single ShardMul  latency: %5zu keys, %6.2f ns; measurement "
      "MAD=%5.2f%%\n",
      num_keys, result.ns, result.mad_percent);
}

HWY_NOINLINE void TestPhastLatency(size_t num_keys) {
  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  ThreadPool pool = MakePool();
  const Phast phast = MakePhast(Span(keys), 0, pool);

  MeasureResult result = Measure([&phast](FuncInput func_input) {
    return phast(static_cast<uint32_t>(func_input));
  });
  printf(
      "Single PHAST     latency: %5zu keys, %6.2f ns; measurement "
      "MAD=%5.2f%%\n",
      num_keys, result.ns, result.mad_percent);
}

HWY_NOINLINE void TestCuckoo2x2Latency(size_t num_keys) {
  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  ThreadPool pool = MakePool();
  const Cuckoo2x2 set = MakeCuckoo2x2(Span(keys), pool);

  MeasureResult result = Measure([&set](FuncInput func_input) {
    return set.Contains(static_cast<uint32_t>(func_input));
  });
  printf(
      "Single Cuckoo2x2 latency: %5zu keys, %6.2f ns; measurement "
      "MAD=%5.2f%%\n",
      num_keys, result.ns, result.mad_percent);
}

HWY_NOINLINE void TestAbslLatency(size_t num_keys) {
#if HWY_HAVE_ABSL
  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  absl::flat_hash_set<uint32_t> set(keys.begin(), keys.end());

  MeasureResult result = Measure([&set](FuncInput func_input) {
    return set.contains(static_cast<uint32_t>(func_input));
  });
  printf(
      "Single Absl      latency: %5zu keys, %6.2f ns; measurement "
      "MAD=%5.2f%%\n",
      num_keys, result.ns, result.mad_percent);
#else
  (void)num_keys;
  HWY_WARN("absl::flat_hash_set not available, skipping test.");
#endif  // HWY_HAVE_ABSL
}

template <bool kUseU16 = false>
HWY_NOINLINE void TestCuckooThroughput(size_t num_keys) {
  const AlignedVector<uint32_t> keys = GenerateKeys(num_keys);
  const size_t before = AllocatedBefore();
  CuckooTraits<> traits;
  auto cuckoo = CuckooBuild(traits, keys.data(), keys.size(), /*epsilon=*/0.1,
                            /*max_attempts=*/100, CuckooBuildAlgo::kMinCost);
  if constexpr (kUseU16) {
    cuckoo.BuildU16Slots();
  }
  const size_t allocated_bytes =
      AllocatedBytes(before, cuckoo.AllocatedBytes());
  if (cuckoo.IsEmpty()) {
    HWY_WARN("Cuckoo build failed, skipping throughput test.\n");
    return;
  }

  const double pct_pri = 100.0 * cuckoo.NumPrimary() / keys.size();
  const double pct_sec =
      100.0 * (keys.size() - cuckoo.NumPrimary()) / keys.size();
  fprintf(stderr,
          "Cuckoo hashing fill stats: %.2f%% primary, %.2f%% secondary\n",
          pct_pri, pct_sec);

  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);

  // Each worker starts at a different offset in the keys to avoid unrealistic
  // cache behavior, without requiring separate per-worker allocations.
  const ScalableTag<uint32_t> du32;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
  const size_t keys_per_chunk = RoundDownTo(num_keys / pool.NumWorkers(), N);
  HWY_ASSERT(num_keys % N == 0);

  AlignedVector<uint8_t> per_worker(pool.NumWorkers() * HWY_ALIGNMENT);
  MeasureResult result = Measure([&](FuncInput func_input) {
    pool.Run(0, pool.NumWorkers(), [&](uint64_t task_idx, size_t worker) {
      bool all_found = true;
      for (size_t i = 0; i < num_keys; i += N) {
        const size_t wrapped_i = (worker * keys_per_chunk + i) % num_keys;
        if constexpr (kUseU16) {
          const auto not_found =
              cuckoo.QueryBatchU16</*kPrecomputeSecondary=*/true>(
                  du32, &keys[wrapped_i]);
          all_found &= AllFalse(du32, not_found);
        } else {
          const auto not_found = cuckoo.QueryBatch(du32, &keys[wrapped_i]);
          all_found &= AllFalse(du32, not_found);
        }
      }
      per_worker[worker * HWY_ALIGNMENT] = all_found;
    });
    return per_worker[Unpredictable1() * HWY_ALIGNMENT];
  });
  for (size_t i = 0; i < pool.NumWorkers(); ++i) {
    HWY_ASSERT(per_worker[i * HWY_ALIGNMENT]);
  }
  const size_t bytes = num_keys * sizeof(uint32_t) * pool.NumWorkers();
  const char* suffix = kUseU16 ? "u16" : "u32";
  printf(
      "Cuckoo 2x16 %s throughput: %4zuKi keys = %4.1f GB/s; "
      "measurement MAD=%4.2f%%, allocated %5zu KiB\n",
      suffix, num_keys / 1024, static_cast<double>(bytes) / result.ns,
      result.mad_percent, allocated_bytes / 1024);
}

// Driver functions: sweep across sizes or run once.
HWY_NOINLINE void TestLatencySweep() {
  if (kSweepLatency) {
    // Powers of ten: 100-10K.
    for (size_t n = 100; n <= 10'000; n *= 10) {
      TestShardMulLatency(n);
      TestPhastLatency(n);
      TestCuckoo2x2Latency(n);
      TestAbslLatency(n);
    }
    // Powers of two
    for (size_t n = 512; n <= 4096; n *= 2) {
      TestShardMulLatency(n);
      TestPhastLatency(n);
      TestCuckoo2x2Latency(n);
      TestAbslLatency(n);
    }
  } else {
    const size_t n = 1000;
    TestShardMulLatency(n);
    TestPhastLatency(n);
    TestCuckoo2x2Latency(n);
    TestAbslLatency(n);
  }
}

HWY_NOINLINE void TestThroughputSweep() {
  if (kSweepThroughput) {
    // Powers of ten: 10K-10M.
    for (size_t n = 10'000; n <= 10'000'000; n *= 10) {
      TestShardMulThroughput(n);
      TestPhastThroughput(n);
      TestCuckoo2x2Throughput(n);
      TestAbslThroughput(n);
      TestCuckooThroughput(n);
      TestCuckooThroughput</*kUseU16=*/false>(n);
    }
    // Powers of two: 32K-4M.
    for (size_t n = (32 << 10); n <= (size_t{4} << 20); n *= 2) {
      TestShardMulThroughput(n);
      TestPhastThroughput(n);
      TestCuckoo2x2Throughput(n);
      TestAbslThroughput(n);
      TestCuckooThroughput(n);
      TestCuckooThroughput</*kUseU16=*/true>(n);
    }
  } else {
    TestShardMulThroughput(kNumKeys);
    TestPhastThroughput(kNumKeys);
    TestCuckoo2x2Throughput(kNumKeys);
    TestAbslThroughput(kNumKeys);
  }
}

#else   // HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128
void TestBW() {}
void TestThroughputSweep() {}
void TestLatencySweep() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(PhastBench);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestThroughputSweep);
// Measure last so they reflect the current (less-boosted) CPU clock rate.
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestBW);
HWY_EXPORT_AND_TEST_BEST_P(PhastBench, TestLatencySweep);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
