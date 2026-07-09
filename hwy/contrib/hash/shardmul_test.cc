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

// Tests for ShardMul u64->u32 reducer.

#include <stdint.h>
#include <stdio.h>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/hash/shardmul.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/shardmul_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/hash/shardmul-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if HWY_TARGET != HWY_SCALAR

static ThreadPool MakePool() {
  return ThreadPool(ThreadPool::NumThreadsFromCores());
}

// Generates 'count' clustered u64 keys simulating packed 8-byte UTF-8 strings.
// Uses multiple base templates (7-8 bytes with zero padding) and randomly
// mutates byte positions per key with values spanning the UTF-8 byte range.
static AlignedVector<uint64_t> GenerateClusteredKeys(size_t count) {
  const ScalableTag<uint32_t> du32;
  const RepartitionToWide<decltype(du32)> du64;
  AlignedVector<uint64_t> keys;
  const size_t candidates = 3 * count / 2;
  keys.reserve(candidates);

  // Several base "templates" - full 8-byte strings, no null padding.
  const uint64_t kPatterns[] = {
      0x48656C6C6F20576Full,  // "Hello Wo"
      0x7468655F6B65795Full,  // "the_key_"
      0xC3A9C3A0C3BCE282ull,  // multi-byte UTF-8 codepoints
      0x6162636465666747ull,  // "abcdefgH"
  };
  constexpr size_t kNumPatterns = sizeof(kPatterns) / sizeof(kPatterns[0]);

  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);

  for (size_t i = 0; i < candidates; ++i) {
    uint64_t key = kPatterns[i % kNumPatterns];
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&key);

    // Randomly choose 1-5 byte positions to mutate.
    const size_t num_mutations = 1 + static_cast<size_t>(rng() % 5);
    for (size_t m = 0; m < num_mutations; ++m) {
      const size_t pos = rng() % 8;
      // UTF-8 byte range: mix of ASCII (0x20-0x7E), continuation (0x80-0xBF),
      // and leading bytes (0xC0-0xF4).
      const uint8_t val = static_cast<uint8_t>(0x20 + (rng() % 213));
      bytes[pos] = val;
    }
    // 25% chance of 7-byte key.
    if (rng() % 4 == 0) {
      key >>= 8;
    }
    keys.push_back(key);
  }

  // Deduplicate (very unlikely to have collisions, but be safe).
  VQSort(keys.data(), keys.size(), SortAscending());
  const size_t remaining = Unique(du64, keys.data(), keys.size());
  HWY_ASSERT(remaining >= count);
  keys.resize(count);
  return keys;
}

HWY_NOINLINE void TestShardMulExtraOutputs() {
  const ScalableTag<uint64_t> du64;
  const RepartitionToNarrow<decltype(du64)> du32;
  const size_t NU32 = Lanes(du32);

  ThreadPool pool = MakePool();
  const size_t num_keys = RoundUpTo(AdjustedReps(AdjustedReps(300'000)), NU32);
  AlignedVector<uint64_t> keys = GenerateClusteredKeys(num_keys);

  // Generate some extra outputs.
  const size_t kNumExtra = AdjustedReps(AdjustedReps(80'000));
  AesCtrEngine engine(/*deterministic=*/true);
  AlignedVector<uint32_t> extra_outputs =
      FillRandom<uint32_t>(kNumExtra, engine, /*seed=*/0);
  VQSort(extra_outputs.data(), extra_outputs.size(), SortAscending());
  // Per birthday paradox, duplicates are likely, which would cause building
  // ShardMul to fail.
  extra_outputs.resize(
      Unique(du32, extra_outputs.data(), extra_outputs.size()));

  const ShardMulData data(BuildShardMul(Span(keys), Span(extra_outputs), pool));
  const ShardMul shard_mul{data};
  HWY_ASSERT(!shard_mul.IsEmpty());

  // Verify all outputs of keys are distinct, and also do not collide with
  // extra_outputs.
  AlignedVector<uint32_t> combined_outputs(keys.size() + extra_outputs.size());
  for (size_t i = 0; i < keys.size(); i += NU32) {
    const Vec<decltype(du64)> keys0 = Load(du64, &keys[i]);
    const Vec<decltype(du64)> keys1 = Load(du64, &keys[i + Lanes(du64)]);
    Store(shard_mul.TwoVec(du32, keys0, keys1), du32, &combined_outputs[i]);
  }
  // Copy extra outputs to the end
  CopyBytes(extra_outputs.data(), combined_outputs.data() + keys.size(),
            extra_outputs.size() * sizeof(uint32_t));

  VQSort(combined_outputs.data(), combined_outputs.size(), SortAscending());
  HWY_ASSERT_M(
      AllUnique(du32, combined_outputs.data(), combined_outputs.size()),
      "Collision with extra outputs detected");
}

HWY_NOINLINE void TestShardMulCollisionFree() {
  const ScalableTag<uint64_t> du64;
  const RepartitionToNarrow<decltype(du64)> du32;
  const size_t NU64 = Lanes(du64);
  const size_t NU32 = Lanes(du32);

  ThreadPool pool = MakePool();
  const size_t num_keys =
      RoundUpTo(AdjustedReps(AdjustedReps(1'000'000)), NU32);
  AlignedVector<uint64_t> keys = GenerateClusteredKeys(num_keys);

  const double t0 = platform::Now();
  const ShardMulData data(BuildShardMul(Span(keys), pool));
  const ShardMul shard_mul{data};
  HWY_ASSERT(!shard_mul.IsEmpty());
  const double elapsed = platform::Now() - t0;
  fprintf(stderr, "  Build: %.2f ms, %zuK keys, %.2f MB/s\n", elapsed * 1E3,
          keys.size() >> 10,
          static_cast<double>(keys.size() * sizeof(uint64_t)) / elapsed * 1E-6);
  fprintf(stderr, "  attempts: %s\n", data.s_bucket_reps.ToString().c_str());

  // Verify all outputs are distinct.
  AlignedVector<uint32_t> outputs(keys.size());
  for (size_t i = 0; i < keys.size(); i += NU32) {
    const Vec<decltype(du64)> keys0 = Load(du64, &keys[i]);
    const Vec<decltype(du64)> keys1 = Load(du64, &keys[i + NU64]);
    Store(shard_mul.TwoVec(du32, keys0, keys1), du32, &outputs[i]);
  }
  VQSort(outputs.data(), outputs.size(), SortAscending());
  HWY_ASSERT_M(AllUnique(du32, outputs.data(), outputs.size()),
               "Collision detected");

  PROFILER_PRINT_RESULTS();
}

HWY_NOINLINE void TestShardMulTwoVec() {
  ThreadPool pool = MakePool();
  const ScalableTag<uint32_t> du32;
  const RepartitionToWide<decltype(du32)> du64;
  const size_t NU32 = Lanes(du32);
  const size_t NU64 = Lanes(du64);

  const size_t num_keys = RoundUpTo(1000, NU32);
  AlignedVector<uint64_t> keys = GenerateClusteredKeys(num_keys);

  const ShardMul shard_mul = MakeShardMul(Span(keys), pool);
  HWY_ASSERT(!shard_mul.IsEmpty());

  HWY_ALIGN uint32_t expected[MaxLanes(du32)];
  HWY_ALIGN uint32_t actual[MaxLanes(du32)];

  for (size_t i = 0; i < num_keys; i += NU32) {
    // Compute scalar reference outputs.
    for (size_t lane = 0; lane < NU32; ++lane) {
      expected[lane] = shard_mul(keys[i + lane]);
    }

    const Vec<decltype(du64)> keys0 = Load(du64, &keys[i]);
    const Vec<decltype(du64)> keys1 = Load(du64, &keys[i + NU64]);
    Store(shard_mul.TwoVec(du32, keys0, keys1), du32, actual);
    HWY_ASSERT_VEC_EQ(du32, expected, Load(du32, actual));
  }
}

#else   // HWY_TARGET == HWY_SCALAR
void TestShardMulExtraOutputs() {}
void TestShardMulCollisionFree() {}
void TestShardMulTwoVec() {}
#endif  // HWY_TARGET == HWY_SCALAR

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(ShardMulTest);
HWY_EXPORT_AND_TEST_BEST_P(ShardMulTest, TestShardMulTwoVec);
HWY_EXPORT_AND_TEST_BEST_P(ShardMulTest, TestShardMulCollisionFree);
HWY_EXPORT_AND_TEST_BEST_P(ShardMulTest, TestShardMulExtraOutputs);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
