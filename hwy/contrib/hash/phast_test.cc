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

// Tests for PHAST perfect hash.

#include <stdint.h>
#include <stdio.h>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/hash/phast.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/nanobenchmark.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/phast_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/hash/phast-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// Phast is not supported on HWY_SCALAR and too slow on HWY_EMU128.
#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestSliceInvariant() {}
HWY_NOINLINE void TestDisjointSlicesNeverCollide() {}
HWY_NOINLINE void TestSliceBoundIsTight() {}
HWY_NOINLINE void TestQueryConsistency() {}
HWY_NOINLINE void TestMultipleSizes() {}
#else

static ThreadPool MakePool() {
  return ThreadPool(ThreadPool::NumThreadsFromCores());
}

// --------------------------------------------------------------------------
// Placement structure. The builder's collision and overlap checks rely on
// PosFromHashAndSeed decomposing into a seed-independent slice base plus an
// offset smaller than slice_length. These tests pin that decomposition so a
// change to Placement/Hash16 that broke it would fail here rather than silently
// invalidate the builder's pruning.

// Every seed places a key inside its own slice.
HWY_NOINLINE void TestSliceInvariant() {
  fprintf(stderr, "=== TestSliceInvariant ===\n");
  const size_t kNumSlots[] = {203, 1021, 12289, 206001};
  const uint32_t kSliceLengths[] = {64, 128, 512, 2048};
  size_t checked = 0;
  for (size_t num_slots : kNumSlots) {
    for (uint32_t slice_length : kSliceLengths) {
      if (num_slots < slice_length) continue;
      const PhastPlacement pp(num_slots, slice_length);
      uint32_t hash = 0x9E3779B9u;
      for (size_t t = 0; t < 500; ++t) {
        hash = hash * 1664525u + 1013904223u;
        const uint32_t base = LemireMod(hash, pp.num_slice_offsets);
        for (uint32_t seed = 0; seed < 256; ++seed) {
          const uint32_t pos = Phast::PosFromHashAndSeed(pp, hash, seed);
          HWY_ASSERT_M(pos >= base, "position below its slice");
          HWY_ASSERT_M(pos - base < slice_length, "position beyond its slice");
          HWY_ASSERT_M(pos < num_slots, "position outside the table");
          ++checked;
        }
      }
    }
  }
  fprintf(stderr, "  OK: %zu (hash, seed) pairs inside their slice\n", checked);
}

// Keys whose slices are disjoint cannot share a slot at any seed. This is what
// lets the builder skip those pairs instead of retesting them per seed.
HWY_NOINLINE void TestDisjointSlicesNeverCollide() {
  fprintf(stderr, "=== TestDisjointSlicesNeverCollide ===\n");
  const PhastPlacement pp(/*num_slots=*/1020001, /*slice_length=*/2048);
  const uint32_t slice_length = pp.slice_mask + 1;
  uint32_t hash = 12345u;
  size_t disjoint = 0;
  for (size_t t = 0; t < 20000; ++t) {
    hash = hash * 1664525u + 1013904223u;
    const uint32_t hi = hash;
    hash = hash * 1664525u + 1013904223u;
    const uint32_t hj = hash;
    const uint32_t bi = LemireMod(hi, pp.num_slice_offsets);
    const uint32_t bj = LemireMod(hj, pp.num_slice_offsets);
    const uint32_t d = bi > bj ? bi - bj : bj - bi;
    if (d < slice_length) continue;  // slices overlap; collision permitted
    ++disjoint;
    for (uint32_t seed = 0; seed < 256; ++seed) {
      HWY_ASSERT_M(Phast::PosFromHashAndSeed(pp, hi, seed) !=
                       Phast::PosFromHashAndSeed(pp, hj, seed),
                   "disjoint slices produced a collision");
    }
  }
  HWY_ASSERT_M(disjoint > 1000, "too few disjoint pairs to be meaningful");
  fprintf(stderr, "  OK: %zu random disjoint pairs, no collision\n", disjoint);

  // Random pairs land far apart and so exercise the claim weakly. The case that
  // discriminates is the tightest one: slices exactly slice_length apart, where
  // a single extra reachable offset would produce a collision. Construct those
  // pairs rather than sampling for them.
  const PhastPlacement tight(/*num_slots=*/203, /*slice_length=*/64);
  const uint32_t width = tight.slice_mask + 1;
  size_t constructed = 0;
  for (uint64_t h = 0; h <= 0xFFFFFFFFull; h += 655357) {
    const uint32_t hi2 = static_cast<uint32_t>(h);
    const uint64_t target =
        static_cast<uint64_t>(LemireMod(hi2, tight.num_slice_offsets)) + width;
    if (target >= tight.num_slice_offsets) continue;
    const uint64_t lo = ((target << 32) + tight.num_slice_offsets - 1) /
                        tight.num_slice_offsets;
    if (lo > 0xFFFFFFFFull) continue;
    const uint32_t hj2 = static_cast<uint32_t>(lo);
    if (LemireMod(hj2, tight.num_slice_offsets) != target) continue;
    ++constructed;
    for (uint32_t seed = 0; seed < 256; ++seed) {
      HWY_ASSERT_M(Phast::PosFromHashAndSeed(tight, hi2, seed) !=
                       Phast::PosFromHashAndSeed(tight, hj2, seed),
                   "slices exactly slice_length apart must not collide");
    }
  }
  HWY_ASSERT_M(constructed > 500, "too few tight pairs constructed");
  fprintf(stderr, "  OK: %zu pairs exactly slice_length apart, no collision\n",
          constructed);
}

// The bound above is `< slice_length`, not `< slice_length - 1`: slices exactly
// slice_length - 1 apart share one slot and that slot is reachable. Regression
// case for a config the builder enumerates (num_keys 200, headroom 1%).
HWY_NOINLINE void TestSliceBoundIsTight() {
  fprintf(stderr, "=== TestSliceBoundIsTight ===\n");
  const PhastPlacement pp(/*num_slots=*/203, /*slice_length=*/64);
  const uint32_t hi = 0x000c0000u, hj = 0x73570000u, seed = 0;
  const uint32_t bi = LemireMod(hi, pp.num_slice_offsets);
  const uint32_t bj = LemireMod(hj, pp.num_slice_offsets);
  HWY_ASSERT_M(bj - bi == pp.slice_mask,
               "witness slices not slice_length-1 apart");
  const uint32_t pi = Phast::PosFromHashAndSeed(pp, hi, seed);
  const uint32_t pj = Phast::PosFromHashAndSeed(pp, hj, seed);
  HWY_ASSERT_M(pi == pj,
               "witness pair must collide; the bound cannot be tightened");
  fprintf(stderr, "  OK: slices %u apart (slice_length-1) collide at slot %u\n",
          bj - bi, pi);
}

template <typename KeyT>
HWY_NOINLINE void TestQueryConsistencyT() {
  fprintf(stderr, "=== TestQueryConsistency (%zu-bit) ===\n", sizeof(KeyT) * 8);
  const size_t num_keys = AdjustedReps(5'000);
  AlignedVector<KeyT> keys(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys[i] = static_cast<KeyT>(i * 37 + 1);  // Distinct, non-sequential.
  }

  ThreadPool pool = MakePool();
  PhastT<KeyT> phast = MakePhast(Span(keys), 0, pool);

  // Query each key twice and verify same result.
  for (size_t i = 0; i < num_keys; ++i) {
    const uint32_t idx1 = phast(keys[i]);
    const uint32_t idx2 = phast(keys[i]);
    HWY_ASSERT_M(idx1 == idx2, "Query not deterministic");
  }
  fprintf(stderr, "  OK: %zu queries consistent\n", num_keys);
}

HWY_NOINLINE void TestQueryConsistency() {
  TestQueryConsistencyT<uint32_t>();
  TestQueryConsistencyT<uint64_t>();
}

// --------------------------------------------------------------------------
// Main test: query all keys, ensure indices distinct and in range.

// Outputs indices for a batch of keys. Considerably higher throughput than
// repeated single queries: 7.8 GB/s on Turin for 1M keys.
template <typename KeyT>
void QueryBatch(const KeyT* HWY_RESTRICT keys, size_t num_keys,
                const PhastT<KeyT>& phast, uint32_t* HWY_RESTRICT indices) {
  const ScalableTag<KeyT> d;
  using V = Vec<decltype(d)>;
  const auto du32 = DemoteTag32(d);
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(d);

  size_t i = 0;
  if (HWY_LIKELY(num_keys >= 2 * N)) {
    for (; i <= num_keys - 2 * N; i += 2 * N) {
      V v0 = Load(d, keys + i + 0 * N);
      V v1 = Load(d, keys + i + 1 * N);
      VU32 idx0, idx1;
      phast(d, v0, v1, idx0, idx1);
      Store(idx0, du32, indices + i + 0 * N);
      Store(idx1, du32, indices + i + 1 * N);
    }
  }
  if (HWY_UNLIKELY(i != num_keys)) {
    const size_t remaining = num_keys - i;
    HWY_DASSERT(remaining < 2 * N);
    const size_t remaining1 = remaining <= N ? 0 : remaining - N;
    V v0 = LoadN(d, keys + i + 0 * N, remaining);
    V v1 = LoadN(d, keys + i + 1 * N, remaining1);
    VU32 idx0, idx1;
    phast(d, v0, v1, idx0, idx1);
    StoreN(idx0, du32, indices + i + 0 * N, remaining);
    StoreN(idx1, du32, indices + i + 1 * N, remaining1);
  }
}

// Mutates input.
void CheckDistinctAndRange(uint32_t* indices, size_t num_indices,
                           size_t num_slots) {
  VQSort(indices, num_indices, SortAscending());
  const ScalableTag<uint32_t> du32;
  HWY_ASSERT_M(num_indices == Unique(du32, indices, num_indices),
               "Collision detected");

  for (size_t i = 0; i < num_indices; ++i) {
    HWY_ASSERT_M(indices[i] < num_slots, "Index out of range");
  }
}

template <typename KeyT>
void TestDistinctAndRange(const size_t num_keys) {
  ThreadPool pool = MakePool();
  AlignedVector<KeyT> keys = FillRandomDistinct<KeyT>(num_keys, 0);

  const double t0 = platform::Now();
  const size_t payload_bytes = 0;
  const PhastT<KeyT> phast = MakePhast(Span(keys), payload_bytes, pool);
  const double elapsed = platform::Now() - t0;
  const PhastData& data = phast.Data();
  fprintf(
      stderr,
      "    Build(%7zu %zu-bit keys): %7.2f ms, %7zu slots, %.2f b/key config "
      "%2zu, attempt %2zu\n",
      num_keys, sizeof(KeyT) * 8, elapsed * 1E3, data.NumSlots(),
      static_cast<double>(phast.Data().AllocatedBytes(payload_bytes)) * 8.0 /
          static_cast<double>(num_keys),
      data.config_idx, data.attempt_idx);

  // Check that all keys map to distinct indices in [0, num_slots).
  AlignedVector<uint32_t> indices(num_keys);
  QueryBatch(keys.data(), num_keys, phast, indices.data());
  CheckDistinctAndRange(indices.data(), num_keys, data.NumSlots());
}

HWY_NOINLINE void TestMultipleSizes() {
  const size_t kMul = 1;  // increase for larger tests.
  fprintf(stderr, "=== TestSmall (32-bit) ===\n");
  // Includes num_keys == 64, where MinSliceLength(num_keys) == num_keys.
  for (size_t num_keys = 1; num_keys < 100; ++num_keys) {
    TestDistinctAndRange<uint32_t>(num_keys);
  }
  TestDistinctAndRange<uint32_t>(
      /*num_keys=*/AdjustedReps(AdjustedReps(100 * kMul)));
  fprintf(stderr, "=== TestSmall (64-bit) ===\n");
  for (size_t num_keys = 1; num_keys < 100; ++num_keys) {
    TestDistinctAndRange<uint64_t>(num_keys);
  }
  TestDistinctAndRange<uint64_t>(
      /*num_keys=*/AdjustedReps(AdjustedReps(100 * kMul)));

  fprintf(stderr, "=== TestMedium (32-bit) ===\n");
  TestDistinctAndRange<uint32_t>(
      /*num_keys=*/AdjustedReps(AdjustedReps(500 * kMul)));
  fprintf(stderr, "=== TestMedium (64-bit) ===\n");
  TestDistinctAndRange<uint64_t>(
      /*num_keys=*/AdjustedReps(AdjustedReps(500 * kMul)));

  fprintf(stderr, "=== TestLarge (32-bit) ===\n");
  TestDistinctAndRange<uint32_t>(
      /*num_keys=*/AdjustedReps(AdjustedReps(2 * kMul)) * 1024);
  fprintf(stderr, "=== TestLarge (64-bit) ===\n");
  TestDistinctAndRange<uint64_t>(
      /*num_keys=*/AdjustedReps(AdjustedReps(2 * kMul)) * 1024);

  PROFILER_PRINT_RESULTS();
}

#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(PhastTest);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestSliceInvariant);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestDisjointSlicesNeverCollide);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestSliceBoundIsTight);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestQueryConsistency);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestMultipleSizes);
HWY_AFTER_TEST();

// An empty key set has no perfect hash; BuildPhast must return the empty
// sentinel (NumSlots() == 0) rather than aborting.
TEST(PhastEmptyTest, EmptyKeysReturnEmpty) {
  ThreadPool pool(0);
  const uint32_t* no_keys32 = nullptr;
  const PhastData data32 =
      BuildPhast(Span<const uint32_t>(no_keys32, 0), 0, pool);
  HWY_ASSERT_EQ(size_t{0}, data32.NumSlots());

  const uint64_t* no_keys64 = nullptr;
  const PhastData data64 =
      BuildPhast(Span<const uint64_t>(no_keys64, 0), 0, pool);
  HWY_ASSERT_EQ(size_t{0}, data64.NumSlots());
}
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
