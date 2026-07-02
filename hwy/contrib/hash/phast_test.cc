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
HWY_NOINLINE void TestQueryConsistency() {}
HWY_NOINLINE void TestMultipleSizes() {}
#else

static ThreadPool MakePool() {
  return ThreadPool(ThreadPool::NumThreadsFromCores());
}

HWY_NOINLINE void TestQueryConsistency() {
  fprintf(stderr, "=== TestQueryConsistency ===\n");
  const size_t num_keys = AdjustedReps(5'000);
  AlignedVector<uint32_t> keys(num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    keys[i] = static_cast<uint32_t>(i * 37 + 1);  // Distinct, non-sequential.
  }

  ThreadPool pool = MakePool();
  Phast phast = MakePhast(Span(keys), 0, pool);

  // Query each key twice and verify same result.
  for (size_t i = 0; i < num_keys; ++i) {
    const uint32_t idx1 = phast(keys[i]);
    const uint32_t idx2 = phast(keys[i]);
    HWY_ASSERT_M(idx1 == idx2, "Query not deterministic");
  }
  fprintf(stderr, "  OK: %zu queries consistent\n", num_keys);
}

// --------------------------------------------------------------------------
// Main test: query all keys, ensure indices distinct and in range.

// Outputs indices for a batch of keys. Considerably higher throughput than
// repeated single queries: 7.8 GB/s on Turin for 1M keys.
void QueryBatch(const uint32_t* HWY_RESTRICT keys, size_t num_keys,
                const Phast& phast, uint32_t* HWY_RESTRICT indices) {
  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

  size_t i = 0;
  if (HWY_LIKELY(num_keys >= 2 * N)) {
    for (; i <= num_keys - 2 * N; i += 2 * N) {
      VU32 v0 = Load(du32, keys + i + 0 * N);
      VU32 v1 = Load(du32, keys + i + 1 * N);
      VU32 idx0, idx1;
      phast(du32, v0, v1, idx0, idx1);
      Store(idx0, du32, indices + i + 0 * N);
      Store(idx1, du32, indices + i + 1 * N);
    }
  }
  if (HWY_UNLIKELY(i != num_keys)) {
    const size_t remaining = num_keys - i;
    HWY_DASSERT(remaining < 2 * N);
    const size_t remaining1 = remaining <= N ? 0 : remaining - N;
    VU32 v0 = LoadN(du32, keys + i + 0 * N, remaining);
    VU32 v1 = LoadN(du32, keys + i + 1 * N, remaining1);
    VU32 idx0, idx1;
    phast(du32, v0, v1, idx0, idx1);
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

void TestDistinctAndRange(const size_t num_keys) {
  ThreadPool pool = MakePool();
  AlignedVector<uint32_t> keys = FillRandomDistinct<uint32_t>(num_keys, 0);

  const double t0 = platform::Now();
  const size_t payload_bytes = 0;
  const Phast phast = MakePhast(Span(keys), payload_bytes, pool);
  const double elapsed = platform::Now() - t0;
  const PhastData& data = phast.Data();
  fprintf(stderr,
          "    Build(%7zu keys): %7.2f ms, %7zu slots, %.2f b/key config %2zu, "
          "attempt %2zu\n",
          num_keys, elapsed * 1E3, data.NumSlots(),
          static_cast<double>(phast.Data().AllocatedBytes(payload_bytes)) *
              8.0 / static_cast<double>(num_keys),
          data.config_idx, data.attempt_idx);

  // Check that all keys map to distinct indices in [0, num_slots).
  AlignedVector<uint32_t> indices(num_keys);
  QueryBatch(keys.data(), num_keys, phast, indices.data());
  CheckDistinctAndRange(indices.data(), num_keys, data.NumSlots());
}

HWY_NOINLINE void TestMultipleSizes() {
  const size_t kMul = 1;  // increase for larger tests.
  fprintf(stderr, "=== TestSmall ===\n");
  for (size_t num_keys = 1; num_keys < 64; ++num_keys) {
    TestDistinctAndRange(num_keys);
  }
  TestDistinctAndRange(/*num_keys=*/AdjustedReps(AdjustedReps(100 * kMul)));
  fprintf(stderr, "=== TestMedium ===\n");
  TestDistinctAndRange(/*num_keys=*/AdjustedReps(AdjustedReps(500 * kMul)));
  fprintf(stderr, "=== TestLarge ===\n");
  TestDistinctAndRange(
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
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestQueryConsistency);
HWY_EXPORT_AND_TEST_BEST_P(PhastTest, TestMultipleSizes);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
