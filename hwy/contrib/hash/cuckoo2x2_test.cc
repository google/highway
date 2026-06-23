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

// Tests for Cuckoo2x2 two-choice Set2.

#include <stdint.h>
#include <stdio.h>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/hash/cuckoo2x2.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/nanobenchmark.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/cuckoo2x2_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/cuckoo2x2-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestScalarContains() {}
HWY_NOINLINE void TestSimdContains() {}
HWY_NOINLINE void TestMultipleSizes() {}
#else

static ThreadPool MakePool() {
  return ThreadPool(ThreadPool::NumThreadsFromCores());
}

// --------------------------------------------------------------------------
// Scalar: verify Contains() for all keys.

HWY_NOINLINE void TestScalarContains() {
  fprintf(stderr, "=== TestScalarContains ===\n");
  const uint32_t num_keys = AdjustedReps(1'000);
  AlignedVector<uint32_t> keys(num_keys);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = i * 37 + 1;
  }

  ThreadPool pool = MakePool();
  Cuckoo2x2 set = MakeCuckoo2x2(Span(keys), pool);

  // All keys must be found.
  for (uint32_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT_M(set.Contains(keys[i]), "Key not found (scalar)");
  }
  fprintf(stderr, "  OK: %u scalar lookups\n", num_keys);
}

// --------------------------------------------------------------------------
// SIMD: verify batch query matches scalar.

void QueryBatch(const uint32_t* HWY_RESTRICT keys, size_t num_keys,
                const Cuckoo2x2& set, uint8_t* HWY_RESTRICT results) {
  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  using MU32 = Mask<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);

  size_t i = 0;
  if (HWY_LIKELY(num_keys >= N)) {
    for (; i <= num_keys - N; i += N) {
      VU32 v = Load(du32, keys + i);
      MU32 nf = set(du32, v);
      // nf = not-found mask: true = NOT in set.
      const VU32 r = VecFromMask(du32, Not(nf));
      for (size_t j = 0; j < N; ++j) {
        results[i + j] = ExtractLane(r, j) != 0 ? 1 : 0;
      }
    }
  }
  // Scalar tail.
  for (; i < num_keys; ++i) {
    results[i] = set.Contains(keys[i]) ? 1 : 0;
  }
}

HWY_NOINLINE void TestSimdContains() {
  fprintf(stderr, "=== TestSimdContains ===\n");
  const uint32_t num_keys = AdjustedReps(1'000);
  AlignedVector<uint32_t> keys(num_keys);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = i * 37 + 1;
  }

  ThreadPool pool = MakePool();
  Cuckoo2x2 set = MakeCuckoo2x2(Span(keys), pool);

  AlignedVector<uint8_t> results(num_keys);
  QueryBatch(keys.data(), num_keys, set, results.data());

  for (uint32_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT_M(results[i] == 1, "Key not found (SIMD)");
  }
  fprintf(stderr, "  OK: %u SIMD lookups\n", num_keys);
}

// --------------------------------------------------------------------------
// Multiple sizes: verify build + query for various key counts.

void TestBuildAndQuery(const size_t num_keys) {
  ThreadPool pool = MakePool();
  AlignedVector<uint32_t> keys = FillRandomDistinct<uint32_t>(num_keys, 0);

  const double t0 = platform::Now();
  Cuckoo2x2 set = MakeCuckoo2x2(Span(keys), pool);
  const double elapsed = platform::Now() - t0;
  const Cuckoo2x2Data& data = set.Data();
  fprintf(stderr,
          "    Build(%7zu keys): %7.2f ms, %7zu buckets, %.2f b/key "
          "config %2zu, attempt %2zu\n",
          num_keys, elapsed * 1E3, data.NumBuckets(),
          static_cast<double>(data.AllocatedBytes()) * 8.0 /
              static_cast<double>(num_keys),
          data.config_idx, data.attempt_idx);

  // Check all keys found via scalar.
  for (size_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT_M(set.Contains(keys[i]), "False negative (scalar)");
  }

  // Check all keys found via SIMD.
  AlignedVector<uint8_t> results(num_keys);
  QueryBatch(keys.data(), num_keys, set, results.data());
  for (size_t i = 0; i < num_keys; ++i) {
    HWY_ASSERT_M(results[i] == 1, "False negative (SIMD)");
  }
}

HWY_NOINLINE void TestMultipleSizes() {
  fprintf(stderr, "=== TestMultipleSizes ===\n");
  TestBuildAndQuery(/*num_keys=*/AdjustedReps(6'000));
  TestBuildAndQuery(/*num_keys=*/AdjustedReps(60'000));

  PROFILER_PRINT_RESULTS();
}

#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128
}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(Cuckoo2x2Test);
HWY_EXPORT_AND_TEST_BEST_P(Cuckoo2x2Test, TestScalarContains);
HWY_EXPORT_AND_TEST_BEST_P(Cuckoo2x2Test, TestSimdContains);
HWY_EXPORT_AND_TEST_BEST_P(Cuckoo2x2Test, TestMultipleSizes);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
