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

// This file sweeps across Cuckoo hash table bucket sizes and load factors to
// see:
// 1. If we can build a valid Cuckoo hash table.
// 2. How many keys can be inserted in primary and secondary buckets.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/hash/cuckoo2x2.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/cuckoo_load_factor_sweep.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/cuckoo-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestAllBucketSizeSweep(size_t /*num_keys*/) {}
#else

static ThreadPool MakePool() {
  static Topology topology;
  if (topology.packages.empty()) return ThreadPool(ThreadPool::MaxThreads());
  // Minus one because these are in addition to the main thread.
  return ThreadPool(ThreadPool::NumThreadsFromCores());
}

static AlignedVector<uint32_t> GenerateKeys(size_t num_keys,
                                            uint64_t seed = 0) {
  if (num_keys >= 1000000) {
    fprintf(stderr, "GenerateKeys(%zu) starting...\n", num_keys);
  }
  AlignedVector<uint32_t> keys(num_keys);
  AesCtrEngine engine(/*deterministic=*/true);
  Triple32 perm(engine, seed);
  for (uint32_t i = 0; i < num_keys; ++i) {
    keys[i] = perm(i);
    // Ensure no key equals the sentinel value.
    if (keys[i] == CuckooTable::kEmpty) perm(num_keys + i);
  }
  if (num_keys >= 1000000) {
    fprintf(stderr, "GenerateKeys(%zu) finished.\n", num_keys);
  }
  return keys;
}

template <uint32_t kBucketSize>
void TestBucketSize(size_t num_keys) {
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);
  auto keys = GenerateKeys(num_keys);
  CuckooBuildStats stats;
  CuckooTraits<WeakTwoMul, kBucketSize> traits;
  auto table = CuckooBuild(traits, keys.data(), static_cast<uint32_t>(num_keys),
                           /*epsilon=*/0.75,
                           /*max_attempts=*/200,
                           /*optimize_primary=*/false, &stats);

  if (!stats.success) {
    HWY_ABORT("  bucket_size=%u, keys=%zu: FAILED after %u attempts\n",
              kBucketSize, num_keys, stats.attempts);
  }

    const uint32_t num_secondary =
        static_cast<uint32_t>(num_keys) - stats.num_primary;
    fprintf(stderr,
            "  bucket_size=%u, keys=%zu: primary=%u (%.1f%%), "
            "secondary=%u (%.1f%%), buckets=%zu\n",
            kBucketSize, num_keys, stats.num_primary,
            100.0 * stats.num_primary / num_keys, num_secondary,
            100.0 * num_secondary / num_keys, table.GetConfig().NumBuckets());

    // Verify query correctness for every key.
    for (size_t i = 0; i < num_keys; ++i) {
      HWY_ASSERT_M(table.QueryOne(keys[i]),
                   "BucketSizeSweep: QueryOne missed a key");
    }

  auto table2x2 = BuildCuckoo2x2(keys, pool);
  fprintf(stderr,
          "  bucket_size=%u, keys=%zu: 2x2 primary=%u (%.1f%%), "
          "secondary=%zu (%.1f%%), buckets=%zu\n",
          kBucketSize, num_keys, table2x2.num_primary,
          100.0 * table2x2.num_primary / num_keys,
          num_keys - table2x2.num_primary,
          100.0 * (num_keys - table2x2.num_primary) / num_keys,
          table2x2.config.NumBuckets());
}

HWY_NOINLINE void TestAllBucketSizeSweep(size_t num_keys) {
  fprintf(stderr, "=== TestBucketSizeSweep (num_keys=%zu, epsilon=1.0) ===\n",
          num_keys);
  TestBucketSize<1>(num_keys);
  TestBucketSize<2>(num_keys);
  TestBucketSize<4>(num_keys);
  TestBucketSize<8>(num_keys);
  TestBucketSize<16>(num_keys);
  TestBucketSize<32>(num_keys);
}

#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(TestAllBucketSizeSweep);

void Run(size_t num_keys) {
  HWY_DYNAMIC_DISPATCH(TestAllBucketSizeSweep)(num_keys);
}
}  // namespace
}  // namespace hwy

int main(int argc, char** argv) {
  size_t num_keys = 224000;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--num_keys=", 11) == 0) {
      num_keys = static_cast<size_t>(strtoull(argv[i] + 11, nullptr, 10));
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      fprintf(stderr, "Usage: %s [--num_keys=224000]\n", argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown flag: %s\nUsage: %s [--num_keys=N]\n", argv[i],
              argv[0]);
      return 1;
    }
  }
  if (num_keys == 0) {
    fprintf(stderr, "Error: --num_keys must be > 0\n");
    return 1;
  }
  hwy::Run(num_keys);
  return 0;
}
#endif  // HWY_ONCE
