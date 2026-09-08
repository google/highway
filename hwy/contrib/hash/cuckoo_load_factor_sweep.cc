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
#include "hwy/timer.h"

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
HWY_NOINLINE void TestAllBucketSizeSweep(size_t /*num_keys*/,
                                         double /*epsilon*/, bool /*pow2*/,
                                         uint32_t /*max_attempts*/) {}
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

static const char* AlgoName(CuckooBuildAlgo algo) {
  switch (algo) {
    case CuckooBuildAlgo::kHopcroftKarp:
      return "HopcroftKarp";
    case CuckooBuildAlgo::kMinCost:
      return "MinCost";
    case CuckooBuildAlgo::kLocalSearch:
      return "LocalSearch";
  }
  return "Unknown";
}

template <uint32_t kBucketSize, bool kPow2>
void TestBucketSizeAndPow2(size_t num_keys, double epsilon,
                           uint32_t max_attempts,
                           const AlignedVector<uint32_t>& keys) {
  for (CuckooBuildAlgo algo :
       {CuckooBuildAlgo::kHopcroftKarp, CuckooBuildAlgo::kMinCost,
        CuckooBuildAlgo::kLocalSearch}) {
    CuckooBuildStats stats;
    const CuckooTraits<WeakTwoMul, kBucketSize, /*kMinBuckets_=*/1, kPow2>
        traits;
    CuckooBuildArgs args;
    args.epsilon = epsilon;
    args.max_attempts = max_attempts;
    args.algo = algo;
    const double t0 = platform::Now();
    auto table = CuckooBuild(
        traits, Span<const uint32_t>(keys.data(), num_keys), args, &stats);
    const double build_ms = (platform::Now() - t0) * 1000.0;

    if (!stats.success) {
      fprintf(stderr,
              "  algo=%-12s pow2=%d bucket_size=%2u keys=%zu eps=%.2f: FAILED "
              "after %u attempts (%.2f ms)\n",
              AlgoName(algo), static_cast<int>(kPow2), kBucketSize, num_keys,
              epsilon, stats.attempts, build_ms);
      continue;
    }

    const uint32_t num_secondary =
        static_cast<uint32_t>(num_keys) - stats.num_primary;
    fprintf(
        stderr,
        "  algo=%-12s pow2=%d bucket_size=%2u keys=%zu eps=%.2f: primary=%u "
        "(%.1f%%), secondary=%u (%.1f%%), buckets=%zu, build_time=%.2f ms\n",
        AlgoName(algo), static_cast<int>(kPow2), kBucketSize, num_keys, epsilon,
        stats.num_primary, 100.0 * stats.num_primary / num_keys, num_secondary,
        100.0 * num_secondary / num_keys, table.GetConfig().NumBuckets(),
        build_ms);

    // Verify query correctness for every key.
    for (size_t i = 0; i < num_keys; ++i) {
      HWY_ASSERT_M(table.QueryOne(keys[i]),
                   "BucketSizeSweep: QueryOne missed a key");
    }
  }
}

template <uint32_t kBucketSize>
void TestBucketSize(size_t num_keys, double epsilon, bool pow2,
                    uint32_t max_attempts, const AlignedVector<uint32_t>& keys,
                    ThreadPool& pool) {
  if (pow2) {
    TestBucketSizeAndPow2<kBucketSize, true>(num_keys, epsilon, max_attempts,
                                             keys);
  } else {
    TestBucketSizeAndPow2<kBucketSize, false>(num_keys, epsilon, max_attempts,
                                              keys);
  }

  const double t_2x2_start = platform::Now();
  auto table2x2 = BuildCuckoo2x2(keys, pool);
  const double build_2x2_ms = (platform::Now() - t_2x2_start) * 1000.0;
  fprintf(stderr,
          "  algo=%-12s pow2=1 bucket_size=%2u keys=%zu eps=%.2f: primary=%u "
          "(%.1f%%), secondary=%zu (%.1f%%), buckets=%zu, build_time=%.2f ms\n",
          "Cuckoo2x2", kBucketSize, num_keys, epsilon, table2x2.num_primary,
          100.0 * table2x2.num_primary / num_keys,
          num_keys - table2x2.num_primary,
          100.0 * (num_keys - table2x2.num_primary) / num_keys,
          table2x2.config.NumBuckets(), build_2x2_ms);
}

HWY_NOINLINE void TestAllBucketSizeSweep(size_t num_keys, double epsilon,
                                         bool pow2, uint32_t max_attempts) {
  fprintf(stderr,
          "=== TestBucketSizeSweep (num_keys=%zu, epsilon=%.2f, pow2=%d, "
          "max_attempts=%u) ===\n",
          num_keys, epsilon, pow2, max_attempts);
  ThreadPool pool = MakePool();
  pool.SetWaitMode(PoolWaitMode::kSpin);
  auto keys = GenerateKeys(num_keys);
  TestBucketSize<1>(num_keys, epsilon, pow2, max_attempts, keys, pool);
  TestBucketSize<2>(num_keys, epsilon, pow2, max_attempts, keys, pool);
  TestBucketSize<4>(num_keys, epsilon, pow2, max_attempts, keys, pool);
  TestBucketSize<8>(num_keys, epsilon, pow2, max_attempts, keys, pool);
  TestBucketSize<16>(num_keys, epsilon, pow2, max_attempts, keys, pool);
  TestBucketSize<32>(num_keys, epsilon, pow2, max_attempts, keys, pool);
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

void Run(size_t num_keys, double epsilon, bool pow2, uint32_t max_attempts) {
  HWY_DYNAMIC_DISPATCH(TestAllBucketSizeSweep)
  (num_keys, epsilon, pow2, max_attempts);
}
}  // namespace
}  // namespace hwy

int main(int argc, char** argv) {
  size_t num_keys = 224000;
  double epsilon = 1.0;
  bool pow2 = true;
  uint32_t max_attempts = 200;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--num_keys=", 11) == 0) {
      num_keys = static_cast<size_t>(strtoull(argv[i] + 11, nullptr, 10));
    } else if (strncmp(argv[i], "--epsilon=", 10) == 0) {
      epsilon = strtod(argv[i] + 10, nullptr);
    } else if (strncmp(argv[i], "--max_attempts=", 15) == 0) {
      max_attempts = static_cast<uint32_t>(strtoull(argv[i] + 15, nullptr, 10));
    } else if (strncmp(argv[i], "--pow2=", 7) == 0) {
      pow2 =
          (strcmp(argv[i] + 7, "1") == 0 || strcmp(argv[i] + 7, "true") == 0);
    } else if (strcmp(argv[i], "--pow2") == 0) {
      pow2 = true;
    } else if (strcmp(argv[i], "--nopow2") == 0) {
      pow2 = false;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      fprintf(stderr,
              "Usage: %s [--num_keys=224000] [--epsilon=1.0] [--pow2=1|0] "
              "[--max_attempts=200]\n",
              argv[0]);
      return 0;
    } else {
      fprintf(stderr,
              "Unknown flag: %s\nUsage: %s [--num_keys=N] [--epsilon=E] "
              "[--pow2=1|0] "
              "[--max_attempts=A]\n",
              argv[i], argv[0]);
      return 1;
    }
  }
  if (num_keys == 0) {
    fprintf(stderr, "Error: --num_keys must be > 0\n");
    return 1;
  }
  hwy::Run(num_keys, epsilon, pow2, max_attempts);
  return 0;
}
#endif  // HWY_ONCE
