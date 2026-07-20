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

#include <stdint.h>
#include <stdio.h>

#include <cmath>

#include "hwy/base.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/nanobenchmark.h"  // Unpredictable1
#include "hwy/stats.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/hash_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/bit_set.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if (HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128) || HWY_IDE

HWY_MAYBE_UNUSED ThreadPool MakePool(size_t max_threads = 31) {
  return ThreadPool(HWY_MIN(ThreadPool::MaxThreads(), max_threads));
}

template <typename T>
HWY_MAYBE_UNUSED void AssertNear(const char* name, double expected, T actual,
                                 double rel = 0.01) {
  const double tol = std::abs(expected * rel);
  if (actual < expected - tol || actual > expected + tol) {
    HWY_ABORT("%s: %f outside [%f, %f]", name, actual, expected - tol,
              expected + tol);
  }
}

template <typename T>
HWY_MAYBE_UNUSED void WarnIfNotNear(const char* name, double expected, T actual,
                                    double rel = 0.01) {
  const double tol = std::abs(expected * rel);
  if (actual < expected - tol || actual > expected + tol) {
    HWY_WARN("%s: %f outside [%f, %f]", name, actual, expected - tol,
             expected + tol);
  }
}

template <class Hash>
static HWY_NOINLINE void TestMasked() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);
  Hash hash(engine, 0);

  using T = typename Hash::LaneType;
  const T mask_val = Hash::kMask;

  // Scalar test
  for (size_t trial = 0; trial < 1000; ++trial) {
    const T input = static_cast<T>(rng()) & mask_val;
    const T output = hash(input);
    if (output > mask_val) {
      HWY_ABORT(
          "Scalar output exceeded mask: %s, trial %zu, input %zx, output "
          "%zx, mask %zx",
          Hash::Name(), trial, static_cast<size_t>(input),
          static_cast<size_t>(output), static_cast<size_t>(mask_val));
    }
  }

  // Vector test
  const ScalableTag<T> d;
  using V = Vec<decltype(d)>;
  const size_t N = Lanes(d);
  AlignedVector<T> in_buf(N);
  AlignedVector<T> out_buf(N);

  for (size_t trial = 0; trial < 100; ++trial) {
    for (size_t i = 0; i < N; ++i) {
      in_buf[i] = static_cast<T>(rng()) & mask_val;
    }
    V in = Load(d, in_buf.data());
    V out = hash.OneVec(d, in);
    Store(out, d, out_buf.data());

    for (size_t i = 0; i < N; ++i) {
      if (out_buf[i] > mask_val) {
        HWY_ABORT(
            "Vector output exceeded mask: %s, trial %zu, lane %zu, input %zx, "
            "output %zx, mask %zx",
            Hash::Name(), trial, i, static_cast<size_t>(in_buf[i]),
            static_cast<size_t>(out_buf[i]), static_cast<size_t>(mask_val));
      }
    }
  }
}

static HWY_NOINLINE void TestAllMasked() {
  // Test a few sizes for 32-bit hashes
  TestMasked<MaskedWeakTwoMul<1>>();
  TestMasked<MaskedWeakTwoMul<7>>();
  TestMasked<MaskedWeakTwoMul<13>>();
  TestMasked<MaskedWeakTwoMul<31>>();

  TestMasked<MaskedTriple32<1>>();
  TestMasked<MaskedTriple32<7>>();
  TestMasked<MaskedTriple32<13>>();
  TestMasked<MaskedTriple32<31>>();

  // Test a few sizes for 64-bit hashes
  TestMasked<MaskedMoremur<1>>();
  TestMasked<MaskedMoremur<7>>();
  TestMasked<MaskedMoremur<31>>();
  TestMasked<MaskedMoremur<63>>();
}

// Strict Avalanche Criterion: flipping any single input bit should cause
// each output bit to flip with ~50% probability.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestAvalanche(const Hash& hash) {
  AesCtrEngine engine(/*deterministic=*/true);

  // Each worker does the same number of trials with differing RNG stream.
  ThreadPool pool = MakePool();

  // 8x prevents false sharing.
  AlignedVector<size_t> all_max_abs_diff(pool.NumWorkers() * 8);

  pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t worker) {
    RngStream rng(engine, task);

    // For each of 32 input bit positions, count how often each output bit
    // flips. `flip_count[input_bit][output_bit]` = output bit flips.
    uint32_t flip_count[32][32] = {};

    constexpr size_t kNumTrials = AdjustedReps(10'000);
    for (size_t trial = 0; trial < kNumTrials; ++trial) {
      const uint32_t base = static_cast<uint32_t>(rng() & 0xFFFFFFFFu);

      for (size_t bit = 0; bit < 32; ++bit) {
        const uint32_t flipped = base ^ (1u << bit);

        const uint32_t h0 = hash(base);
        const uint32_t h1 = hash(flipped);

        uint32_t diff = h0 ^ h1;
        for (size_t obit = 0; obit < 32; ++obit) {
          if (diff & (1u << obit)) {
            flip_count[bit][obit]++;
          }
        }
      }
    }

    // Each (input_bit, output_bit) pair should flip ~50% of the time.
    // Allow 40-60% range for non-cryptographic use.
    const double num_trials = static_cast<double>(kNumTrials);
    const int32_t lo = static_cast<int32_t>(num_trials * 0.40);
    const int32_t hi = static_cast<int32_t>(num_trials * 0.60);
    size_t violations = 0;

    const int32_t expected = static_cast<int32_t>(kNumTrials / 2);
    uint64_t sum_abs_diff = 0;
    int32_t max_abs_diff = 0;

    for (size_t ibit = 0; ibit < 32; ++ibit) {
      for (size_t obit = 0; obit < 32; ++obit) {
        const int32_t actual = static_cast<int32_t>(flip_count[ibit][obit]);
        const int32_t abs_diff = std::abs(actual - expected);
        sum_abs_diff = sum_abs_diff + static_cast<uint64_t>(abs_diff);
        max_abs_diff = HWY_MAX(max_abs_diff, abs_diff);

        if (actual < lo || actual > hi) {
          if (violations < 20) {  // Don't flood output.
            HWY_WARN(
                "Avalanche violation: input_bit=%2zu, output_bit=%2zu, "
                "flip_count=%4ud (expected ~%4d, range [%4d, %4d])",
                ibit, obit, actual, expected, lo, hi);
          }
          violations++;
        }
      }
    }

    all_max_abs_diff[worker * 8] = static_cast<size_t>(max_abs_diff);
    const double avg_abs_diff = static_cast<double>(sum_abs_diff) / (32 * 32);
    WarnIfNotNear("avg abs diff", 40.0, avg_abs_diff, 0.06);

    if (violations > 0) {
      HWY_WARN("Total avalanche violations: %zu / 1024", violations);
      // For a non-cryptographic hash, some violations are acceptable.
      // Fail only if more than 5% of bit pairs are out of range.
      HWY_ASSERT(violations < 52);  // 5% of 1024
    }
  });

  for (size_t i = 1; i < pool.NumWorkers(); ++i) {
    all_max_abs_diff[0] = HWY_MAX(all_max_abs_diff[0], all_max_abs_diff[i]);
  }
  fprintf(stderr, "max_abs_diff: %zu\n", all_max_abs_diff[0]);
}

static HWY_NOINLINE void TestAllAvalanche() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s\n", hash.Name());
    TestAvalanche(hash);
  });
}

// Test that each output bit is approximately unbiased (50% zeros, 50% ones).
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestBias(const Hash& hash) {
  AesCtrEngine engine(/*deterministic=*/true);

  // Each worker does the same number of trials with differing RNG stream.
  ThreadPool pool = MakePool();
  pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t /*worker*/) {
    RngStream rng(engine, task);

    uint32_t bit_count[32] = {};  // Count of 1s in each output bit position.

    constexpr uint32_t kNumTrials = AdjustedReps(100'000);
    for (uint32_t trial = 0; trial < kNumTrials; ++trial) {
      const uint32_t val = static_cast<uint32_t>(rng() & 0xFFFFFFFFu);
      const uint32_t h = hash(val);

      for (size_t bit = 0; bit < 32; ++bit) {
        if (h & (1u << bit)) {
          bit_count[bit]++;
        }
      }
    }

    // Each bit should be ~50% ones. 2.5% tolerance.
    const double kTol = kNumTrials < 20'000 ? 0.05 : 0.025;
    const uint32_t lo = static_cast<uint32_t>(kNumTrials * (0.5 - kTol));
    const uint32_t hi = static_cast<uint32_t>(kNumTrials * (0.5 + kTol));

    for (size_t bit = 0; bit < 32; ++bit) {
      if (bit_count[bit] < lo || bit_count[bit] > hi) {
        HWY_ABORT("Bias: stream %zu, bit=%zu, ones=%u/%u (%.2f%% != ~50%%).",
                  static_cast<size_t>(task), bit, bit_count[bit], kNumTrials,
                  100.0 * bit_count[bit] / kNumTrials);
      }
    }
  });
}

static HWY_NOINLINE void TestAllBias() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s\n", hash.Name());
    TestBias(hash);
  });
}

// Enumerates all 2^32 inputs and computes histogram of hash values.
// Parallelized and vectorized, < 1 sec in opt builds.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestBuckets(const Hash& hash) {
  if constexpr (HWY_IS_DEBUG_BUILD) return;  // too slow

  // Each worker hashes a range of inputs and updates its bucket counts.
  constexpr size_t kNumBuckets = 0x10000;  // one per 64k output values

  ThreadPool pool = MakePool();
  const size_t num_workers = pool.NumWorkers();

  AlignedFreeUniquePtr<uint32_t[]> all_buckets =
      AllocateAligned<uint32_t>(kNumBuckets * num_workers);
  ZeroBytes(all_buckets.get(), kNumBuckets * num_workers * sizeof(uint32_t));

  pool.Run(0, 256, [&](uint64_t task, size_t worker) {
    uint32_t* HWY_RESTRICT buckets = all_buckets.get() + worker * kNumBuckets;

    ScalableTag<uint32_t> du32;
    using VU32 = Vec<decltype(du32)>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
    HWY_ALIGN uint32_t out[2 * MaxLanes(du32)];

    const uint64_t in_begin = task << 24;
    const uint64_t in_end = (task + 1) << 24;
    for (uint64_t in_pos = in_begin; in_pos < in_end; in_pos += 2 * N) {
      VU32 v0 = Iota(du32, static_cast<uint32_t>(in_pos + 0 * N));
      VU32 v1 = Iota(du32, static_cast<uint32_t>(in_pos + 1 * N));
      hash.TwoVec(du32, v0, v1);
      Store(v0, du32, out);
      Store(v1, du32, out + N);
      for (size_t i = 0; i < 2 * N; ++i) {
        const uint32_t h = out[i];
        buckets[h >> 16]++;
      }
    }
  });

  // Reduce to first set of buckets.
  for (size_t worker = 1; worker < num_workers; ++worker) {
    uint32_t* HWY_RESTRICT buckets = all_buckets.get() + worker * kNumBuckets;
    for (size_t i = 0; i < kNumBuckets; ++i) {
      all_buckets[i] += buckets[i];
    }
  }

  // Sanity check: total = 2^32.
  uint64_t sum = 0;
  Stats s_bucket;
  for (size_t i = 0; i < kNumBuckets; ++i) {
    sum += all_buckets[i];
    s_bucket.Notify(static_cast<float>(all_buckets[i]));
  }
  HWY_ASSERT(sum == (uint64_t{1} << 32));

  if (s_bucket.Min() == kNumBuckets && s_bucket.Max() == kNumBuckets) {
    return;  // Bijection
  } else {
    const double mean = s_bucket.Mean();
    const double sd = HWY_MAX(s_bucket.StandardDeviation(), 1E-6);
    const double zmin = (s_bucket.Min() - mean) / sd;
    const double zmax = (s_bucket.Max() - mean) / sd;
    printf("Bucket: %s\n", s_bucket.ToString().c_str());
    AssertNear("bucket  min", 64000, s_bucket.Min(), 0.01);
    AssertNear("bucket  max", 67000, s_bucket.Max(), 0.01);
    AssertNear("bucket mean", 65536, mean, 1E-4);
    // This can be high relative to the expected 256 for a binomial
    // distribution, which is likely explained by correlations from carry
    // chains.
    AssertNear("bucket stdv", 320, sd, 0.25);
    AssertNear("bucket skew", 0.01, s_bucket.Skewness(), 1.0);  // balanced
    AssertNear("bucket kurt", 3.0, s_bucket.Kurtosis(), 0.01);  // no fat tails
    AssertNear("bucket Zmin", -4.3, zmin, 0.05);
    AssertNear("bucket Zmax", 4.1, zmax, 0.05);
  }
}

// Verifies that a masked hash is a bijection on [0, 2^kBits). For kBits <= 24,
// uses a bitset to exhaustively verify all outputs are distinct. For larger
// kBits, uses the bucket histogram approach (same as TestBuckets) restricted to
// valid inputs.
template <size_t kBits, class Hash>
static HWY_NOINLINE void TestMaskedBuckets(const Hash& hash) {
  if constexpr (HWY_IS_DEBUG_BUILD) return;  // too slow

  static_assert(kBits < 32, "Use TestBuckets for full 32-bit hashes");
  using T = typename Hash::LaneType;
  constexpr uint64_t kNumInputs = uint64_t{1} << kBits;
  constexpr T kMask = static_cast<T>(kNumInputs - 1);

  if constexpr (kBits <= 24) {
    // Exact bijection test via bitset. Fast for small domains.
    AlignedFreeUniquePtr<uint8_t[]> seen =
        AllocateAligned<uint8_t>((kNumInputs + 7) / 8);
    ZeroBytes(seen.get(), (kNumInputs + 7) / 8);

    for (uint64_t i = 0; i < kNumInputs; ++i) {
      const T h = hash(static_cast<T>(i));
      HWY_ASSERT_M(h <= kMask, "Output exceeds mask");
      const size_t byte_idx = static_cast<size_t>(h / 8);
      const uint8_t bit_mask = static_cast<uint8_t>(uint8_t{1} << (h % 8));
      if (seen[byte_idx] & bit_mask) {
        HWY_ABORT("%s<%zu>: collision at input %zu, output %zu", Hash::Name(),
                  kBits, static_cast<size_t>(i), static_cast<size_t>(h));
      }
      seen[byte_idx] |= bit_mask;
    }
    fprintf(stderr, "  %s<%zu>: bijection verified (%zu inputs)\n",
            Hash::Name(), kBits, static_cast<size_t>(kNumInputs));
  } else {
    // Bucket histogram approach for larger domains (e.g., kBits=31).
    // Use 2^16 buckets; each should get exactly kNumInputs / 2^16 entries.
    constexpr size_t kNumBuckets = 0x10000;
    constexpr uint32_t kExpectedPerBucket =
        static_cast<uint32_t>(kNumInputs / kNumBuckets);
    // Bucket shift: hash >> kBucketShift maps to bucket index.
    constexpr size_t kBucketShift = kBits - 16;

    ThreadPool pool = MakePool();
    const size_t num_workers = pool.NumWorkers();
    AlignedFreeUniquePtr<uint32_t[]> all_buckets =
        AllocateAligned<uint32_t>(kNumBuckets * num_workers);
    ZeroBytes(all_buckets.get(), kNumBuckets * num_workers * sizeof(uint32_t));

    // Divide the input range into tasks.
    constexpr uint64_t kTaskBits = HWY_MIN(kBits, size_t{8});
    constexpr uint64_t kNumTasks = uint64_t{1} << kTaskBits;
    constexpr uint64_t kPerTask = kNumInputs / kNumTasks;

    pool.Run(0, kNumTasks, [&](uint64_t task, size_t worker) {
      uint32_t* HWY_RESTRICT buckets = all_buckets.get() + worker * kNumBuckets;
      const uint64_t in_begin = task * kPerTask;
      const uint64_t in_end = in_begin + kPerTask;
      for (uint64_t i = in_begin; i < in_end; ++i) {
        const T h = hash(static_cast<T>(i));
        buckets[h >> kBucketShift]++;
      }
    });

    // Reduce to first set of buckets.
    for (size_t worker = 1; worker < num_workers; ++worker) {
      uint32_t* HWY_RESTRICT buckets = all_buckets.get() + worker * kNumBuckets;
      for (size_t i = 0; i < kNumBuckets; ++i) {
        all_buckets[i] += buckets[i];
      }
    }

    // Verify: sum must equal kNumInputs, and for bijection, each bucket must
    // have exactly kExpectedPerBucket entries.
    uint64_t sum = 0;
    bool is_bijection = true;
    for (size_t i = 0; i < kNumBuckets; ++i) {
      sum += all_buckets[i];
      if (all_buckets[i] != kExpectedPerBucket) is_bijection = false;
    }
    HWY_ASSERT(sum == kNumInputs);
    if (is_bijection) {
      fprintf(stderr, "  %s<%zu>: bijection verified (%zu inputs)\n",
              Hash::Name(), kBits, static_cast<size_t>(kNumInputs));
    } else {
      HWY_ABORT("%s<%zu>: NOT a bijection!", Hash::Name(), kBits);
    }
  }
}

static HWY_NOINLINE void TestAllBuckets() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s\n", hash.Name());
    TestBuckets(hash);
  });

  // Also verify masked hashes are bijections on their domain.
  {
    MaskedWeakTwoMul<15> masked15(engine, 0);
    fprintf(stderr, "MaskedWeakTwoMul<15>\n");
    TestMaskedBuckets<15>(masked15);
  }
  {
    MaskedWeakTwoMul<31> masked31(engine, 0);
    fprintf(stderr, "MaskedWeakTwoMul<31>\n");
    TestMaskedBuckets<31>(masked31);
  }
  {
    MaskedMoremur<14> masked14(engine, 0);
    fprintf(stderr, "MaskedMoremur<14>\n");
    TestMaskedBuckets<14>(masked14);
  }
  {
    MaskedMoremur<27> masked27(engine, 0);
    fprintf(stderr, "MaskedMoremur<27>\n");
    TestMaskedBuckets<27>(masked27);
  }
}

// Verify bijection: hash all inputs, check each output was not yet seen using
// a bit array. Takes 12 seconds on Milan.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestBijection(const Hash& hash) {
  // Verified for WeakTwoMul, Triple32, Murmur3, and Speck32. TestBuckets also
  // verifies bijection, but faster, hence disable.
  if (Unpredictable1()) HWY_GTEST_SKIP();

  // Each worker hashes all inputs, but only stores outputs for its task's
  // range of outputs to reduce the working set size to L3. One global bitset
  // avoids per-worker allocations and does not experience false sharing because
  // ranges are large.
  constexpr uint64_t kNumU32 = uint64_t{1} << 32;
  AlignedUniquePtr<BitSet64[]> bits =
      MakeUniqueAlignedArray<BitSet64>(kNumU32 / 64);
  ZeroBytes(bits.get(), kNumU32 / 64 * sizeof(BitSet64));

  // Here, all workers must test the same permutation!

  ThreadPool pool = MakePool(HWY_MIN(pool::kMaxThreads, 255));
  pool.Run(0, 256, [&](uint64_t task, size_t /*worker*/) {
    const ScalableTag<uint32_t> du32;
    using VU32 = Vec<decltype(du32)>;

    HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
    HWY_ALIGN uint32_t out[4 * MaxLanes(du32)];
    const VU32 out_first = Set(du32, static_cast<uint32_t>(task << 24));
    const VU32 out_last =
        Set(du32, static_cast<uint32_t>(((task + 1) << 24) - 1));
    HWY_ASSERT(GetLane(out_last) - GetLane(out_first) == kNumU32 / 256 - 1);

    for (uint64_t in_pos = 0; in_pos < kNumU32; in_pos += 4 * N) {
      VU32 v0 = Iota(du32, static_cast<uint32_t>(in_pos + 0 * N));
      VU32 v1 = Iota(du32, static_cast<uint32_t>(in_pos + 1 * N));
      VU32 v2 = Iota(du32, static_cast<uint32_t>(in_pos + 2 * N));
      VU32 v3 = Iota(du32, static_cast<uint32_t>(in_pos + 3 * N));
      hash.TwoVec(du32, v0, v1);
      hash.TwoVec(du32, v2, v3);
      // Only keep if in range (2x speedup vs. scalar branching)
      const auto keep0 = And(Ge(v0, out_first), Le(v0, out_last));
      const auto keep1 = And(Ge(v1, out_first), Le(v1, out_last));
      const auto keep2 = And(Ge(v2, out_first), Le(v2, out_last));
      const auto keep3 = And(Ge(v3, out_first), Le(v3, out_last));
      size_t kept = CompressStore(v0, keep0, du32, out);
      kept += CompressStore(v1, keep1, du32, out + kept);
      kept += CompressStore(v2, keep2, du32, out + kept);
      kept += CompressStore(v3, keep3, du32, out + kept);
      HWY_DASSERT(kept <= 4 * N);
      for (size_t i = 0; i < kept; ++i) {
        const uint32_t h = out[i];
        HWY_ASSERT(!bits[h / 64].Get(h % 64));
        bits[h / 64].Set(h % 64);
      }
    }
  });
}

static HWY_NOINLINE void TestAllBijection() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s\n", hash.Name());
    TestBijection(hash);
  });
}

// Ensures each lane (per-vector) computes the same permutation.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestLanesEqual(const Hash& hash) {
  const ScalableTag<uint32_t> du32;
  using VU32 = Vec<decltype(du32)>;
  HWY_LANES_CONSTEXPR size_t N = Lanes(du32);
  AesCtrEngine engine(/*deterministic=*/true);

  ThreadPool pool = MakePool();
  pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t /*worker*/) {
    RngStream rng(engine, task);

    HWY_ALIGN uint32_t out[2 * MaxLanes(du32)];

    constexpr size_t kNumTrials = AdjustedReps(50'000);
    for (size_t trial = 0; trial < kNumTrials; ++trial) {
      VU32 vout0 = Set(du32, static_cast<uint32_t>(rng()));
      VU32 vout1 = vout0;
      hash.TwoVec(du32, vout0, vout1);
      Store(vout0, du32, out + 0);
      Store(vout1, du32, out + N);
      for (size_t i = 1; i < N; ++i) {
        if (out[0] != out[i]) {
          HWY_ABORT("Lane %zu mismatch: %08X != %08X.", i, out[0], out[i]);
        }
        if (out[0] != out[N + i]) {
          HWY_ABORT("Lane %zu mismatch2: %08X != %08X.", i, out[0], out[N + i]);
        }
      }
    }
  });
}

static HWY_NOINLINE void TestAllLanesEqual() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s\n", hash.Name());
    TestLanesEqual(hash);
  });
}

// Edge case tests for permutation variants.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestEdgeCases(const Hash& hash) {
  // Perm(0) != 0 (non-trivial).
  HWY_ASSERT(hash(0) != 0);

  // hash(0) != hash(1) (distinct).
  HWY_ASSERT(hash(0) != hash(1));

  // hash(0xFFFFFFFF) should be non-trivial.
  HWY_ASSERT(hash(0xFFFFFFFF) != 0);
  HWY_ASSERT(hash(0xFFFFFFFF) != 0xFFFFFFFF);

  // Consecutive inputs produce different outputs.
  for (uint32_t i = 0; i < 1000; ++i) {
    HWY_ASSERT(hash(i) != hash(i + 1));
  }

  // Different seeds produce different permutations.
  AesCtrEngine engine(/*deterministic=*/true);
  Hash hash2(engine, 1);
  HWY_ASSERT(hash(42u) != hash2(42u));
}

static HWY_NOINLINE void TestAllEdgeCases() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s\n", hash.Name());
    TestEdgeCases(hash);
  });
}

// ---------- 64-bit hash tests ----------

// Minimal edge-case tests for 64-bit hash permutations.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestEdgeCases64(const Hash& hash) {
  // hash(0) != 0 (non-trivial).
  HWY_ASSERT(hash(uint64_t{0}) != 0);

  // hash(0) != hash(1) (distinct).
  HWY_ASSERT(hash(uint64_t{0}) != hash(uint64_t{1}));

  // hash(max) should be non-trivial.
  const uint64_t all_ones = ~uint64_t{0};
  HWY_ASSERT(hash(all_ones) != 0);
  HWY_ASSERT(hash(all_ones) != all_ones);

  // Consecutive inputs produce different outputs.
  for (uint64_t i = 0; i < 1000; ++i) {
    HWY_ASSERT(hash(i) != hash(i + 1));
  }

  // Large inputs also produce distinct outputs.
  const uint64_t large = uint64_t{1} << 40;
  for (uint64_t i = large; i < large + 1000; ++i) {
    HWY_ASSERT(hash(i) != hash(i + 1));
  }

  // Different seeds produce different permutations.
  AesCtrEngine engine(/*deterministic=*/true);
  Hash hash2(engine, 1);
  HWY_ASSERT(hash(uint64_t{42}) != hash2(uint64_t{42}));
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void TestAllEdgeCases64() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash64(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s (64-bit)\n", hash.Name());
    TestEdgeCases64(hash);
  });
}

// Probabilistic bijection test for 64-bit hashes. Since we cannot enumerate
// all 2^64 inputs, we hash many random inputs and check for collisions.
template <class Hash>
static HWY_NOINLINE HWY_MAYBE_UNUSED void TestBijection64(const Hash& hash) {
  AesCtrEngine engine(/*deterministic=*/true);

  ThreadPool pool = MakePool();

  // Each worker hashes random inputs and stores outputs in a sorted array,
  // then we check for duplicates. Any collision very likely indicates a bug.
  constexpr size_t kNumTrials = AdjustedReps(500'000);

  pool.Run(0, pool.NumWorkers(), [&](uint64_t task, size_t /*worker*/) {
    RngStream rng(engine, task);

    const ScalableTag<uint64_t> du64;
    using VU64 = Vec<decltype(du64)>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(du64);
    HWY_ALIGN uint64_t out[2 * MaxLanes(du64)];

    AlignedVector<uint64_t> outputs;
    outputs.reserve(kNumTrials);

    for (size_t trial = 0; trial < kNumTrials; trial += 2 * N) {
      // Generate inputs that are unique with overwhelming probability.
      // Within each vector, inputs = rng_base ^ Iota, which are unique
      // because XOR with a constant is a bijection. Across vectors, the rng
      // bases differ (AES-CTR outputs are pseudo-random), so collisions
      // require a 64-bit coincidence: P(any collision) ~ kNumTrials^2/2^65,
      // which is negligible.
      VU64 v0 = Xor(Set(du64, rng()), Iota(du64, trial));
      VU64 v1 = Xor(Set(du64, rng()), Iota(du64, trial + N));
      hash.TwoVec(du64, v0, v1);
      Store(v0, du64, out);
      Store(v1, du64, out + N);
      const size_t count = HWY_MIN(2 * N, kNumTrials - trial);
      for (size_t i = 0; i < count; ++i) {
        outputs.push_back(out[i]);
      }
    }

    // Sort and check for duplicates.
    VQSort(outputs.data(), outputs.size(), SortAscending());
    for (size_t i = 1; i < outputs.size(); ++i) {
      if (outputs[i] == outputs[i - 1]) {
        HWY_ABORT("Collision in 64-bit hash: stream %zu, value %016llx",
                  static_cast<size_t>(task),
                  static_cast<unsigned long long>(outputs[i]));
      }
    }
  });
}

static HWY_NOINLINE void TestAllBijection64() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash64(engine, 0, [](const auto& hash) {
    fprintf(stderr, "%s (64-bit)\n", hash.Name());
    TestBijection64(hash);
  });
}

#else   // HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128
static void TestAllAvalanche() {}
static void TestAllBias() {}
static void TestAllBuckets() {}
static void TestAllBijection() {}
static void TestAllLanesEqual() {}
static void TestAllEdgeCases() {}
static void TestAllEdgeCases64() {}
static void TestAllBijection64() {}
static void TestAllMasked() {}
#endif  // HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_EMU128

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(HashTest);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllAvalanche);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllBias);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllBuckets);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllBijection);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllLanesEqual);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllEdgeCases);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllEdgeCases64);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllBijection64);
HWY_EXPORT_AND_TEST_BEST_P(HashTest, TestAllMasked);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
