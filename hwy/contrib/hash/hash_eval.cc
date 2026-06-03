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

// For testing quality of hash functions. Parts are derived from smhasher.
// This code is parallelized and mostly vectorized.

#include <stdint.h>
#include <stdio.h>

#include <algorithm>  // std::unique
#include <cmath>
#include <utility>  // std::move
#include <vector>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/thread_pool/index_range.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/hash_eval.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if HWY_TARGET != HWY_SCALAR

using KeyType = uint32_t;
using HashType = uint32_t;
HWY_INLINE_VAR constexpr size_t kKeyBits = sizeof(KeyType) * 8;
HWY_INLINE_VAR constexpr size_t kHashBits = sizeof(HashType) * 8;

//----------------------------------------------------------------------------
// Parallelization, copied from gemma.cpp threading.h.

// Per-worker storage to avoid repeated allocations.
struct EvalCtx {
  // Use all cores of one socket.
  static size_t Threads(const Topology& topology) {
    if (topology.packages.empty()) return ThreadPool::MaxThreads();
    // Minus one because these are in addition to the main thread.
    return topology.packages[0].cores.size() - 1;
  }

  EvalCtx() : pool(Threads(topology)), engine(/*deterministic=*/true) {}

  IndexRangePartition StaticPartition(size_t num_items) {
    size_t size = hwy::DivCeil(num_items, pool.NumWorkers());
    // ComputeHashes requires two vector multiples, except for the last task
    // which is special-cased.
    size = RoundUpTo(size, 2 * Lanes(ScalableTag<HashType>()));
    size = HWY_MIN(size, num_items);
    return IndexRangePartition(IndexRange(0, num_items), size);
  }

  template <typename T>
  static void Reduce(AlignedVector<T>& all) {
    for (size_t i = 1; i < all.size(); ++i) {
      all[0].Assimilate(all[i]);
    }
  }

  Topology topology;
  ThreadPool pool;
  AesCtrEngine engine;
  AlignedVector<uint32_t> dist_bins;
};

// Parallelized and vectorized hash computation, using a callback to generate
// the keys (two aligned vectors at a time): either loading them, or generating
// them via Iota or RNG.
template <class DoHash>
void ComputeHashes(uint64_t seed, AlignedVector<HashType>& hashes, EvalCtx& ctx,
                   const DoHash& do_hash) {
  const IndexRangePartition tasks = ctx.StaticPartition(hashes.size());

  AesCtrEngine engine(/*deterministic=*/true);

  ScalableTag<HashType> dh;
  using VH = Vec<decltype(dh)>;
  HWY_LANES_CONSTEXPR size_t NH = Lanes(dh);

  ctx.pool.Run(
      0, tasks.NumTasks(), [&](const uint64_t task_idx, size_t /*worker*/) {
        RngStream rng(engine, seed + task_idx * 17);

        const IndexRange range = tasks.Range(task_idx);
        size_t i = range.begin();
        if (HWY_LIKELY(range.Num() >= 2 * NH)) {
          for (; i <= range.end() - 2 * NH; i += 2 * NH) {
            VH h0, h1;
            do_hash(dh, NH, rng, i, h0, h1);

            StoreU(h0, dh, &hashes[i]);
            StoreU(h1, dh, &hashes[i + NH]);
          }
        }

        // Same with bounds-checked stores.
        const size_t remaining = range.end() - i;
        if (remaining > 0) {
          VH h0, h1;
          do_hash(dh, NH, rng, i, h0, h1);

          StoreN(h0, dh, &hashes[i], remaining);
          StoreN(h1, dh, &hashes[i + NH], remaining >= NH ? remaining - NH : 0);
        }
      });
}

//----------------------------------------------------------------------------

struct Result {
  static double Start(const char* label) {
    const double t0 = platform::Now();
    printf("%12s: ", label);
    return t0;
  }

  void Update(double score, int start, int width) {
    if (score > worst_score) {
      worst_score = score;
      worst_start = start;
      worst_width = width;
    }
  }

  void Update(const Result& other) {
    collisions = HWY_MAX(collisions, other.collisions);
    Update(other.worst_score, other.worst_start, other.worst_width);
  }

  void Print(double t0) const {
    printf("%7.2f ms ", (platform::Now() - t0) * 1e3);  // elapsed time
    if (static_cast<double>(collisions) > collisions_tolerated) {
      printf("collisions: %zu (> %.1f)!!!! ", collisions, collisions_tolerated);
    }
    printf("worst bias: %2d-bit window at bit %2d - %4.2f%%", worst_width,
           worst_start, worst_score * 100.0);
    if (worst_score > 0.01) {
      printf("!!!!");
    }
    printf("\n");
  }

  size_t collisions = 0;
  double collisions_tolerated = 0.0;

  // From TestDistribution.
  double worst_score = 0.0;
  int worst_start = -1;
  int worst_width = -1;
};

//----------------------------------------------------------------------------
// Avalanche, heavily modified from an old fork of smhasher (MIT license).

class BiasBins {
 public:
  BiasBins(const size_t num_bins) : bins_(num_bins) {}

  uint32_t* Get() { return bins_.data(); }

  void Assimilate(const BiasBins& victim) {
    HWY_ASSERT(bins_.size() == victim.bins_.size());
    for (size_t i = 0; i < bins_.size(); ++i) {
      bins_[i] += victim.bins_[i];
    }
  }

 private:
  AlignedVector<uint32_t> bins_;
};

inline void FlipBit(uint32_t& val, uint32_t bit) {
  bit &= 31;
  val ^= (uint32_t{1} << bit);
}

template <typename HashFunction>
void AccumulateBias(const HashFunction& hash, RngStream& rng, BiasBins& bins) {
  KeyType K = static_cast<KeyType>(rng());
  const HashType ha = hash(K);

  uint32_t* HWY_RESTRICT pos = bins.Get();

  for (size_t k_idx = 0; k_idx < kKeyBits; k_idx++) {
    FlipBit(K, k_idx);
    const HashType hb = hash(K);
    FlipBit(K, k_idx);

    for (size_t h_idx = 0; h_idx < kHashBits; h_idx++) {
      const uint32_t bit_a = (ha >> h_idx) & 1;
      const uint32_t bit_b = (hb >> h_idx) & 1;
      (*pos++) += (bit_a ^ bit_b);
    }
  }
}

template <typename HashFunction>
void TestAvalanche(const HashFunction& hash, EvalCtx& ctx) {
  AesCtrEngine engine(/*deterministic=*/true);
  const double t0 = Result::Start("Avalanche");

  const size_t reps = 1 * 1000 * 1000;
  const IndexRangePartition tasks = ctx.StaticPartition(reps);

  const size_t num_bins = kKeyBits * kHashBits;
  AlignedVector<BiasBins> all_bins;
  all_bins.reserve(ctx.pool.NumWorkers());
  for (size_t i = 0; i < ctx.pool.NumWorkers(); ++i) {
    all_bins.emplace_back(BiasBins(num_bins));
  }

  {
    PROFILER_ZONE("Avalanche.CalcBias");
    ctx.pool.Run(0, tasks.NumTasks(), [&](uint64_t task_idx, size_t worker) {
      BiasBins& my_bins = all_bins[worker];
      RngStream rng(engine, 48273 + 257 * task_idx);
      for (size_t rep : tasks.Range(task_idx)) {
        (void)rep;
        AccumulateBias(hash, rng, my_bins);
      }
    });
  }

  EvalCtx::Reduce(all_bins);

  Result result;
  const double rcp_reps = 1.0 / static_cast<double>(reps);
  for (size_t i = 0; i < num_bins; i++) {
    const double c = all_bins[0].Get()[i] * rcp_reps;
    const double d = fabs(c * 2 - 1);
    result.worst_score = HWY_MAX(result.worst_score, d);
  }

  result.Print(t0);
}

//----------------------------------------------------------------------------
// Check all possible kKeyBits-choose-N differentials for collisions, report
// ones that occur significantly more often than expected.

// Random collisions can happen with probability 1 in 2^32 - if we do more than
// 2^32 tests, we'll probably see some spurious random collisions, so don't
// report them.

// Thread-specific diff lists so that the thread pool threads write into
// their own memory and combine the results when all threads have finished.
struct Diffs {
  void Assimilate(const Diffs& victim) {
    diffs.insert(diffs.end(), victim.diffs.begin(), victim.diffs.end());
  }

  AlignedVector<KeyType> diffs;
};

template <typename HashFunction>
void DiffTestRecurse(const HashFunction& hash, KeyType& k1, KeyType& k2,
                     HashType& h1, HashType& h2, int start, int bitsleft,
                     AlignedVector<KeyType>* diffs) {
  for (int i = start; i < static_cast<int>(kKeyBits); i++) {
    FlipBit(k2, i);
    bitsleft--;

    h2 = hash(k2);

    if (h1 == h2) {
      diffs->push_back(k1 ^ k2);
    }

    if (bitsleft) {
      DiffTestRecurse(hash, k1, k2, h1, h2, i + 1, bitsleft, diffs);
    }

    FlipBit(k2, i);
    bitsleft++;
  }
}

// Sorts the input.
HWY_MAYBE_UNUSED size_t MaxRunLength(AlignedVector<KeyType>& diffs) {
  PROFILER_FUNC;
  if (diffs.empty()) return 1;

  hwy::VQSort(diffs.data(), diffs.size(), hwy::SortAscending());

  size_t max_run_length = 0;

  KeyType d = diffs[0];  // value of the current run in sorted differentials
  size_t run_length = 1;

  for (size_t i = 1; i < diffs.size(); i++) {
    if (diffs[i] == d) {
      run_length++;
      continue;
    }

    // Start a new run but remember the maximum run length.
    max_run_length = HWY_MAX(max_run_length, run_length);
    d = diffs[i];
    run_length = 1;
  }

  max_run_length = HWY_MAX(max_run_length, run_length);
  return max_run_length;
}

template <typename HashFunction>
void TestDiff(const HashFunction& hash, EvalCtx& ctx) {
  PROFILER_FUNC;
  AesCtrEngine engine(/*deterministic=*/true);
  const double t0 = Result::Start("Diff");

  AlignedVector<Diffs> all_diffs;
  all_diffs.resize(ctx.pool.NumWorkers());

  const int diffbits = 5;
  const size_t reps = 1000;
  const IndexRangePartition tasks = ctx.StaticPartition(reps);
  ctx.pool.Run(
      0, tasks.NumTasks(), [&](const uint64_t task_idx, size_t worker) {
        AlignedVector<KeyType>& my_diffs = all_diffs[worker].diffs;
        RngStream rng(engine, 100 + 257 * task_idx);
        for (size_t rep : tasks.Range(task_idx)) {
          (void)rep;
          KeyType k1 = static_cast<KeyType>(rng());
          KeyType k2 = k1;

          HashType h1 = hash(k1);
          HashType h2;
          DiffTestRecurse(hash, k1, k2, h1, h2, 0, diffbits, &my_diffs);
        }
      });

  EvalCtx::Reduce(all_diffs);
  Result result;
  result.collisions = MaxRunLength(all_diffs[0].diffs) - 1;
  result.Print(t0);
}

//----------------------------------------------------------------------------
// Measure the distribution "score" for each possible N-bit span up to 20 bits

// Basically, we're computing a constant that says "The test distribution is as
// uniform, RMS-wise, as a random distribution restricted to (1-X)*100 percent
// of the bins. This makes for a nice uniform way to rate a distribution that
// isn't dependent on the number of bins or the number of keys

// (as long as # keys > # bins * 3 or so, otherwise random fluctuations show up
// as distribution weaknesses)

inline double calcScore(const uint32_t* bins, const size_t bincount,
                        const int keycount) {
  PROFILER_FUNC;

  const double n = static_cast<double>(bincount);
  const double k = static_cast<double>(keycount);

  // compute rms value
  double r = 0.0;
  for (size_t i = 0; i < bincount; i++) {
    double b = bins[i];
    r += b * b;
  }
  r = sqrt(r / n);

  // compute fill factor
  const double denom = n * r * r - k;
  double f = denom == 0.0 ? 0.0 : (k * k - 1) / denom;

  // rescale to (0,1) with 0 = good, 1 = bad
  return 1 - (f / n);
}

// Bulk of CPU time is spent in this function.
Result TestDistribution(const AlignedVector<HashType>& hashes, EvalCtx& ctx) {
  PROFILER_FUNC;

  // We need at least 5 keys per bin to reliably test distribution biases
  // down to 1%, so don't bother to test sparser distributions than that
  int max_width = 20;
  const double num_hashes = static_cast<double>(hashes.size());
  while (num_hashes / static_cast<double>(1 << max_width) < 5.0) {
    max_width--;
  }

  const size_t max_bins = size_t{1} << max_width;
  AlignedVector<uint32_t>& bins = ctx.dist_bins;
  if (max_bins > bins.size()) {
    bins.resize(max_bins);
  }

  Result ret;

  for (int start = 0; start < static_cast<int>(kHashBits); start++) {
    size_t width = max_width;
    size_t num_bins = max_bins;

    // Actually faster serial (8.5s) than parallel (19s). We further accelerate
    // to 5s via SIMD of the rotate and AND.
    {
      PROFILER_ZONE("Dist.Hash");
      ZeroBytes(&bins[0], max_bins * sizeof(bins[0]));

      const ScalableTag<HashType> dh;
      HWY_LANES_CONSTEXPR size_t NH = Lanes(dh);
      using VH = Vec<decltype(dh)>;
      HWY_ALIGN uint32_t indices[MaxLanes(dh)];

      size_t i = 0;
      if (HWY_LIKELY(hashes.size() >= NH)) {
        for (; i <= hashes.size() - NH; i += NH) {
          const VH hash = LoadU(dh, &hashes[i]);
          const VH rot = RotateRightSame(hash, start);
          const VH idx = And(rot, Set(dh, num_bins - 1));
          Store(idx, dh, indices);
          for (size_t j = 0; j < NH; ++j) {
            bins[indices[j]]++;
          }
        }
      }

      if (HWY_UNLIKELY(i != hashes.size())) {
        const size_t remaining = hashes.size() - i;
        const VH hash = LoadN(dh, &hashes[i], remaining);
        const VH rot = RotateRightSame(hash, start);
        const VH idx = And(rot, Set(dh, num_bins - 1));
        Store(idx, dh, indices);
        for (size_t j = 0; j < remaining; ++j) {
          bins[indices[j]]++;
        }
      }
    }

    // Test the distribution, then fold the bins in half,
    // repeat until we're down to 256 bins

    while (num_bins >= 256) {
      const double score = calcScore(bins.data(), num_bins, hashes.size());
      ret.Update(score, start, width);

      width--;
      num_bins /= 2;
      if (width < 8) break;

      for (size_t i = 0; i < num_bins; i++) {
        bins[i] += bins[i + num_bins];
      }
    }
  }

  return ret;
}

size_t CountCollisions(AlignedVector<HashType>& hashes) {
  PROFILER_FUNC;
  hwy::VQSort(hashes.data(), hashes.size(), hwy::SortAscending());
  auto end = std::unique(hashes.begin(), hashes.end());
  return hashes.end() - end;
}

// Non-const `hashes` because `FindCollisions` sorts it.
HWY_MAYBE_UNUSED Result AnalyzeHashes(AlignedVector<HashType>& hashes,
                                      EvalCtx& ctx) {
  // Before FindCollisions sorts hashes.
  Result result = TestDistribution(hashes, ctx);
  result.collisions = CountCollisions(hashes);
  return result;
}

//----------------------------------------------------------------------------

template <class HashFunction>
void TestNotCounter(const HashFunction& hash, EvalCtx& ctx) {
  PROFILER_FUNC;
  const double t0 = Result::Start("NotIota");

  AlignedVector<HashType> hashes(size_t{1} << 18);

  ComputeHashes(0, hashes, ctx,
                [&](auto dh, size_t NH, RngStream& /*rng*/, size_t i, auto& h0,
                    auto& h1) HWY_ATTR {
                  // Hash Not(Iota).
                  using VH = Vec<decltype(dh)>;
                  VH k0 = Not(Iota(dh, i));
                  VH k1 = Not(Iota(dh, i + NH));
                  hash.TwoVec(dh, k0, k1);
                  h0 = k0;
                  h1 = k1;
                });

  const Result result = AnalyzeHashes(hashes, ctx);
  result.Print(t0);
}

template <class HashFunction>
void TestRevCounter(const HashFunction& hash, EvalCtx& ctx) {
  PROFILER_FUNC;
  const double t0 = Result::Start("RevIota");

  AlignedVector<HashType> hashes(size_t{1} << 18);

  ComputeHashes(0, hashes, ctx,
                [&](auto dh, size_t NH, RngStream& /*rng*/, size_t i, auto& h0,
                    auto& h1) HWY_ATTR {
                  // Hash ReverseBits(Iota).
                  using VH = Vec<decltype(dh)>;
                  VH k0 = ReverseBits(Iota(dh, i));
                  VH k1 = ReverseBits(Iota(dh, i + NH));
                  hash.TwoVec(dh, k0, k1);
                  h0 = k0;
                  h1 = k1;
                });

  const Result result = AnalyzeHashes(hashes, ctx);
  result.Print(t0);
}

template <class HashFunction>
void TestMulCounter(const HashFunction& hash, EvalCtx& ctx) {
  PROFILER_FUNC;
  const double t0 = Result::Start("MulIota");

  AlignedVector<HashType> hashes(size_t{1} << 15);

  ComputeHashes(0, hashes, ctx,
                [&](auto dh, size_t NH, RngStream& /*rng*/, size_t i, auto& h0,
                    auto& h1) HWY_ATTR {
                  // Hash Iota*mul.
                  using VH = Vec<decltype(dh)>;
                  const VH kMul = Set(dh, 0x10000);
                  VH k0 = Mul(Iota(dh, i), kMul);
                  VH k1 = Mul(Iota(dh, i + NH), kMul);
                  hash.TwoVec(dh, k0, k1);
                  h0 = k0;
                  h1 = k1;
                });

  const Result result = AnalyzeHashes(hashes, ctx);
  result.Print(t0);
}

template <class HashFunction>
void TestRotCounter(const HashFunction& hash, EvalCtx& ctx) {
  PROFILER_FUNC;
  const double t0 = Result::Start("RotIota");

  AlignedVector<HashType> hashes(size_t{1} << 20);

  Result result;
  for (size_t idx_bit = 0; idx_bit < kKeyBits; idx_bit++) {
    ComputeHashes(0, hashes, ctx,
                  [&](auto dh, size_t NH, RngStream& /*rng*/, size_t i,
                      auto& h0, auto& h1) HWY_ATTR {
                    // Hash a rotated counter.
                    using VH = Vec<decltype(dh)>;
                    VH k0 = RotateLeftSame(Iota(dh, i), idx_bit);
                    VH k1 = RotateLeftSame(Iota(dh, i + NH), idx_bit);
                    hash.TwoVec(dh, k0, k1);
                    h0 = k0;
                    h1 = k1;
                  });

    result.Update(AnalyzeHashes(hashes, ctx));
  }
  result.Print(t0);
}

template <typename HashFunction>
void TestDiffDist(const HashFunction& hash, EvalCtx& ctx) {
  const double t0 = Result::Start("DiffDist");
  AlignedVector<HashType> diffs(256 * 256 * 32);

  Result result;
  for (size_t keybit = 0; keybit < kKeyBits; ++keybit) {
    ComputeHashes(857374 + keybit * 257, diffs, ctx,
                  [&](auto dh, size_t NH, RngStream& rng, size_t /*pos*/,
                      auto& h0, auto& h1) HWY_ATTR {
                    // Random keys.
                    using VH = Vec<decltype(dh)>;
                    HWY_ALIGN KeyType keys[2 * MaxLanes(dh)];
                    for (size_t i = 0; i < 2 * Lanes(dh); ++i) {
                      keys[i] = static_cast<KeyType>(rng());
                    }
                    VH k0 = Load(dh, &keys[0]);
                    VH k1 = Load(dh, &keys[NH]);

                    // Flip the same bit in all keys.
                    const VH flipbit = Set(dh, KeyType{1} << keybit);
                    VH flip0 = Xor(k0, flipbit);
                    VH flip1 = Xor(k1, flipbit);

                    // Return XOR difference of the hashes.
                    hash.TwoVec(dh, k0, k1);
                    hash.TwoVec(dh, flip0, flip1);
                    h0 = Xor(k0, flip0);
                    h1 = Xor(k1, flip1);
                  });

    result.Update(AnalyzeHashes(diffs, ctx));
  }

  const double expected = (static_cast<double>(diffs.size()) *
                           static_cast<double>(diffs.size() - 1)) /
                          pow(2.0, static_cast<double>(kHashBits + 1));
  // We expect duplicate keys, flipped-pair symmetry (if both k and k^D are in
  // the inputs, then both produce equal differentials, and collisions, each
  // equal to the birthday bound (= 1.0), hence triple plus 10% margin.
  result.collisions_tolerated = 3.3 * expected;

  result.Print(t0);
}

//----------------------------------------------------------------------------
// Keyset generators.

// All keys with two non-zero bytes. Fast.
HWY_MAYBE_UNUSED AlignedVector<KeyType> TwoBytesKeygen() {
  const auto chooseK = [](int n, int k) -> double {
    if (k > (n - k)) k = n - k;

    double c = 1.0;
    for (int i = 0; i < k; i++) {
      double t = static_cast<double>(n - i) / static_cast<double>(i + 1);
      c *= t;
    }
    return c;
  };
  const int keycount = (int)chooseK(4, 2) * 255 * 255 + 4 * 255;

  AlignedVector<KeyType> keys;
  keys.reserve(keycount);

  uint8_t bytes[sizeof(KeyType)] = {};

  // Add all keys with one non-zero byte
  for (size_t byteA = 0; byteA < sizeof(KeyType); byteA++) {
    for (int valA = 1; valA <= 255; valA++) {
      bytes[byteA] = (uint8_t)valA;

      KeyType key;
      CopyBytes(bytes, &key, sizeof(key));
      keys.push_back(key);
    }

    bytes[byteA] = 0;
  }

  // Add all keys with two non-zero bytes
  for (size_t byteA = 0; byteA < sizeof(KeyType) - 1; byteA++) {
    for (size_t byteB = byteA + 1; byteB < sizeof(KeyType); byteB++) {
      for (int valA = 1; valA <= 255; valA++) {
        bytes[byteA] = static_cast<uint8_t>(valA);

        for (int valB = 1; valB <= 255; valB++) {
          bytes[byteB] = (uint8_t)valB;
          KeyType key;
          CopyBytes(bytes, &key, sizeof(key));
          keys.push_back(key);
        }

        bytes[byteB] = 0;
      }

      bytes[byteA] = 0;
    }
  }

  return keys;
}

void SparseKeygenR(int start, int bitsleft, KeyType& k,
                   AlignedVector<HashType>& keys) {
  for (int i = start; i < static_cast<int>(kKeyBits); i++) {
    FlipBit(k, i);

    keys.push_back(k);

    if (bitsleft > 1) {
      SparseKeygenR(i + 1, bitsleft - 1, k, keys);
    }

    FlipBit(k, i);
  }
}

HWY_MAYBE_UNUSED AlignedVector<HashType> SparseKeygen() {
  const int kNonzeroBits = 6;

  AlignedVector<HashType> keys;
  keys.reserve(2 * 1000 * 1000);
  KeyType k = 0;
  keys.push_back(k);
  SparseKeygenR(0, kNonzeroBits, k, keys);
  return keys;
}

// Keys with bytes ABAB (including AAAA)
HWY_MAYBE_UNUSED AlignedVector<HashType> CyclicKeygen() {
  AlignedVector<HashType> keys;
  keys.reserve(255 * 255);
  for (int valA = 1; valA <= 255; ++valA) {
    for (int valB = 1; valB <= 255; ++valB) {
      const KeyType cycle = valA * 256 + valB;
      keys.push_back(cycle * 0x00010001u);
    }
  }
  return keys;
}

// Convenience function to compute, analyze and report results.
template <class HashFunction>
HWY_NOINLINE void ReportResults(const char* key_type,
                                AlignedVector<HashType>&& keys,
                                const HashFunction& hash, EvalCtx& ctx) {
  const double t0 = Result::Start(key_type);

  AlignedVector<HashType> hashes(std::move(keys));
  // Padding, plus the task alignment in StaticPartition, ensures we can
  // unconditionally load two vectors in the lambda below.
  const size_t valid_keys = hashes.size();
  hashes.resize(RoundUpTo(valid_keys, 2 * Lanes(ScalableTag<KeyType>{})));

  ComputeHashes(0, hashes, ctx,
                [&](auto dh, size_t NH, RngStream& /*rng*/, size_t i, auto& h0,
                    auto& h1) HWY_ATTR {
                  // Load keys from the generator and hash.
                  h0 = LoadU(dh, &hashes[i]);
                  h1 = LoadU(dh, &hashes[i + NH]);
                  hash.TwoVec(dh, h0, h1);
                });
  hashes.resize(valid_keys);
  const Result result = AnalyzeHashes(hashes, ctx);
  result.Print(t0);
}

template <class HashFunction>
void TestTwoBytes(const HashFunction& hash, EvalCtx& ctx) {
  ReportResults("TwoBytes", TwoBytesKeygen(), hash, ctx);
}

template <class HashFunction>
void TestSparse(const HashFunction& hash, EvalCtx& ctx) {
  ReportResults("Sparse", SparseKeygen(), hash, ctx);
}

template <typename HashFunction>
void TestCyclic(const HashFunction& hash, EvalCtx& ctx) {
  ReportResults("Cyclic", CyclicKeygen(), hash, ctx);
}

// HashFunction provides Name(), HashType operator()(KeyType).
template <class HashFunction>
void RunTests(const HashFunction& hash) {
  EvalCtx ctx;

  printf("\n=============================== %s\n", HashFunction::Name());

  // Quick tests first.
  TestTwoBytes(hash, ctx);
  TestSparse(hash, ctx);
  TestCyclic(hash, ctx);

  TestAvalanche(hash, ctx);
  TestDiff(hash, ctx);

  TestNotCounter(hash, ctx);
  TestRevCounter(hash, ctx);
  TestMulCounter(hash, ctx);
  TestRotCounter(hash, ctx);
  TestDiffDist(hash, ctx);
}

HWY_NOINLINE void RunAll() {
  AesCtrEngine engine(/*deterministic=*/true);
  ForeachHash(engine, 0, [](const auto& hash) { RunTests(hash); });
  PROFILER_PRINT_RESULTS();
}

#else   // HWY_TARGET == HWY_SCALAR
void RunAll() {}
#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(HashEval);
HWY_EXPORT_AND_TEST_BEST_P(HashEval, RunAll);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
