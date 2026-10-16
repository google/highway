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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>  // std::sort
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/algo/shuffle_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/algo/shuffle-inl.h"
#include "hwy/contrib/hash/hash-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// Random bits from `gen` as ShuffleSpan draws them, 64 at a time. With a 32-bit
// generator, the first of two draws is the upper half.
template <class Gen>
class RefBits {
 public:
  explicit RefBits(Gen& gen) : gen_(gen) {}

  uint64_t Bits64() {
    uint64_t bits = gen_() - (Gen::min)();
    HWY_IF_CONSTEXPR(sizeof(typename Gen::result_type) < 8) {
      bits = (bits << 32) | (gen_() - (Gen::min)());
    }
    return bits;
  }

 private:
  Gen& gen_;
};

// The 64 random bits of a position, as ShuffleHash documents: the upper 32 are
// Triple32 of offset + pos * 0x9E3779B9 under the lower half of `key`, whose
// upper half is the offset, and the lower 32 are Triple32 of those under the
// offset.
uint64_t RefPositionBits(uint64_t key, uint32_t pos) {
  const uint32_t offset = static_cast<uint32_t>(key >> 32);
  const uint32_t upper =
      Triple32(static_cast<uint32_t>(key))(offset + pos * 0x9E3779B9u);
  const uint32_t lower = Triple32(offset)(upper);
  return (uint64_t{upper} << 32) | lower;
}

// Sequential Fisher-Yates using the rule ShuffleSpan documents: from the top
// down, a new 64-bit key for each block of 32 positions, and the target is the
// upper 64 bits of RefPositionBits times (i + 1). Positions below 32 take 32
// bits each: the upper, then the lower half of each draw.
template <class Bits>
std::vector<size_t> ReferencePermutation(size_t count, Bits& bits) {
  std::vector<size_t> perm(count);
  for (size_t k = 0; k < count; ++k) perm[k] = k;
  uint64_t key = 0, draw = 0;
  bool upper = true;
  for (size_t i = count; i-- > 1;) {
    const uint32_t i32 = static_cast<uint32_t>(i);
    uint64_t target;
    if (i < 32) {
      if (upper) draw = bits.Bits64();
      const uint64_t half = upper ? draw >> 32 : draw;
      target = MulHigh32(static_cast<uint32_t>(half), i32 + 1);
      upper = !upper;
    } else {
      if (i == count - 1 || i % 32 == 31) key = bits.Bits64();
      Mul128(RefPositionBits(key, i32), uint64_t{i32} + 1, &target);
    }
    std::swap(perm[i], perm[static_cast<size_t>(target)]);
  }
  return perm;
}

// The bucket rule: from the bottom up, a new 64-bit key for each block of 32
// positions, and each position goes in order to the bucket in the upper bits of
// RefPositionBits of it. Then each bucket is shuffled as above.
template <class Bits>
std::vector<size_t> ReferenceBuckets(size_t count, size_t log2_buckets,
                                     Bits& bits) {
  std::vector<std::vector<size_t>> buckets(size_t{1} << log2_buckets);
  uint64_t key = 0;
  for (size_t k = 0; k < count; ++k) {
    if (k % 32 == 0) key = bits.Bits64();
    const uint64_t pos_bits = RefPositionBits(key, static_cast<uint32_t>(k));
    buckets[static_cast<size_t>(pos_bits >> (64 - log2_buckets))].push_back(k);
  }
  std::vector<size_t> perm;
  for (const std::vector<size_t>& bucket : buckets) {
    for (size_t k : ReferencePermutation(bucket.size(), bits)) {
      perm.push_back(bucket[k]);
    }
  }
  return perm;
}

// Order-dependent fingerprint, to compare results without keeping copies.
template <typename T>
uint64_t Fingerprint(const T* p, size_t count) {
  uint64_t print = 0;
  for (size_t k = 0; k < count; ++k) {
    print = (print + static_cast<uint64_t>(p[k]) + 1) * 0x9E3779B97F4A7C15ull;
  }
  return print;
}

void AssertIsPermutation(const std::vector<size_t>& perm) {
  std::vector<bool> seen(perm.size());
  for (size_t p : perm) {
    HWY_ASSERT(p < perm.size() && !seen[p]);
    seen[p] = true;
  }
}

// Values are distinct for counts up to 128 for 8-bit lanes and 2048 otherwise.
template <typename T>
T ValueAt(size_t k) {
  return ConvertScalarTo<T>(k % (sizeof(T) == 1 ? 128 : 2048));
}

// Shuffles `count` values placed at `misalign` inside a guarded buffer, and
// checks the result is `perm` applied to them with nothing outside touched.
template <class D, class Shuffle>
void CheckShuffle(D d, size_t count, size_t misalign,
                  const std::vector<size_t>& perm, const Shuffle& shuffle) {
  using T = TFromD<D>;
  const T guard = ConvertScalarTo<T>(99);
  AlignedFreeUniquePtr<T[]> storage = AllocateAligned<T>(misalign + count + 1);
  HWY_ASSERT(storage);
  for (size_t k = 0; k < misalign + count + 1; ++k) storage[k] = guard;
  T* data = storage.get() + misalign;
  for (size_t k = 0; k < count; ++k) data[k] = ValueAt<T>(k);

  shuffle(d, data, count);

  for (size_t k = 0; k < misalign; ++k) {
    HWY_ASSERT_EQ(guard, storage[k]);
  }
  for (size_t k = 0; k < count; ++k) {
    HWY_ASSERT_EQ(ValueAt<T>(perm[k]), data[k]);
  }
  HWY_ASSERT_EQ(guard, data[count]);
}

template <class D>
std::vector<size_t> CountsFor(D d) {
  std::vector<size_t> counts;
  for (size_t count = 0; count < 3 * Lanes(d) + 5; ++count) {
    counts.push_back(count);
  }
  static constexpr size_t kBlockEdges[] = {31, 32, 33, 63, 64, 65, 97, 1000};
  for (size_t count : kBlockEdges) counts.push_back(count);
  return counts;
}

// 32 bits per draw, so 64 bits take two draws.
class Gen32 {
 public:
  using result_type = uint32_t;
  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return ~result_type{0}; }
  explicit Gen32(uint64_t seed) : gen_(seed) {}
  result_type operator()() { return static_cast<result_type>(gen_() >> 32); }

 private:
  CachedXoshiro<> gen_;
};

// ShuffleSpan must draw its keys in the reference's order, so identically
// seeded generators give the reference permutation on every target.
template <class D, class MakeGen>
void CheckGenerator(D d, const MakeGen& make_gen) {
  using T = TFromD<D>;
  using Gen = decltype(make_gen());
  const size_t misalignments[2] = {0, Lanes(d) / 3 + 1};
  for (size_t count : CountsFor(d)) {
    if (sizeof(T) == 1 && count > 128) continue;
    Gen ref_gen = make_gen();
    RefBits<Gen> bits(ref_gen);
    const std::vector<size_t> perm = ReferencePermutation(count, bits);
    AssertIsPermutation(perm);
    for (size_t misalign : misalignments) {
      Gen gen = make_gen();
      CheckShuffle(d, count, misalign, perm, [&gen](D tag, T* p, size_t n) {
        ShuffleSpan(tag, p, n, gen);
      });
    }
  }
}

struct TestGenerator {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    CheckGenerator(d, []() HWY_ATTR { return CachedXoshiro<>(123); });
    CheckGenerator(d, []() HWY_ATTR { return Gen32(456); });
#if HWY_TARGET != HWY_SCALAR  // RngStream is not supported there.
    const AesCtrEngine engine(/*deterministic=*/true);
    CheckGenerator(d, [&engine]() HWY_ATTR { return RngStream(engine, 789); });
#endif
  }
};

void TestAllGenerator() { ForIntegerTypes(ForPartialVectors<TestGenerator>()); }

// The bucket path against its reference, at sizes a test can afford. A guard
// after `buf` checks that it fits in ShuffleBucketBufNum.
template <class D, class MakeGen>
void CheckBuckets(D d, const MakeGen& make_gen) {
  using T = TFromD<D>;
  using Gen = decltype(make_gen());
  for (size_t count : {size_t{1}, size_t{33}, size_t{128}, size_t{1000}}) {
    if (sizeof(T) == 1 && count > 128) continue;
    for (size_t log2_buckets : {size_t{1}, size_t{3}, size_t{10}}) {
      Gen ref_gen = make_gen();
      RefBits<Gen> ref_bits(ref_gen);
      const std::vector<size_t> perm =
          ReferenceBuckets(count, log2_buckets, ref_bits);
      AssertIsPermutation(perm);
      const size_t buf_num = detail::ShuffleBucketBufNum<T>(count);
      AlignedFreeUniquePtr<T[]> buf = AllocateAligned<T>(buf_num + 1);
      HWY_ASSERT(buf);
      const T guard = ConvertScalarTo<T>(99);
      buf[buf_num] = guard;
      Gen gen = make_gen();
      CheckShuffle(d, count, /*misalign=*/1, perm, [&](D tag, T* p, size_t n) {
        detail::ShuffleDrawnBits<Gen> bits(gen);
        detail::ShuffleBuckets(tag, p, n, bits, log2_buckets, buf.get());
      });
      HWY_ASSERT_EQ(guard, buf[buf_num]);
    }
  }
}

struct TestBuckets {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    CheckBuckets(d, []() HWY_ATTR { return CachedXoshiro<>(123); });
    CheckBuckets(d, []() HWY_ATTR { return Gen32(456); });
  }
};

void TestAllBuckets() { ForIntegerTypes(ForPartialVectors<TestBuckets>()); }

// Every ordering of 4 elements, and every final position of the first and last
// of 96 elements (three blocks), should be about equally likely. Generator
// seeds are fixed, so this cannot flake; bounds are ~6 sigma.
template <class D, class Shuffle>
void CheckUniform(D d, const Shuffle& shuffle) {
  using T = TFromD<D>;
  std::vector<size_t> orderings(256);
  const size_t kOrderingTrials = 24 * 1000;
  CachedXoshiro<> ordering_gen(1);
  for (size_t trial = 0; trial < kOrderingTrials; ++trial) {
    T data[4] = {0, 1, 2, 3};
    shuffle(d, data, 4, ordering_gen);
    size_t code = 0;
    for (T v : data) code = code * 4 + static_cast<size_t>(v);
    ++orderings[code];
  }
  size_t num_seen = 0;
  for (size_t n : orderings) {
    if (n == 0) continue;
    ++num_seen;
    HWY_ASSERT(800 <= n && n <= 1200);
  }
  HWY_ASSERT_EQ(size_t{24}, num_seen);

  const size_t kCount = 96;
  std::vector<size_t> first_pos(kCount), last_pos(kCount);
  std::vector<T> data(kCount);
  CachedXoshiro<> gen(42);
  for (size_t trial = 0; trial < kCount * 1000; ++trial) {
    for (size_t k = 0; k < kCount; ++k) data[k] = ConvertScalarTo<T>(k);
    shuffle(d, data.data(), kCount, gen);
    for (size_t k = 0; k < kCount; ++k) {
      if (data[k] == ConvertScalarTo<T>(0)) ++first_pos[k];
      if (data[k] == ConvertScalarTo<T>(kCount - 1)) ++last_pos[k];
    }
  }
  for (size_t k = 0; k < kCount; ++k) {
    HWY_ASSERT(800 <= first_pos[k] && first_pos[k] <= 1200);
    HWY_ASSERT(800 <= last_pos[k] && last_pos[k] <= 1200);
  }
}

// The reference tests above show results match on every target and vector
// size, so the other tests of whole shuffles use just one.
void TestUniform() {
  const ScalableTag<uint32_t> d;
  CheckUniform(d, [](ScalableTag<uint32_t> tag, uint32_t* p, size_t n,
                     CachedXoshiro<>& g) { ShuffleSpan(tag, p, n, g); });
}

// Two buckets, whose sizes vary between shuffles.
void TestBucketUniform() {
  const ScalableTag<uint32_t> d;
  std::vector<uint32_t> buf(detail::ShuffleBucketBufNum<uint32_t>(96));
  CheckUniform(d, [&buf](ScalableTag<uint32_t> tag, uint32_t* p, size_t n,
                         CachedXoshiro<>& g) {
    detail::ShuffleDrawnBits<CachedXoshiro<>> bits(g);
    detail::ShuffleBuckets(tag, p, n, bits, /*log2_buckets=*/1, buf.data());
  });
}

// At 64 MiB, the overload with `buf` must match the bucket path with 64
// buckets. Large, so only for one type.
void TestThreshold() {
  if (HWY_IS_DEBUG_BUILD) return;  // too slow
  using T = uint64_t;
  const ScalableTag<T> d;
  const size_t count = detail::kShuffleBucketMinBytes / sizeof(T);
  HWY_ASSERT_EQ(size_t{0}, ShuffleSpanBufNum<T>(count - 1));
  HWY_ASSERT_EQ(size_t{6}, detail::ShuffleLog2Buckets<T>(count));
  // 1 GiB reaches the cap of 1024 buckets, and 2 GiB stays there.
  HWY_ASSERT_EQ(size_t{10}, detail::ShuffleLog2Buckets<T>(size_t{1} << 27));
  HWY_ASSERT_EQ(size_t{10}, detail::ShuffleLog2Buckets<T>(size_t{1} << 28));
  const size_t buf_num = ShuffleSpanBufNum<T>(count);
  HWY_ASSERT_EQ(detail::ShuffleBucketBufNum<T>(count), buf_num);

  AlignedFreeUniquePtr<T[]> data = AllocateAligned<T>(count);
  AlignedFreeUniquePtr<T[]> buf = AllocateAligned<T>(buf_num + 1);
  HWY_ASSERT(data && buf);
  buf[buf_num] = 99;
  uint64_t prints[2];
  for (size_t run = 0; run < 2; ++run) {
    for (size_t k = 0; k < count; ++k) data[k] = k;
    CachedXoshiro<> gen(5);
    if (run == 0) {
      ShuffleSpan(d, data.get(), count, gen, buf.get());
    } else {
      detail::ShuffleDrawnBits<CachedXoshiro<>> bits(gen);
      detail::ShuffleBuckets(d, data.get(), count, bits, /*log2_buckets=*/6,
                             buf.get());
    }
    prints[run] = Fingerprint(data.get(), count);
  }
  HWY_ASSERT_EQ(prints[0], prints[1]);
  HWY_ASSERT_EQ(T{99}, buf[buf_num]);
  std::vector<uint8_t> seen(count);
  bool permutation = true;
  for (size_t k = 0; k < count; ++k) {
    const size_t value = static_cast<size_t>(data[k]);
    permutation &= data[k] < count && !seen[value];
    if (data[k] < count) seen[value] = 1;
  }
  HWY_ASSERT(permutation);
}

// The last 32 of 64 shuffled values depend only on one block's key. A 32-bit
// key would repeat about 8 times in 2^18 shuffles; a 64-bit key, practically
// never.
void TestNoRepeats() {
  const ScalableTag<uint32_t> d;
  const size_t num_shuffles = AdjustedReps(size_t{1} << 18);
  std::vector<uint64_t> prints(num_shuffles);
  CachedXoshiro<> gen(9);
  uint32_t data[64];
  for (size_t s = 0; s < num_shuffles; ++s) {
    for (uint32_t k = 0; k < 64; ++k) data[k] = k;
    ShuffleSpan(d, data, 64, gen);
    prints[s] = Fingerprint(data + 32, 32);
  }
  std::sort(prints.begin(), prints.end());
  size_t repeats = 0;
  for (size_t s = 1; s < num_shuffles; ++s) {
    repeats += prints[s] == prints[s - 1];
  }
  HWY_ASSERT_EQ(size_t{0}, repeats);
}

// Supplies given lower halves to ShuffleTarget in place of ShuffleHash.
struct LowerFrom {
  template <class D, class V>
  V Lower(D d, V /*upper*/) const {
    return Load(d, lower);
  }
  const uint32_t* lower;
};

// Carries out of the middle of the product need ranges near 2^32, which no
// array in a test reaches, so check ShuffleTarget directly against Mul128.
struct TestTarget {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    AlignedFreeUniquePtr<uint32_t[]> upper = AllocateAligned<uint32_t>(N);
    AlignedFreeUniquePtr<uint32_t[]> lower = AllocateAligned<uint32_t>(N);
    AlignedFreeUniquePtr<uint32_t[]> range = AllocateAligned<uint32_t>(N);
    AlignedFreeUniquePtr<uint32_t[]> target = AllocateAligned<uint32_t>(N);
    HWY_ASSERT(upper && lower && range && target);
    RandomState rng;
    size_t num_carries = 0;
    for (size_t rep = 0; rep < AdjustedReps(1000); ++rep) {
      for (size_t k = 0; k < N; ++k) {
        const uint64_t bits = Random64(&rng);
        upper[k] = rep % 4 == 0 ? ~0u : static_cast<uint32_t>(bits >> 32);
        lower[k] = rep % 4 == 1 ? ~0u : static_cast<uint32_t>(bits);
        const uint32_t r = static_cast<uint32_t>(Random64(&rng));
        range[k] = rep % 2 == 0 ? ~0u - static_cast<uint32_t>(k) : r | 1u;
      }
      const LowerFrom from{lower.get()};
      Store(detail::ShuffleTarget(d, Load(d, upper.get()), Load(d, range.get()),
                                  from),
            d, target.get());
      for (size_t k = 0; k < N; ++k) {
        uint64_t expected;
        Mul128((uint64_t{upper[k]} << 32) | lower[k], range[k], &expected);
        HWY_ASSERT_EQ(expected, uint64_t{target[k]});
        num_carries += expected != MulHigh32(upper[k], range[k]);
      }
    }
    HWY_ASSERT(num_carries != 0);
  }
};

void TestAllTarget() { ForPartialVectors<TestTarget>()(uint32_t()); }

// ShuffleHash must give RefPositionBits on every target. The lower 32 bits only
// change a target in the rare case above, so the shuffle tests cannot see them.
struct TestPositionBits {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const size_t N = Lanes(d);
    AlignedFreeUniquePtr<uint32_t[]> upper = AllocateAligned<uint32_t>(N);
    AlignedFreeUniquePtr<uint32_t[]> lower = AllocateAligned<uint32_t>(N);
    HWY_ASSERT(upper && lower);
    const Vec<D> steps = Mul(Iota(d, 0u), Set(d, detail::kShuffleStep));
    RandomState rng;
    for (size_t rep = 0; rep < AdjustedReps(1000); ++rep) {
      const uint64_t key = Random64(&rng);
      const uint32_t first = static_cast<uint32_t>(Random64(&rng));
      const detail::ShuffleHash hash(key);
      const Vec<D> up = hash.Upper(d, first, steps);
      Store(up, d, upper.get());
      Store(hash.Lower(d, up), d, lower.get());
      for (size_t k = 0; k < N; ++k) {
        const uint32_t pos = first + static_cast<uint32_t>(k);
        HWY_ASSERT_EQ(RefPositionBits(key, pos),
                      (uint64_t{upper[k]} << 32) | lower[k]);
      }
    }
  }
};

void TestAllPositionBits() {
  ForPartialVectors<TestPositionBits>()(uint32_t());
}

// The path for positions past 2^32 cannot run in a test, so check its
// position rule directly.
void TestIndex64() {
  RandomState rng;
  const uint64_t kFirst = uint64_t{1} << 32;
  for (uint64_t i : {kFirst - 1, kFirst, kFirst + 12345, ~uint64_t{0} - 1}) {
    HWY_ASSERT_EQ(uint64_t{0}, detail::ShuffleIndex64(0, i));
    HWY_ASSERT_EQ(i, detail::ShuffleIndex64(~uint64_t{0}, i));
    for (size_t rep = 0; rep < 1000; ++rep) {
      HWY_ASSERT(detail::ShuffleIndex64(Random64(&rng), i) <= i);
    }
  }
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(ShuffleTest);
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestAllGenerator);
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestAllBuckets);
HWY_EXPORT_AND_TEST_BEST_P(ShuffleTest, TestUniform);
HWY_EXPORT_AND_TEST_BEST_P(ShuffleTest, TestBucketUniform);
HWY_EXPORT_AND_TEST_BEST_P(ShuffleTest, TestThreshold);
HWY_EXPORT_AND_TEST_BEST_P(ShuffleTest, TestNoRepeats);
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestAllTarget);
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestAllPositionBits);
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestIndex64);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
