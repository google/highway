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

#include <random>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/algo/shuffle_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/algo/shuffle-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

// Sequential Fisher-Yates using the position rule ShuffleSpan documents.
// `next()` returns the next 32 random bits, in the order a sequential loop
// consumes them.
template <class Next>
std::vector<size_t> ReferencePermutation(size_t count, Next next) {
  std::vector<size_t> perm(count);
  for (size_t k = 0; k < count; ++k) perm[k] = k;
  for (size_t i = count; i-- > 1;) {
    const uint32_t i32 = static_cast<uint32_t>(i);
    std::swap(perm[i], perm[MulHigh32(next(), i32 + 1)]);
  }
  return perm;
}

void AssertIsPermutation(const std::vector<size_t>& perm) {
  std::vector<bool> seen(perm.size());
  for (size_t p : perm) {
    HWY_ASSERT(p < perm.size() && !seen[p]);
    seen[p] = true;
  }
}

// Values are distinct for every count used below, including for 8-bit and
// float16 lanes.
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
  counts.push_back(1000);
  return counts;
}

// ShuffleSpan must consume draws in the same order as a sequential loop, so
// identically seeded generators give the reference permutation.
template <class D, class Gen>
void CheckGenerator(D d, const Gen& prototype) {
  using T = TFromD<D>;
  const size_t misalignments[2] = {0, Lanes(d) / 3 + 1};
  for (size_t count : CountsFor(d)) {
    Gen ref_gen = prototype;
    const std::vector<size_t> perm = ReferencePermutation(count, [&ref_gen]() {
      return static_cast<uint32_t>(ref_gen() - (Gen::min)());
    });
    AssertIsPermutation(perm);
    for (size_t misalign : misalignments) {
      Gen gen = prototype;
      CheckShuffle(d, count, misalign, perm, [&gen](D tag, T* p, size_t n) {
        ShuffleSpan(tag, p, n, gen);
      });
    }
  }
}

struct TestGenerator {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    CheckGenerator(d, std::mt19937(123));
    CheckGenerator(d, std::mt19937_64(456));
    const AesCtrEngine engine(/*deterministic=*/true);
    CheckGenerator(d, RngStream(engine, 789));
  }
};

void TestAllGenerator() { ForAllTypes(ForPartialVectors<TestGenerator>()); }

// Every ordering of 4 elements, and every final position of the first and last
// of 64 elements (which goes through the vector path), should be about equally
// likely. Generator seeds are fixed, so this cannot flake; bounds are ~6 sigma.
struct TestUniform {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    std::vector<size_t> orderings(256);
    const size_t kOrderingTrials = 24 * 1000;
    std::mt19937 ordering_gen(1);
    for (size_t trial = 0; trial < kOrderingTrials; ++trial) {
      T data[4] = {0, 1, 2, 3};
      ShuffleSpan(d, data, 4, ordering_gen);
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

    const size_t kCount = 64;
    std::vector<size_t> first_pos(kCount), last_pos(kCount);
    std::vector<T> data(kCount);
    std::mt19937 gen(42);
    for (size_t trial = 0; trial < kCount * 1000; ++trial) {
      for (size_t k = 0; k < kCount; ++k) data[k] = ConvertScalarTo<T>(k);
      ShuffleSpan(d, data.data(), kCount, gen);
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
};

void TestAllUniform() { ForPartialVectors<TestUniform>()(uint32_t()); }

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
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestAllUniform);
HWY_EXPORT_AND_TEST_P(ShuffleTest, TestIndex64);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
