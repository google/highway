// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause

#include <stdint.h>
#include <stdio.h>

#include <limits>
#include <set>
#include <vector>

// Skip older x86 and redundant Arm targets (NEON_WITHOUT_AES, SVE2, and
// SVE2_128 still run) to keep CppCompile time well within CI limits.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS                                              \
  (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_NEON | HWY_NEON_BF16 | HWY_SVE | \
   HWY_SVE_256)
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_set32_test.cc"  // NOLINT
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/btree/btree_set.h"
#include "hwy/contrib/btree/btree_test_util-inl.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

enum DefaultSignedEnum { kNeg = -42, kZero = 0, kPos = 42 };
enum DefaultFullRangeEnum {
  kMinVal = -0x7fffffff - 1,
  kNegVal = -1,
  kZeroVal = 0,
  kMaxVal = 0x7fffffff
};
enum UnscopedSignedEnum32 : int32_t {};
enum class ScopedUnsignedEnum32 : uint32_t {};

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestBTreeSetU32() {}
HWY_NOINLINE void TestBTreeSetI32() {}
HWY_NOINLINE void TestBTreeSetEnum32() {}
HWY_NOINLINE void TestBTreeSetSignedBoundaries32() {}
#else

HWY_NOINLINE void TestBTreeSetU32() {
  fprintf(stderr, "Running Public API BTreeSet uint32_t tests...\n");
  RunFullTestSuite<hwy::BTreeSet<uint32_t>, std::set<uint32_t> >();
}

HWY_NOINLINE void TestBTreeSetI32() {
  fprintf(stderr, "Running Public API BTreeSet int32_t tests...\n");
  RunFullTestSuite<hwy::BTreeSet<int32_t>, std::set<int32_t> >();
}

HWY_NOINLINE void TestBTreeSetEnum32() {
  fprintf(stderr, "Running Public API BTreeSet 32-bit enum tests...\n");
  RunFullTestSuite<hwy::BTreeSet<DefaultFullRangeEnum>,
                   std::set<DefaultFullRangeEnum> >();
  RunFullTestSuite<hwy::BTreeSet<UnscopedSignedEnum32>,
                   std::set<UnscopedSignedEnum32> >();
  RunFullTestSuite<hwy::BTreeSet<ScopedUnsignedEnum32>,
                   std::set<ScopedUnsignedEnum32> >();

  // Direct test on an unscoped enum with no explicit underlying type and a
  // negative enumerator.
  hwy::BTreeSet<DefaultSignedEnum> default_enum_set;
  default_enum_set.insert(kPos);
  default_enum_set.insert(kNeg);
  default_enum_set.insert(kZero);
  HWY_ASSERT_EQ(size_t{3}, default_enum_set.size());
  auto it = default_enum_set.begin();
  VerifyEqualValues(*it++, kNeg);
  VerifyEqualValues(*it++, kZero);
  VerifyEqualValues(*it++, kPos);
  HWY_ASSERT(it == default_enum_set.end());
}

template <typename SignedKeyT>
void VerifySignedSetBoundaries() {
  constexpr SignedKeyT kMin = std::numeric_limits<SignedKeyT>::min();
  constexpr SignedKeyT kMax = std::numeric_limits<SignedKeyT>::max();
  const SignedKeyT sorted_keys[] = {kMin, -1000, -1, 0, 1, 1000, kMax};
  constexpr size_t kNumKeys = sizeof(sorted_keys) / sizeof(sorted_keys[0]);

  auto set_sorted = hwy::BTreeSet<SignedKeyT>::Build(sorted_keys, kNumKeys);
  HWY_ASSERT_EQ(kNumKeys, set_sorted.size());

  const SignedKeyT unsorted_dups[] = {kMax, 0,    kMin, -1,    0,
                                      1,    kMin, 1000, -1000, kMax};
  auto set_unsorted = hwy::BTreeSet<SignedKeyT>::Build(
      unsorted_dups, sizeof(unsorted_dups) / sizeof(unsorted_dups[0]));
  HWY_ASSERT_EQ(kNumKeys, set_unsorted.size());

  // Exercise CopyConstruct and CopyAssign on signed storage-erased sets.
  hwy::BTreeSet<SignedKeyT> copied(set_sorted);
  hwy::BTreeSet<SignedKeyT> assigned;
  assigned = set_unsorted;

  for (const auto* s : {&set_sorted, &set_unsorted, &copied, &assigned}) {
    size_t idx = 0;
    for (SignedKeyT k : *s) {
      HWY_ASSERT_EQ(sorted_keys[idx++], k);
    }
    HWY_ASSERT_EQ(kNumKeys, idx);

    // 11 queries = 8-way SIMD micro-batch + 3-element scalar tail.
    const SignedKeyT queries[11] = {kMin, -2000, -1000, -2,   -1,  0,
                                    1,    2,     1000,  2000, kMax};
    const bool expected_found[11] = {true, false, true, false, true, true,
                                     true, false, true, false, true};

    // Test both < 8 tail-only (count = 5) and >= 8 pipelined + tail (11).
    for (size_t count : {size_t{5}, size_t{11}}) {
      bool found[11] = {};
      s->ContainsBatch(queries, count, found);
      typename hwy::BTreeSet<SignedKeyT>::const_iterator find_it[11] = {};
      s->FindBatch(queries, count, find_it);
      typename hwy::BTreeSet<SignedKeyT>::const_iterator lb_it[11] = {};
      s->LowerBoundBatch(queries, count, lb_it);

      for (size_t i = 0; i < count; ++i) {
        HWY_ASSERT_EQ(expected_found[i], found[i]);
        HWY_ASSERT_EQ(expected_found[i], find_it[i] != s->end());
        if (expected_found[i]) {
          HWY_ASSERT_EQ(queries[i], *find_it[i]);
        }
        auto scalar_lb = s->lower_bound(queries[i]);
        HWY_ASSERT(lb_it[i] == scalar_lb);
      }
    }
  }
}

HWY_NOINLINE void TestBTreeSetSignedBoundaries32() {
  VerifySignedSetBoundaries<int32_t>();
}

#endif  // (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(BTreeSet32PublicTest);
HWY_EXPORT_AND_TEST_P(BTreeSet32PublicTest, TestBTreeSetU32);
HWY_EXPORT_AND_TEST_P(BTreeSet32PublicTest, TestBTreeSetI32);
HWY_EXPORT_AND_TEST_P(BTreeSet32PublicTest, TestBTreeSetEnum32);
HWY_EXPORT_AND_TEST_P(BTreeSet32PublicTest, TestBTreeSetSignedBoundaries32);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
