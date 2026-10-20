// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>

#include <set>

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_set_test.cc"  // NOLINT
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
enum class ScopedSignedEnum64 : int64_t {};
enum class ScopedUnsignedEnum64 : uint64_t {};

HWY_NOINLINE void TestBTreeSetU32() {
  fprintf(stderr, "Running Public API BTreeSet uint32_t tests...\n");
  RunFullTestSuite<hwy::BTreeSet<uint32_t>, std::set<uint32_t>>();
}

HWY_NOINLINE void TestBTreeSetI32() {
  fprintf(stderr, "Running Public API BTreeSet int32_t tests...\n");
  RunFullTestSuite<hwy::BTreeSet<int32_t>, std::set<int32_t>>();
}

HWY_NOINLINE void TestBTreeSetU64() {
  fprintf(stderr, "Running Public API BTreeSet uint64_t tests...\n");
  RunFullTestSuite<hwy::BTreeSet<uint64_t>, std::set<uint64_t>>();
}

HWY_NOINLINE void TestBTreeSetI64() {
  fprintf(stderr, "Running Public API BTreeSet int64_t tests...\n");
  RunFullTestSuite<hwy::BTreeSet<int64_t>, std::set<int64_t>>();
}

HWY_NOINLINE void TestBTreeSetPtrAndEnum() {
  fprintf(stderr, "Running Public API BTreeSet pointer and enum tests...\n");
  RunFullTestSuite<hwy::BTreeSet<uint64_t*>, std::set<uint64_t*>>();
  RunFullTestSuite<hwy::BTreeSet<const uint32_t*>, std::set<const uint32_t*>>();
  RunFullTestSuite<hwy::BTreeSet<void*>, std::set<void*>>();
  RunFullTestSuite<hwy::BTreeSet<DefaultFullRangeEnum>,
                   std::set<DefaultFullRangeEnum>>();
  RunFullTestSuite<hwy::BTreeSet<UnscopedSignedEnum32>,
                   std::set<UnscopedSignedEnum32>>();
  RunFullTestSuite<hwy::BTreeSet<ScopedUnsignedEnum32>,
                   std::set<ScopedUnsignedEnum32>>();
  RunFullTestSuite<hwy::BTreeSet<ScopedSignedEnum64>,
                   std::set<ScopedSignedEnum64>>();
  RunFullTestSuite<hwy::BTreeSet<ScopedUnsignedEnum64>,
                   std::set<ScopedUnsignedEnum64>>();

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

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(BTreeSetPublicTest);
HWY_EXPORT_AND_TEST_P(BTreeSetPublicTest, TestBTreeSetU32);
HWY_EXPORT_AND_TEST_P(BTreeSetPublicTest, TestBTreeSetI32);
HWY_EXPORT_AND_TEST_P(BTreeSetPublicTest, TestBTreeSetU64);
HWY_EXPORT_AND_TEST_P(BTreeSetPublicTest, TestBTreeSetI64);
HWY_EXPORT_AND_TEST_P(BTreeSetPublicTest, TestBTreeSetPtrAndEnum);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
