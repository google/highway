// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>

#include <map>

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_map_test.cc"  // NOLINT
// clang-format on

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/btree/btree_map.h"
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

struct Custom32 {
  int16_t a = 0;
  int16_t b = 0;

  constexpr Custom32() = default;
  constexpr Custom32(int v1, int v2)
      : a(static_cast<int16_t>(v1)), b(static_cast<int16_t>(v2)) {}

  bool operator==(const Custom32& other) const {
    return a == other.a && b == other.b;
  }
};
static_assert(sizeof(Custom32) == 4);
static_assert(std::is_trivially_copyable_v<Custom32>);

struct Custom64 {
  int32_t a = 0;
  int32_t b = 0;

  constexpr Custom64() = default;
  constexpr Custom64(int v1, int v2) : a(v1), b(v2) {}

  bool operator==(const Custom64& other) const {
    return a == other.a && b == other.b;
  }
};
static_assert(sizeof(Custom64) == 8);
static_assert(std::is_trivially_copyable_v<Custom64>);

void TestCustomValues() {
  // Test 32-bit custom struct value
  {
    hwy::BTreeMap<uint32_t, Custom32> map32;
    map32[10] = Custom32(1, 2);
    map32[20] = Custom32(3, 4);
    HWY_ASSERT_EQ(size_t{2}, map32.size());
    HWY_ASSERT_EQ(true, (map32[10] == Custom32(1, 2)));
    HWY_ASSERT_EQ(true, (map32[20] == Custom32(3, 4)));

    auto it = map32.find(10);
    HWY_ASSERT_EQ(true, it != map32.end());
    HWY_ASSERT_EQ(true, (it->second == Custom32(1, 2)));

    auto [it_new, inserted_new] = map32.emplace(30, 5, 6);
    HWY_ASSERT_EQ(true, inserted_new);
    HWY_ASSERT_EQ(true, (it_new->second == Custom32(5, 6)));

    // emplace on existing key must NOT overwrite existing value
    auto [it_dup, inserted_dup] = map32.emplace(10, 99, 99);
    HWY_ASSERT_EQ(false, inserted_dup);
    HWY_ASSERT_EQ(true, (it_dup->second == Custom32(1, 2)));
    HWY_ASSERT_EQ(true, (map32[10] == Custom32(1, 2)));

    map32.erase(10);
    map32.erase(30);
    HWY_ASSERT_EQ(size_t{1}, map32.size());
    HWY_ASSERT_EQ(true, map32.find(10) == map32.end());
  }

  // Test 64-bit custom struct value
  {
    hwy::BTreeMap<int64_t, Custom64> map64;
    map64[-100] = Custom64(10, 20);
    map64[200] = Custom64(30, 40);
    HWY_ASSERT_EQ(size_t{2}, map64.size());
    HWY_ASSERT_EQ(true, (map64[-100] == Custom64(10, 20)));
    HWY_ASSERT_EQ(true, (map64[200] == Custom64(30, 40)));

    auto it = map64.find(200);
    HWY_ASSERT_EQ(true, it != map64.end());
    HWY_ASSERT_EQ(true, (it->second == Custom64(30, 40)));

    const Custom64* val_ptr = map64.FindValue(-100);
    HWY_ASSERT_EQ(true, val_ptr != nullptr);
    HWY_ASSERT_EQ(true, (*val_ptr == Custom64(10, 20)));
  }
}

template <typename KeyT, typename ValueT, typename MakeValFunc>
void TestBatchOperationsForType(MakeValFunc make_val) {
  constexpr size_t kNumItems = 100;
  std::vector<KeyT> sorted_keys(kNumItems);
  std::vector<ValueT> sorted_vals(kNumItems);
  for (size_t i = 0; i < kNumItems; ++i) {
    sorted_keys[i] = KeyFromRawInt<KeyT>(i * 10);
    sorted_vals[i] = make_val(i);
  }

  // 1. Test BTreeMap::Build
  auto map = hwy::BTreeMap<KeyT, ValueT>::Build(sorted_keys.data(),
                                                sorted_vals.data(), kNumItems);
  HWY_ASSERT_EQ(kNumItems, map.size());

  // 2. Test ContainsBatch
  const KeyT queries[5] = {KeyFromRawInt<KeyT>(0), KeyFromRawInt<KeyT>(20),
                           KeyFromRawInt<KeyT>(25), KeyFromRawInt<KeyT>(990),
                           KeyFromRawInt<KeyT>(1000)};
  bool contains_out[5] = {};
  map.ContainsBatch(queries, 5, contains_out);
  HWY_ASSERT_EQ(true, contains_out[0]);
  HWY_ASSERT_EQ(true, contains_out[1]);
  HWY_ASSERT_EQ(false, contains_out[2]);
  HWY_ASSERT_EQ(true, contains_out[3]);
  HWY_ASSERT_EQ(false, contains_out[4]);

  // 3. Test FindBatch
  using const_iterator = typename hwy::BTreeMap<KeyT, ValueT>::const_iterator;
  const_iterator find_out[5] = {};
  map.FindBatch(queries, 5, find_out);
  HWY_ASSERT_EQ(true, find_out[0] != map.end());
  VerifyEqualValues(sorted_keys[0], find_out[0]->first);
  HWY_ASSERT_EQ(true, (find_out[0]->second == sorted_vals[0]));

  HWY_ASSERT_EQ(true, find_out[1] != map.end());
  VerifyEqualValues(sorted_keys[2], find_out[1]->first);
  HWY_ASSERT_EQ(true, (find_out[1]->second == sorted_vals[2]));

  HWY_ASSERT_EQ(true, find_out[2] == map.end());
  HWY_ASSERT_EQ(true, find_out[3] != map.end());
  HWY_ASSERT_EQ(true, find_out[4] == map.end());

  // 4. Test LowerBoundBatch
  const_iterator lb_out[5] = {};
  map.LowerBoundBatch(queries, 5, lb_out);
  HWY_ASSERT_EQ(true, lb_out[0] != map.end());
  VerifyEqualValues(sorted_keys[0], lb_out[0]->first);

  HWY_ASSERT_EQ(true, lb_out[1] != map.end());
  VerifyEqualValues(sorted_keys[2], lb_out[1]->first);

  HWY_ASSERT_EQ(true, lb_out[2] != map.end());
  VerifyEqualValues(KeyFromRawInt<KeyT>(30), lb_out[2]->first);

  HWY_ASSERT_EQ(true, lb_out[3] != map.end());
  VerifyEqualValues(sorted_keys[99], lb_out[3]->first);

  HWY_ASSERT_EQ(true, lb_out[4] == map.end());

  // 5. Test LookupBatch
  ValueT lookup_vals[5] = {};
  bool lookup_found[5] = {};
  map.LookupBatch(queries, 5, lookup_vals, lookup_found);
  HWY_ASSERT_EQ(true, lookup_found[0]);
  HWY_ASSERT_EQ(true, (lookup_vals[0] == sorted_vals[0]));

  HWY_ASSERT_EQ(true, lookup_found[1]);
  HWY_ASSERT_EQ(true, (lookup_vals[1] == sorted_vals[2]));

  HWY_ASSERT_EQ(false, lookup_found[2]);

  HWY_ASSERT_EQ(true, lookup_found[3]);
  HWY_ASSERT_EQ(true, (lookup_vals[3] == sorted_vals[99]));

  HWY_ASSERT_EQ(false, lookup_found[4]);
}

void TestBatchOperations() {
  // Fundamental float (32-bit)
  TestBatchOperationsForType<uint32_t, float>(
      [](size_t i) { return static_cast<float>(i) * 1.5f; });
  // Fundamental double (64-bit)
  TestBatchOperationsForType<int64_t, double>(
      [](size_t i) { return static_cast<double>(i) * 2.5; });
  // Custom 32-bit struct
  TestBatchOperationsForType<uint32_t, Custom32>([](size_t i) {
    return Custom32(static_cast<int>(i), static_cast<int>(i * 2));
  });
  // Custom 64-bit struct
  TestBatchOperationsForType<int64_t, Custom64>([](size_t i) {
    return Custom64(static_cast<int>(i), static_cast<int>(i * 3));
  });
  // Pointer and enum keys with custom structs
  TestBatchOperationsForType<void*, Custom64>([](size_t i) {
    return Custom64(static_cast<int>(i), static_cast<int>(i * 3));
  });
  TestBatchOperationsForType<ScopedSignedEnum64, Custom32>([](size_t i) {
    return Custom32(static_cast<int>(i), static_cast<int>(i * 2));
  });
}

template <typename KeyT, typename ValueT>
void TestMapSpecificSemantics() {
  hwy::BTreeMap<KeyT, ValueT> map;
  const KeyT k10 = KeyFromRawInt<KeyT>(10);
  const KeyT k20 = KeyFromRawInt<KeyT>(20);
  const KeyT k30 = KeyFromRawInt<KeyT>(30);
  const KeyT k40 = KeyFromRawInt<KeyT>(40);
  const KeyT k999 = KeyFromRawInt<KeyT>(999);

  // 1. operator[] insertions and updates
  map[k10] = static_cast<ValueT>(100);
  map[k20] = static_cast<ValueT>(200);
  map[k30] = static_cast<ValueT>(300);
  HWY_ASSERT_EQ(size_t{3}, map.size());
  HWY_ASSERT_EQ(static_cast<ValueT>(100), map[k10]);
  HWY_ASSERT_EQ(static_cast<ValueT>(200), map[k20]);
  HWY_ASSERT_EQ(static_cast<ValueT>(300), map[k30]);

  // Overwrite via operator[]
  map[k20] = static_cast<ValueT>(250);
  HWY_ASSERT_EQ(static_cast<ValueT>(250), map[k20]);
  HWY_ASSERT_EQ(size_t{3}, map.size());

  // 2. at() checked access (testing both const and mutable overloads)
  const auto& const_map = map;
  HWY_ASSERT_EQ(static_cast<ValueT>(100), const_map.at(k10));
  HWY_ASSERT_EQ(static_cast<ValueT>(250), const_map.at(k20));
  map.at(k10) = static_cast<ValueT>(105);
  HWY_ASSERT_EQ(static_cast<ValueT>(105), const_map.at(k10));

  // 3. FindValue direct pointer lookup (testing both const and mutable
  // overloads)
  const auto* const_val_ptr = const_map.FindValue(k30);
  HWY_ASSERT_EQ(true, const_val_ptr != nullptr);
  HWY_ASSERT_EQ(static_cast<ValueT>(300), *const_val_ptr);
  HWY_ASSERT_EQ(true, const_map.FindValue(k999) == nullptr);

  auto* mut_val_ptr = map.FindValue(k30);
  HWY_ASSERT_EQ(true, mut_val_ptr != nullptr);
  HWY_ASSERT_EQ(static_cast<ValueT>(300), *mut_val_ptr);
  HWY_ASSERT_EQ(true, map.FindValue(k999) == nullptr);

  // 4. In-place mutable iterator value updates
  for (auto it = map.begin(); it != map.end(); ++it) {
    it->second += static_cast<ValueT>(1);
  }
  HWY_ASSERT_EQ(static_cast<ValueT>(106), map[k10]);
  HWY_ASSERT_EQ(static_cast<ValueT>(251), map[k20]);
  HWY_ASSERT_EQ(static_cast<ValueT>(301), map[k30]);

  // 5. insert_or_assign
  auto res1 = map.insert_or_assign(k10, static_cast<ValueT>(110));
  HWY_ASSERT_EQ(false, res1.second);  // existing key updated
  HWY_ASSERT_EQ(static_cast<ValueT>(110), map[k10]);

  auto res2 = map.insert_or_assign(k40, static_cast<ValueT>(400));
  HWY_ASSERT_EQ(true, res2.second);  // new key inserted
  HWY_ASSERT_EQ(static_cast<ValueT>(400), map[k40]);
  HWY_ASSERT_EQ(size_t{4}, map.size());
}

template <typename KeyT, typename ValueT>
void TestOneCombo() {
  RunFullTestSuite<hwy::BTreeMap<KeyT, ValueT>, std::map<KeyT, ValueT>>();
  TestMapSpecificSemantics<KeyT, ValueT>();
}

#define DEFINE_MAP_KEY_TESTS(KeySuffix, KeyT)        \
  HWY_NOINLINE void TestBTreeMap##KeySuffix##U32() { \
    TestOneCombo<KeyT, uint32_t>();                  \
  }                                                  \
  HWY_NOINLINE void TestBTreeMap##KeySuffix##I32() { \
    TestOneCombo<KeyT, int32_t>();                   \
  }                                                  \
  HWY_NOINLINE void TestBTreeMap##KeySuffix##U64() { \
    TestOneCombo<KeyT, uint64_t>();                  \
  }                                                  \
  HWY_NOINLINE void TestBTreeMap##KeySuffix##I64() { \
    TestOneCombo<KeyT, int64_t>();                   \
  }                                                  \
  HWY_NOINLINE void TestBTreeMap##KeySuffix##F32() { \
    TestOneCombo<KeyT, float>();                     \
  }                                                  \
  HWY_NOINLINE void TestBTreeMap##KeySuffix##F64() { \
    TestOneCombo<KeyT, double>();                    \
  }

DEFINE_MAP_KEY_TESTS(U32, uint32_t)
DEFINE_MAP_KEY_TESTS(I32, int32_t)
DEFINE_MAP_KEY_TESTS(U64, uint64_t)
DEFINE_MAP_KEY_TESTS(I64, int64_t)

#undef DEFINE_MAP_KEY_TESTS

HWY_NOINLINE void TestBTreeMapPtrAndEnum() {
  fprintf(stderr,
          "Running Public API BTreeMap pointer and enum key tests...\n");
  TestOneCombo<uint64_t*, uint64_t>();
  TestOneCombo<const uint32_t*, double>();
  TestOneCombo<void*, uint32_t>();
  TestOneCombo<DefaultFullRangeEnum, int32_t>();
  TestOneCombo<UnscopedSignedEnum32, uint64_t>();
  TestOneCombo<ScopedUnsignedEnum32, float>();
  TestOneCombo<ScopedSignedEnum64, double>();
  TestOneCombo<ScopedUnsignedEnum64, uint64_t>();

  // Direct test on an unscoped enum with no explicit underlying type and a
  // negative enumerator.
  hwy::BTreeMap<DefaultSignedEnum, int32_t> default_enum_map;
  default_enum_map[kPos] = 100;
  default_enum_map[kNeg] = -100;
  default_enum_map[kZero] = 0;
  HWY_ASSERT_EQ(size_t{3}, default_enum_map.size());
  auto it = default_enum_map.begin();
  VerifyEqualValues(it->first, kNeg);
  HWY_ASSERT_EQ(-100, it->second);
  ++it;
  VerifyEqualValues(it->first, kZero);
  HWY_ASSERT_EQ(0, it->second);
  ++it;
  VerifyEqualValues(it->first, kPos);
  HWY_ASSERT_EQ(100, it->second);
}

HWY_NOINLINE void TestMapCustomValues() {
  fprintf(stderr, "Running Custom32/Custom64 value type tests...\n");
  TestCustomValues();
}

HWY_NOINLINE void TestMapBatchOperations() {
  fprintf(stderr,
          "Running batch operations tests (float, double, Custom32, "
          "Custom64)...\n");
  TestBatchOperations();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(BTreeMapPublicTest);
#define EXPORT_MAP_KEY_TESTS(KeySuffix)                                    \
  HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMap##KeySuffix##U32); \
  HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMap##KeySuffix##I32); \
  HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMap##KeySuffix##U64); \
  HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMap##KeySuffix##I64); \
  HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMap##KeySuffix##F32); \
  HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMap##KeySuffix##F64)

EXPORT_MAP_KEY_TESTS(U32);
EXPORT_MAP_KEY_TESTS(I32);
EXPORT_MAP_KEY_TESTS(U64);
EXPORT_MAP_KEY_TESTS(I64);
#undef EXPORT_MAP_KEY_TESTS
HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestBTreeMapPtrAndEnum);
HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestMapCustomValues);
HWY_EXPORT_AND_TEST_P(BTreeMapPublicTest, TestMapBatchOperations);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
