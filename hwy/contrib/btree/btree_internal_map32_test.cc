// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>

#include <map>

// Skip older x86 and redundant Arm targets (NEON_WITHOUT_AES, SVE2, and
// SVE2_128 still run) to keep CppCompile time well within CI limits.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS                                              \
  (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_NEON | HWY_NEON_BF16 | HWY_SVE | \
   HWY_SVE_256)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_internal_map32_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/btree/btree-inl.h"
#include "hwy/contrib/btree/btree_test_util-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

enum UnscopedSignedEnum32 : int32_t {};
enum class ScopedUnsignedEnum32 : uint32_t {};

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestMapAdapterMove32() {}
HWY_NOINLINE void TestBTreeMapU32U64() {}
HWY_NOINLINE void TestBTreeMapI32U64() {}
HWY_NOINLINE void TestBTreeMapU32F32() {}
HWY_NOINLINE void TestBTreeMapEnum32() {}
#else
HWY_NOINLINE void TestMapAdapterMove32() {
  fprintf(stderr, "Running BTreeMap 32-bit adapter move semantics tests...\n");
  DoAdapterMoveTest<BTreeMap<uint32_t, uint64_t> >();
}

HWY_NOINLINE void TestBTreeMapU32U64() {
  fprintf(stderr, "Running BTreeMap uint32_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<uint32_t, uint64_t>,
                   std::map<uint32_t, uint64_t> >();
}

HWY_NOINLINE void TestBTreeMapI32U64() {
  fprintf(stderr, "Running BTreeMap int32_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<int32_t, uint64_t>, std::map<int32_t, uint64_t> >();
}

HWY_NOINLINE void TestBTreeMapU32F32() {
  fprintf(stderr, "Running BTreeMap uint32_t -> float tests...\n");
  RunFullTestSuite<BTreeMap<uint32_t, float>, std::map<uint32_t, float> >();
}

HWY_NOINLINE void TestBTreeMapEnum32() {
  fprintf(stderr, "Running BTreeMap 32-bit enum key tests...\n");
  RunFullTestSuite<BTreeMap<UnscopedSignedEnum32, uint64_t>,
                   std::map<UnscopedSignedEnum32, uint64_t> >();
  RunFullTestSuite<BTreeMap<ScopedUnsignedEnum32, float>,
                   std::map<ScopedUnsignedEnum32, float> >();
}

#endif  // (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(BTreeInternalMap32Test);
HWY_EXPORT_AND_TEST_P(BTreeInternalMap32Test, TestMapAdapterMove32);
HWY_EXPORT_AND_TEST_P(BTreeInternalMap32Test, TestBTreeMapU32U64);
HWY_EXPORT_AND_TEST_P(BTreeInternalMap32Test, TestBTreeMapI32U64);
HWY_EXPORT_AND_TEST_P(BTreeInternalMap32Test, TestBTreeMapU32F32);
HWY_EXPORT_AND_TEST_P(BTreeInternalMap32Test, TestBTreeMapEnum32);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();

#endif  // HWY_ONCE
