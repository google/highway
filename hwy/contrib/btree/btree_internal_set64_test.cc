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

#include <set>

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
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_internal_set64_test.cc"  // NOLINT
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

enum class ScopedSignedEnum64 : int64_t {};
enum class ScopedUnsignedEnum64 : uint64_t {};

#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestSetAdapterMove64() {}
HWY_NOINLINE void TestBTreeSetU64() {}
HWY_NOINLINE void TestBTreeSetI64() {}
HWY_NOINLINE void TestBTreeSetPtrAndEnum64() {}
#else
HWY_NOINLINE void TestSetAdapterMove64() {
  fprintf(stderr, "Running BTreeSet 64-bit adapter move semantics tests...\n");
  DoAdapterMoveTest<BTreeSet<int64_t> >();
  DoAdapterMoveTest<BTreeSet<void*> >();
  DoAdapterMoveTest<BTreeSet<ScopedSignedEnum64> >();
}

HWY_NOINLINE void TestBTreeSetU64() {
  fprintf(stderr, "Running BTreeSet uint64_t tests...\n");
  RunFullTestSuite<BTreeSet<uint64_t>, std::set<uint64_t> >();
}

HWY_NOINLINE void TestBTreeSetI64() {
  fprintf(stderr, "Running BTreeSet int64_t tests...\n");
  RunFullTestSuite<BTreeSet<int64_t>, std::set<int64_t> >();
}

HWY_NOINLINE void TestBTreeSetPtrAndEnum64() {
  fprintf(stderr, "Running BTreeSet pointer and 64-bit enum tests...\n");
  RunFullTestSuite<BTreeSet<uint64_t*>, std::set<uint64_t*> >();
  RunFullTestSuite<BTreeSet<const uint32_t*>, std::set<const uint32_t*> >();
  RunFullTestSuite<BTreeSet<void*>, std::set<void*> >();
  RunFullTestSuite<BTreeSet<ScopedSignedEnum64>,
                   std::set<ScopedSignedEnum64> >();
  RunFullTestSuite<BTreeSet<ScopedUnsignedEnum64>,
                   std::set<ScopedUnsignedEnum64> >();
}

#endif  // (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(BTreeInternalSet64Test);
HWY_EXPORT_AND_TEST_P(BTreeInternalSet64Test, TestSetAdapterMove64);
HWY_EXPORT_AND_TEST_P(BTreeInternalSet64Test, TestBTreeSetU64);
HWY_EXPORT_AND_TEST_P(BTreeInternalSet64Test, TestBTreeSetI64);
HWY_EXPORT_AND_TEST_P(BTreeInternalSet64Test, TestBTreeSetPtrAndEnum64);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();

#endif  // HWY_ONCE
