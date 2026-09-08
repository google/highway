// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
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
#include <set>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/base.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/btree/btree_test.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/btree/btree-inl.h"
#include "hwy/contrib/btree/btree_test_util-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE
HWY_NOINLINE void TestAll() {}
#else
void TestAll() {
  fprintf(stderr, "Running BTreeSet uint32_t tests...\n");
  RunFullTestSuite<BTreeSet<uint32_t>, std::set<uint32_t> >();

  fprintf(stderr, "Running BTreeSet int32_t tests...\n");
  RunFullTestSuite<BTreeSet<int32_t>, std::set<int32_t> >();

  fprintf(stderr, "Running BTreeSet uint64_t tests...\n");
  RunFullTestSuite<BTreeSet<uint64_t>, std::set<uint64_t> >();

  fprintf(stderr, "Running BTreeSet int64_t tests...\n");
  RunFullTestSuite<BTreeSet<int64_t>, std::set<int64_t> >();

  fprintf(stderr, "Running BTreeMap uint32_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<uint32_t, uint64_t>,
                   std::map<uint32_t, uint64_t> >();

  fprintf(stderr, "Running BTreeMap int32_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<int32_t, uint64_t>, std::map<int32_t, uint64_t> >();

  fprintf(stderr, "Running BTreeMap uint64_t -> uint64_t tests...\n");
  RunFullTestSuite<BTreeMap<uint64_t, uint64_t>,
                   std::map<uint64_t, uint64_t> >();

  fprintf(stderr, "Running BTreeMap uint64_t -> double tests...\n");
  RunFullTestSuite<BTreeMap<uint64_t, double>, std::map<uint64_t, double> >();

  fprintf(stderr, "Running BTreeMap int64_t -> double tests...\n");
  RunFullTestSuite<BTreeMap<int64_t, double>, std::map<int64_t, double> >();

  fprintf(stderr, "Running BTreeMap uint32_t -> float tests...\n");
  RunFullTestSuite<BTreeMap<uint32_t, float>, std::map<uint32_t, float> >();

  fprintf(stderr, "All unified BTree tests passed successfully!\n");
}

#endif  // (HWY_TARGET == HWY_SCALAR || HWY_TARGET == HWY_EMU128) && !HWY_IDE

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(BTreeTest);
HWY_EXPORT_AND_TEST_P(BTreeTest, TestAll);
HWY_AFTER_TEST();
}  // namespace hwy
HWY_TEST_MAIN();

#endif  // HWY_ONCE
