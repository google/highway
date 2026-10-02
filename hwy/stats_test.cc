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

#include "hwy/stats.h"

#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

template <typename BinT>
void TestBinsAssimilate() {
  Bins<4, BinT> total;
  Bins<4, BinT> part;
  total.Notify(0);
  part.Notify(1);
  part.Notify(1);
  part.IncrementBy(3, 5);
  total.Assimilate(part);
  HWY_ASSERT_EQ(BinT{1}, total.Bin(0));
  HWY_ASSERT_EQ(BinT{2}, total.Bin(1));
  HWY_ASSERT_EQ(BinT{0}, total.Bin(2));
  HWY_ASSERT_EQ(BinT{5}, total.Bin(3));
  HWY_ASSERT_EQ(size_t{3}, total.NumNonzero());
  total.Print("TestBinsAssimilate");
}

TEST(StatsTest, TestBinsAssimilateU32) { TestBinsAssimilate<uint32_t>(); }
TEST(StatsTest, TestBinsAssimilateU64) { TestBinsAssimilate<uint64_t>(); }

TEST(StatsTest, TestBinsIncrementByU64) {
  Bins<2, uint64_t> bins;
  const uint64_t count = (uint64_t{1} << 32) + 3;
  bins.IncrementBy(1, count);
  HWY_ASSERT_EQ(count, bins.Bin(1));
  HWY_ASSERT_EQ(uint64_t{0}, bins.Bin(0));
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
