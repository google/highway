// Copyright 2024 Google LLC
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

#include "hwy/contrib/thread_pool/topology.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>  // std::find
#include <cstdio>
#include <map>
#include <utility>  // std::make_pair
#include <vector>

#include "hwy/base.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"
#include "hwy/tests/test_util.h"

namespace hwy {
namespace {
using HWY_NAMESPACE::AdjustedReps;

TEST(TopologyTest, TestSetSimple) {
  const size_t max = TotalLogicalProcessors() - 1;

  LogicalProcessorSet set;
  // Defaults to empty.
  HWY_ASSERT(set.Count() == 0);
  set.Foreach(
      [](size_t lp) { HWY_ABORT("Set should be empty but got %zu\n", lp); });
  HWY_ASSERT(!set.Get(0));
  HWY_ASSERT(!set.Get(max));

  // After setting, we can retrieve it.
  set.Set(max);
  HWY_ASSERT(set.Get(max));
  HWY_ASSERT(set.Count() == 1);
  set.Foreach([max](size_t lp) { HWY_ASSERT(lp == max); });

  // SetNonzeroBitsFrom64 does not clear old bits.
  const size_t min = 0;
  set.SetNonzeroBitsFrom64(1ull << min);
  HWY_ASSERT(set.Get(min));
  HWY_ASSERT(set.Get(max));
  HWY_ASSERT(set.Count() == ((min == max) ? 1 : 2));
  set.Foreach([=](size_t lp) { HWY_ASSERT(lp == min || lp == max); });

  // After clearing, it is empty again.
  set.Clear(min);
  set.Clear(max);
  HWY_ASSERT(set.Count() == 0);
  set.Foreach(
      [](size_t lp) { HWY_ABORT("Set should be empty but got %zu\n", lp); });
  HWY_ASSERT(!set.Get(0));
  HWY_ASSERT(!set.Get(max));
}

// Supports membership and random choice, for testing LogicalProcessorSet.
class SlowSet {
 public:
  // Inserting multiple times is a no-op.
  void Set(size_t lp) {
    const auto ib = idx_for_lp_.insert(std::make_pair(lp, vec_.size()));
    if (ib.second) {  // inserted
      vec_.push_back(lp);
      HWY_ASSERT(idx_for_lp_.size() == vec_.size());
    } else {
      // Already have `lp` and it can be found at the stored index.
      HWY_ASSERT(ib.first->first == lp);
      const size_t idx = ib.first->second;
      HWY_ASSERT(vec_[idx] == lp);
    }
    HWY_ASSERT(Get(lp));
  }

  bool Get(size_t lp) const {
    const auto it = idx_for_lp_.find(lp);
    if (it == idx_for_lp_.end()) {
      HWY_ASSERT(std::find(vec_.begin(), vec_.end(), lp) == vec_.end());
      return false;
    }
    HWY_ASSERT(vec_[it->second] == lp);
    return true;
  }

  void Clear(size_t lp) {
    if (!Get(lp)) return;
    const size_t idx = idx_for_lp_[lp];
    idx_for_lp_.erase(lp);
    // Move last into gap, unless it was equal to `lp`.
    const size_t last = vec_.back();
    vec_.pop_back();
    if (last == lp) {
      HWY_ASSERT(idx == vec_.size());  // was the last item
    } else {
      HWY_ASSERT(vec_[idx] == lp);
      vec_[idx] = last;
      idx_for_lp_[last] = idx;
      HWY_ASSERT(Get(last));  // can still find `last`
    }
    HWY_ASSERT(!Get(lp));
  }

  size_t Count() const {
    HWY_ASSERT(idx_for_lp_.size() == vec_.size());
    return vec_.size();
  }

  // Must not call if Count() == 0.
  size_t RandomChoice(RandomState& rng) const {
    HWY_ASSERT(Count() != 0);
    const size_t idx = hwy::Random64(&rng) % vec_.size();
    return vec_[idx];
  }

  void CheckSame(const LogicalProcessorSet& lps) {
    HWY_ASSERT(Count() == lps.Count());
    // Everything lps has, we also have.
    lps.Foreach([this](size_t lp) { HWY_ASSERT(Get(lp)); });
    // Everything we have, lps also has.
    std::for_each(vec_.begin(), vec_.end(),
                  [&lps](size_t lp) { HWY_ASSERT(lps.Get(lp)); });
  }

 private:
  std::vector<size_t> vec_;
  std::map<size_t, size_t> idx_for_lp_;
};

void TestSetRandom(uint64_t grow_prob) {
  const size_t total = TotalLogicalProcessors();
  RandomState rng;

  // Multiple independent random tests:
  for (size_t rep = 0; rep < AdjustedReps(100); ++rep) {
    LogicalProcessorSet lps;
    SlowSet set;
    // Mutate sets via random walk and ensure they are the same afterwards.
    for (size_t i = 0; i < 200; ++i) {
      const uint64_t bits = (Random64(&rng) >> 10) & 0x3FF;
      if (bits > 980 && set.Count() != 0) {
        // Small chance of reinsertion: already present, unchanged after.
        const size_t lp = set.RandomChoice(rng);
        const size_t count = lps.Count();
        HWY_ASSERT(lps.Get(lp));
        set.Set(lp);
        lps.Set(lp);
        HWY_ASSERT(lps.Get(lp));
        HWY_ASSERT(count == lps.Count());
      } else if (bits < grow_prob) {
        // Set random value; no harm if already set.
        const size_t lp = Random64(&rng) % total;
        set.Set(lp);
        lps.Set(lp);
        HWY_ASSERT(lps.Get(lp));
      } else if (set.Count() != 0) {
        // Remove existing item.
        const size_t lp = set.RandomChoice(rng);
        const size_t count = lps.Count();
        HWY_ASSERT(lps.Get(lp));
        set.Clear(lp);
        lps.Clear(lp);
        HWY_ASSERT(!lps.Get(lp));
        HWY_ASSERT(count == lps.Count() + 1);
      }
    }
    set.CheckSame(lps);
  }
}

// Lower probability of growth so that the set is often nearly empty.
TEST(TopologyTest, TestSetRandomShrink) { TestSetRandom(400); }
TEST(TopologyTest, TestSetRandomGrow) { TestSetRandom(600); }

TEST(TopologyTest, TestNum) {
  const size_t total = TotalLogicalProcessors();
  fprintf(stderr, "TotalLogical %zu\n", total);

  LogicalProcessorSet lps;
  if (GetThreadAffinity(lps)) {
    fprintf(stderr, "Active %zu\n", lps.Count());
    HWY_ASSERT(lps.Count() <= total);
  }
}

TEST(TopologyTest, TestTopology) {
  Topology topology;
  if (topology.packages.empty()) return;

  HWY_ASSERT(!topology.lps.empty());

  size_t lps_by_cluster = 0;
  size_t lps_by_core = 0;
  for (size_t p = 0; p < topology.packages.size(); ++p) {
    const Topology::Package& pkg = topology.packages[p];
    HWY_ASSERT(!pkg.clusters.empty());
    HWY_ASSERT(!pkg.cores.empty());
    HWY_ASSERT(pkg.clusters.size() <= pkg.cores.size());

    for (const Topology::Cluster& c : pkg.clusters) {
      lps_by_cluster += c.lps.Count();
    }
    for (const Topology::Core& c : pkg.cores) {
      lps_by_core += c.lps.Count();
    }
  }
  HWY_ASSERT(lps_by_cluster == topology.lps.size());
  HWY_ASSERT(lps_by_core == topology.lps.size());
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
