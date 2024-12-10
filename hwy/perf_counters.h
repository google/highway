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

#ifndef HIGHWAY_HWY_PERF_COUNTERS_H_
#define HIGHWAY_HWY_PERF_COUNTERS_H_

// Reads OS/CPU performance counters.

#include <stddef.h>
#include <stdio.h>

#include "hwy/base.h"  // HWY_ABORT
#include "hwy/bit_set.h"

namespace hwy {
namespace platform {

#pragma pack(push, 1)
class PerfCounters {
 public:
  static constexpr size_t kCapacity = 16;  // = HWY_ALIGNMENT / sizeof(double)

  // Bit indices used to identify counters. The ordering is arbitrary. Some of
  // these counters may be 'removed' in the sense of not being visited by
  // `Foreach`, but their enumerators will remain. New counters may be appended.
  enum Counter {
    kRefCycles = 0,
    kInstructions,
    kBranches,
    kBranchMispredicts,
    kBusCycles,
    kCacheRefs,
    kCacheMisses,
    kL3Loads,
    kL3Stores,
    kPageFaults,  // SW
    kMigrations   // SW
  };  // BitSet64 requires these values to be less than 64.

  // Strings for user-facing messages, not used in the implementation.
  static inline const char* Name(Counter c) {
    switch (c) {
      case kRefCycles:
        return "ref_cycles";
      case kInstructions:
        return "instructions";
      case kBranches:
        return "branches";
      case kBranchMispredicts:
        return "branch_mispredicts";
      case kBusCycles:
        return "bus_cycles";
      case kCacheRefs:
        return "cache_refs";
      case kCacheMisses:
        return "cache_misses";
      case kL3Loads:
        return "l3_load";
      case kL3Stores:
        return "l3_store";
      case kPageFaults:
        return "page_fault";
      case kMigrations:
        return "migration";
      default:
        HWY_ABORT("Bug: unknown counter %d", c);
    }
  }

  // Returns false if counters are unavailable. This is separate from
  // `StartAll` to reduce the overhead of stopping/starting counters, and must
  // be called before `StartAll`. The PMU is accessed via monostate pattern
  // because there is only one, and libraries may want to use `PerfCounters`
  // without passing around a pointer.
  static bool Init();

  // Returns false if counters are unavailable, otherwise starts them. Note that
  // they default to stopped. Without calling this, `Read` returns zeros.
  static bool StartAll();

  // Stops and zeros all counters. This is not necessary if periodically
  // calling `Read` and subtracting the previous counter values, but can
  // increase precision because floating-point has more precision near zero.
  static void StopAllAndReset();

  // Returns negative on error; otherwise the minimum coverage of any counter
  // (the fraction of the time that the counter was actually running vs. the
  // total time between `StartAll` and now or the last `StopAllAndReset`), and
  // overwrites `values_` with the extrapolated values.
  double Read();

  // Calls visitor with f64 value and `Counter` for each valid counter, in
  // increasing numerical order.
  template <class Visitor>
  void Foreach(const Visitor& visitor) {
    Valid().Foreach([&](size_t bit_idx) {
      const Counter c = static_cast<Counter>(bit_idx);
      visitor(values_[IndexForCounter(c)], c);
    });
  }

  void Print() {
    Foreach([](double val, Counter c) {
      fprintf(stderr, "%-20s: %.3E\n", Name(c), val);
    });
  }

 private:
  // Monostate so that `PerfCounters` only stores the values (flyweight
  // pattern). Counter values are used as the bit indices.
  static BitSet64& Valid();
  // Index within `values_` for a given counter.
  static size_t IndexForCounter(Counter c);

  // Floating-point because these are extrapolated (multiplexing). It would be
  // nice for this to fit in one cache line to reduce the cost of reading
  // counters in profiler.h, but some of the values are too large for float and
  // we want more than 8 counters. Ensure all values are sums, not ratios, so
  // that profiler.h can add/subtract them. These are contiguous in memory, in
  // the order that counters were initialized.
  double values_[kCapacity];
};
#pragma pack(pop)

}  // namespace platform
}  // namespace hwy

#endif  // HIGHWAY_HWY_PERF_COUNTERS_H_
