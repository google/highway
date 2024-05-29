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

#include <memory>  // unique_ptr

namespace hwy {
namespace platform {

#pragma pack(push, 1)
class PerfCounters {
 public:
  static constexpr size_t Num() { return 11; }

  template <class Visitor>
  void ForEach(PerfCounters* other, const Visitor& visitor) {
    visitor(this->ref_cycle, other->ref_cycle, "ref_cycle");
    visitor(this->instruction, other->instruction, "instruction");
    visitor(this->branch, other->branch, "branch");
    visitor(this->branch_mispred, other->branch_mispred, "branch_mispred");
    visitor(this->frontend_stall, other->frontend_stall, "frontend_stall");
    visitor(this->backend_stall, other->backend_stall, "backend_stall");
    visitor(this->l3_load, other->l3_load, "l3_load");
    visitor(this->l3_store, other->l3_store, "l3_store");
    visitor(this->l3_load_miss, other->l3_load_miss, "l3_load_miss");
    visitor(this->l3_store_miss, other->l3_store_miss, "l3_store_miss");
    // must be last, see GetCounterConfigs.
    visitor(this->page_fault, other->page_fault, "page_fault");
  }

  // Floating-point because these are extrapolated (multiplexing). We want this
  // to fit in one cache line to reduce cost in profiler.h, hence use individual
  // members with smaller types instead of an array. Ensure all values are sums,
  // not ratios, so that profiler.h can add/subtract them.
  double ref_cycle;
  double instruction;
  float branch;
  float branch_mispred;
  float frontend_stall;  // [cycles]
  float backend_stall;   // [cycles]
  float l3_load;
  float l3_store;
  float l3_load_miss;
  float l3_store_miss;
  float page_fault;
};
#pragma pack(pop)

// Holds state required for reading PerfCounters. Expensive to create.
class PMU {
 public:
  PMU();
  ~PMU();

  // Returns false if counters are unavailable, otherwise starts them.
  bool Start();

  // Returns 0.0 on error; otherwise the minimum coverage of any counter, i.e.,
  // the fraction of the time between Start and Stop that the counter was
  // active, and overwrites `counters` with the extrapolated values since Start.
  double Stop(PerfCounters& counters);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace platform
}  // namespace hwy

#endif  // HIGHWAY_HWY_PERF_COUNTERS_H_
