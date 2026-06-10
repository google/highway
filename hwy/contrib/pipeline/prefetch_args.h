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

#ifndef HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_ARGS_H_
#define HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_ARGS_H_

#include <stddef.h>
#include <stdint.h>

namespace hwy {

struct PrefetchArgs {
  // The iteration distance (in loop iterations) to look ahead for the deep L3
  // prefetch. If 0, no deep prefetch will be issued.
  size_t deep_lookahead = 32;

  // The iteration distance (in loop iterations) to look ahead for the shallow
  // L1 prefetch. If 0, no shallow prefetch will be issued.
  size_t shallow_lookahead = 4;

  // -------------------------------------------------------------------------
  // Telemetry Conduit
  // -------------------------------------------------------------------------
  // A generic callback executed upon pipeline completion to report performance
  // metrics (typically elapsed time). Its signature avoids std::function to
  // maintain zero-overhead C-style linkage and ensure the struct remains
  // trivially copyable without heap allocations.
  //   user_data: Custom context pointer returned to the callback.
  //   elapsed_ticks: Raw cycle ticks taken to evaluate the pipeline loop.
  void (*metric_collector_cb)(void* user_data,
                              uint64_t elapsed_ticks) = nullptr;
  void* user_data = nullptr;

  // -------------------------------------------------------------------------
  // Safe Default Factories
  // -------------------------------------------------------------------------
  // Tuning memory prefetching is notoriously difficult because lookahead bounds
  // change dramatically depending on the spatial distribution of the workload.

  // Random Access / Scatter-Gather (e.g. Hash Table Probing, Graph Walks)
  //
  // Random array accesses constantly suffer TLB (Translation Lookaside Buffer)
  // misses, resulting in massive Page Walk delays. To absorb these colossal
  // ~300-cycle stalls natively inside the L3 queue, the deep lookahead must
  // aggressively stretch out by large margins (e.g. 32-48 iterations).
  static constexpr PrefetchArgs DefaultRandom() {
#if HWY_ARCH_ARM_A64
    return PrefetchArgs{.deep_lookahead = 64, .shallow_lookahead = 8};
#else
    return PrefetchArgs{.deep_lookahead = 32, .shallow_lookahead = 4};
#endif
  }

  // Sequential Scans / Linear Memory (e.g. Matrix Vector, Filter Scans)
  //
  // Linear accesses benefit intimately from native CPU stream-trackers (which
  // already mask bulk DRAM latency). Here, a heavy L3 lookahead is
  // counter-productive; it merely crowds the queue. Instead, we tighten the
  // lookaheads down to safely bridge the narrower L3 -> L1 latency gap
  // (~40 cycles) without overflowing LFBs during heavy SIMD evaluation.
  static constexpr PrefetchArgs DefaultSequential() {
#if HWY_ARCH_ARM_A64
    return PrefetchArgs{.deep_lookahead = 32, .shallow_lookahead = 4};
#else
    return PrefetchArgs{.deep_lookahead = 8, .shallow_lookahead = 2};
#endif
  }
};

// ---------------------------------------------------------------------------
// 2D-Tiled Prefetch Policy
// ---------------------------------------------------------------------------
// Extends the base PrefetchPolicy to include standard 2D tiling constants.
struct Prefetch2DArgs {
  PrefetchArgs prefetch;
  size_t outer_block = 128;
  size_t inner_block = 256;

  // -------------------------------------------------------------------------
  // Safe Default Factories
  // -------------------------------------------------------------------------

  static constexpr Prefetch2DArgs DefaultRandom() {
    Prefetch2DArgs args;
    args.prefetch = PrefetchArgs::DefaultRandom();
    args.outer_block = 128;
    args.inner_block = 256;
    return args;
  }

  static constexpr Prefetch2DArgs DefaultSequential() {
    Prefetch2DArgs args;
    args.prefetch = PrefetchArgs::DefaultSequential();
    args.outer_block = 256;
    args.inner_block = 512;
    return args;
  }
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_ARGS_H_
