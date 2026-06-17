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

#ifndef HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_TYPES_H_
#define HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_TYPES_H_

#include <stddef.h>

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#include <concepts>  // IWYU pragma: keep
#include <type_traits>
#endif

#include <stdint.h>

#include "hwy/base.h"
namespace hwy {
namespace pipeline {

// ---------------------------------------------------------------------------
// PrefetchStrategy
// ---------------------------------------------------------------------------
// Enumerates the structural looping algorithm that the pipeline will compile
// down into. Used to decouple hardware limits from loop execution mechanics.
enum class PrefetchStrategy {
  // Bypasses all explicit software prefetch instructions entirely. Used as a
  // control baseline during A/B testing or when hardware stream trackers are
  // already fully saturating memory bandwidth.
  kNoPrefetch,

  // Issues prefetch instructions exclusively for the deep lookahead distance
  // (targeting L3 cache or DRAM). Ideal for pointer-chasing or irregular
  // scatter-gather workloads where hiding massive DRAM latency is the primary
  // bottleneck.
  kDeepLookaheadOnly,

  // Issues prefetch instructions exclusively for the shallow lookahead distance
  // (targeting L1/L2 cache). Ideal for linear memory scans where L3 is warmed
  // up by hardware prefetchers, but explicit hints are needed to bridge the
  // final L3->L1 latency gap without thrashing Line Fill Buffers (LFBs).
  kShallowLookaheadOnly,

  // Aggregates deep prefetch instructions into discrete mini-batches executed
  // before compute phases. Decouples prefetch dispatch from inner loop compute
  // unrolling, preventing instruction-cache bloat and front-end bottlenecks.
  kMiniBatchDeep,

  // Aggregates shallow prefetch instructions into discrete mini-batches.
  // Ensures L1 cachelines are fetched in tight bursts right before SIMD vector
  // execution, maximizing instruction-level parallelism (ILP).
  kMiniBatchShallow,

  // The flagship, production-grade execution strategy. Simultaneously
  // orchestrates both deep (L3/DRAM) and shallow (L1/L2) prefetching in a
  // coordinated, two-tier pipeline. Keeps the memory hierarchy perfectly
  // saturated across all latency boundaries.
  kDualTier
};

// Identifier for a prefetch call site. It is recommended to be unique for
// different call sites, but it is functionally safe even if collisions occur.
using CallsiteId = uint64_t;

// ---------------------------------------------------------------------------
// PrefetchArgs
// ---------------------------------------------------------------------------
// Bit-field struct encapsulating the two prefetch lookahead parameters for a
// single prefetch call site. Total size is exactly 2 bytes (16 bits).
struct PrefetchArgs {
  // The iteration distance (in loop iterations) to look ahead for the deep L3
  // prefetch (max 511). If 0, no deep prefetch will be issued.
  uint16_t deep_lookahead : 9;

  // The iteration distance (in loop iterations) to look ahead for the shallow
  // L1 prefetch (max 127). If 0, no shallow prefetch will be issued.
  uint16_t shallow_lookahead : 7;

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
// PrefetchTuningHint
// ---------------------------------------------------------------------------
// Hint for the expected memory access pattern.
//
// These hints are primarily used to establish safe baseline configuration
// limits when the tuning framework executes without a registered telemetry
// plugin, or to artificially restrict the scope of candidate grids when
// auto-tuning is enabled.
enum class PrefetchTuningHint {
  // The workload access pattern is unknown or highly variable.
  //   - Without Tuner: Safely defaults to conservative, tight lookahead bounds.
  //   - With Tuner: Evaluates the maximum exhaustive search grid for optimal
  //     configuration.
  kAuto,

  // The workload scatters reads across wide memory distributions (e.g., hash
  // tables, graph node aggregations) resulting in TLB/Page Walk delays.
  //   - Without Tuner: Defaults to aggressive, deep lookahead bounds.
  //   - With Tuner: Artificially restricts candidate grids to deep lookahead
  //     bounds only to optimize profiling.
  kRandom,

  // The workload linearly scans contiguous memory blocks (e.g., Matrix-Vector
  // multiplication).
  //   - Without Tuner: Safely defaults to conservative, tight lookahead bounds.
  //   - With Tuner: Artificially restricts candidate grids to shallow lookahead
  //     bounds only to optimize profiling.
  kSequential
};

// ---------------------------------------------------------------------------
// PrefetchPolicy
// ---------------------------------------------------------------------------
struct DefaultPrefetchPolicy {
  // The specific execution strategy the pipeline will adopt.
  //
  // Note this is mainly used for benchmarking and testing purposes.
  // For the production code, just use `kDualTier` and let the tuning framework
  // optimize the lookahead values, e.g., when deep lookahead is zero, it will
  // automatically fall back to shallow-only.
  static constexpr PrefetchStrategy kStrategy = PrefetchStrategy::kDualTier;

  // Threshold below which the workload short-circuits dynamic tuning queries
  // overhead, and run the loop with the default prefetch strategy and lookahead
  // values per the available hints.
  //
  // Note that the optimal value may vary, depending on the specific hardware,
  // workload, and the tuner capabilities.
  static constexpr size_t kUltraTinyThreshold = 32;

  // The maximum number of explicit cachelines that should be prefetched per
  // iteration (i.e., cachelines containing the data to be used by the upcoming
  // compute task).
  static constexpr size_t kMaxCachelinesPerIter = 4;

  // -------------------------------------------------------------------------
  // Hardware Architecture Matrix
  // -------------------------------------------------------------------------
  // Resolved automatically within the struct using HWY_ARCH_*

#if HWY_ARCH_ARM || HWY_ARCH_ARM_A64
  // ARM platforms vary by their natures, we use 24 here to be conservative.
  static constexpr size_t kNumMSHRs = 24;

#elif HWY_ARCH_X86
  // On x86, we cannot differentiate Intel vs AMD reliably at compile time.
  // Because hitting the edge of the Intel LFB pool (10-12) causes a hard CPU
  // stall, we MUST gracefully pick the most restrictive denominator (12) to
  // ensure safety. AMD Zen architectures uniquely "absorb" the excess prefetch
  // instructions into their 124 MABs, suffering zero penalty from this wrapper.
  static constexpr size_t kNumMSHRs = 12;

#else
  // Safe, generic bounds.
  static constexpr size_t kNumMSHRs = 12;
#endif
};

// ---------------------------------------------------------------------------
// CachelineBundle
// ---------------------------------------------------------------------------
// A lightweight, stack-allocated, fixed-capacity container for collecting
// memory addresses to be prefetched. Provides bulletproof bounds checking in
// production while maintaining strict debug capacity assertions.
template <typename Policy = DefaultPrefetchPolicy>
struct CachelineBundle {
  // Array of explicit memory addresses to prefetch.
  const void* ptrs[Policy::kMaxCachelinesPerIter];

  // The number of valid pointers currently registered in the array.
  size_t count = 0;

  // Registers a discrete, individual memory address to be prefetched.
  // USE CASE: Non-contiguous, scatter-gather accesses (e.g., hash tables, graph
  // walks).
  // Use `AddContiguousRange` for linear memory buffers.
  inline void Add(const void* ptr) {
    HWY_DASSERT(count < Policy::kMaxCachelinesPerIter);
    if (HWY_LIKELY(count < Policy::kMaxCachelinesPerIter)) {
      ptrs[count] = ptr;
      ++count;
    }
  }

  // Registers a dense, contiguous memory range to be prefetched.
  // USE CASE: Linear memory buffers (e.g., embedding vectors, matrix tiles,
  // image spans). Automatically calculates 64-byte cacheline strides and
  // short-circuits if full.
  inline void AddContiguousRange(const void* base_ptr, size_t num_bytes) {
    const char* base = static_cast<const char*>(base_ptr);
    for (size_t offset = 0; offset < num_bytes; offset += 64) {
      if (HWY_LIKELY(count < Policy::kMaxCachelinesPerIter)) {
        Add(base + offset);
      } else {
        break;  // Container is full; short-circuit remaining iterations
      }
    }
  }
};

namespace low_level {

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
// ---------------------------------------------------------------------------
// PrefetchPolicy Concept
// ---------------------------------------------------------------------------
// Mandates that any user-provided Policy struct defines the required hardware
// and execution policy constants.
template <typename T>
concept IsPrefetchPolicy = requires {
  requires ::std::is_integral_v<decltype(T::kMaxCachelinesPerIter)>;
  requires ::std::is_integral_v<decltype(T::kNumMSHRs)>;
  requires ::std::is_same_v<::std::remove_const_t<decltype(T::kStrategy)>,
                            PrefetchStrategy>;
  requires ::std::is_integral_v<decltype(T::kUltraTinyThreshold)>;
};

// ---------------------------------------------------------------------------
// Cachelines Provider Concept
// ---------------------------------------------------------------------------
// To use PrefetchPipelineLoop, the user must provide a callable that adheres
// to the following signature:
//
//   template <typename Policy>
//   void operator()(size_t i, CachelineBundle<Policy>&
//   bundle) const;
//
// Parameters:
//  - i:                The current sequence index evaluated by the loop.
//                      This is guaranteed to be in the range `[start, end)`.
//  - bundle:           Output collection. Call `bundle.Add(ptr)` to
//                      register cachelines. For invalid or conditional
//                      indices, simply do not add anything.
//
// Execution constraints:
//  - Depending on the architecture, prefetching optimizations, or runtime
//    auto-tuning limits (see Policy configurations below), there is no
//    hardware or programmatic guarantee how many times this function will be
//    called for a given `i` (including zero times if prefetches are stripped
//    via `constexpr`). Therefore, this callable MUST be completely pure and
//    strictly side-effect free!
//  - Missing or zero-length entries are natively skipped by the pipeline
//    without incurring branching overhead.
template <typename T, typename Policy>
concept PrefetchPipelineCachelineProvider =
    requires(const T& provider, size_t i, CachelineBundle<Policy>& bundle) {
      { provider(i, bundle) };
    };

// ---------------------------------------------------------------------------
// Pipeline Task Concept
// ---------------------------------------------------------------------------
// To use PrefetchPipelineLoop, the user must provide a callable that adheres
// to the following signature:
//
//   void operator()(size_t i) const;
//
// Parameters:
//  - i: The current sequence index being evaluated by the pipeline.
//
// Execution constraints:
//  - Guaranteed to be called exactly once for each index `i` in the range
//    `[start, end)`. Furthermore, it is guaranteed to be invoked purely
//    sequentially (i.e. `i`, `i+1`, `i+2`), preserving any cross-iteration
//    dependencies or internal accumulator state.
template <typename T>
concept PrefetchPipelineTask = requires(const T& task, size_t i) {
  { task(i) };
};
#endif

}  // namespace low_level
}  // namespace pipeline
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_TYPES_H_
