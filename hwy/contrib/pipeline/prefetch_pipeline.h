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

#ifndef HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_H_
#define HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_H_

#include <stddef.h>

#include "hwy/base.h"
#include "hwy/cache_control.h"
#include "hwy/contrib/pipeline/prefetch_args.h"
#include "hwy/timer.h"

namespace hwy {

// ---------------------------------------------------------------------------
// PrefetchStrategy
// ---------------------------------------------------------------------------
// Enumerates the structural looping algorithm that the pipeline will compile
// down into. Used to decouple hardware limits from loop execution mechanics.
enum class PrefetchStrategy {
  kNoPrefetch,
  kDeepLookaheadOnly,
  kShallowLookaheadOnly,
  kMiniBatchDeep,
  kMiniBatchShallow,
  kDualTier
};

// ---------------------------------------------------------------------------
// PrefetchLimits
// ---------------------------------------------------------------------------
// Provides hardware limitation policies for the pipelining loop to optimize
// execution. A default policy is provided via `DefaultPrefetchLimits`, but
// users can construct their own via template specializations.
struct DefaultPrefetchLimits {
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

  // The specific execution strategy the pipeline will adopt.
  static constexpr PrefetchStrategy kStrategy = PrefetchStrategy::kDualTier;
};

// ---------------------------------------------------------------------------
// PrefetchCachelines
// ---------------------------------------------------------------------------
// A lightweight, stack-allocated, fixed-capacity container for collecting
// discrete memory addresses to be prefetched.
// By strictly accumulating memory pointers individually, it allows precise
// control over the Line Fill Buffer (LFB) utilization in the pipelining loop.
template <size_t kMaxCachelinesPerIter =
              DefaultPrefetchLimits::kMaxCachelinesPerIter>
struct PrefetchCachelines {
  // Array of explicit memory addresses to prefetch.
  const void* ptrs[kMaxCachelinesPerIter];

  // The number of valid pointers currently registered in the array.
  size_t count = 0;

  // Registers a discrete memory address to be prefetched.
  // The user should supply the base pointer of the data they intend to access.
  HWY_INLINE void Add(const void* ptr) {
    HWY_DASSERT(count < kMaxCachelinesPerIter);
    ptrs[count] = ptr;
    ++count;
  }
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
// ---------------------------------------------------------------------------
// Cachelines Provider Concept
// ---------------------------------------------------------------------------
// To use PrefetchPipelineLoop, the user must provide a callable that adheres
// to the following signature:
//
//   template <size_t kMaxCachelinesPerIter>
//   void operator()(size_t i, PrefetchCachelines<kMaxCachelinesPerIter>&
//   cachelines) const;
//
// Parameters:
//  - i:                The current sequence index evaluated by the loop.
//                      This is guaranteed to be in the range `[start, end)`.
//  - cachelines:       Output collection. Call `cachelines.Add(ptr)` to
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
template <typename T, size_t MaxCachelines>
concept PrefetchPipelineCachelineProvider =
    requires(const T& provider, size_t i,
             PrefetchCachelines<MaxCachelines>& cachelines) {
      { provider(i, cachelines) };
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

// PrefetchPipelineLoop
// ---------------------------------------------------------------------------
// Design Philosophy:
// While this helper aims to deeply accelerate predictable memory-bound loops,
// its core tenet is "do no harm in the worst case". By carefully staging
// memory into L3 before migrating it to L1, and rigorously capping the active
// footprint against hardware Line Fill Buffer (LFB) limits, it ensures that
// memory bandwidth is maximized without accidentally thrashing the cache or
// stalling the processor pipelines.
//
// Evaluates a pipelined loop over the range [start, end) with two stages of
// rolling prefetch lookahead: a deep prefetch (e.g. L3) and a shallow prefetch
// (e.g. L1). This correctly stages data transitions from main memory -> L3 ->
// L1 to maximize memory bandwidth and keep CPU Line Fill Buffers from stalling.
//
//   Policy: A struct dictating cache limits and loop constants. Defaults to
//           `DefaultPrefetchLimits`.
//   CachelinesProvider: A callable type matching the signature documented
//                       in `PrefetchPipelineCachelineProvider` above. It
//                       resolves cacheline pointers for a given index `i`.
//   TaskFn: A callable type matching the signature documented in
//           `PrefetchPipelineTask` above. It represents the core evaluation
//           logic for index `i`.
//   args: A `PrefetchArgs` configuration that controls the deep (L3) and
//         shallow (L1) lookahead pipeline distances. Passing a reasonable
//         value is crucial for optimal hardware performance. Ideally, use an
//         auto-tuned configuration or an explicit architecture preset.
template <typename Limits = DefaultPrefetchLimits, typename CachelinesProvider,
          typename TaskFn,
          // Allow overriding the prefetch functions for testing.
          void (*DeepPrefetchFn)(const void*) = DeepPrefetch,
          void (*ShallowPrefetchFn)(const void*) = ShallowPrefetch>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
  requires PrefetchPipelineTask<TaskFn> &&
           PrefetchPipelineCachelineProvider<CachelinesProvider,
                                             Limits::kMaxCachelinesPerIter>
#endif
HWY_INLINE void PrefetchPipelineLoop(size_t start, size_t end,
                                     const CachelinesProvider& get_cachelines,
                                     const TaskFn& task,
                                     const PrefetchArgs& args) {
  const uint64_t t0 =
      args.metric_collector_cb != nullptr ? hwy::timer::Start() : 0;
  // Gracefully degrade inverted configurations if AutoTune generates them.
  size_t actual_shallow = args.shallow_lookahead;
  size_t actual_deep = args.deep_lookahead;
  HWY_DASSERT(actual_deep == 0 || actual_deep > actual_shallow);
  if (actual_shallow >= actual_deep) {
    actual_deep = 0;
  }

  const size_t initial_shallow_prefetch_end =
      HWY_MIN(start + actual_shallow, end);
  const size_t initial_deep_prefetch_end = HWY_MIN(start + actual_deep, end);

  // Reusable cache injection loops:
  auto execute_deep_prefetch =
      [&](const PrefetchCachelines<Limits::kMaxCachelinesPerIter>& cachelines)
          HWY_ATTR_CACHE {
            if (actual_deep == 0) return;
            for (size_t r = 0; r < cachelines.count; ++r) {
              DeepPrefetchFn(cachelines.ptrs[r]);
            }
          };
  auto execute_shallow_prefetch =
      [&](const PrefetchCachelines<Limits::kMaxCachelinesPerIter>& cachelines)
          HWY_ATTR_CACHE {
#if HWY_IS_DEBUG_BUILD
            // Safeguard: The active L1 hardware queue footprint should not
            // exceed the physical Miss Status Holding Registers (MSHRs).
            // (e.g. 10-12 Line Fill Buffers on legacy Intel architectures).
            // Exceeding this generates catastrophic silent memory stalls.
            HWY_DASSERT(cachelines.count * actual_shallow <= Limits::kNumMSHRs);
#endif
            for (size_t r = 0; r < cachelines.count; ++r) {
              ShallowPrefetchFn(cachelines.ptrs[r]);
            }
          };

  // Hoisted state variables to avoid tight-loop reallocation thrashing:
  PrefetchCachelines<Limits::kMaxCachelinesPerIter> cachelines;

  // ------------------------------------------------------------------------
  // Branchless loop execution via compile-time strategy
  // ------------------------------------------------------------------------

  // NOTE: Use a strict `if constexpr ... else if constexpr ... else` chain to
  // discard not matched branches at compile time.
  if constexpr (Limits::kStrategy == PrefetchStrategy::kNoPrefetch) {
    for (size_t i = start; i < end; ++i) {
      task(i);
    }
  } else if constexpr (Limits::kStrategy ==
                       PrefetchStrategy::kShallowLookaheadOnly) {
    const size_t limit_shallow =
        (end > start + actual_shallow) ? end - actual_shallow : start;
    // Startup prefetching
    for (size_t i = start; i < initial_shallow_prefetch_end; ++i) {
      cachelines.count = 0;
      get_cachelines(i, cachelines);
      execute_shallow_prefetch(cachelines);
    }
    // Main sliding loop
    for (size_t i = start; i < limit_shallow; ++i) {
      cachelines.count = 0;
      get_cachelines(i + actual_shallow, cachelines);
      execute_shallow_prefetch(cachelines);
      task(i);
    }
    // Task drain
    for (size_t i = limit_shallow; i < end; ++i) {
      task(i);
    }
  } else if constexpr (Limits::kStrategy ==
                       PrefetchStrategy::kDeepLookaheadOnly) {
    const size_t limit_deep =
        (end > start + actual_deep) ? end - actual_deep : start;
    // Startup prefetching
    for (size_t i = start; i < initial_deep_prefetch_end; ++i) {
      cachelines.count = 0;
      get_cachelines(i, cachelines);
      execute_deep_prefetch(cachelines);
    }
    // Main sliding loop
    for (size_t i = start; i < limit_deep; ++i) {
      cachelines.count = 0;
      get_cachelines(i + actual_deep, cachelines);
      execute_deep_prefetch(cachelines);
      task(i);
    }
    // Task drain
    for (size_t i = limit_deep; i < end; ++i) {
      task(i);
    }
  } else if constexpr (Limits::kStrategy == PrefetchStrategy::kMiniBatchDeep ||
                       Limits::kStrategy ==
                           PrefetchStrategy::kMiniBatchShallow) {
    const size_t batch_size =
        (Limits::kStrategy == PrefetchStrategy::kMiniBatchDeep)
            ? actual_deep
            : actual_shallow;

    for (size_t b = start; b < end; b += batch_size) {
      const size_t b_end = HWY_MIN(b + batch_size, end);
      for (size_t p = b; p < b_end; ++p) {
        cachelines.count = 0;
        get_cachelines(p, cachelines);
        if constexpr (Limits::kStrategy == PrefetchStrategy::kMiniBatchDeep) {
          execute_deep_prefetch(cachelines);
        } else {
          execute_shallow_prefetch(cachelines);
        }
      }
      for (size_t p = b; p < b_end; ++p) {
        task(p);
      }
    }
  } else {
    // Fallback: Default Staggered Pipeline (PrefetchStrategy::kDualTier)
    // A meticulously unrolled Dual-Tier pipeline is functionally necessary here
    // to stage data into L3 before pulling it into L1, preventing LFB stalls
    // on Intel, while providing optimal micro-pipelining on Zen/Maple.

    // Phase 1: Overlapping L1 (shallow) and L3 (deep) startup pipeline horizons
    for (size_t i = start; i < initial_shallow_prefetch_end; ++i) {
      cachelines.count = 0;
      get_cachelines(i, cachelines);
      execute_deep_prefetch(cachelines);
      execute_shallow_prefetch(cachelines);
    }

    // Phase 2: Outstanding L3 (deep) horizons which haven't entered L1 window
    for (size_t i = initial_shallow_prefetch_end; i < initial_deep_prefetch_end;
         ++i) {
      cachelines.count = 0;
      get_cachelines(i, cachelines);
      execute_deep_prefetch(cachelines);
    }

    // Phase 3: Main execution.
    // Instead of a single loop with bounds-checking `if` statements, we split
    // it into three branchless phases. This avoids branch mispredictions in the
    // hot loop and prevents the user provided `get_prefetch_cachelines` from
    // ever receiving an out-of-bounds index (requiring them to defensively
    // handle it).

    const size_t limit_deep = (actual_deep == 0 || start + actual_deep >= end)
                                  ? start
                                  : end - actual_deep;
    const size_t limit_shallow =
        (actual_shallow == 0 || start + actual_shallow >= end)
            ? start
            : end - actual_shallow;

    // 3a: Both deep and shallow prefetches are within bounds.
    for (size_t i = start; i < limit_deep; ++i) {
      cachelines.count = 0;
      get_cachelines(i + actual_deep, cachelines);
      execute_deep_prefetch(cachelines);

      if (actual_shallow > 0) {
        cachelines.count = 0;
        get_cachelines(i + actual_shallow, cachelines);
        execute_shallow_prefetch(cachelines);
      }

      task(i);
    }

    // 3b: Only shallow prefetch is within bounds.
    for (size_t i = limit_deep; i < limit_shallow; ++i) {
      cachelines.count = 0;
      get_cachelines(i + actual_shallow, cachelines);
      execute_shallow_prefetch(cachelines);

      task(i);
    }

    // 3c: No prefetches are within bounds, just finish the tasks.
    const size_t task_drain_start = HWY_MAX(limit_deep, limit_shallow);
    for (size_t i = task_drain_start; i < end; ++i) {
      task(i);
    }
  }

  if (HWY_UNLIKELY(args.metric_collector_cb != nullptr)) {
    args.metric_collector_cb(args.user_data, hwy::timer::Stop() - t0);
  }
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_H_
