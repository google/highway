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

#include <cstddef>

#include "hwy/base.h"
#include "hwy/cache_control.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"

namespace hwy {
namespace pipeline {
namespace low_level {

// ===========================================================================
// WARNING: Low-Level Execution API
// ===========================================================================
// Do NOT call this function directly in production code! Bypassing the tuning
// framework prevents fleet-wide continuous profiling and telemetry collection.
// Prefer `hwy::PrefetchPipelineLoop` in `prefetch_pipeline.h`.
//
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
//           `DefaultPrefetchPolicy`.
//   CachelinesProvider: A callable type matching the signature documented
//                       in `PrefetchPipelineCachelineProvider` concept. It
//                       resolves cacheline pointers for a given index `i`.
//   TaskFn: A callable type matching the signature documented in
//           `PrefetchPipelineTask` concept. It represents the core evaluation
//           logic for index `i`.
//   args: A `PrefetchArgs` configuration that controls the deep (L3) and
//         shallow (L1) lookahead pipeline distances. Passing a reasonable
template <typename Policy = DefaultPrefetchPolicy, typename CachelinesProvider,
          typename TaskFn,
          // Allow overriding the prefetch functions for testing.
          void (*DeepPrefetchFn)(const void*) = PrefetchForFutureUse,
          void (*ShallowPrefetchFn)(const void*) = PrefetchForImmediateuse>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
  requires low_level::IsPrefetchPolicy<Policy> &&
           PrefetchPipelineCachelineProvider<CachelinesProvider, Policy> &&
           PrefetchPipelineTask<TaskFn>
#endif
inline void PrefetchPipelineLoop(size_t start, size_t end,
                                 const CachelinesProvider& populate_cachelines,
                                 const TaskFn& task, const PrefetchArgs& args) {
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
  auto execute_deep_prefetch = [&](const CachelineBundle<Policy>& bundle)
                                   HWY_ATTR_CACHE {
                                     if (actual_deep == 0) return;
                                     for (size_t r = 0; r < bundle.count; ++r) {
                                       DeepPrefetchFn(bundle.ptrs[r]);
                                     }
                                   };
  auto execute_shallow_prefetch =
      [&](const CachelineBundle<Policy>& bundle) HWY_ATTR_CACHE {
#if HWY_IS_DEBUG_BUILD
        // Safeguard: The active L1 hardware queue footprint should not
        // exceed the physical Miss Status Holding Registers (MSHRs).
        // (e.g. 10-12 Line Fill Buffers on legacy Intel architectures).
        // Exceeding this generates catastrophic silent memory stalls.
        HWY_DASSERT(bundle.count * actual_shallow <= Policy::kNumMSHRs);
#endif
        for (size_t r = 0; r < bundle.count; ++r) {
          ShallowPrefetchFn(bundle.ptrs[r]);
        }
      };

  // Hoisted state variables to avoid tight-loop reallocation thrashing:
  CachelineBundle<Policy> bundle;

  // ------------------------------------------------------------------------
  // Branchless loop execution via compile-time strategy
  // ------------------------------------------------------------------------

  // NOTE: Use a strict `if constexpr ... else if constexpr ... else` chain to
  // discard not matched branches at compile time.
  if constexpr (Policy::kStrategy == PrefetchStrategy::kNoPrefetch) {
    for (size_t i = start; i < end; ++i) {
      task(i);
    }
  } else if constexpr (Policy::kStrategy ==
                       PrefetchStrategy::kShallowLookaheadOnly) {
    const size_t limit_shallow =
        (end > start + actual_shallow) ? end - actual_shallow : start;
    // Startup prefetching
    for (size_t i = start; i < initial_shallow_prefetch_end; ++i) {
      bundle.count = 0;
      populate_cachelines(i, bundle);
      execute_shallow_prefetch(bundle);
    }
    // Main sliding loop
    for (size_t i = start; i < limit_shallow; ++i) {
      bundle.count = 0;
      populate_cachelines(i + actual_shallow, bundle);
      execute_shallow_prefetch(bundle);
      task(i);
    }
    // Task drain
    for (size_t i = limit_shallow; i < end; ++i) {
      task(i);
    }
  } else if constexpr (Policy::kStrategy ==
                       PrefetchStrategy::kDeepLookaheadOnly) {
    const size_t limit_deep =
        (end > start + actual_deep) ? end - actual_deep : start;
    // Startup prefetching
    for (size_t i = start; i < initial_deep_prefetch_end; ++i) {
      bundle.count = 0;
      populate_cachelines(i, bundle);
      execute_deep_prefetch(bundle);
    }
    // Main sliding loop
    for (size_t i = start; i < limit_deep; ++i) {
      bundle.count = 0;
      populate_cachelines(i + actual_deep, bundle);
      execute_deep_prefetch(bundle);
      task(i);
    }
    // Task drain
    for (size_t i = limit_deep; i < end; ++i) {
      task(i);
    }
  } else if constexpr (Policy::kStrategy == PrefetchStrategy::kMiniBatchDeep ||
                       Policy::kStrategy ==
                           PrefetchStrategy::kMiniBatchShallow) {
    const size_t batch_size =
        (Policy::kStrategy == PrefetchStrategy::kMiniBatchDeep)
            ? actual_deep
            : actual_shallow;

    for (size_t b = start; b < end; b += batch_size) {
      const size_t b_end = HWY_MIN(b + batch_size, end);
      for (size_t p = b; p < b_end; ++p) {
        bundle.count = 0;
        populate_cachelines(p, bundle);
        if constexpr (Policy::kStrategy == PrefetchStrategy::kMiniBatchDeep) {
          execute_deep_prefetch(bundle);
        } else {
          execute_shallow_prefetch(bundle);
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
      bundle.count = 0;
      populate_cachelines(i, bundle);
      execute_deep_prefetch(bundle);
      execute_shallow_prefetch(bundle);
    }

    // Phase 2: Outstanding L3 (deep) horizons which haven't entered L1 window
    for (size_t i = initial_shallow_prefetch_end; i < initial_deep_prefetch_end;
         ++i) {
      bundle.count = 0;
      populate_cachelines(i, bundle);
      execute_deep_prefetch(bundle);
    }

    // Phase 3: Main execution.
    // Instead of a single loop with bounds-checking `if` statements, we split
    // it into three branchless phases. This avoids branch mispredictions in the
    // hot loop and prevents the user provided `populate_cachelines` from
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
      bundle.count = 0;
      populate_cachelines(i + actual_deep, bundle);
      execute_deep_prefetch(bundle);

      if (actual_shallow > 0) {
        bundle.count = 0;
        populate_cachelines(i + actual_shallow, bundle);
        execute_shallow_prefetch(bundle);
      }

      task(i);
    }

    // 3b: Only shallow prefetch is within bounds.
    for (size_t i = limit_deep; i < limit_shallow; ++i) {
      bundle.count = 0;
      populate_cachelines(i + actual_shallow, bundle);
      execute_shallow_prefetch(bundle);

      task(i);
    }

    // 3c: No prefetches are within bounds, just finish the tasks.
    const size_t task_drain_start = HWY_MAX(limit_deep, limit_shallow);
    for (size_t i = task_drain_start; i < end; ++i) {
      task(i);
    }
  }
}

}  // namespace low_level

// ---------------------------------------------------------------------------
// Context-Aware Loop Wrapper (Production Entry Point)
// ---------------------------------------------------------------------------
// Evaluates a pipelined loop over [start, end) using dynamic lookahead tuning.
//
// Parameters:
//   start, end: The iteration range of the workload.
//   populate_cachelines: Callable providing cacheline addresses for index `i`.
//   task: Callable executing the compute logic for index `i`.
//   file_loc, line_loc: Lexical call-site identifiers used as the primary key
//                       by the autotuning framework (AutoTune / AutoFDO) to
//                       isolate and maintain historical tuning profiles.
//
// Call-Site Stability & Autotuning Best Practices:
// By default, `file_loc` and `line_loc` automatically capture the physical
// source file and line number via `__builtin_FILE()` and `__builtin_LINE()`.
// Under normal development, this provides zero-boilerplate profile isolation.
//
// However, the stability requirements for `file_loc` and `line_loc` largely
// depend on the underlying prefetch tuner mechanism in use.
//
// If your callsite shifts frequently and relies on compile-time AutoFDO or
// requires strict zero-warmup latency stability, it is highly recommended to
// pass a fixed `file_loc` and `line_loc` (e.g., a stable virtual file string
// like "my_project_stable_loop" and a fixed dummy line number like 1000).
//
// WARNING: When providing fixed identifiers, ensure they are globally unique
// across your project. Reusing the same fixed `(file_loc, line_loc)` for
// distinct loops will cause profile aliasing, where the tuner attempts to fit
// conflicting access patterns into a single shared profile bucket.
template <PrefetchTuningHint Hint = PrefetchTuningHint::kAuto,
          typename Policy = DefaultPrefetchPolicy, typename CachelinesProvider,
          typename TaskFn>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
  requires low_level::IsPrefetchPolicy<Policy>
#endif
inline void PrefetchPipelineLoop(size_t start, size_t end,
                                 const CachelinesProvider& populate_cachelines,
                                 const TaskFn& task,
#if defined(__clang__) || defined(__GNUC__)
                                 const char* file_loc = __builtin_FILE(),
                                 int line_loc = __builtin_LINE()) {
#else
                                 const char* file_loc = nullptr,
                                 int line_loc = 0) {
#endif
  const size_t total_elements = end > start ? end - start : 0;

  auto loop_runner = [&](const PrefetchArgs& args) {
    hwy::pipeline::low_level::PrefetchPipelineLoop<Policy>(
        start, end, populate_cachelines, task, args);
  };

  low_level::DispatchTunedWorkload<Hint, Policy>(total_elements, file_loc,
                                                 line_loc, loop_runner);
}

}  // namespace pipeline
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_H_
