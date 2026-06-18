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

#ifndef HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_2D_H_
#define HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_2D_H_

#include <stddef.h>

#include "hwy/base.h"
#include "hwy/contrib/pipeline/prefetch_pipeline.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"

namespace hwy {
namespace pipeline {

// ---------------------------------------------------------------------------
// 2D Cache Tiling Geometry
// ---------------------------------------------------------------------------
// A dedicated POD struct specifying outer and inner loop blocking dimensions.
// Dictates cache locality boundaries independently of prefetch lookahead
// velocity.
struct Tiling2DArgs {
  // In contrast to prefetch lookaheads (which must be aggressively tuned to
  // negotiate volatile memory latency boundaries), the 2D block shapes rarely
  // require dynamic adjustment.
  //
  // `outer_block` dictates how many items in the outer loop are amortized
  // together to ensure their intermediate working state remains resident inside
  // the L1 Data Cache. For most algorithms, an outer block around ~128 items
  // consumes minimal memory (e.g., 512 bytes for 32-bit accumulators) while
  // fully amortizing outer-loop overhead.
  //
  // `inner_block` dictates how many iterations of the innermost phase are
  // processed continuously to saturate instruction-level parallelism. This is
  // usually overridden at compile-time or runtime by the specific compute
  // kernel, which intrinsically knows its own ideal unrolling bounds (e.g., to
  // perfectly fill pipeline stages with Fused Multiply-Add instructions).
  size_t outer_block = 128;
  size_t inner_block = 256;

  // Calculates the total number of discrete 1D tile-iterations to be processed
  // across the 2D geometry.
  //
  // NOTE: This is NOT the total number of 2D tiles (which would be
  // `num_outer_tiles * num_inner_tiles`). Instead, it represents
  // `outer_size * num_inner_tiles`. Because the underlying 1D prefetch pipeline
  // unrolls across `outer_size` rows for each inner tile, this value accurately
  // reflects the total number of discrete inner loop invocations, ensuring
  // correct AutoFDO fission bucket assignment and cost normalization.
  inline size_t TotalElements(size_t outer_size, size_t inner_size) const {
    const size_t num_tile_per_row =
        inner_block > 0 ? (inner_size + inner_block - 1) / inner_block : 1;
    return outer_size * num_tile_per_row;
  }
};

struct Default2DTiledPrefetchPolicy : public DefaultPrefetchPolicy {};

namespace low_level {

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
// ---------------------------------------------------------------------------
// 2D-Tiled Pipeline Callbacks Concept
// ---------------------------------------------------------------------------
// To use PrefetchPipeline2DTiledLoop, the user must provide a callbacks object
// that adheres to the following signatures:
//
//   // Called before evaluating the inner loop phases for an outer block.
//   // Useful for allocating or resetting local accumulators on the stack.
//   void OnOuterBlockStart(size_t outer_idx, size_t outer_end);
//
//   // Called before processing a new inner segment across all items in the
//   // current outer block. Useful for broadcasting data into SIMD registers.
//   void PrepareInnerBlock(size_t outer_idx, size_t outer_end,
//                          size_t inner_idx, size_t inner_end);
//
//   // Evaluates the sequence index `outer_i` and adds memory pointers to the
//   // `collector` collection to be prefetched for the given inner block.
//   template <typename Policy>
//   void PopulateCachelines(
//       size_t outer_i, size_t inner_idx, size_t inner_end,
//       CachelineBundle<Policy>& collector);
//
//   // The core loop body to execute for a given sequence index `outer_i` and
//   // inner segment.
//   void ComputeTask(size_t outer_i, size_t inner_idx, size_t inner_end);
//
//   // Called after all inner segments have been processed for the outer block.
//   // Useful for committing accumulated data.
//   void OnOuterBlockFinish(size_t outer_idx, size_t outer_end);
template <typename T, typename Policy>
concept PrefetchPipeline2DTiledCallbacks =
    requires(T& cb, size_t outer_idx, size_t outer_end, size_t inner_idx,
             size_t inner_end, CachelineBundle<Policy>& collector) {
      { cb.OnOuterBlockStart(outer_idx, outer_end) };
      { cb.PrepareInnerBlock(outer_idx, outer_end, inner_idx, inner_end) };
      { cb.PopulateCachelines(outer_idx, inner_idx, inner_end, collector) };
      { cb.ComputeTask(outer_idx, inner_idx, inner_end) };
      { cb.OnOuterBlockFinish(outer_idx, outer_end) };
    };
#endif

// ===========================================================================
// WARNING: Low-Level Execution API
// ===========================================================================
// Do NOT call this function directly in production code! Bypassing the tuning
// framework prevents fleet-wide continuous profiling and telemetry collection.
// Prefer `hwy::PrefetchPipeline2DTiledLoop` in `prefetch_pipeline_2d.h`.
//
// A generic pipeline for executing 2D block-tiled computations with
// prefetching. This splits processing into localized blocks that prevent cache
// evictions and FPU instruction bottlenecks.
//
// Template Parameters:
//   Policy: A struct extending `Default2DTiledPrefetchPolicy` dictating cache
//           limits, tiling dimensions, and loop constants.
//   Callbacks: A type matching the signature documented in
//              `PrefetchPipeline2DTiledCallbacks` above. Passed by reference to
//              allow state mutation.
//   tiling: A `Tiling2DArgs` object dictating the outer/inner block dimensions.
//   prefetch_args: A `PrefetchArgs` object dictating prefetch lookahead depths.
template <typename Policy = Default2DTiledPrefetchPolicy, typename Callbacks>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
  requires low_level::IsPrefetchPolicy<Policy> &&
           PrefetchPipeline2DTiledCallbacks<Callbacks, Policy>
#endif
inline void PrefetchPipeline2DTiledLoop(size_t outer_size, size_t inner_size,
                                        Callbacks& cb,
                                        const Tiling2DArgs& tiling,
                                        const PrefetchArgs& prefetch_args) {
  for (size_t outer_idx = 0; outer_idx < outer_size;
       outer_idx += tiling.outer_block) {
    const size_t outer_end = (outer_size < outer_idx + tiling.outer_block)
                                 ? outer_size
                                 : outer_idx + tiling.outer_block;
    cb.OnOuterBlockStart(outer_idx, outer_end);

    for (size_t inner_idx = 0; inner_idx < inner_size;
         inner_idx += tiling.inner_block) {
      const size_t inner_end = (inner_size < inner_idx + tiling.inner_block)
                                   ? inner_size
                                   : inner_idx + tiling.inner_block;

      cb.PrepareInnerBlock(outer_idx, outer_end, inner_idx, inner_end);

      hwy::pipeline::low_level::PrefetchPipelineLoop<Policy>(
          outer_idx, outer_end,
          [&](size_t i, auto& bundle) {
            cb.PopulateCachelines(i, inner_idx, inner_end, bundle);
          },
          [&](size_t i) { cb.ComputeTask(i, inner_idx, inner_end); },
          prefetch_args);
    }

    cb.OnOuterBlockFinish(outer_idx, outer_end);
  }
}

}  // namespace low_level

// ---------------------------------------------------------------------------
// PrefetchPipeline2DTiledLoop (Production Entry Point)
// ---------------------------------------------------------------------------
template <PrefetchTuningHint Hint = PrefetchTuningHint::kAuto,
          typename Policy = Default2DTiledPrefetchPolicy, typename Callbacks>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
  requires low_level::IsPrefetchPolicy<Policy>
#endif
inline void PrefetchPipeline2DTiledLoop(size_t outer_size, size_t inner_size,
                                        Callbacks& cb,
                                        Tiling2DArgs tiling = Tiling2DArgs(),
#if defined(__clang__) || defined(__GNUC__)
                                        const char* file_loc = __builtin_FILE(),
                                        int line_loc = __builtin_LINE()) {
#else
                                        const char* file_loc = nullptr,
                                        int line_loc = 0) {
#endif
  auto loop_runner = [&](const PrefetchArgs& args) {
    hwy::pipeline::low_level::PrefetchPipeline2DTiledLoop<Policy>(
        outer_size, inner_size, cb, tiling, args);
  };

  low_level::DispatchTunedWorkload<Hint, Policy>(
      tiling.TotalElements(outer_size, inner_size), file_loc, line_loc,
      loop_runner);
}

}  // namespace pipeline
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_2D_H_
