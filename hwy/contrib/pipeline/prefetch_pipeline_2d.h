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
#include "hwy/contrib/pipeline/prefetch_args.h"
#include "hwy/contrib/pipeline/prefetch_pipeline.h"
#include "hwy/ops/shared-inl.h"
#include "hwy/timer.h"

namespace hwy {

struct Default2DTiledPrefetchLimits : public DefaultPrefetchLimits {};

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
//   // `cachelines` collection to be prefetched for the given inner block.
//   template <size_t kMaxCachelinesPerIter>
//   void GetPrefetchCachelines(
//       size_t outer_i, size_t inner_idx, size_t inner_end,
//       PrefetchCachelines<kMaxCachelinesPerIter>& cachelines);
//
//   // The core loop body to execute for a given sequence index `outer_i` and
//   // inner segment.
//   void ComputeTask(size_t outer_i, size_t inner_idx, size_t inner_end);
//
//   // Called after all inner segments have been processed for the outer block.
//   // Useful for committing accumulated data.
//   void OnOuterBlockFinish(size_t outer_idx, size_t outer_end);
template <typename T, size_t MaxCachelines>
concept PrefetchPipeline2DTiledCallbacks =
    requires(T& cb, size_t outer_idx, size_t outer_end, size_t inner_idx,
             size_t inner_end, PrefetchCachelines<MaxCachelines>& cachelines) {
      { cb.OnOuterBlockStart(outer_idx, outer_end) };
      { cb.PrepareInnerBlock(outer_idx, outer_end, inner_idx, inner_end) };
      { cb.GetPrefetchCachelines(outer_idx, inner_idx, inner_end, cachelines) };
      { cb.ComputeTask(outer_idx, inner_idx, inner_end) };
      { cb.OnOuterBlockFinish(outer_idx, outer_end) };
    };
#endif

// ---------------------------------------------------------------------------
// PrefetchPipeline2DTiledLoop
// ---------------------------------------------------------------------------
// A generic pipeline for executing 2D block-tiled computations with
// prefetching. This splits processing into localized blocks that prevent cache
// evictions and FPU instruction bottlenecks.
//
// Template Parameters:
//   Policy: A struct extending `Default2DTiledPrefetchLimits` dictating cache
//           limits, tiling dimensions, and loop constants.
//   Callbacks: A type matching the signature documented in
//              `PrefetchPipeline2DTiledCallbacks` above. Passed by reference to
//              allow state mutation.
//   args: A `Prefetch2DArgs` object dictating prefetch lookahead and tiling
//         dimensions. Note the telemetry fields, if set, will cover the entire
//         2D loop once.
template <typename Limits = Default2DTiledPrefetchLimits, typename Callbacks>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
  requires PrefetchPipeline2DTiledCallbacks<Callbacks,
                                            Limits::kMaxCachelinesPerIter>
#endif
HWY_INLINE void PrefetchPipeline2DTiledLoop(size_t outer_size,
                                            size_t inner_size, Callbacks& cb,
                                            const Prefetch2DArgs& args) {
  const uint64_t t0 =
      args.prefetch.metric_collector_cb != nullptr ? hwy::timer::Start() : 0;

  // We explicitly disable the metric collection callback for the internal 1D
  // pipeline loop to prevent nested metric tracing. We will trigger the
  // callback manually once the entire 2D loop has completed.
  Prefetch2DArgs inner_args = args;
  inner_args.prefetch.metric_collector_cb = nullptr;

  for (size_t outer_idx = 0; outer_idx < outer_size;
       outer_idx += args.outer_block) {
    const size_t outer_end = (outer_size < outer_idx + args.outer_block)
                                 ? outer_size
                                 : outer_idx + args.outer_block;
    cb.OnOuterBlockStart(outer_idx, outer_end);

    for (size_t inner_idx = 0; inner_idx < inner_size;
         inner_idx += args.inner_block) {
      const size_t inner_end = (inner_size < inner_idx + args.inner_block)
                                   ? inner_size
                                   : inner_idx + args.inner_block;

      cb.PrepareInnerBlock(outer_idx, outer_end, inner_idx, inner_end);

      hwy::PrefetchPipelineLoop<Limits>(
          outer_idx, outer_end,
          [&](size_t i, auto& cachelines) {
            cb.GetPrefetchCachelines(i, inner_idx, inner_end, cachelines);
          },
          [&](size_t i) { cb.ComputeTask(i, inner_idx, inner_end); },
          inner_args.prefetch);
    }

    cb.OnOuterBlockFinish(outer_idx, outer_end);
  }

  if (HWY_UNLIKELY(args.prefetch.metric_collector_cb != nullptr)) {
    args.prefetch.metric_collector_cb(args.prefetch.user_data,
                                      hwy::timer::Stop() - t0);
  }
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_PIPELINE_2D_H_
