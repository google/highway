// Generic One-to-Many 2D-Tiled SIMD Scoring Framework.
//
// This header provides a generic Highway-based 2D block-tiling architecture
// for One-To-Many (OTM) dot product / distance operations. It abstracts the
// pipeline mechanics so that any downstream application can leverage
// high-throughput vector search strategies.
//
// NOTE ON HEADER ARCHITECTURE: While the outer pipeline driver is structurally
// generic, we anticipate that the underlying compute kernels (ScorerKernel)
// will predominantly invoke per-target SIMD vector instructions. By packaging
// this framework as an `-inl.h` header included after `foreach_target.h`, we
// ensure flawless per-target SIMD kernel inlining and Clang AST target
// adaptation for maximum performance and architectural simplicity.
//
// 1. Separation of Concerns:
//    - MemoryLayout: Defines how memory is accessed, prefetching rules, and
//      final score calibration (e.g., metric adjustments).
//    - ScorerKernel: The raw compute abstraction (e.g. BF16/F32 FMA loops).
//    - Pipeline: The nested-loop driver that manages registers and cache.

#include "hwy/contrib/pipeline/prefetch_pipeline_2d.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner_registry.h"
#if defined(HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_INL_H_
#undef HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_INL_H_
#endif

#include <algorithm>
#include <concepts>  // IWYU pragma: keep
#include <cstddef>
#include <vector>

#include "hwy/base.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace dot {

namespace low_level {

// =============================================================================
// Generic Concept Definitions
// =============================================================================

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L

// Concept: Accessor
// ---------------------------------------------------------------------------
// Represents the mapping interface between the flat iteration index (0 to N)
// and the underlying database indices. It also serves as the final sink where
// MemoryLayout writes calibrated scores.
template <typename A>
concept OTM_Accessor = requires(const A c_accessor, size_t i) {
  { c_accessor.size() } -> std::convertible_to<size_t>;
  { c_accessor.Index(i) } -> std::convertible_to<size_t>;
};

// Concept: MemoryLayout
// ---------------------------------------------------------------------------
// Defines the contract for fetching dimensions, strides, and dynamically
// prefetching inner loops, independent of precision type (Float, BF16, Int8, or
// other quantization types).
//
// - Layout is the concrete type satisfying this contract (e.g. an
//   implementation that fetches dimensions and strides for a specific
//   quantization type and layout).
// - Accessor: The type of the accessor, which must satisfy OTM_Accessor.
// - AccumT: The type of the accumulator.
// - B_dp: The number of datapoints in a block.
template <typename Layout, typename Accessor, typename AccumT>
concept OTM_MemoryLayout =
    OTM_Accessor<Accessor> &&
    requires(const Layout layout, size_t dp_idx, size_t dp_end,
             size_t dim_start, size_t dim_end, const AccumT* raw_accumulators,
             Accessor& accessor,
             hwy::pipeline::CachelineBundle<Layout>& collector) {
      // Yields the discrete memory cachelines to prefetch for a chunk
      { layout.PopulateCachelines(dp_idx, dim_start, dim_end, collector) };
      { layout.CalibrateScores(raw_accumulators, dp_idx, dp_end, accessor) };
      // GetBasePtr MUST return a valid readable memory pointer even if dp_idx
      // is out-of-bounds (e.g., mapping invalid indices to index 0). This
      // guarantees that the ScorerKernel never segfaults and avoids branch
      // divergence during unrolled SIMD loads.
      { layout.GetBasePtr(dp_idx) } -> std::convertible_to<const void*>;
    };

// Concept: ScorerKernel
// ---------------------------------------------------------------------------
// The innermost compute block dealing exclusively with raw vectorized loops.
//
// - Kernel is the concrete type satisfying this contract.
template <typename Kernel, typename Layout>
concept OTM_ScorerKernel =
    requires(Kernel& kernel, size_t dp_idx, const Layout& layout,
             size_t dim_offset, size_t inner_end, size_t dim_end) {
      typename Kernel::AccumT;

      // Pipeline Configuration Traits
      // -----------------------------------------------------------------------
      // - kBlockDimensions: Query dimension slicing scale suitable for FMA
      // loops.
      { Kernel::kBlockDimensions } -> std::convertible_to<size_t>;

      // Hook designed for executing pre-block operations (like broadcasting
      // sub-centroid codebooks into SIMD registers prior to inner loop scans).
      { kernel.PrepareDimensionBlock(dim_offset, dim_end) };
      // Core compute evaluating datapoint vector comparisons, accumulating
      // into the requested accum reference.
      {
        kernel.ScoreBlock(dp_idx, layout, dim_offset, inner_end, dim_end,
                          std::declval<typename Kernel::AccumT&>())
      };
    };

#endif  // __cpp_concepts

// ===========================================================================
// WARNING: Low-Level Execution API
// ===========================================================================
// Do NOT call this struct's methods directly in production code! Bypassing the
// tuning framework prevents fleet-wide continuous profiling and telemetry
// collection. Prefer `hwy::OneToMany2DTiledPipeline::Run` below.
//
// A generic pipeline for executing 2D block-tiled one-to-many distance scoring.
// This splits scoring out into localized blocks that prevent cache evictions
// and FPU instruction bottlenecks.
//
// This generic pipeline retrieves hardware cache-bound constants directly from
// the execution kernel and policy descriptors.
//
// Required Traits:
//   ScorerKernelType::kBlockDimensions: The number of dimensions scored per
//       inner phase. This pins active query segments inside SIMD vector
//       registers / L1 cache. Often 256 or 512, representing a pipelined
//       saturation threshold.
struct OneToMany2DTiledPipeline {
 private:
  template <typename MemoryLayout, typename Accessor, typename ScorerKernelType,
            typename AccumT>
  struct Callbacks {
    const MemoryLayout& layout;
    Accessor& accessor;
    ScorerKernelType& kernel;
    size_t simd_end;
    size_t outer_block;
    std::vector<AccumT>& raw_accumulators;

    void OnOuterBlockStart(size_t /*outer_idx*/, size_t /*outer_end*/) {
      std::fill(raw_accumulators.begin(), raw_accumulators.end(), AccumT{0});
    }

    void PrepareInnerBlock(size_t /*outer_idx*/, size_t /*outer_end*/,
                           size_t inner_idx, size_t inner_end) {
      kernel.PrepareDimensionBlock(inner_idx, inner_end);
    }

    template <typename Policy>
    void PopulateCachelines(size_t outer_i, size_t inner_idx, size_t inner_end,
                            hwy::pipeline::CachelineBundle<Policy>& collector) {
      layout.PopulateCachelines(accessor.Index(outer_i), inner_idx, inner_end,
                                collector);
    }

    void ComputeTask(size_t outer_i, size_t inner_idx, size_t inner_end) {
      const size_t mapped_idx = accessor.Index(outer_i);
      kernel.ScoreBlock(mapped_idx, layout, inner_idx,
                        std::min(inner_end, simd_end), inner_end,
                        raw_accumulators[outer_i % outer_block]);
    }

    void OnOuterBlockFinish(size_t outer_idx, size_t outer_end) {
      layout.CalibrateScores(raw_accumulators.data(), outer_idx, outer_end,
                             accessor);
    }
  };

 public:
  template <typename MemoryLayout>
  struct OTM_Policy : hwy::pipeline::Default2DTiledPrefetchPolicy {
    static constexpr size_t kMaxCachelinesPerIter =
        MemoryLayout::kMaxCachelinesPerIter;
  };

  // Executes the 2D-tiled scoring pipeline with explicitly provided arguments.
  // Primarily useful for isolated micro-benchmarks or rigid environments that
  // must bypass dynamic tuning.
  template <typename MemoryLayout, typename Accessor, typename ScorerKernelType>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires OTM_MemoryLayout<MemoryLayout, Accessor,
                              typename ScorerKernelType::AccumT> &&
             OTM_ScorerKernel<ScorerKernelType, MemoryLayout>
#endif
  static HWY_INLINE HWY_ATTR void Run(
      const MemoryLayout& layout, Accessor& accessor, size_t total_dims,
      size_t simd_end, ScorerKernelType& kernel,
      hwy::pipeline::Tiling2DArgs tiling,
      const hwy::pipeline::PrefetchArgs& prefetch_args) {
    using AccumT = typename ScorerKernelType::AccumT;

    const size_t outer_block = tiling.outer_block;
    std::vector<AccumT> raw_accumulators(outer_block);

    Callbacks<MemoryLayout, Accessor, ScorerKernelType, AccumT> cb{
        layout, accessor, kernel, simd_end, outer_block, raw_accumulators};

    hwy::pipeline::low_level::PrefetchPipeline2DTiledLoop<
        OTM_Policy<MemoryLayout>>(accessor.size(), total_dims, cb, tiling,
                                  prefetch_args);
  }
};

}  // namespace low_level

// ===========================================================================
// Public Context-Aware Wrapper
// ===========================================================================
// OneToMany2DTiledPipeline
// ---------------------------------------------------------------------------
// A generic pipeline for executing 2D block-tiled one-to-many distance scoring.
// This splits scoring out into localized blocks that prevent cache evictions
// and FPU instruction bottlenecks.
struct OneToMany2DTiledPipeline {
  // Executes the 2D-tiled scoring pipeline utilizing Context-Aware Tuning.
  // This automatically resolves tuning arguments from the global registry,
  // preserving caller-specified tiling geometry while injecting auto-tuned
  // lookahead velocity.
  template <hwy::pipeline::PrefetchTuningHint Hint =
                hwy::pipeline::PrefetchTuningHint::kAuto,
            typename MemoryLayout, typename Accessor, typename ScorerKernelType>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires hwy::pipeline::low_level::IsPrefetchPolicy<
        low_level::OneToMany2DTiledPipeline::OTM_Policy<MemoryLayout>>
#endif
  static HWY_INLINE HWY_ATTR void Run(
      const MemoryLayout& layout, Accessor& accessor, size_t total_dims,
      size_t simd_end, ScorerKernelType& kernel,
      hwy::pipeline::Tiling2DArgs tiling = hwy::pipeline::Tiling2DArgs(),
#if defined(__clang__) || defined(__GNUC__)
      const char* file_loc = __builtin_FILE(), int line_loc = __builtin_LINE()
#else
      const char* file_loc = nullptr, int line_loc = 0
#endif
  ) {
    auto loop_runner = [&](const hwy::pipeline::PrefetchArgs& args) {
      low_level::OneToMany2DTiledPipeline::Run(layout, accessor, total_dims,
                                               simd_end, kernel, tiling, args);
    };

    hwy::pipeline::low_level::DispatchTunedWorkload<
        Hint, low_level::OneToMany2DTiledPipeline::OTM_Policy<MemoryLayout>>(
        tiling.TotalElements(accessor.size(), total_dims), file_loc, line_loc,
        loop_runner);
  }
};

}  // namespace dot
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif  // HIGHWAY_HWY_CONTRIB_DOT_ONE_TO_MANY_INL_H_
