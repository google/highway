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

#ifndef HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_TUNER_REGISTRY_H_
#define HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_TUNER_REGISTRY_H_

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "hwy/base.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/contrib/pipeline/prefetch_tuner.h"

namespace hwy {
namespace pipeline {
namespace low_level {

// Link-time hook to retrieve a context-aware prefetch tuner if one is linked.
// Returns nullptr if no plugin is available (falling back to generic defaults).
inline const PrefetchTuner*& GetGlobalPrefetchTunerRegistry() {
  static const PrefetchTuner* tuner = nullptr;
  return tuner;
}

// Sets the global prefetch tuner registry.
// This should only be called once at initialization time.
inline void SetGlobalPrefetchTunerRegistry(const PrefetchTuner* tuner) {
  HWY_ASSERT(GetGlobalPrefetchTunerRegistry() == nullptr);
  GetGlobalPrefetchTunerRegistry() = tuner;
}

// Registers a prefetch tuning context with the global tuner registry. Returns a
// callsite ID for the context.
inline CallsiteId RegisterPrefetchContext(
    const PrefetchTuningContext& context) {
  const PrefetchTuner* tuner = GetGlobalPrefetchTunerRegistry();
  if (tuner != nullptr) {
    return tuner->RegisterContext(context);
  }
  return 0;
}

// Creates a prefetch tuning scope for the given callsite ID and context.
// If no tuner is available or if the workload is ultra-tiny, returns a scope
// with default args and cost normalization factor.
inline PrefetchTuningScope CreatePrefetchTuningScopeByCallsiteId(
    CallsiteId callsite_id, const PrefetchTuningContext& context) {
  if (HWY_LIKELY(!context.is_ultra_tiny)) {
    const PrefetchTuner* tuner = GetGlobalPrefetchTunerRegistry();
    if (HWY_LIKELY(tuner != nullptr)) {
      return tuner->CreateScopeByCallsiteId(callsite_id, context);
    }
  }

  // Use default args and cost normalization factor if no tuner is available or
  // if the workload is ultra-tiny (to bypass profiling overhead).
  constexpr float kDefaultCostNormalizationFactor = 1.0f;
  return PrefetchTuningScope(context.hint == PrefetchTuningHint::kRandom
                                 ? PrefetchArgs::DefaultRandom()
                                 : PrefetchArgs::DefaultSequential(),
                             nullptr, nullptr, kDefaultCostNormalizationFactor);
}

// ---------------------------------------------------------------------------
// Dynamic Dispatch Fission Helper
// ---------------------------------------------------------------------------
#if defined(__clang__) || defined(__GNUC__)
#define HWY_NO_CODE_MERGE __attribute__((noinline))
#else
#define HWY_NO_CODE_MERGE
#endif

// Dispatches workload execution across distinct physical call sites based on
// the compile-time memory hint and runtime workload size.
//
// Under the hood, this mechanism pairs with forced compiler fission
// (HWY_NO_CODE_MERGE on the caller's lambda) to guarantee that LLVM AutoFDO
// records separate profile buckets and prefetch tuners allocate independent
// lookahead tables for disparate workload categories (Tiny, Medium, Huge)
// originating from the exact same user call site.
//
// Microarchitectural Footprint: Generates exactly one physical machine code
// instance of the unrolled loop body in the .text section (zero code bloat),
// incurring only ~2-3 cycles of fast integer comparison jumping.
template <PrefetchTuningHint Hint, typename Policy, typename Executor>
HWY_INLINE void DispatchWorkloadFission(size_t total_elements,
                                        Executor&& exec) {
  if (HWY_UNLIKELY(total_elements < Policy::kUltraTinyThreshold)) {
    exec.template operator()<0>(true);
    return;
  }

  constexpr size_t kTinyThreshold = 256;
  constexpr size_t kMediumThreshold = 65536;

  if constexpr (Hint == PrefetchTuningHint::kSequential) {
    if (total_elements <= kTinyThreshold) {
      exec.template operator()<100000>(false);
    } else if (total_elements <= kMediumThreshold) {
      exec.template operator()<200000>(false);
    } else {
      exec.template operator()<300000>(false);
    }
  } else if constexpr (Hint == PrefetchTuningHint::kRandom) {
    if (total_elements <= kTinyThreshold) {
      exec.template operator()<400000>(false);
    } else if (total_elements <= kMediumThreshold) {
      exec.template operator()<500000>(false);
    } else {
      exec.template operator()<600000>(false);
    }
  } else {
    // kAuto fallback
    if (total_elements <= kTinyThreshold) {
      exec.template operator()<700000>(false);
    } else if (total_elements <= kMediumThreshold) {
      exec.template operator()<800000>(false);
    } else {
      exec.template operator()<900000>(false);
    }
  }
}

template <PrefetchTuningHint Hint, typename Policy, typename LoopRunner>
struct UniversalFissionExecutor {
  size_t total_elements;
  const char* file_loc;
  int line_loc;
  LoopRunner runner;

  template <size_t FissionOffset>
  HWY_NO_CODE_MERGE void operator()(bool is_ultra_tiny) const {
    const int derived_line_loc = line_loc + static_cast<int>(FissionOffset);
    static const CallsiteId kCallsiteId = [this, derived_line_loc]() {
      PrefetchTuningContext init_ctx;
      init_ctx.file_loc = file_loc;
      init_ctx.line_loc = derived_line_loc;
      init_ctx.hint = Hint;
      init_ctx.active_cachelines_per_element = Policy::kMaxCachelinesPerIter;
      return RegisterPrefetchContext(init_ctx);
    }();

    PrefetchTuningContext run_ctx;
    run_ctx.hint = Hint;
    run_ctx.total_elements = total_elements;
    run_ctx.active_cachelines_per_element = Policy::kMaxCachelinesPerIter;
    run_ctx.file_loc = file_loc;
    run_ctx.line_loc = derived_line_loc;
    run_ctx.is_ultra_tiny = is_ultra_tiny;

    PrefetchTuningScope tuning_scope =
        CreatePrefetchTuningScopeByCallsiteId(kCallsiteId, run_ctx);

    runner(tuning_scope.GetArgs());
  }
};

template <PrefetchTuningHint Hint, typename Policy, typename LoopRunner>
HWY_INLINE void DispatchTunedWorkload(size_t total_elements,
                                      const char* file_loc, int line_loc,
                                      LoopRunner&& runner) {
  UniversalFissionExecutor<Hint, Policy,
                           typename std::remove_reference<LoopRunner>::type>
      exec{total_elements, file_loc, line_loc,
           std::forward<LoopRunner>(runner)};
  DispatchWorkloadFission<Hint, Policy>(total_elements, exec);
}

}  // namespace low_level
}  // namespace pipeline
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_TUNER_REGISTRY_H_
