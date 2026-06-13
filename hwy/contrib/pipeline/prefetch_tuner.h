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

#ifndef HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_TUNER_H_
#define HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_TUNER_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/base.h"
#include "hwy/contrib/pipeline/prefetch_pipeline_types.h"
#include "hwy/timer.h"

namespace hwy {
namespace pipeline {
namespace low_level {

struct DefaultPrefetchTimer {
  static HWY_INLINE uint64_t Start() { return hwy::timer::Start(); }
  static HWY_INLINE uint64_t Stop() { return hwy::timer::Stop(); }
};

// ---------------------------------------------------------------------------
// Telemetry & Tuning Scopes
// ---------------------------------------------------------------------------
// An RAII scope manager returned by PrefetchTuner plugins.
// Automatically orchestrates timer allocation on construction and metric
// reporting on destruction, decoupling telemetry from the core POD structs.
template <typename Timer = DefaultPrefetchTimer>
class PrefetchTuningScopeT {
 public:
  // Steady-state constructor (Zero overhead, no timer started)
  PrefetchTuningScopeT()
      : args_{},
        cb_(nullptr),
        user_data_(nullptr),
        t0_(0),
        cost_normalization_factor_(1.0f) {}

  // Active profiling constructor (Starts timer automatically)
  PrefetchTuningScopeT(PrefetchArgs args, void (*cb)(void*, float),
                       void* user_data, float cost_normalization_factor)
      : args_(args),
        cb_(cb),
        user_data_(user_data),
        t0_(cb != nullptr ? Timer::Start() : 0),
        cost_normalization_factor_(cost_normalization_factor > 0.0f
                                       ? cost_normalization_factor
                                       : 1.0f) {}

  ~PrefetchTuningScopeT() {
    if (HWY_UNLIKELY(cb_ != nullptr)) {
      float elapsed = static_cast<float>(Timer::Stop() - t0_);
      cb_(user_data_, elapsed / cost_normalization_factor_);
    }
  }

  // Move semantics guarantee exactly-once firing
  PrefetchTuningScopeT(PrefetchTuningScopeT&& other) noexcept
      : args_(other.args_),
        cb_(other.cb_),
        user_data_(other.user_data_),
        t0_(other.t0_),
        cost_normalization_factor_(other.cost_normalization_factor_) {
    other.cb_ = nullptr;
  }

  PrefetchTuningScopeT& operator=(PrefetchTuningScopeT&& other) noexcept {
    if (this != &other) {
      if (cb_ != nullptr) {
        float elapsed = static_cast<float>(Timer::Stop() - t0_);
        cb_(user_data_, elapsed / cost_normalization_factor_);
      }
      args_ = other.args_;
      cb_ = other.cb_;
      user_data_ = other.user_data_;
      t0_ = other.t0_;
      cost_normalization_factor_ = other.cost_normalization_factor_;
      other.cb_ = nullptr;
    }
    return *this;
  }

  // Prevent copying
  PrefetchTuningScopeT(const PrefetchTuningScopeT&) = delete;
  PrefetchTuningScopeT& operator=(const PrefetchTuningScopeT&) = delete;

  // Returns the current prefetch arguments for the scope.
  const PrefetchArgs& GetArgs() const { return args_; }

 private:
  // The prefetch arguments to use for this scope. If the scope is actively
  // profiling, this is the candidate being evaluated. Otherwise, it is the best
  // converged configuration (or default if no tuning is performed).
  PrefetchArgs args_;

  // Telemetry ingestion callback function pointer invoked upon scope
  // destruction. Converts stateless lambdas directly into machine code stubs to
  // guarantee zero heap allocations and zero virtual table dispatch overhead.
  //
  // Parameters:
  //  - udata:           Opaque 64-bit payload (e.g., packed dense ID and
  //                     candidate args) passed through registers.
  //  - normalized_cost: Raw hardware timer ticks elapsed during active
  //                     profiling normalized by cost_normalization_factor.
  void (*cb_)(void* udata, float normalized_cost) = nullptr;

  // Opaque 64-bit payload passed through registers to the callback.
  // This could be a pointer to any allocated memory, or even an arbitrary
  // packed 64-bit value.
  //
  // May be nullptr if no callback is installed.
  void* user_data_ = nullptr;

  // The unnormalized cost (e.g., elapsed timer ticks) offset at the start of
  // profiling.
  uint64_t t0_ = 0;

  // The normalization factor to divide the raw cost by before reporting.
  float cost_normalization_factor_ = 1.0f;
};

// Default RAII scope manager using the default timer.
using PrefetchTuningScope = PrefetchTuningScopeT<>;

// ---------------------------------------------------------------------------
// Tuning Context & Plugin Interface
// ---------------------------------------------------------------------------

// Establishes the context around the data geometry and access patterns before
// executing the pipelined loops.
struct PrefetchTuningContext {
  // Hint for the underlying memory baseline in absence of a plugin.
  PrefetchTuningHint hint = PrefetchTuningHint::kAuto;

  // Flag indicating whether the workload is an ultra-tiny micro-batch.
  bool is_ultra_tiny = false;

  // Total number of discrete workload elements to be processed.
  // E.g., in 1D workloads, the number of outer loop iterations. In 2D
  // workloads, explicitly populated by callers as `num_rows * num_tile_per_row`
  // (total tile-iterations) to ensure accurate AutoFDO fission bucket
  // assignment.
  size_t total_elements = 0;

  // The estimated number of discrete cachelines accessed per workload element.
  // E.g., in 1D workloads, cachelines accessed per element. In 2D workloads,
  // represents cachelines accessed per tile-iteration (typically
  // `Policy::kMaxCachelinesPerIter`).
  size_t active_cachelines_per_element = 0;

  // Captures the precise physical source file and line number of the loop
  // invocation.
  const char* file_loc = nullptr;
  int line_loc = 0;

  // Pointer to an array of static string tags representing the active ambient
  // context.
  const char* const* scope_tags = nullptr;
  size_t num_scope_tags = 0;
};

// A tuning plugin interface used to instantiate dynamically-optimized pipeline
// arguments based on the runtime context prior to loop execution.
class PrefetchTuner {
 public:
  virtual ~PrefetchTuner() = default;

  // Derives dynamically-tuned lookahead depths based on the runtime context.
  virtual PrefetchTuningScope CreateScope(
      const PrefetchTuningContext& context) const = 0;

  // Registers a static call site context out-of-band and returns a callsite ID.
  // This is called exactly once at static initialization time per call site.
  virtual CallsiteId RegisterContext(
      const PrefetchTuningContext& context) const = 0;

  // Fast-path lock-free lookup of winning prefetch arguments by callsite ID.
  virtual PrefetchTuningScope CreateScopeByCallsiteId(
      CallsiteId callsite_id, const PrefetchTuningContext& context) const = 0;
};

}  // namespace low_level
}  // namespace pipeline
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_PIPELINE_PREFETCH_TUNER_H_
