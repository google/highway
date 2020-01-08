// Copyright 2019 Google LLC
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

#ifndef HWY_RUNTIME_DISPATCH_H_
#define HWY_RUNTIME_DISPATCH_H_

// Chooses and calls best available target instantiation. See README.md.

#include <stddef.h>

#include <utility>  // std::forward

#include "hwy/compiler_specific.h"  // HWY_INLINE
#include "hwy/runtime_targets.h"
#include "hwy/targets.h"

namespace hwy {

// Strongly-typed enum ensures the argument to Dispatch is a single target, not
// a bitfield. Only targets in HWY_RUNTIME_TARGETS are defined, which avoids
// unhandled-enumerator warnings.
enum class Target {
#define HWY_X(target) k##target = HWY_##target,
  HWY_FOREACH_TARGET
#undef HWY_X
};

// Returns func.HWY_FUNC(args). Calling a member function allows stateful
// functors. Dispatch overhead is low but prefer to call this infrequently by
// hoisting this call.
template <class Func, typename... Args>
HWY_INLINE auto Dispatch(const Target target, Func&& func, Args&&... args)
    -> decltype(std::forward<Func>(func).F_NONE(std::forward<Args>(args)...)) {
  switch (target) {
#define HWY_X(target)     \
  case Target::k##target: \
    return std::forward<Func>(func).F_##target(std::forward<Args>(args)...);
    HWY_FOREACH_TARGET
#undef HWY_X
  }
}

// All targets supported by the current CPU. Cheap to construct.
class TargetBitfield {
 public:
  TargetBitfield();

  int Bits() const { return bits_; }
  bool Any() const { return bits_ != 0; }

  // The best target available on all supported CPUs.
  static constexpr Target Baseline() {
#if HWY_RUNTIME_TARGETS & HWY_WASM
    return Target::kWASM;
#elif HWY_RUNTIME_TARGETS & HWY_SSE4
    return Target::kSSE4;
#elif HWY_RUNTIME_TARGETS & HWY_ARM8
    return Target::kARM8;
#elif HWY_RUNTIME_TARGETS & HWY_PPC8
    return Target::kPPC;
#else
    return Target::kNONE;
#endif
  }

  // Returns 'best' (widest/most recent) target amongst those supported.
  Target Best() const {
#if HWY_RUNTIME_TARGETS & HWY_WASM
    if (bits_ & HWY_WASM) return Target::kWASM;
#endif
#if HWY_RUNTIME_TARGETS & HWY_AVX512
    if (bits_ & HWY_AVX512) return Target::kAVX512;
#endif
#if HWY_RUNTIME_TARGETS & HWY_AVX2
    if (bits_ & HWY_AVX2) return Target::kAVX2;
#endif
#if HWY_RUNTIME_TARGETS & HWY_SSE4
    if (bits_ & HWY_SSE4) return Target::kSSE4;
#endif
#if HWY_RUNTIME_TARGETS & HWY_PPC8
    if (bits_ & HWY_PPC8) return Target::kPPC8;
#endif
#if HWY_RUNTIME_TARGETS & HWY_ARM8
    if (bits_ & HWY_ARM8) return Target::kARM8;
#endif
    return Target::kNONE;
  }

  void Clear(Target target) { bits_ &= ~static_cast<int>(target); }

  // Calls func.HWY_FUNC(args) for all enabled targets and returns the bitfield
  // of enabled targets.
  template <class Func, typename... Args>
  HWY_INLINE int Foreach(Func&& func, Args&&... args) const {
#define HWY_X(target)                                                   \
  {                                                                     \
    constexpr int kBit = HWY_##target;                                  \
    /* if supported, OR kBit == HWY_NONE (always enabled) */            \
    if ((bits_ & kBit) == kBit) {                                       \
      std::forward<Func>(func).F_##target(std::forward<Args>(args)...); \
    }                                                                   \
  }
    HWY_FOREACH_TARGET
#undef HWY_X

    return bits_ & HWY_RUNTIME_TARGETS;
  }

 private:
  int bits_;
};

}  // namespace hwy

#endif  // HWY_RUNTIME_DISPATCH_H_
