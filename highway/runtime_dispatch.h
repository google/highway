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

#ifndef HIGHWAY_RUNTIME_DISPATCH_H_
#define HIGHWAY_RUNTIME_DISPATCH_H_

// Chooses and calls best available target instantiation. Example usage:
//
// module.h
//   struct Module {
//     template <class Target>  // instantiated for each enabled target
//     void operator()(/*args*/) const;
//   }
//
// caller.cc:
//   #include "third_party/highway/highway/runtime_dispatch.h"
//   Dispatch(TargetBitfield().Best(), Module()/*, args*/);
//
// module.cc:
//   #define SIMD_TARGET_SKIP  // required if module_target is also compiled
//   #include "module_target.cc"  // ensures build system knows dependency
//   #define SIMD_TARGET_INCLUDE "module_target.cc"
//   #include "third_party/highway/highway/foreach_target.h"  // sets SIMD_TARGET=...
//
// module_target.cc:
//   #ifndef SIMD_TARGET_INCLUDE  // included directly, not by foreach_target
//   // Shared definitions/headers, only compiled once:
//   #include "module.h"
//   #include "third_party/highway/highway/in_target.h"  // sets SIMD_TARGET=NONE
//   #endif
//   namespace SIMD_NAMESPACE { namespace {
//     // implementation
//   }}
//   template <>
//   void Module::operator()<SIMD_TARGET>(/*args*/) const {
//     // call implementation
//   }

#include <stddef.h>

#include <utility>  // std::forward

#include "third_party/highway/highway/compiler_specific.h"  // SIMD_INLINE
#include "third_party/highway/highway/runtime_targets.h"
#include "third_party/highway/highway/targets.h"

namespace jxl {

// Strongly-typed enum ensures the argument to Dispatch is a single target, not
// a bitfield. #if avoids unhandled-enumerator warnings.
enum class Target {
#if SIMD_RUNTIME_TARGETS & SIMD_AVX512
  kAVX512 = SIMD_AVX512,
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_AVX2
  kAVX2 = SIMD_AVX2,
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_SSE4
  kSSE4 = SIMD_SSE4,
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_PPC8
  kPPC8 = SIMD_PPC8,
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_ARM8
  kARM8 = SIMD_ARM8,
#endif
  kNONE = SIMD_NONE
};

// Returns func.operator()<Target>(args). Calling a member function template
// instead of a class template allows stateful functors. Dispatch overhead is
// low but prefer to call this infrequently by hoisting this call.
template <class Func, typename... Args>
SIMD_INLINE auto Dispatch(const Target target, Func&& func, Args&&... args)
    -> decltype(std::forward<Func>(func).template operator()<NONE>(
        std::forward<Args>(args)...)) {
  switch (target) {
#if SIMD_RUNTIME_TARGETS & SIMD_AVX512
    case Target::kAVX512:
      return std::forward<Func>(func).template operator()<AVX512>(
          std::forward<Args>(args)...);
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_AVX2
    case Target::kAVX2:
      return std::forward<Func>(func).template operator()<AVX2>(
          std::forward<Args>(args)...);
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_SSE4
    case Target::kSSE4:
      return std::forward<Func>(func).template operator()<SSE4>(
          std::forward<Args>(args)...);
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_PPC8
    case Target::kPPC8:
      return std::forward<Func>(func).template operator()<PPC8>(
          std::forward<Args>(args)...);
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_ARM8
    case Target::kARM8:
      return std::forward<Func>(func).template operator()<ARM8>(
          std::forward<Args>(args)...);
#endif

    case Target::kNONE:
      return std::forward<Func>(func).template operator()<NONE>(
          std::forward<Args>(args)...);
  }
}

// All targets supported by the current CPU. Cheap to construct.
class TargetBitfield {
 public:
  TargetBitfield();

  int Bits() const { return bits_; }
  bool Any() const { return bits_ != 0; }

  // Returns 'best' (widest/most recent) target amongst those supported.
  Target Best() const {
#if SIMD_RUNTIME_TARGETS & SIMD_AVX512
    if (bits_ & SIMD_AVX512) return Target::kAVX512;
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_AVX2
    if (bits_ & SIMD_AVX2) return Target::kAVX2;
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_SSE4
    if (bits_ & SIMD_SSE4) return Target::kSSE4;
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_PPC8
    if (bits_ & SIMD_PPC8) return Target::kPPC8;
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_ARM8
    if (bits_ & SIMD_ARM8) return Target::kARM8;
#endif
    return Target::kNONE;
  }

  void Clear(Target target) { bits_ &= ~static_cast<int>(target); }

  // Calls func.operator()<Target>(args) for all enabled targets and returns
  // which target bits were used.
  template <class Func, typename... Args>
  SIMD_INLINE int Foreach(Func&& func, Args&&... args) const {
    std::forward<Func>(func).template operator()<NONE>(
        std::forward<Args>(args)...);

#if SIMD_RUNTIME_TARGETS & SIMD_SSE4
    if (bits_ & SIMD_SSE4) {
      std::forward<Func>(func).template operator()<SSE4>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_AVX2
    if (bits_ & SIMD_AVX2) {
      std::forward<Func>(func).template operator()<AVX2>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_AVX512
    if (bits_ & SIMD_AVX512) {
      std::forward<Func>(func).template operator()<AVX512>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_PPC8
    if (bits_ & SIMD_PPC8) {
      std::forward<Func>(func).template operator()<PPC8>(
          std::forward<Args>(args)...);
    }
#endif
#if SIMD_RUNTIME_TARGETS & SIMD_ARM8
    if (bits_ & SIMD_ARM8) {
      std::forward<Func>(func).template operator()<ARM8>(
          std::forward<Args>(args)...);
    }
#endif

    return bits_ & SIMD_RUNTIME_TARGETS;
  }

 private:
  int bits_;
};

}  // namespace jxl

#endif  // HIGHWAY_RUNTIME_DISPATCH_H_
