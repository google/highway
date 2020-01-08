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

#ifndef HWY_INCLUDE_HEADERS_H_
#define HWY_INCLUDE_HEADERS_H_

// Do not include directly.
#if !defined(HWY_STATIC_TARGETS) && !defined(HWY_RUNTIME_TARGETS)
#error "Must include static_targets.h or foreach_target.h"
#endif

#include <atomic>

// Alternative for asm volatile("" : : : "memory"), which has no effect.
#define HWY_FENCE std::atomic_thread_fence(std::memory_order_acq_rel)

// How many lanes of type T in a full vector. NOTE: cannot use in #if because
// this uses sizeof.
#define HWY_LANES_OR_0(T) (HWY_BITS / (8 * sizeof(T)))

#define HWY_FULL(T) ::hwy::Desc<T, HWY_LANES_OR_0(T)>

// A vector of up to MAX_N lanes.
#define HWY_CAPPED(T, MAX_N) ::hwy::Desc<T, HWY_MIN(MAX_N, HWY_LANES_OR_0(T))>

#endif  // #ifndef HWY_INCLUDE_HEADERS_H_

#if (HWY_RUNTIME_TARGETS | HWY_STATIC_TARGETS) & HWY_WASM
#include "hwy/wasm.h"
#endif

#if (HWY_RUNTIME_TARGETS | HWY_STATIC_TARGETS) & HWY_ARM8
#include "hwy/arm64_neon.h"
#endif

#if (HWY_RUNTIME_TARGETS | HWY_STATIC_TARGETS) & HWY_SSE4
#include "hwy/x86/sse4.h"
#endif

#if (HWY_RUNTIME_TARGETS | HWY_STATIC_TARGETS) & HWY_AVX2
#include "hwy/x86/avx2.h"
#endif

#if (HWY_RUNTIME_TARGETS | HWY_STATIC_TARGETS) & HWY_AVX512
#include "hwy/x86/avx512.h"
#endif

#include "hwy/scalar.h"
