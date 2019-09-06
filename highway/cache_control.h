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

#ifndef HIGHWAY_CACHE_CONTROL_H_
#define HIGHWAY_CACHE_CONTROL_H_

#include <stdint.h>
#include <string.h>  // memcpy

#include "third_party/highway/highway/arch.h"
#include "third_party/highway/highway/compiler_specific.h"

#if SIMD_ARCH == SIMD_ARCH_X86
#include <emmintrin.h>
#endif

namespace jxl {

SIMD_INLINE void stream(const uint32_t t, uint32_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_stream_si32(reinterpret_cast<int*>(aligned), t);
#else
  memcpy(aligned, &t, sizeof(t));
#endif
}

SIMD_INLINE void stream(const uint64_t t, uint64_t* SIMD_RESTRICT aligned) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_stream_si64(reinterpret_cast<long long*>(aligned), t);
#else
  memcpy(aligned, &t, sizeof(t));
#endif
}

// Delays subsequent loads until prior loads are visible. On Intel CPUs, also
// serves as a full fence (waits for all prior instructions to complete).
// No effect on non-x86.
SIMD_INLINE void load_fence() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_lfence();
#endif
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
SIMD_INLINE void store_fence() {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_sfence();
#endif
}

// Begins loading the cache line containing "p".
template <typename T>
SIMD_INLINE void prefetch(const T* p) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_prefetch(p, _MM_HINT_T0);
#elif SIMD_ARCH == SIMD_ARCH_ARM
  __pld(p);
#endif
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
SIMD_INLINE void flush_cacheline(const void* p) {
#if SIMD_ARCH == SIMD_ARCH_X86
  _mm_clflush(p);
#endif
}

}  // namespace jxl

#endif  // HIGHWAY_CACHE_CONTROL_H_
