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

#ifndef HWY_CACHE_CONTROL_H_
#define HWY_CACHE_CONTROL_H_

#include <stdint.h>
#include <string.h>  // memcpy

#include "hwy/arch.h"
#include "hwy/compiler_specific.h"

#if HWY_ARCH == HWY_ARCH_X86
#include <emmintrin.h>
#endif

namespace hwy {

HWY_INLINE void Stream(const uint32_t t, uint32_t* HWY_RESTRICT aligned) {
#if HWY_ARCH == HWY_ARCH_X86
  _mm_stream_si32(reinterpret_cast<int*>(aligned), t);
#else
  memcpy(aligned, &t, sizeof(t));
#endif
}

HWY_INLINE void Stream(const uint64_t t, uint64_t* HWY_RESTRICT aligned) {
#if defined(__x86_64__) || defined(_M_X64)
  // NOLINTNEXTLINE(google-runtime-int)
  _mm_stream_si64(reinterpret_cast<long long*>(aligned), t);
#elif HWY_ARCH == HWY_ARCH_X86  // i386 case
  _mm_stream_si32(reinterpret_cast<int*>(aligned), static_cast<uint32_t>(t));
  _mm_stream_si32(reinterpret_cast<int*>(aligned) + 1, t >> 32u);
#else
  memcpy(aligned, &t, sizeof(t));
#endif
}

// Delays subsequent loads until prior loads are visible. On Intel CPUs, also
// serves as a full fence (waits for all prior instructions to complete).
// No effect on non-x86.
HWY_INLINE void LoadFence() {
#if HWY_ARCH == HWY_ARCH_X86
  _mm_lfence();
#endif
}

// Ensures previous weakly-ordered stores are visible. No effect on non-x86.
HWY_INLINE void StoreFence() {
#if HWY_ARCH == HWY_ARCH_X86
  _mm_sfence();
#endif
}

// Begins loading the cache line containing "p".
template <typename T>
HWY_INLINE void Prefetch(const T* p) {
#if HWY_ARCH == HWY_ARCH_X86
  _mm_prefetch(p, _MM_HINT_T0);
#elif HWY_ARCH == HWY_ARCH_ARM
  __pld(p);
#else
  (void)p;
#endif
}

// Invalidates and flushes the cache line containing "p". No effect on non-x86.
HWY_INLINE void FlushCacheline(const void* p) {
#if HWY_ARCH == HWY_ARCH_X86
  _mm_clflush(p);
#else
  (void)p;
#endif
}

}  // namespace hwy

#endif  // HWY_CACHE_CONTROL_H_
