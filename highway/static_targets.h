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

#ifndef HIGHWAY_STATIC_TARGETS_H_
#define HIGHWAY_STATIC_TARGETS_H_

// Defines SIMD_STATIC_TARGETS and the best available SIMD_TARGET.

#if defined(SIMD_NAMESPACE) || defined(SIMD_TARGET)
#error "Include static_targets.h before in_target.h"
#endif

#include "third_party/highway/highway/arch.h"
#include "third_party/highway/highway/target_bits.h"

// Any targets in SIMD_STATIC_TARGETS are eligible for use without runtime
// dispatch, so they must be supported by both the compiler and CPU.
#if SIMD_ARCH == SIMD_ARCH_X86
#define SIMD_STATIC_TARGETS (SIMD_SSE4 | SIMD_AVX2 /*| SIMD_AVX512*/)
#elif SIMD_ARCH == SIMD_ARCH_PPC
#define SIMD_STATIC_TARGETS SIMD_PPC8
#elif SIMD_ARCH == SIMD_ARCH_ARM
#define SIMD_STATIC_TARGETS SIMD_ARM8
#elif SIMD_ARCH == SIMD_ARCH_SCALAR
#define SIMD_STATIC_TARGETS SIMD_NONE
#else
#error "Unsupported platform"
#endif

// After SIMD_STATIC_TARGETS
#include "third_party/highway/highway/simd.h"

// SIMD_TARGET determines the value of SIMD_ATTR.
#if SIMD_STATIC_TARGETS & SIMD_AVX512
#define SIMD_TARGET AVX512
#elif SIMD_STATIC_TARGETS & SIMD_AVX2
#define SIMD_TARGET AVX2
#elif SIMD_STATIC_TARGETS & SIMD_SSE4
#define SIMD_TARGET SSE4
#elif SIMD_STATIC_TARGETS & SIMD_PPC8
#define SIMD_TARGET PPC8
#elif SIMD_STATIC_TARGETS & SIMD_ARM8
#define SIMD_TARGET ARM8
#else
#define SIMD_TARGET NONE
#endif

#endif  // HIGHWAY_STATIC_TARGETS_H_
