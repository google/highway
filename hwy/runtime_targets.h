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

#ifndef HIGHWAY_RUNTIME_TARGETS_H_
#define HIGHWAY_RUNTIME_TARGETS_H_

// Chooses which targets to generate via foreach_target.h and potentially call
// by runtime_dispatch.h.

#include "third_party/highway/highway/arch.h"
#include "third_party/highway/highway/target_bits.h"

// Functors will be specialized for all targets in SIMD_RUNTIME_TARGETS, which
// requires compiler support, but only called if supported by the current CPU.
// This bitfield is separate from SIMD_STATIC_TARGETS to allow static_targets.h
// and runtime_dispatch.h to be included in the same translation unit.
#if SIMD_ARCH == SIMD_ARCH_X86
#define SIMD_RUNTIME_TARGETS (SIMD_SSE4 | SIMD_AVX2 | SIMD_AVX512)
#elif SIMD_ARCH == SIMD_ARCH_PPC
#define SIMD_RUNTIME_TARGETS SIMD_PPC8
#elif SIMD_ARCH == SIMD_ARCH_ARM
#define SIMD_RUNTIME_TARGETS SIMD_ARM8
#elif SIMD_ARCH == SIMD_ARCH_SCALAR
#define SIMD_RUNTIME_TARGETS SIMD_NONE
#else
#error "Unsupported platform"
#endif

#endif  // HIGHWAY_RUNTIME_TARGETS_H_
