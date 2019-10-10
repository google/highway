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

// No include guard - requires textual inclusion.

// Includes a specified file for each target in SIMD_RUNTIME_TARGETS. This
// generates an instantiation of Module::operator()<SIMD_TARGET> which may be
// called by runtime_dispatch.h. NOTE: before including this, #include the
// header whose filename is SIMD_TARGET_INCLUDE so build systems are aware of
// the dependency.

#ifndef SIMD_TARGET_INCLUDE
#error "Must set SIMD_TARGET_INCLUDE to name of include file"
#endif
#ifndef SIMD_TARGET_SKIP
#error "Must set SIMD_TARGET_SKIP"
#endif

#ifndef SIMD_RUNTIME_TARGETS
#error "Must include in_target.h first"
// Avoid undefined-identifier warnings in IDE:
#define SIMD_RUNTIME_TARGETS 0
#include "third_party/highway/highway/target_bits.h"
#endif

#if SIMD_RUNTIME_TARGETS & SIMD_AVX512
#undef SIMD_TARGET
#define SIMD_TARGET AVX512
#include SIMD_TARGET_INCLUDE
#endif

#if SIMD_RUNTIME_TARGETS & SIMD_AVX2
#undef SIMD_TARGET
#define SIMD_TARGET AVX2
#include SIMD_TARGET_INCLUDE
#endif

#if SIMD_RUNTIME_TARGETS & SIMD_SSE4
#undef SIMD_TARGET
#define SIMD_TARGET SSE4
#include SIMD_TARGET_INCLUDE
#endif

#if SIMD_RUNTIME_TARGETS & SIMD_PPC8
#undef SIMD_TARGET
#define SIMD_TARGET PPC8
#include SIMD_TARGET_INCLUDE
#endif

#if SIMD_RUNTIME_TARGETS & SIMD_ARM8
#undef SIMD_TARGET
#define SIMD_TARGET ARM8
#include SIMD_TARGET_INCLUDE
#endif

// NOTE: no SIMD_TARGET=NONE -- see in_target.h.

#undef SIMD_TARGET_INCLUDE  // ensures next user sets it
