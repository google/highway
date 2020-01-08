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

#ifndef HWY_STATIC_TARGETS_H_
#define HWY_STATIC_TARGETS_H_

// Defines HWY_STATIC_TARGETS and chooses the best available.

#include "hwy/arch.h"
#include "hwy/targets.h"

#ifndef HWY_STATIC_TARGETS

// Any targets in HWY_STATIC_TARGETS are eligible for use without runtime
// dispatch, so they must be supported by both the compiler and CPU.
#if HWY_ARCH == HWY_ARCH_X86
#if defined(HWY_DISABLE_AVX2)
// Clang6 encounters backend errors when compiling AVX2.
#define HWY_STATIC_TARGETS (HWY_SSE4)
#else
#define HWY_STATIC_TARGETS (HWY_SSE4 | HWY_AVX2)
#endif
#elif HWY_ARCH == HWY_ARCH_PPC
#define HWY_STATIC_TARGETS HWY_PPC8
#elif HWY_ARCH == HWY_ARCH_ARM
#define HWY_STATIC_TARGETS HWY_ARM8
#elif HWY_ARCH == HWY_ARCH_WASM
#define HWY_STATIC_TARGETS HWY_WASM
#elif HWY_ARCH == HWY_ARCH_SCALAR
#define HWY_STATIC_TARGETS HWY_NONE
#else
#error "Unsupported platform"
#endif

#endif  // HWY_STATIC_TARGETS

// After HWY_STATIC_TARGETS:
#include "hwy/include_headers.h"

// It is permissible to include static_targets.h BEFORE foreach_target.h,
// but not the reverse because that would override the intended runtime target.
#ifdef HWY_NAMESPACE
#error "Do not include static_targets.h after foreach_target.h"
#endif

// Choose best available:

//-----------------------------------------------------------------------------
#if HWY_STATIC_TARGETS & HWY_AVX512
#define HWY_BITS 512
#define HWY_ALIGN alignas(64)
#define HWY_HAS_CMP64 1
#define HWY_HAS_GATHER 1
#define HWY_HAS_VARIABLE_SHIFT 1
#define HWY_HAS_INT64 1
#define HWY_HAS_DOUBLE 1
#define HWY_ATTR HWY_ATTR_AVX512
//-----------------------------------------------------------------------------
#elif HWY_STATIC_TARGETS & HWY_AVX2
#define HWY_BITS 256
#define HWY_ALIGN alignas(32)
#define HWY_HAS_CMP64 1
#define HWY_HAS_GATHER 1
#define HWY_HAS_VARIABLE_SHIFT 1
#define HWY_HAS_INT64 1
#define HWY_HAS_DOUBLE 1
#define HWY_ATTR HWY_ATTR_AVX2
//-----------------------------------------------------------------------------
#elif HWY_STATIC_TARGETS & HWY_SSE4
#define HWY_BITS 128
#define HWY_ALIGN alignas(16)
#define HWY_HAS_CMP64 0
#define HWY_HAS_GATHER 0
#define HWY_HAS_VARIABLE_SHIFT 0
#define HWY_HAS_INT64 1
#define HWY_HAS_DOUBLE 1
#define HWY_ATTR HWY_ATTR_SSE4
//-----------------------------------------------------------------------------
#elif HWY_STATIC_TARGETS & HWY_PPC8
#define HWY_BITS 128
#define HWY_ALIGN alignas(16)
#define HWY_HAS_CMP64 1
#define HWY_HAS_GATHER 0
#define HWY_HAS_VARIABLE_SHIFT 1
#define HWY_HAS_INT64 1
#define HWY_HAS_DOUBLE 1
#define HWY_ATTR
//-----------------------------------------------------------------------------
#elif HWY_STATIC_TARGETS & HWY_WASM
#define HWY_BITS 128
#define HWY_ALIGN alignas(16)
#define HWY_HAS_CMP64 0
#define HWY_HAS_GATHER 0
#define HWY_HAS_VARIABLE_SHIFT 0
#define HWY_HAS_INT64 0
#define HWY_HAS_DOUBLE 0
#define HWY_ATTR HWY_ATTR_WASM
//-----------------------------------------------------------------------------
#elif HWY_STATIC_TARGETS & HWY_ARM8
#define HWY_BITS 128
#define HWY_ALIGN alignas(16)
#ifdef __arm__
#define HWY_HAS_CMP64 0
#define HWY_HAS_DOUBLE 0
#else
#define HWY_HAS_CMP64 1
#define HWY_HAS_DOUBLE 1
#endif
#define HWY_HAS_INT64 1
#define HWY_HAS_GATHER 0
#define HWY_HAS_VARIABLE_SHIFT 1
#define HWY_ATTR HWY_ATTR_ARM8
//-----------------------------------------------------------------------------
#else  // NONE
#define HWY_BITS 0
#define HWY_ALIGN
#define HWY_HAS_CMP64 0
#define HWY_HAS_GATHER 1
#define HWY_HAS_VARIABLE_SHIFT 1
#define HWY_HAS_INT64 1
#define HWY_HAS_DOUBLE 1
#define HWY_ATTR
#endif

#endif  // HWY_STATIC_TARGETS_H_
