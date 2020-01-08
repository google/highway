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

#ifndef HWY_RUNTIME_TARGETS_H_
#define HWY_RUNTIME_TARGETS_H_

// Chooses default HWY_RUNTIME_TARGETS and provides support for generating
// implementations (if enabled) and invocations (if supported).

// HWY_RUNTIME_TARGETS is separate from HWY_STATIC_TARGETS to allow
// static_targets.h and runtime_dispatch.h to coexist in a translation unit.

#include "hwy/arch.h"
#include "hwy/compiler_specific.h"
#include "hwy/targets.h"

//-----------------------------------------------------------------------------
// If not already defined, choose default HWY_RUNTIME_TARGETS for this platform.

#ifndef HWY_RUNTIME_TARGETS

#if HWY_ARCH == HWY_ARCH_X86

#if defined(HWY_DISABLE_AVX2)
#define HWY_RUNTIME_TARGETS HWY_SSE4
#elif (HWY_COMPILER_CLANG != 0 && HWY_COMPILER_CLANG < 700) || \
    defined(HWY_DISABLE_AVX512)
// Clang6 encounters backend errors when compiling AVX-512.
#define HWY_RUNTIME_TARGETS (HWY_SSE4 | HWY_AVX2)
#else
#define HWY_RUNTIME_TARGETS (HWY_SSE4 | HWY_AVX2 | HWY_AVX512)
#endif

#elif HWY_ARCH == HWY_ARCH_WASM
#define HWY_RUNTIME_TARGETS HWY_WASM
#elif HWY_ARCH == HWY_ARCH_PPC
#define HWY_RUNTIME_TARGETS HWY_PPC8
#elif HWY_ARCH == HWY_ARCH_ARM
#define HWY_RUNTIME_TARGETS HWY_ARM8
#elif HWY_ARCH == HWY_ARCH_SCALAR
#define HWY_RUNTIME_TARGETS HWY_NONE
#else
#error "Unsupported platform"
#endif

#endif  // HWY_RUNTIME_TARGETS

//-----------------------------------------------------------------------------
// X macros for generating enum/code for each enabled target.

#define HWY_FOR_NONE HWY_X(NONE)

#if HWY_RUNTIME_TARGETS & HWY_ARM8
#define HWY_FOR_ARM8 HWY_X(ARM8)
#else
#define HWY_FOR_ARM8
#endif

#if HWY_RUNTIME_TARGETS & HWY_PPC8
#define HWY_FOR_PPC8 HWY_X(PPC8)
#else
#define HWY_FOR_PPC8
#endif

#if HWY_RUNTIME_TARGETS & HWY_SSE4
#define HWY_FOR_SSE4 HWY_X(SSE4)
#else
#define HWY_FOR_SSE4
#endif

#if HWY_RUNTIME_TARGETS & HWY_AVX2
#define HWY_FOR_AVX2 HWY_X(AVX2)
#else
#define HWY_FOR_AVX2
#endif

#if HWY_RUNTIME_TARGETS & HWY_AVX512
#define HWY_FOR_AVX512 HWY_X(AVX512)
#else
#define HWY_FOR_AVX512
#endif

#if HWY_RUNTIME_TARGETS & HWY_WASM
#define HWY_FOR_WASM HWY_X(WASM)
#else
#define HWY_FOR_WASM
#endif

#define HWY_FOREACH_TARGET \
  HWY_FOR_NONE             \
  HWY_FOR_WASM             \
  HWY_FOR_ARM8             \
  HWY_FOR_PPC8             \
  HWY_FOR_SSE4             \
  HWY_FOR_AVX2             \
  HWY_FOR_AVX512

//-----------------------------------------------------------------------------
// Special support for declaring target-specific member functions.
// This is easier for users than using via HWY_FOREACH_TARGET.

#define HWY_DECLARE_NONE(ret, args) ret F_NONE args;

#if HWY_RUNTIME_TARGETS & HWY_WASM
#define HWY_DECLARE_WASM(ret, args) ret F_WASM args;
#else
#define HWY_DECLARE_WASM(ret, args)
#endif

#if HWY_RUNTIME_TARGETS & HWY_ARM8
#define HWY_DECLARE_ARM8(ret, args) ret F_ARM8 args;
#else
#define HWY_DECLARE_ARM8(ret, args)
#endif

#if HWY_RUNTIME_TARGETS & HWY_PPC8
#define HWY_DECLARE_PPC8(ret, args) ret F_PPC8 args;
#else
#define HWY_DECLARE_PPC8(ret, args)
#endif

#if HWY_RUNTIME_TARGETS & HWY_SSE4
#define HWY_DECLARE_SSE4(ret, args) ret F_SSE4 args;
#else
#define HWY_DECLARE_SSE4(ret, args)
#endif

#if HWY_RUNTIME_TARGETS & HWY_AVX2
#define HWY_DECLARE_AVX2(ret, args) ret F_AVX2 args;
#else
#define HWY_DECLARE_AVX2(ret, args)
#endif

#if HWY_RUNTIME_TARGETS & HWY_AVX512
#define HWY_DECLARE_AVX512(ret, args) ret F_AVX512 args;
#else
#define HWY_DECLARE_AVX512(ret, args)
#endif

#define HWY_DECLARE(ret, args) \
  HWY_DECLARE_NONE(ret, args)  \
  HWY_DECLARE_WASM(ret, args)  \
  HWY_DECLARE_ARM8(ret, args)  \
  HWY_DECLARE_PPC8(ret, args)  \
  HWY_DECLARE_SSE4(ret, args)  \
  HWY_DECLARE_AVX2(ret, args)  \
  HWY_DECLARE_AVX512(ret, args)

//-----------------------------------------------------------------------------

#endif  // HWY_RUNTIME_TARGETS_H_
