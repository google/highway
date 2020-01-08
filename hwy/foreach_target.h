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

#ifndef HWY_FOREACH_TARGET_H_
#define HWY_FOREACH_TARGET_H_

// Includes a specified file for each target in HWY_RUNTIME_TARGETS. This
// generates a definition of Module::HWY_FUNC, called by runtime_dispatch.h.

#ifndef HWY_TARGET_INCLUDE
#error "Must set HWY_TARGET_INCLUDE before including foreach_target.h"
#endif

#include "hwy/runtime_targets.h"
// After runtime_targets:
#include "hwy/include_headers.h"

//-----------------------------------------------------------------------------
// SSE4
#if HWY_RUNTIME_TARGETS & HWY_SSE4

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_SSE4

#undef HWY_FUNC
#define HWY_FUNC F_SSE4

#undef HWY_ATTR
#define HWY_ATTR HWY_ATTR_SSE4

#undef HWY_BITS
#define HWY_BITS 128

#undef HWY_ALIGN
#define HWY_ALIGN alignas(16)

#undef HWY_HAS_CMP64
#define HWY_HAS_CMP64 0

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 0

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 0

#undef HWY_HAS_INT64
#define HWY_HAS_INT64 1

#undef HWY_HAS_DOUBLE
#define HWY_HAS_DOUBLE 1

#include HWY_TARGET_INCLUDE
#endif

//-----------------------------------------------------------------------------
// AVX2
#if HWY_RUNTIME_TARGETS & HWY_AVX2

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_AVX2

#undef HWY_FUNC
#define HWY_FUNC F_AVX2

#undef HWY_ATTR
#define HWY_ATTR HWY_ATTR_AVX2

#undef HWY_BITS
#define HWY_BITS 256

#undef HWY_ALIGN
#define HWY_ALIGN alignas(32)

#undef HWY_HAS_CMP64
#define HWY_HAS_CMP64 1

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 1

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 1

#undef HWY_HAS_INT64
#define HWY_HAS_INT64 1

#undef HWY_HAS_DOUBLE
#define HWY_HAS_DOUBLE 1

#include HWY_TARGET_INCLUDE
#endif

//-----------------------------------------------------------------------------
// AVX-512
#if HWY_RUNTIME_TARGETS & HWY_AVX512

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_AVX512

#undef HWY_FUNC
#define HWY_FUNC F_AVX512

#undef HWY_ATTR
#define HWY_ATTR HWY_ATTR_AVX512

#undef HWY_BITS
#define HWY_BITS 512

#undef HWY_ALIGN
#define HWY_ALIGN alignas(64)

#undef HWY_HAS_CMP64
#define HWY_HAS_CMP64 1

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 1

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 1

#undef HWY_HAS_INT64
#define HWY_HAS_INT64 1

#undef HWY_HAS_DOUBLE
#define HWY_HAS_DOUBLE 1

#include HWY_TARGET_INCLUDE
#endif

//-----------------------------------------------------------------------------
// PPC8
#if HWY_RUNTIME_TARGETS & HWY_PPC8

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_PPC8

#undef HWY_FUNC
#define HWY_FUNC F_PPC8

#undef HWY_ATTR
#define HWY_ATTR

#undef HWY_BITS
#define HWY_BITS 128

#undef HWY_ALIGN
#define HWY_ALIGN alignas(16)

#undef HWY_HAS_CMP64
#define HWY_HAS_CMP64 1

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 0

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 1

#undef HWY_HAS_INT64
#define HWY_HAS_INT64 1

#undef HWY_HAS_DOUBLE
#define HWY_HAS_DOUBLE 1

#include HWY_TARGET_INCLUDE
#endif

//-----------------------------------------------------------------------------
// ARM8
#if HWY_RUNTIME_TARGETS & HWY_ARM8

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_ARM8

#undef HWY_FUNC
#define HWY_FUNC F_ARM8

#undef HWY_ATTR
#define HWY_ATTR HWY_ATTR_ARM8

#undef HWY_BITS
#define HWY_BITS 128

#undef HWY_ALIGN
#define HWY_ALIGN alignas(16)

#undef HWY_HAS_CMP64
#undef HWY_HAS_DOUBLE
#undef HWY_HAS_INT64
#ifdef __arm__
#define HWY_HAS_CMP64 0
#define HWY_HAS_DOUBLE 0
#define HWY_HAS_INT64 0
#else
#define HWY_HAS_CMP64 1
#define HWY_HAS_DOUBLE 1
#define HWY_HAS_INT64 1
#endif

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 0

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 1

#include HWY_TARGET_INCLUDE
#endif

//-----------------------------------------------------------------------------
// WASM
#if HWY_RUNTIME_TARGETS & HWY_WASM

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_WASM

#undef HWY_FUNC
#define HWY_FUNC F_WASM

#undef HWY_ATTR
#define HWY_ATTR HWY_ATTR_WASM

#undef HWY_BITS
#define HWY_BITS 128

#undef HWY_ALIGN
#define HWY_ALIGN alignas(16)

#undef HWY_HAS_CMP64
#define HWY_HAS_CMP64 0

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 0

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 0

#undef HWY_HAS_INT64
#define HWY_HAS_INT64 0

#undef HWY_HAS_DOUBLE
#define HWY_HAS_DOUBLE 0

#include HWY_TARGET_INCLUDE
#endif

//-----------------------------------------------------------------------------
// NONE

#undef HWY_NAMESPACE
#define HWY_NAMESPACE N_NONE

#undef HWY_FUNC
#define HWY_FUNC F_NONE

#undef HWY_ATTR
#define HWY_ATTR

#undef HWY_BITS
#define HWY_BITS 0

#undef HWY_ALIGN
#define HWY_ALIGN

#undef HWY_HAS_CMP64
#define HWY_HAS_CMP64 1

#undef HWY_HAS_GATHER
#define HWY_HAS_GATHER 1

#undef HWY_HAS_VARIABLE_SHIFT
#define HWY_HAS_VARIABLE_SHIFT 1

#undef HWY_HAS_INT64
#define HWY_HAS_INT64 1

#undef HWY_HAS_DOUBLE
#define HWY_HAS_DOUBLE 1

// This is followed by the implementation part of the original CC file, so no
// need to include again. NONE must be last because it is unconditional.

#endif  // #ifndef HWY_FOREACH_TARGET_H_
