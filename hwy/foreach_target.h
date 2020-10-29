// Copyright 2020 Google LLC
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

// Re-includes the translation unit zero or more times to compile for any
// targets except HWY_STATIC_TARGET. Defines unique HWY_TARGET each time so that
// highway.h defines the corresponding macro/namespace.

#ifdef HWY_HIGHWAY_H_
#error "Must include foreach_target.h before highway.h."
#endif

#include "hwy/targets.h"

// Avoid warnings on #include HWY_TARGET_INCLUDE by hiding them from the IDE;
// also skip if only 1 target defined (no re-inclusion will be necessary).
#if !HWY_IDE && (HWY_TARGETS != HWY_STATIC_TARGET)

#if !defined(HWY_TARGET_INCLUDE)
#error ">1 target enabled => define HWY_TARGET_INCLUDE before foreach_target.h"
#endif

// *_inl.h may include other headers, which requires include guards to prevent
// repeated inclusion. The guards must be reset after compiling each target, so
// the header is again visible. This is done by flipping HWY_TARGET_TOGGLE,
// defining it if undefined and vice versa. This macro is initially undefined
// so that IDEs don't gray out the contents of each header.
#ifdef HWY_TARGET_TOGGLE
#error "This macro must not be defined outside foreach_target.h"
#endif

#if (HWY_TARGETS & HWY_SCALAR) && (HWY_STATIC_TARGET != HWY_SCALAR)
#undef HWY_TARGET
#define HWY_TARGET HWY_SCALAR
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_NEON) && (HWY_STATIC_TARGET != HWY_NEON)
#undef HWY_TARGET
#define HWY_TARGET HWY_NEON
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_SSE4) && (HWY_STATIC_TARGET != HWY_SSE4)
#undef HWY_TARGET
#define HWY_TARGET HWY_SSE4
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_AVX2) && (HWY_STATIC_TARGET != HWY_AVX2)
#undef HWY_TARGET
#define HWY_TARGET HWY_AVX2
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_AVX3) && (HWY_STATIC_TARGET != HWY_AVX3)
#undef HWY_TARGET
#define HWY_TARGET HWY_AVX3
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_WASM) && (HWY_STATIC_TARGET != HWY_WASM)
#undef HWY_TARGET
#define HWY_TARGET HWY_WASM
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#if (HWY_TARGETS & HWY_PPC8) && (HWY_STATIC_TARGET != HWY_PPC8)
#undef HWY_TARGET
#define HWY_TARGET HWY_PPC8
#include HWY_TARGET_INCLUDE
#ifdef HWY_TARGET_TOGGLE
#undef HWY_TARGET_TOGGLE
#else
#define HWY_TARGET_TOGGLE
#endif
#endif

#endif  // !HWY_IDE && (HWY_TARGETS != HWY_STATIC_TARGET)

// If we re-include once per enabled target, the translation unit's
// implementation would have to be skipped via #if to avoid redefining symbols.
// We instead skip the re-include for HWY_STATIC_TARGET, and generate its
// implementation when resuming compilation of the translation unit. Reverting
// to the initial value of HWY_TARGET also causes HWY_ONCE to expand to 1.
#undef HWY_TARGET
#define HWY_TARGET HWY_STATIC_TARGET

#endif  // HWY_FOREACH_TARGET_H_
