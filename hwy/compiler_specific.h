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

#ifndef HWY_COMPILER_SPECIFIC_H_
#define HWY_COMPILER_SPECIFIC_H_

// Compiler-specific includes and definitions.

#ifdef _MSC_VER
#define HWY_COMPILER_MSVC _MSC_VER
#else
#define HWY_COMPILER_MSVC 0
#endif

#ifdef __GNUC__
#define HWY_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define HWY_COMPILER_GCC 0
#endif

// Clang can masquerade as MSVC/GCC, in which case both are set.
#ifdef __clang__
#define HWY_COMPILER_CLANG (__clang_major__ * 100 + __clang_minor__)
#else
#define HWY_COMPILER_CLANG 0
#endif

#if !HWY_COMPILER_MSVC && !HWY_COMPILER_GCC && !HWY_COMPILER_CLANG
#error "Unsupported compiler"
#endif

#if HWY_COMPILER_MSVC
#include <intrin.h>

#define HWY_RESTRICT __restrict
#define HWY_INLINE __forceinline
#define HWY_NOINLINE __declspec(noinline)
#define HWY_LIKELY(expr) expr
#define HWY_TRAP __debugbreak
#define HWY_TARGET_ATTR(feature_str)
#define HWY_DIAGNOSTICS(tokens) __pragma(warning(tokens))
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(msc)

#else

#define HWY_RESTRICT __restrict__
#define HWY_INLINE \
  inline __attribute__((always_inline)) __attribute__((flatten))
#define HWY_NOINLINE inline __attribute__((noinline))
#define HWY_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define HWY_TRAP __builtin_trap
#define HWY_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#define HWY_PRAGMA(tokens) _Pragma(#tokens)
#define HWY_DIAGNOSTICS(tokens) HWY_PRAGMA(GCC diagnostic tokens)
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(gcc)

#endif

// Add to #if conditions to prevent IDE from graying out code.
#if (defined __CDT_PARSER__) || (defined __INTELLISENSE__) || \
    (defined Q_CREATOR_RUN)
#define HWY_IDE 1
#else
#define HWY_IDE 0
#endif

// Clang 3.9 generates VINSERTF128 instead of the desired VBROADCASTF128,
// which would free up port5. However, inline assembly isn't supported on
// MSVC, results in incorrect output on GCC 8.3, and raises "invalid output size
// for constraint" errors on Clang (https://gcc.godbolt.org/z/-Jt_-F), hence we
// disable it.
#define HWY_LOADDUP_ASM 0

#endif  // HWY_COMPILER_SPECIFIC_H_
