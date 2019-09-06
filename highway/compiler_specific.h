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

#ifndef HIGHWAY_COMPILER_SPECIFIC_H_
#define HIGHWAY_COMPILER_SPECIFIC_H_

// Compiler-specific includes and definitions.

// SIMD_COMPILER expands to one of the following:
#define SIMD_COMPILER_CLANG 1
#define SIMD_COMPILER_GCC 2
#define SIMD_COMPILER_MSVC 3

#ifdef _MSC_VER
#define SIMD_COMPILER SIMD_COMPILER_MSVC
#elif defined(__clang__)
#define SIMD_COMPILER SIMD_COMPILER_CLANG
#elif defined(__GNUC__)
#define SIMD_COMPILER SIMD_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif

#if SIMD_COMPILER == SIMD_COMPILER_MSVC
#include <intrin.h>

#define SIMD_RESTRICT __restrict
#define SIMD_INLINE __forceinline
#define SIMD_NOINLINE __declspec(noinline)
#define SIMD_LIKELY(expr) expr
#define SIMD_TRAP __debugbreak
#define SIMD_TARGET_ATTR(feature_str)
#define SIMD_DIAGNOSTICS(tokens) __pragma(warning(tokens))
#define SIMD_DIAGNOSTICS_OFF(msc, gcc) SIMD_DIAGNOSTICS(msc)

#else

#define SIMD_RESTRICT __restrict__
#define SIMD_INLINE \
  inline __attribute__((always_inline)) __attribute__((flatten))
#define SIMD_NOINLINE inline __attribute__((noinline))
#define SIMD_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define SIMD_TRAP __builtin_trap
#define SIMD_TARGET_ATTR(feature_str) __attribute__((target(feature_str)))
#define SIMD_PRAGMA(tokens) _Pragma(#tokens)
#define SIMD_DIAGNOSTICS(tokens) SIMD_PRAGMA(GCC diagnostic tokens)
#define SIMD_DIAGNOSTICS_OFF(msc, gcc) SIMD_DIAGNOSTICS(gcc)

#endif

#endif  // HIGHWAY_COMPILER_SPECIFIC_H_
