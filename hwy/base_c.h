// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
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

#ifndef HIGHWAY_HWY_BASE_C_H_
#define HIGHWAY_HWY_BASE_C_H_

// Target-independent, macro-only definitions that are also usable from C
// (C99 or later), e.g. for extern "C" library interfaces. base.h includes this
// header and adds the C++ definitions. Note that HWY_INLINE and HWY_API rely
// on C99 `inline` semantics when used from C; in C headers, prefer HWY_API or
// `static HWY_INLINE` because bare HWY_INLINE has non-internal linkage.

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include "hwy/detect_compiler_arch.h"
#include "hwy/highway_export.h"

// API version (https://semver.org/); keep in sync with CMakeLists.txt and
// meson.build.
#define HWY_MAJOR 1
#define HWY_MINOR 4
#define HWY_PATCH 0

// True if the Highway version >= major.minor.0. Added in 1.2.0.
#define HWY_VERSION_GE(major, minor) \
  (HWY_MAJOR > (major) || (HWY_MAJOR == (major) && HWY_MINOR >= (minor)))
// True if the Highway version < major.minor.0. Added in 1.2.0.
#define HWY_VERSION_LT(major, minor) \
  (HWY_MAJOR < (major) || (HWY_MAJOR == (major) && HWY_MINOR < (minor)))

// "IWYU pragma: keep" does not work for these includes, so hide from the IDE.
#if !HWY_IDE

#if !defined(HWY_NO_LIBCXX)
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS  // before inttypes.h
#endif
#include <inttypes.h>
#endif

#endif  // !HWY_IDE
// IWYU pragma: end_exports

//------------------------------------------------------------------------------
// Compiler-specific definitions

#define HWY_STR_IMPL(macro) #macro
#define HWY_STR(macro) HWY_STR_IMPL(macro)

#if HWY_COMPILER_MSVC

#define HWY_FUNCTION __FUNCSIG__  // function name + template args
#define HWY_RESTRICT __restrict
#define HWY_INLINE_ATTR
#define HWY_INLINE __forceinline
#define HWY_NOINLINE __declspec(noinline)
#define HWY_FLATTEN
#define HWY_PRESERVE_MOST
#define HWY_NORETURN __declspec(noreturn)
#define HWY_LIKELY(expr) (expr)
#define HWY_UNLIKELY(expr) (expr)
#define HWY_UNREACHABLE __assume(0)
#define HWY_PRAGMA(tokens) __pragma(tokens)
#define HWY_DIAGNOSTICS(tokens) HWY_PRAGMA(warning(tokens))
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(msc)
#define HWY_MAYBE_UNUSED
#if (_MSC_VER >= 1700)
#define HWY_MUST_USE_RESULT _Check_return_
#else
#define HWY_MUST_USE_RESULT
#endif

#else

#ifdef __cplusplus
#define HWY_FUNCTION __PRETTY_FUNCTION__  // function name + template args
#else
#define HWY_FUNCTION __func__
#endif
#define HWY_RESTRICT __restrict__
// force inlining without optimization enabled creates very inefficient code
// that can cause compiler timeout
#ifdef __OPTIMIZE__
#define HWY_INLINE_ATTR __attribute__((always_inline))
#else
#define HWY_INLINE_ATTR
#endif
#define HWY_INLINE inline HWY_INLINE_ATTR
#define HWY_NOINLINE __attribute__((noinline))
#define HWY_FLATTEN __attribute__((flatten))
// Clang currently only supports this on x86 and Arm, but more are expected.
#if HWY_HAS_ATTRIBUTE(preserve_most) && (HWY_ARCH_X86_64 || HWY_ARCH_ARM_A64)
#define HWY_PRESERVE_MOST __attribute__((preserve_most))
#else
#define HWY_PRESERVE_MOST
#endif
#define HWY_NORETURN __attribute__((noreturn))
#define HWY_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define HWY_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#if HWY_COMPILER_GCC || HWY_HAS_BUILTIN(__builtin_unreachable)
#define HWY_UNREACHABLE __builtin_unreachable()
#else
#define HWY_UNREACHABLE
#endif
#define HWY_PRAGMA(tokens) _Pragma(#tokens)
#define HWY_DIAGNOSTICS(tokens) HWY_PRAGMA(GCC diagnostic tokens)
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(gcc)
// Encountered "attribute list cannot appear here" when using the C++17
// [[maybe_unused]], so only use the old style attribute for now.
#define HWY_MAYBE_UNUSED __attribute__((unused))
// HWY_NODISCARD is preferred, see comment there.
#define HWY_MUST_USE_RESULT __attribute__((warn_unused_result))

#endif  // !HWY_COMPILER_MSVC

// Reduce inlining in sanitizer builds: shorter compile and less stack space.
#if HWY_IS_ASAN || HWY_IS_HWASAN || HWY_IS_MSAN || HWY_IS_TSAN
#define HWY_MAYBE_INLINE inline
#else
#define HWY_MAYBE_INLINE HWY_INLINE
#endif

#if (HWY_COMPILER_GCC_ACTUAL && HWY_COMPILER_GCC_ACTUAL < 1200) || \
    (HWY_COMPILER_ICC && !HWY_COMPILER_ICX)
// The use of __attribute__((unused)) in private class member variables triggers
// a compiler warning with GCC 11 and earlier and ICC

// GCC 11 and earlier and ICC also do not emit -Wunused-private-field warnings
// for unused private class member variables
#define HWY_MEMBER_VAR_MAYBE_UNUSED
#else
// Clang and ICX need __attribute__((unused)) in unused private class member
// variables to suppress -Wunused-private-field warnings unless this warning is
// ignored by using HWY_DIAGNOSTICS_OFF
#define HWY_MEMBER_VAR_MAYBE_UNUSED HWY_MAYBE_UNUSED
#endif

//------------------------------------------------------------------------------
// Builtin/attributes

// Enables error-checking of format strings.
#if HWY_HAS_ATTRIBUTE(__format__)
#define HWY_FORMAT(idx_fmt, idx_arg) \
  __attribute__((__format__(__printf__, idx_fmt, idx_arg)))
#else
#define HWY_FORMAT(idx_fmt, idx_arg)
#endif

// Returns a void* pointer which the compiler then assumes is N-byte aligned.
// Example: float* HWY_RESTRICT aligned = (float*)HWY_ASSUME_ALIGNED(in, 32);
//
// The assignment semantics are required by GCC/Clang. ICC provides an in-place
// __assume_aligned, whereas MSVC's __assume appears unsuitable.
#if HWY_HAS_BUILTIN(__builtin_assume_aligned)
#define HWY_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
#define HWY_ASSUME_ALIGNED(ptr, align) (ptr) /* not supported */
#endif

// Clang and GCC require attributes on each function into which SIMD intrinsics
// are inlined. Support both per-function annotation (HWY_ATTR) for lambdas and
// automatic annotation via pragmas.
#if HWY_COMPILER_ICC
// As of ICC 2021.{1-9} the pragma is neither implemented nor required.
#define HWY_PUSH_ATTRIBUTES(targets_str)
#define HWY_POP_ATTRIBUTES
#elif HWY_COMPILER_CLANG
#define HWY_PUSH_ATTRIBUTES(targets_str)                                \
  HWY_PRAGMA(clang attribute push(__attribute__((target(targets_str))), \
                                  apply_to = function))
#define HWY_POP_ATTRIBUTES HWY_PRAGMA(clang attribute pop)
#elif HWY_COMPILER_GCC_ACTUAL
#define HWY_PUSH_ATTRIBUTES(targets_str) \
  HWY_PRAGMA(GCC push_options) HWY_PRAGMA(GCC target targets_str)
#define HWY_POP_ATTRIBUTES HWY_PRAGMA(GCC pop_options)
#else
#define HWY_PUSH_ATTRIBUTES(targets_str)
#define HWY_POP_ATTRIBUTES
#endif

//------------------------------------------------------------------------------
// Macros

// Warns if the return value (or an object of the annotated class) is ignored.
// C++17 [[nodiscard]] is a warning (clang -Wc++17-extensions, MSVC C5051) in
// C++14 mode, hence fall back to the older function-only attribute.
//
// HWY_NODISCARD vs. HWY_MUST_USE_RESULT: generally prefer HWY_NODISCARD: it is
// always active on MSVC, whereas HWY_MUST_USE_RESULT is only active in
// /analyze builds, and also allows suppressing the warning via (void) on GCC,
// but cannot be used on function pointer typedefs, for which
// HWY_MUST_USE_RESULT should be used instead.
#if (HWY_CXX_LANG >= 201703L) && HWY_HAS_CPP_ATTRIBUTE(nodiscard)
#define HWY_NODISCARD [[nodiscard]]
#define HWY_NODISCARD_CLASS [[nodiscard]]
#else
#define HWY_NODISCARD HWY_MUST_USE_RESULT
#define HWY_NODISCARD_CLASS
#endif

// Note: it is safe to remove `static` for users who want to use modules, but
// that might be a breaking change for some users, hence we do not by default.
#define HWY_API static HWY_INLINE HWY_FLATTEN

#define HWY_CONCAT_IMPL(a, b) a##b
#define HWY_CONCAT(a, b) HWY_CONCAT_IMPL(a, b)

#define HWY_MIN(a, b) ((a) < (b) ? (a) : (b))
#define HWY_MAX(a, b) ((a) > (b) ? (a) : (b))

#if HWY_COMPILER_GCC_ACTUAL
// nielskm: GCC does not support '#pragma GCC unroll' without the factor.
#define HWY_UNROLL(factor) HWY_PRAGMA(GCC unroll factor)
#define HWY_DEFAULT_UNROLL HWY_UNROLL(4)
#elif HWY_COMPILER_CLANG || HWY_COMPILER_ICC || HWY_COMPILER_ICX
#define HWY_UNROLL(factor) HWY_PRAGMA(unroll factor)
#define HWY_DEFAULT_UNROLL HWY_UNROLL()
#else
#define HWY_UNROLL(factor)
#define HWY_DEFAULT_UNROLL
#endif

#if HWY_COMPILER_CLANG
#define HWY_DISABLE_AUTOVEC HWY_PRAGMA(clang loop vectorize(disable))
#else
#define HWY_DISABLE_AUTOVEC
#endif

// Tell a compiler that the expression always evaluates to true.
// The expression should be free from any side effects.
// Some older compilers may have trouble with complex expressions, therefore
// it is advisable to split multiple conditions into separate assume statements,
// and manually check the generated code.
// OK but could fail:
//   HWY_ASSUME(x == 2 && y == 3);
// Better:
//   HWY_ASSUME(x == 2);
//   HWY_ASSUME(y == 3);
#if (HWY_CXX_LANG >= 202302L) && HWY_HAS_CPP_ATTRIBUTE(assume)
#define HWY_ASSUME(expr) [[assume(expr)]]
#elif HWY_COMPILER_MSVC || HWY_COMPILER_ICC
#define HWY_ASSUME(expr) __assume(expr)
// __builtin_assume() was added in clang 3.6.
#elif HWY_COMPILER_CLANG && HWY_HAS_BUILTIN(__builtin_assume)
#define HWY_ASSUME(expr) __builtin_assume(expr)
// __builtin_unreachable() was added in GCC 4.5, but __has_builtin() was added
// later, so check for the compiler version directly.
#elif HWY_COMPILER_GCC_ACTUAL >= 405
#ifdef __cplusplus
#define HWY_ASSUME(expr) \
  ((expr) ? static_cast<void>(0) : __builtin_unreachable())
#else
#define HWY_ASSUME(expr) ((expr) ? (void)0 : __builtin_unreachable())
#endif
#else
#ifdef __cplusplus
#define HWY_ASSUME(expr) static_cast<void>(0)
#else
#define HWY_ASSUME(expr) ((void)0)
#endif
#endif

// 4 instances of a given literal value, useful as input to LoadDup128.
#define HWY_REP4(literal) literal, literal, literal, literal

//------------------------------------------------------------------------------
// Debug builds

// For enabling HWY_DASSERT and shortening tests in slower debug builds
//
// Note: `HWY_IS_UBSAN` is specifically excluded from engaging debug
// builds. This is in service of Chromium's `-fsanitize=array-bounds` by
// default, where we don't want Highway to unconditionally build in
// debug mode.
//
// See also:
// https://docs.google.com/document/d/1eCtY4AZF-SiFHxhIYWzEytdIx3C24de7ccD6Y5Gn2H8/edit?tab=t.9zkn85hr82ms#heading=h.efcshvfql42c
#if !defined(HWY_IS_DEBUG_BUILD)
// Clang does not define NDEBUG, but it and GCC define __OPTIMIZE__, and recent
// MSVC defines NDEBUG (if not, could instead check _DEBUG).
#if (!defined(__OPTIMIZE__) && !defined(NDEBUG)) ||         \
    (HWY_IS_ASAN || (HWY_IS_SANITIZER && !HWY_IS_UBSAN)) || \
    defined(__clang_analyzer__)
#define HWY_IS_DEBUG_BUILD 1
#else
#define HWY_IS_DEBUG_BUILD 0
#endif
#endif  // HWY_IS_DEBUG_BUILD

#endif  // HIGHWAY_HWY_BASE_C_H_
