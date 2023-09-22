// Copyright 2020 Google LLC
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

#ifndef HIGHWAY_HWY_BASE_H_
#define HIGHWAY_HWY_BASE_H_

// For SIMD module implementations and their callers, target-independent.

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

// Wrapping this into a HWY_HAS_INCLUDE causes clang-format to fail.
#if __cplusplus >= 202100L && defined(__has_include)
#if __has_include(<stdfloat>)
#include <stdfloat>  // std::float16_t
#endif
#endif

#include "hwy/detect_compiler_arch.h"
#include "hwy/highway_export.h"

// "IWYU pragma: keep" does not work for these includes, so hide from the IDE.
#if !HWY_IDE

#if !defined(HWY_NO_LIBCXX)
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS  // before inttypes.h
#endif
#include <inttypes.h>
#endif

#if (HWY_ARCH_X86 && !defined(HWY_NO_LIBCXX)) || HWY_COMPILER_MSVC
#include <atomic>
#endif

#endif  // !HWY_IDE

// IWYU pragma: end_exports

#if HWY_COMPILER_MSVC
#include <string.h>  // memcpy
#endif

//------------------------------------------------------------------------------
// Compiler-specific definitions

#define HWY_STR_IMPL(macro) #macro
#define HWY_STR(macro) HWY_STR_IMPL(macro)

#if HWY_COMPILER_MSVC

#include <intrin.h>

#define HWY_RESTRICT __restrict
#define HWY_INLINE __forceinline
#define HWY_NOINLINE __declspec(noinline)
#define HWY_FLATTEN
#define HWY_NORETURN __declspec(noreturn)
#define HWY_LIKELY(expr) (expr)
#define HWY_UNLIKELY(expr) (expr)
#define HWY_PRAGMA(tokens) __pragma(tokens)
#define HWY_DIAGNOSTICS(tokens) HWY_PRAGMA(warning(tokens))
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(msc)
#define HWY_MAYBE_UNUSED
#define HWY_HAS_ASSUME_ALIGNED 0
#if (_MSC_VER >= 1700)
#define HWY_MUST_USE_RESULT _Check_return_
#else
#define HWY_MUST_USE_RESULT
#endif

#else

#define HWY_RESTRICT __restrict__
// force inlining without optimization enabled creates very inefficient code
// that can cause compiler timeout
#ifdef __OPTIMIZE__
#define HWY_INLINE inline __attribute__((always_inline))
#else
#define HWY_INLINE inline
#endif
#define HWY_NOINLINE __attribute__((noinline))
#define HWY_FLATTEN __attribute__((flatten))
#define HWY_NORETURN __attribute__((noreturn))
#define HWY_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define HWY_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define HWY_PRAGMA(tokens) _Pragma(#tokens)
#define HWY_DIAGNOSTICS(tokens) HWY_PRAGMA(GCC diagnostic tokens)
#define HWY_DIAGNOSTICS_OFF(msc, gcc) HWY_DIAGNOSTICS(gcc)
// Encountered "attribute list cannot appear here" when using the C++17
// [[maybe_unused]], so only use the old style attribute for now.
#define HWY_MAYBE_UNUSED __attribute__((unused))
#define HWY_MUST_USE_RESULT __attribute__((warn_unused_result))

#endif  // !HWY_COMPILER_MSVC

//------------------------------------------------------------------------------
// Builtin/attributes (no more #include after this point due to namespace!)

namespace hwy {

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

#define HWY_API static HWY_INLINE HWY_FLATTEN HWY_MAYBE_UNUSED

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
#if HWY_HAS_CPP_ATTRIBUTE(assume)
#define HWY_ASSUME(expr) [[assume(expr)]]
#elif HWY_COMPILER_MSVC || HWY_COMPILER_ICC
#define HWY_ASSUME(expr) __assume(expr)
// __builtin_assume() was added in clang 3.6.
#elif HWY_COMPILER_CLANG && HWY_HAS_BUILTIN(__builtin_assume)
#define HWY_ASSUME(expr) __builtin_assume(expr)
// __builtin_unreachable() was added in GCC 4.5, but __has_builtin() was added
// later, so check for the compiler version directly.
#elif HWY_COMPILER_GCC_ACTUAL >= 405
#define HWY_ASSUME(expr) \
  ((expr) ? static_cast<void>(0) : __builtin_unreachable())
#else
#define HWY_ASSUME(expr) static_cast<void>(0)
#endif

// Compile-time fence to prevent undesirable code reordering. On Clang x86, the
// typical asm volatile("" : : : "memory") has no effect, whereas atomic fence
// does, without generating code.
#if HWY_ARCH_X86 && !defined(HWY_NO_LIBCXX)
#define HWY_FENCE std::atomic_thread_fence(std::memory_order_acq_rel)
#else
// TODO(janwas): investigate alternatives. On Arm, the above generates barriers.
#define HWY_FENCE
#endif

// 4 instances of a given literal value, useful as input to LoadDup128.
#define HWY_REP4(literal) literal, literal, literal, literal

HWY_DLLEXPORT HWY_NORETURN void HWY_FORMAT(3, 4)
    Abort(const char* file, int line, const char* format, ...);

#define HWY_ABORT(format, ...) \
  ::hwy::Abort(__FILE__, __LINE__, format, ##__VA_ARGS__)

// Always enabled.
#define HWY_ASSERT(condition)             \
  do {                                    \
    if (!(condition)) {                   \
      HWY_ABORT("Assert %s", #condition); \
    }                                     \
  } while (0)

#if HWY_HAS_FEATURE(memory_sanitizer) || defined(MEMORY_SANITIZER)
#define HWY_IS_MSAN 1
#else
#define HWY_IS_MSAN 0
#endif

#if HWY_HAS_FEATURE(address_sanitizer) || defined(ADDRESS_SANITIZER)
#define HWY_IS_ASAN 1
#else
#define HWY_IS_ASAN 0
#endif

#if HWY_HAS_FEATURE(thread_sanitizer) || defined(THREAD_SANITIZER)
#define HWY_IS_TSAN 1
#else
#define HWY_IS_TSAN 0
#endif

// MSAN may cause lengthy build times or false positives e.g. in AVX3 DemoteTo.
// You can disable MSAN by adding this attribute to the function that fails.
#if HWY_IS_MSAN
#define HWY_ATTR_NO_MSAN __attribute__((no_sanitize_memory))
#else
#define HWY_ATTR_NO_MSAN
#endif

// For enabling HWY_DASSERT and shortening tests in slower debug builds
#if !defined(HWY_IS_DEBUG_BUILD)
// Clang does not define NDEBUG, but it and GCC define __OPTIMIZE__, and recent
// MSVC defines NDEBUG (if not, could instead check _DEBUG).
#if (!defined(__OPTIMIZE__) && !defined(NDEBUG)) || HWY_IS_ASAN || \
    HWY_IS_MSAN || HWY_IS_TSAN || defined(__clang_analyzer__)
#define HWY_IS_DEBUG_BUILD 1
#else
#define HWY_IS_DEBUG_BUILD 0
#endif
#endif  // HWY_IS_DEBUG_BUILD

#if HWY_IS_DEBUG_BUILD
#define HWY_DASSERT(condition) HWY_ASSERT(condition)
#else
#define HWY_DASSERT(condition) \
  do {                         \
  } while (0)
#endif

//------------------------------------------------------------------------------
// CopyBytes / ZeroBytes

#if HWY_COMPILER_MSVC
#pragma intrinsic(memcpy)
#pragma intrinsic(memset)
#endif

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
HWY_API void CopyBytes(const From* from, To* to) {
#if HWY_COMPILER_MSVC
  memcpy(to, from, kBytes);
#else
  __builtin_memcpy(to, from, kBytes);
#endif
}

HWY_API void CopyBytes(const void* HWY_RESTRICT from, void* HWY_RESTRICT to,
                       size_t num_of_bytes_to_copy) {
#if HWY_COMPILER_MSVC
  memcpy(to, from, num_of_bytes_to_copy);
#else
  __builtin_memcpy(to, from, num_of_bytes_to_copy);
#endif
}

// Same as CopyBytes, but for same-sized objects; avoids a size argument.
template <typename From, typename To>
HWY_API void CopySameSize(const From* HWY_RESTRICT from, To* HWY_RESTRICT to) {
  static_assert(sizeof(From) == sizeof(To), "");
  CopyBytes<sizeof(From)>(from, to);
}

template <size_t kBytes, typename To>
HWY_API void ZeroBytes(To* to) {
#if HWY_COMPILER_MSVC
  memset(to, 0, kBytes);
#else
  __builtin_memset(to, 0, kBytes);
#endif
}

HWY_API void ZeroBytes(void* to, size_t num_bytes) {
#if HWY_COMPILER_MSVC
  memset(to, 0, num_bytes);
#else
  __builtin_memset(to, 0, num_bytes);
#endif
}

// -----------------------------------------------------------------------------
// BitCastScalar

#if HWY_HAS_BUILTIN(__builtin_bit_cast) || HWY_COMPILER_MSVC >= 1926
#define HWY_BITCASTSCALAR_CONSTEXPR constexpr
#else
#define HWY_BITCASTSCALAR_CONSTEXPR
#endif

template <class To, class From>
HWY_API HWY_BITCASTSCALAR_CONSTEXPR To BitCastScalar(const From& val) {
#if HWY_HAS_BUILTIN(__builtin_bit_cast) || HWY_COMPILER_MSVC >= 1926
  return __builtin_bit_cast(To, val);
#else
  To result;
  CopySameSize(&val, &result);
  return result;
#endif
}

//------------------------------------------------------------------------------
// kMaxVectorSize (undocumented, pending removal)

#if HWY_ARCH_X86
static constexpr HWY_MAYBE_UNUSED size_t kMaxVectorSize = 64;  // AVX-512
#elif HWY_ARCH_RVV && defined(__riscv_v_intrinsic) && \
    __riscv_v_intrinsic >= 11000
// Not actually an upper bound on the size.
static constexpr HWY_MAYBE_UNUSED size_t kMaxVectorSize = 4096;
#else
static constexpr HWY_MAYBE_UNUSED size_t kMaxVectorSize = 16;
#endif

//------------------------------------------------------------------------------
// Alignment

// Potentially useful for LoadDup128 and capped vectors. In other cases, arrays
// should be allocated dynamically via aligned_allocator.h because Lanes() may
// exceed the stack size.
#if HWY_ARCH_X86
#define HWY_ALIGN_MAX alignas(64)
#elif HWY_ARCH_RVV && defined(__riscv_v_intrinsic) && \
    __riscv_v_intrinsic >= 11000
#define HWY_ALIGN_MAX alignas(8)  // only elements need be aligned
#else
#define HWY_ALIGN_MAX alignas(16)
#endif

//------------------------------------------------------------------------------
// Lane types

#pragma pack(push, 1)

// float16_t load/store/conversion intrinsics are always supported on Armv8 and
// VFPv4 (except with MSVC). On Armv7 Clang requires __ARM_FP & 2; GCC requires
// -mfp16-format=ieee.
#if (HWY_ARCH_ARM_A64 && !HWY_COMPILER_MSVC) ||                    \
    (HWY_COMPILER_CLANG && defined(__ARM_FP) && (__ARM_FP & 2)) || \
    (HWY_COMPILER_GCC_ACTUAL && defined(__ARM_FP16_FORMAT_IEEE))
#define HWY_NEON_HAVE_FLOAT16C 1
#else
#define HWY_NEON_HAVE_FLOAT16C 0
#endif

// C11 extension ISO/IEC TS 18661-3:2015 but not supported on all targets.
// Required if HWY_HAVE_FLOAT16, i.e. RVV with zvfh or AVX3_SPR (with
// sufficiently new compiler supporting avx512fp16). Do not use on clang-cl,
// which is missing __extendhfsf2.
#if ((HWY_ARCH_RVV && defined(__riscv_zvfh) && HWY_COMPILER_CLANG) || \
     (HWY_ARCH_X86 && defined(__SSE2__) &&                            \
      ((HWY_COMPILER_CLANG >= 1600 && !HWY_COMPILER_CLANGCL) ||       \
       HWY_COMPILER_GCC_ACTUAL >= 1200)))
#define HWY_HAVE_C11_FLOAT16 1
#else
#define HWY_HAVE_C11_FLOAT16 0
#endif

// If 1, both __bf16 and a limited set of *_bf16 SVE intrinsics are available:
// create/get/set/dup, ld/st, sel, rev, trn, uzp, zip.
#if HWY_ARCH_ARM_A64 && defined(__ARM_FEATURE_SVE_BF16)
#define HWY_SVE_HAVE_BFLOAT16 1
#else
#define HWY_SVE_HAVE_BFLOAT16 0
#endif

// Match [u]int##_t naming scheme so rvv-inl.h macros can obtain the type name
// by concatenating base type and bits. We use a wrapper class instead of a
// typedef to the native type to ensure that the same symbols, e.g. for VQSort,
// are generated regardless of F16 support; see #1684.
struct float16_t {
#if HWY_NEON_HAVE_FLOAT16C  // ACLE's __fp16
  using Raw = __fp16;
#elif HWY_HAVE_C11_FLOAT16                                    // C11 _Float16
  using Raw = _Float16;
#elif __cplusplus > 202002L && defined(__STDCPP_FLOAT16_T__)  // C++23
  using Raw = std::float16_t;
#else
#define HWY_EMULATE_FLOAT16
  using Raw = uint16_t;
  Raw bits;
#endif  // float16_t

// When backed by a native type, ensure the wrapper behaves like the native
// type by forwarding all operators. Unfortunately it seems difficult to reuse
// this code in a base class, so we repeat it in bfloat16_t.
#ifndef HWY_EMULATE_FLOAT16
  Raw raw;

  float16_t() noexcept = default;
  template <typename T>
  constexpr float16_t(T arg) noexcept : raw(static_cast<Raw>(arg)) {}
  float16_t& operator=(Raw arg) noexcept {
    raw = arg;
    return *this;
  }
  constexpr float16_t(const float16_t&) noexcept = default;
  float16_t& operator=(const float16_t&) noexcept = default;
  constexpr operator Raw() const noexcept { return raw; }

  template <typename T>
  float16_t& operator+=(T rhs) noexcept {
    raw = static_cast<Raw>(raw + rhs);
    return *this;
  }

  template <typename T>
  float16_t& operator-=(T rhs) noexcept {
    raw = static_cast<Raw>(raw - rhs);
    return *this;
  }

  template <typename T>
  float16_t& operator*=(T rhs) noexcept {
    raw = static_cast<Raw>(raw * rhs);
    return *this;
  }

  template <typename T>
  float16_t& operator/=(T rhs) noexcept {
    raw = static_cast<Raw>(raw / rhs);
    return *this;
  }

  // pre-decrement operator (--x)
  float16_t& operator--() noexcept {
    raw = static_cast<Raw>(raw - Raw{1});
    return *this;
  }

  // post-decrement operator (x--)
  float16_t operator--(int) noexcept {
    float16_t result = *this;
    raw = static_cast<Raw>(raw - Raw{1});
    return result;
  }

  // pre-increment operator (++x)
  float16_t& operator++() noexcept {
    raw = static_cast<Raw>(raw + Raw{1});
    return *this;
  }

  // post-increment operator (x++)
  float16_t operator++(int) noexcept {
    float16_t result = *this;
    raw = static_cast<Raw>(raw + Raw{1});
    return result;
  }

  constexpr float16_t operator-() const noexcept {
    return float16_t(static_cast<Raw>(-raw));
  }
  constexpr float16_t operator+() const noexcept { return *this; }
#endif  // HWY_EMULATE_FLOAT16
};

#ifndef HWY_EMULATE_FLOAT16
constexpr inline bool operator==(float16_t lhs, float16_t rhs) noexcept {
  return lhs.raw == rhs.raw;
}
constexpr inline bool operator!=(float16_t lhs, float16_t rhs) noexcept {
  return lhs.raw != rhs.raw;
}
constexpr inline bool operator<(float16_t lhs, float16_t rhs) noexcept {
  return lhs.raw < rhs.raw;
}
constexpr inline bool operator<=(float16_t lhs, float16_t rhs) noexcept {
  return lhs.raw <= rhs.raw;
}
constexpr inline bool operator>(float16_t lhs, float16_t rhs) noexcept {
  return lhs.raw > rhs.raw;
}
constexpr inline bool operator>=(float16_t lhs, float16_t rhs) noexcept {
  return lhs.raw >= rhs.raw;
}
#endif  // HWY_EMULATE_FLOAT16

struct bfloat16_t {
#if HWY_SVE_HAVE_BFLOAT16
  using Raw = __bf16;
#elif __cplusplus >= 202100L && defined(__STDCPP_BFLOAT16_T__)  // C++23
  using Raw = std::bfloat16_t;
#else
#define HWY_EMULATE_BFLOAT16
  using Raw = uint16_t;
  Raw bits;
#endif

#ifndef HWY_EMULATE_BFLOAT16
  Raw raw;

  bfloat16_t() noexcept = default;
  template <typename T>
  constexpr bfloat16_t(T arg) noexcept : raw(static_cast<Raw>(arg)) {}
  bfloat16_t& operator=(Raw arg) noexcept {
    raw = arg;
    return *this;
  }
  constexpr bfloat16_t(const bfloat16_t&) noexcept = default;
  bfloat16_t& operator=(const bfloat16_t&) noexcept = default;
  constexpr operator Raw() const noexcept { return raw; }

  template <typename T>
  bfloat16_t& operator+=(T rhs) noexcept {
    raw = static_cast<Raw>(raw + rhs);
    return *this;
  }

  template <typename T>
  bfloat16_t& operator-=(T rhs) noexcept {
    raw = static_cast<Raw>(raw - rhs);
    return *this;
  }

  template <typename T>
  bfloat16_t& operator*=(T rhs) noexcept {
    raw = static_cast<Raw>(raw * rhs);
    return *this;
  }

  template <typename T>
  bfloat16_t& operator/=(T rhs) noexcept {
    raw = static_cast<Raw>(raw / rhs);
    return *this;
  }

  // pre-decrement operator (--x)
  bfloat16_t& operator--() noexcept {
    raw = static_cast<Raw>(raw - Raw{1});
    return *this;
  }

  // post-decrement operator (x--)
  bfloat16_t operator--(int) noexcept {
    bfloat16_t result = *this;
    raw = static_cast<Raw>(raw - Raw{1});
    return result;
  }

  // pre-increment operator (++x)
  bfloat16_t& operator++() noexcept {
    raw = static_cast<Raw>(raw + Raw{1});
    return *this;
  }

  // post-increment operator (x++)
  bfloat16_t operator++(int) noexcept {
    bfloat16_t result = *this;
    raw = static_cast<Raw>(raw + Raw{1});
    return result;
  }

  constexpr bfloat16_t operator-() const noexcept {
    return bfloat16_t(static_cast<Raw>(-raw));
  }
  constexpr bfloat16_t operator+() const noexcept { return *this; }
#endif  // HWY_EMULATE_BFLOAT16
};

#ifndef HWY_EMULATE_BFLOAT16
constexpr inline bool operator==(bfloat16_t lhs, bfloat16_t rhs) noexcept {
  return lhs.raw == rhs.raw;
}
constexpr inline bool operator!=(bfloat16_t lhs, bfloat16_t rhs) noexcept {
  return lhs.raw != rhs.raw;
}
constexpr inline bool operator<(bfloat16_t lhs, bfloat16_t rhs) noexcept {
  return lhs.raw < rhs.raw;
}
constexpr inline bool operator<=(bfloat16_t lhs, bfloat16_t rhs) noexcept {
  return lhs.raw <= rhs.raw;
}
constexpr inline bool operator>(bfloat16_t lhs, bfloat16_t rhs) noexcept {
  return lhs.raw > rhs.raw;
}
constexpr inline bool operator>=(bfloat16_t lhs, bfloat16_t rhs) noexcept {
  return lhs.raw >= rhs.raw;
}
#endif  // HWY_EMULATE_BFLOAT16

#pragma pack(pop)

HWY_API float F32FromF16(float16_t f16) {
#ifdef HWY_EMULATE_FLOAT16
  uint16_t bits16;
  CopySameSize(&f16, &bits16);
  const uint32_t sign = static_cast<uint32_t>(bits16 >> 15);
  const uint32_t biased_exp = (bits16 >> 10) & 0x1F;
  const uint32_t mantissa = bits16 & 0x3FF;

  // Subnormal or zero
  if (biased_exp == 0) {
    const float subnormal =
        (1.0f / 16384) * (static_cast<float>(mantissa) * (1.0f / 1024));
    return sign ? -subnormal : subnormal;
  }

  // Normalized: convert the representation directly (faster than ldexp/tables).
  const uint32_t biased_exp32 = biased_exp + (127 - 15);
  const uint32_t mantissa32 = mantissa << (23 - 10);
  const uint32_t bits32 = (sign << 31) | (biased_exp32 << 23) | mantissa32;

  float result;
  CopySameSize(&bits32, &result);
  return result;
#else
  return static_cast<float>(f16);
#endif
}

HWY_API float16_t F16FromF32(float f32) {
#ifdef HWY_EMULATE_FLOAT16
  uint32_t bits32;
  CopySameSize(&f32, &bits32);
  const uint32_t sign = bits32 >> 31;
  const uint32_t biased_exp32 = (bits32 >> 23) & 0xFF;
  const uint32_t mantissa32 = bits32 & 0x7FFFFF;

  const int32_t exp = HWY_MIN(static_cast<int32_t>(biased_exp32) - 127, 15);

  // Tiny or zero => zero.
  float16_t out;
  if (exp < -24) {
    // restore original sign
    const uint16_t bits = static_cast<uint16_t>(sign << 15);
    CopySameSize(&bits, &out);
    return out;
  }

  uint32_t biased_exp16, mantissa16;

  // exp = [-24, -15] => subnormal
  if (exp < -14) {
    biased_exp16 = 0;
    const uint32_t sub_exp = static_cast<uint32_t>(-14 - exp);
    HWY_DASSERT(1 <= sub_exp && sub_exp < 11);
    mantissa16 = static_cast<uint32_t>((1u << (10 - sub_exp)) +
                                       (mantissa32 >> (13 + sub_exp)));
  } else {
    // exp = [-14, 15]
    biased_exp16 = static_cast<uint32_t>(exp + 15);
    HWY_DASSERT(1 <= biased_exp16 && biased_exp16 < 31);
    mantissa16 = mantissa32 >> 13;
  }

  HWY_DASSERT(mantissa16 < 1024);
  const uint32_t bits16 = (sign << 15) | (biased_exp16 << 10) | mantissa16;
  HWY_DASSERT(bits16 < 0x10000);
  const uint16_t narrowed = static_cast<uint16_t>(bits16);  // big-endian safe
  CopySameSize(&narrowed, &out);
  return out;
#else
  return float16_t(static_cast<float16_t::Raw>(f32));
#endif
}

HWY_API float F32FromBF16(bfloat16_t bf) {
  uint16_t bits16;
  CopyBytes<2>(&bf, &bits16);
  uint32_t bits = bits16;
  bits <<= 16;
  float f;
  CopySameSize(&bits, &f);
  return f;
}

HWY_API float F32FromF16Mem(const void* ptr) {
  float16_t f16;
  CopyBytes<2>(ptr, &f16);
  return F32FromF16(f16);
}

HWY_API float F32FromBF16Mem(const void* ptr) {
  bfloat16_t bf;
  CopyBytes<2>(ptr, &bf);
  return F32FromBF16(bf);
}

HWY_API bfloat16_t BF16FromF32(float f) {
  uint32_t bits;
  CopySameSize(&f, &bits);
  const uint16_t bits16 = static_cast<uint16_t>(bits >> 16);
  bfloat16_t bf;
  CopySameSize(&bits16, &bf);
  return bf;
}

using float32_t = float;
using float64_t = double;

#pragma pack(push, 1)

// Aligned 128-bit type. Cannot use __int128 because clang doesn't yet align it:
// https://reviews.llvm.org/D86310
struct alignas(16) uint128_t {
  uint64_t lo;  // little-endian layout
  uint64_t hi;
};

// 64 bit key plus 64 bit value. Faster than using uint128_t when only the key
// field is to be compared (Lt128Upper instead of Lt128).
struct alignas(16) K64V64 {
  uint64_t value;  // little-endian layout
  uint64_t key;
};

// 32 bit key plus 32 bit value. Allows vqsort recursions to terminate earlier
// than when considering both to be a 64-bit key.
struct alignas(8) K32V32 {
  uint32_t value;  // little-endian layout
  uint32_t key;
};

#pragma pack(pop)

#ifdef HWY_EMULATE_FLOAT16

static inline HWY_MAYBE_UNUSED bool operator<(const float16_t& a,
                                              const float16_t& b) {
  return F32FromF16(a) < F32FromF16(b);
}
// Required for std::greater.
static inline HWY_MAYBE_UNUSED bool operator>(const float16_t& a,
                                              const float16_t& b) {
  return F32FromF16(a) > F32FromF16(b);
}
static inline HWY_MAYBE_UNUSED bool operator==(const float16_t& a,
                                               const float16_t& b) {
  return F32FromF16(a) == F32FromF16(b);
}

#endif  // HWY_EMULATE_FLOAT16

static inline HWY_MAYBE_UNUSED bool operator<(const uint128_t& a,
                                              const uint128_t& b) {
  return (a.hi == b.hi) ? a.lo < b.lo : a.hi < b.hi;
}
// Required for std::greater.
static inline HWY_MAYBE_UNUSED bool operator>(const uint128_t& a,
                                              const uint128_t& b) {
  return b < a;
}
static inline HWY_MAYBE_UNUSED bool operator==(const uint128_t& a,
                                               const uint128_t& b) {
  return a.lo == b.lo && a.hi == b.hi;
}

static inline HWY_MAYBE_UNUSED bool operator<(const K64V64& a,
                                              const K64V64& b) {
  return a.key < b.key;
}
// Required for std::greater.
static inline HWY_MAYBE_UNUSED bool operator>(const K64V64& a,
                                              const K64V64& b) {
  return b < a;
}
static inline HWY_MAYBE_UNUSED bool operator==(const K64V64& a,
                                               const K64V64& b) {
  return a.key == b.key;
}

static inline HWY_MAYBE_UNUSED bool operator<(const K32V32& a,
                                              const K32V32& b) {
  return a.key < b.key;
}
// Required for std::greater.
static inline HWY_MAYBE_UNUSED bool operator>(const K32V32& a,
                                              const K32V32& b) {
  return b < a;
}
static inline HWY_MAYBE_UNUSED bool operator==(const K32V32& a,
                                               const K32V32& b) {
  return a.key == b.key;
}

//------------------------------------------------------------------------------
// Controlling overload resolution (SFINAE)

template <bool Condition>
struct EnableIfT {};
template <>
struct EnableIfT<true> {
  using type = void;
};

template <bool Condition>
using EnableIf = typename EnableIfT<Condition>::type;

template <typename T, typename U>
struct IsSameT {
  enum { value = 0 };
};

template <typename T>
struct IsSameT<T, T> {
  enum { value = 1 };
};

template <typename T, typename U>
HWY_API constexpr bool IsSame() {
  return IsSameT<T, U>::value;
}

template <bool Condition, typename Then, typename Else>
struct IfT {
  using type = Then;
};

template <class Then, class Else>
struct IfT<false, Then, Else> {
  using type = Else;
};

template <bool Condition, typename Then, typename Else>
using If = typename IfT<Condition, Then, Else>::type;

// Insert into template/function arguments to enable this overload only for
// vectors of exactly, at most (LE), or more than (GT) this many bytes.
//
// As an example, checking for a total size of 16 bytes will match both
// Simd<uint8_t, 16, 0> and Simd<uint8_t, 8, 1>.
#define HWY_IF_V_SIZE(T, kN, bytes) \
  hwy::EnableIf<kN * sizeof(T) == bytes>* = nullptr
#define HWY_IF_V_SIZE_LE(T, kN, bytes) \
  hwy::EnableIf<kN * sizeof(T) <= bytes>* = nullptr
#define HWY_IF_V_SIZE_GT(T, kN, bytes) \
  hwy::EnableIf<(kN * sizeof(T) > bytes)>* = nullptr

#define HWY_IF_LANES(kN, lanes) hwy::EnableIf<(kN == lanes)>* = nullptr
#define HWY_IF_LANES_LE(kN, lanes) hwy::EnableIf<(kN <= lanes)>* = nullptr
#define HWY_IF_LANES_GT(kN, lanes) hwy::EnableIf<(kN > lanes)>* = nullptr

#define HWY_IF_UNSIGNED(T) hwy::EnableIf<!IsSigned<T>()>* = nullptr
#define HWY_IF_SIGNED(T)                                                   \
  hwy::EnableIf<IsSigned<T>() && !IsFloat<T>() && !IsSpecialFloat<T>()>* = \
      nullptr
#define HWY_IF_FLOAT(T) hwy::EnableIf<hwy::IsFloat<T>()>* = nullptr
#define HWY_IF_NOT_FLOAT(T) hwy::EnableIf<!hwy::IsFloat<T>()>* = nullptr
#define HWY_IF_FLOAT3264(T) hwy::EnableIf<hwy::IsFloat3264<T>()>* = nullptr
#define HWY_IF_NOT_FLOAT3264(T) hwy::EnableIf<!hwy::IsFloat3264<T>()>* = nullptr
#define HWY_IF_SPECIAL_FLOAT(T) \
  hwy::EnableIf<hwy::IsSpecialFloat<T>()>* = nullptr
#define HWY_IF_NOT_SPECIAL_FLOAT(T) \
  hwy::EnableIf<!hwy::IsSpecialFloat<T>()>* = nullptr
#define HWY_IF_FLOAT_OR_SPECIAL(T) \
  hwy::EnableIf<hwy::IsFloat<T>() || hwy::IsSpecialFloat<T>()>* = nullptr
#define HWY_IF_NOT_FLOAT_NOR_SPECIAL(T) \
  hwy::EnableIf<!hwy::IsFloat<T>() && !hwy::IsSpecialFloat<T>()>* = nullptr
#define HWY_IF_INTEGER(T) hwy::EnableIf<hwy::IsInteger<T>()>* = nullptr

#define HWY_IF_T_SIZE(T, bytes) hwy::EnableIf<sizeof(T) == (bytes)>* = nullptr
#define HWY_IF_NOT_T_SIZE(T, bytes) \
  hwy::EnableIf<sizeof(T) != (bytes)>* = nullptr
// bit_array = 0x102 means 1 or 8 bytes. There is no NONE_OF because it sounds
// too similar. If you want the opposite of this (2 or 4 bytes), ask for those
// bits explicitly (0x14) instead of attempting to 'negate' 0x102.
#define HWY_IF_T_SIZE_ONE_OF(T, bit_array) \
  hwy::EnableIf<((size_t{1} << sizeof(T)) & (bit_array)) != 0>* = nullptr

#define HWY_IF_U8(T) hwy::EnableIf<IsSame<T, uint8_t>()>* = nullptr
#define HWY_IF_U16(T) hwy::EnableIf<IsSame<T, uint16_t>()>* = nullptr
#define HWY_IF_U32(T) hwy::EnableIf<IsSame<T, uint32_t>()>* = nullptr
#define HWY_IF_U64(T) hwy::EnableIf<IsSame<T, uint64_t>()>* = nullptr

#define HWY_IF_I8(T) hwy::EnableIf<IsSame<T, int8_t>()>* = nullptr
#define HWY_IF_I16(T) hwy::EnableIf<IsSame<T, int16_t>()>* = nullptr
#define HWY_IF_I32(T) hwy::EnableIf<IsSame<T, int32_t>()>* = nullptr
#define HWY_IF_I64(T) hwy::EnableIf<IsSame<T, int64_t>()>* = nullptr

#define HWY_IF_BF16(T) hwy::EnableIf<IsSame<T, hwy::bfloat16_t>()>* = nullptr
#define HWY_IF_F16(T) hwy::EnableIf<IsSame<T, hwy::float16_t>()>* = nullptr
#define HWY_IF_F32(T) hwy::EnableIf<IsSame<T, float>()>* = nullptr
#define HWY_IF_F64(T) hwy::EnableIf<IsSame<T, double>()>* = nullptr

// Use instead of HWY_IF_T_SIZE to avoid ambiguity with float16_t/float/double
// overloads.
#define HWY_IF_UI8(T) \
  hwy::EnableIf<IsSame<T, uint8_t>() || IsSame<T, int8_t>()>* = nullptr
#define HWY_IF_UI16(T) \
  hwy::EnableIf<IsSame<T, uint16_t>() || IsSame<T, int16_t>()>* = nullptr
#define HWY_IF_UI32(T) \
  hwy::EnableIf<IsSame<T, uint32_t>() || IsSame<T, int32_t>()>* = nullptr
#define HWY_IF_UI64(T) \
  hwy::EnableIf<IsSame<T, uint64_t>() || IsSame<T, int64_t>()>* = nullptr

#define HWY_IF_LANES_PER_BLOCK(T, N, LANES) \
  hwy::EnableIf<HWY_MIN(sizeof(T) * N, 16) / sizeof(T) == (LANES)>* = nullptr

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

template <class T>
struct RemoveConstT {
  using type = T;
};
template <class T>
struct RemoveConstT<const T> {
  using type = T;
};

template <class T>
using RemoveConst = typename RemoveConstT<T>::type;

template <class T>
struct RemoveVolatileT {
  using type = T;
};
template <class T>
struct RemoveVolatileT<volatile T> {
  using type = T;
};

template <class T>
using RemoveVolatile = typename RemoveVolatileT<T>::type;

template <class T>
struct RemoveRefT {
  using type = T;
};
template <class T>
struct RemoveRefT<T&> {
  using type = T;
};
template <class T>
struct RemoveRefT<T&&> {
  using type = T;
};

template <class T>
using RemoveRef = typename RemoveRefT<T>::type;

template <class T>
using RemoveCvRef = RemoveConst<RemoveVolatile<RemoveRef<T>>>;

//------------------------------------------------------------------------------
// Type relations

namespace detail {

template <typename T>
struct Relations;
template <>
struct Relations<uint8_t> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
  using Wide = uint16_t;
  enum { is_signed = 0, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<int8_t> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
  using Wide = int16_t;
  enum { is_signed = 1, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<uint16_t> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Float = float16_t;
  using Wide = uint32_t;
  using Narrow = uint8_t;
  enum { is_signed = 0, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<int16_t> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Float = float16_t;
  using Wide = int32_t;
  using Narrow = int8_t;
  enum { is_signed = 1, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<uint32_t> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
  using Wide = uint64_t;
  using Narrow = uint16_t;
  enum { is_signed = 0, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<int32_t> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
  using Wide = int64_t;
  using Narrow = int16_t;
  enum { is_signed = 1, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<uint64_t> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
  using Wide = uint128_t;
  using Narrow = uint32_t;
  enum { is_signed = 0, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<int64_t> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
  using Narrow = int32_t;
  enum { is_signed = 1, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<uint128_t> {
  using Unsigned = uint128_t;
  using Narrow = uint64_t;
  enum { is_signed = 0, is_float = 0, is_bf16 = 0 };
};
template <>
struct Relations<float16_t> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Float = float16_t;
  using Wide = float;
  enum { is_signed = 1, is_float = 1, is_bf16 = 0 };
};
template <>
struct Relations<bfloat16_t> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Wide = float;
  enum { is_signed = 1, is_float = 1, is_bf16 = 1 };
};
template <>
struct Relations<float> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
  using Wide = double;
  using Narrow = float16_t;
  enum { is_signed = 1, is_float = 1, is_bf16 = 0 };
};
template <>
struct Relations<double> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
  using Narrow = float;
  enum { is_signed = 1, is_float = 1, is_bf16 = 0 };
};

template <size_t N>
struct TypeFromSize;
template <>
struct TypeFromSize<1> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
};
template <>
struct TypeFromSize<2> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
  using Float = float16_t;
};
template <>
struct TypeFromSize<4> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
};
template <>
struct TypeFromSize<8> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
};
template <>
struct TypeFromSize<16> {
  using Unsigned = uint128_t;
};

}  // namespace detail

// Aliases for types of a different category, but the same size.
template <typename T>
using MakeUnsigned = typename detail::Relations<T>::Unsigned;
template <typename T>
using MakeSigned = typename detail::Relations<T>::Signed;
template <typename T>
using MakeFloat = typename detail::Relations<T>::Float;

// Aliases for types of the same category, but different size.
template <typename T>
using MakeWide = typename detail::Relations<T>::Wide;
template <typename T>
using MakeNarrow = typename detail::Relations<T>::Narrow;

// Obtain type from its size [bytes].
template <size_t N>
using UnsignedFromSize = typename detail::TypeFromSize<N>::Unsigned;
template <size_t N>
using SignedFromSize = typename detail::TypeFromSize<N>::Signed;
template <size_t N>
using FloatFromSize = typename detail::TypeFromSize<N>::Float;

// Avoid confusion with SizeTag where the parameter is a lane size.
using UnsignedTag = SizeTag<0>;
using SignedTag = SizeTag<0x100>;  // integer
using FloatTag = SizeTag<0x200>;
using SpecialTag = SizeTag<0x300>;

template <typename T, class R = detail::Relations<T>>
constexpr auto TypeTag()
    -> hwy::SizeTag<((R::is_signed + R::is_float + R::is_bf16) << 8)> {
  return hwy::SizeTag<((R::is_signed + R::is_float + R::is_bf16) << 8)>();
}

// For when we only want to distinguish FloatTag from everything else.
using NonFloatTag = SizeTag<0x400>;

template <typename T, class R = detail::Relations<T>>
constexpr auto IsFloatTag() -> hwy::SizeTag<(R::is_float ? 0x200 : 0x400)> {
  return hwy::SizeTag<(R::is_float ? 0x200 : 0x400)>();
}

//------------------------------------------------------------------------------
// Type traits

template <typename T>
HWY_API constexpr bool IsFloat3264() {
  return IsSame<T, float>() || IsSame<T, double>();
}

template <typename T>
HWY_API constexpr bool IsFloat() {
  // Cannot use T(1.25) != T(1) for float16_t, which can only be converted to or
  // from a float, not compared. Include float16_t in case HWY_HAVE_FLOAT16=1.
  return IsSame<T, float16_t>() || IsFloat3264<T>();
}

// These types are often special-cased and not supported in all ops.
template <typename T>
HWY_API constexpr bool IsSpecialFloat() {
  return IsSame<T, float16_t>() || IsSame<T, bfloat16_t>();
}

template <class T>
HWY_API constexpr bool IsIntegerLaneType() {
  return false;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<int8_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<uint8_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<int16_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<uint16_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<int32_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<uint32_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<int64_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsIntegerLaneType<uint64_t>() {
  return true;
}

template <class T>
HWY_API constexpr bool IsInteger() {
  // NOTE: Do not add a IsInteger<wchar_t>() specialization below as it is
  // possible for IsSame<wchar_t, uint16_t>() to be true when compiled with MSVC
  // with the /Zc:wchar_t- option.
  return IsIntegerLaneType<T>() || IsSame<T, wchar_t>() ||
         IsSame<T, size_t>() || IsSame<T, ptrdiff_t>() ||
         IsSame<T, intptr_t>() || IsSame<T, uintptr_t>();
}
template <>
HWY_INLINE constexpr bool IsInteger<bool>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<char>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<signed char>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<unsigned char>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<short>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<unsigned short>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<int>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<unsigned>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<long>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<unsigned long>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<long long>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<unsigned long long>() {
  return true;
}
#if defined(__cpp_char8_t) && __cpp_char8_t >= 201811L
template <>
HWY_INLINE constexpr bool IsInteger<char8_t>() {
  return true;
}
#endif
template <>
HWY_INLINE constexpr bool IsInteger<char16_t>() {
  return true;
}
template <>
HWY_INLINE constexpr bool IsInteger<char32_t>() {
  return true;
}

template <typename T>
HWY_API constexpr bool IsSigned() {
  return T(0) > T(-1);
}
template <>
constexpr bool IsSigned<float16_t>() {
  return true;
}
template <>
constexpr bool IsSigned<bfloat16_t>() {
  return true;
}

// Largest/smallest representable integer values.
template <typename T>
HWY_API constexpr T LimitsMax() {
  static_assert(IsInteger<T>(), "Only for integer types");
  using TU = UnsignedFromSize<sizeof(T)>;
  return static_cast<T>(IsSigned<T>() ? (static_cast<TU>(~TU(0)) >> 1)
                                      : static_cast<TU>(~TU(0)));
}
template <typename T>
HWY_API constexpr T LimitsMin() {
  static_assert(IsInteger<T>(), "Only for integer types");
  return IsSigned<T>() ? T(-1) - LimitsMax<T>() : T(0);
}

// Largest/smallest representable value (integer or float). This naming avoids
// confusion with numeric_limits<float>::min() (the smallest positive value).
// Cannot be constexpr because we use CopySameSize for [b]float16_t.
template <typename T>
HWY_API HWY_BITCASTSCALAR_CONSTEXPR T LowestValue() {
  return LimitsMin<T>();
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR bfloat16_t LowestValue<bfloat16_t>() {
  return BitCastScalar<bfloat16_t>(uint16_t{0xFF7Fu});  // -1.1111111 x 2^127
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float16_t LowestValue<float16_t>() {
  return BitCastScalar<float16_t>(uint16_t{0xFBFFu});  // -1.1111111111 x 2^15
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float LowestValue<float>() {
  return -3.402823466e+38F;
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR double LowestValue<double>() {
  return -1.7976931348623158e+308;
}

template <typename T>
HWY_API HWY_BITCASTSCALAR_CONSTEXPR T HighestValue() {
  return LimitsMax<T>();
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR bfloat16_t HighestValue<bfloat16_t>() {
  return BitCastScalar<bfloat16_t>(uint16_t{0x7F7Fu});  // 1.1111111 x 2^127
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float16_t HighestValue<float16_t>() {
  return BitCastScalar<float16_t>(uint16_t{0x7BFFu});  // 1.1111111111 x 2^15
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float HighestValue<float>() {
  return 3.402823466e+38F;
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR double HighestValue<double>() {
  return 1.7976931348623158e+308;
}

// Difference between 1.0 and the next representable value. Equal to
// 1 / (1ULL << MantissaBits<T>()), but hard-coding ensures precision.
template <typename T>
HWY_API HWY_BITCASTSCALAR_CONSTEXPR T Epsilon() {
  return 1;
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR bfloat16_t Epsilon<bfloat16_t>() {
  return BitCastScalar<bfloat16_t>(uint16_t{0x3C00u});  // 0.0078125
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float16_t Epsilon<float16_t>() {
  return BitCastScalar<float16_t>(uint16_t{0x1400u});  // 0.0009765625
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float Epsilon<float>() {
  return 1.192092896e-7f;
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR double Epsilon<double>() {
  return 2.2204460492503131e-16;
}

// Returns width in bits of the mantissa field in IEEE binary16/32/64.
template <typename T>
constexpr int MantissaBits() {
  static_assert(sizeof(T) == 0, "Only instantiate the specializations");
  return 0;
}
template <>
constexpr int MantissaBits<bfloat16_t>() {
  return 7;
}
template <>
constexpr int MantissaBits<float16_t>() {
  return 10;
}
template <>
constexpr int MantissaBits<float>() {
  return 23;
}
template <>
constexpr int MantissaBits<double>() {
  return 52;
}

// Returns the (left-shifted by one bit) IEEE binary16/32/64 representation with
// the largest possible (biased) exponent field. Used by IsInf.
template <typename T>
constexpr MakeSigned<T> MaxExponentTimes2() {
  return -(MakeSigned<T>{1} << (MantissaBits<T>() + 1));
}

// Returns bitmask of the sign bit in IEEE binary16/32/64.
template <typename T>
constexpr MakeUnsigned<T> SignMask() {
  return MakeUnsigned<T>{1} << (sizeof(T) * 8 - 1);
}

// Returns bitmask of the exponent field in IEEE binary16/32/64.
template <typename T>
constexpr MakeUnsigned<T> ExponentMask() {
  return (~(MakeUnsigned<T>{1} << MantissaBits<T>()) + 1) &
         static_cast<MakeUnsigned<T>>(~SignMask<T>());
}

// Returns bitmask of the mantissa field in IEEE binary16/32/64.
template <typename T>
constexpr MakeUnsigned<T> MantissaMask() {
  return (MakeUnsigned<T>{1} << MantissaBits<T>()) - 1;
}

// Returns 1 << mantissa_bits as a floating-point number. All integers whose
// absolute value are less than this can be represented exactly.
template <typename T>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR T MantissaEnd() {
  static_assert(sizeof(T) == 0, "Only instantiate the specializations");
  return 0;
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR bfloat16_t MantissaEnd<bfloat16_t>() {
  return BitCastScalar<bfloat16_t>(uint16_t{0x4300u});  // 1.0 x 2^7
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float16_t MantissaEnd<float16_t>() {
  return BitCastScalar<float16_t>(uint16_t{0x6400u});  // 1.0 x 2^10
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR float MantissaEnd<float>() {
  return 8388608.0f;  // 1 << 23
}
template <>
HWY_INLINE HWY_BITCASTSCALAR_CONSTEXPR double MantissaEnd<double>() {
  // floating point literal with p52 requires C++17.
  return 4503599627370496.0;  // 1 << 52
}

// Returns width in bits of the exponent field in IEEE binary16/32/64.
template <typename T>
constexpr int ExponentBits() {
  // Exponent := remaining bits after deducting sign and mantissa.
  return 8 * sizeof(T) - 1 - MantissaBits<T>();
}

// Returns largest value of the biased exponent field in IEEE binary16/32/64,
// right-shifted so that the LSB is bit zero. Example: 0xFF for float.
// This is expressed as a signed integer for more efficient comparison.
template <typename T>
constexpr MakeSigned<T> MaxExponentField() {
  return (MakeSigned<T>{1} << ExponentBits<T>()) - 1;
}

//------------------------------------------------------------------------------
// Helper functions

template <typename T1, typename T2>
constexpr inline T1 DivCeil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

// Works for any `align`; if a power of two, compiler emits ADD+AND.
constexpr inline size_t RoundUpTo(size_t what, size_t align) {
  return DivCeil(what, align) * align;
}

// Undefined results for x == 0.
HWY_API size_t Num0BitsBelowLS1Bit_Nonzero32(const uint32_t x) {
#if HWY_COMPILER_MSVC
  unsigned long index;  // NOLINT
  _BitScanForward(&index, x);
  return index;
#else   // HWY_COMPILER_MSVC
  return static_cast<size_t>(__builtin_ctz(x));
#endif  // HWY_COMPILER_MSVC
}

HWY_API size_t Num0BitsBelowLS1Bit_Nonzero64(const uint64_t x) {
#if HWY_COMPILER_MSVC
#if HWY_ARCH_X86_64
  unsigned long index;  // NOLINT
  _BitScanForward64(&index, x);
  return index;
#else   // HWY_ARCH_X86_64
  // _BitScanForward64 not available
  uint32_t lsb = static_cast<uint32_t>(x & 0xFFFFFFFF);
  unsigned long index;  // NOLINT
  if (lsb == 0) {
    uint32_t msb = static_cast<uint32_t>(x >> 32u);
    _BitScanForward(&index, msb);
    return 32 + index;
  } else {
    _BitScanForward(&index, lsb);
    return index;
  }
#endif  // HWY_ARCH_X86_64
#else   // HWY_COMPILER_MSVC
  return static_cast<size_t>(__builtin_ctzll(x));
#endif  // HWY_COMPILER_MSVC
}

// Undefined results for x == 0.
HWY_API size_t Num0BitsAboveMS1Bit_Nonzero32(const uint32_t x) {
#if HWY_COMPILER_MSVC
  unsigned long index;  // NOLINT
  _BitScanReverse(&index, x);
  return 31 - index;
#else   // HWY_COMPILER_MSVC
  return static_cast<size_t>(__builtin_clz(x));
#endif  // HWY_COMPILER_MSVC
}

HWY_API size_t Num0BitsAboveMS1Bit_Nonzero64(const uint64_t x) {
#if HWY_COMPILER_MSVC
#if HWY_ARCH_X86_64
  unsigned long index;  // NOLINT
  _BitScanReverse64(&index, x);
  return 63 - index;
#else   // HWY_ARCH_X86_64
  // _BitScanReverse64 not available
  const uint32_t msb = static_cast<uint32_t>(x >> 32u);
  unsigned long index;  // NOLINT
  if (msb == 0) {
    const uint32_t lsb = static_cast<uint32_t>(x & 0xFFFFFFFF);
    _BitScanReverse(&index, lsb);
    return 63 - index;
  } else {
    _BitScanReverse(&index, msb);
    return 31 - index;
  }
#endif  // HWY_ARCH_X86_64
#else   // HWY_COMPILER_MSVC
  return static_cast<size_t>(__builtin_clzll(x));
#endif  // HWY_COMPILER_MSVC
}

template <class T, HWY_IF_INTEGER(RemoveCvRef<T>),
          HWY_IF_T_SIZE_ONE_OF(RemoveCvRef<T>, (1 << 1) | (1 << 2) | (1 << 4))>
HWY_API size_t PopCount(T x) {
  uint32_t u32_x = static_cast<uint32_t>(
      static_cast<UnsignedFromSize<sizeof(RemoveCvRef<T>)>>(x));

#if HWY_COMPILER_GCC || HWY_COMPILER_CLANG
  return static_cast<size_t>(__builtin_popcountl(u32_x));
#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_32 && defined(__AVX__)
  return static_cast<size_t>(_mm_popcnt_u32(u32_x));
#else
  u32_x -= ((u32_x >> 1) & 0x55555555u);
  u32_x = (((u32_x >> 2) & 0x33333333u) + (u32_x & 0x33333333u));
  u32_x = (((u32_x >> 4) + u32_x) & 0x0F0F0F0Fu);
  u32_x += (u32_x >> 8);
  u32_x += (u32_x >> 16);
  return static_cast<size_t>(u32_x & 0x3Fu);
#endif
}

template <class T, HWY_IF_INTEGER(RemoveCvRef<T>),
          HWY_IF_T_SIZE(RemoveCvRef<T>, 8)>
HWY_API size_t PopCount(T x) {
  uint64_t u64_x = static_cast<uint64_t>(
      static_cast<UnsignedFromSize<sizeof(RemoveCvRef<T>)>>(x));

#if HWY_COMPILER_GCC || HWY_COMPILER_CLANG
  return static_cast<size_t>(__builtin_popcountll(u64_x));
#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_64 && defined(__AVX__)
  return _mm_popcnt_u64(u64_x);
#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_32 && defined(__AVX__)
  return _mm_popcnt_u32(static_cast<uint32_t>(u64_x & 0xFFFFFFFFu)) +
         _mm_popcnt_u32(static_cast<uint32_t>(u64_x >> 32));
#else
  u64_x -= ((u64_x >> 1) & 0x5555555555555555ULL);
  u64_x = (((u64_x >> 2) & 0x3333333333333333ULL) +
           (u64_x & 0x3333333333333333ULL));
  u64_x = (((u64_x >> 4) + u64_x) & 0x0F0F0F0F0F0F0F0FULL);
  u64_x += (u64_x >> 8);
  u64_x += (u64_x >> 16);
  u64_x += (u64_x >> 32);
  return static_cast<size_t>(u64_x & 0x7Fu);
#endif
}

// Skip HWY_API due to GCC "function not considered for inlining". Previously
// such errors were caused by underlying type mismatches, but it's not clear
// what is still mismatched despite all the casts.
template <typename TI>
/*HWY_API*/ constexpr size_t FloorLog2(TI x) {
  return x == TI{1}
             ? 0
             : static_cast<size_t>(FloorLog2(static_cast<TI>(x >> 1)) + 1);
}

template <typename TI>
/*HWY_API*/ constexpr size_t CeilLog2(TI x) {
  return x == TI{1}
             ? 0
             : static_cast<size_t>(FloorLog2(static_cast<TI>(x - 1)) + 1);
}

template <typename T, typename T2>
HWY_INLINE constexpr T AddWithWraparound(hwy::FloatTag /*tag*/, T t, T2 n) {
  return t + static_cast<T>(n);
}

template <typename T, typename T2>
HWY_INLINE constexpr T AddWithWraparound(hwy::NonFloatTag /*tag*/, T t, T2 n) {
  using TU = MakeUnsigned<T>;
  return static_cast<T>(
      static_cast<TU>(static_cast<TU>(t) + static_cast<TU>(n)) &
      hwy::LimitsMax<TU>());
}

#if HWY_COMPILER_MSVC && HWY_ARCH_X86_64
#pragma intrinsic(_umul128)
#endif

// 64 x 64 = 128 bit multiplication
HWY_API uint64_t Mul128(uint64_t a, uint64_t b, uint64_t* HWY_RESTRICT upper) {
#if defined(__SIZEOF_INT128__)
  __uint128_t product = (__uint128_t)a * (__uint128_t)b;
  *upper = (uint64_t)(product >> 64);
  return (uint64_t)(product & 0xFFFFFFFFFFFFFFFFULL);
#elif HWY_COMPILER_MSVC && HWY_ARCH_X86_64
  return _umul128(a, b, upper);
#else
  constexpr uint64_t kLo32 = 0xFFFFFFFFU;
  const uint64_t lo_lo = (a & kLo32) * (b & kLo32);
  const uint64_t hi_lo = (a >> 32) * (b & kLo32);
  const uint64_t lo_hi = (a & kLo32) * (b >> 32);
  const uint64_t hi_hi = (a >> 32) * (b >> 32);
  const uint64_t t = (lo_lo >> 32) + (hi_lo & kLo32) + lo_hi;
  *upper = (hi_lo >> 32) + (t >> 32) + hi_hi;
  return (t << 32) | (lo_lo & kLo32);
#endif
}

// Prevents the compiler from eliding the computations that led to "output".
template <class T>
HWY_API void PreventElision(T&& output) {
#if HWY_COMPILER_MSVC
  // MSVC does not support inline assembly anymore (and never supported GCC's
  // RTL constraints). Self-assignment with #pragma optimize("off") might be
  // expected to prevent elision, but it does not with MSVC 2015. Type-punning
  // with volatile pointers generates inefficient code on MSVC 2017.
  static std::atomic<RemoveCvRef<T>> dummy;
  dummy.store(output, std::memory_order_relaxed);
#else
  // Works by indicating to the compiler that "output" is being read and
  // modified. The +r constraint avoids unnecessary writes to memory, but only
  // works for built-in types (typically FuncOutput).
  asm volatile("" : "+r"(output) : : "memory");
#endif
}

}  // namespace hwy

#endif  // HIGHWAY_HWY_BASE_H_
