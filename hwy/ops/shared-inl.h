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

// Implementation details included from each ops/*.h.

// Normal include guard required for macros/symbols in hwy (instead of the
// unique-per-target hwy::NAMESPACE). NOTE: this header also has a per-target
// section after this include guard.
#ifndef HWY_SHARED_INL_H_
#define HWY_SHARED_INL_H_

#include <stddef.h>
#include <stdint.h>

#include "hwy/targets.h"

// Clang 3.9 generates VINSERTF128 instead of the desired VBROADCASTF128,
// which would free up port5. However, inline assembly isn't supported on
// MSVC, results in incorrect output on GCC 8.3, and raises "invalid output size
// for constraint" errors on Clang (https://gcc.godbolt.org/z/-Jt_-F), hence we
// disable it.
#ifndef HWY_LOADDUP_ASM
#define HWY_LOADDUP_ASM 0
#endif

// Shorthand for implementations of Highway ops.
#define HWY_API static HWY_INLINE HWY_FLATTEN HWY_MAYBE_UNUSED

namespace hwy {

// Unfortunately the GCC/Clang intrinsics do not accept int64_t*.
using GatherIndex64 = long long int;  // NOLINT(google-runtime-int)
static_assert(sizeof(GatherIndex64) == 8, "Must be 64-bit type");

//------------------------------------------------------------------------------
// Controlling overload resolution

// Insert into template/function arguments to enable this overload only for
// vectors of AT MOST this many bits.
//
// Note that enabling for exactly 128 bits is unnecessary because a function can
// simply be overloaded with Vec128<T> and Full128<T> descriptor. Enabling for
// other sizes (e.g. 64 bit) can be achieved with Simd<T, 8 / sizeof(T)>.
#define HWY_IF_LE128(T, N) hwy::EnableIf<N * sizeof(T) <= 16>* = nullptr
#define HWY_IF_LE64(T, N) hwy::EnableIf<N * sizeof(T) <= 8>* = nullptr
#define HWY_IF_LE32(T, N) hwy::EnableIf<N * sizeof(T) <= 4>* = nullptr

#define HWY_IF_FLOAT(T) hwy::EnableIf<hwy::IsFloat<T>()>* = nullptr

// Empty struct used as a size tag type.
template <size_t N>
struct SizeTag {};

//------------------------------------------------------------------------------
// Conversion between types of the same size

// Unsigned/signed/floating-point types whose sizes are kSize bytes.
template <size_t kSize>
struct TypesOfSize;
template <>
struct TypesOfSize<1> {
  using Unsigned = uint8_t;
  using Signed = int8_t;
};
template <>
struct TypesOfSize<2> {
  using Unsigned = uint16_t;
  using Signed = int16_t;
};
template <>
struct TypesOfSize<4> {
  using Unsigned = uint32_t;
  using Signed = int32_t;
  using Float = float;
};
template <>
struct TypesOfSize<8> {
  using Unsigned = uint64_t;
  using Signed = int64_t;
  using Float = double;
};

template <typename T>
using MakeUnsigned = typename TypesOfSize<sizeof(T)>::Unsigned;
template <typename T>
using MakeSigned = typename TypesOfSize<sizeof(T)>::Signed;
template <typename T>
using MakeFloat = typename TypesOfSize<sizeof(T)>::Float;

}  // namespace hwy

#endif  // HWY_SHARED_INL_H_

//------------------------------------------------------------------------------
// Per-target definitions (relies on external include guard in highway.h)

namespace macros_only {  // for code folding

#undef HWY_NAMESPACE
#undef HWY_ALIGN
#undef HWY_LANES

#undef HWY_GATHER_LANES
#undef HWY_VARIABLE_SHIFT_LANES
#undef HWY_COMPARE64_LANES
#undef HWY_MINMAX64_LANES

#undef HWY_CAP_INTEGER64
#undef HWY_CAP_FLOAT64
#undef HWY_CAP_GE256
#undef HWY_CAP_GE512

#undef HWY_TARGET_STR

// Before include guard so we redefine HWY_TARGET_STR on each include,
// governed by the current HWY_TARGET.
//-----------------------------------------------------------------------------
// SSE4
#if HWY_TARGET == HWY_SSE4

#define HWY_NAMESPACE N_SSE4
#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_GATHER_LANES(T) 1
#define HWY_VARIABLE_SHIFT_LANES(T) 1
#define HWY_COMPARE64_LANES 1
#define HWY_MINMAX64_LANES 1

#define HWY_CAP_INTEGER64 1
#define HWY_CAP_FLOAT64 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

#define HWY_TARGET_STR "sse2,ssse3,sse4.1"

//-----------------------------------------------------------------------------
// AVX2
#elif HWY_TARGET == HWY_AVX2

#define HWY_NAMESPACE N_AVX2
#define HWY_ALIGN alignas(32)
#define HWY_LANES(T) (32 / sizeof(T))

#define HWY_GATHER_LANES(T) HWY_LANES(T)
#define HWY_VARIABLE_SHIFT_LANES(T) HWY_LANES(T)
#define HWY_COMPARE64_LANES 4
#define HWY_MINMAX64_LANES 1

#define HWY_CAP_INTEGER64 1
#define HWY_CAP_FLOAT64 1
#define HWY_CAP_GE256 1
#define HWY_CAP_GE512 0

#if defined(HWY_DISABLE_BMI2_FMA)
#define HWY_TARGET_STR "avx,avx2"
#else
#define HWY_TARGET_STR "avx,avx2,bmi,bmi2,fma"
#endif

//-----------------------------------------------------------------------------
// AVX3
#elif HWY_TARGET == HWY_AVX3

#define HWY_ALIGN alignas(64)
#define HWY_LANES(T) (64 / sizeof(T))

#define HWY_GATHER_LANES(T) HWY_LANES(T)
#define HWY_VARIABLE_SHIFT_LANES(T) HWY_LANES(T)
#define HWY_COMPARE64_LANES 8
#define HWY_MINMAX64_LANES 8

#define HWY_CAP_INTEGER64 1
#define HWY_CAP_FLOAT64 1
#define HWY_CAP_GE256 1
#define HWY_CAP_GE512 1

#define HWY_NAMESPACE N_AVX3

// Must include AVX2 because an AVX3 test may call AVX2 functions (e.g. when
// converting to half-vectors). HWY_DISABLE_BMI2_FMA is not relevant because if
// we have AVX3, we should also have BMI2/FMA.
#define HWY_TARGET_STR \
  "avx,avx2,bmi,bmi2,fma,avx512f,avx512vl,avx512dq,avx512bw"

//-----------------------------------------------------------------------------
// PPC8
#elif HWY_TARGET == HWY_PPC8

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_GATHER_LANES(T) 1
#define HWY_VARIABLE_SHIFT_LANES(T) HWY_LANES(T)
#define HWY_COMPARE64_LANES 2
#define HWY_MINMAX64_LANES 2

#define HWY_CAP_INTEGER64 1
#define HWY_CAP_FLOAT64 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

#define HWY_NAMESPACE N_PPC8

#define HWY_TARGET_STR "altivec,vsx"

//-----------------------------------------------------------------------------
// NEON
#elif HWY_TARGET == HWY_NEON

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_GATHER_LANES(T) 1
#define HWY_VARIABLE_SHIFT_LANES(T) HWY_LANES(T)
#define HWY_MINMAX64_LANES 1

#define HWY_CAP_INTEGER64 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

#ifdef __arm__
#define HWY_COMPARE64_LANES 1
#define HWY_CAP_FLOAT64 0
#else
#define HWY_COMPARE64_LANES 2
#define HWY_CAP_FLOAT64 1
#endif

#define HWY_NAMESPACE N_NEON

#define HWY_TARGET_STR "crypto"

//-----------------------------------------------------------------------------
// WASM
#elif HWY_TARGET == HWY_WASM

#define HWY_ALIGN alignas(16)
#define HWY_LANES(T) (16 / sizeof(T))

#define HWY_GATHER_LANES(T) 1
#define HWY_VARIABLE_SHIFT_LANES(T) 1
#define HWY_COMPARE64_LANES 1
#define HWY_MINMAX64_LANES 1

#define HWY_CAP_INTEGER64 0
#define HWY_CAP_FLOAT64 0
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

#define HWY_NAMESPACE N_WASM

#define HWY_TARGET_STR "simd128"

//-----------------------------------------------------------------------------
// SCALAR
#elif HWY_TARGET == HWY_SCALAR

#define HWY_ALIGN
#define HWY_LANES(T) 1

#define HWY_GATHER_LANES(T) 1
#define HWY_VARIABLE_SHIFT_LANES(T) 1
#define HWY_COMPARE64_LANES 1
#define HWY_MINMAX64_LANES 1

#define HWY_CAP_INTEGER64 1
#define HWY_CAP_FLOAT64 1
#define HWY_CAP_GE256 0
#define HWY_CAP_GE512 0

#define HWY_NAMESPACE N_SCALAR

// HWY_TARGET_STR remains undefined so HWY_ATTR is a no-op.

#else
#pragma message("HWY_TARGET does not match any known target")
#endif  // HWY_TARGET

// Clang <9 requires this be invoked at file scope, before any namespace.
#undef HWY_BEFORE_NAMESPACE
#if defined(HWY_TARGET_STR)
#define HWY_BEFORE_NAMESPACE()        \
  HWY_PUSH_ATTRIBUTES(HWY_TARGET_STR) \
  static_assert(true, "For requiring trailing semicolon")
#else
// avoids compiler warning if no HWY_TARGET_STR
#define HWY_BEFORE_NAMESPACE() \
  static_assert(true, "For requiring trailing semicolon")
#endif

// Clang <9 requires any namespaces be closed before this macro.
#undef HWY_AFTER_NAMESPACE
#if defined(HWY_TARGET_STR)
#define HWY_AFTER_NAMESPACE() \
  HWY_POP_ATTRIBUTES          \
  static_assert(true, "For requiring trailing semicolon")
#else
// avoids compiler warning if no HWY_TARGET_STR
#define HWY_AFTER_NAMESPACE() \
  static_assert(true, "For requiring trailing semicolon")
#endif

#undef HWY_ATTR
#if defined(HWY_TARGET_STR) && HWY_HAS_ATTRIBUTE(target)
#define HWY_ATTR __attribute__((target(HWY_TARGET_STR)))
#else
#define HWY_ATTR
#endif

}  // namespace macros_only

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// SIMD operations are implemented as overloaded functions selected using a
// "descriptor" D := Simd<T, N>. T is the lane type, N the requested number of
// lanes >= 1 (always a power of two). In the common case, users do not choose N
// directly, but instead use HWY_FULL (the largest available size). N may differ
// from the hardware vector size. If N is less, only that many lanes will be
// loaded/stored.
//
// Only HWY_FULL(T) and N <= 16 / sizeof(T) are guaranteed to be available - the
// latter are useful if >128 bit vectors are unnecessary or undesirable.
//
// Users should not use the N of a Simd<> but instead query the actual number of
// lanes via Lanes(). MaxLanes() is provided for template arguments and array
// dimensions, but this is discouraged because an upper bound might not exist.
template <typename Lane, size_t N>
struct Simd {
  constexpr Simd() = default;
  using T = Lane;
  static_assert((N & (N - 1)) == 0 && N != 0, "N must be a power of two");
};

// Compile-time-constant upper bound (even for variable-length vectors), useful
// for array dimensions.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED constexpr size_t MaxLanes(Simd<T, N>) {
  return N;
}

// (Potentially) non-constant actual size of the vector at runtime, subject to
// the limit imposed by the Simd. Useful for advancing loop counters.
template <typename T, size_t N>
HWY_INLINE HWY_MAYBE_UNUSED size_t Lanes(Simd<T, N>) {
  return N;
}

// The source/destination must not overlap/alias.
template <size_t kBytes, typename From, typename To>
HWY_API void CopyBytes(const From* from, To* to) {
#if HWY_COMPILER_MSVC
  const uint8_t* HWY_RESTRICT from_bytes =
      reinterpret_cast<const uint8_t*>(from);
  uint8_t* HWY_RESTRICT to_bytes = reinterpret_cast<uint8_t*>(to);
  for (size_t i = 0; i < kBytes; ++i) {
    to_bytes[i] = from_bytes[i];
  }
#else
  // Avoids horrible codegen on Clang (series of PINSRB)
  __builtin_memcpy(to, from, kBytes);
#endif
}

HWY_API size_t PopCount(const uint64_t x) {
#if HWY_COMPILER_CLANG || HWY_COMPILER_GCC
  return static_cast<size_t>(__builtin_popcountll(x));
#elif HWY_COMPILER_MSVC
  return _mm_popcnt_u64(x);
#else
#error "not supported"
#endif
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
