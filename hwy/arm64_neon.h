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

#ifndef HWY_ARM64_NEON_H_
#define HWY_ARM64_NEON_H_

// 128-bit ARM64 NEON vectors and operations.

#include <arm_neon.h>
#include <stddef.h>

#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
#include <stdio.h>
#endif

#include "hwy/shared.h"

#define HWY_ATTR_ARM8 HWY_TARGET_ATTR("crypto")

#define HWY_CONCAT_IMPL(a, b) a##b
#define HWY_CONCAT(a, b) HWY_CONCAT_IMPL(a, b)

// Macros used to define single and double function calls for multiple types
// for full and half vectors. These macros are undefined at the end of the file.

// HWY_NEON_BUILD_TPL_* is the template<...> prefix to the function.
#define HWY_NEON_BUILD_TPL_1
#define HWY_NEON_BUILD_TPL_2
#define HWY_NEON_BUILD_TPL_3

// HWY_NEON_BUILD_RET_* is return type.
#define HWY_NEON_BUILD_RET_1(type, size) Vec128<type, size>
#define HWY_NEON_BUILD_RET_2(type, size) Vec128<type, size>
#define HWY_NEON_BUILD_RET_3(type, size) Vec128<type, size>

// HWY_NEON_BUILD_PARAM_* is the list of parameters the function receives.
#define HWY_NEON_BUILD_PARAM_1(type, size) const Vec128<type, size> a
#define HWY_NEON_BUILD_PARAM_2(type, size) \
  const Vec128<type, size> a, const Vec128<type, size> b
#define HWY_NEON_BUILD_PARAM_3(type, size)                \
  const Vec128<type, size> a, const Vec128<type, size> b, \
      const Vec128<type, size> c

// HWY_NEON_BUILD_ARG_* is the list of arguments passed to the underlying
// function.
#define HWY_NEON_BUILD_ARG_1 a.raw
#define HWY_NEON_BUILD_ARG_2 a.raw, b.raw
#define HWY_NEON_BUILD_ARG_3 a.raw, b.raw, c.raw

// We use HWY_NEON_EVAL(func, ...) to delay the evaluation of func until after
// the __VA_ARGS__ have been expanded. This allows "func" to be a macro on
// itself like with some of the library "functions" such as vshlq_u8. For
// example, HWY_NEON_EVAL(vshlq_u8, MY_PARAMS) where MY_PARAMS is defined as
// "a, b" (without the quotes) will end up expanding "vshlq_u8(a, b)" if needed.
// Directly writing vshlq_u8(MY_PARAMS) would fail since vshlq_u8() macro
// expects two arguments.
#define HWY_NEON_EVAL(func, ...) func(__VA_ARGS__)

// Main macro definition that defines a single function for the given type and
// size of vector, using the underlying (prefix##infix##suffix) function and
// the template, return type, parameters and arguments defined by the "args"
// parameters passed here (see HWY_NEON_BUILD_* macros defined before).
#define HWY_NEON_DEF_FUNCTION(type, size, name, prefix, infix, suffix, args) \
  HWY_CONCAT(HWY_NEON_BUILD_TPL_, args)                                      \
  HWY_INLINE HWY_CONCAT(HWY_NEON_BUILD_RET_, args)(type, size)               \
      name(HWY_CONCAT(HWY_NEON_BUILD_PARAM_, args)(type, size)) {            \
    return HWY_CONCAT(HWY_NEON_BUILD_RET_, args)(type, size)(                \
        HWY_NEON_EVAL(prefix##infix##suffix, HWY_NEON_BUILD_ARG_##args));    \
  }

// The HWY_NEON_DEF_FUNCTION_* macros define all the variants of a function
// called "name" using the set of neon functions starting with the given
// "prefix" for all the variants of certain types, as specified next to each
// macro. For example, the prefix "vsub" can be used to define the operator-
// using args=2.

// uint8_t
#define HWY_NEON_DEF_FUNCTION_UINT_8(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(uint8_t, 16, name, prefix##q, infix, u8, args) \
  HWY_NEON_DEF_FUNCTION(uint8_t, 8, name, prefix, infix, u8, args)     \
  HWY_NEON_DEF_FUNCTION(uint8_t, 4, name, prefix, infix, u8, args)     \
  HWY_NEON_DEF_FUNCTION(uint8_t, 2, name, prefix, infix, u8, args)

// int8_t
#define HWY_NEON_DEF_FUNCTION_INT_8(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(int8_t, 16, name, prefix##q, infix, s8, args) \
  HWY_NEON_DEF_FUNCTION(int8_t, 8, name, prefix, infix, s8, args)     \
  HWY_NEON_DEF_FUNCTION(int8_t, 4, name, prefix, infix, s8, args)     \
  HWY_NEON_DEF_FUNCTION(int8_t, 2, name, prefix, infix, s8, args)

// uint16_t
#define HWY_NEON_DEF_FUNCTION_UINT_16(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(uint16_t, 8, name, prefix##q, infix, u16, args) \
  HWY_NEON_DEF_FUNCTION(uint16_t, 4, name, prefix, infix, u16, args)    \
  HWY_NEON_DEF_FUNCTION(uint16_t, 2, name, prefix, infix, u16, args)    \
  HWY_NEON_DEF_FUNCTION(uint16_t, 1, name, prefix, infix, u16, args)

// int16_t
#define HWY_NEON_DEF_FUNCTION_INT_16(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(int16_t, 8, name, prefix##q, infix, s16, args) \
  HWY_NEON_DEF_FUNCTION(int16_t, 4, name, prefix, infix, s16, args)    \
  HWY_NEON_DEF_FUNCTION(int16_t, 2, name, prefix, infix, s16, args)    \
  HWY_NEON_DEF_FUNCTION(int16_t, 1, name, prefix, infix, s16, args)

// uint32_t
#define HWY_NEON_DEF_FUNCTION_UINT_32(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(uint32_t, 4, name, prefix##q, infix, u32, args) \
  HWY_NEON_DEF_FUNCTION(uint32_t, 2, name, prefix, infix, u32, args)    \
  HWY_NEON_DEF_FUNCTION(uint32_t, 1, name, prefix, infix, u32, args)

// int32_t
#define HWY_NEON_DEF_FUNCTION_INT_32(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(int32_t, 4, name, prefix##q, infix, s32, args) \
  HWY_NEON_DEF_FUNCTION(int32_t, 2, name, prefix, infix, s32, args)    \
  HWY_NEON_DEF_FUNCTION(int32_t, 1, name, prefix, infix, s32, args)

// uint64_t
#define HWY_NEON_DEF_FUNCTION_UINT_64(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(uint64_t, 2, name, prefix##q, infix, u64, args) \
  HWY_NEON_DEF_FUNCTION(uint64_t, 1, name, prefix, infix, u64, args)

// int64_t
#define HWY_NEON_DEF_FUNCTION_INT_64(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(int64_t, 2, name, prefix##q, infix, s64, args) \
  HWY_NEON_DEF_FUNCTION(int64_t, 1, name, prefix, infix, s64, args)

// float and double
#if defined(__aarch64__)
#define HWY_NEON_DEF_FUNCTION_ALL_FLOATS(name, prefix, infix, args)   \
  HWY_NEON_DEF_FUNCTION(float, 4, name, prefix##q, infix, f32, args)  \
  HWY_NEON_DEF_FUNCTION(float, 2, name, prefix, infix, f32, args)     \
  HWY_NEON_DEF_FUNCTION(float, 1, name, prefix, infix, f32, args)     \
  HWY_NEON_DEF_FUNCTION(double, 2, name, prefix##q, infix, f64, args) \
  HWY_NEON_DEF_FUNCTION(double, 1, name, prefix, infix, f64, args)
#else
#define HWY_NEON_DEF_FUNCTION_ALL_FLOATS(name, prefix, infix, args)  \
  HWY_NEON_DEF_FUNCTION(float, 4, name, prefix##q, infix, f32, args) \
  HWY_NEON_DEF_FUNCTION(float, 2, name, prefix, infix, f32, args)    \
  HWY_NEON_DEF_FUNCTION(float, 1, name, prefix, infix, f32, args)
#endif

// Helper macros to define for more than one type.
// uint8_t, uint16_t and uint32_t
#define HWY_NEON_DEF_FUNCTION_UINT_8_16_32(name, prefix, infix, args) \
  HWY_NEON_DEF_FUNCTION_UINT_8(name, prefix, infix, args)             \
  HWY_NEON_DEF_FUNCTION_UINT_16(name, prefix, infix, args)            \
  HWY_NEON_DEF_FUNCTION_UINT_32(name, prefix, infix, args)

// int8_t, int16_t and int32_t
#define HWY_NEON_DEF_FUNCTION_INT_8_16_32(name, prefix, infix, args) \
  HWY_NEON_DEF_FUNCTION_INT_8(name, prefix, infix, args)             \
  HWY_NEON_DEF_FUNCTION_INT_16(name, prefix, infix, args)            \
  HWY_NEON_DEF_FUNCTION_INT_32(name, prefix, infix, args)

// uint8_t, uint16_t, uint32_t and uint64_t
#define HWY_NEON_DEF_FUNCTION_UINTS(name, prefix, infix, args)  \
  HWY_NEON_DEF_FUNCTION_UINT_8_16_32(name, prefix, infix, args) \
  HWY_NEON_DEF_FUNCTION_UINT_64(name, prefix, infix, args)

// int8_t, int16_t, int32_t and int64_t
#define HWY_NEON_DEF_FUNCTION_INTS(name, prefix, infix, args)  \
  HWY_NEON_DEF_FUNCTION_INT_8_16_32(name, prefix, infix, args) \
  HWY_NEON_DEF_FUNCTION_INT_64(name, prefix, infix, args)

// All int*_t and uint*_t up to 64
#define HWY_NEON_DEF_FUNCTION_INTS_UINTS(name, prefix, infix, args) \
  HWY_NEON_DEF_FUNCTION_INTS(name, prefix, infix, args)             \
  HWY_NEON_DEF_FUNCTION_UINTS(name, prefix, infix, args)

// All previous types.
#define HWY_NEON_DEF_FUNCTION_ALL_TYPES(name, prefix, infix, args) \
  HWY_NEON_DEF_FUNCTION_INTS_UINTS(name, prefix, infix, args)      \
  HWY_NEON_DEF_FUNCTION_ALL_FLOATS(name, prefix, infix, args)

// Emulation of some intrinsics on armv7.
#if !defined(__aarch64__)
#define vuzp1_s8(x, y) vuzp_s8(x, y).val[0]
#define vuzp1_u8(x, y) vuzp_u8(x, y).val[0]
#define vuzp1_s16(x, y) vuzp_s16(x, y).val[0]
#define vuzp1_u16(x, y) vuzp_u16(x, y).val[0]
#define vuzp1_s32(x, y) vuzp_s32(x, y).val[0]
#define vuzp1_u32(x, y) vuzp_u32(x, y).val[0]
#define vuzp1_f32(x, y) vuzp_f32(x, y).val[0]
#define vuzp1q_s8(x, y) vuzpq_s8(x, y).val[0]
#define vuzp1q_u8(x, y) vuzpq_u8(x, y).val[0]
#define vuzp1q_s16(x, y) vuzpq_s16(x, y).val[0]
#define vuzp1q_u16(x, y) vuzpq_u16(x, y).val[0]
#define vuzp1q_s32(x, y) vuzpq_s32(x, y).val[0]
#define vuzp1q_u32(x, y) vuzpq_u32(x, y).val[0]
#define vuzp1q_f32(x, y) vuzpq_f32(x, y).val[0]
#define vuzp2_s8(x, y) vuzp_s8(x, y).val[1]
#define vuzp2_u8(x, y) vuzp_u8(x, y).val[1]
#define vuzp2_s16(x, y) vuzp_s16(x, y).val[1]
#define vuzp2_u16(x, y) vuzp_u16(x, y).val[1]
#define vuzp2_s32(x, y) vuzp_s32(x, y).val[1]
#define vuzp2_u32(x, y) vuzp_u32(x, y).val[1]
#define vuzp2_f32(x, y) vuzp_f32(x, y).val[1]
#define vuzp2q_s8(x, y) vuzpq_s8(x, y).val[1]
#define vuzp2q_u8(x, y) vuzpq_u8(x, y).val[1]
#define vuzp2q_s16(x, y) vuzpq_s16(x, y).val[1]
#define vuzp2q_u16(x, y) vuzpq_u16(x, y).val[1]
#define vuzp2q_s32(x, y) vuzpq_s32(x, y).val[1]
#define vuzp2q_u32(x, y) vuzpq_u32(x, y).val[1]
#define vuzp2q_f32(x, y) vuzpq_f32(x, y).val[1]
#define vzip1_s8(x, y) vzip_s8(x, y).val[0]
#define vzip1_u8(x, y) vzip_u8(x, y).val[0]
#define vzip1_s16(x, y) vzip_s16(x, y).val[0]
#define vzip1_u16(x, y) vzip_u16(x, y).val[0]
#define vzip1_f32(x, y) vzip_f32(x, y).val[0]
#define vzip1_u32(x, y) vzip_u32(x, y).val[0]
#define vzip1_s32(x, y) vzip_s32(x, y).val[0]
#define vzip1q_s8(x, y) vzipq_s8(x, y).val[0]
#define vzip1q_u8(x, y) vzipq_u8(x, y).val[0]
#define vzip1q_s16(x, y) vzipq_s16(x, y).val[0]
#define vzip1q_u16(x, y) vzipq_u16(x, y).val[0]
#define vzip1q_s32(x, y) vzipq_s32(x, y).val[0]
#define vzip1q_u32(x, y) vzipq_u32(x, y).val[0]
#define vzip1q_f32(x, y) vzipq_f32(x, y).val[0]
#define vzip2_s8(x, y) vzip_s8(x, y).val[1]
#define vzip2_u8(x, y) vzip_u8(x, y).val[1]
#define vzip2_s16(x, y) vzip_s16(x, y).val[1]
#define vzip2_u16(x, y) vzip_u16(x, y).val[1]
#define vzip2_s32(x, y) vzip_s32(x, y).val[1]
#define vzip2_u32(x, y) vzip_u32(x, y).val[1]
#define vzip2_f32(x, y) vzip_f32(x, y).val[1]
#define vzip2q_s8(x, y) vzipq_s8(x, y).val[1]
#define vzip2q_u8(x, y) vzipq_u8(x, y).val[1]
#define vzip2q_s16(x, y) vzipq_s16(x, y).val[1]
#define vzip2q_u16(x, y) vzipq_u16(x, y).val[1]
#define vzip2q_s32(x, y) vzipq_s32(x, y).val[1]
#define vzip2q_u32(x, y) vzipq_u32(x, y).val[1]
#define vzip2q_f32(x, y) vzipq_f32(x, y).val[1]
#endif

namespace hwy {

template <typename T, size_t N>
struct Raw128;

// 128
template <>
struct Raw128<uint8_t, 16> {
  using type = uint8x16_t;
};

template <>
struct Raw128<uint16_t, 8> {
  using type = uint16x8_t;
};

template <>
struct Raw128<uint32_t, 4> {
  using type = uint32x4_t;
};

template <>
struct Raw128<uint64_t, 2> {
  using type = uint64x2_t;
};

template <>
struct Raw128<int8_t, 16> {
  using type = int8x16_t;
};

template <>
struct Raw128<int16_t, 8> {
  using type = int16x8_t;
};

template <>
struct Raw128<int32_t, 4> {
  using type = int32x4_t;
};

template <>
struct Raw128<int64_t, 2> {
  using type = int64x2_t;
};

template <>
struct Raw128<float, 4> {
  using type = float32x4_t;
};

#if defined(__aarch64__)
template <>
struct Raw128<double, 2> {
  using type = float64x2_t;
};
#endif

// 64
template <>
struct Raw128<uint8_t, 8> {
  using type = uint8x8_t;
};

template <>
struct Raw128<uint16_t, 4> {
  using type = uint16x4_t;
};

template <>
struct Raw128<uint32_t, 2> {
  using type = uint32x2_t;
};

template <>
struct Raw128<uint64_t, 1> {
  using type = uint64x1_t;
};

template <>
struct Raw128<int8_t, 8> {
  using type = int8x8_t;
};

template <>
struct Raw128<int16_t, 4> {
  using type = int16x4_t;
};

template <>
struct Raw128<int32_t, 2> {
  using type = int32x2_t;
};

template <>
struct Raw128<int64_t, 1> {
  using type = int64x1_t;
};

template <>
struct Raw128<float, 2> {
  using type = float32x2_t;
};

#if defined(__aarch64__)
template <>
struct Raw128<double, 1> {
  using type = float64x1_t;
};
#endif

// 32 (same as 64)
template <>
struct Raw128<uint8_t, 4> {
  using type = uint8x8_t;
};

template <>
struct Raw128<uint16_t, 2> {
  using type = uint16x4_t;
};

template <>
struct Raw128<uint32_t, 1> {
  using type = uint32x2_t;
};

template <>
struct Raw128<int8_t, 4> {
  using type = int8x8_t;
};

template <>
struct Raw128<int16_t, 2> {
  using type = int16x4_t;
};

template <>
struct Raw128<int32_t, 1> {
  using type = int32x2_t;
};

template <>
struct Raw128<float, 1> {
  using type = float32x2_t;
};

// 16 (same as 64)
template <>
struct Raw128<uint8_t, 2> {
  using type = uint8x8_t;
};

template <>
struct Raw128<uint16_t, 1> {
  using type = uint16x4_t;
};

template <>
struct Raw128<int8_t, 2> {
  using type = int8x8_t;
};

template <>
struct Raw128<int16_t, 1> {
  using type = int16x4_t;
};

template <typename T>
using Full128 = Desc<T, 16 / sizeof(T)>;

template <typename T, size_t N = 16 / sizeof(T)>
class Vec128 {
  using Raw = typename Raw128<T, N>::type;

 public:
  HWY_INLINE Vec128() {}
  Vec128(const Vec128&) = default;
  Vec128& operator=(const Vec128&) = default;
  HWY_INLINE explicit Vec128(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  HWY_INLINE Vec128& operator*=(const Vec128 other) {
    return *this = (*this * other);
  }
  HWY_INLINE Vec128& operator/=(const Vec128 other) {
    return *this = (*this / other);
  }
  HWY_INLINE Vec128& operator+=(const Vec128 other) {
    return *this = (*this + other);
  }
  HWY_INLINE Vec128& operator-=(const Vec128 other) {
    return *this = (*this - other);
  }
  HWY_INLINE Vec128& operator&=(const Vec128 other) {
    return *this = (*this & other);
  }
  HWY_INLINE Vec128& operator|=(const Vec128 other) {
    return *this = (*this | other);
  }
  HWY_INLINE Vec128& operator^=(const Vec128 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// FF..FF or 0, also for floating-point - see README.
template <typename T, size_t N = 16 / sizeof(T)>
class Mask128 {
  using Raw = typename Raw128<T, N>::type;

 public:
  HWY_INLINE Mask128() {}
  Mask128(const Mask128&) = default;
  Mask128& operator=(const Mask128&) = default;
  HWY_INLINE explicit Mask128(const Raw raw) : raw(raw) {}

  Raw raw;
};

// ------------------------------ Cast

// cast_to_u8

// Converts from Vec128<T, N> to Vec128<uint8_t, N * sizeof(T)> using the
// vreinterpret*_u8_*() set of functions.
#define HWY_NEON_BUILD_TPL_HWY_CAST_TO_U8
#define HWY_NEON_BUILD_RET_HWY_CAST_TO_U8(type, size) \
  Vec128<uint8_t, size * sizeof(type)>
#define HWY_NEON_BUILD_PARAM_HWY_CAST_TO_U8(type, size) Vec128<type, size> v
#define HWY_NEON_BUILD_ARG_HWY_CAST_TO_U8 v.raw

// Special case of u8 to u8 since vreinterpret*_u8_u8 is obviously not defined.
template <size_t N>
HWY_INLINE Vec128<uint8_t, N> cast_to_u8(Vec128<uint8_t, N> v) {
  return v;
}

HWY_NEON_DEF_FUNCTION_ALL_FLOATS(cast_to_u8, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_INTS(cast_to_u8, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_UINT_16(cast_to_u8, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_UINT_32(cast_to_u8, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_UINT_64(cast_to_u8, vreinterpret, _u8_, HWY_CAST_TO_U8)

#undef HWY_NEON_BUILD_TPL_HWY_CAST_TO_U8
#undef HWY_NEON_BUILD_RET_HWY_CAST_TO_U8
#undef HWY_NEON_BUILD_PARAM_HWY_CAST_TO_U8
#undef HWY_NEON_BUILD_ARG_HWY_CAST_TO_U8

// cast_u8_to

template <size_t N>
HWY_INLINE Vec128<uint8_t, N> cast_u8_to(Desc<uint8_t, N> /* tag */,
                                         Vec128<uint8_t, N> v) {
  return v;
}

// 64-bit full/part:

// Differentiate between <= 64 bit and (64, 128] (for q suffix on NEON)
#define HWY_IF64(T, N) EnableIf<N != 0 && (N * sizeof(T) <= 8)>* = nullptr

// .. 8-bit
template <size_t N, HWY_IF64(int8_t, N)>
HWY_INLINE Vec128<int8_t, N> cast_u8_to(Desc<int8_t, N> /* tag */,
                                        Vec128<uint8_t, N> v) {
  return Vec128<int8_t, N>(vreinterpret_s8_u8(v.raw));
}
// .. 16-bit
template <size_t N, HWY_IF64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> cast_u8_to(Desc<uint16_t, N> /* tag */,
                                          Vec128<uint8_t, N * 2> v) {
  return Vec128<uint16_t, N>(vreinterpret_u16_u8(v.raw));
}
template <size_t N, HWY_IF64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> cast_u8_to(Desc<int16_t, N> /* tag */,
                                         Vec128<uint8_t, N * 2> v) {
  return Vec128<int16_t, N>(vreinterpret_s16_u8(v.raw));
}
// .. 32-bit
template <size_t N, HWY_IF64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> cast_u8_to(Desc<uint32_t, N> /* tag */,
                                          Vec128<uint8_t, N * 4> v) {
  return Vec128<uint32_t, N>(vreinterpret_u32_u8(v.raw));
}
template <size_t N, HWY_IF64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> cast_u8_to(Desc<int32_t, N> /* tag */,
                                         Vec128<uint8_t, N * 4> v) {
  return Vec128<int32_t, N>(vreinterpret_s32_u8(v.raw));
}
template <size_t N, HWY_IF64(float, N)>
HWY_INLINE Vec128<float, N> cast_u8_to(Desc<float, N> /* tag */,
                                       Vec128<uint8_t, N * 4> v) {
  return Vec128<float, N>(vreinterpret_f32_u8(v.raw));
}
// .. 64-bit
HWY_INLINE Vec128<uint64_t, 1> cast_u8_to(Desc<uint64_t, 1> /* tag */,
                                          Vec128<uint8_t, 1 * 8> v) {
  return Vec128<uint64_t, 1>(vreinterpret_u64_u8(v.raw));
}
HWY_INLINE Vec128<int64_t, 1> cast_u8_to(Desc<int64_t, 1> /* tag */,
                                         Vec128<uint8_t, 1 * 8> v) {
  return Vec128<int64_t, 1>(vreinterpret_s64_u8(v.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double, 1> cast_u8_to(Desc<double, 1> /* tag */,
                                        Vec128<uint8_t, 1 * 8> v) {
  return Vec128<double, 1>(vreinterpret_f64_u8(v.raw));
}
#endif

// 128-bit full/part:

#define HWY_IF128_Q(T, N) EnableIf<(N * sizeof(T) > 8)>* = nullptr

// .. 8-bit
template <size_t N, HWY_IF128_Q(int8_t, N)>
HWY_INLINE Vec128<int8_t, N> cast_u8_to(Desc<int8_t, N> /* tag */,
                                        Vec128<uint8_t, N> v) {
  return Vec128<int8_t, N>(vreinterpretq_s8_u8(v.raw));
}
// .. 16-bit
template <size_t N, HWY_IF128_Q(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> cast_u8_to(Desc<uint16_t, N> /* tag */,
                                          Vec128<uint8_t, N * 2> v) {
  return Vec128<uint16_t, N>(vreinterpretq_u16_u8(v.raw));
}
template <size_t N, HWY_IF128_Q(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> cast_u8_to(Desc<int16_t, N> /* tag */,
                                         Vec128<uint8_t, N * 2> v) {
  return Vec128<int16_t, N>(vreinterpretq_s16_u8(v.raw));
}
// .. 32-bit
template <size_t N, HWY_IF128_Q(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> cast_u8_to(Desc<uint32_t, N> /* tag */,
                                          Vec128<uint8_t, N * 4> v) {
  return Vec128<uint32_t, N>(vreinterpretq_u32_u8(v.raw));
}
template <size_t N, HWY_IF128_Q(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> cast_u8_to(Desc<int32_t, N> /* tag */,
                                         Vec128<uint8_t, N * 4> v) {
  return Vec128<int32_t, N>(vreinterpretq_s32_u8(v.raw));
}
template <size_t N, HWY_IF128_Q(float, N)>
HWY_INLINE Vec128<float, N> cast_u8_to(Desc<float, N> /* tag */,
                                       Vec128<uint8_t, N * 4> v) {
  return Vec128<float, N>(vreinterpretq_f32_u8(v.raw));
}
// .. 64-bit
HWY_INLINE Vec128<uint64_t, 2> cast_u8_to(Desc<uint64_t, 2> /* tag */,
                                          Vec128<uint8_t, 2 * 8> v) {
  return Vec128<uint64_t, 2>(vreinterpretq_u64_u8(v.raw));
}
HWY_INLINE Vec128<int64_t, 2> cast_u8_to(Desc<int64_t, 2> /* tag */,
                                         Vec128<uint8_t, 2 * 8> v) {
  return Vec128<int64_t, 2>(vreinterpretq_s64_u8(v.raw));
}

#if defined(__aarch64__)
HWY_INLINE Vec128<double, 2> cast_u8_to(Desc<double, 2> /* tag */,
                                        Vec128<uint8_t, 2 * 8> v) {
  return Vec128<double, 2>(vreinterpretq_f64_u8(v.raw));
}
#endif

// BitCast
template <typename T, size_t N, typename FromT>
HWY_INLINE Vec128<T, N> BitCast(
    Desc<T, N> d, Vec128<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  const auto u8 = cast_to_u8(v);
  return cast_u8_to(d, u8);
}

// ------------------------------ Set

// Returns a vector with all lanes set to "t".
#define HWY_NEON_BUILD_TPL_HWY_SET1
#define HWY_NEON_BUILD_RET_HWY_SET1(type, size) Vec128<type, size>
#define HWY_NEON_BUILD_PARAM_HWY_SET1(type, size) \
  Desc<type, size> /* tag */, const type t
#define HWY_NEON_BUILD_ARG_HWY_SET1 t

HWY_NEON_DEF_FUNCTION_ALL_TYPES(Set, vdup, _n_, HWY_SET1)

#undef HWY_NEON_BUILD_TPL_HWY_SET1
#undef HWY_NEON_BUILD_RET_HWY_SET1
#undef HWY_NEON_BUILD_PARAM_HWY_SET1
#undef HWY_NEON_BUILD_ARG_HWY_SET1

// Returns an all-zero vector.
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Zero(Desc<T, N> d) {
  return Set(d, 0);
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2>
HWY_INLINE Vec128<T, N> Iota(Desc<T, N> d, const T2 first) {
  alignas(16) T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return Load(d, lanes);
}

// Returns a vector with uninitialized elements.
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Undefined(Desc<T, N> /*d*/) {
  HWY_DIAGNOSTICS(push)
  HWY_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")
  typename Raw128<T, N>::type a;
  return Vec128<T, N>(a);
  HWY_DIAGNOSTICS(pop)
}

// ================================================== ARITHMETIC

// ------------------------------ Addition
HWY_NEON_DEF_FUNCTION_ALL_TYPES(operator+, vadd, _, 2)

// ------------------------------ Subtraction
HWY_NEON_DEF_FUNCTION_ALL_TYPES(operator-, vsub, _, 2)

// ------------------------------ Saturating addition and subtraction
// Only defined for uint8_t, uint16_t and their signed versions, as in other
// architectures.

// Returns a + b clamped to the destination range.
HWY_NEON_DEF_FUNCTION_INT_8(SaturatedAdd, vqadd, _, 2)
HWY_NEON_DEF_FUNCTION_INT_16(SaturatedAdd, vqadd, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_8(SaturatedAdd, vqadd, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_16(SaturatedAdd, vqadd, _, 2)

// Returns a - b clamped to the destination range.
HWY_NEON_DEF_FUNCTION_INT_8(SaturatedSub, vqsub, _, 2)
HWY_NEON_DEF_FUNCTION_INT_16(SaturatedSub, vqsub, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_8(SaturatedSub, vqsub, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_16(SaturatedSub, vqsub, _, 2)

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
HWY_NEON_DEF_FUNCTION_UINT_8(AverageRound, vrhadd, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_16(AverageRound, vrhadd, _, 2)

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
HWY_INLINE Vec128<int8_t, N> Abs(const Vec128<int8_t, N> v) {
  return Vec128<int8_t, N>(vabsq_s8(v.raw));
}
template <size_t N>
HWY_INLINE Vec128<int16_t, N> Abs(const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>(vabsq_s16(v.raw));
}
template <size_t N>
HWY_INLINE Vec128<int32_t, N> Abs(const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>(vabsq_s32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Only defined for ints and uints, except for signed i64 shr.
#define HWY_NEON_BUILD_TPL_HWY_SHIFT template <int kBits>
#define HWY_NEON_BUILD_RET_HWY_SHIFT(type, size) Vec128<type, size>
#define HWY_NEON_BUILD_PARAM_HWY_SHIFT(type, size) const Vec128<type, size> v
#define HWY_NEON_BUILD_ARG_HWY_SHIFT v.raw, kBits

HWY_NEON_DEF_FUNCTION_INTS_UINTS(ShiftLeft, vshl, _n_, HWY_SHIFT)

HWY_NEON_DEF_FUNCTION_UINTS(ShiftRight, vshr, _n_, HWY_SHIFT)
HWY_NEON_DEF_FUNCTION_INT_8_16_32(ShiftRight, vshr, _n_, HWY_SHIFT)

#undef HWY_NEON_BUILD_TPL_HWY_SHIFT
#undef HWY_NEON_BUILD_RET_HWY_SHIFT
#undef HWY_NEON_BUILD_PARAM_HWY_SHIFT
#undef HWY_NEON_BUILD_ARG_HWY_SHIFT

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
template <size_t N>
HWY_INLINE Vec128<uint32_t, N> operator<<(const Vec128<uint32_t, N> v,
                                          const Vec128<uint32_t, N> bits) {
  return Vec128<uint32_t, N>(vshlq_u32(v.raw, bits.raw));
}
template <size_t N>
HWY_INLINE Vec128<uint32_t, N> operator>>(const Vec128<uint32_t, N> v,
                                          const Vec128<uint32_t, N> bits) {
  return Vec128<uint32_t, N>(
      vshlq_u32(v.raw, vnegq_s32(vreinterpretq_s32_u32(bits.raw))));
}
template <size_t N>
HWY_INLINE Vec128<uint64_t, N> operator<<(const Vec128<uint64_t, N> v,
                                          const Vec128<uint64_t, N> bits) {
  return Vec128<uint64_t, N>(vshlq_u64(v.raw, bits.raw));
}
template <size_t N>
HWY_INLINE Vec128<uint64_t, N> operator>>(const Vec128<uint64_t, N> v,
                                          const Vec128<uint64_t, N> bits) {
#if !defined(__aarch64__)
  // A32 doesn't have vnegq_s64().
  return Vec128<uint64_t, N>(
      vshlq_u64(v.raw, vsubq_s64(Set(Desc<int64_t, N>(), 0).raw,
                                 vreinterpretq_s64_u64(bits.raw))));
#else
  return Vec128<uint64_t, N>(
      vshlq_u64(v.raw, vnegq_s64(vreinterpretq_s64_u64(bits.raw))));
#endif
}

// Signed (no i8,i16)
template <size_t N>
HWY_INLINE Vec128<int32_t, N> operator<<(const Vec128<int32_t, N> v,
                                         const Vec128<int32_t, N> bits) {
  return Vec128<int32_t, N>(vshlq_s32(v.raw, bits.raw));
}
template <size_t N>
HWY_INLINE Vec128<int32_t, N> operator>>(const Vec128<int32_t, N> v,
                                         const Vec128<int32_t, N> bits) {
  return Vec128<int32_t, N>(vshlq_s32(v.raw, vnegq_s32(bits.raw)));
}
template <size_t N>
HWY_INLINE Vec128<int64_t, N> operator<<(const Vec128<int64_t, N> v,
                                         const Vec128<int64_t, N> bits) {
  return Vec128<int64_t, N>(vshlq_s64(v.raw, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(Min, vmin, _, 2)

// Signed (no i64)
HWY_NEON_DEF_FUNCTION_INT_8_16_32(Min, vmin, _, 2)

// Float
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Min, vmin, _, 2)

// ------------------------------ Maximum

// Unsigned (no u64)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(Max, vmax, _, 2)

// Signed (no i64)
HWY_NEON_DEF_FUNCTION_INT_8_16_32(Max, vmax, _, 2)

// Float
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Max, vmax, _, 2)

// ------------------------------ Clamping
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Clamp(const Vec128<T, N> v, const Vec128<T, N> lo,
                              const Vec128<T, N> hi) {
  return Min(Max(lo, v), hi);
}
// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
HWY_INLINE Vec128<uint16_t, N> operator*(const Vec128<uint16_t, N> a,
                                         const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(vmulq_u16(a.raw, b.raw));
}
template <size_t N>
HWY_INLINE Vec128<uint32_t, N> operator*(const Vec128<uint32_t, N> a,
                                         const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(vmulq_u32(a.raw, b.raw));
}

// Signed
template <size_t N>
HWY_INLINE Vec128<int16_t, N> operator*(const Vec128<int16_t, N> a,
                                        const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(vmulq_s16(a.raw, b.raw));
}
template <size_t N>
HWY_INLINE Vec128<int32_t, N> operator*(const Vec128<int32_t, N> a,
                                        const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(vmulq_s32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
HWY_INLINE Vec128<int16_t, N> MulHigh(const Vec128<int16_t, N> a,
                                      const Vec128<int16_t, N> b) {
  int32x4_t rlo = vmull_s16(vget_low_s16(a.raw), vget_low_s16(b.raw));
#if defined(__aarch64__)
  int32x4_t rhi = vmull_high_s16(a.raw, b.raw);
#else
  int32x4_t rhi = vmull_s16(vget_high_s16(a.raw), vget_high_s16(b.raw));
#endif
  return Vec128<int16_t, N>(
      vuzp2q_s16(vreinterpretq_s16_s32(rlo), vreinterpretq_s16_s32(rhi)));
}
template <size_t N>
HWY_INLINE Vec128<uint16_t, N> MulHigh(const Vec128<uint16_t, N> a,
                                       const Vec128<uint16_t, N> b) {
  int32x4_t rlo = vmull_u16(vget_low_u16(a.raw), vget_low_u16(b.raw));
#if defined(__aarch64__)
  int32x4_t rhi = vmull_high_u16(a.raw, b.raw);
#else
  int32x4_t rhi = vmull_u16(vget_high_u16(a.raw), vget_high_u16(b.raw));
#endif
  return Vec128<uint16_t, N>(
      vuzp2q_u16(vreinterpretq_u16_u32(rlo), vreinterpretq_u16_u32(rhi)));
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
HWY_INLINE Vec128<int64_t> MulEven(const Vec128<int32_t> a,
                                   const Vec128<int32_t> b) {
  int32x4_t a_packed = vuzp1q_s32(a.raw, a.raw);
  int32x4_t b_packed = vuzp1q_s32(b.raw, b.raw);
  return Vec128<int64_t>(
      vmull_s32(vget_low_s32(a_packed), vget_low_s32(b_packed)));
}
HWY_INLINE Vec128<uint64_t> MulEven(const Vec128<uint32_t> a,
                                    const Vec128<uint32_t> b) {
  uint32x4_t a_packed = vuzp1q_u32(a.raw, a.raw);
  uint32x4_t b_packed = vuzp1q_u32(b.raw, b.raw);
  return Vec128<uint64_t>(
      vmull_u32(vget_low_u32(a_packed), vget_low_u32(b_packed)));
}

// ------------------------------ Floating-point negate

HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Neg, vneg, _, 1)

// ------------------------------ Floating-point mul / div

HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator*, vmul, _, 2)

// Approximate reciprocal
HWY_INLINE Vec128<float, 4> ApproximateReciprocal(const Vec128<float, 4> v) {
  return Vec128<float, 4>(vrecpeq_f32(v.raw));
}
template <size_t N>
HWY_INLINE Vec128<float, N> ApproximateReciprocal(const Vec128<float, N> v) {
  return Vec128<float, N>(vrecpe_f32(v.raw));
}

#if defined(__aarch64__)
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator/, vdiv, _, 2)
#else
// Emulated with approx reciprocal + Newton-Raphson + mul
template <size_t N>
HWY_INLINE Vec128<float, N> operator/(const Vec128<float, N> a,
                                      const Vec128<float, N> b) {
  auto x = ApproximateReciprocal(b);
  // Newton-Raphson on 1/x - b
  const auto two = Set(Desc<float, N>(), 2);
  x = x * (two - b * x);
  x = x * (two - b * x);
  x = x * (two - b * x);
  return a * x;
}
#endif

namespace ext {
// Absolute value of difference.
template <size_t N, HWY_IF64(int8_t, N)>
HWY_INLINE Vec128<float, N> AbsDiff(const Vec128<float, 2> a,
                                    const Vec128<float, 2> b) {
  return Vec128<float, N>(vabd_f32(a.raw, b.raw));
}
HWY_INLINE Vec128<float, 4> AbsDiff(const Vec128<float, 4> a,
                                    const Vec128<float, 4> b) {
  return Vec128<float, 4>(vabdq_f32(a.raw, b.raw));
}
}  // namespace ext

// ------------------------------ Floating-point multiply-add variants

// Returns add + mul * x
#if defined(__aarch64__)
HWY_INLINE Vec128<float, 4> MulAdd(const Vec128<float, 4> mul,
                                   const Vec128<float, 4> x,
                                   const Vec128<float, 4> add) {
  return Vec128<float, 4>(vfmaq_f32(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<float, 2> MulAdd(const Vec128<float, 2> mul,
                                   const Vec128<float, 2> x,
                                   const Vec128<float, 2> add) {
  return Vec128<float, 2>(vfma_f32(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<double, 2> MulAdd(const Vec128<double, 2> mul,
                                    const Vec128<double, 2> x,
                                    const Vec128<double, 2> add) {
  return Vec128<double, 2>(vfmaq_f64(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<double, 1> MulAdd(const Vec128<double, 1> mul,
                                    const Vec128<double, 1> x,
                                    const Vec128<double, 1> add) {
  return Vec128<double, 1>(vfma_f64(add.raw, mul.raw, x.raw));
}
#else
// Emulate FMA for floats.
template <size_t N>
HWY_INLINE Vec128<float, N> MulAdd(const Vec128<float, N> mul,
                                   const Vec128<float, N> x,
                                   const Vec128<float, N> add) {
  return mul * x + add;
}
#endif

// Returns add - mul * x
#if defined(__aarch64__)
template <size_t N>
HWY_INLINE Vec128<float, N> NegMulAdd(const Vec128<float, N> mul,
                                      const Vec128<float, N> x,
                                      const Vec128<float, N> add) {
  return Vec128<float, N>(vfmsq_f32(add.raw, mul.raw, x.raw));
}
template <size_t N>
HWY_INLINE Vec128<double, N> NegMulAdd(const Vec128<double, N> mul,
                                       const Vec128<double, N> x,
                                       const Vec128<double, N> add) {
  return Vec128<double, N>(vfmsq_f64(add.raw, mul.raw, x.raw));
}
#else
// Emulate FMA for floats.
template <size_t N>
HWY_INLINE Vec128<float, N> NegMulAdd(const Vec128<float, N> mul,
                                      const Vec128<float, N> x,
                                      const Vec128<float, N> add) {
  return add - mul * x;
}
#endif

// Slightly more expensive (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
HWY_INLINE Vec128<float, N> MulSub(const Vec128<float, N> mul,
                                   const Vec128<float, N> x,
                                   const Vec128<float, N> sub) {
  return Neg(NegMulAdd(mul, x, sub));
}
template <size_t N>
HWY_INLINE Vec128<double, N> MulSub(const Vec128<double, N> mul,
                                    const Vec128<double, N> x,
                                    const Vec128<double, N> sub) {
  return Neg(NegMulAdd(mul, x, sub));
}

// Returns -mul * x - sub
template <size_t N>
HWY_INLINE Vec128<float, N> NegMulSub(const Vec128<float, N> mul,
                                      const Vec128<float, N> x,
                                      const Vec128<float, N> sub) {
  return Neg(MulAdd(mul, x, sub));
}
template <size_t N>
HWY_INLINE Vec128<double, N> NegMulSub(const Vec128<double, N> mul,
                                       const Vec128<double, N> x,
                                       const Vec128<double, N> sub) {
  return Neg(MulAdd(mul, x, sub));
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
template <size_t N>
HWY_INLINE Vec128<float, N> ApproximateReciprocalSqrt(
    const Vec128<float, N> v) {
  return Vec128<float, N>(vrsqrteq_f32(v.raw));
}

// Full precision square root
#if defined(__aarch64__)
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Sqrt, vsqrt, _, 1)
#else
// Not defined on armv7: emulate with approx reciprocal sqrt + Goldschmidt.
template <size_t N>
HWY_INLINE Vec128<float, N> Sqrt(const Vec128<float, N> v) {
  auto b = v;
  auto Y = ApproximateReciprocalSqrt(v);
  auto x = v * Y;
  const auto half = Set(Desc<float, N>(), 0.5);
  const auto oneandhalf = Set(Desc<float, N>(), 1.5);
  for (size_t i = 0; i < 3; i++) {
    b = b * Y * Y;
    Y = oneandhalf - half * b;
    x = x * Y;
  }
  return IfThenZeroElse(v == Zero(Desc<float, N>()), x);
}
#endif

// ------------------------------ Floating-point rounding

#if defined(__aarch64__)
// Toward nearest integer
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Round, vrndn, _, 1)

// Toward zero, aka truncate
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Trunc, vrnd, _, 1)

// Toward +infinity, aka ceiling
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Ceil, vrndp, _, 1)

// Toward -infinity, aka floor
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Floor, vrndm, _, 1)
#else

template <size_t N>
HWY_INLINE Vec128<float, N> Trunc(const Vec128<float, N> v) {
  const Desc<uint32_t, N> du;
  const Desc<int32_t, N> di;
  const Desc<float, N> df;
  const auto v_bits = BitCast(du, v);
  const auto biased_exp = ShiftRight<23>(v_bits) & Set(du, 0xFF);
  const auto bits_to_remove =
      Set(du, 150) - Max(Min(biased_exp, Set(du, 150)), Set(du, 127));
  const auto mask = (Set(du, 1) << bits_to_remove) - Set(du, 1);
  return BitCast(df, IfThenZeroElse(BitCast(di, biased_exp) < Set(di, 127),
                                    BitCast(di, AndNot(mask, v_bits))));
}

// WARNING: does not quite have the same semantics as what NEON does on
// aarch64. In particular, does not break ties to even.
template <size_t N>
HWY_INLINE Vec128<float, N> Round(const Vec128<float, N> v) {
  const Desc<uint32_t, N> du;
  const Desc<float, N> df;
  const auto sign_mask = BitCast(df, Set(du, 0x80000000u));
  // move 0.5f away from 0 and call truncate.
  return Trunc(v + ((v & sign_mask) | Set(df, 0.5f)));
}

template <size_t N>
HWY_INLINE Vec128<float, N> Ceil(const Vec128<float, N> v) {
  const Desc<uint32_t, N> du;
  const Desc<int32_t, N> di;
  const Desc<float, N> df;
  const auto sign_mask = Set(du, 0x80000000u);
  const auto v_bits = BitCast(du, v);
  const auto biased_exp = ShiftRight<23>(v_bits) & Set(du, 0xFF);
  const auto bits_to_remove =
      Set(du, 150) - Max(Min(biased_exp, Set(du, 150)), Set(du, 127));
  const auto high_bit = Set(du, 1) << bits_to_remove;
  const auto mask = high_bit - Set(du, 1);
  const auto removed_bits = mask & v_bits;
  // number is positive and at least one bit was set in the mantissa
  const auto should_round_up = MaskFromVec(
      BitCast(df, AndNot(VecFromMask(removed_bits == Zero(du)),
                         VecFromMask(Zero(du) == (v_bits & sign_mask)))));
  const auto add_one = IfThenElseZero(should_round_up, Set(df, 1.0f));
  const auto rounded =
      BitCast(df, IfThenZeroElse(BitCast(di, biased_exp) < Set(di, 127),
                                 BitCast(di, AndNot(mask, v_bits))));
  return rounded + add_one;
}

template <size_t N>
HWY_INLINE Vec128<float, N> Floor(const Vec128<float, N> v) {
  const Desc<float, N> df;
  const auto zero = Zero(df);
  return zero - Ceil(zero - v);
}

#endif

// ================================================== COMPARE

#define HWY_NEON_BUILD_TPL_HWY_COMPARE
#define HWY_NEON_BUILD_RET_HWY_COMPARE(type, size) Mask128<type, size>
#define HWY_NEON_BUILD_PARAM_HWY_COMPARE(type, size) \
  const Vec128<type, size> a, const Vec128<type, size> b
#define HWY_NEON_BUILD_ARG_HWY_COMPARE a.raw, b.raw

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator==, vceq, _, HWY_COMPARE)
#if defined(__aarch64__)
HWY_NEON_DEF_FUNCTION_INTS_UINTS(operator==, vceq, _, HWY_COMPARE);
#else
// No 64-bit comparisons on armv7: emulate them.
HWY_NEON_DEF_FUNCTION_INT_8_16_32(operator==, vceq, _, HWY_COMPARE)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(operator==, vceq, _, HWY_COMPARE)

HWY_INLINE Mask128<int64_t, 2> operator==(const Vec128<int64_t, 2> a,
                                          const Vec128<int64_t, 2> b) {
  const Full128<int32_t> d;
  const Full128<int64_t> d64;
  auto a32 = BitCast(d, a);
  auto b32 = BitCast(d, b);
  auto cmp = a32 == b32;
  auto cmp_and = Vec128<int32_t, 4>(vandq_u32(cmp.raw, vrev64q_u32(cmp.raw)));
  return Mask128<int64_t, 2>(BitCast(d64, cmp_and).raw);
}

HWY_INLINE Mask128<uint64_t, 2> operator==(const Vec128<uint64_t, 2> a,
                                           const Vec128<uint64_t, 2> b) {
  const Full128<uint32_t> d;
  const Full128<uint64_t> d64;
  auto a32 = BitCast(d, a);
  auto b32 = BitCast(d, b);
  auto cmp = a32 == b32;
  auto cmp_and = Vec128<uint32_t, 4>(vandq_u32(cmp.raw, vrev64q_u32(cmp.raw)));
  return Mask128<uint64_t, 2>(BitCast(d64, cmp_and).raw);
}

#endif

// ------------------------------ Strict inequality

// Signed/float < (no unsigned)
#if defined(__aarch64__)
HWY_NEON_DEF_FUNCTION_INTS(operator<, vclt, _, HWY_COMPARE)
#else
HWY_NEON_DEF_FUNCTION_INT_8_16_32(operator<, vclt, _, HWY_COMPARE)
#endif
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator<, vclt, _, HWY_COMPARE)

// Signed/float > (no unsigned)
#if defined(__aarch64__)
HWY_NEON_DEF_FUNCTION_INTS(operator>, vcgt, _, HWY_COMPARE)
#else
HWY_NEON_DEF_FUNCTION_INT_8_16_32(operator>, vcgt, _, HWY_COMPARE)
#endif
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator>, vcgt, _, HWY_COMPARE)

// ------------------------------ Weak inequality

// Float <= >=
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator<=, vcle, _, HWY_COMPARE)
HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator>=, vcge, _, HWY_COMPARE)

#undef HWY_NEON_BUILD_TPL_HWY_COMPARE
#undef HWY_NEON_BUILD_RET_HWY_COMPARE
#undef HWY_NEON_BUILD_PARAM_HWY_COMPARE
#undef HWY_NEON_BUILD_ARG_HWY_COMPARE

// ================================================== LOGICAL

// ------------------------------ Bitwise AND
HWY_NEON_DEF_FUNCTION_INTS_UINTS(operator&, vand, _, 2)
// These operator& rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
HWY_INLINE Vec128<float, N> operator&(const Vec128<float, N> a,
                                      const Vec128<float, N> b) {
  const Full128<uint32_t> d;
  return BitCast(Full128<float>(), BitCast(d, a) & BitCast(d, b));
}
template <size_t N>
HWY_INLINE Vec128<double, N> operator&(const Vec128<double, N> a,
                                       const Vec128<double, N> b) {
  const Full128<uint64_t> d;
  return BitCast(Full128<double>(), BitCast(d, a) & BitCast(d, b));
}

// ------------------------------ Bitwise AND-NOT

namespace internal {
// reversed_andnot returns a & ~b.
HWY_NEON_DEF_FUNCTION_INTS_UINTS(reversed_andnot, vbic, _, 2)
}  // namespace internal

// Returns ~not_mask & mask.
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> AndNot(const Vec128<T, N> not_mask,
                               const Vec128<T, N> mask) {
  return internal::reversed_andnot(mask, not_mask);
}

// These AndNot() rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.

template <size_t N>
HWY_INLINE Vec128<float, N> AndNot(const Vec128<float, N> not_mask,
                                   const Vec128<float, N> mask) {
  const Desc<uint32_t, N> du;
  Vec128<uint32_t, N> ret =
      internal::reversed_andnot(BitCast(du, mask), BitCast(du, not_mask));
  return BitCast(Desc<float, N>(), ret);
}

#if defined(__aarch64__)
template <size_t N>
HWY_INLINE Vec128<double, N> AndNot(const Vec128<double, N> not_mask,
                                    const Vec128<double, N> mask) {
  const Desc<uint64_t, N> du;
  Vec128<uint64_t, N> ret =
      internal::reversed_andnot(BitCast(du, mask), BitCast(du, not_mask));
  return BitCast(Desc<double, N>(), ret);
}
#endif

// ------------------------------ Bitwise OR

HWY_NEON_DEF_FUNCTION_INTS_UINTS(operator|, vorr, _, 2)

// These operator| rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
HWY_INLINE Vec128<float, N> operator|(const Vec128<float, N> a,
                                      const Vec128<float, N> b) {
  const Desc<uint32_t, N> d;
  return BitCast(Full128<float>(), BitCast(d, a) | BitCast(d, b));
}
template <size_t N>
HWY_INLINE Vec128<double, N> operator|(const Vec128<double, N> a,
                                       const Vec128<double, N> b) {
  const Desc<uint64_t, N> d;
  return BitCast(Full128<double>(), BitCast(d, a) | BitCast(d, b));
}

// ------------------------------ Bitwise XOR

HWY_NEON_DEF_FUNCTION_INTS_UINTS(operator^, veor, _, 2)

// These operator| rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
HWY_INLINE Vec128<float, N> operator^(const Vec128<float, N> a,
                                      const Vec128<float, N> b) {
  const Desc<uint32_t, N> d;
  return BitCast(Full128<float>(), BitCast(d, a) ^ BitCast(d, b));
}
template <size_t N>
HWY_INLINE Vec128<double, N> operator^(const Vec128<double, N> a,
                                       const Vec128<double, N> b) {
  const Desc<uint64_t, N> d;
  return BitCast(Full128<double>(), BitCast(d, a) ^ BitCast(d, b));
}

// ------------------------------ Make mask

template <typename T, size_t N>
HWY_INLINE Mask128<T, N> TestBit(Vec128<T, N> v, Vec128<T, N> bit) {
  static_assert(!IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

// Mask and Vec are the same (true = FF..FF).
template <typename T, size_t N>
HWY_INLINE Mask128<T, N> MaskFromVec(const Vec128<T, N> v) {
  return Mask128<T, N>(v.raw);
}

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> VecFromMask(const Mask128<T, N> v) {
  return Vec128<T, N>(v.raw);
}

// IfThenElse(mask, yes, no)
// Returns mask ? b : a.
#define HWY_NEON_BUILD_TPL_HWY_IF
#define HWY_NEON_BUILD_RET_HWY_IF(type, size) Vec128<type, size>
#define HWY_NEON_BUILD_PARAM_HWY_IF(type, size)                 \
  const Mask128<type, size> mask, const Vec128<type, size> yes, \
      const Vec128<type, size> no
#define HWY_NEON_BUILD_ARG_HWY_IF mask.raw, yes.raw, no.raw

HWY_NEON_DEF_FUNCTION_ALL_TYPES(IfThenElse, vbsl, _, HWY_IF)

#undef HWY_NEON_BUILD_TPL_HWY_IF
#undef HWY_NEON_BUILD_RET_HWY_IF
#undef HWY_NEON_BUILD_PARAM_HWY_IF
#undef HWY_NEON_BUILD_ARG_HWY_IF

// mask ? yes : 0
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> IfThenElseZero(const Mask128<T, N> mask,
                                       const Vec128<T, N> yes) {
  return yes & VecFromMask(mask);
}

// mask ? 0 : no
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> IfThenZeroElse(const Mask128<T, N> mask,
                                       const Vec128<T, N> no) {
  return AndNot(VecFromMask(mask), no);
}

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> ZeroIfNegative(Vec128<T, N> v) {
  const auto zero = Zero(Desc<T, N>());
  return Max(zero, v);
}

// ================================================== MEMORY

// ------------------------------ Load 128

HWY_INLINE Vec128<uint8_t> LoadU(Full128<uint8_t> /* tag */,
                                 const uint8_t* HWY_RESTRICT aligned) {
  return Vec128<uint8_t>(vld1q_u8(aligned));
}
HWY_INLINE Vec128<uint16_t> LoadU(Full128<uint16_t> /* tag */,
                                  const uint16_t* HWY_RESTRICT aligned) {
  return Vec128<uint16_t>(vld1q_u16(aligned));
}
HWY_INLINE Vec128<uint32_t> LoadU(Full128<uint32_t> /* tag */,
                                  const uint32_t* HWY_RESTRICT aligned) {
  return Vec128<uint32_t>(vld1q_u32(aligned));
}
HWY_INLINE Vec128<uint64_t> LoadU(Full128<uint64_t> /* tag */,
                                  const uint64_t* HWY_RESTRICT aligned) {
  return Vec128<uint64_t>(vld1q_u64(aligned));
}
HWY_INLINE Vec128<int8_t> LoadU(Full128<int8_t> /* tag */,
                                const int8_t* HWY_RESTRICT aligned) {
  return Vec128<int8_t>(vld1q_s8(aligned));
}
HWY_INLINE Vec128<int16_t> LoadU(Full128<int16_t> /* tag */,
                                 const int16_t* HWY_RESTRICT aligned) {
  return Vec128<int16_t>(vld1q_s16(aligned));
}
HWY_INLINE Vec128<int32_t> LoadU(Full128<int32_t> /* tag */,
                                 const int32_t* HWY_RESTRICT aligned) {
  return Vec128<int32_t>(vld1q_s32(aligned));
}
HWY_INLINE Vec128<int64_t> LoadU(Full128<int64_t> /* tag */,
                                 const int64_t* HWY_RESTRICT aligned) {
  return Vec128<int64_t>(vld1q_s64(aligned));
}
HWY_INLINE Vec128<float> LoadU(Full128<float> /* tag */,
                               const float* HWY_RESTRICT aligned) {
  return Vec128<float>(vld1q_f32(aligned));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double> LoadU(Full128<double> /* tag */,
                                const double* HWY_RESTRICT aligned) {
  return Vec128<double>(vld1q_f64(aligned));
}
#endif

template <typename T>
HWY_INLINE Vec128<T> Load(Full128<T> d, const T* HWY_RESTRICT p) {
  return LoadU(d, p);
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
HWY_INLINE Vec128<T> LoadDup128(Full128<T> d, const T* const HWY_RESTRICT p) {
  return LoadU(d, p);
}

// ------------------------------ Load 64

HWY_INLINE Vec128<uint8_t, 8> Load(Desc<uint8_t, 8> /* tag */,
                                   const uint8_t* HWY_RESTRICT p) {
  return Vec128<uint8_t, 8>(vld1_u8(p));
}
HWY_INLINE Vec128<uint16_t, 4> Load(Desc<uint16_t, 4> /* tag */,
                                    const uint16_t* HWY_RESTRICT p) {
  return Vec128<uint16_t, 4>(vld1_u16(p));
}
HWY_INLINE Vec128<uint32_t, 2> Load(Desc<uint32_t, 2> /* tag */,
                                    const uint32_t* HWY_RESTRICT p) {
  return Vec128<uint32_t, 2>(vld1_u32(p));
}
HWY_INLINE Vec128<uint64_t, 1> Load(Desc<uint64_t, 1> /* tag */,
                                    const uint64_t* HWY_RESTRICT p) {
  return Vec128<uint64_t, 1>(vld1_u64(p));
}
HWY_INLINE Vec128<int8_t, 8> Load(Desc<int8_t, 8> /* tag */,
                                  const int8_t* HWY_RESTRICT p) {
  return Vec128<int8_t, 8>(vld1_s8(p));
}
HWY_INLINE Vec128<int16_t, 4> Load(Desc<int16_t, 4> /* tag */,
                                   const int16_t* HWY_RESTRICT p) {
  return Vec128<int16_t, 4>(vld1_s16(p));
}
HWY_INLINE Vec128<int32_t, 2> Load(Desc<int32_t, 2> /* tag */,
                                   const int32_t* HWY_RESTRICT p) {
  return Vec128<int32_t, 2>(vld1_s32(p));
}
HWY_INLINE Vec128<int64_t, 1> Load(Desc<int64_t, 1> /* tag */,
                                   const int64_t* HWY_RESTRICT p) {
  return Vec128<int64_t, 1>(vld1_s64(p));
}
HWY_INLINE Vec128<float, 2> Load(Desc<float, 2> /* tag */,
                                 const float* HWY_RESTRICT p) {
  return Vec128<float, 2>(vld1_f32(p));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double, 1> Load(Desc<double, 1> /* tag */,
                                  const double* HWY_RESTRICT p) {
  return Vec128<double, 1>(vld1_f64(p));
}
#endif

// ------------------------------ Load 32

// In the following load functions, |a| is purposely undefined.
// It is a required parameter to the intrinsic, however
// we don't actually care what is in it, and we don't want
// to introduce extra overhead by initializing it to something.

HWY_INLINE Vec128<uint8_t, 4> Load(Desc<uint8_t, 4> d,
                                   const uint8_t* HWY_RESTRICT p) {
  uint32x2_t a = Undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return Vec128<uint8_t, 4>(vreinterpret_u8_u32(b));
}
HWY_INLINE Vec128<uint16_t, 2> Load(Desc<uint16_t, 2> d,
                                    const uint16_t* HWY_RESTRICT p) {
  uint32x2_t a = Undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return Vec128<uint16_t, 2>(vreinterpret_u16_u32(b));
}
HWY_INLINE Vec128<uint32_t, 1> Load(Desc<uint32_t, 1> d,
                                    const uint32_t* HWY_RESTRICT p) {
  uint32x2_t a = Undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(p, a, 0);
  return Vec128<uint32_t, 1>(b);
}
HWY_INLINE Vec128<int8_t, 4> Load(Desc<int8_t, 4> d,
                                  const int8_t* HWY_RESTRICT p) {
  int32x2_t a = Undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return Vec128<int8_t, 4>(vreinterpret_s8_s32(b));
}
HWY_INLINE Vec128<int16_t, 2> Load(Desc<int16_t, 2> d,
                                   const int16_t* HWY_RESTRICT p) {
  int32x2_t a = Undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return Vec128<int16_t, 2>(vreinterpret_s16_s32(b));
}
HWY_INLINE Vec128<int32_t, 1> Load(Desc<int32_t, 1> d,
                                   const int32_t* HWY_RESTRICT p) {
  int32x2_t a = Undefined(d).raw;
  int32x2_t b = vld1_lane_s32(p, a, 0);
  return Vec128<int32_t, 1>(b);
}
HWY_INLINE Vec128<float, 1> Load(Desc<float, 1> d,
                                 const float* HWY_RESTRICT p) {
  float32x2_t a = Undefined(d).raw;
  float32x2_t b = vld1_lane_f32(p, a, 0);
  return Vec128<float, 1>(b);
}

// ------------------------------ Store 128

HWY_INLINE void StoreU(const Vec128<uint8_t> v, Full128<uint8_t> /* tag */,
                       uint8_t* HWY_RESTRICT aligned) {
  vst1q_u8(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<uint16_t> v, Full128<uint16_t> /* tag */,
                       uint16_t* HWY_RESTRICT aligned) {
  vst1q_u16(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<uint32_t> v, Full128<uint32_t> /* tag */,
                       uint32_t* HWY_RESTRICT aligned) {
  vst1q_u32(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<uint64_t> v, Full128<uint64_t> /* tag */,
                       uint64_t* HWY_RESTRICT aligned) {
  vst1q_u64(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int8_t> v, Full128<int8_t> /* tag */,
                       int8_t* HWY_RESTRICT aligned) {
  vst1q_s8(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int16_t> v, Full128<int16_t> /* tag */,
                       int16_t* HWY_RESTRICT aligned) {
  vst1q_s16(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int32_t> v, Full128<int32_t> /* tag */,
                       int32_t* HWY_RESTRICT aligned) {
  vst1q_s32(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int64_t> v, Full128<int64_t> /* tag */,
                       int64_t* HWY_RESTRICT aligned) {
  vst1q_s64(aligned, v.raw);
}
HWY_INLINE void StoreU(const Vec128<float> v, Full128<float> /* tag */,
                       float* HWY_RESTRICT aligned) {
  vst1q_f32(aligned, v.raw);
}
#if defined(__aarch64__)
HWY_INLINE void StoreU(const Vec128<double> v, Full128<double> /* tag */,
                       double* HWY_RESTRICT aligned) {
  vst1q_f64(aligned, v.raw);
}
#endif

template <typename T, size_t N>
HWY_INLINE void Store(Vec128<T, N> v, Desc<T, N> d, T* HWY_RESTRICT p) {
  StoreU(v, d, p);
}

// ------------------------------ Store 64

HWY_INLINE void Store(const Vec128<uint8_t, 8> v, Desc<uint8_t, 8>,
                      uint8_t* HWY_RESTRICT p) {
  vst1_u8(p, v.raw);
}
HWY_INLINE void Store(const Vec128<uint16_t, 4> v, Desc<uint16_t, 4>,
                      uint16_t* HWY_RESTRICT p) {
  vst1_u16(p, v.raw);
}
HWY_INLINE void Store(const Vec128<uint32_t, 2> v, Desc<uint32_t, 2>,
                      uint32_t* HWY_RESTRICT p) {
  vst1_u32(p, v.raw);
}
HWY_INLINE void Store(const Vec128<uint64_t, 1> v, Desc<uint64_t, 1>,
                      uint64_t* HWY_RESTRICT p) {
  vst1_u64(p, v.raw);
}
HWY_INLINE void Store(const Vec128<int8_t, 8> v, Desc<int8_t, 8>,
                      int8_t* HWY_RESTRICT p) {
  vst1_s8(p, v.raw);
}
HWY_INLINE void Store(const Vec128<int16_t, 4> v, Desc<int16_t, 4>,
                      int16_t* HWY_RESTRICT p) {
  vst1_s16(p, v.raw);
}
HWY_INLINE void Store(const Vec128<int32_t, 2> v, Desc<int32_t, 2>,
                      int32_t* HWY_RESTRICT p) {
  vst1_s32(p, v.raw);
}
HWY_INLINE void Store(const Vec128<int64_t, 1> v, Desc<int64_t, 1>,
                      int64_t* HWY_RESTRICT p) {
  vst1_s64(p, v.raw);
}
HWY_INLINE void Store(const Vec128<float, 2> v, Desc<float, 2>,
                      float* HWY_RESTRICT p) {
  vst1_f32(p, v.raw);
}
#if defined(__aarch64__)
HWY_INLINE void Store(const Vec128<double, 1> v, Desc<double, 1>,
                      double* HWY_RESTRICT p) {
  vst1_f64(p, v.raw);
}
#endif

// ------------------------------ Store 32

HWY_INLINE void Store(const Vec128<uint8_t, 4> v, Desc<uint8_t, 4>,
                      uint8_t* HWY_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u8(v.raw);
  vst1_lane_u32(p, a, 0);
}
HWY_INLINE void Store(const Vec128<uint16_t, 2> v, Desc<uint16_t, 2>,
                      uint16_t* HWY_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u16(v.raw);
  vst1_lane_u32(p, a, 0);
}
HWY_INLINE void Store(const Vec128<uint32_t, 1> v, Desc<uint32_t, 1>,
                      uint32_t* HWY_RESTRICT p) {
  vst1_lane_u32(p, v.raw, 0);
}
HWY_INLINE void Store(const Vec128<int8_t, 4> v, Desc<int8_t, 4>,
                      int8_t* HWY_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s8(v.raw);
  vst1_lane_s32(p, a, 0);
}
HWY_INLINE void Store(const Vec128<int16_t, 2> v, Desc<int16_t, 2>,
                      int16_t* HWY_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s16(v.raw);
  vst1_lane_s32(p, a, 0);
}
HWY_INLINE void Store(const Vec128<int32_t, 1> v, Desc<int32_t, 1>,
                      int32_t* HWY_RESTRICT p) {
  vst1_lane_s32(p, v.raw, 0);
}
HWY_INLINE void Store(const Vec128<float, 1> v, Desc<float, 1>,
                      float* HWY_RESTRICT p) {
  vst1_lane_f32(p, v.raw, 0);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
HWY_INLINE void Stream(const Vec128<T> v, Full128<T> d,
                       T* HWY_RESTRICT aligned) {
  Store(v, d, aligned);
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
HWY_INLINE Vec128<uint16_t> ConvertTo(Full128<uint16_t> /* tag */,
                                      const Vec128<uint8_t, 8> v) {
  return Vec128<uint16_t>(vmovl_u8(v.raw));
}
HWY_INLINE Vec128<uint32_t> ConvertTo(Full128<uint32_t> /* tag */,
                                      const Vec128<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<uint32_t>(vmovl_u16(vget_low_u16(a)));
}
HWY_INLINE Vec128<uint32_t> ConvertTo(Full128<uint32_t> /* tag */,
                                      const Vec128<uint16_t, 4> v) {
  return Vec128<uint32_t>(vmovl_u16(v.raw));
}
HWY_INLINE Vec128<uint64_t> ConvertTo(Full128<uint64_t> /* tag */,
                                      const Vec128<uint32_t, 2> v) {
  return Vec128<uint64_t>(vmovl_u32(v.raw));
}
HWY_INLINE Vec128<int16_t> ConvertTo(Full128<int16_t> /* tag */,
                                     const Vec128<uint8_t, 8> v) {
  return Vec128<int16_t>(vmovl_u8(v.raw));
}
HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                     const Vec128<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<int32_t>(vreinterpretq_s32_u16(vmovl_u16(vget_low_u16(a))));
}
HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                     const Vec128<uint16_t, 4> v) {
  return Vec128<int32_t>(vmovl_u16(v.raw));
}

HWY_INLINE Vec128<uint32_t> U32FromU8(const Vec128<uint8_t> v) {
  return Vec128<uint32_t>(
      vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(v.raw)))));
}

// Signed: replicate sign bit.
HWY_INLINE Vec128<int16_t> ConvertTo(Full128<int16_t> /* tag */,
                                     const Vec128<int8_t, 8> v) {
  return Vec128<int16_t>(vmovl_s8(v.raw));
}
HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                     const Vec128<int8_t, 4> v) {
  int16x8_t a = vmovl_s8(v.raw);
  return Vec128<int32_t>(vmovl_s16(vget_low_s16(a)));
}
HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                     const Vec128<int16_t, 4> v) {
  return Vec128<int32_t>(vmovl_s16(v.raw));
}
HWY_INLINE Vec128<int64_t> ConvertTo(Full128<int64_t> /* tag */,
                                     const Vec128<int32_t, 2> v) {
  return Vec128<int64_t>(vmovl_s32(v.raw));
}

#if defined(__aarch64__)
HWY_INLINE Vec128<double> ConvertTo(Full128<double> /* tag */,
                                    const Vec128<float, 2> v) {
  // vcvt_high_f64_f32 takes a float32x4_t even it only uses 2 of the values.
  const float32x4_t w = vcombine_f32(v.raw, v.raw);
  return Vec128<double>(vcvt_high_f64_f32(w));
}
#endif

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <size_t N>
HWY_INLINE Vec128<uint16_t, N> ConvertTo(Desc<uint16_t, N> /* tag */,
                                         const Vec128<int32_t> v) {
  return Vec128<uint16_t, N>(vqmovun_s32(v.raw));
}
template <size_t N>
HWY_INLINE Vec128<uint8_t, N> ConvertTo(Desc<uint8_t, N> /* tag */,
                                        const Vec128<uint16_t> v) {
  return Vec128<uint8_t, N>(vqmovn_u16(v.raw));
}

template <size_t N>
HWY_INLINE Vec128<uint8_t, N> ConvertTo(Desc<uint8_t, N> /* tag */,
                                        const Vec128<int16_t> v) {
  return Vec128<uint8_t, N>(vqmovun_s16(v.raw));
}

template <size_t N>
HWY_INLINE Vec128<int16_t, N> ConvertTo(Desc<int16_t, N> /* tag */,
                                        const Vec128<int32_t> v) {
  return Vec128<int16_t, N>(vqmovn_s32(v.raw));
}
template <size_t N>
HWY_INLINE Vec128<int8_t, N> ConvertTo(Desc<int8_t, N> /* tag */,
                                       const Vec128<int16_t> v) {
  return Vec128<int8_t, N>(vqmovn_s16(v.raw));
}

HWY_INLINE Vec128<uint8_t, 4> U8FromU32(const Vec128<uint32_t> v) {
  const uint8x16_t org_v = cast_to_u8(v).raw;
  const uint8x16_t w = vuzp1q_u8(org_v, org_v);
  return Vec128<uint8_t, 4>(vget_low_u8(vuzp1q_u8(w, w)));
}

// In the following ConvertTo functions, |b| is purposely undefined.
// The value a needs to be extended to 128 bits so that vqmovn can be
// used and |b| is undefined so that no extra overhead is introduced.
HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")

template <size_t N>
HWY_INLINE Vec128<uint8_t, N> ConvertTo(Desc<uint8_t, N> /* tag */,
                                        const Vec128<int32_t> v) {
  Vec128<uint16_t, N> a = ConvertTo(Desc<uint16_t, N>(), v);
  Vec128<uint16_t, N> b;
  uint16x8_t c = vcombine_u16(a.raw, b.raw);
  return Vec128<uint8_t, N>(vqmovn_u16(c));
}

template <size_t N>
HWY_INLINE Vec128<int8_t, N> ConvertTo(Desc<int8_t, N> /* tag */,
                                       const Vec128<int32_t> v) {
  Vec128<int16_t, N> a = ConvertTo(Desc<int16_t, N>(), v);
  Vec128<int16_t, N> b;
  uint16x8_t c = vcombine_s16(a.raw, b.raw);
  return Vec128<int8_t, N>(vqmovn_s16(c));
}

HWY_DIAGNOSTICS(pop)

// ------------------------------ Convert i32 <=> f32

template <size_t N>
HWY_INLINE Vec128<float, N> ConvertTo(Desc<float, N> /* tag */,
                                      const Vec128<int32_t, N> v) {
  return Vec128<float, N>(vcvtq_f32_s32(v.raw));
}
// Truncates (rounds toward zero).
template <size_t N>
HWY_INLINE Vec128<int32_t, N> ConvertTo(Desc<int32_t, N> /* tag */,
                                        const Vec128<float, N> v) {
  return Vec128<int32_t, N>(vcvtq_s32_f32(v.raw));
}

#if defined(__aarch64__)
template <size_t N>
HWY_INLINE Vec128<int32_t, N> NearestInt(const Vec128<float, N> v) {
  return Vec128<int32_t, N>(vcvtnq_s32_f32(v.raw));
}
#else
template <size_t N>
HWY_INLINE Vec128<int32_t, N> NearestInt(const Vec128<float, N> v) {
  return ConvertTo(Desc<int32_t, N>(), Round(v));
}
#endif

// ================================================== SWIZZLE

// ------------------------------ Extract lane

HWY_INLINE uint8_t GetLane(const Vec128<uint8_t, 16> v) {
  return vget_lane_u8(vget_low_u8(v.raw), 0);
}
template <size_t N>
HWY_INLINE uint8_t GetLane(const Vec128<uint8_t, N> v) {
  return vget_lane_u8(v.raw, 0);
}

HWY_INLINE int8_t GetLane(const Vec128<int8_t, 16> v) {
  return vget_lane_s8(vget_low_s8(v.raw), 0);
}
template <size_t N>
HWY_INLINE int8_t GetLane(const Vec128<int8_t, N> v) {
  return vget_lane_s8(v.raw, 0);
}

HWY_INLINE uint16_t GetLane(const Vec128<uint16_t, 8> v) {
  return vget_lane_u16(vget_low_u16(v.raw), 0);
}
template <size_t N>
HWY_INLINE uint16_t GetLane(const Vec128<uint16_t, N> v) {
  return vget_lane_u16(v.raw, 0);
}

HWY_INLINE int16_t GetLane(const Vec128<int16_t, 8> v) {
  return vget_lane_s16(vget_low_s16(v.raw), 0);
}
template <size_t N>
HWY_INLINE int16_t GetLane(const Vec128<int16_t, N> v) {
  return vget_lane_s16(v.raw, 0);
}

HWY_INLINE uint32_t GetLane(const Vec128<uint32_t, 4> v) {
  return vget_lane_u32(vget_low_u32(v.raw), 0);
}
template <size_t N>
HWY_INLINE uint32_t GetLane(const Vec128<uint32_t, N> v) {
  return vget_lane_u32(v.raw, 0);
}

HWY_INLINE int32_t GetLane(const Vec128<int32_t, 4> v) {
  return vget_lane_s32(vget_low_s32(v.raw), 0);
}
template <size_t N>
HWY_INLINE int32_t GetLane(const Vec128<int32_t, N> v) {
  return vget_lane_s32(v.raw, 0);
}

#if defined(__aarch64__)
HWY_INLINE uint64_t GetLane(const Vec128<uint64_t, 2> v) {
  return vget_lane_u64(vget_low_u64(v.raw), 0);
}
HWY_INLINE uint64_t GetLane(const Vec128<uint64_t, 1> v) {
  return vget_lane_u64(v.raw, 0);
}
HWY_INLINE int64_t GetLane(const Vec128<int64_t, 2> v) {
  return vget_lane_s64(vget_low_s64(v.raw), 0);
}
HWY_INLINE int64_t GetLane(const Vec128<int64_t, 1> v) {
  return vget_lane_s64(v.raw, 0);
}
#endif

HWY_INLINE float GetLane(const Vec128<float, 4> v) {
  return vget_lane_f32(vget_low_f32(v.raw), 0);
}
HWY_INLINE float GetLane(const Vec128<float, 2> v) {
  return vget_lane_f32(v.raw, 0);
}
HWY_INLINE float GetLane(const Vec128<float, 1> v) {
  return vget_lane_f32(v.raw, 0);
}
#if defined(__aarch64__)
HWY_INLINE double GetLane(const Vec128<double, 2> v) {
  return vget_lane_f64(vget_low_f64(v.raw), 0);
}
HWY_INLINE double GetLane(const Vec128<double, 1> v) {
  return vget_lane_f64(v.raw, 0);
}
#endif

// ------------------------------ Extract half

// <= 64 bit: just return different type
template <typename T, size_t N, HWY_IF64(uint8_t, N)>
HWY_INLINE Vec128<T, N / 2> LowerHalf(const Vec128<T, N> v) {
  return Vec128<T, N / 2>(v.raw);
}

template <size_t N, HWY_IF128_Q(uint8_t, N)>
HWY_INLINE Vec128<uint8_t, N / 2> LowerHalf(const Vec128<uint8_t, N> v) {
  return Vec128<uint8_t, N / 2>(vget_low_u8(v.raw));
}
template <size_t N, HWY_IF128_Q(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N / 2> LowerHalf(const Vec128<uint16_t, N> v) {
  return Vec128<uint16_t, N / 2>(vget_low_u16(v.raw));
}
template <size_t N, HWY_IF128_Q(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N / 2> LowerHalf(const Vec128<uint32_t, N> v) {
  return Vec128<uint32_t, N / 2>(vget_low_u32(v.raw));
}
HWY_INLINE Vec128<uint64_t, 1> LowerHalf(const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_low_u64(v.raw));
}
template <size_t N, HWY_IF128_Q(int8_t, N)>
HWY_INLINE Vec128<int8_t, N / 2> LowerHalf(const Vec128<int8_t, N> v) {
  return Vec128<int8_t, N / 2>(vget_low_s8(v.raw));
}
template <size_t N, HWY_IF128_Q(int16_t, N)>
HWY_INLINE Vec128<int16_t, N / 2> LowerHalf(const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N / 2>(vget_low_s16(v.raw));
}
template <size_t N, HWY_IF128_Q(int32_t, N)>
HWY_INLINE Vec128<int32_t, N / 2> LowerHalf(const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N / 2>(vget_low_s32(v.raw));
}
HWY_INLINE Vec128<int64_t, 1> LowerHalf(const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_low_s64(v.raw));
}
template <size_t N, HWY_IF128_Q(float, N)>
HWY_INLINE Vec128<float, N / 2> LowerHalf(const Vec128<float, N> v) {
  return Vec128<float, N / 2>(vget_low_f32(v.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double, 1> LowerHalf(const Vec128<double> v) {
  return Vec128<double, 1>(vget_low_f64(v.raw));
}
#endif

HWY_INLINE Vec128<uint8_t, 8> UpperHalf(const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 8>(vget_high_u8(v.raw));
}
HWY_INLINE Vec128<uint16_t, 4> UpperHalf(const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 4>(vget_high_u16(v.raw));
}
HWY_INLINE Vec128<uint32_t, 2> UpperHalf(const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 2>(vget_high_u32(v.raw));
}
HWY_INLINE Vec128<uint64_t, 1> UpperHalf(const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_high_u64(v.raw));
}
HWY_INLINE Vec128<int8_t, 8> UpperHalf(const Vec128<int8_t> v) {
  return Vec128<int8_t, 8>(vget_high_s8(v.raw));
}
HWY_INLINE Vec128<int16_t, 4> UpperHalf(const Vec128<int16_t> v) {
  return Vec128<int16_t, 4>(vget_high_s16(v.raw));
}
HWY_INLINE Vec128<int32_t, 2> UpperHalf(const Vec128<int32_t> v) {
  return Vec128<int32_t, 2>(vget_high_s32(v.raw));
}
HWY_INLINE Vec128<int64_t, 1> UpperHalf(const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_high_s64(v.raw));
}
HWY_INLINE Vec128<float, 2> UpperHalf(const Vec128<float> v) {
  return Vec128<float, 2>(vget_high_f32(v.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double, 1> UpperHalf(const Vec128<double> v) {
  return Vec128<double, 1>(vget_high_f64(v.raw));
}
#endif

template <typename T>
HWY_INLINE Vec128<T, 8 / sizeof(T)> GetHalf(Lower /* tag */, Vec128<T> v) {
  return LowerHalf(v);
}
template <typename T>
HWY_INLINE Vec128<T, 8 / sizeof(T)> GetHalf(Upper /* tag */, Vec128<T> v) {
  return UpperHalf(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
HWY_INLINE Vec128<T> CombineShiftRightBytes(const Vec128<T> hi,
                                            const Vec128<T> lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  const Full128<uint8_t> d8;
  return BitCast(Full128<T>(),
                 Vec128<uint8_t>(vextq_u8(BitCast(d8, lo).raw,
                                          BitCast(d8, hi).raw, kBytes)));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
HWY_INLINE Vec128<T, N> ShiftLeftBytes(const Vec128<T, N> v) {
  return CombineShiftRightBytes<16 - kBytes>(v, Zero(Full128<T>()));
}

template <int kLanes, typename T>
HWY_INLINE Vec128<T> ShiftLeftLanes(const Vec128<T> v) {
  return ShiftLeftBytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
HWY_INLINE Vec128<T, N> ShiftRightBytes(const Vec128<T, N> v) {
  return CombineShiftRightBytes<kBytes>(Zero(Full128<T>()), v);
}

template <int kLanes, typename T>
HWY_INLINE Vec128<T> ShiftRightLanes(const Vec128<T> v) {
  return ShiftRightBytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Broadcast/splat any lane

#if defined(__aarch64__)
// Unsigned
template <int kLane>
HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<uint16_t>(vdupq_laneq_u16(v.raw, kLane));
}
template <int kLane, size_t N>
HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint16_t>(vdupq_laneq_u16(vcombine_u16(v.raw, v.raw), kLane));
}
template <int kLane>
HWY_INLINE Vec128<uint32_t> Broadcast(const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<uint32_t>(vdupq_laneq_u32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<uint64_t> Broadcast(const Vec128<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<uint64_t>(vdupq_laneq_u64(v.raw, kLane));
}

// Signed
template <int kLane>
HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<int16_t>(vdupq_laneq_s16(v.raw, kLane));
}
template <int kLane, size_t N>
HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int16_t>(vdupq_laneq_s16(vcombine_s16(v.raw, v.raw), kLane));
}
template <int kLane>
HWY_INLINE Vec128<int32_t> Broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<int32_t>(vdupq_laneq_s32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<int64_t> Broadcast(const Vec128<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<int64_t>(vdupq_laneq_s64(v.raw, kLane));
}

// Float
template <int kLane>
HWY_INLINE Vec128<float> Broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<float>(vdupq_laneq_f32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<double> Broadcast(const Vec128<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<double>(vdupq_laneq_f64(v.raw, kLane));
}
#else
// No vdupq_laneq_* on armv7: use vgetq_lane_* + vdupq_n_*.

// Unsigned
template <int kLane>
HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<uint16_t>(vdupq_n_u16(vgetq_lane_u16(v.raw, kLane)));
}
template <int kLane, size_t N>
HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint16_t>(
      vdupq_n_u16(vgetq_lane_u16(vcombine_u16(v.raw, v.raw), kLane)));
}
template <int kLane>
HWY_INLINE Vec128<uint32_t> Broadcast(const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<uint32_t>(vdupq_n_u32(vgetq_lane_u32(v.raw, kLane)));
}
template <int kLane>
HWY_INLINE Vec128<uint64_t> Broadcast(const Vec128<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<uint64_t>(vdupq_n_u64(vgetq_lane_u64(v.raw, kLane)));
}

// Signed
template <int kLane>
HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<int16_t>(vdupq_n_s16(vgetq_lane_s16(v.raw, kLane)));
}
template <int kLane, size_t N>
HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int16_t>(
      vdupq_n_s16(vgetq_lane_s16(vcombine_s16(v.raw, v.raw), kLane)));
}
template <int kLane>
HWY_INLINE Vec128<int32_t> Broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<int32_t>(vdupq_n_s32(vgetq_lane_s32(v.raw, kLane)));
}
template <int kLane>
HWY_INLINE Vec128<int64_t> Broadcast(const Vec128<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<int64_t>(vdupq_n_s64(vgetq_lane_s64(v.raw, kLane)));
}

// Float
template <int kLane>
HWY_INLINE Vec128<float> Broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<float>(vdupq_n_f32(vgetq_lane_f32(v.raw, kLane)));
}
#endif

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
HWY_INLINE Vec128<T> TableLookupBytes(const Vec128<T> bytes,
                                      const Vec128<TI> from) {
  const Full128<uint8_t> d8;
#if defined(__aarch64__)
  return BitCast(Full128<T>(),
                 Vec128<uint8_t>(vqtbl1q_u8(BitCast(d8, bytes).raw,
                                            BitCast(d8, from).raw)));
#else
  uint8x16_t table0 = BitCast(d8, bytes).raw;
  uint8x8x2_t table;
  table.val[0] = vget_low_u8(table0);
  table.val[1] = vget_high_u8(table0);
  uint8x16_t idx = BitCast(d8, from).raw;
  uint8x8_t low = vtbl2_u8(table, vget_low_u8(idx));
  uint8x8_t hi = vtbl2_u8(table, vget_high_u8(idx));
  return BitCast(Full128<T>(), Vec128<uint8_t>(vcombine_u8(low, hi)));
#endif
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec128<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// Shuffle0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// CombineShiftRightBytes but the shuffle_abcd notation is more convenient.

// Swap 32-bit halves in 64-bits
template <typename T>
HWY_INLINE Vec128<T> Shuffle2301(const Vec128<T> v) {
  static constexpr uint8_t bytes[16] = {4,  5,  6,  7,  0, 1, 2,  3,
                                        12, 13, 14, 15, 8, 9, 10, 11};
  return TableLookupBytes(v, Load(Full128<uint8_t>(), bytes));
}

// Swap 64-bit halves
template <typename T>
HWY_INLINE Vec128<T> Shuffle1032(const Vec128<T> v) {
  return CombineShiftRightBytes<8>(v, v);
}
template <typename T>
HWY_INLINE Vec128<T> Shuffle01(const Vec128<T> v) {
  return CombineShiftRightBytes<8>(v, v);
}

// Rotate right 32 bits
template <typename T>
HWY_INLINE Vec128<T> Shuffle0321(const Vec128<T> v) {
  return CombineShiftRightBytes<4>(v, v);
}

// Rotate left 32 bits
template <typename T>
HWY_INLINE Vec128<T> Shuffle2103(const Vec128<T> v) {
  return CombineShiftRightBytes<12>(v, v);
}

// Reverse
template <typename T>
HWY_INLINE Vec128<T> Shuffle0123(const Vec128<T> v) {
  static_assert(sizeof(T) == 4,
                "Shuffle0123 should only be applied to 32-bit types");
  // TODO(janwas): more efficient implementation?,
  // It is possible to use two instructions (vrev64q_u32 and vcombine_u32 of the
  // high/low parts) instead of the extra memory and load.
  static constexpr uint8_t bytes[16] = {12, 13, 14, 15, 8, 9, 10, 11,
                                        4,  5,  6,  7,  0, 1, 2,  3};
  return TableLookupBytes(v, Load(Full128<uint8_t>(), bytes));
}

// ------------------------------ Permute (runtime variable)

// Returned by SetTableIndices for use by TableLookupLanes.
template <typename T>
struct Permute128 {
  uint8x16_t raw;
};

template <typename T>
HWY_INLINE Permute128<T> SetTableIndices(const Full128<T> d,
                                         const int32_t* idx) {
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
  for (size_t i = 0; i < d.N; ++i) {
    if (idx[i] >= static_cast<int32_t>(d.N)) {
      printf("SetTableIndices [%zu] = %d >= %zu\n", i, idx[i], d.N);
      HWY_TRAP();
    }
  }
#else
  (void)d;
#endif

  const Full128<uint8_t> d8;
  alignas(16) uint8_t control[d8.N];
  for (size_t idx_byte = 0; idx_byte < d8.N; ++idx_byte) {
    const size_t idx_lane = idx_byte / sizeof(T);
    const size_t mod = idx_byte % sizeof(T);
    control[idx_byte] = idx[idx_lane] * sizeof(T) + mod;
  }
  return Permute128<T>{Load(d8, control).raw};
}

HWY_INLINE Vec128<uint32_t> TableLookupLanes(const Vec128<uint32_t> v,
                                             const Permute128<uint32_t> idx) {
  return TableLookupBytes(v, Vec128<uint8_t>(idx.raw));
}
HWY_INLINE Vec128<int32_t> TableLookupLanes(const Vec128<int32_t> v,
                                            const Permute128<int32_t> idx) {
  return TableLookupBytes(v, Vec128<uint8_t>(idx.raw));
}
HWY_INLINE Vec128<float> TableLookupLanes(const Vec128<float> v,
                                          const Permute128<float> idx) {
  const Full128<int32_t> di;
  const Full128<float> df;
  return BitCast(df,
                 TableLookupBytes(BitCast(di, v), Vec128<uint8_t>(idx.raw)));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use ZipLo/hi instead (also works with scalar).
HWY_NEON_DEF_FUNCTION_INT_8_16_32(InterleaveLo, vzip1, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(InterleaveLo, vzip1, _, 2)

HWY_NEON_DEF_FUNCTION_INT_8_16_32(InterleaveHi, vzip2, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(InterleaveHi, vzip2, _, 2)

#if defined(__aarch64__)
// For 64 bit types, we only have the "q" version of the function defined as
// interleaving 64-wide registers with 64-wide types in them makes no sense.
HWY_INLINE Vec128<uint64_t> InterleaveLo(const Vec128<uint64_t> a,
                                         const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(vzip1q_u64(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> InterleaveLo(const Vec128<int64_t> a,
                                        const Vec128<int64_t> b) {
  return Vec128<int64_t>(vzip1q_s64(a.raw, b.raw));
}

HWY_INLINE Vec128<uint64_t> InterleaveHi(const Vec128<uint64_t> a,
                                         const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(vzip2q_u64(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> InterleaveHi(const Vec128<int64_t> a,
                                        const Vec128<int64_t> b) {
  return Vec128<int64_t>(vzip2q_s64(a.raw, b.raw));
}
#else
// ARMv7 emulation.
HWY_INLINE Vec128<uint64_t> InterleaveLo(const Vec128<uint64_t> a,
                                         const Vec128<uint64_t> b) {
  auto flip = CombineShiftRightBytes<8>(a, a);
  return CombineShiftRightBytes<8>(b, flip);
}
HWY_INLINE Vec128<int64_t> InterleaveLo(const Vec128<int64_t> a,
                                        const Vec128<int64_t> b) {
  auto flip = CombineShiftRightBytes<8>(a, a);
  return CombineShiftRightBytes<8>(b, flip);
}

HWY_INLINE Vec128<uint64_t> InterleaveHi(const Vec128<uint64_t> a,
                                         const Vec128<uint64_t> b) {
  auto flip = CombineShiftRightBytes<8>(b, b);
  return CombineShiftRightBytes<8>(flip, a);
}
HWY_INLINE Vec128<int64_t> InterleaveHi(const Vec128<int64_t> a,
                                        const Vec128<int64_t> b) {
  auto flip = CombineShiftRightBytes<8>(b, b);
  return CombineShiftRightBytes<8>(flip, a);
}
#endif

// Floats
HWY_INLINE Vec128<float> InterleaveLo(const Vec128<float> a,
                                      const Vec128<float> b) {
  return Vec128<float>(vzip1q_f32(a.raw, b.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double> InterleaveLo(const Vec128<double> a,
                                       const Vec128<double> b) {
  return Vec128<double>(vzip1q_f64(a.raw, b.raw));
}
#endif

HWY_INLINE Vec128<float> InterleaveHi(const Vec128<float> a,
                                      const Vec128<float> b) {
  return Vec128<float>(vzip2q_f32(a.raw, b.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double> InterleaveHi(const Vec128<double> a,
                                       const Vec128<double> b) {
  return Vec128<double>(vzip2q_s64(a.raw, b.raw));
}
#endif

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

HWY_INLINE Vec128<uint16_t> ZipLo(const Vec128<uint8_t> a,
                                  const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(vzip1q_u8(a.raw, b.raw));
}
HWY_INLINE Vec128<uint32_t> ZipLo(const Vec128<uint16_t> a,
                                  const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(vzip1q_u16(a.raw, b.raw));
}
HWY_INLINE Vec128<uint64_t> ZipLo(const Vec128<uint32_t> a,
                                  const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(vzip1q_u32(a.raw, b.raw));
}

HWY_INLINE Vec128<int16_t> ZipLo(const Vec128<int8_t> a,
                                 const Vec128<int8_t> b) {
  return Vec128<int16_t>(vzip1q_s8(a.raw, b.raw));
}
HWY_INLINE Vec128<int32_t> ZipLo(const Vec128<int16_t> a,
                                 const Vec128<int16_t> b) {
  return Vec128<int32_t>(vzip1q_s16(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> ZipLo(const Vec128<int32_t> a,
                                 const Vec128<int32_t> b) {
  return Vec128<int64_t>(vzip1q_s32(a.raw, b.raw));
}

HWY_INLINE Vec128<uint16_t> ZipHi(const Vec128<uint8_t> a,
                                  const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(vzip2q_u8(a.raw, b.raw));
}
HWY_INLINE Vec128<uint32_t> ZipHi(const Vec128<uint16_t> a,
                                  const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(vzip2q_u16(a.raw, b.raw));
}
HWY_INLINE Vec128<uint64_t> ZipHi(const Vec128<uint32_t> a,
                                  const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(vzip2q_u32(a.raw, b.raw));
}

HWY_INLINE Vec128<int16_t> ZipHi(const Vec128<int8_t> a,
                                 const Vec128<int8_t> b) {
  return Vec128<int16_t>(vzip2q_s8(a.raw, b.raw));
}
HWY_INLINE Vec128<int32_t> ZipHi(const Vec128<int16_t> a,
                                 const Vec128<int16_t> b) {
  return Vec128<int32_t>(vzip2q_s16(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> ZipHi(const Vec128<int32_t> a,
                                 const Vec128<int32_t> b) {
  return Vec128<int64_t>(vzip2q_s32(a.raw, b.raw));
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatLoLo(const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return BitCast(Full128<T>(),
                 InterleaveLo(BitCast(d64, lo), BitCast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatHiHi(const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return BitCast(Full128<T>(),
                 InterleaveHi(BitCast(d64, lo), BitCast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatLoHi(const Vec128<T> hi, const Vec128<T> lo) {
  return CombineShiftRightBytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatHiLo(const Vec128<T> hi, const Vec128<T> lo) {
  // TODO(janwas): more efficient implementation?
  alignas(16) const uint8_t kBytes[16] = {
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0};
  const auto vec = BitCast(Full128<T>(), Load(Full128<uint8_t>(), kBytes));
  return IfThenElse(MaskFromVec(vec), lo, hi);
}

// ------------------------------ Odd/even lanes

template <typename T>
HWY_INLINE Vec128<T> OddEven(const Vec128<T> a, const Vec128<T> b) {
  alignas(16) constexpr uint8_t kBytes[16] = {
      ((0 / sizeof(T)) & 1) ? 0 : 0xFF,  ((1 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((2 / sizeof(T)) & 1) ? 0 : 0xFF,  ((3 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((4 / sizeof(T)) & 1) ? 0 : 0xFF,  ((5 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((6 / sizeof(T)) & 1) ? 0 : 0xFF,  ((7 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((8 / sizeof(T)) & 1) ? 0 : 0xFF,  ((9 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((10 / sizeof(T)) & 1) ? 0 : 0xFF, ((11 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((12 / sizeof(T)) & 1) ? 0 : 0xFF, ((13 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((14 / sizeof(T)) & 1) ? 0 : 0xFF, ((15 / sizeof(T)) & 1) ? 0 : 0xFF,
  };
  const auto vec = BitCast(Full128<T>(), Load(Full128<uint8_t>(), kBytes));
  return IfThenElse(MaskFromVec(vec), b, a);
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
HWY_INLINE Vec128<uint64_t> SumsOfU8x8(const Vec128<uint8_t> v) {
  uint16x8_t a = vpaddlq_u8(v.raw);
  uint32x4_t b = vpaddlq_u16(a);
  return Vec128<uint64_t>(vpaddlq_u32(b));
}

#if defined(__aarch64__)
// Supported for 32b and 64b vector types. Returns the sum in each lane.
HWY_INLINE Vec128<uint32_t> SumOfLanes(const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(vdupq_n_u32(vaddvq_u32(v.raw)));
}
HWY_INLINE Vec128<int32_t> SumOfLanes(const Vec128<int32_t> v) {
  return Vec128<int32_t>(vdupq_n_s32(vaddvq_s32(v.raw)));
}
HWY_INLINE Vec128<float> SumOfLanes(const Vec128<float> v) {
  return Vec128<float>(vdupq_n_f32(vaddvq_f32(v.raw)));
}
HWY_INLINE Vec128<uint64_t> SumOfLanes(const Vec128<uint64_t> v) {
  return Vec128<uint64_t>(vdupq_n_u64(vaddvq_u64(v.raw)));
}
HWY_INLINE Vec128<int64_t> SumOfLanes(const Vec128<int64_t> v) {
  return Vec128<int64_t>(vdupq_n_s64(vaddvq_s64(v.raw)));
}
HWY_INLINE Vec128<double> SumOfLanes(const Vec128<double> v) {
  return Vec128<double>(vdupq_n_f64(vaddvq_f64(v.raw)));
}
#else
// ARMv7 version for everything except doubles.
HWY_INLINE Vec128<uint32_t> SumOfLanes(const Vec128<uint32_t> v) {
  uint32x4x2_t v0 = vuzpq_u32(v.raw, v.raw);
  uint32x4_t c0 = vaddq_u32(v0.val[0], v0.val[1]);
  uint32x4x2_t v1 = vuzpq_u32(c0, c0);
  return Vec128<uint32_t>(vaddq_u32(v1.val[0], v1.val[1]));
}
HWY_INLINE Vec128<int32_t> SumOfLanes(const Vec128<int32_t> v) {
  int32x4x2_t v0 = vuzpq_s32(v.raw, v.raw);
  int32x4_t c0 = vaddq_s32(v0.val[0], v0.val[1]);
  int32x4x2_t v1 = vuzpq_s32(c0, c0);
  return Vec128<int32_t>(vaddq_s32(v1.val[0], v1.val[1]));
}
HWY_INLINE Vec128<float> SumOfLanes(const Vec128<float> v) {
  float32x4x2_t v0 = vuzpq_f32(v.raw, v.raw);
  float32x4_t c0 = vaddq_f32(v0.val[0], v0.val[1]);
  float32x4x2_t v1 = vuzpq_f32(c0, c0);
  return Vec128<float>(vaddq_f32(v1.val[0], v1.val[1]));
}
HWY_INLINE Vec128<uint64_t> SumOfLanes(const Vec128<uint64_t> v) {
  return v + CombineShiftRightBytes<8>(v, v);
}
HWY_INLINE Vec128<int64_t> SumOfLanes(const Vec128<int64_t> v) {
  return v + CombineShiftRightBytes<8>(v, v);
}
#endif

// ------------------------------ mask

template <typename T>
HWY_INLINE bool AllFalse(const Mask128<T> v) {
  const auto v64 = BitCast(Full128<uint64_t>(), VecFromMask(v));
  uint32x2_t a = vqmovn_u64(v64.raw);
  return vreinterpret_u64_u32(a)[0] == 0;
}

template <typename T>
HWY_INLINE bool AllTrue(const Mask128<T> v) {
  return AllFalse(VecFromMask(v) == Zero(Full128<T>()));
}

namespace impl {

template <typename T>
HWY_INLINE uint64_t BitsFromMask(SizeTag<1> /*tag*/, const Mask128<T> mask) {
  constexpr uint8x16_t kCollapseMask = {
      1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
  };
  const Full128<uint8_t> du;
  const uint8x16_t values = BitCast(du, VecFromMask(mask)).raw & kCollapseMask;

#if defined(__aarch64__)
  // Can't vaddv - we need two separate bytes (16 bits).
  const uint8x8_t x2 = vget_low_u8(vpaddq_u8(values, values));
  const uint8x8_t x4 = vpadd_u8(x2, x2);
  const uint8x8_t x8 = vpadd_u8(x4, x4);
  return vreinterpret_u16_u8(x8)[0];
#else
  // Don't have vpaddq, so keep doubling lane size.
  const uint16x8_t x2 = vpaddlq_u8(values);
  const uint32x4_t x4 = vpaddlq_u16(x2);
  const uint64x2_t x8 = vpaddlq_u32(x4);
  return (uint64_t(x8[1]) << 8) | x8[0];
#endif
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(SizeTag<2> /*tag*/, const Mask128<T> mask) {
  constexpr uint16x8_t kCollapseMask = {1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80};
  const Full128<uint16_t> du;
  const uint16x8_t values = BitCast(du, VecFromMask(mask)).raw & kCollapseMask;
#if defined(__aarch64__)
  return vaddvq_u16(values);
#else
  const uint32x4_t x2 = vpaddlq_u16(values);
  const uint64x2_t x4 = vpaddlq_u32(x2);
  return x4[0] + x4[1];
#endif
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(SizeTag<4> /*tag*/, const Mask128<T> mask) {
  constexpr uint32x4_t kCollapseMask = {1, 2, 4, 8};
  const Full128<uint32_t> du;
  const uint32x4_t values = BitCast(du, VecFromMask(mask)).raw & kCollapseMask;
#if defined(__aarch64__)
  return vaddvq_u32(values);
#else
  const uint64x2_t x2 = vpaddlq_u32(values);
  return x2[0] + x2[1];
#endif
}

#if defined(__aarch64__)
template <typename T>
HWY_INLINE uint64_t BitsFromMask(SizeTag<8> /*tag*/, const Mask128<T> v) {
  constexpr uint64x2_t kCollapseMask = {1, 2};
  const Full128<uint64_t> du;
  const uint64x2_t values = BitCast(du, VecFromMask(v)).raw & kCollapseMask;
  return vaddvq_u64(values);
}
#endif

// Returns number of lanes whose mask is set.
//
// Masks are either FF..FF or 0. Unfortunately there is no reduce-sub op
// ("vsubv"). ANDing with 1 would work but requires a constant. Negating also
// changes each lane to 1 (if mask set) or 0.

template <typename T>
HWY_INLINE size_t CountTrue(SizeTag<1> /*tag*/, const Mask128<T> mask) {
  const Full128<int8_t> di;
  const int8x16_t ones = vnegq_s8(BitCast(di, VecFromMask(mask)).raw);

#if defined(__aarch64__)
  return vaddvq_s8(ones);
#else
  const int16x8_t x2 = vpaddlq_s8(ones);
  const int32x4_t x4 = vpaddlq_s16(x2);
  const int64x2_t x8 = vpaddlq_s32(x4);
  return x8[0] + x8[1];
#endif
}
template <typename T>
HWY_INLINE size_t CountTrue(SizeTag<2> /*tag*/, const Mask128<T> mask) {
  const Full128<int16_t> di;
  const int16x8_t ones = vnegq_s16(BitCast(di, VecFromMask(mask)).raw);

#if defined(__aarch64__)
  return vaddvq_s16(ones);
#else
  const int32x4_t x2 = vpaddlq_s16(ones);
  const int64x2_t x4 = vpaddlq_s32(x2);
  return x4[0] + x4[1];
#endif
}

template <typename T>
HWY_INLINE size_t CountTrue(SizeTag<4> /*tag*/, const Mask128<T> mask) {
  const Full128<int32_t> di;
  const int32x4_t ones = vnegq_s32(BitCast(di, VecFromMask(mask)).raw);

#if defined(__aarch64__)
  return vaddvq_s32(ones);
#else
  const int64x2_t x2 = vpaddlq_s32(ones);
  return x2[0] + x2[1];
#endif
}

#if defined(__aarch64__)
template <typename T>
HWY_INLINE size_t CountTrue(SizeTag<8> /*tag*/, const Mask128<T> mask) {
  const Full128<int64_t> di;
  const int64x2_t ones = vnegq_s64(BitCast(di, VecFromMask(mask)).raw);
  return vaddvq_s64(ones);
}
#endif

}  // namespace impl

template <typename T>
HWY_INLINE uint64_t BitsFromMask(const Mask128<T> mask) {
  return impl::BitsFromMask(SizeTag<sizeof(T)>(), mask);
}

template <typename T>
HWY_INLINE size_t CountTrue(const Mask128<T> mask) {
  return impl::CountTrue(SizeTag<sizeof(T)>(), mask);
}

}  // namespace ext

}  // namespace hwy

#if !defined(__aarch64__)
#undef vuzp1_s8
#undef vuzp1_u8
#undef vuzp1_s16
#undef vuzp1_u16
#undef vuzp1_s32
#undef vuzp1_u32
#undef vuzp1_f32
#undef vuzp1q_s8
#undef vuzp1q_u8
#undef vuzp1q_s16
#undef vuzp1q_u16
#undef vuzp1q_s32
#undef vuzp1q_u32
#undef vuzp1q_f32
#undef vuzp2_s8
#undef vuzp2_u8
#undef vuzp2_s16
#undef vuzp2_u16
#undef vuzp2_s32
#undef vuzp2_u32
#undef vuzp2_f32
#undef vuzp2q_s8
#undef vuzp2q_u8
#undef vuzp2q_s16
#undef vuzp2q_u16
#undef vuzp2q_s32
#undef vuzp2q_u32
#undef vuzp2q_f32
#undef vzip1_s8
#undef vzip1_u8
#undef vzip1_s16
#undef vzip1_u16
#undef vzip1_s32
#undef vzip1_u32
#undef vzip1_f32
#undef vzip1q_s8
#undef vzip1q_u8
#undef vzip1q_s16
#undef vzip1q_u16
#undef vzip1q_s32
#undef vzip1q_u32
#undef vzip1q_f32
#undef vzip2_s8
#undef vzip2_u8
#undef vzip2_s16
#undef vzip2_u16
#undef vzip2_s32
#undef vzip2_u32
#undef vzip2_f32
#undef vzip2q_s8
#undef vzip2q_u8
#undef vzip2q_s16
#undef vzip2q_u16
#undef vzip2q_s32
#undef vzip2q_u32
#undef vzip2q_f32
#endif

#undef HWY_NEON_BUILD_ARG_1
#undef HWY_NEON_BUILD_ARG_2
#undef HWY_NEON_BUILD_ARG_3
#undef HWY_NEON_BUILD_PARAM_1
#undef HWY_NEON_BUILD_PARAM_2
#undef HWY_NEON_BUILD_PARAM_3
#undef HWY_NEON_BUILD_RET_1
#undef HWY_NEON_BUILD_RET_2
#undef HWY_NEON_BUILD_RET_3
#undef HWY_NEON_BUILD_TPL_1
#undef HWY_NEON_BUILD_TPL_2
#undef HWY_NEON_BUILD_TPL_3
#undef HWY_NEON_DEF_FUNCTION
#undef HWY_NEON_DEF_FUNCTION_ALL_FLOATS
#undef HWY_NEON_DEF_FUNCTION_ALL_TYPES
#undef HWY_NEON_DEF_FUNCTION_INT_8
#undef HWY_NEON_DEF_FUNCTION_INT_16
#undef HWY_NEON_DEF_FUNCTION_INT_32
#undef HWY_NEON_DEF_FUNCTION_INT_8_16_32
#undef HWY_NEON_DEF_FUNCTION_INTS
#undef HWY_NEON_DEF_FUNCTION_INTS_UINTS
#undef HWY_NEON_DEF_FUNCTION_TPL
#undef HWY_NEON_DEF_FUNCTION_UINT_8
#undef HWY_NEON_DEF_FUNCTION_UINT_16
#undef HWY_NEON_DEF_FUNCTION_UINT_32
#undef HWY_NEON_DEF_FUNCTION_UINT_8_16_32
#undef HWY_NEON_DEF_FUNCTION_UINTS
#undef HWY_NEON_EVAL

#undef HWY_CONCAT_IMPL
#undef HWY_CONCAT

#endif  // HWY_ARM64_NEON_H_
