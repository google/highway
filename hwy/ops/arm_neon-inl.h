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

// 128-bit ARM64 NEON vectors and operations.
// External include guard in highway.h - see comment there.

#include <arm_neon.h>

#include "hwy/ops/shared-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

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
  HWY_NEON_DEF_FUNCTION(uint8_t, 2, name, prefix, infix, u8, args)     \
  HWY_NEON_DEF_FUNCTION(uint8_t, 1, name, prefix, infix, u8, args)

// int8_t
#define HWY_NEON_DEF_FUNCTION_INT_8(name, prefix, infix, args)        \
  HWY_NEON_DEF_FUNCTION(int8_t, 16, name, prefix##q, infix, s8, args) \
  HWY_NEON_DEF_FUNCTION(int8_t, 8, name, prefix, infix, s8, args)     \
  HWY_NEON_DEF_FUNCTION(int8_t, 4, name, prefix, infix, s8, args)     \
  HWY_NEON_DEF_FUNCTION(int8_t, 2, name, prefix, infix, s8, args)     \
  HWY_NEON_DEF_FUNCTION(int8_t, 1, name, prefix, infix, s8, args)

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

// 8 (same as 64)
template <>
struct Raw128<uint8_t, 1> {
  using type = uint8x8_t;
};

template <>
struct Raw128<int8_t, 1> {
  using type = int8x8_t;
};

template <typename T>
using Full128 = Simd<T, 16 / sizeof(T)>;

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

// ------------------------------ BitCast

namespace detail {

// Converts from Vec128<T, N> to Vec128<uint8_t, N * sizeof(T)> using the
// vreinterpret*_u8_*() set of functions.
#define HWY_NEON_BUILD_TPL_HWY_CAST_TO_U8
#define HWY_NEON_BUILD_RET_HWY_CAST_TO_U8(type, size) \
  Vec128<uint8_t, size * sizeof(type)>
#define HWY_NEON_BUILD_PARAM_HWY_CAST_TO_U8(type, size) Vec128<type, size> v
#define HWY_NEON_BUILD_ARG_HWY_CAST_TO_U8 v.raw

// Special case of u8 to u8 since vreinterpret*_u8_u8 is obviously not defined.
template <size_t N>
HWY_INLINE Vec128<uint8_t, N> BitCastToByte(Vec128<uint8_t, N> v) {
  return v;
}

HWY_NEON_DEF_FUNCTION_ALL_FLOATS(BitCastToByte, vreinterpret, _u8_,
                                 HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_INTS(BitCastToByte, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_UINT_16(BitCastToByte, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_UINT_32(BitCastToByte, vreinterpret, _u8_, HWY_CAST_TO_U8)
HWY_NEON_DEF_FUNCTION_UINT_64(BitCastToByte, vreinterpret, _u8_, HWY_CAST_TO_U8)

#undef HWY_NEON_BUILD_TPL_HWY_CAST_TO_U8
#undef HWY_NEON_BUILD_RET_HWY_CAST_TO_U8
#undef HWY_NEON_BUILD_PARAM_HWY_CAST_TO_U8
#undef HWY_NEON_BUILD_ARG_HWY_CAST_TO_U8

template <size_t N>
HWY_INLINE Vec128<uint8_t, N> BitCastFromByte(Simd<uint8_t, N> /* tag */,
                                              Vec128<uint8_t, N> v) {
  return v;
}

// 64-bit or less:

template <size_t N, HWY_IF_LE64(int8_t, N)>
HWY_INLINE Vec128<int8_t, N> BitCastFromByte(Simd<int8_t, N> /* tag */,
                                             Vec128<uint8_t, N> v) {
  return Vec128<int8_t, N>(vreinterpret_s8_u8(v.raw));
}
template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> BitCastFromByte(Simd<uint16_t, N> /* tag */,
                                               Vec128<uint8_t, N * 2> v) {
  return Vec128<uint16_t, N>(vreinterpret_u16_u8(v.raw));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> BitCastFromByte(Simd<int16_t, N> /* tag */,
                                              Vec128<uint8_t, N * 2> v) {
  return Vec128<int16_t, N>(vreinterpret_s16_u8(v.raw));
}
template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> BitCastFromByte(Simd<uint32_t, N> /* tag */,
                                               Vec128<uint8_t, N * 4> v) {
  return Vec128<uint32_t, N>(vreinterpret_u32_u8(v.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> BitCastFromByte(Simd<int32_t, N> /* tag */,
                                              Vec128<uint8_t, N * 4> v) {
  return Vec128<int32_t, N>(vreinterpret_s32_u8(v.raw));
}
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> BitCastFromByte(Simd<float, N> /* tag */,
                                            Vec128<uint8_t, N * 4> v) {
  return Vec128<float, N>(vreinterpret_f32_u8(v.raw));
}
HWY_INLINE Vec128<uint64_t, 1> BitCastFromByte(Simd<uint64_t, 1> /* tag */,
                                               Vec128<uint8_t, 1 * 8> v) {
  return Vec128<uint64_t, 1>(vreinterpret_u64_u8(v.raw));
}
HWY_INLINE Vec128<int64_t, 1> BitCastFromByte(Simd<int64_t, 1> /* tag */,
                                              Vec128<uint8_t, 1 * 8> v) {
  return Vec128<int64_t, 1>(vreinterpret_s64_u8(v.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double, 1> BitCastFromByte(Simd<double, 1> /* tag */,
                                             Vec128<uint8_t, 1 * 8> v) {
  return Vec128<double, 1>(vreinterpret_f64_u8(v.raw));
}
#endif

// 128-bit full:

HWY_INLINE Vec128<int8_t> BitCastFromByte(Full128<int8_t> /* tag */,
                                          Vec128<uint8_t> v) {
  return Vec128<int8_t>(vreinterpretq_s8_u8(v.raw));
}
HWY_INLINE Vec128<uint16_t> BitCastFromByte(Full128<uint16_t> /* tag */,
                                            Vec128<uint8_t> v) {
  return Vec128<uint16_t>(vreinterpretq_u16_u8(v.raw));
}
HWY_INLINE Vec128<int16_t> BitCastFromByte(Full128<int16_t> /* tag */,
                                           Vec128<uint8_t> v) {
  return Vec128<int16_t>(vreinterpretq_s16_u8(v.raw));
}
HWY_INLINE Vec128<uint32_t> BitCastFromByte(Full128<uint32_t> /* tag */,
                                            Vec128<uint8_t> v) {
  return Vec128<uint32_t>(vreinterpretq_u32_u8(v.raw));
}
HWY_INLINE Vec128<int32_t> BitCastFromByte(Full128<int32_t> /* tag */,
                                           Vec128<uint8_t> v) {
  return Vec128<int32_t>(vreinterpretq_s32_u8(v.raw));
}
HWY_INLINE Vec128<float> BitCastFromByte(Full128<float> /* tag */,
                                         Vec128<uint8_t> v) {
  return Vec128<float>(vreinterpretq_f32_u8(v.raw));
}
HWY_INLINE Vec128<uint64_t> BitCastFromByte(Full128<uint64_t> /* tag */,
                                            Vec128<uint8_t> v) {
  return Vec128<uint64_t>(vreinterpretq_u64_u8(v.raw));
}
HWY_INLINE Vec128<int64_t> BitCastFromByte(Full128<int64_t> /* tag */,
                                           Vec128<uint8_t> v) {
  return Vec128<int64_t>(vreinterpretq_s64_u8(v.raw));
}

#if defined(__aarch64__)
HWY_INLINE Vec128<double> BitCastFromByte(Full128<double> /* tag */,
                                          Vec128<uint8_t> v) {
  return Vec128<double>(vreinterpretq_f64_u8(v.raw));
}
#endif

}  // namespace detail

template <typename T, size_t N, typename FromT>
HWY_INLINE Vec128<T, N> BitCast(
    Simd<T, N> d, Vec128<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  return detail::BitCastFromByte(d, detail::BitCastToByte(v));
}

// ------------------------------ Set

// Returns a vector with all lanes set to "t".
#define HWY_NEON_BUILD_TPL_HWY_SET1
#define HWY_NEON_BUILD_RET_HWY_SET1(type, size) Vec128<type, size>
#define HWY_NEON_BUILD_PARAM_HWY_SET1(type, size) \
  Simd<type, size> /* tag */, const type t
#define HWY_NEON_BUILD_ARG_HWY_SET1 t

HWY_NEON_DEF_FUNCTION_ALL_TYPES(Set, vdup, _n_, HWY_SET1)

#undef HWY_NEON_BUILD_TPL_HWY_SET1
#undef HWY_NEON_BUILD_RET_HWY_SET1
#undef HWY_NEON_BUILD_PARAM_HWY_SET1
#undef HWY_NEON_BUILD_ARG_HWY_SET1

// Returns an all-zero vector.
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Zero(Simd<T, N> d) {
  return Set(d, 0);
}

// Returns a vector with uninitialized elements.
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Undefined(Simd<T, N> /*d*/) {
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
HWY_INLINE Vec128<int8_t> Abs(const Vec128<int8_t> v) {
  return Vec128<int8_t>(vabsq_s8(v.raw));
}
HWY_INLINE Vec128<int16_t> Abs(const Vec128<int16_t> v) {
  return Vec128<int16_t>(vabsq_s16(v.raw));
}
HWY_INLINE Vec128<int32_t> Abs(const Vec128<int32_t> v) {
  return Vec128<int32_t>(vabsq_s32(v.raw));
}
HWY_INLINE Vec128<float> Abs(const Vec128<float> v) {
  return Vec128<float>{vabsq_f32(v.raw)};
}

template <size_t N, HWY_IF_LE64(int8_t, N)>
HWY_INLINE Vec128<int8_t, N> Abs(const Vec128<int8_t, N> v) {
  return Vec128<int8_t, N>(vabs_s8(v.raw));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> Abs(const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>(vabs_s16(v.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> Abs(const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>(vabs_s32(v.raw));
}
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> Abs(const Vec128<float, N> v) {
  return Vec128<float, N>{vabs_f32(v.raw)};
}

#if defined(__aarch64__)
HWY_INLINE Vec128<double> Abs(const Vec128<double> v) {
  return Vec128<double>{vabsq_f64(v.raw)};
}

HWY_INLINE Vec128<double, 1> Abs(const Vec128<double, 1> v) {
  return Vec128<double, 1>{vabs_f64(v.raw)};
}
#endif

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
HWY_INLINE Vec128<uint32_t> operator<<(const Vec128<uint32_t> v,
                                       const Vec128<uint32_t> bits) {
  return Vec128<uint32_t>(vshlq_u32(v.raw, vreinterpretq_s32_u32(bits.raw)));
}
HWY_INLINE Vec128<uint32_t> operator>>(const Vec128<uint32_t> v,
                                       const Vec128<uint32_t> bits) {
  return Vec128<uint32_t>(
      vshlq_u32(v.raw, vnegq_s32(vreinterpretq_s32_u32(bits.raw))));
}
HWY_INLINE Vec128<uint64_t> operator<<(const Vec128<uint64_t> v,
                                       const Vec128<uint64_t> bits) {
  return Vec128<uint64_t>(vshlq_u64(v.raw, vreinterpretq_s64_u64(bits.raw)));
}
HWY_INLINE Vec128<uint64_t> operator>>(const Vec128<uint64_t> v,
                                       const Vec128<uint64_t> bits) {
#if defined(__aarch64__)
  const int64x2_t neg_bits = vnegq_s64(vreinterpretq_s64_u64(bits.raw));
#else
  // A32 doesn't have vnegq_s64().
  const int64x2_t neg_bits =
      vsubq_s64(Set(Full128<int64_t>(), 0).raw, bits.raw);
#endif
  return Vec128<uint64_t>(vshlq_u64(v.raw, neg_bits));
}

template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> operator<<(const Vec128<uint32_t, N> v,
                                          const Vec128<uint32_t, N> bits) {
  return Vec128<uint32_t, N>(vshl_u32(v.raw, vreinterpret_s32_u32(bits.raw)));
}
template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> operator>>(const Vec128<uint32_t, N> v,
                                          const Vec128<uint32_t, N> bits) {
  return Vec128<uint32_t, N>(
      vshl_u32(v.raw, vneg_s32(vreinterpret_s32_u32(bits.raw))));
}
HWY_INLINE Vec128<uint64_t, 1> operator<<(const Vec128<uint64_t, 1> v,
                                          const Vec128<uint64_t, 1> bits) {
  return Vec128<uint64_t, 1>(vshl_u64(v.raw, vreinterpret_s64_u64(bits.raw)));
}
HWY_INLINE Vec128<uint64_t, 1> operator>>(const Vec128<uint64_t, 1> v,
                                          const Vec128<uint64_t, 1> bits) {
#if defined(__aarch64__)
  const int64x1_t neg_bits = vneg_s64(vreinterpret_s64_u64(bits.raw));
#else
  // A32 doesn't have vneg_s64().
  const int64x1_t neg_bits = vsub_s64(Set(Simd<int64_t, 1>(), 0).raw, bits.raw);
#endif
  return Vec128<uint64_t, 1>(vshl_u64(v.raw, neg_bits));
}

// Signed (no i8,i16)
HWY_INLINE Vec128<int32_t> operator<<(const Vec128<int32_t> v,
                                      const Vec128<int32_t> bits) {
  return Vec128<int32_t>(vshlq_s32(v.raw, bits.raw));
}
HWY_INLINE Vec128<int32_t> operator>>(const Vec128<int32_t> v,
                                      const Vec128<int32_t> bits) {
  return Vec128<int32_t>(vshlq_s32(v.raw, vnegq_s32(bits.raw)));
}
HWY_INLINE Vec128<int64_t> operator<<(const Vec128<int64_t> v,
                                      const Vec128<int64_t> bits) {
  return Vec128<int64_t>(vshlq_s64(v.raw, bits.raw));
}

template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> operator<<(const Vec128<int32_t, N> v,
                                         const Vec128<int32_t, N> bits) {
  return Vec128<int32_t, N>(vshl_s32(v.raw, bits.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> operator>>(const Vec128<int32_t, N> v,
                                         const Vec128<int32_t, N> bits) {
  return Vec128<int32_t, N>(vshl_s32(v.raw, vneg_s32(bits.raw)));
}
HWY_INLINE Vec128<int64_t, 1> operator<<(const Vec128<int64_t, 1> v,
                                         const Vec128<int64_t, 1> bits) {
  return Vec128<int64_t, 1>(vshl_s64(v.raw, bits.raw));
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

// ------------------------------ Integer multiplication

// Unsigned
HWY_INLINE Vec128<uint16_t> operator*(const Vec128<uint16_t> a,
                                      const Vec128<uint16_t> b) {
  return Vec128<uint16_t>(vmulq_u16(a.raw, b.raw));
}
HWY_INLINE Vec128<uint32_t> operator*(const Vec128<uint32_t> a,
                                      const Vec128<uint32_t> b) {
  return Vec128<uint32_t>(vmulq_u32(a.raw, b.raw));
}

template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> operator*(const Vec128<uint16_t, N> a,
                                         const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(vmul_u16(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> operator*(const Vec128<uint32_t, N> a,
                                         const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(vmul_u32(a.raw, b.raw));
}

// Signed
HWY_INLINE Vec128<int16_t> operator*(const Vec128<int16_t> a,
                                     const Vec128<int16_t> b) {
  return Vec128<int16_t>(vmulq_s16(a.raw, b.raw));
}
HWY_INLINE Vec128<int32_t> operator*(const Vec128<int32_t> a,
                                     const Vec128<int32_t> b) {
  return Vec128<int32_t>(vmulq_s32(a.raw, b.raw));
}

template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<int16_t, N> operator*(const Vec128<int16_t, N> a,
                                        const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(vmul_s16(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> operator*(const Vec128<int32_t, N> a,
                                        const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(vmul_s32(a.raw, b.raw));
}

// Returns the upper 16 bits of a * b in each lane.
HWY_INLINE Vec128<int16_t> MulHigh(const Vec128<int16_t> a,
                                   const Vec128<int16_t> b) {
  int32x4_t rlo = vmull_s16(vget_low_s16(a.raw), vget_low_s16(b.raw));
#if defined(__aarch64__)
  int32x4_t rhi = vmull_high_s16(a.raw, b.raw);
#else
  int32x4_t rhi = vmull_s16(vget_high_s16(a.raw), vget_high_s16(b.raw));
#endif
  return Vec128<int16_t>(
      vuzp2q_s16(vreinterpretq_s16_s32(rlo), vreinterpretq_s16_s32(rhi)));
}
HWY_INLINE Vec128<uint16_t> MulHigh(const Vec128<uint16_t> a,
                                    const Vec128<uint16_t> b) {
  uint32x4_t rlo = vmull_u16(vget_low_u16(a.raw), vget_low_u16(b.raw));
#if defined(__aarch64__)
  uint32x4_t rhi = vmull_high_u16(a.raw, b.raw);
#else
  uint32x4_t rhi = vmull_u16(vget_high_u16(a.raw), vget_high_u16(b.raw));
#endif
  return Vec128<uint16_t>(
      vuzp2q_u16(vreinterpretq_u16_u32(rlo), vreinterpretq_u16_u32(rhi)));
}

template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> MulHigh(const Vec128<int16_t, N> a,
                                      const Vec128<int16_t, N> b) {
  int16x8_t hi_lo = vreinterpretq_s16_s32(vmull_s16(a.raw, b.raw));
  return Vec128<int16_t, N>(vget_low_s16(vuzp2q_s16(hi_lo, hi_lo)));
}
template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> MulHigh(const Vec128<uint16_t, N> a,
                                       const Vec128<uint16_t, N> b) {
  uint16x8_t hi_lo = vreinterpretq_u16_u32(vmull_u16(a.raw, b.raw));
  return Vec128<uint16_t, N>(vget_low_u16(vuzp2q_u16(hi_lo, hi_lo)));
}

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

template <size_t N>
HWY_INLINE Vec128<int64_t, (N + 1) / 2> MulEven(const Vec128<int32_t, N> a,
                                                const Vec128<int32_t, N> b) {
  int32x2_t a_packed = vuzp1_s32(a.raw, a.raw);
  int32x2_t b_packed = vuzp1_s32(b.raw, b.raw);
  return Vec128<int64_t, (N + 1) / 2>(
      vget_low_s64(vmull_s32(a_packed, b_packed)));
}
template <size_t N>
HWY_INLINE Vec128<uint64_t, (N + 1) / 2> MulEven(const Vec128<uint32_t, N> a,
                                                 const Vec128<uint32_t, N> b) {
  uint32x2_t a_packed = vuzp1_u32(a.raw, a.raw);
  uint32x2_t b_packed = vuzp1_u32(b.raw, b.raw);
  return Vec128<uint64_t, (N + 1) / 2>(
      vget_low_u64(vmull_u32(a_packed, b_packed)));
}

// ------------------------------ Floating-point negate

HWY_NEON_DEF_FUNCTION_ALL_FLOATS(Neg, vneg, _, 1)
HWY_NEON_DEF_FUNCTION_INT_8_16_32(Neg, vneg, _, 1)

HWY_INLINE Vec128<int64_t, 1> Neg(const Vec128<int64_t, 1> v) {
#if defined(__aarch64__)
  return Vec128<int64_t, 1>(vneg_s64(v.raw));
#else
  return Zero(Simd<int64_t, 1>()) - v;
#endif
}

HWY_INLINE Vec128<int64_t> Neg(const Vec128<int64_t> v) {
#if defined(__aarch64__)
  return Vec128<int64_t>(vnegq_s64(v.raw));
#else
  return Zero(Full128<int64_t>()) - v;
#endif
}

// ------------------------------ Floating-point mul / div

HWY_NEON_DEF_FUNCTION_ALL_FLOATS(operator*, vmul, _, 2)

// Approximate reciprocal
HWY_INLINE Vec128<float> ApproximateReciprocal(const Vec128<float> v) {
  return Vec128<float>(vrecpeq_f32(v.raw));
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
  const auto two = Set(Simd<float, N>(), 2);
  x = x * (two - b * x);
  x = x * (two - b * x);
  x = x * (two - b * x);
  return a * x;
}
#endif

// Absolute value of difference.
HWY_INLINE Vec128<float> AbsDiff(const Vec128<float> a, const Vec128<float> b) {
  return Vec128<float>(vabdq_f32(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> AbsDiff(const Vec128<float, N> a,
                                    const Vec128<float, N> b) {
  return Vec128<float, N>(vabd_f32(a.raw, b.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns add + mul * x
#if defined(__aarch64__)
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> MulAdd(const Vec128<float, N> mul,
                                   const Vec128<float, N> x,
                                   const Vec128<float, N> add) {
  return Vec128<float, N>(vfma_f32(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<float> MulAdd(const Vec128<float> mul, const Vec128<float> x,
                                const Vec128<float> add) {
  return Vec128<float>(vfmaq_f32(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<double, 1> MulAdd(const Vec128<double, 1> mul,
                                    const Vec128<double, 1> x,
                                    const Vec128<double, 1> add) {
  return Vec128<double, 1>(vfma_f64(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<double> MulAdd(const Vec128<double> mul,
                                 const Vec128<double> x,
                                 const Vec128<double> add) {
  return Vec128<double>(vfmaq_f64(add.raw, mul.raw, x.raw));
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
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> NegMulAdd(const Vec128<float, N> mul,
                                      const Vec128<float, N> x,
                                      const Vec128<float, N> add) {
  return Vec128<float, N>(vfms_f32(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<float> NegMulAdd(const Vec128<float> mul,
                                   const Vec128<float> x,
                                   const Vec128<float> add) {
  return Vec128<float>(vfmsq_f32(add.raw, mul.raw, x.raw));
}

HWY_INLINE Vec128<double, 1> NegMulAdd(const Vec128<double, 1> mul,
                                       const Vec128<double, 1> x,
                                       const Vec128<double, 1> add) {
  return Vec128<double, 1>(vfms_f64(add.raw, mul.raw, x.raw));
}
HWY_INLINE Vec128<double> NegMulAdd(const Vec128<double> mul,
                                    const Vec128<double> x,
                                    const Vec128<double> add) {
  return Vec128<double>(vfmsq_f64(add.raw, mul.raw, x.raw));
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

// Returns mul * x - sub
template <size_t N>
HWY_INLINE Vec128<float, N> MulSub(const Vec128<float, N> mul,
                                   const Vec128<float, N> x,
                                   const Vec128<float, N> sub) {
  return MulAdd(mul, x, Neg(sub));
}
template <size_t N>
HWY_INLINE Vec128<double, N> MulSub(const Vec128<double, N> mul,
                                    const Vec128<double, N> x,
                                    const Vec128<double, N> sub) {
  return MulAdd(mul, x, Neg(sub));
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

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
HWY_INLINE Vec128<float> ApproximateReciprocalSqrt(const Vec128<float> v) {
  return Vec128<float>(vrsqrteq_f32(v.raw));
}
template <size_t N>
HWY_INLINE Vec128<float, N> ApproximateReciprocalSqrt(
    const Vec128<float, N> v) {
  return Vec128<float, N>(vrsqrte_f32(v.raw));
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
  const auto half = Set(Simd<float, N>(), 0.5);
  const auto oneandhalf = Set(Simd<float, N>(), 1.5);
  for (size_t i = 0; i < 3; i++) {
    b = b * Y * Y;
    Y = oneandhalf - half * b;
    x = x * Y;
  }
  return IfThenZeroElse(v == Zero(Simd<float, N>()), x);
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
  const Simd<uint32_t, N> du;
  const Simd<int32_t, N> di;
  const Simd<float, N> df;
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
  const Simd<uint32_t, N> du;
  const Simd<float, N> df;
  const auto sign_mask = BitCast(df, Set(du, 0x80000000u));
  // move 0.5f away from 0 and call truncate.
  return Trunc(v + ((v & sign_mask) | Set(df, 0.5f)));
}

template <size_t N>
HWY_INLINE Vec128<float, N> Ceil(const Vec128<float, N> v) {
  const Simd<uint32_t, N> du;
  const Simd<int32_t, N> di;
  const Simd<float, N> df;
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
  const Simd<float, N> df;
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
// No 64-bit comparisons on armv7: emulate them below, after Shuffle2301.
HWY_NEON_DEF_FUNCTION_INT_8_16_32(operator==, vceq, _, HWY_COMPARE)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(operator==, vceq, _, HWY_COMPARE)
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
HWY_NEON_DEF_FUNCTION_INTS_UINTS(And, vand, _, 2)
// These operator& rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
HWY_INLINE Vec128<float, N> And(const Vec128<float, N> a,
                                const Vec128<float, N> b) {
  const Simd<uint32_t, N> d;
  return BitCast(Simd<float, N>(), BitCast(d, a) & BitCast(d, b));
}
template <size_t N>
HWY_INLINE Vec128<double, N> And(const Vec128<double, N> a,
                                 const Vec128<double, N> b) {
  const Simd<uint64_t, N> d;
  return BitCast(Simd<double, N>(), BitCast(d, a) & BitCast(d, b));
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
  const Simd<uint32_t, N> du;
  Vec128<uint32_t, N> ret =
      internal::reversed_andnot(BitCast(du, mask), BitCast(du, not_mask));
  return BitCast(Simd<float, N>(), ret);
}

#if defined(__aarch64__)
template <size_t N>
HWY_INLINE Vec128<double, N> AndNot(const Vec128<double, N> not_mask,
                                    const Vec128<double, N> mask) {
  const Simd<uint64_t, N> du;
  Vec128<uint64_t, N> ret =
      internal::reversed_andnot(BitCast(du, mask), BitCast(du, not_mask));
  return BitCast(Simd<double, N>(), ret);
}
#endif

// ------------------------------ Bitwise OR

HWY_NEON_DEF_FUNCTION_INTS_UINTS(Or, vorr, _, 2)

// These operator| rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
HWY_INLINE Vec128<float, N> Or(const Vec128<float, N> a,
                               const Vec128<float, N> b) {
  const Simd<uint32_t, N> d;
  return BitCast(Simd<float, N>(), BitCast(d, a) | BitCast(d, b));
}
template <size_t N>
HWY_INLINE Vec128<double, N> Or(const Vec128<double, N> a,
                                const Vec128<double, N> b) {
  const Simd<uint64_t, N> d;
  return BitCast(Simd<double, N>(), BitCast(d, a) | BitCast(d, b));
}

// ------------------------------ Bitwise XOR

HWY_NEON_DEF_FUNCTION_INTS_UINTS(Xor, veor, _, 2)

// These operator| rely on the special cases for uint32_t and uint64_t just
// defined by HWY_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
HWY_INLINE Vec128<float, N> Xor(const Vec128<float, N> a,
                                const Vec128<float, N> b) {
  const Simd<uint32_t, N> d;
  return BitCast(Simd<float, N>(), BitCast(d, a) ^ BitCast(d, b));
}
template <size_t N>
HWY_INLINE Vec128<double, N> Xor(const Vec128<double, N> a,
                                 const Vec128<double, N> b) {
  const Simd<uint64_t, N> d;
  return BitCast(Simd<double, N>(), BitCast(d, a) ^ BitCast(d, b));
}

// ------------------------------ Operator overloads (internal-only if float)

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> operator&(const Vec128<T, N> a, const Vec128<T, N> b) {
  return And(a, b);
}

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> operator|(const Vec128<T, N> a, const Vec128<T, N> b) {
  return Or(a, b);
}

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> operator^(const Vec128<T, N> a, const Vec128<T, N> b) {
  return Xor(a, b);
}

// ------------------------------ CopySign

template <typename T, size_t N>
HWY_API Vec128<T, N> CopySign(const Vec128<T, N> magn,
                              const Vec128<T, N> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  const auto msb = SignBit(Simd<T, N>());
  return Or(AndNot(msb, magn), And(msb, sign));
}

template <typename T, size_t N>
HWY_API Vec128<T, N> CopySignToAbs(const Vec128<T, N> abs,
                                   const Vec128<T, N> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  return Or(abs, And(SignBit(Simd<T, N>()), sign));
}

// ------------------------------ Make mask

template <typename T, size_t N>
HWY_INLINE Mask128<T, N> TestBit(Vec128<T, N> v, Vec128<T, N> bit) {
  static_assert(!hwy::IsFloat<T>(), "Only integer vectors supported");
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
  const auto zero = Zero(Simd<T, N>());
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

// ------------------------------ Load 64

HWY_INLINE Vec128<uint8_t, 8> LoadU(Simd<uint8_t, 8> /* tag */,
                                    const uint8_t* HWY_RESTRICT p) {
  return Vec128<uint8_t, 8>(vld1_u8(p));
}
HWY_INLINE Vec128<uint16_t, 4> LoadU(Simd<uint16_t, 4> /* tag */,
                                     const uint16_t* HWY_RESTRICT p) {
  return Vec128<uint16_t, 4>(vld1_u16(p));
}
HWY_INLINE Vec128<uint32_t, 2> LoadU(Simd<uint32_t, 2> /* tag */,
                                     const uint32_t* HWY_RESTRICT p) {
  return Vec128<uint32_t, 2>(vld1_u32(p));
}
HWY_INLINE Vec128<uint64_t, 1> LoadU(Simd<uint64_t, 1> /* tag */,
                                     const uint64_t* HWY_RESTRICT p) {
  return Vec128<uint64_t, 1>(vld1_u64(p));
}
HWY_INLINE Vec128<int8_t, 8> LoadU(Simd<int8_t, 8> /* tag */,
                                   const int8_t* HWY_RESTRICT p) {
  return Vec128<int8_t, 8>(vld1_s8(p));
}
HWY_INLINE Vec128<int16_t, 4> LoadU(Simd<int16_t, 4> /* tag */,
                                    const int16_t* HWY_RESTRICT p) {
  return Vec128<int16_t, 4>(vld1_s16(p));
}
HWY_INLINE Vec128<int32_t, 2> LoadU(Simd<int32_t, 2> /* tag */,
                                    const int32_t* HWY_RESTRICT p) {
  return Vec128<int32_t, 2>(vld1_s32(p));
}
HWY_INLINE Vec128<int64_t, 1> LoadU(Simd<int64_t, 1> /* tag */,
                                    const int64_t* HWY_RESTRICT p) {
  return Vec128<int64_t, 1>(vld1_s64(p));
}
HWY_INLINE Vec128<float, 2> LoadU(Simd<float, 2> /* tag */,
                                  const float* HWY_RESTRICT p) {
  return Vec128<float, 2>(vld1_f32(p));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double, 1> LoadU(Simd<double, 1> /* tag */,
                                   const double* HWY_RESTRICT p) {
  return Vec128<double, 1>(vld1_f64(p));
}
#endif

// ------------------------------ Load 32

// In the following load functions, |a| is purposely undefined.
// It is a required parameter to the intrinsic, however
// we don't actually care what is in it, and we don't want
// to introduce extra overhead by initializing it to something.

HWY_INLINE Vec128<uint8_t, 4> LoadU(Simd<uint8_t, 4> d,
                                    const uint8_t* HWY_RESTRICT p) {
  uint32x2_t a = Undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return Vec128<uint8_t, 4>(vreinterpret_u8_u32(b));
}
HWY_INLINE Vec128<uint16_t, 2> LoadU(Simd<uint16_t, 2> d,
                                     const uint16_t* HWY_RESTRICT p) {
  uint32x2_t a = Undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return Vec128<uint16_t, 2>(vreinterpret_u16_u32(b));
}
HWY_INLINE Vec128<uint32_t, 1> LoadU(Simd<uint32_t, 1> d,
                                     const uint32_t* HWY_RESTRICT p) {
  uint32x2_t a = Undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(p, a, 0);
  return Vec128<uint32_t, 1>(b);
}
HWY_INLINE Vec128<int8_t, 4> LoadU(Simd<int8_t, 4> d,
                                   const int8_t* HWY_RESTRICT p) {
  int32x2_t a = Undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return Vec128<int8_t, 4>(vreinterpret_s8_s32(b));
}
HWY_INLINE Vec128<int16_t, 2> LoadU(Simd<int16_t, 2> d,
                                    const int16_t* HWY_RESTRICT p) {
  int32x2_t a = Undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return Vec128<int16_t, 2>(vreinterpret_s16_s32(b));
}
HWY_INLINE Vec128<int32_t, 1> LoadU(Simd<int32_t, 1> d,
                                    const int32_t* HWY_RESTRICT p) {
  int32x2_t a = Undefined(d).raw;
  int32x2_t b = vld1_lane_s32(p, a, 0);
  return Vec128<int32_t, 1>(b);
}
HWY_INLINE Vec128<float, 1> LoadU(Simd<float, 1> d,
                                  const float* HWY_RESTRICT p) {
  float32x2_t a = Undefined(d).raw;
  float32x2_t b = vld1_lane_f32(p, a, 0);
  return Vec128<float, 1>(b);
}

// ------------------------------ Load 16

HWY_INLINE Vec128<uint8_t, 2> LoadU(Simd<uint8_t, 2> d,
                                    const uint8_t* HWY_RESTRICT p) {
  uint16x4_t a = Undefined(d).raw;
  uint16x4_t b = vld1_lane_u16(reinterpret_cast<const uint16_t*>(p), a, 0);
  return Vec128<uint8_t, 2>(vreinterpret_u8_u16(b));
}
HWY_INLINE Vec128<uint16_t, 1> LoadU(Simd<uint16_t, 1> d,
                                     const uint16_t* HWY_RESTRICT p) {
  uint16x4_t a = Undefined(d).raw;
  uint16x4_t b = vld1_lane_u16(p, a, 0);
  return Vec128<uint16_t, 1>(b);
}

HWY_INLINE Vec128<int8_t, 2> LoadU(Simd<int8_t, 2> d,
                                   const int8_t* HWY_RESTRICT p) {
  int16x4_t a = Undefined(d).raw;
  int16x4_t b = vld1_lane_s16(reinterpret_cast<const int16_t*>(p), a, 0);
  return Vec128<int8_t, 2>(vreinterpret_s8_s16(b));
}
HWY_INLINE Vec128<int16_t, 1> LoadU(Simd<int16_t, 1> d,
                                    const int16_t* HWY_RESTRICT p) {
  int16x4_t a = Undefined(d).raw;
  int16x4_t b = vld1_lane_s16(p, a, 0);
  return Vec128<int16_t, 1>(b);
}

// ------------------------------ Load 8

HWY_INLINE Vec128<uint8_t, 1> LoadU(Simd<uint8_t, 1> d,
                                    const uint8_t* HWY_RESTRICT p) {
  uint8x8_t a = Undefined(d).raw;
  uint8x8_t b = vld1_lane_u8(p, a, 0);
  return Vec128<uint8_t, 1>(b);
}

HWY_INLINE Vec128<int8_t, 1> LoadU(Simd<int8_t, 1> d,
                                   const int8_t* HWY_RESTRICT p) {
  int8x8_t a = Undefined(d).raw;
  int8x8_t b = vld1_lane_s8(p, a, 0);
  return Vec128<int8_t, 1>(b);
}

// On ARM, Load is the same as LoadU.
template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Load(Simd<T, N> d, const T* HWY_RESTRICT p) {
  return LoadU(d, p);
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T, size_t N, HWY_IF_LE128(T, N)>
HWY_INLINE Vec128<T, N> LoadDup128(Simd<T, N> d,
                                   const T* const HWY_RESTRICT p) {
  return LoadU(d, p);
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

// ------------------------------ Store 64

HWY_INLINE void StoreU(const Vec128<uint8_t, 8> v, Simd<uint8_t, 8> /* tag */,
                       uint8_t* HWY_RESTRICT p) {
  vst1_u8(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<uint16_t, 4> v, Simd<uint16_t, 4> /* tag */,
                       uint16_t* HWY_RESTRICT p) {
  vst1_u16(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<uint32_t, 2> v, Simd<uint32_t, 2> /* tag */,
                       uint32_t* HWY_RESTRICT p) {
  vst1_u32(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<uint64_t, 1> v, Simd<uint64_t, 1> /* tag */,
                       uint64_t* HWY_RESTRICT p) {
  vst1_u64(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int8_t, 8> v, Simd<int8_t, 8> /* tag */,
                       int8_t* HWY_RESTRICT p) {
  vst1_s8(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int16_t, 4> v, Simd<int16_t, 4> /* tag */,
                       int16_t* HWY_RESTRICT p) {
  vst1_s16(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int32_t, 2> v, Simd<int32_t, 2> /* tag */,
                       int32_t* HWY_RESTRICT p) {
  vst1_s32(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<int64_t, 1> v, Simd<int64_t, 1> /* tag */,
                       int64_t* HWY_RESTRICT p) {
  vst1_s64(p, v.raw);
}
HWY_INLINE void StoreU(const Vec128<float, 2> v, Simd<float, 2> /* tag */,
                       float* HWY_RESTRICT p) {
  vst1_f32(p, v.raw);
}
#if defined(__aarch64__)
HWY_INLINE void StoreU(const Vec128<double, 1> v, Simd<double, 1> /* tag */,
                       double* HWY_RESTRICT p) {
  vst1_f64(p, v.raw);
}
#endif

// ------------------------------ Store 32

HWY_INLINE void StoreU(const Vec128<uint8_t, 4> v, Simd<uint8_t, 4>,
                       uint8_t* HWY_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u8(v.raw);
  vst1_lane_u32(p, a, 0);
}
HWY_INLINE void StoreU(const Vec128<uint16_t, 2> v, Simd<uint16_t, 2>,
                       uint16_t* HWY_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u16(v.raw);
  vst1_lane_u32(p, a, 0);
}
HWY_INLINE void StoreU(const Vec128<uint32_t, 1> v, Simd<uint32_t, 1>,
                       uint32_t* HWY_RESTRICT p) {
  vst1_lane_u32(p, v.raw, 0);
}
HWY_INLINE void StoreU(const Vec128<int8_t, 4> v, Simd<int8_t, 4>,
                       int8_t* HWY_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s8(v.raw);
  vst1_lane_s32(p, a, 0);
}
HWY_INLINE void StoreU(const Vec128<int16_t, 2> v, Simd<int16_t, 2>,
                       int16_t* HWY_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s16(v.raw);
  vst1_lane_s32(p, a, 0);
}
HWY_INLINE void StoreU(const Vec128<int32_t, 1> v, Simd<int32_t, 1>,
                       int32_t* HWY_RESTRICT p) {
  vst1_lane_s32(p, v.raw, 0);
}
HWY_INLINE void StoreU(const Vec128<float, 1> v, Simd<float, 1>,
                       float* HWY_RESTRICT p) {
  vst1_lane_f32(p, v.raw, 0);
}

// ------------------------------ Store 16

HWY_INLINE void StoreU(const Vec128<uint8_t, 2> v, Simd<uint8_t, 2>,
                       uint8_t* HWY_RESTRICT p) {
  uint16x4_t a = vreinterpret_u16_u8(v.raw);
  vst1_lane_u16(p, a, 0);
}
HWY_INLINE void StoreU(const Vec128<uint16_t, 1> v, Simd<uint16_t, 1>,
                       uint16_t* HWY_RESTRICT p) {
  vst1_lane_u16(p, v.raw, 0);
}
HWY_INLINE void StoreU(const Vec128<int8_t, 2> v, Simd<int8_t, 2>,
                       int8_t* HWY_RESTRICT p) {
  int16x4_t a = vreinterpret_s16_s8(v.raw);
  vst1_lane_s16(p, a, 0);
}
HWY_INLINE void StoreU(const Vec128<int16_t, 1> v, Simd<int16_t, 1>,
                       int16_t* HWY_RESTRICT p) {
  vst1_lane_s16(p, v.raw, 0);
}

// ------------------------------ Store 8

HWY_INLINE void StoreU(const Vec128<uint8_t, 1> v, Simd<uint8_t, 1>,
                       uint8_t* HWY_RESTRICT p) {
  vst1_lane_u8(p, v.raw, 0);
}
HWY_INLINE void StoreU(const Vec128<int8_t, 1> v, Simd<int8_t, 1>,
                       int8_t* HWY_RESTRICT p) {
  vst1_lane_s8(p, v.raw, 0);
}

// On ARM, Store is the same as StoreU.
template <typename T, size_t N>
HWY_INLINE void Store(Vec128<T, N> v, Simd<T, N> d, T* HWY_RESTRICT p) {
  StoreU(v, d, p);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T, size_t N>
HWY_INLINE void Stream(const Vec128<T, N> v, Simd<T, N> d,
                       T* HWY_RESTRICT aligned) {
  Store(v, d, aligned);
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend to full vector.
HWY_INLINE Vec128<uint16_t> PromoteTo(Full128<uint16_t> /* tag */,
                                      const Vec128<uint8_t, 8> v) {
  return Vec128<uint16_t>(vmovl_u8(v.raw));
}
HWY_INLINE Vec128<uint32_t> PromoteTo(Full128<uint32_t> /* tag */,
                                      const Vec128<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<uint32_t>(vmovl_u16(vget_low_u16(a)));
}
HWY_INLINE Vec128<uint32_t> PromoteTo(Full128<uint32_t> /* tag */,
                                      const Vec128<uint16_t, 4> v) {
  return Vec128<uint32_t>(vmovl_u16(v.raw));
}
HWY_INLINE Vec128<uint64_t> PromoteTo(Full128<uint64_t> /* tag */,
                                      const Vec128<uint32_t, 2> v) {
  return Vec128<uint64_t>(vmovl_u32(v.raw));
}
HWY_INLINE Vec128<int16_t> PromoteTo(Full128<int16_t> /* tag */,
                                     const Vec128<uint8_t, 8> v) {
  return Vec128<int16_t>(vmovl_u8(v.raw));
}
HWY_INLINE Vec128<int32_t> PromoteTo(Full128<int32_t> /* tag */,
                                     const Vec128<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<int32_t>(vreinterpretq_s32_u16(vmovl_u16(vget_low_u16(a))));
}
HWY_INLINE Vec128<int32_t> PromoteTo(Full128<int32_t> /* tag */,
                                     const Vec128<uint16_t, 4> v) {
  return Vec128<int32_t>(vmovl_u16(v.raw));
}

// Unsigned: zero-extend to half vector.
template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> PromoteTo(Simd<uint16_t, N> /* tag */,
                                         const Vec128<uint8_t, N> v) {
  return Vec128<uint16_t, N>(vget_low_u16(vmovl_u8(v.raw)));
}
template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> PromoteTo(Simd<uint32_t, N> /* tag */,
                                         const Vec128<uint8_t, N> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<uint32_t, N>(vget_low_u32(vmovl_u16(vget_low_u16(a))));
}
template <size_t N>
HWY_INLINE Vec128<uint32_t, N> PromoteTo(Simd<uint32_t, N> /* tag */,
                                         const Vec128<uint16_t, N> v) {
  return Vec128<uint32_t, N>(vget_low_u32(vmovl_u16(v.raw)));
}
template <size_t N, HWY_IF_LE64(uint64_t, N)>
HWY_INLINE Vec128<uint64_t, N> PromoteTo(Simd<uint64_t, N> /* tag */,
                                         const Vec128<uint32_t, N> v) {
  return Vec128<uint64_t, N>(vget_low_u64(vmovl_u32(v.raw)));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> PromoteTo(Simd<int16_t, N> /* tag */,
                                        const Vec128<uint8_t, N> v) {
  return Vec128<int16_t, N>(vget_low_s16(vmovl_u8(v.raw)));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> PromoteTo(Simd<int32_t, N> /* tag */,
                                        const Vec128<uint8_t, N> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  uint32x4_t b = vmovl_u16(vget_low_u16(a));
  return Vec128<int32_t, N>(vget_low_s32(vreinterpretq_s32_u32(b)));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> PromoteTo(Simd<int32_t, N> /* tag */,
                                        const Vec128<uint16_t, N> v) {
  uint32x4_t a = vmovl_u16(v.raw);
  return Vec128<int32_t, N>(vget_low_s32(vreinterpretq_s32_u32(a)));
}

HWY_INLINE Vec128<uint32_t> U32FromU8(const Vec128<uint8_t> v) {
  return Vec128<uint32_t>(
      vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(v.raw)))));
}

// Signed: replicate sign bit to full vector.
HWY_INLINE Vec128<int16_t> PromoteTo(Full128<int16_t> /* tag */,
                                     const Vec128<int8_t, 8> v) {
  return Vec128<int16_t>(vmovl_s8(v.raw));
}
HWY_INLINE Vec128<int32_t> PromoteTo(Full128<int32_t> /* tag */,
                                     const Vec128<int8_t, 4> v) {
  int16x8_t a = vmovl_s8(v.raw);
  return Vec128<int32_t>(vmovl_s16(vget_low_s16(a)));
}
HWY_INLINE Vec128<int32_t> PromoteTo(Full128<int32_t> /* tag */,
                                     const Vec128<int16_t, 4> v) {
  return Vec128<int32_t>(vmovl_s16(v.raw));
}
HWY_INLINE Vec128<int64_t> PromoteTo(Full128<int64_t> /* tag */,
                                     const Vec128<int32_t, 2> v) {
  return Vec128<int64_t>(vmovl_s32(v.raw));
}

// Signed: replicate sign bit to half vector.
template <size_t N>
HWY_INLINE Vec128<int16_t, N> PromoteTo(Simd<int16_t, N> /* tag */,
                                        const Vec128<int8_t, N> v) {
  return Vec128<int16_t, N>(vget_low_s16(vmovl_s8(v.raw)));
}
template <size_t N>
HWY_INLINE Vec128<int32_t, N> PromoteTo(Simd<int32_t, N> /* tag */,
                                        const Vec128<int8_t, N> v) {
  int16x8_t a = vmovl_s8(v.raw);
  int32x4_t b = vmovl_s16(vget_low_s16(a));
  return Vec128<int32_t, N>(vget_low_s32(b));
}
template <size_t N>
HWY_INLINE Vec128<int32_t, N> PromoteTo(Simd<int32_t, N> /* tag */,
                                        const Vec128<int16_t, N> v) {
  return Vec128<int32_t, N>(vget_low_s32(vmovl_s16(v.raw)));
}
template <size_t N>
HWY_INLINE Vec128<int64_t, N> PromoteTo(Simd<int64_t, N> /* tag */,
                                        const Vec128<int32_t, N> v) {
  return Vec128<int64_t, N>(vget_low_s64(vmovl_s32(v.raw)));
}

#if defined(__aarch64__)
HWY_INLINE Vec128<double> PromoteTo(Full128<double> /* tag */,
                                    const Vec128<float, 2> v) {
  return Vec128<double>(vcvt_f64_f32(v.raw));
}

HWY_INLINE Vec128<double, 1> PromoteTo(Simd<double, 1> /* tag */,
                                       const Vec128<float, 1> v) {
  return Vec128<double, 1>(vget_low_f64(vcvt_f64_f32(v.raw)));
}

HWY_INLINE Vec128<double> PromoteTo(Full128<double> /* tag */,
                                    const Vec128<int32_t, 2> v) {
  const int64x2_t i64 = vmovl_s32(v.raw);
  return Vec128<double>(vcvtq_f64_s64(i64));
}

HWY_INLINE Vec128<double, 1> PromoteTo(Simd<double, 1> /* tag */,
                                       const Vec128<int32_t, 1> v) {
  const int64x1_t i64 = vget_low_s64(vmovl_s32(v.raw));
  return Vec128<double, 1>(vcvt_f64_s64(i64));
}

#endif

// ------------------------------ Demotions (full -> part w/ narrow lanes)

// From full vector to half or quarter
HWY_INLINE Vec128<uint16_t, 4> DemoteTo(Simd<uint16_t, 4> /* tag */,
                                        const Vec128<int32_t> v) {
  return Vec128<uint16_t, 4>(vqmovun_s32(v.raw));
}
HWY_INLINE Vec128<int16_t, 4> DemoteTo(Simd<int16_t, 4> /* tag */,
                                       const Vec128<int32_t> v) {
  return Vec128<int16_t, 4>(vqmovn_s32(v.raw));
}
HWY_INLINE Vec128<uint8_t, 4> DemoteTo(Simd<uint8_t, 4> /* tag */,
                                       const Vec128<int32_t> v) {
  const uint16x4_t a = vqmovun_s32(v.raw);
  return Vec128<uint8_t, 4>(vqmovn_u16(vcombine_u16(a, a)));
}
HWY_INLINE Vec128<uint8_t, 8> DemoteTo(Simd<uint8_t, 8> /* tag */,
                                       const Vec128<int16_t> v) {
  return Vec128<uint8_t, 8>(vqmovun_s16(v.raw));
}
HWY_INLINE Vec128<int8_t, 4> DemoteTo(Simd<int8_t, 4> /* tag */,
                                      const Vec128<int32_t> v) {
  const int16x4_t a = vqmovn_s32(v.raw);
  return Vec128<int8_t, 4>(vqmovn_s16(vcombine_s16(a, a)));
}
HWY_INLINE Vec128<int8_t, 8> DemoteTo(Simd<int8_t, 8> /* tag */,
                                      const Vec128<int16_t> v) {
  return Vec128<int8_t, 8>(vqmovn_s16(v.raw));
}

// From half vector to partial half
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<uint16_t, N> DemoteTo(Simd<uint16_t, N> /* tag */,
                                        const Vec128<int32_t, N> v) {
  return Vec128<uint16_t, N>(vqmovun_s32(vcombine_s32(v.raw, v.raw)));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int16_t, N> DemoteTo(Simd<int16_t, N> /* tag */,
                                       const Vec128<int32_t, N> v) {
  return Vec128<int16_t, N>(vqmovn_s32(vcombine_s32(v.raw, v.raw)));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<uint8_t, N> DemoteTo(Simd<uint8_t, N> /* tag */,
                                       const Vec128<int32_t, N> v) {
  const uint16x4_t a = vqmovun_s32(vcombine_s32(v.raw, v.raw));
  return Vec128<uint8_t, N>(vqmovn_u16(vcombine_u16(a, a)));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<uint8_t, N> DemoteTo(Simd<uint8_t, N> /* tag */,
                                       const Vec128<int16_t, N> v) {
  return Vec128<uint8_t, N>(vqmovun_s16(vcombine_s16(v.raw, v.raw)));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int8_t, N> DemoteTo(Simd<int8_t, N> /* tag */,
                                      const Vec128<int32_t, N> v) {
  const int16x4_t a = vqmovn_s32(vcombine_s32(v.raw, v.raw));
  return Vec128<int8_t, N>(vqmovn_s16(vcombine_s16(a, a)));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int8_t, N> DemoteTo(Simd<int8_t, N> /* tag */,
                                      const Vec128<int16_t, N> v) {
  return Vec128<int8_t, N>(vqmovn_s16(vcombine_s16(v.raw, v.raw)));
}

#if defined(__aarch64__)
HWY_INLINE Vec128<float, 2> DemoteTo(Simd<float, 2> /* tag */,
                                     const Vec128<double> v) {
  return Vec128<float, 2>(vcvt_f32_f64(v.raw));
}
HWY_INLINE Vec128<float, 1> DemoteTo(Simd<float, 1> /* tag */,
                                     const Vec128<double, 1> v) {
  return Vec128<float, 1>(vcvt_f32_f64(vcombine_f64(v.raw, v.raw)));
}

HWY_INLINE Vec128<int32_t, 2> DemoteTo(Simd<int32_t, 2> /* tag */,
                                       const Vec128<double> v) {
  const int64x2_t i64 = vcvtq_s64_f64(v.raw);
  return Vec128<int32_t, 2>(vqmovn_s64(i64));
}
HWY_INLINE Vec128<int32_t, 1> DemoteTo(Simd<int32_t, 1> /* tag */,
                                       const Vec128<double, 1> v) {
  const int64x1_t i64 = vcvt_s64_f64(v.raw);
  // There is no i64x1 -> i32x1 narrow, so expand to int64x2_t first.
  const int64x2_t i64x2 = vcombine_s64(i64, i64);
  return Vec128<int32_t, 1>(vqmovn_s64(i64x2));
}

#endif

HWY_INLINE Vec128<uint8_t, 4> U8FromU32(const Vec128<uint32_t> v) {
  const uint8x16_t org_v = detail::BitCastToByte(v).raw;
  const uint8x16_t w = vuzp1q_u8(org_v, org_v);
  return Vec128<uint8_t, 4>(vget_low_u8(vuzp1q_u8(w, w)));
}

// In the following DemoteTo functions, |b| is purposely undefined.
// The value a needs to be extended to 128 bits so that vqmovn can be
// used and |b| is undefined so that no extra overhead is introduced.
HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")

template <size_t N>
HWY_INLINE Vec128<uint8_t, N> DemoteTo(Simd<uint8_t, N> /* tag */,
                                       const Vec128<int32_t> v) {
  Vec128<uint16_t, N> a = DemoteTo(Simd<uint16_t, N>(), v);
  Vec128<uint16_t, N> b;
  uint16x8_t c = vcombine_u16(a.raw, b.raw);
  return Vec128<uint8_t, N>(vqmovn_u16(c));
}

template <size_t N>
HWY_INLINE Vec128<int8_t, N> DemoteTo(Simd<int8_t, N> /* tag */,
                                      const Vec128<int32_t> v) {
  Vec128<int16_t, N> a = DemoteTo(Simd<int16_t, N>(), v);
  Vec128<int16_t, N> b;
  uint16x8_t c = vcombine_s16(a.raw, b.raw);
  return Vec128<int8_t, N>(vqmovn_s16(c));
}

HWY_DIAGNOSTICS(pop)

// ------------------------------ Convert integer <=> floating-point

HWY_INLINE Vec128<float> ConvertTo(Full128<float> /* tag */,
                                   const Vec128<int32_t> v) {
  return Vec128<float>(vcvtq_f32_s32(v.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<float, N> ConvertTo(Simd<float, N> /* tag */,
                                      const Vec128<int32_t, N> v) {
  return Vec128<float, N>(vcvt_f32_s32(v.raw));
}

// Truncates (rounds toward zero).
HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                     const Vec128<float> v) {
  return Vec128<int32_t>(vcvtq_s32_f32(v.raw));
}
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<int32_t, N> ConvertTo(Simd<int32_t, N> /* tag */,
                                        const Vec128<float, N> v) {
  return Vec128<int32_t, N>(vcvt_s32_f32(v.raw));
}

#if defined(__aarch64__)

HWY_INLINE Vec128<double> ConvertTo(Full128<double> /* tag */,
                                    const Vec128<int64_t> v) {
  return Vec128<double>(vcvtq_f64_s64(v.raw));
}
HWY_INLINE Vec128<double, 1> ConvertTo(Simd<double, 1> /* tag */,
                                       const Vec128<int64_t, 1> v) {
  return Vec128<double, 1>(vcvt_f64_s64(v.raw));
}

// Truncates (rounds toward zero).
HWY_INLINE Vec128<int64_t> ConvertTo(Full128<int64_t> /* tag */,
                                     const Vec128<double> v) {
  return Vec128<int64_t>(vcvtq_s64_f64(v.raw));
}
HWY_INLINE Vec128<int64_t, 1> ConvertTo(Simd<int64_t, 1> /* tag */,
                                        const Vec128<double, 1> v) {
  return Vec128<int64_t, 1>(vcvt_s64_f64(v.raw));
}

HWY_INLINE Vec128<int32_t> NearestInt(const Vec128<float> v) {
  return Vec128<int32_t>(vcvtnq_s32_f32(v.raw));
}
template <size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<int32_t, N> NearestInt(const Vec128<float, N> v) {
  return Vec128<int32_t, N>(vcvtn_s32_f32(v.raw));
}

#else

template <size_t N>
HWY_INLINE Vec128<int32_t, N> NearestInt(const Vec128<float, N> v) {
  return ConvertTo(Simd<int32_t, N>(), Round(v));
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
template <typename T, size_t N, HWY_IF_LE64(uint8_t, N)>
HWY_INLINE Vec128<T, N / 2> LowerHalf(const Vec128<T, N> v) {
  return Vec128<T, N / 2>(v.raw);
}

HWY_INLINE Vec128<uint8_t, 8> LowerHalf(const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 8>(vget_low_u8(v.raw));
}
HWY_INLINE Vec128<uint16_t, 4> LowerHalf(const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 4>(vget_low_u16(v.raw));
}
HWY_INLINE Vec128<uint32_t, 2> LowerHalf(const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 2>(vget_low_u32(v.raw));
}
HWY_INLINE Vec128<uint64_t, 1> LowerHalf(const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_low_u64(v.raw));
}
HWY_INLINE Vec128<int8_t, 8> LowerHalf(const Vec128<int8_t> v) {
  return Vec128<int8_t, 8>(vget_low_s8(v.raw));
}
HWY_INLINE Vec128<int16_t, 4> LowerHalf(const Vec128<int16_t> v) {
  return Vec128<int16_t, 4>(vget_low_s16(v.raw));
}
HWY_INLINE Vec128<int32_t, 2> LowerHalf(const Vec128<int32_t> v) {
  return Vec128<int32_t, 2>(vget_low_s32(v.raw));
}
HWY_INLINE Vec128<int64_t, 1> LowerHalf(const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_low_s64(v.raw));
}
HWY_INLINE Vec128<float, 2> LowerHalf(const Vec128<float> v) {
  return Vec128<float, 2>(vget_low_f32(v.raw));
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

namespace impl {

// Need to partially specialize because CombineShiftRightBytes<16> and <0> are
// compile errors.
template <int kBytes>
struct ShiftLeftBytesT {
  template <class T, size_t N>
  HWY_INLINE Vec128<T, N> operator()(const Vec128<T, N> v) {
    return CombineShiftRightBytes<16 - kBytes>(v, Zero(Full128<T>()));
  }
};
template <>
struct ShiftLeftBytesT<0> {
  template <class T, size_t N>
  HWY_INLINE Vec128<T, N> operator()(const Vec128<T, N> v) {
    return v;
  }
};

template <int kBytes>
struct ShiftRightBytesT {
  template <class T, size_t N>
  HWY_INLINE Vec128<T, N> operator()(const Vec128<T, N> v) {
    return CombineShiftRightBytes<kBytes>(Zero(Full128<T>()), v);
  }
};
template <>
struct ShiftRightBytesT<0> {
  template <class T, size_t N>
  HWY_INLINE Vec128<T, N> operator()(const Vec128<T, N> v) {
    return v;
  }
};

}  // namespace impl

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
HWY_INLINE Vec128<T, N> ShiftLeftBytes(const Vec128<T, N> v) {
  return impl::ShiftLeftBytesT<kBytes>()(v);
}

template <int kLanes, typename T, size_t N>
HWY_INLINE Vec128<T, N> ShiftLeftLanes(const Vec128<T, N> v) {
  const Simd<uint8_t, N * sizeof(T)> d8;
  const Simd<T, N> d;
  return BitCast(d, ShiftLeftBytes<kLanes * sizeof(T)>(BitCast(d8, v)));
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
HWY_INLINE Vec128<T, N> ShiftRightBytes(const Vec128<T, N> v) {
  return impl::ShiftRightBytesT<kBytes>()(v);
}

template <int kLanes, typename T, size_t N>
HWY_INLINE Vec128<T, N> ShiftRightLanes(const Vec128<T, N> v) {
  const Simd<uint8_t, N * sizeof(T)> d8;
  const Simd<T, N> d;
  return BitCast(d, ShiftRightBytes<kLanes * sizeof(T)>(BitCast(d8, v)));
}

// ------------------------------ Broadcast/splat any lane

#if defined(__aarch64__)
// Unsigned
template <int kLane>
HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<uint16_t>(vdupq_laneq_u16(v.raw, kLane));
}
template <int kLane, size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> Broadcast(const Vec128<uint16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint16_t, N>(vdup_lane_u16(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<uint32_t> Broadcast(const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<uint32_t>(vdupq_laneq_u32(v.raw, kLane));
}
template <int kLane, size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> Broadcast(const Vec128<uint32_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint32_t, N>(vdup_lane_u32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<uint64_t> Broadcast(const Vec128<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<uint64_t>(vdupq_laneq_u64(v.raw, kLane));
}
// Vec128<uint64_t, 1> is defined below.

// Signed
template <int kLane>
HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<int16_t>(vdupq_laneq_s16(v.raw, kLane));
}
template <int kLane, size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> Broadcast(const Vec128<int16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int16_t, N>(vdup_lane_s16(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<int32_t> Broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<int32_t>(vdupq_laneq_s32(v.raw, kLane));
}
template <int kLane, size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> Broadcast(const Vec128<int32_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int32_t, N>(vdup_lane_s32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<int64_t> Broadcast(const Vec128<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<int64_t>(vdupq_laneq_s64(v.raw, kLane));
}
// Vec128<int64_t, 1> is defined below.

// Float
template <int kLane>
HWY_INLINE Vec128<float> Broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<float>(vdupq_laneq_f32(v.raw, kLane));
}
template <int kLane, size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> Broadcast(const Vec128<float, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<float, N>(vdup_lane_f32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<double> Broadcast(const Vec128<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<double>(vdupq_laneq_f64(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<double, 1> Broadcast(const Vec128<double, 1> v) {
  static_assert(0 <= kLane && kLane < 1, "Invalid lane");
  return v;
}

#else
// No vdupq_laneq_* on armv7: use vgetq_lane_* + vdupq_n_*.

// Unsigned
template <int kLane>
HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<uint16_t>(vdupq_n_u16(vgetq_lane_u16(v.raw, kLane)));
}
template <int kLane, size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint16_t, N> Broadcast(const Vec128<uint16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint16_t, N>(vdup_lane_u16(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<uint32_t> Broadcast(const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<uint32_t>(vdupq_n_u32(vgetq_lane_u32(v.raw, kLane)));
}
template <int kLane, size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint32_t, N> Broadcast(const Vec128<uint32_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint32_t, N>(vdup_lane_u32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<uint64_t> Broadcast(const Vec128<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<uint64_t>(vdupq_n_u64(vgetq_lane_u64(v.raw, kLane)));
}
// Vec128<uint64_t, 1> is defined below.

// Signed
template <int kLane>
HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<int16_t>(vdupq_n_s16(vgetq_lane_s16(v.raw, kLane)));
}
template <int kLane, size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int16_t, N> Broadcast(const Vec128<int16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int16_t, N>(vdup_lane_s16(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<int32_t> Broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<int32_t>(vdupq_n_s32(vgetq_lane_s32(v.raw, kLane)));
}
template <int kLane, size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int32_t, N> Broadcast(const Vec128<int32_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int32_t, N>(vdup_lane_s32(v.raw, kLane));
}
template <int kLane>
HWY_INLINE Vec128<int64_t> Broadcast(const Vec128<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<int64_t>(vdupq_n_s64(vgetq_lane_s64(v.raw, kLane)));
}
// Vec128<int64_t, 1> is defined below.

// Float
template <int kLane>
HWY_INLINE Vec128<float> Broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<float>(vdupq_n_f32(vgetq_lane_f32(v.raw, kLane)));
}
template <int kLane, size_t N, HWY_IF_LE64(float, N)>
HWY_INLINE Vec128<float, N> Broadcast(const Vec128<float, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<float, N>(vdup_lane_f32(v.raw, kLane));
}

#endif

template <int kLane>
HWY_INLINE Vec128<uint64_t, 1> Broadcast(const Vec128<uint64_t, 1> v) {
  static_assert(0 <= kLane && kLane < 1, "Invalid lane");
  return v;
}
template <int kLane>
HWY_INLINE Vec128<int64_t, 1> Broadcast(const Vec128<int64_t, 1> v) {
  static_assert(0 <= kLane && kLane < 1, "Invalid lane");
  return v;
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes, i.e.
// lane indices in [0, 16).
template <typename T>
HWY_API Vec128<T> TableLookupBytes(const Vec128<T> bytes,
                                   const Vec128<T> from) {
  const Full128<T> d;
  const Repartition<uint8_t, decltype(d)> d8;
#if defined(__aarch64__)
  return BitCast(d, Vec128<uint8_t>(vqtbl1q_u8(BitCast(d8, bytes).raw,
                                               BitCast(d8, from).raw)));
#else
  uint8x16_t table0 = BitCast(d8, bytes).raw;
  uint8x8x2_t table;
  table.val[0] = vget_low_u8(table0);
  table.val[1] = vget_high_u8(table0);
  uint8x16_t idx = BitCast(d8, from).raw;
  uint8x8_t low = vtbl2_u8(table, vget_low_u8(idx));
  uint8x8_t hi = vtbl2_u8(table, vget_high_u8(idx));
  return BitCast(d, Vec128<uint8_t>(vcombine_u8(low, hi)));
#endif
}

template <typename T, size_t N, typename TI, HWY_IF_LE64(T, N)>
HWY_INLINE Vec128<T, N> TableLookupBytes(
    const Vec128<T, N> bytes,
    const Vec128<TI, N * sizeof(T) / sizeof(TI)> from) {
  const Simd<T, N> d;
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, decltype(Zero(d8))(vtbl1_u8(BitCast(d8, bytes).raw,
                                                BitCast(d8, from).raw)));
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec128<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// Shuffle0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// CombineShiftRightBytes but the shuffle_abcd notation is more convenient.

// Swap 32-bit halves in 64-bits
HWY_INLINE Vec128<uint32_t, 2> Shuffle2301(const Vec128<uint32_t, 2> v) {
  return Vec128<uint32_t, 2>(vrev64_u32(v.raw));
}
HWY_INLINE Vec128<int32_t, 2> Shuffle2301(const Vec128<int32_t, 2> v) {
  return Vec128<int32_t, 2>(vrev64_s32(v.raw));
}
HWY_INLINE Vec128<float, 2> Shuffle2301(const Vec128<float, 2> v) {
  return Vec128<float, 2>(vrev64_f32(v.raw));
}
HWY_INLINE Vec128<uint32_t> Shuffle2301(const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(vrev64q_u32(v.raw));
}
HWY_INLINE Vec128<int32_t> Shuffle2301(const Vec128<int32_t> v) {
  return Vec128<int32_t>(vrev64q_s32(v.raw));
}
HWY_INLINE Vec128<float> Shuffle2301(const Vec128<float> v) {
  return Vec128<float>(vrev64q_f32(v.raw));
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
  const Full128<uint8_t> d8;
  const Full128<T> d;
  return TableLookupBytes(v, BitCast(d, Load(d8, bytes)));
}

// ------------------------------ TableLookupLanes

// Returned by SetTableIndices for use by TableLookupLanes.
template <typename T>
struct Indices128 {
  uint8x16_t raw;
};

template <typename T>
HWY_INLINE Indices128<T> SetTableIndices(const Full128<T>, const int32_t* idx) {
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
  const size_t N = 16 / sizeof(T);
  for (size_t i = 0; i < N; ++i) {
    HWY_DASSERT(0 <= idx[i] && idx[i] < static_cast<int32_t>(N));
  }
#endif

  const Full128<uint8_t> d8;
  alignas(16) uint8_t control[16];
  for (size_t idx_byte = 0; idx_byte < 16; ++idx_byte) {
    const size_t idx_lane = idx_byte / sizeof(T);
    const size_t mod = idx_byte % sizeof(T);
    control[idx_byte] = idx[idx_lane] * sizeof(T) + mod;
  }
  return Indices128<T>{Load(d8, control).raw};
}

HWY_INLINE Vec128<uint32_t> TableLookupLanes(const Vec128<uint32_t> v,
                                             const Indices128<uint32_t> idx) {
  return TableLookupBytes(v, Vec128<uint32_t>(idx.raw));
}
HWY_INLINE Vec128<int32_t> TableLookupLanes(const Vec128<int32_t> v,
                                            const Indices128<int32_t> idx) {
  return TableLookupBytes(v, Vec128<int32_t>(idx.raw));
}
HWY_INLINE Vec128<float> TableLookupLanes(const Vec128<float> v,
                                          const Indices128<float> idx) {
  const Full128<int32_t> di;
  const Full128<float> df;
  return BitCast(df,
                 TableLookupBytes(BitCast(di, v), Vec128<int32_t>(idx.raw)));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use ZipLower/Upper instead (also works with scalar).
HWY_NEON_DEF_FUNCTION_INT_8_16_32(InterleaveLower, vzip1, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(InterleaveLower, vzip1, _, 2)

HWY_NEON_DEF_FUNCTION_INT_8_16_32(InterleaveUpper, vzip2, _, 2)
HWY_NEON_DEF_FUNCTION_UINT_8_16_32(InterleaveUpper, vzip2, _, 2)

#if defined(__aarch64__)
// For 64 bit types, we only have the "q" version of the function defined as
// interleaving 64-wide registers with 64-wide types in them makes no sense.
HWY_INLINE Vec128<uint64_t> InterleaveLower(const Vec128<uint64_t> a,
                                            const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(vzip1q_u64(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> InterleaveLower(const Vec128<int64_t> a,
                                           const Vec128<int64_t> b) {
  return Vec128<int64_t>(vzip1q_s64(a.raw, b.raw));
}

HWY_INLINE Vec128<uint64_t> InterleaveUpper(const Vec128<uint64_t> a,
                                            const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(vzip2q_u64(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> InterleaveUpper(const Vec128<int64_t> a,
                                           const Vec128<int64_t> b) {
  return Vec128<int64_t>(vzip2q_s64(a.raw, b.raw));
}
#else
// ARMv7 emulation.
HWY_INLINE Vec128<uint64_t> InterleaveLower(const Vec128<uint64_t> a,
                                            const Vec128<uint64_t> b) {
  auto flip = CombineShiftRightBytes<8>(a, a);
  return CombineShiftRightBytes<8>(b, flip);
}
HWY_INLINE Vec128<int64_t> InterleaveLower(const Vec128<int64_t> a,
                                           const Vec128<int64_t> b) {
  auto flip = CombineShiftRightBytes<8>(a, a);
  return CombineShiftRightBytes<8>(b, flip);
}

HWY_INLINE Vec128<uint64_t> InterleaveUpper(const Vec128<uint64_t> a,
                                            const Vec128<uint64_t> b) {
  auto flip = CombineShiftRightBytes<8>(b, b);
  return CombineShiftRightBytes<8>(flip, a);
}
HWY_INLINE Vec128<int64_t> InterleaveUpper(const Vec128<int64_t> a,
                                           const Vec128<int64_t> b) {
  auto flip = CombineShiftRightBytes<8>(b, b);
  return CombineShiftRightBytes<8>(flip, a);
}
#endif

// Floats
HWY_INLINE Vec128<float> InterleaveLower(const Vec128<float> a,
                                         const Vec128<float> b) {
  return Vec128<float>(vzip1q_f32(a.raw, b.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double> InterleaveLower(const Vec128<double> a,
                                          const Vec128<double> b) {
  return Vec128<double>(vzip1q_f64(a.raw, b.raw));
}
#endif

HWY_INLINE Vec128<float> InterleaveUpper(const Vec128<float> a,
                                         const Vec128<float> b) {
  return Vec128<float>(vzip2q_f32(a.raw, b.raw));
}
#if defined(__aarch64__)
HWY_INLINE Vec128<double> InterleaveUpper(const Vec128<double> a,
                                          const Vec128<double> b) {
  return Vec128<double>(vzip2q_s64(a.raw, b.raw));
}
#endif

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

// Full vectors
HWY_INLINE Vec128<uint16_t> ZipLower(const Vec128<uint8_t> a,
                                     const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(vzip1q_u8(a.raw, b.raw));
}
HWY_INLINE Vec128<uint32_t> ZipLower(const Vec128<uint16_t> a,
                                     const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(vzip1q_u16(a.raw, b.raw));
}
HWY_INLINE Vec128<uint64_t> ZipLower(const Vec128<uint32_t> a,
                                     const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(vzip1q_u32(a.raw, b.raw));
}

HWY_INLINE Vec128<int16_t> ZipLower(const Vec128<int8_t> a,
                                    const Vec128<int8_t> b) {
  return Vec128<int16_t>(vzip1q_s8(a.raw, b.raw));
}
HWY_INLINE Vec128<int32_t> ZipLower(const Vec128<int16_t> a,
                                    const Vec128<int16_t> b) {
  return Vec128<int32_t>(vzip1q_s16(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> ZipLower(const Vec128<int32_t> a,
                                    const Vec128<int32_t> b) {
  return Vec128<int64_t>(vzip1q_s32(a.raw, b.raw));
}

HWY_INLINE Vec128<uint16_t> ZipUpper(const Vec128<uint8_t> a,
                                     const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(vzip2q_u8(a.raw, b.raw));
}
HWY_INLINE Vec128<uint32_t> ZipUpper(const Vec128<uint16_t> a,
                                     const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(vzip2q_u16(a.raw, b.raw));
}
HWY_INLINE Vec128<uint64_t> ZipUpper(const Vec128<uint32_t> a,
                                     const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(vzip2q_u32(a.raw, b.raw));
}

HWY_INLINE Vec128<int16_t> ZipUpper(const Vec128<int8_t> a,
                                    const Vec128<int8_t> b) {
  return Vec128<int16_t>(vzip2q_s8(a.raw, b.raw));
}
HWY_INLINE Vec128<int32_t> ZipUpper(const Vec128<int16_t> a,
                                    const Vec128<int16_t> b) {
  return Vec128<int32_t>(vzip2q_s16(a.raw, b.raw));
}
HWY_INLINE Vec128<int64_t> ZipUpper(const Vec128<int32_t> a,
                                    const Vec128<int32_t> b) {
  return Vec128<int64_t>(vzip2q_s32(a.raw, b.raw));
}

// Half vectors or less
template <size_t N, HWY_IF_LE64(uint8_t, N)>
HWY_INLINE Vec128<uint16_t, (N + 1) / 2> ZipLower(const Vec128<uint8_t, N> a,
                                                  const Vec128<uint8_t, N> b) {
  return Vec128<uint16_t, (N + 1) / 2>(vzip1_u8(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint32_t, (N + 1) / 2> ZipLower(const Vec128<uint16_t, N> a,
                                                  const Vec128<uint16_t, N> b) {
  return Vec128<uint32_t, (N + 1) / 2>(vzip1_u16(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint64_t, (N + 1) / 2> ZipLower(const Vec128<uint32_t, N> a,
                                                  const Vec128<uint32_t, N> b) {
  return Vec128<uint64_t, (N + 1) / 2>(vzip1_u32(a.raw, b.raw));
}

template <size_t N, HWY_IF_LE64(int8_t, N)>
HWY_INLINE Vec128<int16_t, (N + 1) / 2> ZipLower(const Vec128<int8_t, N> a,
                                                 const Vec128<int8_t, N> b) {
  return Vec128<int16_t, (N + 1) / 2>(vzip1_s8(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int32_t, (N + 1) / 2> ZipLower(const Vec128<int16_t, N> a,
                                                 const Vec128<int16_t, N> b) {
  return Vec128<int32_t, (N + 1) / 2>(vzip1_s16(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int64_t, (N + 1) / 2> ZipLower(const Vec128<int32_t, N> a,
                                                 const Vec128<int32_t, N> b) {
  return Vec128<int64_t, (N + 1) / 2>(vzip1_s32(a.raw, b.raw));
}

template <size_t N, HWY_IF_LE64(uint8_t, N)>
HWY_INLINE Vec128<uint16_t, N / 2> ZipUpper(const Vec128<uint8_t, N> a,
                                            const Vec128<uint8_t, N> b) {
  return Vec128<uint16_t, N / 2>(vzip2_u8(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(uint16_t, N)>
HWY_INLINE Vec128<uint32_t, N / 2> ZipUpper(const Vec128<uint16_t, N> a,
                                            const Vec128<uint16_t, N> b) {
  return Vec128<uint32_t, N / 2>(vzip2_u16(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(uint32_t, N)>
HWY_INLINE Vec128<uint64_t, N / 2> ZipUpper(const Vec128<uint32_t, N> a,
                                            const Vec128<uint32_t, N> b) {
  return Vec128<uint64_t, N / 2>(vzip2_u32(a.raw, b.raw));
}

template <size_t N, HWY_IF_LE64(int8_t, N)>
HWY_INLINE Vec128<int16_t, N / 2> ZipUpper(const Vec128<int8_t, N> a,
                                           const Vec128<int8_t, N> b) {
  return Vec128<int16_t, N / 2>(vzip2_s8(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(int16_t, N)>
HWY_INLINE Vec128<int32_t, N / 2> ZipUpper(const Vec128<int16_t, N> a,
                                           const Vec128<int16_t, N> b) {
  return Vec128<int32_t, N / 2>(vzip2_s16(a.raw, b.raw));
}
template <size_t N, HWY_IF_LE64(int32_t, N)>
HWY_INLINE Vec128<int64_t, N / 2> ZipUpper(const Vec128<int32_t, N> a,
                                           const Vec128<int32_t, N> b) {
  return Vec128<int64_t, N / 2>(vzip2_s32(a.raw, b.raw));
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatLowerLower(const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return BitCast(Full128<T>(),
                 InterleaveLower(BitCast(d64, lo), BitCast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatUpperUpper(const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return BitCast(Full128<T>(),
                 InterleaveUpper(BitCast(d64, lo), BitCast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatLowerUpper(const Vec128<T> hi, const Vec128<T> lo) {
  return CombineShiftRightBytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
HWY_INLINE Vec128<T> ConcatUpperLower(const Vec128<T> hi, const Vec128<T> lo) {
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

// Returns a vector with lane i=[0, N) set to "first" + i.
template <typename T, size_t N, typename T2>
Vec128<T, N> Iota(const Simd<T, N> d, const T2 first) {
  HWY_ALIGN T lanes[16 / sizeof(T)];
  for (size_t i = 0; i < 16 / sizeof(T); ++i) {
    lanes[i] = static_cast<T>(first + static_cast<T2>(i));
  }
  return Load(d, lanes);
}

// ------------------------------ Gather (requires GetLane)

template <typename T, size_t N, typename Offset>
HWY_API Vec128<T, N> GatherOffset(const Simd<T, N> d,
                                  const T* HWY_RESTRICT base,
                                  const Vec128<Offset, N> offset) {
  static_assert(N == 1, "NEON does not support full gather");
  static_assert(sizeof(T) == sizeof(Offset), "T must match Offset");
  const uintptr_t address = reinterpret_cast<uintptr_t>(base) + GetLane(offset);
  T val;
  CopyBytes<sizeof(T)>(reinterpret_cast<const T*>(address), &val);
  return Set(d, val);
}

template <typename T, size_t N, typename Index>
HWY_API Vec128<T, N> GatherIndex(const Simd<T, N> d, const T* HWY_RESTRICT base,
                                 const Vec128<Index, N> index) {
  static_assert(N == 1, "NEON does not support full gather");
  static_assert(sizeof(T) == sizeof(Index), "T must match Index");
  return Set(d, base[GetLane(index)]);
}

// ------------------------------ ARMv7 int64 equality (requires Shuffle2301)

#if !defined(__aarch64__)

template <size_t N>
HWY_INLINE Mask128<int64_t, N> operator==(const Vec128<int64_t, N> a,
                                          const Vec128<int64_t, N> b) {
  const Simd<int32_t, N * 2> d32;
  const Simd<int64_t, N> d64;
  const auto cmp32 = VecFromMask(BitCast(d32, a) == BitCast(d32, b));
  const auto cmp64 = cmp32 & Shuffle2301(cmp32);
  return MaskFromVec(BitCast(d64, cmp64));
}

template <size_t N>
HWY_INLINE Mask128<uint64_t, N> operator==(const Vec128<uint64_t, N> a,
                                           const Vec128<uint64_t, N> b) {
  const Simd<uint32_t, N * 2> d32;
  const Simd<uint64_t, N> d64;
  const auto cmp32 = VecFromMask(BitCast(d32, a) == BitCast(d32, b));
  const auto cmp64 = cmp32 & Shuffle2301(cmp32);
  return MaskFromVec(BitCast(d64, cmp64));
}

#endif

// ------------------------------ Compress

namespace detail {

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Idx32x4FromBits(const uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 16);

  // There are only 4 lanes, so we can afford to load the index vector directly.
  alignas(16) constexpr uint8_t packed_array[16 * 16] = {
      0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  //
      4,  5,  6,  7,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  0,  1,  2,  3,  //
      8,  9,  10, 11, 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  8,  9,  10, 11, 0,  1,  2,  3,  0,  1,  2,  3,  //
      4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,  //
      12, 13, 14, 15, 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  12, 13, 14, 15, 0,  1,  2,  3,  0,  1,  2,  3,  //
      4,  5,  6,  7,  12, 13, 14, 15, 0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  4,  5,  6,  7,  12, 13, 14, 15, 0,  1,  2,  3,  //
      8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  0,  1,  2,  3,  //
      0,  1,  2,  3,  8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  //
      4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  //
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15};

  const Simd<T, N> d;
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, packed_array + 16 * mask_bits));
}

#if HWY_CAP_INTEGER64 || HWY_CAP_FLOAT64

template <typename T, size_t N>
HWY_INLINE Vec128<T, N> Idx64x2FromBits(const uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 4);

  // There are only 2 lanes, so we can afford to load the index vector directly.
  alignas(16) constexpr uint8_t packed_array[4 * 16] = {
      0, 1, 2,  3,  4,  5,  6,  7,  0, 1, 2,  3,  4,  5,  6,  7,  //
      0, 1, 2,  3,  4,  5,  6,  7,  0, 1, 2,  3,  4,  5,  6,  7,  //
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2,  3,  4,  5,  6,  7,  //
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15};

  const Simd<T, N> d;
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, packed_array + 16 * mask_bits));
}

#endif

// Helper function called by both Compress and CompressStore - avoids a
// redundant BitsFromMask in the latter.

template <size_t N>
HWY_API Vec128<uint32_t, N> Compress(Vec128<uint32_t, N> v,
                                     const uint64_t mask_bits) {
  const auto idx = detail::Idx32x4FromBits<uint32_t, N>(mask_bits);
  return TableLookupBytes(v, idx);
}
template <size_t N>
HWY_API Vec128<int32_t, N> Compress(Vec128<int32_t, N> v,
                                    const uint64_t mask_bits) {
  const auto idx = detail::Idx32x4FromBits<int32_t, N>(mask_bits);
  return TableLookupBytes(v, idx);
}

#if HWY_CAP_INTEGER64

template <size_t N>
HWY_API Vec128<uint64_t, N> Compress(Vec128<uint64_t, N> v,
                                     const uint64_t mask_bits) {
  const auto idx = detail::Idx64x2FromBits<uint64_t, N>(mask_bits);
  return TableLookupBytes(v, idx);
}
template <size_t N>
HWY_API Vec128<int64_t, N> Compress(Vec128<int64_t, N> v,
                                    const uint64_t mask_bits) {
  const auto idx = detail::Idx64x2FromBits<int64_t, N>(mask_bits);
  return TableLookupBytes(v, idx);
}

#endif

template <size_t N>
HWY_API Vec128<float, N> Compress(Vec128<float, N> v,
                                  const uint64_t mask_bits) {
  const auto idx = detail::Idx32x4FromBits<int32_t, N>(mask_bits);
  const Simd<float, N> df;
  const Simd<int32_t, N> di;
  return BitCast(df, TableLookupBytes(BitCast(di, v), idx));
}

#if HWY_CAP_FLOAT64

template <size_t N>
HWY_API Vec128<double, N> Compress(Vec128<double, N> v,
                                   const uint64_t mask_bits) {
  const auto idx = detail::Idx64x2FromBits<int64_t, N>(mask_bits);
  const Simd<double, N> df;
  const Simd<int64_t, N> di;
  return BitCast(df, TableLookupBytes(BitCast(di, v), idx));
}

#endif

}  // namespace detail

template <typename T, size_t N>
HWY_API Vec128<T, N> Compress(Vec128<T, N> v, const Mask128<T, N> mask) {
  return detail::Compress(v, BitsFromMask(mask));
}

// ------------------------------ CompressStore

template <typename T, size_t N>
HWY_API size_t CompressStore(Vec128<T, N> v, const Mask128<T, N> mask,
                             Simd<T, N> d, T* HWY_RESTRICT aligned) {
  const uint64_t mask_bits = BitsFromMask(mask);
  Store(detail::Compress(v, mask_bits), d, aligned);
  return PopCount(mask_bits);
}

// ------------------------------ Reductions

// Returns 64-bit sums of 8-byte groups.
HWY_INLINE Vec128<uint64_t> SumsOfU8x8(const Vec128<uint8_t> v) {
  uint16x8_t a = vpaddlq_u8(v.raw);
  uint32x4_t b = vpaddlq_u16(a);
  return Vec128<uint64_t>(vpaddlq_u32(b));
}

HWY_INLINE Vec128<uint64_t, 1> SumsOfU8x8(const Vec128<uint8_t, 8> v) {
  uint16x4_t a = vpaddl_u8(v.raw);
  uint32x2_t b = vpaddl_u16(a);
  return Vec128<uint64_t, 1>(vpaddl_u32(b));
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

namespace detail {

// For u32/i32/f32.
template <typename T, size_t N>
HWY_API Vec128<T, N> MinOfLanes(hwy::SizeTag<4> /* tag */,
                                const Vec128<T, N> v3210) {
  const Vec128<T> v1032 = Shuffle1032(v3210);
  const Vec128<T> v31_20_31_20 = Min(v3210, v1032);
  const Vec128<T> v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return Min(v20_31_20_31, v31_20_31_20);
}
template <typename T, size_t N>
HWY_API Vec128<T, N> MaxOfLanes(hwy::SizeTag<4> /* tag */,
                                const Vec128<T, N> v3210) {
  const Vec128<T> v1032 = Shuffle1032(v3210);
  const Vec128<T> v31_20_31_20 = Max(v3210, v1032);
  const Vec128<T> v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return Max(v20_31_20_31, v31_20_31_20);
}

// For u64/i64[/f64].
template <typename T, size_t N>
HWY_API Vec128<T, N> MinOfLanes(hwy::SizeTag<8> /* tag */,
                                const Vec128<T, N> v10) {
  const Vec128<T> v01 = Shuffle01(v10);
  return Min(v10, v01);
}
template <typename T, size_t N>
HWY_API Vec128<T, N> MaxOfLanes(hwy::SizeTag<8> /* tag */,
                                const Vec128<T, N> v10) {
  const Vec128<T> v01 = Shuffle01(v10);
  return Max(v10, v01);
}

}  // namespace detail

template <typename T, size_t N>
HWY_API Vec128<T, N> MinOfLanes(const Vec128<T, N> v) {
  return detail::MinOfLanes(hwy::SizeTag<sizeof(T)>(), v);
}
template <typename T, size_t N>
HWY_API Vec128<T, N> MaxOfLanes(const Vec128<T, N> v) {
  return detail::MaxOfLanes(hwy::SizeTag<sizeof(T)>(), v);
}

// ------------------------------ Mask

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
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<1> /*tag*/,
                                 const Mask128<T> mask) {
  alignas(16) constexpr uint8_t kSliceLanes[16] = {
      1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
  };
  const Full128<uint8_t> du;
  const Vec128<uint8_t> values =
      BitCast(du, VecFromMask(mask)) & Load(du, kSliceLanes);

#if defined(__aarch64__)
  // Can't vaddv - we need two separate bytes (16 bits).
  const uint8x8_t x2 = vget_low_u8(vpaddq_u8(values.raw, values.raw));
  const uint8x8_t x4 = vpadd_u8(x2, x2);
  const uint8x8_t x8 = vpadd_u8(x4, x4);
  return vreinterpret_u16_u8(x8)[0];
#else
  // Don't have vpaddq, so keep doubling lane size.
  const uint16x8_t x2 = vpaddlq_u8(values.raw);
  const uint32x4_t x4 = vpaddlq_u16(x2);
  const uint64x2_t x8 = vpaddlq_u32(x4);
  return (uint64_t(x8[1]) << 8) | x8[0];
#endif
}

template <typename T, size_t N, HWY_IF_LE64(T, N)>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<1> /*tag*/,
                                 const Mask128<T, N> mask) {
  // Upper lanes of partial loads are undefined. OnlyActive will fix this if
  // we load all kSliceLanes so the upper lanes do not pollute the valid bits.
  alignas(8) constexpr uint8_t kSliceLanes[8] = {1,    2,    4,    8,
                                                 0x10, 0x20, 0x40, 0x80};
  const Simd<uint8_t, N> du;
  const Vec128<uint8_t, N> slice(Load(Simd<uint8_t, 8>(), kSliceLanes).raw);
  const Vec128<uint8_t, N> values = BitCast(du, VecFromMask(mask)) & slice;

#if defined(__aarch64__)
  return vaddv_u8(values.raw);
#else
  const uint16x4_t x2 = vpaddl_u8(values.raw);
  const uint32x2_t x4 = vpaddl_u16(x2);
  const uint64x1_t x8 = vpaddl_u32(x4);
  return vget_lane_u64(x8, 0);
#endif
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<2> /*tag*/,
                                 const Mask128<T> mask) {
  alignas(16) constexpr uint16_t kSliceLanes[8] = {1,    2,    4,    8,
                                                   0x10, 0x20, 0x40, 0x80};
  const Full128<uint16_t> du;
  const Vec128<uint16_t> values =
      BitCast(du, VecFromMask(mask)) & Load(du, kSliceLanes);
#if defined(__aarch64__)
  return vaddvq_u16(values.raw);
#else
  const uint32x4_t x2 = vpaddlq_u16(values.raw);
  const uint64x2_t x4 = vpaddlq_u32(x2);
  return vgetq_lane_u64(x4, 0) + vgetq_lane_u64(x4, 1);
#endif
}

template <typename T, size_t N, HWY_IF_LE64(T, N)>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<2> /*tag*/,
                                 const Mask128<T, N> mask) {
  // Upper lanes of partial loads are undefined. OnlyActive will fix this if
  // we load all kSliceLanes so the upper lanes do not pollute the valid bits.
  alignas(8) constexpr uint16_t kSliceLanes[4] = {1, 2, 4, 8};
  const Simd<uint16_t, N> du;
  const Vec128<uint16_t, N> slice(Load(Simd<uint16_t, 4>(), kSliceLanes).raw);
  const Vec128<uint16_t, N> values = BitCast(du, VecFromMask(mask)) & slice;
#if defined(__aarch64__)
  return vaddv_u16(values.raw);
#else
  const uint32x2_t x2 = vpaddl_u16(values.raw);
  const uint64x1_t x4 = vpaddl_u32(x2);
  return vget_lane_u64(x4, 0);
#endif
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<4> /*tag*/,
                                 const Mask128<T> mask) {
  alignas(16) constexpr uint32_t kSliceLanes[4] = {1, 2, 4, 8};
  const Full128<uint32_t> du;
  const Vec128<uint32_t> values =
      BitCast(du, VecFromMask(mask)) & Load(du, kSliceLanes);
#if defined(__aarch64__)
  return vaddvq_u32(values.raw);
#else
  const uint64x2_t x2 = vpaddlq_u32(values.raw);
  return vgetq_lane_u64(x2, 0) + vgetq_lane_u64(x2, 1);
#endif
}

template <typename T, size_t N, HWY_IF_LE64(T, N)>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<4> /*tag*/,
                                 const Mask128<T, N> mask) {
  // Upper lanes of partial loads are undefined. OnlyActive will fix this if
  // we load all kSliceLanes so the upper lanes do not pollute the valid bits.
  alignas(8) constexpr uint32_t kSliceLanes[2] = {1, 2};
  const Simd<uint32_t, N> du;
  const Vec128<uint32_t, N> slice(Load(Simd<uint32_t, 2>(), kSliceLanes).raw);
  const Vec128<uint32_t, N> values = BitCast(du, VecFromMask(mask)) & slice;
#if defined(__aarch64__)
  return vaddv_u32(values.raw);
#else
  const uint64x1_t x2 = vpaddl_u32(values.raw);
  return vget_lane_u64(x2, 0);
#endif
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<8> /*tag*/, const Mask128<T> v) {
  alignas(16) constexpr uint64_t kSliceLanes[2] = {1, 2};
  const Full128<uint64_t> du;
  const Vec128<uint64_t> values =
      BitCast(du, VecFromMask(v)) & Load(du, kSliceLanes);
#if defined(__aarch64__)
  return vaddvq_u64(values.raw);
#else
  return vgetq_lane_u64(values.raw, 0) + vgetq_lane_u64(values.raw, 1);
#endif
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<8> /*tag*/,
                                 const Mask128<T, 1> v) {
  const Simd<uint64_t, 1> du;
  const Vec128<uint64_t, 1> values = BitCast(du, VecFromMask(v)) & Set(du, 1);
  return vget_lane_u64(values.raw, 0);
}

// Returns the lowest N for the BitsFromMask result.
template <typename T, size_t N>
constexpr uint64_t OnlyActive(uint64_t bits) {
  return ((N * sizeof(T)) >= 8) ? bits : (bits & ((1ull << N) - 1));
}

// Returns number of lanes whose mask is set.
//
// Masks are either FF..FF or 0. Unfortunately there is no reduce-sub op
// ("vsubv"). ANDing with 1 would work but requires a constant. Negating also
// changes each lane to 1 (if mask set) or 0.

template <typename T>
HWY_INLINE size_t CountTrue(hwy::SizeTag<1> /*tag*/, const Mask128<T> mask) {
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
HWY_INLINE size_t CountTrue(hwy::SizeTag<2> /*tag*/, const Mask128<T> mask) {
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
HWY_INLINE size_t CountTrue(hwy::SizeTag<4> /*tag*/, const Mask128<T> mask) {
  const Full128<int32_t> di;
  const int32x4_t ones = vnegq_s32(BitCast(di, VecFromMask(mask)).raw);

#if defined(__aarch64__)
  return vaddvq_s32(ones);
#else
  const int64x2_t x2 = vpaddlq_s32(ones);
  return x2[0] + x2[1];
#endif
}

template <typename T>
HWY_INLINE size_t CountTrue(hwy::SizeTag<8> /*tag*/, const Mask128<T> mask) {
#if defined(__aarch64__)
  const Full128<int64_t> di;
  const int64x2_t ones = vnegq_s64(BitCast(di, VecFromMask(mask)).raw);
  return vaddvq_s64(ones);
#else
  const Full128<int64_t> di;
  const int64x2_t ones = vshrq_n_u64(BitCast(di, VecFromMask(mask)).raw, 63);
  return ones[0] + ones[1];
#endif
}

}  // namespace impl

template <typename T, size_t N>
HWY_INLINE uint64_t BitsFromMask(const Mask128<T, N> mask) {
  return impl::OnlyActive<T, N>(
      impl::BitsFromMask(hwy::SizeTag<sizeof(T)>(), mask));
}

template <typename T>
HWY_INLINE size_t CountTrue(const Mask128<T> mask) {
  return impl::CountTrue(hwy::SizeTag<sizeof(T)>(), mask);
}

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

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
