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

#ifndef HIGHWAY_ARM64_NEON_H_
#define HIGHWAY_ARM64_NEON_H_

// 128-bit ARM64 NEON vectors and operations.

#include <arm_neon.h>

#include "third_party/highway/highway/shared.h"

// Macros used to define single and double function calls for multiple types
// for full and half vectors. These macros are undefined at the end of the file.

// SIMD_NEON_BUILD_TPL_* is the template<...> prefix to the function.
#define SIMD_NEON_BUILD_TPL_1
#define SIMD_NEON_BUILD_TPL_2
#define SIMD_NEON_BUILD_TPL_3

// SIMD_NEON_BUILD_RET_* is return type.
#define SIMD_NEON_BUILD_RET_1(type, size) Vec128<type, size>
#define SIMD_NEON_BUILD_RET_2(type, size) Vec128<type, size>
#define SIMD_NEON_BUILD_RET_3(type, size) Vec128<type, size>

// SIMD_NEON_BUILD_PARAM_* is the list of parameters the function receives.
#define SIMD_NEON_BUILD_PARAM_1(type, size) const Vec128<type, size> a
#define SIMD_NEON_BUILD_PARAM_2(type, size) \
  const Vec128<type, size> a, const Vec128<type, size> b
#define SIMD_NEON_BUILD_PARAM_3(type, size)               \
  const Vec128<type, size> a, const Vec128<type, size> b, \
      const Vec128<type, size> c

// SIMD_NEON_BUILD_ARG_* is the list of arguments passed to the underlying
// function.
#define SIMD_NEON_BUILD_ARG_1 a.raw
#define SIMD_NEON_BUILD_ARG_2 a.raw, b.raw
#define SIMD_NEON_BUILD_ARG_3 a.raw, b.raw, c.raw

// We use SIMD_NEON_EVAL(func, ...) to delay the evaluation of func until after
// the __VA_ARGS__ have been expanded. This allows "func" to be a macro on
// itself like with some of the library "functions" such as vshlq_u8. For
// example, SIMD_NEON_EVAL(vshlq_u8, MY_PARAMS) where MY_PARAMS is defined as
// "a, b" (without the quotes) will end up expanding "vshlq_u8(a, b)" if needed.
// Directly writing vshlq_u8(MY_PARAMS) would fail since vshlq_u8() macro
// expects two arguments.
#define SIMD_NEON_EVAL(func, ...) func(__VA_ARGS__)

// Main macro definition that defines a single function for the given type and
// size of vector, using the underlying (prefix##infix##suffix) function and
// the template, return type, parameters and arguments defined by the "args"
// parameters passed here (see SIMD_NEON_BUILD_* macros defined before).
#define SIMD_NEON_DEF_FUNCTION(type, size, name, prefix, infix, suffix, args) \
  SIMD_CONCAT(SIMD_NEON_BUILD_TPL_, args)                                     \
  SIMD_INLINE SIMD_CONCAT(SIMD_NEON_BUILD_RET_, args)(type, size)             \
      name(SIMD_CONCAT(SIMD_NEON_BUILD_PARAM_, args)(type, size)) {           \
    return SIMD_CONCAT(SIMD_NEON_BUILD_RET_, args)(type, size)(               \
        SIMD_NEON_EVAL(prefix##infix##suffix, SIMD_NEON_BUILD_ARG_##args));   \
  }

// The SIMD_NEON_DEF_FUNCTION_* macros define all the variants of a function
// called "name" using the set of neon functions starting with the given
// "prefix" for all the variants of certain types, as specified next to each
// macro. For example, the prefix "vsub" can be used to define the operator-
// using args=2.

// uint8_t
#define SIMD_NEON_DEF_FUNCTION_UINT_8(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(uint8_t, 16, name, prefix##q, infix, u8, args) \
  SIMD_NEON_DEF_FUNCTION(uint8_t, 8, name, prefix, infix, u8, args)     \
  SIMD_NEON_DEF_FUNCTION(uint8_t, 4, name, prefix, infix, u8, args)     \
  SIMD_NEON_DEF_FUNCTION(uint8_t, 2, name, prefix, infix, u8, args)

// int8_t
#define SIMD_NEON_DEF_FUNCTION_INT_8(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(int8_t, 16, name, prefix##q, infix, s8, args) \
  SIMD_NEON_DEF_FUNCTION(int8_t, 8, name, prefix, infix, s8, args)     \
  SIMD_NEON_DEF_FUNCTION(int8_t, 4, name, prefix, infix, s8, args)     \
  SIMD_NEON_DEF_FUNCTION(int8_t, 2, name, prefix, infix, s8, args)

// uint16_t
#define SIMD_NEON_DEF_FUNCTION_UINT_16(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(uint16_t, 8, name, prefix##q, infix, u16, args) \
  SIMD_NEON_DEF_FUNCTION(uint16_t, 4, name, prefix, infix, u16, args)    \
  SIMD_NEON_DEF_FUNCTION(uint16_t, 2, name, prefix, infix, u16, args)    \
  SIMD_NEON_DEF_FUNCTION(uint16_t, 1, name, prefix, infix, u16, args)

// int16_t
#define SIMD_NEON_DEF_FUNCTION_INT_16(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(int16_t, 8, name, prefix##q, infix, s16, args) \
  SIMD_NEON_DEF_FUNCTION(int16_t, 4, name, prefix, infix, s16, args)    \
  SIMD_NEON_DEF_FUNCTION(int16_t, 2, name, prefix, infix, s16, args)    \
  SIMD_NEON_DEF_FUNCTION(int16_t, 1, name, prefix, infix, s16, args)

// uint32_t
#define SIMD_NEON_DEF_FUNCTION_UINT_32(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(uint32_t, 4, name, prefix##q, infix, u32, args) \
  SIMD_NEON_DEF_FUNCTION(uint32_t, 2, name, prefix, infix, u32, args)    \
  SIMD_NEON_DEF_FUNCTION(uint32_t, 1, name, prefix, infix, u32, args)

// int32_t
#define SIMD_NEON_DEF_FUNCTION_INT_32(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(int32_t, 4, name, prefix##q, infix, s32, args) \
  SIMD_NEON_DEF_FUNCTION(int32_t, 2, name, prefix, infix, s32, args)    \
  SIMD_NEON_DEF_FUNCTION(int32_t, 1, name, prefix, infix, s32, args)

// uint64_t
#define SIMD_NEON_DEF_FUNCTION_UINT_64(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(uint64_t, 2, name, prefix##q, infix, u64, args) \
  SIMD_NEON_DEF_FUNCTION(uint64_t, 1, name, prefix, infix, u64, args)

// int64_t
#define SIMD_NEON_DEF_FUNCTION_INT_64(name, prefix, infix, args)        \
  SIMD_NEON_DEF_FUNCTION(int64_t, 2, name, prefix##q, infix, s64, args) \
  SIMD_NEON_DEF_FUNCTION(int64_t, 1, name, prefix, infix, s64, args)

// float and double
#define SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(name, prefix, infix, args)   \
  SIMD_NEON_DEF_FUNCTION(float, 4, name, prefix##q, infix, f32, args)  \
  SIMD_NEON_DEF_FUNCTION(float, 2, name, prefix, infix, f32, args)     \
  SIMD_NEON_DEF_FUNCTION(float, 1, name, prefix, infix, f32, args)     \
  SIMD_NEON_DEF_FUNCTION(double, 2, name, prefix##q, infix, f64, args) \
  SIMD_NEON_DEF_FUNCTION(double, 1, name, prefix, infix, f64, args)

// Helper macros to define for more than one type.
// uint8_t, uint16_t and uint32_t
#define SIMD_NEON_DEF_FUNCTION_UINT_8_16_32(name, prefix, infix, args) \
  SIMD_NEON_DEF_FUNCTION_UINT_8(name, prefix, infix, args)             \
  SIMD_NEON_DEF_FUNCTION_UINT_16(name, prefix, infix, args)            \
  SIMD_NEON_DEF_FUNCTION_UINT_32(name, prefix, infix, args)

// int8_t, int16_t and int32_t
#define SIMD_NEON_DEF_FUNCTION_INT_8_16_32(name, prefix, infix, args) \
  SIMD_NEON_DEF_FUNCTION_INT_8(name, prefix, infix, args)             \
  SIMD_NEON_DEF_FUNCTION_INT_16(name, prefix, infix, args)            \
  SIMD_NEON_DEF_FUNCTION_INT_32(name, prefix, infix, args)

// uint8_t, uint16_t, uint32_t and uint64_t
#define SIMD_NEON_DEF_FUNCTION_UINTS(name, prefix, infix, args)  \
  SIMD_NEON_DEF_FUNCTION_UINT_8_16_32(name, prefix, infix, args) \
  SIMD_NEON_DEF_FUNCTION_UINT_64(name, prefix, infix, args)

// int8_t, int16_t, int32_t and int64_t
#define SIMD_NEON_DEF_FUNCTION_INTS(name, prefix, infix, args)  \
  SIMD_NEON_DEF_FUNCTION_INT_8_16_32(name, prefix, infix, args) \
  SIMD_NEON_DEF_FUNCTION_INT_64(name, prefix, infix, args)

// All int*_t and uint*_t up to 64
#define SIMD_NEON_DEF_FUNCTION_INTS_UINTS(name, prefix, infix, args) \
  SIMD_NEON_DEF_FUNCTION_INTS(name, prefix, infix, args)             \
  SIMD_NEON_DEF_FUNCTION_UINTS(name, prefix, infix, args)

// All previous types.
#define SIMD_NEON_DEF_FUNCTION_ALL_TYPES(name, prefix, infix, args) \
  SIMD_NEON_DEF_FUNCTION_INTS_UINTS(name, prefix, infix, args)      \
  SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(name, prefix, infix, args)

namespace jxl {

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

template <>
struct Raw128<double, 2> {
  using type = float64x2_t;
};

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

template <>
struct Raw128<double, 1> {
  using type = float64x1_t;
};

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
  SIMD_INLINE Vec128() {}
  Vec128(const Vec128&) = default;
  Vec128& operator=(const Vec128&) = default;
  SIMD_INLINE explicit Vec128(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_INLINE Vec128& operator*=(const Vec128 other) {
    return *this = (*this * other);
  }
  SIMD_INLINE Vec128& operator/=(const Vec128 other) {
    return *this = (*this / other);
  }
  SIMD_INLINE Vec128& operator+=(const Vec128 other) {
    return *this = (*this + other);
  }
  SIMD_INLINE Vec128& operator-=(const Vec128 other) {
    return *this = (*this - other);
  }
  SIMD_INLINE Vec128& operator&=(const Vec128 other) {
    return *this = (*this & other);
  }
  SIMD_INLINE Vec128& operator|=(const Vec128 other) {
    return *this = (*this | other);
  }
  SIMD_INLINE Vec128& operator^=(const Vec128 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// ------------------------------ Cast

// cast_to_u8

// Converts from Vec128<T, N> to Vec128<uint8_t, N * sizeof(T)> using the
// vreinterpret*_u8_*() set of functions.
#define SIMD_NEON_BUILD_TPL_SIMD_CAST_TO_U8
#define SIMD_NEON_BUILD_RET_SIMD_CAST_TO_U8(type, size) \
  Vec128<uint8_t, size * sizeof(type)>
#define SIMD_NEON_BUILD_PARAM_SIMD_CAST_TO_U8(type, size) Vec128<type, size> v
#define SIMD_NEON_BUILD_ARG_SIMD_CAST_TO_U8 v.raw

// Special case of u8 to u8 since vreinterpret*_u8_u8 is obviously not defined.
template <size_t N>
SIMD_INLINE Vec128<uint8_t, N> cast_to_u8(Vec128<uint8_t, N> v) {
  return v;
}

SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(cast_to_u8, vreinterpret, _u8_,
                                  SIMD_CAST_TO_U8)
SIMD_NEON_DEF_FUNCTION_INTS(cast_to_u8, vreinterpret, _u8_, SIMD_CAST_TO_U8)
SIMD_NEON_DEF_FUNCTION_UINT_16(cast_to_u8, vreinterpret, _u8_, SIMD_CAST_TO_U8)
SIMD_NEON_DEF_FUNCTION_UINT_32(cast_to_u8, vreinterpret, _u8_, SIMD_CAST_TO_U8)
SIMD_NEON_DEF_FUNCTION_UINT_64(cast_to_u8, vreinterpret, _u8_, SIMD_CAST_TO_U8)

#undef SIMD_NEON_BUILD_TPL_SIMD_CAST_TO_U8
#undef SIMD_NEON_BUILD_RET_SIMD_CAST_TO_U8
#undef SIMD_NEON_BUILD_PARAM_SIMD_CAST_TO_U8
#undef SIMD_NEON_BUILD_ARG_SIMD_CAST_TO_U8

// cast_u8_to
// TODO(deymo): Add all the missing cast_u8_to() missing variants.

template <size_t N>
SIMD_INLINE Vec128<uint8_t, N> cast_u8_to(Desc<uint8_t, N> /* tag */,
                                          Vec128<uint8_t, N> v) {
  return v;
}
template <size_t N>
SIMD_INLINE Vec128<uint16_t, N> cast_u8_to(Desc<uint16_t, N> /* tag */,
                                           Vec128<uint8_t, N * 2> v) {
  return Vec128<uint16_t, N>(vreinterpretq_u16_u8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint32_t, N> cast_u8_to(Desc<uint32_t, N> /* tag */,
                                           Vec128<uint8_t, N * 4> v) {
  return Vec128<uint32_t, N>(vreinterpretq_u32_u8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint64_t, N> cast_u8_to(Desc<uint64_t, N> /* tag */,
                                           Vec128<uint8_t, N * 8> v) {
  return Vec128<uint64_t, N>(vreinterpretq_u64_u8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int8_t, N> cast_u8_to(Desc<int8_t, N> /* tag */,
                                         Vec128<uint8_t, N> v) {
  return Vec128<int8_t, N>(vreinterpretq_s8_u8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int16_t, N> cast_u8_to(Desc<int16_t, N> /* tag */,
                                          Vec128<uint8_t, N * 2> v) {
  return Vec128<int16_t, N>(vreinterpretq_s16_u8(v.raw));
}
SIMD_INLINE Vec128<int32_t, 4> cast_u8_to(Desc<int32_t, 4> /* tag */,
                                          Vec128<uint8_t, 4 * 4> v) {
  return Vec128<int32_t, 4>(vreinterpretq_s32_u8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> cast_u8_to(Desc<int32_t, N> /* tag */,
                                          Vec128<uint8_t, N * 4> v) {
  return Vec128<int32_t, N>(vreinterpret_s32_u8(v.raw));
}

SIMD_INLINE Vec128<int64_t, 2> cast_u8_to(Desc<int64_t, 2> /* tag */,
                                          Vec128<uint8_t, 2 * 8> v) {
  return Vec128<int64_t, 2>(vreinterpretq_s64_u8(v.raw));
}
SIMD_INLINE Vec128<int64_t, 1> cast_u8_to(Desc<int64_t, 1> /* tag */,
                                          Vec128<uint8_t, 1 * 8> v) {
  return Vec128<int64_t, 1>(vreinterpret_s64_u8(v.raw));
}

SIMD_INLINE Vec128<float, 4> cast_u8_to(Desc<float, 4> /* tag */,
                                        Vec128<uint8_t, 4 * 4> v) {
  return Vec128<float, 4>(vreinterpretq_f32_u8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<float, N> cast_u8_to(Desc<float, N> /* tag */,
                                        Vec128<uint8_t, N * 4> v) {
  return Vec128<float, N>(vreinterpret_f32_u8(v.raw));
}

SIMD_INLINE Vec128<double, 2> cast_u8_to(Desc<double, 2> /* tag */,
                                         Vec128<uint8_t, 2 * 8> v) {
  return Vec128<double, 2>(vreinterpretq_f64_u8(v.raw));
}
SIMD_INLINE Vec128<double, 1> cast_u8_to(Desc<double, 1> /* tag */,
                                         Vec128<uint8_t, 1 * 8> v) {
  return Vec128<double, 1>(vreinterpret_f64_u8(v.raw));
}

// bit_cast
template <typename T, size_t N, typename FromT>
SIMD_INLINE Vec128<T, N> bit_cast(
    Desc<T, N> d, Vec128<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  const auto u8 = cast_to_u8(v);
  return cast_u8_to(d, u8);
}

// ------------------------------ Set

// Returns a vector with all lanes set to "t".
#define SIMD_NEON_BUILD_TPL_SIMD_SET1
#define SIMD_NEON_BUILD_RET_SIMD_SET1(type, size) Vec128<type, size>
#define SIMD_NEON_BUILD_PARAM_SIMD_SET1(type, size) \
  Desc<type, size> /* tag */, const type t
#define SIMD_NEON_BUILD_ARG_SIMD_SET1 t

SIMD_NEON_DEF_FUNCTION_ALL_TYPES(set1, vdup, _n_, SIMD_SET1)

#undef SIMD_NEON_BUILD_TPL_SIMD_SET1
#undef SIMD_NEON_BUILD_RET_SIMD_SET1
#undef SIMD_NEON_BUILD_PARAM_SIMD_SET1
#undef SIMD_NEON_BUILD_ARG_SIMD_SET1

// Returns an all-zero vector.
template <typename T, size_t N>
SIMD_INLINE Vec128<T, N> setzero(Desc<T, N> d) {
  return set1(d, 0);
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2>
SIMD_INLINE Vec128<T, N> iota(Desc<T, N> d, const T2 first) {
  SIMD_ALIGN T lanes[N];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector with uninitialized elements.
template <typename T, size_t N>
SIMD_INLINE Vec128<T, N> undefined(Desc<T, N> d) {
  SIMD_DIAGNOSTICS(push)
  SIMD_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")
  typename Raw128<T, N>::type a;
  return Vec128<T, N>(a);
  SIMD_DIAGNOSTICS(pop)
}

// ================================================== ARITHMETIC

// ------------------------------ Addition
SIMD_NEON_DEF_FUNCTION_ALL_TYPES(operator+, vadd, _, 2)

// ------------------------------ Subtraction
SIMD_NEON_DEF_FUNCTION_ALL_TYPES(operator-, vsub, _, 2)

// ------------------------------ Saturating addition and subtraction
// Only defined for uint8_t, uint16_t and their signed versions, as in other
// architectures.

// Returns a + b clamped to the destination range.
SIMD_NEON_DEF_FUNCTION_INT_8(saturated_add, vqadd, _, 2)
SIMD_NEON_DEF_FUNCTION_INT_16(saturated_add, vqadd, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_8(saturated_add, vqadd, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_16(saturated_add, vqadd, _, 2)

// Returns a - b clamped to the destination range.
SIMD_NEON_DEF_FUNCTION_INT_8(saturated_subtract, vqsub, _, 2)
SIMD_NEON_DEF_FUNCTION_INT_16(saturated_subtract, vqsub, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_8(saturated_subtract, vqsub, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_16(saturated_subtract, vqsub, _, 2)

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
SIMD_NEON_DEF_FUNCTION_UINT_8(average_round, vrhadd, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_16(average_round, vrhadd, _, 2)

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
SIMD_INLINE Vec128<int8_t, N> abs(const Vec128<int8_t, N> v) {
  return Vec128<int8_t, N>(vabsq_s8(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int16_t, N> abs(const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>(vabsq_s16(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> abs(const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>(vabsq_s32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Only defined for ints and uints, except for signed i64 shr.
#define SIMD_NEON_BUILD_TPL_SIMD_SHIFT template <int kBits>
#define SIMD_NEON_BUILD_RET_SIMD_SHIFT(type, size) Vec128<type, size>
#define SIMD_NEON_BUILD_PARAM_SIMD_SHIFT(type, size) const Vec128<type, size> v
#define SIMD_NEON_BUILD_ARG_SIMD_SHIFT v.raw, kBits

SIMD_NEON_DEF_FUNCTION_INTS_UINTS(shift_left, vshl, _n_, SIMD_SHIFT)

SIMD_NEON_DEF_FUNCTION_UINTS(shift_right, vshr, _n_, SIMD_SHIFT)
SIMD_NEON_DEF_FUNCTION_INT_8_16_32(shift_right, vshr, _n_, SIMD_SHIFT)

#undef SIMD_NEON_BUILD_TPL_SIMD_SHIFT
#undef SIMD_NEON_BUILD_RET_SIMD_SHIFT
#undef SIMD_NEON_BUILD_PARAM_SIMD_SHIFT
#undef SIMD_NEON_BUILD_ARG_SIMD_SHIFT

// ------------------------------ Shift lanes by same variable #bits

// Extra overhead, use _var instead unless SSE4 support is required.

template <typename T>
struct shift_left_count {
  Vec128<T> v;
};

template <typename T>
struct shift_right_count {
  Vec128<T> v;
};

template <typename T>
SIMD_INLINE shift_left_count<T> set_shift_left_count(Full128<T> d,
                                                     const int bits) {
  return shift_left_count<T>{set1(d, bits)};
}

template <typename T>
SIMD_INLINE shift_right_count<T> set_shift_right_count(Full128<T> d,
                                                       const int bits) {
  return shift_right_count<T>{set1(d, -bits)};
}

// Unsigned (no u8)
template <size_t N>
SIMD_INLINE Vec128<uint16_t, N> shift_left_same(
    const Vec128<uint16_t, N> v, const shift_left_count<uint16_t> bits) {
  return Vec128<uint16_t, N>(vshlq_u16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint16_t, N> shift_right_same(
    const Vec128<uint16_t, N> v, const shift_right_count<uint16_t> bits) {
  return Vec128<uint16_t, N>(vshlq_u16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint32_t, N> shift_left_same(
    const Vec128<uint32_t, N> v, const shift_left_count<uint32_t> bits) {
  return Vec128<uint32_t, N>(vshlq_u32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint32_t, N> shift_right_same(
    const Vec128<uint32_t, N> v, const shift_right_count<uint32_t> bits) {
  return Vec128<uint32_t, N>(vshlq_u32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint64_t, N> shift_left_same(
    const Vec128<uint64_t, N> v, const shift_left_count<uint64_t> bits) {
  return Vec128<uint64_t, N>(vshlq_u64(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint64_t, N> shift_right_same(
    const Vec128<uint64_t, N> v, const shift_right_count<uint64_t> bits) {
  return Vec128<uint64_t, N>(vshlq_u64(v.raw, bits.v.raw));
}

// Signed (no i8,i64)
template <size_t N>
SIMD_INLINE Vec128<int16_t, N> shift_left_same(
    const Vec128<int16_t, N> v, const shift_left_count<int16_t> bits) {
  return Vec128<int16_t, N>(vshlq_s16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int16_t, N> shift_right_same(
    const Vec128<int16_t, N> v, const shift_right_count<int16_t> bits) {
  return Vec128<int16_t, N>(vshlq_s16(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> shift_left_same(
    const Vec128<int32_t, N> v, const shift_left_count<int32_t> bits) {
  return Vec128<int32_t, N>(vshlq_s32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> shift_right_same(
    const Vec128<int32_t, N> v, const shift_right_count<int32_t> bits) {
  return Vec128<int32_t, N>(vshlq_s32(v.raw, bits.v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int64_t, N> shift_left_same(
    const Vec128<int64_t, N> v, const shift_left_count<int64_t> bits) {
  return Vec128<int64_t, N>(vshlq_s64(v.raw, bits.v.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
template <size_t N>
SIMD_INLINE Vec128<uint32_t, N> operator<<(const Vec128<uint32_t, N> v,
                                           const Vec128<uint32_t, N> bits) {
  return Vec128<uint32_t, N>(vshlq_u32(v.raw, bits.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint32_t, N> operator>>(const Vec128<uint32_t, N> v,
                                           const Vec128<uint32_t, N> bits) {
  return Vec128<uint32_t, N>(
      vshlq_u32(v.raw, vnegq_s32(vreinterpretq_s32_u32(bits.raw))));
}
template <size_t N>
SIMD_INLINE Vec128<uint64_t, N> operator<<(const Vec128<uint64_t, N> v,
                                           const Vec128<uint64_t, N> bits) {
  return Vec128<uint64_t, N>(vshlq_u64(v.raw, bits.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint64_t, N> operator>>(const Vec128<uint64_t, N> v,
                                           const Vec128<uint64_t, N> bits) {
  return Vec128<uint64_t, N>(
      vshlq_u64(v.raw, vnegq_s64(vreinterpretq_s64_u64(bits.raw))));
}

// Signed (no i8,i16)
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> operator<<(const Vec128<int32_t, N> v,
                                          const Vec128<int32_t, N> bits) {
  return Vec128<int32_t, N>(vshlq_s32(v.raw, bits.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> operator>>(const Vec128<int32_t, N> v,
                                          const Vec128<int32_t, N> bits) {
  return Vec128<int32_t, N>(vshlq_s32(v.raw, vnegq_s32(bits.raw)));
}
template <size_t N>
SIMD_INLINE Vec128<int64_t, N> operator<<(const Vec128<int64_t, N> v,
                                          const Vec128<int64_t, N> bits) {
  return Vec128<int64_t, N>(vshlq_s64(v.raw, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
SIMD_NEON_DEF_FUNCTION_UINT_8_16_32(min, vmin, _, 2)

// Signed (no i64)
SIMD_NEON_DEF_FUNCTION_INT_8_16_32(min, vmin, _, 2)

// Float
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(min, vmin, _, 2)

// ------------------------------ Maximum

// Unsigned (no u64)
SIMD_NEON_DEF_FUNCTION_UINT_8_16_32(max, vmax, _, 2)

// Signed (no i64)
SIMD_NEON_DEF_FUNCTION_INT_8_16_32(max, vmax, _, 2)

// Float
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(max, vmax, _, 2)

// ------------------------------ Clamping
template <typename T, size_t N>
SIMD_INLINE Vec128<T, N> clamp(const Vec128<T, N> v, const Vec128<T, N> lo,
                               const Vec128<T, N> hi) {
  return min(max(lo, v), hi);
}
// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
SIMD_INLINE Vec128<uint16_t, N> operator*(const Vec128<uint16_t, N> a,
                                          const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(vmulq_u16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint32_t, N> operator*(const Vec128<uint32_t, N> a,
                                          const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(vmulq_u32(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_INLINE Vec128<int16_t, N> operator*(const Vec128<int16_t, N> a,
                                         const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(vmulq_s16(a.raw, b.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> operator*(const Vec128<int32_t, N> a,
                                         const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(vmulq_s32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
SIMD_INLINE Vec128<int16_t, N> mul_high(const Vec128<int16_t, N> a,
                                        const Vec128<int16_t, N> b) {
  int32x4_t rlo = vmull_s16(vget_low_s16(a.raw), vget_low_s16(b.raw));
  int32x4_t rhi = vmull_high_s16(a.raw, b.raw);
  return Vec128<int16_t, N>(
      vuzp2q_s16(vreinterpretq_s16_s32(rlo), vreinterpretq_s16_s32(rhi)));
}
template <size_t N>
SIMD_INLINE Vec128<uint16_t, N> mul_high(const Vec128<uint16_t, N> a,
                                         const Vec128<uint16_t, N> b) {
  int32x4_t rlo = vmull_u16(vget_low_u16(a.raw), vget_low_u16(b.raw));
  int32x4_t rhi = vmull_high_u16(a.raw, b.raw);
  return Vec128<uint16_t, N>(
      vuzp2q_u16(vreinterpretq_u16_u32(rlo), vreinterpretq_u16_u32(rhi)));
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_INLINE Vec128<int64_t> mul_even(const Vec128<int32_t> a,
                                     const Vec128<int32_t> b) {
  int32x4_t a_packed = vuzp1q_s32(a.raw, a.raw);
  int32x4_t b_packed = vuzp1q_s32(b.raw, b.raw);
  return Vec128<int64_t>(
      vmull_s32(vget_low_s32(a_packed), vget_low_s32(b_packed)));
}
SIMD_INLINE Vec128<uint64_t> mul_even(const Vec128<uint32_t> a,
                                      const Vec128<uint32_t> b) {
  uint32x4_t a_packed = vuzp1q_u32(a.raw, a.raw);
  uint32x4_t b_packed = vuzp1q_u32(b.raw, b.raw);
  return Vec128<uint64_t>(
      vmull_u32(vget_low_u32(a_packed), vget_low_u32(b_packed)));
}

// ------------------------------ Floating-point negate

SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(neg, vneg, _, 1)

// ------------------------------ Floating-point mul / div

SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(operator*, vmul, _, 2)
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(operator/, vdiv, _, 2)

// Approximate reciprocal
SIMD_INLINE Vec128<float, 4> approximate_reciprocal(const Vec128<float, 4> v) {
  return Vec128<float, 4>(vrecpeq_f32(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<float, N> approximate_reciprocal(const Vec128<float, N> v) {
  return Vec128<float, N>(vrecpe_f32(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns add + mul * x
SIMD_INLINE Vec128<float, 4> mul_add(const Vec128<float, 4> mul,
                                     const Vec128<float, 4> x,
                                     const Vec128<float, 4> add) {
  return Vec128<float, 4>(vfmaq_f32(add.raw, mul.raw, x.raw));
}
SIMD_INLINE Vec128<float, 2> mul_add(const Vec128<float, 2> mul,
                                     const Vec128<float, 2> x,
                                     const Vec128<float, 2> add) {
  return Vec128<float, 2>(vfma_f32(add.raw, mul.raw, x.raw));
}
SIMD_INLINE Vec128<double, 2> mul_add(const Vec128<double, 2> mul,
                                      const Vec128<double, 2> x,
                                      const Vec128<double, 2> add) {
  return Vec128<double, 2>(vfmaq_f64(add.raw, mul.raw, x.raw));
}
SIMD_INLINE Vec128<double, 1> mul_add(const Vec128<double, 1> mul,
                                      const Vec128<double, 1> x,
                                      const Vec128<double, 1> add) {
  return Vec128<double, 1>(vfma_f64(add.raw, mul.raw, x.raw));
}

// Returns add - mul * x
template <size_t N>
SIMD_INLINE Vec128<float, N> nmul_add(const Vec128<float, N> mul,
                                      const Vec128<float, N> x,
                                      const Vec128<float, N> add) {
  return Vec128<float, N>(vfmsq_f32(add.raw, mul.raw, x.raw));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> nmul_add(const Vec128<double, N> mul,
                                       const Vec128<double, N> x,
                                       const Vec128<double, N> add) {
  return Vec128<double, N>(vfmsq_f64(add.raw, mul.raw, x.raw));
}

// Slightly more expensive (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
SIMD_INLINE Vec128<float, N> mul_subtract(const Vec128<float, N> mul,
                                          const Vec128<float, N> x,
                                          const Vec128<float, N> sub) {
  return neg(nmul_add(mul, x, sub));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> mul_subtract(const Vec128<double, N> mul,
                                           const Vec128<double, N> x,
                                           const Vec128<double, N> sub) {
  return neg(nmul_add(mul, x, sub));
}

// Returns -mul * x - sub
template <size_t N>
SIMD_INLINE Vec128<float, N> nmul_subtract(const Vec128<float, N> mul,
                                           const Vec128<float, N> x,
                                           const Vec128<float, N> sub) {
  return neg(mul_add(mul, x, sub));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> nmul_subtract(const Vec128<double, N> mul,
                                            const Vec128<double, N> x,
                                            const Vec128<double, N> sub) {
  return neg(mul_add(mul, x, sub));
}

}  // namespace ext

// Returns x + add
template <size_t N>
SIMD_INLINE Vec128<float, N> fadd(Vec128<float, N> x, const Vec128<float, N> k1,
                                  const Vec128<float, N> add) {
  return x + add;
}
template <size_t N>
SIMD_INLINE Vec128<double, N> fadd(Vec128<double, N> x,
                                   const Vec128<double, N> k1,
                                   const Vec128<double, N> add) {
  return x + add;
}

// Returns x - sub
template <size_t N>
SIMD_INLINE Vec128<float, N> fsub(Vec128<float, N> x, const Vec128<float, N> k1,
                                  const Vec128<float, N> sub) {
  return x - sub;
}
template <size_t N>
SIMD_INLINE Vec128<double, N> fsub(Vec128<double, N> x,
                                   const Vec128<double, N> k1,
                                   const Vec128<double, N> sub) {
  return x - sub;
}

// Returns -sub + x
template <size_t N>
SIMD_INLINE Vec128<float, N> fnadd(Vec128<float, N> sub,
                                   const Vec128<float, N> k1,
                                   const Vec128<float, N> x) {
  return x - sub;
}
template <size_t N>
SIMD_INLINE Vec128<double, N> fnadd(Vec128<double, N> sub,
                                    const Vec128<double, N> k1,
                                    const Vec128<double, N> x) {
  return x - sub;
}

// ------------------------------ Floating-point square root

// Full precision square root
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(sqrt, vsqrt, _, 1)

// Approximate reciprocal square root
template <size_t N>
SIMD_INLINE Vec128<float, N> approximate_reciprocal_sqrt(
    const Vec128<float, N> v) {
  return Vec128<float, N>(vrsqrteq_f32(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(round, vrndn, _, 1)

// Toward zero, aka truncate
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(trunc, vrnd, _, 1)

// Toward +infinity, aka ceiling
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(ceil, vrndp, _, 1)

// Toward -infinity, aka floor
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(floor, vrndm, _, 1)

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality
SIMD_NEON_DEF_FUNCTION_ALL_TYPES(operator==, vceq, _, 2)

// ------------------------------ Strict inequality

// Signed/float < (no unsigned)
SIMD_NEON_DEF_FUNCTION_INTS(operator<, vclt, _, 2)
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(operator<, vclt, _, 2)

// Signed/float > (no unsigned)
SIMD_NEON_DEF_FUNCTION_INTS(operator>, vcgt, _, 2)
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(operator>, vcgt, _, 2)

// ------------------------------ Weak inequality

// Float <= >=
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(operator<=, vcle, _, 2)
SIMD_NEON_DEF_FUNCTION_ALL_FLOATS(operator>=, vcge, _, 2)

// ================================================== LOGICAL

// ------------------------------ Bitwise AND
SIMD_NEON_DEF_FUNCTION_INTS_UINTS(operator&, vand, _, 2)
// These operator& rely on the special cases for uint32_t and uint64_t just
// defined by SIMD_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
SIMD_INLINE Vec128<float, N> operator&(const Vec128<float, N> a,
                                       const Vec128<float, N> b) {
  const Full128<uint32_t> d;
  return bit_cast(Full128<float>(), bit_cast(d, a) & bit_cast(d, b));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> operator&(const Vec128<double, N> a,
                                        const Vec128<double, N> b) {
  const Full128<uint64_t> d;
  return bit_cast(Full128<double>(), bit_cast(d, a) & bit_cast(d, b));
}

// ------------------------------ Bitwise AND-NOT

namespace internal {
// reversed_andnot returns a & ~b.
SIMD_NEON_DEF_FUNCTION_INTS_UINTS(reversed_andnot, vbic, _, 2)
}  // namespace internal

// Returns ~not_mask & mask.
template <typename T, size_t N>
SIMD_INLINE Vec128<T, N> andnot(const Vec128<T, N> not_mask,
                                const Vec128<T, N> mask) {
  return internal::reversed_andnot(mask, not_mask);
}

// These andnot() rely on the special cases for uint32_t and uint64_t just
// defined by SIMD_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
SIMD_INLINE Vec128<float, N> andnot(const Vec128<float, N> not_mask,
                                    const Vec128<float, N> mask) {
  const Desc<uint32_t, N> du;
  Vec128<uint32_t, N> ret =
      internal::reversed_andnot(bit_cast(du, mask), bit_cast(du, not_mask));
  return Vec128<float, N>(vreinterpretq_f32_u32(ret.raw));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> andnot(const Vec128<double, N> not_mask,
                                     const Vec128<double, N> mask) {
  const Desc<uint64_t, N> du;
  Vec128<uint64_t, N> ret =
      internal::reversed_andnot(bit_cast(du, mask), bit_cast(du, not_mask));
  return Vec128<double, N>(vreinterpretq_f64_u64(ret.raw));
}

// ------------------------------ Bitwise OR

SIMD_NEON_DEF_FUNCTION_INTS_UINTS(operator|, vorr, _, 2)

// These operator| rely on the special cases for uint32_t and uint64_t just
// defined by SIMD_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
SIMD_INLINE Vec128<float, N> operator|(const Vec128<float, N> a,
                                       const Vec128<float, N> b) {
  const Desc<uint32_t, N> d;
  return bit_cast(Full128<float>(), bit_cast(d, a) | bit_cast(d, b));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> operator|(const Vec128<double, N> a,
                                        const Vec128<double, N> b) {
  const Desc<uint64_t, N> d;
  return bit_cast(Full128<double>(), bit_cast(d, a) | bit_cast(d, b));
}

// ------------------------------ Bitwise XOR

SIMD_NEON_DEF_FUNCTION_INTS_UINTS(operator^, veor, _, 2)

// These operator| rely on the special cases for uint32_t and uint64_t just
// defined by SIMD_NEON_DEF_FUNCTION_INTS_UINTS() macro.
template <size_t N>
SIMD_INLINE Vec128<float, N> operator^(const Vec128<float, N> a,
                                       const Vec128<float, N> b) {
  const Desc<uint32_t, N> d;
  return bit_cast(Full128<float>(), bit_cast(d, a) ^ bit_cast(d, b));
}
template <size_t N>
SIMD_INLINE Vec128<double, N> operator^(const Vec128<double, N> a,
                                        const Vec128<double, N> b) {
  const Desc<uint64_t, N> d;
  return bit_cast(Full128<double>(), bit_cast(d, a) ^ bit_cast(d, b));
}

// ------------------------------ Select/blend

// Returns a mask for use by if_then_else().
// blendv_ps/pd only check the sign bit, so this is a no-op on x86.
template <size_t N>
SIMD_INLINE Vec128<float, N> mask_from_sign(const Vec128<float, N> v) {
  const Desc<float, N> df;
  const Desc<int32_t, N> di;
  return bit_cast(df, shift_right<31>(bit_cast(di, v)));
}
// There's no shift_right defined for int64_t, so we need to specialize this
// case using the underlying vshrq_n_s64 and vshr_n_s64 functions.
SIMD_INLINE Vec128<double, 2> mask_from_sign(const Vec128<double, 2> v) {
  const Desc<double, 2> df;
  const Desc<int64_t, 2> di;
  return bit_cast(df, Vec128<double, 2>(vshrq_n_s64(bit_cast(di, v).raw, 63)));
}
SIMD_INLINE Vec128<double, 1> mask_from_sign(const Vec128<double, 1> v) {
  const Desc<double, 1> df;
  const Desc<int64_t, 1> di;
  return bit_cast(df, Vec128<double, 1>(vshr_n_s64(bit_cast(di, v).raw, 63)));
}

// if_then_else(mask, yes, no)
// Returns mask ? b : a. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
SIMD_NEON_DEF_FUNCTION_ALL_TYPES(if_then_else, vbsl, _, 3)

// ================================================== MEMORY

// ------------------------------ Load 128

SIMD_INLINE Vec128<uint8_t> load_u(Full128<uint8_t> /* tag */,
                                   const uint8_t* SIMD_RESTRICT aligned) {
  return Vec128<uint8_t>(vld1q_u8(aligned));
}
SIMD_INLINE Vec128<uint16_t> load_u(Full128<uint16_t> /* tag */,
                                    const uint16_t* SIMD_RESTRICT aligned) {
  return Vec128<uint16_t>(vld1q_u16(aligned));
}
SIMD_INLINE Vec128<uint32_t> load_u(Full128<uint32_t> /* tag */,
                                    const uint32_t* SIMD_RESTRICT aligned) {
  return Vec128<uint32_t>(vld1q_u32(aligned));
}
SIMD_INLINE Vec128<uint64_t> load_u(Full128<uint64_t> /* tag */,
                                    const uint64_t* SIMD_RESTRICT aligned) {
  return Vec128<uint64_t>(vld1q_u64(aligned));
}
SIMD_INLINE Vec128<int8_t> load_u(Full128<int8_t> /* tag */,
                                  const int8_t* SIMD_RESTRICT aligned) {
  return Vec128<int8_t>(vld1q_s8(aligned));
}
SIMD_INLINE Vec128<int16_t> load_u(Full128<int16_t> /* tag */,
                                   const int16_t* SIMD_RESTRICT aligned) {
  return Vec128<int16_t>(vld1q_s16(aligned));
}
SIMD_INLINE Vec128<int32_t> load_u(Full128<int32_t> /* tag */,
                                   const int32_t* SIMD_RESTRICT aligned) {
  return Vec128<int32_t>(vld1q_s32(aligned));
}
SIMD_INLINE Vec128<int64_t> load_u(Full128<int64_t> /* tag */,
                                   const int64_t* SIMD_RESTRICT aligned) {
  return Vec128<int64_t>(vld1q_s64(aligned));
}
SIMD_INLINE Vec128<float> load_u(Full128<float> /* tag */,
                                 const float* SIMD_RESTRICT aligned) {
  return Vec128<float>(vld1q_f32(aligned));
}
SIMD_INLINE Vec128<double> load_u(Full128<double> /* tag */,
                                  const double* SIMD_RESTRICT aligned) {
  return Vec128<double>(vld1q_f64(aligned));
}

template <typename T>
SIMD_INLINE Vec128<T> load(Full128<T> d, const T* SIMD_RESTRICT p) {
  return load_u(d, p);
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_INLINE Vec128<T> load_dup128(Full128<T> d,
                                  const T* const SIMD_RESTRICT p) {
  return load_u(d, p);
}

// ------------------------------ Load 64

SIMD_INLINE Vec128<uint8_t, 8> load(Desc<uint8_t, 8> /* tag */,
                                    const uint8_t* SIMD_RESTRICT p) {
  return Vec128<uint8_t, 8>(vld1_u8(p));
}
SIMD_INLINE Vec128<uint16_t, 4> load(Desc<uint16_t, 4> /* tag */,
                                     const uint16_t* SIMD_RESTRICT p) {
  return Vec128<uint16_t, 4>(vld1_u16(p));
}
SIMD_INLINE Vec128<uint32_t, 2> load(Desc<uint32_t, 2> /* tag */,
                                     const uint32_t* SIMD_RESTRICT p) {
  return Vec128<uint32_t, 2>(vld1_u32(p));
}
SIMD_INLINE Vec128<uint64_t, 1> load(Desc<uint64_t, 1> /* tag */,
                                     const uint64_t* SIMD_RESTRICT p) {
  return Vec128<uint64_t, 1>(vld1_u64(p));
}
SIMD_INLINE Vec128<int8_t, 8> load(Desc<int8_t, 8> /* tag */,
                                   const int8_t* SIMD_RESTRICT p) {
  return Vec128<int8_t, 8>(vld1_s8(p));
}
SIMD_INLINE Vec128<int16_t, 4> load(Desc<int16_t, 4> /* tag */,
                                    const int16_t* SIMD_RESTRICT p) {
  return Vec128<int16_t, 4>(vld1_s16(p));
}
SIMD_INLINE Vec128<int32_t, 2> load(Desc<int32_t, 2> /* tag */,
                                    const int32_t* SIMD_RESTRICT p) {
  return Vec128<int32_t, 2>(vld1_s32(p));
}
SIMD_INLINE Vec128<int64_t, 1> load(Desc<int64_t, 1> /* tag */,
                                    const int64_t* SIMD_RESTRICT p) {
  return Vec128<int64_t, 1>(vld1_s64(p));
}
SIMD_INLINE Vec128<float, 2> load(Desc<float, 2> /* tag */,
                                  const float* SIMD_RESTRICT p) {
  return Vec128<float, 2>(vld1_f32(p));
}
SIMD_INLINE Vec128<double, 1> load(Desc<double, 1> /* tag */,
                                   const double* SIMD_RESTRICT p) {
  return Vec128<double, 1>(vld1_f64(p));
}

// ------------------------------ Load 32

// In the following load functions, |a| is purposely undefined.
// It is a required parameter to the intrinsic, however
// we don't actually care what is in it, and we don't want
// to introduce extra overhead by initializing it to something.

SIMD_INLINE Vec128<uint8_t, 4> load(Desc<uint8_t, 4> d,
                                    const uint8_t* SIMD_RESTRICT p) {
  uint32x2_t a = undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return Vec128<uint8_t, 4>(vreinterpret_u8_u32(b));
}
SIMD_INLINE Vec128<uint16_t, 2> load(Desc<uint16_t, 2> d,
                                     const uint16_t* SIMD_RESTRICT p) {
  uint32x2_t a = undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(reinterpret_cast<const uint32_t*>(p), a, 0);
  return Vec128<uint16_t, 2>(vreinterpret_u16_u32(b));
}
SIMD_INLINE Vec128<uint32_t, 1> load(Desc<uint32_t, 1> d,
                                     const uint32_t* SIMD_RESTRICT p) {
  uint32x2_t a = undefined(d).raw;
  uint32x2_t b = vld1_lane_u32(p, a, 0);
  return Vec128<uint32_t, 1>(b);
}
SIMD_INLINE Vec128<int8_t, 4> load(Desc<int8_t, 4> d,
                                   const int8_t* SIMD_RESTRICT p) {
  int32x2_t a = undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return Vec128<int8_t, 4>(vreinterpret_s8_s32(b));
}
SIMD_INLINE Vec128<int16_t, 2> load(Desc<int16_t, 2> d,
                                    const int16_t* SIMD_RESTRICT p) {
  int32x2_t a = undefined(d).raw;
  int32x2_t b = vld1_lane_s32(reinterpret_cast<const int32_t*>(p), a, 0);
  return Vec128<int16_t, 2>(vreinterpret_s16_s32(b));
}
SIMD_INLINE Vec128<int32_t, 1> load(Desc<int32_t, 1> d,
                                    const int32_t* SIMD_RESTRICT p) {
  int32x2_t a = undefined(d).raw;
  int32x2_t b = vld1_lane_s32(p, a, 0);
  return Vec128<int32_t, 1>(b);
}
SIMD_INLINE Vec128<float, 1> load(Desc<float, 1> d,
                                  const float* SIMD_RESTRICT p) {
  float32x2_t a = undefined(d).raw;
  float32x2_t b = vld1_lane_f32(p, a, 0);
  return Vec128<float, 1>(b);
}

// ------------------------------ Store 128

SIMD_INLINE void store_u(const Vec128<uint8_t> v, Full128<uint8_t> /* tag */,
                         uint8_t* SIMD_RESTRICT aligned) {
  vst1q_u8(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<uint16_t> v, Full128<uint16_t> /* tag */,
                         uint16_t* SIMD_RESTRICT aligned) {
  vst1q_u16(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<uint32_t> v, Full128<uint32_t> /* tag */,
                         uint32_t* SIMD_RESTRICT aligned) {
  vst1q_u32(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<uint64_t> v, Full128<uint64_t> /* tag */,
                         uint64_t* SIMD_RESTRICT aligned) {
  vst1q_u64(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<int8_t> v, Full128<int8_t> /* tag */,
                         int8_t* SIMD_RESTRICT aligned) {
  vst1q_s8(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<int16_t> v, Full128<int16_t> /* tag */,
                         int16_t* SIMD_RESTRICT aligned) {
  vst1q_s16(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<int32_t> v, Full128<int32_t> /* tag */,
                         int32_t* SIMD_RESTRICT aligned) {
  vst1q_s32(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<int64_t> v, Full128<int64_t> /* tag */,
                         int64_t* SIMD_RESTRICT aligned) {
  vst1q_s64(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<float> v, Full128<float> /* tag */,
                         float* SIMD_RESTRICT aligned) {
  vst1q_f32(aligned, v.raw);
}
SIMD_INLINE void store_u(const Vec128<double> v, Full128<double> /* tag */,
                         double* SIMD_RESTRICT aligned) {
  vst1q_f64(aligned, v.raw);
}

template <typename T, size_t N>
SIMD_INLINE void store(Vec128<T, N> v, Desc<T, N> d, T* SIMD_RESTRICT p) {
  store_u(v, d, p);
}

// ------------------------------ Store 64

SIMD_INLINE void store(const Vec128<uint8_t, 8> v, Desc<uint8_t, 8>,
                       uint8_t* SIMD_RESTRICT p) {
  vst1_u8(p, v.raw);
}
SIMD_INLINE void store(const Vec128<uint16_t, 4> v, Desc<uint16_t, 4>,
                       uint16_t* SIMD_RESTRICT p) {
  vst1_u16(p, v.raw);
}
SIMD_INLINE void store(const Vec128<uint32_t, 2> v, Desc<uint32_t, 2>,
                       uint32_t* SIMD_RESTRICT p) {
  vst1_u32(p, v.raw);
}
SIMD_INLINE void store(const Vec128<uint64_t, 1> v, Desc<uint64_t, 1>,
                       uint64_t* SIMD_RESTRICT p) {
  vst1_u64(p, v.raw);
}
SIMD_INLINE void store(const Vec128<int8_t, 8> v, Desc<int8_t, 8>,
                       int8_t* SIMD_RESTRICT p) {
  vst1_s8(p, v.raw);
}
SIMD_INLINE void store(const Vec128<int16_t, 4> v, Desc<int16_t, 4>,
                       int16_t* SIMD_RESTRICT p) {
  vst1_s16(p, v.raw);
}
SIMD_INLINE void store(const Vec128<int32_t, 2> v, Desc<int32_t, 2>,
                       int32_t* SIMD_RESTRICT p) {
  vst1_s32(p, v.raw);
}
SIMD_INLINE void store(const Vec128<int64_t, 1> v, Desc<int64_t, 1>,
                       int64_t* SIMD_RESTRICT p) {
  vst1_s64(p, v.raw);
}
SIMD_INLINE void store(const Vec128<float, 2> v, Desc<float, 2>,
                       float* SIMD_RESTRICT p) {
  vst1_f32(p, v.raw);
}
SIMD_INLINE void store(const Vec128<double, 1> v, Desc<double, 1>,
                       double* SIMD_RESTRICT p) {
  vst1_f64(p, v.raw);
}

// ------------------------------ Store 32

SIMD_INLINE void store(const Vec128<uint8_t, 4> v, Desc<uint8_t, 4>,
                       uint8_t* SIMD_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u8(v.raw);
  vst1_lane_u32(p, a, 0);
}
SIMD_INLINE void store(const Vec128<uint16_t, 2> v, Desc<uint16_t, 2>,
                       uint16_t* SIMD_RESTRICT p) {
  uint32x2_t a = vreinterpret_u32_u16(v.raw);
  vst1_lane_u32(p, a, 0);
}
SIMD_INLINE void store(const Vec128<uint32_t, 1> v, Desc<uint32_t, 1>,
                       uint32_t* SIMD_RESTRICT p) {
  vst1_lane_u32(p, v.raw, 0);
}
SIMD_INLINE void store(const Vec128<int8_t, 4> v, Desc<int8_t, 4>,
                       int8_t* SIMD_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s8(v.raw);
  vst1_lane_s32(p, a, 0);
}
SIMD_INLINE void store(const Vec128<int16_t, 2> v, Desc<int16_t, 2>,
                       int16_t* SIMD_RESTRICT p) {
  int32x2_t a = vreinterpret_s32_s16(v.raw);
  vst1_lane_s32(p, a, 0);
}
SIMD_INLINE void store(const Vec128<int32_t, 1> v, Desc<int32_t, 1>,
                       int32_t* SIMD_RESTRICT p) {
  vst1_lane_s32(p, v.raw, 0);
}
SIMD_INLINE void store(const Vec128<float, 1> v, Desc<float, 1>,
                       float* SIMD_RESTRICT p) {
  vst1_lane_f32(p, v.raw, 0);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_INLINE void stream(const Vec128<T> v, Full128<T> d,
                        T* SIMD_RESTRICT aligned) {
  store(v, d, aligned);
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
SIMD_INLINE Vec128<uint16_t> convert_to(Full128<uint16_t> /* tag */,
                                        const Vec128<uint8_t, 8> v) {
  return Vec128<uint16_t>(vmovl_u8(v.raw));
}
SIMD_INLINE Vec128<uint32_t> convert_to(Full128<uint32_t> /* tag */,
                                        const Vec128<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<uint32_t>(vmovl_u16(vget_low_u16(a)));
}
SIMD_INLINE Vec128<uint32_t> convert_to(Full128<uint32_t> /* tag */,
                                        const Vec128<uint16_t, 4> v) {
  return Vec128<uint32_t>(vmovl_u16(v.raw));
}
SIMD_INLINE Vec128<uint64_t> convert_to(Full128<uint64_t> /* tag */,
                                        const Vec128<uint32_t, 2> v) {
  return Vec128<uint64_t>(vmovl_u32(v.raw));
}
SIMD_INLINE Vec128<int16_t> convert_to(Full128<int16_t> /* tag */,
                                       const Vec128<uint8_t, 8> v) {
  return Vec128<int16_t>(vmovl_u8(v.raw));
}
SIMD_INLINE Vec128<int32_t> convert_to(Full128<int32_t> /* tag */,
                                       const Vec128<uint8_t, 4> v) {
  uint16x8_t a = vmovl_u8(v.raw);
  return Vec128<int32_t>(vreinterpretq_s32_u16(vmovl_u16(vget_low_u16(a))));
}
SIMD_INLINE Vec128<int32_t> convert_to(Full128<int32_t> /* tag */,
                                       const Vec128<uint16_t, 4> v) {
  return Vec128<int32_t>(vmovl_u16(v.raw));
}

SIMD_INLINE Vec128<uint32_t> u32_from_u8(const Vec128<uint8_t> v) {
  return Vec128<uint32_t>(
      vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(v.raw)))));
}

// Signed: replicate sign bit.
SIMD_INLINE Vec128<int16_t> convert_to(Full128<int16_t> /* tag */,
                                       const Vec128<int8_t, 8> v) {
  return Vec128<int16_t>(vmovl_s8(v.raw));
}
SIMD_INLINE Vec128<int32_t> convert_to(Full128<int32_t> /* tag */,
                                       const Vec128<int8_t, 4> v) {
  int16x8_t a = vmovl_s8(v.raw);
  return Vec128<int32_t>(vmovl_s16(vget_low_s16(a)));
}
SIMD_INLINE Vec128<int32_t> convert_to(Full128<int32_t> /* tag */,
                                       const Vec128<int16_t, 4> v) {
  return Vec128<int32_t>(vmovl_s16(v.raw));
}
SIMD_INLINE Vec128<int64_t> convert_to(Full128<int64_t> /* tag */,
                                       const Vec128<int32_t, 2> v) {
  return Vec128<int64_t>(vmovl_s32(v.raw));
}

SIMD_INLINE Vec128<double> convert_to(Full128<double> /* tag */,
                                      const Vec128<float, 2> v) {
  // vcvt_high_f64_f32 takes a float32x4_t even it only uses 2 of the values.
  const float32x4_t w = vcombine_f32(v.raw, v.raw);
  return Vec128<double>(vcvt_high_f64_f32(w));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <size_t N>
SIMD_INLINE Vec128<uint16_t, N> convert_to(Desc<uint16_t, N> /* tag */,
                                           const Vec128<int32_t> v) {
  return Vec128<uint16_t, N>(vqmovun_s32(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<uint8_t, N> convert_to(Desc<uint8_t, N> /* tag */,
                                          const Vec128<uint16_t> v) {
  return Vec128<uint8_t, N>(vqmovn_u16(v.raw));
}

template <size_t N>
SIMD_INLINE Vec128<uint8_t, N> convert_to(Desc<uint8_t, N> /* tag */,
                                          const Vec128<int16_t> v) {
  return Vec128<uint8_t, N>(vqmovun_s16(v.raw));
}

template <size_t N>
SIMD_INLINE Vec128<int16_t, N> convert_to(Desc<int16_t, N> /* tag */,
                                          const Vec128<int32_t> v) {
  return Vec128<int16_t, N>(vqmovn_s32(v.raw));
}
template <size_t N>
SIMD_INLINE Vec128<int8_t, N> convert_to(Desc<int8_t, N> /* tag */,
                                         const Vec128<int16_t> v) {
  return Vec128<int8_t, N>(vqmovn_s16(v.raw));
}

SIMD_INLINE Vec128<uint8_t, 4> u8_from_u32(const Vec128<uint32_t> v) {
  const uint8x16_t org_v = cast_to_u8(v).raw;
  const uint8x16_t w = vuzp1q_u8(org_v, org_v);
  return Vec128<uint8_t, 4>(vget_low_u8(vuzp1q_u8(w, w)));
}

// In the following convert_to functions, |b| is purposely undefined.
// The value a needs to be extended to 128 bits so that vqmovn can be
// used and |b| is undefined so that no extra overhead is introduced.
SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")

template <size_t N>
SIMD_INLINE Vec128<uint8_t, N> convert_to(Desc<uint8_t, N> /* tag */,
                                          const Vec128<int32_t> v) {
  Vec128<uint16_t, N> a = convert_to(Desc<uint16_t, N>(), v);
  Vec128<uint16_t, N> b;
  uint16x8_t c = vcombine_u16(a.raw, b.raw);
  return Vec128<uint8_t, N>(vqmovn_u16(c));
}

template <size_t N>
SIMD_INLINE Vec128<int8_t, N> convert_to(Desc<int8_t, N> /* tag */,
                                         const Vec128<int32_t> v) {
  Vec128<int16_t, N> a = convert_to(Desc<int16_t, N>(), v);
  Vec128<int16_t, N> b;
  uint16x8_t c = vcombine_s16(a.raw, b.raw);
  return Vec128<int8_t, N>(vqmovn_s16(c));
}

SIMD_DIAGNOSTICS(pop)

// ------------------------------ Convert i32 <=> f32

template <size_t N>
SIMD_INLINE Vec128<float, N> convert_to(Desc<float, N> /* tag */,
                                        const Vec128<int32_t, N> v) {
  return Vec128<float, N>(vcvtq_f32_s32(v.raw));
}
// Truncates (rounds toward zero).
template <size_t N>
SIMD_INLINE Vec128<int32_t, N> convert_to(Desc<int32_t, N> /* tag */,
                                          const Vec128<float, N> v) {
  return Vec128<int32_t, N>(vcvtq_s32_f32(v.raw));
}

template <size_t N>
SIMD_INLINE Vec128<int32_t, N> nearest_int(const Vec128<float, N> v) {
  return Vec128<int32_t, N>(vcvtnq_s32_f32(v.raw));
}

// ================================================== SWIZZLE

// ------------------------------ 'Extract' other half (see any_part)

// These copy hi into lo
SIMD_INLINE Vec128<uint8_t, 8> other_half(const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 8>(vget_high_u8(v.raw));
}
SIMD_INLINE Vec128<int8_t, 8> other_half(const Vec128<int8_t> v) {
  return Vec128<int8_t, 8>(vget_high_s8(v.raw));
}
SIMD_INLINE Vec128<uint16_t, 4> other_half(const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 4>(vget_high_u16(v.raw));
}
SIMD_INLINE Vec128<int16_t, 4> other_half(const Vec128<int16_t> v) {
  return Vec128<int16_t, 4>(vget_high_s16(v.raw));
}
SIMD_INLINE Vec128<uint32_t, 2> other_half(const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 2>(vget_high_u32(v.raw));
}
SIMD_INLINE Vec128<int32_t, 2> other_half(const Vec128<int32_t> v) {
  return Vec128<int32_t, 2>(vget_high_s32(v.raw));
}
SIMD_INLINE Vec128<uint64_t, 1> other_half(const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_high_u64(v.raw));
}
SIMD_INLINE Vec128<int64_t, 1> other_half(const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_high_s64(v.raw));
}
SIMD_INLINE Vec128<float, 2> other_half(const Vec128<float> v) {
  return Vec128<float, 2>(vget_high_f32(v.raw));
}
SIMD_INLINE Vec128<double, 1> other_half(const Vec128<double> v) {
  return Vec128<double, 1>(vget_high_f64(v.raw));
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_INLINE Vec128<T> combine_shift_right_bytes(const Vec128<T> hi,
                                                const Vec128<T> lo) {
  static_assert(0 < kBytes && kBytes < 16, "kBytes must be in [1, 15]");
  const Full128<uint8_t> d8;
  return bit_cast(Full128<T>(),
                  Vec128<uint8_t>(vextq_u8(bit_cast(d8, lo).raw,
                                           bit_cast(d8, hi).raw, kBytes)));
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T, size_t N>
SIMD_INLINE Vec128<T, N> shift_left_bytes(const Vec128<T, N> v) {
  return combine_shift_right_bytes<16 - kBytes>(v, setzero(Full128<T>()));
}

template <int kLanes, typename T>
SIMD_INLINE Vec128<T> shift_left_lanes(const Vec128<T> v) {
  return shift_left_bytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T, size_t N>
SIMD_INLINE Vec128<T, N> shift_right_bytes(const Vec128<T, N> v) {
  return combine_shift_right_bytes<kBytes>(setzero(Full128<T>()), v);
}

template <int kLanes, typename T>
SIMD_INLINE Vec128<T> shift_right_lanes(const Vec128<T> v) {
  return shift_right_bytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_INLINE Vec128<uint16_t> broadcast(const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<uint16_t>(vdupq_laneq_u16(v.raw, kLane));
}
template <int kLane, size_t N>
SIMD_INLINE Vec128<uint16_t> broadcast(const Vec128<uint16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<uint16_t>(vdupq_laneq_u16(vcombine_u16(v.raw, v.raw), kLane));
}
template <int kLane>
SIMD_INLINE Vec128<uint32_t> broadcast(const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<uint32_t>(vdupq_laneq_u32(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE Vec128<uint64_t> broadcast(const Vec128<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<uint64_t>(vdupq_laneq_u64(v.raw, kLane));
}

// Signed
template <int kLane>
SIMD_INLINE Vec128<int16_t> broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  return Vec128<int16_t>(vdupq_laneq_s16(v.raw, kLane));
}
template <int kLane, size_t N>
SIMD_INLINE Vec128<int16_t> broadcast(const Vec128<int16_t, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<int16_t>(vdupq_laneq_s16(vcombine_s16(v.raw, v.raw), kLane));
}
template <int kLane>
SIMD_INLINE Vec128<int32_t> broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<int32_t>(vdupq_laneq_s32(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE Vec128<int64_t> broadcast(const Vec128<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<int64_t>(vdupq_laneq_s64(v.raw, kLane));
}

// Float
template <int kLane>
SIMD_INLINE Vec128<float> broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<float>(vdupq_laneq_f32(v.raw, kLane));
}
template <int kLane>
SIMD_INLINE Vec128<double> broadcast(const Vec128<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<double>(vdupq_laneq_f64(v.raw, kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_INLINE Vec128<T> table_lookup_bytes(const Vec128<T> bytes,
                                         const Vec128<TI> from) {
  const Full128<uint8_t> d8;
  return bit_cast(Full128<T>(),
                  Vec128<uint8_t>(vqtbl1q_u8(bit_cast(d8, bytes).raw,
                                             bit_cast(d8, from).raw)));
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec128<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// combine_shift_right_bytes but the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
template <typename T>
SIMD_INLINE Vec128<T> shuffle_1032(const Vec128<T> v) {
  return combine_shift_right_bytes<8>(v, v);
}
template <typename T>
SIMD_INLINE Vec128<T> shuffle_01(const Vec128<T> v) {
  return combine_shift_right_bytes<8>(v, v);
}

// Rotate right 32 bits
template <typename T>
SIMD_INLINE Vec128<T> shuffle_0321(const Vec128<T> v) {
  return combine_shift_right_bytes<4>(v, v);
}

// Rotate left 32 bits
template <typename T>
SIMD_INLINE Vec128<T> shuffle_2103(const Vec128<T> v) {
  return combine_shift_right_bytes<12>(v, v);
}

// Reverse
template <typename T>
SIMD_INLINE Vec128<T> shuffle_0123(const Vec128<T> v) {
  static_assert(sizeof(T) == 4,
                "shuffle_0123 should only be applied to 32-bit types");
  // TODO(janwas): more efficient implementation?,
  // It is possible to use two instructions (vrev64q_u32 and vcombine_u32 of the
  // high/low parts) instead of the extra memory and load.
  static constexpr uint8_t bytes[16] = {12, 13, 14, 15, 8, 9, 10, 11,
                                        4,  5,  6,  7,  0, 1, 2,  3};
  return table_lookup_bytes(v, load(Full128<uint8_t>(), bytes));
}

// ------------------------------ Permute (runtime variable)

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct Permute128 {
  uint8x16_t raw;
};

template <typename T>
SIMD_INLINE Permute128<T> set_table_indices(const Full128<T> d,
                                            const int32_t* idx) {
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
  for (size_t i = 0; i < d.N; ++i) {
    if (idx[i] >= static_cast<int32_t>(d.N)) {
      printf("set_table_indices [%zu] = %d >= %zu\n", i, idx[i], d.N);
      SIMD_TRAP();
    }
  }
#else
  (void)d;
#endif

  const Full128<uint8_t> d8;
  SIMD_ALIGN uint8_t control[d8.N];
  for (size_t idx_byte = 0; idx_byte < d8.N; ++idx_byte) {
    const size_t idx_lane = idx_byte / sizeof(T);
    const size_t mod = idx_byte % sizeof(T);
    control[idx_byte] = idx[idx_lane] * sizeof(T) + mod;
  }
  return Permute128<T>{load(d8, control).raw};
}

SIMD_INLINE Vec128<uint32_t> table_lookup_lanes(
    const Vec128<uint32_t> v, const Permute128<uint32_t> idx) {
  return table_lookup_bytes(v, Vec128<uint8_t>(idx.raw));
}
SIMD_INLINE Vec128<int32_t> table_lookup_lanes(const Vec128<int32_t> v,
                                               const Permute128<int32_t> idx) {
  return table_lookup_bytes(v, Vec128<uint8_t>(idx.raw));
}
SIMD_INLINE Vec128<float> table_lookup_lanes(const Vec128<float> v,
                                             const Permute128<float> idx) {
  const Full128<int32_t> di;
  const Full128<float> df;
  return bit_cast(
      df, table_lookup_bytes(bit_cast(di, v), Vec128<uint8_t>(idx.raw)));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).
SIMD_NEON_DEF_FUNCTION_INT_8_16_32(interleave_lo, vzip1, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_8_16_32(interleave_lo, vzip1, _, 2)

SIMD_NEON_DEF_FUNCTION_INT_8_16_32(interleave_hi, vzip2, _, 2)
SIMD_NEON_DEF_FUNCTION_UINT_8_16_32(interleave_hi, vzip2, _, 2)

// For 64 bit types, we only have the "q" version of the function defined as
// interleaving 64-wide registers with 64-wide types in them makes no sense.
SIMD_INLINE Vec128<uint64_t> interleave_lo(const Vec128<uint64_t> a,
                                           const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(vzip1q_u64(a.raw, b.raw));
}
SIMD_INLINE Vec128<int64_t> interleave_lo(const Vec128<int64_t> a,
                                          const Vec128<int64_t> b) {
  return Vec128<int64_t>(vzip1q_s64(a.raw, b.raw));
}

SIMD_INLINE Vec128<uint64_t> interleave_hi(const Vec128<uint64_t> a,
                                           const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(vzip2q_u64(a.raw, b.raw));
}
SIMD_INLINE Vec128<int64_t> interleave_hi(const Vec128<int64_t> a,
                                          const Vec128<int64_t> b) {
  return Vec128<int64_t>(vzip2q_s64(a.raw, b.raw));
}

// Floats
SIMD_INLINE Vec128<float> interleave_lo(const Vec128<float> a,
                                        const Vec128<float> b) {
  return Vec128<float>(vzip1q_f32(a.raw, b.raw));
}
SIMD_INLINE Vec128<double> interleave_lo(const Vec128<double> a,
                                         const Vec128<double> b) {
  return Vec128<double>(vzip1q_f64(a.raw, b.raw));
}

SIMD_INLINE Vec128<float> interleave_hi(const Vec128<float> a,
                                        const Vec128<float> b) {
  return Vec128<float>(vzip2q_f32(a.raw, b.raw));
}
SIMD_INLINE Vec128<double> interleave_hi(const Vec128<double> a,
                                         const Vec128<double> b) {
  return Vec128<double>(vzip2q_s64(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_INLINE Vec128<uint16_t> zip_lo(const Vec128<uint8_t> a,
                                    const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(vzip1q_u8(a.raw, b.raw));
}
SIMD_INLINE Vec128<uint32_t> zip_lo(const Vec128<uint16_t> a,
                                    const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(vzip1q_u16(a.raw, b.raw));
}
SIMD_INLINE Vec128<uint64_t> zip_lo(const Vec128<uint32_t> a,
                                    const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(vzip1q_u32(a.raw, b.raw));
}

SIMD_INLINE Vec128<int16_t> zip_lo(const Vec128<int8_t> a,
                                   const Vec128<int8_t> b) {
  return Vec128<int16_t>(vzip1q_s8(a.raw, b.raw));
}
SIMD_INLINE Vec128<int32_t> zip_lo(const Vec128<int16_t> a,
                                   const Vec128<int16_t> b) {
  return Vec128<int32_t>(vzip1q_s16(a.raw, b.raw));
}
SIMD_INLINE Vec128<int64_t> zip_lo(const Vec128<int32_t> a,
                                   const Vec128<int32_t> b) {
  return Vec128<int64_t>(vzip1q_s32(a.raw, b.raw));
}

SIMD_INLINE Vec128<uint16_t> zip_hi(const Vec128<uint8_t> a,
                                    const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(vzip2q_u8(a.raw, b.raw));
}
SIMD_INLINE Vec128<uint32_t> zip_hi(const Vec128<uint16_t> a,
                                    const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(vzip2q_u16(a.raw, b.raw));
}
SIMD_INLINE Vec128<uint64_t> zip_hi(const Vec128<uint32_t> a,
                                    const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(vzip2q_u32(a.raw, b.raw));
}

SIMD_INLINE Vec128<int16_t> zip_hi(const Vec128<int8_t> a,
                                   const Vec128<int8_t> b) {
  return Vec128<int16_t>(vzip2q_s8(a.raw, b.raw));
}
SIMD_INLINE Vec128<int32_t> zip_hi(const Vec128<int16_t> a,
                                   const Vec128<int16_t> b) {
  return Vec128<int32_t>(vzip2q_s16(a.raw, b.raw));
}
SIMD_INLINE Vec128<int64_t> zip_hi(const Vec128<int32_t> a,
                                   const Vec128<int32_t> b) {
  return Vec128<int64_t>(vzip2q_s32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns a part with value "t".
template <typename T>
SIMD_INLINE Vec128<T, 1> set_lane(const T t) {
  return set1(Desc<T, 1>(), t);
}

// Gets the single value stored in a vector/part.
SIMD_INLINE uint8_t get_lane(const Vec128<uint8_t, 16> v) {
  return vget_lane_u8(vget_low_u8(v.raw), 0);
}
template <size_t N>
SIMD_INLINE uint8_t get_lane(const Vec128<uint8_t, N> v) {
  return vget_lane_u8(v.raw, 0);
}

SIMD_INLINE int8_t get_lane(const Vec128<int8_t, 16> v) {
  return vget_lane_s8(vget_low_s8(v.raw), 0);
}
template <size_t N>
SIMD_INLINE int8_t get_lane(const Vec128<int8_t, N> v) {
  return vget_lane_s8(v.raw, 0);
}

SIMD_INLINE uint16_t get_lane(const Vec128<uint16_t, 8> v) {
  return vget_lane_u16(vget_low_u16(v.raw), 0);
}
template <size_t N>
SIMD_INLINE uint16_t get_lane(const Vec128<uint16_t, N> v) {
  return vget_lane_u16(v.raw, 0);
}

SIMD_INLINE int16_t get_lane(const Vec128<int16_t, 8> v) {
  return vget_lane_s16(vget_low_s16(v.raw), 0);
}
template <size_t N>
SIMD_INLINE int16_t get_lane(const Vec128<int16_t, N> v) {
  return vget_lane_s16(v.raw, 0);
}

SIMD_INLINE uint32_t get_lane(const Vec128<uint32_t, 4> v) {
  return vget_lane_u32(vget_low_u32(v.raw), 0);
}
template <size_t N>
SIMD_INLINE uint32_t get_lane(const Vec128<uint32_t, N> v) {
  return vget_lane_u32(v.raw, 0);
}

SIMD_INLINE int32_t get_lane(const Vec128<int32_t, 4> v) {
  return vget_lane_s32(vget_low_s32(v.raw), 0);
}
template <size_t N>
SIMD_INLINE int32_t get_lane(const Vec128<int32_t, N> v) {
  return vget_lane_s32(v.raw, 0);
}

SIMD_INLINE uint64_t get_lane(const Vec128<uint64_t, 2> v) {
  return vget_lane_u64(vget_low_u64(v.raw), 0);
}
SIMD_INLINE uint64_t get_lane(const Vec128<uint64_t, 1> v) {
  return vget_lane_u64(v.raw, 0);
}
SIMD_INLINE int64_t get_lane(const Vec128<int64_t, 2> v) {
  return vget_lane_s64(vget_low_s64(v.raw), 0);
}
SIMD_INLINE int64_t get_lane(const Vec128<int64_t, 1> v) {
  return vget_lane_s64(v.raw, 0);
}

SIMD_INLINE float get_lane(const Vec128<float, 4> v) {
  return vget_lane_f32(vget_low_f32(v.raw), 0);
}
SIMD_INLINE float get_lane(const Vec128<float, 2> v) {
  return vget_lane_f32(v.raw, 0);
}
SIMD_INLINE float get_lane(const Vec128<float, 1> v) {
  return vget_lane_f32(v.raw, 0);
}
SIMD_INLINE double get_lane(const Vec128<double, 2> v) {
  return vget_lane_f64(vget_low_f64(v.raw), 0);
}
SIMD_INLINE double get_lane(const Vec128<double, 1> v) {
  return vget_lane_f64(v.raw, 0);
}

// Returns part of a vector (unspecified whether upper or lower).
// 64-bit result
SIMD_INLINE Vec128<uint8_t, 8> any_part(Desc<uint8_t, 8> /* tag */,
                                        const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 8>(vget_low_u8(v.raw));
}
SIMD_INLINE Vec128<uint16_t, 4> any_part(Desc<uint16_t, 4> /* tag */,
                                         const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 4>(vget_low_u16(v.raw));
}
SIMD_INLINE Vec128<uint32_t, 2> any_part(Desc<uint32_t, 2> /* tag */,
                                         const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 2>(vget_low_u32(v.raw));
}
SIMD_INLINE Vec128<uint64_t, 1> any_part(Desc<uint64_t, 1> /* tag */,
                                         const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_low_u64(v.raw));
}
SIMD_INLINE Vec128<int8_t, 8> any_part(Desc<int8_t, 8> /* tag */,
                                       const Vec128<int8_t> v) {
  return Vec128<int8_t, 8>(vget_low_s8(v.raw));
}
SIMD_INLINE Vec128<int16_t, 4> any_part(Desc<int16_t, 4> /* tag */,
                                        const Vec128<int16_t> v) {
  return Vec128<int16_t, 4>(vget_low_s16(v.raw));
}
SIMD_INLINE Vec128<int32_t, 2> any_part(Desc<int32_t, 2> /* tag */,
                                        const Vec128<int32_t> v) {
  return Vec128<int32_t, 2>(vget_low_s32(v.raw));
}
SIMD_INLINE Vec128<int64_t, 1> any_part(Desc<int64_t, 1> /* tag */,
                                        const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_low_s64(v.raw));
}
SIMD_INLINE Vec128<float, 2> any_part(Desc<float, 2> /* tag */,
                                      const Vec128<float> v) {
  return Vec128<float, 2>(vget_low_f32(v.raw));
}
SIMD_INLINE Vec128<double, 1> any_part(Desc<double, 1> /* tag */,
                                       const Vec128<double> v) {
  return Vec128<double, 1>(vget_low_f64(v.raw));
}

// 32-bit result
SIMD_INLINE Vec128<uint8_t, 4> any_part(Desc<uint8_t, 4> /* tag */,
                                        const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 4>(vget_low_u8(v.raw));
}
SIMD_INLINE Vec128<uint16_t, 2> any_part(Desc<uint16_t, 2> /* tag */,
                                         const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 2>(vget_low_u16(v.raw));
}
SIMD_INLINE Vec128<uint32_t, 1> any_part(Desc<uint32_t, 1> /* tag */,
                                         const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 1>(vget_low_u32(v.raw));
}
SIMD_INLINE Vec128<int8_t, 4> any_part(Desc<int8_t, 4> /* tag */,
                                       const Vec128<int8_t> v) {
  return Vec128<int8_t, 4>(vget_low_s8(v.raw));
}
SIMD_INLINE Vec128<int16_t, 2> any_part(Desc<int16_t, 2> /* tag */,
                                        const Vec128<int16_t> v) {
  return Vec128<int16_t, 2>(vget_low_s16(v.raw));
}
SIMD_INLINE Vec128<int32_t, 1> any_part(Desc<int32_t, 1> /* tag */,
                                        const Vec128<int32_t> v) {
  return Vec128<int32_t, 1>(vget_low_s32(v.raw));
}
SIMD_INLINE Vec128<float, 1> any_part(Desc<float, 1> /* tag */,
                                      const Vec128<float> v) {
  return Vec128<float, 1>(vget_low_f32(v.raw));
}

// 16-bit result
SIMD_INLINE Vec128<uint8_t, 2> any_part(Desc<uint8_t, 2> /* tag */,
                                        const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 2>(vget_low_u8(v.raw));
}
SIMD_INLINE Vec128<uint16_t, 1> any_part(Desc<uint16_t, 1> /* tag */,
                                         const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 1>(vget_low_u16(v.raw));
}
SIMD_INLINE Vec128<int8_t, 2> any_part(Desc<int8_t, 2> /* tag */,
                                       const Vec128<int8_t> v) {
  return Vec128<int8_t, 2>(vget_low_s8(v.raw));
}
SIMD_INLINE Vec128<int16_t, 1> any_part(Desc<int16_t, 1> /* tag */,
                                        const Vec128<int16_t> v) {
  return Vec128<int16_t, 1>(vget_low_s16(v.raw));
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_INLINE Vec128<T> broadcast_part(Full128<T> /* tag */,
                                     const Vec128<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return broadcast<kLane>(v);
}

// Returns upper/lower half of a vector.
SIMD_INLINE Vec128<uint8_t, 8> lower_half(const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 8>(vget_low_u8(v.raw));
}
SIMD_INLINE Vec128<uint16_t, 4> lower_half(const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 4>(vget_low_u16(v.raw));
}
SIMD_INLINE Vec128<uint32_t, 2> lower_half(const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 2>(vget_low_u32(v.raw));
}
SIMD_INLINE Vec128<uint64_t, 1> lower_half(const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_low_u64(v.raw));
}
SIMD_INLINE Vec128<int8_t, 8> lower_half(const Vec128<int8_t> v) {
  return Vec128<int8_t, 8>(vget_low_s8(v.raw));
}
SIMD_INLINE Vec128<int16_t, 4> lower_half(const Vec128<int16_t> v) {
  return Vec128<int16_t, 4>(vget_low_s16(v.raw));
}
SIMD_INLINE Vec128<int32_t, 2> lower_half(const Vec128<int32_t> v) {
  return Vec128<int32_t, 2>(vget_low_s32(v.raw));
}
SIMD_INLINE Vec128<int64_t, 1> lower_half(const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_low_s64(v.raw));
}
SIMD_INLINE Vec128<float, 2> lower_half(const Vec128<float> v) {
  return Vec128<float, 2>(vget_low_f32(v.raw));
}
SIMD_INLINE Vec128<double, 1> lower_half(const Vec128<double> v) {
  return Vec128<double, 1>(vget_low_f64(v.raw));
}

SIMD_INLINE Vec128<uint8_t, 8> upper_half(const Vec128<uint8_t> v) {
  return Vec128<uint8_t, 8>(vget_high_u8(v.raw));
}
SIMD_INLINE Vec128<uint16_t, 4> upper_half(const Vec128<uint16_t> v) {
  return Vec128<uint16_t, 4>(vget_high_u16(v.raw));
}
SIMD_INLINE Vec128<uint32_t, 2> upper_half(const Vec128<uint32_t> v) {
  return Vec128<uint32_t, 2>(vget_high_u32(v.raw));
}
SIMD_INLINE Vec128<uint64_t, 1> upper_half(const Vec128<uint64_t> v) {
  return Vec128<uint64_t, 1>(vget_high_u64(v.raw));
}
SIMD_INLINE Vec128<int8_t, 8> upper_half(const Vec128<int8_t> v) {
  return Vec128<int8_t, 8>(vget_high_s8(v.raw));
}
SIMD_INLINE Vec128<int16_t, 4> upper_half(const Vec128<int16_t> v) {
  return Vec128<int16_t, 4>(vget_high_s16(v.raw));
}
SIMD_INLINE Vec128<int32_t, 2> upper_half(const Vec128<int32_t> v) {
  return Vec128<int32_t, 2>(vget_high_s32(v.raw));
}
SIMD_INLINE Vec128<int64_t, 1> upper_half(const Vec128<int64_t> v) {
  return Vec128<int64_t, 1>(vget_high_s64(v.raw));
}
SIMD_INLINE Vec128<float, 2> upper_half(const Vec128<float> v) {
  return Vec128<float, 2>(vget_high_f32(v.raw));
}
SIMD_INLINE Vec128<double, 1> upper_half(const Vec128<double> v) {
  return Vec128<double, 1>(vget_high_f64(v.raw));
}

template <typename T>
SIMD_INLINE Vec128<T, 8 / sizeof(T)> get_half(Lower /* tag */, Vec128<T> v) {
  return lower_half(v);
}
template <typename T>
SIMD_INLINE Vec128<T, 8 / sizeof(T)> get_half(Upper /* tag */, Vec128<T> v) {
  return upper_half(v);
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_INLINE Vec128<T> concat_lo_lo(const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return bit_cast(Full128<T>(),
                  interleave_lo(bit_cast(d64, lo), bit_cast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_INLINE Vec128<T> concat_hi_hi(const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return bit_cast(Full128<T>(),
                  interleave_hi(bit_cast(d64, lo), bit_cast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
SIMD_INLINE Vec128<T> concat_lo_hi(const Vec128<T> hi, const Vec128<T> lo) {
  return combine_shift_right_bytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_INLINE Vec128<T> concat_hi_lo(const Vec128<T> hi, const Vec128<T> lo) {
  // TODO(janwas): more efficient implementation?
  SIMD_ALIGN const uint8_t mask[16] = {
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0};
  return if_then_else(bit_cast(Full128<T>(), load(Full128<uint8_t>(), mask)),
                      lo, hi);
}

// ------------------------------ Odd/even lanes

template <typename T>
SIMD_INLINE Vec128<T> odd_even(const Vec128<T> a, const Vec128<T> b) {
  const Full128<uint8_t> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {
      ((0 / sizeof(T)) & 1) ? 0 : 0xFF,  ((1 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((2 / sizeof(T)) & 1) ? 0 : 0xFF,  ((3 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((4 / sizeof(T)) & 1) ? 0 : 0xFF,  ((5 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((6 / sizeof(T)) & 1) ? 0 : 0xFF,  ((7 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((8 / sizeof(T)) & 1) ? 0 : 0xFF,  ((9 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((10 / sizeof(T)) & 1) ? 0 : 0xFF, ((11 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((12 / sizeof(T)) & 1) ? 0 : 0xFF, ((13 / sizeof(T)) & 1) ? 0 : 0xFF,
      ((14 / sizeof(T)) & 1) ? 0 : 0xFF, ((15 / sizeof(T)) & 1) ? 0 : 0xFF,
  };
  return if_then_else(cast_u8_to(Full128<T>(), load(d8, mask)), b, a);
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_INLINE uint64_t movemask(const Vec128<uint8_t> v) {
  static constexpr uint8x16_t kCollapseMask = {
      1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
  };
  int8x16_t signed_v = vreinterpretq_s8_u8(v.raw);
  int8x16_t signed_mask = vshrq_n_s8(signed_v, 7);
  uint8x16_t values = vreinterpretq_u8_s8(signed_mask) & kCollapseMask;

  uint8x8_t c0 = vget_low_u8(vpaddq_u8(values, values));
  uint8x8_t c1 = vpadd_u8(c0, c0);
  uint8x8_t c2 = vpadd_u8(c1, c1);

  return vreinterpret_u16_u8(c2)[0];
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_INLINE uint64_t movemask(const Vec128<float> v) {
  static constexpr uint32x4_t kCollapseMask = {1, 2, 4, 8};
  int32x4_t signed_v = vreinterpretq_s32_f32(v.raw);
  int32x4_t signed_mask = vshrq_n_s32(signed_v, 31);
  uint32x4_t values = vreinterpretq_u32_s32(signed_mask) & kCollapseMask;
  return vaddvq_u32(values);
}
SIMD_INLINE uint64_t movemask(const Vec128<double> v) {
  static constexpr uint64x2_t kCollapseMask = {1, 2};
  int64x2_t signed_v = vreinterpretq_s64_f64(v.raw);
  int64x2_t signed_mask = vshrq_n_s64(signed_v, 63);
  uint64x2_t values = vreinterpretq_u64_s64(signed_mask) & kCollapseMask;
  return vaddvq_u64(values);
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero.
template <typename T>
SIMD_INLINE bool all_zero(const Vec128<T> v) {
  const auto v64 = bit_cast(Full128<uint64_t>(), v);
  uint32x2_t a = vqmovn_u64(v64.raw);
  return vreinterpret_u64_u32(a)[0] == 0;
}

// ------------------------------ minpos

// Returns index and min value in lanes 1 and 0.
SIMD_INLINE Vec128<uint16_t> minpos(const Vec128<uint16_t> v) {
  // v = ABCDEFGH
  uint16x8_t mask = {0, 1, 2, 3, 4, 5, 6, 7};
  uint32x4_t a = vreinterpretq_u32_u16(vzip1q_u16(mask, v.raw));  // 0A1B2C3D
  uint32x4_t b = vreinterpretq_u32_u16(vzip2q_u16(mask, v.raw));  // 4E5F6G7H
  a = vminq_u32(a, b);
#if defined(__aarch64__)
  uint32_t pos_min = vminvq_u32(a);
  a = set1(Full128<uint32_t>(), pos_min).raw;
#else
  a = vpminq_u32(a, a);
  a = vpminq_u32(a, a);
#endif
  // At the end the minimum position is in the first 16-bit lane and the minimum
  // value in the second 16-bit lane; but we need them switched. Since every
  // 32-bit lane contains the same value we can just right-shift a 64-bit lane
  // 16-bits.
  return Vec128<uint16_t>(
      vreinterpretq_u16_u64(vshrq_n_u64(vreinterpretq_u64_u32(a), 16)));
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_INLINE Vec128<uint64_t> sums_of_u8x8(const Vec128<uint8_t> v) {
  uint16x8_t a = vpaddlq_u8(v.raw);
  uint32x4_t b = vpaddlq_u16(a);
  return Vec128<uint64_t>(vpaddlq_u32(b));
}

// Supported for 32b and 64b vector types. Returns the sum in each lane.
SIMD_INLINE Vec128<uint32_t> sum_of_lanes(const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(vdupq_n_u32(vaddvq_u32(v.raw)));
}
SIMD_INLINE Vec128<int32_t> sum_of_lanes(const Vec128<int32_t> v) {
  return Vec128<int32_t>(vdupq_n_s32(vaddvq_s32(v.raw)));
}
SIMD_INLINE Vec128<float> sum_of_lanes(const Vec128<float> v) {
  return Vec128<float>(vdupq_n_f32(vaddvq_f32(v.raw)));
}
SIMD_INLINE Vec128<uint64_t> sum_of_lanes(const Vec128<uint64_t> v) {
  return Vec128<uint64_t>(vdupq_n_u64(vaddvq_u64(v.raw)));
}
SIMD_INLINE Vec128<int64_t> sum_of_lanes(const Vec128<int64_t> v) {
  return Vec128<int64_t>(vdupq_n_s64(vaddvq_s64(v.raw)));
}
SIMD_INLINE Vec128<double> sum_of_lanes(const Vec128<double> v) {
  return Vec128<double>(vdupq_n_f64(vaddvq_f64(v.raw)));
}

// ------------------------------ MPSADBW

template <int idx_ref>
SIMD_INLINE Vec128<uint16_t> mpsadbw(const Vec128<uint8_t> window,
                                     const Vec128<uint8_t> ref) {
  static_assert(idx_ref < 4, "a_offset must be 0");
  const Full128<uint8_t> d8;
  const Full128<uint16_t> d16;

  // Let S(w,r) = |window[w] - ref[r + idx_ref * 4]|.
  // The result of MPSADBW is defined as 16-bit values:
  //   MPSADBW[i=0..7] := sum{q=0..3} S(i+q, q):
  // Note that only the 4 uint8_t values from ref starting at "idx_ref * 4" are
  // ever used. Example:
  // 0123   <- the four w indices of the first (not yet) shifted window
  //  1234
  //   ..   789a  <- w indices of last window (bytes 11..15 are unused)
  // MPSADBW[7] = S(7,0) + S(8,1) + S(9,2) + S(10,3).

  // To compute each value of the result we need to sum four 8-bit S(w, r)
  // values. After adding two 8-bit values we would need to store 16-bits and
  // then keep adding to these 16-bit values. The strategy here is to add two
  // 8-bit S(w, r) results (like S(i, 0) and S(i+1, 1)) into a 16-bit, and then
  // the other two S(w, r) results (S(i+2, 2) and S(i+3, 3)) into two separated
  // 16-bit values, and then add these two values together.
  // For this we will use vaddl_u8 and vaddl_high_u8, which add two 16 uint8_t
  // vectors vertically and store the result in 16 uint16_ts. This requires two
  // vectors to store the result (low and high) generated by vaddl_u8 and
  // vaddl_high_u8 respectively. Finally, we use vpaddq_u16 to add the uint16_t
  // pairwise horizontally into uint16_t results.

  // For this to work we need to compute s0 and s1 as the absolute difference
  // between the specific window and ref values, which requires to arrange
  // the window and ref bytes as follows:

  // For ref, the first vector will have ref[idx_ref * 4 + 0] and
  // ref[idx_ref * 4 + 2] repeated 8 times, and the second vector the other two
  // values also repeated 8 times:
  // ref01 = 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
  // ref23 = 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3
  const uint8x16_t ref01 =
      bit_cast(d8, broadcast<2 * idx_ref>(bit_cast(d16, ref))).raw;
  const uint8x16_t ref23 =
      bit_cast(d8, broadcast<2 * idx_ref + 1>(bit_cast(d16, ref))).raw;

  // Then, for the window we need the following scheme, where one half of the
  // 0123 group is in the first window vector (w0) and the other half in w1.
  // w0: 0 1 | 1 2 | 2 3 | 3 4 | 4 5 | 5 6 | 6 7 | 7 8
  // w1: 2 3 | 3 4 | 4 5 | 5 6 | 6 7 | 7 8 | 8 9 | 9 10
  SIMD_ALIGN static const uint8_t shuf0[16] = {0, 1, 1, 2, 2, 3, 3, 4,
                                               4, 5, 5, 6, 6, 7, 7, 8};
  SIMD_ALIGN static const uint8_t shuf1[16] = {2, 3, 3, 4, 4, 5, 5, 6,
                                               6, 7, 7, 8, 8, 9, 9, 10};
  const uint8x16_t w0 = table_lookup_bytes(window, load(d8, shuf0)).raw;
  const uint8x16_t w1 = table_lookup_bytes(window, load(d8, shuf1)).raw;

  // We compute the absolute different of |w1:w0 - ref01:ref23| as 32 uint8_t
  // values.
  const uint8x16_t s0 = vabdq_u8(w0, ref01);
  const uint8x16_t s1 = vabdq_u8(w1, ref23);

  // ph:pl contains 16-bit values with the sum of the pairs of absolute
  // differences in s1 and s0. Note that we need two vectors for the result
  // because we are storing 16-bit partial sums.
  const uint16x8_t pl = vaddl_u8(vget_low_u8(s0), vget_low_u8(s1));
  const uint16x8_t ph = vaddl_high_u8(s0, s1);

  return Vec128<uint16_t>(vpaddq_u16(pl, ph));
}

// TODO(user): wrappers for all intrinsics (in neon namespace).
}  // namespace ext

}  // namespace jxl

#undef SIMD_NEON_BUILD_ARG_1
#undef SIMD_NEON_BUILD_ARG_2
#undef SIMD_NEON_BUILD_ARG_3
#undef SIMD_NEON_BUILD_PARAM_1
#undef SIMD_NEON_BUILD_PARAM_2
#undef SIMD_NEON_BUILD_PARAM_3
#undef SIMD_NEON_BUILD_RET_1
#undef SIMD_NEON_BUILD_RET_2
#undef SIMD_NEON_BUILD_RET_3
#undef SIMD_NEON_BUILD_TPL_1
#undef SIMD_NEON_BUILD_TPL_2
#undef SIMD_NEON_BUILD_TPL_3
#undef SIMD_NEON_DEF_FUNCTION
#undef SIMD_NEON_DEF_FUNCTION_ALL_FLOATS
#undef SIMD_NEON_DEF_FUNCTION_ALL_TYPES
#undef SIMD_NEON_DEF_FUNCTION_INT_8
#undef SIMD_NEON_DEF_FUNCTION_INT_16
#undef SIMD_NEON_DEF_FUNCTION_INT_32
#undef SIMD_NEON_DEF_FUNCTION_INT_8_16_32
#undef SIMD_NEON_DEF_FUNCTION_INTS
#undef SIMD_NEON_DEF_FUNCTION_INTS_UINTS
#undef SIMD_NEON_DEF_FUNCTION_TPL
#undef SIMD_NEON_DEF_FUNCTION_UINT_8
#undef SIMD_NEON_DEF_FUNCTION_UINT_16
#undef SIMD_NEON_DEF_FUNCTION_UINT_32
#undef SIMD_NEON_DEF_FUNCTION_UINT_8_16_32
#undef SIMD_NEON_DEF_FUNCTION_UINTS
#undef SIMD_NEON_EVAL

#endif  // HIGHWAY_ARM64_NEON_H_
