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

#ifndef HWY_WASM_H_
#define HWY_WASM_H_

// 128-bit WASM vectors and operations.

#include <wasm_simd128.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
#include <stdio.h>
#endif

#include "hwy/shared.h"

#define HWY_ATTR_WASM HWY_TARGET_ATTR("simd128")
namespace hwy {

template <typename T>
struct Raw128 {
  using type = __v128_u;
};
template <>
struct Raw128<float> {
  using type = __f32x4;
};

template <typename T>
using Full128 = Desc<T, 16 / sizeof(T)>;

template <typename T, size_t N = 16 / sizeof(T)>
class Vec128 {
  using Raw = typename Raw128<T>::type;

 public:
  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  HWY_ATTR_WASM HWY_INLINE Vec128& operator*=(const Vec128 other) {
    return *this = (*this * other);
  }
  HWY_ATTR_WASM HWY_INLINE Vec128& operator/=(const Vec128 other) {
    return *this = (*this / other);
  }
  HWY_ATTR_WASM HWY_INLINE Vec128& operator+=(const Vec128 other) {
    return *this = (*this + other);
  }
  HWY_ATTR_WASM HWY_INLINE Vec128& operator-=(const Vec128 other) {
    return *this = (*this - other);
  }
  HWY_ATTR_WASM HWY_INLINE Vec128& operator&=(const Vec128 other) {
    return *this = (*this & other);
  }
  HWY_ATTR_WASM HWY_INLINE Vec128& operator|=(const Vec128 other) {
    return *this = (*this | other);
  }
  HWY_ATTR_WASM HWY_INLINE Vec128& operator^=(const Vec128 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// Integer: FF..FF or 0. Float: MSB, all other bits undefined - see README.
template <typename T, size_t N = 16 / sizeof(T)>
class Mask128 {
  using Raw = typename Raw128<T>::type;

 public:
  Raw raw;
};

// ------------------------------ Cast

HWY_ATTR_WASM HWY_INLINE __v128_u BitCastToInteger(__v128_u v) { return v; }
HWY_ATTR_WASM HWY_INLINE __v128_u BitCastToInteger(__f32x4 v) {
  return static_cast<__v128_u>(v);
}
HWY_ATTR_WASM HWY_INLINE __v128_u BitCastToInteger(__f64x2 v) {
  return static_cast<__v128_u>(v);
}

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N * sizeof(T)> cast_to_u8(
    Vec128<T, N> v) {
  return Vec128<uint8_t, N * sizeof(T)>{BitCastToInteger(v.raw)};
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromInteger128 {
  HWY_ATTR_WASM HWY_INLINE __v128_u operator()(__v128_u v) { return v; }
};
template <>
struct BitCastFromInteger128<float> {
  HWY_ATTR_WASM HWY_INLINE __f32x4 operator()(__v128_u v) {
    return static_cast<__f32x4>(v);
  }
};

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> cast_u8_to(
    Desc<T, N> /* tag */, Vec128<uint8_t, N * sizeof(T)> v) {
  return Vec128<T, N>{BitCastFromInteger128<T>()(v.raw)};
}

template <typename T, size_t N, typename FromT>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> BitCast(
    Desc<T, N> d, Vec128<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  return cast_u8_to(d, cast_to_u8(v));
}

// ------------------------------ Set

// Returns an all-zero vector/part.
template <typename T, size_t N, HWY_IF128(T, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> Zero(Desc<T, N> /* tag */) {
  return Vec128<T, N>{wasm_i32x4_splat(0)};
}
template <size_t N, HWY_IF128(float, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Zero(Desc<float, N> /* tag */) {
  return Vec128<float, N>{wasm_f32x4_splat(0.0f)};
}

// Returns a vector/part with all lanes set to "t".
template <size_t N, HWY_IF128(uint8_t, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> Set(Desc<uint8_t, N> /* tag */,
                                                const uint8_t t) {
  return Vec128<uint8_t, N>{wasm_i8x16_splat(t)};
}
template <size_t N, HWY_IF128(uint16_t, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> Set(Desc<uint16_t, N> /* tag */,
                                                 const uint16_t t) {
  return Vec128<uint16_t, N>{wasm_i16x8_splat(t)};
}
template <size_t N, HWY_IF128(uint32_t, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> Set(Desc<uint32_t, N> /* tag */,
                                                 const uint32_t t) {
  return Vec128<uint32_t, N>{wasm_i32x4_splat(t)};
}

template <size_t N, HWY_IF128(int8_t, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> Set(Desc<int8_t, N> /* tag */,
                                               const int8_t t) {
  return Vec128<int8_t, N>{wasm_i8x16_splat(t)};
}
template <size_t N, HWY_IF128(int16_t, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> Set(Desc<int16_t, N> /* tag */,
                                                const int16_t t) {
  return Vec128<int16_t, N>{wasm_i16x8_splat(t)};
}
template <size_t N, HWY_IF128(int32_t, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> Set(Desc<int32_t, N> /* tag */,
                                                const int32_t t) {
  return Vec128<int32_t, N>{wasm_i32x4_splat(t)};
}

template <size_t N, HWY_IF128(float, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Set(Desc<float, N> /* tag */,
                                              const float t) {
  return Vec128<float, N>{wasm_f32x4_splat(t)};
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2, HWY_IF128(T, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> Iota(Desc<T, N> d, const T2 first) {
  alignas(16) T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = first + i;
  }
  return Load(d, lanes);
}

HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T, size_t N, HWY_IF128(T, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> Undefined(Desc<T, N> /* tag */) {
  __v128_u raw;
  return Vec128<T, N>{raw};
}
template <size_t N, HWY_IF128(float, N)>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Undefined(Desc<float, N> /* tag */) {
  __f32x4 raw;
  return Vec128<float, N>{raw};
}

HWY_DIAGNOSTICS(pop)

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> operator+(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>{wasm_i8x16_add(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> operator+(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>{wasm_i16x8_add(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> operator+(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>{wasm_i32x4_add(a.raw, b.raw)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> operator+(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>{wasm_i8x16_add(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> operator+(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>{wasm_i16x8_add(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> operator+(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>{wasm_i32x4_add(a.raw, b.raw)};
}

// Float
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> operator+(const Vec128<float, N> a,
                                                    const Vec128<float, N> b) {
  return Vec128<float, N>{wasm_f32x4_add(a.raw, b.raw)};
}

// ------------------------------ Subtraction

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> operator-(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>{wasm_i8x16_sub(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> operator-(Vec128<uint16_t, N> a,
                                                       Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>{wasm_i16x8_sub(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> operator-(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>{wasm_i32x4_sub(a.raw, b.raw)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> operator-(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>{wasm_i8x16_sub(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> operator-(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>{wasm_i16x8_sub(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> operator-(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>{wasm_i32x4_sub(a.raw, b.raw)};
}

// Float
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> operator-(const Vec128<float, N> a,
                                                    const Vec128<float, N> b) {
  return Vec128<float, N>{wasm_f32x4_sub(a.raw, b.raw)};
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> SaturatedAdd(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>{wasm_u8x16_add_saturate(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> SaturatedAdd(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>{wasm_u16x8_add_saturate(a.raw, b.raw)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> SaturatedAdd(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>{wasm_i8x16_add_saturate(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> SaturatedAdd(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>{wasm_i16x8_add_saturate(a.raw, b.raw)};
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> SaturatedSub(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>{wasm_u8x16_sub_saturate(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> SaturatedSub(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>{wasm_u16x8_sub_saturate(a.raw, b.raw)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> SaturatedSub(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>{wasm_i8x16_sub_saturate(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> SaturatedSub(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>{wasm_i16x8_sub_saturate(a.raw, b.raw)};
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> AverageRound(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  // TODO(eustas): declared in JS spec, but not in include file.
  alignas(16) uint8_t a_lanes[16];
  alignas(16) uint8_t b_lanes[16];
  alignas(16) uint8_t output[16];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 16; ++i) {
    output[i] = (a_lanes[i] + b_lanes[i] + 1) >> 1;
  }
  return Vec128<uint8_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> AverageRound(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  // TODO(eustas): declared in JS spec, but not in include file.
  alignas(16) uint16_t a_lanes[8];
  alignas(16) uint16_t b_lanes[8];
  alignas(16) uint16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    output[i] = (a_lanes[i] + b_lanes[i] + 1) >> 1;
  }
  return Vec128<uint16_t, N>{wasm_v128_load(output)};
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> Abs(const Vec128<int8_t, N> v) {
  // TODO(eustas): return unsigned_min(v, ~v + 1)?
  alignas(16) int8_t input[16];
  alignas(16) int8_t output[16];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 16; ++i) {
    output[i] = std::abs(input[i]);
  }
  // what should happen to -128?
  return Vec128<int8_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> Abs(const Vec128<int16_t, N> v) {
  // TODO(eustas): return unsigned_min(v, ~v + 1)?
  alignas(16) int16_t input[8];
  alignas(16) int16_t output[8];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 8; ++i) {
    output[i] = std::abs(input[i]);
  }
  return Vec128<int16_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> Abs(const Vec128<int32_t, N> v) {
  // TODO(eustas): return unsigned_min(v, ~v + 1)?
  alignas(16) int32_t input[4];
  alignas(16) int32_t output[4];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::abs(input[i]);
  }
  // what should happen to -128?
  return Vec128<int32_t, N>{wasm_v128_load(output)};
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> ShiftLeft(
    const Vec128<uint16_t, N> v) {
  return Vec128<uint16_t, N>{wasm_i16x8_shl(v.raw, kBits)};
}
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> ShiftRight(
    const Vec128<uint16_t, N> v) {
  return Vec128<uint16_t, N>{wasm_u16x8_shr(v.raw, kBits)};
}
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> ShiftLeft(
    const Vec128<uint32_t, N> v) {
  return Vec128<uint32_t, N>{wasm_i32x4_shl(v.raw, kBits)};
}
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> ShiftRight(
    const Vec128<uint32_t, N> v) {
  return Vec128<uint32_t, N>{wasm_u32x4_shr(v.raw, kBits)};
}

// Signed
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> ShiftLeft(
    const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>{wasm_i16x8_shl(v.raw, kBits)};
}
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> ShiftRight(
    const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>{wasm_i16x8_shr(v.raw, kBits)};
}
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> ShiftLeft(
    const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>{wasm_i32x4_shl(v.raw, kBits)};
}
template <int kBits, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> ShiftRight(
    const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>{wasm_i32x4_shr(v.raw, kBits)};
}

// ------------------------------ Shift lanes by same variable #bits

// Unsigned (no u8)
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> ShiftLeftSame(
    const Vec128<uint16_t, N> v, const int bits) {
  return Vec128<uint16_t, N>{wasm_i16x8_shl(v.raw, bits)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> ShiftRightSame(
    const Vec128<uint16_t, N> v, const int bits) {
  return Vec128<uint16_t, N>{wasm_u16x8_shr(v.raw, bits)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> ShiftLeftSame(
    const Vec128<uint32_t, N> v, const int bits) {
  return Vec128<uint32_t, N>{wasm_i32x4_shl(v.raw, bits)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> ShiftRightSame(
    const Vec128<uint32_t, N> v, const int bits) {
  return Vec128<uint32_t, N>{wasm_u32x4_shr(v.raw, bits)};
}

// Signed (no i8)
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> ShiftLeftSame(
    const Vec128<int16_t, N> v, const int bits) {
  return Vec128<int16_t, N>{wasm_i16x8_shl(v.raw, bits)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> ShiftRightSame(
    const Vec128<int16_t, N> v, const int bits) {
  return Vec128<int16_t, N>{wasm_i16x8_shr(v.raw, bits)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> ShiftLeftSame(
    const Vec128<int32_t, N> v, const int bits) {
  return Vec128<int32_t, N>{wasm_i32x4_shl(v.raw, bits)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> ShiftRightSame(
    const Vec128<int32_t, N> v, const int bits) {
  return Vec128<int32_t, N>{wasm_i32x4_shr(v.raw, bits)};
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsupported.

// ------------------------------ Minimum

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> Min(const Vec128<uint8_t, N> a,
                                                const Vec128<uint8_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint8_t a_lanes[16];
  alignas(16) uint8_t b_lanes[16];
  alignas(16) uint8_t output[16];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 16; ++i) {
    output[i] = std::min(a_lanes[i], b_lanes[i]);
  }
  return Vec128<uint8_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> Min(const Vec128<uint16_t, N> a,
                                                 const Vec128<uint16_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint16_t a_lanes[8];
  alignas(16) uint16_t b_lanes[8];
  alignas(16) uint16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    output[i] = std::min(a_lanes[i], b_lanes[i]);
  }
  return Vec128<uint16_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> Min(const Vec128<uint32_t, N> a,
                                                 const Vec128<uint32_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint32_t a_lanes[4];
  alignas(16) uint32_t b_lanes[4];
  alignas(16) uint32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::min(a_lanes[i], b_lanes[i]);
  }
  return Vec128<uint32_t, N>{wasm_v128_load(output)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> Min(const Vec128<int8_t, N> a,
                                               const Vec128<int8_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int8_t a_lanes[16];
  alignas(16) int8_t b_lanes[16];
  alignas(16) int8_t output[16];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 16; ++i) {
    output[i] = std::min(a_lanes[i], b_lanes[i]);
  }
  return Vec128<int8_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> Min(const Vec128<int16_t, N> a,
                                                const Vec128<int16_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int16_t a_lanes[8];
  alignas(16) int16_t b_lanes[8];
  alignas(16) int16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    output[i] = std::min(a_lanes[i], b_lanes[i]);
  }
  return Vec128<int16_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> Min(const Vec128<int32_t, N> a,
                                                const Vec128<int32_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int32_t a_lanes[4];
  alignas(16) int32_t b_lanes[4];
  alignas(16) int32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::min(a_lanes[i], b_lanes[i]);
  }
  return Vec128<int32_t, N>{wasm_v128_load(output)};
}

// Float
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Min(const Vec128<float, N> a,
                                              const Vec128<float, N> b) {
  return Vec128<float, N>{__builtin_wasm_min_f32x4(a.raw, b.raw)};
}

// ------------------------------ Maximum

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> Max(const Vec128<uint8_t, N> a,
                                                const Vec128<uint8_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint8_t a_lanes[16];
  alignas(16) uint8_t b_lanes[16];
  alignas(16) uint8_t output[16];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 16; ++i) {
    output[i] = std::max(a_lanes[i], b_lanes[i]);
  }
  return Vec128<uint8_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> Max(const Vec128<uint16_t, N> a,
                                                 const Vec128<uint16_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint16_t a_lanes[8];
  alignas(16) uint16_t b_lanes[8];
  alignas(16) uint16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    output[i] = std::max(a_lanes[i], b_lanes[i]);
  }
  return Vec128<uint16_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> Max(const Vec128<uint32_t, N> a,
                                                 const Vec128<uint32_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint32_t a_lanes[4];
  alignas(16) uint32_t b_lanes[4];
  alignas(16) uint32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::max(a_lanes[i], b_lanes[i]);
  }
  return Vec128<uint32_t, N>{wasm_v128_load(output)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> Max(const Vec128<int8_t, N> a,
                                               const Vec128<int8_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int8_t a_lanes[16];
  alignas(16) int8_t b_lanes[16];
  alignas(16) int8_t output[16];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 16; ++i) {
    output[i] = std::max(a_lanes[i], b_lanes[i]);
  }
  return Vec128<int8_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> Max(const Vec128<int16_t, N> a,
                                                const Vec128<int16_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int16_t a_lanes[8];
  alignas(16) int16_t b_lanes[8];
  alignas(16) int16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    output[i] = std::max(a_lanes[i], b_lanes[i]);
  }
  return Vec128<int16_t, N>{wasm_v128_load(output)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> Max(const Vec128<int32_t, N> a,
                                                const Vec128<int32_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int32_t a_lanes[4];
  alignas(16) int32_t b_lanes[4];
  alignas(16) int32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::max(a_lanes[i], b_lanes[i]);
  }
  return Vec128<int32_t, N>{wasm_v128_load(output)};
}

// Float
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Max(const Vec128<float, N> a,
                                              const Vec128<float, N> b) {
  return Vec128<float, N>{__builtin_wasm_max_f32x4(a.raw, b.raw)};
}

// Returns the closest value to v within [lo, hi].
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> Clamp(const Vec128<T, N> v,
                                            const Vec128<T, N> lo,
                                            const Vec128<T, N> hi) {
  return Min(Max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> operator*(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>{wasm_i16x8_mul(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t, N> operator*(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>{wasm_i32x4_mul(a.raw, b.raw)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> operator*(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>{wasm_i16x8_mul(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> operator*(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>{wasm_i32x4_mul(a.raw, b.raw)};
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> MulHigh(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint16_t a_lanes[8];
  alignas(16) uint16_t b_lanes[8];
  alignas(16) uint16_t c_lanes[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    uint32_t ab = static_cast<uint32_t>(a_lanes[i]) * b_lanes[i];
    c_lanes[i] = static_cast<uint16_t>(ab >> 16);
  }
  return Vec128<uint16_t>{wasm_v128_load(c_lanes)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> MulHigh(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int16_t a_lanes[8];
  alignas(16) int16_t b_lanes[8];
  alignas(16) int16_t c_lanes[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 8; ++i) {
    int32_t ab = static_cast<int32_t>(a_lanes[i]) * b_lanes[i];
    c_lanes[i] = static_cast<int16_t>(ab >> 16);
  }
  return Vec128<int16_t>{wasm_v128_load(c_lanes)};
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
HWY_ATTR_WASM HWY_INLINE Vec128<int64_t> MulEven(const Vec128<int32_t> a,
                                                 const Vec128<int32_t> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) int32_t a_lanes[4];
  alignas(16) int32_t b_lanes[4];
  alignas(16) int64_t c_lanes[2];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 2; ++i) {
    c_lanes[i] = static_cast<int64_t>(a_lanes[2 * i]) * b_lanes[2 * i];
  }
  return Vec128<int64_t>{wasm_v128_load(c_lanes)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint64_t> MulEven(const Vec128<uint32_t> a,
                                                  const Vec128<uint32_t> b) {
  // TODO(eustas): replace, when implemented in WASM.
  alignas(16) uint32_t a_lanes[4];
  alignas(16) uint32_t b_lanes[4];
  alignas(16) uint64_t c_lanes[2];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 2; ++i) {
    c_lanes[i] = static_cast<uint64_t>(a_lanes[2 * i]) * b_lanes[2 * i];
  }
  return Vec128<uint64_t>{wasm_v128_load(c_lanes)};
}

// ------------------------------ Floating-point negate

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Neg(const Vec128<float, N> v) {
  const Desc<float, N> df;
  const Desc<uint32_t, N> du;
  const auto sign = BitCast(df, Set(du, 0x80000000u));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> operator*(Vec128<float, N> a,
                                                    Vec128<float, N> b) {
  return Vec128<float, N>{wasm_f32x4_mul(a.raw, b.raw)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> operator/(const Vec128<float, N> a,
                                                    const Vec128<float, N> b) {
  return Vec128<float, N>{wasm_f32x4_div(a.raw, b.raw)};
}

// Approximate reciprocal
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> ApproximateReciprocal(
    const Vec128<float, N> v) {
  // TODO(eustas): replace, when implemented in WASM.
  const Vec128<float, N> one = Vec128<float, N>{wasm_f32x4_splat(1.0f)};
  return one / v;
}

namespace ext {
// Absolute value of difference.
HWY_ATTR_WASM HWY_INLINE Vec128<float> AbsDiff(const Vec128<float> a,
                                               const Vec128<float> b) {
  const auto mask =
      BitCast(Full128<float>(), Set(Full128<uint32_t>(), 0x7FFFFFFFu));
  return Vec128<float>{wasm_v128_and(mask.raw, (a - b).raw)};
}
}  // namespace ext

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> MulAdd(const Vec128<float, N> mul,
                                                 const Vec128<float, N> x,
                                                 const Vec128<float, N> add) {
  // TODO(eustas): replace, when implemented in WASM.
  // TODO(eustas): is it wasm_f32x4_qfma?
  return mul * x + add;
}

// Returns add - mul * x
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> NegMulAdd(
    const Vec128<float, N> mul, const Vec128<float, N> x,
    const Vec128<float, N> add) {
  // TODO(eustas): replace, when implemented in WASM.
  return add - mul * x;
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> MulSub(const Vec128<float, N> mul,
                                                 const Vec128<float, N> x,
                                                 const Vec128<float, N> sub) {
  // TODO(eustas): replace, when implemented in WASM.
  // TODO(eustas): is it wasm_f32x4_qfms?
  return mul * x - sub;
}

// Returns -mul * x - sub
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> NegMulSub(
    const Vec128<float, N> mul, const Vec128<float, N> x,
    const Vec128<float, N> sub) {
  // TODO(eustas): replace, when implemented in WASM.
  return Neg(mul) * x - sub;
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Sqrt(const Vec128<float, N> v) {
  alignas(16) float input[4];
  alignas(16) float output[4];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::sqrtf(input[i]);
  }
  return Vec128<float>{wasm_v128_load(output)};
  // TODO(eustas): requires target feature "unimplemented-simd128"
  // return Vec128<float, N>{wasm_f32x4_sqrt(v.raw)};
}

// Approximate reciprocal square root
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> ApproximateReciprocalSqrt(
    const Vec128<float, N> v) {
  // TODO(eustas): find cheaper a way to calculate this.
  const Vec128<float, N> one = Vec128<float, N>{wasm_f32x4_splat(1.0f)};
  return one / Sqrt(v);
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, ties to even
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Round(const Vec128<float, N> v) {
  // TODO(eustas): workaround does not work - wasm_i32x4_trunc_saturate_f32x4
  //               limits feasible input to +-2^31.
  // TODO(eustas): how to do "ties to even"?
  // TODO(eustas): 8 ops; isn't it cheaper to store/convert/load?
  // const __f32x4 c00 = wasm_f32x4_splat(0.0f);
  // const __f32x4 corr = wasm_f32x4_convert_i32x4(wasm_f32x4_le(v.raw, c00));
  // const __f32x4 c05 = wasm_f32x4_splat(0.5f);
  // +0.5 for non-negative lane, -0.5 for other.
  // const __f32x4 delta = wasm_f32x4_add(c05, corr);
  // Shift input by 0.5 away from 0.
  // const __f32x4 fixed = wasm_f32x4_add(v.raw, delta);
  // const __i32x4 result = wasm_i32x4_trunc_saturate_f32x4(fixed);
  // return Vec128<float, N>{wasm_f32x4_convert_i32x4(result)};
  alignas(16) float input[4];
  alignas(16) float output[4];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::round(input[i]);
  }
  return Vec128<float>{wasm_v128_load(output)};
}

// Toward zero, aka truncate
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Trunc(const Vec128<float, N> v) {
  // TODO(eustas): impossible
  // const __i32x4 result = wasm_i32x4_trunc_saturate_f32x4(v.raw);
  // return Vec128<float, N>{wasm_f32x4_convert_i32x4(result)};
  alignas(16) float input[4];
  alignas(16) float output[4];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::trunc(input[i]);
  }
  return Vec128<float>{wasm_v128_load(output)};
}

// Toward +infinity, aka ceiling
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Ceil(const Vec128<float, N> v) {
  // TODO(eustas): impossible
  // const __i32x4 truncated = wasm_i32x4_trunc_saturate_f32x4(v.raw);
  // const __f32x4 draft = wasm_f32x4_convert_i32x4(truncated);
  // -1 for positive with fractional part, 0 for negative and integer.
  // const __i32x4 delta = wasm_f32x4_gt(v.raw, draft);
  // const __f32x4 corr = wasm_f32x4_convert_i32x4(delta);
  // return Vec128<float, N>{wasm_f32x4_sub(draft, corr)};
  alignas(16) float input[4];
  alignas(16) float output[4];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::ceil(input[i]);
  }
  return Vec128<float>{wasm_v128_load(output)};
}

// Toward -infinity, aka floor
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> Floor(const Vec128<float, N> v) {
  // TODO(eustas): impossible
  // const __i32x4 truncated = wasm_i32x4_trunc_saturate_f32x4(v.raw);
  // const __f32x4 draft = wasm_f32x4_convert_i32x4(truncated);
  // -1 for negative with fractional part, 0 for positive and integer.
  // const __i32x4 delta = wasm_f32x4_lt(v.raw, draft);
  // const __f32x4 corr = wasm_f32x4_convert_i32x4(delta);
  // return Vec128<float, N>{wasm_f32x4_add(draft, corr)};
  alignas(16) float input[4];
  alignas(16) float output[4];
  wasm_v128_store(input, v.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[i] = std::floor(input[i]);
  }
  return Vec128<float>{wasm_v128_load(output)};
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<uint8_t, N> operator==(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Mask128<uint8_t, N>{wasm_i8x16_eq(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<uint16_t, N> operator==(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Mask128<uint16_t, N>{wasm_i16x8_eq(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<uint32_t, N> operator==(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Mask128<uint32_t, N>{wasm_i32x4_eq(a.raw, b.raw)};
}

// Signed
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int8_t, N> operator==(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Mask128<int8_t, N>{wasm_i8x16_eq(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int16_t, N> operator==(Vec128<int16_t, N> a,
                                                        Vec128<int16_t, N> b) {
  return Mask128<int16_t, N>{wasm_i16x8_eq(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int32_t, N> operator==(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Mask128<int32_t, N>{wasm_i32x4_eq(a.raw, b.raw)};
}

// Float
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<float, N> operator==(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Mask128<float, N>{wasm_f32x4_eq(a.raw, b.raw)};
}

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<T, N> TestBit(Vec128<T, N> v,
                                               Vec128<T, N> bit) {
  static_assert(!IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

// ------------------------------ Strict inequality

// Signed/float <
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int8_t, N> operator<(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Mask128<int8_t, N>{wasm_i8x16_lt(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int16_t, N> operator<(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Mask128<int16_t, N>{wasm_i16x8_lt(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int32_t, N> operator<(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Mask128<int32_t, N>{wasm_i32x4_lt(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<float, N> operator<(const Vec128<float, N> a,
                                                     const Vec128<float, N> b) {
  return Mask128<float, N>{wasm_f32x4_lt(a.raw, b.raw)};
}

// Signed/float >
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int8_t, N> operator>(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Mask128<int8_t, N>{wasm_i8x16_gt(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int16_t, N> operator>(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Mask128<int16_t, N>{wasm_i16x8_gt(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<int32_t, N> operator>(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Mask128<int32_t, N>{wasm_i32x4_gt(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<float, N> operator>(const Vec128<float, N> a,
                                                     const Vec128<float, N> b) {
  return Mask128<float, N>{wasm_f32x4_gt(a.raw, b.raw)};
}

// ------------------------------ Weak inequality

// Float <= >=
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<float, N> operator<=(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Mask128<float, N>{wasm_f32x4_le(a.raw, b.raw)};
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<float, N> operator>=(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Mask128<float, N>{wasm_f32x4_ge(a.raw, b.raw)};
}

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> operator&(Vec128<T, N> a,
                                                Vec128<T, N> b) {
  return Vec128<T, N>{wasm_v128_and(a.raw, b.raw)};
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> AndNot(Vec128<T, N> not_mask,
                                             Vec128<T, N> mask) {
  return Vec128<T, N>{wasm_v128_and(wasm_v128_not(not_mask.raw), mask.raw)};
}

// ------------------------------ Bitwise OR

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> operator|(Vec128<T, N> a,
                                                Vec128<T, N> b) {
  return Vec128<T, N>{wasm_v128_or(a.raw, b.raw)};
}

// ------------------------------ Bitwise XOR

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> operator^(Vec128<T, N> a,
                                                Vec128<T, N> b) {
  return Vec128<T, N>{wasm_v128_xor(a.raw, b.raw)};
}

// ------------------------------ Mask

// Mask and Vec are the same (true = FF..FF).
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Mask128<T, N> MaskFromVec(const Vec128<T, N> v) {
  return Mask128<T, N>{v.raw};
}

template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> VecFromMask(const Mask128<T, N> v) {
  return Vec128<T, N>{v.raw};
}

// mask ? yes : no
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> IfThenElse(Mask128<T, N> mask,
                                                 Vec128<T, N> yes,
                                                 Vec128<T, N> no) {
  return Vec128<T, N>{wasm_v128_bitselect(yes.raw, no.raw, mask.raw)};
}

// mask ? yes : 0
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> IfThenElseZero(Mask128<T, N> mask,
                                                     Vec128<T, N> yes) {
  return yes & VecFromMask(mask);
}

// mask ? 0 : no
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> IfThenZeroElse(Mask128<T, N> mask,
                                                     Vec128<T, N> no) {
  return AndNot(VecFromMask(mask), no);
}

template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> ZeroIfNegative(Vec128<T, N> v) {
  const Desc<T, N> d;
  return IfThenElse(MaskFromVec(v), Zero(d), v);
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> Load(Full128<T> /* tag */,
                                        const T* HWY_RESTRICT aligned) {
  return Vec128<T>{wasm_v128_load(aligned)};
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> LoadU(Full128<T> /* tag */,
                                         const T* HWY_RESTRICT p) {
  return Vec128<T>{wasm_v128_load(p)};
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T, 8 / sizeof(T)> Load(
    Desc<T, 8 / sizeof(T)> /* tag */, const T* HWY_RESTRICT p) {
  // TODO(eustas): In WASM memory model do we care about overflow?
  // TODO(eustas): Use memcpy for safe load?
  // TODO(eustas): Use memcpy + zero init to set the upper lanes to 0.
  return Vec128<T, 8 / sizeof(T)>{wasm_v128_load(p)};
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T, 4 / sizeof(T)> Load(
    Desc<T, 4 / sizeof(T)> /* tag */, const T* HWY_RESTRICT p) {
  return Vec128<T, 4 / sizeof(T)>{wasm_v128_load(p)};
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> LoadDup128(Full128<T> /* tag */,
                                              const T* HWY_RESTRICT p) {
  return Vec128<T>{wasm_v128_load(p)};
}

// ------------------------------ Store

template <typename T>
HWY_ATTR_WASM HWY_INLINE void Store(Vec128<T> v, Full128<T> /* tag */,
                                    T* HWY_RESTRICT aligned) {
  wasm_v128_store(aligned, v.raw);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE void StoreU(Vec128<T> v, Full128<T> /* tag */,
                                     T* HWY_RESTRICT p) {
  wasm_v128_store(p, v.raw);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE void Store(Vec128<T, 8 / sizeof(T)> v,
                                    Desc<T, 8 / sizeof(T)> /* tag */,
                                    T* HWY_RESTRICT p) {
  CopyBytes<8>(&v, p);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE void Store(Vec128<T, 4 / sizeof(T)> v,
                                    Desc<T, 4 / sizeof(T)> /* tag */,
                                    T* HWY_RESTRICT p) {
  CopyBytes<4>(&v, p);
}
HWY_ATTR_WASM HWY_INLINE void Store(const Vec128<float, 1> v,
                                    Desc<float, 1> /* tag */,
                                    float* HWY_RESTRICT p) {
  *p = wasm_f32x4_extract_lane(v.raw, 0);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
HWY_ATTR_WASM HWY_INLINE void Stream(Vec128<T> v, Full128<T> /* tag */,
                                     T* HWY_RESTRICT aligned) {
  wasm_v128_store(aligned, v.raw);
}

// ------------------------------ Gather

// Unsupported.

// ================================================== SWIZZLE

// ------------------------------ Extract lane

// Gets the single value stored in a vector/part.
template <size_t N>
HWY_ATTR_WASM HWY_INLINE uint16_t GetLane(const Vec128<uint16_t, N> v) {
  return wasm_i16x8_extract_lane(v.raw, 0);
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE int16_t GetLane(const Vec128<int16_t, N> v) {
  return wasm_i16x8_extract_lane(v.raw, 0);
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE uint32_t GetLane(const Vec128<uint32_t, N> v) {
  return wasm_i32x4_extract_lane(v.raw, 0);
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE int32_t GetLane(const Vec128<int32_t, N> v) {
  return wasm_i32x4_extract_lane(v.raw, 0);
}
template <size_t N>
HWY_ATTR_WASM HWY_INLINE float GetLane(const Vec128<float, N> v) {
  return wasm_f32x4_extract_lane(v.raw, 0);
}

// ------------------------------ Extract half

// Returns upper/lower half of a vector.
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N / 2> LowerHalf(Vec128<T, N> v) {
  return Vec128<T, N / 2>{v.raw};
}

// These copy hi into lo (smaller instruction encoding than shifts).
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T, 8 / sizeof(T)> UpperHalf(Vec128<T> v) {
  // TODO(eustas): use swizzle?
  return Vec128<T, 8 / sizeof(T)>{wasm_v8x16_shuffle(v.raw, v.raw, 8, 9, 10, 11,
                                                     12, 13, 14, 15, 8, 9, 10,
                                                     11, 12, 13, 14, 15)};
}
template <>
HWY_ATTR_WASM HWY_INLINE Vec128<float, 2> UpperHalf(Vec128<float> v) {
  // TODO(eustas): use swizzle?
  return Vec128<float, 2>{wasm_v8x16_shuffle(v.raw, v.raw, 24, 25, 26, 27, 28,
                                             29, 30, 31, 8, 9, 10, 11, 12, 13,
                                             14, 15)};
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T, 8 / sizeof(T)> GetHalf(Lower /* tag */,
                                                          Vec128<T> v) {
  return LowerHalf(v);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T, 8 / sizeof(T)> GetHalf(Upper /* tag */,
                                                          Vec128<T> v) {
  return UpperHalf(v);
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ShiftLeftBytes(const Vec128<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  const __i8x16 zero = wasm_i8x16_splat(0);
  switch (kBytes) {
    case 0:
      return v;

    case 1:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 0, 1, 2, 3, 4, 5, 6,
                                          7, 8, 9, 10, 11, 12, 13, 14)};

    case 2:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 0, 1, 2, 3, 4, 5,
                                          6, 7, 8, 9, 10, 11, 12, 13)};

    case 3:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 0, 1, 2, 3,
                                          4, 5, 6, 7, 8, 9, 10, 11, 12)};

    case 4:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 0, 1, 2,
                                          3, 4, 5, 6, 7, 8, 9, 10, 11)};

    case 5:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 0, 1,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10)};

    case 6:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9)};

    case 7:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 0, 1, 2, 3, 4, 5, 6, 7, 8)};

    case 8:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 0, 1, 2, 3, 4, 5, 6, 7)};

    case 9:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 0, 1, 2, 3, 4, 5, 6)};

    case 10:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 0, 1, 2, 3, 4, 5)};

    case 11:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 0, 1, 2, 3, 4)};

    case 12:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 0, 1, 2, 3)};

    case 13:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 0, 1, 2)};

    case 14:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 0,
                                          1)};

    case 15:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 16, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          0)};
  }
  return Vec128<T>{zero};
}

template <int kLanes, typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ShiftLeftLanes(const Vec128<T> v) {
  return ShiftLeftBytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ShiftRightBytes(const Vec128<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  const __i8x16 zero = wasm_i8x16_splat(0);
  switch (kBytes) {
    case 0:
      return v;

    case 1:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 1, 2, 3, 4, 5, 6, 7, 8,
                                          9, 10, 11, 12, 13, 14, 15, 16)};

    case 2:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 2, 3, 4, 5, 6, 7, 8, 9,
                                          10, 11, 12, 13, 14, 15, 16, 16)};

    case 3:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 3, 4, 5, 6, 7, 8, 9, 10,
                                          11, 12, 13, 14, 15, 16, 16, 16)};

    case 4:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 4, 5, 6, 7, 8, 9, 10, 11,
                                          12, 13, 14, 15, 16, 16, 16, 16)};

    case 5:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 5, 6, 7, 8, 9, 10, 11,
                                          12, 13, 14, 15, 16, 16, 16, 16, 16)};

    case 6:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 6, 7, 8, 9, 10, 11, 12,
                                          13, 14, 15, 16, 16, 16, 16, 16, 16)};

    case 7:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 7, 8, 9, 10, 11, 12, 13,
                                          14, 15, 16, 16, 16, 16, 16, 16, 16)};

    case 8:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 8, 9, 10, 11, 12, 13, 14,
                                          15, 16, 16, 16, 16, 16, 16, 16, 16)};

    case 9:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 9, 10, 11, 12, 13, 14,
                                          15, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};

    case 10:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 10, 11, 12, 13, 14, 15,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};

    case 11:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 11, 12, 13, 14, 15, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};

    case 12:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 12, 13, 14, 15, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};

    case 13:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 13, 14, 15, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};

    case 14:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 14, 15, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};

    case 15:
      return Vec128<T>{wasm_v8x16_shuffle(v.raw, zero, 15, 16, 16, 16, 16, 16,
                                          16, 16, 16, 16, 16, 16, 16, 16, 16,
                                          16)};
  }
  return Vec128<T>{zero};
}

template <int kLanes, typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ShiftRightLanes(const Vec128<T> v) {
  return ShiftRightBytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> CombineShiftRightBytes(const Vec128<T> hi,
                                                          const Vec128<T> lo) {
  const auto l = ShiftRightBytes<kBytes>(lo);
  const auto h = ShiftLeftBytes<16 - kBytes>(hi);
  return Vec128<T>{wasm_v128_or(l.raw, h.raw)};
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t> Broadcast(const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  constexpr int l0 = kLane * 2;
  constexpr int l1 = kLane * 2 + 1;
  return Vec128<uint16_t>{wasm_v8x16_shuffle(v.raw, v.raw, l0, l1, l0, l1, l0,
                                             l1, l0, l1, l0, l1, l0, l1, l0, l1,
                                             l0, l1)};
}
template <int kLane>
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> Broadcast(const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  constexpr int l0 = kLane * 4;
  constexpr int l1 = kLane * 4 + 1;
  constexpr int l2 = kLane * 4 + 2;
  constexpr int l3 = kLane * 4 + 3;
  return Vec128<uint32_t>{wasm_v8x16_shuffle(v.raw, v.raw, l0, l1, l2, l3, l0,
                                             l1, l2, l3, l0, l1, l2, l3, l0, l1,
                                             l2, l3)};
}

// Signed
template <int kLane>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> Broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  constexpr int l0 = kLane * 2;
  constexpr int l1 = kLane * 2 + 1;
  return Vec128<int16_t>{wasm_v8x16_shuffle(v.raw, v.raw, l0, l1, l0, l1, l0,
                                            l1, l0, l1, l0, l1, l0, l1, l0, l1,
                                            l0, l1)};
}
template <int kLane>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> Broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  constexpr int l0 = kLane * 4;
  constexpr int l1 = kLane * 4 + 1;
  constexpr int l2 = kLane * 4 + 2;
  constexpr int l3 = kLane * 4 + 3;
  return Vec128<int32_t>{wasm_v8x16_shuffle(v.raw, v.raw, l0, l1, l2, l3, l0,
                                            l1, l2, l3, l0, l1, l2, l3, l0, l1,
                                            l2, l3)};
}

// Float
template <int kLane>
HWY_ATTR_WASM HWY_INLINE Vec128<float> Broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  constexpr int l0 = kLane * 4;
  constexpr int l1 = kLane * 4 + 1;
  constexpr int l2 = kLane * 4 + 2;
  constexpr int l3 = kLane * 4 + 3;
  return Vec128<float>{wasm_v8x16_shuffle(v.raw, v.raw, l0, l1, l2, l3, l0, l1,
                                          l2, l3, l0, l1, l2, l3, l0, l1, l2,
                                          l3)};
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
HWY_ATTR_WASM HWY_INLINE Vec128<T> TableLookupBytes(const Vec128<T> bytes,
                                                    const Vec128<TI> from) {
  // TODO(eustas): use swizzle? what about 0x80+ indices?
  alignas(16) uint8_t control[16];
  alignas(16) uint8_t input[16];
  alignas(16) uint8_t output[16];
  wasm_v128_store(control, from.raw);
  wasm_v128_store(input, bytes.raw);
  // TODO(eustas): wasm_v8x16_shuffle does not work: params have to be
  // constants.
  for (size_t i = 0; i < 16; ++i) {
    const int idx = control[i];
    output[i] = (idx >= 0x80) ? 0 : input[idx];
  }
  return Vec128<T>{wasm_v128_load(output)};
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec128<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// Shuffle0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// CombineShiftRightBytes but the shuffle_abcd notation is more convenient.

// Swap 32-bit halves in 64-bit halves.
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> Shuffle2301(
    const Vec128<uint32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<uint32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 4, 5, 6, 7, 0, 1, 2,
                                             3, 12, 13, 14, 15, 8, 9, 10, 11)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> Shuffle2301(const Vec128<int32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<int32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 4, 5, 6, 7, 0, 1, 2,
                                            3, 12, 13, 14, 15, 8, 9, 10, 11)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<float> Shuffle2301(const Vec128<float> v) {
  // TODO(eustas): use swizzle?
  return Vec128<float>{wasm_v8x16_shuffle(v.raw, v.raw, 4, 5, 6, 7, 0, 1, 2, 3,
                                          12, 13, 14, 15, 8, 9, 10, 11)};
}

// Swap 64-bit halves
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> Shuffle1032(
    const Vec128<uint32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<uint32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 8, 9, 10, 11, 12, 13,
                                             14, 15, 0, 1, 2, 3, 4, 5, 6, 7)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> Shuffle1032(const Vec128<int32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<int32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 8, 9, 10, 11, 12, 13,
                                            14, 15, 0, 1, 2, 3, 4, 5, 6, 7)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<float> Shuffle1032(const Vec128<float> v) {
  // TODO(eustas): use swizzle?
  return Vec128<float>{wasm_v8x16_shuffle(v.raw, v.raw, 8, 9, 10, 11, 12, 13,
                                          14, 15, 0, 1, 2, 3, 4, 5, 6, 7)};
}

// Rotate right 32 bits
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> Shuffle0321(
    const Vec128<uint32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<uint32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 4, 5, 6, 7, 8, 9, 10,
                                             11, 12, 13, 14, 15, 0, 1, 2, 3)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> Shuffle0321(const Vec128<int32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<int32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 4, 5, 6, 7, 8, 9, 10,
                                            11, 12, 13, 14, 15, 0, 1, 2, 3)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<float> Shuffle0321(const Vec128<float> v) {
  // TODO(eustas): use swizzle?
  return Vec128<float>{wasm_v8x16_shuffle(v.raw, v.raw, 4, 5, 6, 7, 8, 9, 10,
                                          11, 12, 13, 14, 15, 0, 1, 2, 3)};
}
// Rotate left 32 bits
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> Shuffle2103(
    const Vec128<uint32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<uint32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 12, 13, 14, 15, 0, 1,
                                             2, 3, 4, 5, 6, 7, 8, 9, 10, 11)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> Shuffle2103(const Vec128<int32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<int32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 12, 13, 14, 15, 0, 1,
                                            2, 3, 4, 5, 6, 7, 8, 9, 10, 11)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<float> Shuffle2103(const Vec128<float> v) {
  // TODO(eustas): use swizzle?
  return Vec128<float>{wasm_v8x16_shuffle(v.raw, v.raw, 12, 13, 14, 15, 0, 1, 2,
                                          3, 4, 5, 6, 7, 8, 9, 10, 11)};
}

// Reverse
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> Shuffle0123(
    const Vec128<uint32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<uint32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 12, 13, 14, 15, 8, 9,
                                             10, 11, 4, 5, 6, 7, 0, 1, 2, 3)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> Shuffle0123(const Vec128<int32_t> v) {
  // TODO(eustas): use swizzle?
  return Vec128<int32_t>{wasm_v8x16_shuffle(v.raw, v.raw, 12, 13, 14, 15, 8, 9,
                                            10, 11, 4, 5, 6, 7, 0, 1, 2, 3)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<float> Shuffle0123(const Vec128<float> v) {
  // TODO(eustas): use swizzle?
  return Vec128<float>{wasm_v8x16_shuffle(v.raw, v.raw, 12, 13, 14, 15, 8, 9,
                                          10, 11, 4, 5, 6, 7, 0, 1, 2, 3)};
}

// ------------------------------ Permute (runtime variable)

// Returned by SetTableIndices for use by TableLookupLanes.
template <typename T>
struct permute_sse4 {
  __v128_u raw;
};

template <typename T>
HWY_ATTR_WASM HWY_INLINE permute_sse4<T> SetTableIndices(Full128<T> d,
                                                         const int32_t* idx) {
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
  for (size_t i = 0; i < d.N; ++i) {
    // TODO(eustas): also assume idx[i] >= 0
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
  // TODO(eustas): permute_sse4?
  return permute_sse4<T>{Load(d8, control).raw};
}

HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> TableLookupLanes(
    const Vec128<uint32_t> v, const permute_sse4<uint32_t> idx) {
  return TableLookupBytes(v, Vec128<uint8_t>{idx.raw});
}

HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> TableLookupLanes(
    const Vec128<int32_t> v, const permute_sse4<int32_t> idx) {
  return TableLookupBytes(v, Vec128<uint8_t>{idx.raw});
}

HWY_ATTR_WASM HWY_INLINE Vec128<float> TableLookupLanes(
    const Vec128<float> v, const permute_sse4<float> idx) {
  return TableLookupBytes(v, Vec128<uint8_t>{idx.raw});
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use ZipLo/hi instead (also works with scalar).

HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t> InterleaveLo(const Vec128<uint8_t> a,
                                                      const Vec128<uint8_t> b) {
  return Vec128<uint8_t>{wasm_v8x16_shuffle(a.raw, b.raw, 0, 16, 1, 17, 2, 18,
                                            3, 19, 4, 20, 5, 21, 6, 22, 7, 23)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t> InterleaveLo(
    const Vec128<uint16_t> a, const Vec128<uint16_t> b) {
  return Vec128<uint16_t>{wasm_v8x16_shuffle(
      a.raw, b.raw, 0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> InterleaveLo(
    const Vec128<uint32_t> a, const Vec128<uint32_t> b) {
  return Vec128<uint32_t>{wasm_v8x16_shuffle(
      a.raw, b.raw, 0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23)};
}

HWY_ATTR_WASM HWY_INLINE Vec128<int8_t> InterleaveLo(const Vec128<int8_t> a,
                                                     const Vec128<int8_t> b) {
  return Vec128<int8_t>{wasm_v8x16_shuffle(a.raw, b.raw, 0, 16, 1, 17, 2, 18, 3,
                                           19, 4, 20, 5, 21, 6, 22, 7, 23)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> InterleaveLo(const Vec128<int16_t> a,
                                                      const Vec128<int16_t> b) {
  return Vec128<int16_t>{wasm_v8x16_shuffle(
      a.raw, b.raw, 0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> InterleaveLo(const Vec128<int32_t> a,
                                                      const Vec128<int32_t> b) {
  return Vec128<int32_t>{wasm_v8x16_shuffle(
      a.raw, b.raw, 0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23)};
}

HWY_ATTR_WASM HWY_INLINE Vec128<float> InterleaveLo(const Vec128<float> a,
                                                    const Vec128<float> b) {
  return Vec128<float>{wasm_v8x16_shuffle(a.raw, b.raw, 0, 1, 2, 3, 16, 17, 18,
                                          19, 4, 5, 6, 7, 20, 21, 22, 23)};
}

HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t> InterleaveHi(const Vec128<uint8_t> a,
                                                      const Vec128<uint8_t> b) {
  return Vec128<uint8_t>{wasm_v8x16_shuffle(a.raw, b.raw, 8, 24, 9, 25, 10, 26,
                                            11, 27, 12, 28, 13, 29, 14, 30, 15,
                                            31)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t> InterleaveHi(
    const Vec128<uint16_t> a, const Vec128<uint16_t> b) {
  // TODO(eustas): remove emulation, when segfault is fixed.
  alignas(16) uint16_t a_lanes[8];
  alignas(16) uint16_t b_lanes[8];
  alignas(16) uint16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[2 * i] = a_lanes[4 + i];
    output[2 * i + 1] = b_lanes[4 + i];
  }
  return Vec128<uint16_t>{wasm_v128_load(output)};
  // return Vec128<uint16_t>{wasm_v8x16_shuffle(a.raw, b.raw,
  //                                            8,  9, 24, 25,
  //                                           10, 11, 26, 27,
  //                                           12, 13, 28, 29,
  //                                           14, 15, 30, 31)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> InterleaveHi(
    const Vec128<uint32_t> a, const Vec128<uint32_t> b) {
  // TODO(eustas): remove emulation, when segfault is fixed.
  alignas(16) uint32_t a_lanes[4];
  alignas(16) uint32_t b_lanes[4];
  alignas(16) uint32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 2; ++i) {
    output[2 * i] = a_lanes[2 + i];
    output[2 * i + 1] = b_lanes[2 + i];
  }
  return Vec128<uint32_t>{wasm_v128_load(output)};
  // return Vec128<uint32_t>{wasm_v8x16_shuffle(a.raw, b.raw,
  //                                            8,  9, 10, 11,
  //                                           24, 25, 26, 27,
  //                                           12, 13, 14, 15,
  //                                           28, 29, 30, 31)};
}

HWY_ATTR_WASM HWY_INLINE Vec128<int8_t> InterleaveHi(const Vec128<int8_t> a,
                                                     const Vec128<int8_t> b) {
  return Vec128<int8_t>{wasm_v8x16_shuffle(a.raw, b.raw, 8, 24, 9, 25, 10, 26,
                                           11, 27, 12, 28, 13, 29, 14, 30, 15,
                                           31)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> InterleaveHi(const Vec128<int16_t> a,
                                                      const Vec128<int16_t> b) {
  // TODO(eustas): remove emulation, when segfault is fixed.
  alignas(16) int16_t a_lanes[8];
  alignas(16) int16_t b_lanes[8];
  alignas(16) int16_t output[8];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 4; ++i) {
    output[2 * i] = a_lanes[4 + i];
    output[2 * i + 1] = b_lanes[4 + i];
  }
  return Vec128<int16_t>{wasm_v128_load(output)};
  // return Vec128<int16_t>{wasm_v8x16_shuffle(a.raw, b.raw,
  //                                           8,  9, 24, 25,
  //                                          10, 11, 26, 27,
  //                                          12, 13, 28, 29,
  //                                          14, 15, 30, 31)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> InterleaveHi(const Vec128<int32_t> a,
                                                      const Vec128<int32_t> b) {
  // TODO(eustas): remove emulation, when segfault is fixed.
  alignas(16) int32_t a_lanes[4];
  alignas(16) int32_t b_lanes[4];
  alignas(16) int32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 2; ++i) {
    output[2 * i] = a_lanes[2 + i];
    output[2 * i + 1] = b_lanes[2 + i];
  }
  return Vec128<int32_t>{wasm_v128_load(output)};
  // return Vec128<int32_t>{wasm_v8x16_shuffle(a.raw, b.raw,
  //                                           8,  9, 10, 11,
  //                                          24, 25, 26, 27,
  //                                          12, 13, 14, 15,
  //                                          28, 29, 30, 31)};
}

HWY_ATTR_WASM HWY_INLINE Vec128<float> InterleaveHi(const Vec128<float> a,
                                                    const Vec128<float> b) {
  // TODO(eustas): remove emulation, when segfault is fixed.
  alignas(16) int32_t a_lanes[4];
  alignas(16) int32_t b_lanes[4];
  alignas(16) int32_t output[4];
  wasm_v128_store(a_lanes, a.raw);
  wasm_v128_store(b_lanes, b.raw);
  for (size_t i = 0; i < 2; ++i) {
    output[2 * i] = a_lanes[2 + i];
    output[2 * i + 1] = b_lanes[2 + i];
  }
  return Vec128<float>{wasm_v128_load(output)};
  // return Vec128<float>{wasm_v8x16_shuffle(a.raw, b.raw,
  //                                         8,  9, 10, 11,
  //                                        24, 25, 26, 27,
  //                                        12, 13, 14, 15,
  //                                        28, 29, 30, 31)};
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t> ZipLo(const Vec128<uint8_t> a,
                                                const Vec128<uint8_t> b) {
  return Vec128<uint16_t>{InterleaveLo(a, b).raw};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> ZipLo(const Vec128<uint16_t> a,
                                                const Vec128<uint16_t> b) {
  return Vec128<uint32_t>{InterleaveLo(a, b).raw};
}

HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> ZipLo(const Vec128<int8_t> a,
                                               const Vec128<int8_t> b) {
  return Vec128<int16_t>{InterleaveLo(a, b).raw};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> ZipLo(const Vec128<int16_t> a,
                                               const Vec128<int16_t> b) {
  return Vec128<int32_t>{InterleaveLo(a, b).raw};
}

HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t> ZipHi(const Vec128<uint8_t> a,
                                                const Vec128<uint8_t> b) {
  return Vec128<uint16_t>{InterleaveHi(a, b).raw};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> ZipHi(const Vec128<uint16_t> a,
                                                const Vec128<uint16_t> b) {
  return Vec128<uint32_t>{InterleaveHi(a, b).raw};
}

HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> ZipHi(const Vec128<int8_t> a,
                                               const Vec128<int8_t> b) {
  return Vec128<int16_t>{InterleaveHi(a, b).raw};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> ZipHi(const Vec128<int16_t> a,
                                               const Vec128<int16_t> b) {
  return Vec128<int32_t>{InterleaveHi(a, b).raw};
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ConcatLoLo(const Vec128<T> hi,
                                              const Vec128<T> lo) {
  return Vec128<T>{wasm_v8x16_shuffle(lo.raw, hi.raw, 0, 1, 2, 3, 4, 5, 6, 7,
                                      16, 17, 18, 19, 20, 21, 22, 23)};
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ConcatHiHi(const Vec128<T> hi,
                                              const Vec128<T> lo) {
  return Vec128<T>{wasm_v8x16_shuffle(lo.raw, hi.raw, 8, 9, 10, 11, 12, 13, 14,
                                      15, 24, 25, 26, 27, 28, 29, 30, 31)};
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ConcatLoHi(const Vec128<T> hi,
                                              const Vec128<T> lo) {
  return CombineShiftRightBytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> ConcatHiLo(const Vec128<T> hi,
                                              const Vec128<T> lo) {
  return Vec128<T>{wasm_v8x16_shuffle(lo.raw, hi.raw, 0, 1, 2, 3, 4, 5, 6, 7,
                                      24, 25, 26, 27, 28, 29, 30, 31)};
}
template <>
HWY_ATTR_WASM HWY_INLINE Vec128<float> ConcatHiLo(const Vec128<float> hi,
                                                  const Vec128<float> lo) {
  return Vec128<float>{wasm_v8x16_shuffle(hi.raw, lo.raw, 16, 17, 18, 19, 20,
                                          21, 22, 23, 8, 9, 10, 11, 12, 13, 14,
                                          15)};
}

// ------------------------------ Odd/even lanes

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> odd_even_impl(SizeTag<1> /* tag */,
                                                 const Vec128<T> a,
                                                 const Vec128<T> b) {
  const Full128<T> d;
  const Full128<uint8_t> d8;
  alignas(16) constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                            0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return IfThenElse(MaskFromVec(BitCast(d, Load(d8, mask))), b, a);
}
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> odd_even_impl(SizeTag<2> /* tag */,
                                                 const Vec128<T> a,
                                                 const Vec128<T> b) {
  return Vec128<T>{wasm_v8x16_shuffle(a.raw, b.raw, 16, 17, 2, 3, 20, 21, 6, 7,
                                      24, 25, 10, 11, 28, 29, 14, 15)};
}
template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> odd_even_impl(SizeTag<4> /* tag */,
                                                 const Vec128<T> a,
                                                 const Vec128<T> b) {
  return Vec128<T>{wasm_v8x16_shuffle(a.raw, b.raw, 16, 17, 18, 19, 4, 5, 6, 7,
                                      24, 25, 26, 27, 12, 13, 14, 15)};
}
// TODO(eustas): implement
// template <typename T>
// HWY_ATTR_WASM HWY_INLINE Vec128<T> odd_even_impl(SizeTag<8> /* tag */,
//                                                 const Vec128<T> a,
//                                                 const Vec128<T> b)

template <typename T>
HWY_ATTR_WASM HWY_INLINE Vec128<T> OddEven(const Vec128<T> a,
                                           const Vec128<T> b) {
  return odd_even_impl(SizeTag<sizeof(T)>(), a, b);
}
template <>
HWY_ATTR_WASM HWY_INLINE Vec128<float> OddEven<float>(const Vec128<float> a,
                                                      const Vec128<float> b) {
  return Vec128<float>{wasm_v8x16_shuffle(a.raw, b.raw, 16, 17, 18, 19, 4, 5, 6,
                                          7, 24, 25, 26, 27, 12, 13, 14, 15)};
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t> ConvertTo(
    Full128<uint16_t> /* tag */, const Vec128<uint8_t, 8> v) {
  return Vec128<uint16_t>{wasm_i16x8_widen_low_u8x16(v.raw)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> ConvertTo(
    Full128<uint32_t> /* tag */, const Vec128<uint8_t, 4> v) {
  return Vec128<uint32_t>{
      wasm_i32x4_widen_low_u16x8(wasm_i16x8_widen_low_u8x16(v.raw))};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> ConvertTo(Full128<int16_t> /* tag */,
                                                   const Vec128<uint8_t, 8> v) {
  return Vec128<int16_t>{wasm_i16x8_widen_low_u8x16(v.raw)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                                   const Vec128<uint8_t, 4> v) {
  return Vec128<int32_t>{
      wasm_i32x4_widen_low_u16x8(wasm_i16x8_widen_low_u8x16(v.raw))};
}
HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> ConvertTo(
    Full128<uint32_t> /* tag */, const Vec128<uint16_t, 4> v) {
  return Vec128<uint32_t>{__builtin_wasm_widen_low_u_i32x4_i16x8(v.raw)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> ConvertTo(
    Full128<int32_t> /* tag */, const Vec128<uint16_t, 4> v) {
  return Vec128<int32_t>{__builtin_wasm_widen_low_u_i32x4_i16x8(v.raw)};
}

HWY_ATTR_WASM HWY_INLINE Vec128<uint32_t> U32FromU8(const Vec128<uint8_t> v) {
  return Vec128<uint32_t>{
      wasm_i32x4_widen_low_u16x8(wasm_i16x8_widen_low_u8x16(v.raw))};
}

// Signed: replicate sign bit.
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t> ConvertTo(Full128<int16_t> /* tag */,
                                                   const Vec128<int8_t, 8> v) {
  return Vec128<int16_t>{wasm_i16x8_widen_low_i8x16(v.raw)};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                                   const Vec128<int8_t, 4> v) {
  return Vec128<int32_t>{
      wasm_i32x4_widen_low_i16x8(wasm_i16x8_widen_low_i8x16(v.raw))};
}
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t> ConvertTo(Full128<int32_t> /* tag */,
                                                   const Vec128<int16_t, 4> v) {
  return Vec128<int32_t>{wasm_i32x4_widen_low_i16x8(v.raw)};
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint16_t, N> ConvertTo(
    Desc<uint16_t, N> /* tag */, const Vec128<int32_t, N> v) {
  return Vec128<uint16_t, N>{wasm_u16x8_narrow_i32x4(v.raw, v.raw)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> ConvertTo(
    Desc<uint8_t, N> /* tag */, const Vec128<int32_t> v) {
  const auto intermediate = wasm_i16x8_narrow_i32x4(v.raw, v.raw);
  return Vec128<uint8_t, N>{
      wasm_u8x16_narrow_i16x8(intermediate, intermediate)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, N> ConvertTo(
    Desc<uint8_t, N> /* tag */, const Vec128<int16_t> v) {
  return Vec128<uint8_t, N>{wasm_u8x16_narrow_i16x8(v.raw, v.raw)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int16_t, N> ConvertTo(
    Desc<int16_t, N> /* tag */, const Vec128<int32_t> v) {
  return Vec128<int16_t, N>{wasm_i16x8_narrow_i32x4(v.raw, v.raw)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> ConvertTo(Desc<int8_t, N> /* tag */,
                                                     const Vec128<int32_t> v) {
  const auto intermediate = wasm_i16x8_narrow_i32x4(v.raw, v.raw);
  return Vec128<int8_t, N>{wasm_i8x16_narrow_i16x8(intermediate, intermediate)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int8_t, N> ConvertTo(Desc<int8_t, N> /* tag */,
                                                     const Vec128<int16_t> v) {
  return Vec128<int8_t, N>{wasm_i8x16_narrow_i16x8(v.raw, v.raw)};
}

// For already range-limited input [0, 255].
HWY_ATTR_WASM HWY_INLINE Vec128<uint8_t, 4> U8FromU32(
    const Vec128<uint32_t> v) {
  const auto intermediate = wasm_i16x8_narrow_i32x4(v.raw, v.raw);
  return Vec128<uint8_t, 4>{
      wasm_u8x16_narrow_i16x8(intermediate, intermediate)};
}

// ------------------------------ Convert i32 <=> f32

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<float, N> ConvertTo(
    Desc<float, N> /* tag */, const Vec128<int32_t, N> v) {
  return Vec128<float, N>{wasm_f32x4_convert_i32x4(v.raw)};
}
// Truncates (rounds toward zero).
template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> ConvertTo(
    Desc<int32_t, N> /* tag */, const Vec128<float, N> v) {
  return Vec128<int32_t, N>{wasm_i32x4_trunc_saturate_f32x4(v.raw)};
}

template <size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<int32_t, N> NearestInt(
    const Vec128<float, N> v) {
  const __f32x4 c00 = wasm_f32x4_splat(0.0f);
  const __f32x4 corr = wasm_f32x4_convert_i32x4(wasm_f32x4_le(v.raw, c00));
  const __f32x4 c05 = wasm_f32x4_splat(0.5f);
  // +0.5 for non-negative lane, -0.5 for other.
  const __f32x4 delta = wasm_f32x4_add(c05, corr);
  // Shift input by 0.5 away from 0.
  const __f32x4 fixed = wasm_f32x4_add(v.raw, delta);
  return Vec128<int32_t, N>{wasm_i32x4_trunc_saturate_f32x4(fixed)};
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ Mask

template <typename T>
HWY_ATTR_WASM HWY_INLINE bool AllFalse(const Mask128<T> v) {
  return !wasm_i8x16_any_true(v.raw);
}
HWY_ATTR_WASM HWY_INLINE bool AllFalse(const Mask128<float> v) {
  return !wasm_i32x4_any_true(v.raw);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE bool AllTrue(const Mask128<T> v) {
  return wasm_i8x16_all_true(v.raw);
}
HWY_ATTR_WASM HWY_INLINE bool AllTrue(const Mask128<float> v) {
  return wasm_i32x4_all_true(v.raw);
}

namespace impl {

template <typename T>
HWY_ATTR_WASM HWY_INLINE uint64_t BitsFromMask(SizeTag<1> /*tag*/,
                                               const Mask128<T> mask) {
  const __i8x16 slice =
      wasm_i8x16_make(1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8);
  // Each u32 lane has byte[i] = (1 << i) or 0.
  const __i8x16 v8_4_2_1 = wasm_v128_and(mask.raw, slice);
  // OR together 4 bytes of each u32 to get the 4 bits.
  const __i16x8 v2_1_z_z = wasm_i32x4_shl(v8_4_2_1, 16);
  const __i16x8 v82_41_2_1 = wasm_v128_or(v8_4_2_1, v2_1_z_z);
  const __i16x8 v41_2_1_0 = wasm_i32x4_shl(v82_41_2_1, 8);
  const __i16x8 v8421_421_21_10 = wasm_v128_or(v82_41_2_1, v41_2_1_0);
  const __i16x8 nibble_per_u32 = wasm_i32x4_shr(v8421_421_21_10, 24);
  // Assemble four nibbles into 16 bits.
  alignas(16) uint32_t lanes[4];
  wasm_v128_store(lanes, nibble_per_u32);
  return lanes[0] | (lanes[1] << 4) | (lanes[2] << 8) | (lanes[3] << 12);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE uint64_t BitsFromMask(SizeTag<2> /*tag*/,
                                               const Mask128<T> mask) {
  // Remove useless lower half of each u16 while preserving the sign bit.
  const __i16x8 zero = wasm_i16x8_splat(0);
  const Mask128<T> mask8{wasm_i8x16_narrow_i16x8(mask.raw, zero)};
  return BitsFromMask(SizeTag<1>(), mask8);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE uint64_t BitsFromMask(SizeTag<4> /*tag*/,
                                               const Mask128<T> mask) {
  const __i32x4 mask_i = static_cast<__i32x4>(mask.raw);
  const __i32x4 slice = wasm_i32x4_make(1, 2, 4, 8);
  const __i32x4 sliced_mask = wasm_v128_and(mask_i, slice);
  alignas(16) uint32_t lanes[4];
  wasm_v128_store(lanes, sliced_mask);
  return lanes[0] | lanes[1] | lanes[2] | lanes[3];
}

}  // namespace impl

template <typename T>
HWY_ATTR_WASM HWY_INLINE uint64_t BitsFromMask(const Mask128<T> mask) {
  return impl::BitsFromMask(SizeTag<sizeof(T)>(), mask);
}

template <typename T>
HWY_ATTR_WASM HWY_INLINE size_t CountTrue(const Mask128<T> v) {
  const __i32x4 mask =
      wasm_i32x4_make(0x01010101, 0x01010101, 0x02020202, 0x02020202);
  const __i8x16 shifted_bits = wasm_v128_and(v.raw, mask);
  alignas(16) uint64_t lanes[2];
  wasm_v128_store(lanes, shifted_bits);
  return PopCount(lanes[0] | lanes[1]) / sizeof(T);
}

HWY_ATTR_WASM HWY_INLINE size_t CountTrue(const Mask128<float> v) {
  const __i32x4 var_shift = wasm_i32x4_make(1, 2, 4, 8);
  const __i32x4 shifted_bits = wasm_v128_and(v.raw, var_shift);
  alignas(16) uint64_t lanes[2];
  wasm_v128_store(lanes, shifted_bits);
  return PopCount(lanes[0] | lanes[1]);
}

// ------------------------------ Horizontal sum (reduction)

// TODO(eustas): optimize
// Returns 64-bit sums of 8-byte groups.
HWY_ATTR_WASM HWY_INLINE Vec128<uint64_t> SumsOfU8x8(const Vec128<uint8_t> v) {
  alignas(16) uint8_t lanes[16];
  wasm_v128_store(lanes, v.raw);
  uint32_t sums[2] = {0};
  for (size_t i = 0; i < 16; ++i) {
    sums[i / 8] += lanes[i];
  }
  return Vec128<uint64_t>{wasm_i32x4_make(sums[0], 0, sums[1], 0)};
}

// For u32/i32/f32.
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> horz_sum_impl(SizeTag<4> /* tag */,
                                                    const Vec128<T, N> v3210) {
  const Vec128<T> v1032 = Shuffle1032(v3210);
  const Vec128<T> v31_20_31_20 = v3210 + v1032;
  const Vec128<T> v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

// For u64/i64/f64.
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> horz_sum_impl(SizeTag<8> /* tag */,
                                                    const Vec128<T, N> v10) {
  const Vec128<T> v01 = Shuffle01(v10);
  return v10 + v01;
}

// Supported for u/i/f 32/64. Returns the sum in each lane.
template <typename T, size_t N>
HWY_ATTR_WASM HWY_INLINE Vec128<T, N> SumOfLanes(const Vec128<T, N> v) {
  return horz_sum_impl(SizeTag<sizeof(T)>(), v);
}

}  // namespace ext

}  // namespace hwy

#endif  // HWY_WASM_H_
