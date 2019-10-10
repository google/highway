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

#ifndef HIGHWAY_X86_SSE4_H_
#define HIGHWAY_X86_SSE4_H_

// 128-bit SSE4 vectors and operations.

#include <smmintrin.h>

#include "third_party/highway/highway/shared.h"

namespace jxl {

template <typename T>
struct Raw128 {
  using type = __m128i;
};
template <>
struct Raw128<float> {
  using type = __m128;
};
template <>
struct Raw128<double> {
  using type = __m128d;
};

template <typename T>
using Full128 = Desc<T, 16 / sizeof(T)>;

template <typename T, size_t N = 16 / sizeof(T)>
class Vec128 {
  using Raw = typename Raw128<T>::type;

 public:
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128() = default;
  Vec128(const Vec128&) = default;
  Vec128& operator=(const Vec128&) = default;
  SIMD_ATTR_SSE4 SIMD_INLINE explicit Vec128(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator*=(const Vec128 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator/=(const Vec128 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator+=(const Vec128 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator-=(const Vec128 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator&=(const Vec128 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator|=(const Vec128 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_SSE4 SIMD_INLINE Vec128& operator^=(const Vec128 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// ------------------------------ Cast

SIMD_ATTR_SSE4 SIMD_INLINE __m128i BitCastToInteger(__m128i v) { return v; }
SIMD_ATTR_SSE4 SIMD_INLINE __m128i BitCastToInteger(__m128 v) {
  return _mm_castps_si128(v);
}
SIMD_ATTR_SSE4 SIMD_INLINE __m128i BitCastToInteger(__m128d v) {
  return _mm_castpd_si128(v);
}

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N * sizeof(T)> cast_to_u8(
    Vec128<T, N> v) {
  return Vec128<uint8_t, N * sizeof(T)>(BitCastToInteger(v.raw));
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromInteger128 {
  SIMD_ATTR_SSE4 SIMD_INLINE __m128i operator()(__m128i v) { return v; }
};
template <>
struct BitCastFromInteger128<float> {
  SIMD_ATTR_SSE4 SIMD_INLINE __m128 operator()(__m128i v) {
    return _mm_castsi128_ps(v);
  }
};
template <>
struct BitCastFromInteger128<double> {
  SIMD_ATTR_SSE4 SIMD_INLINE __m128d operator()(__m128i v) {
    return _mm_castsi128_pd(v);
  }
};

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> cast_u8_to(
    Desc<T, N> /* tag */, Vec128<uint8_t, N * sizeof(T)> v) {
  return Vec128<T, N>(BitCastFromInteger128<T>()(v.raw));
}

template <typename T, size_t N, typename FromT>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> bit_cast(
    Desc<T, N> d, Vec128<FromT, N * sizeof(T) / sizeof(FromT)> v) {
  return cast_u8_to(d, cast_to_u8(v));
}

// ------------------------------ Set

// Returns an all-zero vector/part.
template <typename T, size_t N, SIMD_IF128(T, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> setzero(Desc<T, N> /* tag */) {
  return Vec128<T, N>(_mm_setzero_si128());
}
template <size_t N, SIMD_IF128(float, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> setzero(Desc<float, N> /* tag */) {
  return Vec128<float, N>(_mm_setzero_ps());
}
template <size_t N, SIMD_IF128(double, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> setzero(
    Desc<double, N> /* tag */) {
  return Vec128<double, N>(_mm_setzero_pd());
}

// Returns a vector/part with all lanes set to "t".
template <size_t N, SIMD_IF128(uint8_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> set1(Desc<uint8_t, N> /* tag */,
                                                   const uint8_t t) {
  return Vec128<uint8_t, N>(_mm_set1_epi8(t));
}
template <size_t N, SIMD_IF128(uint16_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> set1(Desc<uint16_t, N> /* tag */,
                                                    const uint16_t t) {
  return Vec128<uint16_t, N>(_mm_set1_epi16(t));
}
template <size_t N, SIMD_IF128(uint32_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> set1(Desc<uint32_t, N> /* tag */,
                                                    const uint32_t t) {
  return Vec128<uint32_t, N>(_mm_set1_epi32(t));
}
template <size_t N, SIMD_IF128(uint64_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> set1(Desc<uint64_t, N> /* tag */,
                                                    const uint64_t t) {
  return Vec128<uint64_t, N>(_mm_set1_epi64x(t));
}
template <size_t N, SIMD_IF128(int8_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> set1(Desc<int8_t, N> /* tag */,
                                                  const int8_t t) {
  return Vec128<int8_t, N>(_mm_set1_epi8(t));
}
template <size_t N, SIMD_IF128(int16_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> set1(Desc<int16_t, N> /* tag */,
                                                   const int16_t t) {
  return Vec128<int16_t, N>(_mm_set1_epi16(t));
}
template <size_t N, SIMD_IF128(int32_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> set1(Desc<int32_t, N> /* tag */,
                                                   const int32_t t) {
  return Vec128<int32_t, N>(_mm_set1_epi32(t));
}
template <size_t N, SIMD_IF128(int64_t, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, N> set1(Desc<int64_t, N> /* tag */,
                                                   const int64_t t) {
  return Vec128<int64_t, N>(_mm_set1_epi64x(t));
}
template <size_t N, SIMD_IF128(float, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> set1(Desc<float, N> /* tag */,
                                                 const float t) {
  return Vec128<float, N>(_mm_set1_ps(t));
}
template <size_t N, SIMD_IF128(double, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> set1(Desc<double, N> /* tag */,
                                                  const double t) {
  return Vec128<double, N>(_mm_set1_pd(t));
}

// Returns a vector with lane i=[0, N) set to "first" + i. Unique per-lane
// values are required to detect lane-crossing bugs.
template <typename T, size_t N, typename T2, SIMD_IF128(T, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> iota(Desc<T, N> d, const T2 first) {
  SIMD_ALIGN T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T, size_t N, SIMD_IF128(T, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> undefined(Desc<T, N> /* tag */) {
#ifdef __clang__
  return Vec128<T, N>(_mm_undefined_si128());
#else
  __m128i raw;
  return Vec128<T, N>(raw);
#endif
}
template <size_t N, SIMD_IF128(float, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> undefined(
    Desc<float, N> /* tag */) {
#ifdef __clang__
  return Vec128<float, N>(_mm_undefined_ps());
#else
  __m128 raw;
  return Vec128<float, N>(raw);
#endif
}
template <size_t N, SIMD_IF128(double, N)>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> undefined(
    Desc<double, N> /* tag */) {
#ifdef __clang__
  return Vec128<double, N>(_mm_undefined_pd());
#else
  __m128d raw;
  return Vec128<double, N>(raw);
#endif
}

SIMD_DIAGNOSTICS(pop)

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> operator+(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_add_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> operator+(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_add_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> operator+(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(_mm_add_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> operator+(
    const Vec128<uint64_t, N> a, const Vec128<uint64_t, N> b) {
  return Vec128<uint64_t, N>(_mm_add_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> operator+(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_add_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> operator+(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_add_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> operator+(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_add_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, N> operator+(
    const Vec128<int64_t, N> a, const Vec128<int64_t, N> b) {
  return Vec128<int64_t, N>(_mm_add_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator+(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_add_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator+(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_add_pd(a.raw, b.raw));
}

// ------------------------------ Subtraction

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> operator-(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_sub_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> operator-(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_sub_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> operator-(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(_mm_sub_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> operator-(
    const Vec128<uint64_t, N> a, const Vec128<uint64_t, N> b) {
  return Vec128<uint64_t, N>(_mm_sub_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> operator-(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_sub_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> operator-(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_sub_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> operator-(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_sub_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, N> operator-(
    const Vec128<int64_t, N> a, const Vec128<int64_t, N> b) {
  return Vec128<int64_t, N>(_mm_sub_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator-(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_sub_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator-(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_sub_pd(a.raw, b.raw));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> saturated_add(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_adds_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> saturated_add(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_adds_epu16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> saturated_add(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_adds_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> saturated_add(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_adds_epi16(a.raw, b.raw));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> saturated_subtract(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_subs_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> saturated_subtract(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_subs_epu16(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> saturated_subtract(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_subs_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> saturated_subtract(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_subs_epi16(a.raw, b.raw));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> average_round(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_avg_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> average_round(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_avg_epu16(a.raw, b.raw));
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> abs(const Vec128<int8_t, N> v) {
  return Vec128<int8_t, N>(_mm_abs_epi8(v.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> abs(const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>(_mm_abs_epi16(v.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> abs(const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>(_mm_abs_epi32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> shift_left(
    const Vec128<uint16_t, N> v) {
  return Vec128<uint16_t, N>(_mm_slli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> shift_right(
    const Vec128<uint16_t, N> v) {
  return Vec128<uint16_t, N>(_mm_srli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> shift_left(
    const Vec128<uint32_t, N> v) {
  return Vec128<uint32_t, N>(_mm_slli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> shift_right(
    const Vec128<uint32_t, N> v) {
  return Vec128<uint32_t, N>(_mm_srli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> shift_left(
    const Vec128<uint64_t, N> v) {
  return Vec128<uint64_t, N>(_mm_slli_epi64(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> shift_right(
    const Vec128<uint64_t, N> v) {
  return Vec128<uint64_t, N>(_mm_srli_epi64(v.raw, kBits));
}

// Signed (no i64 shift_right)
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> shift_left(
    const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>(_mm_slli_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> shift_right(
    const Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>(_mm_srai_epi16(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> shift_left(
    const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>(_mm_slli_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> shift_right(
    const Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>(_mm_srai_epi32(v.raw, kBits));
}
template <int kBits, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, N> shift_left(
    const Vec128<int64_t, N> v) {
  return Vec128<int64_t, N>(_mm_slli_epi64(v.raw, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

// Returned by set_shift_*_count; opaque.
template <typename T>
struct ShiftLeftCount128 {
  __m128i raw;
};

template <typename T>
struct ShiftRightCount128 {
  __m128i raw;
};

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE ShiftLeftCount128<T> set_shift_left_count(
    Full128<T> /* tag */, const int bits) {
  return ShiftLeftCount128<T>{_mm_cvtsi32_si128(bits)};
}

// Same as ShiftLeftCount128 on x86, but different on ARM.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE ShiftRightCount128<T> set_shift_right_count(
    Full128<T> /* tag */, const int bits) {
  return ShiftRightCount128<T>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> shift_left_same(
    const Vec128<uint16_t, N> v, const ShiftLeftCount128<uint16_t> bits) {
  return Vec128<uint16_t, N>(_mm_sll_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> shift_right_same(
    const Vec128<uint16_t, N> v, const ShiftRightCount128<uint16_t> bits) {
  return Vec128<uint16_t, N>(_mm_srl_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> shift_left_same(
    const Vec128<uint32_t, N> v, const ShiftLeftCount128<uint32_t> bits) {
  return Vec128<uint32_t, N>(_mm_sll_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> shift_right_same(
    const Vec128<uint32_t, N> v, const ShiftRightCount128<uint32_t> bits) {
  return Vec128<uint32_t, N>(_mm_srl_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> shift_left_same(
    const Vec128<uint64_t, N> v, const ShiftLeftCount128<uint64_t> bits) {
  return Vec128<uint64_t, N>(_mm_sll_epi64(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> shift_right_same(
    const Vec128<uint64_t, N> v, const ShiftRightCount128<uint64_t> bits) {
  return Vec128<uint64_t, N>(_mm_srl_epi64(v.raw, bits.raw));
}

// Signed (no i8,i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> shift_left_same(
    const Vec128<int16_t, N> v, const ShiftLeftCount128<int16_t> bits) {
  return Vec128<int16_t, N>(_mm_sll_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> shift_right_same(
    const Vec128<int16_t, N> v, const ShiftRightCount128<int16_t> bits) {
  return Vec128<int16_t, N>(_mm_sra_epi16(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> shift_left_same(
    const Vec128<int32_t, N> v, const ShiftLeftCount128<int32_t> bits) {
  return Vec128<int32_t, N>(_mm_sll_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> shift_right_same(
    const Vec128<int32_t, N> v, const ShiftRightCount128<int32_t> bits) {
  return Vec128<int32_t, N>(_mm_sra_epi32(v.raw, bits.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, N> shift_left_same(
    const Vec128<int64_t, N> v, const ShiftLeftCount128<int64_t> bits) {
  return Vec128<int64_t, N>(_mm_sll_epi64(v.raw, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsupported.

// ------------------------------ Minimum

// Unsigned (no u64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> min(const Vec128<uint8_t, N> a,
                                                  const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_min_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> min(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_min_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> min(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(_mm_min_epu32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> min(const Vec128<int8_t, N> a,
                                                 const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_min_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> min(const Vec128<int16_t, N> a,
                                                  const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_min_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> min(const Vec128<int32_t, N> a,
                                                  const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_min_epi32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> min(const Vec128<float, N> a,
                                                const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_min_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> min(const Vec128<double, N> a,
                                                 const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_min_pd(a.raw, b.raw));
}

// ------------------------------ Maximum

// Unsigned (no u64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> max(const Vec128<uint8_t, N> a,
                                                  const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_max_epu8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> max(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_max_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> max(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(_mm_max_epu32(a.raw, b.raw));
}

// Signed (no i64)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> max(const Vec128<int8_t, N> a,
                                                 const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_max_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> max(const Vec128<int16_t, N> a,
                                                  const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_max_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> max(const Vec128<int32_t, N> a,
                                                  const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_max_epi32(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> max(const Vec128<float, N> a,
                                                const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_max_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> max(const Vec128<double, N> a,
                                                 const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_max_pd(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> clamp(const Vec128<T, N> v,
                                              const Vec128<T, N> lo,
                                              const Vec128<T, N> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> operator*(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_mullo_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> operator*(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(_mm_mullo_epi32(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> operator*(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_mullo_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> operator*(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_mullo_epi32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> mul_high(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_mulhi_epu16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> mul_high(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_mulhi_epi16(a.raw, b.raw));
}

}  // namespace ext

// Returns (((a * b) >> 14) + 1) >> 1.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> mul_high_round(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_mulhrs_epi16(a.raw, b.raw));
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> mul_even(const Vec128<int32_t> a,
                                                    const Vec128<int32_t> b) {
  return Vec128<int64_t>(_mm_mul_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> mul_even(const Vec128<uint32_t> a,
                                                     const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(_mm_mul_epu32(a.raw, b.raw));
}

// ------------------------------ Floating-point negate

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> neg(const Vec128<float, N> v) {
  const Desc<float, N> df;
  const Desc<uint32_t, N> du;
  const auto sign = bit_cast(df, set1(du, 0x80000000u));
  return v ^ sign;
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> neg(const Vec128<double, N> v) {
  const Desc<double, N> df;
  const Desc<uint64_t, N> du;
  const auto sign = bit_cast(df, set1(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator*(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_mul_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> operator*(
    const Vec128<float, 1> a, const Vec128<float, 1> b) {
  return Vec128<float, 1>(_mm_mul_ss(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator*(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_mul_pd(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, 1> operator*(
    const Vec128<double, 1> a, const Vec128<double, 1> b) {
  return Vec128<double, 1>(_mm_mul_sd(a.raw, b.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator/(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_div_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> operator/(
    const Vec128<float, 1> a, const Vec128<float, 1> b) {
  return Vec128<float, 1>(_mm_div_ss(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator/(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_div_pd(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, 1> operator/(
    const Vec128<double, 1> a, const Vec128<double, 1> b) {
  return Vec128<double, 1>(_mm_div_sd(a.raw, b.raw));
}

// Approximate reciprocal
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> approximate_reciprocal(
    const Vec128<float, N> v) {
  return Vec128<float, N>(_mm_rcp_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> approximate_reciprocal(
    const Vec128<float, 1> v) {
  return Vec128<float, 1>(_mm_rcp_ss(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> mul_add(
    const Vec128<float, N> mul, const Vec128<float, N> x,
    const Vec128<float, N> add) {
  return mul * x + add;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> mul_add(
    const Vec128<double, N> mul, const Vec128<double, N> x,
    const Vec128<double, N> add) {
  return mul * x + add;
}

// Returns add - mul * x
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> nmul_add(
    const Vec128<float, N> mul, const Vec128<float, N> x,
    const Vec128<float, N> add) {
  return add - mul * x;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> nmul_add(
    const Vec128<double, N> mul, const Vec128<double, N> x,
    const Vec128<double, N> add) {
  return add - mul * x;
}

// Returns x + add
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> fadd(Vec128<float, N> x,
                                                 const Vec128<float, N> k1,
                                                 const Vec128<float, N> add) {
  return x + add;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> fadd(Vec128<double, N> x,
                                                  const Vec128<double, N> k1,
                                                  const Vec128<double, N> add) {
  return x + add;
}

// Returns x - sub
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> fsub(Vec128<float, N> x,
                                                 const Vec128<float, N> k1,
                                                 const Vec128<float, N> sub) {
  return x - sub;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> fsub(Vec128<double, N> x,
                                                  const Vec128<double, N> k1,
                                                  const Vec128<double, N> sub) {
  return x - sub;
}

// Returns -sub + x (clobbers sub register)
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> fnadd(Vec128<float, N> sub,
                                                  const Vec128<float, N> k1,
                                                  const Vec128<float, N> x) {
  return x - sub;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> fnadd(Vec128<double, N> sub,
                                                   const Vec128<double, N> k1,
                                                   const Vec128<double, N> x) {
  return x - sub;
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> mul_subtract(
    const Vec128<float, N> mul, const Vec128<float, N> x,
    const Vec128<float, N> sub) {
  return mul * x - sub;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> mul_subtract(
    const Vec128<double, N> mul, const Vec128<double, N> x,
    const Vec128<double, N> sub) {
  return mul * x - sub;
}

// Returns -mul * x - sub
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> nmul_subtract(
    const Vec128<float, N> mul, const Vec128<float, N> x,
    const Vec128<float, N> sub) {
  return neg(mul) * x - sub;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> nmul_subtract(
    const Vec128<double, N> mul, const Vec128<double, N> x,
    const Vec128<double, N> sub) {
  return neg(mul) * x - sub;
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> sqrt(const Vec128<float, N> v) {
  return Vec128<float, N>(_mm_sqrt_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> sqrt(const Vec128<float, 1> v) {
  return Vec128<float, 1>(_mm_sqrt_ss(v.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> sqrt(const Vec128<double, N> v) {
  return Vec128<double, N>(_mm_sqrt_pd(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, 1> sqrt(const Vec128<double, 1> v) {
  return Vec128<double, 1>(_mm_sqrt_sd(_mm_setzero_pd(), v.raw));
}

// Approximate reciprocal square root
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> approximate_reciprocal_sqrt(
    const Vec128<float, N> v) {
  return Vec128<float, N>(_mm_rsqrt_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> approximate_reciprocal_sqrt(
    const Vec128<float, 1> v) {
  return Vec128<float, 1>(_mm_rsqrt_ss(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, ties to even
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> round(const Vec128<float, N> v) {
  return Vec128<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> round(const Vec128<double, N> v) {
  return Vec128<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward zero, aka truncate
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> trunc(const Vec128<float, N> v) {
  return Vec128<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> trunc(const Vec128<double, N> v) {
  return Vec128<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> ceil(const Vec128<float, N> v) {
  return Vec128<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> ceil(const Vec128<double, N> v) {
  return Vec128<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

// Toward -infinity, aka floor
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> floor(const Vec128<float, N> v) {
  return Vec128<float, N>(
      _mm_round_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> floor(const Vec128<double, N> v) {
  return Vec128<double, N>(
      _mm_round_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> operator==(
    const Vec128<uint8_t, N> a, const Vec128<uint8_t, N> b) {
  return Vec128<uint8_t, N>(_mm_cmpeq_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> operator==(
    const Vec128<uint16_t, N> a, const Vec128<uint16_t, N> b) {
  return Vec128<uint16_t, N>(_mm_cmpeq_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, N> operator==(
    const Vec128<uint32_t, N> a, const Vec128<uint32_t, N> b) {
  return Vec128<uint32_t, N>(_mm_cmpeq_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, N> operator==(
    const Vec128<uint64_t, N> a, const Vec128<uint64_t, N> b) {
  return Vec128<uint64_t, N>(_mm_cmpeq_epi64(a.raw, b.raw));
}

// Signed
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> operator==(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_cmpeq_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> operator==(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_cmpeq_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> operator==(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_cmpeq_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, N> operator==(
    const Vec128<int64_t, N> a, const Vec128<int64_t, N> b) {
  return Vec128<int64_t, N>(_mm_cmpeq_epi64(a.raw, b.raw));
}

// Float
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator==(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_cmpeq_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator==(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_cmpeq_pd(a.raw, b.raw));
}

// ------------------------------ Strict inequality

// Signed/float <
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> operator<(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_cmpgt_epi8(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> operator<(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_cmpgt_epi16(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> operator<(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_cmpgt_epi32(b.raw, a.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator<(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_cmplt_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator<(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_cmplt_pd(a.raw, b.raw));
}

// Signed/float >
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> operator>(
    const Vec128<int8_t, N> a, const Vec128<int8_t, N> b) {
  return Vec128<int8_t, N>(_mm_cmpgt_epi8(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> operator>(
    const Vec128<int16_t, N> a, const Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>(_mm_cmpgt_epi16(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> operator>(
    const Vec128<int32_t, N> a, const Vec128<int32_t, N> b) {
  return Vec128<int32_t, N>(_mm_cmpgt_epi32(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator>(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_cmpgt_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator>(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_cmpgt_pd(a.raw, b.raw));
}

// ------------------------------ Weak inequality

// Float <= >=
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator<=(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_cmple_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator<=(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_cmple_pd(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator>=(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_cmpge_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator>=(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_cmpge_pd(a.raw, b.raw));
}

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> operator&(const Vec128<T, N> a,
                                                  const Vec128<T, N> b) {
  return Vec128<T, N>(_mm_and_si128(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator&(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_and_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator&(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_and_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> andnot(const Vec128<T, N> not_mask,
                                               const Vec128<T, N> mask) {
  return Vec128<T, N>(_mm_andnot_si128(not_mask.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> andnot(
    const Vec128<float, N> not_mask, const Vec128<float, N> mask) {
  return Vec128<float, N>(_mm_andnot_ps(not_mask.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> andnot(
    const Vec128<double, N> not_mask, const Vec128<double, N> mask) {
  return Vec128<double, N>(_mm_andnot_pd(not_mask.raw, mask.raw));
}

// ------------------------------ Bitwise OR

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> operator|(const Vec128<T, N> a,
                                                  const Vec128<T, N> b) {
  return Vec128<T, N>(_mm_or_si128(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator|(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_or_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator|(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_or_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise XOR

template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> operator^(const Vec128<T, N> a,
                                                  const Vec128<T, N> b) {
  return Vec128<T, N>(_mm_xor_si128(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> operator^(
    const Vec128<float, N> a, const Vec128<float, N> b) {
  return Vec128<float, N>(_mm_xor_ps(a.raw, b.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> operator^(
    const Vec128<double, N> a, const Vec128<double, N> b) {
  return Vec128<double, N>(_mm_xor_pd(a.raw, b.raw));
}

// ------------------------------ Select/blend

// Returns a mask for use by if_then_else().
// blendv_ps/pd only check the sign bit, so this is a no-op on x86.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> mask_from_sign(const Vec128<T, N> v) {
  return v;
}

// Returns mask ? t : f. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> if_then_else(const Vec128<T, N> mask,
                                                     const Vec128<T, N> yes,
                                                     const Vec128<T, N> no) {
  return Vec128<T, N>(_mm_blendv_epi8(no.raw, yes.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> if_then_else(
    const Vec128<float, N> mask, const Vec128<float, N> yes,
    const Vec128<float, N> no) {
  return Vec128<float, N>(_mm_blendv_ps(no.raw, yes.raw, mask.raw));
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, N> if_then_else(
    const Vec128<double, N> mask, const Vec128<double, N> yes,
    const Vec128<double, N> no) {
  return Vec128<double, N>(_mm_blendv_pd(no.raw, yes.raw, mask.raw));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> load(Full128<T> /* tag */,
                                          const T* SIMD_RESTRICT aligned) {
  return Vec128<T>(_mm_load_si128(reinterpret_cast<const __m128i*>(aligned)));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> load(
    Full128<float> /* tag */, const float* SIMD_RESTRICT aligned) {
  return Vec128<float>(_mm_load_ps(aligned));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> load(
    Full128<double> /* tag */, const double* SIMD_RESTRICT aligned) {
  return Vec128<double>(_mm_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> load_u(Full128<T> /* tag */,
                                            const T* SIMD_RESTRICT p) {
  return Vec128<T>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> load_u(Full128<float> /* tag */,
                                                const float* SIMD_RESTRICT p) {
  return Vec128<float>(_mm_loadu_ps(p));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> load_u(
    Full128<double> /* tag */, const double* SIMD_RESTRICT p) {
  return Vec128<double>(_mm_loadu_pd(p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, 8 / sizeof(T)> load(
    Desc<T, 8 / sizeof(T)> /* tag */, const T* SIMD_RESTRICT p) {
  return Vec128<T, 8 / sizeof(T)>(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 2> load(Desc<float, 2> /* tag */,
                                                 const float* SIMD_RESTRICT p) {
  const __m128 hi = _mm_setzero_ps();
  return Vec128<float, 2>(_mm_loadl_pi(hi, reinterpret_cast<const __m64*>(p)));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, 1> load(
    Desc<double, 1> /* tag */, const double* SIMD_RESTRICT p) {
  const __m128d hi = _mm_setzero_pd();
  return Vec128<double, 1>(_mm_loadl_pd(hi, p));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, 4 / sizeof(T)> load(
    Desc<T, 4 / sizeof(T)> /* tag */, const T* SIMD_RESTRICT p) {
  // TODO(janwas): load_ss?
  int32_t bits;
  CopyBytes<4>(p, &bits);
  return Vec128<T, 4 / sizeof(T)>(_mm_cvtsi32_si128(bits));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> load(Desc<float, 1> /* tag */,
                                                 const float* SIMD_RESTRICT p) {
  return Vec128<float, 1>(_mm_load_ss(p));
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> load_dup128(
    Full128<T> d, const T* const SIMD_RESTRICT p) {
  return load_u(d, p);
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<T> v, Full128<T> /* tag */,
                                      T* SIMD_RESTRICT aligned) {
  _mm_store_si128(reinterpret_cast<__m128i*>(aligned), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<float> v,
                                      Full128<float> /* tag */,
                                      float* SIMD_RESTRICT aligned) {
  _mm_store_ps(aligned, v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<double> v,
                                      Full128<double> /* tag */,
                                      double* SIMD_RESTRICT aligned) {
  _mm_store_pd(aligned, v.raw);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store_u(const Vec128<T> v, Full128<T> /* tag */,
                                        T* SIMD_RESTRICT p) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store_u(const Vec128<float> v,
                                        Full128<float> /* tag */,
                                        float* SIMD_RESTRICT p) {
  _mm_storeu_ps(p, v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store_u(const Vec128<double> v,
                                        Full128<double> /* tag */,
                                        double* SIMD_RESTRICT p) {
  _mm_storeu_pd(p, v.raw);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<T, 8 / sizeof(T)> v,
                                      Desc<T, 8 / sizeof(T)> /* tag */,
                                      T* SIMD_RESTRICT p) {
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<float, 2> v,
                                      Desc<float, 2> /* tag */,
                                      float* SIMD_RESTRICT p) {
  _mm_storel_pi(reinterpret_cast<__m64*>(p), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<double, 1> v,
                                      Desc<double, 1> /* tag */,
                                      double* SIMD_RESTRICT p) {
  _mm_storel_pd(p, v.raw);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<T, 4 / sizeof(T)> v,
                                      Desc<T, 4 / sizeof(T)> /* tag */,
                                      T* SIMD_RESTRICT p) {
  // _mm_storeu_si32 is documented but unavailable in Clang; CopyBytes generates
  // bad code; type-punning is unsafe; this actually generates MOVD.
  _mm_store_ss(reinterpret_cast<float * SIMD_RESTRICT>(p),
               _mm_castsi128_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE void store(const Vec128<float, 1> v,
                                      Desc<float, 1> /* tag */,
                                      float* SIMD_RESTRICT p) {
  _mm_store_ss(p, v.raw);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const Vec128<T> v, Full128<T> /* tag */,
                                       T* SIMD_RESTRICT aligned) {
  _mm_stream_si128(reinterpret_cast<__m128i*>(aligned), v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const Vec128<float> v,
                                       Full128<float> /* tag */,
                                       float* SIMD_RESTRICT aligned) {
  _mm_stream_ps(aligned, v.raw);
}
SIMD_ATTR_SSE4 SIMD_INLINE void stream(const Vec128<double> v,
                                       Full128<double> /* tag */,
                                       double* SIMD_RESTRICT aligned) {
  _mm_stream_pd(aligned, v.raw);
}

// ------------------------------ Gather

// Unsupported.

// ================================================== SWIZZLE

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> shift_left_bytes(const Vec128<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  return Vec128<T>(_mm_slli_si128(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> shift_left_lanes(const Vec128<T> v) {
  return shift_left_bytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> shift_right_bytes(const Vec128<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  return Vec128<T>(_mm_srli_si128(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> shift_right_lanes(const Vec128<T> v) {
  return shift_right_bytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> combine_shift_right_bytes(
    const Vec128<T> hi, const Vec128<T> lo) {
  const Full128<uint8_t> d8;
  const Vec128<uint8_t> extracted_bytes(
      _mm_alignr_epi8(bit_cast(d8, hi).raw, bit_cast(d8, lo).raw, kBytes));
  return bit_cast(Full128<T>(), extracted_bytes);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> broadcast(
    const Vec128<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m128i lo = _mm_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec128<uint16_t>(_mm_unpacklo_epi64(lo, lo));
  } else {
    const __m128i hi = _mm_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec128<uint16_t>(_mm_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> broadcast(
    const Vec128<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<uint32_t>(_mm_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> broadcast(
    const Vec128<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<uint64_t>(_mm_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> broadcast(const Vec128<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m128i lo = _mm_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec128<int16_t>(_mm_unpacklo_epi64(lo, lo));
  } else {
    const __m128i hi = _mm_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec128<int16_t>(_mm_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> broadcast(const Vec128<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<int32_t>(_mm_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> broadcast(const Vec128<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<int64_t>(_mm_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> broadcast(const Vec128<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec128<float>(_mm_shuffle_ps(v.raw, v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> broadcast(const Vec128<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec128<double>(_mm_shuffle_pd(v.raw, v.raw, 3 * kLane));
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> table_lookup_bytes(const Vec128<T> bytes,
                                                        const Vec128<TI> from) {
  return Vec128<T>(_mm_shuffle_epi8(bytes.raw, from.raw));
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec128<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// shuffle_0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// combine_shift_right_bytes but the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> shuffle_1032(
    const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> shuffle_1032(
    const Vec128<int32_t> v) {
  return Vec128<int32_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> shuffle_1032(const Vec128<float> v) {
  return Vec128<float>(_mm_shuffle_ps(v.raw, v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> shuffle_01(
    const Vec128<uint64_t> v) {
  return Vec128<uint64_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> shuffle_01(const Vec128<int64_t> v) {
  return Vec128<int64_t>(_mm_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> shuffle_01(const Vec128<double> v) {
  return Vec128<double>(_mm_shuffle_pd(v.raw, v.raw, 1));
}

// Rotate right 32 bits
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> shuffle_0321(
    const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(_mm_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> shuffle_0321(
    const Vec128<int32_t> v) {
  return Vec128<int32_t>(_mm_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> shuffle_0321(const Vec128<float> v) {
  return Vec128<float>(_mm_shuffle_ps(v.raw, v.raw, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> shuffle_2103(
    const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(_mm_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> shuffle_2103(
    const Vec128<int32_t> v) {
  return Vec128<int32_t>(_mm_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> shuffle_2103(const Vec128<float> v) {
  return Vec128<float>(_mm_shuffle_ps(v.raw, v.raw, 0x93));
}

// Reverse
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> shuffle_0123(
    const Vec128<uint32_t> v) {
  return Vec128<uint32_t>(_mm_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> shuffle_0123(
    const Vec128<int32_t> v) {
  return Vec128<int32_t>(_mm_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> shuffle_0123(const Vec128<float> v) {
  return Vec128<float>(_mm_shuffle_ps(v.raw, v.raw, 0x1B));
}

// ------------------------------ Permute (runtime variable)

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct permute_sse4 {
  __m128i raw;
};

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE permute_sse4<T> set_table_indices(
    const Full128<T> d, const int32_t* idx) {
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
  return permute_sse4<T>{load(d8, control).raw};
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> table_lookup_lanes(
    const Vec128<uint32_t> v, const permute_sse4<uint32_t> idx) {
  return table_lookup_bytes(v, Vec128<uint8_t>(idx.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> table_lookup_lanes(
    const Vec128<int32_t> v, const permute_sse4<int32_t> idx) {
  return table_lookup_bytes(v, Vec128<uint8_t>(idx.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> table_lookup_lanes(
    const Vec128<float> v, const permute_sse4<float> idx) {
  const Full128<int32_t> di;
  const Full128<float> df;
  return bit_cast(
      df, table_lookup_bytes(bit_cast(di, v), Vec128<uint8_t>(idx.raw)));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t> interleave_lo(
    const Vec128<uint8_t> a, const Vec128<uint8_t> b) {
  return Vec128<uint8_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> interleave_lo(
    const Vec128<uint16_t> a, const Vec128<uint16_t> b) {
  return Vec128<uint16_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> interleave_lo(
    const Vec128<uint32_t> a, const Vec128<uint32_t> b) {
  return Vec128<uint32_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> interleave_lo(
    const Vec128<uint64_t> a, const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(_mm_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t> interleave_lo(
    const Vec128<int8_t> a, const Vec128<int8_t> b) {
  return Vec128<int8_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> interleave_lo(
    const Vec128<int16_t> a, const Vec128<int16_t> b) {
  return Vec128<int16_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> interleave_lo(
    const Vec128<int32_t> a, const Vec128<int32_t> b) {
  return Vec128<int32_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> interleave_lo(
    const Vec128<int64_t> a, const Vec128<int64_t> b) {
  return Vec128<int64_t>(_mm_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> interleave_lo(const Vec128<float> a,
                                                       const Vec128<float> b) {
  return Vec128<float>(_mm_unpacklo_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> interleave_lo(
    const Vec128<double> a, const Vec128<double> b) {
  return Vec128<double>(_mm_unpacklo_pd(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t> interleave_hi(
    const Vec128<uint8_t> a, const Vec128<uint8_t> b) {
  return Vec128<uint8_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> interleave_hi(
    const Vec128<uint16_t> a, const Vec128<uint16_t> b) {
  return Vec128<uint16_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> interleave_hi(
    const Vec128<uint32_t> a, const Vec128<uint32_t> b) {
  return Vec128<uint32_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> interleave_hi(
    const Vec128<uint64_t> a, const Vec128<uint64_t> b) {
  return Vec128<uint64_t>(_mm_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t> interleave_hi(
    const Vec128<int8_t> a, const Vec128<int8_t> b) {
  return Vec128<int8_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> interleave_hi(
    const Vec128<int16_t> a, const Vec128<int16_t> b) {
  return Vec128<int16_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> interleave_hi(
    const Vec128<int32_t> a, const Vec128<int32_t> b) {
  return Vec128<int32_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> interleave_hi(
    const Vec128<int64_t> a, const Vec128<int64_t> b) {
  return Vec128<int64_t>(_mm_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> interleave_hi(const Vec128<float> a,
                                                       const Vec128<float> b) {
  return Vec128<float>(_mm_unpackhi_ps(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> interleave_hi(
    const Vec128<double> a, const Vec128<double> b) {
  return Vec128<double>(_mm_unpackhi_pd(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> zip_lo(const Vec128<uint8_t> a,
                                                   const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> zip_lo(const Vec128<uint16_t> a,
                                                   const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> zip_lo(const Vec128<uint32_t> a,
                                                   const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> zip_lo(const Vec128<int8_t> a,
                                                  const Vec128<int8_t> b) {
  return Vec128<int16_t>(_mm_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> zip_lo(const Vec128<int16_t> a,
                                                  const Vec128<int16_t> b) {
  return Vec128<int32_t>(_mm_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> zip_lo(const Vec128<int32_t> a,
                                                  const Vec128<int32_t> b) {
  return Vec128<int64_t>(_mm_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> zip_hi(const Vec128<uint8_t> a,
                                                   const Vec128<uint8_t> b) {
  return Vec128<uint16_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> zip_hi(const Vec128<uint16_t> a,
                                                   const Vec128<uint16_t> b) {
  return Vec128<uint32_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> zip_hi(const Vec128<uint32_t> a,
                                                   const Vec128<uint32_t> b) {
  return Vec128<uint64_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> zip_hi(const Vec128<int8_t> a,
                                                  const Vec128<int8_t> b) {
  return Vec128<int16_t>(_mm_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> zip_hi(const Vec128<int16_t> a,
                                                  const Vec128<int16_t> b) {
  return Vec128<int32_t>(_mm_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> zip_hi(const Vec128<int32_t> a,
                                                  const Vec128<int32_t> b) {
  return Vec128<int64_t>(_mm_unpackhi_epi32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns a part with value "t".
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, 1> set_lane(const uint16_t t) {
  return Vec128<uint16_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, 1> set_lane(const int16_t t) {
  return Vec128<int16_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t, 1> set_lane(const uint32_t t) {
  return Vec128<uint32_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, 1> set_lane(const int32_t t) {
  return Vec128<int32_t, 1>(_mm_cvtsi32_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 1> set_lane(const float t) {
  return Vec128<float, 1>(_mm_set_ss(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t, 1> set_lane(const uint64_t t) {
  return Vec128<uint64_t, 1>(_mm_cvtsi64_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t, 1> set_lane(const int64_t t) {
  return Vec128<int64_t, 1>(_mm_cvtsi64_si128(t));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, 1> set_lane(const double t) {
  return Vec128<double, 1>(_mm_set_sd(t));
}

// Gets the single value stored in a vector/part.
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint16_t get_lane(const Vec128<uint16_t, N> v) {
  return _mm_cvtsi128_si32(v.raw) & 0xFFFF;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int16_t get_lane(const Vec128<int16_t, N> v) {
  return _mm_cvtsi128_si32(v.raw) & 0xFFFF;
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint32_t get_lane(const Vec128<uint32_t, N> v) {
  return _mm_cvtsi128_si32(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int32_t get_lane(const Vec128<int32_t, N> v) {
  return _mm_cvtsi128_si32(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE float get_lane(const Vec128<float, N> v) {
  return _mm_cvtss_f32(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE uint64_t get_lane(const Vec128<uint64_t, N> v) {
  return _mm_cvtsi128_si64(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE int64_t get_lane(const Vec128<int64_t, N> v) {
  return _mm_cvtsi128_si64(v.raw);
}
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE double get_lane(const Vec128<double, N> v) {
  return _mm_cvtsd_f64(v.raw);
}

// Returns part of a vector (unspecified whether upper or lower).
template <typename T, size_t N, size_t VN>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> any_part(Desc<T, N> /* tag */,
                                                 const Vec128<T, VN> v) {
  return Vec128<T, N>(v.raw);
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> broadcast_part(Full128<T> /* tag */,
                                                    const Vec128<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return broadcast<kLane>(Vec128<T>(v.raw));
}

// Returns upper/lower half of a vector.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, 8 / sizeof(T)> lower_half(Vec128<T> v) {
  return Vec128<T, 8 / sizeof(T)>(v.raw);
}

// These copy hi into lo (smaller instruction encoding than shifts).
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, 8 / sizeof(T)> upper_half(Vec128<T> v) {
  return Vec128<T, 8 / sizeof(T)>(_mm_unpackhi_epi64(v.raw, v.raw));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, 2> upper_half(Vec128<float> v) {
  return Vec128<float, 2>(_mm_movehl_ps(v.raw, v.raw));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double, 1> upper_half(Vec128<double> v) {
  return Vec128<double, 1>(_mm_unpackhi_pd(v.raw, v.raw));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, 8 / sizeof(T)> get_half(Lower /* tag */,
                                                             Vec128<T> v) {
  return lower_half(v);
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, 8 / sizeof(T)> get_half(Upper /* tag */,
                                                             Vec128<T> v) {
  return upper_half(v);
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> concat_lo_lo(const Vec128<T> hi,
                                                  const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return bit_cast(Full128<T>(),
                  interleave_lo(bit_cast(d64, lo), bit_cast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> concat_hi_hi(const Vec128<T> hi,
                                                  const Vec128<T> lo) {
  const Full128<uint64_t> d64;
  return bit_cast(Full128<T>(),
                  interleave_hi(bit_cast(d64, lo), bit_cast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> concat_lo_hi(const Vec128<T> hi,
                                                  const Vec128<T> lo) {
  return combine_shift_right_bytes<8>(hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> concat_hi_lo(const Vec128<T> hi,
                                                  const Vec128<T> lo) {
  return Vec128<T>(_mm_blend_epi16(hi.raw, lo.raw, 0x0F));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> concat_hi_lo(const Vec128<float> hi,
                                                      const Vec128<float> lo) {
  return Vec128<float>(_mm_blend_ps(hi.raw, lo.raw, 3));
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> concat_hi_lo(
    const Vec128<double> hi, const Vec128<double> lo) {
  return Vec128<double>(_mm_blend_pd(hi.raw, lo.raw, 1));
}

// ------------------------------ Odd/even lanes

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> odd_even_impl(SizeTag<1> /* tag */,
                                                   const Vec128<T> a,
                                                   const Vec128<T> b) {
  const Full128<T> d;
  const Full128<uint8_t> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                           0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return if_then_else(bit_cast(d, load(d8, mask)), b, a);
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> odd_even_impl(SizeTag<2> /* tag */,
                                                   const Vec128<T> a,
                                                   const Vec128<T> b) {
  return Vec128<T>(_mm_blend_epi16(a.raw, b.raw, 0x55));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> odd_even_impl(SizeTag<4> /* tag */,
                                                   const Vec128<T> a,
                                                   const Vec128<T> b) {
  return Vec128<T>(_mm_blend_epi16(a.raw, b.raw, 0x33));
}
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> odd_even_impl(SizeTag<8> /* tag */,
                                                   const Vec128<T> a,
                                                   const Vec128<T> b) {
  return Vec128<T>(_mm_blend_epi16(a.raw, b.raw, 0x0F));
}

template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T> odd_even(const Vec128<T> a,
                                              const Vec128<T> b) {
  return odd_even_impl(SizeTag<sizeof(T)>(), a, b);
}
template <>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float> odd_even<float>(
    const Vec128<float> a, const Vec128<float> b) {
  return Vec128<float>(_mm_blend_ps(a.raw, b.raw, 5));
}

template <>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> odd_even<double>(
    const Vec128<double> a, const Vec128<double> b) {
  return Vec128<double>(_mm_blend_pd(a.raw, b.raw, 1));
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<double> convert_to(Full128<double> /* tag */,
                                                     const Vec128<float, 2> v) {
  return Vec128<double>(_mm_cvtps_pd(v.raw));
}

// Unsigned: zero-extend.
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> convert_to(
    Full128<uint16_t> /* tag */, const Vec128<uint8_t, 8> v) {
  return Vec128<uint16_t>(_mm_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> convert_to(
    Full128<uint32_t> /* tag */, const Vec128<uint8_t, 4> v) {
  return Vec128<uint32_t>(_mm_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> convert_to(
    Full128<int16_t> /* tag */, const Vec128<uint8_t, 8> v) {
  return Vec128<int16_t>(_mm_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> convert_to(
    Full128<int32_t> /* tag */, const Vec128<uint8_t, 4> v) {
  return Vec128<int32_t>(_mm_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> convert_to(
    Full128<uint32_t> /* tag */, const Vec128<uint16_t, 4> v) {
  return Vec128<uint32_t>(_mm_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> convert_to(
    Full128<int32_t> /* tag */, const Vec128<uint16_t, 4> v) {
  return Vec128<int32_t>(_mm_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> convert_to(
    Full128<uint64_t> /* tag */, const Vec128<uint32_t, 2> v) {
  return Vec128<uint64_t>(_mm_cvtepu32_epi64(v.raw));
}

SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint32_t> u32_from_u8(
    const Vec128<uint8_t> v) {
  return Vec128<uint32_t>(_mm_cvtepu8_epi32(v.raw));
}

// Signed: replicate sign bit.
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t> convert_to(
    Full128<int16_t> /* tag */, const Vec128<int8_t, 8> v) {
  return Vec128<int16_t>(_mm_cvtepi8_epi16(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> convert_to(
    Full128<int32_t> /* tag */, const Vec128<int8_t, 4> v) {
  return Vec128<int32_t>(_mm_cvtepi8_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t> convert_to(
    Full128<int32_t> /* tag */, const Vec128<int16_t, 4> v) {
  return Vec128<int32_t>(_mm_cvtepi16_epi32(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int64_t> convert_to(
    Full128<int64_t> /* tag */, const Vec128<int32_t, 2> v) {
  return Vec128<int64_t>(_mm_cvtepi32_epi64(v.raw));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t, N> convert_to(
    Desc<uint16_t, N> /* tag */, const Vec128<int32_t, N> v) {
  return Vec128<uint16_t, N>(_mm_packus_epi32(v.raw, v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> convert_to(
    Desc<uint8_t, N> /* tag */, const Vec128<int32_t> v) {
  const __m128i u16 = _mm_packus_epi32(v.raw, v.raw);
  return Vec128<uint8_t, N>(_mm_packus_epi16(u16, u16));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, N> convert_to(
    Desc<uint8_t, N> /* tag */, const Vec128<int16_t> v) {
  return Vec128<uint8_t, N>(_mm_packus_epi16(v.raw, v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int16_t, N> convert_to(
    Desc<int16_t, N> /* tag */, const Vec128<int32_t> v) {
  return Vec128<int16_t, N>(_mm_packs_epi32(v.raw, v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> convert_to(
    Desc<int8_t, N> /* tag */, const Vec128<int32_t> v) {
  const __m128i i16 = _mm_packs_epi32(v.raw, v.raw);
  return Vec128<int8_t, N>(_mm_packs_epi16(i16, i16));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int8_t, N> convert_to(
    Desc<int8_t, N> /* tag */, const Vec128<int16_t> v) {
  return Vec128<int8_t, N>(_mm_packs_epi16(v.raw, v.raw));
}

// For already range-limited input [0, 255].
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint8_t, 4> u8_from_u32(
    const Vec128<uint32_t> v) {
  const Full128<uint32_t> d32;
  const Full128<uint8_t> d8;
  SIMD_ALIGN static constexpr uint32_t k8From32[4] = {0x0C080400u, 0x0C080400u,
                                                      0x0C080400u, 0x0C080400u};
  // Replicate bytes into all 32 bit lanes for any_part.
  const auto quad = table_lookup_bytes(v, load(d32, k8From32));
  return any_part(Desc<uint8_t, 4>(), bit_cast(d8, quad));
}

// ------------------------------ Convert i32 <=> f32

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<float, N> convert_to(
    Desc<float, N> /* tag */, const Vec128<int32_t, N> v) {
  return Vec128<float, N>(_mm_cvtepi32_ps(v.raw));
}
// Truncates (rounds toward zero).
template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> convert_to(
    Desc<int32_t, N> /* tag */, const Vec128<float, N> v) {
  return Vec128<int32_t, N>(_mm_cvttps_epi32(v.raw));
}

template <size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<int32_t, N> nearest_int(
    const Vec128<float, N> v) {
  return Vec128<int32_t, N>(_mm_cvtps_epi32(v.raw));
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_SSE4 SIMD_INLINE uint64_t movemask(const Vec128<uint8_t> v) {
  return static_cast<unsigned>(_mm_movemask_epi8(v.raw));
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_SSE4 SIMD_INLINE uint64_t movemask(const Vec128<float> v) {
  return static_cast<unsigned>(_mm_movemask_ps(v.raw));
}
SIMD_ATTR_SSE4 SIMD_INLINE uint64_t movemask(const Vec128<double> v) {
  return static_cast<unsigned>(_mm_movemask_pd(v.raw));
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero. Supported for all integer V.
template <typename T>
SIMD_ATTR_SSE4 SIMD_INLINE bool all_zero(const Vec128<T> v) {
  return static_cast<bool>(_mm_testz_si128(v.raw, v.raw));
}

// ------------------------------ minpos

// Returns index and min value in lanes 1 and 0.
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> minpos(const Vec128<uint16_t> v) {
  return Vec128<uint16_t>(_mm_minpos_epu16(v.raw));
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint64_t> sums_of_u8x8(
    const Vec128<uint8_t> v) {
  return Vec128<uint64_t>(_mm_sad_epu8(v.raw, _mm_setzero_si128()));
}

// Returns N sums of differences of byte quadruplets, starting from byte offset
// i = [0, N) in window (11 consecutive bytes) and idx_ref * 4 in ref.
template <int idx_ref>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<uint16_t> mpsadbw(
    const Vec128<uint8_t> window, const Vec128<uint8_t> ref) {
  static_assert(idx_ref < 4, "a_offset must be 0");
  return Vec128<uint16_t>(_mm_mpsadbw_epu8(window.raw, ref.raw, idx_ref));
}

// For u32/i32/f32.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> horz_sum_impl(
    SizeTag<4> /* tag */, const Vec128<T, N> v3210) {
  const Vec128<T> v1032 = shuffle_1032(v3210);
  const Vec128<T> v31_20_31_20 = v3210 + v1032;
  const Vec128<T> v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

// For u64/i64/f64.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> horz_sum_impl(SizeTag<8> /* tag */,
                                                      const Vec128<T, N> v10) {
  const Vec128<T> v01 = shuffle_01(v10);
  return v10 + v01;
}

// Supported for u/i/f 32/64. Returns the sum in each lane.
template <typename T, size_t N>
SIMD_ATTR_SSE4 SIMD_INLINE Vec128<T, N> sum_of_lanes(const Vec128<T, N> v) {
  return horz_sum_impl(SizeTag<sizeof(T)>(), v);
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).
}  // namespace jxl

#endif  // HIGHWAY_X86_SSE4_H_
