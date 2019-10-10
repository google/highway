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

#ifndef HIGHWAY_X86_AVX512_H_
#define HIGHWAY_X86_AVX512_H_

// 512-bit AVX512 vectors and operations.
// WARNING: most operations do not cross 128-bit block boundaries. In
// particular, "broadcast", pack and zip behavior may be surprising.

#include <immintrin.h>

#include "third_party/highway/highway/shared.h"
#include "third_party/highway/highway/x86_avx2.h"

namespace jxl {

template <typename T>
struct Raw512 {
  using type = __m512i;
};
template <>
struct Raw512<float> {
  using type = __m512;
};
template <>
struct Raw512<double> {
  using type = __m512d;
};

template <typename T>
using Full512 = Desc<T, 64 / sizeof(T)>;

template <typename T>
class Vec512 {
  using Raw = typename Raw512<T>::type;

 public:
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512() = default;
  Vec512(const Vec512&) = default;
  Vec512& operator=(const Vec512&) = default;
  SIMD_ATTR_AVX512 SIMD_INLINE explicit Vec512(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator*=(const Vec512 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator/=(const Vec512 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator+=(const Vec512 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator-=(const Vec512 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator&=(const Vec512 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator|=(const Vec512 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_AVX512 SIMD_INLINE Vec512& operator^=(const Vec512 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// ------------------------------ Cast

SIMD_ATTR_AVX512 SIMD_INLINE __m512i BitCastToInteger(__m512i v) { return v; }
SIMD_ATTR_AVX512 SIMD_INLINE __m512i BitCastToInteger(__m512 v) {
  return _mm512_castps_si512(v);
}
SIMD_ATTR_AVX512 SIMD_INLINE __m512i BitCastToInteger(__m512d v) {
  return _mm512_castpd_si512(v);
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> cast_to_u8(Vec512<T> v) {
  return Vec512<uint8_t>(BitCastToInteger(v.raw));
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromInteger512 {
  SIMD_ATTR_AVX512 SIMD_INLINE __m512i operator()(__m512i v) { return v; }
};
template <>
struct BitCastFromInteger512<float> {
  SIMD_ATTR_AVX512 SIMD_INLINE __m512 operator()(__m512i v) {
    return _mm512_castsi512_ps(v);
  }
};
template <>
struct BitCastFromInteger512<double> {
  SIMD_ATTR_AVX512 SIMD_INLINE __m512d operator()(__m512i v) {
    return _mm512_castsi512_pd(v);
  }
};

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> cast_u8_to(Full512<T> /* tag */,
                                                  Vec512<uint8_t> v) {
  return Vec512<T>(BitCastFromInteger512<T>()(v.raw));
}

template <typename T, typename FromT>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> bit_cast(Full512<T> d, Vec512<FromT> v) {
  return cast_u8_to(d, cast_to_u8(v));
}

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> setzero(Full512<T> /* tag */) {
  return Vec512<T>(_mm512_setzero_si512());
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> setzero(Full512<float> /* tag */) {
  return Vec512<float>(_mm512_setzero_ps());
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> setzero(Full512<double> /* tag */) {
  return Vec512<double>(_mm512_setzero_pd());
}

template <typename T, typename T2>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> iota(Full512<T> d, const T2 first) {
  SIMD_ALIGN T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector with all lanes set to "t".
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> set1(Full512<uint8_t> /* tag */,
                                                  const uint8_t t) {
  return Vec512<uint8_t>(_mm512_set1_epi8(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> set1(Full512<uint16_t> /* tag */,
                                                   const uint16_t t) {
  return Vec512<uint16_t>(_mm512_set1_epi16(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> set1(Full512<uint32_t> /* tag */,
                                                   const uint32_t t) {
  return Vec512<uint32_t>(_mm512_set1_epi32(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> set1(Full512<uint64_t> /* tag */,
                                                   const uint64_t t) {
  return Vec512<uint64_t>(_mm512_set1_epi64(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> set1(Full512<int8_t> /* tag */,
                                                 const int8_t t) {
  return Vec512<int8_t>(_mm512_set1_epi8(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> set1(Full512<int16_t> /* tag */,
                                                  const int16_t t) {
  return Vec512<int16_t>(_mm512_set1_epi16(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> set1(Full512<int32_t> /* tag */,
                                                  const int32_t t) {
  return Vec512<int32_t>(_mm512_set1_epi32(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> set1(Full512<int64_t> /* tag */,
                                                  const int64_t t) {
  return Vec512<int64_t>(_mm512_set1_epi64(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> set1(Full512<float> /* tag */,
                                                const float t) {
  return Vec512<float>(_mm512_set1_ps(t));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> set1(Full512<double> /* tag */,
                                                 const double t) {
  return Vec512<double>(_mm512_set1_pd(t));
}

SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> undefined(Full512<T> /* tag */) {
#ifdef __clang__
  return Vec512<T>(_mm512_undefined_epi32());
#else
  __m512i raw;
  return Vec512<T>(raw);
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> undefined(Full512<float> /* tag */) {
#ifdef __clang__
  return Vec512<float>(_mm512_undefined_ps());
#else
  __m512 raw;
  return Vec512<float>(raw);
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> undefined(
    Full512<double> /* tag */) {
#ifdef __clang__
  return Vec512<double>(_mm512_undefined_pd());
#else
  __m512d raw;
  return Vec512<double>(raw);
#endif
}

SIMD_DIAGNOSTICS(pop)

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> operator&(const Vec512<T> a,
                                                 const Vec512<T> b) {
  return Vec512<T>(_mm512_and_si512(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator&(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_and_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator&(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_and_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> andnot(const Vec512<T> not_mask,
                                              const Vec512<T> mask) {
  return Vec512<T>(_mm512_andnot_si512(not_mask.raw, mask.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> andnot(const Vec512<float> not_mask,
                                                  const Vec512<float> mask) {
  return Vec512<float>(_mm512_andnot_ps(not_mask.raw, mask.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> andnot(
    const Vec512<double> not_mask, const Vec512<double> mask) {
  return Vec512<double>(_mm512_andnot_pd(not_mask.raw, mask.raw));
}

// ------------------------------ Bitwise OR

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> operator|(const Vec512<T> a,
                                                 const Vec512<T> b) {
  return Vec512<T>(_mm512_or_si512(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator|(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_or_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator|(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_or_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise XOR

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> operator^(const Vec512<T> a,
                                                 const Vec512<T> b) {
  return Vec512<T>(_mm512_xor_si512(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator^(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_xor_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator^(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_xor_pd(a.raw, b.raw));
}

// ------------------------------ Select/blend

// Returns a mask for use by if_then_else().
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> mask_from_sign(const Vec512<T> v) {
  // TODO(janwas): use cmp_mask when return type is actually mask.
  return v < setzero(Full512<T>());
}

// Returns mask ? b : a. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> if_then_else(const Vec512<T> mask,
                                                    const Vec512<T> yes,
                                                    const Vec512<T> no) {
  // TODO(janwas): replace with mask
  return (mask & yes) | andnot(mask, no);
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> if_then_else(
    const Vec512<float> mask, const Vec512<float> yes, const Vec512<float> no) {
  return (mask & yes) | andnot(mask, no);
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> if_then_else(
    const Vec512<double> mask, const Vec512<double> yes,
    const Vec512<double> no) {
  return (mask & yes) | andnot(mask, no);
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> operator+(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_add_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> operator+(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_add_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> operator+(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_add_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> operator+(
    const Vec512<uint64_t> a, const Vec512<uint64_t> b) {
  return Vec512<uint64_t>(_mm512_add_epi64(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> operator+(const Vec512<int8_t> a,
                                                      const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_add_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> operator+(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_add_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator+(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_add_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> operator+(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  return Vec512<int64_t>(_mm512_add_epi64(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator+(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_add_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator+(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_add_pd(a.raw, b.raw));
}

// ------------------------------ Subtraction

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> operator-(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_sub_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> operator-(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_sub_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> operator-(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_sub_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> operator-(
    const Vec512<uint64_t> a, const Vec512<uint64_t> b) {
  return Vec512<uint64_t>(_mm512_sub_epi64(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> operator-(const Vec512<int8_t> a,
                                                      const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_sub_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> operator-(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_sub_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator-(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_sub_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> operator-(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  return Vec512<int64_t>(_mm512_sub_epi64(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator-(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_sub_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator-(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_sub_pd(a.raw, b.raw));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> saturated_add(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_adds_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> saturated_add(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_adds_epu16(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> saturated_add(
    const Vec512<int8_t> a, const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_adds_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> saturated_add(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_adds_epi16(a.raw, b.raw));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> saturated_subtract(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_subs_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> saturated_subtract(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_subs_epu16(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> saturated_subtract(
    const Vec512<int8_t> a, const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_subs_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> saturated_subtract(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_subs_epi16(a.raw, b.raw));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> average_round(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_avg_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> average_round(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_avg_epu16(a.raw, b.raw));
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> abs(const Vec512<int8_t> v) {
  return Vec512<int8_t>(_mm512_abs_epi8(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> abs(const Vec512<int16_t> v) {
  return Vec512<int16_t>(_mm512_abs_epi16(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> abs(const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_abs_epi32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> shift_left(
    const Vec512<uint16_t> v) {
  return Vec512<uint16_t>(_mm512_slli_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> shift_right(
    const Vec512<uint16_t> v) {
  return Vec512<uint16_t>(_mm512_srli_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shift_left(
    const Vec512<uint32_t> v) {
  return Vec512<uint32_t>(_mm512_slli_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shift_right(
    const Vec512<uint32_t> v) {
  return Vec512<uint32_t>(_mm512_srli_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> shift_left(
    const Vec512<uint64_t> v) {
  return Vec512<uint64_t>(_mm512_slli_epi64(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> shift_right(
    const Vec512<uint64_t> v) {
  return Vec512<uint64_t>(_mm512_srli_epi64(v.raw, kBits));
}

// Signed (no i64 shift_right)
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> shift_left(
    const Vec512<int16_t> v) {
  return Vec512<int16_t>(_mm512_slli_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> shift_right(
    const Vec512<int16_t> v) {
  return Vec512<int16_t>(_mm512_srai_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shift_left(
    const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_slli_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shift_right(
    const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_srai_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> shift_left(
    const Vec512<int64_t> v) {
  return Vec512<int64_t>(_mm512_slli_epi64(v.raw, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

// Returned by set_shift_*_count; opaque.
template <typename T>
struct ShiftLeftCount512 {
  __m128i raw;
};

template <typename T>
struct ShiftRightCount512 {
  __m128i raw;
};

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE ShiftLeftCount512<T> set_shift_left_count(
    Full512<T> /* tag */, const int bits) {
  return ShiftLeftCount512<T>{_mm_cvtsi32_si128(bits)};
}

// Same as shift_left_count on x86, but different on ARM.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE ShiftRightCount512<T> set_shift_right_count(
    Full512<T> /* tag */, const int bits) {
  return ShiftRightCount512<T>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> shift_left_same(
    const Vec512<uint16_t> v, const ShiftLeftCount512<uint16_t> bits) {
  return Vec512<uint16_t>(_mm512_sll_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> shift_right_same(
    const Vec512<uint16_t> v, const ShiftRightCount512<uint16_t> bits) {
  return Vec512<uint16_t>(_mm512_srl_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shift_left_same(
    const Vec512<uint32_t> v, const ShiftLeftCount512<uint32_t> bits) {
  return Vec512<uint32_t>(_mm512_sll_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shift_right_same(
    const Vec512<uint32_t> v, const ShiftRightCount512<uint32_t> bits) {
  return Vec512<uint32_t>(_mm512_srl_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> shift_left_same(
    const Vec512<uint64_t> v, const ShiftLeftCount512<uint64_t> bits) {
  return Vec512<uint64_t>(_mm512_sll_epi64(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> shift_right_same(
    const Vec512<uint64_t> v, const ShiftRightCount512<uint64_t> bits) {
  return Vec512<uint64_t>(_mm512_srl_epi64(v.raw, bits.raw));
}

// Signed (no i8,i64)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> shift_left_same(
    const Vec512<int16_t> v, const ShiftLeftCount512<int16_t> bits) {
  return Vec512<int16_t>(_mm512_sll_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> shift_right_same(
    const Vec512<int16_t> v, const ShiftRightCount512<int16_t> bits) {
  return Vec512<int16_t>(_mm512_sra_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shift_left_same(
    const Vec512<int32_t> v, const ShiftLeftCount512<int32_t> bits) {
  return Vec512<int32_t>(_mm512_sll_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shift_right_same(
    const Vec512<int32_t> v, const ShiftRightCount512<int32_t> bits) {
  return Vec512<int32_t>(_mm512_sra_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> shift_left_same(
    const Vec512<int64_t> v, const ShiftLeftCount512<int64_t> bits) {
  return Vec512<int64_t>(_mm512_sll_epi64(v.raw, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> operator<<(
    const Vec512<uint32_t> v, const Vec512<uint32_t> bits) {
  return Vec512<uint32_t>(_mm512_sllv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> operator>>(
    const Vec512<uint32_t> v, const Vec512<uint32_t> bits) {
  return Vec512<uint32_t>(_mm512_srlv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> operator<<(
    const Vec512<uint64_t> v, const Vec512<uint64_t> bits) {
  return Vec512<uint64_t>(_mm512_sllv_epi64(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> operator>>(
    const Vec512<uint64_t> v, const Vec512<uint64_t> bits) {
  return Vec512<uint64_t>(_mm512_srlv_epi64(v.raw, bits.raw));
}

// Signed (no i8,i16,i64)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator<<(
    const Vec512<int32_t> v, const Vec512<int32_t> bits) {
  return Vec512<int32_t>(_mm512_sllv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator>>(
    const Vec512<int32_t> v, const Vec512<int32_t> bits) {
  return Vec512<int32_t>(_mm512_srav_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> operator<<(
    const Vec512<int64_t> v, const Vec512<int64_t> bits) {
  return Vec512<int64_t>(_mm512_sllv_epi64(v.raw, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> min(const Vec512<uint8_t> a,
                                                 const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_min_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> min(const Vec512<uint16_t> a,
                                                  const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_min_epu16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> min(const Vec512<uint32_t> a,
                                                  const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_min_epu32(a.raw, b.raw));
}

// Signed (no i64)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> min(const Vec512<int8_t> a,
                                                const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_min_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> min(const Vec512<int16_t> a,
                                                 const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_min_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> min(const Vec512<int32_t> a,
                                                 const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_min_epi32(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> min(const Vec512<float> a,
                                               const Vec512<float> b) {
  return Vec512<float>(_mm512_min_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> min(const Vec512<double> a,
                                                const Vec512<double> b) {
  return Vec512<double>(_mm512_min_pd(a.raw, b.raw));
}

// ------------------------------ Maximum

// Unsigned (no u64)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> max(const Vec512<uint8_t> a,
                                                 const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_max_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> max(const Vec512<uint16_t> a,
                                                  const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_max_epu16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> max(const Vec512<uint32_t> a,
                                                  const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_max_epu32(a.raw, b.raw));
}

// Signed (no i64)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> max(const Vec512<int8_t> a,
                                                const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_max_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> max(const Vec512<int16_t> a,
                                                 const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_max_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> max(const Vec512<int32_t> a,
                                                 const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_max_epi32(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> max(const Vec512<float> a,
                                               const Vec512<float> b) {
  return Vec512<float>(_mm512_max_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> max(const Vec512<double> a,
                                                const Vec512<double> b) {
  return Vec512<double>(_mm512_max_pd(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> clamp(const Vec512<T> v,
                                             const Vec512<T> lo,
                                             const Vec512<T> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> operator*(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_mullo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> operator*(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_mullo_epi32(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> operator*(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_mullo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator*(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_mullo_epi32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> mul_high(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_mulhi_epu16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> mul_high(const Vec512<int16_t> a,
                                                      const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_mulhi_epi16(a.raw, b.raw));
}

}  // namespace ext

// Returns (((a * b) >> 14) + 1) >> 1.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> mul_high_round(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_mulhrs_epi16(a.raw, b.raw));
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> mul_even(const Vec512<int32_t> a,
                                                      const Vec512<int32_t> b) {
  return Vec512<int64_t>(_mm512_mul_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> mul_even(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  return Vec512<uint64_t>(_mm512_mul_epu32(a.raw, b.raw));
}

// ------------------------------ Floating-point negate

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> neg(const Vec512<float> v) {
  const Full512<float> df;
  const Full512<uint32_t> du;
  const auto sign = bit_cast(df, set1(du, 0x80000000u));
  return v ^ sign;
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> neg(const Vec512<double> v) {
  const Full512<double> df;
  const Full512<uint64_t> du;
  const auto sign = bit_cast(df, set1(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator*(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_mul_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator*(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_mul_pd(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator/(const Vec512<float> a,
                                                     const Vec512<float> b) {
  return Vec512<float>(_mm512_div_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator/(const Vec512<double> a,
                                                      const Vec512<double> b) {
  return Vec512<double>(_mm512_div_pd(a.raw, b.raw));
}

// Approximate reciprocal
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> approximate_reciprocal(
    const Vec512<float> v) {
  return Vec512<float>(_mm512_rcp14_ps(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> mul_add(const Vec512<float> mul,
                                                   const Vec512<float> x,
                                                   const Vec512<float> add) {
  return Vec512<float>(_mm512_fmadd_ps(mul.raw, x.raw, add.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> mul_add(const Vec512<double> mul,
                                                    const Vec512<double> x,
                                                    const Vec512<double> add) {
  return Vec512<double>(_mm512_fmadd_pd(mul.raw, x.raw, add.raw));
}

// Returns add - mul * x
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> nmul_add(const Vec512<float> mul,
                                                    const Vec512<float> x,
                                                    const Vec512<float> add) {
  return Vec512<float>(_mm512_fnmadd_ps(mul.raw, x.raw, add.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> nmul_add(const Vec512<double> mul,
                                                     const Vec512<double> x,
                                                     const Vec512<double> add) {
  return Vec512<double>(_mm512_fnmadd_pd(mul.raw, x.raw, add.raw));
}

// Expresses addition/subtraction as FMA for higher throughput (but also
// higher latency) on HSW/BDW. Requires inline assembly because clang > 6
// 'optimizes' FMA by 1.0 to addition/subtraction. x86 offers 132, 213, 231
// forms (1=F, 2=M, 3=A); the first is also the destination.

// Returns x + add
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> fadd(Vec512<float> x,
                                                const Vec512<float> k1,
                                                const Vec512<float> add) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX512__)
  asm("vfmadd132ps %2, %1, %0" : "+x"(x.raw) : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return Vec512<float>(_mm512_fmadd_ps(k1.raw, x.raw, add.raw));
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> fadd(Vec512<double> x,
                                                 const Vec512<double> k1,
                                                 const Vec512<double> add) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX512__)
  asm("vfmadd132pd %2, %1, %0" : "+x"(x.raw) : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return Vec512<double>(_mm512_fmadd_pd(k1.raw, x.raw, add.raw));
#endif
}

// Returns x - sub
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> fsub(Vec512<float> x,
                                                const Vec512<float> k1,
                                                const Vec512<float> sub) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX512__)
  asm("vfmsub132ps %2, %1, %0" : "+x"(x.raw) : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return Vec512<float>(_mm512_fmsub_ps(k1.raw, x.raw, sub.raw));
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> fsub(Vec512<double> x,
                                                 const Vec512<double> k1,
                                                 const Vec512<double> sub) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX512__)
  asm("vfmsub132pd %2, %1, %0" : "+x"(x.raw) : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return Vec512<double>(_mm512_fmsub_pd(k1.raw, x.raw, sub.raw));
#endif
}

// Returns -sub + x (clobbers sub register)
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> fnadd(Vec512<float> sub,
                                                 const Vec512<float> k1,
                                                 const Vec512<float> x) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX512__)
  asm("vfnmadd132ps %2, %1, %0" : "+x"(sub.raw) : "x"(x.raw), "x"(k1.raw));
  return sub;
#else
  return Vec512<float>(_mm512_fnmadd_ps(sub.raw, k1.raw, x.raw));
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> fnadd(Vec512<double> sub,
                                                  const Vec512<double> k1,
                                                  const Vec512<double> x) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX512__)
  asm("vfnmadd132pd %2, %1, %0" : "+x"(sub.raw) : "x"(x.raw), "x"(k1.raw));
  return sub;
#else
  return Vec512<double>(_mm512_fnmadd_pd(sub.raw, k1.raw, x.raw));
#endif
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> mul_subtract(
    const Vec512<float> mul, const Vec512<float> x, const Vec512<float> sub) {
  return Vec512<float>(_mm512_fmsub_ps(mul.raw, x.raw, sub.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> mul_subtract(
    const Vec512<double> mul, const Vec512<double> x,
    const Vec512<double> sub) {
  return Vec512<double>(_mm512_fmsub_pd(mul.raw, x.raw, sub.raw));
}

// Returns -mul * x - sub
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> nmul_subtract(
    const Vec512<float> mul, const Vec512<float> x, const Vec512<float> sub) {
  return Vec512<float>(_mm512_fnmsub_ps(mul.raw, x.raw, sub.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> nmul_subtract(
    const Vec512<double> mul, const Vec512<double> x,
    const Vec512<double> sub) {
  return Vec512<double>(_mm512_fnmsub_pd(mul.raw, x.raw, sub.raw));
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> sqrt(const Vec512<float> v) {
  return Vec512<float>(_mm512_sqrt_ps(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> sqrt(const Vec512<double> v) {
  return Vec512<double>(_mm512_sqrt_pd(v.raw));
}

// Approximate reciprocal square root
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> approximate_reciprocal_sqrt(
    const Vec512<float> v) {
  return Vec512<float>(_mm512_rsqrt14_ps(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, tie to even
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> round(const Vec512<float> v) {
  return Vec512<float>(_mm512_roundscale_ps(
      v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> round(const Vec512<double> v) {
  return Vec512<double>(_mm512_roundscale_pd(
      v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward zero, aka truncate
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> trunc(const Vec512<float> v) {
  return Vec512<float>(
      _mm512_roundscale_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> trunc(const Vec512<double> v) {
  return Vec512<double>(
      _mm512_roundscale_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> ceil(const Vec512<float> v) {
  return Vec512<float>(
      _mm512_roundscale_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> ceil(const Vec512<double> v) {
  return Vec512<double>(
      _mm512_roundscale_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

// Toward -infinity, aka floor
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> floor(const Vec512<float> v) {
  return Vec512<float>(
      _mm512_roundscale_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> floor(const Vec512<double> v) {
  return Vec512<double>(
      _mm512_roundscale_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> operator==(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  const __mmask64 mask = _mm512_cmpeq_epi8_mask(a.raw, b.raw);
  return Vec512<uint8_t>(_mm512_movm_epi8(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> operator==(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  const __mmask32 mask = _mm512_cmpeq_epi16_mask(a.raw, b.raw);
  return Vec512<uint16_t>(_mm512_movm_epi16(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> operator==(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  const __mmask16 mask = _mm512_cmpeq_epi32_mask(a.raw, b.raw);
  return Vec512<uint32_t>(_mm512_movm_epi32(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> operator==(
    const Vec512<uint64_t> a, const Vec512<uint64_t> b) {
  const __mmask8 mask = _mm512_cmpeq_epi64_mask(a.raw, b.raw);
  return Vec512<uint64_t>(_mm512_movm_epi64(mask));
}

// Signed
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> operator==(const Vec512<int8_t> a,
                                                       const Vec512<int8_t> b) {
  const __mmask64 mask = _mm512_cmpeq_epi8_mask(a.raw, b.raw);
  return Vec512<int8_t>(_mm512_movm_epi8(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> operator==(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  const __mmask32 mask = _mm512_cmpeq_epi16_mask(a.raw, b.raw);
  return Vec512<int16_t>(_mm512_movm_epi16(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator==(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  const __mmask16 mask = _mm512_cmpeq_epi32_mask(a.raw, b.raw);
  return Vec512<int32_t>(_mm512_movm_epi32(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> operator==(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  const __mmask8 mask = _mm512_cmpeq_epi64_mask(a.raw, b.raw);
  return Vec512<int64_t>(_mm512_movm_epi64(mask));
}

// Float
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator==(const Vec512<float> a,
                                                      const Vec512<float> b) {
  const __mmask16 mask = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_EQ_OQ);
  return bit_cast(Full512<float>(), Vec512<int32_t>(_mm512_movm_epi32(mask)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator==(const Vec512<double> a,
                                                       const Vec512<double> b) {
  const __mmask8 mask = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_EQ_OQ);
  return bit_cast(Full512<double>(), Vec512<int64_t>(_mm512_movm_epi64(mask)));
}

// ------------------------------ Strict inequality

// Signed/float <
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> operator<(const Vec512<int8_t> a,
                                                      const Vec512<int8_t> b) {
  const __mmask64 mask = _mm512_cmpgt_epi8_mask(b.raw, a.raw);
  return Vec512<int8_t>(_mm512_movm_epi8(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> operator<(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  const __mmask32 mask = _mm512_cmpgt_epi16_mask(b.raw, a.raw);
  return Vec512<int16_t>(_mm512_movm_epi16(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator<(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  const __mmask16 mask = _mm512_cmpgt_epi32_mask(b.raw, a.raw);
  return Vec512<int32_t>(_mm512_movm_epi32(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> operator<(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  const __mmask8 mask = _mm512_cmpgt_epi64_mask(b.raw, a.raw);
  return Vec512<int64_t>(_mm512_movm_epi64(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator<(const Vec512<float> a,
                                                     const Vec512<float> b) {
  const __mmask16 mask = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_LT_OQ);
  return bit_cast(Full512<float>(), Vec512<int32_t>(_mm512_movm_epi32(mask)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator<(const Vec512<double> a,
                                                      const Vec512<double> b) {
  const __mmask8 mask = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_LT_OQ);
  return bit_cast(Full512<double>(), Vec512<int64_t>(_mm512_movm_epi64(mask)));
}

// Signed/float >
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> operator>(const Vec512<int8_t> a,
                                                      const Vec512<int8_t> b) {
  const __mmask64 mask = _mm512_cmpgt_epi8_mask(a.raw, b.raw);
  return Vec512<int8_t>(_mm512_movm_epi8(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> operator>(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  const __mmask32 mask = _mm512_cmpgt_epi16_mask(a.raw, b.raw);
  return Vec512<int16_t>(_mm512_movm_epi16(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> operator>(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  const __mmask16 mask = _mm512_cmpgt_epi32_mask(a.raw, b.raw);
  return Vec512<int32_t>(_mm512_movm_epi32(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> operator>(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  const __mmask8 mask = _mm512_cmpgt_epi64_mask(a.raw, b.raw);
  return Vec512<int64_t>(_mm512_movm_epi64(mask));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator>(const Vec512<float> a,
                                                     const Vec512<float> b) {
  const __mmask16 mask = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_GT_OQ);
  return bit_cast(Full512<float>(), Vec512<int32_t>(_mm512_movm_epi32(mask)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator>(const Vec512<double> a,
                                                      const Vec512<double> b) {
  const __mmask8 mask = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_GT_OQ);
  return bit_cast(Full512<double>(), Vec512<int64_t>(_mm512_movm_epi64(mask)));
}

// ------------------------------ Weak inequality

// Float <= >=
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator<=(const Vec512<float> a,
                                                      const Vec512<float> b) {
  const __mmask16 mask = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_LE_OQ);
  return bit_cast(Full512<float>(), Vec512<int32_t>(_mm512_movm_epi32(mask)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator<=(const Vec512<double> a,
                                                       const Vec512<double> b) {
  const __mmask8 mask = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_LE_OQ);
  return bit_cast(Full512<double>(), Vec512<int64_t>(_mm512_movm_epi64(mask)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> operator>=(const Vec512<float> a,
                                                      const Vec512<float> b) {
  const __mmask16 mask = _mm512_cmp_ps_mask(a.raw, b.raw, _CMP_GE_OQ);
  return bit_cast(Full512<float>(), Vec512<int32_t>(_mm512_movm_epi32(mask)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> operator>=(const Vec512<double> a,
                                                       const Vec512<double> b) {
  const __mmask8 mask = _mm512_cmp_pd_mask(a.raw, b.raw, _CMP_GE_OQ);
  return bit_cast(Full512<double>(), Vec512<int64_t>(_mm512_movm_epi64(mask)));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> load(Full512<T> /* tag */,
                                            const T* SIMD_RESTRICT aligned) {
  return Vec512<T>(
      _mm512_load_si512(reinterpret_cast<const __m512i*>(aligned)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> load(
    Full512<float> /* tag */, const float* SIMD_RESTRICT aligned) {
  return Vec512<float>(_mm512_load_ps(aligned));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> load(
    Full512<double> /* tag */, const double* SIMD_RESTRICT aligned) {
  return Vec512<double>(_mm512_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> load_u(Full512<T> /* tag */,
                                              const T* SIMD_RESTRICT p) {
  return Vec512<T>(_mm512_loadu_si512(reinterpret_cast<const __m512i*>(p)));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> load_u(
    Full512<float> /* tag */, const float* SIMD_RESTRICT p) {
  return Vec512<float>(_mm512_loadu_ps(p));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> load_u(
    Full512<double> /* tag */, const double* SIMD_RESTRICT p) {
  return Vec512<double>(_mm512_loadu_pd(p));
}

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> load_dup128(
    Full512<T> /* tag */, const T* const SIMD_RESTRICT p) {
  // Clang 3.9 generates VINSERTF128 which is slower, but inline assembly leads
  // to "invalid output size for constraint" without -mavx512:
  // https://gcc.godbolt.org/z/-Jt_-F
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX512__)
  __m512i out;
  asm("vbroadcasti128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec512<T>(out);
#else
  const auto x4 = load_u(Full128<T>(), p);
  return Vec512<T>(_mm512_broadcast_i32x4(x4.raw));
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> load_dup128(
    Full512<float> /* tag */, const float* const SIMD_RESTRICT p) {
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX512__)
  __m512 out;
  asm("vbroadcastf128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec512<float>(out);
#else
  const __m128 x4 = _mm_load_ps(p);
  return Vec512<float>(_mm512_broadcast_f32x4(x4));
#endif
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> load_dup128(
    Full512<double> /* tag */, const double* const SIMD_RESTRICT p) {
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX512__)
  __m512d out;
  asm("vbroadcastf128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec512<double>(out);
#else
  const __m128d x2 = _mm_load_pd(p);
  return Vec512<double>(_mm512_broadcast_f64x2(x2));
#endif
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE void store(const Vec512<T> v, Full512<T> /* tag */,
                                        T* SIMD_RESTRICT aligned) {
  _mm512_store_si512(reinterpret_cast<__m512i*>(aligned), v.raw);
}
SIMD_ATTR_AVX512 SIMD_INLINE void store(const Vec512<float> v,
                                        Full512<float> /* tag */,
                                        float* SIMD_RESTRICT aligned) {
  _mm512_store_ps(aligned, v.raw);
}
SIMD_ATTR_AVX512 SIMD_INLINE void store(const Vec512<double> v,
                                        Full512<double> /* tag */,
                                        double* SIMD_RESTRICT aligned) {
  _mm512_store_pd(aligned, v.raw);
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE void store_u(const Vec512<T> v,
                                          Full512<T> /* tag */,
                                          T* SIMD_RESTRICT p) {
  _mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v.raw);
}
SIMD_ATTR_AVX512 SIMD_INLINE void store_u(const Vec512<float> v,
                                          Full512<float> /* tag */,
                                          float* SIMD_RESTRICT p) {
  _mm512_storeu_ps(p, v.raw);
}
SIMD_ATTR_AVX512 SIMD_INLINE void store_u(const Vec512<double> v,
                                          Full512<double>,
                                          double* SIMD_RESTRICT p) {
  _mm512_storeu_pd(p, v.raw);
}

// ------------------------------ Non-temporal stores

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE void stream(const Vec512<T> v,
                                         Full512<T> /* tag */,
                                         T* SIMD_RESTRICT aligned) {
  _mm512_stream_si512(reinterpret_cast<__m512i*>(aligned), v.raw);
}
SIMD_ATTR_AVX512 SIMD_INLINE void stream(const Vec512<float> v,
                                         Full512<float> /* tag */,
                                         float* SIMD_RESTRICT aligned) {
  _mm512_stream_ps(aligned, v.raw);
}
SIMD_ATTR_AVX512 SIMD_INLINE void stream(const Vec512<double> v,
                                         Full512<double>,
                                         double* SIMD_RESTRICT aligned) {
  _mm512_stream_pd(aligned, v.raw);
}

// ------------------------------ Gather

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> gather_offset_impl(
    SizeTag<4> /* tag */, Full512<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec512<int32_t> offset) {
  return Vec512<T>(_mm512_i32gather_epi32(offset.raw, base, 1));
}
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> gather_index_impl(
    SizeTag<4> /* tag */, Full512<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec512<int32_t> index) {
  return Vec512<T>(_mm512_i32gather_epi32(index.raw, base, 4));
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> gather_offset_impl(
    SizeTag<8> /* tag */, Full512<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec512<int64_t> offset) {
  return Vec512<T>(_mm512_i64gather_epi64(offset.raw, base, 1));
}
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> gather_index_impl(
    SizeTag<8> /* tag */, Full512<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec512<int64_t> index) {
  return Vec512<T>(_mm512_i64gather_epi64(index.raw, base, 8));
}

template <typename T, typename Offset>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> gather_offset(
    Full512<T> d, const T* SIMD_RESTRICT base, const Vec512<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  return gather_offset_impl(SizeTag<sizeof(T)>(), d, base, offset);
}
template <typename T, typename Index>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> gather_index(Full512<T> d,
                                                    const T* SIMD_RESTRICT base,
                                                    const Vec512<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  return gather_index_impl(SizeTag<sizeof(T)>(), d, base, index);
}

template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> gather_offset<float>(
    Full512<float> /* tag */, const float* SIMD_RESTRICT base,
    const Vec512<int32_t> offset) {
  return Vec512<float>(_mm512_i32gather_ps(offset.raw, base, 1));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> gather_index<float>(
    Full512<float> /* tag */, const float* SIMD_RESTRICT base,
    const Vec512<int32_t> index) {
  return Vec512<float>(_mm512_i32gather_ps(index.raw, base, 4));
}

template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> gather_offset<double>(
    Full512<double> /* tag */, const double* SIMD_RESTRICT base,
    const Vec512<int64_t> offset) {
  return Vec512<double>(_mm512_i64gather_pd(offset.raw, base, 1));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> gather_index<double>(
    Full512<double> /* tag */, const double* SIMD_RESTRICT base,
    const Vec512<int64_t> index) {
  return Vec512<double>(_mm512_i64gather_pd(index.raw, base, 8));
}

}  // namespace ext

// ================================================== SWIZZLE

// ------------------------------ Extract half

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<T> lower_half(Vec512<T> v) {
  return Vec256<T>(_mm512_castsi512_si256(v.raw));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<float> lower_half(Vec512<float> v) {
  return Vec256<float>(_mm512_castps512_ps256(v.raw));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<double> lower_half(Vec512<double> v) {
  return Vec256<double>(_mm512_castpd512_pd256(v.raw));
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<T> upper_half(Vec512<T> v) {
  return Vec256<T>(_mm512_extracti32x8_epi32(v.raw, 1));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<float> upper_half(Vec512<float> v) {
  return Vec256<float>(_mm512_extractf32x8_ps(v.raw, 1));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<double> upper_half(Vec512<double> v) {
  return Vec256<double>(_mm512_extractf64x4_pd(v.raw, 1));
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<T> get_half(Lower /* tag */, Vec512<T> v) {
  return lower_half(v);
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<T> get_half(Upper /* tag */, Vec512<T> v) {
  return upper_half(v);
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> shift_left_bytes(const Vec512<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  return Vec512<T>(_mm512_bslli_epi128(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> shift_left_lanes(const Vec512<T> v) {
  return shift_left_bytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> shift_right_bytes(const Vec512<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  return Vec512<T>(_mm512_bsrli_epi128(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> shift_right_lanes(const Vec512<T> v) {
  return shift_right_bytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> combine_shift_right_bytes(
    const Vec512<T> hi, const Vec512<T> lo) {
  const Full512<uint8_t> d8;
  const Vec512<uint8_t> extracted_bytes(
      _mm512_alignr_epi8(bit_cast(d8, hi).raw, bit_cast(d8, lo).raw, kBytes));
  return bit_cast(Full512<T>(), extracted_bytes);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> broadcast(
    const Vec512<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m512i lo = _mm512_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec512<uint16_t>(_mm512_unpacklo_epi64(lo, lo));
  } else {
    const __m512i hi =
        _mm512_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec512<uint16_t>(_mm512_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> broadcast(
    const Vec512<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec512<uint32_t>(_mm512_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> broadcast(
    const Vec512<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec512<uint64_t>(_mm512_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> broadcast(
    const Vec512<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m512i lo = _mm512_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec512<int16_t>(_mm512_unpacklo_epi64(lo, lo));
  } else {
    const __m512i hi =
        _mm512_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec512<int16_t>(_mm512_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> broadcast(
    const Vec512<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec512<int32_t>(_mm512_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> broadcast(
    const Vec512<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec512<int64_t>(_mm512_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> broadcast(const Vec512<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec512<float>(_mm512_shuffle_ps(v.raw, v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> broadcast(const Vec512<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec512<double>(_mm512_shuffle_pd(v.raw, v.raw, 0xFF * kLane));
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec512<int32_t> have lanes 7,6,5,4,3,2,1,0 (0 is
// least-significant). shuffle_0321 rotates four-lane blocks one lane to the
// right (the previous least-significant lane is now most-significant =>
// 47650321). These could also be implemented via combine_shift_right_bytes but
// the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shuffle_1032(
    const Vec512<uint32_t> v) {
  return Vec512<uint32_t>(_mm512_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shuffle_1032(
    const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> shuffle_1032(const Vec512<float> v) {
  // Shorter encoding than _mm512_permute_ps.
  return Vec512<float>(_mm512_shuffle_ps(v.raw, v.raw, 0x4E));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> shuffle_01(
    const Vec512<uint64_t> v) {
  return Vec512<uint64_t>(_mm512_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> shuffle_01(
    const Vec512<int64_t> v) {
  return Vec512<int64_t>(_mm512_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> shuffle_01(const Vec512<double> v) {
  // Shorter encoding than _mm512_permute_pd.
  return Vec512<double>(_mm512_shuffle_pd(v.raw, v.raw, 0x55));
}

// Rotate right 32 bits
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shuffle_0321(
    const Vec512<uint32_t> v) {
  return Vec512<uint32_t>(_mm512_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shuffle_0321(
    const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> shuffle_0321(const Vec512<float> v) {
  return Vec512<float>(_mm512_shuffle_ps(v.raw, v.raw, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shuffle_2103(
    const Vec512<uint32_t> v) {
  return Vec512<uint32_t>(_mm512_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shuffle_2103(
    const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> shuffle_2103(const Vec512<float> v) {
  return Vec512<float>(_mm512_shuffle_ps(v.raw, v.raw, 0x93));
}

// Reverse
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> shuffle_0123(
    const Vec512<uint32_t> v) {
  return Vec512<uint32_t>(_mm512_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> shuffle_0123(
    const Vec512<int32_t> v) {
  return Vec512<int32_t>(_mm512_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> shuffle_0123(const Vec512<float> v) {
  return Vec512<float>(_mm512_shuffle_ps(v.raw, v.raw, 0x1B));
}

// ------------------------------ Permute (runtime variable)

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct Permute512 {
  __m512i raw;
};

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Permute512<T> set_table_indices(
    const Full512<T> d, const int32_t* idx) {
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
  return Permute512<T>{load_u(Full512<int32_t>(), idx).raw};
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> table_lookup_lanes(
    const Vec512<uint32_t> v, const Permute512<uint32_t> idx) {
  return Vec512<uint32_t>(_mm512_permutexvar_epi32(idx.raw, v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> table_lookup_lanes(
    const Vec512<int32_t> v, const Permute512<int32_t> idx) {
  return Vec512<int32_t>(_mm512_permutexvar_epi32(idx.raw, v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> table_lookup_lanes(
    const Vec512<float> v, const Permute512<float> idx) {
  return Vec512<float>(_mm512_permutexvar_ps(idx.raw, v.raw));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> interleave_lo(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> interleave_lo(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> interleave_lo(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> interleave_lo(
    const Vec512<uint64_t> a, const Vec512<uint64_t> b) {
  return Vec512<uint64_t>(_mm512_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> interleave_lo(
    const Vec512<int8_t> a, const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> interleave_lo(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> interleave_lo(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> interleave_lo(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  return Vec512<int64_t>(_mm512_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> interleave_lo(
    const Vec512<float> a, const Vec512<float> b) {
  return Vec512<float>(_mm512_unpacklo_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> interleave_lo(
    const Vec512<double> a, const Vec512<double> b) {
  return Vec512<double>(_mm512_unpacklo_pd(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint8_t> interleave_hi(
    const Vec512<uint8_t> a, const Vec512<uint8_t> b) {
  return Vec512<uint8_t>(_mm512_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> interleave_hi(
    const Vec512<uint16_t> a, const Vec512<uint16_t> b) {
  return Vec512<uint16_t>(_mm512_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> interleave_hi(
    const Vec512<uint32_t> a, const Vec512<uint32_t> b) {
  return Vec512<uint32_t>(_mm512_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> interleave_hi(
    const Vec512<uint64_t> a, const Vec512<uint64_t> b) {
  return Vec512<uint64_t>(_mm512_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int8_t> interleave_hi(
    const Vec512<int8_t> a, const Vec512<int8_t> b) {
  return Vec512<int8_t>(_mm512_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> interleave_hi(
    const Vec512<int16_t> a, const Vec512<int16_t> b) {
  return Vec512<int16_t>(_mm512_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> interleave_hi(
    const Vec512<int32_t> a, const Vec512<int32_t> b) {
  return Vec512<int32_t>(_mm512_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> interleave_hi(
    const Vec512<int64_t> a, const Vec512<int64_t> b) {
  return Vec512<int64_t>(_mm512_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> interleave_hi(
    const Vec512<float> a, const Vec512<float> b) {
  return Vec512<float>(_mm512_unpackhi_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> interleave_hi(
    const Vec512<double> a, const Vec512<double> b) {
  return Vec512<double>(_mm512_unpackhi_pd(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> zip_lo(const Vec512<uint8_t> a,
                                                     const Vec512<uint8_t> b) {
  return Vec512<uint16_t>(_mm512_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> zip_lo(const Vec512<uint16_t> a,
                                                     const Vec512<uint16_t> b) {
  return Vec512<uint32_t>(_mm512_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> zip_lo(const Vec512<uint32_t> a,
                                                     const Vec512<uint32_t> b) {
  return Vec512<uint64_t>(_mm512_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> zip_lo(const Vec512<int8_t> a,
                                                    const Vec512<int8_t> b) {
  return Vec512<int16_t>(_mm512_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> zip_lo(const Vec512<int16_t> a,
                                                    const Vec512<int16_t> b) {
  return Vec512<int32_t>(_mm512_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> zip_lo(const Vec512<int32_t> a,
                                                    const Vec512<int32_t> b) {
  return Vec512<int64_t>(_mm512_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> zip_hi(const Vec512<uint8_t> a,
                                                     const Vec512<uint8_t> b) {
  return Vec512<uint16_t>(_mm512_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> zip_hi(const Vec512<uint16_t> a,
                                                     const Vec512<uint16_t> b) {
  return Vec512<uint32_t>(_mm512_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> zip_hi(const Vec512<uint32_t> a,
                                                     const Vec512<uint32_t> b) {
  return Vec512<uint64_t>(_mm512_unpackhi_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> zip_hi(const Vec512<int8_t> a,
                                                    const Vec512<int8_t> b) {
  return Vec512<int16_t>(_mm512_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> zip_hi(const Vec512<int16_t> a,
                                                    const Vec512<int16_t> b) {
  return Vec512<int32_t>(_mm512_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> zip_hi(const Vec512<int32_t> a,
                                                    const Vec512<int32_t> b) {
  return Vec512<int64_t>(_mm512_unpackhi_epi32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns part of a vector (unspecified whether upper or lower).
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> any_part(Full512<T> /* tag */,
                                                Vec256<T> v) {
  return v;  // full
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<T> any_part(Full256<T> /* tag */,
                                                Vec512<T> v) {
  return Vec256<T>(_mm512_castsi512_si256(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<float> any_part(Full256<float> /* tag */,
                                                    Vec512<float> v) {
  return Vec256<float>(_mm512_castps512_ps256(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec256<double> any_part(Full256<double> /* tag */,
                                                     Vec512<double> v) {
  return Vec256<double>(_mm512_castpd512_pd256(v.raw));
}

template <typename T, size_t N, SIMD_IF128(T, N)>
SIMD_ATTR_AVX512 SIMD_INLINE Vec128<T, N> any_part(Desc<T, N> /* tag */,
                                                   Vec512<T> v) {
  return Vec128<T, N>(_mm512_castsi512_si128(v.raw));
}
template <size_t N, SIMD_IF128(float, N)>
SIMD_ATTR_AVX512 SIMD_INLINE Vec128<float, N> any_part(Desc<float, N> /* tag */,
                                                       Vec512<float> v) {
  return Vec128<float, N>(_mm512_castps512_ps128(v.raw));
}
template <size_t N, SIMD_IF128(double, N)>
SIMD_ATTR_AVX512 SIMD_INLINE Vec128<double, N> any_part(
    Desc<double, N> /* tag */, Vec512<double> v) {
  return Vec128<double, N>(_mm512_castpd512_pd128(v.raw));
}

// Gets the single value stored in a vector/part.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE T get_lane(const Vec512<T> v) {
  return get_lane(any_part(Full128<T>(), v));
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> broadcast_part(Full512<T> /* tag */,
                                                      const Vec128<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(Vec128<T>(v.raw));
  // Same as _mm512_castsi128_si512, but with guaranteed zero-extension.
  const auto lo = _mm512_zextsi128_si512(v128.raw);
  return Vec512<T>(_mm512_broadcast_i32x4(lo));
}
template <int kLane, size_t N>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> broadcast_part(
    Full512<float> /* tag */, const Vec128<float, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(Vec128<float>(v.raw)).raw;
  // Same as _mm512_castps128_ps256, but with guaranteed zero-extension.
  const auto lo = _mm512_zextps128_ps256(v128);
  return Vec512<float>(_mm512_broadcast_f32x4(lo));
}
template <int kLane, size_t N>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> broadcast_part(
    Full512<double> /* tag */, const Vec128<double, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(Vec128<double>(v.raw)).raw;
  // Same as _mm512_castpd128_pd256, but with guaranteed zero-extension.
  const auto lo = _mm512_zextpd128_pd256(v128);
  return Vec512<double>(_mm512_broadcast_f64x2(lo));
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> concat_lo_lo(const Vec512<T> hi,
                                                    const Vec512<T> lo) {
  return Vec512<T>(_mm512_shuffle_i32x4(lo.raw, hi.raw, 0x44));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> concat_lo_lo(
    const Vec512<float> hi, const Vec512<float> lo) {
  return Vec512<float>(_mm512_shuffle_f32x4(lo.raw, hi.raw, 0x44));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> concat_lo_lo(
    const Vec512<double> hi, const Vec512<double> lo) {
  return Vec512<double>(_mm512_shuffle_f64x2(lo.raw, hi.raw, 0x44));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> concat_hi_hi(const Vec512<T> hi,
                                                    const Vec512<T> lo) {
  return Vec512<T>(_mm512_shuffle_i32x4(lo.raw, hi.raw, 0xEE));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> concat_hi_hi(
    const Vec512<float> hi, const Vec512<float> lo) {
  return Vec512<float>(_mm512_shuffle_f32x4(lo.raw, hi.raw, 0xEE));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> concat_hi_hi(
    const Vec512<double> hi, const Vec512<double> lo) {
  return Vec512<double>(_mm512_shuffle_f64x2(lo.raw, hi.raw, 0xEE));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves / swap blocks)
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> concat_lo_hi(const Vec512<T> hi,
                                                    const Vec512<T> lo) {
  return Vec512<T>(_mm512_shuffle_i32x4(lo.raw, hi.raw, 0x4E));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> concat_lo_hi(
    const Vec512<float> hi, const Vec512<float> lo) {
  return Vec512<float>(_mm512_shuffle_f32x4(lo.raw, hi.raw, 0x4E));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> concat_lo_hi(
    const Vec512<double> hi, const Vec512<double> lo) {
  return Vec512<double>(_mm512_shuffle_f64x2(lo.raw, hi.raw, 0x4E));
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> concat_hi_lo(const Vec512<T> hi,
                                                    const Vec512<T> lo) {
  // There are no imm8 blend in AVX512. Use blend16 because 32-bit masks
  // are efficiently loaded from 32-bit regs.
  const __mmask32 mask = /*_cvtu32_mask32 */ (0x0000FFFF);
  return Vec512<T>(_mm512_mask_blend_epi16(mask, hi.raw, lo.raw));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> concat_hi_lo(
    const Vec512<float> hi, const Vec512<float> lo) {
  const __mmask16 mask = /*_cvtu32_mask16 */ (0x00FF);
  return Vec512<float>(_mm512_mask_blend_ps(mask, hi.raw, lo.raw));
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> concat_hi_lo(
    const Vec512<double> hi, const Vec512<double> lo) {
  const __mmask8 mask = /*_cvtu32_mask8 */ (0x0F);
  return Vec512<double>(_mm512_mask_blend_pd(mask, hi.raw, lo.raw));
}

// ------------------------------ Odd/even lanes

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> odd_even_impl(SizeTag<1> /* tag */,
                                                     const Vec512<T> a,
                                                     const Vec512<T> b) {
  const Full512<T> d;
  const Full512<uint8_t> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                           0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return if_then_else(bit_cast(d, load_dup128(d8, mask)), b, a);
}
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> odd_even_impl(SizeTag<2> /* tag */,
                                                     const Vec512<T> a,
                                                     const Vec512<T> b) {
  const __mmask32 mask = /*_cvtu32_mask32 */ (0x55555555);
  return Vec512<T>(_mm512_mask_blend_epi16(mask, a.raw, b.raw));
}
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> odd_even_impl(SizeTag<4> /* tag */,
                                                     const Vec512<T> a,
                                                     const Vec512<T> b) {
  const __mmask16 mask = /*_cvtu32_mask16 */ (0x5555);
  return Vec512<T>(_mm512_mask_blend_epi32(mask, a.raw, b.raw));
}
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> odd_even_impl(SizeTag<8> /* tag */,
                                                     const Vec512<T> a,
                                                     const Vec512<T> b) {
  const __mmask8 mask = /*_cvtu32_mask8 */ (0x55);
  return Vec512<T>(_mm512_mask_blend_epi64(mask, a.raw, b.raw));
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> odd_even(const Vec512<T> a,
                                                const Vec512<T> b) {
  return odd_even_impl(SizeTag<sizeof(T)>(), a, b);
}
template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> odd_even<float>(
    const Vec512<float> a, const Vec512<float> b) {
  const __mmask16 mask = /*_cvtu32_mask16 */ (0x5555);
  return Vec512<float>(_mm512_mask_blend_ps(mask, a.raw, b.raw));
}

template <>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> odd_even<double>(
    const Vec512<double> a, const Vec512<double> b) {
  const __mmask8 mask = /*_cvtu32_mask8 */ (0x55);
  return Vec512<double>(_mm512_mask_blend_pd(mask, a.raw, b.raw));
}

// ================================================== CONVERT

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> table_lookup_bytes(
    const Vec512<T> bytes, const Vec512<TI> from) {
  return Vec512<T>(_mm512_shuffle_epi8(bytes.raw, from.raw));
}

// ------------------------------ Promotions (part w/ narrow lanes -> full)

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<double> convert_to(
    Full512<double> /* tag */, Vec256<float> v) {
  return Vec512<double>(_mm512_cvtps_pd(v.raw));
}

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo would be faster.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint16_t> convert_to(
    Full512<uint16_t> /* tag */, Vec256<uint8_t> v) {
  return Vec512<uint16_t>(_mm512_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> convert_to(
    Full512<uint32_t> /* tag */, Vec128<uint8_t> v) {
  return Vec512<uint32_t>(_mm512_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> convert_to(
    Full512<int16_t> /* tag */, Vec256<uint8_t> v) {
  return Vec512<int16_t>(_mm512_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> convert_to(
    Full512<int32_t> /* tag */, Vec128<uint8_t> v) {
  return Vec512<int32_t>(_mm512_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> convert_to(
    Full512<uint32_t> /* tag */, Vec256<uint16_t> v) {
  return Vec512<uint32_t>(_mm512_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> convert_to(
    Full512<int32_t> /* tag */, Vec256<uint16_t> v) {
  return Vec512<int32_t>(_mm512_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> convert_to(
    Full512<uint64_t> /* tag */, Vec256<uint32_t> v) {
  return Vec512<uint64_t>(_mm512_cvtepu32_epi64(v.raw));
}

// Special case for "v" with all blocks equal (e.g. from broadcast_block or
// load_dup128): single-cycle latency instead of 3.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint32_t> u32_from_u8(
    const Vec512<uint8_t> v) {
  const Full512<uint32_t> d32;
  SIMD_ALIGN static constexpr uint32_t k32From8[16] = {
      0xFFFFFF00UL, 0xFFFFFF01UL, 0xFFFFFF02UL, 0xFFFFFF03UL,
      0xFFFFFF04UL, 0xFFFFFF05UL, 0xFFFFFF06UL, 0xFFFFFF07UL,
      0xFFFFFF08UL, 0xFFFFFF09UL, 0xFFFFFF0AUL, 0xFFFFFF0BUL,
      0xFFFFFF0CUL, 0xFFFFFF0DUL, 0xFFFFFF0EUL, 0xFFFFFF0FUL};
  return table_lookup_bytes(bit_cast(d32, v), load(d32, k32From8));
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo followed by
// signed shift would be faster.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int16_t> convert_to(
    Full512<int16_t> /* tag */, Vec256<int8_t> v) {
  return Vec512<int16_t>(_mm512_cvtepi8_epi16(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> convert_to(
    Full512<int32_t> /* tag */, Vec128<int8_t> v) {
  return Vec512<int32_t>(_mm512_cvtepi8_epi32(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> convert_to(
    Full512<int32_t> /* tag */, Vec256<int16_t> v) {
  return Vec512<int32_t>(_mm512_cvtepi16_epi32(v.raw));
}
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int64_t> convert_to(
    Full512<int64_t> /* tag */, Vec256<int32_t> v) {
  return Vec512<int64_t>(_mm512_cvtepi32_epi64(v.raw));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

SIMD_ATTR_AVX512 SIMD_INLINE Vec256<uint16_t> convert_to(
    Full256<uint16_t> /* tag */, const Vec512<int32_t> v) {
  const Vec512<uint16_t> u16(_mm512_packus_epi32(v.raw, v.raw));

  // Compress even u64 lanes into 256 bit.
  SIMD_ALIGN static constexpr uint64_t kLanes[8] = {0, 2, 4, 6, 0, 2, 4, 6};
  const auto idx64 = load(Full512<uint64_t>(), kLanes);
  const Vec512<uint16_t> even(_mm512_permutexvar_epi64(idx64.raw, u16.raw));
  return lower_half(even);
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec128<uint8_t, 16> convert_to(
    Full128<uint8_t> /* tag */, const Vec512<int32_t> v) {
  const Vec512<uint16_t> u16(_mm512_packus_epi32(v.raw, v.raw));
  const Vec512<uint8_t> u8(_mm512_packus_epi16(u16.raw, u16.raw));

  SIMD_ALIGN static constexpr uint32_t kLanes[16] = {0, 4, 8, 12, 0, 4, 8, 12,
                                                     0, 4, 8, 12, 0, 4, 8, 12};
  const auto idx32 = load(Full512<uint32_t>(), kLanes);
  const Vec512<uint8_t> fixed(_mm512_permutexvar_epi32(idx32.raw, u8.raw));
  return any_part(Full128<uint8_t>(), fixed);
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec256<int16_t> convert_to(
    Full256<int16_t> /* tag */, const Vec512<int32_t> v) {
  const Vec512<int16_t> i16(_mm512_packs_epi32(v.raw, v.raw));

  // Compress even u64 lanes into 256 bit.
  SIMD_ALIGN static constexpr uint64_t kLanes[8] = {0, 2, 4, 6, 0, 2, 4, 6};
  const auto idx64 = load(Full512<uint64_t>(), kLanes);
  const Vec512<int16_t> even(_mm512_permutexvar_epi64(idx64.raw, i16.raw));
  return lower_half(even);
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec128<int8_t, 16> convert_to(
    Full128<int8_t> /* tag */, const Vec512<int32_t> v) {
  const Vec512<int16_t> i16(_mm512_packs_epi32(v.raw, v.raw));
  const Vec512<int8_t> i8(_mm512_packs_epi16(i16.raw, i16.raw));

  SIMD_ALIGN static constexpr uint32_t kLanes[16] = {0, 4, 8, 12, 0, 4, 8, 12,
                                                     0, 4, 8, 12, 0, 4, 8, 12};
  const auto idx32 = load(Full512<uint32_t>(), kLanes);
  const Vec512<int8_t> fixed(_mm512_permutexvar_epi32(idx32.raw, i8.raw));
  return any_part(Full128<int8_t>(), fixed);
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec256<uint8_t> convert_to(
    Full256<uint8_t> /* tag */, const Vec512<int16_t> v) {
  const Vec512<uint8_t> u8(_mm512_packus_epi16(v.raw, v.raw));

  // Compress even u64 lanes into 256 bit.
  SIMD_ALIGN static constexpr uint64_t kLanes[8] = {0, 2, 4, 6, 0, 2, 4, 6};
  const auto idx64 = load(Full512<uint64_t>(), kLanes);
  const Vec512<uint8_t> even(_mm512_permutexvar_epi64(idx64.raw, u8.raw));
  return lower_half(even);
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec256<int8_t> convert_to(
    Full256<int8_t> /* tag */, const Vec512<int16_t> v) {
  const Vec512<int8_t> u8(_mm512_packs_epi16(v.raw, v.raw));

  // Compress even u64 lanes into 256 bit.
  SIMD_ALIGN static constexpr uint64_t kLanes[8] = {0, 2, 4, 6, 0, 2, 4, 6};
  const auto idx64 = load(Full512<uint64_t>(), kLanes);
  const Vec512<int8_t> even(_mm512_permutexvar_epi64(idx64.raw, u8.raw));
  return lower_half(even);
}

// For already range-limited input [0, 255].
SIMD_ATTR_AVX512 SIMD_INLINE Vec128<uint8_t, 16> u8_from_u32(
    const Vec512<uint32_t> v) {
  const Full512<uint32_t> d32;
  // In each 128 bit block, gather the lower byte of 4 uint32_t lanes into the
  // lowest 4 bytes.
  SIMD_ALIGN static constexpr uint32_t k8From32[4] = {0x0C080400u, ~0u, ~0u,
                                                      ~0u};
  const auto quads = table_lookup_bytes(v, load_dup128(d32, k8From32));
  // Gather the lowest 4 bytes of 4 128-bit blocks.
  SIMD_ALIGN static constexpr uint32_t kIndex32[16] = {
      0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12};
  const Vec512<uint8_t> bytes(
      _mm512_permutexvar_epi32(load(d32, kIndex32).raw, quads.raw));
  return any_part(Full128<uint8_t>(), bytes);
}

// ------------------------------ Convert i32 <=> f32

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<float> convert_to(Full512<float> /* tag */,
                                                      const Vec512<int32_t> v) {
  return Vec512<float>(_mm512_cvtepi32_ps(v.raw));
}
// Truncates (rounds toward zero).
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> convert_to(
    Full512<int32_t> /* tag */, const Vec512<float> v) {
  return Vec512<int32_t>(_mm512_cvttps_epi32(v.raw));
}

SIMD_ATTR_AVX512 SIMD_INLINE Vec512<int32_t> nearest_int(
    const Vec512<float> v) {
  return Vec512<int32_t>(_mm512_cvtps_epi32(v.raw));
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..31 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_AVX512 SIMD_INLINE uint64_t movemask(const Vec512<uint8_t> v) {
  return _mm512_cmplt_epi8_mask(v.raw, _mm512_setzero_si512());
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_AVX512 SIMD_INLINE uint64_t movemask(const Vec512<float> v) {
  // cmp < 0 doesn't set if -0.0 and <= -0.0 sets despite 0.0 input.
  return _mm512_test_epi32_mask(_mm512_castps_si512(v.raw),
                                _mm512_set1_epi32(-0x7fffffff - 1));
}
SIMD_ATTR_AVX512 SIMD_INLINE uint64_t movemask(const Vec512<double> v) {
  // cmp < 0 doesn't set if -0.0 and <= -0.0 sets despite 0.0 input.
  return _mm512_test_epi64_mask(_mm512_castpd_si512(v.raw),
                                _mm512_set1_epi64(-0x7fffffffffffffff - 1));
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero. Supported for all integer V.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE bool all_zero(const Vec512<T> v) {
  return 0 == _mm512_test_epi16_mask(v.raw, v.raw);
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<uint64_t> sums_of_u8x8(
    const Vec512<uint8_t> v) {
  return Vec512<uint64_t>(_mm512_sad_epu8(v.raw, _mm512_setzero_si512()));
}

// mpsadbw2 is not supported - AVX512 has different semantics.

// Returns sum{lane[i]} in each lane. "v3210" is a replicated 128-bit block.
// Same logic as x86_sse4.h, but with Vec512 arguments.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> horz_sum_impl(SizeTag<4> /* tag */,
                                                     const Vec512<T> v3210) {
  const auto v1032 = shuffle_1032(v3210);
  const auto v31_20_31_20 = v3210 + v1032;
  const auto v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> horz_sum_impl(SizeTag<8> /* tag */,
                                                     const Vec512<T> v10) {
  const auto v01 = shuffle_01(v10);
  return v10 + v01;
}

// Swaps adjacent 128-bit blocks.
SIMD_ATTR_AVX512 SIMD_INLINE __m512i Blocks2301(__m512i v) {
  return _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
}
SIMD_ATTR_AVX512 SIMD_INLINE __m512 Blocks2301(__m512 v) {
  return _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(2, 3, 0, 1));
}
SIMD_ATTR_AVX512 SIMD_INLINE __m512d Blocks2301(__m512d v) {
  return _mm512_shuffle_f64x2(v, v, _MM_SHUFFLE(2, 3, 0, 1));
}

// Swaps upper/lower pairs of 128-bit blocks.
SIMD_ATTR_AVX512 SIMD_INLINE __m512i Blocks1032(__m512i v) {
  return _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2));
}
SIMD_ATTR_AVX512 SIMD_INLINE __m512 Blocks1032(__m512 v) {
  return _mm512_shuffle_f32x4(v, v, _MM_SHUFFLE(1, 0, 3, 2));
}
SIMD_ATTR_AVX512 SIMD_INLINE __m512d Blocks1032(__m512d v) {
  return _mm512_shuffle_f64x2(v, v, _MM_SHUFFLE(1, 0, 3, 2));
}

// Supported for {uif}32x8, {uif}64x4. Returns the sum in each lane.
template <typename T>
SIMD_ATTR_AVX512 SIMD_INLINE Vec512<T> sum_of_lanes(const Vec512<T> v3210) {
  // Sum of all 128-bit blocks in each 128-bit block.
  const Vec512<T> v2301(Blocks2301(v3210.raw));
  const Vec512<T> v32_32_10_10 = v2301 + v3210;
  const Vec512<T> v10_10_32_32(Blocks1032(v32_32_10_10.raw));
  return horz_sum_impl(SizeTag<sizeof(T)>(), v32_32_10_10 + v10_10_32_32);
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).
}  // namespace jxl

#endif  // HIGHWAY_X86_AVX512_H_
