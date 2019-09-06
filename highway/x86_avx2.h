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

#ifndef HIGHWAY_X86_AVX2_H_
#define HIGHWAY_X86_AVX2_H_

// 256-bit AVX2 vectors and operations.
// WARNING: most operations do not cross 128-bit block boundaries. In
// particular, "broadcast", pack and zip behavior may be surprising.

#include <immintrin.h>

#include "third_party/highway/highway/shared.h"
#include "third_party/highway/highway/x86_sse4.h"

namespace jxl {

template <typename T>
struct Raw256 {
  using type = __m256i;
};
template <>
struct Raw256<float> {
  using type = __m256;
};
template <>
struct Raw256<double> {
  using type = __m256d;
};

template <typename T>
using Full256 = Desc<T, 32 / sizeof(T)>;

template <typename T>
class Vec256 {
  using Raw = typename Raw256<T>::type;

 public:
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256() = default;
  Vec256(const Vec256&) = default;
  Vec256& operator=(const Vec256&) = default;
  SIMD_ATTR_AVX2 SIMD_INLINE explicit Vec256(const Raw raw) : raw(raw) {}

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator*=(const Vec256 other) {
    return *this = (*this * other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator/=(const Vec256 other) {
    return *this = (*this / other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator+=(const Vec256 other) {
    return *this = (*this + other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator-=(const Vec256 other) {
    return *this = (*this - other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator&=(const Vec256 other) {
    return *this = (*this & other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator|=(const Vec256 other) {
    return *this = (*this | other);
  }
  SIMD_ATTR_AVX2 SIMD_INLINE Vec256& operator^=(const Vec256 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// ------------------------------ Cast

SIMD_ATTR_AVX2 SIMD_INLINE __m256i BitCastToInteger(__m256i v) { return v; }
SIMD_ATTR_AVX2 SIMD_INLINE __m256i BitCastToInteger(__m256 v) {
  return _mm256_castps_si256(v);
}
SIMD_ATTR_AVX2 SIMD_INLINE __m256i BitCastToInteger(__m256d v) {
  return _mm256_castpd_si256(v);
}

// cast_to_u8
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> cast_to_u8(Vec256<T> v) {
  return Vec256<uint8_t>(BitCastToInteger(v.raw));
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromInteger256 {
  SIMD_ATTR_AVX2 SIMD_INLINE __m256i operator()(__m256i v) { return v; }
};
template <>
struct BitCastFromInteger256<float> {
  SIMD_ATTR_AVX2 SIMD_INLINE __m256 operator()(__m256i v) {
    return _mm256_castsi256_ps(v);
  }
};
template <>
struct BitCastFromInteger256<double> {
  SIMD_ATTR_AVX2 SIMD_INLINE __m256d operator()(__m256i v) {
    return _mm256_castsi256_pd(v);
  }
};

// cast_u8_to
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> cast_u8_to(Full256<T> /* tag */,
                                                Vec256<uint8_t> v) {
  return Vec256<T>(BitCastFromInteger256<T>()(v.raw));
}

// bit_cast
template <typename T, typename FromT>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> bit_cast(Full256<T> d, Vec256<FromT> v) {
  return cast_u8_to(d, cast_to_u8(v));
}

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> setzero(Full256<T> /* tag */) {
  return Vec256<T>(_mm256_setzero_si256());
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> setzero(Full256<float> /* tag */) {
  return Vec256<float>(_mm256_setzero_ps());
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> setzero(Full256<double> /* tag */) {
  return Vec256<double>(_mm256_setzero_pd());
}

template <typename T, typename T2>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> iota(Full256<T> d, const T2 first) {
  SIMD_ALIGN T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = first + i;
  }
  return load(d, lanes);
}

// Returns a vector with all lanes set to "t".
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> set1(Full256<uint8_t> /* tag */,
                                                const uint8_t t) {
  return Vec256<uint8_t>(_mm256_set1_epi8(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> set1(Full256<uint16_t> /* tag */,
                                                 const uint16_t t) {
  return Vec256<uint16_t>(_mm256_set1_epi16(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> set1(Full256<uint32_t> /* tag */,
                                                 const uint32_t t) {
  return Vec256<uint32_t>(_mm256_set1_epi32(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> set1(Full256<uint64_t> /* tag */,
                                                 const uint64_t t) {
  return Vec256<uint64_t>(_mm256_set1_epi64x(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> set1(Full256<int8_t> /* tag */,
                                               const int8_t t) {
  return Vec256<int8_t>(_mm256_set1_epi8(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> set1(Full256<int16_t> /* tag */,
                                                const int16_t t) {
  return Vec256<int16_t>(_mm256_set1_epi16(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> set1(Full256<int32_t> /* tag */,
                                                const int32_t t) {
  return Vec256<int32_t>(_mm256_set1_epi32(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> set1(Full256<int64_t> /* tag */,
                                                const int64_t t) {
  return Vec256<int64_t>(_mm256_set1_epi64x(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> set1(Full256<float> /* tag */,
                                              const float t) {
  return Vec256<float>(_mm256_set1_ps(t));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> set1(Full256<double> /* tag */,
                                               const double t) {
  return Vec256<double>(_mm256_set1_pd(t));
}

SIMD_DIAGNOSTICS(push)
SIMD_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> undefined(Full256<T> /* tag */) {
#ifdef __clang__
  return Vec256<T>(_mm256_undefined_si256());
#else
  __m256i raw;
  return Vec256<T>(raw);
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> undefined(Full256<float> /* tag */) {
#ifdef __clang__
  return Vec256<float>(_mm256_undefined_ps());
#else
  __m256 raw;
  return Vec256<float>(raw);
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> undefined(Full256<double> /* tag */) {
#ifdef __clang__
  return Vec256<double>(_mm256_undefined_pd());
#else
  __m256d raw;
  return Vec256<double>(raw);
#endif
}

SIMD_DIAGNOSTICS(pop)

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> operator&(const Vec256<T> a,
                                               const Vec256<T> b) {
  return Vec256<T>(_mm256_and_si256(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator&(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_and_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator&(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_and_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> andnot(const Vec256<T> not_mask,
                                            const Vec256<T> mask) {
  return Vec256<T>(_mm256_andnot_si256(not_mask.raw, mask.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> andnot(const Vec256<float> not_mask,
                                                const Vec256<float> mask) {
  return Vec256<float>(_mm256_andnot_ps(not_mask.raw, mask.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> andnot(const Vec256<double> not_mask,
                                                 const Vec256<double> mask) {
  return Vec256<double>(_mm256_andnot_pd(not_mask.raw, mask.raw));
}

// ------------------------------ Bitwise OR

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> operator|(const Vec256<T> a,
                                               const Vec256<T> b) {
  return Vec256<T>(_mm256_or_si256(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator|(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_or_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator|(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_or_pd(a.raw, b.raw));
}

// ------------------------------ Bitwise XOR

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> operator^(const Vec256<T> a,
                                               const Vec256<T> b) {
  return Vec256<T>(_mm256_xor_si256(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator^(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_xor_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator^(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_xor_pd(a.raw, b.raw));
}

// ------------------------------ Select/blend

// Returns a mask for use by if_then_else().
// blendv_ps/pd only check the sign bit, so this is a no-op on x86.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> mask_from_sign(const Vec256<T> v) {
  return v;
}

// Returns mask ? b : a. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> if_then_else(const Vec256<T> mask,
                                                  const Vec256<T> yes,
                                                  const Vec256<T> no) {
  return Vec256<T>(_mm256_blendv_epi8(no.raw, yes.raw, mask.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> if_then_else(const Vec256<float> mask,
                                                      const Vec256<float> yes,
                                                      const Vec256<float> no) {
  return Vec256<float>(_mm256_blendv_ps(no.raw, yes.raw, mask.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> if_then_else(
    const Vec256<double> mask, const Vec256<double> yes,
    const Vec256<double> no) {
  return Vec256<double>(_mm256_blendv_pd(no.raw, yes.raw, mask.raw));
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> operator+(const Vec256<uint8_t> a,
                                                     const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_add_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> operator+(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_add_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> operator+(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_add_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> operator+(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>(_mm256_add_epi64(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> operator+(const Vec256<int8_t> a,
                                                    const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_add_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> operator+(const Vec256<int16_t> a,
                                                     const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_add_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator+(const Vec256<int32_t> a,
                                                     const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_add_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> operator+(const Vec256<int64_t> a,
                                                     const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_add_epi64(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator+(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_add_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator+(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_add_pd(a.raw, b.raw));
}

// ------------------------------ Subtraction

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> operator-(const Vec256<uint8_t> a,
                                                     const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_sub_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> operator-(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_sub_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> operator-(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_sub_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> operator-(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>(_mm256_sub_epi64(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> operator-(const Vec256<int8_t> a,
                                                    const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_sub_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> operator-(const Vec256<int16_t> a,
                                                     const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_sub_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator-(const Vec256<int32_t> a,
                                                     const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_sub_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> operator-(const Vec256<int64_t> a,
                                                     const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_sub_epi64(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator-(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_sub_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator-(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_sub_pd(a.raw, b.raw));
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> saturated_add(
    const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_adds_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> saturated_add(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_adds_epu16(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> saturated_add(
    const Vec256<int8_t> a, const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_adds_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> saturated_add(
    const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_adds_epi16(a.raw, b.raw));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> saturated_subtract(
    const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_subs_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> saturated_subtract(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_subs_epu16(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> saturated_subtract(
    const Vec256<int8_t> a, const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_subs_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> saturated_subtract(
    const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_subs_epi16(a.raw, b.raw));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> average_round(
    const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_avg_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> average_round(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_avg_epu16(a.raw, b.raw));
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> abs(const Vec256<int8_t> v) {
  return Vec256<int8_t>(_mm256_abs_epi8(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> abs(const Vec256<int16_t> v) {
  return Vec256<int16_t>(_mm256_abs_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> abs(const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_abs_epi32(v.raw));
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> shift_left(
    const Vec256<uint16_t> v) {
  return Vec256<uint16_t>(_mm256_slli_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> shift_right(
    const Vec256<uint16_t> v) {
  return Vec256<uint16_t>(_mm256_srli_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shift_left(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>(_mm256_slli_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shift_right(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>(_mm256_srli_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> shift_left(
    const Vec256<uint64_t> v) {
  return Vec256<uint64_t>(_mm256_slli_epi64(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> shift_right(
    const Vec256<uint64_t> v) {
  return Vec256<uint64_t>(_mm256_srli_epi64(v.raw, kBits));
}

// Signed (no i64 shift_right)
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> shift_left(const Vec256<int16_t> v) {
  return Vec256<int16_t>(_mm256_slli_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> shift_right(
    const Vec256<int16_t> v) {
  return Vec256<int16_t>(_mm256_srai_epi16(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shift_left(const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_slli_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shift_right(
    const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_srai_epi32(v.raw, kBits));
}
template <int kBits>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> shift_left(const Vec256<int64_t> v) {
  return Vec256<int64_t>(_mm256_slli_epi64(v.raw, kBits));
}

// ------------------------------ Shift lanes by same variable #bits

// Returned by set_shift_*_count; opaque.
template <typename T>
struct ShiftLeftCount256 {
  __m128i raw;
};

template <typename T>
struct ShiftRightCount256 {
  __m128i raw;
};

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE ShiftLeftCount256<T> set_shift_left_count(
    Full256<T> /* tag */, const int bits) {
  return ShiftLeftCount256<T>{_mm_cvtsi32_si128(bits)};
}

// Same as shift_left_count on x86, but different on ARM.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE ShiftRightCount256<T> set_shift_right_count(
    Full256<T> /* tag */, const int bits) {
  return ShiftRightCount256<T>{_mm_cvtsi32_si128(bits)};
}

// Unsigned (no u8)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> shift_left_same(
    const Vec256<uint16_t> v, const ShiftLeftCount256<uint16_t> bits) {
  return Vec256<uint16_t>(_mm256_sll_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> shift_right_same(
    const Vec256<uint16_t> v, const ShiftRightCount256<uint16_t> bits) {
  return Vec256<uint16_t>(_mm256_srl_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shift_left_same(
    const Vec256<uint32_t> v, const ShiftLeftCount256<uint32_t> bits) {
  return Vec256<uint32_t>(_mm256_sll_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shift_right_same(
    const Vec256<uint32_t> v, const ShiftRightCount256<uint32_t> bits) {
  return Vec256<uint32_t>(_mm256_srl_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> shift_left_same(
    const Vec256<uint64_t> v, const ShiftLeftCount256<uint64_t> bits) {
  return Vec256<uint64_t>(_mm256_sll_epi64(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> shift_right_same(
    const Vec256<uint64_t> v, const ShiftRightCount256<uint64_t> bits) {
  return Vec256<uint64_t>(_mm256_srl_epi64(v.raw, bits.raw));
}

// Signed (no i8,i64)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> shift_left_same(
    const Vec256<int16_t> v, const ShiftLeftCount256<int16_t> bits) {
  return Vec256<int16_t>(_mm256_sll_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> shift_right_same(
    const Vec256<int16_t> v, const ShiftRightCount256<int16_t> bits) {
  return Vec256<int16_t>(_mm256_sra_epi16(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shift_left_same(
    const Vec256<int32_t> v, const ShiftLeftCount256<int32_t> bits) {
  return Vec256<int32_t>(_mm256_sll_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shift_right_same(
    const Vec256<int32_t> v, const ShiftRightCount256<int32_t> bits) {
  return Vec256<int32_t>(_mm256_sra_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> shift_left_same(
    const Vec256<int64_t> v, const ShiftLeftCount256<int64_t> bits) {
  return Vec256<int64_t>(_mm256_sll_epi64(v.raw, bits.raw));
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> operator<<(
    const Vec256<uint32_t> v, const Vec256<uint32_t> bits) {
  return Vec256<uint32_t>(_mm256_sllv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> operator>>(
    const Vec256<uint32_t> v, const Vec256<uint32_t> bits) {
  return Vec256<uint32_t>(_mm256_srlv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> operator<<(
    const Vec256<uint64_t> v, const Vec256<uint64_t> bits) {
  return Vec256<uint64_t>(_mm256_sllv_epi64(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> operator>>(
    const Vec256<uint64_t> v, const Vec256<uint64_t> bits) {
  return Vec256<uint64_t>(_mm256_srlv_epi64(v.raw, bits.raw));
}

// Signed (no i8,i16,i64)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator<<(
    const Vec256<int32_t> v, const Vec256<int32_t> bits) {
  return Vec256<int32_t>(_mm256_sllv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator>>(
    const Vec256<int32_t> v, const Vec256<int32_t> bits) {
  return Vec256<int32_t>(_mm256_srav_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> operator<<(
    const Vec256<int64_t> v, const Vec256<int64_t> bits) {
  return Vec256<int64_t>(_mm256_sllv_epi64(v.raw, bits.raw));
}

// Variable shift for SSE registers.
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint32_t> operator>>(
    const Vec128<uint32_t> v, const Vec128<uint32_t> bits) {
  return Vec128<uint32_t>(_mm_srlv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint32_t> operator<<(
    const Vec128<uint32_t> v, const Vec128<uint32_t> bits) {
  return Vec128<uint32_t>(_mm_sllv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint64_t> operator<<(
    const Vec128<uint64_t> v, const Vec128<uint64_t> bits) {
  return Vec128<uint64_t>(_mm_sllv_epi64(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint64_t> operator>>(
    const Vec128<uint64_t> v, const Vec128<uint64_t> bits) {
  return Vec128<uint64_t>(_mm_srlv_epi64(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<int32_t> operator<<(
    const Vec128<int32_t> v, const Vec128<int32_t> bits) {
  return Vec128<int32_t>(_mm_sllv_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<int32_t> operator>>(
    const Vec128<int32_t> v, const Vec128<int32_t> bits) {
  return Vec128<int32_t>(_mm_srav_epi32(v.raw, bits.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<int64_t> operator<<(
    const Vec128<int64_t> v, const Vec128<int64_t> bits) {
  return Vec128<int64_t>(_mm_sllv_epi64(v.raw, bits.raw));
}

// ------------------------------ Minimum

// Unsigned (no u64)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> min(const Vec256<uint8_t> a,
                                               const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_min_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> min(const Vec256<uint16_t> a,
                                                const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_min_epu16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> min(const Vec256<uint32_t> a,
                                                const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_min_epu32(a.raw, b.raw));
}

// Signed (no i64)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> min(const Vec256<int8_t> a,
                                              const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_min_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> min(const Vec256<int16_t> a,
                                               const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_min_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> min(const Vec256<int32_t> a,
                                               const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_min_epi32(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> min(const Vec256<float> a,
                                             const Vec256<float> b) {
  return Vec256<float>(_mm256_min_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> min(const Vec256<double> a,
                                              const Vec256<double> b) {
  return Vec256<double>(_mm256_min_pd(a.raw, b.raw));
}

// ------------------------------ Maximum

// Unsigned (no u64)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> max(const Vec256<uint8_t> a,
                                               const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_max_epu8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> max(const Vec256<uint16_t> a,
                                                const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_max_epu16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> max(const Vec256<uint32_t> a,
                                                const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_max_epu32(a.raw, b.raw));
}

// Signed (no i64)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> max(const Vec256<int8_t> a,
                                              const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_max_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> max(const Vec256<int16_t> a,
                                               const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_max_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> max(const Vec256<int32_t> a,
                                               const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_max_epi32(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> max(const Vec256<float> a,
                                             const Vec256<float> b) {
  return Vec256<float>(_mm256_max_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> max(const Vec256<double> a,
                                              const Vec256<double> b) {
  return Vec256<double>(_mm256_max_pd(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> clamp(const Vec256<T> v,
                                           const Vec256<T> lo,
                                           const Vec256<T> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> operator*(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_mullo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> operator*(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_mullo_epi32(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> operator*(const Vec256<int16_t> a,
                                                     const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_mullo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator*(const Vec256<int32_t> a,
                                                     const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_mullo_epi32(a.raw, b.raw));
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> mul_high(const Vec256<uint16_t> a,
                                                     const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_mulhi_epu16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> mul_high(const Vec256<int16_t> a,
                                                    const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_mulhi_epi16(a.raw, b.raw));
}

}  // namespace ext

// Returns (((a * b) >> 14) + 1) >> 1.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> mul_high_round(
    const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_mulhrs_epi16(a.raw, b.raw));
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> mul_even(const Vec256<int32_t> a,
                                                    const Vec256<int32_t> b) {
  return Vec256<int64_t>(_mm256_mul_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> mul_even(const Vec256<uint32_t> a,
                                                     const Vec256<uint32_t> b) {
  return Vec256<uint64_t>(_mm256_mul_epu32(a.raw, b.raw));
}

// ------------------------------ Floating-point negate

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> neg(const Vec256<float> v) {
  const Full256<float> df;
  const Full256<uint32_t> du;
  const auto sign = bit_cast(df, set1(du, 0x80000000u));
  return v ^ sign;
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> neg(const Vec256<double> v) {
  const Full256<double> df;
  const Full256<uint64_t> du;
  const auto sign = bit_cast(df, set1(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator*(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_mul_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator*(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_mul_pd(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator/(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_div_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator/(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_div_pd(a.raw, b.raw));
}

// Approximate reciprocal
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> approximate_reciprocal(
    const Vec256<float> v) {
  return Vec256<float>(_mm256_rcp_ps(v.raw));
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> mul_add(const Vec256<float> mul,
                                                 const Vec256<float> x,
                                                 const Vec256<float> add) {
  return Vec256<float>(_mm256_fmadd_ps(mul.raw, x.raw, add.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> mul_add(const Vec256<double> mul,
                                                  const Vec256<double> x,
                                                  const Vec256<double> add) {
  return Vec256<double>(_mm256_fmadd_pd(mul.raw, x.raw, add.raw));
}

// Returns add - mul * x
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> nmul_add(const Vec256<float> mul,
                                                  const Vec256<float> x,
                                                  const Vec256<float> add) {
  return Vec256<float>(_mm256_fnmadd_ps(mul.raw, x.raw, add.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> nmul_add(const Vec256<double> mul,
                                                   const Vec256<double> x,
                                                   const Vec256<double> add) {
  return Vec256<double>(_mm256_fnmadd_pd(mul.raw, x.raw, add.raw));
}

// Expresses addition/subtraction as FMA for higher throughput (but also
// higher latency) on HSW/BDW. Requires inline assembly because clang > 6
// 'optimizes' FMA by 1.0 to addition/subtraction. x86 offers 132, 213, 231
// forms (1=F, 2=M, 3=A); the first is also the destination.

// Returns x + add
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> fadd(Vec256<float> x,
                                              const Vec256<float> k1,
                                              const Vec256<float> add) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm("vfmadd132ps %2, %1, %0" : "+x"(x.raw) : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return Vec256<float>(_mm256_fmadd_ps(k1.raw, x.raw, add.raw));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> fadd(Vec256<double> x,
                                               const Vec256<double> k1,
                                               const Vec256<double> add) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm("vfmadd132pd %2, %1, %0" : "+x"(x.raw) : "x"(add.raw), "x"(k1.raw));
  return x;
#else
  return Vec256<double>(_mm256_fmadd_pd(k1.raw, x.raw, add.raw));
#endif
}

// Returns x - sub
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> fsub(Vec256<float> x,
                                              const Vec256<float> k1,
                                              const Vec256<float> sub) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm("vfmsub132ps %2, %1, %0" : "+x"(x.raw) : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return Vec256<float>(_mm256_fmsub_ps(k1.raw, x.raw, sub.raw));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> fsub(Vec256<double> x,
                                               const Vec256<double> k1,
                                               const Vec256<double> sub) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm("vfmsub132pd %2, %1, %0" : "+x"(x.raw) : "x"(sub.raw), "x"(k1.raw));
  return x;
#else
  return Vec256<double>(_mm256_fmsub_pd(k1.raw, x.raw, sub.raw));
#endif
}

// Returns -sub + x (clobbers sub register)
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> fnadd(Vec256<float> sub,
                                               const Vec256<float> k1,
                                               const Vec256<float> x) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm("vfnmadd132ps %2, %1, %0" : "+x"(sub.raw) : "x"(x.raw), "x"(k1.raw));
  return sub;
#else
  return Vec256<float>(_mm256_fnmadd_ps(sub.raw, k1.raw, x.raw));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> fnadd(Vec256<double> sub,
                                                const Vec256<double> k1,
                                                const Vec256<double> x) {
#if SIMD_COMPILER != SIMD_COMPILER_MSVC && defined(__AVX2__)
  asm("vfnmadd132pd %2, %1, %0" : "+x"(sub.raw) : "x"(x.raw), "x"(k1.raw));
  return sub;
#else
  return Vec256<double>(_mm256_fnmadd_pd(sub.raw, k1.raw, x.raw));
#endif
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> mul_subtract(const Vec256<float> mul,
                                                      const Vec256<float> x,
                                                      const Vec256<float> sub) {
  return Vec256<float>(_mm256_fmsub_ps(mul.raw, x.raw, sub.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> mul_subtract(
    const Vec256<double> mul, const Vec256<double> x,
    const Vec256<double> sub) {
  return Vec256<double>(_mm256_fmsub_pd(mul.raw, x.raw, sub.raw));
}

// Returns -mul * x - sub
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> nmul_subtract(
    const Vec256<float> mul, const Vec256<float> x, const Vec256<float> sub) {
  return Vec256<float>(_mm256_fnmsub_ps(mul.raw, x.raw, sub.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> nmul_subtract(
    const Vec256<double> mul, const Vec256<double> x,
    const Vec256<double> sub) {
  return Vec256<double>(_mm256_fnmsub_pd(mul.raw, x.raw, sub.raw));
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> sqrt(const Vec256<float> v) {
  return Vec256<float>(_mm256_sqrt_ps(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> sqrt(const Vec256<double> v) {
  return Vec256<double>(_mm256_sqrt_pd(v.raw));
}

// Approximate reciprocal square root
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> approximate_reciprocal_sqrt(
    const Vec256<float> v) {
  return Vec256<float>(_mm256_rsqrt_ps(v.raw));
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, tie to even
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> round(const Vec256<float> v) {
  return Vec256<float>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> round(const Vec256<double> v) {
  return Vec256<double>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Toward zero, aka truncate
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> trunc(const Vec256<float> v) {
  return Vec256<float>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> trunc(const Vec256<double> v) {
  return Vec256<double>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
}

// Toward +infinity, aka ceiling
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> ceil(const Vec256<float> v) {
  return Vec256<float>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> ceil(const Vec256<double> v) {
  return Vec256<double>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
}

// Toward -infinity, aka floor
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> floor(const Vec256<float> v) {
  return Vec256<float>(
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> floor(const Vec256<double> v) {
  return Vec256<double>(
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> operator==(const Vec256<uint8_t> a,
                                                      const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_cmpeq_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> operator==(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_cmpeq_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> operator==(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_cmpeq_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> operator==(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>(_mm256_cmpeq_epi64(a.raw, b.raw));
}

// Signed
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> operator==(const Vec256<int8_t> a,
                                                     const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_cmpeq_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> operator==(const Vec256<int16_t> a,
                                                      const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_cmpeq_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator==(const Vec256<int32_t> a,
                                                      const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_cmpeq_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> operator==(const Vec256<int64_t> a,
                                                      const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_cmpeq_epi64(a.raw, b.raw));
}

// Float
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator==(const Vec256<float> a,
                                                    const Vec256<float> b) {
  return Vec256<float>(_mm256_cmp_ps(a.raw, b.raw, _CMP_EQ_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator==(const Vec256<double> a,
                                                     const Vec256<double> b) {
  return Vec256<double>(_mm256_cmp_pd(a.raw, b.raw, _CMP_EQ_OQ));
}

// ------------------------------ Strict inequality

// Signed/float <
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> operator<(const Vec256<int8_t> a,
                                                    const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_cmpgt_epi8(b.raw, a.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> operator<(const Vec256<int16_t> a,
                                                     const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_cmpgt_epi16(b.raw, a.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator<(const Vec256<int32_t> a,
                                                     const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_cmpgt_epi32(b.raw, a.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> operator<(const Vec256<int64_t> a,
                                                     const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_cmpgt_epi64(b.raw, a.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator<(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_cmp_ps(a.raw, b.raw, _CMP_LT_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator<(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_cmp_pd(a.raw, b.raw, _CMP_LT_OQ));
}

// Signed/float >
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> operator>(const Vec256<int8_t> a,
                                                    const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_cmpgt_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> operator>(const Vec256<int16_t> a,
                                                     const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_cmpgt_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> operator>(const Vec256<int32_t> a,
                                                     const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_cmpgt_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> operator>(const Vec256<int64_t> a,
                                                     const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_cmpgt_epi64(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator>(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Vec256<float>(_mm256_cmp_ps(a.raw, b.raw, _CMP_GT_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator>(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Vec256<double>(_mm256_cmp_pd(a.raw, b.raw, _CMP_GT_OQ));
}

// ------------------------------ Weak inequality

// Float <= >=
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator<=(const Vec256<float> a,
                                                    const Vec256<float> b) {
  return Vec256<float>(_mm256_cmp_ps(a.raw, b.raw, _CMP_LE_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator<=(const Vec256<double> a,
                                                     const Vec256<double> b) {
  return Vec256<double>(_mm256_cmp_pd(a.raw, b.raw, _CMP_LE_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> operator>=(const Vec256<float> a,
                                                    const Vec256<float> b) {
  return Vec256<float>(_mm256_cmp_ps(a.raw, b.raw, _CMP_GE_OQ));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> operator>=(const Vec256<double> a,
                                                     const Vec256<double> b) {
  return Vec256<double>(_mm256_cmp_pd(a.raw, b.raw, _CMP_GE_OQ));
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> load(Full256<T> /* tag */,
                                          const T* SIMD_RESTRICT aligned) {
  return Vec256<T>(
      _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned)));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> load(
    Full256<float> /* tag */, const float* SIMD_RESTRICT aligned) {
  return Vec256<float>(_mm256_load_ps(aligned));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> load(
    Full256<double> /* tag */, const double* SIMD_RESTRICT aligned) {
  return Vec256<double>(_mm256_load_pd(aligned));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> load_u(Full256<T> /* tag */,
                                            const T* SIMD_RESTRICT p) {
  return Vec256<T>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> load_u(Full256<float> /* tag */,
                                                const float* SIMD_RESTRICT p) {
  return Vec256<float>(_mm256_loadu_ps(p));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> load_u(
    Full256<double> /* tag */, const double* SIMD_RESTRICT p) {
  return Vec256<double>(_mm256_loadu_pd(p));
}

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> load_dup128(
    Full256<T> /* tag */, const T* const SIMD_RESTRICT p) {
  // Clang 3.9 generates VINSERTF128 which is slower, but inline assembly leads
  // to "invalid output size for constraint" without -mavx2:
  // https://gcc.godbolt.org/z/-Jt_-F
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX2__)
  __m256i out;
  asm("vbroadcasti128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec256<T>(out);
#else
  return Vec256<T>(_mm256_broadcastsi128_si256(load_u(Full128<T>(), p).raw));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> load_dup128(
    Full256<float> /* tag */, const float* const SIMD_RESTRICT p) {
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX2__)
  __m256 out;
  asm("vbroadcastf128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec256<float>(out);
#else
  return Vec256<float>(_mm256_broadcast_ps(reinterpret_cast<const __m128*>(p)));
#endif
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> load_dup128(
    Full256<double> /* tag */, const double* const SIMD_RESTRICT p) {
#if (SIMD_COMPILER != SIMD_COMPILER_MSVC) && defined(__AVX2__)
  __m256d out;
  asm("vbroadcastf128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec256<double>(out);
#else
  return Vec256<double>(
      _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p)));
#endif
}

// ------------------------------ Store

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store(const Vec256<T> v, Full256<T> /* tag */,
                                      T* SIMD_RESTRICT aligned) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store(const Vec256<float> v,
                                      Full256<float> /* tag */,
                                      float* SIMD_RESTRICT aligned) {
  _mm256_store_ps(aligned, v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store(const Vec256<double> v,
                                      Full256<double> /* tag */,
                                      double* SIMD_RESTRICT aligned) {
  _mm256_store_pd(aligned, v.raw);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void store_u(const Vec256<T> v, Full256<T> /* tag */,
                                        T* SIMD_RESTRICT p) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store_u(const Vec256<float> v,
                                        Full256<float> /* tag */,
                                        float* SIMD_RESTRICT p) {
  _mm256_storeu_ps(p, v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void store_u(const Vec256<double> v,
                                        Full256<double> /* tag */,
                                        double* SIMD_RESTRICT p) {
  _mm256_storeu_pd(p, v.raw);
}

// ------------------------------ Non-temporal stores

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const Vec256<T> v, Full256<T> /* tag */,
                                       T* SIMD_RESTRICT aligned) {
  _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const Vec256<float> v,
                                       Full256<float> /* tag */,
                                       float* SIMD_RESTRICT aligned) {
  _mm256_stream_ps(aligned, v.raw);
}
SIMD_ATTR_AVX2 SIMD_INLINE void stream(const Vec256<double> v,
                                       Full256<double> /* tag */,
                                       double* SIMD_RESTRICT aligned) {
  _mm256_stream_pd(aligned, v.raw);
}

// ------------------------------ Gather

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> gather_offset_impl(
    SizeTag<4> /* tag */, Full256<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec256<int32_t> offset) {
  return Vec256<T>(_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), offset.raw, 1));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> gather_index_impl(
    SizeTag<4> /* tag */, Full256<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec256<int32_t> index) {
  return Vec256<T>(_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), index.raw, 4));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> gather_offset_impl(
    SizeTag<8> /* tag */, Full256<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec256<int64_t> offset) {
  return Vec256<T>(_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), offset.raw, 1));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> gather_index_impl(
    SizeTag<8> /* tag */, Full256<T> /* tag */, const T* SIMD_RESTRICT base,
    const Vec256<int64_t> index) {
  return Vec256<T>(_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), index.raw, 8));
}

template <typename T, typename Offset>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> gather_offset(
    Full256<T> d, const T* SIMD_RESTRICT base, const Vec256<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  return gather_offset_impl(SizeTag<sizeof(T)>(), d, base, offset);
}
template <typename T, typename Index>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> gather_index(Full256<T> d,
                                                  const T* SIMD_RESTRICT base,
                                                  const Vec256<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  return gather_index_impl(SizeTag<sizeof(T)>(), d, base, index);
}

template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> gather_offset<float>(
    Full256<float> /* tag */, const float* SIMD_RESTRICT base,
    const Vec256<int32_t> offset) {
  return Vec256<float>(_mm256_i32gather_ps(base, offset.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> gather_index<float>(
    Full256<float> /* tag */, const float* SIMD_RESTRICT base,
    const Vec256<int32_t> index) {
  return Vec256<float>(_mm256_i32gather_ps(base, index.raw, 4));
}

template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> gather_offset<double>(
    Full256<double> /* tag */, const double* SIMD_RESTRICT base,
    const Vec256<int64_t> offset) {
  return Vec256<double>(_mm256_i64gather_pd(base, offset.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> gather_index<double>(
    Full256<double> /* tag */, const double* SIMD_RESTRICT base,
    const Vec256<int64_t> index) {
  return Vec256<double>(_mm256_i64gather_pd(base, index.raw, 8));
}

}  // namespace ext

// ================================================== SWIZZLE

// ------------------------------ Extract half

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<T> lower_half(Vec256<T> v) {
  return Vec128<T>(_mm256_castsi256_si128(v.raw));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<float> lower_half(Vec256<float> v) {
  return Vec128<float>(_mm256_castps256_ps128(v.raw));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<double> lower_half(Vec256<double> v) {
  return Vec128<double>(_mm256_castpd256_pd128(v.raw));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<T> upper_half(Vec256<T> v) {
  return Vec128<T>(_mm256_extracti128_si256(v.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<float> upper_half(Vec256<float> v) {
  return Vec128<float>(_mm256_extractf128_ps(v.raw, 1));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<double> upper_half(Vec256<double> v) {
  return Vec128<double>(_mm256_extractf128_pd(v.raw, 1));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<T> get_half(Lower /* tag */, Vec256<T> v) {
  return lower_half(v);
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<T> get_half(Upper /* tag */, Vec256<T> v) {
  return upper_half(v);
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> shift_left_bytes(const Vec256<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bslli_epi128.
  return Vec256<T>(_mm256_slli_si256(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> shift_left_lanes(const Vec256<T> v) {
  return shift_left_bytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> shift_right_bytes(const Vec256<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bsrli_epi128.
  return Vec256<T>(_mm256_srli_si256(v.raw, kBytes));
}

template <int kLanes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> shift_right_lanes(const Vec256<T> v) {
  return shift_right_bytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> combine_shift_right_bytes(
    const Vec256<T> hi, const Vec256<T> lo) {
  const Full256<uint8_t> d8;
  const Vec256<uint8_t> extracted_bytes(
      _mm256_alignr_epi8(bit_cast(d8, hi).raw, bit_cast(d8, lo).raw, kBytes));
  return bit_cast(Full256<T>(), extracted_bytes);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> broadcast(
    const Vec256<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m256i lo = _mm256_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec256<uint16_t>(_mm256_unpacklo_epi64(lo, lo));
  } else {
    const __m256i hi =
        _mm256_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec256<uint16_t>(_mm256_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> broadcast(
    const Vec256<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> broadcast(
    const Vec256<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<uint64_t>(_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Signed
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> broadcast(const Vec256<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m256i lo = _mm256_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec256<int16_t>(_mm256_unpacklo_epi64(lo, lo));
  } else {
    const __m256i hi =
        _mm256_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec256<int16_t>(_mm256_unpackhi_epi64(hi, hi));
  }
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> broadcast(const Vec256<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<int32_t>(_mm256_shuffle_epi32(v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> broadcast(const Vec256<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<int64_t>(_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44));
}

// Float
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> broadcast(const Vec256<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x55 * kLane));
}
template <int kLane>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> broadcast(const Vec256<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<double>(_mm256_shuffle_pd(v.raw, v.raw, 15 * kLane));
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec256<int32_t> have lanes 7,6,5,4,3,2,1,0 (0 is
// least-significant). shuffle_0321 rotates four-lane blocks one lane to the
// right (the previous least-significant lane is now most-significant =>
// 47650321). These could also be implemented via combine_shift_right_bytes but
// the shuffle_abcd notation is more convenient.

// Swap 64-bit halves
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shuffle_1032(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shuffle_1032(
    const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> shuffle_1032(const Vec256<float> v) {
  // Shorter encoding than _mm256_permute_ps.
  return Vec256<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> shuffle_01(
    const Vec256<uint64_t> v) {
  return Vec256<uint64_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> shuffle_01(const Vec256<int64_t> v) {
  return Vec256<int64_t>(_mm256_shuffle_epi32(v.raw, 0x4E));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> shuffle_01(const Vec256<double> v) {
  // Shorter encoding than _mm256_permute_pd.
  return Vec256<double>(_mm256_shuffle_pd(v.raw, v.raw, 5));
}

// Rotate right 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shuffle_0321(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shuffle_0321(
    const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_shuffle_epi32(v.raw, 0x39));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> shuffle_0321(const Vec256<float> v) {
  return Vec256<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x39));
}
// Rotate left 32 bits
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shuffle_2103(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shuffle_2103(
    const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_shuffle_epi32(v.raw, 0x93));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> shuffle_2103(const Vec256<float> v) {
  return Vec256<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x93));
}

// Reverse
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> shuffle_0123(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>(_mm256_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> shuffle_0123(
    const Vec256<int32_t> v) {
  return Vec256<int32_t>(_mm256_shuffle_epi32(v.raw, 0x1B));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> shuffle_0123(const Vec256<float> v) {
  return Vec256<float>(_mm256_shuffle_ps(v.raw, v.raw, 0x1B));
}

// ------------------------------ Permute (runtime variable)

// Returned by set_table_indices for use by table_lookup_lanes.
template <typename T>
struct Permute256 {
  __m256i raw;
};

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Permute256<T> set_table_indices(const Full256<T> d,
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
  return Permute256<T>{load_u(Full256<int32_t>(), idx).raw};
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> table_lookup_lanes(
    const Vec256<uint32_t> v, const Permute256<uint32_t> idx) {
  return Vec256<uint32_t>(_mm256_permutevar8x32_epi32(v.raw, idx.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> table_lookup_lanes(
    const Vec256<int32_t> v, const Permute256<int32_t> idx) {
  return Vec256<int32_t>(_mm256_permutevar8x32_epi32(v.raw, idx.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> table_lookup_lanes(
    const Vec256<float> v, const Permute256<float> idx) {
  return Vec256<float>(_mm256_permutevar8x32_ps(v.raw, idx.raw));
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use zip_lo/hi instead (also works with scalar).

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> interleave_lo(
    const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> interleave_lo(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> interleave_lo(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> interleave_lo(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>(_mm256_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> interleave_lo(
    const Vec256<int8_t> a, const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> interleave_lo(
    const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> interleave_lo(
    const Vec256<int32_t> a, const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> interleave_lo(
    const Vec256<int64_t> a, const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_unpacklo_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> interleave_lo(const Vec256<float> a,
                                                       const Vec256<float> b) {
  return Vec256<float>(_mm256_unpacklo_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> interleave_lo(
    const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>(_mm256_unpacklo_pd(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint8_t> interleave_hi(
    const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> interleave_hi(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> interleave_hi(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> interleave_hi(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>(_mm256_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int8_t> interleave_hi(
    const Vec256<int8_t> a, const Vec256<int8_t> b) {
  return Vec256<int8_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> interleave_hi(
    const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> interleave_hi(
    const Vec256<int32_t> a, const Vec256<int32_t> b) {
  return Vec256<int32_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> interleave_hi(
    const Vec256<int64_t> a, const Vec256<int64_t> b) {
  return Vec256<int64_t>(_mm256_unpackhi_epi64(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> interleave_hi(const Vec256<float> a,
                                                       const Vec256<float> b) {
  return Vec256<float>(_mm256_unpackhi_ps(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> interleave_hi(
    const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>(_mm256_unpackhi_pd(a.raw, b.raw));
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> zip_lo(const Vec256<uint8_t> a,
                                                   const Vec256<uint8_t> b) {
  return Vec256<uint16_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> zip_lo(const Vec256<uint16_t> a,
                                                   const Vec256<uint16_t> b) {
  return Vec256<uint32_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> zip_lo(const Vec256<uint32_t> a,
                                                   const Vec256<uint32_t> b) {
  return Vec256<uint64_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> zip_lo(const Vec256<int8_t> a,
                                                  const Vec256<int8_t> b) {
  return Vec256<int16_t>(_mm256_unpacklo_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> zip_lo(const Vec256<int16_t> a,
                                                  const Vec256<int16_t> b) {
  return Vec256<int32_t>(_mm256_unpacklo_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> zip_lo(const Vec256<int32_t> a,
                                                  const Vec256<int32_t> b) {
  return Vec256<int64_t>(_mm256_unpacklo_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> zip_hi(const Vec256<uint8_t> a,
                                                   const Vec256<uint8_t> b) {
  return Vec256<uint16_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> zip_hi(const Vec256<uint16_t> a,
                                                   const Vec256<uint16_t> b) {
  return Vec256<uint32_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> zip_hi(const Vec256<uint32_t> a,
                                                   const Vec256<uint32_t> b) {
  return Vec256<uint64_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> zip_hi(const Vec256<int8_t> a,
                                                  const Vec256<int8_t> b) {
  return Vec256<int16_t>(_mm256_unpackhi_epi8(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> zip_hi(const Vec256<int16_t> a,
                                                  const Vec256<int16_t> b) {
  return Vec256<int32_t>(_mm256_unpackhi_epi16(a.raw, b.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> zip_hi(const Vec256<int32_t> a,
                                                  const Vec256<int32_t> b) {
  return Vec256<int64_t>(_mm256_unpackhi_epi32(a.raw, b.raw));
}

// ------------------------------ Parts

// Returns part of a vector (unspecified whether upper or lower).
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> any_part(Full256<T> /* tag */,
                                              const Vec256<T> v) {
  return v;  // full
}

template <typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<T, N> any_part(Desc<T, N> /* tag */,
                                                 const Vec256<T> v) {
  return Vec128<T, N>(_mm256_castsi256_si128(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<float, N> any_part(Desc<float, N> /* tag */,
                                                     Vec256<float> v) {
  return Vec128<float, N>(_mm256_castps256_ps128(v.raw));
}
template <size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<double, N> any_part(Desc<double, N> /* tag */,
                                                      Vec256<double> v) {
  return Vec128<double, N>(_mm256_castpd256_pd128(v.raw));
}

// Gets the single value stored in a vector/part.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE T get_lane(const Vec256<T> v) {
  return get_lane(any_part(Full128<T>(), v));
}

// Returns full vector with the given part's lane broadcasted. Note that
// callers cannot use broadcast directly because part lane order is undefined.
template <int kLane, typename T, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> broadcast_part(Full256<T> /* tag */,
                                                    const Vec128<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(Vec128<T>(v.raw));
  // Same as _mm256_castsi128_si256, but with guaranteed zero-extension.
  const auto lo = _mm256_zextsi128_si256(v128.raw);
  // Same instruction as _mm256_permute2f128_si256.
  return Vec256<T>(_mm256_permute2x128_si256(lo, lo, 0));
}
template <int kLane, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> broadcast_part(
    Full256<float> /* tag */, const Vec128<float, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(Vec128<float>(v.raw)).raw;
  // Same as _mm256_castps128_ps256, but with guaranteed zero-extension.
  const auto lo = _mm256_zextps128_ps256(v128);
  return Vec256<float>(_mm256_permute2f128_ps(lo, lo, 0));
}
template <int kLane, size_t N>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> broadcast_part(
    Full256<double> /* tag */, const Vec128<double, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  const auto v128 = broadcast<kLane>(Vec128<double>(v.raw)).raw;
  // Same as _mm256_castpd128_pd256, but with guaranteed zero-extension.
  const auto lo = _mm256_zextpd128_pd256(v128);
  return Vec256<double>(_mm256_permute2f128_pd(lo, lo, 0));
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> concat_lo_lo(const Vec256<T> hi,
                                                  const Vec256<T> lo) {
  return Vec256<T>(_mm256_permute2x128_si256(lo.raw, hi.raw, 0x20));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> concat_lo_lo(const Vec256<float> hi,
                                                      const Vec256<float> lo) {
  return Vec256<float>(_mm256_permute2f128_ps(lo.raw, hi.raw, 0x20));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> concat_lo_lo(
    const Vec256<double> hi, const Vec256<double> lo) {
  return Vec256<double>(_mm256_permute2f128_pd(lo.raw, hi.raw, 0x20));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> concat_hi_hi(const Vec256<T> hi,
                                                  const Vec256<T> lo) {
  return Vec256<T>(_mm256_permute2x128_si256(lo.raw, hi.raw, 0x31));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> concat_hi_hi(const Vec256<float> hi,
                                                      const Vec256<float> lo) {
  return Vec256<float>(_mm256_permute2f128_ps(lo.raw, hi.raw, 0x31));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> concat_hi_hi(
    const Vec256<double> hi, const Vec256<double> lo) {
  return Vec256<double>(_mm256_permute2f128_pd(lo.raw, hi.raw, 0x31));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves / swap blocks)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> concat_lo_hi(const Vec256<T> hi,
                                                  const Vec256<T> lo) {
  return Vec256<T>(_mm256_permute2x128_si256(lo.raw, hi.raw, 0x21));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> concat_lo_hi(const Vec256<float> hi,
                                                      const Vec256<float> lo) {
  return Vec256<float>(_mm256_permute2f128_ps(lo.raw, hi.raw, 0x21));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> concat_lo_hi(
    const Vec256<double> hi, const Vec256<double> lo) {
  return Vec256<double>(_mm256_permute2f128_pd(lo.raw, hi.raw, 0x21));
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> concat_hi_lo(const Vec256<T> hi,
                                                  const Vec256<T> lo) {
  return Vec256<T>(_mm256_blend_epi32(hi.raw, lo.raw, 0x0F));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> concat_hi_lo(const Vec256<float> hi,
                                                      const Vec256<float> lo) {
  return Vec256<float>(_mm256_blend_ps(hi.raw, lo.raw, 0x0F));
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> concat_hi_lo(
    const Vec256<double> hi, const Vec256<double> lo) {
  return Vec256<double>(_mm256_blend_pd(hi.raw, lo.raw, 3));
}

// ------------------------------ Odd/even lanes

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> odd_even_impl(SizeTag<1> /* tag */,
                                                   const Vec256<T> a,
                                                   const Vec256<T> b) {
  const Full256<T> d;
  const Full256<uint8_t> d8;
  SIMD_ALIGN constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                           0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return if_then_else(bit_cast(d, load_dup128(d8, mask)), b, a);
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> odd_even_impl(SizeTag<2> /* tag */,
                                                   const Vec256<T> a,
                                                   const Vec256<T> b) {
  return Vec256<T>(_mm256_blend_epi16(a.raw, b.raw, 0x55));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> odd_even_impl(SizeTag<4> /* tag */,
                                                   const Vec256<T> a,
                                                   const Vec256<T> b) {
  return Vec256<T>(_mm256_blend_epi32(a.raw, b.raw, 0x55));
}
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> odd_even_impl(SizeTag<8> /* tag */,
                                                   const Vec256<T> a,
                                                   const Vec256<T> b) {
  return Vec256<T>(_mm256_blend_epi32(a.raw, b.raw, 0x33));
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> odd_even(const Vec256<T> a,
                                              const Vec256<T> b) {
  return odd_even_impl(SizeTag<sizeof(T)>(), a, b);
}
template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> odd_even<float>(
    const Vec256<float> a, const Vec256<float> b) {
  return Vec256<float>(_mm256_blend_ps(a.raw, b.raw, 0x55));
}

template <>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> odd_even<double>(
    const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>(_mm256_blend_pd(a.raw, b.raw, 5));
}

// ================================================== CONVERT

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> table_lookup_bytes(const Vec256<T> bytes,
                                                        const Vec256<TI> from) {
  return Vec256<T>(_mm256_shuffle_epi8(bytes.raw, from.raw));
}

// ------------------------------ Promotions (part w/ narrow lanes -> full)

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<double> convert_to(Full256<double> /* tag */,
                                                     const Vec128<float, 4> v) {
  return Vec256<double>(_mm256_cvtps_pd(v.raw));
}

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> convert_to(
    Full256<uint16_t> /* tag */, Vec128<uint8_t> v) {
  return Vec256<uint16_t>(_mm256_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> convert_to(
    Full256<uint32_t> /* tag */, Vec128<uint8_t, 8> v) {
  return Vec256<uint32_t>(_mm256_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> convert_to(
    Full256<int16_t> /* tag */, Vec128<uint8_t> v) {
  return Vec256<int16_t>(_mm256_cvtepu8_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> convert_to(
    Full256<int32_t> /* tag */, Vec128<uint8_t, 8> v) {
  return Vec256<int32_t>(_mm256_cvtepu8_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> convert_to(
    Full256<uint32_t> /* tag */, Vec128<uint16_t> v) {
  return Vec256<uint32_t>(_mm256_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> convert_to(
    Full256<int32_t> /* tag */, Vec128<uint16_t> v) {
  return Vec256<int32_t>(_mm256_cvtepu16_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> convert_to(
    Full256<uint64_t> /* tag */, Vec128<uint32_t> v) {
  return Vec256<uint64_t>(_mm256_cvtepu32_epi64(v.raw));
}

// Special case for "v" with all blocks equal (e.g. from broadcast_block or
// load_dup128): single-cycle latency instead of 3.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint32_t> u32_from_u8(
    const Vec256<uint8_t> v) {
  const Full256<uint32_t> d32;
  SIMD_ALIGN static constexpr uint32_t k32From8[8] = {
      0xFFFFFF00UL, 0xFFFFFF01UL, 0xFFFFFF02UL, 0xFFFFFF03UL,
      0xFFFFFF04UL, 0xFFFFFF05UL, 0xFFFFFF06UL, 0xFFFFFF07UL};
  return table_lookup_bytes(bit_cast(d32, v), load(d32, k32From8));
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then zip_hi/lo followed by
// signed shift would be faster.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int16_t> convert_to(
    Full256<int16_t> /* tag */, Vec128<int8_t> v) {
  return Vec256<int16_t>(_mm256_cvtepi8_epi16(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> convert_to(
    Full256<int32_t> /* tag */, Vec128<int8_t, 8> v) {
  return Vec256<int32_t>(_mm256_cvtepi8_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> convert_to(
    Full256<int32_t> /* tag */, Vec128<int16_t> v) {
  return Vec256<int32_t>(_mm256_cvtepi16_epi32(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int64_t> convert_to(
    Full256<int64_t> /* tag */, Vec128<int32_t> v) {
  return Vec256<int64_t>(_mm256_cvtepi32_epi64(v.raw));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint16_t> convert_to(
    Full128<uint16_t> /* tag */, const Vec256<int32_t> v) {
  const __m256i u16 = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenating lower halves of both 128-bit blocks afterward is more
  // efficient than an extra input with low block = high block of v.
  return Vec128<uint16_t>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u16, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint8_t, 8> convert_to(
    Desc<uint8_t, 8> /* tag */, const Vec256<int32_t> v) {
  const __m256i u16_blocks = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i u16_concat = _mm256_permute4x64_epi64(u16_blocks, 0x88);
  const __m128i u16 = _mm256_castsi256_si128(u16_concat);
  return Vec128<uint8_t, 8>(_mm_packus_epi16(u16, u16));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec128<int16_t> convert_to(
    Full128<int16_t> /* tag */, const Vec256<int32_t> v) {
  const __m256i i16 = _mm256_packs_epi32(v.raw, v.raw);
  return Vec128<int16_t>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i16, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec128<int8_t, 8> convert_to(
    Desc<int8_t, 8> /* tag */, const Vec256<int32_t> v) {
  const __m256i i16_blocks = _mm256_packs_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i i16_concat = _mm256_permute4x64_epi64(i16_blocks, 0x88);
  const __m128i i16 = _mm256_castsi256_si128(i16_concat);
  return Vec128<int8_t, 8>(_mm_packs_epi16(i16, i16));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint8_t> convert_to(
    Full128<uint8_t> /* tag */, const Vec256<int16_t> v) {
  const __m256i u8 = _mm256_packus_epi16(v.raw, v.raw);
  return Vec128<uint8_t>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u8, 0x88)));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec128<int8_t> convert_to(Full128<int8_t> /* tag */,
                                                     const Vec256<int16_t> v) {
  const __m256i i8 = _mm256_packs_epi16(v.raw, v.raw);
  return Vec128<int8_t>(
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i8, 0x88)));
}

// For already range-limited input [0, 255].
SIMD_ATTR_AVX2 SIMD_INLINE Vec128<uint8_t, 8> u8_from_u32(
    const Vec256<uint32_t> v) {
  const Full256<uint32_t> d32;
  SIMD_ALIGN static constexpr uint32_t k8From32[8] = {
      0x0C080400u, ~0u, ~0u, ~0u, ~0u, 0x0C080400u, ~0u, ~0u};
  // Place first four bytes in lo[0], remaining 4 in hi[1].
  const auto quad = table_lookup_bytes(v, load(d32, k8From32));
  // Interleave both quadruplets - OR instead of unpack reduces port5 pressure.
  const auto lo = lower_half(quad);
  const auto hi = upper_half(quad);
  const auto pair = lower_half(lo | hi);
  return bit_cast(Desc<uint8_t, 8>(), pair);
}

// ------------------------------ Convert i32 <=> f32

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<float> convert_to(Full256<float> /* tag */,
                                                    const Vec256<int32_t> v) {
  return Vec256<float>(_mm256_cvtepi32_ps(v.raw));
}
// Truncates (rounds toward zero).
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> convert_to(
    Full256<int32_t> /* tag */, const Vec256<float> v) {
  return Vec256<int32_t>(_mm256_cvttps_epi32(v.raw));
}

SIMD_ATTR_AVX2 SIMD_INLINE Vec256<int32_t> nearest_int(const Vec256<float> v) {
  return Vec256<int32_t>(_mm256_cvtps_epi32(v.raw));
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ movemask

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..31 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_ATTR_AVX2 SIMD_INLINE uint64_t movemask(const Vec256<uint8_t> v) {
  // Prevent sign-extension of 32-bit masks because the intrinsic returns int.
  return static_cast<uint32_t>(_mm256_movemask_epi8(v.raw));
}

// Returns the most significant bit of each float/double lane (see above).
SIMD_ATTR_AVX2 SIMD_INLINE uint64_t movemask(const Vec256<float> v) {
  return static_cast<unsigned>(_mm256_movemask_ps(v.raw));
}
SIMD_ATTR_AVX2 SIMD_INLINE uint64_t movemask(const Vec256<double> v) {
  return static_cast<unsigned>(_mm256_movemask_pd(v.raw));
}

// ------------------------------ all_zero

// Returns whether all lanes are equal to zero. Supported for all integer V.
// (Floating-point VTESTP* only test the sign bit!)
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE bool all_zero(const Vec256<T> v) {
  return static_cast<bool>(_mm256_testz_si256(v.raw, v.raw));
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint64_t> sums_of_u8x8(
    const Vec256<uint8_t> v) {
  return Vec256<uint64_t>(_mm256_sad_epu8(v.raw, _mm256_setzero_si256()));
}

// Returns N sums of differences of byte quadruplets, starting from byte offset
// i = [0, N) in window (11 consecutive bytes) and idx_ref * 4 in ref.
// This version computes two independent SAD with separate idx_ref.
template <int idx_ref1, int idx_ref0>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<uint16_t> mpsadbw2(
    const Vec256<uint8_t> window, const Vec256<uint8_t> ref) {
  static_assert(idx_ref0 < 4 && idx_ref1 < 4, "a_offset must be 0");
  return Vec256<uint16_t>(
      _mm256_mpsadbw_epu8(window.raw, ref.raw, (idx_ref1 << 3) + idx_ref0));
}

// Returns sum{lane[i]} in each lane. "v3210" is a replicated 128-bit block.
// Same logic as x86_sse4.h, but with Vec256 arguments.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> horz_sum_impl(SizeTag<4> /* tag */,
                                                   const Vec256<T> v3210) {
  const auto v1032 = shuffle_1032(v3210);
  const auto v31_20_31_20 = v3210 + v1032;
  const auto v20_31_20_31 = shuffle_0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> horz_sum_impl(SizeTag<8> /* tag */,
                                                   const Vec256<T> v10) {
  const auto v01 = shuffle_01(v10);
  return v10 + v01;
}

// Supported for {uif}32x8, {uif}64x4. Returns the sum in each lane.
template <typename T>
SIMD_ATTR_AVX2 SIMD_INLINE Vec256<T> sum_of_lanes(const Vec256<T> vHL) {
  const Vec256<T> vLH = concat_lo_hi(vHL, vHL);
  return horz_sum_impl(SizeTag<sizeof(T)>(), vLH + vHL);
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).
}  // namespace jxl

#endif  // HIGHWAY_X86_AVX2_H_
