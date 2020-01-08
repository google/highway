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

#ifndef HWY_X86_AVX2_H_
#define HWY_X86_AVX2_H_

// 256-bit AVX2 vectors and operations.
// WARNING: most operations do not cross 128-bit block boundaries. In
// particular, "Broadcast", pack and zip behavior may be surprising.

#include <immintrin.h>

#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
#include <stdio.h>
#endif

#include "hwy/shared.h"
#include "hwy/x86/sse4.h"

#ifdef HWY_DISABLE_BMI2  // See runtime_dispatch.cc
#define HWY_ATTR_AVX2 HWY_TARGET_ATTR("avx,avx2,fma")
#else
#define HWY_ATTR_AVX2 HWY_TARGET_ATTR("bmi,bmi2,avx,avx2,fma")
#endif

namespace hwy {

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
  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator*=(const Vec256 other) {
    return *this = (*this * other);
  }
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator/=(const Vec256 other) {
    return *this = (*this / other);
  }
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator+=(const Vec256 other) {
    return *this = (*this + other);
  }
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator-=(const Vec256 other) {
    return *this = (*this - other);
  }
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator&=(const Vec256 other) {
    return *this = (*this & other);
  }
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator|=(const Vec256 other) {
    return *this = (*this | other);
  }
  HWY_ATTR_AVX2 HWY_INLINE Vec256& operator^=(const Vec256 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

// Integer: FF..FF or 0. Float: MSB, all other bits undefined - see README.
template <typename T>
class Mask256 {
  using Raw = typename Raw256<T>::type;

 public:
  Raw raw;
};

// ------------------------------ Cast

HWY_ATTR_AVX2 HWY_INLINE __m256i BitCastToInteger(__m256i v) { return v; }
HWY_ATTR_AVX2 HWY_INLINE __m256i BitCastToInteger(__m256 v) {
  return _mm256_castps_si256(v);
}
HWY_ATTR_AVX2 HWY_INLINE __m256i BitCastToInteger(__m256d v) {
  return _mm256_castpd_si256(v);
}

// cast_to_u8
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> cast_to_u8(Vec256<T> v) {
  return Vec256<uint8_t>{BitCastToInteger(v.raw)};
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromInteger256 {
  HWY_ATTR_AVX2 HWY_INLINE __m256i operator()(__m256i v) { return v; }
};
template <>
struct BitCastFromInteger256<float> {
  HWY_ATTR_AVX2 HWY_INLINE __m256 operator()(__m256i v) {
    return _mm256_castsi256_ps(v);
  }
};
template <>
struct BitCastFromInteger256<double> {
  HWY_ATTR_AVX2 HWY_INLINE __m256d operator()(__m256i v) {
    return _mm256_castsi256_pd(v);
  }
};

// cast_u8_to
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> cast_u8_to(Full256<T> /* tag */,
                                              Vec256<uint8_t> v) {
  return Vec256<T>{BitCastFromInteger256<T>()(v.raw)};
}

// BitCast
template <typename T, typename FromT>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> BitCast(Full256<T> d, Vec256<FromT> v) {
  return cast_u8_to(d, cast_to_u8(v));
}

// ------------------------------ Set

// Returns an all-zero vector.
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> Zero(Full256<T> /* tag */) {
  return Vec256<T>{_mm256_setzero_si256()};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Zero(Full256<float> /* tag */) {
  return Vec256<float>{_mm256_setzero_ps()};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Zero(Full256<double> /* tag */) {
  return Vec256<double>{_mm256_setzero_pd()};
}

// Returns a vector with all lanes set to "t".
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> Set(Full256<uint8_t> /* tag */,
                                             const uint8_t t) {
  return Vec256<uint8_t>{_mm256_set1_epi8(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> Set(Full256<uint16_t> /* tag */,
                                              const uint16_t t) {
  return Vec256<uint16_t>{_mm256_set1_epi16(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Set(Full256<uint32_t> /* tag */,
                                              const uint32_t t) {
  return Vec256<uint32_t>{_mm256_set1_epi32(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> Set(Full256<uint64_t> /* tag */,
                                              const uint64_t t) {
  return Vec256<uint64_t>{_mm256_set1_epi64x(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> Set(Full256<int8_t> /* tag */,
                                            const int8_t t) {
  return Vec256<int8_t>{_mm256_set1_epi8(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> Set(Full256<int16_t> /* tag */,
                                             const int16_t t) {
  return Vec256<int16_t>{_mm256_set1_epi16(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Set(Full256<int32_t> /* tag */,
                                             const int32_t t) {
  return Vec256<int32_t>{_mm256_set1_epi32(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> Set(Full256<int64_t> /* tag */,
                                             const int64_t t) {
  return Vec256<int64_t>{_mm256_set1_epi64x(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Set(Full256<float> /* tag */,
                                           const float t) {
  return Vec256<float>{_mm256_set1_ps(t)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Set(Full256<double> /* tag */,
                                            const double t) {
  return Vec256<double>{_mm256_set1_pd(t)};
}

template <typename T, typename T2>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> Iota(Full256<T> d, T2 first) {
  alignas(32) T lanes[d.N];
  for (size_t i = 0; i < d.N; ++i) {
    lanes[i] = first + i;
  }
  return Load(d, lanes);
}

HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> Undefined(Full256<T> /* tag */) {
#ifdef __clang__
  return Vec256<T>{_mm256_undefined_si256()};
#else
  __m256i raw;
  return Vec256<T>{raw};
#endif
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Undefined(Full256<float> /* tag */) {
#ifdef __clang__
  return Vec256<float>{_mm256_undefined_ps()};
#else
  __m256 raw;
  return Vec256<float>{raw};
#endif
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Undefined(Full256<double> /* tag */) {
#ifdef __clang__
  return Vec256<double>{_mm256_undefined_pd()};
#else
  __m256d raw;
  return Vec256<double>{raw};
#endif
}

HWY_DIAGNOSTICS(pop)

// ================================================== LOGICAL

// ------------------------------ Bitwise AND

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> operator&(Vec256<T> a, Vec256<T> b) {
  return Vec256<T>{_mm256_and_si256(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator&(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_and_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator&(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_and_pd(a.raw, b.raw)};
}

// ------------------------------ Bitwise AND-NOT

// Returns ~not_mask & mask.
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> AndNot(Vec256<T> not_mask, Vec256<T> mask) {
  return Vec256<T>{_mm256_andnot_si256(not_mask.raw, mask.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> AndNot(const Vec256<float> not_mask,
                                              const Vec256<float> mask) {
  return Vec256<float>{_mm256_andnot_ps(not_mask.raw, mask.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> AndNot(const Vec256<double> not_mask,
                                               const Vec256<double> mask) {
  return Vec256<double>{_mm256_andnot_pd(not_mask.raw, mask.raw)};
}

// ------------------------------ Bitwise OR

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> operator|(Vec256<T> a, Vec256<T> b) {
  return Vec256<T>{_mm256_or_si256(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator|(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_or_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator|(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_or_pd(a.raw, b.raw)};
}

// ------------------------------ Bitwise XOR

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> operator^(Vec256<T> a, Vec256<T> b) {
  return Vec256<T>{_mm256_xor_si256(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator^(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_xor_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator^(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_xor_pd(a.raw, b.raw)};
}

// ------------------------------ Mask

// Mask and Vec are the same (true = FF..FF).
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Mask256<T> MaskFromVec(const Vec256<T> v) {
  return Mask256<T>{v.raw};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> VecFromMask(const Mask256<T> v) {
  return Vec256<T>{v.raw};
}

// mask ? yes : no
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> IfThenElse(const Mask256<T> mask,
                                              const Vec256<T> yes,
                                              const Vec256<T> no) {
  return Vec256<T>{_mm256_blendv_epi8(no.raw, yes.raw, mask.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> IfThenElse(const Mask256<float> mask,
                                                  const Vec256<float> yes,
                                                  const Vec256<float> no) {
  return Vec256<float>{_mm256_blendv_ps(no.raw, yes.raw, mask.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> IfThenElse(const Mask256<double> mask,
                                                   const Vec256<double> yes,
                                                   const Vec256<double> no) {
  return Vec256<double>{_mm256_blendv_pd(no.raw, yes.raw, mask.raw)};
}

// mask ? yes : 0
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> IfThenElseZero(Mask256<T> mask,
                                                  Vec256<T> yes) {
  return yes & VecFromMask(mask);
}

// mask ? 0 : no
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> IfThenZeroElse(Mask256<T> mask,
                                                  Vec256<T> no) {
  return AndNot(VecFromMask(mask), no);
}

template <typename T, HWY_IF_FLOAT(T)>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ZeroIfNegative(Vec256<T> v) {
  const auto zero = Zero(Full256<T>());
  return IfThenElse(MaskFromVec(v), zero, v);
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> operator+(const Vec256<uint8_t> a,
                                                   const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_add_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> operator+(const Vec256<uint16_t> a,
                                                    const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_add_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> operator+(const Vec256<uint32_t> a,
                                                    const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_add_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> operator+(const Vec256<uint64_t> a,
                                                    const Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_add_epi64(a.raw, b.raw)};
}

// Signed
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> operator+(const Vec256<int8_t> a,
                                                  const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_add_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> operator+(const Vec256<int16_t> a,
                                                   const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_add_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> operator+(const Vec256<int32_t> a,
                                                   const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_add_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> operator+(const Vec256<int64_t> a,
                                                   const Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_add_epi64(a.raw, b.raw)};
}

// Float
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator+(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_add_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator+(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_add_pd(a.raw, b.raw)};
}

// ------------------------------ Subtraction

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> operator-(const Vec256<uint8_t> a,
                                                   const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_sub_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> operator-(const Vec256<uint16_t> a,
                                                    const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_sub_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> operator-(const Vec256<uint32_t> a,
                                                    const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_sub_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> operator-(const Vec256<uint64_t> a,
                                                    const Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_sub_epi64(a.raw, b.raw)};
}

// Signed
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> operator-(const Vec256<int8_t> a,
                                                  const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_sub_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> operator-(const Vec256<int16_t> a,
                                                   const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_sub_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> operator-(const Vec256<int32_t> a,
                                                   const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_sub_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> operator-(const Vec256<int64_t> a,
                                                   const Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_sub_epi64(a.raw, b.raw)};
}

// Float
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator-(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_sub_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator-(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_sub_pd(a.raw, b.raw)};
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> SaturatedAdd(const Vec256<uint8_t> a,
                                                      const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_adds_epu8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> SaturatedAdd(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_adds_epu16(a.raw, b.raw)};
}

// Signed
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> SaturatedAdd(const Vec256<int8_t> a,
                                                     const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_adds_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> SaturatedAdd(const Vec256<int16_t> a,
                                                      const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_adds_epi16(a.raw, b.raw)};
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> SaturatedSub(const Vec256<uint8_t> a,
                                                      const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_subs_epu8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> SaturatedSub(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_subs_epu16(a.raw, b.raw)};
}

// Signed
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> SaturatedSub(const Vec256<int8_t> a,
                                                     const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_subs_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> SaturatedSub(const Vec256<int16_t> a,
                                                      const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_subs_epi16(a.raw, b.raw)};
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> AverageRound(const Vec256<uint8_t> a,
                                                      const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_avg_epu8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> AverageRound(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_avg_epu16(a.raw, b.raw)};
}

// ------------------------------ Absolute value

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> Abs(const Vec256<int8_t> v) {
  return Vec256<int8_t>{_mm256_abs_epi8(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> Abs(const Vec256<int16_t> v) {
  return Vec256<int16_t>{_mm256_abs_epi16(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Abs(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_abs_epi32(v.raw)};
}

// ------------------------------ Shift lanes by constant #bits

// Unsigned
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> ShiftLeft(const Vec256<uint16_t> v) {
  return Vec256<uint16_t>{_mm256_slli_epi16(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> ShiftRight(const Vec256<uint16_t> v) {
  return Vec256<uint16_t>{_mm256_srli_epi16(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> ShiftLeft(const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_slli_epi32(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> ShiftRight(const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_srli_epi32(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> ShiftLeft(const Vec256<uint64_t> v) {
  return Vec256<uint64_t>{_mm256_slli_epi64(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> ShiftRight(const Vec256<uint64_t> v) {
  return Vec256<uint64_t>{_mm256_srli_epi64(v.raw, kBits)};
}

// Signed (no i64 ShiftRight)
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> ShiftLeft(const Vec256<int16_t> v) {
  return Vec256<int16_t>{_mm256_slli_epi16(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> ShiftRight(const Vec256<int16_t> v) {
  return Vec256<int16_t>{_mm256_srai_epi16(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ShiftLeft(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_slli_epi32(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ShiftRight(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_srai_epi32(v.raw, kBits)};
}
template <int kBits>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> ShiftLeft(const Vec256<int64_t> v) {
  return Vec256<int64_t>{_mm256_slli_epi64(v.raw, kBits)};
}

// ------------------------------ Shift lanes by independent variable #bits

// Unsigned (no u8,u16)
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> operator<<(
    const Vec256<uint32_t> v, const Vec256<uint32_t> bits) {
  return Vec256<uint32_t>{_mm256_sllv_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> operator>>(
    const Vec256<uint32_t> v, const Vec256<uint32_t> bits) {
  return Vec256<uint32_t>{_mm256_srlv_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> operator<<(
    const Vec256<uint64_t> v, const Vec256<uint64_t> bits) {
  return Vec256<uint64_t>{_mm256_sllv_epi64(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> operator>>(
    const Vec256<uint64_t> v, const Vec256<uint64_t> bits) {
  return Vec256<uint64_t>{_mm256_srlv_epi64(v.raw, bits.raw)};
}

// Signed (no i8,i16,i64)
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> operator<<(
    const Vec256<int32_t> v, const Vec256<int32_t> bits) {
  return Vec256<int32_t>{_mm256_sllv_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> operator>>(
    const Vec256<int32_t> v, const Vec256<int32_t> bits) {
  return Vec256<int32_t>{_mm256_srav_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> operator<<(
    const Vec256<int64_t> v, const Vec256<int64_t> bits) {
  return Vec256<int64_t>{_mm256_sllv_epi64(v.raw, bits.raw)};
}

// Variable shift for SSE registers.
HWY_ATTR_AVX2 HWY_INLINE Vec128<uint32_t> operator>>(
    const Vec128<uint32_t> v, const Vec128<uint32_t> bits) {
  return Vec128<uint32_t>{_mm_srlv_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec128<uint32_t> operator<<(
    const Vec128<uint32_t> v, const Vec128<uint32_t> bits) {
  return Vec128<uint32_t>{_mm_sllv_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec128<uint64_t> operator<<(
    const Vec128<uint64_t> v, const Vec128<uint64_t> bits) {
  return Vec128<uint64_t>{_mm_sllv_epi64(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec128<uint64_t> operator>>(
    const Vec128<uint64_t> v, const Vec128<uint64_t> bits) {
  return Vec128<uint64_t>{_mm_srlv_epi64(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec128<int32_t> operator<<(
    const Vec128<int32_t> v, const Vec128<int32_t> bits) {
  return Vec128<int32_t>{_mm_sllv_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec128<int32_t> operator>>(
    const Vec128<int32_t> v, const Vec128<int32_t> bits) {
  return Vec128<int32_t>{_mm_srav_epi32(v.raw, bits.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec128<int64_t> operator<<(
    const Vec128<int64_t> v, const Vec128<int64_t> bits) {
  return Vec128<int64_t>{_mm_sllv_epi64(v.raw, bits.raw)};
}

// ------------------------------ Minimum

// Unsigned (no u64)
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> Min(const Vec256<uint8_t> a,
                                             const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_min_epu8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> Min(const Vec256<uint16_t> a,
                                              const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_min_epu16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Min(const Vec256<uint32_t> a,
                                              const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_min_epu32(a.raw, b.raw)};
}

// Signed (no i64)
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> Min(const Vec256<int8_t> a,
                                            const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_min_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> Min(const Vec256<int16_t> a,
                                             const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_min_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Min(const Vec256<int32_t> a,
                                             const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_min_epi32(a.raw, b.raw)};
}

// Float
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Min(const Vec256<float> a,
                                           const Vec256<float> b) {
  return Vec256<float>{_mm256_min_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Min(const Vec256<double> a,
                                            const Vec256<double> b) {
  return Vec256<double>{_mm256_min_pd(a.raw, b.raw)};
}

// ------------------------------ Maximum

// Unsigned (no u64)
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> Max(const Vec256<uint8_t> a,
                                             const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_max_epu8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> Max(const Vec256<uint16_t> a,
                                              const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_max_epu16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Max(const Vec256<uint32_t> a,
                                              const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_max_epu32(a.raw, b.raw)};
}

// Signed (no i64)
HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> Max(const Vec256<int8_t> a,
                                            const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_max_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> Max(const Vec256<int16_t> a,
                                             const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_max_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Max(const Vec256<int32_t> a,
                                             const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_max_epi32(a.raw, b.raw)};
}

// Float
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Max(const Vec256<float> a,
                                           const Vec256<float> b) {
  return Vec256<float>{_mm256_max_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Max(const Vec256<double> a,
                                            const Vec256<double> b) {
  return Vec256<double>{_mm256_max_pd(a.raw, b.raw)};
}

// Returns the closest value to v within [lo, hi].
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> Clamp(const Vec256<T> v, const Vec256<T> lo,
                                         const Vec256<T> hi) {
  return Min(Max(lo, v), hi);
}

// ------------------------------ Integer multiplication

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> operator*(const Vec256<uint16_t> a,
                                                    const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_mullo_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> operator*(const Vec256<uint32_t> a,
                                                    const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_mullo_epi32(a.raw, b.raw)};
}

// Signed
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> operator*(const Vec256<int16_t> a,
                                                   const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_mullo_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> operator*(const Vec256<int32_t> a,
                                                   const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_mullo_epi32(a.raw, b.raw)};
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> MulHigh(const Vec256<uint16_t> a,
                                                  const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_mulhi_epu16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> MulHigh(const Vec256<int16_t> a,
                                                 const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_mulhi_epi16(a.raw, b.raw)};
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> MulEven(const Vec256<int32_t> a,
                                                 const Vec256<int32_t> b) {
  return Vec256<int64_t>{_mm256_mul_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> MulEven(const Vec256<uint32_t> a,
                                                  const Vec256<uint32_t> b) {
  return Vec256<uint64_t>{_mm256_mul_epu32(a.raw, b.raw)};
}

// ------------------------------ Floating-point negate

HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Neg(const Vec256<float> v) {
  const Full256<float> df;
  const Full256<uint32_t> du;
  const auto sign = BitCast(df, Set(du, 0x80000000u));
  return v ^ sign;
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Neg(const Vec256<double> v) {
  const Full256<double> df;
  const Full256<uint64_t> du;
  const auto sign = BitCast(df, Set(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ Floating-point mul / div

HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator*(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_mul_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator*(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_mul_pd(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<float> operator/(const Vec256<float> a,
                                                 const Vec256<float> b) {
  return Vec256<float>{_mm256_div_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> operator/(const Vec256<double> a,
                                                  const Vec256<double> b) {
  return Vec256<double>{_mm256_div_pd(a.raw, b.raw)};
}

// Approximate reciprocal
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ApproximateReciprocal(
    const Vec256<float> v) {
  return Vec256<float>{_mm256_rcp_ps(v.raw)};
}

namespace ext {
// Absolute value of difference.
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> AbsDiff(const Vec256<float> a,
                                               const Vec256<float> b) {
  const auto mask =
      BitCast(Full256<float>(), Set(Full256<uint32_t>(), 0x7FFFFFFFu));
  return mask & (a - b);
}
}  // namespace ext

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> MulAdd(const Vec256<float> mul,
                                              const Vec256<float> x,
                                              const Vec256<float> add) {
  return Vec256<float>{_mm256_fmadd_ps(mul.raw, x.raw, add.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> MulAdd(const Vec256<double> mul,
                                               const Vec256<double> x,
                                               const Vec256<double> add) {
  return Vec256<double>{_mm256_fmadd_pd(mul.raw, x.raw, add.raw)};
}

// Returns add - mul * x
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> NegMulAdd(const Vec256<float> mul,
                                                 const Vec256<float> x,
                                                 const Vec256<float> add) {
  return Vec256<float>{_mm256_fnmadd_ps(mul.raw, x.raw, add.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> NegMulAdd(const Vec256<double> mul,
                                                  const Vec256<double> x,
                                                  const Vec256<double> add) {
  return Vec256<double>{_mm256_fnmadd_pd(mul.raw, x.raw, add.raw)};
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

// Returns mul * x - sub
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> MulSub(const Vec256<float> mul,
                                              const Vec256<float> x,
                                              const Vec256<float> sub) {
  return Vec256<float>{_mm256_fmsub_ps(mul.raw, x.raw, sub.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> MulSub(const Vec256<double> mul,
                                               const Vec256<double> x,
                                               const Vec256<double> sub) {
  return Vec256<double>{_mm256_fmsub_pd(mul.raw, x.raw, sub.raw)};
}

// Returns -mul * x - sub
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> NegMulSub(const Vec256<float> mul,
                                                 const Vec256<float> x,
                                                 const Vec256<float> sub) {
  return Vec256<float>{_mm256_fnmsub_ps(mul.raw, x.raw, sub.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> NegMulSub(const Vec256<double> mul,
                                                  const Vec256<double> x,
                                                  const Vec256<double> sub) {
  return Vec256<double>{_mm256_fnmsub_pd(mul.raw, x.raw, sub.raw)};
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Full precision square root
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Sqrt(const Vec256<float> v) {
  return Vec256<float>{_mm256_sqrt_ps(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Sqrt(const Vec256<double> v) {
  return Vec256<double>{_mm256_sqrt_pd(v.raw)};
}

// Approximate reciprocal square root
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ApproximateReciprocalSqrt(
    const Vec256<float> v) {
  return Vec256<float>{_mm256_rsqrt_ps(v.raw)};
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, tie to even
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Round(const Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Round(const Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}

// Toward zero, aka truncate
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Trunc(const Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Trunc(const Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
}

// Toward +infinity, aka ceiling
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Ceil(const Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Ceil(const Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)};
}

// Toward -infinity, aka floor
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Floor(const Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Floor(const Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)};
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

// ------------------------------ Equality

// Unsigned
HWY_ATTR_AVX2 HWY_INLINE Mask256<uint8_t> operator==(const Vec256<uint8_t> a,
                                                     const Vec256<uint8_t> b) {
  return Mask256<uint8_t>{_mm256_cmpeq_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<uint16_t> operator==(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Mask256<uint16_t>{_mm256_cmpeq_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<uint32_t> operator==(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Mask256<uint32_t>{_mm256_cmpeq_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<uint64_t> operator==(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Mask256<uint64_t>{_mm256_cmpeq_epi64(a.raw, b.raw)};
}

// Signed
HWY_ATTR_AVX2 HWY_INLINE Mask256<int8_t> operator==(const Vec256<int8_t> a,
                                                    const Vec256<int8_t> b) {
  return Mask256<int8_t>{_mm256_cmpeq_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int16_t> operator==(const Vec256<int16_t> a,
                                                     const Vec256<int16_t> b) {
  return Mask256<int16_t>{_mm256_cmpeq_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int32_t> operator==(const Vec256<int32_t> a,
                                                     const Vec256<int32_t> b) {
  return Mask256<int32_t>{_mm256_cmpeq_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int64_t> operator==(const Vec256<int64_t> a,
                                                     const Vec256<int64_t> b) {
  return Mask256<int64_t>{_mm256_cmpeq_epi64(a.raw, b.raw)};
}

// Float
HWY_ATTR_AVX2 HWY_INLINE Mask256<float> operator==(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_EQ_OQ)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<double> operator==(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_EQ_OQ)};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Mask256<T> TestBit(const Vec256<T> v,
                                            const Vec256<T> bit) {
  static_assert(!IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

// ------------------------------ Strict inequality

// Pre-9.3 GCC immintrin.h uses char, which may be unsigned, causing cmpgt_epi8
// to perform an unsigned comparison instead of the intended signed. Workaround
// is to cast to an explicitly signed type. See https://godbolt.org/z/PL7Ujy
#if HWY_COMPILER_GCC != 0 && HWY_COMPILER_GCC < 930
#define HWY_AVX2_GCC_CMPGT8_WORKAROUND 1
#else
#define HWY_AVX2_GCC_CMPGT8_WORKAROUND 0
#endif

// Signed/float <
HWY_ATTR_AVX2 HWY_INLINE Mask256<int8_t> operator<(const Vec256<int8_t> a,
                                                   const Vec256<int8_t> b) {
#if HWY_AVX2_GCC_CMPGT8_WORKAROUND
  using i8x32 = signed char __attribute__((__vector_size__(32)));
  return Mask256<int8_t>{static_cast<__m256i>(reinterpret_cast<i8x32>(a.raw) <
                                              reinterpret_cast<i8x32>(b.raw))};
#else
  return Mask256<int8_t>{_mm256_cmpgt_epi8(b.raw, a.raw)};
#endif
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int16_t> operator<(const Vec256<int16_t> a,
                                                    const Vec256<int16_t> b) {
  return Mask256<int16_t>{_mm256_cmpgt_epi16(b.raw, a.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int32_t> operator<(const Vec256<int32_t> a,
                                                    const Vec256<int32_t> b) {
  return Mask256<int32_t>{_mm256_cmpgt_epi32(b.raw, a.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int64_t> operator<(const Vec256<int64_t> a,
                                                    const Vec256<int64_t> b) {
  return Mask256<int64_t>{_mm256_cmpgt_epi64(b.raw, a.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<float> operator<(const Vec256<float> a,
                                                  const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_LT_OQ)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<double> operator<(const Vec256<double> a,
                                                   const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_LT_OQ)};
}

// Signed/float >
HWY_ATTR_AVX2 HWY_INLINE Mask256<int8_t> operator>(const Vec256<int8_t> a,
                                                   const Vec256<int8_t> b) {
#if HWY_AVX2_GCC_CMPGT8_WORKAROUND
  using i8x32 = signed char __attribute__((__vector_size__(32)));
  return Mask256<int8_t>{static_cast<__m256i>(reinterpret_cast<i8x32>(a.raw) >
                                              reinterpret_cast<i8x32>(b.raw))};
#else
  return Mask256<int8_t>{_mm256_cmpgt_epi8(a.raw, b.raw)};
#endif
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int16_t> operator>(const Vec256<int16_t> a,
                                                    const Vec256<int16_t> b) {
  return Mask256<int16_t>{_mm256_cmpgt_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int32_t> operator>(const Vec256<int32_t> a,
                                                    const Vec256<int32_t> b) {
  return Mask256<int32_t>{_mm256_cmpgt_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<int64_t> operator>(const Vec256<int64_t> a,
                                                    const Vec256<int64_t> b) {
  return Mask256<int64_t>{_mm256_cmpgt_epi64(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<float> operator>(const Vec256<float> a,
                                                  const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_GT_OQ)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<double> operator>(const Vec256<double> a,
                                                   const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_GT_OQ)};
}

// ------------------------------ Weak inequality

// Float <= >=
HWY_ATTR_AVX2 HWY_INLINE Mask256<float> operator<=(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_LE_OQ)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<double> operator<=(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_LE_OQ)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<float> operator>=(const Vec256<float> a,
                                                   const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_GE_OQ)};
}
HWY_ATTR_AVX2 HWY_INLINE Mask256<double> operator>=(const Vec256<double> a,
                                                    const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_GE_OQ)};
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> Load(Full256<T> /* tag */,
                                        const T* HWY_RESTRICT aligned) {
  return Vec256<T>{
      _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned))};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Load(Full256<float> /* tag */,
                                            const float* HWY_RESTRICT aligned) {
  return Vec256<float>{_mm256_load_ps(aligned)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Load(
    Full256<double> /* tag */, const double* HWY_RESTRICT aligned) {
  return Vec256<double>{_mm256_load_pd(aligned)};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> LoadU(Full256<T> /* tag */,
                                         const T* HWY_RESTRICT p) {
  return Vec256<T>{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> LoadU(Full256<float> /* tag */,
                                             const float* HWY_RESTRICT p) {
  return Vec256<float>{_mm256_loadu_ps(p)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> LoadU(Full256<double> /* tag */,
                                              const double* HWY_RESTRICT p) {
  return Vec256<double>{_mm256_loadu_pd(p)};
}

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> LoadDup128(Full256<T> /* tag */,
                                              const T* HWY_RESTRICT p) {
#if HWY_LOADDUP_ASM
  __m256i out;
  asm("vbroadcasti128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec256<T>{out};
#else
  return Vec256<T>{_mm256_broadcastsi128_si256(LoadU(Full128<T>(), p).raw)};
#endif
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> LoadDup128(
    Full256<float> /* tag */, const float* const HWY_RESTRICT p) {
#if HWY_LOADDUP_ASM
  __m256 out;
  asm("vbroadcastf128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec256<float>{out};
#else
  return Vec256<float>{_mm256_broadcast_ps(reinterpret_cast<const __m128*>(p))};
#endif
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> LoadDup128(
    Full256<double> /* tag */, const double* const HWY_RESTRICT p) {
#if HWY_LOADDUP_ASM
  __m256d out;
  asm("vbroadcastf128 %1, %[reg]" : [ reg ] "=x"(out) : "m"(p[0]));
  return Vec256<double>{out};
#else
  return Vec256<double>{
      _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p))};
#endif
}

// ------------------------------ Store

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE void Store(Vec256<T> v, Full256<T> /* tag */,
                                    T* HWY_RESTRICT aligned) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
HWY_ATTR_AVX2 HWY_INLINE void Store(const Vec256<float> v,
                                    Full256<float> /* tag */,
                                    float* HWY_RESTRICT aligned) {
  _mm256_store_ps(aligned, v.raw);
}
HWY_ATTR_AVX2 HWY_INLINE void Store(const Vec256<double> v,
                                    Full256<double> /* tag */,
                                    double* HWY_RESTRICT aligned) {
  _mm256_store_pd(aligned, v.raw);
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE void StoreU(Vec256<T> v, Full256<T> /* tag */,
                                     T* HWY_RESTRICT p) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v.raw);
}
HWY_ATTR_AVX2 HWY_INLINE void StoreU(const Vec256<float> v,
                                     Full256<float> /* tag */,
                                     float* HWY_RESTRICT p) {
  _mm256_storeu_ps(p, v.raw);
}
HWY_ATTR_AVX2 HWY_INLINE void StoreU(const Vec256<double> v,
                                     Full256<double> /* tag */,
                                     double* HWY_RESTRICT p) {
  _mm256_storeu_pd(p, v.raw);
}

// ------------------------------ Non-temporal stores

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE void Stream(Vec256<T> v, Full256<T> /* tag */,
                                     T* HWY_RESTRICT aligned) {
  _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
HWY_ATTR_AVX2 HWY_INLINE void Stream(const Vec256<float> v,
                                     Full256<float> /* tag */,
                                     float* HWY_RESTRICT aligned) {
  _mm256_stream_ps(aligned, v.raw);
}
HWY_ATTR_AVX2 HWY_INLINE void Stream(const Vec256<double> v,
                                     Full256<double> /* tag */,
                                     double* HWY_RESTRICT aligned) {
  _mm256_stream_pd(aligned, v.raw);
}

// ------------------------------ Gather

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> gather_offset_impl(
    SizeTag<4> /* tag */, Full256<T> /* tag */, const T* HWY_RESTRICT base,
    const Vec256<int32_t> offset) {
  return Vec256<T>{_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), offset.raw, 1)};
}
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> gather_index_impl(
    SizeTag<4> /* tag */, Full256<T> /* tag */, const T* HWY_RESTRICT base,
    const Vec256<int32_t> index) {
  return Vec256<T>{_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), index.raw, 4)};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> gather_offset_impl(
    SizeTag<8> /* tag */, Full256<T> /* tag */, const T* HWY_RESTRICT base,
    const Vec256<int64_t> offset) {
  return Vec256<T>{_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), offset.raw, 1)};
}
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> gather_index_impl(
    SizeTag<8> /* tag */, Full256<T> /* tag */, const T* HWY_RESTRICT base,
    const Vec256<int64_t> index) {
  return Vec256<T>{_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), index.raw, 8)};
}

template <typename T, typename Offset>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> GatherOffset(Full256<T> d,
                                                const T* HWY_RESTRICT base,
                                                const Vec256<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  return gather_offset_impl(SizeTag<sizeof(T)>(), d, base, offset);
}
template <typename T, typename Index>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> GatherIndex(Full256<T> d,
                                               const T* HWY_RESTRICT base,
                                               const Vec256<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  return gather_index_impl(SizeTag<sizeof(T)>(), d, base, index);
}

template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> GatherOffset<float>(
    Full256<float> /* tag */, const float* HWY_RESTRICT base,
    const Vec256<int32_t> offset) {
  return Vec256<float>{_mm256_i32gather_ps(base, offset.raw, 1)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> GatherIndex<float>(
    Full256<float> /* tag */, const float* HWY_RESTRICT base,
    const Vec256<int32_t> index) {
  return Vec256<float>{_mm256_i32gather_ps(base, index.raw, 4)};
}

template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> GatherOffset<double>(
    Full256<double> /* tag */, const double* HWY_RESTRICT base,
    const Vec256<int64_t> offset) {
  return Vec256<double>{_mm256_i64gather_pd(base, offset.raw, 1)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> GatherIndex<double>(
    Full256<double> /* tag */, const double* HWY_RESTRICT base,
    const Vec256<int64_t> index) {
  return Vec256<double>{_mm256_i64gather_pd(base, index.raw, 8)};
}

}  // namespace ext

// ================================================== SWIZZLE

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE T GetLane(const Vec256<T> v) {
  return GetLane(LowerHalf(v));
}

// ------------------------------ Extract half

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec128<T> LowerHalf(Vec256<T> v) {
  return Vec128<T>{_mm256_castsi256_si128(v.raw)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec128<float> LowerHalf(Vec256<float> v) {
  return Vec128<float>{_mm256_castps256_ps128(v.raw)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec128<double> LowerHalf(Vec256<double> v) {
  return Vec128<double>{_mm256_castpd256_pd128(v.raw)};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec128<T> UpperHalf(Vec256<T> v) {
  return Vec128<T>{_mm256_extracti128_si256(v.raw, 1)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec128<float> UpperHalf(Vec256<float> v) {
  return Vec128<float>{_mm256_extractf128_ps(v.raw, 1)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec128<double> UpperHalf(Vec256<double> v) {
  return Vec128<double>{_mm256_extractf128_pd(v.raw, 1)};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec128<T> GetHalf(Lower /* tag */, Vec256<T> v) {
  return LowerHalf(v);
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec128<T> GetHalf(Upper /* tag */, Vec256<T> v) {
  return UpperHalf(v);
}

// ------------------------------ Shift vector by constant #bytes

// 0x01..0F, kBytes = 1 => 0x02..0F00
template <int kBytes, typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ShiftLeftBytes(const Vec256<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bslli_epi128.
  return Vec256<T>{_mm256_slli_si256(v.raw, kBytes)};
}

template <int kLanes, typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ShiftLeftLanes(const Vec256<T> v) {
  return ShiftLeftBytes<kLanes * sizeof(T)>(v);
}

// 0x01..0F, kBytes = 1 => 0x0001..0E
template <int kBytes, typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ShiftRightBytes(const Vec256<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bsrli_epi128.
  return Vec256<T>{_mm256_srli_si256(v.raw, kBytes)};
}

template <int kLanes, typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ShiftRightLanes(const Vec256<T> v) {
  return ShiftRightBytes<kLanes * sizeof(T)>(v);
}

// ------------------------------ Extract from 2x 128-bit at constant offset

// Extracts 128 bits from <hi, lo> by skipping the least-significant kBytes.
template <int kBytes, typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> CombineShiftRightBytes(const Vec256<T> hi,
                                                          const Vec256<T> lo) {
  const Full256<uint8_t> d8;
  const Vec256<uint8_t> extracted_bytes{
      _mm256_alignr_epi8(BitCast(d8, hi).raw, BitCast(d8, lo).raw, kBytes)};
  return BitCast(Full256<T>(), extracted_bytes);
}

// ------------------------------ Broadcast/splat any lane

// Unsigned
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> Broadcast(const Vec256<uint16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m256i lo = _mm256_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec256<uint16_t>{_mm256_unpacklo_epi64(lo, lo)};
  } else {
    const __m256i hi =
        _mm256_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec256<uint16_t>{_mm256_unpackhi_epi64(hi, hi)};
  }
}
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Broadcast(const Vec256<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x55 * kLane)};
}
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> Broadcast(const Vec256<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<uint64_t>{_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44)};
}

// Signed
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> Broadcast(const Vec256<int16_t> v) {
  static_assert(0 <= kLane && kLane < 8, "Invalid lane");
  if (kLane < 4) {
    const __m256i lo = _mm256_shufflelo_epi16(v.raw, (0x55 * kLane) & 0xFF);
    return Vec256<int16_t>{_mm256_unpacklo_epi64(lo, lo)};
  } else {
    const __m256i hi =
        _mm256_shufflehi_epi16(v.raw, (0x55 * (kLane - 4)) & 0xFF);
    return Vec256<int16_t>{_mm256_unpackhi_epi64(hi, hi)};
  }
}
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Broadcast(const Vec256<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x55 * kLane)};
}
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> Broadcast(const Vec256<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<int64_t>{_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44)};
}

// Float
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Broadcast(Vec256<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x55 * kLane)};
}
template <int kLane>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Broadcast(const Vec256<double> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<double>{_mm256_shuffle_pd(v.raw, v.raw, 15 * kLane)};
}

// ------------------------------ Hard-coded shuffles

// Notation: let Vec256<int32_t> have lanes 7,6,5,4,3,2,1,0 (0 is
// least-significant). Shuffle0321 rotates four-lane blocks one lane to the
// right (the previous least-significant lane is now most-significant =>
// 47650321). These could also be implemented via CombineShiftRightBytes but
// the shuffle_abcd notation is more convenient.

// Swap 32-bit halves in 64-bit halves.
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Shuffle2301(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0xB1)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Shuffle2301(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0xB1)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Shuffle2301(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0xB1)};
}

// Swap 64-bit halves
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Shuffle1032(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Shuffle1032(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Shuffle1032(const Vec256<float> v) {
  // Shorter encoding than _mm256_permute_ps.
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x4E)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> Shuffle01(const Vec256<uint64_t> v) {
  return Vec256<uint64_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> Shuffle01(const Vec256<int64_t> v) {
  return Vec256<int64_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> Shuffle01(const Vec256<double> v) {
  // Shorter encoding than _mm256_permute_pd.
  return Vec256<double>{_mm256_shuffle_pd(v.raw, v.raw, 5)};
}

// Rotate right 32 bits
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Shuffle0321(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x39)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Shuffle0321(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x39)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Shuffle0321(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x39)};
}
// Rotate left 32 bits
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Shuffle2103(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x93)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Shuffle2103(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x93)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> Shuffle2103(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x93)};
}

// Reverse
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> Shuffle0123(
    const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x1B)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> Shuffle0123(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x1B)};
}
HWY_ATTR_AVX2
HWY_INLINE Vec256<float> Shuffle0123(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x1B)};
}

// ------------------------------ Permute (runtime variable)

// Returned by SetTableIndices for use by TableLookupLanes.
template <typename T>
struct Permute256 {
  __m256i raw;
};

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Permute256<T> SetTableIndices(const Full256<T> d,
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
  return Permute256<T>{LoadU(Full256<int32_t>(), idx).raw};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> TableLookupLanes(
    const Vec256<uint32_t> v, const Permute256<uint32_t> idx) {
  return Vec256<uint32_t>{_mm256_permutevar8x32_epi32(v.raw, idx.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> TableLookupLanes(
    const Vec256<int32_t> v, const Permute256<int32_t> idx) {
  return Vec256<int32_t>{_mm256_permutevar8x32_epi32(v.raw, idx.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> TableLookupLanes(
    const Vec256<float> v, const Permute256<float> idx) {
  return Vec256<float>{_mm256_permutevar8x32_ps(v.raw, idx.raw)};
}

// ------------------------------ Interleave lanes

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use ZipLo/hi instead (also works with scalar).

HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> InterleaveLo(const Vec256<uint8_t> a,
                                                      const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_unpacklo_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> InterleaveLo(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_unpacklo_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> InterleaveLo(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_unpacklo_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> InterleaveLo(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_unpacklo_epi64(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> InterleaveLo(const Vec256<int8_t> a,
                                                     const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_unpacklo_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> InterleaveLo(const Vec256<int16_t> a,
                                                      const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_unpacklo_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> InterleaveLo(const Vec256<int32_t> a,
                                                      const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_unpacklo_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> InterleaveLo(const Vec256<int64_t> a,
                                                      const Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_unpacklo_epi64(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<float> InterleaveLo(const Vec256<float> a,
                                                    const Vec256<float> b) {
  return Vec256<float>{_mm256_unpacklo_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> InterleaveLo(const Vec256<double> a,
                                                     const Vec256<double> b) {
  return Vec256<double>{_mm256_unpacklo_pd(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<uint8_t> InterleaveHi(const Vec256<uint8_t> a,
                                                      const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_unpackhi_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> InterleaveHi(
    const Vec256<uint16_t> a, const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_unpackhi_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> InterleaveHi(
    const Vec256<uint32_t> a, const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_unpackhi_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> InterleaveHi(
    const Vec256<uint64_t> a, const Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_unpackhi_epi64(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<int8_t> InterleaveHi(const Vec256<int8_t> a,
                                                     const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_unpackhi_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> InterleaveHi(const Vec256<int16_t> a,
                                                      const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_unpackhi_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> InterleaveHi(const Vec256<int32_t> a,
                                                      const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_unpackhi_epi32(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> InterleaveHi(const Vec256<int64_t> a,
                                                      const Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_unpackhi_epi64(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<float> InterleaveHi(const Vec256<float> a,
                                                    const Vec256<float> b) {
  return Vec256<float>{_mm256_unpackhi_ps(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> InterleaveHi(const Vec256<double> a,
                                                     const Vec256<double> b) {
  return Vec256<double>{_mm256_unpackhi_pd(a.raw, b.raw)};
}

// ------------------------------ Zip lanes

// Same as interleave_*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.

HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> ZipLo(const Vec256<uint8_t> a,
                                                const Vec256<uint8_t> b) {
  return Vec256<uint16_t>{_mm256_unpacklo_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> ZipLo(const Vec256<uint16_t> a,
                                                const Vec256<uint16_t> b) {
  return Vec256<uint32_t>{_mm256_unpacklo_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> ZipLo(const Vec256<uint32_t> a,
                                                const Vec256<uint32_t> b) {
  return Vec256<uint64_t>{_mm256_unpacklo_epi32(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> ZipLo(const Vec256<int8_t> a,
                                               const Vec256<int8_t> b) {
  return Vec256<int16_t>{_mm256_unpacklo_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ZipLo(const Vec256<int16_t> a,
                                               const Vec256<int16_t> b) {
  return Vec256<int32_t>{_mm256_unpacklo_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> ZipLo(const Vec256<int32_t> a,
                                               const Vec256<int32_t> b) {
  return Vec256<int64_t>{_mm256_unpacklo_epi32(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> ZipHi(const Vec256<uint8_t> a,
                                                const Vec256<uint8_t> b) {
  return Vec256<uint16_t>{_mm256_unpackhi_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> ZipHi(const Vec256<uint16_t> a,
                                                const Vec256<uint16_t> b) {
  return Vec256<uint32_t>{_mm256_unpackhi_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> ZipHi(const Vec256<uint32_t> a,
                                                const Vec256<uint32_t> b) {
  return Vec256<uint64_t>{_mm256_unpackhi_epi32(a.raw, b.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> ZipHi(const Vec256<int8_t> a,
                                               const Vec256<int8_t> b) {
  return Vec256<int16_t>{_mm256_unpackhi_epi8(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ZipHi(const Vec256<int16_t> a,
                                               const Vec256<int16_t> b) {
  return Vec256<int32_t>{_mm256_unpackhi_epi16(a.raw, b.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> ZipHi(const Vec256<int32_t> a,
                                               const Vec256<int32_t> b) {
  return Vec256<int64_t>{_mm256_unpackhi_epi32(a.raw, b.raw)};
}

// ------------------------------ Blocks

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ConcatLoLo(const Vec256<T> hi,
                                              const Vec256<T> lo) {
  return Vec256<T>{_mm256_permute2x128_si256(lo.raw, hi.raw, 0x20)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ConcatLoLo(const Vec256<float> hi,
                                                  const Vec256<float> lo) {
  return Vec256<float>{_mm256_permute2f128_ps(lo.raw, hi.raw, 0x20)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> ConcatLoLo(const Vec256<double> hi,
                                                   const Vec256<double> lo) {
  return Vec256<double>{_mm256_permute2f128_pd(lo.raw, hi.raw, 0x20)};
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ConcatHiHi(const Vec256<T> hi,
                                              const Vec256<T> lo) {
  return Vec256<T>{_mm256_permute2x128_si256(lo.raw, hi.raw, 0x31)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ConcatHiHi(const Vec256<float> hi,
                                                  const Vec256<float> lo) {
  return Vec256<float>{_mm256_permute2f128_ps(lo.raw, hi.raw, 0x31)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> ConcatHiHi(const Vec256<double> hi,
                                                   const Vec256<double> lo) {
  return Vec256<double>{_mm256_permute2f128_pd(lo.raw, hi.raw, 0x31)};
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves / swap blocks)
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ConcatLoHi(const Vec256<T> hi,
                                              const Vec256<T> lo) {
  return Vec256<T>{_mm256_permute2x128_si256(lo.raw, hi.raw, 0x21)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ConcatLoHi(const Vec256<float> hi,
                                                  const Vec256<float> lo) {
  return Vec256<float>{_mm256_permute2f128_ps(lo.raw, hi.raw, 0x21)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> ConcatLoHi(const Vec256<double> hi,
                                                   const Vec256<double> lo) {
  return Vec256<double>{_mm256_permute2f128_pd(lo.raw, hi.raw, 0x21)};
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> ConcatHiLo(const Vec256<T> hi,
                                              const Vec256<T> lo) {
  return Vec256<T>{_mm256_blend_epi32(hi.raw, lo.raw, 0x0F)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ConcatHiLo(const Vec256<float> hi,
                                                  const Vec256<float> lo) {
  return Vec256<float>{_mm256_blend_ps(hi.raw, lo.raw, 0x0F)};
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> ConcatHiLo(const Vec256<double> hi,
                                                   const Vec256<double> lo) {
  return Vec256<double>{_mm256_blend_pd(hi.raw, lo.raw, 3)};
}

// ------------------------------ Odd/even lanes

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> odd_even_impl(SizeTag<1> /* tag */,
                                                 const Vec256<T> a,
                                                 const Vec256<T> b) {
  const Full256<T> d;
  const Full256<uint8_t> d8;
  alignas(32) constexpr uint8_t mask[16] = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                            0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return IfThenElse(MaskFromVec(BitCast(d, LoadDup128(d8, mask))), b, a);
}
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> odd_even_impl(SizeTag<2> /* tag */,
                                                 const Vec256<T> a,
                                                 const Vec256<T> b) {
  return Vec256<T>{_mm256_blend_epi16(a.raw, b.raw, 0x55)};
}
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> odd_even_impl(SizeTag<4> /* tag */,
                                                 const Vec256<T> a,
                                                 const Vec256<T> b) {
  return Vec256<T>{_mm256_blend_epi32(a.raw, b.raw, 0x55)};
}
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> odd_even_impl(SizeTag<8> /* tag */,
                                                 const Vec256<T> a,
                                                 const Vec256<T> b) {
  return Vec256<T>{_mm256_blend_epi32(a.raw, b.raw, 0x33)};
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> OddEven(const Vec256<T> a,
                                           const Vec256<T> b) {
  return odd_even_impl(SizeTag<sizeof(T)>(), a, b);
}
template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<float> OddEven<float>(const Vec256<float> a,
                                                      const Vec256<float> b) {
  return Vec256<float>{_mm256_blend_ps(a.raw, b.raw, 0x55)};
}

template <>
HWY_ATTR_AVX2 HWY_INLINE Vec256<double> OddEven<double>(
    const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>{_mm256_blend_pd(a.raw, b.raw, 5)};
}

// ================================================== CONVERT

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes:
// either valid indices in [0, 16) or >= 0x80 to zero the i-th output byte.
template <typename T, typename TI>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> TableLookupBytes(const Vec256<T> bytes,
                                                    const Vec256<TI> from) {
  return Vec256<T>{_mm256_shuffle_epi8(bytes.raw, from.raw)};
}

// ------------------------------ Promotions (part w/ narrow lanes -> full)

HWY_ATTR_AVX2 HWY_INLINE Vec256<double> ConvertTo(Full256<double> /* tag */,
                                                  const Vec128<float, 4> v) {
  return Vec256<double>{_mm256_cvtps_pd(v.raw)};
}

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then ZipHi/lo would be faster.
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint16_t> ConvertTo(Full256<uint16_t> /* tag */,
                                                    Vec128<uint8_t> v) {
  return Vec256<uint16_t>{_mm256_cvtepu8_epi16(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> ConvertTo(Full256<uint32_t> /* tag */,
                                                    Vec128<uint8_t, 8> v) {
  return Vec256<uint32_t>{_mm256_cvtepu8_epi32(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> ConvertTo(Full256<int16_t> /* tag */,
                                                   Vec128<uint8_t> v) {
  return Vec256<int16_t>{_mm256_cvtepu8_epi16(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ConvertTo(Full256<int32_t> /* tag */,
                                                   Vec128<uint8_t, 8> v) {
  return Vec256<int32_t>{_mm256_cvtepu8_epi32(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> ConvertTo(Full256<uint32_t> /* tag */,
                                                    Vec128<uint16_t> v) {
  return Vec256<uint32_t>{_mm256_cvtepu16_epi32(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ConvertTo(Full256<int32_t> /* tag */,
                                                   Vec128<uint16_t> v) {
  return Vec256<int32_t>{_mm256_cvtepu16_epi32(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> ConvertTo(Full256<uint64_t> /* tag */,
                                                    Vec128<uint32_t> v) {
  return Vec256<uint64_t>{_mm256_cvtepu32_epi64(v.raw)};
}

// Special case for "v" with all blocks equal (e.g. from LoadDup128):
// single-cycle latency instead of 3.
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint32_t> U32FromU8(const Vec256<uint8_t> v) {
  const Full256<uint32_t> d32;
  alignas(32) static constexpr uint32_t k32From8[8] = {
      0xFFFFFF00UL, 0xFFFFFF01UL, 0xFFFFFF02UL, 0xFFFFFF03UL,
      0xFFFFFF04UL, 0xFFFFFF05UL, 0xFFFFFF06UL, 0xFFFFFF07UL};
  return TableLookupBytes(BitCast(d32, v), Load(d32, k32From8));
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then ZipHi/lo followed by
// signed shift would be faster.
HWY_ATTR_AVX2 HWY_INLINE Vec256<int16_t> ConvertTo(Full256<int16_t> /* tag */,
                                                   Vec128<int8_t> v) {
  return Vec256<int16_t>{_mm256_cvtepi8_epi16(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ConvertTo(Full256<int32_t> /* tag */,
                                                   Vec128<int8_t, 8> v) {
  return Vec256<int32_t>{_mm256_cvtepi8_epi32(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ConvertTo(Full256<int32_t> /* tag */,
                                                   Vec128<int16_t> v) {
  return Vec256<int32_t>{_mm256_cvtepi16_epi32(v.raw)};
}
HWY_ATTR_AVX2 HWY_INLINE Vec256<int64_t> ConvertTo(Full256<int64_t> /* tag */,
                                                   Vec128<int32_t> v) {
  return Vec256<int64_t>{_mm256_cvtepi32_epi64(v.raw)};
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

HWY_ATTR_AVX2 HWY_INLINE Vec128<uint16_t> ConvertTo(Full128<uint16_t> /* tag */,
                                                    const Vec256<int32_t> v) {
  const __m256i u16 = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenating lower halves of both 128-bit blocks afterward is more
  // efficient than an extra input with low block = high block of v.
  return Vec128<uint16_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u16, 0x88))};
}

HWY_ATTR_AVX2 HWY_INLINE Vec128<uint8_t, 8> ConvertTo(
    Desc<uint8_t, 8> /* tag */, const Vec256<int32_t> v) {
  const __m256i u16_blocks = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i u16_concat = _mm256_permute4x64_epi64(u16_blocks, 0x88);
  const __m128i u16 = _mm256_castsi256_si128(u16_concat);
  return Vec128<uint8_t, 8>{_mm_packus_epi16(u16, u16)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec128<int16_t> ConvertTo(Full128<int16_t> /* tag */,
                                                   const Vec256<int32_t> v) {
  const __m256i i16 = _mm256_packs_epi32(v.raw, v.raw);
  return Vec128<int16_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i16, 0x88))};
}

HWY_ATTR_AVX2 HWY_INLINE Vec128<int8_t, 8> ConvertTo(Desc<int8_t, 8> /* tag */,
                                                     const Vec256<int32_t> v) {
  const __m256i i16_blocks = _mm256_packs_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i i16_concat = _mm256_permute4x64_epi64(i16_blocks, 0x88);
  const __m128i i16 = _mm256_castsi256_si128(i16_concat);
  return Vec128<int8_t, 8>{_mm_packs_epi16(i16, i16)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec128<uint8_t> ConvertTo(Full128<uint8_t> /* tag */,
                                                   const Vec256<int16_t> v) {
  const __m256i u8 = _mm256_packus_epi16(v.raw, v.raw);
  return Vec128<uint8_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u8, 0x88))};
}

HWY_ATTR_AVX2 HWY_INLINE Vec128<int8_t> ConvertTo(Full128<int8_t> /* tag */,
                                                  const Vec256<int16_t> v) {
  const __m256i i8 = _mm256_packs_epi16(v.raw, v.raw);
  return Vec128<int8_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i8, 0x88))};
}

// For already range-limited input [0, 255].
HWY_ATTR_AVX2 HWY_INLINE Vec128<uint8_t, 8> U8FromU32(
    const Vec256<uint32_t> v) {
  const Full256<uint32_t> d32;
  alignas(32) static constexpr uint32_t k8From32[8] = {
      0x0C080400u, ~0u, ~0u, ~0u, ~0u, 0x0C080400u, ~0u, ~0u};
  // Place first four bytes in lo[0], remaining 4 in hi[1].
  const auto quad = TableLookupBytes(v, Load(d32, k8From32));
  // Interleave both quadruplets - OR instead of unpack reduces port5 pressure.
  const auto lo = LowerHalf(quad);
  const auto hi = UpperHalf(quad);
  const auto pair = LowerHalf(lo | hi);
  return BitCast(Desc<uint8_t, 8>(), pair);
}

// ------------------------------ Convert i32 <=> f32

HWY_ATTR_AVX2 HWY_INLINE Vec256<float> ConvertTo(Full256<float> /* tag */,
                                                 const Vec256<int32_t> v) {
  return Vec256<float>{_mm256_cvtepi32_ps(v.raw)};
}
// Truncates (rounds toward zero).
HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> ConvertTo(Full256<int32_t> /* tag */,
                                                   const Vec256<float> v) {
  return Vec256<int32_t>{_mm256_cvttps_epi32(v.raw)};
}

HWY_ATTR_AVX2 HWY_INLINE Vec256<int32_t> NearestInt(const Vec256<float> v) {
  return Vec256<int32_t>{_mm256_cvtps_epi32(v.raw)};
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// ------------------------------ Mask

namespace impl {

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE uint64_t BitsFromMask(SizeTag<1> /*tag*/,
                                               const Mask256<T> mask) {
  const Full256<uint8_t> d;
  const auto sign_bits = BitCast(d, VecFromMask(mask)).raw;
  // Prevent sign-extension of 32-bit masks because the intrinsic returns int.
  return static_cast<uint32_t>(_mm256_movemask_epi8(sign_bits));
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE uint64_t BitsFromMask(SizeTag<2> /*tag*/,
                                               const Mask256<T> mask) {
  const uint64_t sign_bits8 = BitsFromMask(SizeTag<1>(), mask);
  // Skip the bits from the lower byte of each u16 (better not to use the
  // same packs_epi16 as sse4.h, because that requires an extra swizzle here).
  return _pext_u64(sign_bits8, 0xAAAAAAAAull);
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE uint64_t BitsFromMask(SizeTag<4> /*tag*/,
                                               const Mask256<T> mask) {
  const Full256<float> d;
  const auto sign_bits = BitCast(d, VecFromMask(mask)).raw;
  return static_cast<unsigned>(_mm256_movemask_ps(sign_bits));
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE uint64_t BitsFromMask(SizeTag<8> /*tag*/,
                                               const Mask256<T> mask) {
  const Full256<double> d;
  const auto sign_bits = BitCast(d, VecFromMask(mask)).raw;
  return static_cast<unsigned>(_mm256_movemask_pd(sign_bits));
}

}  // namespace impl

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE uint64_t BitsFromMask(const Mask256<T> mask) {
  return impl::BitsFromMask(SizeTag<sizeof(T)>(), mask);
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE bool AllFalse(const Mask256<T> mask) {
  // Cheaper than PTEST, which is 2 uop / 3L.
  return BitsFromMask(mask) == 0;
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE bool AllTrue(const Mask256<T> mask) {
  constexpr uint64_t kAllBits = (1ull << Full256<T>::N) - 1;
  return BitsFromMask(mask) == kAllBits;
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE size_t CountTrue(const Mask256<T> mask) {
  return PopCount(BitsFromMask(mask));
}

// ------------------------------ Horizontal sum (reduction)

// Returns 64-bit sums of 8-byte groups.
HWY_ATTR_AVX2 HWY_INLINE Vec256<uint64_t> SumsOfU8x8(const Vec256<uint8_t> v) {
  return Vec256<uint64_t>{_mm256_sad_epu8(v.raw, _mm256_setzero_si256())};
}

// Returns sum{lane[i]} in each lane. "v3210" is a replicated 128-bit block.
// Same logic as x86_sse4.h, but with Vec256 arguments.
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> horz_sum_impl(SizeTag<4> /* tag */,
                                                 const Vec256<T> v3210) {
  const auto v1032 = Shuffle1032(v3210);
  const auto v31_20_31_20 = v3210 + v1032;
  const auto v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}

template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> horz_sum_impl(SizeTag<8> /* tag */,
                                                 const Vec256<T> v10) {
  const auto v01 = Shuffle01(v10);
  return v10 + v01;
}

// Supported for {uif}32x8, {uif}64x4. Returns the sum in each lane.
template <typename T>
HWY_ATTR_AVX2 HWY_INLINE Vec256<T> SumOfLanes(const Vec256<T> vHL) {
  const Vec256<T> vLH = ConcatLoHi(vHL, vHL);
  return horz_sum_impl(SizeTag<sizeof(T)>(), vLH + vHL);
}

}  // namespace ext

// TODO(janwas): wrappers for all intrinsics (in x86 namespace).
}  // namespace hwy

#endif  // HWY_X86_AVX2_H_
