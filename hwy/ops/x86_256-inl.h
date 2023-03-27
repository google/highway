// Copyright 2019 Google LLC
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

// 256-bit vectors and AVX2 instructions, plus some AVX512-VL operations when
// compiling for that target.
// External include guard in highway.h - see comment there.

// WARNING: most operations do not cross 128-bit block boundaries. In
// particular, "Broadcast", pack and zip behavior may be surprising.

// Must come before HWY_DIAGNOSTICS and HWY_COMPILER_CLANGCL
#include "hwy/base.h"

// Avoid uninitialized warnings in GCC's avx512fintrin.h - see
// https://github.com/google/highway/issues/710)
HWY_DIAGNOSTICS(push)
#if HWY_COMPILER_GCC_ACTUAL
HWY_DIAGNOSTICS_OFF(disable : 4701, ignored "-Wuninitialized")
HWY_DIAGNOSTICS_OFF(disable : 4703 6001 26494, ignored "-Wmaybe-uninitialized")
#endif

// Must come before HWY_COMPILER_CLANGCL
#include <immintrin.h>  // AVX2+

#if HWY_COMPILER_CLANGCL
// Including <immintrin.h> should be enough, but Clang's headers helpfully skip
// including these headers when _MSC_VER is defined, like when using clang-cl.
// Include these directly here.
#include <avxintrin.h>
// avxintrin defines __m256i and must come before avx2intrin.
#include <avx2intrin.h>
#include <bmi2intrin.h>  // _pext_u64
#include <f16cintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif  // HWY_COMPILER_CLANGCL

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#if HWY_IS_MSAN
#include <sanitizer/msan_interface.h>
#endif

// For half-width vectors. Already includes base.h and shared-inl.h.
#include "hwy/ops/x86_128-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace detail {

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

}  // namespace detail

template <typename T>
class Vec256 {
  using Raw = typename detail::Raw256<T>::type;

 public:
  using PrivateT = T;                                  // only for DFromV
  static constexpr size_t kPrivateN = 32 / sizeof(T);  // only for DFromV

  // Compound assignment. Only usable if there is a corresponding non-member
  // binary operator overload. For example, only f32 and f64 support division.
  HWY_INLINE Vec256& operator*=(const Vec256 other) {
    return *this = (*this * other);
  }
  HWY_INLINE Vec256& operator/=(const Vec256 other) {
    return *this = (*this / other);
  }
  HWY_INLINE Vec256& operator+=(const Vec256 other) {
    return *this = (*this + other);
  }
  HWY_INLINE Vec256& operator-=(const Vec256 other) {
    return *this = (*this - other);
  }
  HWY_INLINE Vec256& operator&=(const Vec256 other) {
    return *this = (*this & other);
  }
  HWY_INLINE Vec256& operator|=(const Vec256 other) {
    return *this = (*this | other);
  }
  HWY_INLINE Vec256& operator^=(const Vec256 other) {
    return *this = (*this ^ other);
  }

  Raw raw;
};

#if HWY_TARGET <= HWY_AVX3

namespace detail {

// Template arg: sizeof(lane type)
template <size_t size>
struct RawMask256 {};
template <>
struct RawMask256<1> {
  using type = __mmask32;
};
template <>
struct RawMask256<2> {
  using type = __mmask16;
};
template <>
struct RawMask256<4> {
  using type = __mmask8;
};
template <>
struct RawMask256<8> {
  using type = __mmask8;
};

}  // namespace detail

template <typename T>
struct Mask256 {
  using Raw = typename detail::RawMask256<sizeof(T)>::type;

  static Mask256<T> FromBits(uint64_t mask_bits) {
    return Mask256<T>{static_cast<Raw>(mask_bits)};
  }

  Raw raw;
};

#else  // AVX2

// FF..FF or 0.
template <typename T>
struct Mask256 {
  typename detail::Raw256<T>::type raw;
};

#endif  // AVX2

#if HWY_TARGET <= HWY_AVX3
namespace detail {

// Used by Expand() emulation, which is required for both AVX3 and AVX2.
template <typename T>
HWY_INLINE uint64_t BitsFromMask(const Mask256<T> mask) {
  return mask.raw;
}

}  // namespace detail
#endif  // HWY_TARGET <= HWY_AVX3

template <typename T>
using Full256 = Simd<T, 32 / sizeof(T), 0>;

// ------------------------------ BitCast

namespace detail {

HWY_INLINE __m256i BitCastToInteger(__m256i v) { return v; }
HWY_INLINE __m256i BitCastToInteger(__m256 v) { return _mm256_castps_si256(v); }
HWY_INLINE __m256i BitCastToInteger(__m256d v) {
  return _mm256_castpd_si256(v);
}

template <typename T>
HWY_INLINE Vec256<uint8_t> BitCastToByte(Vec256<T> v) {
  return Vec256<uint8_t>{BitCastToInteger(v.raw)};
}

// Cannot rely on function overloading because return types differ.
template <typename T>
struct BitCastFromInteger256 {
  HWY_INLINE __m256i operator()(__m256i v) { return v; }
};
template <>
struct BitCastFromInteger256<float> {
  HWY_INLINE __m256 operator()(__m256i v) { return _mm256_castsi256_ps(v); }
};
template <>
struct BitCastFromInteger256<double> {
  HWY_INLINE __m256d operator()(__m256i v) { return _mm256_castsi256_pd(v); }
};

template <class D, typename T = TFromD<D>>
HWY_INLINE Vec256<T> BitCastFromByte(D /* tag */, Vec256<uint8_t> v) {
  return Vec256<T>{BitCastFromInteger256<T>()(v.raw)};
}

}  // namespace detail

template <class D, typename FromT>
HWY_API Vec256<TFromD<D>> BitCast(D d, Vec256<FromT> v) {
  return detail::BitCastFromByte(d, detail::BitCastToByte(v));
}

// ------------------------------ Zero

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_NOT_FLOAT_D(D)>
HWY_API Vec256<TFromD<D>> Zero(D /* tag */) {
  return Vec256<TFromD<D>>{_mm256_setzero_si256()};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_F32_D(D)>
HWY_API Vec256<float> Zero(D /* tag */) {
  return Vec256<float>{_mm256_setzero_ps()};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_F64_D(D)>
HWY_API Vec256<double> Zero(D /* tag */) {
  return Vec256<double>{_mm256_setzero_pd()};
}

// ------------------------------ Set

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_T_SIZE_D(D, 1)>
HWY_API VFromD<D> Set(D /* tag */, TFromD<D> t) {
  return VFromD<D>{_mm256_set1_epi8(static_cast<char>(t))};  // NOLINT
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_T_SIZE_D(D, 2)>
HWY_API VFromD<D> Set(D /* tag */, TFromD<D> t) {
  return VFromD<D>{_mm256_set1_epi16(static_cast<short>(t))};  // NOLINT
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_UI32_D(D)>
HWY_API VFromD<D> Set(D /* tag */, TFromD<D> t) {
  return VFromD<D>{_mm256_set1_epi32(static_cast<int>(t))};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_UI64_D(D)>
HWY_API VFromD<D> Set(D /* tag */, TFromD<D> t) {
  return VFromD<D>{_mm256_set1_epi64x(static_cast<long long>(t))};  // NOLINT
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_F32_D(D)>
HWY_API Vec256<float> Set(D /* tag */, float t) {
  return Vec256<float>{_mm256_set1_ps(t)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_F64_D(D)>
HWY_API Vec256<double> Set(D /* tag */, double t) {
  return Vec256<double>{_mm256_set1_pd(t)};
}

HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")

// Returns a vector with uninitialized elements.
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_NOT_FLOAT_D(D)>
HWY_API Vec256<TFromD<D>> Undefined(D /* tag */) {
  // Available on Clang 6.0, GCC 6.2, ICC 16.03, MSVC 19.14. All but ICC
  // generate an XOR instruction.
  return Vec256<TFromD<D>>{_mm256_undefined_si256()};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_F32_D(D)>
HWY_API Vec256<float> Undefined(D /* tag */) {
  return Vec256<float>{_mm256_undefined_ps()};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_F64_D(D)>
HWY_API Vec256<double> Undefined(D /* tag */) {
  return Vec256<double>{_mm256_undefined_pd()};
}

HWY_DIAGNOSTICS(pop)

// ================================================== LOGICAL

// ------------------------------ And

template <typename T>
HWY_API Vec256<T> And(Vec256<T> a, Vec256<T> b) {
  return Vec256<T>{_mm256_and_si256(a.raw, b.raw)};
}

HWY_API Vec256<float> And(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_and_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> And(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_and_pd(a.raw, b.raw)};
}

// ------------------------------ AndNot

// Returns ~not_mask & mask.
template <typename T>
HWY_API Vec256<T> AndNot(Vec256<T> not_mask, Vec256<T> mask) {
  return Vec256<T>{_mm256_andnot_si256(not_mask.raw, mask.raw)};
}
HWY_API Vec256<float> AndNot(Vec256<float> not_mask, Vec256<float> mask) {
  return Vec256<float>{_mm256_andnot_ps(not_mask.raw, mask.raw)};
}
HWY_API Vec256<double> AndNot(Vec256<double> not_mask, Vec256<double> mask) {
  return Vec256<double>{_mm256_andnot_pd(not_mask.raw, mask.raw)};
}

// ------------------------------ Or

template <typename T>
HWY_API Vec256<T> Or(Vec256<T> a, Vec256<T> b) {
  return Vec256<T>{_mm256_or_si256(a.raw, b.raw)};
}

HWY_API Vec256<float> Or(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_or_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> Or(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_or_pd(a.raw, b.raw)};
}

// ------------------------------ Xor

template <typename T>
HWY_API Vec256<T> Xor(Vec256<T> a, Vec256<T> b) {
  return Vec256<T>{_mm256_xor_si256(a.raw, b.raw)};
}

HWY_API Vec256<float> Xor(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_xor_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> Xor(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_xor_pd(a.raw, b.raw)};
}

// ------------------------------ Not
template <typename T>
HWY_API Vec256<T> Not(const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  using TU = MakeUnsigned<T>;
#if HWY_TARGET <= HWY_AVX3
  const __m256i vu = BitCast(RebindToUnsigned<decltype(d)>(), v).raw;
  return BitCast(d, Vec256<TU>{_mm256_ternarylogic_epi32(vu, vu, vu, 0x55)});
#else
  return Xor(v, BitCast(d, Vec256<TU>{_mm256_set1_epi32(-1)}));
#endif
}

// ------------------------------ Xor3
template <typename T>
HWY_API Vec256<T> Xor3(Vec256<T> x1, Vec256<T> x2, Vec256<T> x3) {
#if HWY_TARGET <= HWY_AVX3
  const DFromV<decltype(x1)> d;
  const RebindToUnsigned<decltype(d)> du;
  using VU = VFromD<decltype(du)>;
  const __m256i ret = _mm256_ternarylogic_epi64(
      BitCast(du, x1).raw, BitCast(du, x2).raw, BitCast(du, x3).raw, 0x96);
  return BitCast(d, VU{ret});
#else
  return Xor(x1, Xor(x2, x3));
#endif
}

// ------------------------------ Or3
template <typename T>
HWY_API Vec256<T> Or3(Vec256<T> o1, Vec256<T> o2, Vec256<T> o3) {
#if HWY_TARGET <= HWY_AVX3
  const DFromV<decltype(o1)> d;
  const RebindToUnsigned<decltype(d)> du;
  using VU = VFromD<decltype(du)>;
  const __m256i ret = _mm256_ternarylogic_epi64(
      BitCast(du, o1).raw, BitCast(du, o2).raw, BitCast(du, o3).raw, 0xFE);
  return BitCast(d, VU{ret});
#else
  return Or(o1, Or(o2, o3));
#endif
}

// ------------------------------ OrAnd
template <typename T>
HWY_API Vec256<T> OrAnd(Vec256<T> o, Vec256<T> a1, Vec256<T> a2) {
#if HWY_TARGET <= HWY_AVX3
  const DFromV<decltype(o)> d;
  const RebindToUnsigned<decltype(d)> du;
  using VU = VFromD<decltype(du)>;
  const __m256i ret = _mm256_ternarylogic_epi64(
      BitCast(du, o).raw, BitCast(du, a1).raw, BitCast(du, a2).raw, 0xF8);
  return BitCast(d, VU{ret});
#else
  return Or(o, And(a1, a2));
#endif
}

// ------------------------------ IfVecThenElse
template <typename T>
HWY_API Vec256<T> IfVecThenElse(Vec256<T> mask, Vec256<T> yes, Vec256<T> no) {
#if HWY_TARGET <= HWY_AVX3
  const DFromV<decltype(yes)> d;
  const RebindToUnsigned<decltype(d)> du;
  using VU = VFromD<decltype(du)>;
  return BitCast(d, VU{_mm256_ternarylogic_epi64(BitCast(du, mask).raw,
                                                 BitCast(du, yes).raw,
                                                 BitCast(du, no).raw, 0xCA)});
#else
  return IfThenElse(MaskFromVec(mask), yes, no);
#endif
}

// ------------------------------ Operator overloads (internal-only if float)

template <typename T>
HWY_API Vec256<T> operator&(const Vec256<T> a, const Vec256<T> b) {
  return And(a, b);
}

template <typename T>
HWY_API Vec256<T> operator|(const Vec256<T> a, const Vec256<T> b) {
  return Or(a, b);
}

template <typename T>
HWY_API Vec256<T> operator^(const Vec256<T> a, const Vec256<T> b) {
  return Xor(a, b);
}

// ------------------------------ PopulationCount

// 8/16 require BITALG, 32/64 require VPOPCNTDQ.
#if HWY_TARGET <= HWY_AVX3_DL

#ifdef HWY_NATIVE_POPCNT
#undef HWY_NATIVE_POPCNT
#else
#define HWY_NATIVE_POPCNT
#endif

namespace detail {

template <typename T>
HWY_INLINE Vec256<T> PopulationCount(hwy::SizeTag<1> /* tag */, Vec256<T> v) {
  return Vec256<T>{_mm256_popcnt_epi8(v.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> PopulationCount(hwy::SizeTag<2> /* tag */, Vec256<T> v) {
  return Vec256<T>{_mm256_popcnt_epi16(v.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> PopulationCount(hwy::SizeTag<4> /* tag */, Vec256<T> v) {
  return Vec256<T>{_mm256_popcnt_epi32(v.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> PopulationCount(hwy::SizeTag<8> /* tag */, Vec256<T> v) {
  return Vec256<T>{_mm256_popcnt_epi64(v.raw)};
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> PopulationCount(Vec256<T> v) {
  return detail::PopulationCount(hwy::SizeTag<sizeof(T)>(), v);
}

#endif  // HWY_TARGET <= HWY_AVX3_DL

// ================================================== SIGN

// ------------------------------ CopySign

template <typename T>
HWY_API Vec256<T> CopySign(const Vec256<T> magn, const Vec256<T> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");

  const DFromV<decltype(magn)> d;
  const auto msb = SignBit(d);

#if HWY_TARGET <= HWY_AVX3
  const Rebind<MakeUnsigned<T>, decltype(d)> du;
  // Truth table for msb, magn, sign | bitwise msb ? sign : mag
  //                  0    0     0   |  0
  //                  0    0     1   |  0
  //                  0    1     0   |  1
  //                  0    1     1   |  1
  //                  1    0     0   |  0
  //                  1    0     1   |  1
  //                  1    1     0   |  0
  //                  1    1     1   |  1
  // The lane size does not matter because we are not using predication.
  const __m256i out = _mm256_ternarylogic_epi32(
      BitCast(du, msb).raw, BitCast(du, magn).raw, BitCast(du, sign).raw, 0xAC);
  return BitCast(d, decltype(Zero(du)){out});
#else
  return Or(AndNot(msb, magn), And(msb, sign));
#endif
}

template <typename T>
HWY_API Vec256<T> CopySignToAbs(const Vec256<T> abs, const Vec256<T> sign) {
#if HWY_TARGET <= HWY_AVX3
  // AVX3 can also handle abs < 0, so no extra action needed.
  return CopySign(abs, sign);
#else
  const DFromV<decltype(abs)> d;
  return Or(abs, And(SignBit(d), sign));
#endif
}

// ================================================== MASK

#if HWY_TARGET <= HWY_AVX3

// ------------------------------ IfThenElse

// Returns mask ? b : a.

namespace detail {

// Templates for signed/unsigned integer of a particular size.
template <typename T>
HWY_INLINE Vec256<T> IfThenElse(hwy::SizeTag<1> /* tag */, Mask256<T> mask,
                                Vec256<T> yes, Vec256<T> no) {
  return Vec256<T>{_mm256_mask_mov_epi8(no.raw, mask.raw, yes.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenElse(hwy::SizeTag<2> /* tag */, Mask256<T> mask,
                                Vec256<T> yes, Vec256<T> no) {
  return Vec256<T>{_mm256_mask_mov_epi16(no.raw, mask.raw, yes.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenElse(hwy::SizeTag<4> /* tag */, Mask256<T> mask,
                                Vec256<T> yes, Vec256<T> no) {
  return Vec256<T>{_mm256_mask_mov_epi32(no.raw, mask.raw, yes.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenElse(hwy::SizeTag<8> /* tag */, Mask256<T> mask,
                                Vec256<T> yes, Vec256<T> no) {
  return Vec256<T>{_mm256_mask_mov_epi64(no.raw, mask.raw, yes.raw)};
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> IfThenElse(Mask256<T> mask, Vec256<T> yes, Vec256<T> no) {
  return detail::IfThenElse(hwy::SizeTag<sizeof(T)>(), mask, yes, no);
}
HWY_API Vec256<float> IfThenElse(Mask256<float> mask, Vec256<float> yes,
                                 Vec256<float> no) {
  return Vec256<float>{_mm256_mask_mov_ps(no.raw, mask.raw, yes.raw)};
}
HWY_API Vec256<double> IfThenElse(Mask256<double> mask, Vec256<double> yes,
                                  Vec256<double> no) {
  return Vec256<double>{_mm256_mask_mov_pd(no.raw, mask.raw, yes.raw)};
}

namespace detail {

template <typename T>
HWY_INLINE Vec256<T> IfThenElseZero(hwy::SizeTag<1> /* tag */, Mask256<T> mask,
                                    Vec256<T> yes) {
  return Vec256<T>{_mm256_maskz_mov_epi8(mask.raw, yes.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenElseZero(hwy::SizeTag<2> /* tag */, Mask256<T> mask,
                                    Vec256<T> yes) {
  return Vec256<T>{_mm256_maskz_mov_epi16(mask.raw, yes.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenElseZero(hwy::SizeTag<4> /* tag */, Mask256<T> mask,
                                    Vec256<T> yes) {
  return Vec256<T>{_mm256_maskz_mov_epi32(mask.raw, yes.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenElseZero(hwy::SizeTag<8> /* tag */, Mask256<T> mask,
                                    Vec256<T> yes) {
  return Vec256<T>{_mm256_maskz_mov_epi64(mask.raw, yes.raw)};
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> IfThenElseZero(Mask256<T> mask, Vec256<T> yes) {
  return detail::IfThenElseZero(hwy::SizeTag<sizeof(T)>(), mask, yes);
}
HWY_API Vec256<float> IfThenElseZero(Mask256<float> mask, Vec256<float> yes) {
  return Vec256<float>{_mm256_maskz_mov_ps(mask.raw, yes.raw)};
}
HWY_API Vec256<double> IfThenElseZero(Mask256<double> mask,
                                      Vec256<double> yes) {
  return Vec256<double>{_mm256_maskz_mov_pd(mask.raw, yes.raw)};
}

namespace detail {

template <typename T>
HWY_INLINE Vec256<T> IfThenZeroElse(hwy::SizeTag<1> /* tag */, Mask256<T> mask,
                                    Vec256<T> no) {
  // xor_epi8/16 are missing, but we have sub, which is just as fast for u8/16.
  return Vec256<T>{_mm256_mask_sub_epi8(no.raw, mask.raw, no.raw, no.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenZeroElse(hwy::SizeTag<2> /* tag */, Mask256<T> mask,
                                    Vec256<T> no) {
  return Vec256<T>{_mm256_mask_sub_epi16(no.raw, mask.raw, no.raw, no.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenZeroElse(hwy::SizeTag<4> /* tag */, Mask256<T> mask,
                                    Vec256<T> no) {
  return Vec256<T>{_mm256_mask_xor_epi32(no.raw, mask.raw, no.raw, no.raw)};
}
template <typename T>
HWY_INLINE Vec256<T> IfThenZeroElse(hwy::SizeTag<8> /* tag */, Mask256<T> mask,
                                    Vec256<T> no) {
  return Vec256<T>{_mm256_mask_xor_epi64(no.raw, mask.raw, no.raw, no.raw)};
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> IfThenZeroElse(Mask256<T> mask, Vec256<T> no) {
  return detail::IfThenZeroElse(hwy::SizeTag<sizeof(T)>(), mask, no);
}
HWY_API Vec256<float> IfThenZeroElse(Mask256<float> mask, Vec256<float> no) {
  return Vec256<float>{_mm256_mask_xor_ps(no.raw, mask.raw, no.raw, no.raw)};
}
HWY_API Vec256<double> IfThenZeroElse(Mask256<double> mask, Vec256<double> no) {
  return Vec256<double>{_mm256_mask_xor_pd(no.raw, mask.raw, no.raw, no.raw)};
}

template <typename T>
HWY_API Vec256<T> ZeroIfNegative(const Vec256<T> v) {
  static_assert(IsSigned<T>(), "Only for float");
  // AVX3 MaskFromVec only looks at the MSB
  return IfThenZeroElse(MaskFromVec(v), v);
}

// ------------------------------ Mask logical

namespace detail {

template <typename T>
HWY_INLINE Mask256<T> And(hwy::SizeTag<1> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kand_mask32(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask32>(a.raw & b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> And(hwy::SizeTag<2> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kand_mask16(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask16>(a.raw & b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> And(hwy::SizeTag<4> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kand_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(a.raw & b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> And(hwy::SizeTag<8> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kand_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(a.raw & b.raw)};
#endif
}

template <typename T>
HWY_INLINE Mask256<T> AndNot(hwy::SizeTag<1> /*tag*/, const Mask256<T> a,
                             const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kandn_mask32(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask32>(~a.raw & b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> AndNot(hwy::SizeTag<2> /*tag*/, const Mask256<T> a,
                             const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kandn_mask16(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask16>(~a.raw & b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> AndNot(hwy::SizeTag<4> /*tag*/, const Mask256<T> a,
                             const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kandn_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(~a.raw & b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> AndNot(hwy::SizeTag<8> /*tag*/, const Mask256<T> a,
                             const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kandn_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(~a.raw & b.raw)};
#endif
}

template <typename T>
HWY_INLINE Mask256<T> Or(hwy::SizeTag<1> /*tag*/, const Mask256<T> a,
                         const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kor_mask32(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask32>(a.raw | b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> Or(hwy::SizeTag<2> /*tag*/, const Mask256<T> a,
                         const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kor_mask16(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask16>(a.raw | b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> Or(hwy::SizeTag<4> /*tag*/, const Mask256<T> a,
                         const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kor_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(a.raw | b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> Or(hwy::SizeTag<8> /*tag*/, const Mask256<T> a,
                         const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kor_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(a.raw | b.raw)};
#endif
}

template <typename T>
HWY_INLINE Mask256<T> Xor(hwy::SizeTag<1> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxor_mask32(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask32>(a.raw ^ b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> Xor(hwy::SizeTag<2> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxor_mask16(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask16>(a.raw ^ b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> Xor(hwy::SizeTag<4> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxor_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(a.raw ^ b.raw)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> Xor(hwy::SizeTag<8> /*tag*/, const Mask256<T> a,
                          const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxor_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(a.raw ^ b.raw)};
#endif
}

template <typename T>
HWY_INLINE Mask256<T> ExclusiveNeither(hwy::SizeTag<1> /*tag*/,
                                       const Mask256<T> a, const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxnor_mask32(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask32>(~(a.raw ^ b.raw) & 0xFFFFFFFF)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> ExclusiveNeither(hwy::SizeTag<2> /*tag*/,
                                       const Mask256<T> a, const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxnor_mask16(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask16>(~(a.raw ^ b.raw) & 0xFFFF)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> ExclusiveNeither(hwy::SizeTag<4> /*tag*/,
                                       const Mask256<T> a, const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{_kxnor_mask8(a.raw, b.raw)};
#else
  return Mask256<T>{static_cast<__mmask8>(~(a.raw ^ b.raw) & 0xFF)};
#endif
}
template <typename T>
HWY_INLINE Mask256<T> ExclusiveNeither(hwy::SizeTag<8> /*tag*/,
                                       const Mask256<T> a, const Mask256<T> b) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return Mask256<T>{static_cast<__mmask8>(_kxnor_mask8(a.raw, b.raw) & 0xF)};
#else
  return Mask256<T>{static_cast<__mmask8>(~(a.raw ^ b.raw) & 0xF)};
#endif
}

}  // namespace detail

template <typename T>
HWY_API Mask256<T> And(const Mask256<T> a, Mask256<T> b) {
  return detail::And(hwy::SizeTag<sizeof(T)>(), a, b);
}

template <typename T>
HWY_API Mask256<T> AndNot(const Mask256<T> a, Mask256<T> b) {
  return detail::AndNot(hwy::SizeTag<sizeof(T)>(), a, b);
}

template <typename T>
HWY_API Mask256<T> Or(const Mask256<T> a, Mask256<T> b) {
  return detail::Or(hwy::SizeTag<sizeof(T)>(), a, b);
}

template <typename T>
HWY_API Mask256<T> Xor(const Mask256<T> a, Mask256<T> b) {
  return detail::Xor(hwy::SizeTag<sizeof(T)>(), a, b);
}

template <typename T>
HWY_API Mask256<T> Not(const Mask256<T> m) {
  // Flip only the valid bits.
  constexpr size_t N = 32 / sizeof(T);
  return Xor(m, Mask256<T>::FromBits((1ull << N) - 1));
}

template <typename T>
HWY_API Mask256<T> ExclusiveNeither(const Mask256<T> a, Mask256<T> b) {
  return detail::ExclusiveNeither(hwy::SizeTag<sizeof(T)>(), a, b);
}

#else  // AVX2

// ------------------------------ Mask

// Mask and Vec are the same (true = FF..FF).
template <typename T>
HWY_API Mask256<T> MaskFromVec(const Vec256<T> v) {
  return Mask256<T>{v.raw};
}

template <typename T>
HWY_API Vec256<T> VecFromMask(const Mask256<T> v) {
  return Vec256<T>{v.raw};
}

template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> VecFromMask(D /* tag */, const Mask256<T> v) {
  return Vec256<T>{v.raw};
}

// ------------------------------ IfThenElse

// mask ? yes : no
template <typename T>
HWY_API Vec256<T> IfThenElse(const Mask256<T> mask, const Vec256<T> yes,
                             const Vec256<T> no) {
  return Vec256<T>{_mm256_blendv_epi8(no.raw, yes.raw, mask.raw)};
}
HWY_API Vec256<float> IfThenElse(const Mask256<float> mask,
                                 const Vec256<float> yes,
                                 const Vec256<float> no) {
  return Vec256<float>{_mm256_blendv_ps(no.raw, yes.raw, mask.raw)};
}
HWY_API Vec256<double> IfThenElse(const Mask256<double> mask,
                                  const Vec256<double> yes,
                                  const Vec256<double> no) {
  return Vec256<double>{_mm256_blendv_pd(no.raw, yes.raw, mask.raw)};
}

// mask ? yes : 0
template <typename T>
HWY_API Vec256<T> IfThenElseZero(Mask256<T> mask, Vec256<T> yes) {
  const DFromV<decltype(yes)> d;
  return yes & VecFromMask(d, mask);
}

// mask ? 0 : no
template <typename T>
HWY_API Vec256<T> IfThenZeroElse(Mask256<T> mask, Vec256<T> no) {
  const DFromV<decltype(no)> d;
  return AndNot(VecFromMask(d, mask), no);
}

template <typename T>
HWY_API Vec256<T> ZeroIfNegative(Vec256<T> v) {
  static_assert(IsSigned<T>(), "Only for float");
  const DFromV<decltype(v)> d;
  const auto zero = Zero(d);
  // AVX2 IfThenElse only looks at the MSB for 32/64-bit lanes
  return IfThenElse(MaskFromVec(v), zero, v);
}

// ------------------------------ Mask logical

template <typename T>
HWY_API Mask256<T> Not(const Mask256<T> m) {
  const Full256<T> d;
  return MaskFromVec(Not(VecFromMask(d, m)));
}

template <typename T>
HWY_API Mask256<T> And(const Mask256<T> a, Mask256<T> b) {
  const Full256<T> d;
  return MaskFromVec(And(VecFromMask(d, a), VecFromMask(d, b)));
}

template <typename T>
HWY_API Mask256<T> AndNot(const Mask256<T> a, Mask256<T> b) {
  const Full256<T> d;
  return MaskFromVec(AndNot(VecFromMask(d, a), VecFromMask(d, b)));
}

template <typename T>
HWY_API Mask256<T> Or(const Mask256<T> a, Mask256<T> b) {
  const Full256<T> d;
  return MaskFromVec(Or(VecFromMask(d, a), VecFromMask(d, b)));
}

template <typename T>
HWY_API Mask256<T> Xor(const Mask256<T> a, Mask256<T> b) {
  const Full256<T> d;
  return MaskFromVec(Xor(VecFromMask(d, a), VecFromMask(d, b)));
}

template <typename T>
HWY_API Mask256<T> ExclusiveNeither(const Mask256<T> a, Mask256<T> b) {
  const Full256<T> d;
  return MaskFromVec(AndNot(VecFromMask(d, a), Not(VecFromMask(d, b))));
}

#endif  // HWY_TARGET <= HWY_AVX3

// ================================================== COMPARE

#if HWY_TARGET <= HWY_AVX3

// Comparisons set a mask bit to 1 if the condition is true, else 0.

template <typename TFrom, class DTo, typename TTo = TFromD<DTo>>
HWY_API Mask256<TTo> RebindMask(DTo /*tag*/, Mask256<TFrom> m) {
  static_assert(sizeof(TFrom) == sizeof(TTo), "Must have same size");
  return Mask256<TTo>{m.raw};
}

namespace detail {

template <typename T>
HWY_INLINE Mask256<T> TestBit(hwy::SizeTag<1> /*tag*/, const Vec256<T> v,
                              const Vec256<T> bit) {
  return Mask256<T>{_mm256_test_epi8_mask(v.raw, bit.raw)};
}
template <typename T>
HWY_INLINE Mask256<T> TestBit(hwy::SizeTag<2> /*tag*/, const Vec256<T> v,
                              const Vec256<T> bit) {
  return Mask256<T>{_mm256_test_epi16_mask(v.raw, bit.raw)};
}
template <typename T>
HWY_INLINE Mask256<T> TestBit(hwy::SizeTag<4> /*tag*/, const Vec256<T> v,
                              const Vec256<T> bit) {
  return Mask256<T>{_mm256_test_epi32_mask(v.raw, bit.raw)};
}
template <typename T>
HWY_INLINE Mask256<T> TestBit(hwy::SizeTag<8> /*tag*/, const Vec256<T> v,
                              const Vec256<T> bit) {
  return Mask256<T>{_mm256_test_epi64_mask(v.raw, bit.raw)};
}

}  // namespace detail

template <typename T>
HWY_API Mask256<T> TestBit(const Vec256<T> v, const Vec256<T> bit) {
  static_assert(!hwy::IsFloat<T>(), "Only integer vectors supported");
  return detail::TestBit(hwy::SizeTag<sizeof(T)>(), v, bit);
}

// ------------------------------ Equality

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi8_mask(a.raw, b.raw)};
}
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi16_mask(a.raw, b.raw)};
}
template <typename T, HWY_IF_UI32(T)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi32_mask(a.raw, b.raw)};
}
template <typename T, HWY_IF_UI64(T)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi64_mask(a.raw, b.raw)};
}

HWY_API Mask256<float> operator==(Vec256<float> a, Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps_mask(a.raw, b.raw, _CMP_EQ_OQ)};
}

HWY_API Mask256<double> operator==(Vec256<double> a, Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd_mask(a.raw, b.raw, _CMP_EQ_OQ)};
}

// ------------------------------ Inequality

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Mask256<T> operator!=(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpneq_epi8_mask(a.raw, b.raw)};
}
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Mask256<T> operator!=(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpneq_epi16_mask(a.raw, b.raw)};
}
template <typename T, HWY_IF_UI32(T)>
HWY_API Mask256<T> operator!=(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpneq_epi32_mask(a.raw, b.raw)};
}
template <typename T, HWY_IF_UI64(T)>
HWY_API Mask256<T> operator!=(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpneq_epi64_mask(a.raw, b.raw)};
}

HWY_API Mask256<float> operator!=(Vec256<float> a, Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps_mask(a.raw, b.raw, _CMP_NEQ_OQ)};
}

HWY_API Mask256<double> operator!=(Vec256<double> a, Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd_mask(a.raw, b.raw, _CMP_NEQ_OQ)};
}

// ------------------------------ Strict inequality

HWY_API Mask256<int8_t> operator>(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Mask256<int8_t>{_mm256_cmpgt_epi8_mask(a.raw, b.raw)};
}
HWY_API Mask256<int16_t> operator>(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Mask256<int16_t>{_mm256_cmpgt_epi16_mask(a.raw, b.raw)};
}
HWY_API Mask256<int32_t> operator>(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Mask256<int32_t>{_mm256_cmpgt_epi32_mask(a.raw, b.raw)};
}
HWY_API Mask256<int64_t> operator>(Vec256<int64_t> a, Vec256<int64_t> b) {
  return Mask256<int64_t>{_mm256_cmpgt_epi64_mask(a.raw, b.raw)};
}

HWY_API Mask256<uint8_t> operator>(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Mask256<uint8_t>{_mm256_cmpgt_epu8_mask(a.raw, b.raw)};
}
HWY_API Mask256<uint16_t> operator>(const Vec256<uint16_t> a,
                                    const Vec256<uint16_t> b) {
  return Mask256<uint16_t>{_mm256_cmpgt_epu16_mask(a.raw, b.raw)};
}
HWY_API Mask256<uint32_t> operator>(const Vec256<uint32_t> a,
                                    const Vec256<uint32_t> b) {
  return Mask256<uint32_t>{_mm256_cmpgt_epu32_mask(a.raw, b.raw)};
}
HWY_API Mask256<uint64_t> operator>(const Vec256<uint64_t> a,
                                    const Vec256<uint64_t> b) {
  return Mask256<uint64_t>{_mm256_cmpgt_epu64_mask(a.raw, b.raw)};
}

HWY_API Mask256<float> operator>(Vec256<float> a, Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps_mask(a.raw, b.raw, _CMP_GT_OQ)};
}
HWY_API Mask256<double> operator>(Vec256<double> a, Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd_mask(a.raw, b.raw, _CMP_GT_OQ)};
}

// ------------------------------ Weak inequality

HWY_API Mask256<float> operator>=(Vec256<float> a, Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps_mask(a.raw, b.raw, _CMP_GE_OQ)};
}
HWY_API Mask256<double> operator>=(Vec256<double> a, Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd_mask(a.raw, b.raw, _CMP_GE_OQ)};
}

HWY_API Mask256<int8_t> operator>=(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Mask256<int8_t>{_mm256_cmpge_epi8_mask(a.raw, b.raw)};
}
HWY_API Mask256<int16_t> operator>=(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Mask256<int16_t>{_mm256_cmpge_epi16_mask(a.raw, b.raw)};
}
HWY_API Mask256<int32_t> operator>=(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Mask256<int32_t>{_mm256_cmpge_epi32_mask(a.raw, b.raw)};
}
HWY_API Mask256<int64_t> operator>=(Vec256<int64_t> a, Vec256<int64_t> b) {
  return Mask256<int64_t>{_mm256_cmpge_epi64_mask(a.raw, b.raw)};
}

HWY_API Mask256<uint8_t> operator>=(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Mask256<uint8_t>{_mm256_cmpge_epu8_mask(a.raw, b.raw)};
}
HWY_API Mask256<uint16_t> operator>=(const Vec256<uint16_t> a,
                                     const Vec256<uint16_t> b) {
  return Mask256<uint16_t>{_mm256_cmpge_epu16_mask(a.raw, b.raw)};
}
HWY_API Mask256<uint32_t> operator>=(const Vec256<uint32_t> a,
                                     const Vec256<uint32_t> b) {
  return Mask256<uint32_t>{_mm256_cmpge_epu32_mask(a.raw, b.raw)};
}
HWY_API Mask256<uint64_t> operator>=(const Vec256<uint64_t> a,
                                     const Vec256<uint64_t> b) {
  return Mask256<uint64_t>{_mm256_cmpge_epu64_mask(a.raw, b.raw)};
}

// ------------------------------ Mask

namespace detail {

template <typename T>
HWY_INLINE Mask256<T> MaskFromVec(hwy::SizeTag<1> /*tag*/, const Vec256<T> v) {
  return Mask256<T>{_mm256_movepi8_mask(v.raw)};
}
template <typename T>
HWY_INLINE Mask256<T> MaskFromVec(hwy::SizeTag<2> /*tag*/, const Vec256<T> v) {
  return Mask256<T>{_mm256_movepi16_mask(v.raw)};
}
template <typename T>
HWY_INLINE Mask256<T> MaskFromVec(hwy::SizeTag<4> /*tag*/, const Vec256<T> v) {
  return Mask256<T>{_mm256_movepi32_mask(v.raw)};
}
template <typename T>
HWY_INLINE Mask256<T> MaskFromVec(hwy::SizeTag<8> /*tag*/, const Vec256<T> v) {
  return Mask256<T>{_mm256_movepi64_mask(v.raw)};
}

}  // namespace detail

template <typename T>
HWY_API Mask256<T> MaskFromVec(const Vec256<T> v) {
  return detail::MaskFromVec(hwy::SizeTag<sizeof(T)>(), v);
}
// There do not seem to be native floating-point versions of these instructions.
HWY_API Mask256<float> MaskFromVec(const Vec256<float> v) {
  const Full256<int32_t> di;
  return Mask256<float>{MaskFromVec(BitCast(di, v)).raw};
}
HWY_API Mask256<double> MaskFromVec(const Vec256<double> v) {
  const Full256<int64_t> di;
  return Mask256<double>{MaskFromVec(BitCast(di, v)).raw};
}

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> VecFromMask(const Mask256<T> v) {
  return Vec256<T>{_mm256_movm_epi8(v.raw)};
}

template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> VecFromMask(const Mask256<T> v) {
  return Vec256<T>{_mm256_movm_epi16(v.raw)};
}

template <typename T, HWY_IF_UI32(T)>
HWY_API Vec256<T> VecFromMask(const Mask256<T> v) {
  return Vec256<T>{_mm256_movm_epi32(v.raw)};
}

template <typename T, HWY_IF_UI64(T)>
HWY_API Vec256<T> VecFromMask(const Mask256<T> v) {
  return Vec256<T>{_mm256_movm_epi64(v.raw)};
}

HWY_API Vec256<float> VecFromMask(const Mask256<float> v) {
  return Vec256<float>{_mm256_castsi256_ps(_mm256_movm_epi32(v.raw))};
}

HWY_API Vec256<double> VecFromMask(const Mask256<double> v) {
  return Vec256<double>{_mm256_castsi256_pd(_mm256_movm_epi64(v.raw))};
}

template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> VecFromMask(D /* tag */, const Mask256<T> v) {
  return VecFromMask(v);
}

#else  // AVX2

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

template <typename TFrom, class DTo, typename TTo = TFromD<DTo>>
HWY_API Mask256<TTo> RebindMask(DTo d_to, Mask256<TFrom> m) {
  static_assert(sizeof(TFrom) == sizeof(TTo), "Must have same size");
  const Full256<TFrom> dfrom;
  return MaskFromVec(BitCast(d_to, VecFromMask(dfrom, m)));
}

template <typename T>
HWY_API Mask256<T> TestBit(const Vec256<T> v, const Vec256<T> bit) {
  static_assert(!hwy::IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

// ------------------------------ Equality

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi8(a.raw, b.raw)};
}

template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi16(a.raw, b.raw)};
}

template <typename T, HWY_IF_UI32(T)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi32(a.raw, b.raw)};
}

template <typename T, HWY_IF_UI64(T)>
HWY_API Mask256<T> operator==(const Vec256<T> a, const Vec256<T> b) {
  return Mask256<T>{_mm256_cmpeq_epi64(a.raw, b.raw)};
}

HWY_API Mask256<float> operator==(const Vec256<float> a,
                                  const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_EQ_OQ)};
}

HWY_API Mask256<double> operator==(const Vec256<double> a,
                                   const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_EQ_OQ)};
}

// ------------------------------ Inequality

template <typename T>
HWY_API Mask256<T> operator!=(const Vec256<T> a, const Vec256<T> b) {
  return Not(a == b);
}
HWY_API Mask256<float> operator!=(const Vec256<float> a,
                                  const Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_NEQ_OQ)};
}
HWY_API Mask256<double> operator!=(const Vec256<double> a,
                                   const Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_NEQ_OQ)};
}

// ------------------------------ Strict inequality

// Tag dispatch instead of SFINAE for MSVC 2017 compatibility
namespace detail {

// Pre-9.3 GCC immintrin.h uses char, which may be unsigned, causing cmpgt_epi8
// to perform an unsigned comparison instead of the intended signed. Workaround
// is to cast to an explicitly signed type. See https://godbolt.org/z/PL7Ujy
#if HWY_COMPILER_GCC_ACTUAL != 0 && HWY_COMPILER_GCC_ACTUAL < 903
#define HWY_AVX2_GCC_CMPGT8_WORKAROUND 1
#else
#define HWY_AVX2_GCC_CMPGT8_WORKAROUND 0
#endif

HWY_API Mask256<int8_t> Gt(hwy::SignedTag /*tag*/, Vec256<int8_t> a,
                           Vec256<int8_t> b) {
#if HWY_AVX2_GCC_CMPGT8_WORKAROUND
  using i8x32 = signed char __attribute__((__vector_size__(32)));
  return Mask256<int8_t>{static_cast<__m256i>(reinterpret_cast<i8x32>(a.raw) >
                                              reinterpret_cast<i8x32>(b.raw))};
#else
  return Mask256<int8_t>{_mm256_cmpgt_epi8(a.raw, b.raw)};
#endif
}
HWY_API Mask256<int16_t> Gt(hwy::SignedTag /*tag*/, Vec256<int16_t> a,
                            Vec256<int16_t> b) {
  return Mask256<int16_t>{_mm256_cmpgt_epi16(a.raw, b.raw)};
}
HWY_API Mask256<int32_t> Gt(hwy::SignedTag /*tag*/, Vec256<int32_t> a,
                            Vec256<int32_t> b) {
  return Mask256<int32_t>{_mm256_cmpgt_epi32(a.raw, b.raw)};
}
HWY_API Mask256<int64_t> Gt(hwy::SignedTag /*tag*/, Vec256<int64_t> a,
                            Vec256<int64_t> b) {
  return Mask256<int64_t>{_mm256_cmpgt_epi64(a.raw, b.raw)};
}

template <typename T>
HWY_INLINE Mask256<T> Gt(hwy::UnsignedTag /*tag*/, Vec256<T> a, Vec256<T> b) {
  const Full256<T> du;
  const RebindToSigned<decltype(du)> di;
  const Vec256<T> msb = Set(du, (LimitsMax<T>() >> 1) + 1);
  return RebindMask(du, BitCast(di, Xor(a, msb)) > BitCast(di, Xor(b, msb)));
}

HWY_API Mask256<float> Gt(hwy::FloatTag /*tag*/, Vec256<float> a,
                          Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_GT_OQ)};
}
HWY_API Mask256<double> Gt(hwy::FloatTag /*tag*/, Vec256<double> a,
                           Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_GT_OQ)};
}

}  // namespace detail

template <typename T>
HWY_API Mask256<T> operator>(Vec256<T> a, Vec256<T> b) {
  return detail::Gt(hwy::TypeTag<T>(), a, b);
}

// ------------------------------ Weak inequality

namespace detail {

template <typename T>
HWY_INLINE Mask256<T> Ge(hwy::SignedTag tag, Vec256<T> a, Vec256<T> b) {
  return Not(Gt(tag, b, a));
}

template <typename T>
HWY_INLINE Mask256<T> Ge(hwy::UnsignedTag tag, Vec256<T> a, Vec256<T> b) {
  return Not(Gt(tag, b, a));
}

HWY_INLINE Mask256<float> Ge(hwy::FloatTag /*tag*/, Vec256<float> a,
                             Vec256<float> b) {
  return Mask256<float>{_mm256_cmp_ps(a.raw, b.raw, _CMP_GE_OQ)};
}
HWY_INLINE Mask256<double> Ge(hwy::FloatTag /*tag*/, Vec256<double> a,
                              Vec256<double> b) {
  return Mask256<double>{_mm256_cmp_pd(a.raw, b.raw, _CMP_GE_OQ)};
}

}  // namespace detail

template <typename T>
HWY_API Mask256<T> operator>=(Vec256<T> a, Vec256<T> b) {
  return detail::Ge(hwy::TypeTag<T>(), a, b);
}

#endif  // HWY_TARGET <= HWY_AVX3

// ------------------------------ Reversed comparisons

template <typename T>
HWY_API Mask256<T> operator<(const Vec256<T> a, const Vec256<T> b) {
  return b > a;
}

template <typename T>
HWY_API Mask256<T> operator<=(const Vec256<T> a, const Vec256<T> b) {
  return b >= a;
}

// ------------------------------ Min (Gt, IfThenElse)

// Unsigned
HWY_API Vec256<uint8_t> Min(const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_min_epu8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> Min(const Vec256<uint16_t> a,
                             const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_min_epu16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> Min(const Vec256<uint32_t> a,
                             const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_min_epu32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> Min(const Vec256<uint64_t> a,
                             const Vec256<uint64_t> b) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<uint64_t>{_mm256_min_epu64(a.raw, b.raw)};
#else
  const Full256<uint64_t> du;
  const Full256<int64_t> di;
  const auto msb = Set(du, 1ull << 63);
  const auto gt = RebindMask(du, BitCast(di, a ^ msb) > BitCast(di, b ^ msb));
  return IfThenElse(gt, b, a);
#endif
}

// Signed
HWY_API Vec256<int8_t> Min(const Vec256<int8_t> a, const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_min_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> Min(const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_min_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> Min(const Vec256<int32_t> a, const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_min_epi32(a.raw, b.raw)};
}
HWY_API Vec256<int64_t> Min(const Vec256<int64_t> a, const Vec256<int64_t> b) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<int64_t>{_mm256_min_epi64(a.raw, b.raw)};
#else
  return IfThenElse(a < b, a, b);
#endif
}

// Float
HWY_API Vec256<float> Min(const Vec256<float> a, const Vec256<float> b) {
  return Vec256<float>{_mm256_min_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> Min(const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>{_mm256_min_pd(a.raw, b.raw)};
}

// ------------------------------ Max (Gt, IfThenElse)

// Unsigned
HWY_API Vec256<uint8_t> Max(const Vec256<uint8_t> a, const Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_max_epu8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> Max(const Vec256<uint16_t> a,
                             const Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_max_epu16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> Max(const Vec256<uint32_t> a,
                             const Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_max_epu32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> Max(const Vec256<uint64_t> a,
                             const Vec256<uint64_t> b) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<uint64_t>{_mm256_max_epu64(a.raw, b.raw)};
#else
  const Full256<uint64_t> du;
  const Full256<int64_t> di;
  const auto msb = Set(du, 1ull << 63);
  const auto gt = RebindMask(du, BitCast(di, a ^ msb) > BitCast(di, b ^ msb));
  return IfThenElse(gt, a, b);
#endif
}

// Signed
HWY_API Vec256<int8_t> Max(const Vec256<int8_t> a, const Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_max_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> Max(const Vec256<int16_t> a, const Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_max_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> Max(const Vec256<int32_t> a, const Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_max_epi32(a.raw, b.raw)};
}
HWY_API Vec256<int64_t> Max(const Vec256<int64_t> a, const Vec256<int64_t> b) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<int64_t>{_mm256_max_epi64(a.raw, b.raw)};
#else
  return IfThenElse(a < b, b, a);
#endif
}

// Float
HWY_API Vec256<float> Max(const Vec256<float> a, const Vec256<float> b) {
  return Vec256<float>{_mm256_max_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> Max(const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>{_mm256_max_pd(a.raw, b.raw)};
}

// ------------------------------ FirstN (Iota, Lt)

template <class D, class M = MFromD<D>, HWY_IF_V_SIZE_D(D, 32)>
HWY_API M FirstN(const D d, size_t n) {
#if HWY_TARGET <= HWY_AVX3
  (void)d;
  constexpr size_t kN = MaxLanes(d);
#if HWY_ARCH_X86_64
  const uint64_t all = (1ull << kN) - 1;
  // BZHI only looks at the lower 8 bits of n!
  return M::FromBits((n > 255) ? all : _bzhi_u64(all, n));
#else
  const uint32_t all = static_cast<uint32_t>((1ull << kN) - 1);
  // BZHI only looks at the lower 8 bits of n!
  return M::FromBits((n > 255) ? all
                               : _bzhi_u32(all, static_cast<uint32_t>(n)));
#endif  // HWY_ARCH_X86_64
#else
  const RebindToSigned<decltype(d)> di;  // Signed comparisons are cheaper.
  using TI = TFromD<decltype(di)>;
  return RebindMask(d, Iota(di, 0) < Set(di, static_cast<TI>(n)));
#endif
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

// Unsigned
HWY_API Vec256<uint8_t> operator+(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_add_epi8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> operator+(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_add_epi16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> operator+(Vec256<uint32_t> a, Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_add_epi32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> operator+(Vec256<uint64_t> a, Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_add_epi64(a.raw, b.raw)};
}

// Signed
HWY_API Vec256<int8_t> operator+(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_add_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> operator+(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_add_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> operator+(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_add_epi32(a.raw, b.raw)};
}
HWY_API Vec256<int64_t> operator+(Vec256<int64_t> a, Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_add_epi64(a.raw, b.raw)};
}

// Float
HWY_API Vec256<float> operator+(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_add_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> operator+(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_add_pd(a.raw, b.raw)};
}

// ------------------------------ Subtraction

// Unsigned
HWY_API Vec256<uint8_t> operator-(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_sub_epi8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> operator-(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_sub_epi16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> operator-(Vec256<uint32_t> a, Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_sub_epi32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> operator-(Vec256<uint64_t> a, Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_sub_epi64(a.raw, b.raw)};
}

// Signed
HWY_API Vec256<int8_t> operator-(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_sub_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> operator-(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_sub_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> operator-(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_sub_epi32(a.raw, b.raw)};
}
HWY_API Vec256<int64_t> operator-(Vec256<int64_t> a, Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_sub_epi64(a.raw, b.raw)};
}

// Float
HWY_API Vec256<float> operator-(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_sub_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> operator-(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_sub_pd(a.raw, b.raw)};
}

// ------------------------------ SumsOf8
HWY_API Vec256<uint64_t> SumsOf8(Vec256<uint8_t> v) {
  return Vec256<uint64_t>{_mm256_sad_epu8(v.raw, _mm256_setzero_si256())};
}

HWY_API Vec256<uint64_t> SumsOf8AbsDiff(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint64_t>{_mm256_sad_epu8(a.raw, b.raw)};
}

// ------------------------------ SaturatedAdd

// Returns a + b clamped to the destination range.

// Unsigned
HWY_API Vec256<uint8_t> SaturatedAdd(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_adds_epu8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> SaturatedAdd(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_adds_epu16(a.raw, b.raw)};
}

// Signed
HWY_API Vec256<int8_t> SaturatedAdd(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_adds_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> SaturatedAdd(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_adds_epi16(a.raw, b.raw)};
}

// ------------------------------ SaturatedSub

// Returns a - b clamped to the destination range.

// Unsigned
HWY_API Vec256<uint8_t> SaturatedSub(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_subs_epu8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> SaturatedSub(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_subs_epu16(a.raw, b.raw)};
}

// Signed
HWY_API Vec256<int8_t> SaturatedSub(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_subs_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> SaturatedSub(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_subs_epi16(a.raw, b.raw)};
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

// Unsigned
HWY_API Vec256<uint8_t> AverageRound(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_avg_epu8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> AverageRound(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_avg_epu16(a.raw, b.raw)};
}

// ------------------------------ Abs (Sub)

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
HWY_API Vec256<int8_t> Abs(Vec256<int8_t> v) {
#if HWY_COMPILER_MSVC
  // Workaround for incorrect codegen? (wrong result)
  const DFromV<decltype(v)> d;
  const auto zero = Zero(d);
  return Vec256<int8_t>{_mm256_max_epi8(v.raw, (zero - v).raw)};
#else
  return Vec256<int8_t>{_mm256_abs_epi8(v.raw)};
#endif
}
HWY_API Vec256<int16_t> Abs(const Vec256<int16_t> v) {
  return Vec256<int16_t>{_mm256_abs_epi16(v.raw)};
}
HWY_API Vec256<int32_t> Abs(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_abs_epi32(v.raw)};
}
// i64 is implemented after BroadcastSignBit.

HWY_API Vec256<float> Abs(const Vec256<float> v) {
  const DFromV<decltype(v)> d;
  const Vec256<int32_t> mask{_mm256_set1_epi32(0x7FFFFFFF)};
  return v & BitCast(d, mask);
}
HWY_API Vec256<double> Abs(const Vec256<double> v) {
  const DFromV<decltype(v)> d;
  const Vec256<int64_t> mask{_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)};
  return v & BitCast(d, mask);
}

// ------------------------------ Integer multiplication

// Unsigned
HWY_API Vec256<uint16_t> operator*(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_mullo_epi16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> operator*(Vec256<uint32_t> a, Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_mullo_epi32(a.raw, b.raw)};
}

// Signed
HWY_API Vec256<int16_t> operator*(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_mullo_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> operator*(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_mullo_epi32(a.raw, b.raw)};
}

// Returns the upper 16 bits of a * b in each lane.
HWY_API Vec256<uint16_t> MulHigh(Vec256<uint16_t> a, Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_mulhi_epu16(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> MulHigh(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_mulhi_epi16(a.raw, b.raw)};
}

HWY_API Vec256<int16_t> MulFixedPoint15(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_mulhrs_epi16(a.raw, b.raw)};
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
HWY_API Vec256<int64_t> MulEven(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Vec256<int64_t>{_mm256_mul_epi32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> MulEven(Vec256<uint32_t> a, Vec256<uint32_t> b) {
  return Vec256<uint64_t>{_mm256_mul_epu32(a.raw, b.raw)};
}

// ------------------------------ ShiftLeft

template <int kBits>
HWY_API Vec256<uint16_t> ShiftLeft(Vec256<uint16_t> v) {
  return Vec256<uint16_t>{_mm256_slli_epi16(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<uint32_t> ShiftLeft(Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_slli_epi32(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<uint64_t> ShiftLeft(Vec256<uint64_t> v) {
  return Vec256<uint64_t>{_mm256_slli_epi64(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<int16_t> ShiftLeft(Vec256<int16_t> v) {
  return Vec256<int16_t>{_mm256_slli_epi16(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<int32_t> ShiftLeft(Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_slli_epi32(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<int64_t> ShiftLeft(Vec256<int64_t> v) {
  return Vec256<int64_t>{_mm256_slli_epi64(v.raw, kBits)};
}

template <int kBits, typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> ShiftLeft(const Vec256<T> v) {
  const Full256<T> d8;
  const RepartitionToWide<decltype(d8)> d16;
  const auto shifted = BitCast(d8, ShiftLeft<kBits>(BitCast(d16, v)));
  return kBits == 1
             ? (v + v)
             : (shifted & Set(d8, static_cast<T>((0xFF << kBits) & 0xFF)));
}

// ------------------------------ ShiftRight

template <int kBits>
HWY_API Vec256<uint16_t> ShiftRight(Vec256<uint16_t> v) {
  return Vec256<uint16_t>{_mm256_srli_epi16(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<uint32_t> ShiftRight(Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_srli_epi32(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<uint64_t> ShiftRight(Vec256<uint64_t> v) {
  return Vec256<uint64_t>{_mm256_srli_epi64(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<uint8_t> ShiftRight(Vec256<uint8_t> v) {
  const Full256<uint8_t> d8;
  // Use raw instead of BitCast to support N=1.
  const Vec256<uint8_t> shifted{ShiftRight<kBits>(Vec256<uint16_t>{v.raw}).raw};
  return shifted & Set(d8, 0xFF >> kBits);
}

template <int kBits>
HWY_API Vec256<int16_t> ShiftRight(Vec256<int16_t> v) {
  return Vec256<int16_t>{_mm256_srai_epi16(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<int32_t> ShiftRight(Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_srai_epi32(v.raw, kBits)};
}

template <int kBits>
HWY_API Vec256<int8_t> ShiftRight(Vec256<int8_t> v) {
  const Full256<int8_t> di;
  const Full256<uint8_t> du;
  const auto shifted = BitCast(di, ShiftRight<kBits>(BitCast(du, v)));
  const auto shifted_sign = BitCast(di, Set(du, 0x80 >> kBits));
  return (shifted ^ shifted_sign) - shifted_sign;
}

// i64 is implemented after BroadcastSignBit.

// ------------------------------ RotateRight

template <int kBits>
HWY_API Vec256<uint32_t> RotateRight(const Vec256<uint32_t> v) {
  static_assert(0 <= kBits && kBits < 32, "Invalid shift count");
#if HWY_TARGET <= HWY_AVX3
  return Vec256<uint32_t>{_mm256_ror_epi32(v.raw, kBits)};
#else
  if (kBits == 0) return v;
  return Or(ShiftRight<kBits>(v), ShiftLeft<HWY_MIN(31, 32 - kBits)>(v));
#endif
}

template <int kBits>
HWY_API Vec256<uint64_t> RotateRight(const Vec256<uint64_t> v) {
  static_assert(0 <= kBits && kBits < 64, "Invalid shift count");
#if HWY_TARGET <= HWY_AVX3
  return Vec256<uint64_t>{_mm256_ror_epi64(v.raw, kBits)};
#else
  if (kBits == 0) return v;
  return Or(ShiftRight<kBits>(v), ShiftLeft<HWY_MIN(63, 64 - kBits)>(v));
#endif
}

// ------------------------------ BroadcastSignBit (ShiftRight, compare, mask)

HWY_API Vec256<int8_t> BroadcastSignBit(const Vec256<int8_t> v) {
  const DFromV<decltype(v)> d;
  return VecFromMask(v < Zero(d));
}

HWY_API Vec256<int16_t> BroadcastSignBit(const Vec256<int16_t> v) {
  return ShiftRight<15>(v);
}

HWY_API Vec256<int32_t> BroadcastSignBit(const Vec256<int32_t> v) {
  return ShiftRight<31>(v);
}

HWY_API Vec256<int64_t> BroadcastSignBit(const Vec256<int64_t> v) {
#if HWY_TARGET == HWY_AVX2
  const DFromV<decltype(v)> d;
  return VecFromMask(v < Zero(d));
#else
  return Vec256<int64_t>{_mm256_srai_epi64(v.raw, 63)};
#endif
}

template <int kBits>
HWY_API Vec256<int64_t> ShiftRight(const Vec256<int64_t> v) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<int64_t>{
      _mm256_srai_epi64(v.raw, static_cast<Shift64Count>(kBits))};
#else
  const Full256<int64_t> di;
  const Full256<uint64_t> du;
  const auto right = BitCast(di, ShiftRight<kBits>(BitCast(du, v)));
  const auto sign = ShiftLeft<64 - kBits>(BroadcastSignBit(v));
  return right | sign;
#endif
}

HWY_API Vec256<int64_t> Abs(const Vec256<int64_t> v) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<int64_t>{_mm256_abs_epi64(v.raw)};
#else
  const DFromV<decltype(v)> d;
  const auto zero = Zero(d);
  return IfThenElse(MaskFromVec(BroadcastSignBit(v)), zero - v, v);
#endif
}

// ------------------------------ IfNegativeThenElse (BroadcastSignBit)
HWY_API Vec256<int8_t> IfNegativeThenElse(Vec256<int8_t> v, Vec256<int8_t> yes,
                                          Vec256<int8_t> no) {
  // int8: AVX2 IfThenElse only looks at the MSB.
  return IfThenElse(MaskFromVec(v), yes, no);
}

template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> IfNegativeThenElse(Vec256<T> v, Vec256<T> yes, Vec256<T> no) {
  static_assert(IsSigned<T>(), "Only works for signed/float");
  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;

  // 16-bit: no native blendv, so copy sign to lower byte's MSB.
  v = BitCast(d, BroadcastSignBit(BitCast(di, v)));
  return IfThenElse(MaskFromVec(v), yes, no);
}

template <typename T, HWY_IF_NOT_T_SIZE(T, 2)>
HWY_API Vec256<T> IfNegativeThenElse(Vec256<T> v, Vec256<T> yes, Vec256<T> no) {
  static_assert(IsSigned<T>(), "Only works for signed/float");
  const DFromV<decltype(v)> d;
  const RebindToFloat<decltype(d)> df;

  // 32/64-bit: use float IfThenElse, which only looks at the MSB.
  const MFromD<decltype(df)> msb = MaskFromVec(BitCast(df, v));
  return BitCast(d, IfThenElse(msb, BitCast(df, yes), BitCast(df, no)));
}

// ------------------------------ ShiftLeftSame

HWY_API Vec256<uint16_t> ShiftLeftSame(const Vec256<uint16_t> v,
                                       const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<uint16_t>{_mm256_slli_epi16(v.raw, bits)};
  }
#endif
  return Vec256<uint16_t>{_mm256_sll_epi16(v.raw, _mm_cvtsi32_si128(bits))};
}
HWY_API Vec256<uint32_t> ShiftLeftSame(const Vec256<uint32_t> v,
                                       const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<uint32_t>{_mm256_slli_epi32(v.raw, bits)};
  }
#endif
  return Vec256<uint32_t>{_mm256_sll_epi32(v.raw, _mm_cvtsi32_si128(bits))};
}
HWY_API Vec256<uint64_t> ShiftLeftSame(const Vec256<uint64_t> v,
                                       const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<uint64_t>{_mm256_slli_epi64(v.raw, bits)};
  }
#endif
  return Vec256<uint64_t>{_mm256_sll_epi64(v.raw, _mm_cvtsi32_si128(bits))};
}

HWY_API Vec256<int16_t> ShiftLeftSame(const Vec256<int16_t> v, const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<int16_t>{_mm256_slli_epi16(v.raw, bits)};
  }
#endif
  return Vec256<int16_t>{_mm256_sll_epi16(v.raw, _mm_cvtsi32_si128(bits))};
}

HWY_API Vec256<int32_t> ShiftLeftSame(const Vec256<int32_t> v, const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<int32_t>{_mm256_slli_epi32(v.raw, bits)};
  }
#endif
  return Vec256<int32_t>{_mm256_sll_epi32(v.raw, _mm_cvtsi32_si128(bits))};
}

HWY_API Vec256<int64_t> ShiftLeftSame(const Vec256<int64_t> v, const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<int64_t>{_mm256_slli_epi64(v.raw, bits)};
  }
#endif
  return Vec256<int64_t>{_mm256_sll_epi64(v.raw, _mm_cvtsi32_si128(bits))};
}

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> ShiftLeftSame(const Vec256<T> v, const int bits) {
  const Full256<T> d8;
  const RepartitionToWide<decltype(d8)> d16;
  const auto shifted = BitCast(d8, ShiftLeftSame(BitCast(d16, v), bits));
  return shifted & Set(d8, static_cast<T>((0xFF << bits) & 0xFF));
}

// ------------------------------ ShiftRightSame (BroadcastSignBit)

HWY_API Vec256<uint16_t> ShiftRightSame(const Vec256<uint16_t> v,
                                        const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<uint16_t>{_mm256_srli_epi16(v.raw, bits)};
  }
#endif
  return Vec256<uint16_t>{_mm256_srl_epi16(v.raw, _mm_cvtsi32_si128(bits))};
}
HWY_API Vec256<uint32_t> ShiftRightSame(const Vec256<uint32_t> v,
                                        const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<uint32_t>{_mm256_srli_epi32(v.raw, bits)};
  }
#endif
  return Vec256<uint32_t>{_mm256_srl_epi32(v.raw, _mm_cvtsi32_si128(bits))};
}
HWY_API Vec256<uint64_t> ShiftRightSame(const Vec256<uint64_t> v,
                                        const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<uint64_t>{_mm256_srli_epi64(v.raw, bits)};
  }
#endif
  return Vec256<uint64_t>{_mm256_srl_epi64(v.raw, _mm_cvtsi32_si128(bits))};
}

HWY_API Vec256<uint8_t> ShiftRightSame(Vec256<uint8_t> v, const int bits) {
  const Full256<uint8_t> d8;
  const RepartitionToWide<decltype(d8)> d16;
  const auto shifted = BitCast(d8, ShiftRightSame(BitCast(d16, v), bits));
  return shifted & Set(d8, static_cast<uint8_t>(0xFF >> bits));
}

HWY_API Vec256<int16_t> ShiftRightSame(const Vec256<int16_t> v,
                                       const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<int16_t>{_mm256_srai_epi16(v.raw, bits)};
  }
#endif
  return Vec256<int16_t>{_mm256_sra_epi16(v.raw, _mm_cvtsi32_si128(bits))};
}

HWY_API Vec256<int32_t> ShiftRightSame(const Vec256<int32_t> v,
                                       const int bits) {
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<int32_t>{_mm256_srai_epi32(v.raw, bits)};
  }
#endif
  return Vec256<int32_t>{_mm256_sra_epi32(v.raw, _mm_cvtsi32_si128(bits))};
}
HWY_API Vec256<int64_t> ShiftRightSame(const Vec256<int64_t> v,
                                       const int bits) {
#if HWY_TARGET <= HWY_AVX3
#if HWY_COMPILER_GCC
  if (__builtin_constant_p(bits)) {
    return Vec256<int64_t>{
        _mm256_srai_epi64(v.raw, static_cast<Shift64Count>(bits))};
  }
#endif
  return Vec256<int64_t>{_mm256_sra_epi64(v.raw, _mm_cvtsi32_si128(bits))};
#else
  const Full256<int64_t> di;
  const Full256<uint64_t> du;
  const auto right = BitCast(di, ShiftRightSame(BitCast(du, v), bits));
  const auto sign = ShiftLeftSame(BroadcastSignBit(v), 64 - bits);
  return right | sign;
#endif
}

HWY_API Vec256<int8_t> ShiftRightSame(Vec256<int8_t> v, const int bits) {
  const Full256<int8_t> di;
  const Full256<uint8_t> du;
  const auto shifted = BitCast(di, ShiftRightSame(BitCast(du, v), bits));
  const auto shifted_sign =
      BitCast(di, Set(du, static_cast<uint8_t>(0x80 >> bits)));
  return (shifted ^ shifted_sign) - shifted_sign;
}

// ------------------------------ Neg (Xor, Sub)

// Tag dispatch instead of SFINAE for MSVC 2017 compatibility
namespace detail {

template <typename T>
HWY_INLINE Vec256<T> Neg(hwy::FloatTag /*tag*/, const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  return Xor(v, SignBit(d));
}

// Not floating-point
template <typename T>
HWY_INLINE Vec256<T> Neg(hwy::NonFloatTag /*tag*/, const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  return Zero(d) - v;
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> Neg(const Vec256<T> v) {
  return detail::Neg(hwy::IsFloatTag<T>(), v);
}

// ------------------------------ Floating-point mul / div

HWY_API Vec256<float> operator*(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_mul_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> operator*(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_mul_pd(a.raw, b.raw)};
}

HWY_API Vec256<float> operator/(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_div_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> operator/(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_div_pd(a.raw, b.raw)};
}

// Approximate reciprocal
HWY_API Vec256<float> ApproximateReciprocal(Vec256<float> v) {
  return Vec256<float>{_mm256_rcp_ps(v.raw)};
}

// Absolute value of difference.
HWY_API Vec256<float> AbsDiff(Vec256<float> a, Vec256<float> b) {
  return Abs(a - b);
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
HWY_API Vec256<float> MulAdd(Vec256<float> mul, Vec256<float> x,
                             Vec256<float> add) {
#ifdef HWY_DISABLE_BMI2_FMA
  return mul * x + add;
#else
  return Vec256<float>{_mm256_fmadd_ps(mul.raw, x.raw, add.raw)};
#endif
}
HWY_API Vec256<double> MulAdd(Vec256<double> mul, Vec256<double> x,
                              Vec256<double> add) {
#ifdef HWY_DISABLE_BMI2_FMA
  return mul * x + add;
#else
  return Vec256<double>{_mm256_fmadd_pd(mul.raw, x.raw, add.raw)};
#endif
}

// Returns add - mul * x
HWY_API Vec256<float> NegMulAdd(Vec256<float> mul, Vec256<float> x,
                                Vec256<float> add) {
#ifdef HWY_DISABLE_BMI2_FMA
  return add - mul * x;
#else
  return Vec256<float>{_mm256_fnmadd_ps(mul.raw, x.raw, add.raw)};
#endif
}
HWY_API Vec256<double> NegMulAdd(Vec256<double> mul, Vec256<double> x,
                                 Vec256<double> add) {
#ifdef HWY_DISABLE_BMI2_FMA
  return add - mul * x;
#else
  return Vec256<double>{_mm256_fnmadd_pd(mul.raw, x.raw, add.raw)};
#endif
}

// Returns mul * x - sub
HWY_API Vec256<float> MulSub(Vec256<float> mul, Vec256<float> x,
                             Vec256<float> sub) {
#ifdef HWY_DISABLE_BMI2_FMA
  return mul * x - sub;
#else
  return Vec256<float>{_mm256_fmsub_ps(mul.raw, x.raw, sub.raw)};
#endif
}
HWY_API Vec256<double> MulSub(Vec256<double> mul, Vec256<double> x,
                              Vec256<double> sub) {
#ifdef HWY_DISABLE_BMI2_FMA
  return mul * x - sub;
#else
  return Vec256<double>{_mm256_fmsub_pd(mul.raw, x.raw, sub.raw)};
#endif
}

// Returns -mul * x - sub
HWY_API Vec256<float> NegMulSub(Vec256<float> mul, Vec256<float> x,
                                Vec256<float> sub) {
#ifdef HWY_DISABLE_BMI2_FMA
  return Neg(mul * x) - sub;
#else
  return Vec256<float>{_mm256_fnmsub_ps(mul.raw, x.raw, sub.raw)};
#endif
}
HWY_API Vec256<double> NegMulSub(Vec256<double> mul, Vec256<double> x,
                                 Vec256<double> sub) {
#ifdef HWY_DISABLE_BMI2_FMA
  return Neg(mul * x) - sub;
#else
  return Vec256<double>{_mm256_fnmsub_pd(mul.raw, x.raw, sub.raw)};
#endif
}

// ------------------------------ Floating-point square root

// Full precision square root
HWY_API Vec256<float> Sqrt(Vec256<float> v) {
  return Vec256<float>{_mm256_sqrt_ps(v.raw)};
}
HWY_API Vec256<double> Sqrt(Vec256<double> v) {
  return Vec256<double>{_mm256_sqrt_pd(v.raw)};
}

// Approximate reciprocal square root
HWY_API Vec256<float> ApproximateReciprocalSqrt(Vec256<float> v) {
  return Vec256<float>{_mm256_rsqrt_ps(v.raw)};
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, tie to even
HWY_API Vec256<float> Round(Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
HWY_API Vec256<double> Round(Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}

// Toward zero, aka truncate
HWY_API Vec256<float> Trunc(Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
}
HWY_API Vec256<double> Trunc(Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
}

// Toward +infinity, aka ceiling
HWY_API Vec256<float> Ceil(Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)};
}
HWY_API Vec256<double> Ceil(Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)};
}

// Toward -infinity, aka floor
HWY_API Vec256<float> Floor(Vec256<float> v) {
  return Vec256<float>{
      _mm256_round_ps(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)};
}
HWY_API Vec256<double> Floor(Vec256<double> v) {
  return Vec256<double>{
      _mm256_round_pd(v.raw, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)};
}

// ------------------------------ Floating-point classification

HWY_API Mask256<float> IsNaN(Vec256<float> v) {
#if HWY_TARGET <= HWY_AVX3
  return Mask256<float>{_mm256_fpclass_ps_mask(v.raw, 0x81)};
#else
  return Mask256<float>{_mm256_cmp_ps(v.raw, v.raw, _CMP_UNORD_Q)};
#endif
}
HWY_API Mask256<double> IsNaN(Vec256<double> v) {
#if HWY_TARGET <= HWY_AVX3
  return Mask256<double>{_mm256_fpclass_pd_mask(v.raw, 0x81)};
#else
  return Mask256<double>{_mm256_cmp_pd(v.raw, v.raw, _CMP_UNORD_Q)};
#endif
}

#if HWY_TARGET <= HWY_AVX3

HWY_API Mask256<float> IsInf(Vec256<float> v) {
  return Mask256<float>{_mm256_fpclass_ps_mask(v.raw, 0x18)};
}
HWY_API Mask256<double> IsInf(Vec256<double> v) {
  return Mask256<double>{_mm256_fpclass_pd_mask(v.raw, 0x18)};
}

HWY_API Mask256<float> IsFinite(Vec256<float> v) {
  // fpclass doesn't have a flag for positive, so we have to check for inf/NaN
  // and negate the mask.
  return Not(Mask256<float>{_mm256_fpclass_ps_mask(v.raw, 0x99)});
}
HWY_API Mask256<double> IsFinite(Vec256<double> v) {
  return Not(Mask256<double>{_mm256_fpclass_pd_mask(v.raw, 0x99)});
}

#else

template <typename T>
HWY_API Mask256<T> IsInf(const Vec256<T> v) {
  static_assert(IsFloat<T>(), "Only for float");
  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;
  const VFromD<decltype(di)> vi = BitCast(di, v);
  // 'Shift left' to clear the sign bit, check for exponent=max and mantissa=0.
  return RebindMask(d, Eq(Add(vi, vi), Set(di, hwy::MaxExponentTimes2<T>())));
}

// Returns whether normal/subnormal/zero.
template <typename T>
HWY_API Mask256<T> IsFinite(const Vec256<T> v) {
  static_assert(IsFloat<T>(), "Only for float");
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du;
  const RebindToSigned<decltype(d)> di;  // cheaper than unsigned comparison
  const VFromD<decltype(du)> vu = BitCast(du, v);
  // Shift left to clear the sign bit, then right so we can compare with the
  // max exponent (cannot compare with MaxExponentTimes2 directly because it is
  // negative and non-negative floats would be greater). MSVC seems to generate
  // incorrect code if we instead add vu + vu.
  const VFromD<decltype(di)> exp =
      BitCast(di, ShiftRight<hwy::MantissaBits<T>() + 1>(ShiftLeft<1>(vu)));
  return RebindMask(d, Lt(exp, Set(di, hwy::MaxExponentField<T>())));
}

#endif  // HWY_TARGET <= HWY_AVX3

// ================================================== MEMORY

// ------------------------------ Load

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API Vec256<T> Load(D /* tag */, const T* HWY_RESTRICT aligned) {
  return Vec256<T>{
      _mm256_load_si256(reinterpret_cast<const __m256i*>(aligned))};
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> Load(D /* tag */, const float* HWY_RESTRICT aligned) {
  return Vec256<float>{_mm256_load_ps(aligned)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> Load(D /* tag */, const double* HWY_RESTRICT aligned) {
  return Vec256<double>{_mm256_load_pd(aligned)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API Vec256<T> LoadU(D /* tag */, const T* HWY_RESTRICT p) {
  return Vec256<T>{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))};
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> LoadU(D /* tag */, const float* HWY_RESTRICT p) {
  return Vec256<float>{_mm256_loadu_ps(p)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> LoadU(D /* tag */, const double* HWY_RESTRICT p) {
  return Vec256<double>{_mm256_loadu_pd(p)};
}

// ------------------------------ MaskedLoad

#if HWY_TARGET <= HWY_AVX3

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D /* tag */,
                             const T* HWY_RESTRICT p) {
  return Vec256<T>{_mm256_maskz_loadu_epi8(m.raw, p)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D /* tag */,
                             const T* HWY_RESTRICT p) {
  return Vec256<T>{_mm256_maskz_loadu_epi16(m.raw, p)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D /* tag */,
                             const T* HWY_RESTRICT p) {
  return Vec256<T>{_mm256_maskz_loadu_epi32(m.raw, p)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D /* tag */,
                             const T* HWY_RESTRICT p) {
  return Vec256<T>{_mm256_maskz_loadu_epi64(m.raw, p)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> MaskedLoad(Mask256<float> m, D /* tag */,
                                 const float* HWY_RESTRICT p) {
  return Vec256<float>{_mm256_maskz_loadu_ps(m.raw, p)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> MaskedLoad(Mask256<double> m, D /* tag */,
                                  const double* HWY_RESTRICT p) {
  return Vec256<double>{_mm256_maskz_loadu_pd(m.raw, p)};
}

#else  //  AVX2

// There is no maskload_epi8/16, so blend instead.
template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_T_SIZE_ONE_OF(T, (1 << 1) | (1 << 2))>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D d, const T* HWY_RESTRICT p) {
  return IfThenElseZero(m, LoadU(d, p));
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D /* tag */,
                             const T* HWY_RESTRICT p) {
  auto pi = reinterpret_cast<const int*>(p);  // NOLINT
  return Vec256<T>{_mm256_maskload_epi32(pi, m.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, D /* tag */,
                             const T* HWY_RESTRICT p) {
  auto pi = reinterpret_cast<const long long*>(p);  // NOLINT
  return Vec256<T>{_mm256_maskload_epi64(pi, m.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> MaskedLoad(Mask256<float> m, D d,
                                 const float* HWY_RESTRICT p) {
  const Vec256<int32_t> mi =
      BitCast(RebindToSigned<decltype(d)>(), VecFromMask(d, m));
  return Vec256<float>{_mm256_maskload_ps(p, mi.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> MaskedLoad(Mask256<double> m, D d,
                                  const double* HWY_RESTRICT p) {
  const Vec256<int64_t> mi =
      BitCast(RebindToSigned<decltype(d)>(), VecFromMask(d, m));
  return Vec256<double>{_mm256_maskload_pd(p, mi.raw)};
}

#endif

// ------------------------------ LoadDup128

// Loads 128 bit and duplicates into both 128-bit halves. This avoids the
// 3-cycle cost of moving data between 128-bit halves and avoids port 5.
template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API Vec256<T> LoadDup128(D /* tag */, const T* HWY_RESTRICT p) {
  const Full128<T> d128;
#if HWY_COMPILER_MSVC && HWY_COMPILER_MSVC < 1931
  // Workaround for incorrect results with _mm256_broadcastsi128_si256. Note
  // that MSVC also lacks _mm256_zextsi128_si256, but cast (which leaves the
  // upper half undefined) is fine because we're overwriting that anyway.
  // This workaround seems in turn to generate incorrect code in MSVC 2022
  // (19.31), so use broadcastsi128 there.
  const __m128i v128 = LoadU(d128, p).raw;
  return Vec256<T>{
      _mm256_inserti128_si256(_mm256_castsi128_si256(v128), v128, 1)};
#else
  return Vec256<T>{_mm256_broadcastsi128_si256(LoadU(d128, p).raw)};
#endif
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> LoadDup128(D /* tag */, const float* HWY_RESTRICT p) {
#if HWY_COMPILER_MSVC && HWY_COMPILER_MSVC < 1931
  const Full128<float> d128;
  const __m128 v128 = LoadU(d128, p).raw;
  return Vec256<float>{
      _mm256_insertf128_ps(_mm256_castps128_ps256(v128), v128, 1)};
#else
  return Vec256<float>{_mm256_broadcast_ps(reinterpret_cast<const __m128*>(p))};
#endif
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> LoadDup128(D /* tag */, const double* HWY_RESTRICT p) {
#if HWY_COMPILER_MSVC && HWY_COMPILER_MSVC < 1931
  const Full128<double> d128;
  const __m128d v128 = LoadU(d128, p).raw;
  return Vec256<double>{
      _mm256_insertf128_pd(_mm256_castpd128_pd256(v128), v128, 1)};
#else
  return Vec256<double>{
      _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p))};
#endif
}

// ------------------------------ Store

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API void Store(Vec256<T> v, D /* tag */, T* HWY_RESTRICT aligned) {
  _mm256_store_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void Store(Vec256<float> v, D /* tag */, float* HWY_RESTRICT aligned) {
  _mm256_store_ps(aligned, v.raw);
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void Store(Vec256<double> v, D /* tag */,
                   double* HWY_RESTRICT aligned) {
  _mm256_store_pd(aligned, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API void StoreU(Vec256<T> v, D /* tag */, T* HWY_RESTRICT p) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v.raw);
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void StoreU(Vec256<float> v, D /* tag */, float* HWY_RESTRICT p) {
  _mm256_storeu_ps(p, v.raw);
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void StoreU(Vec256<double> v, D /* tag */, double* HWY_RESTRICT p) {
  _mm256_storeu_pd(p, v.raw);
}

// ------------------------------ BlendedStore

#if HWY_TARGET <= HWY_AVX3

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_T_SIZE(T, 1)>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D /* tag */,
                          T* HWY_RESTRICT p) {
  _mm256_mask_storeu_epi8(p, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_T_SIZE(T, 2)>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D /* tag */,
                          T* HWY_RESTRICT p) {
  _mm256_mask_storeu_epi16(p, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D /* tag */,
                          T* HWY_RESTRICT p) {
  _mm256_mask_storeu_epi32(p, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D /* tag */,
                          T* HWY_RESTRICT p) {
  _mm256_mask_storeu_epi64(p, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void BlendedStore(Vec256<float> v, Mask256<float> m, D /* tag */,
                          float* HWY_RESTRICT p) {
  _mm256_mask_storeu_ps(p, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void BlendedStore(Vec256<double> v, Mask256<double> m, D /* tag */,
                          double* HWY_RESTRICT p) {
  _mm256_mask_storeu_pd(p, m.raw, v.raw);
}

#else  //  AVX2

// Intel SDM says "No AC# reported for any mask bit combinations". However, AMD
// allows AC# if "Alignment checking enabled and: 256-bit memory operand not
// 32-byte aligned". Fortunately AC# is not enabled by default and requires both
// OS support (CR0) and the application to set rflags.AC. We assume these remain
// disabled because x86/x64 code and compiler output often contain misaligned
// scalar accesses, which would also fault.
//
// Caveat: these are slow on AMD Jaguar/Bulldozer.

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_T_SIZE_ONE_OF(T, (1 << 1) | (1 << 2))>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D d, T* HWY_RESTRICT p) {
  // There is no maskload_epi8/16. Blending is also unsafe because loading a
  // full vector that crosses the array end causes asan faults. Resort to scalar
  // code; the caller should instead use memcpy, assuming m is FirstN(d, n).
  const RebindToUnsigned<decltype(d)> du;
  using TU = TFromD<decltype(du)>;
  alignas(32) TU buf[MaxLanes(d)];
  alignas(32) TU mask[MaxLanes(d)];
  Store(BitCast(du, v), du, buf);
  Store(BitCast(du, VecFromMask(d, m)), du, mask);
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    if (mask[i]) {
      CopySameSize(buf + i, p + i);
    }
  }
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D /* tag */,
                          T* HWY_RESTRICT p) {
  auto pi = reinterpret_cast<int*>(p);  // NOLINT
  _mm256_maskstore_epi32(pi, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, D /* tag */,
                          T* HWY_RESTRICT p) {
  auto pi = reinterpret_cast<long long*>(p);  // NOLINT
  _mm256_maskstore_epi64(pi, m.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void BlendedStore(Vec256<float> v, Mask256<float> m, D d,
                          float* HWY_RESTRICT p) {
  const Vec256<int32_t> mi =
      BitCast(RebindToSigned<decltype(d)>(), VecFromMask(d, m));
  _mm256_maskstore_ps(p, mi.raw, v.raw);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void BlendedStore(Vec256<double> v, Mask256<double> m, D d,
                          double* HWY_RESTRICT p) {
  const Vec256<int64_t> mi =
      BitCast(RebindToSigned<decltype(d)>(), VecFromMask(d, m));
  _mm256_maskstore_pd(p, mi.raw, v.raw);
}

#endif

// ------------------------------ Non-temporal stores

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_NOT_FLOAT(T)>
HWY_API void Stream(Vec256<T> v, D /* tag */, T* HWY_RESTRICT aligned) {
  _mm256_stream_si256(reinterpret_cast<__m256i*>(aligned), v.raw);
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void Stream(Vec256<float> v, D /* tag */, float* HWY_RESTRICT aligned) {
  _mm256_stream_ps(aligned, v.raw);
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void Stream(Vec256<double> v, D /* tag */,
                    double* HWY_RESTRICT aligned) {
  _mm256_stream_pd(aligned, v.raw);
}

// ------------------------------ Scatter

// Work around warnings in the intrinsic definitions (passing -1 as a mask).
HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4245 4365, ignored "-Wsign-conversion")

#if HWY_TARGET <= HWY_AVX3

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_API void ScatterOffset(Vec256<T> v, D /* tag */, T* HWY_RESTRICT base,
                           Vec256<int32_t> offset) {
  _mm256_i32scatter_epi32(base, offset.raw, v.raw, 1);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_API void ScatterIndex(Vec256<T> v, D /* tag */, T* HWY_RESTRICT base,
                          Vec256<int32_t> index) {
  _mm256_i32scatter_epi32(base, index.raw, v.raw, 4);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_API void ScatterOffset(Vec256<T> v, D /* tag */, T* HWY_RESTRICT base,
                           Vec256<int64_t> offset) {
  _mm256_i64scatter_epi64(base, offset.raw, v.raw, 1);
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_API void ScatterIndex(Vec256<T> v, D /* tag */, T* HWY_RESTRICT base,
                          Vec256<int64_t> index) {
  _mm256_i64scatter_epi64(base, index.raw, v.raw, 8);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void ScatterOffset(Vec256<float> v, D /* tag */,
                           float* HWY_RESTRICT base,
                           const Vec256<int32_t> offset) {
  _mm256_i32scatter_ps(base, offset.raw, v.raw, 1);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void ScatterIndex(Vec256<float> v, D /* tag */,
                          float* HWY_RESTRICT base,
                          const Vec256<int32_t> index) {
  _mm256_i32scatter_ps(base, index.raw, v.raw, 4);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void ScatterOffset(Vec256<double> v, D /* tag */,
                           double* HWY_RESTRICT base,
                           const Vec256<int64_t> offset) {
  _mm256_i64scatter_pd(base, offset.raw, v.raw, 1);
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API void ScatterIndex(Vec256<double> v, D /* tag */,
                          double* HWY_RESTRICT base,
                          const Vec256<int64_t> index) {
  _mm256_i64scatter_pd(base, index.raw, v.raw, 8);
}

#else

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          typename Offset>
HWY_API void ScatterOffset(Vec256<T> v, D d, T* HWY_RESTRICT base,
                           const Vec256<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "Must match for portability");

  alignas(32) T lanes[MaxLanes(d)];
  Store(v, d, lanes);

  alignas(32) Offset offset_lanes[MaxLanes(d)];
  Store(offset, Full256<Offset>(), offset_lanes);

  uint8_t* base_bytes = reinterpret_cast<uint8_t*>(base);
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    CopyBytes<sizeof(T)>(&lanes[i], base_bytes + offset_lanes[i]);
  }
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          typename Index>
HWY_API void ScatterIndex(Vec256<T> v, D d, T* HWY_RESTRICT base,
                          const Vec256<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "Must match for portability");

  alignas(32) T lanes[MaxLanes(d)];
  Store(v, d, lanes);

  alignas(32) Index index_lanes[MaxLanes(d)];
  Store(index, Full256<Index>(), index_lanes);

  for (size_t i = 0; i < MaxLanes(d); ++i) {
    base[index_lanes[i]] = lanes[i];
  }
}

#endif

// ------------------------------ Gather

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_INLINE Vec256<T> GatherOffset(D /* tag */, const T* HWY_RESTRICT base,
                                  Vec256<int32_t> offset) {
  return Vec256<T>{_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), offset.raw, 1)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI32(T)>
HWY_INLINE Vec256<T> GatherIndex(D /* tag */, const T* HWY_RESTRICT base,
                                 Vec256<int32_t> index) {
  return Vec256<T>{_mm256_i32gather_epi32(
      reinterpret_cast<const int32_t*>(base), index.raw, 4)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_INLINE Vec256<T> GatherOffset(D /* tag */, const T* HWY_RESTRICT base,
                                  Vec256<int64_t> offset) {
  return Vec256<T>{_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), offset.raw, 1)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>,
          HWY_IF_UI64(T)>
HWY_INLINE Vec256<T> GatherIndex(D /* tag */, const T* HWY_RESTRICT base,
                                 Vec256<int64_t> index) {
  return Vec256<T>{_mm256_i64gather_epi64(
      reinterpret_cast<const GatherIndex64*>(base), index.raw, 8)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> GatherOffset(D /* tag */, const float* HWY_RESTRICT base,
                                   Vec256<int32_t> offset) {
  return Vec256<float>{_mm256_i32gather_ps(base, offset.raw, 1)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<float> GatherIndex(D /* tag */, const float* HWY_RESTRICT base,
                                  Vec256<int32_t> index) {
  return Vec256<float>{_mm256_i32gather_ps(base, index.raw, 4)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> GatherOffset(D /* tag */,
                                    const double* HWY_RESTRICT base,
                                    Vec256<int64_t> offset) {
  return Vec256<double>{_mm256_i64gather_pd(base, offset.raw, 1)};
}
template <class D, HWY_IF_V_SIZE_D(D, 32)>
HWY_API Vec256<double> GatherIndex(D /* tag */, const double* HWY_RESTRICT base,
                                   Vec256<int64_t> index) {
  return Vec256<double>{_mm256_i64gather_pd(base, index.raw, 8)};
}

HWY_DIAGNOSTICS(pop)

// ================================================== SWIZZLE

// ------------------------------ LowerHalf

template <class D, typename T = TFromD<D>, HWY_IF_NOT_FLOAT(T)>
HWY_API Vec128<T> LowerHalf(D /* tag */, Vec256<T> v) {
  return Vec128<T>{_mm256_castsi256_si128(v.raw)};
}
template <class D>
HWY_API Vec128<float> LowerHalf(D /* tag */, Vec256<float> v) {
  return Vec128<float>{_mm256_castps256_ps128(v.raw)};
}
template <class D>
HWY_API Vec128<double> LowerHalf(D /* tag */, Vec256<double> v) {
  return Vec128<double>{_mm256_castpd256_pd128(v.raw)};
}

template <typename T>
HWY_API Vec128<T> LowerHalf(Vec256<T> v) {
  const Full128<T> dh;
  return LowerHalf(dh, v);
}

// ------------------------------ UpperHalf

template <class D, typename T = TFromD<D>>
HWY_API Vec128<T> UpperHalf(D /* tag */, Vec256<T> v) {
  return Vec128<T>{_mm256_extracti128_si256(v.raw, 1)};
}
template <class D>
HWY_API Vec128<float> UpperHalf(D /* tag */, Vec256<float> v) {
  return Vec128<float>{_mm256_extractf128_ps(v.raw, 1)};
}
template <class D>
HWY_API Vec128<double> UpperHalf(D /* tag */, Vec256<double> v) {
  return Vec128<double>{_mm256_extractf128_pd(v.raw, 1)};
}

// ------------------------------ ExtractLane (Store)
template <typename T>
HWY_API T ExtractLane(const Vec256<T> v, size_t i) {
  const DFromV<decltype(v)> d;
  HWY_DASSERT(i < Lanes(d));
  alignas(32) T lanes[32 / sizeof(T)];
  Store(v, d, lanes);
  return lanes[i];
}

// ------------------------------ InsertLane (Store)
template <typename T>
HWY_API Vec256<T> InsertLane(const Vec256<T> v, size_t i, T t) {
  const DFromV<decltype(v)> d;
  HWY_DASSERT(i < Lanes(d));
  alignas(64) T lanes[64 / sizeof(T)];
  Store(v, d, lanes);
  lanes[i] = t;
  return Load(d, lanes);
}

// ------------------------------ GetLane (LowerHalf)
template <typename T>
HWY_API T GetLane(const Vec256<T> v) {
  return GetLane(LowerHalf(v));
}

// ------------------------------ ZeroExtendVector

// Unfortunately the initial _mm256_castsi128_si256 intrinsic leaves the upper
// bits undefined. Although it makes sense for them to be zero (VEX encoded
// 128-bit instructions zero the upper lanes to avoid large penalties), a
// compiler could decide to optimize out code that relies on this.
//
// The newer _mm256_zextsi128_si256 intrinsic fixes this by specifying the
// zeroing, but it is not available on MSVC until 15.7 nor GCC until 10.1. For
// older GCC, we can still obtain the desired code thanks to pattern
// recognition; note that the expensive insert instruction is not actually
// generated, see https://gcc.godbolt.org/z/1MKGaP.

#if !defined(HWY_HAVE_ZEXT)
#if (HWY_COMPILER_MSVC && HWY_COMPILER_MSVC >= 1915) ||  \
    (HWY_COMPILER_CLANG && HWY_COMPILER_CLANG >= 500) || \
    (HWY_COMPILER_GCC_ACTUAL && HWY_COMPILER_GCC_ACTUAL >= 1000)
#define HWY_HAVE_ZEXT 1
#else
#define HWY_HAVE_ZEXT 0
#endif
#endif  // defined(HWY_HAVE_ZEXT)

template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> ZeroExtendVector(D /* tag */, Vec128<T> lo) {
#if HWY_HAVE_ZEXT
  return Vec256<T>{_mm256_zextsi128_si256(lo.raw)};
#else
  return Vec256<T>{_mm256_inserti128_si256(_mm256_setzero_si256(), lo.raw, 0)};
#endif
}
template <class D>
HWY_API Vec256<float> ZeroExtendVector(D /* tag */, Vec128<float> lo) {
#if HWY_HAVE_ZEXT
  return Vec256<float>{_mm256_zextps128_ps256(lo.raw)};
#else
  return Vec256<float>{_mm256_insertf128_ps(_mm256_setzero_ps(), lo.raw, 0)};
#endif
}
template <class D>
HWY_API Vec256<double> ZeroExtendVector(D /* tag */, Vec128<double> lo) {
#if HWY_HAVE_ZEXT
  return Vec256<double>{_mm256_zextpd128_pd256(lo.raw)};
#else
  return Vec256<double>{_mm256_insertf128_pd(_mm256_setzero_pd(), lo.raw, 0)};
#endif
}

// ------------------------------ Combine

template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> Combine(D d, Vec128<T> hi, Vec128<T> lo) {
  const auto lo256 = ZeroExtendVector(d, lo);
  return Vec256<T>{_mm256_inserti128_si256(lo256.raw, hi.raw, 1)};
}
template <class D>
HWY_API Vec256<float> Combine(D d, Vec128<float> hi, Vec128<float> lo) {
  const auto lo256 = ZeroExtendVector(d, lo);
  return Vec256<float>{_mm256_insertf128_ps(lo256.raw, hi.raw, 1)};
}
template <class D>
HWY_API Vec256<double> Combine(D d, Vec128<double> hi, Vec128<double> lo) {
  const auto lo256 = ZeroExtendVector(d, lo);
  return Vec256<double>{_mm256_insertf128_pd(lo256.raw, hi.raw, 1)};
}

// ------------------------------ ShiftLeftBytes

template <int kBytes, class D, typename T = TFromD<D>>
HWY_API Vec256<T> ShiftLeftBytes(D /* tag */, const Vec256<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bslli_epi128.
  return Vec256<T>{_mm256_slli_si256(v.raw, kBytes)};
}

template <int kBytes, typename T>
HWY_API Vec256<T> ShiftLeftBytes(const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  return ShiftLeftBytes<kBytes>(d, v);
}

// ------------------------------ ShiftLeftLanes

template <int kLanes, class D, typename T = TFromD<D>>
HWY_API Vec256<T> ShiftLeftLanes(D d, const Vec256<T> v) {
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, ShiftLeftBytes<kLanes * sizeof(T)>(BitCast(d8, v)));
}

template <int kLanes, typename T>
HWY_API Vec256<T> ShiftLeftLanes(const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  return ShiftLeftLanes<kLanes>(d, v);
}

// ------------------------------ ShiftRightBytes
template <int kBytes, class D, typename T = TFromD<D>>
HWY_API Vec256<T> ShiftRightBytes(D /* tag */, const Vec256<T> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  // This is the same operation as _mm256_bsrli_epi128.
  return Vec256<T>{_mm256_srli_si256(v.raw, kBytes)};
}

// ------------------------------ ShiftRightLanes
template <int kLanes, class D, typename T = TFromD<D>>
HWY_API Vec256<T> ShiftRightLanes(D d, const Vec256<T> v) {
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, ShiftRightBytes<kLanes * sizeof(T)>(d8, BitCast(d8, v)));
}

// ------------------------------ CombineShiftRightBytes
template <int kBytes, class D, typename T = TFromD<D>>
HWY_API Vec256<T> CombineShiftRightBytes(D d, Vec256<T> hi, Vec256<T> lo) {
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Vec256<uint8_t>{_mm256_alignr_epi8(
                        BitCast(d8, hi).raw, BitCast(d8, lo).raw, kBytes)});
}

// ------------------------------ Broadcast

// Unsigned
template <int kLane>
HWY_API Vec256<uint16_t> Broadcast(const Vec256<uint16_t> v) {
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
HWY_API Vec256<uint32_t> Broadcast(const Vec256<uint32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x55 * kLane)};
}
template <int kLane>
HWY_API Vec256<uint64_t> Broadcast(const Vec256<uint64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<uint64_t>{_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44)};
}

// Signed
template <int kLane>
HWY_API Vec256<int16_t> Broadcast(const Vec256<int16_t> v) {
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
HWY_API Vec256<int32_t> Broadcast(const Vec256<int32_t> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x55 * kLane)};
}
template <int kLane>
HWY_API Vec256<int64_t> Broadcast(const Vec256<int64_t> v) {
  static_assert(0 <= kLane && kLane < 2, "Invalid lane");
  return Vec256<int64_t>{_mm256_shuffle_epi32(v.raw, kLane ? 0xEE : 0x44)};
}

// Float
template <int kLane>
HWY_API Vec256<float> Broadcast(Vec256<float> v) {
  static_assert(0 <= kLane && kLane < 4, "Invalid lane");
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x55 * kLane)};
}
template <int kLane>
HWY_API Vec256<double> Broadcast(const Vec256<double> v) {
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
template <typename T, HWY_IF_UI32(T)>
HWY_API Vec256<T> Shuffle2301(const Vec256<T> v) {
  return Vec256<T>{_mm256_shuffle_epi32(v.raw, 0xB1)};
}
HWY_API Vec256<float> Shuffle2301(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0xB1)};
}

// Used by generic_ops-inl.h
namespace detail {

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Shuffle2301(const Vec256<T> a, const Vec256<T> b) {
  const DFromV<decltype(a)> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int m = _MM_SHUFFLE(2, 3, 0, 1);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(BitCast(df, a).raw,
                                                    BitCast(df, b).raw, m)});
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Shuffle1230(const Vec256<T> a, const Vec256<T> b) {
  const DFromV<decltype(a)> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int m = _MM_SHUFFLE(1, 2, 3, 0);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(BitCast(df, a).raw,
                                                    BitCast(df, b).raw, m)});
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Shuffle3012(const Vec256<T> a, const Vec256<T> b) {
  const DFromV<decltype(a)> d;
  const RebindToFloat<decltype(d)> df;
  constexpr int m = _MM_SHUFFLE(3, 0, 1, 2);
  return BitCast(d, Vec256<float>{_mm256_shuffle_ps(BitCast(df, a).raw,
                                                    BitCast(df, b).raw, m)});
}

}  // namespace detail

// Swap 64-bit halves
HWY_API Vec256<uint32_t> Shuffle1032(const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_API Vec256<int32_t> Shuffle1032(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_API Vec256<float> Shuffle1032(const Vec256<float> v) {
  // Shorter encoding than _mm256_permute_ps.
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x4E)};
}
HWY_API Vec256<uint64_t> Shuffle01(const Vec256<uint64_t> v) {
  return Vec256<uint64_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_API Vec256<int64_t> Shuffle01(const Vec256<int64_t> v) {
  return Vec256<int64_t>{_mm256_shuffle_epi32(v.raw, 0x4E)};
}
HWY_API Vec256<double> Shuffle01(const Vec256<double> v) {
  // Shorter encoding than _mm256_permute_pd.
  return Vec256<double>{_mm256_shuffle_pd(v.raw, v.raw, 5)};
}

// Rotate right 32 bits
HWY_API Vec256<uint32_t> Shuffle0321(const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x39)};
}
HWY_API Vec256<int32_t> Shuffle0321(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x39)};
}
HWY_API Vec256<float> Shuffle0321(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x39)};
}
// Rotate left 32 bits
HWY_API Vec256<uint32_t> Shuffle2103(const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x93)};
}
HWY_API Vec256<int32_t> Shuffle2103(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x93)};
}
HWY_API Vec256<float> Shuffle2103(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x93)};
}

// Reverse
HWY_API Vec256<uint32_t> Shuffle0123(const Vec256<uint32_t> v) {
  return Vec256<uint32_t>{_mm256_shuffle_epi32(v.raw, 0x1B)};
}
HWY_API Vec256<int32_t> Shuffle0123(const Vec256<int32_t> v) {
  return Vec256<int32_t>{_mm256_shuffle_epi32(v.raw, 0x1B)};
}
HWY_API Vec256<float> Shuffle0123(const Vec256<float> v) {
  return Vec256<float>{_mm256_shuffle_ps(v.raw, v.raw, 0x1B)};
}

// ------------------------------ TableLookupLanes

// Returned by SetTableIndices/IndicesFromVec for use by TableLookupLanes.
template <typename T>
struct Indices256 {
  __m256i raw;
};

// Native 8x32 instruction: indices remain unchanged
template <class D, typename T = TFromD<D>, typename TI, HWY_IF_T_SIZE(T, 4)>
HWY_API Indices256<T> IndicesFromVec(D /* tag */, Vec256<TI> vec) {
  static_assert(sizeof(T) == sizeof(TI), "Index size must match lane");
#if HWY_IS_DEBUG_BUILD
  const Full256<TI> di;
  HWY_DASSERT(AllFalse(di, Lt(vec, Zero(di))) &&
              AllTrue(di, Lt(vec, Set(di, static_cast<TI>(32 / sizeof(T))))));
#endif
  return Indices256<T>{vec.raw};
}

// 64-bit lanes: convert indices to 8x32 unless AVX3 is available
template <class D, typename T = TFromD<D>, typename TI, HWY_IF_T_SIZE(T, 8)>
HWY_API Indices256<T> IndicesFromVec(D d, Vec256<TI> idx64) {
  static_assert(sizeof(T) == sizeof(TI), "Index size must match lane");
  const Rebind<TI, decltype(d)> di;
  (void)di;  // potentially unused
#if HWY_IS_DEBUG_BUILD
  HWY_DASSERT(AllFalse(di, Lt(idx64, Zero(di))) &&
              AllTrue(di, Lt(idx64, Set(di, static_cast<TI>(32 / sizeof(T))))));
#endif

#if HWY_TARGET <= HWY_AVX3
  (void)d;
  return Indices256<T>{idx64.raw};
#else
  const Repartition<float, decltype(d)> df;  // 32-bit!
  // Replicate 64-bit index into upper 32 bits
  const Vec256<TI> dup =
      BitCast(di, Vec256<float>{_mm256_moveldup_ps(BitCast(df, idx64).raw)});
  // For each idx64 i, idx32 are 2*i and 2*i+1.
  const Vec256<TI> idx32 = dup + dup + Set(di, TI(1) << 32);
  return Indices256<T>{idx32.raw};
#endif
}

template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>, typename TI>
HWY_API Indices256<T> SetTableIndices(D d, const TI* idx) {
  const Rebind<TI, decltype(d)> di;
  return IndicesFromVec(d, LoadU(di, idx));
}

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> TableLookupLanes(Vec256<T> v, Indices256<T> idx) {
  return Vec256<T>{_mm256_permutevar8x32_epi32(v.raw, idx.raw)};
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> TableLookupLanes(Vec256<T> v, Indices256<T> idx) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<T>{_mm256_permutexvar_epi64(idx.raw, v.raw)};
#else
  return Vec256<T>{_mm256_permutevar8x32_epi32(v.raw, idx.raw)};
#endif
}

HWY_API Vec256<float> TableLookupLanes(const Vec256<float> v,
                                       const Indices256<float> idx) {
  return Vec256<float>{_mm256_permutevar8x32_ps(v.raw, idx.raw)};
}

HWY_API Vec256<double> TableLookupLanes(const Vec256<double> v,
                                        const Indices256<double> idx) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<double>{_mm256_permutexvar_pd(idx.raw, v.raw)};
#else
  const Full256<double> df;
  const Full256<uint64_t> du;
  return BitCast(df, Vec256<uint64_t>{_mm256_permutevar8x32_epi32(
                         BitCast(du, v).raw, idx.raw)});
#endif
}

// ------------------------------ SwapAdjacentBlocks

template <typename T>
HWY_API Vec256<T> SwapAdjacentBlocks(Vec256<T> v) {
  return Vec256<T>{_mm256_permute2x128_si256(v.raw, v.raw, 0x01)};
}

HWY_API Vec256<float> SwapAdjacentBlocks(Vec256<float> v) {
  return Vec256<float>{_mm256_permute2f128_ps(v.raw, v.raw, 0x01)};
}

HWY_API Vec256<double> SwapAdjacentBlocks(Vec256<double> v) {
  return Vec256<double>{_mm256_permute2f128_pd(v.raw, v.raw, 0x01)};
}

// ------------------------------ Reverse (RotateRight)

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Reverse(D d, const Vec256<T> v) {
  alignas(32) static constexpr int32_t kReverse[8] = {7, 6, 5, 4, 3, 2, 1, 0};
  return TableLookupLanes(v, SetTableIndices(d, kReverse));
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> Reverse(D d, const Vec256<T> v) {
  alignas(32) static constexpr int64_t kReverse[4] = {3, 2, 1, 0};
  return TableLookupLanes(v, SetTableIndices(d, kReverse));
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> Reverse(D d, const Vec256<T> v) {
#if HWY_TARGET <= HWY_AVX3
  const RebindToSigned<decltype(d)> di;
  alignas(32) static constexpr int16_t kReverse[16] = {
      15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  const Vec256<int16_t> idx = Load(di, kReverse);
  return BitCast(d, Vec256<int16_t>{
                        _mm256_permutexvar_epi16(idx.raw, BitCast(di, v).raw)});
#else
  const RepartitionToWide<RebindToUnsigned<decltype(d)>> du32;
  const Vec256<uint32_t> rev32 = Reverse(du32, BitCast(du32, v));
  return BitCast(d, RotateRight<16>(rev32));
#endif
}

// ------------------------------ Reverse2 (in x86_128)

// ------------------------------ Reverse4 (SwapAdjacentBlocks)

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> Reverse4(D d, const Vec256<T> v) {
#if HWY_TARGET <= HWY_AVX3
  const RebindToSigned<decltype(d)> di;
  alignas(32) static constexpr int16_t kReverse4[16] = {
      3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};
  const Vec256<int16_t> idx = Load(di, kReverse4);
  return BitCast(d, Vec256<int16_t>{
                        _mm256_permutexvar_epi16(idx.raw, BitCast(di, v).raw)});
#else
  const RepartitionToWide<decltype(d)> dw;
  return Reverse2(d, BitCast(d, Shuffle2301(BitCast(dw, v))));
#endif
}

// 32 bit Reverse4 defined in x86_128.

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> Reverse4(D /* tag */, const Vec256<T> v) {
  // Could also use _mm256_permute4x64_epi64.
  return SwapAdjacentBlocks(Shuffle01(v));
}

// ------------------------------ Reverse8

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> Reverse8(D d, const Vec256<T> v) {
#if HWY_TARGET <= HWY_AVX3
  const RebindToSigned<decltype(d)> di;
  alignas(32) static constexpr int16_t kReverse8[16] = {
      7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8};
  const Vec256<int16_t> idx = Load(di, kReverse8);
  return BitCast(d, Vec256<int16_t>{
                        _mm256_permutexvar_epi16(idx.raw, BitCast(di, v).raw)});
#else
  const RepartitionToWide<decltype(d)> dw;
  return Reverse2(d, BitCast(d, Shuffle0123(BitCast(dw, v))));
#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Reverse8(D d, const Vec256<T> v) {
  return Reverse(d, v);
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> Reverse8(D /* tag */, const Vec256<T> /* v */) {
  HWY_ASSERT(0);  // AVX2 does not have 8 64-bit lanes
}

// ------------------------------ InterleaveLower

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use ZipLower/Upper instead (also works with scalar).

HWY_API Vec256<uint8_t> InterleaveLower(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_unpacklo_epi8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> InterleaveLower(Vec256<uint16_t> a,
                                         Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_unpacklo_epi16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> InterleaveLower(Vec256<uint32_t> a,
                                         Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_unpacklo_epi32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> InterleaveLower(Vec256<uint64_t> a,
                                         Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_unpacklo_epi64(a.raw, b.raw)};
}

HWY_API Vec256<int8_t> InterleaveLower(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_unpacklo_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> InterleaveLower(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_unpacklo_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> InterleaveLower(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_unpacklo_epi32(a.raw, b.raw)};
}
HWY_API Vec256<int64_t> InterleaveLower(Vec256<int64_t> a, Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_unpacklo_epi64(a.raw, b.raw)};
}

HWY_API Vec256<float> InterleaveLower(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_unpacklo_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> InterleaveLower(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_unpacklo_pd(a.raw, b.raw)};
}

// ------------------------------ InterleaveUpper

// All functions inside detail lack the required D parameter.
namespace detail {

HWY_API Vec256<uint8_t> InterleaveUpper(Vec256<uint8_t> a, Vec256<uint8_t> b) {
  return Vec256<uint8_t>{_mm256_unpackhi_epi8(a.raw, b.raw)};
}
HWY_API Vec256<uint16_t> InterleaveUpper(Vec256<uint16_t> a,
                                         Vec256<uint16_t> b) {
  return Vec256<uint16_t>{_mm256_unpackhi_epi16(a.raw, b.raw)};
}
HWY_API Vec256<uint32_t> InterleaveUpper(Vec256<uint32_t> a,
                                         Vec256<uint32_t> b) {
  return Vec256<uint32_t>{_mm256_unpackhi_epi32(a.raw, b.raw)};
}
HWY_API Vec256<uint64_t> InterleaveUpper(Vec256<uint64_t> a,
                                         Vec256<uint64_t> b) {
  return Vec256<uint64_t>{_mm256_unpackhi_epi64(a.raw, b.raw)};
}

HWY_API Vec256<int8_t> InterleaveUpper(Vec256<int8_t> a, Vec256<int8_t> b) {
  return Vec256<int8_t>{_mm256_unpackhi_epi8(a.raw, b.raw)};
}
HWY_API Vec256<int16_t> InterleaveUpper(Vec256<int16_t> a, Vec256<int16_t> b) {
  return Vec256<int16_t>{_mm256_unpackhi_epi16(a.raw, b.raw)};
}
HWY_API Vec256<int32_t> InterleaveUpper(Vec256<int32_t> a, Vec256<int32_t> b) {
  return Vec256<int32_t>{_mm256_unpackhi_epi32(a.raw, b.raw)};
}
HWY_API Vec256<int64_t> InterleaveUpper(Vec256<int64_t> a, Vec256<int64_t> b) {
  return Vec256<int64_t>{_mm256_unpackhi_epi64(a.raw, b.raw)};
}

HWY_API Vec256<float> InterleaveUpper(Vec256<float> a, Vec256<float> b) {
  return Vec256<float>{_mm256_unpackhi_ps(a.raw, b.raw)};
}
HWY_API Vec256<double> InterleaveUpper(Vec256<double> a, Vec256<double> b) {
  return Vec256<double>{_mm256_unpackhi_pd(a.raw, b.raw)};
}

}  // namespace detail

template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> InterleaveUpper(D /* tag */, Vec256<T> a, Vec256<T> b) {
  return detail::InterleaveUpper(a, b);
}

// ------------------------------ ZipLower/ZipUpper (InterleaveLower)

// Same as Interleave*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.
template <typename T, typename TW = MakeWide<T>>
HWY_API Vec256<TW> ZipLower(Vec256<T> a, Vec256<T> b) {
  const Full256<TW> dw;
  return BitCast(dw, InterleaveLower(a, b));
}
template <class DW, typename TN = MakeNarrow<TFromD<DW>>>
HWY_API VFromD<DW> ZipLower(DW dw, Vec256<TN> a, Vec256<TN> b) {
  return BitCast(dw, InterleaveLower(a, b));
}

template <class DW, typename TN = MakeNarrow<TFromD<DW>>>
HWY_API VFromD<DW> ZipUpper(DW dw, Vec256<TN> a, Vec256<TN> b) {
  const RepartitionToNarrow<decltype(dw)> dn;
  return BitCast(dw, InterleaveUpper(dn, a, b));
}

// ------------------------------ Blocks (LowerHalf, ZeroExtendVector)

// _mm256_broadcastsi128_si256 has 7 cycle latency on ICL.
// _mm256_permute2x128_si256 is slow on Zen1 (8 uops), so we avoid it (at no
// extra cost) for LowerLower and UpperLower.

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> ConcatLowerLower(D d, Vec256<T> hi, Vec256<T> lo) {
  const Half<decltype(d)> d2;
  return Vec256<T>{_mm256_inserti128_si256(lo.raw, LowerHalf(d2, hi).raw, 1)};
}
template <class D>
HWY_API Vec256<float> ConcatLowerLower(D d, Vec256<float> hi,
                                       Vec256<float> lo) {
  const Half<decltype(d)> d2;
  return Vec256<float>{_mm256_insertf128_ps(lo.raw, LowerHalf(d2, hi).raw, 1)};
}
template <class D>
HWY_API Vec256<double> ConcatLowerLower(D d, Vec256<double> hi,
                                        Vec256<double> lo) {
  const Half<decltype(d)> d2;
  return Vec256<double>{_mm256_insertf128_pd(lo.raw, LowerHalf(d2, hi).raw, 1)};
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves / swap blocks)
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> ConcatLowerUpper(D /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return Vec256<T>{_mm256_permute2x128_si256(lo.raw, hi.raw, 0x21)};
}
template <class D>
HWY_API Vec256<float> ConcatLowerUpper(D /* tag */, Vec256<float> hi,
                                       Vec256<float> lo) {
  return Vec256<float>{_mm256_permute2f128_ps(lo.raw, hi.raw, 0x21)};
}
template <class D>
HWY_API Vec256<double> ConcatLowerUpper(D /* tag */, Vec256<double> hi,
                                        Vec256<double> lo) {
  return Vec256<double>{_mm256_permute2f128_pd(lo.raw, hi.raw, 0x21)};
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> ConcatUpperLower(D /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return Vec256<T>{_mm256_blend_epi32(hi.raw, lo.raw, 0x0F)};
}
template <class D>
HWY_API Vec256<float> ConcatUpperLower(D /* tag */, Vec256<float> hi,
                                       Vec256<float> lo) {
  return Vec256<float>{_mm256_blend_ps(hi.raw, lo.raw, 0x0F)};
}
template <class D>
HWY_API Vec256<double> ConcatUpperLower(D /* tag */, Vec256<double> hi,
                                        Vec256<double> lo) {
  return Vec256<double>{_mm256_blend_pd(hi.raw, lo.raw, 3)};
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> ConcatUpperUpper(D /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return Vec256<T>{_mm256_permute2x128_si256(lo.raw, hi.raw, 0x31)};
}
template <class D>
HWY_API Vec256<float> ConcatUpperUpper(D /* tag */, Vec256<float> hi,
                                       Vec256<float> lo) {
  return Vec256<float>{_mm256_permute2f128_ps(lo.raw, hi.raw, 0x31)};
}
template <class D>
HWY_API Vec256<double> ConcatUpperUpper(D /* tag */, Vec256<double> hi,
                                        Vec256<double> lo) {
  return Vec256<double>{_mm256_permute2f128_pd(lo.raw, hi.raw, 0x31)};
}

// ------------------------------ ConcatOdd

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> ConcatOdd(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3_DL
  alignas(32) static constexpr uint8_t kIdx[32] = {
      1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
      33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63};
  return BitCast(
      d, Vec256<uint16_t>{_mm256_permutex2var_epi8(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RepartitionToWide<decltype(du)> dw;
  // Unsigned 8-bit shift so we can pack.
  const Vec256<uint16_t> uH = ShiftRight<8>(BitCast(dw, hi));
  const Vec256<uint16_t> uL = ShiftRight<8>(BitCast(dw, lo));
  const __m256i u8 = _mm256_packus_epi16(uL.raw, uH.raw);
  return Vec256<T>{_mm256_permute4x64_epi64(u8, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> ConcatOdd(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(32) static constexpr uint16_t kIdx[16] = {
      1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
  return BitCast(
      d, Vec256<uint16_t>{_mm256_permutex2var_epi16(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RepartitionToWide<decltype(du)> dw;
  // Unsigned 16-bit shift so we can pack.
  const Vec256<uint32_t> uH = ShiftRight<16>(BitCast(dw, hi));
  const Vec256<uint32_t> uL = ShiftRight<16>(BitCast(dw, lo));
  const __m256i u16 = _mm256_packus_epi32(uL.raw, uH.raw);
  return Vec256<T>{_mm256_permute4x64_epi64(u16, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> ConcatOdd(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(32) static constexpr uint32_t kIdx[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  return BitCast(
      d, Vec256<uint32_t>{_mm256_permutex2var_epi32(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RebindToFloat<decltype(d)> df;
  const Vec256<float> v3131{_mm256_shuffle_ps(
      BitCast(df, lo).raw, BitCast(df, hi).raw, _MM_SHUFFLE(3, 1, 3, 1))};
  return Vec256<T>{_mm256_permute4x64_epi64(BitCast(du, v3131).raw,
                                            _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

template <class D>
HWY_API Vec256<float> ConcatOdd(D d, Vec256<float> hi, Vec256<float> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(32) static constexpr uint32_t kIdx[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  return Vec256<float>{
      _mm256_permutex2var_ps(lo.raw, Load(du, kIdx).raw, hi.raw)};
#else
  const Vec256<float> v3131{
      _mm256_shuffle_ps(lo.raw, hi.raw, _MM_SHUFFLE(3, 1, 3, 1))};
  return BitCast(d, Vec256<uint32_t>{_mm256_permute4x64_epi64(
                        BitCast(du, v3131).raw, _MM_SHUFFLE(3, 1, 2, 0))});
#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> ConcatOdd(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(64) static constexpr uint64_t kIdx[4] = {1, 3, 5, 7};
  return BitCast(
      d, Vec256<uint64_t>{_mm256_permutex2var_epi64(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RebindToFloat<decltype(d)> df;
  const Vec256<double> v31{
      _mm256_shuffle_pd(BitCast(df, lo).raw, BitCast(df, hi).raw, 15)};
  return Vec256<T>{
      _mm256_permute4x64_epi64(BitCast(du, v31).raw, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

template <class D>
HWY_API Vec256<double> ConcatOdd(D d, Vec256<double> hi, Vec256<double> lo) {
#if HWY_TARGET <= HWY_AVX3
  const RebindToUnsigned<decltype(d)> du;
  alignas(64) static constexpr uint64_t kIdx[4] = {1, 3, 5, 7};
  return Vec256<double>{
      _mm256_permutex2var_pd(lo.raw, Load(du, kIdx).raw, hi.raw)};
#else
  (void)d;
  const Vec256<double> v31{_mm256_shuffle_pd(lo.raw, hi.raw, 15)};
  return Vec256<double>{
      _mm256_permute4x64_pd(v31.raw, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

// ------------------------------ ConcatEven

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> ConcatEven(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3_DL
  alignas(64) static constexpr uint8_t kIdx[32] = {
      0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
      32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62};
  return BitCast(
      d, Vec256<uint32_t>{_mm256_permutex2var_epi8(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RepartitionToWide<decltype(du)> dw;
  // Isolate lower 8 bits per u16 so we can pack.
  const Vec256<uint16_t> mask = Set(dw, 0x00FF);
  const Vec256<uint16_t> uH = And(BitCast(dw, hi), mask);
  const Vec256<uint16_t> uL = And(BitCast(dw, lo), mask);
  const __m256i u8 = _mm256_packus_epi16(uL.raw, uH.raw);
  return Vec256<T>{_mm256_permute4x64_epi64(u8, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> ConcatEven(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(64) static constexpr uint16_t kIdx[16] = {
      0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
  return BitCast(
      d, Vec256<uint32_t>{_mm256_permutex2var_epi16(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RepartitionToWide<decltype(du)> dw;
  // Isolate lower 16 bits per u32 so we can pack.
  const Vec256<uint32_t> mask = Set(dw, 0x0000FFFF);
  const Vec256<uint32_t> uH = And(BitCast(dw, hi), mask);
  const Vec256<uint32_t> uL = And(BitCast(dw, lo), mask);
  const __m256i u16 = _mm256_packus_epi32(uL.raw, uH.raw);
  return Vec256<T>{_mm256_permute4x64_epi64(u16, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> ConcatEven(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(64) static constexpr uint32_t kIdx[8] = {0, 2, 4, 6, 8, 10, 12, 14};
  return BitCast(
      d, Vec256<uint32_t>{_mm256_permutex2var_epi32(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RebindToFloat<decltype(d)> df;
  const Vec256<float> v2020{_mm256_shuffle_ps(
      BitCast(df, lo).raw, BitCast(df, hi).raw, _MM_SHUFFLE(2, 0, 2, 0))};
  return Vec256<T>{_mm256_permute4x64_epi64(BitCast(du, v2020).raw,
                                            _MM_SHUFFLE(3, 1, 2, 0))};

#endif
}

template <class D>
HWY_API Vec256<float> ConcatEven(D d, Vec256<float> hi, Vec256<float> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(64) static constexpr uint32_t kIdx[8] = {0, 2, 4, 6, 8, 10, 12, 14};
  return Vec256<float>{
      _mm256_permutex2var_ps(lo.raw, Load(du, kIdx).raw, hi.raw)};
#else
  const Vec256<float> v2020{
      _mm256_shuffle_ps(lo.raw, hi.raw, _MM_SHUFFLE(2, 0, 2, 0))};
  return BitCast(d, Vec256<uint32_t>{_mm256_permute4x64_epi64(
                        BitCast(du, v2020).raw, _MM_SHUFFLE(3, 1, 2, 0))});

#endif
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> ConcatEven(D d, Vec256<T> hi, Vec256<T> lo) {
  const RebindToUnsigned<decltype(d)> du;
#if HWY_TARGET <= HWY_AVX3
  alignas(64) static constexpr uint64_t kIdx[4] = {0, 2, 4, 6};
  return BitCast(
      d, Vec256<uint64_t>{_mm256_permutex2var_epi64(
             BitCast(du, lo).raw, Load(du, kIdx).raw, BitCast(du, hi).raw)});
#else
  const RebindToFloat<decltype(d)> df;
  const Vec256<double> v20{
      _mm256_shuffle_pd(BitCast(df, lo).raw, BitCast(df, hi).raw, 0)};
  return Vec256<T>{
      _mm256_permute4x64_epi64(BitCast(du, v20).raw, _MM_SHUFFLE(3, 1, 2, 0))};

#endif
}

template <class D>
HWY_API Vec256<double> ConcatEven(D d, Vec256<double> hi, Vec256<double> lo) {
#if HWY_TARGET <= HWY_AVX3
  const RebindToUnsigned<decltype(d)> du;
  alignas(64) static constexpr uint64_t kIdx[4] = {0, 2, 4, 6};
  return Vec256<double>{
      _mm256_permutex2var_pd(lo.raw, Load(du, kIdx).raw, hi.raw)};
#else
  (void)d;
  const Vec256<double> v20{_mm256_shuffle_pd(lo.raw, hi.raw, 0)};
  return Vec256<double>{
      _mm256_permute4x64_pd(v20.raw, _MM_SHUFFLE(3, 1, 2, 0))};
#endif
}

// ------------------------------ DupEven (InterleaveLower)

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> DupEven(Vec256<T> v) {
  return Vec256<T>{_mm256_shuffle_epi32(v.raw, _MM_SHUFFLE(2, 2, 0, 0))};
}
HWY_API Vec256<float> DupEven(Vec256<float> v) {
  return Vec256<float>{
      _mm256_shuffle_ps(v.raw, v.raw, _MM_SHUFFLE(2, 2, 0, 0))};
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> DupEven(const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  return InterleaveLower(d, v, v);
}

// ------------------------------ DupOdd (InterleaveUpper)

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> DupOdd(Vec256<T> v) {
  return Vec256<T>{_mm256_shuffle_epi32(v.raw, _MM_SHUFFLE(3, 3, 1, 1))};
}
HWY_API Vec256<float> DupOdd(Vec256<float> v) {
  return Vec256<float>{
      _mm256_shuffle_ps(v.raw, v.raw, _MM_SHUFFLE(3, 3, 1, 1))};
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> DupOdd(const Vec256<T> v) {
  const DFromV<decltype(v)> d;
  return InterleaveUpper(d, v, v);
}

// ------------------------------ OddEven

namespace detail {

template <typename T>
HWY_INLINE Vec256<T> OddEven(hwy::SizeTag<1> /* tag */, const Vec256<T> a,
                             const Vec256<T> b) {
  const DFromV<decltype(a)> d;
  const Full256<uint8_t> d8;
  alignas(32) static constexpr uint8_t mask[16] = {
      0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return IfThenElse(MaskFromVec(BitCast(d, LoadDup128(d8, mask))), b, a);
}
template <typename T>
HWY_INLINE Vec256<T> OddEven(hwy::SizeTag<2> /* tag */, const Vec256<T> a,
                             const Vec256<T> b) {
  return Vec256<T>{_mm256_blend_epi16(a.raw, b.raw, 0x55)};
}
template <typename T>
HWY_INLINE Vec256<T> OddEven(hwy::SizeTag<4> /* tag */, const Vec256<T> a,
                             const Vec256<T> b) {
  return Vec256<T>{_mm256_blend_epi32(a.raw, b.raw, 0x55)};
}
template <typename T>
HWY_INLINE Vec256<T> OddEven(hwy::SizeTag<8> /* tag */, const Vec256<T> a,
                             const Vec256<T> b) {
  return Vec256<T>{_mm256_blend_epi32(a.raw, b.raw, 0x33)};
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> OddEven(const Vec256<T> a, const Vec256<T> b) {
  return detail::OddEven(hwy::SizeTag<sizeof(T)>(), a, b);
}
HWY_API Vec256<float> OddEven(const Vec256<float> a, const Vec256<float> b) {
  return Vec256<float>{_mm256_blend_ps(a.raw, b.raw, 0x55)};
}

HWY_API Vec256<double> OddEven(const Vec256<double> a, const Vec256<double> b) {
  return Vec256<double>{_mm256_blend_pd(a.raw, b.raw, 5)};
}

// ------------------------------ OddEvenBlocks

template <typename T>
Vec256<T> OddEvenBlocks(Vec256<T> odd, Vec256<T> even) {
  return Vec256<T>{_mm256_blend_epi32(odd.raw, even.raw, 0xFu)};
}

HWY_API Vec256<float> OddEvenBlocks(Vec256<float> odd, Vec256<float> even) {
  return Vec256<float>{_mm256_blend_ps(odd.raw, even.raw, 0xFu)};
}

HWY_API Vec256<double> OddEvenBlocks(Vec256<double> odd, Vec256<double> even) {
  return Vec256<double>{_mm256_blend_pd(odd.raw, even.raw, 0x3u)};
}

// ------------------------------ ReverseBlocks (ConcatLowerUpper)

template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> ReverseBlocks(D d, Vec256<T> v) {
  return ConcatLowerUpper(d, v, v);
}

// ------------------------------ TableLookupBytes (ZeroExtendVector)

// Both full
template <typename T, typename TI>
HWY_API Vec256<TI> TableLookupBytes(Vec256<T> bytes, Vec256<TI> from) {
  return Vec256<TI>{_mm256_shuffle_epi8(bytes.raw, from.raw)};
}

// Partial index vector
template <typename T, typename TI, size_t NI>
HWY_API Vec128<TI, NI> TableLookupBytes(Vec256<T> bytes, Vec128<TI, NI> from) {
  const Full256<TI> di;
  const Half<decltype(di)> dih;
  // First expand to full 128, then 256.
  const auto from_256 = ZeroExtendVector(di, Vec128<TI>{from.raw});
  const auto tbl_full = TableLookupBytes(bytes, from_256);
  // Shrink to 128, then partial.
  return Vec128<TI, NI>{LowerHalf(dih, tbl_full).raw};
}

// Partial table vector
template <typename T, size_t N, typename TI>
HWY_API Vec256<TI> TableLookupBytes(Vec128<T, N> bytes, Vec256<TI> from) {
  const Full256<T> d;
  // First expand to full 128, then 256.
  const auto bytes_256 = ZeroExtendVector(d, Vec128<T>{bytes.raw});
  return TableLookupBytes(bytes_256, from);
}

// Partial both are handled by x86_128.

// ------------------------------ Shl (Mul, ZipLower)

namespace detail {

#if HWY_TARGET > HWY_AVX3 && !HWY_IDE  // AVX2 or older

// Returns 2^v for use as per-lane multipliers to emulate 16-bit shifts.
template <typename T>
HWY_INLINE Vec256<MakeUnsigned<T>> Pow2(const Vec256<T> v) {
  static_assert(sizeof(T) == 2, "Only for 16-bit");
  const DFromV<decltype(v)> d;
  const RepartitionToWide<decltype(d)> dw;
  const Rebind<float, decltype(dw)> df;
  const auto zero = Zero(d);
  // Move into exponent (this u16 will become the upper half of an f32)
  const auto exp = ShiftLeft<23 - 16>(v);
  const auto upper = exp + Set(d, 0x3F80);  // upper half of 1.0f
  // Insert 0 into lower halves for reinterpreting as binary32.
  const auto f0 = ZipLower(dw, zero, upper);
  const auto f1 = ZipUpper(dw, zero, upper);
  // Do not use ConvertTo because it checks for overflow, which is redundant
  // because we only care about v in [0, 16).
  const Vec256<int32_t> bits0{_mm256_cvttps_epi32(BitCast(df, f0).raw)};
  const Vec256<int32_t> bits1{_mm256_cvttps_epi32(BitCast(df, f1).raw)};
  return Vec256<MakeUnsigned<T>>{_mm256_packus_epi32(bits0.raw, bits1.raw)};
}

#endif  // HWY_TARGET > HWY_AVX3

HWY_INLINE Vec256<uint16_t> Shl(hwy::UnsignedTag /*tag*/, Vec256<uint16_t> v,
                                Vec256<uint16_t> bits) {
#if HWY_TARGET <= HWY_AVX3 || HWY_IDE
  return Vec256<uint16_t>{_mm256_sllv_epi16(v.raw, bits.raw)};
#else
  return v * Pow2(bits);
#endif
}

HWY_INLINE Vec256<uint32_t> Shl(hwy::UnsignedTag /*tag*/, Vec256<uint32_t> v,
                                Vec256<uint32_t> bits) {
  return Vec256<uint32_t>{_mm256_sllv_epi32(v.raw, bits.raw)};
}

HWY_INLINE Vec256<uint64_t> Shl(hwy::UnsignedTag /*tag*/, Vec256<uint64_t> v,
                                Vec256<uint64_t> bits) {
  return Vec256<uint64_t>{_mm256_sllv_epi64(v.raw, bits.raw)};
}

template <typename T>
HWY_INLINE Vec256<T> Shl(hwy::SignedTag /*tag*/, Vec256<T> v, Vec256<T> bits) {
  // Signed left shifts are the same as unsigned.
  const Full256<T> di;
  const Full256<MakeUnsigned<T>> du;
  return BitCast(di,
                 Shl(hwy::UnsignedTag(), BitCast(du, v), BitCast(du, bits)));
}

}  // namespace detail

template <typename T>
HWY_API Vec256<T> operator<<(Vec256<T> v, Vec256<T> bits) {
  return detail::Shl(hwy::TypeTag<T>(), v, bits);
}

// ------------------------------ Shr (MulHigh, IfThenElse, Not)

HWY_API Vec256<uint16_t> operator>>(Vec256<uint16_t> v, Vec256<uint16_t> bits) {
#if HWY_TARGET <= HWY_AVX3 || HWY_IDE
  return Vec256<uint16_t>{_mm256_srlv_epi16(v.raw, bits.raw)};
#else
  Full256<uint16_t> d;
  // For bits=0, we cannot mul by 2^16, so fix the result later.
  auto out = MulHigh(v, detail::Pow2(Set(d, 16) - bits));
  // Replace output with input where bits == 0.
  return IfThenElse(bits == Zero(d), v, out);
#endif
}

HWY_API Vec256<uint32_t> operator>>(Vec256<uint32_t> v, Vec256<uint32_t> bits) {
  return Vec256<uint32_t>{_mm256_srlv_epi32(v.raw, bits.raw)};
}

HWY_API Vec256<uint64_t> operator>>(Vec256<uint64_t> v, Vec256<uint64_t> bits) {
  return Vec256<uint64_t>{_mm256_srlv_epi64(v.raw, bits.raw)};
}

HWY_API Vec256<int16_t> operator>>(Vec256<int16_t> v, Vec256<int16_t> bits) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<int16_t>{_mm256_srav_epi16(v.raw, bits.raw)};
#else
  const DFromV<decltype(v)> d;
  return detail::SignedShr(d, v, bits);
#endif
}

HWY_API Vec256<int32_t> operator>>(Vec256<int32_t> v, Vec256<int32_t> bits) {
  return Vec256<int32_t>{_mm256_srav_epi32(v.raw, bits.raw)};
}

HWY_API Vec256<int64_t> operator>>(Vec256<int64_t> v, Vec256<int64_t> bits) {
#if HWY_TARGET <= HWY_AVX3
  return Vec256<int64_t>{_mm256_srav_epi64(v.raw, bits.raw)};
#else
  const DFromV<decltype(v)> d;
  return detail::SignedShr(d, v, bits);
#endif
}

HWY_INLINE Vec256<uint64_t> MulEven(const Vec256<uint64_t> a,
                                    const Vec256<uint64_t> b) {
  const Full256<uint64_t> du64;
  const RepartitionToNarrow<decltype(du64)> du32;
  const auto maskL = Set(du64, 0xFFFFFFFFULL);
  const auto a32 = BitCast(du32, a);
  const auto b32 = BitCast(du32, b);
  // Inputs for MulEven: we only need the lower 32 bits
  const auto aH = Shuffle2301(a32);
  const auto bH = Shuffle2301(b32);

  // Knuth double-word multiplication. We use 32x32 = 64 MulEven and only need
  // the even (lower 64 bits of every 128-bit block) results. See
  // https://github.com/hcs0/Hackers-Delight/blob/master/muldwu.c.tat
  const auto aLbL = MulEven(a32, b32);
  const auto w3 = aLbL & maskL;

  const auto t2 = MulEven(aH, b32) + ShiftRight<32>(aLbL);
  const auto w2 = t2 & maskL;
  const auto w1 = ShiftRight<32>(t2);

  const auto t = MulEven(a32, bH) + w2;
  const auto k = ShiftRight<32>(t);

  const auto mulH = MulEven(aH, bH) + w1 + k;
  const auto mulL = ShiftLeft<32>(t) + w3;
  return InterleaveLower(mulL, mulH);
}

HWY_INLINE Vec256<uint64_t> MulOdd(const Vec256<uint64_t> a,
                                   const Vec256<uint64_t> b) {
  const Full256<uint64_t> du64;
  const RepartitionToNarrow<decltype(du64)> du32;
  const auto maskL = Set(du64, 0xFFFFFFFFULL);
  const auto a32 = BitCast(du32, a);
  const auto b32 = BitCast(du32, b);
  // Inputs for MulEven: we only need bits [95:64] (= upper half of input)
  const auto aH = Shuffle2301(a32);
  const auto bH = Shuffle2301(b32);

  // Same as above, but we're using the odd results (upper 64 bits per block).
  const auto aLbL = MulEven(a32, b32);
  const auto w3 = aLbL & maskL;

  const auto t2 = MulEven(aH, b32) + ShiftRight<32>(aLbL);
  const auto w2 = t2 & maskL;
  const auto w1 = ShiftRight<32>(t2);

  const auto t = MulEven(a32, bH) + w2;
  const auto k = ShiftRight<32>(t);

  const auto mulH = MulEven(aH, bH) + w1 + k;
  const auto mulL = ShiftLeft<32>(t) + w3;
  return InterleaveUpper(du64, mulL, mulH);
}

// ------------------------------ ReorderWidenMulAccumulate
template <class D, HWY_IF_SIGNED_D(D)>
HWY_API Vec256<int32_t> ReorderWidenMulAccumulate(D /*d32*/, Vec256<int16_t> a,
                                                  Vec256<int16_t> b,
                                                  const Vec256<int32_t> sum0,
                                                  Vec256<int32_t>& /*sum1*/) {
  return sum0 + Vec256<int32_t>{_mm256_madd_epi16(a.raw, b.raw)};
}

// ------------------------------ RearrangeToOddPlusEven
HWY_API Vec256<int32_t> RearrangeToOddPlusEven(const Vec256<int32_t> sum0,
                                               Vec256<int32_t> /*sum1*/) {
  return sum0;  // invariant already holds
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

template <class D, HWY_IF_F64_D(D)>
HWY_API Vec256<double> PromoteTo(D /* tag */, Vec128<float> v) {
  return Vec256<double>{_mm256_cvtps_pd(v.raw)};
}

template <class D, HWY_IF_F64_D(D)>
HWY_API Vec256<double> PromoteTo(D /* tag */, Vec128<int32_t> v) {
  return Vec256<double>{_mm256_cvtepi32_pd(v.raw)};
}

// Unsigned: zero-extend.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then Zip* would be faster.
template <class D, HWY_IF_U16_D(D)>
HWY_API Vec256<uint16_t> PromoteTo(D /* tag */, Vec128<uint8_t> v) {
  return Vec256<uint16_t>{_mm256_cvtepu8_epi16(v.raw)};
}
template <class D, HWY_IF_U32_D(D)>
HWY_API Vec256<uint32_t> PromoteTo(D /* tag */, Vec128<uint8_t, 8> v) {
  return Vec256<uint32_t>{_mm256_cvtepu8_epi32(v.raw)};
}
template <class D, HWY_IF_I16_D(D)>
HWY_API Vec256<int16_t> PromoteTo(D /* tag */, Vec128<uint8_t> v) {
  return Vec256<int16_t>{_mm256_cvtepu8_epi16(v.raw)};
}
template <class D, HWY_IF_I32_D(D)>
HWY_API Vec256<int32_t> PromoteTo(D /* tag */, Vec128<uint8_t, 8> v) {
  return Vec256<int32_t>{_mm256_cvtepu8_epi32(v.raw)};
}
template <class D, HWY_IF_U32_D(D)>
HWY_API Vec256<uint32_t> PromoteTo(D /* tag */, Vec128<uint16_t> v) {
  return Vec256<uint32_t>{_mm256_cvtepu16_epi32(v.raw)};
}
template <class D, HWY_IF_I32_D(D)>
HWY_API Vec256<int32_t> PromoteTo(D /* tag */, Vec128<uint16_t> v) {
  return Vec256<int32_t>{_mm256_cvtepu16_epi32(v.raw)};
}
template <class D, HWY_IF_U64_D(D)>
HWY_API Vec256<uint64_t> PromoteTo(D /* tag */, Vec128<uint32_t> v) {
  return Vec256<uint64_t>{_mm256_cvtepu32_epi64(v.raw)};
}

// Signed: replicate sign bit.
// Note: these have 3 cycle latency; if inputs are already split across the
// 128 bit blocks (in their upper/lower halves), then ZipUpper/lo followed by
// signed shift would be faster.
template <class D, HWY_IF_I16_D(D)>
HWY_API Vec256<int16_t> PromoteTo(D /* tag */, Vec128<int8_t> v) {
  return Vec256<int16_t>{_mm256_cvtepi8_epi16(v.raw)};
}
template <class D, HWY_IF_I32_D(D)>
HWY_API Vec256<int32_t> PromoteTo(D /* tag */, Vec128<int8_t, 8> v) {
  return Vec256<int32_t>{_mm256_cvtepi8_epi32(v.raw)};
}
template <class D, HWY_IF_I32_D(D)>
HWY_API Vec256<int32_t> PromoteTo(D /* tag */, Vec128<int16_t> v) {
  return Vec256<int32_t>{_mm256_cvtepi16_epi32(v.raw)};
}
template <class D, HWY_IF_I64_D(D)>
HWY_API Vec256<int64_t> PromoteTo(D /* tag */, Vec128<int32_t> v) {
  return Vec256<int64_t>{_mm256_cvtepi32_epi64(v.raw)};
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <class D, HWY_IF_U16_D(D)>
HWY_API Vec128<uint16_t> DemoteTo(D /* tag */, Vec256<int32_t> v) {
  const __m256i u16 = _mm256_packus_epi32(v.raw, v.raw);
  // Concatenating lower halves of both 128-bit blocks afterward is more
  // efficient than an extra input with low block = high block of v.
  return Vec128<uint16_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u16, 0x88))};
}

template <class D, HWY_IF_U16_D(D)>
HWY_API Vec128<uint16_t> DemoteTo(D dn, Vec256<uint32_t> v) {
  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;
  return DemoteTo(dn, BitCast(di, Min(v, Set(d, 0x7FFFFFFFu))));
}

template <class D, HWY_IF_I16_D(D)>
HWY_API Vec128<int16_t> DemoteTo(D /* tag */, Vec256<int32_t> v) {
  const __m256i i16 = _mm256_packs_epi32(v.raw, v.raw);
  return Vec128<int16_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i16, 0x88))};
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec64<uint8_t> DemoteTo(D /* tag */, Vec256<int32_t> v) {
  const __m256i i16_blocks = _mm256_packs_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i i16_concat = _mm256_permute4x64_epi64(i16_blocks, 0x88);
  const __m128i i16 = _mm256_castsi256_si128(i16_concat);
  return Vec64<uint8_t>{_mm_packus_epi16(i16, i16)};
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec64<uint8_t> DemoteTo(D dn, Vec256<uint32_t> v) {
#if HWY_TARGET <= HWY_AVX3
  (void)dn;
  return Vec64<uint8_t>{_mm256_cvtusepi32_epi8(v.raw)};
#else
  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;
  return DemoteTo(dn, BitCast(di, Min(v, Set(d, 0x7FFFFFFFu))));
#endif
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec128<uint8_t> DemoteTo(D /* tag */, Vec256<int16_t> v) {
  const __m256i u8 = _mm256_packus_epi16(v.raw, v.raw);
  return Vec128<uint8_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(u8, 0x88))};
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec128<uint8_t> DemoteTo(D dn, Vec256<uint16_t> v) {
  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;
  return DemoteTo(dn, BitCast(di, Min(v, Set(d, 0x7FFFu))));
}

template <class D, HWY_IF_I8_D(D)>
HWY_API Vec64<int8_t> DemoteTo(D /* tag */, Vec256<int32_t> v) {
  const __m256i i16_blocks = _mm256_packs_epi32(v.raw, v.raw);
  // Concatenate lower 64 bits of each 128-bit block
  const __m256i i16_concat = _mm256_permute4x64_epi64(i16_blocks, 0x88);
  const __m128i i16 = _mm256_castsi256_si128(i16_concat);
  return Vec128<int8_t, 8>{_mm_packs_epi16(i16, i16)};
}

template <class D, HWY_IF_I8_D(D)>
HWY_API Vec128<int8_t> DemoteTo(D /* tag */, Vec256<int16_t> v) {
  const __m256i i8 = _mm256_packs_epi16(v.raw, v.raw);
  return Vec128<int8_t>{
      _mm256_castsi256_si128(_mm256_permute4x64_epi64(i8, 0x88))};
}

#if HWY_TARGET <= HWY_AVX3
template <class D, HWY_IF_I32_D(D)>
HWY_API Vec128<int32_t> DemoteTo(D /* tag */, Vec256<int64_t> v) {
  return Vec128<int32_t>{_mm256_cvtsepi64_epi32(v.raw)};
}
template <class D, HWY_IF_I16_D(D)>
HWY_API Vec64<int16_t> DemoteTo(D /* tag */, Vec256<int64_t> v) {
  return Vec64<int16_t>{_mm256_cvtsepi64_epi16(v.raw)};
}
template <class D, HWY_IF_I8_D(D)>
HWY_API Vec32<int8_t> DemoteTo(D /* tag */, Vec256<int64_t> v) {
  return Vec32<int8_t>{_mm256_cvtsepi64_epi8(v.raw)};
}

template <class D, HWY_IF_U32_D(D)>
HWY_API Vec128<uint32_t> DemoteTo(D /* tag */, Vec256<int64_t> v) {
  const auto neg_mask = MaskFromVec(v);
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  const __mmask8 non_neg_mask = _knot_mask8(neg_mask.raw);
#else
  const __mmask8 non_neg_mask = static_cast<__mmask8>(~neg_mask.raw);
#endif
  return Vec128<uint32_t>{_mm256_maskz_cvtusepi64_epi32(non_neg_mask, v.raw)};
}
template <class D, HWY_IF_U16_D(D)>
HWY_API Vec64<uint16_t> DemoteTo(D /* tag */, Vec256<int64_t> v) {
  const auto neg_mask = MaskFromVec(v);
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  const __mmask8 non_neg_mask = _knot_mask8(neg_mask.raw);
#else
  const __mmask8 non_neg_mask = static_cast<__mmask8>(~neg_mask.raw);
#endif
  return Vec64<uint16_t>{_mm256_maskz_cvtusepi64_epi16(non_neg_mask, v.raw)};
}
template <class D, HWY_IF_U8_D(D)>
HWY_API Vec32<uint8_t> DemoteTo(D /* tag */, Vec256<int64_t> v) {
  const auto neg_mask = MaskFromVec(v);
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  const __mmask8 non_neg_mask = _knot_mask8(neg_mask.raw);
#else
  const __mmask8 non_neg_mask = static_cast<__mmask8>(~neg_mask.raw);
#endif
  return Vec32<uint8_t>{_mm256_maskz_cvtusepi64_epi8(non_neg_mask, v.raw)};
}

template <class D, HWY_IF_U32_D(D)>
HWY_API Vec128<uint32_t> DemoteTo(D /* tag */, Vec256<uint64_t> v) {
  return Vec128<uint32_t>{_mm256_cvtusepi64_epi32(v.raw)};
}
template <class D, HWY_IF_U16_D(D)>
HWY_API Vec64<uint16_t> DemoteTo(D /* tag */, Vec256<uint64_t> v) {
  return Vec64<uint16_t>{_mm256_cvtusepi64_epi16(v.raw)};
}
template <class D, HWY_IF_U8_D(D)>
HWY_API Vec32<uint8_t> DemoteTo(D /* tag */, Vec256<uint64_t> v) {
  return Vec32<uint8_t>{_mm256_cvtusepi64_epi8(v.raw)};
}
#endif  // HWY_TARGET <= HWY_AVX3

  // Avoid "value of intrinsic immediate argument '8' is out of range '0 - 7'".
  // 8 is the correct value of _MM_FROUND_NO_EXC, which is allowed here.
HWY_DIAGNOSTICS(push)
HWY_DIAGNOSTICS_OFF(disable : 4556, ignored "-Wsign-conversion")

template <class D, HWY_IF_F16_D(D)>
HWY_API Vec128<float16_t> DemoteTo(D df16, Vec256<float> v) {
#ifdef HWY_DISABLE_F16C
  const RebindToUnsigned<decltype(df16)> du16;
  const Rebind<uint32_t, decltype(df16)> du;
  const RebindToSigned<decltype(du)> di;
  const auto bits32 = BitCast(du, v);
  const auto sign = ShiftRight<31>(bits32);
  const auto biased_exp32 = ShiftRight<23>(bits32) & Set(du, 0xFF);
  const auto mantissa32 = bits32 & Set(du, 0x7FFFFF);

  const auto k15 = Set(di, 15);
  const auto exp = Min(BitCast(di, biased_exp32) - Set(di, 127), k15);
  const auto is_tiny = exp < Set(di, -24);

  const auto is_subnormal = exp < Set(di, -14);
  const auto biased_exp16 =
      BitCast(du, IfThenZeroElse(is_subnormal, exp + k15));
  const auto sub_exp = BitCast(du, Set(di, -14) - exp);  // [1, 11)
  const auto sub_m = (Set(du, 1) << (Set(du, 10) - sub_exp)) +
                     (mantissa32 >> (Set(du, 13) + sub_exp));
  const auto mantissa16 = IfThenElse(RebindMask(du, is_subnormal), sub_m,
                                     ShiftRight<13>(mantissa32));  // <1024

  const auto sign16 = ShiftLeft<15>(sign);
  const auto normal16 = sign16 | ShiftLeft<10>(biased_exp16) | mantissa16;
  const auto bits16 = IfThenZeroElse(is_tiny, BitCast(di, normal16));
  return BitCast(df16, DemoteTo(du16, bits16));
#else
  (void)df16;
  return Vec128<float16_t>{_mm256_cvtps_ph(v.raw, _MM_FROUND_NO_EXC)};
#endif
}

HWY_DIAGNOSTICS(pop)

template <class D, HWY_IF_BF16_D(D)>
HWY_API Vec128<bfloat16_t> DemoteTo(D dbf16, Vec256<float> v) {
  // TODO(janwas): _mm256_cvtneps_pbh once we have avx512bf16.
  const Rebind<int32_t, decltype(dbf16)> di32;
  const Rebind<uint32_t, decltype(dbf16)> du32;  // for logical shift right
  const Rebind<uint16_t, decltype(dbf16)> du16;
  const auto bits_in_32 = BitCast(di32, ShiftRight<16>(BitCast(du32, v)));
  return BitCast(dbf16, DemoteTo(du16, bits_in_32));
}

HWY_API Vec256<bfloat16_t> ReorderDemote2To(Full256<bfloat16_t> dbf16,
                                            Vec256<float> a, Vec256<float> b) {
  // TODO(janwas): _mm256_cvtne2ps_pbh once we have avx512bf16.
  const RebindToUnsigned<decltype(dbf16)> du16;
  const Repartition<uint32_t, decltype(dbf16)> du32;
  const Vec256<uint32_t> b_in_even = ShiftRight<16>(BitCast(du32, b));
  return BitCast(dbf16, OddEven(BitCast(du16, a), BitCast(du16, b_in_even)));
}

template <class D, HWY_IF_I16_D(D)>
HWY_API Vec256<int16_t> ReorderDemote2To(D /*d16*/, Vec256<int32_t> a,
                                         Vec256<int32_t> b) {
  return Vec256<int16_t>{_mm256_packs_epi32(a.raw, b.raw)};
}

template <class D, HWY_IF_U16_D(D)>
HWY_API Vec256<uint16_t> ReorderDemote2To(D /*d16*/, Vec256<int32_t> a,
                                          Vec256<int32_t> b) {
  return Vec256<uint16_t>{_mm256_packus_epi32(a.raw, b.raw)};
}

template <class D, HWY_IF_U16_D(D)>
HWY_API Vec256<uint16_t> ReorderDemote2To(D dn, Vec256<uint32_t> a,
                                          Vec256<uint32_t> b) {
  const DFromV<decltype(a)> d;
  const RebindToSigned<decltype(d)> di;
  const auto max_i32 = Set(d, 0x7FFFFFFFu);
  return ReorderDemote2To(dn, BitCast(di, Min(a, max_i32)),
                              BitCast(di, Min(b, max_i32)));
}

template <class D, HWY_IF_I8_D(D)>
HWY_API Vec256<int8_t> ReorderDemote2To(D /*d16*/, Vec256<int16_t> a,
                                        Vec256<int16_t> b) {
  return Vec256<int8_t>{_mm256_packs_epi16(a.raw, b.raw)};
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec256<uint8_t> ReorderDemote2To(D /*d16*/, Vec256<int16_t> a,
                                         Vec256<int16_t> b) {
  return Vec256<uint8_t>{_mm256_packus_epi16(a.raw, b.raw)};
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec256<uint8_t> ReorderDemote2To(D dn, Vec256<uint16_t> a,
                                         Vec256<uint16_t> b) {
  const DFromV<decltype(a)> d;
  const RebindToSigned<decltype(d)> di;
  const auto max_i16 = Set(d, 0x7FFFu);
  return ReorderDemote2To(dn, BitCast(di, Min(a, max_i16)),
                              BitCast(di, Min(b, max_i16)));
}

#if HWY_TARGET > HWY_AVX3
template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_I32_D(D)>
HWY_API Vec256<int32_t> ReorderDemote2To(D dn, Vec256<int64_t> a,
                                         Vec256<int64_t> b) {
  const DFromV<decltype(a)> di64;
  const RebindToUnsigned<decltype(di64)> du64;
  const Half<decltype(dn)> dnh;
  const Repartition<float, decltype(dn)> dn_f;

  // Negative values are saturated by first saturating their bitwise inverse
  // and then inverting the saturation result
  const auto invert_mask_a = BitCast(du64, BroadcastSignBit(a));
  const auto invert_mask_b = BitCast(du64, BroadcastSignBit(b));
  const auto saturated_a = Xor(invert_mask_a,
    detail::DemoteFromU64Saturate(dnh, Xor(invert_mask_a, BitCast(du64, a))));
  const auto saturated_b = Xor(invert_mask_b,
    detail::DemoteFromU64Saturate(dnh, Xor(invert_mask_b, BitCast(du64, b))));

  return BitCast(dn, Vec256<float>{_mm256_shuffle_ps(
    BitCast(dn_f, saturated_a).raw, BitCast(dn_f, saturated_b).raw,
    _MM_SHUFFLE(2, 0, 2, 0))});
}

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_U32_D(D)>
HWY_API Vec256<uint32_t> ReorderDemote2To(D dn, Vec256<int64_t> a,
                                          Vec256<int64_t> b) {
  const DFromV<decltype(a)> di64;
  const RebindToUnsigned<decltype(di64)> du64;
  const Half<decltype(dn)> dnh;
  const Repartition<float, decltype(dn)> dn_f;

  const auto saturated_a = detail::DemoteFromU64Saturate(dnh,
    BitCast(du64, AndNot(BroadcastSignBit(a), a)));
  const auto saturated_b = detail::DemoteFromU64Saturate(dnh,
    BitCast(du64, AndNot(BroadcastSignBit(b), b)));

  return BitCast(dn, Vec256<float>{_mm256_shuffle_ps(
    BitCast(dn_f, saturated_a).raw, BitCast(dn_f, saturated_b).raw,
    _MM_SHUFFLE(2, 0, 2, 0))});
}

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_U32_D(D)>
HWY_API Vec256<uint32_t> ReorderDemote2To(D dn, Vec256<uint64_t> a,
                                          Vec256<uint64_t> b) {
  const Half<decltype(dn)> dnh;
  const Repartition<float, decltype(dn)> dn_f;

  const auto saturated_a = detail::DemoteFromU64Saturate(dnh, a);
  const auto saturated_b = detail::DemoteFromU64Saturate(dnh, b);

  return BitCast(dn, Vec256<float>{_mm256_shuffle_ps(
    BitCast(dn_f, saturated_a).raw, BitCast(dn_f, saturated_b).raw,
    _MM_SHUFFLE(2, 0, 2, 0))});
}
#endif  // HWY_TARGET > HWY_AVX3

template <class D, class V, HWY_IF_NOT_FLOAT_NOR_SPECIAL(TFromD<D>),
          HWY_IF_V_SIZE_D(D, 32), HWY_IF_NOT_FLOAT_NOR_SPECIAL_V(V),
          HWY_IF_T_SIZE_V(V, sizeof(TFromD<D>) * 2),
          HWY_IF_LANES_D(D, HWY_MAX_LANES_D(DFromV<V>) * 2),
          HWY_IF_T_SIZE_ONE_OF_V(V,
                                 (1 << 1) | (1 << 2) | (1 << 4) |
                                     ((HWY_TARGET > HWY_AVX3) ? (1 << 8) : 0))>
HWY_API VFromD<D> OrderedDemote2To(D d, V a, V b) {
  return VFromD<D>{_mm256_permute4x64_epi64(
    ReorderDemote2To(d, a, b).raw, _MM_SHUFFLE(3, 1, 2, 0))};
}

template <class D, HWY_IF_F32_D(D)>
HWY_API Vec128<float> DemoteTo(D /* tag */, Vec256<double> v) {
  return Vec128<float>{_mm256_cvtpd_ps(v.raw)};
}

template <class D, HWY_IF_I32_D(D)>
HWY_API Vec128<int32_t> DemoteTo(D /* tag */, Vec256<double> v) {
  const Full256<double> d64;
  const auto clamped = detail::ClampF64ToI32Max(d64, v);
  return Vec128<int32_t>{_mm256_cvttpd_epi32(clamped.raw)};
}

// For already range-limited input [0, 255].
HWY_API Vec128<uint8_t, 8> U8FromU32(const Vec256<uint32_t> v) {
  const Full256<uint32_t> d32;
  const Full64<uint8_t> d8;
  alignas(32) static constexpr uint32_t k8From32[8] = {
      0x0C080400u, ~0u, ~0u, ~0u, ~0u, 0x0C080400u, ~0u, ~0u};
  // Place first four bytes in lo[0], remaining 4 in hi[1].
  const auto quad = TableLookupBytes(v, Load(d32, k8From32));
  // Interleave both quadruplets - OR instead of unpack reduces port5 pressure.
  const auto lo = LowerHalf(quad);
  const auto hi = UpperHalf(Half<decltype(d32)>(), quad);
  return BitCast(d8, LowerHalf(lo | hi));
}

// ------------------------------ Truncations

namespace detail {

// LO and HI each hold four indices of bytes within a 128-bit block.
template <uint32_t LO, uint32_t HI, typename T>
HWY_INLINE Vec128<uint32_t> LookupAndConcatHalves(Vec256<T> v) {
  const Full256<uint32_t> d32;

#if HWY_TARGET <= HWY_AVX3_DL
  alignas(32) static constexpr uint32_t kMap[8] = {
      LO, HI, 0x10101010 + LO, 0x10101010 + HI, 0, 0, 0, 0};
  const auto result = _mm256_permutexvar_epi8(v.raw, Load(d32, kMap).raw);
#else
  alignas(32) static constexpr uint32_t kMap[8] = {LO,  HI,  ~0u, ~0u,
                                                   ~0u, ~0u, LO,  HI};
  const auto quad = TableLookupBytes(v, Load(d32, kMap));
  const auto result = _mm256_permute4x64_epi64(quad.raw, 0xCC);
  // Possible alternative:
  // const auto lo = LowerHalf(quad);
  // const auto hi = UpperHalf(Half<decltype(d32)>(), quad);
  // const auto result = lo | hi;
#endif

  return Vec128<uint32_t>{_mm256_castsi256_si128(result)};
}

// LO and HI each hold two indices of bytes within a 128-bit block.
template <uint16_t LO, uint16_t HI, typename T>
HWY_INLINE Vec128<uint32_t, 2> LookupAndConcatQuarters(Vec256<T> v) {
  const Full256<uint16_t> d16;

#if HWY_TARGET <= HWY_AVX3_DL
  alignas(32) static constexpr uint16_t kMap[16] = {
      LO, HI, 0x1010 + LO, 0x1010 + HI, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const auto result = _mm256_permutexvar_epi8(v.raw, Load(d16, kMap).raw);
  return LowerHalf(Vec128<uint32_t>{_mm256_castsi256_si128(result)});
#else
  constexpr uint16_t ff = static_cast<uint16_t>(~0u);
  alignas(32) static constexpr uint16_t kMap[16] = {
      LO, ff, HI, ff, ff, ff, ff, ff, ff, ff, ff, ff, LO, ff, HI, ff};
  const auto quad = TableLookupBytes(v, Load(d16, kMap));
  const auto mixed = _mm256_permute4x64_epi64(quad.raw, 0xCC);
  const auto half = _mm256_castsi256_si128(mixed);
  return LowerHalf(Vec128<uint32_t>{_mm_packus_epi32(half, half)});
#endif
}

}  // namespace detail

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec32<uint8_t> TruncateTo(D /* tag */, Vec256<uint64_t> v) {
  const Full256<uint32_t> d32;
#if HWY_TARGET <= HWY_AVX3_DL
  alignas(32) static constexpr uint32_t kMap[8] = {0x18100800u, 0, 0, 0,
                                                   0,           0, 0, 0};
  const auto result = _mm256_permutexvar_epi8(v.raw, Load(d32, kMap).raw);
  return LowerHalf(LowerHalf(LowerHalf(Vec256<uint8_t>{result})));
#else
  alignas(32) static constexpr uint32_t kMap[8] = {0xFFFF0800u, ~0u, ~0u, ~0u,
                                                   0x0800FFFFu, ~0u, ~0u, ~0u};
  const auto quad = TableLookupBytes(v, Load(d32, kMap));
  const auto lo = LowerHalf(quad);
  const auto hi = UpperHalf(Half<decltype(d32)>(), quad);
  const auto result = lo | hi;
  return LowerHalf(LowerHalf(Vec128<uint8_t>{result.raw}));
#endif
}

template <class D, HWY_IF_U16_D(D)>
HWY_API Vec64<uint16_t> TruncateTo(D /* tag */, Vec256<uint64_t> v) {
  const auto result = detail::LookupAndConcatQuarters<0x100, 0x908>(v);
  return Vec64<uint16_t>{result.raw};
}

template <class D, HWY_IF_U32_D(D)>
HWY_API Vec128<uint32_t> TruncateTo(D /* tag */, Vec256<uint64_t> v) {
  const Full256<uint32_t> d32;
  alignas(32) static constexpr uint32_t kEven[8] = {0, 2, 4, 6, 0, 2, 4, 6};
  const auto v32 =
      TableLookupLanes(BitCast(d32, v), SetTableIndices(d32, kEven));
  return LowerHalf(Vec256<uint32_t>{v32.raw});
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec64<uint8_t> TruncateTo(D /* tag */, Vec256<uint32_t> v) {
  const auto full = detail::LookupAndConcatQuarters<0x400, 0xC08>(v);
  return Vec64<uint8_t>{full.raw};
}

template <class D, HWY_IF_U16_D(D)>
HWY_API Vec128<uint16_t> TruncateTo(D /* tag */, Vec256<uint32_t> v) {
  const auto full = detail::LookupAndConcatHalves<0x05040100, 0x0D0C0908>(v);
  return Vec128<uint16_t>{full.raw};
}

template <class D, HWY_IF_U8_D(D)>
HWY_API Vec128<uint8_t> TruncateTo(D /* tag */, Vec256<uint16_t> v) {
  const auto full = detail::LookupAndConcatHalves<0x06040200, 0x0E0C0A08>(v);
  return Vec128<uint8_t>{full.raw};
}

// ------------------------------ Integer <=> fp (ShiftRight, OddEven)

template <class D, HWY_IF_F32_D(D)>
HWY_API Vec256<float> ConvertTo(D /* tag */, Vec256<int32_t> v) {
  return Vec256<float>{_mm256_cvtepi32_ps(v.raw)};
}

template <class D, HWY_IF_F64_D(D)>
HWY_API Vec256<double> ConvertTo(D dd, const Vec256<int64_t> v) {
#if HWY_TARGET <= HWY_AVX3
  (void)dd;
  return Vec256<double>{_mm256_cvtepi64_pd(v.raw)};
#else
  // Based on wim's approach (https://stackoverflow.com/questions/41144668/)
  const Repartition<uint32_t, decltype(dd)> d32;
  const Repartition<uint64_t, decltype(dd)> d64;

  // Toggle MSB of lower 32-bits and insert exponent for 2^84 + 2^63
  const auto k84_63 = Set(d64, 0x4530000080000000ULL);
  const auto v_upper = BitCast(dd, ShiftRight<32>(BitCast(d64, v)) ^ k84_63);

  // Exponent is 2^52, lower 32 bits from v (=> 32-bit OddEven)
  const auto k52 = Set(d32, 0x43300000);
  const auto v_lower = BitCast(dd, OddEven(k52, BitCast(d32, v)));

  const auto k84_63_52 = BitCast(dd, Set(d64, 0x4530000080100000ULL));
  return (v_upper - k84_63_52) + v_lower;  // order matters!
#endif
}

template <class D, HWY_IF_F32_D(D)>
HWY_API Vec256<float> ConvertTo(D df, Vec256<uint32_t> v) {
#if HWY_TARGET <= HWY_AVX3
  (void)df;
  return Vec256<float>{_mm256_cvtepu32_ps(v.raw)};
#else
  // Based on wim's approach (https://stackoverflow.com/questions/34066228/)
  const RebindToUnsigned<decltype(df)> du32;
  const RebindToSigned<decltype(df)> d32;

  const auto msk_lo = Set(du32, 0xFFFF);
  const auto cnst2_16_flt = Set(df, 65536.0f);  // 2^16

  // Extract the 16 lowest/highest significant bits of v and cast to signed int
  const auto v_lo = BitCast(d32, And(v, msk_lo));
  const auto v_hi = BitCast(d32, ShiftRight<16>(v));

  return MulAdd(cnst2_16_flt, ConvertTo(df, v_hi), ConvertTo(df, v_lo));
#endif
}

template <class D, HWY_IF_F64_D(D)>
HWY_API Vec256<double> ConvertTo(D dd, Vec256<uint64_t> v) {
#if HWY_TARGET <= HWY_AVX3
  (void)dd;
  return Vec256<double>{_mm256_cvtepu64_pd(v.raw)};
#else
  // Based on wim's approach (https://stackoverflow.com/questions/41144668/)
  const RebindToUnsigned<decltype(dd)> d64;
  using VU = VFromD<decltype(d64)>;

  const VU msk_lo = Set(d64, 0xFFFFFFFFULL);
  const auto cnst2_32_dbl = Set(dd, 4294967296.0);  // 2^32

  // Extract the 32 lowest significant bits of v
  const VU v_lo = And(v, msk_lo);
  const VU v_hi = ShiftRight<32>(v);

  auto uint64_to_double256_fast = [&dd](Vec256<uint64_t> w) HWY_ATTR {
    w = Or(w, Vec256<uint64_t>{
                  detail::BitCastToInteger(Set(dd, 0x0010000000000000).raw)});
    return BitCast(dd, w) - Set(dd, 0x0010000000000000);
  };

  const auto v_lo_dbl = uint64_to_double256_fast(v_lo);
  return MulAdd(cnst2_32_dbl, uint64_to_double256_fast(v_hi), v_lo_dbl);
#endif
}

// Truncates (rounds toward zero).
template <class D, HWY_IF_I32_D(D)>
HWY_API Vec256<int32_t> ConvertTo(D d, Vec256<float> v) {
  return detail::FixConversionOverflow(d, v, _mm256_cvttps_epi32(v.raw));
}

template <class D, HWY_IF_I64_D(D)>
HWY_API Vec256<int64_t> ConvertTo(D di, Vec256<double> v) {
#if HWY_TARGET <= HWY_AVX3
  return detail::FixConversionOverflow(di, v, _mm256_cvttpd_epi64(v.raw));
#else
  using VI = decltype(Zero(di));
  const VI k0 = Zero(di);
  const VI k1 = Set(di, 1);
  const VI k51 = Set(di, 51);

  // Exponent indicates whether the number can be represented as int64_t.
  const VI biased_exp = ShiftRight<52>(BitCast(di, v)) & Set(di, 0x7FF);
  const VI exp = biased_exp - Set(di, 0x3FF);
  const auto in_range = exp < Set(di, 63);

  // If we were to cap the exponent at 51 and add 2^52, the number would be in
  // [2^52, 2^53) and mantissa bits could be read out directly. We need to
  // round-to-0 (truncate), but changing rounding mode in MXCSR hits a
  // compiler reordering bug: https://gcc.godbolt.org/z/4hKj6c6qc . We instead
  // manually shift the mantissa into place (we already have many of the
  // inputs anyway).
  const VI shift_mnt = Max(k51 - exp, k0);
  const VI shift_int = Max(exp - k51, k0);
  const VI mantissa = BitCast(di, v) & Set(di, (1ULL << 52) - 1);
  // Include implicit 1-bit; shift by one more to ensure it's in the mantissa.
  const VI int52 = (mantissa | Set(di, 1ULL << 52)) >> (shift_mnt + k1);
  // For inputs larger than 2^52, insert zeros at the bottom.
  const VI shifted = int52 << shift_int;
  // Restore the one bit lost when shifting in the implicit 1-bit.
  const VI restored = shifted | ((mantissa & k1) << (shift_int - k1));

  // Saturate to LimitsMin (unchanged when negating below) or LimitsMax.
  const VI sign_mask = BroadcastSignBit(BitCast(di, v));
  const VI limit = Set(di, LimitsMax<int64_t>()) - sign_mask;
  const VI magnitude = IfThenElse(in_range, restored, limit);

  // If the input was negative, negate the integer (two's complement).
  return (magnitude ^ sign_mask) - sign_mask;
#endif
}

HWY_API Vec256<int32_t> NearestInt(const Vec256<float> v) {
  const Full256<int32_t> di;
  return detail::FixConversionOverflow(di, v, _mm256_cvtps_epi32(v.raw));
}

template <class D, HWY_IF_F32_D(D)>
HWY_API Vec256<float> PromoteTo(D df32, Vec128<float16_t> v) {
#ifdef HWY_DISABLE_F16C
  const RebindToSigned<decltype(df32)> di32;
  const RebindToUnsigned<decltype(df32)> du32;
  // Expand to u32 so we can shift.
  const auto bits16 = PromoteTo(du32, Vec128<uint16_t>{v.raw});
  const auto sign = ShiftRight<15>(bits16);
  const auto biased_exp = ShiftRight<10>(bits16) & Set(du32, 0x1F);
  const auto mantissa = bits16 & Set(du32, 0x3FF);
  const auto subnormal =
      BitCast(du32, ConvertTo(df32, BitCast(di32, mantissa)) *
                        Set(df32, 1.0f / 16384 / 1024));

  const auto biased_exp32 = biased_exp + Set(du32, 127 - 15);
  const auto mantissa32 = ShiftLeft<23 - 10>(mantissa);
  const auto normal = ShiftLeft<23>(biased_exp32) | mantissa32;
  const auto bits32 = IfThenElse(biased_exp == Zero(du32), subnormal, normal);
  return BitCast(df32, ShiftLeft<31>(sign) | bits32);
#else
  (void)df32;
  return Vec256<float>{_mm256_cvtph_ps(v.raw)};
#endif
}

template <class D, HWY_IF_F32_D(D)>
HWY_API Vec256<float> PromoteTo(D df32, Vec128<bfloat16_t> v) {
  const Rebind<uint16_t, decltype(df32)> du16;
  const RebindToSigned<decltype(df32)> di32;
  return BitCast(df32, ShiftLeft<16>(PromoteTo(di32, BitCast(du16, v))));
}

// ================================================== CRYPTO

#if !defined(HWY_DISABLE_PCLMUL_AES)

// Per-target flag to prevent generic_ops-inl.h from defining AESRound.
#ifdef HWY_NATIVE_AES
#undef HWY_NATIVE_AES
#else
#define HWY_NATIVE_AES
#endif

HWY_API Vec256<uint8_t> AESRound(Vec256<uint8_t> state,
                                 Vec256<uint8_t> round_key) {
#if HWY_TARGET <= HWY_AVX3_DL
  return Vec256<uint8_t>{_mm256_aesenc_epi128(state.raw, round_key.raw)};
#else
  const Full256<uint8_t> d;
  const Half<decltype(d)> d2;
  return Combine(d, AESRound(UpperHalf(d2, state), UpperHalf(d2, round_key)),
                 AESRound(LowerHalf(state), LowerHalf(round_key)));
#endif
}

HWY_API Vec256<uint8_t> AESLastRound(Vec256<uint8_t> state,
                                     Vec256<uint8_t> round_key) {
#if HWY_TARGET <= HWY_AVX3_DL
  return Vec256<uint8_t>{_mm256_aesenclast_epi128(state.raw, round_key.raw)};
#else
  const Full256<uint8_t> d;
  const Half<decltype(d)> d2;
  return Combine(d,
                 AESLastRound(UpperHalf(d2, state), UpperHalf(d2, round_key)),
                 AESLastRound(LowerHalf(state), LowerHalf(round_key)));
#endif
}

HWY_API Vec256<uint64_t> CLMulLower(Vec256<uint64_t> a, Vec256<uint64_t> b) {
#if HWY_TARGET <= HWY_AVX3_DL
  return Vec256<uint64_t>{_mm256_clmulepi64_epi128(a.raw, b.raw, 0x00)};
#else
  const Full256<uint64_t> d;
  const Half<decltype(d)> d2;
  return Combine(d, CLMulLower(UpperHalf(d2, a), UpperHalf(d2, b)),
                 CLMulLower(LowerHalf(a), LowerHalf(b)));
#endif
}

HWY_API Vec256<uint64_t> CLMulUpper(Vec256<uint64_t> a, Vec256<uint64_t> b) {
#if HWY_TARGET <= HWY_AVX3_DL
  return Vec256<uint64_t>{_mm256_clmulepi64_epi128(a.raw, b.raw, 0x11)};
#else
  const Full256<uint64_t> d;
  const Half<decltype(d)> d2;
  return Combine(d, CLMulUpper(UpperHalf(d2, a), UpperHalf(d2, b)),
                 CLMulUpper(LowerHalf(a), LowerHalf(b)));
#endif
}

#endif  // HWY_DISABLE_PCLMUL_AES

// ================================================== MISC

#if HWY_TARGET <= HWY_AVX3

// ------------------------------ LoadMaskBits

// `p` points to at least 8 readable bytes, not all of which need be valid.
template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API Mask256<T> LoadMaskBits(D d, const uint8_t* HWY_RESTRICT bits) {
  constexpr size_t kN = MaxLanes(d);
  constexpr size_t kNumBytes = (kN + 7) / 8;

  uint64_t mask_bits = 0;
  CopyBytes<kNumBytes>(bits, &mask_bits);

  if (kN < 8) {
    mask_bits &= (1ull << kN) - 1;
  }

  return Mask256<T>::FromBits(mask_bits);
}

// ------------------------------ StoreMaskBits

// `p` points to at least 8 writable bytes.
template <class D, typename T = TFromD<D>>
HWY_API size_t StoreMaskBits(D d, Mask256<T> mask, uint8_t* bits) {
  constexpr size_t kN = MaxLanes(d);
  constexpr size_t kNumBytes = (kN + 7) / 8;

  CopyBytes<kNumBytes>(&mask.raw, bits);

  // Non-full byte, need to clear the undefined upper bits.
  if (kN < 8) {
    const int mask_bits = static_cast<int>((1ull << kN) - 1);
    bits[0] = static_cast<uint8_t>(bits[0] & mask_bits);
  }
  return kNumBytes;
}

// ------------------------------ Mask testing

template <class D, typename T = TFromD<D>>
HWY_API size_t CountTrue(D /* tag */, Mask256<T> mask) {
  return PopCount(static_cast<uint64_t>(mask.raw));
}

template <class D, typename T = TFromD<D>>
HWY_API size_t FindKnownFirstTrue(D /* tag */, Mask256<T> mask) {
  return Num0BitsBelowLS1Bit_Nonzero32(mask.raw);
}

template <class D, typename T = TFromD<D>>
HWY_API intptr_t FindFirstTrue(D d, Mask256<T> mask) {
  return mask.raw ? static_cast<intptr_t>(FindKnownFirstTrue(d, mask))
                  : intptr_t{-1};
}

// Beware: the suffix indicates the number of mask bits, not lane size!

namespace detail {

template <typename T>
HWY_INLINE bool AllFalse(hwy::SizeTag<1> /*tag*/, const Mask256<T> mask) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return _kortestz_mask32_u8(mask.raw, mask.raw);
#else
  return mask.raw == 0;
#endif
}
template <typename T>
HWY_INLINE bool AllFalse(hwy::SizeTag<2> /*tag*/, const Mask256<T> mask) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return _kortestz_mask16_u8(mask.raw, mask.raw);
#else
  return mask.raw == 0;
#endif
}
template <typename T>
HWY_INLINE bool AllFalse(hwy::SizeTag<4> /*tag*/, const Mask256<T> mask) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return _kortestz_mask8_u8(mask.raw, mask.raw);
#else
  return mask.raw == 0;
#endif
}
template <typename T>
HWY_INLINE bool AllFalse(hwy::SizeTag<8> /*tag*/, const Mask256<T> mask) {
  return (uint64_t{mask.raw} & 0xF) == 0;
}

}  // namespace detail

template <class D, typename T = TFromD<D>>
HWY_API bool AllFalse(D /* tag */, Mask256<T> mask) {
  return detail::AllFalse(hwy::SizeTag<sizeof(T)>(), mask);
}

namespace detail {

template <typename T>
HWY_INLINE bool AllTrue(hwy::SizeTag<1> /*tag*/, const Mask256<T> mask) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return _kortestc_mask32_u8(mask.raw, mask.raw);
#else
  return mask.raw == 0xFFFFFFFFu;
#endif
}
template <typename T>
HWY_INLINE bool AllTrue(hwy::SizeTag<2> /*tag*/, const Mask256<T> mask) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return _kortestc_mask16_u8(mask.raw, mask.raw);
#else
  return mask.raw == 0xFFFFu;
#endif
}
template <typename T>
HWY_INLINE bool AllTrue(hwy::SizeTag<4> /*tag*/, const Mask256<T> mask) {
#if HWY_COMPILER_HAS_MASK_INTRINSICS
  return _kortestc_mask8_u8(mask.raw, mask.raw);
#else
  return mask.raw == 0xFFu;
#endif
}
template <typename T>
HWY_INLINE bool AllTrue(hwy::SizeTag<8> /*tag*/, const Mask256<T> mask) {
  // Cannot use _kortestc because we have less than 8 mask bits.
  return mask.raw == 0xFu;
}

}  // namespace detail

template <class D, typename T = TFromD<D>>
HWY_API bool AllTrue(D /* tag */, const Mask256<T> mask) {
  return detail::AllTrue(hwy::SizeTag<sizeof(T)>(), mask);
}

// ------------------------------ Compress

// 16-bit is defined in x86_512 so we can use 512-bit vectors.

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Compress(Vec256<T> v, Mask256<T> mask) {
  return Vec256<T>{_mm256_maskz_compress_epi32(mask.raw, v.raw)};
}

HWY_API Vec256<float> Compress(Vec256<float> v, Mask256<float> mask) {
  return Vec256<float>{_mm256_maskz_compress_ps(mask.raw, v.raw)};
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> Compress(Vec256<T> v, Mask256<T> mask) {
  // See CompressIsPartition.
  alignas(16) static constexpr uint64_t packed_array[16] = {
      // PrintCompress64x4NibbleTables
      0x00003210, 0x00003210, 0x00003201, 0x00003210, 0x00003102, 0x00003120,
      0x00003021, 0x00003210, 0x00002103, 0x00002130, 0x00002031, 0x00002310,
      0x00001032, 0x00001320, 0x00000321, 0x00003210};

  // For lane i, shift the i-th 4-bit index down to bits [0, 2) -
  // _mm256_permutexvar_epi64 will ignore the upper bits.
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du64;
  const auto packed = Set(du64, packed_array[mask.raw]);
  alignas(64) static constexpr uint64_t shifts[4] = {0, 4, 8, 12};
  const auto indices = Indices256<T>{(packed >> Load(du64, shifts)).raw};
  return TableLookupLanes(v, indices);
}

// ------------------------------ CompressNot (Compress)

// Implemented in x86_512 for lane size != 8.

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> CompressNot(Vec256<T> v, Mask256<T> mask) {
  // See CompressIsPartition.
  alignas(16) static constexpr uint64_t packed_array[16] = {
      // PrintCompressNot64x4NibbleTables
      0x00003210, 0x00000321, 0x00001320, 0x00001032, 0x00002310, 0x00002031,
      0x00002130, 0x00002103, 0x00003210, 0x00003021, 0x00003120, 0x00003102,
      0x00003210, 0x00003201, 0x00003210, 0x00003210};

  // For lane i, shift the i-th 4-bit index down to bits [0, 2) -
  // _mm256_permutexvar_epi64 will ignore the upper bits.
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du64;
  const auto packed = Set(du64, packed_array[mask.raw]);
  alignas(32) static constexpr uint64_t shifts[4] = {0, 4, 8, 12};
  const auto indices = Indices256<T>{(packed >> Load(du64, shifts)).raw};
  return TableLookupLanes(v, indices);
}

// ------------------------------ CompressStore (defined in x86_512)
// ------------------------------ CompressBlendedStore (defined in x86_512)
// ------------------------------ CompressBitsStore (defined in x86_512)

#else  // AVX2

// ------------------------------ LoadMaskBits (TestBit)

namespace detail {

// 256 suffix avoids ambiguity with x86_128 without needing HWY_IF_V_SIZE.
template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_INLINE Mask256<T> LoadMaskBits256(uint64_t mask_bits) {
  const Full256<T> d;
  const RebindToUnsigned<decltype(d)> du;
  const Repartition<uint32_t, decltype(d)> du32;
  const auto vbits = BitCast(du, Set(du32, static_cast<uint32_t>(mask_bits)));

  // Replicate bytes 8x such that each byte contains the bit that governs it.
  const Repartition<uint64_t, decltype(d)> du64;
  alignas(32) static constexpr uint64_t kRep8[4] = {
      0x0000000000000000ull, 0x0101010101010101ull, 0x0202020202020202ull,
      0x0303030303030303ull};
  const auto rep8 = TableLookupBytes(vbits, BitCast(du, Load(du64, kRep8)));

  alignas(32) static constexpr uint8_t kBit[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                                   1, 2, 4, 8, 16, 32, 64, 128};
  return RebindMask(d, TestBit(rep8, LoadDup128(du, kBit)));
}

template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_INLINE Mask256<T> LoadMaskBits256(uint64_t mask_bits) {
  const Full256<T> d;
  const RebindToUnsigned<decltype(d)> du;
  alignas(32) static constexpr uint16_t kBit[16] = {
      1,     2,     4,     8,     16,     32,     64,     128,
      0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000};
  const auto vmask_bits = Set(du, static_cast<uint16_t>(mask_bits));
  return RebindMask(d, TestBit(vmask_bits, Load(du, kBit)));
}

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Mask256<T> LoadMaskBits256(uint64_t mask_bits) {
  const Full256<T> d;
  const RebindToUnsigned<decltype(d)> du;
  alignas(32) static constexpr uint32_t kBit[8] = {1, 2, 4, 8, 16, 32, 64, 128};
  const auto vmask_bits = Set(du, static_cast<uint32_t>(mask_bits));
  return RebindMask(d, TestBit(vmask_bits, Load(du, kBit)));
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Mask256<T> LoadMaskBits256(uint64_t mask_bits) {
  const Full256<T> d;
  const RebindToUnsigned<decltype(d)> du;
  alignas(32) static constexpr uint64_t kBit[8] = {1, 2, 4, 8};
  return RebindMask(d, TestBit(Set(du, mask_bits), Load(du, kBit)));
}

}  // namespace detail

// `p` points to at least 8 readable bytes, not all of which need be valid.
template <class D, HWY_IF_V_SIZE_D(D, 32), typename T = TFromD<D>>
HWY_API Mask256<T> LoadMaskBits(D d, const uint8_t* HWY_RESTRICT bits) {
  constexpr size_t kN = MaxLanes(d);
  constexpr size_t kNumBytes = (kN + 7) / 8;

  uint64_t mask_bits = 0;
  CopyBytes<kNumBytes>(bits, &mask_bits);

  if (kN < 8) {
    mask_bits &= (1ull << kN) - 1;
  }

  return detail::LoadMaskBits256<T>(mask_bits);
}

// ------------------------------ StoreMaskBits

namespace detail {

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_INLINE uint64_t BitsFromMask(const Mask256<T> mask) {
  const Full256<T> d;
  const Full256<uint8_t> d8;
  const auto sign_bits = BitCast(d8, VecFromMask(d, mask)).raw;
  // Prevent sign-extension of 32-bit masks because the intrinsic returns int.
  return static_cast<uint32_t>(_mm256_movemask_epi8(sign_bits));
}

template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_INLINE uint64_t BitsFromMask(const Mask256<T> mask) {
#if HWY_ARCH_X86_64
  const Full256<T> d;
  const Full256<uint8_t> d8;
  const Mask256<uint8_t> mask8 = MaskFromVec(BitCast(d8, VecFromMask(d, mask)));
  const uint64_t sign_bits8 = BitsFromMask(mask8);
  // Skip the bits from the lower byte of each u16 (better not to use the
  // same packs_epi16 as SSE4, because that requires an extra swizzle here).
  return _pext_u64(sign_bits8, 0xAAAAAAAAull);
#else
  // Slow workaround for 32-bit builds, which lack _pext_u64.
  // Remove useless lower half of each u16 while preserving the sign bit.
  // Bytes [0, 8) and [16, 24) have the same sign bits as the input lanes.
  const auto sign_bits = _mm256_packs_epi16(mask.raw, _mm256_setzero_si256());
  // Move odd qwords (value zero) to top so they don't affect the mask value.
  const auto compressed =
      _mm256_permute4x64_epi64(sign_bits, _MM_SHUFFLE(3, 1, 2, 0));
  return static_cast<unsigned>(_mm256_movemask_epi8(compressed));
#endif  // HWY_ARCH_X86_64
}

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE uint64_t BitsFromMask(const Mask256<T> mask) {
  const Full256<T> d;
  const Full256<float> df;
  const auto sign_bits = BitCast(df, VecFromMask(d, mask)).raw;
  return static_cast<unsigned>(_mm256_movemask_ps(sign_bits));
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE uint64_t BitsFromMask(const Mask256<T> mask) {
  const Full256<T> d;
  const Full256<double> df;
  const auto sign_bits = BitCast(df, VecFromMask(d, mask)).raw;
  return static_cast<unsigned>(_mm256_movemask_pd(sign_bits));
}

}  // namespace detail

// `p` points to at least 8 writable bytes.
template <class D, typename T = TFromD<D>>
HWY_API size_t StoreMaskBits(D /* tag */, Mask256<T> mask, uint8_t* bits) {
  constexpr size_t N = 32 / sizeof(T);
  constexpr size_t kNumBytes = (N + 7) / 8;

  const uint64_t mask_bits = detail::BitsFromMask(mask);
  CopyBytes<kNumBytes>(&mask_bits, bits);
  return kNumBytes;
}

// ------------------------------ Mask testing

// Specialize for 16-bit lanes to avoid unnecessary pext. This assumes each mask
// lane is 0 or ~0.
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API bool AllFalse(D d, const Mask256<T> mask) {
  const Repartition<uint8_t, decltype(d)> d8;
  const Mask256<uint8_t> mask8 = MaskFromVec(BitCast(d8, VecFromMask(d, mask)));
  return detail::BitsFromMask(mask8) == 0;
}

template <class D, typename T = TFromD<D>, HWY_IF_NOT_T_SIZE(T, 2)>
HWY_API bool AllFalse(D /* tag */, const Mask256<T> mask) {
  // Cheaper than PTEST, which is 2 uop / 3L.
  return detail::BitsFromMask(mask) == 0;
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API bool AllTrue(D d, const Mask256<T> mask) {
  const Repartition<uint8_t, decltype(d)> d8;
  const Mask256<uint8_t> mask8 = MaskFromVec(BitCast(d8, VecFromMask(d, mask)));
  return detail::BitsFromMask(mask8) == (1ull << 32) - 1;
}
template <class D, typename T = TFromD<D>, HWY_IF_NOT_T_SIZE(T, 2)>
HWY_API bool AllTrue(D /* tag */, const Mask256<T> mask) {
  constexpr uint64_t kAllBits = (1ull << (32 / sizeof(T))) - 1;
  return detail::BitsFromMask(mask) == kAllBits;
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API size_t CountTrue(D d, const Mask256<T> mask) {
  const Repartition<uint8_t, decltype(d)> d8;
  const Mask256<uint8_t> mask8 = MaskFromVec(BitCast(d8, VecFromMask(d, mask)));
  return PopCount(detail::BitsFromMask(mask8)) >> 1;
}
template <class D, typename T = TFromD<D>, HWY_IF_NOT_T_SIZE(T, 2)>
HWY_API size_t CountTrue(D /* tag */, const Mask256<T> mask) {
  return PopCount(detail::BitsFromMask(mask));
}

template <class D, typename T = TFromD<D>>
HWY_API size_t FindKnownFirstTrue(D /* tag */, Mask256<T> mask) {
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  return Num0BitsBelowLS1Bit_Nonzero64(mask_bits);
}

template <class D, typename T = TFromD<D>>
HWY_API intptr_t FindFirstTrue(D /* tag */, Mask256<T> mask) {
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  return mask_bits ? intptr_t(Num0BitsBelowLS1Bit_Nonzero64(mask_bits)) : -1;
}

// ------------------------------ Compress, CompressBits

namespace detail {

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec256<uint32_t> IndicesFromBits256(uint64_t mask_bits) {
  const Full256<uint32_t> d32;
  // We need a masked Iota(). With 8 lanes, there are 256 combinations and a LUT
  // of SetTableIndices would require 8 KiB, a large part of L1D. The other
  // alternative is _pext_u64, but this is extremely slow on Zen2 (18 cycles)
  // and unavailable in 32-bit builds. We instead compress each index into 4
  // bits, for a total of 1 KiB.
  alignas(16) static constexpr uint32_t packed_array[256] = {
      // PrintCompress32x8Tables
      0x76543210, 0x76543218, 0x76543209, 0x76543298, 0x7654310a, 0x765431a8,
      0x765430a9, 0x76543a98, 0x7654210b, 0x765421b8, 0x765420b9, 0x76542b98,
      0x765410ba, 0x76541ba8, 0x76540ba9, 0x7654ba98, 0x7653210c, 0x765321c8,
      0x765320c9, 0x76532c98, 0x765310ca, 0x76531ca8, 0x76530ca9, 0x7653ca98,
      0x765210cb, 0x76521cb8, 0x76520cb9, 0x7652cb98, 0x76510cba, 0x7651cba8,
      0x7650cba9, 0x765cba98, 0x7643210d, 0x764321d8, 0x764320d9, 0x76432d98,
      0x764310da, 0x76431da8, 0x76430da9, 0x7643da98, 0x764210db, 0x76421db8,
      0x76420db9, 0x7642db98, 0x76410dba, 0x7641dba8, 0x7640dba9, 0x764dba98,
      0x763210dc, 0x76321dc8, 0x76320dc9, 0x7632dc98, 0x76310dca, 0x7631dca8,
      0x7630dca9, 0x763dca98, 0x76210dcb, 0x7621dcb8, 0x7620dcb9, 0x762dcb98,
      0x7610dcba, 0x761dcba8, 0x760dcba9, 0x76dcba98, 0x7543210e, 0x754321e8,
      0x754320e9, 0x75432e98, 0x754310ea, 0x75431ea8, 0x75430ea9, 0x7543ea98,
      0x754210eb, 0x75421eb8, 0x75420eb9, 0x7542eb98, 0x75410eba, 0x7541eba8,
      0x7540eba9, 0x754eba98, 0x753210ec, 0x75321ec8, 0x75320ec9, 0x7532ec98,
      0x75310eca, 0x7531eca8, 0x7530eca9, 0x753eca98, 0x75210ecb, 0x7521ecb8,
      0x7520ecb9, 0x752ecb98, 0x7510ecba, 0x751ecba8, 0x750ecba9, 0x75ecba98,
      0x743210ed, 0x74321ed8, 0x74320ed9, 0x7432ed98, 0x74310eda, 0x7431eda8,
      0x7430eda9, 0x743eda98, 0x74210edb, 0x7421edb8, 0x7420edb9, 0x742edb98,
      0x7410edba, 0x741edba8, 0x740edba9, 0x74edba98, 0x73210edc, 0x7321edc8,
      0x7320edc9, 0x732edc98, 0x7310edca, 0x731edca8, 0x730edca9, 0x73edca98,
      0x7210edcb, 0x721edcb8, 0x720edcb9, 0x72edcb98, 0x710edcba, 0x71edcba8,
      0x70edcba9, 0x7edcba98, 0x6543210f, 0x654321f8, 0x654320f9, 0x65432f98,
      0x654310fa, 0x65431fa8, 0x65430fa9, 0x6543fa98, 0x654210fb, 0x65421fb8,
      0x65420fb9, 0x6542fb98, 0x65410fba, 0x6541fba8, 0x6540fba9, 0x654fba98,
      0x653210fc, 0x65321fc8, 0x65320fc9, 0x6532fc98, 0x65310fca, 0x6531fca8,
      0x6530fca9, 0x653fca98, 0x65210fcb, 0x6521fcb8, 0x6520fcb9, 0x652fcb98,
      0x6510fcba, 0x651fcba8, 0x650fcba9, 0x65fcba98, 0x643210fd, 0x64321fd8,
      0x64320fd9, 0x6432fd98, 0x64310fda, 0x6431fda8, 0x6430fda9, 0x643fda98,
      0x64210fdb, 0x6421fdb8, 0x6420fdb9, 0x642fdb98, 0x6410fdba, 0x641fdba8,
      0x640fdba9, 0x64fdba98, 0x63210fdc, 0x6321fdc8, 0x6320fdc9, 0x632fdc98,
      0x6310fdca, 0x631fdca8, 0x630fdca9, 0x63fdca98, 0x6210fdcb, 0x621fdcb8,
      0x620fdcb9, 0x62fdcb98, 0x610fdcba, 0x61fdcba8, 0x60fdcba9, 0x6fdcba98,
      0x543210fe, 0x54321fe8, 0x54320fe9, 0x5432fe98, 0x54310fea, 0x5431fea8,
      0x5430fea9, 0x543fea98, 0x54210feb, 0x5421feb8, 0x5420feb9, 0x542feb98,
      0x5410feba, 0x541feba8, 0x540feba9, 0x54feba98, 0x53210fec, 0x5321fec8,
      0x5320fec9, 0x532fec98, 0x5310feca, 0x531feca8, 0x530feca9, 0x53feca98,
      0x5210fecb, 0x521fecb8, 0x520fecb9, 0x52fecb98, 0x510fecba, 0x51fecba8,
      0x50fecba9, 0x5fecba98, 0x43210fed, 0x4321fed8, 0x4320fed9, 0x432fed98,
      0x4310feda, 0x431feda8, 0x430feda9, 0x43feda98, 0x4210fedb, 0x421fedb8,
      0x420fedb9, 0x42fedb98, 0x410fedba, 0x41fedba8, 0x40fedba9, 0x4fedba98,
      0x3210fedc, 0x321fedc8, 0x320fedc9, 0x32fedc98, 0x310fedca, 0x31fedca8,
      0x30fedca9, 0x3fedca98, 0x210fedcb, 0x21fedcb8, 0x20fedcb9, 0x2fedcb98,
      0x10fedcba, 0x1fedcba8, 0x0fedcba9, 0xfedcba98};

  // No need to mask because _mm256_permutevar8x32_epi32 ignores bits 3..31.
  // Just shift each copy of the 32 bit LUT to extract its 4-bit fields.
  // If broadcasting 32-bit from memory incurs the 3-cycle block-crossing
  // latency, it may be faster to use LoadDup128 and PSHUFB.
  const auto packed = Set(d32, packed_array[mask_bits]);
  alignas(32) static constexpr uint32_t shifts[8] = {0,  4,  8,  12,
                                                     16, 20, 24, 28};
  return packed >> Load(d32, shifts);
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Vec256<uint32_t> IndicesFromBits256(uint64_t mask_bits) {
  const Full256<uint32_t> d32;

  // For 64-bit, we still need 32-bit indices because there is no 64-bit
  // permutevar, but there are only 4 lanes, so we can afford to skip the
  // unpacking and load the entire index vector directly.
  alignas(32) static constexpr uint32_t u32_indices[128] = {
      // PrintCompress64x4PairTables
      0,  1,  2,  3,  4,  5,  6, 7, 8, 9, 2,  3,  4,  5,  6,  7,
      10, 11, 0,  1,  4,  5,  6, 7, 8, 9, 10, 11, 4,  5,  6,  7,
      12, 13, 0,  1,  2,  3,  6, 7, 8, 9, 12, 13, 2,  3,  6,  7,
      10, 11, 12, 13, 0,  1,  6, 7, 8, 9, 10, 11, 12, 13, 6,  7,
      14, 15, 0,  1,  2,  3,  4, 5, 8, 9, 14, 15, 2,  3,  4,  5,
      10, 11, 14, 15, 0,  1,  4, 5, 8, 9, 10, 11, 14, 15, 4,  5,
      12, 13, 14, 15, 0,  1,  2, 3, 8, 9, 12, 13, 14, 15, 2,  3,
      10, 11, 12, 13, 14, 15, 0, 1, 8, 9, 10, 11, 12, 13, 14, 15};
  return Load(d32, u32_indices + 8 * mask_bits);
}

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec256<uint32_t> IndicesFromNotBits256(uint64_t mask_bits) {
  const Full256<uint32_t> d32;
  // We need a masked Iota(). With 8 lanes, there are 256 combinations and a LUT
  // of SetTableIndices would require 8 KiB, a large part of L1D. The other
  // alternative is _pext_u64, but this is extremely slow on Zen2 (18 cycles)
  // and unavailable in 32-bit builds. We instead compress each index into 4
  // bits, for a total of 1 KiB.
  alignas(16) static constexpr uint32_t packed_array[256] = {
      // PrintCompressNot32x8Tables
      0xfedcba98, 0x8fedcba9, 0x9fedcba8, 0x98fedcba, 0xafedcb98, 0xa8fedcb9,
      0xa9fedcb8, 0xa98fedcb, 0xbfedca98, 0xb8fedca9, 0xb9fedca8, 0xb98fedca,
      0xbafedc98, 0xba8fedc9, 0xba9fedc8, 0xba98fedc, 0xcfedba98, 0xc8fedba9,
      0xc9fedba8, 0xc98fedba, 0xcafedb98, 0xca8fedb9, 0xca9fedb8, 0xca98fedb,
      0xcbfeda98, 0xcb8feda9, 0xcb9feda8, 0xcb98feda, 0xcbafed98, 0xcba8fed9,
      0xcba9fed8, 0xcba98fed, 0xdfecba98, 0xd8fecba9, 0xd9fecba8, 0xd98fecba,
      0xdafecb98, 0xda8fecb9, 0xda9fecb8, 0xda98fecb, 0xdbfeca98, 0xdb8feca9,
      0xdb9feca8, 0xdb98feca, 0xdbafec98, 0xdba8fec9, 0xdba9fec8, 0xdba98fec,
      0xdcfeba98, 0xdc8feba9, 0xdc9feba8, 0xdc98feba, 0xdcafeb98, 0xdca8feb9,
      0xdca9feb8, 0xdca98feb, 0xdcbfea98, 0xdcb8fea9, 0xdcb9fea8, 0xdcb98fea,
      0xdcbafe98, 0xdcba8fe9, 0xdcba9fe8, 0xdcba98fe, 0xefdcba98, 0xe8fdcba9,
      0xe9fdcba8, 0xe98fdcba, 0xeafdcb98, 0xea8fdcb9, 0xea9fdcb8, 0xea98fdcb,
      0xebfdca98, 0xeb8fdca9, 0xeb9fdca8, 0xeb98fdca, 0xebafdc98, 0xeba8fdc9,
      0xeba9fdc8, 0xeba98fdc, 0xecfdba98, 0xec8fdba9, 0xec9fdba8, 0xec98fdba,
      0xecafdb98, 0xeca8fdb9, 0xeca9fdb8, 0xeca98fdb, 0xecbfda98, 0xecb8fda9,
      0xecb9fda8, 0xecb98fda, 0xecbafd98, 0xecba8fd9, 0xecba9fd8, 0xecba98fd,
      0xedfcba98, 0xed8fcba9, 0xed9fcba8, 0xed98fcba, 0xedafcb98, 0xeda8fcb9,
      0xeda9fcb8, 0xeda98fcb, 0xedbfca98, 0xedb8fca9, 0xedb9fca8, 0xedb98fca,
      0xedbafc98, 0xedba8fc9, 0xedba9fc8, 0xedba98fc, 0xedcfba98, 0xedc8fba9,
      0xedc9fba8, 0xedc98fba, 0xedcafb98, 0xedca8fb9, 0xedca9fb8, 0xedca98fb,
      0xedcbfa98, 0xedcb8fa9, 0xedcb9fa8, 0xedcb98fa, 0xedcbaf98, 0xedcba8f9,
      0xedcba9f8, 0xedcba98f, 0xfedcba98, 0xf8edcba9, 0xf9edcba8, 0xf98edcba,
      0xfaedcb98, 0xfa8edcb9, 0xfa9edcb8, 0xfa98edcb, 0xfbedca98, 0xfb8edca9,
      0xfb9edca8, 0xfb98edca, 0xfbaedc98, 0xfba8edc9, 0xfba9edc8, 0xfba98edc,
      0xfcedba98, 0xfc8edba9, 0xfc9edba8, 0xfc98edba, 0xfcaedb98, 0xfca8edb9,
      0xfca9edb8, 0xfca98edb, 0xfcbeda98, 0xfcb8eda9, 0xfcb9eda8, 0xfcb98eda,
      0xfcbaed98, 0xfcba8ed9, 0xfcba9ed8, 0xfcba98ed, 0xfdecba98, 0xfd8ecba9,
      0xfd9ecba8, 0xfd98ecba, 0xfdaecb98, 0xfda8ecb9, 0xfda9ecb8, 0xfda98ecb,
      0xfdbeca98, 0xfdb8eca9, 0xfdb9eca8, 0xfdb98eca, 0xfdbaec98, 0xfdba8ec9,
      0xfdba9ec8, 0xfdba98ec, 0xfdceba98, 0xfdc8eba9, 0xfdc9eba8, 0xfdc98eba,
      0xfdcaeb98, 0xfdca8eb9, 0xfdca9eb8, 0xfdca98eb, 0xfdcbea98, 0xfdcb8ea9,
      0xfdcb9ea8, 0xfdcb98ea, 0xfdcbae98, 0xfdcba8e9, 0xfdcba9e8, 0xfdcba98e,
      0xfedcba98, 0xfe8dcba9, 0xfe9dcba8, 0xfe98dcba, 0xfeadcb98, 0xfea8dcb9,
      0xfea9dcb8, 0xfea98dcb, 0xfebdca98, 0xfeb8dca9, 0xfeb9dca8, 0xfeb98dca,
      0xfebadc98, 0xfeba8dc9, 0xfeba9dc8, 0xfeba98dc, 0xfecdba98, 0xfec8dba9,
      0xfec9dba8, 0xfec98dba, 0xfecadb98, 0xfeca8db9, 0xfeca9db8, 0xfeca98db,
      0xfecbda98, 0xfecb8da9, 0xfecb9da8, 0xfecb98da, 0xfecbad98, 0xfecba8d9,
      0xfecba9d8, 0xfecba98d, 0xfedcba98, 0xfed8cba9, 0xfed9cba8, 0xfed98cba,
      0xfedacb98, 0xfeda8cb9, 0xfeda9cb8, 0xfeda98cb, 0xfedbca98, 0xfedb8ca9,
      0xfedb9ca8, 0xfedb98ca, 0xfedbac98, 0xfedba8c9, 0xfedba9c8, 0xfedba98c,
      0xfedcba98, 0xfedc8ba9, 0xfedc9ba8, 0xfedc98ba, 0xfedcab98, 0xfedca8b9,
      0xfedca9b8, 0xfedca98b, 0xfedcba98, 0xfedcb8a9, 0xfedcb9a8, 0xfedcb98a,
      0xfedcba98, 0xfedcba89, 0xfedcba98, 0xfedcba98};

  // No need to mask because <_mm256_permutevar8x32_epi32> ignores bits 3..31.
  // Just shift each copy of the 32 bit LUT to extract its 4-bit fields.
  // If broadcasting 32-bit from memory incurs the 3-cycle block-crossing
  // latency, it may be faster to use LoadDup128 and PSHUFB.
  const Vec256<uint32_t> packed = Set(d32, packed_array[mask_bits]);
  alignas(32) static constexpr uint32_t shifts[8] = {0,  4,  8,  12,
                                                     16, 20, 24, 28};
  return packed >> Load(d32, shifts);
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Vec256<uint32_t> IndicesFromNotBits256(uint64_t mask_bits) {
  const Full256<uint32_t> d32;

  // For 64-bit, we still need 32-bit indices because there is no 64-bit
  // permutevar, but there are only 4 lanes, so we can afford to skip the
  // unpacking and load the entire index vector directly.
  alignas(32) static constexpr uint32_t u32_indices[128] = {
      // PrintCompressNot64x4PairTables
      8, 9, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 8,  9,
      8, 9, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 8,  9,  10, 11,
      8, 9, 10, 11, 14, 15, 12, 13, 10, 11, 14, 15, 8,  9,  12, 13,
      8, 9, 14, 15, 10, 11, 12, 13, 14, 15, 8,  9,  10, 11, 12, 13,
      8, 9, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 8,  9,  14, 15,
      8, 9, 12, 13, 10, 11, 14, 15, 12, 13, 8,  9,  10, 11, 14, 15,
      8, 9, 10, 11, 12, 13, 14, 15, 10, 11, 8,  9,  12, 13, 14, 15,
      8, 9, 10, 11, 12, 13, 14, 15, 8,  9,  10, 11, 12, 13, 14, 15};
  return Load(d32, u32_indices + 8 * mask_bits);
}

template <typename T, HWY_IF_NOT_T_SIZE(T, 2)>
HWY_INLINE Vec256<T> Compress(Vec256<T> v, const uint64_t mask_bits) {
  const DFromV<decltype(v)> d;
  const Repartition<uint32_t, decltype(d)> du32;

  HWY_DASSERT(mask_bits < (1ull << (32 / sizeof(T))));
  // 32-bit indices because we only have _mm256_permutevar8x32_epi32 (there is
  // no instruction for 4x64).
  const Indices256<uint32_t> indices{IndicesFromBits256<T>(mask_bits).raw};
  return BitCast(d, TableLookupLanes(BitCast(du32, v), indices));
}

// LUTs are infeasible for 2^16 possible masks, so splice together two
// half-vector Compress.
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_INLINE Vec256<T> Compress(Vec256<T> v, const uint64_t mask_bits) {
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du;
  const auto vu16 = BitCast(du, v);  // (required for float16_t inputs)
  const Half<decltype(du)> duh;
  const auto half0 = LowerHalf(duh, vu16);
  const auto half1 = UpperHalf(duh, vu16);

  const uint64_t mask_bits0 = mask_bits & 0xFF;
  const uint64_t mask_bits1 = mask_bits >> 8;
  const auto compressed0 = detail::CompressBits(half0, mask_bits0);
  const auto compressed1 = detail::CompressBits(half1, mask_bits1);

  alignas(32) uint16_t all_true[16] = {};
  // Store mask=true lanes, left to right.
  const size_t num_true0 = PopCount(mask_bits0);
  Store(compressed0, duh, all_true);
  StoreU(compressed1, duh, all_true + num_true0);

  if (hwy::HWY_NAMESPACE::CompressIsPartition<T>::value) {
    // Store mask=false lanes, right to left. The second vector fills the upper
    // half with right-aligned false lanes. The first vector is shifted
    // rightwards to overwrite the true lanes of the second.
    alignas(32) uint16_t all_false[16] = {};
    const size_t num_true1 = PopCount(mask_bits1);
    Store(compressed1, duh, all_false + 8);
    StoreU(compressed0, duh, all_false + num_true1);

    const auto mask = FirstN(du, num_true0 + num_true1);
    return BitCast(d,
                   IfThenElse(mask, Load(du, all_true), Load(du, all_false)));
  } else {
    // Only care about the mask=true lanes.
    return BitCast(d, Load(du, all_true));
  }
}

template <typename T, HWY_IF_T_SIZE_ONE_OF(T, (1 << 4) | (1 << 8))>
HWY_INLINE Vec256<T> CompressNot(Vec256<T> v, const uint64_t mask_bits) {
  const DFromV<decltype(v)> d;
  const Repartition<uint32_t, decltype(d)> du32;

  HWY_DASSERT(mask_bits < (1ull << (32 / sizeof(T))));
  // 32-bit indices because we only have _mm256_permutevar8x32_epi32 (there is
  // no instruction for 4x64).
  const Indices256<uint32_t> indices{IndicesFromNotBits256<T>(mask_bits).raw};
  return BitCast(d, TableLookupLanes(BitCast(du32, v), indices));
}

// LUTs are infeasible for 2^16 possible masks, so splice together two
// half-vector Compress.
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_INLINE Vec256<T> CompressNot(Vec256<T> v, const uint64_t mask_bits) {
  // Compress ensures only the lower 16 bits are set, so flip those.
  return Compress(v, mask_bits ^ 0xFFFF);
}

}  // namespace detail

template <typename T, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API Vec256<T> Compress(Vec256<T> v, Mask256<T> m) {
  return detail::Compress(v, detail::BitsFromMask(m));
}

template <typename T, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API Vec256<T> CompressNot(Vec256<T> v, Mask256<T> m) {
  return detail::CompressNot(v, detail::BitsFromMask(m));
}

HWY_API Vec256<uint64_t> CompressBlocksNot(Vec256<uint64_t> v,
                                           Mask256<uint64_t> mask) {
  return CompressNot(v, mask);
}

template <typename T, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API Vec256<T> CompressBits(Vec256<T> v, const uint8_t* HWY_RESTRICT bits) {
  constexpr size_t N = 32 / sizeof(T);
  constexpr size_t kNumBytes = (N + 7) / 8;

  uint64_t mask_bits = 0;
  CopyBytes<kNumBytes>(bits, &mask_bits);

  if (N < 8) {
    mask_bits &= (1ull << N) - 1;
  }

  return detail::Compress(v, mask_bits);
}

// ------------------------------ CompressStore, CompressBitsStore

template <class D, typename T = TFromD<D>, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API size_t CompressStore(Vec256<T> v, Mask256<T> m, D d,
                             T* HWY_RESTRICT unaligned) {
  const uint64_t mask_bits = detail::BitsFromMask(m);
  const size_t count = PopCount(mask_bits);
  StoreU(detail::Compress(v, mask_bits), d, unaligned);
  detail::MaybeUnpoison(unaligned, count);
  return count;
}

template <class D, typename T = TFromD<D>,
          HWY_IF_T_SIZE_ONE_OF(T, (1 << 4) | (1 << 8))>
HWY_API size_t CompressBlendedStore(Vec256<T> v, Mask256<T> m, D d,
                                    T* HWY_RESTRICT unaligned) {
  const uint64_t mask_bits = detail::BitsFromMask(m);
  const size_t count = PopCount(mask_bits);

  const Repartition<uint32_t, decltype(d)> du32;
  HWY_DASSERT(mask_bits < (1ull << (32 / sizeof(T))));
  // 32-bit indices because we only have _mm256_permutevar8x32_epi32 (there is
  // no instruction for 4x64). Nibble MSB encodes FirstN.
  const Vec256<uint32_t> idx_mask = detail::IndicesFromBits256<T>(mask_bits);
  // Shift nibble MSB into MSB
  const Mask256<uint32_t> mask32 = MaskFromVec(ShiftLeft<28>(idx_mask));
  // First cast to unsigned (RebindMask cannot change lane size)
  const Mask256<MakeUnsigned<T>> mask_u{mask32.raw};
  const Mask256<T> mask = RebindMask(d, mask_u);
  const Vec256<T> compressed = BitCast(
      d,
      TableLookupLanes(BitCast(du32, v), Indices256<uint32_t>{idx_mask.raw}));

  BlendedStore(compressed, mask, d, unaligned);
  detail::MaybeUnpoison(unaligned, count);
  return count;
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API size_t CompressBlendedStore(Vec256<T> v, Mask256<T> m, D d,
                                    T* HWY_RESTRICT unaligned) {
  const uint64_t mask_bits = detail::BitsFromMask(m);
  const size_t count = PopCount(mask_bits);
  const Vec256<T> compressed = detail::Compress(v, mask_bits);

#if HWY_MEM_OPS_MIGHT_FAULT  // true if HWY_IS_MSAN
  // BlendedStore tests mask for each lane, but we know that the mask is
  // FirstN, so we can just copy.
  alignas(32) T buf[16];
  Store(compressed, d, buf);
  memcpy(unaligned, buf, count * sizeof(T));
#else
  BlendedStore(compressed, FirstN(d, count), d, unaligned);
#endif
  return count;
}

template <typename T, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API size_t CompressBitsStore(Vec256<T> v, const uint8_t* HWY_RESTRICT bits,
                                 Full256<T> d, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 32 / sizeof(T);
  constexpr size_t kNumBytes = (N + 7) / 8;

  uint64_t mask_bits = 0;
  CopyBytes<kNumBytes>(bits, &mask_bits);

  if (N < 8) {
    mask_bits &= (1ull << N) - 1;
  }
  const size_t count = PopCount(mask_bits);

  StoreU(detail::Compress(v, mask_bits), d, unaligned);
  detail::MaybeUnpoison(unaligned, count);
  return count;
}

#endif  // HWY_TARGET <= HWY_AVX3

// ------------------------------ Expand

// Always define Expand/LoadExpand because generic_ops only does so for Vec128.

namespace detail {

#if HWY_TARGET <= HWY_AVX3_DL || HWY_IDE  // VBMI2

HWY_INLINE Vec256<uint8_t> NativeExpand(Vec256<uint8_t> v,
                                        Mask256<uint8_t> mask) {
  return Vec256<uint8_t>{_mm256_maskz_expand_epi8(mask.raw, v.raw)};
}

HWY_INLINE Vec256<uint16_t> NativeExpand(Vec256<uint16_t> v,
                                         Mask256<uint16_t> mask) {
  return Vec256<uint16_t>{_mm256_maskz_expand_epi16(mask.raw, v.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_U8_D(D)>
HWY_INLINE Vec256<uint8_t> NativeLoadExpand(
    Mask256<uint8_t> mask, D /* d */, const uint8_t* HWY_RESTRICT unaligned) {
  return Vec256<uint8_t>{_mm256_maskz_expandloadu_epi8(mask.raw, unaligned)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_U16_D(D)>
HWY_INLINE Vec256<uint16_t> NativeLoadExpand(
    Mask256<uint16_t> mask, D /* d */, const uint16_t* HWY_RESTRICT unaligned) {
  return Vec256<uint16_t>{_mm256_maskz_expandloadu_epi16(mask.raw, unaligned)};
}

#endif  // HWY_TARGET <= HWY_AVX3_DL
#if HWY_TARGET <= HWY_AVX3 || HWY_IDE

HWY_INLINE Vec256<uint32_t> NativeExpand(Vec256<uint32_t> v,
                                         Mask256<uint32_t> mask) {
  return Vec256<uint32_t>{_mm256_maskz_expand_epi32(mask.raw, v.raw)};
}

HWY_INLINE Vec256<uint64_t> NativeExpand(Vec256<uint64_t> v,
                                         Mask256<uint64_t> mask) {
  return Vec256<uint64_t>{_mm256_maskz_expand_epi64(mask.raw, v.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_U32_D(D)>
HWY_INLINE Vec256<uint32_t> NativeLoadExpand(
    Mask256<uint32_t> mask, D /* d */, const uint32_t* HWY_RESTRICT unaligned) {
  return Vec256<uint32_t>{_mm256_maskz_expandloadu_epi32(mask.raw, unaligned)};
}

template <class D, HWY_IF_V_SIZE_D(D, 32), HWY_IF_U64_D(D)>
HWY_INLINE Vec256<uint64_t> NativeLoadExpand(
    Mask256<uint64_t> mask, D /* d */, const uint64_t* HWY_RESTRICT unaligned) {
  return Vec256<uint64_t>{_mm256_maskz_expandloadu_epi64(mask.raw, unaligned)};
}

#endif  // HWY_TARGET <= HWY_AVX3

}  // namespace detail

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec256<T> Expand(Vec256<T> v, Mask256<T> mask) {
  const DFromV<decltype(v)> d;
#if HWY_TARGET <= HWY_AVX3_DL  // VBMI2
  const RebindToUnsigned<decltype(d)> du;
  const MFromD<decltype(du)> mu = RebindMask(du, mask);
  return BitCast(d, detail::NativeExpand(BitCast(du, v), mu));
#else
  // LUTs are infeasible for so many mask combinations, so Combine two
  // half-vector Expand.
  const Half<decltype(d)> dh;
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  constexpr size_t N = 32 / sizeof(T);
  const size_t countL = PopCount(mask_bits & ((1 << (N / 2)) - 1));
  const Mask128<T> maskL = MaskFromVec(LowerHalf(VecFromMask(d, mask)));
  const Vec128<T> expandL = Expand(LowerHalf(v), maskL);
  // We have to shift the input by a variable number of bytes, but there isn't
  // a table-driven option for that until VBMI, and CPUs with that likely also
  // have VBMI2 and thus native Expand.
  alignas(32) T lanes[N];
  Store(v, d, lanes);
  const Mask128<T> maskH = MaskFromVec(UpperHalf(dh, VecFromMask(d, mask)));
  const Vec128<T> expandH = Expand(LoadU(dh, lanes + countL), maskH);
  return Combine(d, expandH, expandL);
#endif
}

// If AVX3, this is already implemented by x86_512.
#if HWY_TARGET != HWY_AVX3

template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec256<T> Expand(Vec256<T> v, Mask256<T> mask) {
  const Full256<T> d;
#if HWY_TARGET <= HWY_AVX3_DL  // VBMI2
  const RebindToUnsigned<decltype(d)> du;
  return BitCast(d, detail::NativeExpand(BitCast(du, v), RebindMask(du, mask)));
#else   // AVX2
  // LUTs are infeasible for 2^16 possible masks, so splice together two
  // half-vector Expand.
  const Half<decltype(d)> dh;
  const Mask128<T> maskL = MaskFromVec(LowerHalf(VecFromMask(d, mask)));
  const Vec128<T> expandL = Expand(LowerHalf(v), maskL);
  // We have to shift the input by a variable number of u16. permutevar_epi16
  // requires AVX3 and if we had that, we'd use native u32 Expand. The only
  // alternative is re-loading, which incurs a store to load forwarding stall.
  alignas(32) T lanes[32 / sizeof(T)];
  Store(v, d, lanes);
  const Vec128<T> vH = LoadU(dh, lanes + CountTrue(dh, maskL));
  const Mask128<T> maskH = MaskFromVec(UpperHalf(dh, VecFromMask(d, mask)));
  const Vec128<T> expandH = Expand(vH, maskH);
  return Combine(d, expandH, expandL);
#endif  // AVX2
}

#endif  // HWY_TARGET != HWY_AVX3

template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec256<T> Expand(Vec256<T> v, Mask256<T> mask) {
  const Full256<T> d;
#if HWY_TARGET <= HWY_AVX3
  const RebindToUnsigned<decltype(d)> du;
  const MFromD<decltype(du)> mu = RebindMask(du, mask);
  return BitCast(d, detail::NativeExpand(BitCast(du, v), mu));
#else
  const RebindToUnsigned<decltype(d)> du;
  const uint64_t mask_bits = detail::BitsFromMask(mask);

  alignas(16) constexpr uint32_t packed_array[256] = {
      // PrintExpand32x8Nibble.
      0xffffffff, 0xfffffff0, 0xffffff0f, 0xffffff10, 0xfffff0ff, 0xfffff1f0,
      0xfffff10f, 0xfffff210, 0xffff0fff, 0xffff1ff0, 0xffff1f0f, 0xffff2f10,
      0xffff10ff, 0xffff21f0, 0xffff210f, 0xffff3210, 0xfff0ffff, 0xfff1fff0,
      0xfff1ff0f, 0xfff2ff10, 0xfff1f0ff, 0xfff2f1f0, 0xfff2f10f, 0xfff3f210,
      0xfff10fff, 0xfff21ff0, 0xfff21f0f, 0xfff32f10, 0xfff210ff, 0xfff321f0,
      0xfff3210f, 0xfff43210, 0xff0fffff, 0xff1ffff0, 0xff1fff0f, 0xff2fff10,
      0xff1ff0ff, 0xff2ff1f0, 0xff2ff10f, 0xff3ff210, 0xff1f0fff, 0xff2f1ff0,
      0xff2f1f0f, 0xff3f2f10, 0xff2f10ff, 0xff3f21f0, 0xff3f210f, 0xff4f3210,
      0xff10ffff, 0xff21fff0, 0xff21ff0f, 0xff32ff10, 0xff21f0ff, 0xff32f1f0,
      0xff32f10f, 0xff43f210, 0xff210fff, 0xff321ff0, 0xff321f0f, 0xff432f10,
      0xff3210ff, 0xff4321f0, 0xff43210f, 0xff543210, 0xf0ffffff, 0xf1fffff0,
      0xf1ffff0f, 0xf2ffff10, 0xf1fff0ff, 0xf2fff1f0, 0xf2fff10f, 0xf3fff210,
      0xf1ff0fff, 0xf2ff1ff0, 0xf2ff1f0f, 0xf3ff2f10, 0xf2ff10ff, 0xf3ff21f0,
      0xf3ff210f, 0xf4ff3210, 0xf1f0ffff, 0xf2f1fff0, 0xf2f1ff0f, 0xf3f2ff10,
      0xf2f1f0ff, 0xf3f2f1f0, 0xf3f2f10f, 0xf4f3f210, 0xf2f10fff, 0xf3f21ff0,
      0xf3f21f0f, 0xf4f32f10, 0xf3f210ff, 0xf4f321f0, 0xf4f3210f, 0xf5f43210,
      0xf10fffff, 0xf21ffff0, 0xf21fff0f, 0xf32fff10, 0xf21ff0ff, 0xf32ff1f0,
      0xf32ff10f, 0xf43ff210, 0xf21f0fff, 0xf32f1ff0, 0xf32f1f0f, 0xf43f2f10,
      0xf32f10ff, 0xf43f21f0, 0xf43f210f, 0xf54f3210, 0xf210ffff, 0xf321fff0,
      0xf321ff0f, 0xf432ff10, 0xf321f0ff, 0xf432f1f0, 0xf432f10f, 0xf543f210,
      0xf3210fff, 0xf4321ff0, 0xf4321f0f, 0xf5432f10, 0xf43210ff, 0xf54321f0,
      0xf543210f, 0xf6543210, 0x0fffffff, 0x1ffffff0, 0x1fffff0f, 0x2fffff10,
      0x1ffff0ff, 0x2ffff1f0, 0x2ffff10f, 0x3ffff210, 0x1fff0fff, 0x2fff1ff0,
      0x2fff1f0f, 0x3fff2f10, 0x2fff10ff, 0x3fff21f0, 0x3fff210f, 0x4fff3210,
      0x1ff0ffff, 0x2ff1fff0, 0x2ff1ff0f, 0x3ff2ff10, 0x2ff1f0ff, 0x3ff2f1f0,
      0x3ff2f10f, 0x4ff3f210, 0x2ff10fff, 0x3ff21ff0, 0x3ff21f0f, 0x4ff32f10,
      0x3ff210ff, 0x4ff321f0, 0x4ff3210f, 0x5ff43210, 0x1f0fffff, 0x2f1ffff0,
      0x2f1fff0f, 0x3f2fff10, 0x2f1ff0ff, 0x3f2ff1f0, 0x3f2ff10f, 0x4f3ff210,
      0x2f1f0fff, 0x3f2f1ff0, 0x3f2f1f0f, 0x4f3f2f10, 0x3f2f10ff, 0x4f3f21f0,
      0x4f3f210f, 0x5f4f3210, 0x2f10ffff, 0x3f21fff0, 0x3f21ff0f, 0x4f32ff10,
      0x3f21f0ff, 0x4f32f1f0, 0x4f32f10f, 0x5f43f210, 0x3f210fff, 0x4f321ff0,
      0x4f321f0f, 0x5f432f10, 0x4f3210ff, 0x5f4321f0, 0x5f43210f, 0x6f543210,
      0x10ffffff, 0x21fffff0, 0x21ffff0f, 0x32ffff10, 0x21fff0ff, 0x32fff1f0,
      0x32fff10f, 0x43fff210, 0x21ff0fff, 0x32ff1ff0, 0x32ff1f0f, 0x43ff2f10,
      0x32ff10ff, 0x43ff21f0, 0x43ff210f, 0x54ff3210, 0x21f0ffff, 0x32f1fff0,
      0x32f1ff0f, 0x43f2ff10, 0x32f1f0ff, 0x43f2f1f0, 0x43f2f10f, 0x54f3f210,
      0x32f10fff, 0x43f21ff0, 0x43f21f0f, 0x54f32f10, 0x43f210ff, 0x54f321f0,
      0x54f3210f, 0x65f43210, 0x210fffff, 0x321ffff0, 0x321fff0f, 0x432fff10,
      0x321ff0ff, 0x432ff1f0, 0x432ff10f, 0x543ff210, 0x321f0fff, 0x432f1ff0,
      0x432f1f0f, 0x543f2f10, 0x432f10ff, 0x543f21f0, 0x543f210f, 0x654f3210,
      0x3210ffff, 0x4321fff0, 0x4321ff0f, 0x5432ff10, 0x4321f0ff, 0x5432f1f0,
      0x5432f10f, 0x6543f210, 0x43210fff, 0x54321ff0, 0x54321f0f, 0x65432f10,
      0x543210ff, 0x654321f0, 0x6543210f, 0x76543210,
  };

  // For lane i, shift the i-th 4-bit index down to bits [0, 3).
  const Vec256<uint32_t> packed = Set(du, packed_array[mask_bits]);
  alignas(32) constexpr uint32_t shifts[8] = {0, 4, 8, 12, 16, 20, 24, 28};
  // TableLookupLanes ignores upper bits; avoid bounds-check in IndicesFromVec.
  const Indices256<uint32_t> indices{(packed >> Load(du, shifts)).raw};
  const Vec256<uint32_t> expand = TableLookupLanes(BitCast(du, v), indices);
  // TableLookupLanes cannot also zero masked-off lanes, so do that now.
  return IfThenElseZero(mask, BitCast(d, expand));
#endif
}

template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec256<T> Expand(Vec256<T> v, Mask256<T> mask) {
  const Full256<T> d;
#if HWY_TARGET <= HWY_AVX3
  const RebindToUnsigned<decltype(d)> du;
  const MFromD<decltype(du)> mu = RebindMask(du, mask);
  return BitCast(d, detail::NativeExpand(BitCast(du, v), mu));
#else
  const RebindToUnsigned<decltype(d)> du;
  const uint64_t mask_bits = detail::BitsFromMask(mask);

  alignas(16) constexpr uint64_t packed_array[16] = {
      // PrintExpand64x4Nibble.
      0x0000ffff, 0x0000fff0, 0x0000ff0f, 0x0000ff10, 0x0000f0ff, 0x0000f1f0,
      0x0000f10f, 0x0000f210, 0x00000fff, 0x00001ff0, 0x00001f0f, 0x00002f10,
      0x000010ff, 0x000021f0, 0x0000210f, 0x00003210};

  // For lane i, shift the i-th 4-bit index down to bits [0, 2).
  const Vec256<uint64_t> packed = Set(du, packed_array[mask_bits]);
  alignas(32) constexpr uint64_t shifts[8] = {0, 4, 8, 12, 16, 20, 24, 28};
#if HWY_TARGET <= HWY_AVX3  // native 64-bit TableLookupLanes
  // TableLookupLanes ignores upper bits; avoid bounds-check in IndicesFromVec.
  const Indices256<uint64_t> indices{(packed >> Load(du, shifts)).raw};
#else
  // 64-bit TableLookupLanes on AVX2 requires IndicesFromVec, which checks
  // bounds, so clear the upper bits.
  const Vec256<uint64_t> masked = And(packed >> Load(du, shifts), Set(du, 3));
  const Indices256<uint64_t> indices = IndicesFromVec(du, masked);
#endif
  const Vec256<uint64_t> expand = TableLookupLanes(BitCast(du, v), indices);
  // TableLookupLanes cannot also zero masked-off lanes, so do that now.
  return IfThenElseZero(mask, BitCast(d, expand));
#endif
}

// ------------------------------ LoadExpand

template <class D, HWY_IF_V_SIZE_D(D, 32),
          HWY_IF_T_SIZE_ONE_OF_D(D, (1 << 1) | (1 << 2))>
HWY_API VFromD<D> LoadExpand(MFromD<D> mask, D d,
                             const TFromD<D>* HWY_RESTRICT unaligned) {
#if HWY_TARGET <= HWY_AVX3_DL  // VBMI2
  const RebindToUnsigned<decltype(d)> du;
  using TU = TFromD<decltype(du)>;
  const TU* HWY_RESTRICT pu = reinterpret_cast<const TU*>(unaligned);
  const MFromD<decltype(du)> mu = RebindMask(du, mask);
  return BitCast(d, detail::NativeLoadExpand(mu, du, pu));
#else
  return Expand(LoadU(d, unaligned), mask);
#endif
}

template <class D, HWY_IF_V_SIZE_D(D, 32),
          HWY_IF_T_SIZE_ONE_OF_D(D, (1 << 4) | (1 << 8))>
HWY_API VFromD<D> LoadExpand(MFromD<D> mask, D d,
                             const TFromD<D>* HWY_RESTRICT unaligned) {
#if HWY_TARGET <= HWY_AVX3
  const RebindToUnsigned<decltype(d)> du;
  using TU = TFromD<decltype(du)>;
  const TU* HWY_RESTRICT pu = reinterpret_cast<const TU*>(unaligned);
  const MFromD<decltype(du)> mu = RebindMask(du, mask);
  return BitCast(d, detail::NativeLoadExpand(mu, du, pu));
#else
  return Expand(LoadU(d, unaligned), mask);
#endif
}

// ------------------------------ LoadInterleaved3/4

// Implemented in generic_ops, we just overload LoadTransposedBlocks3/4.

namespace detail {
// Input:
// 1 0 (<- first block of unaligned)
// 3 2
// 5 4
// Output:
// 3 0
// 4 1
// 5 2
template <class D, typename T = TFromD<D>>
HWY_API void LoadTransposedBlocks3(D d, const T* HWY_RESTRICT unaligned,
                                   Vec256<T>& A, Vec256<T>& B, Vec256<T>& C) {
  constexpr size_t N = 32 / sizeof(T);
  const Vec256<T> v10 = LoadU(d, unaligned + 0 * N);  // 1 0
  const Vec256<T> v32 = LoadU(d, unaligned + 1 * N);
  const Vec256<T> v54 = LoadU(d, unaligned + 2 * N);

  A = ConcatUpperLower(d, v32, v10);
  B = ConcatLowerUpper(d, v54, v10);
  C = ConcatUpperLower(d, v54, v32);
}

// Input (128-bit blocks):
// 1 0 (first block of unaligned)
// 3 2
// 5 4
// 7 6
// Output:
// 4 0 (LSB of vA)
// 5 1
// 6 2
// 7 3
template <class D, typename T = TFromD<D>>
HWY_API void LoadTransposedBlocks4(D d, const T* HWY_RESTRICT unaligned,
                                   Vec256<T>& vA, Vec256<T>& vB, Vec256<T>& vC,
                                   Vec256<T>& vD) {
  constexpr size_t N = 32 / sizeof(T);
  const Vec256<T> v10 = LoadU(d, unaligned + 0 * N);
  const Vec256<T> v32 = LoadU(d, unaligned + 1 * N);
  const Vec256<T> v54 = LoadU(d, unaligned + 2 * N);
  const Vec256<T> v76 = LoadU(d, unaligned + 3 * N);

  vA = ConcatLowerLower(d, v54, v10);
  vB = ConcatUpperUpper(d, v54, v10);
  vC = ConcatLowerLower(d, v76, v32);
  vD = ConcatUpperUpper(d, v76, v32);
}
}  // namespace detail

// ------------------------------ StoreInterleaved2/3/4 (ConcatUpperLower)

// Implemented in generic_ops, we just overload StoreTransposedBlocks2/3/4.

namespace detail {
// Input (128-bit blocks):
// 2 0 (LSB of i)
// 3 1
// Output:
// 1 0
// 3 2
template <class D, typename T = TFromD<D>>
HWY_API void StoreTransposedBlocks2(Vec256<T> i, Vec256<T> j, D d,
                                    T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 32 / sizeof(T);
  const auto out0 = ConcatLowerLower(d, j, i);
  const auto out1 = ConcatUpperUpper(d, j, i);
  StoreU(out0, d, unaligned + 0 * N);
  StoreU(out1, d, unaligned + 1 * N);
}

// Input (128-bit blocks):
// 3 0 (LSB of i)
// 4 1
// 5 2
// Output:
// 1 0
// 3 2
// 5 4
template <class D, typename T = TFromD<D>>
HWY_API void StoreTransposedBlocks3(Vec256<T> i, Vec256<T> j, Vec256<T> k, D d,
                                    T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 32 / sizeof(T);
  const auto out0 = ConcatLowerLower(d, j, i);
  const auto out1 = ConcatUpperLower(d, i, k);
  const auto out2 = ConcatUpperUpper(d, k, j);
  StoreU(out0, d, unaligned + 0 * N);
  StoreU(out1, d, unaligned + 1 * N);
  StoreU(out2, d, unaligned + 2 * N);
}

// Input (128-bit blocks):
// 4 0 (LSB of i)
// 5 1
// 6 2
// 7 3
// Output:
// 1 0
// 3 2
// 5 4
// 7 6
template <class D, typename T = TFromD<D>>
HWY_API void StoreTransposedBlocks4(Vec256<T> i, Vec256<T> j, Vec256<T> k,
                                    Vec256<T> l, D d,
                                    T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 32 / sizeof(T);
  // Write lower halves, then upper.
  const auto out0 = ConcatLowerLower(d, j, i);
  const auto out1 = ConcatLowerLower(d, l, k);
  StoreU(out0, d, unaligned + 0 * N);
  StoreU(out1, d, unaligned + 1 * N);
  const auto out2 = ConcatUpperUpper(d, j, i);
  const auto out3 = ConcatUpperUpper(d, l, k);
  StoreU(out2, d, unaligned + 2 * N);
  StoreU(out3, d, unaligned + 3 * N);
}
}  // namespace detail

// ------------------------------ Reductions

namespace detail {

// Returns sum{lane[i]} in each lane. "v3210" is a replicated 128-bit block.
// Same logic as x86/128.h, but with Vec256 arguments.
template <typename T>
HWY_INLINE Vec256<T> SumOfLanes(hwy::SizeTag<4> /* tag */,
                                const Vec256<T> v3210) {
  const auto v1032 = Shuffle1032(v3210);
  const auto v31_20_31_20 = v3210 + v1032;
  const auto v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}
template <typename T>
HWY_INLINE Vec256<T> MinOfLanes(hwy::SizeTag<4> /* tag */,
                                const Vec256<T> v3210) {
  const auto v1032 = Shuffle1032(v3210);
  const auto v31_20_31_20 = Min(v3210, v1032);
  const auto v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return Min(v20_31_20_31, v31_20_31_20);
}
template <typename T>
HWY_INLINE Vec256<T> MaxOfLanes(hwy::SizeTag<4> /* tag */,
                                const Vec256<T> v3210) {
  const auto v1032 = Shuffle1032(v3210);
  const auto v31_20_31_20 = Max(v3210, v1032);
  const auto v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return Max(v20_31_20_31, v31_20_31_20);
}

template <typename T>
HWY_INLINE Vec256<T> SumOfLanes(hwy::SizeTag<8> /* tag */,
                                const Vec256<T> v10) {
  const auto v01 = Shuffle01(v10);
  return v10 + v01;
}
template <typename T>
HWY_INLINE Vec256<T> MinOfLanes(hwy::SizeTag<8> /* tag */,
                                const Vec256<T> v10) {
  const auto v01 = Shuffle01(v10);
  return Min(v10, v01);
}
template <typename T>
HWY_INLINE Vec256<T> MaxOfLanes(hwy::SizeTag<8> /* tag */,
                                const Vec256<T> v10) {
  const auto v01 = Shuffle01(v10);
  return Max(v10, v01);
}

HWY_API Vec256<uint16_t> SumOfLanes(hwy::SizeTag<2> /* tag */,
                                    Vec256<uint16_t> v) {
  const Full256<uint16_t> d;
  const RepartitionToWide<decltype(d)> d32;
  const auto even = And(BitCast(d32, v), Set(d32, 0xFFFF));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto sum = SumOfLanes(hwy::SizeTag<4>(), even + odd);
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(sum)), BitCast(d, sum));
}
HWY_API Vec256<int16_t> SumOfLanes(hwy::SizeTag<2> /* tag */,
                                   Vec256<int16_t> v) {
  const Full256<int16_t> d;
  const RepartitionToWide<decltype(d)> d32;
  // Sign-extend
  const auto even = ShiftRight<16>(ShiftLeft<16>(BitCast(d32, v)));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto sum = SumOfLanes(hwy::SizeTag<4>(), even + odd);
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(sum)), BitCast(d, sum));
}

HWY_API Vec256<uint16_t> MinOfLanes(hwy::SizeTag<2> /* tag */,
                                    Vec256<uint16_t> v) {
  const Full256<uint16_t> d;
  const RepartitionToWide<decltype(d)> d32;
  const auto even = And(BitCast(d32, v), Set(d32, 0xFFFF));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MinOfLanes(hwy::SizeTag<4>(), Min(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}
HWY_API Vec256<int16_t> MinOfLanes(hwy::SizeTag<2> /* tag */,
                                   Vec256<int16_t> v) {
  const Full256<int16_t> d;
  const RepartitionToWide<decltype(d)> d32;
  // Sign-extend
  const auto even = ShiftRight<16>(ShiftLeft<16>(BitCast(d32, v)));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MinOfLanes(hwy::SizeTag<4>(), Min(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}

HWY_API Vec256<uint16_t> MaxOfLanes(hwy::SizeTag<2> /* tag */,
                                    Vec256<uint16_t> v) {
  const Full256<uint16_t> d;
  const RepartitionToWide<decltype(d)> d32;
  const auto even = And(BitCast(d32, v), Set(d32, 0xFFFF));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MaxOfLanes(hwy::SizeTag<4>(), Max(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}
HWY_API Vec256<int16_t> MaxOfLanes(hwy::SizeTag<2> /* tag */,
                                   Vec256<int16_t> v) {
  const Full256<int16_t> d;
  const RepartitionToWide<decltype(d)> d32;
  // Sign-extend
  const auto even = ShiftRight<16>(ShiftLeft<16>(BitCast(d32, v)));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MaxOfLanes(hwy::SizeTag<4>(), Max(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}
}  // namespace detail

// Supported for {uif}{32,64},{ui}16. Returns the broadcasted result.
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> SumOfLanes(D d, const Vec256<T> vHL) {
  const Vec256<T> vLH = ConcatLowerUpper(d, vHL, vHL);
  return detail::SumOfLanes(hwy::SizeTag<sizeof(T)>(), vLH + vHL);
}
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> MinOfLanes(D d, const Vec256<T> vHL) {
  const Vec256<T> vLH = ConcatLowerUpper(d, vHL, vHL);
  return detail::MinOfLanes(hwy::SizeTag<sizeof(T)>(), Min(vLH, vHL));
}
template <class D, typename T = TFromD<D>>
HWY_API Vec256<T> MaxOfLanes(D d, const Vec256<T> vHL) {
  const Vec256<T> vLH = ConcatLowerUpper(d, vHL, vHL);
  return detail::MaxOfLanes(hwy::SizeTag<sizeof(T)>(), Max(vLH, vHL));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

// Note that the GCC warnings are not suppressed if we only wrap the *intrin.h -
// the warning seems to be issued at the call site of intrinsics, i.e. our code.
HWY_DIAGNOSTICS(pop)
