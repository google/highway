// Copyright 2023 Google LLC
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

// 128-bit vectors for VSX
// External include guard in highway.h - see comment there.

// Must come before HWY_DIAGNOSTICS and HWY_COMPILER_GCC_ACTUAL
#include "hwy/base.h"

#pragma push_macro("vector")
#pragma push_macro("pixel")
#pragma push_macro("bool")

#undef vector
#undef pixel
#undef bool

#include <altivec.h>

#pragma pop_macro("vector")
#pragma pop_macro("pixel")
#pragma pop_macro("bool")

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include "hwy/ops/shared-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace detail {

template <typename T>
struct Raw128;

// Each Raw128 specialization defines the following typedefs:
// - type:
//   the backing Altivec/VSX raw vector type of the Vec128<T, N> type
// - RawBoolVectType:
//   the backing Altivec/VSX raw __bool vector type of the Mask128<T, N> type
// - RawVectLaneType:
//   the lane type of Raw128<T>::type
// - AlignedLoadStoreRawVectType:
//   the GCC/Clang vector type that is used for an aligned load of a full
//   16-byte vector
// - UnalignedLoadStoreRawVectType:
//   the GCC/Clang vector type that is used for an unaligned load of a full
//   16-byte vector
#define HWY_VSX_RAW128_SPECIALIZE(LANE_TYPE, RAW_VECT_LANE_TYPE, \
                                  RAW_BOOL_VECT_LANE_TYPE) \
template<> \
struct Raw128<LANE_TYPE> { \
  using type = __vector RAW_VECT_LANE_TYPE; \
  using RawBoolVectType = __vector __bool RAW_BOOL_VECT_LANE_TYPE; \
  using RawVectLaneType = RAW_VECT_LANE_TYPE; \
  typedef LANE_TYPE AlignedLoadStoreRawVectType __attribute__(( \
    __vector_size__(16), __aligned__(16), __may_alias__)); \
  typedef LANE_TYPE UnalignedLoadStoreRawVectType __attribute__(( \
    __vector_size__(16), __aligned__(alignof(LANE_TYPE)), __may_alias__)); \
};

HWY_VSX_RAW128_SPECIALIZE(int8_t, signed char, char)
HWY_VSX_RAW128_SPECIALIZE(uint8_t, unsigned char, char)
HWY_VSX_RAW128_SPECIALIZE(int16_t, signed short, short)
HWY_VSX_RAW128_SPECIALIZE(uint16_t, unsigned short, short)
HWY_VSX_RAW128_SPECIALIZE(int32_t, signed int, int)
HWY_VSX_RAW128_SPECIALIZE(uint32_t, unsigned int, int)
HWY_VSX_RAW128_SPECIALIZE(int64_t, signed long long, long long)
HWY_VSX_RAW128_SPECIALIZE(uint64_t, unsigned long long, long long)
HWY_VSX_RAW128_SPECIALIZE(float, float, int)
HWY_VSX_RAW128_SPECIALIZE(double, double, long long)

template<>
struct Raw128<bfloat16_t> : public Raw128<uint16_t> {};

template<>
struct Raw128<float16_t> : public Raw128<uint16_t> {};

#undef HWY_ALTIVEC_RAW128_SPECIALIZE

}  // namespace detail

template <typename T, size_t N = 16 / sizeof(T)>
class Vec128 {
  using Raw = typename detail::Raw128<T>::type;

 public:
  using PrivateT = T;                     // only for DFromV
  static constexpr size_t kPrivateN = N;  // only for DFromV

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

template <typename T>
using Vec64 = Vec128<T, 8 / sizeof(T)>;

template <typename T>
using Vec32 = Vec128<T, 4 / sizeof(T)>;

template <typename T>
using Vec16 = Vec128<T, 2 / sizeof(T)>;

// FF..FF or 0.
template <typename T, size_t N = 16 / sizeof(T)>
struct Mask128 {
  typename detail::Raw128<T>::RawBoolVectType raw;
};

template <class V>
using DFromV = Simd<typename V::PrivateT, V::kPrivateN, 0>;

template <class V>
using TFromV = typename V::PrivateT;

// ------------------------------ Zero

// Returns an all-zero vector/part.
template <class D>
HWY_API Vec128<TFromD<D>, HWY_MAX_LANES_D(D)> Zero(D /* tag */) {
  return Vec128<TFromD<D>, HWY_MAX_LANES_D(D)>{
    (typename detail::Raw128<TFromD<D>>::type)vec_splats(0)};
}

template <class D>
using VFromD = decltype(Zero(D()));

// ------------------------------ BitCast

template <class D, typename FromT>
HWY_API VFromD<D> BitCast(D /*d*/,
                          Vec128<FromT, Repartition<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{(typename detail::Raw128<TFromD<D>>::type)v.raw};
}

// ------------------------------ Set

// Returns a vector/part with all lanes set to "t".
template <class D, HWY_IF_NOT_SPECIAL_FLOAT(TFromD<D>)>
HWY_API VFromD<D> Set(D /* tag */, TFromD<D> t) {
  using RawVectLaneType =
    typename detail::Raw128<TFromD<D>>::RawVectLaneType;
  return VFromD<D>{vec_splats(static_cast<RawVectLaneType>(t))};  // NOLINT
}

// Returns a vector with uninitialized elements.
template <class D>
HWY_API VFromD<D> Undefined(D /*d*/) {
  HWY_DIAGNOSTICS(push)
  HWY_DIAGNOSTICS_OFF(disable : 4700, ignored "-Wuninitialized")
  typename detail::Raw128<TFromD<D>>::type raw = raw;
  return VFromD<D>{raw};
  HWY_DIAGNOSTICS(pop)
}

// ------------------------------ GetLane

// Gets the single value stored in a vector/part.

template <typename T, size_t N>
HWY_API T GetLane(Vec128<T, N> v) {
  return static_cast<T>(v.raw[0]);
}

// ================================================== LOGICAL

// ------------------------------ And

template <typename T, size_t N>
HWY_API Vec128<T, N> And(Vec128<T, N> a, Vec128<T, N> b) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_and(
    (typename detail::Raw128<MakeUnsigned<T>>::type)a.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)b.raw)};
}

// ------------------------------ AndNot

// Returns ~not_mask & mask.
template <typename T, size_t N>
HWY_API Vec128<T, N> AndNot(Vec128<T, N> not_mask, Vec128<T, N> mask) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_andc(
    (typename detail::Raw128<MakeUnsigned<T>>::type)mask.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)not_mask.raw)};
}

// ------------------------------ Or

template <typename T, size_t N>
HWY_API Vec128<T, N> Or(Vec128<T, N> a, Vec128<T, N> b) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_or(
    (typename detail::Raw128<MakeUnsigned<T>>::type)a.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)b.raw)};
}

// ------------------------------ Xor

template <typename T, size_t N>
HWY_API Vec128<T, N> Xor(Vec128<T, N> a, Vec128<T, N> b) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_xor(
    (typename detail::Raw128<MakeUnsigned<T>>::type)a.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)b.raw)};
}

// ------------------------------ Not
template <typename T, size_t N>
HWY_API Vec128<T, N> Not(Vec128<T, N> v) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_nor(
    (typename detail::Raw128<MakeUnsigned<T>>::type)v.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)v.raw)};
}

// ------------------------------ Xor3
template <typename T, size_t N>
HWY_API Vec128<T, N> Xor3(Vec128<T, N> x1, Vec128<T, N> x2, Vec128<T, N> x3) {
  return Xor(x1, Xor(x2, x3));
}

// ------------------------------ Or3
template <typename T, size_t N>
HWY_API Vec128<T, N> Or3(Vec128<T, N> o1, Vec128<T, N> o2, Vec128<T, N> o3) {
  return Or(o1, Or(o2, o3));
}

// ------------------------------ OrAnd
template <typename T, size_t N>
HWY_API Vec128<T, N> OrAnd(Vec128<T, N> o, Vec128<T, N> a1, Vec128<T, N> a2) {
  return Or(o, And(a1, a2));
}

// ------------------------------ IfVecThenElse
template <typename T, size_t N>
HWY_API Vec128<T, N> IfVecThenElse(Vec128<T, N> mask, Vec128<T, N> yes,
                                   Vec128<T, N> no) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_sel(
    (typename detail::Raw128<MakeUnsigned<T>>::type)no.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)yes.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)mask.raw)};
}

// ------------------------------ Operator overloads (internal-only if float)

template <typename T, size_t N>
HWY_API Vec128<T, N> operator&(Vec128<T, N> a, Vec128<T, N> b) {
  return And(a, b);
}

template <typename T, size_t N>
HWY_API Vec128<T, N> operator|(Vec128<T, N> a, Vec128<T, N> b) {
  return Or(a, b);
}

template <typename T, size_t N>
HWY_API Vec128<T, N> operator^(Vec128<T, N> a, Vec128<T, N> b) {
  return Xor(a, b);
}

// ================================================== SIGN

// ------------------------------ Neg

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_INLINE Vec128<T, N> Neg(Vec128<T, N> v) {
  return Vec128<T, N>{vec_neg(v.raw)};
}

// ------------------------------ Abs

// Returns absolute value, except that LimitsMin() maps to LimitsMax() + 1.
template <class T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Vec128<T, N> Abs(Vec128<T, N> v) {
  return Vec128<T, N>{vec_abs(v.raw)};
}

// ------------------------------ CopySign

template <typename T, size_t N>
HWY_API Vec128<T, N> CopySign(Vec128<T, N> magn, Vec128<T, N> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  return Vec128<T, N>{vec_cpsgn(sign.raw, magn.raw)};
}

template <typename T, size_t N>
HWY_API Vec128<T, N> CopySignToAbs(Vec128<T, N> abs, Vec128<T, N> sign) {
  // PPC8 can also handle abs < 0, so no extra action needed.
  return CopySign(abs, sign);
}

// ================================================== MEMORY (1)

// Note: type punning is safe because the types are tagged with may_alias.
// (https://godbolt.org/z/fqrWjfjsP)

// ------------------------------ Load

template <class D, HWY_IF_V_SIZE_D(D, 16), typename T = TFromD<D>>
HWY_API Vec128<T> Load(D /* tag */, const T* HWY_RESTRICT aligned) {
  using LoadRawVectType =
    typename detail::Raw128<T>::AlignedLoadStoreRawVectType;
  using ResultRawVectType =
    typename detail::Raw128<T>::type;
  return Vec128<T>{(ResultRawVectType)
    *reinterpret_cast<const LoadRawVectType*>(aligned)};
}

// Any <= 64 bit
template <class D, HWY_IF_V_SIZE_LE_D(D, 8), typename T = TFromD<D>>
HWY_API VFromD<D> Load(D d, const T* HWY_RESTRICT p) {
  using BitsT = UnsignedFromSize<d.MaxBytes()>;

  BitsT bits;
  const Repartition<BitsT, decltype(d)> d_bits;
  CopyBytes<d.MaxBytes()>(p, &bits);
  return BitCast(d, Set(d_bits, bits));
}

// ================================================== MASK

// ------------------------------ Mask

// Mask and Vec are both backed by vector types (true = FF..FF).
template <typename T, size_t N>
HWY_API Mask128<T, N> MaskFromVec(Vec128<T, N> v) {
  return Mask128<T, N>{(typename detail::Raw128<T>::RawBoolVectType)v.raw};
}

template <class D>
using MFromD = decltype(MaskFromVec(VFromD<D>()));

template <typename T, size_t N>
HWY_API Vec128<T, N> VecFromMask(Mask128<T, N> v) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)v.raw};
}

template <class D>
HWY_API VFromD<D> VecFromMask(D /* tag */, MFromD<D> v) {
  return VFromD<D>{(typename detail::Raw128<TFromD<D>>::type)v.raw};
}

// mask ? yes : no
template <typename T, size_t N>
HWY_API Vec128<T, N> IfThenElse(Mask128<T, N> mask, Vec128<T, N> yes,
                                Vec128<T, N> no) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_sel(
    (typename detail::Raw128<MakeUnsigned<T>>::type)no.raw,
    (typename detail::Raw128<MakeUnsigned<T>>::type)yes.raw,
    mask.raw)};
}

// mask ? yes : 0
template <typename T, size_t N>
HWY_API Vec128<T, N> IfThenElseZero(Mask128<T, N> mask, Vec128<T, N> yes) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_and(
    (typename detail::Raw128<MakeUnsigned<T>>::type)yes.raw, mask.raw)};
}

// mask ? 0 : no
template <typename T, size_t N>
HWY_API Vec128<T, N> IfThenZeroElse(Mask128<T, N> mask, Vec128<T, N> no) {
  return Vec128<T, N>{(typename detail::Raw128<T>::type)vec_andc(
    (typename detail::Raw128<MakeUnsigned<T>>::type)no.raw, mask.raw)};
}

// ------------------------------ Mask logical

template <typename T, size_t N>
HWY_API Mask128<T, N> Not(Mask128<T, N> m) {
  return Mask128<T, N>{vec_nor(m.raw, m.raw)};
}

template <typename T, size_t N>
HWY_API Mask128<T, N> And(Mask128<T, N> a, Mask128<T, N> b) {
  return Mask128<T, N>{vec_and(a.raw, b.raw)};
}

template <typename T, size_t N>
HWY_API Mask128<T, N> AndNot(Mask128<T, N> a, Mask128<T, N> b) {
  return Mask128<T, N>{vec_andc(b.raw, a.raw)};
}

template <typename T, size_t N>
HWY_API Mask128<T, N> Or(Mask128<T, N> a, Mask128<T, N> b) {
  return Mask128<T, N>{vec_or(a.raw, b.raw)};
}

template <typename T, size_t N>
HWY_API Mask128<T, N> Xor(Mask128<T, N> a, Mask128<T, N> b) {
  return Mask128<T, N>{vec_xor(a.raw, b.raw)};
}

template <typename T, size_t N>
HWY_API Mask128<T, N> ExclusiveNeither(Mask128<T, N> a, Mask128<T, N> b) {
  return Mask128<T, N>{vec_nor(a.raw, b.raw)};
}

// ------------------------------ BroadcastSignBit

template <size_t N>
HWY_API Vec128<int8_t, N> BroadcastSignBit(Vec128<int8_t, N> v) {
  return Vec128<int8_t, N>{
    vec_sra(v.raw, vec_splats(static_cast<unsigned char>(7)))};
}

template <size_t N>
HWY_API Vec128<int16_t, N> BroadcastSignBit(Vec128<int16_t, N> v) {
  return Vec128<int16_t, N>{
    vec_sra(v.raw, vec_splats(static_cast<unsigned short>(15)))};
}

template <size_t N>
HWY_API Vec128<int32_t, N> BroadcastSignBit(Vec128<int32_t, N> v) {
  return Vec128<int32_t, N>{vec_sra(v.raw, vec_splats(31u))};
}

template <size_t N>
HWY_API Vec128<int64_t, N> BroadcastSignBit(Vec128<int64_t, N> v) {
  return Vec128<int64_t, N>{vec_sra(v.raw, vec_splats(63ULL))};
}

// ------------------------------ ShiftLeftSame

template <typename T, size_t N, HWY_IF_NOT_FLOAT_NOR_SPECIAL(T)>
HWY_API Vec128<T, N> ShiftLeftSame(Vec128<T, N> v, const int bits) {
  using UnsignedLaneT =
    typename detail::Raw128<MakeUnsigned<T>>::RawVectLaneType;
  return Vec128<T, N>{vec_sl(v.raw, vec_splats(
    static_cast<UnsignedLaneT>(bits)))};
}

// ------------------------------ ShiftRightSame

template <typename T, size_t N, HWY_IF_UNSIGNED(T)>
HWY_API Vec128<T, N> ShiftRightSame(Vec128<T, N> v,
                                    const int bits) {
  using UnsignedLaneT =
    typename detail::Raw128<MakeUnsigned<T>>::RawVectLaneType;
  return Vec128<T, N>{vec_sr(v.raw, vec_splats(
    static_cast<UnsignedLaneT>(bits)))};
}

template <typename T, size_t N, HWY_IF_SIGNED(T)>
HWY_API Vec128<T, N> ShiftRightSame(Vec128<T, N> v, const int bits) {
  using UnsignedLaneT =
    typename detail::Raw128<MakeUnsigned<T>>::RawVectLaneType;
  return Vec128<T, N>{vec_sra(v.raw, vec_splats(
    static_cast<UnsignedLaneT>(bits)))};
}

// ------------------------------ ShiftLeft

template <int kBits, typename T, size_t N, HWY_IF_NOT_FLOAT_NOR_SPECIAL(T)>
HWY_API Vec128<T, N> ShiftLeft(Vec128<T, N> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return ShiftLeftSame(v, kBits);
}

// ------------------------------ ShiftRight

template <int kBits, typename T, size_t N, HWY_IF_NOT_FLOAT_NOR_SPECIAL(T)>
HWY_API Vec128<T, N> ShiftRight(Vec128<T, N> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return ShiftRightSame(v, kBits);
}

// ================================================== SWIZZLE (1)

// ------------------------------ TableLookupBytes
template <typename T, size_t N, typename TI, size_t NI>
HWY_API Vec128<TI, NI> TableLookupBytes(Vec128<T, N> bytes,
                                        Vec128<TI, NI> from) {
  return Vec128<TI, NI>{(typename detail::Raw128<TI>::type)vec_perm(
    bytes.raw, bytes.raw, (__vector unsigned char)from.raw)};
}

// ------------------------------ TableLookupBytesOr0
// For all vector widths; Altivec/VSX needs zero out
template <class V, class VI>
HWY_API VI TableLookupBytesOr0(const V bytes, const VI from) {
  const VI zeroOutMask{(typename detail::Raw128<TFromV<VI>>::type)
                       vec_sra((__vector signed char)from.raw,
                               vec_splats(static_cast<unsigned char>(7)))};
  return AndNot(zeroOutMask, TableLookupBytes(bytes, from));
}

// ------------------------------ Shuffles (ShiftRight, TableLookupBytes)

// Notation: let Vec128<int32_t> have lanes 3,2,1,0 (0 is least-significant).
// Shuffle0321 rotates one lane to the right (the previous least-significant
// lane is now most-significant). These could also be implemented via
// CombineShiftRightBytes but the shuffle_abcd notation is more convenient.

// Swap 32-bit halves in 64-bit halves.
template <typename T, size_t N>
HWY_API Vec128<T, N> Shuffle2301(Vec128<T, N> v) {
  static_assert(sizeof(T) == 4, "Only for 32-bit lanes");
  static_assert(N == 2 || N == 4, "Does not make sense for N=1");
  const __vector unsigned char kShuffle =
    {4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
  return Vec128<T, N>{vec_perm(v.raw, v.raw, kShuffle)};
}

// These are used by generic_ops-inl to implement LoadInterleaved3. As with
// Intel's shuffle* intrinsics and InterleaveLower, the lower half of the output
// comes from the first argument.
namespace detail {

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec32<T> Shuffle2301(Vec32<T> a, Vec32<T> b) {
  const __vector unsigned char kShuffle16 = {1, 0, 19, 18};
  return Vec32<T>{vec_perm(a.raw, b.raw, kShuffle16)};
}
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec64<T> Shuffle2301(Vec64<T> a, Vec64<T> b) {
  const __vector unsigned char kShuffle = {2, 3, 0, 1, 22, 23, 20, 21};
  return Vec64<T>{vec_perm(a.raw, b.raw, kShuffle)};
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle2301(Vec128<T> a, Vec128<T> b) {
  const __vector unsigned char kShuffle =
    {4, 5, 6, 7, 0, 1, 2, 3, 28, 29, 30, 31, 24, 25, 26, 27};
  return Vec128<T>{vec_perm(a.raw, b.raw, kShuffle)};
}

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec32<T> Shuffle1230(Vec32<T> a, Vec32<T> b) {
  const __vector unsigned char kShuffle = {0, 3, 18, 17};
  return Vec32<T>{vec_perm(a.raw, b.raw, kShuffle)};
}
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec64<T> Shuffle1230(Vec64<T> a, Vec64<T> b) {
  const __vector unsigned char kShuffle = {0, 1, 6, 7, 20, 21, 18, 19};
  return Vec64<T>{vec_perm(a.raw, b.raw, kShuffle)};
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle1230(Vec128<T> a, Vec128<T> b) {
  const __vector unsigned char kShuffle =
    {0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 20, 21, 22, 23};
  return Vec128<T>{vec_perm(a.raw, b.raw, kShuffle)};
}

template <typename T, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec32<T> Shuffle3012(Vec32<T> a, Vec32<T> b) {
  const __vector unsigned char kShuffle = {2, 1, 16, 19};
  return Vec32<T>{vec_perm(a.raw, b.raw, kShuffle)};
}
template <typename T, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec64<T> Shuffle3012(Vec64<T> a, Vec64<T> b) {
  const __vector unsigned char kShuffle = {4, 5, 2, 3, 16, 17, 22, 23};
  return Vec64<T>{vec_perm(a.raw, b.raw, kShuffle)};
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle3012(Vec128<T> a, Vec128<T> b) {
  const __vector unsigned char kShuffle =
    {8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31};
  return Vec128<T>{vec_perm(a.raw, b.raw, kShuffle)};
}

}  // namespace detail

// Swap 64-bit halves
template<class T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle1032(Vec128<T> v) {
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_reve(
    (__vector unsigned long long)v.raw)};
}
template<class T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec128<T> Shuffle01(Vec128<T> v) {
  return Vec128<T>{vec_reve(v.raw)};
}

// Rotate right 32 bits
template<class T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle0321(Vec128<T> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<T>{vec_sld(v.raw, v.raw, 12)};
#else
  return Vec128<T>{vec_sld(v.raw, v.raw, 4)};
#endif
}
// Rotate left 32 bits
template<class T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle2103(Vec128<T> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<T>{vec_sld(v.raw, v.raw, 4)};
#else
  return Vec128<T>{vec_sld(v.raw, v.raw, 12)};
#endif
}

// Reverse
template<class T, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Shuffle0123(Vec128<T> v) {
  return Vec128<T>{vec_reve(v.raw)};
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

template <class DTo, typename TFrom, size_t NFrom>
HWY_API MFromD<DTo> RebindMask(DTo /*dto*/, Mask128<TFrom, NFrom> m) {
  static_assert(sizeof(TFrom) == sizeof(TFromD<DTo>), "Must have same size");
  return MFromD<DTo>{m.raw};
}

template <typename T, size_t N>
HWY_API Mask128<T, N> TestBit(Vec128<T, N> v, Vec128<T, N> bit) {
  static_assert(!hwy::IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

// ------------------------------ Equality

template <typename T, size_t N>
HWY_API Mask128<T, N> operator==(Vec128<T, N> a,
                                 Vec128<T, N> b) {
  return Mask128<T, N>{vec_cmpeq(a.raw, b.raw)};
}

// ------------------------------ Inequality

// This cannot have T as a template argument, otherwise it is not more
// specialized than rewritten operator== in C++20, leading to compile
// errors: https://gcc.godbolt.org/z/xsrPhPvPT.
template <size_t N>
HWY_API Mask128<uint8_t, N> operator!=(Vec128<uint8_t, N> a,
                                       Vec128<uint8_t, N> b) {
#if HWY_TARGET <= HWY_PPC9
  return Mask128<uint8_t, N>{vec_cmpne(a.raw, b.raw)};
#else
  return Not(a == b);
#endif
}
template <size_t N>
HWY_API Mask128<uint16_t, N> operator!=(Vec128<uint16_t, N> a,
                                        Vec128<uint16_t, N> b) {
#if HWY_TARGET <= HWY_PPC9
  return Mask128<uint16_t, N>{vec_cmpne(a.raw, b.raw)};
#else
  return Not(a == b);
#endif
}
template <size_t N>
HWY_API Mask128<uint32_t, N> operator!=(Vec128<uint32_t, N> a,
                                        Vec128<uint32_t, N> b) {
#if HWY_TARGET <= HWY_PPC9
  return Mask128<uint32_t, N>{vec_cmpne(a.raw, b.raw)};
#else
  return Not(a == b);
#endif
}
template <size_t N>
HWY_API Mask128<uint64_t, N> operator!=(Vec128<uint64_t, N> a,
                                        Vec128<uint64_t, N> b) {
  return Not(a == b);
}
template <size_t N>
HWY_API Mask128<int8_t, N> operator!=(Vec128<int8_t, N> a,
                                      Vec128<int8_t, N> b) {
#if HWY_TARGET <= HWY_PPC9
  return Mask128<int8_t, N>{vec_cmpne(a.raw, b.raw)};
#else
  return Not(a == b);
#endif
}
template <size_t N>
HWY_API Mask128<int16_t, N> operator!=(Vec128<int16_t, N> a,
                                       Vec128<int16_t, N> b) {
#if HWY_TARGET <= HWY_PPC9
  return Mask128<int16_t, N>{vec_cmpne(a.raw, b.raw)};
#else
  return Not(a == b);
#endif
}
template <size_t N>
HWY_API Mask128<int32_t, N> operator!=(Vec128<int32_t, N> a,
                                       Vec128<int32_t, N> b) {
#if HWY_TARGET <= HWY_PPC9
  return Mask128<int32_t, N>{vec_cmpne(a.raw, b.raw)};
#else
  return Not(a == b);
#endif
}
template <size_t N>
HWY_API Mask128<int64_t, N> operator!=(Vec128<int64_t, N> a,
                                       Vec128<int64_t, N> b) {
  return Not(a == b);
}

template <size_t N>
HWY_API Mask128<float, N> operator!=(Vec128<float, N> a,
                                     Vec128<float, N> b) {
  return Not(a == b);
}

template <size_t N>
HWY_API Mask128<double, N> operator!=(Vec128<double, N> a,
                                      Vec128<double, N> b) {
  return Not(a == b);
}

// ------------------------------ Strict inequality

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_INLINE Mask128<T, N> operator>(Vec128<T, N> a, Vec128<T, N> b) {
  return Mask128<T, N>{vec_cmpgt(a.raw, b.raw)};
}

// ------------------------------ Weak inequality

template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Mask128<T, N> operator>=(Vec128<T, N> a,
                                 Vec128<T, N> b) {
  return Mask128<T, N>{vec_cmpge(a.raw, b.raw)};
}

// ------------------------------ Reversed comparisons

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Mask128<T, N> operator<(Vec128<T, N> a, Vec128<T, N> b) {
  return b > a;
}

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Mask128<T, N> operator<=(Vec128<T, N> a, Vec128<T, N> b) {
  return b >= a;
}

// ================================================== MEMORY (2)

// ------------------------------ Load
template <class D, HWY_IF_V_SIZE_D(D, 16), typename T = TFromD<D>>
HWY_API Vec128<T> LoadU(D /* tag */, const T* HWY_RESTRICT p) {
  using LoadRawVectType =
    typename detail::Raw128<T>::UnalignedLoadStoreRawVectType;
  using ResultRawVectType =
    typename detail::Raw128<T>::type;
  return Vec128<T>{(ResultRawVectType)
    *reinterpret_cast<const LoadRawVectType*>(p)};
}

// For < 128 bit, LoadU == Load.
template <class D, HWY_IF_V_SIZE_LE_D(D, 8), typename T = TFromD<D>>
HWY_API VFromD<D> LoadU(D d, const T* HWY_RESTRICT p) {
  return Load(d, p);
}

// 128-bit SIMD => nothing to duplicate, same as an unaligned load.
template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> LoadDup128(D d, const T* HWY_RESTRICT p) {
  return LoadU(d, p);
}

// Returns a vector with lane i=[0, N) set to "first" + i.
template <class D, typename T = TFromD<D>, typename T2>
HWY_API VFromD<D> Iota(D d, const T2 first) {
  HWY_ALIGN T lanes[MaxLanes(d)];
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    lanes[i] =
        AddWithWraparound(hwy::IsFloatTag<T>(), static_cast<T>(first), i);
  }
  return Load(d, lanes);
}

// ------------------------------ FirstN (Iota, Lt)

template <class D>
HWY_API MFromD<D> FirstN(D d, size_t num) {
  const RebindToSigned<decltype(d)> di;  // Signed comparisons are cheaper.
  using TI = TFromD<decltype(di)>;
  return RebindMask(d, Iota(di, 0) < Set(di, static_cast<TI>(num)));
}

// ------------------------------ MaskedLoad

template <class D, typename T = TFromD<D>>
HWY_API VFromD<D> MaskedLoad(MFromD<D> m, D d, const T* HWY_RESTRICT p) {
  return IfThenElseZero(m, Load(d, p));
}

// ------------------------------ Store

template <class D, HWY_IF_V_SIZE_D(D, 16), typename T = TFromD<D>>
HWY_API void Store(Vec128<T> v, D /* tag */, T* HWY_RESTRICT aligned) {
  using StoreRawVectT =
    typename detail::Raw128<T>::AlignedLoadStoreRawVectType;
  *reinterpret_cast<StoreRawVectT*>(aligned) = (StoreRawVectT)v.raw;
}

template <class D, HWY_IF_V_SIZE_D(D, 16), typename T = TFromD<D>>
HWY_API void StoreU(Vec128<T> v, D /* tag */, T* HWY_RESTRICT p) {
  using StoreRawVectT =
    typename detail::Raw128<T>::UnalignedLoadStoreRawVectType;
  *reinterpret_cast<StoreRawVectT*>(p) = (StoreRawVectT)v.raw;
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8), typename T = TFromD<D>>
HWY_API void Store(VFromD<D> v, D d, T* HWY_RESTRICT p) {
  using BitsT = UnsignedFromSize<d.MaxBytes()>;

  const Repartition<BitsT, decltype(d)> d_bits;
  const BitsT bits = GetLane(BitCast(d_bits, v));
  CopyBytes<d.MaxBytes()>(&bits, p);
}

// For < 128 bit, StoreU == Store.
template <class D, HWY_IF_V_SIZE_LE_D(D, 8), typename T = TFromD<D>>
HWY_API void StoreU(VFromD<D> v, D d, T* HWY_RESTRICT p) {
  Store(v, d, p);
}

// ------------------------------ BlendedStore

template <class D>
HWY_API void BlendedStore(VFromD<D> v, MFromD<D> m, D d,
                          TFromD<D>* HWY_RESTRICT p) {
  const RebindToSigned<decltype(d)> di;  // for testing mask if T=bfloat16_t.
  using TI = TFromD<decltype(di)>;
  alignas(16) TI buf[MaxLanes(d)];
  alignas(16) TI mask[MaxLanes(d)];
  Store(BitCast(di, v), di, buf);
  Store(BitCast(di, VecFromMask(d, m)), di, mask);
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    if (mask[i]) {
      CopySameSize(buf + i, p + i);
    }
  }
}

// ================================================== ARITHMETIC

// ------------------------------ Addition

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Vec128<T, N> operator+(Vec128<T, N> a,
                               Vec128<T, N> b) {
  return Vec128<T, N>{vec_add(a.raw, b.raw)};
}

// ------------------------------ Subtraction


template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Vec128<T, N> operator-(Vec128<T, N> a,
                               Vec128<T, N> b) {
  return Vec128<T, N>{vec_sub(a.raw, b.raw)};
}

// ------------------------------ SumsOf8
namespace detail {

inline __attribute__((__always_inline__))
__vector unsigned int AltivecVsum4ubs(__vector unsigned char a,
                                      __vector unsigned int b) {
#ifdef __OPTIMIZE__
  if(__builtin_constant_p(a[0]) && __builtin_constant_p(a[1]) &&
     __builtin_constant_p(a[2]) && __builtin_constant_p(a[3]) &&
     __builtin_constant_p(a[4]) && __builtin_constant_p(a[5]) &&
     __builtin_constant_p(a[6]) && __builtin_constant_p(a[7]) &&
     __builtin_constant_p(a[8]) && __builtin_constant_p(a[9]) &&
     __builtin_constant_p(a[10]) && __builtin_constant_p(a[11]) &&
     __builtin_constant_p(a[12]) && __builtin_constant_p(a[13]) &&
     __builtin_constant_p(a[14]) && __builtin_constant_p(a[15]) &&
     __builtin_constant_p(b[0]) && __builtin_constant_p(b[1]) &&
     __builtin_constant_p(b[2]) && __builtin_constant_p(b[3])) {
    const uint64_t sum0 = static_cast<uint64_t>(a[0]) +
                          static_cast<uint64_t>(a[1]) +
                          static_cast<uint64_t>(a[2]) +
                          static_cast<uint64_t>(a[3]) +
                          static_cast<uint64_t>(b[0]);
    const uint64_t sum1 = static_cast<uint64_t>(a[4]) +
                          static_cast<uint64_t>(a[5]) +
                          static_cast<uint64_t>(a[6]) +
                          static_cast<uint64_t>(a[7]) +
                          static_cast<uint64_t>(b[1]);
    const uint64_t sum2 = static_cast<uint64_t>(a[8]) +
                          static_cast<uint64_t>(a[9]) +
                          static_cast<uint64_t>(a[10]) +
                          static_cast<uint64_t>(a[11]) +
                          static_cast<uint64_t>(b[2]);
    const uint64_t sum3 = static_cast<uint64_t>(a[12]) +
                          static_cast<uint64_t>(a[13]) +
                          static_cast<uint64_t>(a[14]) +
                          static_cast<uint64_t>(a[15]) +
                          static_cast<uint64_t>(b[3]);
    return (__vector unsigned int){
      static_cast<unsigned int>(sum0 <= 0xFFFFFFFFu ? sum0 : 0xFFFFFFFFu),
      static_cast<unsigned int>(sum1 <= 0xFFFFFFFFu ? sum1 : 0xFFFFFFFFu),
      static_cast<unsigned int>(sum2 <= 0xFFFFFFFFu ? sum2 : 0xFFFFFFFFu),
      static_cast<unsigned int>(sum3 <= 0xFFFFFFFFu ? sum3 : 0xFFFFFFFFu)
      };
  } else
#endif
  {
    return vec_vsum4ubs(a, b);
  }
}

inline __attribute__((__always_inline__))
__vector signed int AltivecVsum2sws(__vector signed int a,
                                    __vector signed int b) {
#ifdef __OPTIMIZE__
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kDestLaneOffset = 0;
#else
  constexpr int kDestLaneOffset = 1;
#endif
  if(__builtin_constant_p(a[0]) && __builtin_constant_p(a[1]) &&
     __builtin_constant_p(a[2]) && __builtin_constant_p(a[3]) &&
     __builtin_constant_p(b[kDestLaneOffset]) &&
     __builtin_constant_p(b[kDestLaneOffset + 2])) {
    const int64_t sum0 = static_cast<int64_t>(a[0]) +
                         static_cast<int64_t>(a[1]) +
                         static_cast<int64_t>(b[kDestLaneOffset]);
    const int64_t sum1 = static_cast<int64_t>(a[2]) +
                         static_cast<int64_t>(a[3]) +
                         static_cast<int64_t>(b[kDestLaneOffset + 2]);
    const int32_t sign0 = static_cast<int32_t>(sum0 >> 63);
    const int32_t sign1 = static_cast<int32_t>(sum1 >> 63);
    const Full128<int32_t> di32;
    return BitCast(di32, Vec128<uint64_t>{(__vector unsigned long long){
                             (sign0 == (sum0 >> 31))
                                 ? static_cast<uint32_t>(sum0)
                                 : static_cast<uint32_t>(sign0 ^ 0x7FFFFFFF),
                             (sign1 == (sum1 >> 31))
                                 ? static_cast<uint32_t>(sum1)
                                 : static_cast<uint32_t>(sign1 ^ 0x7FFFFFFF)}})
        .raw;
  } else
#endif
  {
    __vector signed int sum;

    // Inline assembly is used for vsum2sws to avoid unnecessary shuffling
    // on little-endian PowerPC targets as the result of the vsum2sws
    // instruction will already be in the correct lanes on little-endian
    // PowerPC targets.
    __asm__("vsum2sws %0,%1,%2"
            : "=v" (sum)
            : "v" (a), "v" (b));

    return sum;
  }
}

}  // namespace detail

template <size_t N>
HWY_API Vec128<uint64_t, N / 8> SumsOf8(Vec128<uint8_t, N> v) {
  const Full128<int32_t> di32;
  const Vec128<uint32_t> zero{vec_splats(0u)};

  return Vec128<uint64_t, N / 8>{
      (__vector unsigned long long)detail::AltivecVsum2sws(
          (__vector signed int)detail::AltivecVsum4ubs(v.raw, zero.raw),
          BitCast(di32, zero).raw)};
}

// ------------------------------ SaturatedAdd

// Returns a + b clamped to the destination range.

// Unsigned
template <typename T, size_t N, HWY_IF_NOT_FLOAT(T),
                                HWY_IF_T_SIZE_ONE_OF(T, 0x6)>
HWY_API Vec128<T, N> SaturatedAdd(Vec128<T, N> a, Vec128<T, N> b) {
  return Vec128<T, N>{vec_adds(a.raw, b.raw)};
}

// ------------------------------ SaturatedSub

// Returns a - b clamped to the destination range.

template <typename T, size_t N, HWY_IF_NOT_FLOAT(T),
                                HWY_IF_T_SIZE_ONE_OF(T, 0x6)>
HWY_API Vec128<T, N> SaturatedSub(Vec128<T, N> a, Vec128<T, N> b) {
  return Vec128<T, N>{vec_subs(a.raw, b.raw)};
}

// ------------------------------ AverageRound

// Returns (a + b + 1) / 2

template <typename T, size_t N, HWY_IF_UNSIGNED(T),
                                HWY_IF_T_SIZE_ONE_OF(T, 0x6)>
HWY_API Vec128<T, N> AverageRound(Vec128<T, N> a, Vec128<T, N> b) {
  return Vec128<T, N>{vec_avg(a.raw, b.raw)};
}

// ------------------------------ Multiplication

template <typename T, size_t N, hwy::EnableIf<(IsFloat<T>() ||
  (!IsSpecialFloat<T>() && (sizeof(T) == 2 || sizeof(T) == 4)))>* = nullptr>
HWY_API Vec128<T, N> operator*(Vec128<T, N> a,
                               Vec128<T, N> b) {
  return Vec128<T, N>{a.raw * b.raw};
}

// Returns the upper 16 bits of a * b in each lane.
template <typename T, size_t N, HWY_IF_T_SIZE(T, 2), HWY_IF_NOT_FLOAT(T)>
HWY_API Vec128<T, N> MulHigh(Vec128<T, N> a,
                             Vec128<T, N> b) {
  const auto p1 = vec_mule(a.raw, b.raw);
  const auto p2 = vec_mulo(a.raw, b.raw);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kShuffle =
    {2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
#else
  const __vector unsigned char kShuffle =
    {0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
#endif
  return Vec128<T, N>{(typename detail::Raw128<T>::type)
    vec_perm(p1, p2, kShuffle)};
}

template <size_t N>
HWY_API Vec128<int16_t, N> MulFixedPoint15(Vec128<int16_t, N> a,
                                           Vec128<int16_t, N> b) {
  return Vec128<int16_t, N>{vec_mradds(a.raw, b.raw, vec_splats(short{0}))};
}

// Multiplies even lanes (0, 2 ..) and places the double-wide result into
// even and the upper half into its odd neighbor lane.
template <typename T, size_t N, HWY_IF_T_SIZE(T, 4), HWY_IF_NOT_FLOAT(T)>
HWY_API Vec128<MakeWide<T>, (N + 1) / 2> MulEven(Vec128<T, N> a,
                                                 Vec128<T, N> b) {
  return Vec128<MakeWide<T>, (N + 1) / 2>{vec_mule(a.raw, b.raw)};
}

// ------------------------------ RotateRight

template <int kBits, size_t N>
HWY_API Vec128<uint32_t, N> RotateRight(Vec128<uint32_t, N> v) {
  static_assert(0 <= kBits && kBits < 32, "Invalid shift count");
  if (kBits == 0) return v;
  return Vec128<uint32_t, N>{
    vec_rl(v.raw, vec_splats(static_cast<unsigned>(32 - kBits)))};
}

template <int kBits, size_t N>
HWY_API Vec128<uint64_t, N> RotateRight(Vec128<uint64_t, N> v) {
  static_assert(0 <= kBits && kBits < 64, "Invalid shift count");
  if (kBits == 0) return v;
  return Vec128<uint64_t, N>{
    vec_rl(v.raw, vec_splats(static_cast<unsigned long long>(64 - kBits)))};
}

// ------------------------------ ZeroIfNegative (BroadcastSignBit)
template <typename T, size_t N>
HWY_API Vec128<T, N> ZeroIfNegative(Vec128<T, N> v) {
  static_assert(IsFloat<T>(), "Only works for float");
  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;
  const auto mask = MaskFromVec(BitCast(d, BroadcastSignBit(BitCast(di, v))));
  return IfThenElse(mask, Zero(d), v);
}

// ------------------------------ IfNegativeThenElse
template <typename T, size_t N>
HWY_API Vec128<T, N> IfNegativeThenElse(Vec128<T, N> v,
                                        Vec128<T, N> yes,
                                        Vec128<T, N> no) {
  static_assert(IsSigned<T>(), "Only works for signed/float");

  const DFromV<decltype(v)> d;
  const RebindToSigned<decltype(d)> di;
  return IfThenElse(MaskFromVec(BitCast(d, BroadcastSignBit(BitCast(di, v)))),
                    yes, no);
}

// Absolute value of difference.
template <size_t N>
HWY_API Vec128<float, N> AbsDiff(Vec128<float, N> a,
                                 Vec128<float, N> b) {
  return Abs(a - b);
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> MulAdd(Vec128<T, N> mul,
                            Vec128<T, N> x,
                            Vec128<T, N> add) {
  return Vec128<T, N>{vec_madd(mul.raw, x.raw, add.raw)};
}

// Returns add - mul * x
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> NegMulAdd(Vec128<T, N> mul,
                               Vec128<T, N> x,
                               Vec128<T, N> add) {
  // NOTE: the vec_nmsub operation below computes -(mul * x - add),
  // which is equivalent to add - mul * x in the round-to-nearest
  // and round-towards-zero rounding modes
  return Vec128<T, N>{vec_nmsub(mul.raw, x.raw, add.raw)};
}

// Returns mul * x - sub
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> MulSub(Vec128<T, N> mul,
                            Vec128<T, N> x,
                            Vec128<T, N> sub) {
  return Vec128<T, N>{vec_msub(mul.raw, x.raw, sub.raw)};
}

// Returns -mul * x - sub
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> NegMulSub(Vec128<T, N> mul,
                               Vec128<T, N> x,
                               Vec128<T, N> sub) {
  // NOTE: The vec_nmadd operation below computes -(mul * x + sub),
  // which is equivalent to -mul * x - sub in the round-to-nearest
  // and round-towards-zero rounding modes
  return Vec128<T, N>{vec_nmadd(mul.raw, x.raw, sub.raw)};
}

// ------------------------------ Floating-point div
// Approximate reciprocal
template <size_t N>
HWY_API Vec128<float, N> ApproximateReciprocal(Vec128<float, N> v) {
  return Vec128<float, N>{vec_re(v.raw)};
}

template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> operator/(Vec128<T, N> a,
                               Vec128<T, N> b) {
  return Vec128<T, N>{vec_div(a.raw, b.raw)};
}

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
template <size_t N>
HWY_API Vec128<float, N> ApproximateReciprocalSqrt(Vec128<float, N> v) {
  return Vec128<float, N>{vec_rsqrte(v.raw)};
}

// Full precision square root
template <class T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> Sqrt(Vec128<T, N> v) {
  return Vec128<T, N>{vec_sqrt(v.raw)};
}

// ------------------------------ Min (Gt, IfThenElse)

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Vec128<T, N> Min(Vec128<T, N> a,
                         Vec128<T, N> b) {
  return Vec128<T, N>{vec_min(a.raw, b.raw)};
}

// ------------------------------ Max (Gt, IfThenElse)

template <typename T, size_t N, HWY_IF_NOT_SPECIAL_FLOAT(T)>
HWY_API Vec128<T, N> Max(Vec128<T, N> a,
                         Vec128<T, N> b) {
  return Vec128<T, N>{vec_max(a.raw, b.raw)};
}

// ================================================== MEMORY (3)

// ------------------------------ Non-temporal stores


template <class D>
HWY_API void Stream(VFromD<D> v, D d, TFromD<D>* HWY_RESTRICT aligned) {
  __builtin_prefetch(aligned, 1, 0);
  Store(v, d, aligned);
}

// ------------------------------ Scatter

template <class D, HWY_IF_V_SIZE_LE_D(D, 16), typename T = TFromD<D>, class VI>
HWY_API void ScatterOffset(VFromD<D> v, D d, T* HWY_RESTRICT base, VI offset) {
  using TI = TFromV<VI>;
  static_assert(sizeof(T) == sizeof(TI), "Index/lane size must match");

  alignas(16) T lanes[MaxLanes(d)];
  Store(v, d, lanes);

  alignas(16) TI offset_lanes[MaxLanes(d)];
  Store(offset, Rebind<TI, decltype(d)>(), offset_lanes);

  uint8_t* base_bytes = reinterpret_cast<uint8_t*>(base);
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    CopyBytes<sizeof(T)>(&lanes[i], base_bytes + offset_lanes[i]);
  }
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 16), typename T = TFromD<D>, class VI>
HWY_API void ScatterIndex(VFromD<D> v, D d, T* HWY_RESTRICT base, VI index) {
  using TI = TFromV<VI>;
  static_assert(sizeof(T) == sizeof(TI), "Index/lane size must match");

  alignas(16) T lanes[MaxLanes(d)];
  Store(v, d, lanes);

  alignas(16) TI index_lanes[MaxLanes(d)];
  Store(index, Rebind<TI, decltype(d)>(), index_lanes);

  for (size_t i = 0; i < MaxLanes(d); ++i) {
    base[index_lanes[i]] = lanes[i];
  }
}

// ------------------------------ Gather (Load/Store)

template <class D, typename T = TFromD<D>, class VI>
HWY_API VFromD<D> GatherOffset(D d, const T* HWY_RESTRICT base, VI offset) {
  using TI = TFromV<VI>;
  static_assert(sizeof(T) == sizeof(TI), "Index/lane size must match");

  alignas(16) TI offset_lanes[MaxLanes(d)];
  Store(offset, Rebind<TI, decltype(d)>(), offset_lanes);

  alignas(16) T lanes[MaxLanes(d)];
  const uint8_t* base_bytes = reinterpret_cast<const uint8_t*>(base);
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    CopyBytes<sizeof(T)>(base_bytes + offset_lanes[i], &lanes[i]);
  }
  return Load(d, lanes);
}

template <class D, typename T = TFromD<D>, class VI>
HWY_API VFromD<D> GatherIndex(D d, const T* HWY_RESTRICT base, VI index) {
  using TI = TFromV<VI>;
  static_assert(sizeof(T) == sizeof(TI), "Index/lane size must match");

  alignas(16) TI index_lanes[MaxLanes(d)];
  Store(index, Rebind<TI, decltype(d)>(), index_lanes);

  alignas(16) T lanes[MaxLanes(d)];
  for (size_t i = 0; i < MaxLanes(d); ++i) {
    lanes[i] = base[index_lanes[i]];
  }
  return Load(d, lanes);
}

// ================================================== SWIZZLE (2)

// ------------------------------ LowerHalf

// Returns upper/lower half of a vector.
template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> LowerHalf(D /* tag */, VFromD<Twice<D>> v) {
  return VFromD<D>{v.raw};
}
template <typename T, size_t N>
HWY_API Vec128<T, N / 2> LowerHalf(Vec128<T, N> v) {
  return Vec128<T, N / 2>{v.raw};
}

// ------------------------------ ShiftLeftBytes

// NOTE: The ShiftLeftBytes operation moves the elements of v to the right
// by kBytes bytes and zeroes out the first kBytes bytes of v on both
// little-endian and big-endian PPC targets
// (same behavior as the HWY_EMU128 ShiftLeftBytes operation on both
// little-endian and big-endian targets)

template <int kBytes, class D>
HWY_API VFromD<D> ShiftLeftBytes(D d, VFromD<D> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  if(kBytes == 0) {
    return v;
  } else {
    const auto zeros = Zero(d);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return VFromD<D>{vec_sld(v.raw, zeros.raw, kBytes)};
#else
    return VFromD<D>{vec_sld(zeros.raw, v.raw, (-kBytes) & 15)};
#endif
  }
}

template <int kBytes, typename T, size_t N>
HWY_API Vec128<T, N> ShiftLeftBytes(Vec128<T, N> v) {
  return ShiftLeftBytes<kBytes>(DFromV<decltype(v)>(), v);
}

// ------------------------------ ShiftLeftLanes

// NOTE: The ShiftLeftLanes operation moves the elements of v to the right
// by kLanes lanes and zeroes out the first kLanes lanes of v on both
// little-endian and big-endian PPC targets
// (same behavior as the HWY_EMU128 ShiftLeftLanes operation on both
// little-endian and big-endian targets)

template <int kLanes, class D, typename T = TFromD<D>>
HWY_API VFromD<D> ShiftLeftLanes(D d, VFromD<D> v) {
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, ShiftLeftBytes<kLanes * sizeof(T)>(BitCast(d8, v)));
}

template <int kLanes, typename T, size_t N>
HWY_API Vec128<T, N> ShiftLeftLanes(Vec128<T, N> v) {
  return ShiftLeftLanes<kLanes>(DFromV<decltype(v)>(), v);
}

// ------------------------------ ShiftRightBytes

// NOTE: The ShiftRightBytes operation moves the elements of v to the left
// by kBytes bytes and zeroes out the last kBytes bytes of v on both
// little-endian and big-endian PPC targets
// (same behavior as the HWY_EMU128 ShiftRightBytes operation on both
// little-endian and big-endian targets)

template <int kBytes, class D>
HWY_API VFromD<D> ShiftRightBytes(D d, VFromD<D> v) {
  static_assert(0 <= kBytes && kBytes <= 16, "Invalid kBytes");
  if(kBytes == 0)
    return v;

  // For partial vectors, clear upper lanes so we shift in zeros.
  if (d.MaxBytes() != 16) {
    const Full128<TFromD<D>> dfull;
    VFromD<decltype(dfull)> vfull{v.raw};
    v = VFromD<D>{IfThenElseZero(FirstN(dfull, MaxLanes(d)), vfull).raw};
  }

  const auto zeros = Zero(d);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return VFromD<D>{vec_sld(zeros.raw, v.raw, (-kBytes) & 15)};
#else
  return VFromD<D>{vec_sld(v.raw, zeros.raw, kBytes)};
#endif
}

// ------------------------------ ShiftRightLanes

// NOTE: The ShiftRightLanes operation moves the elements of v to the left
// by kLanes lanes and zeroes out the last kLanes lanes of v on both
// little-endian and big-endian PPC targets
// (same behavior as the HWY_EMU128 ShiftRightLanes operation on both
// little-endian and big-endian targets)

template <int kLanes, class D>
HWY_API VFromD<D> ShiftRightLanes(D d, VFromD<D> v) {
  const Repartition<uint8_t, decltype(d)> d8;
  constexpr size_t kBytes = kLanes * sizeof(TFromD<D>);
  return BitCast(d, ShiftRightBytes<kBytes>(d8, BitCast(d8, v)));
}

// ------------------------------ UpperHalf (ShiftRightBytes)

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> UpperHalf(D d, VFromD<Twice<D>> v) {
  return LowerHalf(d, ShiftRightBytes<d.MaxBytes()>(Twice<D>(), v));
}

// ------------------------------ ExtractLane (UpperHalf)

template <typename T, size_t N>
HWY_API T ExtractLane(Vec128<T, N> v, size_t i) {
  return static_cast<T>(v.raw[i]);
}

// ------------------------------ InsertLane (UpperHalf)

template <typename T, size_t N>
HWY_API Vec128<T, N> InsertLane(Vec128<T, N> v, size_t i, T t) {
  typename detail::Raw128<T>::type raw_result = v.raw;
  raw_result[i] = t;
  return Vec128<T, N>{raw_result};
}

// ------------------------------ CombineShiftRightBytes

// NOTE: The CombineShiftRightBytes operation below moves the elements of lo to
// the left by kBytes bytes and moves the elements of hi right by (d.MaxBytes()
// - kBytes) bytes on both little-endian and big-endian PPC targets.

template <int kBytes, class D, HWY_IF_V_SIZE_D(D, 16), typename T = TFromD<D>>
HWY_API Vec128<T> CombineShiftRightBytes(D /*d*/, Vec128<T> hi, Vec128<T> lo) {
  constexpr size_t kSize = 16;
  static_assert(0 < kBytes && kBytes < kSize, "kBytes invalid");
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<T>{vec_sld(hi.raw, lo.raw, (-kBytes) & 15)};
#else
  return Vec128<T>{vec_sld(lo.raw, hi.raw, kBytes)};
#endif
}

template <int kBytes, class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> CombineShiftRightBytes(D d, VFromD<D> hi, VFromD<D> lo) {
  constexpr size_t kSize = d.MaxBytes();
  static_assert(0 < kBytes && kBytes < kSize, "kBytes invalid");
  const Repartition<uint8_t, decltype(d)> d8;
  using V8 = Vec128<uint8_t>;
  const DFromV<V8> dfull8;
  const Repartition<TFromD<D>, decltype(dfull8)> dfull;
  const V8 hi8{BitCast(d8, hi).raw};
  // Move into most-significant bytes
  const V8 lo8 = ShiftLeftBytes<16 - kSize>(V8{BitCast(d8, lo).raw});
  const V8 r = CombineShiftRightBytes<16 - kSize + kBytes>(dfull8, hi8, lo8);
  return VFromD<D>{BitCast(dfull, r).raw};
}

// ------------------------------ Broadcast/splat any lane

template <int kLane, typename T, size_t N>
HWY_API Vec128<T, N> Broadcast(Vec128<T, N> v) {
  static_assert(0 <= kLane && kLane < N, "Invalid lane");
  return Vec128<T, N>{vec_splat(v.raw, kLane)};
}

// ------------------------------ TableLookupLanes (Shuffle01)

// Returned by SetTableIndices/IndicesFromVec for use by TableLookupLanes.
template <typename T, size_t N = 16 / sizeof(T)>
struct Indices128 {
  __vector unsigned char raw;
};

template <class D, typename T = TFromD<D>, typename TI, size_t kN,
          HWY_IF_T_SIZE(T, 4)>
HWY_API Indices128<T, kN> IndicesFromVec(D d, Vec128<TI, kN> vec) {
  static_assert(sizeof(T) == sizeof(TI), "Index size must match lane");
#if HWY_IS_DEBUG_BUILD
  const Rebind<TI, decltype(d)> di;
  HWY_DASSERT(AllFalse(di, Lt(vec, Zero(di))) &&
              AllTrue(di, Lt(vec, Set(di, kN))));
#endif

  const Repartition<uint8_t, decltype(d)> d8;
  using V8 = VFromD<decltype(d8)>;
  const __vector unsigned char kByteOffsets = {0, 1, 2, 3, 0, 1, 2, 3,
                                               0, 1, 2, 3, 0, 1, 2, 3};

  // Broadcast each lane index to all 4 bytes of T
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kBroadcastLaneBytes = {
      0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12};
#else
  const __vector unsigned char kBroadcastLaneBytes = {
      3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15};
#endif
  const V8 lane_indices{(__vector unsigned char)vec_perm(vec.raw, vec.raw,
    kBroadcastLaneBytes)};

  // Shift to bytes
  const Repartition<uint16_t, decltype(d)> d16;
  const V8 byte_indices = BitCast(d8, ShiftLeft<2>(BitCast(d16, lane_indices)));

  return Indices128<T, kN>{vec_add(byte_indices.raw, kByteOffsets)};
}

template <class D, typename T = TFromD<D>, typename TI, size_t kN,
          HWY_IF_T_SIZE(T, 8)>
HWY_API Indices128<T, kN> IndicesFromVec(D d, Vec128<TI, kN> vec) {
  static_assert(sizeof(T) == sizeof(TI), "Index size must match lane");
#if HWY_IS_DEBUG_BUILD
  const Rebind<TI, decltype(d)> di;
  HWY_DASSERT(AllFalse(di, Lt(vec, Zero(di))) &&
              AllTrue(di, Lt(vec, Set(di, kN))));
#endif

  const Repartition<uint8_t, decltype(d)> d8;
  using V8 = VFromD<decltype(d8)>;
  const __vector unsigned char kByteOffsets = {0, 1, 2, 3, 4, 5, 6, 7,
                                               0, 1, 2, 3, 4, 5, 6, 7};

  // Broadcast each lane index to all 8 bytes of T
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kBroadcastLaneBytes = {
      0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8};
#else
  const __vector unsigned char kBroadcastLaneBytes = {
      7, 7, 7, 7, 7, 7, 7, 7, 15, 15, 15, 15, 15, 15, 15, 15};
#endif
  const V8 lane_indices{(__vector unsigned char)vec_perm(vec.raw, vec.raw,
    kBroadcastLaneBytes)};

  // Shift to bytes
  const Repartition<uint16_t, decltype(d)> d16;
  const V8 byte_indices = BitCast(d8, ShiftLeft<3>(BitCast(d16, lane_indices)));

  return Indices128<T, kN>{vec_add(byte_indices.raw, kByteOffsets)};
}

template <class D,  typename TI>
HWY_API Indices128<TFromD<D>, HWY_MAX_LANES_D(D)> SetTableIndices(
    D d, const TI* idx) {
  const Rebind<TI, decltype(d)> di;
  return IndicesFromVec(d, LoadU(di, idx));
}

template <typename T, size_t N>
HWY_API Vec128<T, N> TableLookupLanes(Vec128<T, N> v, Indices128<T, N> idx) {
  return TableLookupBytes(v, Vec128<T, N>{
    (typename detail::Raw128<T>::type)idx.raw});
}

// Single lane: no change
template <typename T>
HWY_API Vec128<T, 1> TableLookupLanes(Vec128<T, 1> v,
                                      Indices128<T, 1> /* idx */) {
  return v;
}

// ------------------------------ ReverseBlocks

// Single block: no change
template <class D>
HWY_API VFromD<D> ReverseBlocks(D /* tag */, VFromD<D> v) {
  return v;
}

// ------------------------------ Reverse (Shuffle0123, Shuffle2301)

// Single lane: no change
template <class D, typename T = TFromD<D>, HWY_IF_LANES_D(D, 1)>
HWY_API Vec128<T, 1> Reverse(D /* tag */, Vec128<T, 1> v) {
  return v;
}

// 32-bit x2: shuffle
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec64<T> Reverse(D /* tag */, Vec64<T> v) {
  return Vec64<T>{Shuffle2301(Vec128<T>{v.raw}).raw};
}

// 64-bit x2: shuffle
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec128<T> Reverse(D /* tag */, Vec128<T> v) {
  return Shuffle01(v);
}

// 32-bit x4: shuffle
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> Reverse(D /* tag */, Vec128<T> v) {
  return Shuffle0123(v);
}

// 16-bit x8: shuffle
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec128<T> Reverse(D /* tag */, Vec128<T> v) {
  return Vec128<T>{vec_reve(v.raw)};
}

// 16-bit x4: shuffle
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec64<T> Reverse(D /* tag */, Vec64<T> v) {
  const __vector unsigned char kShuffle = {
    6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9};
  return Vec64<T>{vec_perm(v.raw, v.raw, kShuffle)};
}

// 16-bit x2: rotate bytes
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec32<T> Reverse(D d, Vec32<T> v) {
  const RepartitionToWide<RebindToUnsigned<decltype(d)>> du32;
  return BitCast(d, RotateRight<16>(Reverse(du32, BitCast(du32, v))));
}

// ------------------------------ Reverse2 (for all vector sizes)

// Single lane: no change
template <class D, typename T = TFromD<D>, HWY_IF_LANES_D(D, 1)>
HWY_API Vec128<T, 1> Reverse2(D /* tag */, Vec128<T, 1> v) {
  return v;
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API VFromD<D> Reverse2(D /*d*/, VFromD<D> v) {
  const __vector unsigned char kShuffle = {1, 0, 3,  2,  5,  4,  7,  6,
                                           9, 8, 11, 10, 13, 12, 15, 14};
  return VFromD<D>{vec_perm(v.raw, v.raw, kShuffle)};
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API VFromD<D> Reverse2(D d, VFromD<D> v) {
  const Repartition<uint32_t, decltype(d)> du32;
  return BitCast(d, RotateRight<16>(BitCast(du32, v)));
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API VFromD<D> Reverse2(D /* tag */, VFromD<D> v) {
  return Shuffle2301(v);
}

template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 8)>
HWY_API VFromD<D> Reverse2(D /* tag */, VFromD<D> v) {
  return Shuffle01(v);
}

// ------------------------------ Reverse4

template <class D, HWY_IF_T_SIZE_D(D, 2)>
HWY_API VFromD<D> Reverse4(D /*d*/, VFromD<D> v) {
  const __vector unsigned char kShuffle = {
    6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9};
  return VFromD<D>{vec_perm(v.raw, v.raw, kShuffle)};
}

// 32-bit, any vector size: use Shuffle0123
template <class D, HWY_IF_T_SIZE_D(D, 4)>
HWY_API VFromD<D> Reverse4(D /* tag */, VFromD<D> v) {
  return Shuffle0123(v);
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 16), HWY_IF_T_SIZE_D(D, 8)>
HWY_API VFromD<D> Reverse4(D /* tag */, VFromD<D> /* v */) {
  HWY_ASSERT(0);  // don't have 4 u64 lanes
}

// ------------------------------ Reverse8

template <class D, HWY_IF_T_SIZE_D(D, 2)>
HWY_API VFromD<D> Reverse8(D /*d*/, VFromD<D> v) {
  return VFromD<D>{vec_reve(v.raw)};
}

template <class D, HWY_IF_NOT_T_SIZE_D(D, 2)>
HWY_API VFromD<D> Reverse8(D /*d*/, VFromD<D> /*v*/) {
  HWY_ASSERT(0);  // don't have 8 lanes unless 16-bit
}

// ------------------------------ InterleaveLower

// Interleaves lanes from halves of the 128-bit blocks of "a" (which provides
// the least-significant lane) and "b". To concatenate two half-width integers
// into one, use ZipLower/Upper instead (also works with scalar).

template <typename T, size_t N>
HWY_API Vec128<T, N> InterleaveLower(Vec128<T, N> a,
                                     Vec128<T, N> b) {
  return Vec128<T, N>{vec_mergeh(a.raw, b.raw)};
}

// Additional overload for the optional tag
template <class D>
HWY_API VFromD<D> InterleaveLower(D /* tag */, VFromD<D> a, VFromD<D> b) {
  return InterleaveLower(a, b);
}

// ------------------------------ InterleaveUpper (UpperHalf)

// Full
template <class D, typename T = TFromD<D>>
HWY_API Vec128<T> InterleaveUpper(D /* tag */, Vec128<T> a, Vec128<T> b) {
  return Vec128<T>{vec_mergel(a.raw, b.raw)};
}

// Partial
template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> InterleaveUpper(D d, VFromD<D> a, VFromD<D> b) {
  const Half<decltype(d)> d2;
  return InterleaveLower(d, VFromD<D>{UpperHalf(d2, a).raw},
                         VFromD<D>{UpperHalf(d2, b).raw});
}

// ------------------------------ ZipLower/ZipUpper (InterleaveLower)

// Same as Interleave*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.
template <class V, class DW = RepartitionToWide<DFromV<V>>>
HWY_API VFromD<DW> ZipLower(V a, V b) {
  return BitCast(DW(), InterleaveLower(a, b));
}
template <class V, class D = DFromV<V>, class DW = RepartitionToWide<D>>
HWY_API VFromD<DW> ZipLower(DW dw, V a, V b) {
  return BitCast(dw, InterleaveLower(D(), a, b));
}

template <class V, class D = DFromV<V>, class DW = RepartitionToWide<D>>
HWY_API VFromD<DW> ZipUpper(DW dw, V a, V b) {
  return BitCast(dw, InterleaveUpper(D(), a, b));
}

// ================================================== COMBINE

// ------------------------------ Combine (InterleaveLower)

// N = N/2 + N/2 (upper half undefined)
template <class D, HWY_IF_V_SIZE_LE_D(D, 16), class VH = VFromD<Half<D>>>
HWY_API VFromD<D> Combine(D d, VH hi_half, VH lo_half) {
  const Half<decltype(d)> dh;
  // Treat half-width input as one lane, and expand to two lanes.
  using VU = Vec128<UnsignedFromSize<dh.MaxBytes()>, 2>;
  const VU lo{(typename detail::Raw128<TFromV<VU>>::type)lo_half.raw};
  const VU hi{(typename detail::Raw128<TFromV<VU>>::type)hi_half.raw};
  return BitCast(d, InterleaveLower(lo, hi));
}

// ------------------------------ ZeroExtendVector (Combine, IfThenElseZero)

template <class D>
HWY_API VFromD<D> ZeroExtendVector(D d, VFromD<Half<D>> lo) {
  const Half<D> dh;
  return IfThenElseZero(FirstN(d, MaxLanes(dh)), VFromD<D>{lo.raw});
}

// ------------------------------ Concat full (InterleaveLower)

// hiH,hiL loH,loL |-> hiL,loL (= lower halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec128<T> ConcatLowerLower(D d, Vec128<T> hi, Vec128<T> lo) {
  const Repartition<uint64_t, decltype(d)> d64;
  return BitCast(d, InterleaveLower(BitCast(d64, lo), BitCast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiH,loH (= upper halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec128<T> ConcatUpperUpper(D d, Vec128<T> hi, Vec128<T> lo) {
  const Repartition<uint64_t, decltype(d)> d64;
  return BitCast(d, InterleaveUpper(d64, BitCast(d64, lo), BitCast(d64, hi)));
}

// hiH,hiL loH,loL |-> hiL,loH (= inner halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec128<T> ConcatLowerUpper(D d, Vec128<T> hi, Vec128<T> lo) {
  return CombineShiftRightBytes<8>(d, hi, lo);
}

// hiH,hiL loH,loL |-> hiH,loL (= outer halves)
template <class D, typename T = TFromD<D>>
HWY_API Vec128<T> ConcatUpperLower(D /*d*/, Vec128<T> hi, Vec128<T> lo) {
  const __vector unsigned char kShuffle = {
    0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31};
  return Vec128<T>{vec_perm(lo.raw, hi.raw, kShuffle)};
}

// ------------------------------ Concat partial (Combine, LowerHalf)

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> ConcatLowerLower(D d, VFromD<D> hi, VFromD<D> lo) {
  const Half<decltype(d)> d2;
  return Combine(d, LowerHalf(d2, hi), LowerHalf(d2, lo));
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> ConcatUpperUpper(D d, VFromD<D> hi, VFromD<D> lo) {
  const Half<decltype(d)> d2;
  return Combine(d, UpperHalf(d2, hi), UpperHalf(d2, lo));
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> ConcatLowerUpper(D d, VFromD<D> hi,
                                   VFromD<D> lo) {
  const Half<decltype(d)> d2;
  return Combine(d, LowerHalf(d2, hi), UpperHalf(d2, lo));
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API VFromD<D> ConcatUpperLower(D d, VFromD<D> hi, VFromD<D> lo) {
  const Half<decltype(d)> d2;
  return Combine(d, UpperHalf(d2, hi), LowerHalf(d2, lo));
}

// ------------------------------ ConcatOdd

// 8-bit full
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec128<T> ConcatOdd(D d, Vec128<T> hi, Vec128<T> lo) {
  const Repartition<uint16_t, decltype(d)> dw;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // Right-shift 8 bits per u16 so we can pack.
  const Vec128<uint16_t> uH = ShiftRight<8>(BitCast(dw, hi));
  const Vec128<uint16_t> uL = ShiftRight<8>(BitCast(dw, lo));
#else
  const Vec128<uint16_t> uH = BitCast(dw, hi);
  const Vec128<uint16_t> uL = BitCast(dw, lo);
#endif
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_pack(uL.raw, uH.raw)};
}

// 8-bit x8
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec64<T> ConcatOdd(D /*d*/, Vec64<T> hi, Vec64<T> lo) {
  // Don't care about upper half, no need to zero.
  const __vector unsigned char kCompactOddU8 = {1, 3, 5, 7, 17, 19, 21, 23};
  return Vec64<T>{vec_perm(lo.raw, hi.raw, kCompactOddU8)};
}

// 8-bit x4
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec32<T> ConcatOdd(D /*d*/, Vec32<T> hi, Vec32<T> lo) {
  // Don't care about upper half, no need to zero.
  const __vector unsigned char kCompactOddU8 = {1, 3, 17, 19};
  return Vec32<T>{vec_perm(lo.raw, hi.raw, kCompactOddU8)};
}

// 16-bit full
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec128<T> ConcatOdd(D d, Vec128<T> hi, Vec128<T> lo) {
  const Repartition<uint32_t, decltype(d)> dw;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const Vec128<uint32_t> uH = ShiftRight<16>(BitCast(dw, hi));
  const Vec128<uint32_t> uL = ShiftRight<16>(BitCast(dw, lo));
#else
  const Vec128<uint32_t> uH = BitCast(dw, hi);
  const Vec128<uint32_t> uL = BitCast(dw, lo);
#endif
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_pack(uL.raw, uH.raw)};
}

// 16-bit x4
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec64<T> ConcatOdd(D /*d*/, Vec64<T> hi, Vec64<T> lo) {
  // Don't care about upper half, no need to zero.
  const __vector unsigned char kCompactOddU16 = {2, 3, 6, 7, 18, 19, 22, 23};
  return Vec64<T>{vec_perm(lo.raw, hi.raw, kCompactOddU16)};
}

// 32-bit full
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> ConcatOdd(D /*d*/, Vec128<T> hi, Vec128<T> lo) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kShuffle = {
    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
  return Vec128<T>{vec_perm(lo.raw, hi.raw, kShuffle)};
#else
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_pack(
    (__vector unsigned long long)lo.raw, (__vector unsigned long long)hi.raw)};
#endif
}

// Any type x2
template <class D, typename T = TFromD<D>, HWY_IF_LANES_D(D, 2)>
HWY_API Vec128<T, 2> ConcatOdd(D d, Vec128<T, 2> hi, Vec128<T, 2> lo) {
  return InterleaveUpper(d, lo, hi);
}

// ------------------------------ ConcatEven (InterleaveLower)

// 8-bit full
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec128<T> ConcatEven(D d, Vec128<T> hi, Vec128<T> lo) {
  const Repartition<uint16_t, decltype(d)> dw;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const Vec128<uint16_t> uH = BitCast(dw, hi);
  const Vec128<uint16_t> uL = BitCast(dw, lo);
#else
  // Right-shift 8 bits per u16 so we can pack.
  const Vec128<uint16_t> uH = ShiftRight<8>(BitCast(dw, hi));
  const Vec128<uint16_t> uL = ShiftRight<8>(BitCast(dw, lo));
#endif
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_pack(uL.raw, uH.raw)};
}

// 8-bit x8
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec64<T> ConcatEven(D /*d*/, Vec64<T> hi, Vec64<T> lo) {
  // Don't care about upper half, no need to zero.
  const __vector unsigned char kCompactEvenU8 = {0, 2, 4, 6, 16, 18, 20, 22};
  return Vec64<T>{vec_perm(lo.raw, hi.raw, kCompactEvenU8)};
}

// 8-bit x4
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 1)>
HWY_API Vec32<T> ConcatEven(D /*d*/, Vec32<T> hi, Vec32<T> lo) {
  // Don't care about upper half, no need to zero.
  const __vector unsigned char kCompactEvenU8 = {0, 2, 16, 18};
  return Vec32<T>{vec_perm(lo.raw, hi.raw, kCompactEvenU8)};
}

// 16-bit full
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec128<T> ConcatEven(D d, Vec128<T> hi, Vec128<T> lo) {
  // Isolate lower 16 bits per u32 so we can pack.
  const Repartition<uint32_t, decltype(d)> dw;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const Vec128<uint32_t> uH = BitCast(dw, hi);
  const Vec128<uint32_t> uL = BitCast(dw, lo);
#else
  const Vec128<uint32_t> uH = ShiftRight<16>(BitCast(dw, hi));
  const Vec128<uint32_t> uL = ShiftRight<16>(BitCast(dw, lo));
#endif
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_pack(uL.raw, uH.raw)};
}

// 16-bit x4
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 2)>
HWY_API Vec64<T> ConcatEven(D /*d*/, Vec64<T> hi, Vec64<T> lo) {
  // Don't care about upper half, no need to zero.
  const __vector unsigned char kCompactEvenU16 = {0, 1, 4, 5, 16, 17, 20, 21};
  return Vec64<T>{vec_perm(lo.raw, hi.raw, kCompactEvenU16)};
}

// 32-bit full
template <class D, typename T = TFromD<D>, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T> ConcatEven(D /*d*/, Vec128<T> hi, Vec128<T> lo) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<T>{(typename detail::Raw128<T>::type)vec_pack(
    (__vector unsigned long long)lo.raw, (__vector unsigned long long)hi.raw)};
#else
  constexpr __vector unsigned char kShuffle = {
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
  return Vec128<T>{vec_perm(lo.raw, hi.raw, kShuffle)};
#endif
}

// Any T x2
template <typename D, typename T = TFromD<D>, HWY_IF_LANES_D(D, 2)>
HWY_API Vec128<T, 2> ConcatEven(D d, Vec128<T, 2> hi, Vec128<T, 2> lo) {
  return InterleaveLower(d, lo, hi);
}

// ------------------------------ DupEven (InterleaveLower)

template <typename T, size_t N, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T, N> DupEven(Vec128<T, N> v) {
  return Vec128<T, N>{vec_mergee(v.raw, v.raw)};
}

template <typename T, size_t N, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec128<T, N> DupEven(Vec128<T, N> v) {
  return InterleaveLower(DFromV<decltype(v)>(), v, v);
}

// ------------------------------ DupOdd (InterleaveUpper)

template <typename T, size_t N, HWY_IF_T_SIZE(T, 4)>
HWY_API Vec128<T, N> DupOdd(Vec128<T, N> v) {
  return Vec128<T, N>{vec_mergeo(v.raw, v.raw)};
}

template <typename T, size_t N, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec128<T, N> DupOdd(Vec128<T, N> v) {
  return InterleaveUpper(DFromV<decltype(v)>(), v, v);
}

// ------------------------------ OddEven (IfThenElse)

template <typename T, size_t N, HWY_IF_T_SIZE(T, 1)>
HWY_INLINE Vec128<T, N> OddEven(Vec128<T, N> a, Vec128<T, N> b) {
  const DFromV<decltype(a)> d;
  const __vector unsigned char mask = {0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0,
                                       0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0};
  return IfVecThenElse(BitCast(Simd<T, N, 0>(), Vec128<uint8_t, N>{mask}), b,
                       a);
}

template <typename T, size_t N, HWY_IF_T_SIZE(T, 2)>
HWY_INLINE Vec128<T, N> OddEven(Vec128<T, N> a, Vec128<T, N> b) {
  const __vector unsigned char mask = {0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0,
                                       0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0};
  return IfVecThenElse(BitCast(Simd<T, N, 0>(), Vec128<uint8_t, N * 2>{mask}),
                       b, a);
}

template <typename T, size_t N, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T, N> OddEven(Vec128<T, N> a, Vec128<T, N> b) {
  const __vector unsigned char mask = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0,
                                       0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
  return IfVecThenElse(BitCast(Simd<T, N, 0>(), Vec128<uint8_t, N * 4>{mask}),
                       b, a);
}

template <typename T, size_t N, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Vec128<T, N> OddEven(Vec128<T, N> a, Vec128<T, N> b) {
  // Same as ConcatUpperLower for full vectors; do not call that because this
  // is more efficient for 64x1 vectors.
  const __vector unsigned char mask = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0, 0, 0, 0, 0, 0, 0, 0};
  return IfVecThenElse(BitCast(Simd<T, N, 0>(), Vec128<uint8_t, N * 8>{mask}),
                       b, a);
}

// ------------------------------ OddEvenBlocks
template <typename T, size_t N>
HWY_API Vec128<T, N> OddEvenBlocks(Vec128<T, N> /* odd */, Vec128<T, N> even) {
  return even;
}

// ------------------------------ SwapAdjacentBlocks

template <typename T, size_t N>
HWY_API Vec128<T, N> SwapAdjacentBlocks(Vec128<T, N> v) {
  return v;
}

// ------------------------------ Shl

namespace detail {
template <typename T, size_t N>
HWY_API Vec128<T, N> Shl(hwy::UnsignedTag /*tag*/, Vec128<T, N> v,
                         Vec128<T, N> bits) {
  return Vec128<T, N>{vec_sl(v.raw, bits.raw)};
}

// Signed left shift is the same as unsigned.
template <typename T, size_t N>
HWY_API Vec128<T, N> Shl(hwy::SignedTag /*tag*/, Vec128<T, N> v,
                         Vec128<T, N> bits) {
  const DFromV<decltype(v)> di;
  const RebindToUnsigned<decltype(di)> du;
  return BitCast(di,
                 Shl(hwy::UnsignedTag(), BitCast(du, v), BitCast(du, bits)));
}

}  // namespace detail

template <typename T, size_t N, HWY_IF_NOT_FLOAT(T)>
HWY_API Vec128<T, N> operator<<(Vec128<T, N> v, Vec128<T, N> bits) {
  return detail::Shl(hwy::TypeTag<T>(), v, bits);
}

// ------------------------------ Shr

namespace detail {
template <typename T, size_t N>
HWY_API Vec128<T, N> Shr(hwy::UnsignedTag /*tag*/, Vec128<T, N> v,
                         Vec128<T, N> bits) {
  return Vec128<T, N>{vec_sr(v.raw, bits.raw)};
}

template <typename T, size_t N>
HWY_API Vec128<T, N> Shr(hwy::SignedTag /*tag*/, Vec128<T, N> v,
                         Vec128<T, N> bits) {
  const DFromV<decltype(v)> di;
  const RebindToUnsigned<decltype(di)> du;
  return Vec128<T, N>{vec_sra(v.raw, BitCast(du, bits).raw)};
}

}  // namespace detail

template <typename T, size_t N>
HWY_API Vec128<T, N> operator>>(Vec128<T, N> v, Vec128<T, N> bits) {
  return detail::Shr(hwy::TypeTag<T>(), v, bits);
}

// ------------------------------ MulEven/Odd 64x64 (UpperHalf)

HWY_INLINE Vec128<uint64_t> MulEven(Vec128<uint64_t> a,
                                    Vec128<uint64_t> b) {
#if HWY_TARGET <= HWY_PPC10 && defined(__SIZEOF_INT128__)
  const auto mul128_result =
    (__vector unsigned long long)vec_mule(a.raw, b.raw);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<uint64_t>{mul128_result};
#else
  // Need to swap the two halves of mul128_result on big-endian targets as
  // the upper 64 bits of the product are in lane 0 of mul128_result and
  // the lower 64 bits of the product are in lane 1 of mul128_result
  return Vec128<uint64_t>{vec_sld(mul128_result, mul128_result, 8)};
#endif
#else
  alignas(16) uint64_t mul[2];
  mul[0] = Mul128(GetLane(a), GetLane(b), &mul[1]);
  return Load(Full128<uint64_t>(), mul);
#endif
}

HWY_INLINE Vec128<uint64_t> MulOdd(Vec128<uint64_t> a,
                                   Vec128<uint64_t> b) {
#if HWY_TARGET <= HWY_PPC10 && defined(__SIZEOF_INT128__)
  const auto mul128_result =
    (__vector unsigned long long)vec_mulo(a.raw, b.raw);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<uint64_t>{mul128_result};
#else
  // Need to swap the two halves of mul128_result on big-endian targets as
  // the upper 64 bits of the product are in lane 0 of mul128_result and
  // the lower 64 bits of the product are in lane 1 of mul128_result
  return Vec128<uint64_t>{vec_sld(mul128_result, mul128_result, 8)};
#endif
#else
  alignas(16) uint64_t mul[2];
  const Full64<uint64_t> d2;
  mul[0] =
      Mul128(GetLane(UpperHalf(d2, a)), GetLane(UpperHalf(d2, b)), &mul[1]);
  return Load(Full128<uint64_t>(), mul);
#endif
}

// ------------------------------ ReorderWidenMulAccumulate (MulAdd, ZipLower)

template <class D32, HWY_IF_F32_D(D32),
          class V16 = VFromD<Repartition<bfloat16_t, D32>>>
HWY_API VFromD<D32> ReorderWidenMulAccumulate(D32 df32, V16 a, V16 b,
                                              VFromD<D32> sum0,
                                              VFromD<D32>& sum1) {
  const RebindToUnsigned<decltype(df32)> du32;
  // Lane order within sum0/1 is undefined, hence we can avoid the
  // longer-latency lane-crossing PromoteTo. Using shift/and instead of Zip
  // leads to the odd/even order that RearrangeToOddPlusEven prefers.
  using VU32 = VFromD<decltype(du32)>;
  const VU32 odd = Set(du32, 0xFFFF0000u);
  const VU32 ae = ShiftLeft<16>(BitCast(du32, a));
  const VU32 ao = And(BitCast(du32, a), odd);
  const VU32 be = ShiftLeft<16>(BitCast(du32, b));
  const VU32 bo = And(BitCast(du32, b), odd);
  sum1 = MulAdd(BitCast(df32, ao), BitCast(df32, bo), sum1);
  return MulAdd(BitCast(df32, ae), BitCast(df32, be), sum0);
}

// Even if N=1, the input is always at least 2 lanes, hence vec_msum is safe.
template <class D32, HWY_IF_I32_D(D32),
          class V16 = VFromD<RepartitionToNarrow<D32>>>
HWY_API VFromD<D32> ReorderWidenMulAccumulate(D32 /* tag */, V16 a, V16 b,
                                              VFromD<D32> sum0,
                                              VFromD<D32>& /*sum1*/) {
  return VFromD<D32>{vec_msum(a.raw, b.raw, sum0.raw)};
}

// ------------------------------ RearrangeToOddPlusEven
template <size_t N>
HWY_API Vec128<int32_t, N> RearrangeToOddPlusEven(Vec128<int32_t, N> sum0,
                                                  Vec128<int32_t, N> /*sum1*/) {
  return sum0;  // invariant already holds
}

template <class VW>
HWY_API VW RearrangeToOddPlusEven(const VW sum0, const VW sum1) {
  return Add(sum0, sum1);
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned to signed/unsigned: zero-extend.
template <class D, typename FromT, HWY_IF_T_SIZE_D(D, 2 * sizeof(FromT)),
          HWY_IF_NOT_FLOAT_D(D), HWY_IF_UNSIGNED(FromT)>
HWY_API VFromD<D> PromoteTo(D /* d */,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  const DFromV<decltype(v)> dn;
  const auto zero = Zero(dn);

  using Raw = typename detail::Raw128<TFromD<D>>::type;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return VFromD<D>{(Raw)vec_mergeh(v.raw, zero.raw)};
#else
  return VFromD<D>{(Raw)vec_mergeh(zero.raw, v.raw)};
#endif
}

// Signed: replicate sign bit.
template <class D, typename FromT, HWY_IF_T_SIZE_D(D, 2 * sizeof(FromT)),
          HWY_IF_NOT_FLOAT_D(D), HWY_IF_SIGNED(FromT)>
HWY_API VFromD<D> PromoteTo(D /* d */,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  using Raw = typename detail::Raw128<TFromD<D>>::type;
  return VFromD<D>{(Raw)vec_unpackh(v.raw)};
}

// 8-bit to 32-bit: First, promote to 16-bit, and then convert to 32-bit.
template <class D, typename FromT, HWY_IF_T_SIZE_D(D, 4), HWY_IF_NOT_FLOAT_D(D),
          HWY_IF_T_SIZE(FromT, 1)>
HWY_API VFromD<D> PromoteTo(D d32,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  const DFromV<decltype(v)> d8;
  const Rebind<MakeWide<FromT>, decltype(d8)> d16;
  return PromoteTo(d32, PromoteTo(d16, v));
}

// Workaround for origin tracking bug in Clang msan prior to 11.0
// (spurious "uninitialized memory" for TestF16 with "ORIGIN: invalid")
#if HWY_IS_MSAN && (HWY_COMPILER_CLANG != 0 && HWY_COMPILER_CLANG < 1100)
#define HWY_INLINE_F16 HWY_NOINLINE
#else
#define HWY_INLINE_F16 HWY_INLINE
#endif
template <class D, HWY_IF_F32_D(D)>
HWY_INLINE_F16 VFromD<D> PromoteTo(D df32, VFromD<Rebind<float16_t, D>> v) {
#if HWY_TARGET <= HWY_PPC9
  (void)df32;
  return VFromD<D>{vec_extract_fp32_from_shorth(v.raw)};
#else
  const RebindToSigned<decltype(df32)> di32;
  const RebindToUnsigned<decltype(df32)> du32;
  // Expand to u32 so we can shift.
  const auto bits16 = PromoteTo(du32, VFromD<Rebind<uint16_t, D>>{v.raw});
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
#endif
}

template <class D, HWY_IF_F32_D(D)>
HWY_API VFromD<D> PromoteTo(D df32, VFromD<Rebind<bfloat16_t, D>> v) {
  const Rebind<uint16_t, decltype(df32)> du16;
  const RebindToSigned<decltype(df32)> di32;
  return BitCast(df32, ShiftLeft<16>(PromoteTo(di32, BitCast(du16, v))));
}

template <class D, HWY_IF_F64_D(D)>
HWY_API VFromD<D> PromoteTo(D /* tag */, VFromD<Rebind<float, D>> v) {
  const __vector float raw_v = vec_mergeh(v.raw, v.raw);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return VFromD<D>{vec_doubleo(raw_v)};
#else
  return VFromD<D>{vec_doublee(raw_v)};
#endif
}

template <class D, HWY_IF_F64_D(D)>
HWY_API VFromD<D> PromoteTo(D /* tag */, VFromD<Rebind<int32_t, D>> v) {
  const __vector signed int raw_v = vec_mergeh(v.raw, v.raw);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return VFromD<D>{vec_doubleo(raw_v)};
#else
  return VFromD<D>{vec_doublee(raw_v)};
#endif
}

// ------------------------------ Truncations

template <class D, typename FromT, HWY_IF_UNSIGNED_D(D), HWY_IF_UNSIGNED(FromT),
          hwy::EnableIf<(sizeof(FromT) >= sizeof(TFromD<D>) * 2)>* = nullptr,
          HWY_IF_LANES_D(D, 1)>
HWY_API VFromD<D> TruncateTo(D /* tag */, Vec128<FromT, 1> v) {
  return VFromD<D>{(typename detail::Raw128<TFromD<D>>::type)v.raw};
}

template <class D, typename FromT, HWY_IF_UNSIGNED_D(D), HWY_IF_UNSIGNED(FromT),
          HWY_IF_T_SIZE(FromT, sizeof(TFromD<D>) * 2), HWY_IF_LANES_GT_D(D, 1)>
HWY_API VFromD<D> TruncateTo(D /* tag */,
                             Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_pack(v.raw, v.raw)};
}

template <class D, typename FromT, HWY_IF_UNSIGNED_D(D), HWY_IF_UNSIGNED(FromT),
          hwy::EnableIf<(sizeof(FromT) >= sizeof(TFromD<D>) * 4)>* = nullptr,
          HWY_IF_LANES_GT_D(D, 1)>
HWY_API VFromD<D> TruncateTo(D d,
                             Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  const Rebind<MakeNarrow<FromT>, decltype(d)> d2;
  return TruncateTo(d, TruncateTo(d2, v));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

template <class D, typename FromT, HWY_IF_V_SIZE_LE_D(D, 8),
          HWY_IF_UNSIGNED_D(D), HWY_IF_T_SIZE_ONE_OF_D(D, (1 << 1) | (1 << 2)),
          HWY_IF_SIGNED(FromT), HWY_IF_T_SIZE(FromT, sizeof(TFromD<D>) * 2)>
HWY_API VFromD<D> DemoteTo(D /* tag */,
                           Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_packsu(v.raw, v.raw)};
}

template <class D, typename FromT, HWY_IF_V_SIZE_LE_D(D, 8), HWY_IF_SIGNED_D(D),
          HWY_IF_SIGNED(FromT), HWY_IF_T_SIZE_ONE_OF_D(D, (1 << 1) | (1 << 2)),
          HWY_IF_T_SIZE(FromT, sizeof(TFromD<D>) * 2)>
HWY_API VFromD<D> DemoteTo(D /* tag */,
                           Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_packs(v.raw, v.raw)};
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 4), HWY_IF_T_SIZE_D(D, 1)>
HWY_API VFromD<D> DemoteTo(D d, VFromD<Rebind<int32_t, D>> v) {
  const Rebind<int16_t, decltype(d)> di16;
  return DemoteTo(d, DemoteTo(di16, v));
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8), HWY_IF_F16_D(D)>
HWY_API VFromD<D> DemoteTo(D df16, VFromD<Rebind<float, D>> v) {
#if HWY_TARGET <= HWY_PPC9 && HWY_COMPILER_GCC_ACTUAL
  // Do not use vec_pack_to_short_fp32 on clang as there is a bug in the clang
  // version of vec_pack_to_short_fp32
  (void)df16;
  return VFromD<D>{vec_pack_to_short_fp32(v.raw, v.raw)};
#else
  const Rebind<uint32_t, decltype(df16)> du;
  const RebindToUnsigned<decltype(df16)> du16;
#if HWY_TARGET <= HWY_PPC9 && HWY_HAS_BUILTIN(__builtin_vsx_xvcvsphp)
  // Work around bug in the clang implementation of vec_pack_to_short_fp32
  // by using the __builtin_vsx_xvcvsphp builtin on PPC9/PPC10 targets
  // if the __builtin_vsx_xvcvsphp intrinsic is available
  const VFromD<decltype(du)> bits16{
    (__vector unsigned int)__builtin_vsx_xvcvsphp(v.raw)};
#else
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
  const auto bits16 = IfThenZeroElse(RebindMask(du, is_tiny), normal16);
#endif  // HWY_TARGET <= HWY_PPC9 && HWY_HAS_BUILTIN(__builtin_vsx_xvcvsphp)
  return BitCast(df16, TruncateTo(du16, bits16));
#endif  // HWY_TARGET <= HWY_PPC9 && HWY_COMPILER_GCC_ACTUAL
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8), HWY_IF_BF16_D(D)>
HWY_API VFromD<D> DemoteTo(D dbf16, VFromD<Rebind<float, D>> v) {
  const Rebind<uint32_t, decltype(dbf16)> du32;  // for logical shift right
  const Rebind<uint16_t, decltype(dbf16)> du16;
  const auto bits_in_32 = ShiftRight<16>(BitCast(du32, v));
  return BitCast(dbf16, TruncateTo(du16, bits_in_32));
}

template <class D, HWY_IF_BF16_D(D), class V32 = VFromD<Repartition<float, D>>>
HWY_API VFromD<D> ReorderDemote2To(D dbf16, V32 a, V32 b) {
  const RebindToUnsigned<decltype(dbf16)> du16;
  const Repartition<uint32_t, decltype(dbf16)> du32;
  const auto b_in_even = ShiftRight<16>(BitCast(du32, b));
  return BitCast(dbf16, OddEven(BitCast(du16, a), BitCast(du16, b_in_even)));
}

// Specializations for partial vectors because packs_epi32 sets lanes above 2*N.
template <class D, HWY_IF_I16_D(D)>
HWY_API Vec32<int16_t> ReorderDemote2To(D dn, Vec32<int32_t> a,
                                        Vec32<int32_t> b) {
  const Half<decltype(dn)> dnh;
  // Pretend the result has twice as many lanes so we can InterleaveLower.
  const Vec32<int16_t> an{DemoteTo(dnh, a).raw};
  const Vec32<int16_t> bn{DemoteTo(dnh, b).raw};
  return InterleaveLower(an, bn);
}
template <class D, HWY_IF_I16_D(D)>
HWY_API Vec64<int16_t> ReorderDemote2To(D dn, Vec64<int32_t> a,
                                        Vec64<int32_t> b) {
  const Half<decltype(dn)> dnh;
  // Pretend the result has twice as many lanes so we can InterleaveLower.
  const Vec64<int16_t> an{DemoteTo(dnh, a).raw};
  const Vec64<int16_t> bn{DemoteTo(dnh, b).raw};
  return InterleaveLower(an, bn);
}
template <class D, HWY_IF_I16_D(D)>
HWY_API Vec128<int16_t> ReorderDemote2To(D /* tag */, Vec128<int32_t> a,
                                         Vec128<int32_t> b) {
  return Vec128<int16_t>{vec_packs(a.raw, b.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 4), HWY_IF_F32_D(D)>
HWY_API Vec32<float> DemoteTo(D /* tag */, Vec64<double> v) {
  return Vec32<float>{vec_floate(v.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 8), HWY_IF_F32_D(D)>
HWY_API Vec64<float> DemoteTo(D d, Vec128<double> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const Vec128<float> f64_to_f32{vec_floate(v.raw)};
#else
  const Vec128<float> f64_to_f32{vec_floato(v.raw)};
#endif

  const RebindToUnsigned<D> du;
  const Rebind<uint64_t, D> du64;
  return Vec64<float>{
      BitCast(d, TruncateTo(du, BitCast(du64, f64_to_f32))).raw};
}

template <class D, HWY_IF_V_SIZE_D(D, 4), HWY_IF_I32_D(D)>
HWY_API Vec32<int32_t> DemoteTo(D /* tag */, Vec64<double> v) {
  return Vec32<int32_t>{vec_signede(v.raw)};
}

template <class D, HWY_IF_V_SIZE_D(D, 8), HWY_IF_I32_D(D)>
HWY_API Vec64<int32_t> DemoteTo(D /* tag */, Vec128<double> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const Vec128<int32_t> f64_to_i32{vec_signede(v.raw)};
#else
  const Vec128<int32_t> f64_to_i32{vec_signedo(v.raw)};
#endif

  const Rebind<int64_t, D> di64;
  const Vec128<int64_t> vi64 = BitCast(di64, f64_to_i32);
  return Vec64<int32_t>{vec_pack(vi64.raw, vi64.raw)};
}

// For already range-limited input [0, 255].
template <size_t N>
HWY_API Vec128<uint8_t, N> U8FromU32(Vec128<uint32_t, N> v) {
  const __vector unsigned short v16 = vec_pack(v.raw, v.raw);
  return Vec128<uint8_t, N>{vec_pack(v16, v16)};
}
// ------------------------------ Integer <=> fp (ShiftRight, OddEven)

template <class D, typename FromT, HWY_IF_F32_D(D), HWY_IF_NOT_FLOAT(FromT),
          HWY_IF_T_SIZE_D(D, sizeof(FromT))>
HWY_API VFromD<D> ConvertTo(D /* tag */,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_ctf(v.raw, 0)};
}

template <class D, typename FromT, HWY_IF_F64_D(D), HWY_IF_NOT_FLOAT(FromT),
          HWY_IF_T_SIZE_D(D, sizeof(FromT))>
HWY_API VFromD<D> ConvertTo(D /* tag */,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_double(v.raw)};
}

// Truncates (rounds toward zero).
template <class D, typename FromT, HWY_IF_SIGNED_D(D), HWY_IF_FLOAT(FromT),
          HWY_IF_T_SIZE_D(D, sizeof(FromT))>
HWY_API VFromD<D> ConvertTo(D /* tag */,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_cts(v.raw, 0)};
}

template <class D, typename FromT, HWY_IF_UNSIGNED_D(D), HWY_IF_FLOAT(FromT),
          HWY_IF_T_SIZE_D(D, sizeof(FromT))>
HWY_API VFromD<D> ConvertTo(D /* tag */,
                            Vec128<FromT, Rebind<FromT, D>().MaxLanes()> v) {
  return VFromD<D>{vec_ctu(v.raw, 0)};
}

template <size_t N>
HWY_API Vec128<int32_t, N> NearestInt(Vec128<float, N> v) {
  return Vec128<int32_t, N>{vec_cts(vec_round(v.raw), 0)};
}

// ------------------------------ Floating-point rounding (ConvertTo)

// Toward nearest integer, ties to even
template <size_t N>
HWY_API Vec128<float, N> Round(Vec128<float, N> v) {
  return Vec128<float, N>{vec_round(v.raw)};
}

template <size_t N>
HWY_API Vec128<double, N> Round(Vec128<double, N> v) {
  return Vec128<double, N>{vec_rint(v.raw)};
}

// Toward zero, aka truncate
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> Trunc(Vec128<T, N> v) {
  return Vec128<T, N>{vec_trunc(v.raw)};
}

// Toward +infinity, aka ceiling
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> Ceil(Vec128<T, N> v) {
  return Vec128<T, N>{vec_ceil(v.raw)};
}

// Toward -infinity, aka floor
template <typename T, size_t N, HWY_IF_FLOAT(T)>
HWY_API Vec128<T, N> Floor(Vec128<T, N> v) {
  return Vec128<T, N>{vec_floor(v.raw)};
}

// ------------------------------ Floating-point classification

template <typename T, size_t N>
HWY_API Mask128<T, N> IsNaN(Vec128<T, N> v) {
  static_assert(IsFloat<T>(), "Only for float");
  return v != v;
}

template <typename T, size_t N>
HWY_API Mask128<T, N> IsInf(Vec128<T, N> v) {
  static_assert(IsFloat<T>(), "Only for float");
  using TU = MakeUnsigned<T>;
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du;
  const VFromD<decltype(du)> vu = BitCast(du, v);
  // 'Shift left' to clear the sign bit, check for exponent=max and mantissa=0.
  return RebindMask(d, Eq(Add(vu, vu),
    Set(du, static_cast<TU>(hwy::MaxExponentTimes2<T>()))));
}

// Returns whether normal/subnormal/zero.
template <typename T, size_t N>
HWY_API Mask128<T, N> IsFinite(Vec128<T, N> v) {
  static_assert(IsFloat<T>(), "Only for float");
  using TU = MakeUnsigned<T>;
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du;
  const VFromD<decltype(du)> vu = BitCast(du, v);
  // 'Shift left' to clear the sign bit, check for exponent<max.
  return RebindMask(d, Lt(Add(vu, vu),
    Set(du, static_cast<TU>(hwy::MaxExponentTimes2<T>()))));
}

// ================================================== CRYPTO

#if !defined(HWY_DISABLE_PPC8_CRYPTO)

// Per-target flag to prevent generic_ops-inl.h from defining AESRound.
#ifdef HWY_NATIVE_AES
#undef HWY_NATIVE_AES
#else
#define HWY_NATIVE_AES
#endif

namespace detail {
#if HWY_COMPILER_CLANG
using VSXCipherRawVectType = __vector unsigned long long;
#else
using VSXCipherRawVectType = __vector unsigned char;
#endif
}

HWY_API Vec128<uint8_t> AESRound(Vec128<uint8_t> state,
                                 Vec128<uint8_t> round_key) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<uint8_t>{vec_reve((__vector unsigned char)vec_cipher_be(
    (detail::VSXCipherRawVectType)vec_reve(state.raw),
    (detail::VSXCipherRawVectType)vec_reve(round_key.raw)))};
#else
  return Vec128<uint8_t>{(__vector unsigned char)
    vec_cipher_be((detail::VSXCipherRawVectType)state.raw,
                  (detail::VSXCipherRawVectType)round_key.raw)};
#endif
}

HWY_API Vec128<uint8_t> AESLastRound(Vec128<uint8_t> state,
                                     Vec128<uint8_t> round_key) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return Vec128<uint8_t>{vec_reve((__vector unsigned char)vec_cipherlast_be(
    (detail::VSXCipherRawVectType)vec_reve(state.raw),
    (detail::VSXCipherRawVectType)vec_reve(round_key.raw)))};
#else
  return Vec128<uint8_t>{(__vector unsigned char)vec_cipherlast_be(
    (detail::VSXCipherRawVectType)state.raw,
    (detail::VSXCipherRawVectType)round_key.raw)};
#endif
}

template <size_t N>
HWY_API Vec128<uint64_t, N> CLMulLower(Vec128<uint64_t, N> a,
                                       Vec128<uint64_t, N> b) {
  // NOTE: Lane 1 of both a and b need to be zeroed out for the
  // vec_pmsum_be operation below as the vec_pmsum_be operation
  // does a carryless multiplication of each 64-bit half and then
  // adds the two halves using an bitwise XOR operation.

  const DFromV<decltype(a)> d;
  const auto zero = Zero(d);

  return Vec128<uint64_t, N>{(typename detail::Raw128<uint64_t>::type)
    vec_pmsum_be(vec_mergeh(a.raw, zero.raw),
                 vec_mergeh(b.raw, zero.raw))};
}

template <size_t N>
HWY_API Vec128<uint64_t, N> CLMulUpper(Vec128<uint64_t, N> a,
                                       Vec128<uint64_t, N> b) {
  // NOTE: Lane 0 of both a and b need to be zeroed out for the
  // vec_pmsum_be operation below as the vec_pmsum_be operation
  // does a carryless multiplication of each 64-bit half and then
  // adds the two halves using an bitwise XOR operation.

  const DFromV<decltype(a)> d;
  const auto zero = Zero(d);

  return Vec128<uint64_t, N>{(typename detail::Raw128<uint64_t>::type)
    vec_pmsum_be(vec_mergel(zero.raw, a.raw),
                 vec_mergel(zero.raw, b.raw))};
}

#endif  // !defined(HWY_DISABLE_PPC8_CRYPTO)

// ================================================== MISC

// ------------------------------ LoadMaskBits (TestBit)

namespace detail {
template <class D, HWY_IF_T_SIZE_D(D, 1)>
HWY_INLINE MFromD<D> LoadMaskBits128(D /*d*/, uint64_t mask_bits) {
  const __vector unsigned char vbits =
    (__vector unsigned char)vec_splats(static_cast<uint16_t>(mask_bits));

  // Replicate bytes 8x such that each byte contains the bit that governs it.
  const __vector unsigned char kRep8 = {0, 0, 0, 0, 0, 0, 0, 0,
                                        1, 1, 1, 1, 1, 1, 1, 1};
  const Vec128<uint8_t> rep8{vec_perm(vbits, vbits, kRep8)};

  const __vector unsigned char kBit = {1, 2, 4, 8, 16, 32, 64, 128,
                                       1, 2, 4, 8, 16, 32, 64, 128};
  return MFromD<D>{TestBit(rep8, Vec128<uint8_t>{kBit}).raw};
}

template <class D, HWY_IF_T_SIZE_D(D, 2)>
HWY_INLINE MFromD<D> LoadMaskBits128(D /*d*/, uint64_t mask_bits) {
  const __vector unsigned short kBit = {1, 2, 4, 8, 16, 32, 64, 128};
  const auto vmask_bits =
    Set(Full128<uint16_t>(), static_cast<uint16_t>(mask_bits));
  return MFromD<D>{TestBit(vmask_bits, Vec128<uint16_t>{kBit}).raw};
}

template <class D, HWY_IF_T_SIZE_D(D, 4)>
HWY_INLINE MFromD<D> LoadMaskBits128(D /*d*/, uint64_t mask_bits) {
  const __vector unsigned int kBit = {1, 2, 4, 8};
  const auto vmask_bits =
    Set(Full128<uint32_t>(), static_cast<uint32_t>(mask_bits));
  return MFromD<D>{TestBit(vmask_bits, Vec128<uint32_t>{kBit}).raw};
}

template <class D, HWY_IF_T_SIZE_D(D, 8)>
HWY_INLINE MFromD<D> LoadMaskBits128(D /*d*/, uint64_t mask_bits) {
  const __vector unsigned long long kBit = {1, 2};
  const auto vmask_bits =
    Set(Full128<uint64_t>(), static_cast<uint64_t>(mask_bits));
  return MFromD<D>{TestBit(vmask_bits, Vec128<uint64_t>{kBit}).raw};
}

}

// `p` points to at least 8 readable bytes, not all of which need be valid.
template <class D, HWY_IF_LANES_LE_D(D, 8)>
HWY_API MFromD<D> LoadMaskBits(D d, const uint8_t* HWY_RESTRICT bits) {
  // If there are 8 or fewer lanes, simply convert bits[0] to a uint64_t
  uint64_t mask_bits = bits[0];

  constexpr size_t kN = MaxLanes(d);
  if(kN < 8)
    mask_bits &= (1u << kN) - 1;

  return detail::LoadMaskBits128(d, mask_bits);
}

template <class D, HWY_IF_LANES_D(D, 16)>
HWY_API MFromD<D> LoadMaskBits(D d, const uint8_t* HWY_RESTRICT bits) {
  // First, copy the mask bits to a uint16_t as there as there are at most
  // 16 lanes in a vector.

  // Copying the mask bits to a uint16_t first will also ensure that the
  // mask bits are loaded into the lower 16 bits on big-endian PPC targets.
  uint16_t u16_mask_bits;
  CopyBytes<sizeof(uint16_t)>(bits, &u16_mask_bits);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return detail::LoadMaskBits128(d, u16_mask_bits);
#else
  // On big-endian targets, u16_mask_bits need to be byte swapped as bits
  // contains the mask bits in little-endian byte order

  // GCC/Clang will optimize the load of u16_mask_bits and byte swap to a
  // single lhbrx instruction on big-endian PPC targets when optimizations
  // are enabled.
#if HWY_HAS_BUILTIN(__builtin_bswap16)
  return detail::LoadMaskBits128(d, __builtin_bswap16(u16_mask_bits));
#else
  return detail::LoadMaskBits128(d,
    static_cast<uint16_t>((u16_mask_bits << 8) | (u16_mask_bits >> 8)));
#endif
#endif
}

template <typename T>
struct CompressIsPartition {
  // generic_ops-inl does not guarantee IsPartition for 8-bit.
  enum { value = (sizeof(T) != 1) };
};

// ------------------------------ StoreMaskBits

namespace detail {

template <typename T, size_t N>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<1> /*tag*/,
                                 Mask128<T, N> mask) {
  const __vector unsigned char sign_bits =
    (__vector unsigned char)mask.raw;
#if HWY_TARGET <= HWY_PPC10
  return static_cast<unsigned>(vec_extractm(sign_bits));
#else
  const __vector unsigned char kBitShuffle = {
    120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0
  };

  const __vector unsigned long long extracted_bits_vect =
    (__vector unsigned long long)vec_vbpermq(sign_bits, kBitShuffle);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return extracted_bits_vect[1];
#else
  return extracted_bits_vect[0];
#endif
#endif  // HWY_TARGET <= HWY_PPC10
}

template <typename T, size_t N>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<2> /*tag*/,
                                 Mask128<T, N> mask) {
  const __vector unsigned char sign_bits =
    (__vector unsigned char)mask.raw;
#if HWY_TARGET <= HWY_PPC10
  return static_cast<unsigned>(
    vec_extractm((__vector unsigned short)sign_bits));
#else
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kBitShuffle = {
    112, 96, 80, 64, 48, 32, 16, 0, 128, 128, 128, 128, 128, 128, 128, 128};
#else
  const __vector unsigned char kBitShuffle = {
    128, 128, 128, 128, 128, 128, 128, 128, 112, 96, 80, 64, 48, 32, 16, 0};
#endif

  const __vector unsigned long long extracted_bits_vect =
    (__vector unsigned long long)vec_vbpermq(sign_bits, kBitShuffle);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return extracted_bits_vect[1];
#else
  return extracted_bits_vect[0];
#endif
#endif  // HWY_TARGET <= HWY_PPC10
}

template <typename T, size_t N>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<4> /*tag*/,
                                 Mask128<T, N> mask) {
  const __vector unsigned char sign_bits =
    (__vector unsigned char)mask.raw;
#if HWY_TARGET <= HWY_PPC10
  return static_cast<unsigned>(vec_extractm((__vector unsigned int)sign_bits));
#else
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kBitShuffle = {
    96, 64, 32, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
#else
  const __vector unsigned char kBitShuffle = {
     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 96, 64, 32, 0};
#endif

  const __vector unsigned long long extracted_bits_vect =
    (__vector unsigned long long)vec_vbpermq(sign_bits, kBitShuffle);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return extracted_bits_vect[1];
#else
  return extracted_bits_vect[0];
#endif
#endif  // HWY_TARGET <= HWY_PPC10
}

template <typename T, size_t N>
HWY_INLINE uint64_t BitsFromMask(hwy::SizeTag<8> /*tag*/,
                                 Mask128<T, N> mask) {
  const __vector unsigned char sign_bits =
    (__vector unsigned char)mask.raw;
#if HWY_TARGET <= HWY_PPC10
  return static_cast<unsigned>(
    vec_extractm((__vector unsigned long long)sign_bits));
#else
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned char kBitShuffle = {
     64, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
#else
  const __vector unsigned char kBitShuffle = {
     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 0};
#endif

  const __vector unsigned long long extracted_bits_vect =
    (__vector unsigned long long)vec_vbpermq(sign_bits, kBitShuffle);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return extracted_bits_vect[1];
#else
  return extracted_bits_vect[0];
#endif
#endif  // HWY_TARGET <= HWY_PPC10
}

// Returns the lowest N of the mask bits.
template <typename T, size_t N>
constexpr uint64_t OnlyActive(uint64_t mask_bits) {
  return ((N * sizeof(T)) == 16) ? mask_bits : mask_bits & ((1ull << N) - 1);
}

template <typename T, size_t N>
HWY_INLINE uint64_t BitsFromMask(Mask128<T, N> mask) {
  return OnlyActive<T, N>(BitsFromMask(hwy::SizeTag<sizeof(T)>(), mask));
}

}  // namespace detail

// `p` points to at least 8 writable bytes.
template <class D, HWY_IF_LANES_LE_D(D, 8)>
HWY_API size_t StoreMaskBits(D /*d*/, MFromD<D> mask, uint8_t* bits) {
  // For vectors with 8 or fewer lanes, simply cast the result of BitsFromMask
  // to an uint8_t and store the result in bits[0].
  bits[0] = static_cast<uint8_t>(detail::BitsFromMask(mask));
  return sizeof(uint8_t);
}

template <class D, HWY_IF_LANES_D(D, 16)>
HWY_API size_t StoreMaskBits(D /*d*/, MFromD<D> mask, uint8_t* bits) {
  const auto mask_bits = detail::BitsFromMask(mask);

  // First convert mask_bits to a uint16_t as we only want to store
  // the lower 16 bits of mask_bits as there are 16 lanes in mask.

  // Converting mask_bits to a uint16_t first will also ensure that
  // the lower 16 bits of mask_bits are stored instead of the upper 16 bits
  // of mask_bits on big-endian PPC targets.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const uint16_t u16_mask_bits = static_cast<uint16_t>(mask_bits);
#else
  // On big-endian targets, the bytes of mask_bits need to be swapped
  // as StoreMaskBits expects the mask bits to be stored in little-endian
  // byte order.

  // GCC will also optimize the byte swap and CopyBytes operations below
  // to a single sthbrx instruction when optimizations are enabled on
  // big-endian PPC targets
#if HWY_HAS_BUILTIN(__builtin_bswap16)
  const uint16_t u16_mask_bits =
    __builtin_bswap16(static_cast<uint16_t>(mask_bits));
#else
  const uint16_t u16_mask_bits = static_cast<uint16_t>(
    (mask_bits << 8) | (static_cast<uint16_t>(mask_bits) >> 8));
#endif
#endif

  CopyBytes<sizeof(uint16_t)>(&u16_mask_bits, bits);
  return sizeof(uint16_t);
}

// ------------------------------ Mask testing

template <class D, HWY_IF_V_SIZE_D(D, 16)>
HWY_API bool AllFalse(D /* tag */, MFromD<D> mask) {
  using UnsignedRawVectType =
    typename detail::Raw128<MakeUnsigned<TFromD<D>>>::type;
  return static_cast<bool>(vec_all_eq((UnsignedRawVectType)mask.raw,
                                      (UnsignedRawVectType)vec_splats(0)));
}

template <class D, HWY_IF_V_SIZE_D(D, 16)>
HWY_API bool AllTrue(D /* tag */, MFromD<D> mask) {
  using UnsignedRawVectType =
    typename detail::Raw128<MakeUnsigned<TFromD<D>>>::type;
  return static_cast<bool>(vec_all_eq((UnsignedRawVectType)mask.raw,
                                      (UnsignedRawVectType)vec_splats(-1)));
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API bool AllFalse(D d, MFromD<D> mask) {
  const Full128<TFromD<D>> d_full;
  constexpr size_t kN = MaxLanes(d);
  return AllFalse(d_full,
    MFromD<decltype(d_full)>{vec_and(mask.raw, FirstN(d_full, kN).raw)});
}

template <class D, HWY_IF_V_SIZE_LE_D(D, 8)>
HWY_API bool AllTrue(D d, MFromD<D> mask) {
  const Full128<TFromD<D>> d_full;
  constexpr size_t kN = MaxLanes(d);
  return AllTrue(d_full,
    MFromD<decltype(d_full)>{vec_or(mask.raw, Not(FirstN(d_full, kN)).raw)});
}

template <class D>
HWY_API size_t CountTrue(D /* tag */, MFromD<D> mask) {
  return PopCount(detail::BitsFromMask(mask));
}

template <class D>
HWY_API size_t FindKnownFirstTrue(D /* tag */, MFromD<D> mask) {
  return Num0BitsBelowLS1Bit_Nonzero64(detail::BitsFromMask(mask));
}

template <class D>
HWY_API intptr_t FindFirstTrue(D /* tag */, MFromD<D> mask) {
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  return mask_bits ? intptr_t(Num0BitsBelowLS1Bit_Nonzero64(mask_bits)) : -1;
}

// ------------------------------ Compress, CompressBits

namespace detail {

// Also works for N < 8 because the first 16 4-tuples only reference bytes 0-6.
template <class D, HWY_IF_T_SIZE_D(D, 2)>
HWY_INLINE VFromD<D> IndicesFromBits128(D d, uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 256);
  const Rebind<uint8_t, decltype(d)> d8;
  const Twice<decltype(d8)> d8t;
  const RebindToUnsigned<decltype(d)> du;

  // To reduce cache footprint, store lane indices and convert to byte indices
  // (2*lane + 0..1), with the doubling baked into the table. It's not clear
  // that the additional cost of unpacking nibbles is worthwhile.
  alignas(16) static constexpr uint8_t table[2048] = {
      // PrintCompress16x8Tables
      0,  2,  4,  6,  8,  10, 12, 14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      2,  0,  4,  6,  8,  10, 12, 14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      4,  0,  2,  6,  8,  10, 12, 14, /**/ 0, 4,  2,  6,  8,  10, 12, 14,  //
      2,  4,  0,  6,  8,  10, 12, 14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      6,  0,  2,  4,  8,  10, 12, 14, /**/ 0, 6,  2,  4,  8,  10, 12, 14,  //
      2,  6,  0,  4,  8,  10, 12, 14, /**/ 0, 2,  6,  4,  8,  10, 12, 14,  //
      4,  6,  0,  2,  8,  10, 12, 14, /**/ 0, 4,  6,  2,  8,  10, 12, 14,  //
      2,  4,  6,  0,  8,  10, 12, 14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      8,  0,  2,  4,  6,  10, 12, 14, /**/ 0, 8,  2,  4,  6,  10, 12, 14,  //
      2,  8,  0,  4,  6,  10, 12, 14, /**/ 0, 2,  8,  4,  6,  10, 12, 14,  //
      4,  8,  0,  2,  6,  10, 12, 14, /**/ 0, 4,  8,  2,  6,  10, 12, 14,  //
      2,  4,  8,  0,  6,  10, 12, 14, /**/ 0, 2,  4,  8,  6,  10, 12, 14,  //
      6,  8,  0,  2,  4,  10, 12, 14, /**/ 0, 6,  8,  2,  4,  10, 12, 14,  //
      2,  6,  8,  0,  4,  10, 12, 14, /**/ 0, 2,  6,  8,  4,  10, 12, 14,  //
      4,  6,  8,  0,  2,  10, 12, 14, /**/ 0, 4,  6,  8,  2,  10, 12, 14,  //
      2,  4,  6,  8,  0,  10, 12, 14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      10, 0,  2,  4,  6,  8,  12, 14, /**/ 0, 10, 2,  4,  6,  8,  12, 14,  //
      2,  10, 0,  4,  6,  8,  12, 14, /**/ 0, 2,  10, 4,  6,  8,  12, 14,  //
      4,  10, 0,  2,  6,  8,  12, 14, /**/ 0, 4,  10, 2,  6,  8,  12, 14,  //
      2,  4,  10, 0,  6,  8,  12, 14, /**/ 0, 2,  4,  10, 6,  8,  12, 14,  //
      6,  10, 0,  2,  4,  8,  12, 14, /**/ 0, 6,  10, 2,  4,  8,  12, 14,  //
      2,  6,  10, 0,  4,  8,  12, 14, /**/ 0, 2,  6,  10, 4,  8,  12, 14,  //
      4,  6,  10, 0,  2,  8,  12, 14, /**/ 0, 4,  6,  10, 2,  8,  12, 14,  //
      2,  4,  6,  10, 0,  8,  12, 14, /**/ 0, 2,  4,  6,  10, 8,  12, 14,  //
      8,  10, 0,  2,  4,  6,  12, 14, /**/ 0, 8,  10, 2,  4,  6,  12, 14,  //
      2,  8,  10, 0,  4,  6,  12, 14, /**/ 0, 2,  8,  10, 4,  6,  12, 14,  //
      4,  8,  10, 0,  2,  6,  12, 14, /**/ 0, 4,  8,  10, 2,  6,  12, 14,  //
      2,  4,  8,  10, 0,  6,  12, 14, /**/ 0, 2,  4,  8,  10, 6,  12, 14,  //
      6,  8,  10, 0,  2,  4,  12, 14, /**/ 0, 6,  8,  10, 2,  4,  12, 14,  //
      2,  6,  8,  10, 0,  4,  12, 14, /**/ 0, 2,  6,  8,  10, 4,  12, 14,  //
      4,  6,  8,  10, 0,  2,  12, 14, /**/ 0, 4,  6,  8,  10, 2,  12, 14,  //
      2,  4,  6,  8,  10, 0,  12, 14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      12, 0,  2,  4,  6,  8,  10, 14, /**/ 0, 12, 2,  4,  6,  8,  10, 14,  //
      2,  12, 0,  4,  6,  8,  10, 14, /**/ 0, 2,  12, 4,  6,  8,  10, 14,  //
      4,  12, 0,  2,  6,  8,  10, 14, /**/ 0, 4,  12, 2,  6,  8,  10, 14,  //
      2,  4,  12, 0,  6,  8,  10, 14, /**/ 0, 2,  4,  12, 6,  8,  10, 14,  //
      6,  12, 0,  2,  4,  8,  10, 14, /**/ 0, 6,  12, 2,  4,  8,  10, 14,  //
      2,  6,  12, 0,  4,  8,  10, 14, /**/ 0, 2,  6,  12, 4,  8,  10, 14,  //
      4,  6,  12, 0,  2,  8,  10, 14, /**/ 0, 4,  6,  12, 2,  8,  10, 14,  //
      2,  4,  6,  12, 0,  8,  10, 14, /**/ 0, 2,  4,  6,  12, 8,  10, 14,  //
      8,  12, 0,  2,  4,  6,  10, 14, /**/ 0, 8,  12, 2,  4,  6,  10, 14,  //
      2,  8,  12, 0,  4,  6,  10, 14, /**/ 0, 2,  8,  12, 4,  6,  10, 14,  //
      4,  8,  12, 0,  2,  6,  10, 14, /**/ 0, 4,  8,  12, 2,  6,  10, 14,  //
      2,  4,  8,  12, 0,  6,  10, 14, /**/ 0, 2,  4,  8,  12, 6,  10, 14,  //
      6,  8,  12, 0,  2,  4,  10, 14, /**/ 0, 6,  8,  12, 2,  4,  10, 14,  //
      2,  6,  8,  12, 0,  4,  10, 14, /**/ 0, 2,  6,  8,  12, 4,  10, 14,  //
      4,  6,  8,  12, 0,  2,  10, 14, /**/ 0, 4,  6,  8,  12, 2,  10, 14,  //
      2,  4,  6,  8,  12, 0,  10, 14, /**/ 0, 2,  4,  6,  8,  12, 10, 14,  //
      10, 12, 0,  2,  4,  6,  8,  14, /**/ 0, 10, 12, 2,  4,  6,  8,  14,  //
      2,  10, 12, 0,  4,  6,  8,  14, /**/ 0, 2,  10, 12, 4,  6,  8,  14,  //
      4,  10, 12, 0,  2,  6,  8,  14, /**/ 0, 4,  10, 12, 2,  6,  8,  14,  //
      2,  4,  10, 12, 0,  6,  8,  14, /**/ 0, 2,  4,  10, 12, 6,  8,  14,  //
      6,  10, 12, 0,  2,  4,  8,  14, /**/ 0, 6,  10, 12, 2,  4,  8,  14,  //
      2,  6,  10, 12, 0,  4,  8,  14, /**/ 0, 2,  6,  10, 12, 4,  8,  14,  //
      4,  6,  10, 12, 0,  2,  8,  14, /**/ 0, 4,  6,  10, 12, 2,  8,  14,  //
      2,  4,  6,  10, 12, 0,  8,  14, /**/ 0, 2,  4,  6,  10, 12, 8,  14,  //
      8,  10, 12, 0,  2,  4,  6,  14, /**/ 0, 8,  10, 12, 2,  4,  6,  14,  //
      2,  8,  10, 12, 0,  4,  6,  14, /**/ 0, 2,  8,  10, 12, 4,  6,  14,  //
      4,  8,  10, 12, 0,  2,  6,  14, /**/ 0, 4,  8,  10, 12, 2,  6,  14,  //
      2,  4,  8,  10, 12, 0,  6,  14, /**/ 0, 2,  4,  8,  10, 12, 6,  14,  //
      6,  8,  10, 12, 0,  2,  4,  14, /**/ 0, 6,  8,  10, 12, 2,  4,  14,  //
      2,  6,  8,  10, 12, 0,  4,  14, /**/ 0, 2,  6,  8,  10, 12, 4,  14,  //
      4,  6,  8,  10, 12, 0,  2,  14, /**/ 0, 4,  6,  8,  10, 12, 2,  14,  //
      2,  4,  6,  8,  10, 12, 0,  14, /**/ 0, 2,  4,  6,  8,  10, 12, 14,  //
      14, 0,  2,  4,  6,  8,  10, 12, /**/ 0, 14, 2,  4,  6,  8,  10, 12,  //
      2,  14, 0,  4,  6,  8,  10, 12, /**/ 0, 2,  14, 4,  6,  8,  10, 12,  //
      4,  14, 0,  2,  6,  8,  10, 12, /**/ 0, 4,  14, 2,  6,  8,  10, 12,  //
      2,  4,  14, 0,  6,  8,  10, 12, /**/ 0, 2,  4,  14, 6,  8,  10, 12,  //
      6,  14, 0,  2,  4,  8,  10, 12, /**/ 0, 6,  14, 2,  4,  8,  10, 12,  //
      2,  6,  14, 0,  4,  8,  10, 12, /**/ 0, 2,  6,  14, 4,  8,  10, 12,  //
      4,  6,  14, 0,  2,  8,  10, 12, /**/ 0, 4,  6,  14, 2,  8,  10, 12,  //
      2,  4,  6,  14, 0,  8,  10, 12, /**/ 0, 2,  4,  6,  14, 8,  10, 12,  //
      8,  14, 0,  2,  4,  6,  10, 12, /**/ 0, 8,  14, 2,  4,  6,  10, 12,  //
      2,  8,  14, 0,  4,  6,  10, 12, /**/ 0, 2,  8,  14, 4,  6,  10, 12,  //
      4,  8,  14, 0,  2,  6,  10, 12, /**/ 0, 4,  8,  14, 2,  6,  10, 12,  //
      2,  4,  8,  14, 0,  6,  10, 12, /**/ 0, 2,  4,  8,  14, 6,  10, 12,  //
      6,  8,  14, 0,  2,  4,  10, 12, /**/ 0, 6,  8,  14, 2,  4,  10, 12,  //
      2,  6,  8,  14, 0,  4,  10, 12, /**/ 0, 2,  6,  8,  14, 4,  10, 12,  //
      4,  6,  8,  14, 0,  2,  10, 12, /**/ 0, 4,  6,  8,  14, 2,  10, 12,  //
      2,  4,  6,  8,  14, 0,  10, 12, /**/ 0, 2,  4,  6,  8,  14, 10, 12,  //
      10, 14, 0,  2,  4,  6,  8,  12, /**/ 0, 10, 14, 2,  4,  6,  8,  12,  //
      2,  10, 14, 0,  4,  6,  8,  12, /**/ 0, 2,  10, 14, 4,  6,  8,  12,  //
      4,  10, 14, 0,  2,  6,  8,  12, /**/ 0, 4,  10, 14, 2,  6,  8,  12,  //
      2,  4,  10, 14, 0,  6,  8,  12, /**/ 0, 2,  4,  10, 14, 6,  8,  12,  //
      6,  10, 14, 0,  2,  4,  8,  12, /**/ 0, 6,  10, 14, 2,  4,  8,  12,  //
      2,  6,  10, 14, 0,  4,  8,  12, /**/ 0, 2,  6,  10, 14, 4,  8,  12,  //
      4,  6,  10, 14, 0,  2,  8,  12, /**/ 0, 4,  6,  10, 14, 2,  8,  12,  //
      2,  4,  6,  10, 14, 0,  8,  12, /**/ 0, 2,  4,  6,  10, 14, 8,  12,  //
      8,  10, 14, 0,  2,  4,  6,  12, /**/ 0, 8,  10, 14, 2,  4,  6,  12,  //
      2,  8,  10, 14, 0,  4,  6,  12, /**/ 0, 2,  8,  10, 14, 4,  6,  12,  //
      4,  8,  10, 14, 0,  2,  6,  12, /**/ 0, 4,  8,  10, 14, 2,  6,  12,  //
      2,  4,  8,  10, 14, 0,  6,  12, /**/ 0, 2,  4,  8,  10, 14, 6,  12,  //
      6,  8,  10, 14, 0,  2,  4,  12, /**/ 0, 6,  8,  10, 14, 2,  4,  12,  //
      2,  6,  8,  10, 14, 0,  4,  12, /**/ 0, 2,  6,  8,  10, 14, 4,  12,  //
      4,  6,  8,  10, 14, 0,  2,  12, /**/ 0, 4,  6,  8,  10, 14, 2,  12,  //
      2,  4,  6,  8,  10, 14, 0,  12, /**/ 0, 2,  4,  6,  8,  10, 14, 12,  //
      12, 14, 0,  2,  4,  6,  8,  10, /**/ 0, 12, 14, 2,  4,  6,  8,  10,  //
      2,  12, 14, 0,  4,  6,  8,  10, /**/ 0, 2,  12, 14, 4,  6,  8,  10,  //
      4,  12, 14, 0,  2,  6,  8,  10, /**/ 0, 4,  12, 14, 2,  6,  8,  10,  //
      2,  4,  12, 14, 0,  6,  8,  10, /**/ 0, 2,  4,  12, 14, 6,  8,  10,  //
      6,  12, 14, 0,  2,  4,  8,  10, /**/ 0, 6,  12, 14, 2,  4,  8,  10,  //
      2,  6,  12, 14, 0,  4,  8,  10, /**/ 0, 2,  6,  12, 14, 4,  8,  10,  //
      4,  6,  12, 14, 0,  2,  8,  10, /**/ 0, 4,  6,  12, 14, 2,  8,  10,  //
      2,  4,  6,  12, 14, 0,  8,  10, /**/ 0, 2,  4,  6,  12, 14, 8,  10,  //
      8,  12, 14, 0,  2,  4,  6,  10, /**/ 0, 8,  12, 14, 2,  4,  6,  10,  //
      2,  8,  12, 14, 0,  4,  6,  10, /**/ 0, 2,  8,  12, 14, 4,  6,  10,  //
      4,  8,  12, 14, 0,  2,  6,  10, /**/ 0, 4,  8,  12, 14, 2,  6,  10,  //
      2,  4,  8,  12, 14, 0,  6,  10, /**/ 0, 2,  4,  8,  12, 14, 6,  10,  //
      6,  8,  12, 14, 0,  2,  4,  10, /**/ 0, 6,  8,  12, 14, 2,  4,  10,  //
      2,  6,  8,  12, 14, 0,  4,  10, /**/ 0, 2,  6,  8,  12, 14, 4,  10,  //
      4,  6,  8,  12, 14, 0,  2,  10, /**/ 0, 4,  6,  8,  12, 14, 2,  10,  //
      2,  4,  6,  8,  12, 14, 0,  10, /**/ 0, 2,  4,  6,  8,  12, 14, 10,  //
      10, 12, 14, 0,  2,  4,  6,  8,  /**/ 0, 10, 12, 14, 2,  4,  6,  8,   //
      2,  10, 12, 14, 0,  4,  6,  8,  /**/ 0, 2,  10, 12, 14, 4,  6,  8,   //
      4,  10, 12, 14, 0,  2,  6,  8,  /**/ 0, 4,  10, 12, 14, 2,  6,  8,   //
      2,  4,  10, 12, 14, 0,  6,  8,  /**/ 0, 2,  4,  10, 12, 14, 6,  8,   //
      6,  10, 12, 14, 0,  2,  4,  8,  /**/ 0, 6,  10, 12, 14, 2,  4,  8,   //
      2,  6,  10, 12, 14, 0,  4,  8,  /**/ 0, 2,  6,  10, 12, 14, 4,  8,   //
      4,  6,  10, 12, 14, 0,  2,  8,  /**/ 0, 4,  6,  10, 12, 14, 2,  8,   //
      2,  4,  6,  10, 12, 14, 0,  8,  /**/ 0, 2,  4,  6,  10, 12, 14, 8,   //
      8,  10, 12, 14, 0,  2,  4,  6,  /**/ 0, 8,  10, 12, 14, 2,  4,  6,   //
      2,  8,  10, 12, 14, 0,  4,  6,  /**/ 0, 2,  8,  10, 12, 14, 4,  6,   //
      4,  8,  10, 12, 14, 0,  2,  6,  /**/ 0, 4,  8,  10, 12, 14, 2,  6,   //
      2,  4,  8,  10, 12, 14, 0,  6,  /**/ 0, 2,  4,  8,  10, 12, 14, 6,   //
      6,  8,  10, 12, 14, 0,  2,  4,  /**/ 0, 6,  8,  10, 12, 14, 2,  4,   //
      2,  6,  8,  10, 12, 14, 0,  4,  /**/ 0, 2,  6,  8,  10, 12, 14, 4,   //
      4,  6,  8,  10, 12, 14, 0,  2,  /**/ 0, 4,  6,  8,  10, 12, 14, 2,   //
      2,  4,  6,  8,  10, 12, 14, 0,  /**/ 0, 2,  4,  6,  8,  10, 12, 14};

  const VFromD<decltype(d8t)> byte_idx{Load(d8, table + mask_bits * 8).raw};
  const VFromD<decltype(du)> pairs = ZipLower(byte_idx, byte_idx);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr uint16_t kPairIndexIncrement = 0x0100;
#else
  constexpr uint16_t kPairIndexIncrement = 0x0001;
#endif

  return BitCast(d, pairs + Set(du, kPairIndexIncrement));
}

template <class D, HWY_IF_T_SIZE_D(D, 2)>
HWY_INLINE VFromD<D> IndicesFromNotBits128(D d, uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 256);
  const Rebind<uint8_t, decltype(d)> d8;
  const Twice<decltype(d8)> d8t;
  const RebindToUnsigned<decltype(d)> du;

  // To reduce cache footprint, store lane indices and convert to byte indices
  // (2*lane + 0..1), with the doubling baked into the table. It's not clear
  // that the additional cost of unpacking nibbles is worthwhile.
  alignas(16) static constexpr uint8_t table[2048] = {
      // PrintCompressNot16x8Tables
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  4,  6,  8,  10, 12, 14, 0,   //
      0, 4,  6,  8,  10, 12, 14, 2,  /**/ 4,  6,  8,  10, 12, 14, 0,  2,   //
      0, 2,  6,  8,  10, 12, 14, 4,  /**/ 2,  6,  8,  10, 12, 14, 0,  4,   //
      0, 6,  8,  10, 12, 14, 2,  4,  /**/ 6,  8,  10, 12, 14, 0,  2,  4,   //
      0, 2,  4,  8,  10, 12, 14, 6,  /**/ 2,  4,  8,  10, 12, 14, 0,  6,   //
      0, 4,  8,  10, 12, 14, 2,  6,  /**/ 4,  8,  10, 12, 14, 0,  2,  6,   //
      0, 2,  8,  10, 12, 14, 4,  6,  /**/ 2,  8,  10, 12, 14, 0,  4,  6,   //
      0, 8,  10, 12, 14, 2,  4,  6,  /**/ 8,  10, 12, 14, 0,  2,  4,  6,   //
      0, 2,  4,  6,  10, 12, 14, 8,  /**/ 2,  4,  6,  10, 12, 14, 0,  8,   //
      0, 4,  6,  10, 12, 14, 2,  8,  /**/ 4,  6,  10, 12, 14, 0,  2,  8,   //
      0, 2,  6,  10, 12, 14, 4,  8,  /**/ 2,  6,  10, 12, 14, 0,  4,  8,   //
      0, 6,  10, 12, 14, 2,  4,  8,  /**/ 6,  10, 12, 14, 0,  2,  4,  8,   //
      0, 2,  4,  10, 12, 14, 6,  8,  /**/ 2,  4,  10, 12, 14, 0,  6,  8,   //
      0, 4,  10, 12, 14, 2,  6,  8,  /**/ 4,  10, 12, 14, 0,  2,  6,  8,   //
      0, 2,  10, 12, 14, 4,  6,  8,  /**/ 2,  10, 12, 14, 0,  4,  6,  8,   //
      0, 10, 12, 14, 2,  4,  6,  8,  /**/ 10, 12, 14, 0,  2,  4,  6,  8,   //
      0, 2,  4,  6,  8,  12, 14, 10, /**/ 2,  4,  6,  8,  12, 14, 0,  10,  //
      0, 4,  6,  8,  12, 14, 2,  10, /**/ 4,  6,  8,  12, 14, 0,  2,  10,  //
      0, 2,  6,  8,  12, 14, 4,  10, /**/ 2,  6,  8,  12, 14, 0,  4,  10,  //
      0, 6,  8,  12, 14, 2,  4,  10, /**/ 6,  8,  12, 14, 0,  2,  4,  10,  //
      0, 2,  4,  8,  12, 14, 6,  10, /**/ 2,  4,  8,  12, 14, 0,  6,  10,  //
      0, 4,  8,  12, 14, 2,  6,  10, /**/ 4,  8,  12, 14, 0,  2,  6,  10,  //
      0, 2,  8,  12, 14, 4,  6,  10, /**/ 2,  8,  12, 14, 0,  4,  6,  10,  //
      0, 8,  12, 14, 2,  4,  6,  10, /**/ 8,  12, 14, 0,  2,  4,  6,  10,  //
      0, 2,  4,  6,  12, 14, 8,  10, /**/ 2,  4,  6,  12, 14, 0,  8,  10,  //
      0, 4,  6,  12, 14, 2,  8,  10, /**/ 4,  6,  12, 14, 0,  2,  8,  10,  //
      0, 2,  6,  12, 14, 4,  8,  10, /**/ 2,  6,  12, 14, 0,  4,  8,  10,  //
      0, 6,  12, 14, 2,  4,  8,  10, /**/ 6,  12, 14, 0,  2,  4,  8,  10,  //
      0, 2,  4,  12, 14, 6,  8,  10, /**/ 2,  4,  12, 14, 0,  6,  8,  10,  //
      0, 4,  12, 14, 2,  6,  8,  10, /**/ 4,  12, 14, 0,  2,  6,  8,  10,  //
      0, 2,  12, 14, 4,  6,  8,  10, /**/ 2,  12, 14, 0,  4,  6,  8,  10,  //
      0, 12, 14, 2,  4,  6,  8,  10, /**/ 12, 14, 0,  2,  4,  6,  8,  10,  //
      0, 2,  4,  6,  8,  10, 14, 12, /**/ 2,  4,  6,  8,  10, 14, 0,  12,  //
      0, 4,  6,  8,  10, 14, 2,  12, /**/ 4,  6,  8,  10, 14, 0,  2,  12,  //
      0, 2,  6,  8,  10, 14, 4,  12, /**/ 2,  6,  8,  10, 14, 0,  4,  12,  //
      0, 6,  8,  10, 14, 2,  4,  12, /**/ 6,  8,  10, 14, 0,  2,  4,  12,  //
      0, 2,  4,  8,  10, 14, 6,  12, /**/ 2,  4,  8,  10, 14, 0,  6,  12,  //
      0, 4,  8,  10, 14, 2,  6,  12, /**/ 4,  8,  10, 14, 0,  2,  6,  12,  //
      0, 2,  8,  10, 14, 4,  6,  12, /**/ 2,  8,  10, 14, 0,  4,  6,  12,  //
      0, 8,  10, 14, 2,  4,  6,  12, /**/ 8,  10, 14, 0,  2,  4,  6,  12,  //
      0, 2,  4,  6,  10, 14, 8,  12, /**/ 2,  4,  6,  10, 14, 0,  8,  12,  //
      0, 4,  6,  10, 14, 2,  8,  12, /**/ 4,  6,  10, 14, 0,  2,  8,  12,  //
      0, 2,  6,  10, 14, 4,  8,  12, /**/ 2,  6,  10, 14, 0,  4,  8,  12,  //
      0, 6,  10, 14, 2,  4,  8,  12, /**/ 6,  10, 14, 0,  2,  4,  8,  12,  //
      0, 2,  4,  10, 14, 6,  8,  12, /**/ 2,  4,  10, 14, 0,  6,  8,  12,  //
      0, 4,  10, 14, 2,  6,  8,  12, /**/ 4,  10, 14, 0,  2,  6,  8,  12,  //
      0, 2,  10, 14, 4,  6,  8,  12, /**/ 2,  10, 14, 0,  4,  6,  8,  12,  //
      0, 10, 14, 2,  4,  6,  8,  12, /**/ 10, 14, 0,  2,  4,  6,  8,  12,  //
      0, 2,  4,  6,  8,  14, 10, 12, /**/ 2,  4,  6,  8,  14, 0,  10, 12,  //
      0, 4,  6,  8,  14, 2,  10, 12, /**/ 4,  6,  8,  14, 0,  2,  10, 12,  //
      0, 2,  6,  8,  14, 4,  10, 12, /**/ 2,  6,  8,  14, 0,  4,  10, 12,  //
      0, 6,  8,  14, 2,  4,  10, 12, /**/ 6,  8,  14, 0,  2,  4,  10, 12,  //
      0, 2,  4,  8,  14, 6,  10, 12, /**/ 2,  4,  8,  14, 0,  6,  10, 12,  //
      0, 4,  8,  14, 2,  6,  10, 12, /**/ 4,  8,  14, 0,  2,  6,  10, 12,  //
      0, 2,  8,  14, 4,  6,  10, 12, /**/ 2,  8,  14, 0,  4,  6,  10, 12,  //
      0, 8,  14, 2,  4,  6,  10, 12, /**/ 8,  14, 0,  2,  4,  6,  10, 12,  //
      0, 2,  4,  6,  14, 8,  10, 12, /**/ 2,  4,  6,  14, 0,  8,  10, 12,  //
      0, 4,  6,  14, 2,  8,  10, 12, /**/ 4,  6,  14, 0,  2,  8,  10, 12,  //
      0, 2,  6,  14, 4,  8,  10, 12, /**/ 2,  6,  14, 0,  4,  8,  10, 12,  //
      0, 6,  14, 2,  4,  8,  10, 12, /**/ 6,  14, 0,  2,  4,  8,  10, 12,  //
      0, 2,  4,  14, 6,  8,  10, 12, /**/ 2,  4,  14, 0,  6,  8,  10, 12,  //
      0, 4,  14, 2,  6,  8,  10, 12, /**/ 4,  14, 0,  2,  6,  8,  10, 12,  //
      0, 2,  14, 4,  6,  8,  10, 12, /**/ 2,  14, 0,  4,  6,  8,  10, 12,  //
      0, 14, 2,  4,  6,  8,  10, 12, /**/ 14, 0,  2,  4,  6,  8,  10, 12,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  4,  6,  8,  10, 12, 0,  14,  //
      0, 4,  6,  8,  10, 12, 2,  14, /**/ 4,  6,  8,  10, 12, 0,  2,  14,  //
      0, 2,  6,  8,  10, 12, 4,  14, /**/ 2,  6,  8,  10, 12, 0,  4,  14,  //
      0, 6,  8,  10, 12, 2,  4,  14, /**/ 6,  8,  10, 12, 0,  2,  4,  14,  //
      0, 2,  4,  8,  10, 12, 6,  14, /**/ 2,  4,  8,  10, 12, 0,  6,  14,  //
      0, 4,  8,  10, 12, 2,  6,  14, /**/ 4,  8,  10, 12, 0,  2,  6,  14,  //
      0, 2,  8,  10, 12, 4,  6,  14, /**/ 2,  8,  10, 12, 0,  4,  6,  14,  //
      0, 8,  10, 12, 2,  4,  6,  14, /**/ 8,  10, 12, 0,  2,  4,  6,  14,  //
      0, 2,  4,  6,  10, 12, 8,  14, /**/ 2,  4,  6,  10, 12, 0,  8,  14,  //
      0, 4,  6,  10, 12, 2,  8,  14, /**/ 4,  6,  10, 12, 0,  2,  8,  14,  //
      0, 2,  6,  10, 12, 4,  8,  14, /**/ 2,  6,  10, 12, 0,  4,  8,  14,  //
      0, 6,  10, 12, 2,  4,  8,  14, /**/ 6,  10, 12, 0,  2,  4,  8,  14,  //
      0, 2,  4,  10, 12, 6,  8,  14, /**/ 2,  4,  10, 12, 0,  6,  8,  14,  //
      0, 4,  10, 12, 2,  6,  8,  14, /**/ 4,  10, 12, 0,  2,  6,  8,  14,  //
      0, 2,  10, 12, 4,  6,  8,  14, /**/ 2,  10, 12, 0,  4,  6,  8,  14,  //
      0, 10, 12, 2,  4,  6,  8,  14, /**/ 10, 12, 0,  2,  4,  6,  8,  14,  //
      0, 2,  4,  6,  8,  12, 10, 14, /**/ 2,  4,  6,  8,  12, 0,  10, 14,  //
      0, 4,  6,  8,  12, 2,  10, 14, /**/ 4,  6,  8,  12, 0,  2,  10, 14,  //
      0, 2,  6,  8,  12, 4,  10, 14, /**/ 2,  6,  8,  12, 0,  4,  10, 14,  //
      0, 6,  8,  12, 2,  4,  10, 14, /**/ 6,  8,  12, 0,  2,  4,  10, 14,  //
      0, 2,  4,  8,  12, 6,  10, 14, /**/ 2,  4,  8,  12, 0,  6,  10, 14,  //
      0, 4,  8,  12, 2,  6,  10, 14, /**/ 4,  8,  12, 0,  2,  6,  10, 14,  //
      0, 2,  8,  12, 4,  6,  10, 14, /**/ 2,  8,  12, 0,  4,  6,  10, 14,  //
      0, 8,  12, 2,  4,  6,  10, 14, /**/ 8,  12, 0,  2,  4,  6,  10, 14,  //
      0, 2,  4,  6,  12, 8,  10, 14, /**/ 2,  4,  6,  12, 0,  8,  10, 14,  //
      0, 4,  6,  12, 2,  8,  10, 14, /**/ 4,  6,  12, 0,  2,  8,  10, 14,  //
      0, 2,  6,  12, 4,  8,  10, 14, /**/ 2,  6,  12, 0,  4,  8,  10, 14,  //
      0, 6,  12, 2,  4,  8,  10, 14, /**/ 6,  12, 0,  2,  4,  8,  10, 14,  //
      0, 2,  4,  12, 6,  8,  10, 14, /**/ 2,  4,  12, 0,  6,  8,  10, 14,  //
      0, 4,  12, 2,  6,  8,  10, 14, /**/ 4,  12, 0,  2,  6,  8,  10, 14,  //
      0, 2,  12, 4,  6,  8,  10, 14, /**/ 2,  12, 0,  4,  6,  8,  10, 14,  //
      0, 12, 2,  4,  6,  8,  10, 14, /**/ 12, 0,  2,  4,  6,  8,  10, 14,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  4,  6,  8,  10, 0,  12, 14,  //
      0, 4,  6,  8,  10, 2,  12, 14, /**/ 4,  6,  8,  10, 0,  2,  12, 14,  //
      0, 2,  6,  8,  10, 4,  12, 14, /**/ 2,  6,  8,  10, 0,  4,  12, 14,  //
      0, 6,  8,  10, 2,  4,  12, 14, /**/ 6,  8,  10, 0,  2,  4,  12, 14,  //
      0, 2,  4,  8,  10, 6,  12, 14, /**/ 2,  4,  8,  10, 0,  6,  12, 14,  //
      0, 4,  8,  10, 2,  6,  12, 14, /**/ 4,  8,  10, 0,  2,  6,  12, 14,  //
      0, 2,  8,  10, 4,  6,  12, 14, /**/ 2,  8,  10, 0,  4,  6,  12, 14,  //
      0, 8,  10, 2,  4,  6,  12, 14, /**/ 8,  10, 0,  2,  4,  6,  12, 14,  //
      0, 2,  4,  6,  10, 8,  12, 14, /**/ 2,  4,  6,  10, 0,  8,  12, 14,  //
      0, 4,  6,  10, 2,  8,  12, 14, /**/ 4,  6,  10, 0,  2,  8,  12, 14,  //
      0, 2,  6,  10, 4,  8,  12, 14, /**/ 2,  6,  10, 0,  4,  8,  12, 14,  //
      0, 6,  10, 2,  4,  8,  12, 14, /**/ 6,  10, 0,  2,  4,  8,  12, 14,  //
      0, 2,  4,  10, 6,  8,  12, 14, /**/ 2,  4,  10, 0,  6,  8,  12, 14,  //
      0, 4,  10, 2,  6,  8,  12, 14, /**/ 4,  10, 0,  2,  6,  8,  12, 14,  //
      0, 2,  10, 4,  6,  8,  12, 14, /**/ 2,  10, 0,  4,  6,  8,  12, 14,  //
      0, 10, 2,  4,  6,  8,  12, 14, /**/ 10, 0,  2,  4,  6,  8,  12, 14,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  4,  6,  8,  0,  10, 12, 14,  //
      0, 4,  6,  8,  2,  10, 12, 14, /**/ 4,  6,  8,  0,  2,  10, 12, 14,  //
      0, 2,  6,  8,  4,  10, 12, 14, /**/ 2,  6,  8,  0,  4,  10, 12, 14,  //
      0, 6,  8,  2,  4,  10, 12, 14, /**/ 6,  8,  0,  2,  4,  10, 12, 14,  //
      0, 2,  4,  8,  6,  10, 12, 14, /**/ 2,  4,  8,  0,  6,  10, 12, 14,  //
      0, 4,  8,  2,  6,  10, 12, 14, /**/ 4,  8,  0,  2,  6,  10, 12, 14,  //
      0, 2,  8,  4,  6,  10, 12, 14, /**/ 2,  8,  0,  4,  6,  10, 12, 14,  //
      0, 8,  2,  4,  6,  10, 12, 14, /**/ 8,  0,  2,  4,  6,  10, 12, 14,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  4,  6,  0,  8,  10, 12, 14,  //
      0, 4,  6,  2,  8,  10, 12, 14, /**/ 4,  6,  0,  2,  8,  10, 12, 14,  //
      0, 2,  6,  4,  8,  10, 12, 14, /**/ 2,  6,  0,  4,  8,  10, 12, 14,  //
      0, 6,  2,  4,  8,  10, 12, 14, /**/ 6,  0,  2,  4,  8,  10, 12, 14,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  4,  0,  6,  8,  10, 12, 14,  //
      0, 4,  2,  6,  8,  10, 12, 14, /**/ 4,  0,  2,  6,  8,  10, 12, 14,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 2,  0,  4,  6,  8,  10, 12, 14,  //
      0, 2,  4,  6,  8,  10, 12, 14, /**/ 0,  2,  4,  6,  8,  10, 12, 14};

  const VFromD<decltype(d8t)> byte_idx{Load(d8, table + mask_bits * 8).raw};
  const VFromD<decltype(du)> pairs = ZipLower(byte_idx, byte_idx);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr uint16_t kPairIndexIncrement = 0x0100;
#else
  constexpr uint16_t kPairIndexIncrement = 0x0001;
#endif

  return BitCast(d, pairs + Set(du, kPairIndexIncrement));
}

template <class D, HWY_IF_T_SIZE_D(D, 4)>
HWY_INLINE VFromD<D> IndicesFromBits128(D d, uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 16);

  // There are only 4 lanes, so we can afford to load the index vector directly.
  alignas(16) static constexpr uint8_t u8_indices[256] = {
      // PrintCompress32x4Tables
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  //
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  //
      4,  5,  6,  7,  0,  1,  2,  3,  8,  9,  10, 11, 12, 13, 14, 15,  //
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  //
      8,  9,  10, 11, 0,  1,  2,  3,  4,  5,  6,  7,  12, 13, 14, 15,  //
      0,  1,  2,  3,  8,  9,  10, 11, 4,  5,  6,  7,  12, 13, 14, 15,  //
      4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,  12, 13, 14, 15,  //
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,  //
      12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,  //
      0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,  8,  9,  10, 11,  //
      4,  5,  6,  7,  12, 13, 14, 15, 0,  1,  2,  3,  8,  9,  10, 11,  //
      0,  1,  2,  3,  4,  5,  6,  7,  12, 13, 14, 15, 8,  9,  10, 11,  //
      8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,   //
      0,  1,  2,  3,  8,  9,  10, 11, 12, 13, 14, 15, 4,  5,  6,  7,   //
      4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,   //
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15};

  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, u8_indices + 16 * mask_bits));
}

template <class D, HWY_IF_T_SIZE_D(D, 4)>
HWY_INLINE VFromD<D> IndicesFromNotBits128(D d, uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 16);

  // There are only 4 lanes, so we can afford to load the index vector directly.
  alignas(16) static constexpr uint8_t u8_indices[256] = {
      // PrintCompressNot32x4Tables
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 4,  5,
      6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  0,  1,  2,  3,
      8,  9,  10, 11, 12, 13, 14, 15, 4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
      12, 13, 14, 15, 8,  9,  10, 11, 4,  5,  6,  7,  12, 13, 14, 15, 0,  1,
      2,  3,  8,  9,  10, 11, 0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,
      8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
      10, 11, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,  12, 13, 14, 15, 0,  1,
      2,  3,  8,  9,  10, 11, 4,  5,  6,  7,  12, 13, 14, 15, 8,  9,  10, 11,
      0,  1,  2,  3,  4,  5,  6,  7,  12, 13, 14, 15, 0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 4,  5,  6,  7,  0,  1,  2,  3,
      8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
      10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 14, 15};

  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, u8_indices + 16 * mask_bits));
}

template <class D, HWY_IF_T_SIZE_D(D, 8)>
HWY_INLINE VFromD<D> IndicesFromBits128(D d, uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 4);

  // There are only 2 lanes, so we can afford to load the index vector directly.
  alignas(16) static constexpr uint8_t u8_indices[64] = {
      // PrintCompress64x2Tables
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2,  3,  4,  5,  6,  7,
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15};

  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, u8_indices + 16 * mask_bits));
}

template <class D, HWY_IF_T_SIZE_D(D, 8)>
HWY_INLINE VFromD<D> IndicesFromNotBits128(D d, uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 4);

  // There are only 2 lanes, so we can afford to load the index vector directly.
  alignas(16) static constexpr uint8_t u8_indices[64] = {
      // PrintCompressNot64x2Tables
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2,  3,  4,  5,  6,  7,
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15};

  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, u8_indices + 16 * mask_bits));
}

template <typename T, size_t N, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API Vec128<T, N> CompressBits(Vec128<T, N> v, uint64_t mask_bits) {
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du;

  HWY_DASSERT(mask_bits < (1ull << N));
  const auto indices = BitCast(du, detail::IndicesFromBits128(d, mask_bits));
  return BitCast(d, TableLookupBytes(BitCast(du, v), indices));
}

template <typename T, size_t N, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API Vec128<T, N> CompressNotBits(Vec128<T, N> v, uint64_t mask_bits) {
  const DFromV<decltype(v)> d;
  const RebindToUnsigned<decltype(d)> du;

  HWY_DASSERT(mask_bits < (1ull << N));
  const auto indices = BitCast(du, detail::IndicesFromNotBits128(d, mask_bits));
  return BitCast(d, TableLookupBytes(BitCast(du, v), indices));
}

}  // namespace detail

// Single lane: no-op
template <typename T>
HWY_API Vec128<T, 1> Compress(Vec128<T, 1> v, Mask128<T, 1> /*m*/) {
  return v;
}

// Two lanes: conditional swap
template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec128<T> Compress(Vec128<T> v, Mask128<T> mask) {
  // If mask[1] = 1 and mask[0] = 0, then swap both halves, else keep.
  const Full128<T> d;
  const Vec128<T> m = VecFromMask(d, mask);
  const Vec128<T> maskL = DupEven(m);
  const Vec128<T> maskH = DupOdd(m);
  const Vec128<T> swap = AndNot(maskL, maskH);
  return IfVecThenElse(swap, Shuffle01(v), v);
}

// General case, 2 or 4 bytes
template <typename T, size_t N, HWY_IF_T_SIZE_ONE_OF(T, (1 << 2) | (1 << 4))>
HWY_API Vec128<T, N> Compress(Vec128<T, N> v, Mask128<T, N> mask) {
  return detail::CompressBits(v, detail::BitsFromMask(mask));
}

// ------------------------------ CompressNot

// Single lane: no-op
template <typename T>
HWY_API Vec128<T, 1> CompressNot(Vec128<T, 1> v, Mask128<T, 1> /*m*/) {
  return v;
}

// Two lanes: conditional swap
template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_API Vec128<T> CompressNot(Vec128<T> v, Mask128<T> mask) {
  // If mask[1] = 0 and mask[0] = 1, then swap both halves, else keep.
  const Full128<T> d;
  const Vec128<T> m = VecFromMask(d, mask);
  const Vec128<T> maskL = DupEven(m);
  const Vec128<T> maskH = DupOdd(m);
  const Vec128<T> swap = AndNot(maskH, maskL);
  return IfVecThenElse(swap, Shuffle01(v), v);
}

// General case, 2 or 4 bytes
template <typename T, size_t N, HWY_IF_T_SIZE_ONE_OF(T, (1 << 2) | (1 << 4))>
HWY_API Vec128<T, N> CompressNot(Vec128<T, N> v, Mask128<T, N> mask) {
  // For partial vectors, we cannot pull the Not() into the table because
  // BitsFromMask clears the upper bits.
  if (N < 16 / sizeof(T)) {
    return detail::CompressBits(v, detail::BitsFromMask(Not(mask)));
  }
  return detail::CompressNotBits(v, detail::BitsFromMask(mask));
}

// ------------------------------ CompressBlocksNot
HWY_API Vec128<uint64_t> CompressBlocksNot(Vec128<uint64_t> v,
                                           Mask128<uint64_t> /* m */) {
  return v;
}

template <typename T, size_t N, HWY_IF_NOT_T_SIZE(T, 1)>
HWY_API Vec128<T, N> CompressBits(Vec128<T, N> v,
                                  const uint8_t* HWY_RESTRICT bits) {
  uint64_t mask_bits = 0;
  constexpr size_t kNumBytes = (N + 7) / 8;
  CopyBytes<kNumBytes>(bits, &mask_bits);
  if (N < 8) {
    mask_bits &= (1ull << N) - 1;
  }

  return detail::CompressBits(v, mask_bits);
}

// ------------------------------ CompressStore, CompressBitsStore

template <class D, HWY_IF_NOT_T_SIZE_D(D, 1)>
HWY_API size_t CompressStore(VFromD<D> v, MFromD<D> m, D d,
                             TFromD<D>* HWY_RESTRICT unaligned) {
  const RebindToUnsigned<decltype(d)> du;

  const uint64_t mask_bits = detail::BitsFromMask(m);
  HWY_DASSERT(mask_bits < (1ull << MaxLanes(d)));
  const size_t count = PopCount(mask_bits);

  const auto indices = BitCast(du, detail::IndicesFromBits128(d, mask_bits));
  const auto compressed = BitCast(d, TableLookupBytes(BitCast(du, v), indices));
  StoreU(compressed, d, unaligned);
  return count;
}

template <class D, HWY_IF_NOT_T_SIZE_D(D, 1)>
HWY_API size_t CompressBlendedStore(VFromD<D> v, MFromD<D> m, D d,
                                    TFromD<D>* HWY_RESTRICT unaligned) {
  const RebindToUnsigned<decltype(d)> du;

  const uint64_t mask_bits = detail::BitsFromMask(m);
  HWY_DASSERT(mask_bits < (1ull << MaxLanes(d)));
  const size_t count = PopCount(mask_bits);

  const auto indices = BitCast(du, detail::IndicesFromBits128(d, mask_bits));
  const auto compressed = BitCast(d, TableLookupBytes(BitCast(du, v), indices));
  BlendedStore(compressed, FirstN(d, count), d, unaligned);
  return count;
}

template <class D, HWY_IF_NOT_T_SIZE_D(D, 1)>
HWY_API size_t CompressBitsStore(VFromD<D> v, const uint8_t* HWY_RESTRICT bits,
                                 D d, TFromD<D>* HWY_RESTRICT unaligned) {
  const RebindToUnsigned<decltype(d)> du;

  uint64_t mask_bits = 0;
  constexpr size_t kN = MaxLanes(d);
  constexpr size_t kNumBytes = (kN + 7) / 8;
  CopyBytes<kNumBytes>(bits, &mask_bits);
  if (kN < 8) {
    mask_bits &= (1ull << kN) - 1;
  }
  const size_t count = PopCount(mask_bits);

  const auto indices = BitCast(du, detail::IndicesFromBits128(d, mask_bits));
  const auto compressed = BitCast(d, TableLookupBytes(BitCast(du, v), indices));
  StoreU(compressed, d, unaligned);

  return count;
}

// ------------------------------ StoreInterleaved2/3/4

// HWY_NATIVE_LOAD_STORE_INTERLEAVED not set, hence defined in
// generic_ops-inl.h.

// ------------------------------ Reductions

namespace detail {

// N=1 for any T: no-op
template <typename T>
HWY_INLINE Vec128<T, 1> SumOfLanes(Vec128<T, 1> v) {
  return v;
}
template <typename T>
HWY_INLINE Vec128<T, 1> MinOfLanes(Vec128<T, 1> v) {
  return v;
}
template <typename T>
HWY_INLINE Vec128<T, 1> MaxOfLanes(Vec128<T, 1> v) {
  return v;
}

// u32/i32/f32:

// N=2
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T, 2> SumOfLanes(Vec128<T, 2> v10) {
  // NOTE: AltivecVsum2sws cannot be used here as AltivecVsum2sws
  // computes the signed saturated sum of the lanes.
  return v10 + Shuffle2301(v10);
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T, 2> MinOfLanes(Vec128<T, 2> v10) {
  return Min(v10, Shuffle2301(v10));
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T, 2> MaxOfLanes(Vec128<T, 2> v10) {
  return Max(v10, Shuffle2301(v10));
}

// N=4 (full)
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T> SumOfLanes(Vec128<T> v3210) {
  // NOTE: AltivecVsumsws cannot be used here as AltivecVsumsws
  // computes the signed saturated sum of the lanes.
  const Vec128<T> v1032 = Shuffle1032(v3210);
  const Vec128<T> v31_20_31_20 = v3210 + v1032;
  const Vec128<T> v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return v20_31_20_31 + v31_20_31_20;
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T> MinOfLanes(Vec128<T> v3210) {
  const Vec128<T> v1032 = Shuffle1032(v3210);
  const Vec128<T> v31_20_31_20 = Min(v3210, v1032);
  const Vec128<T> v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return Min(v20_31_20_31, v31_20_31_20);
}
template <typename T, HWY_IF_T_SIZE(T, 4)>
HWY_INLINE Vec128<T> MaxOfLanes(Vec128<T> v3210) {
  const Vec128<T> v1032 = Shuffle1032(v3210);
  const Vec128<T> v31_20_31_20 = Max(v3210, v1032);
  const Vec128<T> v20_31_20_31 = Shuffle0321(v31_20_31_20);
  return Max(v20_31_20_31, v31_20_31_20);
}

// u64/i64/f64:

// N=2 (full)
template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Vec128<T> SumOfLanes(Vec128<T> v10) {
  const Vec128<T> v01 = Shuffle01(v10);
  return v10 + v01;
}
template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Vec128<T> MinOfLanes(Vec128<T> v10) {
  const Vec128<T> v01 = Shuffle01(v10);
  return Min(v10, v01);
}
template <typename T, HWY_IF_T_SIZE(T, 8)>
HWY_INLINE Vec128<T> MaxOfLanes(Vec128<T> v10) {
  const Vec128<T> v01 = Shuffle01(v10);
  return Max(v10, v01);
}

inline __attribute__((__always_inline__))
__vector signed int AltivecVsum4shs(__vector signed short a,
                                    __vector signed int b) {
#ifdef __OPTIMIZE__
  if(__builtin_constant_p(a[0]) && __builtin_constant_p(a[1]) &&
     __builtin_constant_p(a[2]) && __builtin_constant_p(a[3]) &&
     __builtin_constant_p(a[4]) && __builtin_constant_p(a[5]) &&
     __builtin_constant_p(a[6]) && __builtin_constant_p(a[7]) &&
     __builtin_constant_p(b[0]) && __builtin_constant_p(b[1]) &&
     __builtin_constant_p(b[2]) && __builtin_constant_p(b[3])) {
    const int64_t sum0 = static_cast<int64_t>(a[0]) +
                         static_cast<int64_t>(a[1]) +
                         static_cast<int64_t>(b[0]);
    const int64_t sum1 = static_cast<int64_t>(a[2]) +
                         static_cast<int64_t>(a[3]) +
                         static_cast<int64_t>(b[1]);
    const int64_t sum2 = static_cast<int64_t>(a[4]) +
                         static_cast<int64_t>(a[5]) +
                         static_cast<int64_t>(b[2]);
    const int64_t sum3 = static_cast<int64_t>(a[6]) +
                         static_cast<int64_t>(a[7]) +
                         static_cast<int64_t>(b[3]);
    const int32_t sign0 = static_cast<int32_t>(sum0 >> 63);
    const int32_t sign1 = static_cast<int32_t>(sum1 >> 63);
    const int32_t sign2 = static_cast<int32_t>(sum2 >> 63);
    const int32_t sign3 = static_cast<int32_t>(sum3 >> 63);
    using LoadI32VectType =
      typename detail::Raw128<int32_t>::AlignedLoadStoreRawVectType;
    return (__vector signed int)((LoadI32VectType){
      (sign0 == (sum0 >> 31)) ? static_cast<int32_t>(sum0) :
                                static_cast<int32_t>(sign0 ^ 0x7FFFFFFF),
      (sign1 == (sum1 >> 31)) ? static_cast<int32_t>(sum1) :
                                static_cast<int32_t>(sign1 ^ 0x7FFFFFFF),
      (sign2 == (sum2 >> 31)) ? static_cast<int32_t>(sum2) :
                                static_cast<int32_t>(sign2 ^ 0x7FFFFFFF),
      (sign3 == (sum3 >> 31)) ? static_cast<int32_t>(sum3) :
                                static_cast<int32_t>(sign3 ^ 0x7FFFFFFF)
      });
  } else
#endif
  {
    return vec_vsum4shs(a, b);
  }
}

inline __attribute__((__always_inline__))
__vector signed int AltivecVsum4sbs(__vector signed char a,
                                    __vector signed int b) {
#ifdef __OPTIMIZE__
  if(__builtin_constant_p(a[0]) && __builtin_constant_p(a[1]) &&
     __builtin_constant_p(a[2]) && __builtin_constant_p(a[3]) &&
     __builtin_constant_p(a[4]) && __builtin_constant_p(a[5]) &&
     __builtin_constant_p(a[6]) && __builtin_constant_p(a[7]) &&
     __builtin_constant_p(a[8]) && __builtin_constant_p(a[9]) &&
     __builtin_constant_p(a[10]) && __builtin_constant_p(a[11]) &&
     __builtin_constant_p(a[12]) && __builtin_constant_p(a[13]) &&
     __builtin_constant_p(a[14]) && __builtin_constant_p(a[15]) &&
     __builtin_constant_p(b[0]) && __builtin_constant_p(b[1]) &&
     __builtin_constant_p(b[2]) && __builtin_constant_p(b[3])) {
    const int64_t sum0 = static_cast<int64_t>(a[0]) +
                         static_cast<int64_t>(a[1]) +
                         static_cast<int64_t>(a[2]) +
                         static_cast<int64_t>(a[3]) +
                         static_cast<int64_t>(b[0]);
    const int64_t sum1 = static_cast<int64_t>(a[4]) +
                         static_cast<int64_t>(a[5]) +
                         static_cast<int64_t>(a[6]) +
                         static_cast<int64_t>(a[7]) +
                         static_cast<int64_t>(b[1]);
    const int64_t sum2 = static_cast<int64_t>(a[8]) +
                         static_cast<int64_t>(a[9]) +
                         static_cast<int64_t>(a[10]) +
                         static_cast<int64_t>(a[11]) +
                         static_cast<int64_t>(b[2]);
    const int64_t sum3 = static_cast<int64_t>(a[12]) +
                         static_cast<int64_t>(a[13]) +
                         static_cast<int64_t>(a[14]) +
                         static_cast<int64_t>(a[15]) +
                         static_cast<int64_t>(b[3]);
    const int32_t sign0 = static_cast<int32_t>(sum0 >> 63);
    const int32_t sign1 = static_cast<int32_t>(sum1 >> 63);
    const int32_t sign2 = static_cast<int32_t>(sum2 >> 63);
    const int32_t sign3 = static_cast<int32_t>(sum3 >> 63);
    using LoadI32VectType =
      typename detail::Raw128<int32_t>::AlignedLoadStoreRawVectType;
    return (__vector signed int)((LoadI32VectType){
      (sign0 == (sum0 >> 31)) ? static_cast<int32_t>(sum0) :
                                static_cast<int32_t>(sign0 ^ 0x7FFFFFFF),
      (sign1 == (sum1 >> 31)) ? static_cast<int32_t>(sum1) :
                                static_cast<int32_t>(sign1 ^ 0x7FFFFFFF),
      (sign2 == (sum2 >> 31)) ? static_cast<int32_t>(sum2) :
                                static_cast<int32_t>(sign2 ^ 0x7FFFFFFF),
      (sign3 == (sum3 >> 31)) ? static_cast<int32_t>(sum3) :
                                static_cast<int32_t>(sign3 ^ 0x7FFFFFFF)
      });
  } else
#endif
  {
    return vec_vsum4sbs(a, b);
  }
}

inline __attribute__((__always_inline__))
__vector signed int AltivecVsumsws(__vector signed int a,
                                   __vector signed int b) {
#ifdef __OPTIMIZE__
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kDestLaneOffset = 0;
#else
  constexpr int kDestLaneOffset = 3;
#endif
  if(__builtin_constant_p(a[0]) && __builtin_constant_p(a[1]) &&
     __builtin_constant_p(a[2]) && __builtin_constant_p(a[3]) &&
     __builtin_constant_p(b[kDestLaneOffset])) {
    const int64_t sum = static_cast<int64_t>(a[0]) +
                         static_cast<int64_t>(a[1]) +
                         static_cast<int64_t>(a[2]) +
                         static_cast<int64_t>(a[3]) +
                         static_cast<int64_t>(b[kDestLaneOffset]);
    const int32_t sign = static_cast<int32_t>(sum >> 63);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (__vector signed int){
      (sign == (sum >> 31)) ? static_cast<int32_t>(sum) :
                              static_cast<int32_t>(sign ^ 0x7FFFFFFF),
      0, 0, 0};
#else
    return (__vector signed int){0, 0, 0,
      (sign == (sum >> 31)) ? static_cast<int32_t>(sum) :
                              static_cast<int32_t>(sign ^ 0x7FFFFFFF)};
#endif
  } else
#endif
  {
    __vector signed int sum;

    // Inline assembly is used for vsumsws to avoid unnecessary shuffling
    // on little-endian PowerPC targets as the result of the vsumsws
    // instruction will already be in the correct lanes on little-endian
    // PowerPC targets.
    __asm__("vsumsws %0,%1,%2"
            : "=v" (sum)
            : "v" (a), "v" (b));

    return sum;
  }
}

inline __attribute__((__always_inline__, __artificial__))
__vector signed int AltivecU16SumsOf2(__vector unsigned short v) {
  return AltivecVsum4shs(vec_xor((__vector signed short)v,
                                 vec_splats(static_cast<short>(-32768))),
                         vec_splats(65536));
}

HWY_API Vec128<uint16_t, 2> SumOfLanes(Vec128<uint16_t, 2> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 1;
#endif
  return Vec128<uint16_t, 2>{vec_splat(
    (__vector unsigned short)AltivecU16SumsOf2(v.raw), kSumLaneIdx)};
}

HWY_API Vec128<uint16_t, 4> SumOfLanes(Vec128<uint16_t, 4> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 3;
#endif
  return Vec128<uint16_t, 4>{vec_splat((__vector unsigned short)AltivecVsum2sws(
    AltivecU16SumsOf2(v.raw), vec_splats(0)), kSumLaneIdx)};
}

HWY_API Vec128<uint16_t, 8> SumOfLanes(Vec128<uint16_t, 8> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 7;
#endif
  return Vec128<uint16_t, 8>{vec_splat((__vector unsigned short)AltivecVsumsws(
    AltivecU16SumsOf2(v.raw), vec_splats(0)), kSumLaneIdx)};
}

HWY_API Vec128<int16_t, 2> SumOfLanes(Vec128<int16_t, 2> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 1;
#endif
  return Vec128<int16_t, 2>{vec_splat(
    (__vector signed short)AltivecVsum4shs(v.raw, vec_splats(0)),
    kSumLaneIdx)};
}

HWY_API Vec128<int16_t, 4> SumOfLanes(Vec128<int16_t, 4> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 3;
#endif
  return Vec128<int16_t, 4>{vec_splat((__vector signed short)AltivecVsum2sws(
    AltivecVsum4shs(v.raw, vec_splats(0)), vec_splats(0)), kSumLaneIdx)};
}

HWY_API Vec128<int16_t, 8> SumOfLanes(Vec128<int16_t, 8> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 7;
#endif
  const __vector signed int zero = vec_splats(0);
  return Vec128<int16_t, 8>{vec_splat((__vector signed short)AltivecVsumsws(
    AltivecVsum4shs(v.raw, zero), zero), kSumLaneIdx)};
}

// u8, N=2, N=4, N=8, N=16:
HWY_API Vec128<uint8_t, 2> SumOfLanes(Vec16<uint8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 3;
#endif

  const __vector unsigned int zero = vec_splats(0u);
  return Vec128<uint8_t, 2>{vec_splat((__vector unsigned char)AltivecVsum4ubs(
                                      (__vector unsigned char)vec_mergeh(
                                        (__vector unsigned short)v.raw,
                                        (__vector unsigned short)zero),
                                      zero), kSumLaneIdx)};
}

HWY_API Vec32<uint8_t> SumOfLanes(Vec32<uint8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 3;
#endif
  return Vec32<uint8_t>{vec_splat((__vector unsigned char)AltivecVsum4ubs(
                                  v.raw, vec_splats(0u)), kSumLaneIdx)};
}

HWY_API Vec64<uint8_t> SumOfLanes(Vec64<uint8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 7;
#endif
  return Vec64<uint8_t>{vec_splat((__vector unsigned char)SumsOf8(v).raw,
                                  kSumLaneIdx)};
}

HWY_API Vec128<uint8_t> SumOfLanes(Vec128<uint8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 15;
#endif

  const __vector unsigned int zero = vec_splats(0u);
  return Vec128<uint8_t>{vec_splat(
    (__vector unsigned char)AltivecVsumsws(
      (__vector signed int)AltivecVsum4ubs(v.raw, zero),
      (__vector signed int)zero),
    kSumLaneIdx)};
}

HWY_API Vec16<int8_t> SumOfLanes(Vec16<int8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 3;
#endif

  const __vector signed int zero = vec_splats(0);
  return Vec128<int8_t, 2>{vec_splat((__vector signed char)AltivecVsum4sbs(
                                     (__vector signed char)vec_mergeh(
                                       (__vector unsigned short)v.raw,
                                       (__vector unsigned short)zero),
                                     zero), kSumLaneIdx)};
}

HWY_API Vec32<int8_t> SumOfLanes(Vec32<int8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 3;
#endif
  return Vec32<int8_t>{vec_splat((__vector signed char)AltivecVsum4sbs(
                                 v.raw, vec_splats(0)), kSumLaneIdx)};
}

HWY_API Vec64<int8_t> SumOfLanes(Vec64<int8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 7;
#endif

  const __vector signed int zero = vec_splats(0);
  return Vec64<int8_t>{vec_splat((__vector signed char)AltivecVsum2sws(
    AltivecVsum4sbs(v.raw, zero), zero), kSumLaneIdx)};
}

HWY_API Vec128<int8_t> SumOfLanes(Vec128<int8_t> v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr int kSumLaneIdx = 0;
#else
  constexpr int kSumLaneIdx = 15;
#endif

  const __vector signed int zero = vec_splats(0);
  return Vec128<int8_t>{vec_splat((__vector signed char)AltivecVsumsws(
    AltivecVsum4sbs(v.raw, zero), zero), kSumLaneIdx)};
}

template <size_t N, HWY_IF_V_SIZE_GT(uint8_t, N, 4)>
HWY_API Vec128<uint8_t, N> MaxOfLanes(Vec128<uint8_t, N> v) {
  const DFromV<decltype(v)> d;
  const RepartitionToWide<decltype(d)> d16;
  const RepartitionToWide<decltype(d16)> d32;
  Vec128<uint8_t, N> vm = Max(v, Reverse2(d, v));
  vm = Max(vm, BitCast(d, Reverse2(d16, BitCast(d16, vm))));
  vm = Max(vm, BitCast(d, Reverse2(d32, BitCast(d32, vm))));
  if (N > 8) {
    const RepartitionToWide<decltype(d32)> d64;
    vm = Max(vm, BitCast(d, Reverse2(d64, BitCast(d64, vm))));
  }
  return vm;
}

template <size_t N, HWY_IF_V_SIZE_GT(uint8_t, N, 4)>
HWY_API Vec128<uint8_t, N> MinOfLanes(Vec128<uint8_t, N> v) {
  const DFromV<decltype(v)> d;
  const RepartitionToWide<decltype(d)> d16;
  const RepartitionToWide<decltype(d16)> d32;
  Vec128<uint8_t, N> vm = Min(v, Reverse2(d, v));
  vm = Min(vm, BitCast(d, Reverse2(d16, BitCast(d16, vm))));
  vm = Min(vm, BitCast(d, Reverse2(d32, BitCast(d32, vm))));
  if (N > 8) {
    const RepartitionToWide<decltype(d32)> d64;
    vm = Min(vm, BitCast(d, Reverse2(d64, BitCast(d64, vm))));
  }
  return vm;
}

template <size_t N, HWY_IF_V_SIZE_GT(int8_t, N, 4)>
HWY_API Vec128<int8_t, N> MaxOfLanes(Vec128<int8_t, N> v) {
  const DFromV<decltype(v)> d;
  const RepartitionToWide<decltype(d)> d16;
  const RepartitionToWide<decltype(d16)> d32;
  Vec128<int8_t, N> vm = Max(v, Reverse2(d, v));
  vm = Max(vm, BitCast(d, Reverse2(d16, BitCast(d16, vm))));
  vm = Max(vm, BitCast(d, Reverse2(d32, BitCast(d32, vm))));
  if (N > 8) {
    const RepartitionToWide<decltype(d32)> d64;
    vm = Max(vm, BitCast(d, Reverse2(d64, BitCast(d64, vm))));
  }
  return vm;
}

template <size_t N, HWY_IF_V_SIZE_GT(int8_t, N, 4)>
HWY_API Vec128<int8_t, N> MinOfLanes(Vec128<int8_t, N> v) {
  const DFromV<decltype(v)> d;
  const RepartitionToWide<decltype(d)> d16;
  const RepartitionToWide<decltype(d16)> d32;
  Vec128<int8_t, N> vm = Min(v, Reverse2(d, v));
  vm = Min(vm, BitCast(d, Reverse2(d16, BitCast(d16, vm))));
  vm = Min(vm, BitCast(d, Reverse2(d32, BitCast(d32, vm))));
  if (N > 8) {
    const RepartitionToWide<decltype(d32)> d64;
    vm = Min(vm, BitCast(d, Reverse2(d64, BitCast(d64, vm))));
  }
  return vm;
}

template <size_t N, HWY_IF_V_SIZE_GT(uint16_t, N, 2)>
HWY_API Vec128<uint16_t, N> MinOfLanes(Vec128<uint16_t, N> v) {
  const Simd<uint16_t, N, 0> d;
  const RepartitionToWide<decltype(d)> d32;
  const auto even = And(BitCast(d32, v), Set(d32, 0xFFFF));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MinOfLanes(Min(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}
template <size_t N, HWY_IF_V_SIZE_GT(int16_t, N, 2)>
HWY_API Vec128<int16_t, N> MinOfLanes(Vec128<int16_t, N> v) {
  const Simd<int16_t, N, 0> d;
  const RepartitionToWide<decltype(d)> d32;
  // Sign-extend
  const auto even = ShiftRight<16>(ShiftLeft<16>(BitCast(d32, v)));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MinOfLanes(Min(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}

template <size_t N, HWY_IF_V_SIZE_GT(uint16_t, N, 2)>
HWY_API Vec128<uint16_t, N> MaxOfLanes(Vec128<uint16_t, N> v) {
  const Simd<uint16_t, N, 0> d;
  const RepartitionToWide<decltype(d)> d32;
  const auto even = And(BitCast(d32, v), Set(d32, 0xFFFF));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MaxOfLanes(Max(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}
template <size_t N, HWY_IF_V_SIZE_GT(int16_t, N, 2)>
HWY_API Vec128<int16_t, N> MaxOfLanes(Vec128<int16_t, N> v) {
  const Simd<int16_t, N, 0> d;
  const RepartitionToWide<decltype(d)> d32;
  // Sign-extend
  const auto even = ShiftRight<16>(ShiftLeft<16>(BitCast(d32, v)));
  const auto odd = ShiftRight<16>(BitCast(d32, v));
  const auto min = MaxOfLanes(Max(even, odd));
  // Also broadcast into odd lanes.
  return OddEven(BitCast(d, ShiftLeft<16>(min)), BitCast(d, min));
}

}  // namespace detail

// Supported for u/i/f 32/64. Returns the same value in each lane.
template <class D>
HWY_API VFromD<D> SumOfLanes(D /* tag */, VFromD<D> v) {
  return detail::SumOfLanes(v);
}
template <class D>
HWY_API VFromD<D> MinOfLanes(D /* tag */, VFromD<D> v) {
  return detail::MinOfLanes(v);
}
template <class D>
HWY_API VFromD<D> MaxOfLanes(D /* tag */, VFromD<D> v) {
  return detail::MaxOfLanes(v);
}

// ------------------------------ Lt128

namespace detail {

// Returns vector-mask for Lt128.
template <class D, class V = VFromD<D>>
HWY_INLINE V Lt128Vec(D d, V a, V b) {
  static_assert(!IsSigned<TFromD<D>>() && sizeof(TFromD<D>) == 8,
                "D must be u64");
#if HWY_TARGET <= HWY_PPC10 && defined(__SIZEOF_INT128__)
  (void)d;

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const __vector unsigned __int128 a_u128 =
    (__vector unsigned __int128)a.raw;
  const __vector unsigned __int128 b_u128 =
    (__vector unsigned __int128)b.raw;
#else
  // NOTE: Need to swap the halves of both a and b on big-endian targets
  // as the upper 64 bits of a and b are in lane 1 and the lower 64 bits
  // of a and b are in lane 0 whereas the vec_cmplt operation below expects
  // the upper 64 bits in lane 0 and the lower 64 bits in lane 1 on
  // big-endian PPC targets.
  const __vector unsigned __int128 a_u128 =
    (__vector unsigned __int128)vec_sld(a.raw, a.raw, 8);
  const __vector unsigned __int128 b_u128 =
    (__vector unsigned __int128)vec_sld(b.raw, b.raw, 8);
#endif
  return V{(__vector unsigned long long)vec_cmplt(a_u128, b_u128)};
#else
  // Truth table of Eq and Lt for Hi and Lo u64.
  // (removed lines with (=H && cH) or (=L && cL) - cannot both be true)
  // =H =L cH cL  | out = cH | (=H & cL)
  //  0  0  0  0  |  0
  //  0  0  0  1  |  0
  //  0  0  1  0  |  1
  //  0  0  1  1  |  1
  //  0  1  0  0  |  0
  //  0  1  0  1  |  0
  //  0  1  1  0  |  1
  //  1  0  0  0  |  0
  //  1  0  0  1  |  1
  //  1  1  0  0  |  0
  const auto eqHL = Eq(a, b);
  const V ltHL = VecFromMask(d, Lt(a, b));
  const V ltLX = ShiftLeftLanes<1>(ltHL);
  const V vecHx = IfThenElse(eqHL, ltLX, ltHL);
  return InterleaveUpper(d, vecHx, vecHx);
#endif
}

// Returns vector-mask for Eq128.
template <class D, class V = VFromD<D>>
HWY_INLINE V Eq128Vec(D d, V a, V b) {
  static_assert(!IsSigned<TFromD<D>>() && sizeof(TFromD<D>) == 8,
                "D must be u64");
#if HWY_TARGET <= HWY_PPC10 && defined(__SIZEOF_INT128__)
  (void)d;
  return V{(__vector unsigned long long)vec_cmpeq(
             (__vector unsigned __int128)a.raw,
             (__vector unsigned __int128)b.raw)};
#else
  const auto eqHL = VecFromMask(d, Eq(a, b));
  const auto eqLH = Reverse2(d, eqHL);
  return And(eqHL, eqLH);
#endif
}

template <class D, class V = VFromD<D>>
HWY_INLINE V Ne128Vec(D d, V a, V b) {
  static_assert(!IsSigned<TFromD<D>>() && sizeof(TFromD<D>) == 8,
                "D must be u64");
#if HWY_TARGET <= HWY_PPC10 && defined(__SIZEOF_INT128__)
  (void)d;
  return V{(__vector unsigned long long)vec_cmpne(
             (__vector unsigned __int128)a.raw,
             (__vector unsigned __int128)b.raw)};
#else
  const auto neHL = VecFromMask(d, Ne(a, b));
  const auto neLH = Reverse2(d, neHL);
  return Or(neHL, neLH);
#endif
}

template <class D, class V = VFromD<D>>
HWY_INLINE V Lt128UpperVec(D d, V a, V b) {
  const V ltHL = VecFromMask(d, Lt(a, b));
  return InterleaveUpper(d, ltHL, ltHL);
}

template <class D, class V = VFromD<D>>
HWY_INLINE V Eq128UpperVec(D d, V a, V b) {
  const V eqHL = VecFromMask(d, Eq(a, b));
  return InterleaveUpper(d, eqHL, eqHL);
}

template <class D, class V = VFromD<D>>
HWY_INLINE V Ne128UpperVec(D d, V a, V b) {
  const V neHL = VecFromMask(d, Ne(a, b));
  return InterleaveUpper(d, neHL, neHL);
}

}  // namespace detail

template <class D, class V = VFromD<D>>
HWY_API MFromD<D> Lt128(D d, V a, V b) {
  return MaskFromVec(detail::Lt128Vec(d, a, b));
}

template <class D, class V = VFromD<D>>
HWY_API MFromD<D> Eq128(D d, V a, V b) {
  return MaskFromVec(detail::Eq128Vec(d, a, b));
}

template <class D, class V = VFromD<D>>
HWY_API MFromD<D> Ne128(D d, V a, V b) {
  return MaskFromVec(detail::Ne128Vec(d, a, b));
}

template <class D, class V = VFromD<D>>
HWY_API MFromD<D> Lt128Upper(D d, V a, V b) {
  return MaskFromVec(detail::Lt128UpperVec(d, a, b));
}

template <class D, class V = VFromD<D>>
HWY_API MFromD<D> Eq128Upper(D d, V a, V b) {
  return MaskFromVec(detail::Eq128UpperVec(d, a, b));
}

template <class D, class V = VFromD<D>>
HWY_API MFromD<D> Ne128Upper(D d, V a, V b) {
  return MaskFromVec(detail::Ne128UpperVec(d, a, b));
}

// ------------------------------ Min128, Max128 (Lt128)

// Avoids the extra MaskFromVec in Lt128.
template <class D, class V = VFromD<D>>
HWY_API V Min128(D d, const V a, const V b) {
  return IfVecThenElse(detail::Lt128Vec(d, a, b), a, b);
}

template <class D, class V = VFromD<D>>
HWY_API V Max128(D d, const V a, const V b) {
  return IfVecThenElse(detail::Lt128Vec(d, b, a), a, b);
}

template <class D, class V = VFromD<D>>
HWY_API V Min128Upper(D d, const V a, const V b) {
  return IfVecThenElse(detail::Lt128UpperVec(d, a, b), a, b);
}

template <class D, class V = VFromD<D>>
HWY_API V Max128Upper(D d, const V a, const V b) {
  return IfVecThenElse(detail::Lt128UpperVec(d, b, a), a, b);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
