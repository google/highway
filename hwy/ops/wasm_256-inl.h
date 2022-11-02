// Copyright 2021 Google LLC
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

// 256-bit WASM vectors and operations. Experimental.
// External include guard in highway.h - see comment there.

// For half-width vectors. Already includes base.h and shared-inl.h.
#include "hwy/ops/wasm_128-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <typename T>
class Vec256 {
 public:
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

  Vec128<T> v0;
  Vec128<T> v1;
};

template <typename T>
struct Mask256 {
  Mask128<T> m0;
  Mask128<T> m1;
};

// ------------------------------ BitCast

template <typename T, typename FromT>
HWY_API Vec256<T> BitCast(Full256<T> d, Vec256<FromT> v) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = BitCast(dh, v.v0);
  ret.v1 = BitCast(dh, v.v1);
  return ret;
}

// ------------------------------ Zero

template <typename T>
HWY_API Vec256<T> Zero(Full256<T> /* tag */) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = ret.v1 = Zero(dh);
  return ret;
}

template <class D>
using VFromD = decltype(Zero(D()));

// ------------------------------ Set

// Returns a vector/part with all lanes set to "t".
template <typename T>
HWY_API Vec256<T> Set(Full256<T> /* tag */, const T t) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = ret.v1 = Set(dh, t);
  return ret;
}

template <typename T>
HWY_API Vec256<T> Undefined(Full256<T> d) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = ret.v1 = Undefined(dh);
  return ret;
}

template <typename T, typename T2>
Vec256<T> Iota(const Full256<T> d, const T2 first) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = Iota(dh, first);
  ret.v1 = Iota(dh, first + Lanes(dh));
  return ret;
}

// ================================================== ARITHMETIC

template <typename T>
HWY_API Vec256<T> operator+(Vec256<T> a, const Vec256<T> b) {
  a.v0 += b.v0;
  a.v1 += b.v1;
  return a;
}

template <typename T>
HWY_API Vec256<T> operator-(Vec256<T> a, const Vec256<T> b) {
  a.v0 -= b.v0;
  a.v1 -= b.v1;
  return a;
}

// ------------------------------ SumsOf8
HWY_API Vec256<uint64_t> SumsOf8(const Vec256<uint8_t> v) {
  HWY_ABORT("not implemented");
}

template <typename T>
HWY_API Vec256<T> SaturatedAdd(Vec256<T> a, const Vec256<T> b) {
  a.v0 = SaturatedAdd(a.v0, b.v0);
  a.v1 = SaturatedAdd(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> SaturatedSub(Vec256<T> a, const Vec256<T> b) {
  a.v0 = SaturatedSub(a.v0, b.v0);
  a.v1 = SaturatedSub(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> AverageRound(Vec256<T> a, const Vec256<T> b) {
  a.v0 = AverageRound(a.v0, b.v0);
  a.v1 = AverageRound(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> Abs(Vec256<T> v) {
  v.v0 = Abs(v.v0);
  v.v1 = Abs(v.v1);
  return v;
}

// ------------------------------ Shift lanes by constant #bits

template <int kBits, typename T>
HWY_API Vec256<T> ShiftLeft(Vec256<T> v) {
  v.v0 = ShiftLeft<kBits>(v.v0);
  v.v1 = ShiftLeft<kBits>(v.v1);
  return v;
}

template <int kBits, typename T>
HWY_API Vec256<T> ShiftRight(Vec256<T> v) {
  v.v0 = ShiftRight<kBits>(v.v0);
  v.v1 = ShiftRight<kBits>(v.v1);
  return v;
}

// ------------------------------ RotateRight (ShiftRight, Or)
template <int kBits, typename T>
HWY_API Vec256<T> RotateRight(const Vec256<T> v) {
  constexpr size_t kSizeInBits = sizeof(T) * 8;
  static_assert(0 <= kBits && kBits < kSizeInBits, "Invalid shift count");
  if (kBits == 0) return v;
  return Or(ShiftRight<kBits>(v), ShiftLeft<kSizeInBits - kBits>(v));
}

// ------------------------------ Shift lanes by same variable #bits

template <typename T>
HWY_API Vec256<T> ShiftLeftSame(Vec256<T> v, const int bits) {
  v.v0 = ShiftLeftSame(v.v0, bits);
  v.v1 = ShiftLeftSame(v.v1, bits);
  return v;
}

template <typename T>
HWY_API Vec256<T> ShiftRightSame(Vec256<T> v, const int bits) {
  v.v0 = ShiftRightSame(v.v0, bits);
  v.v1 = ShiftRightSame(v.v1, bits);
  return v;
}

// ------------------------------ Min, Max
template <typename T>
HWY_API Vec256<T> Min(Vec256<T> a, const Vec256<T> b) {
  a.v0 = Min(a.v0, b.v0);
  a.v1 = Min(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> Max(Vec256<T> a, const Vec256<T> b) {
  a.v0 = Max(a.v0, b.v0);
  a.v1 = Max(a.v1, b.v1);
  return a;
}
// ------------------------------ Integer multiplication

template <typename T>
HWY_API Vec256<T> operator*(Vec256<T> a, const Vec256<T> b) {
  a.v0 *= b.v0;
  a.v1 *= b.v1;
  return a;
}

template <typename T>
HWY_API Vec256<T> MulHigh(Vec256<T> a, const Vec256<T> b) {
  a.v0 = MulHigh(a.v0, b.v0);
  a.v1 = MulHigh(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> MulFixedPoint15(Vec256<T> a, const Vec256<T> b) {
  a.v0 = MulHigh(a.v0, b.v0);
  a.v1 = MulHigh(a.v1, b.v1);
  return a;
}

template <typename TW, typename TN>
HWY_API Vec256<TW> MulEven(Vec256<TN> a, const Vec256<TN> b) {
  Vec256<TW> ret;
  ret.v0 = MulEven(a.v0, b.v0);
  ret.v1 = MulEven(a.v1, b.v1);
  return ret;
}

template <typename TW, typename TN>
HWY_API Vec256<TW> MulOdd(Vec256<TN> a, const Vec256<TN> b) {
  Vec256<TW> ret;
  ret.v0 = MulOdd(a.v0, b.v0);
  ret.v1 = MulOdd(a.v1, b.v1);
  return ret;
}

// ------------------------------ Negate
template <typename T>
HWY_API Vec256<T> Neg(Vec256<T> v) {
  v.v0 = Neg(v.v0);
  v.v1 = Neg(v.v1);
  return v;
}

// ------------------------------ Floating-point division
template <typename T>
HWY_API Vec256<T> operator/(Vec256<T> a, const Vec256<T> b) {
  a.v0 /= b.v0;
  a.v1 /= b.v1;
  return a;
}

// Approximate reciprocal
HWY_API Vec256<float> ApproximateReciprocal(const Vec256<float> v) {
  const Vec256<float> one = Set(Full256<float>(), 1.0f);
  return one / v;
}

// Absolute value of difference.
HWY_API Vec256<float> AbsDiff(const Vec256<float> a, const Vec256<float> b) {
  return Abs(a - b);
}

// ------------------------------ Floating-point multiply-add variants

// Returns mul * x + add
HWY_API Vec256<float> MulAdd(const Vec256<float> mul, const Vec256<float> x,
                             const Vec256<float> add) {
  // TODO(eustas): replace, when implemented in WASM.
  // TODO(eustas): is it wasm_f32x4_qfma?
  return mul * x + add;
}

// Returns add - mul * x
HWY_API Vec256<float> NegMulAdd(const Vec256<float> mul, const Vec256<float> x,
                                const Vec256<float> add) {
  // TODO(eustas): replace, when implemented in WASM.
  return add - mul * x;
}

// Returns mul * x - sub
HWY_API Vec256<float> MulSub(const Vec256<float> mul, const Vec256<float> x,
                             const Vec256<float> sub) {
  // TODO(eustas): replace, when implemented in WASM.
  // TODO(eustas): is it wasm_f32x4_qfms?
  return mul * x - sub;
}

// Returns -mul * x - sub
HWY_API Vec256<float> NegMulSub(const Vec256<float> mul, const Vec256<float> x,
                                const Vec256<float> sub) {
  // TODO(eustas): replace, when implemented in WASM.
  return Neg(mul) * x - sub;
}

// ------------------------------ Floating-point square root

template <typename T>
HWY_API Vec256<T> Sqrt(Vec256<T> v) {
  v.v0 = Sqrt(v.v0);
  v.v1 = Sqrt(v.v1);
  return v;
}

// Approximate reciprocal square root
HWY_API Vec256<float> ApproximateReciprocalSqrt(const Vec256<float> v) {
  // TODO(eustas): find cheaper a way to calculate this.
  const Vec256<float> one = Set(Full256<float>(), 1.0f);
  return one / Sqrt(v);
}

// ------------------------------ Floating-point rounding

// Toward nearest integer, ties to even
HWY_API Vec256<float> Round(const Vec256<float> v) {
  v.v0 = Round(v.v0);
  v.v1 = Round(v.v1);
  return v;
}

// Toward zero, aka truncate
HWY_API Vec256<float> Trunc(const Vec256<float> v) {
  v.v0 = Trunc(v.v0);
  v.v1 = Trunc(v.v1);
  return v;
}

// Toward +infinity, aka ceiling
HWY_API Vec256<float> Ceil(const Vec256<float> v) {
  v.v0 = Ceil(v.v0);
  v.v1 = Ceil(v.v1);
  return v;
}

// Toward -infinity, aka floor
HWY_API Vec256<float> Floor(const Vec256<float> v) {
  v.v0 = Floor(v.v0);
  v.v1 = Floor(v.v1);
  return v;
}

// ------------------------------ Floating-point classification

template <typename T>
HWY_API Mask256<T> IsNaN(const Vec256<T> v) {
  return v != v;
}

template <typename T, HWY_IF_FLOAT(T)>
HWY_API Mask256<T> IsInf(const Vec256<T> v) {
  const Full256<T> d;
  const RebindToSigned<decltype(d)> di;
  const VFromD<decltype(di)> vi = BitCast(di, v);
  // 'Shift left' to clear the sign bit, check for exponent=max and mantissa=0.
  return RebindMask(d, Eq(Add(vi, vi), Set(di, hwy::MaxExponentTimes2<T>())));
}

// Returns whether normal/subnormal/zero.
template <typename T, HWY_IF_FLOAT(T)>
HWY_API Mask256<T> IsFinite(const Vec256<T> v) {
  const Full256<T> d;
  const RebindToUnsigned<decltype(d)> du;
  const RebindToSigned<decltype(d)> di;  // cheaper than unsigned comparison
  const VFromD<decltype(du)> vu = BitCast(du, v);
  // 'Shift left' to clear the sign bit, then right so we can compare with the
  // max exponent (cannot compare with MaxExponentTimes2 directly because it is
  // negative and non-negative floats would be greater).
  const VFromD<decltype(di)> exp =
      BitCast(di, ShiftRight<hwy::MantissaBits<T>() + 1>(Add(vu, vu)));
  return RebindMask(d, Lt(exp, Set(di, hwy::MaxExponentField<T>())));
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.

template <typename TFrom, typename TTo>
HWY_API Mask256<TTo> RebindMask(Full256<TTo> /*tag*/, Mask256<TFrom> m) {
  static_assert(sizeof(TFrom) == sizeof(TTo), "Must have same size");
  return Mask256<TTo>{Mask128<TTo>{m.m0.raw}, Mask128<TTo>{m.m1.raw}};
}

template <typename T>
HWY_API Mask256<T> TestBit(Vec256<T> v, Vec256<T> bit) {
  static_assert(!hwy::IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

template <typename T>
HWY_API Vec256<T> operator==(Vec256<T> a, const Vec256<T> b) {
  a.v0 = operator==(a.v0, b.v0);
  a.v1 = operator==(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> operator!=(Vec256<T> a, const Vec256<T> b) {
  a.v0 = operator!=(a.v0, b.v0);
  a.v1 = operator!=(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> operator<(Vec256<T> a, const Vec256<T> b) {
  a.v0 = operator<(a.v0, b.v0);
  a.v1 = operator<(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> operator>(Vec256<T> a, const Vec256<T> b) {
  a.v0 = operator>(a.v0, b.v0);
  a.v1 = operator>(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> operator<=(Vec256<T> a, const Vec256<T> b) {
  a.v0 = operator<=(a.v0, b.v0);
  a.v1 = operator<=(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> operator>=(Vec256<T> a, const Vec256<T> b) {
  a.v0 = operator>=(a.v0, b.v0);
  a.v1 = operator>=(a.v1, b.v1);
  return a;
}

// ------------------------------ FirstN (Iota, Lt)

template <typename T>
HWY_API Mask256<T> FirstN(const Full256<T> d, size_t num) {
  const RebindToSigned<decltype(d)> di;  // Signed comparisons may be cheaper.
  return RebindMask(d, Iota(di, 0) < Set(di, static_cast<MakeSigned<T>>(num)));
}

// ================================================== LOGICAL

template <typename T>
HWY_API Vec256<T> Not(Vec256<T> v) {
  v.v0 = Not(v.v0);
  v.v1 = Not(v.v1);
  return v;
}

template <typename T>
HWY_API Vec256<T> And(Vec256<T> a, Vec256<T> b) {
  a.v0 = And(a.v0, b.v0);
  a.v1 = And(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> AndNot(Vec256<T> not_mask, Vec256<T> mask) {
  a.v0 = AndNot(a.v0, b.v0);
  a.v1 = AndNot(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> Or(Vec256<T> a, Vec256<T> b) {
  a.v0 = Or(a.v0, b.v0);
  a.v1 = Or(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> Xor(Vec256<T> a, Vec256<T> b) {
  a.v0 = Xor(a.v0, b.v0);
  a.v1 = Xor(a.v1, b.v1);
  return a;
}

template <typename T>
HWY_API Vec256<T> Or3(Vec256<T> o1, Vec256<T> o2, Vec256<T> o3) {
  return Or(o1, Or(o2, o3));
}

template <typename T>
HWY_API Vec256<T> OrAnd(Vec256<T> o, Vec256<T> a1, Vec256<T> a2) {
  return Or(o, And(a1, a2));
}

template <typename T>
HWY_API Vec256<T> IfVecThenElse(Vec256<T> mask, Vec256<T> yes, Vec256<T> no) {
  return IfThenElse(MaskFromVec(mask), yes, no);
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

// ------------------------------ CopySign

template <typename T>
HWY_API Vec256<T> CopySign(const Vec256<T> magn, const Vec256<T> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  const auto msb = SignBit(Full256<T>());
  return Or(AndNot(msb, magn), And(msb, sign));
}

template <typename T>
HWY_API Vec256<T> CopySignToAbs(const Vec256<T> abs, const Vec256<T> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  return Or(abs, And(SignBit(Full256<T>()), sign));
}

// ------------------------------ BroadcastSignBit (compare)

template <typename T, HWY_IF_NOT_LANE_SIZE(T, 1)>
HWY_API Vec256<T> BroadcastSignBit(const Vec256<T> v) {
  return ShiftRight<sizeof(T) * 8 - 1>(v);
}
HWY_API Vec256<int8_t> BroadcastSignBit(const Vec256<int8_t> v) {
  return VecFromMask(Full256<int8_t>(), v < Zero(Full256<int8_t>()));
}

// ------------------------------ Mask

// Mask and Vec are the same (true = FF..FF).
template <typename T>
HWY_API Mask256<T> MaskFromVec(const Vec256<T> v) {
  Mask256<T> m;
  m.m0 = MaskFromVec(v.v0);
  m.m1 = MaskFromVec(v.v1);
  return m;
}

template <typename T>
HWY_API Vec256<T> VecFromMask(Full256<T> /* tag */, Mask256<T> m) {
  Vec256<T> v;
  v.v0 = MaskFromVec(m.m0);
  v.v1 = MaskFromVec(m.m1);
  return m;
}

// mask ? yes : no
template <typename T>
HWY_API Vec256<T> IfThenElse(Mask256<T> mask, Vec256<T> yes, Vec256<T> no) {
  yes.v0 = IfThenElse(mask.m0, yes.v0, no.v0);
  yes.v1 = IfThenElse(mask.m1, yes.v1, no.v1);
  return yes;
}

// mask ? yes : 0
template <typename T>
HWY_API Vec256<T> IfThenElseZero(Mask256<T> mask, Vec256<T> yes) {
  return yes & VecFromMask(Full256<T>(), mask);
}

// mask ? 0 : no
template <typename T>
HWY_API Vec256<T> IfThenZeroElse(Mask256<T> mask, Vec256<T> no) {
  return AndNot(VecFromMask(Full256<T>(), mask), no);
}

template <typename T>
HWY_API Vec256<T> IfNegativeThenElse(Vec256<T> v, Vec256<T> yes, Vec256<T> no) {
  v.v0 = IfNegativeThenElse(v.v0, yes.v0, no.v0);
  v.v1 = IfNegativeThenElse(v.v1, yes.v1, no.v1);
  return v;
}

template <typename T, HWY_IF_FLOAT(T)>
HWY_API Vec256<T> ZeroIfNegative(Vec256<T> v) {
  return IfThenZeroElse(v < Zero(d), v);
}

// ------------------------------ Mask logical

template <typename T>
HWY_API Mask256<T> Not(const Mask256<T> m) {
  return MaskFromVec(Not(VecFromMask(Full256<T>(), m)));
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

// ------------------------------ Shl (BroadcastSignBit, IfThenElse)
template <typename T>
HWY_API Vec256<T> operator<<(Vec256<T> v, const Vec256<T> bits) {
  v.v0 = operator<<(v.v0, bits.v0);
  v.v1 = operator<<(v.v1, bits.v1);
  return v;
}

// ------------------------------ Shr (BroadcastSignBit, IfThenElse)
template <typename T>
HWY_API Vec256<T> operator>>(Vec256<T> v, const Vec256<T> bits) {
  v.v0 = operator>>(v.v0, bits.v0);
  v.v1 = operator>>(v.v1, bits.v1);
  return v;
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
HWY_API Vec256<T> Load(Full256<T> d, const T* HWY_RESTRICT aligned) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = Load(dh, aligned);
  ret.v1 = Load(dh, aligned + Lanes(dh));
  return ret;
}

template <typename T>
HWY_API Vec256<T> MaskedLoad(Mask256<T> m, Full256<T> d,
                             const T* HWY_RESTRICT aligned) {
  return IfThenElseZero(m, Load(d, aligned));
}

// LoadU == Load.
template <typename T>
HWY_API Vec256<T> LoadU(Full256<T> d, const T* HWY_RESTRICT p) {
  return Load(d, p);
}

template <typename T>
HWY_API Vec256<T> LoadDup128(Full256<T> d, const T* HWY_RESTRICT p) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v0 = ret.v1 = Load(dh, aligned);
  return ret;
}

// ------------------------------ Store

template <typename T>
HWY_API void Store(Vec256<T> v, Full256<T> d, T* HWY_RESTRICT aligned) {
  const Half<decltype(d)> dh;
  Store(v.v0, dh, aligned);
  Store(v.v1, dh, aligned + Lanes(dh));
}

// StoreU == Store.
template <typename T>
HWY_API void StoreU(Vec256<T> v, Full256<T> d, T* HWY_RESTRICT p) {
  Store(v, d, p);
}

template <typename T>
HWY_API void BlendedStore(Vec256<T> v, Mask256<T> m, Full256<T> d,
                          T* HWY_RESTRICT p) {
  StoreU(IfThenElse(m, v, LoadU(d, p)), d, p);
}

// ------------------------------ Non-temporal stores

// Same as aligned stores on non-x86.

template <typename T>
HWY_API void Stream(Vec256<T> v, Full256<T> d, T* HWY_RESTRICT aligned) {
  Store(v, d, aligned);
}

// ------------------------------ Scatter (Store)

template <typename T, typename Offset>
HWY_API void ScatterOffset(Vec256<T> v, Full256<T> d, T* HWY_RESTRICT base,
                           const Vec256<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "Must match for portability");

  alignas(32) T lanes[32 / sizeof(T)];
  Store(v, d, lanes);

  alignas(32) Offset offset_lanes[32 / sizeof(T)];
  Store(offset, Full256<Offset>(), offset_lanes);

  uint8_t* base_bytes = reinterpret_cast<uint8_t*>(base);
  for (size_t i = 0; i < N; ++i) {
    CopyBytes<sizeof(T)>(&lanes[i], base_bytes + offset_lanes[i]);
  }
}

template <typename T, typename Index>
HWY_API void ScatterIndex(Vec256<T> v, Full256<T> d, T* HWY_RESTRICT base,
                          const Vec256<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "Must match for portability");

  alignas(32) T lanes[32 / sizeof(T)];
  Store(v, d, lanes);

  alignas(32) Index index_lanes[32 / sizeof(T)];
  Store(index, Full256<Index>(), index_lanes);

  for (size_t i = 0; i < N; ++i) {
    base[index_lanes[i]] = lanes[i];
  }
}

// ------------------------------ Gather (Load/Store)

template <typename T, typename Offset>
HWY_API Vec256<T> GatherOffset(const Full256<T> d, const T* HWY_RESTRICT base,
                               const Vec256<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "Must match for portability");

  alignas(32) Offset offset_lanes[32 / sizeof(T)];
  Store(offset, Full256<Offset>(), offset_lanes);

  alignas(32) T lanes[32 / sizeof(T)];
  const uint8_t* base_bytes = reinterpret_cast<const uint8_t*>(base);
  for (size_t i = 0; i < N; ++i) {
    CopyBytes<sizeof(T)>(base_bytes + offset_lanes[i], &lanes[i]);
  }
  return Load(d, lanes);
}

template <typename T, typename Index>
HWY_API Vec256<T> GatherIndex(const Full256<T> d, const T* HWY_RESTRICT base,
                              const Vec256<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "Must match for portability");

  alignas(32) Index index_lanes[32 / sizeof(T)];
  Store(index, Full256<Index>(), index_lanes);

  alignas(32) T lanes[32 / sizeof(T)];
  for (size_t i = 0; i < N; ++i) {
    lanes[i] = base[index_lanes[i]];
  }
  return Load(d, lanes);
}

// ================================================== SWIZZLE

// ------------------------------ ExtractLane
template <typename T, size_t N>
HWY_API T ExtractLane(const Vec128<T, N> v, size_t i) {
  alignas(32) T lanes[32 / sizeof(T)];
  Store(v, Full256<T>(), lanes);
  return lanes[i];
}

// ------------------------------ InsertLane
template <typename T, size_t N>
HWY_API Vec128<T, N> InsertLane(const Vec128<T, N> v, size_t i, T t) {
  Full256<T> d;
  alignas(32) T lanes[32 / sizeof(T)];
  Store(v, d, lanes);
  lanes[i] = t;
  return Load(d, lanes);
}

// ------------------------------ LowerHalf

template <typename T>
HWY_API Vec128<T> LowerHalf(Full128<T> /* tag */, Vec256<T> v) {
  return v.v0;
}

template <typename T>
HWY_API Vec128<T> LowerHalf(Vec256<T> v) {
  return v.v0;
}

// ------------------------------ ShiftLeftBytes

template <int kBytes, typename T>
HWY_API Vec256<T> ShiftLeftBytes(Full256<T> d, Vec256<T> v) {
  const Half<decltype(d)> dh;
  v.v0 = ShiftLeftBytes<kBytes>(v.v0);
  v.v1 = ShiftLeftBytes<kBytes>(v.v1);
  return v;
}

template <int kBytes, typename T>
HWY_API Vec256<T> ShiftLeftBytes(Vec256<T> v) {
  return ShiftLeftBytes<kBytes>(Full256<T>(), v);
}

// ------------------------------ ShiftLeftLanes

template <int kLanes, typename T>
HWY_API Vec256<T> ShiftLeftLanes(Full256<T> d, const Vec256<T> v) {
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, ShiftLeftBytes<kLanes * sizeof(T)>(BitCast(d8, v)));
}

template <int kLanes, typename T>
HWY_API Vec256<T> ShiftLeftLanes(const Vec256<T> v) {
  return ShiftLeftLanes<kLanes>(Full256<T>(), v);
}

// ------------------------------ ShiftRightBytes
template <int kBytes, typename T>
HWY_API Vec256<T> ShiftRightBytes(Full256<T> d, Vec256<T> v) {
  const Half<decltype(d)> dh;
  v.v0 = ShiftRightBytes<kBytes>(v.v0);
  v.v1 = ShiftRightBytes<kBytes>(v.v1);
  return v;
}

// ------------------------------ ShiftRightLanes
template <int kLanes, typename T>
HWY_API Vec256<T> ShiftRightLanes(Full256<T> d, const Vec256<T> v) {
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, ShiftRightBytes<kLanes * sizeof(T)>(BitCast(d8, v)));
}

// ------------------------------ UpperHalf (ShiftRightBytes)

template <typename T>
HWY_API Vec128<T> UpperHalf(Full128<T> /* tag */, const Vec256<T> v) {
  return v.v1;
}

// ------------------------------ CombineShiftRightBytes

template <int kBytes, typename T, class V = Vec256<T>>
HWY_API V CombineShiftRightBytes(Full256<T> d, V hi, V lo) {
  const Half<decltype(d)> dh;
  hi.v0 = CombineShiftRightBytes<kBytes>(hi.v0, lo.v0);
  hi.v1 = CombineShiftRightBytes<kBytes>(hi.v1, lo.v1);
  return hi;
}

// ------------------------------ Broadcast/splat any lane

template <int kLane, typename T>
HWY_API Vec256<T> Broadcast(const Vec256<T> v) {
  Vec256<T> ret;
  ret.v0 = Broadcast<kLane>(v.v0);
  ret.v1 = Broadcast<kLane>(v.v1);
  return ret;
}

// ------------------------------ TableLookupBytes

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes, i.e.
// lane indices in [0, 16).
template <typename T, typename TI>
HWY_API Vec256<TI> TableLookupBytes(const Vec256<T> bytes, Vec256<TI> from) {
  from.v0 = TableLookupBytes(bytes.v0, from.v0);
  from.v1 = TableLookupBytes(bytes.v1, from.v1);
  return from;
}

template <typename T, typename TI>
HWY_API Vec256<TI> TableLookupBytesOr0(const Vec256<T> bytes,
                                       const Vec256<TI> from) {
  from.v0 = TableLookupBytesOr0(bytes.v0, from.v0);
  from.v1 = TableLookupBytesOr0(bytes.v1, from.v1);
  return from;
}

// ------------------------------ Hard-coded shuffles

template <typename T>
HWY_API Vec256<T> Shuffle2301(Vec256<T> v) {
  v.v0 = Shuffle2301(v.v0);
  v.v1 = Shuffle2301(v.v1);
  return v;
}

template <typename T>
HWY_API Vec256<T> Shuffle1032(Vec256<T> v) {
  v.v0 = Shuffle1032(v.v0);
  v.v1 = Shuffle1032(v.v1);
  return v;
}

template <typename T>
HWY_API Vec256<T> Shuffle0321(Vec256<T> v) {
  v.v0 = Shuffle0321(v.v0);
  v.v1 = Shuffle0321(v.v1);
  return v;
}

template <typename T>
HWY_API Vec256<T> Shuffle2103(Vec256<T> v) {
  v.v0 = Shuffle2103(v.v0);
  v.v1 = Shuffle2103(v.v1);
  return v;
}

template <typename T>
HWY_API Vec256<T> Shuffle0123(Vec256<T> v) {
  v.v0 = Shuffle0123(v.v0);
  v.v1 = Shuffle0123(v.v1);
  return v;
}

// ------------------------------ TableLookupLanes

// Returned by SetTableIndices for use by TableLookupLanes.
template <typename T>
struct Indices256 {
  __v128_u raw;
};

template <typename T, typename TI>
HWY_API Indices256<T> IndicesFromVec(Full256<T> d, Vec256<TI> vec) {
  static_assert(sizeof(T) == sizeof(TI), "Index size must match lane");
  return Indices256<T>{};
}

template <typename T, typename TI>
HWY_API Indices256<T> SetTableIndices(Full256<T> d, const TI* idx) {
  const Rebind<TI, decltype(d)> di;
  return IndicesFromVec(d, LoadU(di, idx));
}

template <typename T>
HWY_API Vec256<T> TableLookupLanes(Vec256<T> v, Indices256<T> idx) {
  using TI = MakeSigned<T>;
  const Full256<T> d;
  const Full256<TI> di;
  return BitCast(d, TableLookupBytes(BitCast(di, v), Vec256<TI>{idx.raw}));
}

// ------------------------------ Reverse
template <typename T>
HWY_API Vec256<T> Reverse(Full256<T> d, const Vec256<T> v) {
  const Half<decltype(d)> dh;
  Vec256<T> ret;
  ret.v1 = Reverse(dh, v.v0);  // note reversed v1 member order
  ret.v0 = Reverse(dh, v.v1);
  return ret;
}

// ------------------------------ Reverse2
template <typename T>
HWY_API Vec256<T> Reverse2(Full256<T> d, Vec256<T> v) {
  const Half<decltype(d)> dh;
  v.v0 = Reverse2(dh, v.v0);
  v.v1 = Reverse2(dh, v.v1);
  return v;
}

// ------------------------------ Reverse4
template <typename T>
HWY_API Vec256<T> Reverse4(Full256<T> d, const Vec256<T> v) {
  const Half<decltype(d)> dh;
  v.v0 = Reverse4(dh, v.v0);
  v.v1 = Reverse4(dh, v.v1);
  return v;
}

// ------------------------------ Reverse8
template <typename T>
HWY_API Vec256<T> Reverse8(Full256<T> d, const Vec256<T> v) {
  const Half<decltype(d)> dh;
  v.v0 = Reverse8(dh, v.v0);
  v.v1 = Reverse8(dh, v.v1);
  return v;
}

// ------------------------------ InterleaveLower

template <typename T>
HWY_API Vec256<T> InterleaveLower(Vec256<T> a, Vec256<T> b) {
  a.v0 = InterleaveLower(a.v0, b.v0);
  a.v1 = InterleaveLower(a.v1, b.v1);
  return a;
}

// Additional overload for the optional tag.
template <typename T, class V = Vec256<T>>
HWY_API V InterleaveLower(Full256<T> /* tag */, V a, V b) {
  return InterleaveLower(a, b);
}

// ------------------------------ InterleaveUpper (UpperHalf)

template <typename T, class V = Vec256<T>>
HWY_API V InterleaveUpper(Full256<T> d, V a, V b) {
  const Half<decltype(d)> dh;
  a.v0 = InterleaveUpper(dh, a.v0, b.v0);
  a.v1 = InterleaveUpper(dh, a.v1, b.v1);
  return a;
}

// ------------------------------ ZipLower/ZipUpper (InterleaveLower)

// Same as Interleave*, except that the return lanes are double-width integers;
// this is necessary because the single-lane scalar cannot return two values.
template <typename T, class DW = RepartitionToWide<Full256<T>>>
HWY_API VFromD<DW> ZipLower(Vec256<T> a, Vec256<T> b) {
  return BitCast(DW(), InterleaveLower(a, b));
}
template <typename T, class D = Full256<T>, class DW = RepartitionToWide<D>>
HWY_API VFromD<DW> ZipLower(DW dw, Vec256<T> a, Vec256<T> b) {
  return BitCast(dw, InterleaveLower(D(), a, b));
}

template <typename T, class D = Full256<T>, class DW = RepartitionToWide<D>>
HWY_API VFromD<DW> ZipUpper(DW dw, Vec256<T> a, Vec256<T> b) {
  return BitCast(dw, InterleaveUpper(D(), a, b));
}

// ================================================== COMBINE

// ------------------------------ Combine (InterleaveLower)
template <typename T>
HWY_API Vec256<T> Combine(Full256<T> /* d */, Vec128<T> hi, Vec128<T> lo) {
  Vec256<T> ret;
  ret.v1 = hi;
  ret.v0 = lo;
  return ret;
}

// ------------------------------ ZeroExtendVector (Combine)
template <typename T>
HWY_API Vec256<T> ZeroExtendVector(Full256<T> d, Vec128<T> lo) {
  const Half<decltype(d)> dh;
  return Combine(d, Zero(dh), lo);
}

// ------------------------------ ConcatLowerLower
template <typename T>
HWY_API Vec256<T> ConcatLowerLower(Full256<T> /* tag */, const Vec256<T> hi,
                                   const Vec256<T> lo) {
  Vec256<T> ret;
  ret.hi = hi.v0;
  ret.lo = lo.v0;
  return ret;
}

// ------------------------------ ConcatUpperUpper
template <typename T>
HWY_API Vec256<T> ConcatUpperUpper(Full256<T> /* tag */, const Vec256<T> hi,
                                   const Vec256<T> lo) {
  Vec256<T> ret;
  ret.hi = hi.v1;
  ret.lo = lo.v1;
  return ret;
}

// ------------------------------ ConcatLowerUpper
template <typename T>
HWY_API Vec256<T> ConcatLowerUpper(Full256<T> d, const Vec256<T> hi,
                                   const Vec256<T> lo) {
  Vec256<T> ret;
  ret.hi = hi.v0;
  ret.lo = lo.v1;
  return ret;
}

// ------------------------------ ConcatUpperLower
template <typename T>
HWY_API Vec256<T> ConcatUpperLower(Full256<T> d, const Vec256<T> hi,
                                   const Vec256<T> lo) {
  Vec256<T> ret;
  ret.hi = hi.v1;
  ret.lo = lo.v0;
  return ret;
}

// ------------------------------ ConcatOdd

// 32-bit
template <typename T, HWY_IF_LANE_SIZE(T, 4)>
HWY_API Vec256<T> ConcatOdd(Full256<T> /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return Vec256<T>{wasm_i32x4_shuffle(lo.raw, hi.raw, 1, 3, 5, 7)};
}

// 64-bit full - no partial because we need at least two inputs to have
// even/odd.
template <typename T, HWY_IF_LANE_SIZE(T, 8)>
HWY_API Vec256<T> ConcatOdd(Full256<T> /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return InterleaveUpper(Full256<T>(), lo, hi);
}

// ------------------------------ ConcatEven (InterleaveLower)

// 32-bit full
template <typename T, HWY_IF_LANE_SIZE(T, 4)>
HWY_API Vec256<T> ConcatEven(Full256<T> /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return Vec256<T>{wasm_i32x4_shuffle(lo.raw, hi.raw, 0, 2, 4, 6)};
}

// 64-bit full - no partial because we need at least two inputs to have
// even/odd.
template <typename T, HWY_IF_LANE_SIZE(T, 8)>
HWY_API Vec256<T> ConcatEven(Full256<T> /* tag */, Vec256<T> hi, Vec256<T> lo) {
  return InterleaveLower(Full256<T>(), lo, hi);
}

// ------------------------------ DupEven
template <typename T>
HWY_API Vec256<T> DupEven(Vec256<T> v) {
  v.v0 = DupEven(v.v0);
  v.v1 = DupEven(v.v1);
  return v;
}

// ------------------------------ DupOdd
template <typename T>
HWY_API Vec256<T> DupOdd(Vec256<T> v) {
  v.v0 = DupOdd(v.v0);
  v.v1 = DupOdd(v.v1);
  return v;
}

// ------------------------------ OddEven

template <typename T>
HWY_API Vec256<T> OddEven(Vec256<T> a, const Vec256<T> b) {
  a.v0 = OddEven(a.v0, b.v0);
  a.v1 = OddEven(a.v1, b.v1);
  return a;
}

// ------------------------------ OddEvenBlocks
template <typename T>
HWY_API Vec256<T> OddEvenBlocks(Vec256<T> odd, Vec256<T> even) {
  odd.v0 = even.v0;
  return odd;
}

// ------------------------------ SwapAdjacentBlocks

template <typename T>
HWY_API Vec256<T> SwapAdjacentBlocks(Vec256<T> v) {
  Vec256<T> ret;
  ret.v0 = v.v1;  // swapped order
  ret.v1 = v.v0;
  return ret;
}

// ------------------------------ ReverseBlocks

template <typename T>
HWY_API Vec256<T> ReverseBlocks(Full256<T> /* tag */, const Vec256<T> v) {
  return SwapAdjacentBlocks(v);  // 2 blocks, so Swap = Reverse
}

// ================================================== CONVERT

// ------------------------------ Promotions (part w/ narrow lanes -> full)

// Unsigned: zero-extend.
HWY_API Vec256<uint16_t> PromoteTo(Full256<uint16_t> /* tag */,
                                   const Vec128<uint8_t> v) {
  return Vec256<uint16_t>{wasm_u16x8_extend_low_u8x16(v.raw)};
}
HWY_API Vec256<uint32_t> PromoteTo(Full256<uint32_t> /* tag */,
                                   const Vec128<uint8_t> v) {
  return Vec256<uint32_t>{
      wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(v.raw))};
}
HWY_API Vec256<int16_t> PromoteTo(Full256<int16_t> /* tag */,
                                  const Vec128<uint8_t> v) {
  return Vec256<int16_t>{wasm_u16x8_extend_low_u8x16(v.raw)};
}
HWY_API Vec256<int32_t> PromoteTo(Full256<int32_t> /* tag */,
                                  const Vec128<uint8_t> v) {
  return Vec256<int32_t>{
      wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(v.raw))};
}
HWY_API Vec256<uint32_t> PromoteTo(Full256<uint32_t> /* tag */,
                                   const Vec128<uint16_t> v) {
  return Vec256<uint32_t>{wasm_u32x4_extend_low_u16x8(v.raw)};
}
HWY_API Vec256<int32_t> PromoteTo(Full256<int32_t> /* tag */,
                                  const Vec128<uint16_t> v) {
  return Vec256<int32_t>{wasm_u32x4_extend_low_u16x8(v.raw)};
}

// Signed: replicate sign bit.
HWY_API Vec256<int16_t> PromoteTo(Full256<int16_t> /* tag */,
                                  const Vec128<int8_t> v) {
  return Vec256<int16_t>{wasm_i16x8_extend_low_i8x16(v.raw)};
}
HWY_API Vec256<int32_t> PromoteTo(Full256<int32_t> /* tag */,
                                  const Vec128<int8_t> v) {
  return Vec256<int32_t>{
      wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(v.raw))};
}
HWY_API Vec256<int32_t> PromoteTo(Full256<int32_t> /* tag */,
                                  const Vec128<int16_t> v) {
  return Vec256<int32_t>{wasm_i32x4_extend_low_i16x8(v.raw)};
}

HWY_API Vec256<double> PromoteTo(Full256<double> /* tag */,
                                 const Vec128<int32_t> v) {
  return Vec256<double>{wasm_f64x2_convert_low_i32x4(v.raw)};
}

HWY_API Vec256<float> PromoteTo(Full256<float> /* tag */,
                                const Vec128<float16_t> v) {
  const Full256<int32_t> di32;
  const Full256<uint32_t> du32;
  const Full256<float> df32;
  // Expand to u32 so we can shift.
  const auto bits16 = PromoteTo(du32, Vec256<uint16_t>{v.raw});
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
}

HWY_API Vec256<float> PromoteTo(Full256<float> df32,
                                const Vec128<bfloat16_t> v) {
  const Rebind<uint16_t, decltype(df32)> du16;
  const RebindToSigned<decltype(df32)> di32;
  return BitCast(df32, ShiftLeft<16>(PromoteTo(di32, BitCast(du16, v))));
}

// ------------------------------ Demotions (full -> part w/ narrow lanes)

HWY_API Vec128<uint16_t> DemoteTo(Full128<uint16_t> /* tag */,
                                  const Vec256<int32_t> v) {
  return Vec128<uint16_t>{wasm_u16x8_narrow_i32x4(v.raw, v.raw)};
}

HWY_API Vec128<int16_t> DemoteTo(Full128<int16_t> /* tag */,
                                 const Vec256<int32_t> v) {
  return Vec128<int16_t>{wasm_i16x8_narrow_i32x4(v.raw, v.raw)};
}

HWY_API Vec128<uint8_t> DemoteTo(Full128<uint8_t> /* tag */,
                                 const Vec256<int32_t> v) {
  const auto intermediate = wasm_i16x8_narrow_i32x4(v.raw, v.raw);
  return Vec128<uint8_t>{wasm_u8x16_narrow_i16x8(intermediate, intermediate)};
}

HWY_API Vec128<uint8_t> DemoteTo(Full128<uint8_t> /* tag */,
                                 const Vec256<int16_t> v) {
  return Vec128<uint8_t>{wasm_u8x16_narrow_i16x8(v.raw, v.raw)};
}

HWY_API Vec128<int8_t> DemoteTo(Full128<int8_t> /* tag */,
                                const Vec256<int32_t> v) {
  const auto intermediate = wasm_i16x8_narrow_i32x4(v.raw, v.raw);
  return Vec128<int8_t>{wasm_i8x16_narrow_i16x8(intermediate, intermediate)};
}

HWY_API Vec128<int8_t> DemoteTo(Full128<int8_t> /* tag */,
                                const Vec256<int16_t> v) {
  return Vec128<int8_t>{wasm_i8x16_narrow_i16x8(v.raw, v.raw)};
}

HWY_API Vec128<int32_t> DemoteTo(Full128<int32_t> /* di */,
                                 const Vec256<double> v) {
  return Vec128<int32_t>{wasm_i32x4_trunc_sat_f64x2_zero(v.raw)};
}

HWY_API Vec128<float16_t> DemoteTo(Full128<float16_t> /* tag */,
                                   const Vec256<float> v) {
  const Full256<int32_t> di;
  const Full256<uint32_t> du;
  const Full256<uint16_t> du16;
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
  return Vec128<float16_t>{DemoteTo(du16, bits16).raw};
}

HWY_API Vec128<bfloat16_t> DemoteTo(Full128<bfloat16_t> dbf16,
                                    const Vec256<float> v) {
  const Rebind<int32_t, decltype(dbf16)> di32;
  const Rebind<uint32_t, decltype(dbf16)> du32;  // for logical shift right
  const Rebind<uint16_t, decltype(dbf16)> du16;
  const auto bits_in_32 = BitCast(di32, ShiftRight<16>(BitCast(du32, v)));
  return BitCast(dbf16, DemoteTo(du16, bits_in_32));
}

HWY_API Vec128<bfloat16_t> ReorderDemote2To(Full128<bfloat16_t> dbf16,
                                            Vec256<float> a, Vec256<float> b) {
  const RebindToUnsigned<decltype(dbf16)> du16;
  const Repartition<uint32_t, decltype(dbf16)> du32;
  const Vec256<uint32_t> b_in_even = ShiftRight<16>(BitCast(du32, b));
  return BitCast(dbf16, OddEven(BitCast(du16, a), BitCast(du16, b_in_even)));
}

HWY_API Vec512<int16_t> ReorderDemote2To(Full512<int16_t> /*d16*/,
                                         Vec512<int32_t> a, Vec512<int32_t> b) {
  return Vec512<int16_t>{wasm_i16x8_narrow_i32x4(a.raw, b.raw)};
}

// For already range-limited input [0, 255].
HWY_API Vec256<uint8_t> U8FromU32(const Vec256<uint32_t> v) {
  const auto intermediate = wasm_i16x8_narrow_i32x4(v.raw, v.raw);
  return Vec256<uint8_t>{wasm_u8x16_narrow_i16x8(intermediate, intermediate)};
}

// ------------------------------ Truncations

HWY_API Vec256<uint8_t, 4> TruncateTo(Simd<uint8_t, 4, 0> /* tag */,
                                      const Vec256<uint64_t> v) {
  return Vec256<uint8_t, 4>{wasm_i8x16_shuffle(v.v0.raw, v.v1.raw, 0, 8, 16, 24,
                                               0, 8, 16, 24, 0, 8, 16, 24, 0, 8,
                                               16, 24)};
}

HWY_API Vec256<uint16_t, 4> TruncateTo(Simd<uint16_t, 4, 0> /* tag */,
                                       const Vec256<uint64_t> v) {
  return Vec256<uint16_t, 4>{wasm_i8x16_shuffle(v.v0.raw, v.v1.raw, 0, 1, 8, 9,
                                                16, 17, 24, 25, 0, 1, 8, 9, 16,
                                                17, 24, 25)};
}

HWY_API Vec256<uint32_t, 4> TruncateTo(Simd<uint32_t, 4, 0> /* tag */,
                                       const Vec256<uint64_t> v) {
  return Vec256<uint32_t, 4>{wasm_i8x16_shuffle(v.v0.raw, v.v1.raw, 0, 1, 2, 3,
                                                8, 9, 10, 11, 16, 17, 18, 19,
                                                24, 25, 26, 27)};
}

HWY_API Vec256<uint8_t, 8> TruncateTo(Simd<uint8_t, 8, 0> /* tag */,
                                      const Vec256<uint32_t> v) {
  return Vec256<uint8_t, 8>{wasm_i8x16_shuffle(v.v0.raw, v.v1.raw, 0, 4, 8, 12,
                                               16, 20, 24, 28, 0, 4, 8, 12, 16,
                                               20, 24, 28)};
}

HWY_API Vec256<uint16_t, 8> TruncateTo(Simd<uint16_t, 8, 0> /* tag */,
                                       const Vec256<uint32_t> v) {
  return Vec256<uint16_t, 8>{wasm_i8x16_shuffle(v.v0.raw, v.v1.raw, 0, 1, 4, 5,
                                                8, 9, 12, 13, 16, 17, 20, 21,
                                                24, 25, 28, 29)};
}

HWY_API Vec256<uint8_t, 16> TruncateTo(Simd<uint8_t, 16, 0> /* tag */,
                                       const Vec256<uint16_t> v) {
  return Vec256<uint8_t, 16>{wasm_i8x16_shuffle(v.v0.raw, v.v1.raw, 0, 2, 4, 6,
                                                8, 10, 12, 14, 16, 18, 20, 22,
                                                24, 26, 28, 30)};
}

// ------------------------------ Convert i32 <=> f32 (Round)

HWY_API Vec256<float> ConvertTo(Full256<float> /* tag */,
                                const Vec256<int32_t> v) {
  return Vec256<float>{wasm_f32x4_convert_i32x4(v.raw)};
}
HWY_API Vec256<float> ConvertTo(Full256<float> /* tag */,
                                const Vec256<uint32_t> v) {
  return Vec256<float>{wasm_f32x4_convert_u32x4(v.raw)};
}
// Truncates (rounds toward zero).
HWY_API Vec256<int32_t> ConvertTo(Full256<int32_t> /* tag */,
                                  const Vec256<float> v) {
  return Vec256<int32_t>{wasm_i32x4_trunc_sat_f32x4(v.raw)};
}

HWY_API Vec256<int32_t> NearestInt(const Vec256<float> v) {
  return ConvertTo(Full256<int32_t>(), Round(v));
}

// ================================================== MISC

// ------------------------------ LoadMaskBits (TestBit)

// `p` points to at least 8 readable bytes, not all of which need be valid.
// TODO(janwas): special-case 64-bit, only 4 bits
template <typename T>
HWY_API Mask256<T> LoadMaskBits(Full256<T> d,
                                const uint8_t* HWY_RESTRICT bits) {
  uint64_t mask_bits = 0;
  CopyBytes<(N + 7) / 8>(bits, &mask_bits);
  return detail::LoadMaskBits(d, mask_bits);
}

// ------------------------------ Mask

// `p` points to at least 8 writable bytes.
template <typename T>
HWY_API size_t StoreMaskBits(const Full256<T> /* tag */, const Mask256<T> mask,
                             uint8_t* bits) {
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  const size_t kNumBytes = (N + 7) / 8;
  CopyBytes<kNumBytes>(&mask_bits, bits);
  return kNumBytes;
}

template <typename T>
HWY_API size_t CountTrue(const Full256<T> d, const Mask256<T> m) {
  const Half<decltype(d)> dh;
  return CountTrue(dh, m.m0) + CountTrue(dh, m.m1);
}

template <typename T>
HWY_API bool AllFalse(const Full256<T> d, const Mask128<T> m) {
  const Half<decltype(d)> dh;
  return AllFalse(dh, m.m0) && AllFalse(dh, m.m1);
}

template <typename T>
HWY_API bool AllTrue(const Full256<T> d, const Mask128<T> m) {
  const Half<decltype(d)> dh;
  return AllTrue(dh, m.m0) && AllTrue(dh, m.m1);
}

template <typename T>
HWY_API size_t FindKnownFirstTrue(const Full256<T> /* tag */,
                                  const Mask256<T> mask) {
  const uint64_t bits = detail::BitsFromMask(mask);
  return Num0BitsBelowLS1Bit_Nonzero64(bits);
}

template <typename T>
HWY_API intptr_t FindFirstTrue(const Full256<T> /* tag */,
                               const Mask256<T> mask) {
  const uint64_t bits = detail::BitsFromMask(mask);
  return bits ? Num0BitsBelowLS1Bit_Nonzero64(bits) : -1;
}

// ------------------------------ Compress

namespace detail {

template <typename T>
HWY_INLINE Vec256<T> Idx16x8FromBits(const uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 256);
  const Full256<T> d;
  const Rebind<uint8_t, decltype(d)> d8;
  const Full256<uint16_t> du;

  // We need byte indices for TableLookupBytes (one vector's worth for each of
  // 256 combinations of 8 mask bits). Loading them directly requires 4 KiB. We
  // can instead store lane indices and convert to byte indices (2*lane + 0..1),
  // with the doubling baked into the table. Unpacking nibbles is likely more
  // costly than the higher cache footprint from storing bytes.
  alignas(32) constexpr uint8_t table[256 * 8] = {
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,
      0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,
      0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  2,  4,  0,  0,  0,  0,
      0,  0,  0,  2,  4,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0,
      0,  6,  0,  0,  0,  0,  0,  0,  2,  6,  0,  0,  0,  0,  0,  0,  0,  2,
      6,  0,  0,  0,  0,  0,  4,  6,  0,  0,  0,  0,  0,  0,  0,  4,  6,  0,
      0,  0,  0,  0,  2,  4,  6,  0,  0,  0,  0,  0,  0,  2,  4,  6,  0,  0,
      0,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  0,  0,  0,  0,
      2,  8,  0,  0,  0,  0,  0,  0,  0,  2,  8,  0,  0,  0,  0,  0,  4,  8,
      0,  0,  0,  0,  0,  0,  0,  4,  8,  0,  0,  0,  0,  0,  2,  4,  8,  0,
      0,  0,  0,  0,  0,  2,  4,  8,  0,  0,  0,  0,  6,  8,  0,  0,  0,  0,
      0,  0,  0,  6,  8,  0,  0,  0,  0,  0,  2,  6,  8,  0,  0,  0,  0,  0,
      0,  2,  6,  8,  0,  0,  0,  0,  4,  6,  8,  0,  0,  0,  0,  0,  0,  4,
      6,  8,  0,  0,  0,  0,  2,  4,  6,  8,  0,  0,  0,  0,  0,  2,  4,  6,
      8,  0,  0,  0,  10, 0,  0,  0,  0,  0,  0,  0,  0,  10, 0,  0,  0,  0,
      0,  0,  2,  10, 0,  0,  0,  0,  0,  0,  0,  2,  10, 0,  0,  0,  0,  0,
      4,  10, 0,  0,  0,  0,  0,  0,  0,  4,  10, 0,  0,  0,  0,  0,  2,  4,
      10, 0,  0,  0,  0,  0,  0,  2,  4,  10, 0,  0,  0,  0,  6,  10, 0,  0,
      0,  0,  0,  0,  0,  6,  10, 0,  0,  0,  0,  0,  2,  6,  10, 0,  0,  0,
      0,  0,  0,  2,  6,  10, 0,  0,  0,  0,  4,  6,  10, 0,  0,  0,  0,  0,
      0,  4,  6,  10, 0,  0,  0,  0,  2,  4,  6,  10, 0,  0,  0,  0,  0,  2,
      4,  6,  10, 0,  0,  0,  8,  10, 0,  0,  0,  0,  0,  0,  0,  8,  10, 0,
      0,  0,  0,  0,  2,  8,  10, 0,  0,  0,  0,  0,  0,  2,  8,  10, 0,  0,
      0,  0,  4,  8,  10, 0,  0,  0,  0,  0,  0,  4,  8,  10, 0,  0,  0,  0,
      2,  4,  8,  10, 0,  0,  0,  0,  0,  2,  4,  8,  10, 0,  0,  0,  6,  8,
      10, 0,  0,  0,  0,  0,  0,  6,  8,  10, 0,  0,  0,  0,  2,  6,  8,  10,
      0,  0,  0,  0,  0,  2,  6,  8,  10, 0,  0,  0,  4,  6,  8,  10, 0,  0,
      0,  0,  0,  4,  6,  8,  10, 0,  0,  0,  2,  4,  6,  8,  10, 0,  0,  0,
      0,  2,  4,  6,  8,  10, 0,  0,  12, 0,  0,  0,  0,  0,  0,  0,  0,  12,
      0,  0,  0,  0,  0,  0,  2,  12, 0,  0,  0,  0,  0,  0,  0,  2,  12, 0,
      0,  0,  0,  0,  4,  12, 0,  0,  0,  0,  0,  0,  0,  4,  12, 0,  0,  0,
      0,  0,  2,  4,  12, 0,  0,  0,  0,  0,  0,  2,  4,  12, 0,  0,  0,  0,
      6,  12, 0,  0,  0,  0,  0,  0,  0,  6,  12, 0,  0,  0,  0,  0,  2,  6,
      12, 0,  0,  0,  0,  0,  0,  2,  6,  12, 0,  0,  0,  0,  4,  6,  12, 0,
      0,  0,  0,  0,  0,  4,  6,  12, 0,  0,  0,  0,  2,  4,  6,  12, 0,  0,
      0,  0,  0,  2,  4,  6,  12, 0,  0,  0,  8,  12, 0,  0,  0,  0,  0,  0,
      0,  8,  12, 0,  0,  0,  0,  0,  2,  8,  12, 0,  0,  0,  0,  0,  0,  2,
      8,  12, 0,  0,  0,  0,  4,  8,  12, 0,  0,  0,  0,  0,  0,  4,  8,  12,
      0,  0,  0,  0,  2,  4,  8,  12, 0,  0,  0,  0,  0,  2,  4,  8,  12, 0,
      0,  0,  6,  8,  12, 0,  0,  0,  0,  0,  0,  6,  8,  12, 0,  0,  0,  0,
      2,  6,  8,  12, 0,  0,  0,  0,  0,  2,  6,  8,  12, 0,  0,  0,  4,  6,
      8,  12, 0,  0,  0,  0,  0,  4,  6,  8,  12, 0,  0,  0,  2,  4,  6,  8,
      12, 0,  0,  0,  0,  2,  4,  6,  8,  12, 0,  0,  10, 12, 0,  0,  0,  0,
      0,  0,  0,  10, 12, 0,  0,  0,  0,  0,  2,  10, 12, 0,  0,  0,  0,  0,
      0,  2,  10, 12, 0,  0,  0,  0,  4,  10, 12, 0,  0,  0,  0,  0,  0,  4,
      10, 12, 0,  0,  0,  0,  2,  4,  10, 12, 0,  0,  0,  0,  0,  2,  4,  10,
      12, 0,  0,  0,  6,  10, 12, 0,  0,  0,  0,  0,  0,  6,  10, 12, 0,  0,
      0,  0,  2,  6,  10, 12, 0,  0,  0,  0,  0,  2,  6,  10, 12, 0,  0,  0,
      4,  6,  10, 12, 0,  0,  0,  0,  0,  4,  6,  10, 12, 0,  0,  0,  2,  4,
      6,  10, 12, 0,  0,  0,  0,  2,  4,  6,  10, 12, 0,  0,  8,  10, 12, 0,
      0,  0,  0,  0,  0,  8,  10, 12, 0,  0,  0,  0,  2,  8,  10, 12, 0,  0,
      0,  0,  0,  2,  8,  10, 12, 0,  0,  0,  4,  8,  10, 12, 0,  0,  0,  0,
      0,  4,  8,  10, 12, 0,  0,  0,  2,  4,  8,  10, 12, 0,  0,  0,  0,  2,
      4,  8,  10, 12, 0,  0,  6,  8,  10, 12, 0,  0,  0,  0,  0,  6,  8,  10,
      12, 0,  0,  0,  2,  6,  8,  10, 12, 0,  0,  0,  0,  2,  6,  8,  10, 12,
      0,  0,  4,  6,  8,  10, 12, 0,  0,  0,  0,  4,  6,  8,  10, 12, 0,  0,
      2,  4,  6,  8,  10, 12, 0,  0,  0,  2,  4,  6,  8,  10, 12, 0,  14, 0,
      0,  0,  0,  0,  0,  0,  0,  14, 0,  0,  0,  0,  0,  0,  2,  14, 0,  0,
      0,  0,  0,  0,  0,  2,  14, 0,  0,  0,  0,  0,  4,  14, 0,  0,  0,  0,
      0,  0,  0,  4,  14, 0,  0,  0,  0,  0,  2,  4,  14, 0,  0,  0,  0,  0,
      0,  2,  4,  14, 0,  0,  0,  0,  6,  14, 0,  0,  0,  0,  0,  0,  0,  6,
      14, 0,  0,  0,  0,  0,  2,  6,  14, 0,  0,  0,  0,  0,  0,  2,  6,  14,
      0,  0,  0,  0,  4,  6,  14, 0,  0,  0,  0,  0,  0,  4,  6,  14, 0,  0,
      0,  0,  2,  4,  6,  14, 0,  0,  0,  0,  0,  2,  4,  6,  14, 0,  0,  0,
      8,  14, 0,  0,  0,  0,  0,  0,  0,  8,  14, 0,  0,  0,  0,  0,  2,  8,
      14, 0,  0,  0,  0,  0,  0,  2,  8,  14, 0,  0,  0,  0,  4,  8,  14, 0,
      0,  0,  0,  0,  0,  4,  8,  14, 0,  0,  0,  0,  2,  4,  8,  14, 0,  0,
      0,  0,  0,  2,  4,  8,  14, 0,  0,  0,  6,  8,  14, 0,  0,  0,  0,  0,
      0,  6,  8,  14, 0,  0,  0,  0,  2,  6,  8,  14, 0,  0,  0,  0,  0,  2,
      6,  8,  14, 0,  0,  0,  4,  6,  8,  14, 0,  0,  0,  0,  0,  4,  6,  8,
      14, 0,  0,  0,  2,  4,  6,  8,  14, 0,  0,  0,  0,  2,  4,  6,  8,  14,
      0,  0,  10, 14, 0,  0,  0,  0,  0,  0,  0,  10, 14, 0,  0,  0,  0,  0,
      2,  10, 14, 0,  0,  0,  0,  0,  0,  2,  10, 14, 0,  0,  0,  0,  4,  10,
      14, 0,  0,  0,  0,  0,  0,  4,  10, 14, 0,  0,  0,  0,  2,  4,  10, 14,
      0,  0,  0,  0,  0,  2,  4,  10, 14, 0,  0,  0,  6,  10, 14, 0,  0,  0,
      0,  0,  0,  6,  10, 14, 0,  0,  0,  0,  2,  6,  10, 14, 0,  0,  0,  0,
      0,  2,  6,  10, 14, 0,  0,  0,  4,  6,  10, 14, 0,  0,  0,  0,  0,  4,
      6,  10, 14, 0,  0,  0,  2,  4,  6,  10, 14, 0,  0,  0,  0,  2,  4,  6,
      10, 14, 0,  0,  8,  10, 14, 0,  0,  0,  0,  0,  0,  8,  10, 14, 0,  0,
      0,  0,  2,  8,  10, 14, 0,  0,  0,  0,  0,  2,  8,  10, 14, 0,  0,  0,
      4,  8,  10, 14, 0,  0,  0,  0,  0,  4,  8,  10, 14, 0,  0,  0,  2,  4,
      8,  10, 14, 0,  0,  0,  0,  2,  4,  8,  10, 14, 0,  0,  6,  8,  10, 14,
      0,  0,  0,  0,  0,  6,  8,  10, 14, 0,  0,  0,  2,  6,  8,  10, 14, 0,
      0,  0,  0,  2,  6,  8,  10, 14, 0,  0,  4,  6,  8,  10, 14, 0,  0,  0,
      0,  4,  6,  8,  10, 14, 0,  0,  2,  4,  6,  8,  10, 14, 0,  0,  0,  2,
      4,  6,  8,  10, 14, 0,  12, 14, 0,  0,  0,  0,  0,  0,  0,  12, 14, 0,
      0,  0,  0,  0,  2,  12, 14, 0,  0,  0,  0,  0,  0,  2,  12, 14, 0,  0,
      0,  0,  4,  12, 14, 0,  0,  0,  0,  0,  0,  4,  12, 14, 0,  0,  0,  0,
      2,  4,  12, 14, 0,  0,  0,  0,  0,  2,  4,  12, 14, 0,  0,  0,  6,  12,
      14, 0,  0,  0,  0,  0,  0,  6,  12, 14, 0,  0,  0,  0,  2,  6,  12, 14,
      0,  0,  0,  0,  0,  2,  6,  12, 14, 0,  0,  0,  4,  6,  12, 14, 0,  0,
      0,  0,  0,  4,  6,  12, 14, 0,  0,  0,  2,  4,  6,  12, 14, 0,  0,  0,
      0,  2,  4,  6,  12, 14, 0,  0,  8,  12, 14, 0,  0,  0,  0,  0,  0,  8,
      12, 14, 0,  0,  0,  0,  2,  8,  12, 14, 0,  0,  0,  0,  0,  2,  8,  12,
      14, 0,  0,  0,  4,  8,  12, 14, 0,  0,  0,  0,  0,  4,  8,  12, 14, 0,
      0,  0,  2,  4,  8,  12, 14, 0,  0,  0,  0,  2,  4,  8,  12, 14, 0,  0,
      6,  8,  12, 14, 0,  0,  0,  0,  0,  6,  8,  12, 14, 0,  0,  0,  2,  6,
      8,  12, 14, 0,  0,  0,  0,  2,  6,  8,  12, 14, 0,  0,  4,  6,  8,  12,
      14, 0,  0,  0,  0,  4,  6,  8,  12, 14, 0,  0,  2,  4,  6,  8,  12, 14,
      0,  0,  0,  2,  4,  6,  8,  12, 14, 0,  10, 12, 14, 0,  0,  0,  0,  0,
      0,  10, 12, 14, 0,  0,  0,  0,  2,  10, 12, 14, 0,  0,  0,  0,  0,  2,
      10, 12, 14, 0,  0,  0,  4,  10, 12, 14, 0,  0,  0,  0,  0,  4,  10, 12,
      14, 0,  0,  0,  2,  4,  10, 12, 14, 0,  0,  0,  0,  2,  4,  10, 12, 14,
      0,  0,  6,  10, 12, 14, 0,  0,  0,  0,  0,  6,  10, 12, 14, 0,  0,  0,
      2,  6,  10, 12, 14, 0,  0,  0,  0,  2,  6,  10, 12, 14, 0,  0,  4,  6,
      10, 12, 14, 0,  0,  0,  0,  4,  6,  10, 12, 14, 0,  0,  2,  4,  6,  10,
      12, 14, 0,  0,  0,  2,  4,  6,  10, 12, 14, 0,  8,  10, 12, 14, 0,  0,
      0,  0,  0,  8,  10, 12, 14, 0,  0,  0,  2,  8,  10, 12, 14, 0,  0,  0,
      0,  2,  8,  10, 12, 14, 0,  0,  4,  8,  10, 12, 14, 0,  0,  0,  0,  4,
      8,  10, 12, 14, 0,  0,  2,  4,  8,  10, 12, 14, 0,  0,  0,  2,  4,  8,
      10, 12, 14, 0,  6,  8,  10, 12, 14, 0,  0,  0,  0,  6,  8,  10, 12, 14,
      0,  0,  2,  6,  8,  10, 12, 14, 0,  0,  0,  2,  6,  8,  10, 12, 14, 0,
      4,  6,  8,  10, 12, 14, 0,  0,  0,  4,  6,  8,  10, 12, 14, 0,  2,  4,
      6,  8,  10, 12, 14, 0,  0,  2,  4,  6,  8,  10, 12, 14};

  const Vec256<uint8_t> byte_idx{Load(d8, table + mask_bits * 8).raw};
  const Vec256<uint16_t> pairs = ZipLower(byte_idx, byte_idx);
  return BitCast(d, pairs + Set(du, 0x0100));
}

template <typename T>
HWY_INLINE Vec256<T> Idx32x4FromBits(const uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 16);

  // There are only 4 lanes, so we can afford to load the index vector directly.
  alignas(32) constexpr uint8_t packed_array[16 * 16] = {
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

  const Full256<T> d;
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, packed_array + 16 * mask_bits));
}

#if HWY_HAVE_INTEGER64 || HWY_HAVE_FLOAT64

template <typename T>
HWY_INLINE Vec256<T> Idx64x2FromBits(const uint64_t mask_bits) {
  HWY_DASSERT(mask_bits < 4);

  // There are only 2 lanes, so we can afford to load the index vector directly.
  alignas(32) constexpr uint8_t packed_array[4 * 16] = {
      0, 1, 2,  3,  4,  5,  6,  7,  0, 1, 2,  3,  4,  5,  6,  7,  //
      0, 1, 2,  3,  4,  5,  6,  7,  0, 1, 2,  3,  4,  5,  6,  7,  //
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2,  3,  4,  5,  6,  7,  //
      0, 1, 2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15};

  const Full256<T> d;
  const Repartition<uint8_t, decltype(d)> d8;
  return BitCast(d, Load(d8, packed_array + 16 * mask_bits));
}

#endif

// Helper functions called by both Compress and CompressStore - avoids a
// redundant BitsFromMask in the latter.

template <typename T>
HWY_INLINE Vec256<T> Compress(hwy::SizeTag<2> /*tag*/, Vec256<T> v,
                              const uint64_t mask_bits) {
  const auto idx = detail::Idx16x8FromBits<T>(mask_bits);
  using D = Full256<T>;
  const RebindToSigned<D> di;
  return BitCast(D(), TableLookupBytes(BitCast(di, v), BitCast(di, idx)));
}

template <typename T>
HWY_INLINE Vec256<T> Compress(hwy::SizeTag<4> /*tag*/, Vec256<T> v,
                              const uint64_t mask_bits) {
  const auto idx = detail::Idx32x4FromBits<T>(mask_bits);
  using D = Full256<T>;
  const RebindToSigned<D> di;
  return BitCast(D(), TableLookupBytes(BitCast(di, v), BitCast(di, idx)));
}

#if HWY_HAVE_INTEGER64 || HWY_HAVE_FLOAT64

template <typename T>
HWY_INLINE Vec256<uint64_t> Compress(hwy::SizeTag<8> /*tag*/,
                                     Vec256<uint64_t> v,
                                     const uint64_t mask_bits) {
  const auto idx = detail::Idx64x2FromBits<uint64_t>(mask_bits);
  using D = Full256<T>;
  const RebindToSigned<D> di;
  return BitCast(D(), TableLookupBytes(BitCast(di, v), BitCast(di, idx)));
}

#endif

}  // namespace detail

template <typename T>
struct CompressIsPartition {
  enum { value = 1 };
};

template <typename T>
HWY_API Vec256<T> Compress(Vec256<T> v, const Mask256<T> mask) {
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  return detail::Compress(hwy::SizeTag<sizeof(T)>(), v, mask_bits);
}

// ------------------------------ CompressNot
template <typename T>
HWY_API Vec256<T> Compress(Vec256<T> v, const Mask256<T> mask) {
  return Compress(v, Not(mask));
}

// ------------------------------ CompressBlocksNot
HWY_API Vec256<uint64_t> CompressBlocksNot(Vec256<uint64_t> v,
                                           Mask256<uint64_t> mask) {
  HWY_ASSERT(0);  // Not implemented
}

// ------------------------------ CompressBits

template <typename T>
HWY_API Vec256<T> CompressBits(Vec256<T> v, const uint8_t* HWY_RESTRICT bits) {
  uint64_t mask_bits = 0;
  constexpr size_t kNumBytes = (N + 7) / 8;
  CopyBytes<kNumBytes>(bits, &mask_bits);
  if (N < 8) {
    mask_bits &= (1ull << N) - 1;
  }

  return detail::Compress(hwy::SizeTag<sizeof(T)>(), v, mask_bits);
}

// ------------------------------ CompressStore
template <typename T>
HWY_API size_t CompressStore(Vec256<T> v, const Mask256<T> mask, Full256<T> d,
                             T* HWY_RESTRICT unaligned) {
  const uint64_t mask_bits = detail::BitsFromMask(mask);
  const auto c = detail::Compress(hwy::SizeTag<sizeof(T)>(), v, mask_bits);
  StoreU(c, d, unaligned);
  return PopCount(mask_bits);
}

// ------------------------------ CompressBlendedStore
template <typename T>
HWY_API size_t CompressBlendedStore(Vec256<T> v, Mask256<T> m, Full256<T> d,
                                    T* HWY_RESTRICT unaligned) {
  const RebindToUnsigned<decltype(d)> du;  // so we can support fp16/bf16
  using TU = TFromD<decltype(du)>;
  const uint64_t mask_bits = detail::BitsFromMask(m);
  const size_t count = PopCount(mask_bits);
  const Mask256<TU> store_mask = FirstN(du, count);
  const Vec256<TU> compressed =
      detail::Compress(hwy::SizeTag<sizeof(T)>(), BitCast(du, v), mask_bits);
  const Vec256<TU> prev = BitCast(du, LoadU(d, unaligned));
  StoreU(BitCast(d, IfThenElse(store_mask, compressed, prev)), d, unaligned);
  return count;
}

// ------------------------------ CompressBitsStore

template <typename T>
HWY_API size_t CompressBitsStore(Vec256<T> v, const uint8_t* HWY_RESTRICT bits,
                                 Full256<T> d, T* HWY_RESTRICT unaligned) {
  uint64_t mask_bits = 0;
  constexpr size_t kNumBytes = (N + 7) / 8;
  CopyBytes<kNumBytes>(bits, &mask_bits);
  if (N < 8) {
    mask_bits &= (1ull << N) - 1;
  }

  const auto c = detail::Compress(hwy::SizeTag<sizeof(T)>(), v, mask_bits);
  StoreU(c, d, unaligned);
  return PopCount(mask_bits);
}

// ------------------------------ StoreInterleaved2/3/4

// HWY_NATIVE_LOAD_STORE_INTERLEAVED not set, hence defined in
// generic_ops-inl.h.

// ------------------------------ ReorderWidenMulAccumulate
template <typename TN, typename TW>
HWY_API Vec256<TW> ReorderWidenMulAccumulate(Full256<TW> d, Vec256<TN> a,
                                             Vec256<TN> b, Vec256<TW> sum0,
                                             Vec256<TW>& sum1) {
  const Half<decltype(d)> dh;
  sum0.v0 = ReorderWidenMulAccumulate(dh, a.v0, b.v0, sum0.v0, sum1.v0);
  sum0.v1 = ReorderWidenMulAccumulate(dh, a.v1, b.v1, sum0.v1, sum1.v1);
  return sum0;
}

// ------------------------------ Reductions

template <typename T>
HWY_API Vec256<T> SumOfLanes(Full256<T> /* tag */, const Vec256<T> v) {
  const Half<decltype(d)> dh;
  return SumOfLanes(dh, Add(v.v0, v.v1));
}

template <typename T>
HWY_API Vec256<T> MinOfLanes(Full256<T> /* tag */, const Vec256<T> v) {
  const Half<decltype(d)> dh;
  return MinOfLanes(dh, Min(v.v0, v.v1));
}

template <typename T>
HWY_API Vec256<T> MaxOfLanes(Full256<T> /* tag */, const Vec256<T> v) {
  const Half<decltype(d)> dh;
  return MaxOfLanes(dh, Max(v.v0, v.v1));
}

// ------------------------------ Lt128

template <typename T>
HWY_INLINE Mask256<T> Lt128(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Lt128(dh, a.v0, b.v0);
  ret.m1 = Lt128(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Mask256<T> Lt128Upper(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Lt128Upper(dh, a.v0, b.v0);
  ret.m1 = Lt128Upper(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Mask256<T> Eq128(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Eq128(dh, a.v0, b.v0);
  ret.m1 = Eq128(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Mask256<T> Eq128Upper(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Eq128Upper(dh, a.v0, b.v0);
  ret.m1 = Eq128Upper(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Mask256<T> Ne128(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Ne128(dh, a.v0, b.v0);
  ret.m1 = Ne128(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Mask256<T> Ne128Upper(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Ne128Upper(dh, a.v0, b.v0);
  ret.m1 = Ne128Upper(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Vec256<T> Min128(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Min128(dh, a.v0, b.v0);
  ret.m1 = Min128(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Vec256<T> Max128(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Max128(dh, a.v0, b.v0);
  ret.m1 = Max128(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Vec256<T> Min128Upper(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Min128Upper(dh, a.v0, b.v0);
  ret.m1 = Min128Upper(dh, a.v1, b.v1);
  return ret;
}

template <typename T>
HWY_INLINE Vec256<T> Max128Upper(Full256<T> d, Vec256<T> a, Vec256<T> b) {
  const Half<decltype(d)> dh;
  Mask256<T> ret;
  ret.m0 = Max128Upper(dh, a.v0, b.v0);
  ret.m1 = Max128Upper(dh, a.v1, b.v1);
  return ret;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
