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

#ifndef HWY_SCALAR_H_
#define HWY_SCALAR_H_

// Single-element vectors and operations.

#include "hwy/compiler_specific.h"
#include "hwy/shared.h"

namespace hwy {

// Shorthand for a scalar; note that Vec0<T> is the actual data class.
template <typename T>
using Scalar = Desc<T, 0>;

// (Wrapper class required for overloading comparison operators.)
template <typename T>
struct Vec0 {
  HWY_INLINE Vec0() = default;
  Vec0(const Vec0&) = default;
  Vec0& operator=(const Vec0&) = default;
  HWY_INLINE explicit Vec0(const T t) : raw(t) {}

  HWY_INLINE Vec0& operator*=(const Vec0 other) {
    return *this = (*this * other);
  }
  HWY_INLINE Vec0& operator/=(const Vec0 other) {
    return *this = (*this / other);
  }
  HWY_INLINE Vec0& operator+=(const Vec0 other) {
    return *this = (*this + other);
  }
  HWY_INLINE Vec0& operator-=(const Vec0 other) {
    return *this = (*this - other);
  }
  HWY_INLINE Vec0& operator&=(const Vec0 other) {
    return *this = (*this & other);
  }
  HWY_INLINE Vec0& operator|=(const Vec0 other) {
    return *this = (*this | other);
  }
  HWY_INLINE Vec0& operator^=(const Vec0 other) {
    return *this = (*this ^ other);
  }

  T raw;
};

// The unsigned integer type whose size is kSize bytes.
template <size_t kSize>
struct MakeUnsignedT;
template <>
struct MakeUnsignedT<1> {
  using type = uint8_t;
};
template <>
struct MakeUnsignedT<2> {
  using type = uint16_t;
};
template <>
struct MakeUnsignedT<4> {
  using type = uint32_t;
};
template <>
struct MakeUnsignedT<8> {
  using type = uint64_t;
};

template <typename T>
using MakeUnsigned = typename MakeUnsignedT<sizeof(T)>::type;

// 0 or FF..FF, same size as Vec0.
template <typename T>
class Mask0 {
  using Raw = MakeUnsigned<T>;

 public:
  static HWY_INLINE Mask0<T> FromBool(bool b) {
    Mask0<T> mask;
    mask.bits = b ? ~Raw(0) : 0;
    return mask;
  }

  Raw bits;
};

// ------------------------------ Cast

template <typename T, typename FromT>
HWY_INLINE Vec0<T> BitCast(Scalar<T> /* tag */, Vec0<FromT> v) {
  static_assert(sizeof(T) <= sizeof(FromT), "Promoting is undefined");
  T to;
  CopyBytes<sizeof(FromT)>(&v.raw, &to);
  return Vec0<T>(to);
}

// ------------------------------ Set

template <typename T>
HWY_INLINE Vec0<T> Zero(Scalar<T> /* tag */) {
  return Vec0<T>(T(0));
}

template <typename T, typename T2>
HWY_INLINE Vec0<T> Set(Scalar<T> /* tag */, const T2 t) {
  return Vec0<T>(t);
}

template <typename T, typename T2>
HWY_INLINE Vec0<T> Iota(Scalar<T> /* tag */, const T2 first) {
  return Vec0<T>(first);
}

template <typename T>
HWY_INLINE Vec0<T> Undefined(Scalar<T> /* tag */) {
  return Vec0<T>(0);
}

// ================================================== SHIFTS

// ------------------------------ Shift lanes by constant #bits

template <int kBits, typename T>
HWY_INLINE Vec0<T> ShiftLeft(const Vec0<T> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return Vec0<T>(static_cast<MakeUnsigned<T>>(v.raw) << kBits);
}

template <int kBits, typename T>
HWY_INLINE Vec0<T> ShiftRight(const Vec0<T> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return Vec0<T>(v.raw >> kBits);
}

// ------------------------------ Shift lanes by independent variable #bits

// Single-lane => same as above except for the argument type.
template <typename T>
HWY_INLINE Vec0<T> operator<<(const Vec0<T> v, const Vec0<T> bits) {
  return Vec0<T>(static_cast<MakeUnsigned<T>>(v.raw) << bits.raw);
}
template <typename T>
HWY_INLINE Vec0<T> operator>>(const Vec0<T> v, const Vec0<T> bits) {
  return Vec0<T>(v.raw >> bits.raw);
}

// ================================================== LOGICAL

template <typename Bits>
struct BitwiseOp {
  template <typename T, class Op>
  Vec0<T> operator()(const Vec0<T> a, const Vec0<T> b, const Op& op) const {
    static_assert(sizeof(T) == sizeof(Bits), "Float/int size mismatch");
    Bits ia, ib;
    CopyBytes<sizeof(Bits)>(&a, &ia);
    CopyBytes<sizeof(Bits)>(&b, &ib);
    ia = op(ia, ib);
    T ret;
    CopyBytes<sizeof(Bits)>(&ia, &ret);
    return Vec0<T>(ret);
  }
};

// ------------------------------ Bitwise AND

template <typename T>
HWY_INLINE Vec0<T> operator&(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(a.raw & b.raw);
}
template <>
HWY_INLINE Vec0<float> operator&(const Vec0<float> a, const Vec0<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i & j; });
}
template <>
HWY_INLINE Vec0<double> operator&(const Vec0<double> a, const Vec0<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i & j; });
}

// ------------------------------ Bitwise AND-NOT

// Returns ~a & b.
template <typename T>
HWY_INLINE Vec0<T> AndNot(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(~a.raw & b.raw);
}
template <>
HWY_INLINE Vec0<float> AndNot(const Vec0<float> a, const Vec0<float> b) {
  return BitwiseOp<int32_t>()(a, b,
                              [](int32_t i, int32_t j) { return ~i & j; });
}
template <>
HWY_INLINE Vec0<double> AndNot(const Vec0<double> a, const Vec0<double> b) {
  return BitwiseOp<int64_t>()(a, b,
                              [](int64_t i, int64_t j) { return ~i & j; });
}

// ------------------------------ Bitwise OR

template <typename T>
HWY_INLINE Vec0<T> operator|(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(a.raw | b.raw);
}
template <>
HWY_INLINE Vec0<float> operator|(const Vec0<float> a, const Vec0<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i | j; });
}
template <>
HWY_INLINE Vec0<double> operator|(const Vec0<double> a, const Vec0<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i | j; });
}

// ------------------------------ Bitwise XOR

template <typename T>
HWY_INLINE Vec0<T> operator^(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(a.raw ^ b.raw);
}
template <>
HWY_INLINE Vec0<float> operator^(const Vec0<float> a, const Vec0<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i ^ j; });
}
template <>
HWY_INLINE Vec0<double> operator^(const Vec0<double> a, const Vec0<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i ^ j; });
}

// ------------------------------ Mask

// v must be 0 or FF..FF.
template <typename T>
HWY_INLINE Mask0<T> MaskFromVec(const Vec0<T> v) {
  Mask0<T> mask;
  memcpy(&mask.bits, &v.raw, sizeof(mask.bits));
  return mask;
}

template <typename T>
Vec0<T> VecFromMask(const Mask0<T> mask) {
  Vec0<T> v;
  memcpy(&v.raw, &mask.bits, sizeof(v.raw));
  return v;
}

// Returns mask ? yes : no.
template <typename T>
HWY_INLINE Vec0<T> IfThenElse(const Mask0<T> mask, const Vec0<T> yes,
                              const Vec0<T> no) {
  return mask.bits ? yes : no;
}

template <typename T>
HWY_INLINE Vec0<T> IfThenElseZero(const Mask0<T> mask, const Vec0<T> yes) {
  return mask.bits ? yes : Vec0<T>(0);
}

template <typename T>
HWY_INLINE Vec0<T> IfThenZeroElse(const Mask0<T> mask, const Vec0<T> no) {
  return mask.bits ? Vec0<T>(0) : no;
}

template <typename T>
HWY_INLINE Vec0<T> ZeroIfNegative(const Vec0<T> v) {
  return v.raw < 0 ? Vec0<T>(0) : v;
}

// ================================================== ARITHMETIC

template <typename T>
HWY_INLINE Vec0<T> operator+(Vec0<T> a, Vec0<T> b) {
  const uint64_t a64 = static_cast<int64_t>(a.raw);
  const uint64_t b64 = static_cast<int64_t>(b.raw);
  return Vec0<T>((a64 + b64) & ~T(0));
}
HWY_INLINE Vec0<float> operator+(const Vec0<float> a, const Vec0<float> b) {
  return Vec0<float>(a.raw + b.raw);
}
HWY_INLINE Vec0<double> operator+(const Vec0<double> a, const Vec0<double> b) {
  return Vec0<double>(a.raw + b.raw);
}

template <typename T>
HWY_INLINE Vec0<T> operator-(Vec0<T> a, Vec0<T> b) {
  const uint64_t a64 = static_cast<int64_t>(a.raw);
  const uint64_t b64 = static_cast<int64_t>(b.raw);
  return Vec0<T>((a64 - b64) & ~T(0));
}
HWY_INLINE Vec0<float> operator-(const Vec0<float> a, const Vec0<float> b) {
  return Vec0<float>(a.raw - b.raw);
}
HWY_INLINE Vec0<double> operator-(const Vec0<double> a, const Vec0<double> b) {
  return Vec0<double>(a.raw - b.raw);
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
HWY_INLINE Vec0<uint8_t> SaturatedAdd(const Vec0<uint8_t> a,
                                      const Vec0<uint8_t> b) {
  return Vec0<uint8_t>(HWY_MIN(HWY_MAX(0, a.raw + b.raw), 255));
}
HWY_INLINE Vec0<uint16_t> SaturatedAdd(const Vec0<uint16_t> a,
                                       const Vec0<uint16_t> b) {
  return Vec0<uint16_t>(HWY_MIN(HWY_MAX(0, a.raw + b.raw), 65535));
}

// Signed
HWY_INLINE Vec0<int8_t> SaturatedAdd(const Vec0<int8_t> a,
                                     const Vec0<int8_t> b) {
  return Vec0<int8_t>(HWY_MIN(HWY_MAX(-128, a.raw + b.raw), 127));
}
HWY_INLINE Vec0<int16_t> SaturatedAdd(const Vec0<int16_t> a,
                                      const Vec0<int16_t> b) {
  return Vec0<int16_t>(HWY_MIN(HWY_MAX(-32768, a.raw + b.raw), 32767));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
HWY_INLINE Vec0<uint8_t> SaturatedSub(const Vec0<uint8_t> a,
                                      const Vec0<uint8_t> b) {
  return Vec0<uint8_t>(HWY_MIN(HWY_MAX(0, a.raw - b.raw), 255));
}
HWY_INLINE Vec0<uint16_t> SaturatedSub(const Vec0<uint16_t> a,
                                       const Vec0<uint16_t> b) {
  return Vec0<uint16_t>(HWY_MIN(HWY_MAX(0, a.raw - b.raw), 65535));
}

// Signed
HWY_INLINE Vec0<int8_t> SaturatedSub(const Vec0<int8_t> a,
                                     const Vec0<int8_t> b) {
  return Vec0<int8_t>(HWY_MIN(HWY_MAX(-128, a.raw - b.raw), 127));
}
HWY_INLINE Vec0<int16_t> SaturatedSub(const Vec0<int16_t> a,
                                      const Vec0<int16_t> b) {
  return Vec0<int16_t>(HWY_MIN(HWY_MAX(-32768, a.raw - b.raw), 32767));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

HWY_INLINE Vec0<uint8_t> AverageRound(const Vec0<uint8_t> a,
                                      const Vec0<uint8_t> b) {
  return Vec0<uint8_t>((a.raw + b.raw + 1) / 2);
}
HWY_INLINE Vec0<uint16_t> AverageRound(const Vec0<uint16_t> a,
                                       const Vec0<uint16_t> b) {
  return Vec0<uint16_t>((a.raw + b.raw + 1) / 2);
}

// ------------------------------ Absolute value

template <typename T>
HWY_INLINE Vec0<T> Abs(const Vec0<T> a) {
  const T i = a.raw;
  return (i >= 0 || i == LimitsMin<T>()) ? a : Vec0<T>(-i);
}

// ------------------------------ min/max

template <typename T>
HWY_INLINE Vec0<T> Min(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(HWY_MIN(a.raw, b.raw));
}

template <typename T>
HWY_INLINE Vec0<T> Max(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(HWY_MAX(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T>
HWY_INLINE Vec0<T> Clamp(const Vec0<T> v, const Vec0<T> lo, const Vec0<T> hi) {
  return Min(Max(lo, v), hi);
}

// ------------------------------ Floating-point negate

HWY_INLINE Vec0<float> Neg(const Vec0<float> v) {
  const Scalar<float> df;
  const Scalar<uint32_t> du;
  const auto sign = BitCast(df, Set(du, 0x80000000u));
  return v ^ sign;
}

HWY_INLINE Vec0<double> Neg(const Vec0<double> v) {
  const Scalar<double> df;
  const Scalar<uint64_t> du;
  const auto sign = BitCast(df, Set(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ mul/div

template <typename T>
HWY_INLINE Vec0<T> operator*(const Vec0<T> a, const Vec0<T> b) {
  if (IsFloat<T>()) {
    return Vec0<T>(static_cast<T>(double(a.raw) * b.raw));
  } else if (IsSigned<T>()) {
    return Vec0<T>(static_cast<T>(int64_t(a.raw) * b.raw));
  } else {
    return Vec0<T>(static_cast<T>(uint64_t(a.raw) * b.raw));
  }
}

template <typename T>
HWY_INLINE Vec0<T> operator/(const Vec0<T> a, const Vec0<T> b) {
  return Vec0<T>(a.raw / b.raw);
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
HWY_INLINE Vec0<int16_t> MulHigh(const Vec0<int16_t> a, const Vec0<int16_t> b) {
  return Vec0<int16_t>((a.raw * b.raw) >> 16);
}
HWY_INLINE Vec0<uint16_t> MulHigh(const Vec0<uint16_t> a,
                                  const Vec0<uint16_t> b) {
  // Cast to uint32_t first to prevent overflow. Otherwise the result of
  // uint16_t * uint16_t is in "int" which may overflow. In practice the result
  // is the same but this way it is also defined.
  return Vec0<uint16_t>(
      (static_cast<uint32_t>(a.raw) * static_cast<uint32_t>(b.raw)) >> 16);
}

}  // namespace ext

// Multiplies even lanes (0, 2 ..) and returns the double-wide result.
HWY_INLINE Vec0<int64_t> MulEven(const Vec0<int32_t> a, const Vec0<int32_t> b) {
  const int64_t a64 = a.raw;
  return Vec0<int64_t>(a64 * b.raw);
}
HWY_INLINE Vec0<uint64_t> MulEven(const Vec0<uint32_t> a,
                                  const Vec0<uint32_t> b) {
  const uint64_t a64 = a.raw;
  return Vec0<uint64_t>(a64 * b.raw);
}

// Approximate reciprocal
HWY_INLINE Vec0<float> ApproximateReciprocal(const Vec0<float> v) {
  return Vec0<float>(1.0f / v.raw);
}

// ------------------------------ Floating-point multiply-add variants

template <typename T>
HWY_INLINE Vec0<T> MulAdd(const Vec0<T> mul, const Vec0<T> x,
                          const Vec0<T> add) {
  return mul * x + add;
}

template <typename T>
HWY_INLINE Vec0<T> NegMulAdd(const Vec0<T> mul, const Vec0<T> x,
                             const Vec0<T> add) {
  return add - mul * x;
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

template <typename T>
HWY_INLINE Vec0<T> MulSub(const Vec0<T> mul, const Vec0<T> x,
                          const Vec0<T> sub) {
  return mul * x - sub;
}

template <typename T>
HWY_INLINE Vec0<T> NegMulSub(const Vec0<T> mul, const Vec0<T> x,
                             const Vec0<T> sub) {
  return Neg(mul) * x - sub;
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
HWY_INLINE Vec0<float> ApproximateReciprocalSqrt(const Vec0<float> v) {
  float f = v.raw;
  const float half = f * 0.5f;
  uint32_t bits;
  CopyBytes<4>(&f, &bits);
  // Initial guess based on log2(f)
  bits = 0x5F3759DF - (bits >> 1);
  CopyBytes<4>(&bits, &f);
  // One Newton-Raphson iteration
  return Vec0<float>(f * (1.5f - (half * f * f)));
}

// Square root
HWY_INLINE Vec0<float> Sqrt(const Vec0<float> v) {
  return ApproximateReciprocalSqrt(v) * v;
}
HWY_INLINE Vec0<double> Sqrt(const Vec0<double> v) {
  return Vec0<double>(Sqrt(Vec0<float>(v.raw)).raw);
}

// ------------------------------ Floating-point rounding

// Approximation of round-to-nearest for numbers representable as integers.
HWY_INLINE Vec0<float> Round(const Vec0<float> v) {
  const float bias = v.raw < 0.0f ? -0.5f : 0.5f;
  return Vec0<float>(static_cast<int32_t>(v.raw + bias));
}
HWY_INLINE Vec0<double> Round(const Vec0<double> v) {
  const double bias = v.raw < 0.0 ? -0.5 : 0.5;
  return Vec0<double>(static_cast<int64_t>(v.raw + bias));
}

HWY_INLINE Vec0<float> Trunc(const Vec0<float> v) {
  return Vec0<float>(static_cast<int32_t>(v.raw));
}
HWY_INLINE Vec0<double> Trunc(const Vec0<double> v) {
  return Vec0<double>(static_cast<int64_t>(v.raw));
}

template <typename Float, typename Bits, int kMantissaBits, int kExponentBits,
          class V>
V Ceiling(const V v) {
  const Bits kExponentMask = (1ull << kExponentBits) - 1;
  const Bits kMantissaMask = (1ull << kMantissaBits) - 1;
  const Bits kBias = kExponentMask / 2;

  Float f = v.raw;
  const bool positive = f > 0.0f;

  Bits bits;
  CopyBytes<sizeof(Bits)>(&v, &bits);

  const int exponent = ((bits >> kMantissaBits) & kExponentMask) - kBias;
  // Already an integer.
  if (exponent >= kMantissaBits) return v;
  // |v| <= 1 => 0 or 1.
  if (exponent < 0) return V(positive);

  const Bits mantissa_mask = kMantissaMask >> exponent;
  // Already an integer
  if ((bits & mantissa_mask) == 0) return v;

  // Clear fractional bits and round up
  if (positive) bits += (kMantissaMask + 1) >> exponent;
  bits &= ~mantissa_mask;

  CopyBytes<sizeof(Bits)>(&bits, &f);
  return V(f);
}

template <typename Float, typename Bits, int kMantissaBits, int kExponentBits,
          class V>
V Floor(const V v) {
  const Bits kExponentMask = (1ull << kExponentBits) - 1;
  const Bits kMantissaMask = (1ull << kMantissaBits) - 1;
  const Bits kBias = kExponentMask / 2;

  Float f = v.raw;
  const bool negative = f < 0.0f;

  Bits bits;
  CopyBytes<sizeof(Bits)>(&v, &bits);

  const int exponent = ((bits >> kMantissaBits) & kExponentMask) - kBias;
  // Already an integer.
  if (exponent >= kMantissaBits) return v;
  // |v| <= 1 => -1 or 0.
  if (exponent < 0) return V(negative ? -1.0 : 0.0f);

  const Bits mantissa_mask = kMantissaMask >> exponent;
  // Already an integer
  if ((bits & mantissa_mask) == 0) return v;

  // Clear fractional bits and round down
  if (negative) bits += (kMantissaMask + 1) >> exponent;
  bits &= ~mantissa_mask;

  CopyBytes<sizeof(Bits)>(&bits, &f);
  return V(f);
}

// Toward +infinity, aka ceiling
HWY_INLINE Vec0<float> Ceil(const Vec0<float> v) {
  return Ceiling<float, uint32_t, 23, 8>(v);
}
HWY_INLINE Vec0<double> Ceil(const Vec0<double> v) {
  return Ceiling<double, uint64_t, 52, 11>(v);
}

// Toward -infinity, aka floor
HWY_INLINE Vec0<float> Floor(const Vec0<float> v) {
  return Floor<float, uint32_t, 23, 8>(v);
}
HWY_INLINE Vec0<double> Floor(const Vec0<double> v) {
  return Floor<double, uint64_t, 52, 11>(v);
}

// ================================================== COMPARE

template <typename T>
HWY_INLINE Mask0<T> operator==(const Vec0<T> a, const Vec0<T> b) {
  return Mask0<T>::FromBool(a.raw == b.raw);
}

template <typename T>
HWY_INLINE Mask0<T> TestBit(const Vec0<T> v, const Vec0<T> bit) {
  static_assert(!IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

template <typename T>
HWY_INLINE Mask0<T> operator<(const Vec0<T> a, const Vec0<T> b) {
  return Mask0<T>::FromBool(a.raw < b.raw);
}
template <typename T>
HWY_INLINE Mask0<T> operator>(const Vec0<T> a, const Vec0<T> b) {
  return Mask0<T>::FromBool(a.raw > b.raw);
}

template <typename T>
HWY_INLINE Mask0<T> operator<=(const Vec0<T> a, const Vec0<T> b) {
  return Mask0<T>::FromBool(a.raw <= b.raw);
}
template <typename T>
HWY_INLINE Mask0<T> operator>=(const Vec0<T> a, const Vec0<T> b) {
  return Mask0<T>::FromBool(a.raw >= b.raw);
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
HWY_INLINE Vec0<T> Load(Scalar<T> /* tag */, const T* HWY_RESTRICT aligned) {
  T t;
  CopyBytes<sizeof(T)>(aligned, &t);
  return Vec0<T>(t);
}

template <typename T>
HWY_INLINE Vec0<T> LoadU(Scalar<T> d, const T* HWY_RESTRICT p) {
  return Load(d, p);
}

// In some use cases, "load single lane" is sufficient; otherwise avoid this.
template <typename T>
HWY_INLINE Vec0<T> LoadDup128(Scalar<T> d, const T* HWY_RESTRICT aligned) {
  return Load(d, aligned);
}

// ------------------------------ Store

template <typename T>
HWY_INLINE void Store(const Vec0<T> v, Scalar<T> /* tag */,
                      T* HWY_RESTRICT aligned) {
  CopyBytes<sizeof(T)>(&v.raw, aligned);
}

template <typename T>
HWY_INLINE void StoreU(const Vec0<T> v, Scalar<T> d, T* HWY_RESTRICT p) {
  return Store(v, d, p);
}

// ------------------------------ "Non-temporal" stores

template <typename T>
HWY_INLINE void Stream(const Vec0<T> v, Scalar<T> d, T* HWY_RESTRICT aligned) {
  return Store(v, d, aligned);
}

// ------------------------------ Gather

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T, typename Offset>
HWY_INLINE Vec0<T> GatherOffset(Scalar<T> d, const T* base,
                                const Vec0<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  const uintptr_t addr = reinterpret_cast<uintptr_t>(base) + offset.raw;
  return Load(d, reinterpret_cast<const T*>(addr));
}

template <typename T, typename Index>
HWY_INLINE Vec0<T> GatherIndex(Scalar<T> d, const T* HWY_RESTRICT base,
                               const Vec0<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  return Load(d, base + index.raw);
}

}  // namespace ext

// ================================================== CONVERT

template <typename FromT, typename ToT>
HWY_INLINE Vec0<ToT> ConvertTo(Scalar<ToT> /* tag */, Vec0<FromT> from) {
  return Vec0<ToT>(static_cast<ToT>(from.raw));
}

HWY_INLINE Vec0<float> ConvertTo(Scalar<float> /* tag */,
                                 const Vec0<int32_t> v) {
  return Vec0<float>(v.raw);
}

// Truncates (rounds toward zero).
HWY_INLINE Vec0<int32_t> ConvertTo(Scalar<int32_t> /* tag */,
                                   const Vec0<float> v) {
  return Vec0<int32_t>(static_cast<int>(v.raw));
}

HWY_INLINE Vec0<uint32_t> U32FromU8(const Vec0<uint8_t> v) {
  return ConvertTo(Scalar<uint32_t>(), v);
}

HWY_INLINE Vec0<uint8_t> U8FromU32(const Vec0<uint32_t> v) {
  return ConvertTo(Scalar<uint8_t>(), v);
}

// Approximation of round-to-nearest for numbers representable as int32_t.
HWY_INLINE Vec0<int32_t> NearestInt(const Vec0<float> v) {
  const float f = v.raw;
  const float bias = f < 0.0f ? -0.5f : 0.5f;
  return Vec0<int32_t>(static_cast<int>(f + bias));
}

// ================================================== SWIZZLE

// Unsupported: shift_*_bytes, CombineShiftRightBytes, interleave_*,
// shuffle_*, SumsOfU8x8, SumOfLanes, upper/lower/GetHalf - these require
// more than one lane and/or actual 128-bit vectors.

template <typename T>
HWY_INLINE T GetLane(const Vec0<T> v) {
  return v.raw;
}

// ------------------------------ Broadcast/splat any lane

template <int kLane, typename T>
HWY_INLINE Vec0<T> Broadcast(const Vec0<T> v) {
  static_assert(kLane == 0, "Scalar only has one lane");
  return v;
}

// ------------------------------ Zip/unpack

HWY_INLINE Vec0<uint16_t> ZipLo(const Vec0<uint8_t> a, const Vec0<uint8_t> b) {
  return Vec0<uint16_t>((uint32_t(b.raw) << 8) + a.raw);
}
HWY_INLINE Vec0<uint32_t> ZipLo(const Vec0<uint16_t> a,
                                const Vec0<uint16_t> b) {
  return Vec0<uint32_t>((uint32_t(b.raw) << 16) + a.raw);
}
HWY_INLINE Vec0<uint64_t> ZipLo(const Vec0<uint32_t> a,
                                const Vec0<uint32_t> b) {
  return Vec0<uint64_t>((uint64_t(b.raw) << 32) + a.raw);
}
HWY_INLINE Vec0<int16_t> ZipLo(const Vec0<int8_t> a, const Vec0<int8_t> b) {
  return Vec0<int16_t>((uint32_t(b.raw) << 8) + a.raw);
}
HWY_INLINE Vec0<int32_t> ZipLo(const Vec0<int16_t> a, const Vec0<int16_t> b) {
  return Vec0<int32_t>((uint32_t(b.raw) << 16) + a.raw);
}
HWY_INLINE Vec0<int64_t> ZipLo(const Vec0<int32_t> a, const Vec0<int32_t> b) {
  return Vec0<int64_t>((uint64_t(b.raw) << 32) + a.raw);
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
HWY_INLINE uint64_t movemask(const Vec0<uint8_t> v) { return v.raw >> 7; }

// Returns the most significant bit of each float/double lane (see above).
HWY_INLINE uint64_t movemask(const Vec0<float> v) {
  // Cannot return (v < 0) because +0.0 == -0.0.
  const auto bits = BitCast(Scalar<uint32_t>(), v);
  return GetLane(ShiftRight<31>(bits));
}
HWY_INLINE uint64_t movemask(const Vec0<double> v) {
  // Cannot return (v < 0) because +0.0 == -0.0.
  const auto bits = BitCast(Scalar<uint64_t>(), v);
  return GetLane(ShiftRight<63>(bits));
}

template <typename T>
HWY_INLINE bool AllFalse(const Mask0<T> v) {
  return v.bits == 0;
}

template <typename T>
HWY_INLINE bool AllTrue(const Mask0<T> v) {
  return v.bits != 0;
}

template <typename T>
HWY_INLINE size_t CountTrue(const Mask0<T> v) {
  return v.bits == 0 ? 0 : 1;
}

// Sum of all lanes, i.e. the only one.
template <typename T>
HWY_INLINE Vec0<T> SumOfLanes(const Vec0<T> v0) {
  return v0;
}

}  // namespace ext
}  // namespace hwy

#endif  // HWY_SCALAR_H_
