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

// Single-element vectors and operations.
// External include guard in highway.h - see comment there.

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include <cmath>

#include "hwy/ops/shared-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Single instruction, single data.
template <typename T>
using Sisd = Simd<T, 1>;

// (Wrapper class required for overloading comparison operators.)
template <typename T>
struct Vec1 {
  HWY_INLINE Vec1() = default;
  Vec1(const Vec1&) = default;
  Vec1& operator=(const Vec1&) = default;
  HWY_INLINE explicit Vec1(const T t) : raw(t) {}

  HWY_INLINE Vec1& operator*=(const Vec1 other) {
    return *this = (*this * other);
  }
  HWY_INLINE Vec1& operator/=(const Vec1 other) {
    return *this = (*this / other);
  }
  HWY_INLINE Vec1& operator+=(const Vec1 other) {
    return *this = (*this + other);
  }
  HWY_INLINE Vec1& operator-=(const Vec1 other) {
    return *this = (*this - other);
  }
  HWY_INLINE Vec1& operator&=(const Vec1 other) {
    return *this = (*this & other);
  }
  HWY_INLINE Vec1& operator|=(const Vec1 other) {
    return *this = (*this | other);
  }
  HWY_INLINE Vec1& operator^=(const Vec1 other) {
    return *this = (*this ^ other);
  }

  T raw;
};

// 0 or FF..FF, same size as Vec1.
template <typename T>
class Mask1 {
  using Raw = hwy::MakeUnsigned<T>;

 public:
  static HWY_INLINE Mask1<T> FromBool(bool b) {
    Mask1<T> mask;
    mask.bits = b ? ~Raw(0) : 0;
    return mask;
  }

  Raw bits;
};

// ------------------------------ BitCast

template <typename T, typename FromT>
HWY_INLINE Vec1<T> BitCast(Sisd<T> /* tag */, Vec1<FromT> v) {
  static_assert(sizeof(T) <= sizeof(FromT), "Promoting is undefined");
  T to;
  CopyBytes<sizeof(FromT)>(&v.raw, &to);
  return Vec1<T>(to);
}

// ------------------------------ Set

template <typename T>
HWY_INLINE Vec1<T> Zero(Sisd<T> /* tag */) {
  return Vec1<T>(T(0));
}

template <typename T, typename T2>
HWY_INLINE Vec1<T> Set(Sisd<T> /* tag */, const T2 t) {
  return Vec1<T>(static_cast<T>(t));
}

template <typename T>
HWY_INLINE Vec1<T> Undefined(Sisd<T> d) {
  return Zero(d);
}

template <typename T, typename T2>
Vec1<T> Iota(const Sisd<T> /* tag */, const T2 first) {
  return Vec1<T>(static_cast<T>(first));
}

// ================================================== SHIFTS

// ------------------------------ Shift lanes by constant #bits

template <int kBits, typename T>
HWY_INLINE Vec1<T> ShiftLeft(const Vec1<T> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return Vec1<T>(static_cast<hwy::MakeUnsigned<T>>(v.raw) << kBits);
}

template <int kBits, typename T>
HWY_INLINE Vec1<T> ShiftRight(const Vec1<T> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return Vec1<T>(v.raw >> kBits);
}

// ------------------------------ Shift lanes by independent variable #bits

// Single-lane => same as above except for the argument type.
template <typename T>
HWY_INLINE Vec1<T> operator<<(const Vec1<T> v, const Vec1<T> bits) {
  return Vec1<T>(static_cast<hwy::MakeUnsigned<T>>(v.raw) << bits.raw);
}
template <typename T>
HWY_INLINE Vec1<T> operator>>(const Vec1<T> v, const Vec1<T> bits) {
  return Vec1<T>(v.raw >> bits.raw);
}

// ================================================== LOGICAL

template <typename Bits>
struct BitwiseOp {
  template <typename T, class Op>
  Vec1<T> operator()(const Vec1<T> a, const Vec1<T> b, const Op& op) const {
    static_assert(sizeof(T) == sizeof(Bits), "Float/int size mismatch");
    Bits ia, ib;
    CopyBytes<sizeof(Bits)>(&a, &ia);
    CopyBytes<sizeof(Bits)>(&b, &ib);
    ia = op(ia, ib);
    T ret;
    CopyBytes<sizeof(Bits)>(&ia, &ret);
    return Vec1<T>(ret);
  }
};

// ------------------------------ Bitwise AND

template <typename T>
HWY_INLINE Vec1<T> operator&(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(a.raw & b.raw);
}

HWY_INLINE Vec1<float> And(const Vec1<float> a, const Vec1<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i & j; });
}
HWY_INLINE Vec1<double> And(const Vec1<double> a, const Vec1<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i & j; });
}

// ------------------------------ Bitwise AND-NOT

// Returns ~a & b.
template <typename T>
HWY_INLINE Vec1<T> AndNot(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(~a.raw & b.raw);
}
HWY_INLINE Vec1<float> AndNot(const Vec1<float> a, const Vec1<float> b) {
  return BitwiseOp<int32_t>()(a, b,
                              [](int32_t i, int32_t j) { return ~i & j; });
}
HWY_INLINE Vec1<double> AndNot(const Vec1<double> a, const Vec1<double> b) {
  return BitwiseOp<int64_t>()(a, b,
                              [](int64_t i, int64_t j) { return ~i & j; });
}

// ------------------------------ Bitwise OR

template <typename T>
HWY_INLINE Vec1<T> operator|(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(a.raw | b.raw);
}

HWY_INLINE Vec1<float> Or(const Vec1<float> a, const Vec1<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i | j; });
}
HWY_INLINE Vec1<double> Or(const Vec1<double> a, const Vec1<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i | j; });
}

// ------------------------------ Bitwise XOR

template <typename T>
HWY_INLINE Vec1<T> operator^(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(a.raw ^ b.raw);
}

HWY_INLINE Vec1<float> Xor(const Vec1<float> a, const Vec1<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i ^ j; });
}
HWY_INLINE Vec1<double> Xor(const Vec1<double> a, const Vec1<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i ^ j; });
}

// ------------------------------ CopySign

template <typename T>
HWY_API Vec1<T> CopySign(const Vec1<T> magn, const Vec1<T> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  const auto msb = SignBit(Sisd<T>());
  return Or(AndNot(msb, magn), And(msb, sign));
}

template <typename T>
HWY_API Vec1<T> CopySignToAbs(const Vec1<T> abs, const Vec1<T> sign) {
  static_assert(IsFloat<T>(), "Only makes sense for floating-point");
  return Or(abs, And(SignBit(Sisd<T>()), sign));
}

// ------------------------------ Mask

// v must be 0 or FF..FF.
template <typename T>
HWY_INLINE Mask1<T> MaskFromVec(const Vec1<T> v) {
  Mask1<T> mask;
  memcpy(&mask.bits, &v.raw, sizeof(mask.bits));
  return mask;
}

template <typename T>
Vec1<T> VecFromMask(const Mask1<T> mask) {
  Vec1<T> v;
  memcpy(&v.raw, &mask.bits, sizeof(v.raw));
  return v;
}

// Returns mask ? yes : no.
template <typename T>
HWY_INLINE Vec1<T> IfThenElse(const Mask1<T> mask, const Vec1<T> yes,
                              const Vec1<T> no) {
  return mask.bits ? yes : no;
}

template <typename T>
HWY_INLINE Vec1<T> IfThenElseZero(const Mask1<T> mask, const Vec1<T> yes) {
  return mask.bits ? yes : Vec1<T>(0);
}

template <typename T>
HWY_INLINE Vec1<T> IfThenZeroElse(const Mask1<T> mask, const Vec1<T> no) {
  return mask.bits ? Vec1<T>(0) : no;
}

template <typename T>
HWY_INLINE Vec1<T> ZeroIfNegative(const Vec1<T> v) {
  return v.raw < 0 ? Vec1<T>(0) : v;
}

// ================================================== ARITHMETIC

template <typename T>
HWY_INLINE Vec1<T> operator+(Vec1<T> a, Vec1<T> b) {
  const uint64_t a64 = static_cast<int64_t>(a.raw);
  const uint64_t b64 = static_cast<int64_t>(b.raw);
  return Vec1<T>((a64 + b64) & ~T(0));
}
HWY_INLINE Vec1<float> operator+(const Vec1<float> a, const Vec1<float> b) {
  return Vec1<float>(a.raw + b.raw);
}
HWY_INLINE Vec1<double> operator+(const Vec1<double> a, const Vec1<double> b) {
  return Vec1<double>(a.raw + b.raw);
}

template <typename T>
HWY_INLINE Vec1<T> operator-(Vec1<T> a, Vec1<T> b) {
  const uint64_t a64 = static_cast<int64_t>(a.raw);
  const uint64_t b64 = static_cast<int64_t>(b.raw);
  return Vec1<T>((a64 - b64) & ~T(0));
}
HWY_INLINE Vec1<float> operator-(const Vec1<float> a, const Vec1<float> b) {
  return Vec1<float>(a.raw - b.raw);
}
HWY_INLINE Vec1<double> operator-(const Vec1<double> a, const Vec1<double> b) {
  return Vec1<double>(a.raw - b.raw);
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
HWY_INLINE Vec1<uint8_t> SaturatedAdd(const Vec1<uint8_t> a,
                                      const Vec1<uint8_t> b) {
  return Vec1<uint8_t>(HWY_MIN(HWY_MAX(0, a.raw + b.raw), 255));
}
HWY_INLINE Vec1<uint16_t> SaturatedAdd(const Vec1<uint16_t> a,
                                       const Vec1<uint16_t> b) {
  return Vec1<uint16_t>(HWY_MIN(HWY_MAX(0, a.raw + b.raw), 65535));
}

// Signed
HWY_INLINE Vec1<int8_t> SaturatedAdd(const Vec1<int8_t> a,
                                     const Vec1<int8_t> b) {
  return Vec1<int8_t>(HWY_MIN(HWY_MAX(-128, a.raw + b.raw), 127));
}
HWY_INLINE Vec1<int16_t> SaturatedAdd(const Vec1<int16_t> a,
                                      const Vec1<int16_t> b) {
  return Vec1<int16_t>(HWY_MIN(HWY_MAX(-32768, a.raw + b.raw), 32767));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
HWY_INLINE Vec1<uint8_t> SaturatedSub(const Vec1<uint8_t> a,
                                      const Vec1<uint8_t> b) {
  return Vec1<uint8_t>(HWY_MIN(HWY_MAX(0, a.raw - b.raw), 255));
}
HWY_INLINE Vec1<uint16_t> SaturatedSub(const Vec1<uint16_t> a,
                                       const Vec1<uint16_t> b) {
  return Vec1<uint16_t>(HWY_MIN(HWY_MAX(0, a.raw - b.raw), 65535));
}

// Signed
HWY_INLINE Vec1<int8_t> SaturatedSub(const Vec1<int8_t> a,
                                     const Vec1<int8_t> b) {
  return Vec1<int8_t>(HWY_MIN(HWY_MAX(-128, a.raw - b.raw), 127));
}
HWY_INLINE Vec1<int16_t> SaturatedSub(const Vec1<int16_t> a,
                                      const Vec1<int16_t> b) {
  return Vec1<int16_t>(HWY_MIN(HWY_MAX(-32768, a.raw - b.raw), 32767));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

HWY_INLINE Vec1<uint8_t> AverageRound(const Vec1<uint8_t> a,
                                      const Vec1<uint8_t> b) {
  return Vec1<uint8_t>((a.raw + b.raw + 1) / 2);
}
HWY_INLINE Vec1<uint16_t> AverageRound(const Vec1<uint16_t> a,
                                       const Vec1<uint16_t> b) {
  return Vec1<uint16_t>((a.raw + b.raw + 1) / 2);
}

// ------------------------------ Absolute value

template <typename T>
HWY_INLINE Vec1<T> Abs(const Vec1<T> a) {
  const T i = a.raw;
  return (i >= 0 || i == hwy::LimitsMin<T>()) ? a : Vec1<T>(-i);
}
HWY_INLINE Vec1<float> Abs(const Vec1<float> a) {
  return Vec1<float>(std::abs(a.raw));
}
HWY_INLINE Vec1<double> Abs(const Vec1<double> a) {
  return Vec1<double>(std::abs(a.raw));
}

// ------------------------------ min/max

template <typename T>
HWY_INLINE Vec1<T> Min(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(HWY_MIN(a.raw, b.raw));
}

template <typename T>
HWY_INLINE Vec1<T> Max(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(HWY_MAX(a.raw, b.raw));
}

// ------------------------------ Floating-point negate

template <typename T, HWY_IF_FLOAT(T)>
HWY_INLINE Vec1<T> Neg(const Vec1<T> v) {
  return Xor(v, SignBit(Sisd<T>()));
}

template <typename T, HWY_IF_NOT_FLOAT(T)>
HWY_INLINE Vec1<T> Neg(const Vec1<T> v) {
  return Zero(Sisd<T>()) - v;
}

// ------------------------------ mul/div

template <typename T>
HWY_INLINE Vec1<T> operator*(const Vec1<T> a, const Vec1<T> b) {
  if (hwy::IsFloat<T>()) {
    return Vec1<T>(static_cast<T>(double(a.raw) * b.raw));
  } else if (hwy::IsSigned<T>()) {
    return Vec1<T>(static_cast<T>(int64_t(a.raw) * b.raw));
  } else {
    return Vec1<T>(static_cast<T>(uint64_t(a.raw) * b.raw));
  }
}

template <typename T>
HWY_INLINE Vec1<T> operator/(const Vec1<T> a, const Vec1<T> b) {
  return Vec1<T>(a.raw / b.raw);
}

// Returns the upper 16 bits of a * b in each lane.
HWY_INLINE Vec1<int16_t> MulHigh(const Vec1<int16_t> a, const Vec1<int16_t> b) {
  return Vec1<int16_t>((a.raw * b.raw) >> 16);
}
HWY_INLINE Vec1<uint16_t> MulHigh(const Vec1<uint16_t> a,
                                  const Vec1<uint16_t> b) {
  // Cast to uint32_t first to prevent overflow. Otherwise the result of
  // uint16_t * uint16_t is in "int" which may overflow. In practice the result
  // is the same but this way it is also defined.
  return Vec1<uint16_t>(
      (static_cast<uint32_t>(a.raw) * static_cast<uint32_t>(b.raw)) >> 16);
}

// Multiplies even lanes (0, 2 ..) and returns the double-wide result.
HWY_INLINE Vec1<int64_t> MulEven(const Vec1<int32_t> a, const Vec1<int32_t> b) {
  const int64_t a64 = a.raw;
  return Vec1<int64_t>(a64 * b.raw);
}
HWY_INLINE Vec1<uint64_t> MulEven(const Vec1<uint32_t> a,
                                  const Vec1<uint32_t> b) {
  const uint64_t a64 = a.raw;
  return Vec1<uint64_t>(a64 * b.raw);
}

// Approximate reciprocal
HWY_INLINE Vec1<float> ApproximateReciprocal(const Vec1<float> v) {
  // Zero inputs are allowed, but callers are responsible for replacing the
  // return value with something else (typically using IfThenElse). This check
  // avoids a ubsan error. The return value is arbitrary.
  if (v.raw == 0.0f) return Vec1<float>(0.0f);
  return Vec1<float>(1.0f / v.raw);
}

// Absolute value of difference.
HWY_INLINE Vec1<float> AbsDiff(const Vec1<float> a, const Vec1<float> b) {
  return Abs(a - b);
}

// ------------------------------ Floating-point multiply-add variants

template <typename T>
HWY_INLINE Vec1<T> MulAdd(const Vec1<T> mul, const Vec1<T> x,
                          const Vec1<T> add) {
  return mul * x + add;
}

template <typename T>
HWY_INLINE Vec1<T> NegMulAdd(const Vec1<T> mul, const Vec1<T> x,
                             const Vec1<T> add) {
  return add - mul * x;
}

template <typename T>
HWY_INLINE Vec1<T> MulSub(const Vec1<T> mul, const Vec1<T> x,
                          const Vec1<T> sub) {
  return mul * x - sub;
}

template <typename T>
HWY_INLINE Vec1<T> NegMulSub(const Vec1<T> mul, const Vec1<T> x,
                             const Vec1<T> sub) {
  return Neg(mul) * x - sub;
}

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
HWY_INLINE Vec1<float> ApproximateReciprocalSqrt(const Vec1<float> v) {
  float f = v.raw;
  const float half = f * 0.5f;
  uint32_t bits;
  CopyBytes<4>(&f, &bits);
  // Initial guess based on log2(f)
  bits = 0x5F3759DF - (bits >> 1);
  CopyBytes<4>(&bits, &f);
  // One Newton-Raphson iteration
  return Vec1<float>(f * (1.5f - (half * f * f)));
}

// Square root
HWY_INLINE Vec1<float> Sqrt(const Vec1<float> v) {
  return Vec1<float>(std::sqrt(v.raw));
}
HWY_INLINE Vec1<double> Sqrt(const Vec1<double> v) {
  return Vec1<double>(std::sqrt(v.raw));
}

// ------------------------------ Floating-point rounding

// Approximation of round-to-nearest for numbers representable as integers.
HWY_INLINE Vec1<float> Round(const Vec1<float> v) {
  const float bias = v.raw < 0.0f ? -0.5f : 0.5f;
  const int32_t rounded = static_cast<int32_t>(v.raw + bias);
  if (rounded == 0) return Vec1<float>(v.raw < 0.0f ? -0.0f : 0.0f);
  return Vec1<float>(static_cast<float>(rounded));
}
HWY_INLINE Vec1<double> Round(const Vec1<double> v) {
  const double bias = v.raw < 0.0 ? -0.5 : 0.5;
  const int64_t rounded = static_cast<int64_t>(v.raw + bias);
  if (rounded == 0) return Vec1<double>(v.raw < 0.0 ? -0.0 : 0.0);
  return Vec1<double>(std::copysign(rounded, v.raw));
}

HWY_INLINE Vec1<float> Trunc(const Vec1<float> v) {
  const int32_t truncated = static_cast<int32_t>(v.raw);
  if (truncated == 0) return Vec1<float>(v.raw < 0.0f ? -0.0f : 0.0f);
  return Vec1<float>(static_cast<float>(truncated));
}
HWY_INLINE Vec1<double> Trunc(const Vec1<double> v) {
  const int64_t truncated = static_cast<int64_t>(v.raw);
  if (truncated == 0) return Vec1<double>(v.raw < 0.0 ? -0.0 : 0.0);
  return Vec1<double>(static_cast<double>(truncated));
}

template <typename Float, typename Bits, int kMantissaBits, int kExponentBits,
          class V>
V Ceiling(const V v) {
  const Bits kExponentMask = (1ull << kExponentBits) - 1;
  const Bits kMantissaMask = (1ull << kMantissaBits) - 1;
  const Bits kBias = kExponentMask / 2;

  Float f = v.raw;
  const bool positive = f > Float(0.0);

  Bits bits;
  CopyBytes<sizeof(Bits)>(&v, &bits);

  const int exponent = ((bits >> kMantissaBits) & kExponentMask) - kBias;
  // Already an integer.
  if (exponent >= kMantissaBits) return v;
  // |v| <= 1 => 0 or 1.
  if (exponent < 0) return positive ? V(1) : V(-0.0);

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
  const bool negative = f < Float(0.0);

  Bits bits;
  CopyBytes<sizeof(Bits)>(&v, &bits);

  const int exponent = ((bits >> kMantissaBits) & kExponentMask) - kBias;
  // Already an integer.
  if (exponent >= kMantissaBits) return v;
  // |v| <= 1 => -1 or 0.
  if (exponent < 0) return V(negative ? Float(-1.0) : Float(0.0));

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
HWY_INLINE Vec1<float> Ceil(const Vec1<float> v) {
  return Ceiling<float, uint32_t, 23, 8>(v);
}
HWY_INLINE Vec1<double> Ceil(const Vec1<double> v) {
  return Ceiling<double, uint64_t, 52, 11>(v);
}

// Toward -infinity, aka floor
HWY_INLINE Vec1<float> Floor(const Vec1<float> v) {
  return Floor<float, uint32_t, 23, 8>(v);
}
HWY_INLINE Vec1<double> Floor(const Vec1<double> v) {
  return Floor<double, uint64_t, 52, 11>(v);
}

// ================================================== COMPARE

template <typename T>
HWY_INLINE Mask1<T> operator==(const Vec1<T> a, const Vec1<T> b) {
  return Mask1<T>::FromBool(a.raw == b.raw);
}

template <typename T>
HWY_INLINE Mask1<T> TestBit(const Vec1<T> v, const Vec1<T> bit) {
  static_assert(!hwy::IsFloat<T>(), "Only integer vectors supported");
  return (v & bit) == bit;
}

template <typename T>
HWY_INLINE Mask1<T> operator<(const Vec1<T> a, const Vec1<T> b) {
  return Mask1<T>::FromBool(a.raw < b.raw);
}
template <typename T>
HWY_INLINE Mask1<T> operator>(const Vec1<T> a, const Vec1<T> b) {
  return Mask1<T>::FromBool(a.raw > b.raw);
}

template <typename T>
HWY_INLINE Mask1<T> operator<=(const Vec1<T> a, const Vec1<T> b) {
  return Mask1<T>::FromBool(a.raw <= b.raw);
}
template <typename T>
HWY_INLINE Mask1<T> operator>=(const Vec1<T> a, const Vec1<T> b) {
  return Mask1<T>::FromBool(a.raw >= b.raw);
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
HWY_INLINE Vec1<T> Load(Sisd<T> /* tag */, const T* HWY_RESTRICT aligned) {
  T t;
  CopyBytes<sizeof(T)>(aligned, &t);
  return Vec1<T>(t);
}

template <typename T>
HWY_INLINE Vec1<T> LoadU(Sisd<T> d, const T* HWY_RESTRICT p) {
  return Load(d, p);
}

// In some use cases, "load single lane" is sufficient; otherwise avoid this.
template <typename T>
HWY_INLINE Vec1<T> LoadDup128(Sisd<T> d, const T* HWY_RESTRICT aligned) {
  return Load(d, aligned);
}

// ------------------------------ Store

template <typename T>
HWY_INLINE void Store(const Vec1<T> v, Sisd<T> /* tag */,
                      T* HWY_RESTRICT aligned) {
  CopyBytes<sizeof(T)>(&v.raw, aligned);
}

template <typename T>
HWY_INLINE void StoreU(const Vec1<T> v, Sisd<T> d, T* HWY_RESTRICT p) {
  return Store(v, d, p);
}

// ------------------------------ "Non-temporal" stores

template <typename T>
HWY_INLINE void Stream(const Vec1<T> v, Sisd<T> d, T* HWY_RESTRICT aligned) {
  return Store(v, d, aligned);
}

// ------------------------------ Gather

template <typename T, typename Offset>
HWY_INLINE Vec1<T> GatherOffset(Sisd<T> d, const T* base,
                                const Vec1<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  const uintptr_t addr = reinterpret_cast<uintptr_t>(base) + offset.raw;
  return Load(d, reinterpret_cast<const T*>(addr));
}

template <typename T, typename Index>
HWY_INLINE Vec1<T> GatherIndex(Sisd<T> d, const T* HWY_RESTRICT base,
                               const Vec1<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  return Load(d, base + index.raw);
}

// ================================================== CONVERT

// ConvertTo and DemoteTo with floating-point input and integer output truncate
// (rounding toward zero).

template <typename FromT, typename ToT>
HWY_INLINE Vec1<ToT> PromoteTo(Sisd<ToT> /* tag */, Vec1<FromT> from) {
  static_assert(sizeof(ToT) > sizeof(FromT), "Not promoting");
  return Vec1<ToT>(static_cast<ToT>(from.raw));
}

template <typename FromT, typename ToT>
HWY_INLINE Vec1<ToT> DemoteTo(Sisd<ToT> /* tag */, Vec1<FromT> from) {
  static_assert(sizeof(ToT) < sizeof(FromT), "Not demoting");
  // Prevent ubsan errors when converting float to integers
  if (IsFloat<FromT>() && !IsFloat<ToT>()) {
    const double val = static_cast<double>(from.raw);
    if (std::isinf(val) ||
        std::fabs(val) > static_cast<double>(LimitsMax<ToT>())) {
      return Vec1<ToT>(std::signbit(val) ? LimitsMin<ToT>() : LimitsMax<ToT>());
    }
  }
  return Vec1<ToT>(static_cast<ToT>(from.raw));
}

template <typename FromT, typename ToT>
HWY_INLINE Vec1<ToT> ConvertTo(Sisd<ToT> /* tag */, Vec1<FromT> from) {
  // Prevent ubsan errors when converting float to integers
  if (IsFloat<FromT>() && !IsFloat<ToT>()) {
    const double val = static_cast<double>(from.raw);
    if (std::isinf(val) ||
        std::fabs(val) > static_cast<double>(LimitsMax<ToT>())) {
      return Vec1<ToT>(std::signbit(val) ? LimitsMin<ToT>() : LimitsMax<ToT>());
    }
  }
  return Vec1<ToT>(static_cast<ToT>(from.raw));
}

HWY_INLINE Vec1<uint32_t> U32FromU8(const Vec1<uint8_t> v) {
  return PromoteTo(Sisd<uint32_t>(), v);
}

HWY_INLINE Vec1<uint8_t> U8FromU32(const Vec1<uint32_t> v) {
  return DemoteTo(Sisd<uint8_t>(), v);
}

// Approximation of round-to-nearest for numbers representable as int32_t.
HWY_INLINE Vec1<int32_t> NearestInt(const Vec1<float> v) {
  const float f = v.raw;
  const float bias = f < 0.0f ? -0.5f : 0.5f;
  return Vec1<int32_t>(static_cast<int>(f + bias));
}

// ================================================== SWIZZLE

// Unsupported: Shift*Bytes, CombineShiftRightBytes, Interleave*,
// Shuffle*, SumsOfU8x8, UpperHalf - these require more than one lane and/or
// actual 128-bit vectors.

template <typename T>
HWY_INLINE T GetLane(const Vec1<T> v) {
  return v.raw;
}

template <typename T>
HWY_INLINE Vec1<T> LowerHalf(Vec1<T> v) {
  return v;
}

// ------------------------------ Broadcast/splat any lane

template <int kLane, typename T>
HWY_INLINE Vec1<T> Broadcast(const Vec1<T> v) {
  static_assert(kLane == 0, "Scalar only has one lane");
  return v;
}

// ------------------------------ Shuffle bytes with variable indices

// Returns vector of bytes[from[i]]. "from" is also interpreted as bytes, i.e.
// indices in [0, sizeof(T)).
template <typename T>
HWY_API Vec1<T> TableLookupBytes(const Vec1<T> in, const Vec1<T> from) {
  uint8_t in_bytes[sizeof(T)];
  uint8_t from_bytes[sizeof(T)];
  uint8_t out_bytes[sizeof(T)];
  memcpy(&in_bytes, &in, sizeof(T));
  memcpy(&from_bytes, &from, sizeof(T));
  for (size_t i = 0; i < sizeof(T); ++i) {
    out_bytes[i] = in_bytes[from_bytes[i]];
  }
  T out;
  memcpy(&out, &out_bytes, sizeof(T));
  return Vec1<T>{out};
}

// ------------------------------ TableLookupLanes

// Returned by SetTableIndices for use by TableLookupLanes.
template <typename T>
struct Indices1 {
  int raw;
};

template <typename T>
HWY_API Indices1<T> SetTableIndices(Sisd<T>, const int32_t* idx) {
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
  HWY_DASSERT(idx[0] == 0);
#endif
  return Indices1<T>{idx[0]};
}

template <typename T>
HWY_API Vec1<T> TableLookupLanes(const Vec1<T> v, const Indices1<T> /* idx */) {
  return v;
}

// ------------------------------ Zip/unpack

HWY_INLINE Vec1<uint16_t> ZipLower(const Vec1<uint8_t> a,
                                   const Vec1<uint8_t> b) {
  return Vec1<uint16_t>(static_cast<uint16_t>((uint32_t(b.raw) << 8) + a.raw));
}
HWY_INLINE Vec1<uint32_t> ZipLower(const Vec1<uint16_t> a,
                                   const Vec1<uint16_t> b) {
  return Vec1<uint32_t>((uint32_t(b.raw) << 16) + a.raw);
}
HWY_INLINE Vec1<uint64_t> ZipLower(const Vec1<uint32_t> a,
                                   const Vec1<uint32_t> b) {
  return Vec1<uint64_t>((uint64_t(b.raw) << 32) + a.raw);
}
HWY_INLINE Vec1<int16_t> ZipLower(const Vec1<int8_t> a, const Vec1<int8_t> b) {
  return Vec1<int16_t>(static_cast<int16_t>((int32_t(b.raw) << 8) + a.raw));
}
HWY_INLINE Vec1<int32_t> ZipLower(const Vec1<int16_t> a,
                                  const Vec1<int16_t> b) {
  return Vec1<int32_t>((int32_t(b.raw) << 16) + a.raw);
}
HWY_INLINE Vec1<int64_t> ZipLower(const Vec1<int32_t> a,
                                  const Vec1<int32_t> b) {
  return Vec1<int64_t>((int64_t(b.raw) << 32) + a.raw);
}

// ------------------------------ Mask

template <typename T>
HWY_INLINE bool AllFalse(const Mask1<T> mask) {
  return mask.bits == 0;
}

template <typename T>
HWY_INLINE bool AllTrue(const Mask1<T> mask) {
  return mask.bits != 0;
}

template <typename T>
HWY_INLINE uint64_t BitsFromMask(const Mask1<T> mask) {
  return mask.bits & 1;
}

template <typename T>
HWY_INLINE size_t CountTrue(const Mask1<T> mask) {
  return mask.bits == 0 ? 0 : 1;
}

template <typename T>
HWY_API Vec1<T> Compress(Vec1<T> v, const Mask1<T> /* mask */) {
  // Upper lanes are undefined, so result is the same independent of mask.
  return v;
}

// ------------------------------ CompressStore

template <typename T>
HWY_API size_t CompressStore(Vec1<T> v, const Mask1<T> mask, Sisd<T> d,
                             T* HWY_RESTRICT aligned) {
  Store(Compress(v, mask), d, aligned);
  return CountTrue(mask);
}

// ------------------------------ Reductions

// Sum of all lanes, i.e. the only one.
template <typename T>
HWY_INLINE Vec1<T> SumOfLanes(const Vec1<T> v0) {
  return v0;
}
template <typename T>
HWY_INLINE Vec1<T> MinOfLanes(const Vec1<T> v) {
  return v;
}
template <typename T>
HWY_INLINE Vec1<T> MaxOfLanes(const Vec1<T> v) {
  return v;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
