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

#ifndef HIGHWAY_SCALAR_H_
#define HIGHWAY_SCALAR_H_

// Single-element vectors and operations.

#include "third_party/highway/highway/compiler_specific.h"
#include "third_party/highway/highway/shared.h"

namespace jxl {

// Shorthand for a scalar; note that scalar<T> is the actual data class.
template <typename T>
using Scalar = Desc<T, 0>;

// Returned by set_shift_*_count; do not use directly.
struct scalar_shift_left_count {
  int count;
};
struct scalar_shift_right_count {
  int count;
};

// (Wrapper class required for overloading comparison operators.)
template <typename T>
struct scalar {
  SIMD_INLINE scalar() = default;
  scalar(const scalar&) = default;
  scalar& operator=(const scalar&) = default;
  SIMD_INLINE explicit scalar(const T t) : raw(t) {}

  SIMD_INLINE scalar& operator*=(const scalar other) {
    return *this = (*this * other);
  }
  SIMD_INLINE scalar& operator/=(const scalar other) {
    return *this = (*this / other);
  }
  SIMD_INLINE scalar& operator+=(const scalar other) {
    return *this = (*this + other);
  }
  SIMD_INLINE scalar& operator-=(const scalar other) {
    return *this = (*this - other);
  }
  SIMD_INLINE scalar& operator&=(const scalar other) {
    return *this = (*this & other);
  }
  SIMD_INLINE scalar& operator|=(const scalar other) {
    return *this = (*this | other);
  }
  SIMD_INLINE scalar& operator^=(const scalar other) {
    return *this = (*this ^ other);
  }

  T raw;
};

// ------------------------------ Cast

template <typename T, typename FromT>
SIMD_INLINE scalar<T> bit_cast(Scalar<T> /* tag */, scalar<FromT> v) {
  static_assert(sizeof(T) <= sizeof(FromT), "Promoting is undefined");
  T to;
  CopyBytes<sizeof(FromT)>(&v.raw, &to);
  return scalar<T>(to);
}

// ------------------------------ Set

template <typename T>
SIMD_INLINE scalar<T> setzero(Scalar<T> /* tag */) {
  return scalar<T>(T(0));
}

template <typename T, typename T2>
SIMD_INLINE scalar<T> set1(Scalar<T> /* tag */, const T2 t) {
  return scalar<T>(t);
}

template <typename T, typename T2>
SIMD_INLINE scalar<T> iota(Scalar<T> /* tag */, const T2 first) {
  return scalar<T>(first);
}

template <typename T>
SIMD_INLINE scalar<T> undefined(Scalar<T> /* tag */) {
  return scalar<T>(0);
}

// ================================================== SHIFTS

// MakeUnsignedT<T, B>::type is the unsigned version of T. This is only defined
// for all unsigned types and the explicit signed types here.
template <typename T, bool Signed>
struct MakeUnsignedT;

template <typename T>
struct MakeUnsignedT<T, false> {
  // Partial specialization for all the unsigned types.
  typedef T type;
};

template <>
struct MakeUnsignedT<int8_t, true> {
  typedef uint8_t type;
};

template <>
struct MakeUnsignedT<int16_t, true> {
  typedef uint16_t type;
};

template <>
struct MakeUnsignedT<int32_t, true> {
  typedef uint32_t type;
};

template <>
struct MakeUnsignedT<int64_t, true> {
  typedef uint64_t type;
};

template <typename T>
using make_unsigned = typename MakeUnsignedT<T, IsSigned<T>()>::type;

// ------------------------------ Shift lanes by constant #bits

template <int kBits, typename T>
SIMD_INLINE scalar<T> shift_left(const scalar<T> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return scalar<T>(static_cast<make_unsigned<T>>(v.raw) << kBits);
}

template <int kBits, typename T>
SIMD_INLINE scalar<T> shift_right(const scalar<T> v) {
  static_assert(0 <= kBits && kBits < sizeof(T) * 8, "Invalid shift");
  return scalar<T>(v.raw >> kBits);
}

// ------------------------------ Shift lanes by same variable #bits

template <typename T>
SIMD_INLINE scalar_shift_left_count set_shift_left_count(Scalar<T> /* tag */,
                                                         const int bits) {
  return scalar_shift_left_count{bits};
}

template <typename T>
SIMD_INLINE scalar_shift_right_count set_shift_right_count(Scalar<T> /* tag */,
                                                           const int bits) {
  return scalar_shift_right_count{bits};
}

template <typename T>
SIMD_INLINE scalar<T> shift_left_same(const scalar<T> v,
                                      const scalar_shift_left_count bits) {
  return scalar<T>(static_cast<make_unsigned<T>>(v.raw) << bits.count);
}
template <typename T>
SIMD_INLINE scalar<T> shift_right_same(const scalar<T> v,
                                       const scalar_shift_right_count bits) {
  return scalar<T>(v.raw >> bits.count);
}

// ------------------------------ Shift lanes by independent variable #bits

// Single-lane => same as above except for the argument type.
template <typename T>
SIMD_INLINE scalar<T> operator<<(const scalar<T> v, const scalar<T> bits) {
  return scalar<T>(static_cast<make_unsigned<T>>(v.raw) << bits.raw);
}
template <typename T>
SIMD_INLINE scalar<T> operator>>(const scalar<T> v, const scalar<T> bits) {
  return scalar<T>(v.raw >> bits.raw);
}

// ================================================== LOGICAL

template <typename Bits>
struct BitwiseOp {
  template <typename T, class Op>
  scalar<T> operator()(const scalar<T> a, const scalar<T> b,
                       const Op& op) const {
    static_assert(sizeof(T) == sizeof(Bits), "Float/int size mismatch");
    Bits ia, ib;
    CopyBytes<sizeof(Bits)>(&a, &ia);
    CopyBytes<sizeof(Bits)>(&b, &ib);
    ia = op(ia, ib);
    T ret;
    CopyBytes<sizeof(Bits)>(&ia, &ret);
    return scalar<T>(ret);
  }
};

// ------------------------------ Bitwise AND

template <typename T>
SIMD_INLINE scalar<T> operator&(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(a.raw & b.raw);
}
template <>
SIMD_INLINE scalar<float> operator&(const scalar<float> a,
                                    const scalar<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i & j; });
}
template <>
SIMD_INLINE scalar<double> operator&(const scalar<double> a,
                                     const scalar<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i & j; });
}

// ------------------------------ Bitwise AND-NOT

// Returns ~a & b.
template <typename T>
SIMD_INLINE scalar<T> andnot(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(~a.raw & b.raw);
}
template <>
SIMD_INLINE scalar<float> andnot(const scalar<float> a, const scalar<float> b) {
  return BitwiseOp<int32_t>()(a, b,
                              [](int32_t i, int32_t j) { return ~i & j; });
}
template <>
SIMD_INLINE scalar<double> andnot(const scalar<double> a,
                                  const scalar<double> b) {
  return BitwiseOp<int64_t>()(a, b,
                              [](int64_t i, int64_t j) { return ~i & j; });
}

// ------------------------------ Bitwise OR

template <typename T>
SIMD_INLINE scalar<T> operator|(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(a.raw | b.raw);
}
template <>
SIMD_INLINE scalar<float> operator|(const scalar<float> a,
                                    const scalar<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i | j; });
}
template <>
SIMD_INLINE scalar<double> operator|(const scalar<double> a,
                                     const scalar<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i | j; });
}

// ------------------------------ Bitwise XOR

template <typename T>
SIMD_INLINE scalar<T> operator^(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(a.raw ^ b.raw);
}
template <>
SIMD_INLINE scalar<float> operator^(const scalar<float> a,
                                    const scalar<float> b) {
  return BitwiseOp<int32_t>()(a, b, [](int32_t i, int32_t j) { return i ^ j; });
}
template <>
SIMD_INLINE scalar<double> operator^(const scalar<double> a,
                                     const scalar<double> b) {
  return BitwiseOp<int64_t>()(a, b, [](int64_t i, int64_t j) { return i ^ j; });
}

// ------------------------------ Select/blend

// Returns a mask for use by if_then_else().
SIMD_INLINE scalar<float> mask_from_sign(const scalar<float> v) {
  const Scalar<float> df;
  const Scalar<int32_t> di;
  return bit_cast(df, shift_right<31>(bit_cast(di, v)));
}
SIMD_INLINE scalar<double> mask_from_sign(const scalar<double> v) {
  const Scalar<double> df;
  const Scalar<int64_t> di;
  return bit_cast(df, shift_right<63>(bit_cast(di, v)));
}

// Returns mask ? yes : no. "mask" must either have been returned by
// selector_from_mask, or callers must ensure its lanes are T(0) or ~T(0).
template <typename T>
SIMD_INLINE scalar<T> if_then_else(const scalar<T> mask, const scalar<T> yes,
                                   const scalar<T> no) {
  return (mask & yes) | andnot(mask, no);
}

// ================================================== ARITHMETIC

template <typename T>
SIMD_INLINE scalar<T> operator+(const scalar<T> a, const scalar<T> b) {
  const uint64_t a64 = static_cast<int64_t>(a.raw);
  const uint64_t b64 = static_cast<int64_t>(b.raw);
  return scalar<T>((a64 + b64) & ~T(0));
}
SIMD_INLINE scalar<float> operator+(const scalar<float> a,
                                    const scalar<float> b) {
  return scalar<float>(a.raw + b.raw);
}
SIMD_INLINE scalar<double> operator+(const scalar<double> a,
                                     const scalar<double> b) {
  return scalar<double>(a.raw + b.raw);
}

template <typename T>
SIMD_INLINE scalar<T> operator-(const scalar<T> a, const scalar<T> b) {
  const uint64_t a64 = static_cast<int64_t>(a.raw);
  const uint64_t b64 = static_cast<int64_t>(b.raw);
  return scalar<T>((a64 - b64) & ~T(0));
}
SIMD_INLINE scalar<float> operator-(const scalar<float> a,
                                    const scalar<float> b) {
  return scalar<float>(a.raw - b.raw);
}
SIMD_INLINE scalar<double> operator-(const scalar<double> a,
                                     const scalar<double> b) {
  return scalar<double>(a.raw - b.raw);
}

// ------------------------------ Saturating addition

// Returns a + b clamped to the destination range.

// Unsigned
SIMD_INLINE scalar<uint8_t> saturated_add(const scalar<uint8_t> a,
                                          const scalar<uint8_t> b) {
  return scalar<uint8_t>(SIMD_MIN(SIMD_MAX(0, a.raw + b.raw), 255));
}
SIMD_INLINE scalar<uint16_t> saturated_add(const scalar<uint16_t> a,
                                           const scalar<uint16_t> b) {
  return scalar<uint16_t>(SIMD_MIN(SIMD_MAX(0, a.raw + b.raw), 65535));
}

// Signed
SIMD_INLINE scalar<int8_t> saturated_add(const scalar<int8_t> a,
                                         const scalar<int8_t> b) {
  return scalar<int8_t>(SIMD_MIN(SIMD_MAX(-128, a.raw + b.raw), 127));
}
SIMD_INLINE scalar<int16_t> saturated_add(const scalar<int16_t> a,
                                          const scalar<int16_t> b) {
  return scalar<int16_t>(SIMD_MIN(SIMD_MAX(-32768, a.raw + b.raw), 32767));
}

// ------------------------------ Saturating subtraction

// Returns a - b clamped to the destination range.

// Unsigned
SIMD_INLINE scalar<uint8_t> saturated_subtract(const scalar<uint8_t> a,
                                               const scalar<uint8_t> b) {
  return scalar<uint8_t>(SIMD_MIN(SIMD_MAX(0, a.raw - b.raw), 255));
}
SIMD_INLINE scalar<uint16_t> saturated_subtract(const scalar<uint16_t> a,
                                                const scalar<uint16_t> b) {
  return scalar<uint16_t>(SIMD_MIN(SIMD_MAX(0, a.raw - b.raw), 65535));
}

// Signed
SIMD_INLINE scalar<int8_t> saturated_subtract(const scalar<int8_t> a,
                                              const scalar<int8_t> b) {
  return scalar<int8_t>(SIMD_MIN(SIMD_MAX(-128, a.raw - b.raw), 127));
}
SIMD_INLINE scalar<int16_t> saturated_subtract(const scalar<int16_t> a,
                                               const scalar<int16_t> b) {
  return scalar<int16_t>(SIMD_MIN(SIMD_MAX(-32768, a.raw - b.raw), 32767));
}

// ------------------------------ Average

// Returns (a + b + 1) / 2

SIMD_INLINE scalar<uint8_t> average_round(const scalar<uint8_t> a,
                                          const scalar<uint8_t> b) {
  return scalar<uint8_t>((a.raw + b.raw + 1) / 2);
}
SIMD_INLINE scalar<uint16_t> average_round(const scalar<uint16_t> a,
                                           const scalar<uint16_t> b) {
  return scalar<uint16_t>((a.raw + b.raw + 1) / 2);
}

// ------------------------------ Absolute value

template <typename T>
SIMD_INLINE scalar<T> abs(const scalar<T> a) {
  const T i = a.raw;
  return (i >= 0 || i == LimitsMin<T>()) ? a : scalar<T>(-i);
}

// ------------------------------ min/max

template <typename T>
SIMD_INLINE scalar<T> min(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(SIMD_MIN(a.raw, b.raw));
}

template <typename T>
SIMD_INLINE scalar<T> max(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(SIMD_MAX(a.raw, b.raw));
}

// Returns the closest value to v within [lo, hi].
template <typename T>
SIMD_INLINE scalar<T> clamp(const scalar<T> v, const scalar<T> lo,
                            const scalar<T> hi) {
  return min(max(lo, v), hi);
}

// ------------------------------ Floating-point negate

SIMD_INLINE scalar<float> neg(const scalar<float> v) {
  const Scalar<float> df;
  const Scalar<uint32_t> du;
  const auto sign = bit_cast(df, set1(du, 0x80000000u));
  return v ^ sign;
}

SIMD_INLINE scalar<double> neg(const scalar<double> v) {
  const Scalar<double> df;
  const Scalar<uint64_t> du;
  const auto sign = bit_cast(df, set1(du, 0x8000000000000000ull));
  return v ^ sign;
}

// ------------------------------ mul/div

template <typename T>
SIMD_INLINE scalar<T> operator*(const scalar<T> a, const scalar<T> b) {
  if (IsFloat<T>()) {
    return scalar<T>(static_cast<T>(double(a.raw) * b.raw));
  } else if (IsSigned<T>()) {
    return scalar<T>(static_cast<T>(int64_t(a.raw) * b.raw));
  } else {
    return scalar<T>(static_cast<T>(uint64_t(a.raw) * b.raw));
  }
}

template <typename T>
SIMD_INLINE scalar<T> operator/(const scalar<T> a, const scalar<T> b) {
  return scalar<T>(a.raw / b.raw);
}

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns the upper 16 bits of a * b in each lane.
SIMD_INLINE scalar<int16_t> mul_high(const scalar<int16_t> a,
                                     const scalar<int16_t> b) {
  return scalar<int16_t>((a.raw * b.raw) >> 16);
}
SIMD_INLINE scalar<uint16_t> mul_high(const scalar<uint16_t> a,
                                      const scalar<uint16_t> b) {
  // Cast to uint32_t first to prevent overflow. Otherwise the result of
  // uint16_t * uint16_t is in "int" which may overflow. In practice the result
  // is the same but this way it is also defined.
  return scalar<uint16_t>(
      (static_cast<uint32_t>(a.raw) * static_cast<uint32_t>(b.raw)) >> 16);
}

}  // namespace ext

// Returns (((a * b) >> 14) + 1) >> 1.
SIMD_INLINE scalar<int16_t> mul_high_round(const scalar<int16_t> a,
                                           const scalar<int16_t> b) {
  const int rounded = ((a.raw * b.raw) + (1 << 14)) >> 15;
  const int clamped = SIMD_MIN(SIMD_MAX(-32768, rounded), 32767);
  return scalar<int16_t>(clamped);
}

// Multiplies even lanes (0, 2 ..) and returns the double-wide result.
SIMD_INLINE scalar<int64_t> mul_even(const scalar<int32_t> a,
                                     const scalar<int32_t> b) {
  const int64_t a64 = a.raw;
  return scalar<int64_t>(a64 * b.raw);
}
SIMD_INLINE scalar<uint64_t> mul_even(const scalar<uint32_t> a,
                                      const scalar<uint32_t> b) {
  const uint64_t a64 = a.raw;
  return scalar<uint64_t>(a64 * b.raw);
}

// Approximate reciprocal
SIMD_INLINE scalar<float> approximate_reciprocal(const scalar<float> v) {
  return scalar<float>(1.0f / v.raw);
}

// ------------------------------ Floating-point multiply-add variants

template <typename T>
SIMD_INLINE scalar<T> mul_add(const scalar<T> mul, const scalar<T> x,
                              const scalar<T> add) {
  return mul * x + add;
}

template <typename T>
SIMD_INLINE scalar<T> nmul_add(const scalar<T> mul, const scalar<T> x,
                               const scalar<T> add) {
  return add - mul * x;
}

template <typename T>
SIMD_INLINE scalar<T> fadd(const scalar<T> x, const scalar<T> k1,
                           const scalar<T> add) {
  return x + add;
}

template <typename T>
SIMD_INLINE scalar<T> fsub(const scalar<T> x, const scalar<T> k1,
                           const scalar<T> sub) {
  return x - sub;
}

// (parameter order swapped)
template <typename T>
SIMD_INLINE scalar<T> fnadd(const scalar<T> sub, const scalar<T> k1,
                            const scalar<T> x) {
  return x - sub;
}

// Slightly more expensive on ARM (extra negate)
namespace ext {

template <typename T>
SIMD_INLINE scalar<T> mul_subtract(const scalar<T> mul, const scalar<T> x,
                                   const scalar<T> sub) {
  return mul * x - sub;
}

template <typename T>
SIMD_INLINE scalar<T> nmul_subtract(const scalar<T> mul, const scalar<T> x,
                                    const scalar<T> sub) {
  return neg(mul) * x - sub;
}

}  // namespace ext

// ------------------------------ Floating-point square root

// Approximate reciprocal square root
SIMD_INLINE scalar<float> approximate_reciprocal_sqrt(const scalar<float> v) {
  float f = v.raw;
  const float half = f * 0.5f;
  uint32_t bits;
  CopyBytes<4>(&f, &bits);
  // Initial guess based on log2(f)
  bits = 0x5F3759DF - (bits >> 1);
  CopyBytes<4>(&bits, &f);
  // One Newton-Raphson iteration
  return scalar<float>(f * (1.5f - (half * f * f)));
}

// Square root
SIMD_INLINE scalar<float> sqrt(const scalar<float> v) {
  return approximate_reciprocal_sqrt(v) * v;
}
SIMD_INLINE scalar<double> sqrt(const scalar<double> v) {
  return scalar<double>(sqrt(scalar<float>(v.raw)).raw);
}

// ------------------------------ Floating-point rounding

// Approximation of round-to-nearest for numbers representable as integers.
SIMD_INLINE scalar<float> round(const scalar<float> v) {
  const float bias = v.raw < 0.0f ? -0.5f : 0.5f;
  return scalar<float>(static_cast<int32_t>(v.raw + bias));
}
SIMD_INLINE scalar<double> round(const scalar<double> v) {
  const double bias = v.raw < 0.0 ? -0.5 : 0.5;
  return scalar<double>(static_cast<int64_t>(v.raw + bias));
}

SIMD_INLINE scalar<float> trunc(const scalar<float> v) {
  return scalar<float>(static_cast<int32_t>(v.raw));
}
SIMD_INLINE scalar<double> trunc(const scalar<double> v) {
  return scalar<double>(static_cast<int64_t>(v.raw));
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
SIMD_INLINE scalar<float> ceil(const scalar<float> v) {
  return Ceiling<float, uint32_t, 23, 8>(v);
}
SIMD_INLINE scalar<double> ceil(const scalar<double> v) {
  return Ceiling<double, uint64_t, 52, 11>(v);
}

// Toward -infinity, aka floor
SIMD_INLINE scalar<float> floor(const scalar<float> v) {
  return Floor<float, uint32_t, 23, 8>(v);
}
SIMD_INLINE scalar<double> floor(const scalar<double> v) {
  return Floor<double, uint64_t, 52, 11>(v);
}

// ================================================== COMPARE

// Comparisons fill a lane with 1-bits if the condition is true, else 0.
template <typename T>
scalar<T> ComparisonResult(const bool result) {
  T ret;
  SetBytes(result ? 0xFF : 0, &ret);
  return scalar<T>(ret);
}

template <typename T>
SIMD_INLINE scalar<T> operator==(const scalar<T> a, const scalar<T> b) {
  return ComparisonResult<T>(a.raw == b.raw);
}

template <typename T>
SIMD_INLINE scalar<T> operator<(const scalar<T> a, const scalar<T> b) {
  return ComparisonResult<T>(a.raw < b.raw);
}
template <typename T>
SIMD_INLINE scalar<T> operator>(const scalar<T> a, const scalar<T> b) {
  return ComparisonResult<T>(a.raw > b.raw);
}

template <typename T>
SIMD_INLINE scalar<T> operator<=(const scalar<T> a, const scalar<T> b) {
  return ComparisonResult<T>(a.raw <= b.raw);
}
template <typename T>
SIMD_INLINE scalar<T> operator>=(const scalar<T> a, const scalar<T> b) {
  return ComparisonResult<T>(a.raw >= b.raw);
}

// ================================================== MEMORY

// ------------------------------ Load

template <typename T>
SIMD_INLINE scalar<T> load(Scalar<T> /* tag */,
                           const T* SIMD_RESTRICT aligned) {
  T t;
  CopyBytes<sizeof(T)>(aligned, &t);
  return scalar<T>(t);
}

template <typename T>
SIMD_INLINE scalar<T> load_u(Scalar<T> d, const T* SIMD_RESTRICT p) {
  return load(d, p);
}

// In some use cases, "load single lane" is sufficient; otherwise avoid this.
template <typename T>
SIMD_INLINE scalar<T> load_dup128(Scalar<T> d, const T* SIMD_RESTRICT aligned) {
  return load(d, aligned);
}

// ------------------------------ Store

template <typename T>
SIMD_INLINE void store(const scalar<T> v, Scalar<T> /* tag */,
                       T* SIMD_RESTRICT aligned) {
  CopyBytes<sizeof(T)>(&v.raw, aligned);
}

template <typename T>
SIMD_INLINE void store_u(const scalar<T> v, Scalar<T> d, T* SIMD_RESTRICT p) {
  return store(v, d, p);
}

// ------------------------------ "Non-temporal" stores

template <typename T>
SIMD_INLINE void stream(const scalar<T> v, Scalar<T> d,
                        T* SIMD_RESTRICT aligned) {
  return store(v, d, aligned);
}

// ------------------------------ Gather

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

template <typename T, typename Offset>
SIMD_INLINE scalar<T> gather_offset(Scalar<T> d, const T* base,
                                    const scalar<Offset> offset) {
  static_assert(sizeof(T) == sizeof(Offset), "SVE requires same size base/ofs");
  const uintptr_t addr = reinterpret_cast<uintptr_t>(base) + offset.raw;
  return load(d, reinterpret_cast<const T*>(addr));
}

template <typename T, typename Index>
SIMD_INLINE scalar<T> gather_index(Scalar<T> d, const T* SIMD_RESTRICT base,
                                   const scalar<Index> index) {
  static_assert(sizeof(T) == sizeof(Index), "SVE requires same size base/idx");
  return load(d, base + index.raw);
}

}  // namespace ext

// ================================================== CONVERT

template <typename FromT, typename ToT>
SIMD_INLINE scalar<ToT> convert_to(Scalar<ToT> /* tag */,
                                   const scalar<FromT> from) {
  return scalar<ToT>(static_cast<ToT>(from.raw));
}

SIMD_INLINE scalar<float> convert_to(Scalar<float> /* tag */,
                                     const scalar<int32_t> v) {
  return scalar<float>(v.raw);
}

// Truncates (rounds toward zero).
SIMD_INLINE scalar<int32_t> convert_to(Scalar<int32_t> /* tag */,
                                       const scalar<float> v) {
  return scalar<int32_t>(static_cast<int>(v.raw));
}

SIMD_INLINE scalar<uint32_t> u32_from_u8(const scalar<uint8_t> v) {
  return convert_to(Scalar<uint32_t>(), v);
}

SIMD_INLINE scalar<uint8_t> u8_from_u32(const scalar<uint32_t> v) {
  return convert_to(Scalar<uint8_t>(), v);
}

// Approximation of round-to-nearest for numbers representable as int32_t.
SIMD_INLINE scalar<int32_t> nearest_int(const scalar<float> v) {
  const float f = v.raw;
  const float bias = f < 0.0f ? -0.5f : 0.5f;
  return scalar<int32_t>(static_cast<int>(f + bias));
}

// ================================================== SWIZZLE

// Unsupported: shift_*_bytes, combine_shift_right_bytes, interleave_*,
// other_half, shuffle_*, sums_of_u8x8, sum_of_lanes - these require more than
// one lane and/or actual 128-bit vectors.

// ------------------------------ Broadcast/splat any lane

template <int kLane, typename T>
SIMD_INLINE scalar<T> broadcast(const scalar<T> v) {
  static_assert(kLane == 0, "Scalar only has one lane");
  return v;
}

// ------------------------------ Zip/unpack

SIMD_INLINE scalar<uint16_t> zip_lo(const scalar<uint8_t> a,
                                    const scalar<uint8_t> b) {
  return scalar<uint16_t>((uint32_t(b.raw) << 8) + a.raw);
}
SIMD_INLINE scalar<uint32_t> zip_lo(const scalar<uint16_t> a,
                                    const scalar<uint16_t> b) {
  return scalar<uint32_t>((uint32_t(b.raw) << 16) + a.raw);
}
SIMD_INLINE scalar<uint64_t> zip_lo(const scalar<uint32_t> a,
                                    const scalar<uint32_t> b) {
  return scalar<uint64_t>((uint64_t(b.raw) << 32) + a.raw);
}
SIMD_INLINE scalar<int16_t> zip_lo(const scalar<int8_t> a,
                                   const scalar<int8_t> b) {
  return scalar<int16_t>((uint32_t(b.raw) << 8) + a.raw);
}
SIMD_INLINE scalar<int32_t> zip_lo(const scalar<int16_t> a,
                                   const scalar<int16_t> b) {
  return scalar<int32_t>((uint32_t(b.raw) << 16) + a.raw);
}
SIMD_INLINE scalar<int64_t> zip_lo(const scalar<int32_t> a,
                                   const scalar<int32_t> b) {
  return scalar<int64_t>((uint64_t(b.raw) << 32) + a.raw);
}

template <typename T>
SIMD_INLINE auto zip_hi(const scalar<T> a, const scalar<T> b)
    -> decltype(zip_lo(a, b)) {
  return zip_lo(a, b);
}

// ------------------------------ Parts

template <typename T, typename T2>
SIMD_INLINE scalar<T> set_lane(const T2 t) {
  return scalar<T>(t);
}

template <typename T>
SIMD_INLINE T get_lane(const scalar<T> v) {
  return v.raw;
}

template <typename T>
SIMD_INLINE scalar<T> any_part(Scalar<T> /* tag */, const scalar<T> v) {
  return v;
}

template <int kLane, typename T>
SIMD_INLINE scalar<T> broadcast_part(Scalar<T> /* tag */, const scalar<T> v) {
  static_assert(kLane == 0, "Invalid kLane");
  return v;
}

// ================================================== MISC

// "Extensions": useful but not quite performance-portable operations. We add
// functions to this namespace in multiple places.
namespace ext {

// Returns a bit array of the most significant bit of each byte in "v", i.e.
// sum_i=0..15 of (v[i] >> 7) << i; v[0] is the least-significant byte of "v".
// This is useful for testing/branching based on comparison results.
SIMD_INLINE uint64_t movemask(const scalar<uint8_t> v) { return v.raw >> 7; }

// Returns the most significant bit of each float/double lane (see above).
SIMD_INLINE uint64_t movemask(const scalar<float> v) {
  // Cannot return (v < 0) because +0.0 == -0.0.
  const Scalar<uint32_t> du;
  const auto bits = bit_cast(du, v);
  return get_lane(shift_right<31>(bits));
}
SIMD_INLINE uint64_t movemask(const scalar<double> v) {
  // Cannot return (v < 0) because +0.0 == -0.0.
  const Scalar<uint64_t> du;
  const auto bits = bit_cast(du, v);
  return get_lane(shift_right<63>(bits));
}

// Returns whether all lanes are equal to zero. Supported for all integer T.
template <typename T>
SIMD_INLINE bool all_zero(const scalar<T> v) {
  return v.raw == 0;
}

// Sum of all lanes, i.e. the only one.
template <typename T>
SIMD_INLINE scalar<T> sum_of_lanes(const scalar<T> v0) {
  return v0;
}

}  // namespace ext
}  // namespace jxl

#endif  // HIGHWAY_SCALAR_H_
