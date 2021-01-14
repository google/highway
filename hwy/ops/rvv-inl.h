// Copyright 2021 Google LLC
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

// RISC-V V vectors (length not known at compile time).
// External include guard in highway.h - see comment there.

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "hwy/ops/shared-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Private shorthand so that definitions of Set etc. fit on a single line.
using DU8M1 = Simd<uint8_t, HWY_LANES(uint8_t)>;
using DU8M2 = Simd<uint8_t, HWY_LANES(uint8_t) * 2>;
using DU8M4 = Simd<uint8_t, HWY_LANES(uint8_t) * 4>;
using DU8M8 = Simd<uint8_t, HWY_LANES(uint8_t) * 8>;
using DU16M1 = Simd<uint16_t, HWY_LANES(uint16_t)>;
using DU16M2 = Simd<uint16_t, HWY_LANES(uint16_t) * 2>;
using DU16M4 = Simd<uint16_t, HWY_LANES(uint16_t) * 4>;
using DU16M8 = Simd<uint16_t, HWY_LANES(uint16_t) * 8>;
using DU32M1 = Simd<uint32_t, HWY_LANES(uint32_t)>;
using DU32M2 = Simd<uint32_t, HWY_LANES(uint32_t) * 2>;
using DU32M4 = Simd<uint32_t, HWY_LANES(uint32_t) * 4>;
using DU32M8 = Simd<uint32_t, HWY_LANES(uint32_t) * 8>;
using DU64M1 = Simd<uint64_t, HWY_LANES(uint64_t)>;
using DU64M2 = Simd<uint64_t, HWY_LANES(uint64_t) * 2>;
using DU64M4 = Simd<uint64_t, HWY_LANES(uint64_t) * 4>;
using DU64M8 = Simd<uint64_t, HWY_LANES(uint64_t) * 8>;

using DI8M1 = Simd<int8_t, HWY_LANES(int8_t)>;
using DI8M2 = Simd<int8_t, HWY_LANES(int8_t) * 2>;
using DI8M4 = Simd<int8_t, HWY_LANES(int8_t) * 4>;
using DI8M8 = Simd<int8_t, HWY_LANES(int8_t) * 8>;
using DI16M1 = Simd<int16_t, HWY_LANES(int16_t)>;
using DI16M2 = Simd<int16_t, HWY_LANES(int16_t) * 2>;
using DI16M4 = Simd<int16_t, HWY_LANES(int16_t) * 4>;
using DI16M8 = Simd<int16_t, HWY_LANES(int16_t) * 8>;
using DI32M1 = Simd<int32_t, HWY_LANES(int32_t)>;
using DI32M2 = Simd<int32_t, HWY_LANES(int32_t) * 2>;
using DI32M4 = Simd<int32_t, HWY_LANES(int32_t) * 4>;
using DI32M8 = Simd<int32_t, HWY_LANES(int32_t) * 8>;
using DI64M1 = Simd<int64_t, HWY_LANES(int64_t)>;
using DI64M2 = Simd<int64_t, HWY_LANES(int64_t) * 2>;
using DI64M4 = Simd<int64_t, HWY_LANES(int64_t) * 4>;
using DI64M8 = Simd<int64_t, HWY_LANES(int64_t) * 8>;

using DF32M1 = Simd<float, HWY_LANES(float)>;
using DF32M2 = Simd<float, HWY_LANES(float) * 2>;
using DF32M4 = Simd<float, HWY_LANES(float) * 4>;
using DF32M8 = Simd<float, HWY_LANES(float) * 8>;
using DF64M1 = Simd<double, HWY_LANES(double)>;
using DF64M2 = Simd<double, HWY_LANES(double) * 2>;
using DF64M4 = Simd<double, HWY_LANES(double) * 4>;
using DF64M8 = Simd<double, HWY_LANES(double) * 8>;

using VU8M1 = vuint8m1_t;
using VU8M2 = vuint8m2_t;
using VU8M4 = vuint8m4_t;
using VU8M8 = vuint8m8_t;
using VU16M1 = vuint16m1_t;
using VU16M2 = vuint16m2_t;
using VU16M4 = vuint16m4_t;
using VU16M8 = vuint16m8_t;
using VU32M1 = vuint32m1_t;
using VU32M2 = vuint32m2_t;
using VU32M4 = vuint32m4_t;
using VU32M8 = vuint32m8_t;
using VU64M1 = vuint64m1_t;
using VU64M2 = vuint64m2_t;
using VU64M4 = vuint64m4_t;
using VU64M8 = vuint64m8_t;

using VI8M1 = vint8m1_t;
using VI8M2 = vint8m2_t;
using VI8M4 = vint8m4_t;
using VI8M8 = vint8m8_t;
using VI16M1 = vint16m1_t;
using VI16M2 = vint16m2_t;
using VI16M4 = vint16m4_t;
using VI16M8 = vint16m8_t;
using VI32M1 = vint32m1_t;
using VI32M2 = vint32m2_t;
using VI32M4 = vint32m4_t;
using VI32M8 = vint32m8_t;
using VI64M1 = vint64m1_t;
using VI64M2 = vint64m2_t;
using VI64M4 = vint64m4_t;
using VI64M8 = vint64m8_t;

using VF32M1 = vfloat32m1_t;
using VF32M2 = vfloat32m2_t;
using VF32M4 = vfloat32m4_t;
using VF32M8 = vfloat32m8_t;
using VF64M1 = vfloat64m1_t;
using VF64M2 = vfloat64m2_t;
using VF64M4 = vfloat64m4_t;
using VF64M8 = vfloat64m8_t;

// For all combinations of RVV lane types and register groups. We only use these
// X macros for non-function template specializations so that definitions are
// searchable and code remains readable/debuggable.
#define HWY_RVV_FOREACH_LSHIFT(X_MACRO, RVV_TYPE, TYPE) \
  X_MACRO(RVV_TYPE, TYPE, 0, m1)                        \
  X_MACRO(RVV_TYPE, TYPE, 1, m2)                        \
  X_MACRO(RVV_TYPE, TYPE, 2, m4)                        \
  X_MACRO(RVV_TYPE, TYPE, 3, m8)

#define HWY_RVV_FOREACH(X_MACRO)                    \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, uint8, uint8_t)   \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, uint16, uint16_t) \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, uint32, uint32_t) \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, uint64, uint64_t) \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, int8, int8_t)     \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, int16, int16_t)   \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, int32, int32_t)   \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, int64, int64_t)   \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, float32, float)   \
  HWY_RVV_FOREACH_LSHIFT(X_MACRO, float64, double)

template <class V>
struct DFromV_t {};

// TODO(janwas): do we want fractional LMUL (negative LShift)?
// Mixed-precision code can use LMUL>1 and that should be enough unless they
// need many registers.
template <typename T, int LShift>
using Desc = Simd<T, LShift >= 0 ? (HWY_LANES(T) << LShift)
                                 : (HWY_LANES(T) >> (-LShift))>;

#define HWY_SPECIALIZE(RVV_TYPE, TYPE, LSHIFT, M_SUFFIX) \
  template <>                                            \
  struct DFromV_t<v##RVV_TYPE##M_SUFFIX##_t> {           \
    using type = Desc<TYPE, LSHIFT>;                     \
  };
HWY_RVV_FOREACH(HWY_SPECIALIZE)
#undef HWY_SPECIALIZE

template <class V>
using DFromV = typename DFromV_t<V>::type;

template <class V>
using TFromV = TFromD<DFromV<V>>;

#define HWY_IF_UNSIGNED_V(V) hwy::EnableIf<!IsSigned<TFromV<V>>()>* = nullptr
#define HWY_IF_SIGNED_V(V) \
  hwy::EnableIf<IsSigned<TFromV<V>>() && !IsFloat<TFromV<V>>()>* = nullptr
#define HWY_IF_FLOAT_V(V) hwy::EnableIf<IsFloat<TFromV<V>>()>* = nullptr

// ------------------------------ Lanes

// WARNING: this should just query VLMAX/sizeof(T), but actually changes it!
// vlenb is not exposed through intrinsics and vreadvl is not VLMAX.
HWY_API size_t Lanes(DU8M1 /* d */) { return vsetvlmax_e8m1(); }
HWY_API size_t Lanes(DU8M2 /* d */) { return vsetvlmax_e8m2(); }
HWY_API size_t Lanes(DU8M4 /* d */) { return vsetvlmax_e8m4(); }
HWY_API size_t Lanes(DU8M8 /* d */) { return vsetvlmax_e8m8(); }

HWY_API size_t Lanes(DU16M1 /* d */) { return vsetvlmax_e16m1(); }
HWY_API size_t Lanes(DU16M2 /* d */) { return vsetvlmax_e16m2(); }
HWY_API size_t Lanes(DU16M4 /* d */) { return vsetvlmax_e16m4(); }
HWY_API size_t Lanes(DU16M8 /* d */) { return vsetvlmax_e16m8(); }

HWY_API size_t Lanes(DU32M1 /* d */) { return vsetvlmax_e32m1(); }
HWY_API size_t Lanes(DU32M2 /* d */) { return vsetvlmax_e32m2(); }
HWY_API size_t Lanes(DU32M4 /* d */) { return vsetvlmax_e32m4(); }
HWY_API size_t Lanes(DU32M8 /* d */) { return vsetvlmax_e32m8(); }

HWY_API size_t Lanes(DU64M1 /* d */) { return vsetvlmax_e64m1(); }
HWY_API size_t Lanes(DU64M2 /* d */) { return vsetvlmax_e64m2(); }
HWY_API size_t Lanes(DU64M4 /* d */) { return vsetvlmax_e64m4(); }
HWY_API size_t Lanes(DU64M8 /* d */) { return vsetvlmax_e64m8(); }

HWY_API size_t Lanes(DI8M1 /* d */) { return vsetvlmax_e8m1(); }
HWY_API size_t Lanes(DI8M2 /* d */) { return vsetvlmax_e8m2(); }
HWY_API size_t Lanes(DI8M4 /* d */) { return vsetvlmax_e8m4(); }
HWY_API size_t Lanes(DI8M8 /* d */) { return vsetvlmax_e8m8(); }

HWY_API size_t Lanes(DI16M1 /* d */) { return vsetvlmax_e16m1(); }
HWY_API size_t Lanes(DI16M2 /* d */) { return vsetvlmax_e16m2(); }
HWY_API size_t Lanes(DI16M4 /* d */) { return vsetvlmax_e16m4(); }
HWY_API size_t Lanes(DI16M8 /* d */) { return vsetvlmax_e16m8(); }

HWY_API size_t Lanes(DI32M1 /* d */) { return vsetvlmax_e32m1(); }
HWY_API size_t Lanes(DI32M2 /* d */) { return vsetvlmax_e32m2(); }
HWY_API size_t Lanes(DI32M4 /* d */) { return vsetvlmax_e32m4(); }
HWY_API size_t Lanes(DI32M8 /* d */) { return vsetvlmax_e32m8(); }

HWY_API size_t Lanes(DI64M1 /* d */) { return vsetvlmax_e64m1(); }
HWY_API size_t Lanes(DI64M2 /* d */) { return vsetvlmax_e64m2(); }
HWY_API size_t Lanes(DI64M4 /* d */) { return vsetvlmax_e64m4(); }
HWY_API size_t Lanes(DI64M8 /* d */) { return vsetvlmax_e64m8(); }

HWY_API size_t Lanes(DF32M1 /* d */) { return vsetvlmax_e32m1(); }
HWY_API size_t Lanes(DF32M2 /* d */) { return vsetvlmax_e32m2(); }
HWY_API size_t Lanes(DF32M4 /* d */) { return vsetvlmax_e32m4(); }
HWY_API size_t Lanes(DF32M8 /* d */) { return vsetvlmax_e32m8(); }

HWY_API size_t Lanes(DF64M1 /* d */) { return vsetvlmax_e64m1(); }
HWY_API size_t Lanes(DF64M2 /* d */) { return vsetvlmax_e64m2(); }
HWY_API size_t Lanes(DF64M4 /* d */) { return vsetvlmax_e64m4(); }
HWY_API size_t Lanes(DF64M8 /* d */) { return vsetvlmax_e64m8(); }

// ------------------------------ Zero

HWY_API VU8M1 Zero(DU8M1 /* d */) { return vzero_u8m1(); }
HWY_API VU8M2 Zero(DU8M2 /* d */) { return vzero_u8m2(); }
HWY_API VU8M4 Zero(DU8M4 /* d */) { return vzero_u8m4(); }
HWY_API VU8M8 Zero(DU8M8 /* d */) { return vzero_u8m8(); }

HWY_API VI8M1 Zero(DI8M1 /* d */) { return vzero_i8m1(); }
HWY_API VI8M2 Zero(DI8M2 /* d */) { return vzero_i8m2(); }
HWY_API VI8M4 Zero(DI8M4 /* d */) { return vzero_i8m4(); }
HWY_API VI8M8 Zero(DI8M8 /* d */) { return vzero_i8m8(); }

HWY_API VU16M1 Zero(DU16M1 /* d */) { return vzero_u16m1(); }
HWY_API VU16M2 Zero(DU16M2 /* d */) { return vzero_u16m2(); }
HWY_API VU16M4 Zero(DU16M4 /* d */) { return vzero_u16m4(); }
HWY_API VU16M8 Zero(DU16M8 /* d */) { return vzero_u16m8(); }

HWY_API VI16M1 Zero(DI16M1 /* d */) { return vzero_i16m1(); }
HWY_API VI16M2 Zero(DI16M2 /* d */) { return vzero_i16m2(); }
HWY_API VI16M4 Zero(DI16M4 /* d */) { return vzero_i16m4(); }
HWY_API VI16M8 Zero(DI16M8 /* d */) { return vzero_i16m8(); }

HWY_API VU32M1 Zero(DU32M1 /* d */) { return vzero_u32m1(); }
HWY_API VU32M2 Zero(DU32M2 /* d */) { return vzero_u32m2(); }
HWY_API VU32M4 Zero(DU32M4 /* d */) { return vzero_u32m4(); }
HWY_API VU32M8 Zero(DU32M8 /* d */) { return vzero_u32m8(); }

HWY_API VI32M1 Zero(DI32M1 /* d */) { return vzero_i32m1(); }
HWY_API VI32M2 Zero(DI32M2 /* d */) { return vzero_i32m2(); }
HWY_API VI32M4 Zero(DI32M4 /* d */) { return vzero_i32m4(); }
HWY_API VI32M8 Zero(DI32M8 /* d */) { return vzero_i32m8(); }

HWY_API VU64M1 Zero(DU64M1 /* d */) { return vzero_u64m1(); }
HWY_API VU64M2 Zero(DU64M2 /* d */) { return vzero_u64m2(); }
HWY_API VU64M4 Zero(DU64M4 /* d */) { return vzero_u64m4(); }
HWY_API VU64M8 Zero(DU64M8 /* d */) { return vzero_u64m8(); }

HWY_API VI64M1 Zero(DI64M1 /* d */) { return vzero_i64m1(); }
HWY_API VI64M2 Zero(DI64M2 /* d */) { return vzero_i64m2(); }
HWY_API VI64M4 Zero(DI64M4 /* d */) { return vzero_i64m4(); }
HWY_API VI64M8 Zero(DI64M8 /* d */) { return vzero_i64m8(); }

HWY_API VF32M1 Zero(DF32M1 /* d */) { return vzero_f32m1(); }
HWY_API VF32M2 Zero(DF32M2 /* d */) { return vzero_f32m2(); }
HWY_API VF32M4 Zero(DF32M4 /* d */) { return vzero_f32m4(); }
HWY_API VF32M8 Zero(DF32M8 /* d */) { return vzero_f32m8(); }

HWY_API VF64M1 Zero(DF64M1 /* d */) { return vzero_f64m1(); }
HWY_API VF64M2 Zero(DF64M2 /* d */) { return vzero_f64m2(); }
HWY_API VF64M4 Zero(DF64M4 /* d */) { return vzero_f64m4(); }
HWY_API VF64M8 Zero(DF64M8 /* d */) { return vzero_f64m8(); }

template <class D>
using VFromD = decltype(Zero(D()));

// ------------------------------ Set

HWY_API VU8M1 Set(DU8M1 /* d */, uint8_t t) { return vmv_v_x_u8m1(t); }
HWY_API VU8M2 Set(DU8M2 /* d */, uint8_t t) { return vmv_v_x_u8m2(t); }
HWY_API VU8M4 Set(DU8M4 /* d */, uint8_t t) { return vmv_v_x_u8m4(t); }
HWY_API VU8M8 Set(DU8M8 /* d */, uint8_t t) { return vmv_v_x_u8m8(t); }

HWY_API VU16M1 Set(DU16M1 /* d */, uint16_t t) { return vmv_v_x_u16m1(t); }
HWY_API VU16M2 Set(DU16M2 /* d */, uint16_t t) { return vmv_v_x_u16m2(t); }
HWY_API VU16M4 Set(DU16M4 /* d */, uint16_t t) { return vmv_v_x_u16m4(t); }
HWY_API VU16M8 Set(DU16M8 /* d */, uint16_t t) { return vmv_v_x_u16m8(t); }

HWY_API VU32M1 Set(DU32M1 /* d */, uint32_t t) { return vmv_v_x_u32m1(t); }
HWY_API VU32M2 Set(DU32M2 /* d */, uint32_t t) { return vmv_v_x_u32m2(t); }
HWY_API VU32M4 Set(DU32M4 /* d */, uint32_t t) { return vmv_v_x_u32m4(t); }
HWY_API VU32M8 Set(DU32M8 /* d */, uint32_t t) { return vmv_v_x_u32m8(t); }

HWY_API VU64M1 Set(DU64M1 /* d */, uint64_t t) { return vmv_v_x_u64m1(t); }
HWY_API VU64M2 Set(DU64M2 /* d */, uint64_t t) { return vmv_v_x_u64m2(t); }
HWY_API VU64M4 Set(DU64M4 /* d */, uint64_t t) { return vmv_v_x_u64m4(t); }
HWY_API VU64M8 Set(DU64M8 /* d */, uint64_t t) { return vmv_v_x_u64m8(t); }

HWY_API VI8M1 Set(DI8M1 /* d */, int8_t t) { return vmv_v_x_i8m1(t); }
HWY_API VI8M2 Set(DI8M2 /* d */, int8_t t) { return vmv_v_x_i8m2(t); }
HWY_API VI8M4 Set(DI8M4 /* d */, int8_t t) { return vmv_v_x_i8m4(t); }
HWY_API VI8M8 Set(DI8M8 /* d */, int8_t t) { return vmv_v_x_i8m8(t); }

HWY_API VI16M1 Set(DI16M1 /* d */, int16_t t) { return vmv_v_x_i16m1(t); }
HWY_API VI16M2 Set(DI16M2 /* d */, int16_t t) { return vmv_v_x_i16m2(t); }
HWY_API VI16M4 Set(DI16M4 /* d */, int16_t t) { return vmv_v_x_i16m4(t); }
HWY_API VI16M8 Set(DI16M8 /* d */, int16_t t) { return vmv_v_x_i16m8(t); }

HWY_API VI32M1 Set(DI32M1 /* d */, int32_t t) { return vmv_v_x_i32m1(t); }
HWY_API VI32M2 Set(DI32M2 /* d */, int32_t t) { return vmv_v_x_i32m2(t); }
HWY_API VI32M4 Set(DI32M4 /* d */, int32_t t) { return vmv_v_x_i32m4(t); }
HWY_API VI32M8 Set(DI32M8 /* d */, int32_t t) { return vmv_v_x_i32m8(t); }

HWY_API VI64M1 Set(DI64M1 /* d */, int64_t t) { return vmv_v_x_i64m1(t); }
HWY_API VI64M2 Set(DI64M2 /* d */, int64_t t) { return vmv_v_x_i64m2(t); }
HWY_API VI64M4 Set(DI64M4 /* d */, int64_t t) { return vmv_v_x_i64m4(t); }
HWY_API VI64M8 Set(DI64M8 /* d */, int64_t t) { return vmv_v_x_i64m8(t); }

HWY_API VF32M1 Set(DF32M1 /* d */, float t) { return vfmv_v_f_f32m1(t); }
HWY_API VF32M2 Set(DF32M2 /* d */, float t) { return vfmv_v_f_f32m2(t); }
HWY_API VF32M4 Set(DF32M4 /* d */, float t) { return vfmv_v_f_f32m4(t); }
HWY_API VF32M8 Set(DF32M8 /* d */, float t) { return vfmv_v_f_f32m8(t); }

HWY_API VF64M1 Set(DF64M1 /* d */, double t) { return vfmv_v_f_f64m1(t); }
HWY_API VF64M2 Set(DF64M2 /* d */, double t) { return vfmv_v_f_f64m2(t); }
HWY_API VF64M4 Set(DF64M4 /* d */, double t) { return vfmv_v_f_f64m4(t); }
HWY_API VF64M8 Set(DF64M8 /* d */, double t) { return vfmv_v_f_f64m8(t); }

// ------------------------------ Undefined

// RVV vundefined is 'poisoned' such that even XORing a _variable_ initialized
// by it gives unpredictable results. It should only be used for maskoff, so
// keep it internal. For the Highway op, just use Zero (single instruction).
namespace detail {
HWY_API VU8M1 Undefined(DU8M1 /* d */) { return vundefined_u8m1(); }
HWY_API VU8M2 Undefined(DU8M2 /* d */) { return vundefined_u8m2(); }
HWY_API VU8M4 Undefined(DU8M4 /* d */) { return vundefined_u8m4(); }
HWY_API VU8M8 Undefined(DU8M8 /* d */) { return vundefined_u8m8(); }

HWY_API VU16M1 Undefined(DU16M1 /* d */) { return vundefined_u16m1(); }
HWY_API VU16M2 Undefined(DU16M2 /* d */) { return vundefined_u16m2(); }
HWY_API VU16M4 Undefined(DU16M4 /* d */) { return vundefined_u16m4(); }
HWY_API VU16M8 Undefined(DU16M8 /* d */) { return vundefined_u16m8(); }

HWY_API VU32M1 Undefined(DU32M1 /* d */) { return vundefined_u32m1(); }
HWY_API VU32M2 Undefined(DU32M2 /* d */) { return vundefined_u32m2(); }
HWY_API VU32M4 Undefined(DU32M4 /* d */) { return vundefined_u32m4(); }
HWY_API VU32M8 Undefined(DU32M8 /* d */) { return vundefined_u32m8(); }

HWY_API VU64M1 Undefined(DU64M1 /* d */) { return vundefined_u64m1(); }
HWY_API VU64M2 Undefined(DU64M2 /* d */) { return vundefined_u64m2(); }
HWY_API VU64M4 Undefined(DU64M4 /* d */) { return vundefined_u64m4(); }
HWY_API VU64M8 Undefined(DU64M8 /* d */) { return vundefined_u64m8(); }

HWY_API VI8M1 Undefined(DI8M1 /* d */) { return vundefined_i8m1(); }
HWY_API VI8M2 Undefined(DI8M2 /* d */) { return vundefined_i8m2(); }
HWY_API VI8M4 Undefined(DI8M4 /* d */) { return vundefined_i8m4(); }
HWY_API VI8M8 Undefined(DI8M8 /* d */) { return vundefined_i8m8(); }

HWY_API VI16M1 Undefined(DI16M1 /* d */) { return vundefined_i16m1(); }
HWY_API VI16M2 Undefined(DI16M2 /* d */) { return vundefined_i16m2(); }
HWY_API VI16M4 Undefined(DI16M4 /* d */) { return vundefined_i16m4(); }
HWY_API VI16M8 Undefined(DI16M8 /* d */) { return vundefined_i16m8(); }

HWY_API VI32M1 Undefined(DI32M1 /* d */) { return vundefined_i32m1(); }
HWY_API VI32M2 Undefined(DI32M2 /* d */) { return vundefined_i32m2(); }
HWY_API VI32M4 Undefined(DI32M4 /* d */) { return vundefined_i32m4(); }
HWY_API VI32M8 Undefined(DI32M8 /* d */) { return vundefined_i32m8(); }

HWY_API VI64M1 Undefined(DI64M1 /* d */) { return vundefined_i64m1(); }
HWY_API VI64M2 Undefined(DI64M2 /* d */) { return vundefined_i64m2(); }
HWY_API VI64M4 Undefined(DI64M4 /* d */) { return vundefined_i64m4(); }
HWY_API VI64M8 Undefined(DI64M8 /* d */) { return vundefined_i64m8(); }

HWY_API VF32M1 Undefined(DF32M1 /* d */) { return vundefined_f32m1(); }
HWY_API VF32M2 Undefined(DF32M2 /* d */) { return vundefined_f32m2(); }
HWY_API VF32M4 Undefined(DF32M4 /* d */) { return vundefined_f32m4(); }
HWY_API VF32M8 Undefined(DF32M8 /* d */) { return vundefined_f32m8(); }

HWY_API VF64M1 Undefined(DF64M1 /* d */) { return vundefined_f64m1(); }
HWY_API VF64M2 Undefined(DF64M2 /* d */) { return vundefined_f64m2(); }
HWY_API VF64M4 Undefined(DF64M4 /* d */) { return vundefined_f64m4(); }
HWY_API VF64M8 Undefined(DF64M8 /* d */) { return vundefined_f64m8(); }
}  // namespace detail

template <class D>
HWY_API VFromD<D> Undefined(D d) {
  return Zero(d);
}

// ------------------------------ BitCast

namespace detail {

HWY_API VU8M1 BitCastToByte(VU8M1 v) { return v; }
HWY_API VU8M2 BitCastToByte(VU8M2 v) { return v; }
HWY_API VU8M4 BitCastToByte(VU8M4 v) { return v; }
HWY_API VU8M8 BitCastToByte(VU8M8 v) { return v; }

HWY_API VU8M1 BitCastToByte(VU16M1 v) { return vreinterpret_v_u16m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VU16M2 v) { return vreinterpret_v_u16m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VU16M4 v) { return vreinterpret_v_u16m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VU16M8 v) { return vreinterpret_v_u16m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VU32M1 v) { return vreinterpret_v_u32m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VU32M2 v) { return vreinterpret_v_u32m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VU32M4 v) { return vreinterpret_v_u32m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VU32M8 v) { return vreinterpret_v_u32m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VU64M1 v) { return vreinterpret_v_u64m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VU64M2 v) { return vreinterpret_v_u64m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VU64M4 v) { return vreinterpret_v_u64m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VU64M8 v) { return vreinterpret_v_u64m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VI8M1 v) { return vreinterpret_v_i8m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VI8M2 v) { return vreinterpret_v_i8m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VI8M4 v) { return vreinterpret_v_i8m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VI8M8 v) { return vreinterpret_v_i8m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VI16M1 v) { return vreinterpret_v_i16m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VI16M2 v) { return vreinterpret_v_i16m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VI16M4 v) { return vreinterpret_v_i16m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VI16M8 v) { return vreinterpret_v_i16m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VI32M1 v) { return vreinterpret_v_i32m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VI32M2 v) { return vreinterpret_v_i32m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VI32M4 v) { return vreinterpret_v_i32m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VI32M8 v) { return vreinterpret_v_i32m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VI64M1 v) { return vreinterpret_v_i64m1_u8m1(v); }
HWY_API VU8M2 BitCastToByte(VI64M2 v) { return vreinterpret_v_i64m2_u8m2(v); }
HWY_API VU8M4 BitCastToByte(VI64M4 v) { return vreinterpret_v_i64m4_u8m4(v); }
HWY_API VU8M8 BitCastToByte(VI64M8 v) { return vreinterpret_v_i64m8_u8m8(v); }

HWY_API VU8M1 BitCastToByte(VF32M1 v) {
  return vreinterpret_v_u32m1_u8m1(vreinterpret_v_f32m1_u32m1(v));
}
HWY_API VU8M2 BitCastToByte(VF32M2 v) {
  return vreinterpret_v_u32m2_u8m2(vreinterpret_v_f32m2_u32m2(v));
}
HWY_API VU8M4 BitCastToByte(VF32M4 v) {
  return vreinterpret_v_u32m4_u8m4(vreinterpret_v_f32m4_u32m4(v));
}
HWY_API VU8M8 BitCastToByte(VF32M8 v) {
  return vreinterpret_v_u32m8_u8m8(vreinterpret_v_f32m8_u32m8(v));
}

HWY_API VU8M1 BitCastToByte(VF64M1 v) {
  return vreinterpret_v_u64m1_u8m1(vreinterpret_v_f64m1_u64m1(v));
}
HWY_API VU8M2 BitCastToByte(VF64M2 v) {
  return vreinterpret_v_u64m2_u8m2(vreinterpret_v_f64m2_u64m2(v));
}
HWY_API VU8M4 BitCastToByte(VF64M4 v) {
  return vreinterpret_v_u64m4_u8m4(vreinterpret_v_f64m4_u64m4(v));
}
HWY_API VU8M8 BitCastToByte(VF64M8 v) {
  return vreinterpret_v_u64m8_u8m8(vreinterpret_v_f64m8_u64m8(v));
}

HWY_API VU8M1 BitCastFromByte(DU8M1 /* d */, VU8M1 v) { return v; }
HWY_API VU8M2 BitCastFromByte(DU8M2 /* d */, VU8M2 v) { return v; }
HWY_API VU8M4 BitCastFromByte(DU8M4 /* d */, VU8M4 v) { return v; }
HWY_API VU8M8 BitCastFromByte(DU8M8 /* d */, VU8M8 v) { return v; }

HWY_API VU16M1 BitCastFromByte(DU16M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_u16m1(v);
}
HWY_API VU16M2 BitCastFromByte(DU16M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_u16m2(v);
}
HWY_API VU16M4 BitCastFromByte(DU16M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_u16m4(v);
}
HWY_API VU16M8 BitCastFromByte(DU16M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_u16m8(v);
}

HWY_API VU32M1 BitCastFromByte(DU32M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_u32m1(v);
}
HWY_API VU32M2 BitCastFromByte(DU32M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_u32m2(v);
}
HWY_API VU32M4 BitCastFromByte(DU32M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_u32m4(v);
}
HWY_API VU32M8 BitCastFromByte(DU32M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_u32m8(v);
}

HWY_API VU64M1 BitCastFromByte(DU64M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_u64m1(v);
}
HWY_API VU64M2 BitCastFromByte(DU64M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_u64m2(v);
}
HWY_API VU64M4 BitCastFromByte(DU64M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_u64m4(v);
}
HWY_API VU64M8 BitCastFromByte(DU64M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_u64m8(v);
}

HWY_API VI8M1 BitCastFromByte(DI8M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_i8m1(v);
}
HWY_API VI8M2 BitCastFromByte(DI8M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_i8m2(v);
}
HWY_API VI8M4 BitCastFromByte(DI8M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_i8m4(v);
}
HWY_API VI8M8 BitCastFromByte(DI8M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_i8m8(v);
}

HWY_API VI16M1 BitCastFromByte(DI16M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_i16m1(v);
}
HWY_API VI16M2 BitCastFromByte(DI16M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_i16m2(v);
}
HWY_API VI16M4 BitCastFromByte(DI16M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_i16m4(v);
}
HWY_API VI16M8 BitCastFromByte(DI16M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_i16m8(v);
}

HWY_API VI32M1 BitCastFromByte(DI32M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_i32m1(v);
}
HWY_API VI32M2 BitCastFromByte(DI32M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_i32m2(v);
}
HWY_API VI32M4 BitCastFromByte(DI32M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_i32m4(v);
}
HWY_API VI32M8 BitCastFromByte(DI32M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_i32m8(v);
}

HWY_API VI64M1 BitCastFromByte(DI64M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u8m1_i64m1(v);
}
HWY_API VI64M2 BitCastFromByte(DI64M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u8m2_i64m2(v);
}
HWY_API VI64M4 BitCastFromByte(DI64M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u8m4_i64m4(v);
}
HWY_API VI64M8 BitCastFromByte(DI64M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u8m8_i64m8(v);
}

HWY_API VF32M1 BitCastFromByte(DF32M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u32m1_f32m1(vreinterpret_v_u8m1_u32m1(v));
}
HWY_API VF32M2 BitCastFromByte(DF32M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u32m2_f32m2(vreinterpret_v_u8m2_u32m2(v));
}
HWY_API VF32M4 BitCastFromByte(DF32M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u32m4_f32m4(vreinterpret_v_u8m4_u32m4(v));
}
HWY_API VF32M8 BitCastFromByte(DF32M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u32m8_f32m8(vreinterpret_v_u8m8_u32m8(v));
}

HWY_API VF64M1 BitCastFromByte(DF64M1 /* d */, VU8M1 v) {
  return vreinterpret_v_u64m1_f64m1(vreinterpret_v_u8m1_u64m1(v));
}
HWY_API VF64M2 BitCastFromByte(DF64M2 /* d */, VU8M2 v) {
  return vreinterpret_v_u64m2_f64m2(vreinterpret_v_u8m2_u64m2(v));
}
HWY_API VF64M4 BitCastFromByte(DF64M4 /* d */, VU8M4 v) {
  return vreinterpret_v_u64m4_f64m4(vreinterpret_v_u8m4_u64m4(v));
}
HWY_API VF64M8 BitCastFromByte(DF64M8 /* d */, VU8M8 v) {
  return vreinterpret_v_u64m8_f64m8(vreinterpret_v_u8m8_u64m8(v));
}

// Highway specifies same type but RVV requires unsigned shift count.
template <class V, class DU = RebindToUnsigned<DFromV<V>>>
HWY_API VFromD<DU> BitCastToUnsigned(V v) {
  return BitCast(DU(), v);
}

}  // namespace detail

template <class D, class FromV>
HWY_API VFromD<D> BitCast(D d, FromV v) {
  return detail::BitCastFromByte(d, detail::BitCastToByte(v));
}

// ------------------------------ Iota

namespace detail {

HWY_API VU8M1 Iota0(const DU8M1 d) { return vid_v_u8m1(); }
HWY_API VU8M2 Iota0(const DU8M2 d) { return vid_v_u8m2(); }
HWY_API VU8M4 Iota0(const DU8M4 d) { return vid_v_u8m4(); }
HWY_API VU8M8 Iota0(const DU8M8 d) { return vid_v_u8m8(); }

HWY_API VU16M1 Iota0(const DU16M1 d) { return vid_v_u16m1(); }
HWY_API VU16M2 Iota0(const DU16M2 d) { return vid_v_u16m2(); }
HWY_API VU16M4 Iota0(const DU16M4 d) { return vid_v_u16m4(); }
HWY_API VU16M8 Iota0(const DU16M8 d) { return vid_v_u16m8(); }

HWY_API VU32M1 Iota0(const DU32M1 d) { return vid_v_u32m1(); }
HWY_API VU32M2 Iota0(const DU32M2 d) { return vid_v_u32m2(); }
HWY_API VU32M4 Iota0(const DU32M4 d) { return vid_v_u32m4(); }
HWY_API VU32M8 Iota0(const DU32M8 d) { return vid_v_u32m8(); }

HWY_API VU64M1 Iota0(const DU64M1 d) { return vid_v_u64m1(); }
HWY_API VU64M2 Iota0(const DU64M2 d) { return vid_v_u64m2(); }
HWY_API VU64M4 Iota0(const DU64M4 d) { return vid_v_u64m4(); }
HWY_API VU64M8 Iota0(const DU64M8 d) { return vid_v_u64m8(); }

template <class D, class DU = RebindToUnsigned<D>>
HWY_API VFromD<DU> Iota0(const D d) {
  return BitCastToUnsigned(Iota0(DU()));
}

}  // namespace detail

// ================================================== LOGICAL

// ------------------------------ Not

HWY_API VU32M1 Not(const VU32M1 v) { return vnot_v_u32m1(v); }
HWY_API VU32M2 Not(const VU32M2 v) { return vnot_v_u32m2(v); }
HWY_API VU32M4 Not(const VU32M4 v) { return vnot_v_u32m4(v); }
HWY_API VU32M8 Not(const VU32M8 v) { return vnot_v_u32m8(v); }

template <class V, HWY_IF_FLOAT_V(V)>
HWY_API V Not(const V v) {
  using DF = DFromV<V>;
  using DU = RebindToUnsigned<DF>;
  return BitCast(DF(), Not(BitCast(DU(), v)));
}

// ------------------------------ And

// Non-vector version (ideally immediate) for use with Iota0
namespace detail {

HWY_API VU8M1 And(VU8M1 a, uint8_t b) { return vand_vx_u8m1(a, b); }
HWY_API VU8M2 And(VU8M2 a, uint8_t b) { return vand_vx_u8m2(a, b); }
HWY_API VU8M4 And(VU8M4 a, uint8_t b) { return vand_vx_u8m4(a, b); }
HWY_API VU8M8 And(VU8M8 a, uint8_t b) { return vand_vx_u8m8(a, b); }

HWY_API VU16M1 And(VU16M1 a, uint16_t b) { return vand_vx_u16m1(a, b); }
HWY_API VU16M2 And(VU16M2 a, uint16_t b) { return vand_vx_u16m2(a, b); }
HWY_API VU16M4 And(VU16M4 a, uint16_t b) { return vand_vx_u16m4(a, b); }
HWY_API VU16M8 And(VU16M8 a, uint16_t b) { return vand_vx_u16m8(a, b); }

HWY_API VU32M1 And(VU32M1 a, uint32_t b) { return vand_vx_u32m1(a, b); }
HWY_API VU32M2 And(VU32M2 a, uint32_t b) { return vand_vx_u32m2(a, b); }
HWY_API VU32M4 And(VU32M4 a, uint32_t b) { return vand_vx_u32m4(a, b); }
HWY_API VU32M8 And(VU32M8 a, uint32_t b) { return vand_vx_u32m8(a, b); }

HWY_API VU64M1 And(VU64M1 a, uint64_t b) { return vand_vx_u64m1(a, b); }
HWY_API VU64M2 And(VU64M2 a, uint64_t b) { return vand_vx_u64m2(a, b); }
HWY_API VU64M4 And(VU64M4 a, uint64_t b) { return vand_vx_u64m4(a, b); }
HWY_API VU64M8 And(VU64M8 a, uint64_t b) { return vand_vx_u64m8(a, b); }

HWY_API VI8M1 And(VI8M1 a, int8_t b) { return vand_vx_i8m1(a, b); }
HWY_API VI8M2 And(VI8M2 a, int8_t b) { return vand_vx_i8m2(a, b); }
HWY_API VI8M4 And(VI8M4 a, int8_t b) { return vand_vx_i8m4(a, b); }
HWY_API VI8M8 And(VI8M8 a, int8_t b) { return vand_vx_i8m8(a, b); }

HWY_API VI16M1 And(VI16M1 a, int16_t b) { return vand_vx_i16m1(a, b); }
HWY_API VI16M2 And(VI16M2 a, int16_t b) { return vand_vx_i16m2(a, b); }
HWY_API VI16M4 And(VI16M4 a, int16_t b) { return vand_vx_i16m4(a, b); }
HWY_API VI16M8 And(VI16M8 a, int16_t b) { return vand_vx_i16m8(a, b); }

HWY_API VI32M1 And(VI32M1 a, int32_t b) { return vand_vx_i32m1(a, b); }
HWY_API VI32M2 And(VI32M2 a, int32_t b) { return vand_vx_i32m2(a, b); }
HWY_API VI32M4 And(VI32M4 a, int32_t b) { return vand_vx_i32m4(a, b); }
HWY_API VI32M8 And(VI32M8 a, int32_t b) { return vand_vx_i32m8(a, b); }

HWY_API VI64M1 And(VI64M1 a, int64_t b) { return vand_vx_i64m1(a, b); }
HWY_API VI64M2 And(VI64M2 a, int64_t b) { return vand_vx_i64m2(a, b); }
HWY_API VI64M4 And(VI64M4 a, int64_t b) { return vand_vx_i64m4(a, b); }
HWY_API VI64M8 And(VI64M8 a, int64_t b) { return vand_vx_i64m8(a, b); }

}  // namespace detail

HWY_API VU8M1 And(VU8M1 a, VU8M1 b) { return vand_vv_u8m1(a, b); }
HWY_API VU8M2 And(VU8M2 a, VU8M2 b) { return vand_vv_u8m2(a, b); }
HWY_API VU8M4 And(VU8M4 a, VU8M4 b) { return vand_vv_u8m4(a, b); }
HWY_API VU8M8 And(VU8M8 a, VU8M8 b) { return vand_vv_u8m8(a, b); }

HWY_API VU16M1 And(VU16M1 a, VU16M1 b) { return vand_vv_u16m1(a, b); }
HWY_API VU16M2 And(VU16M2 a, VU16M2 b) { return vand_vv_u16m2(a, b); }
HWY_API VU16M4 And(VU16M4 a, VU16M4 b) { return vand_vv_u16m4(a, b); }
HWY_API VU16M8 And(VU16M8 a, VU16M8 b) { return vand_vv_u16m8(a, b); }

HWY_API VU32M1 And(VU32M1 a, VU32M1 b) { return vand_vv_u32m1(a, b); }
HWY_API VU32M2 And(VU32M2 a, VU32M2 b) { return vand_vv_u32m2(a, b); }
HWY_API VU32M4 And(VU32M4 a, VU32M4 b) { return vand_vv_u32m4(a, b); }
HWY_API VU32M8 And(VU32M8 a, VU32M8 b) { return vand_vv_u32m8(a, b); }

HWY_API VU64M1 And(VU64M1 a, VU64M1 b) { return vand_vv_u64m1(a, b); }
HWY_API VU64M2 And(VU64M2 a, VU64M2 b) { return vand_vv_u64m2(a, b); }
HWY_API VU64M4 And(VU64M4 a, VU64M4 b) { return vand_vv_u64m4(a, b); }
HWY_API VU64M8 And(VU64M8 a, VU64M8 b) { return vand_vv_u64m8(a, b); }

HWY_API VI8M1 And(VI8M1 a, VI8M1 b) { return vand_vv_i8m1(a, b); }
HWY_API VI8M2 And(VI8M2 a, VI8M2 b) { return vand_vv_i8m2(a, b); }
HWY_API VI8M4 And(VI8M4 a, VI8M4 b) { return vand_vv_i8m4(a, b); }
HWY_API VI8M8 And(VI8M8 a, VI8M8 b) { return vand_vv_i8m8(a, b); }

HWY_API VI16M1 And(VI16M1 a, VI16M1 b) { return vand_vv_i16m1(a, b); }
HWY_API VI16M2 And(VI16M2 a, VI16M2 b) { return vand_vv_i16m2(a, b); }
HWY_API VI16M4 And(VI16M4 a, VI16M4 b) { return vand_vv_i16m4(a, b); }
HWY_API VI16M8 And(VI16M8 a, VI16M8 b) { return vand_vv_i16m8(a, b); }

HWY_API VI32M1 And(VI32M1 a, VI32M1 b) { return vand_vv_i32m1(a, b); }
HWY_API VI32M2 And(VI32M2 a, VI32M2 b) { return vand_vv_i32m2(a, b); }
HWY_API VI32M4 And(VI32M4 a, VI32M4 b) { return vand_vv_i32m4(a, b); }
HWY_API VI32M8 And(VI32M8 a, VI32M8 b) { return vand_vv_i32m8(a, b); }

HWY_API VI64M1 And(VI64M1 a, VI64M1 b) { return vand_vv_i64m1(a, b); }
HWY_API VI64M2 And(VI64M2 a, VI64M2 b) { return vand_vv_i64m2(a, b); }
HWY_API VI64M4 And(VI64M4 a, VI64M4 b) { return vand_vv_i64m4(a, b); }
HWY_API VI64M8 And(VI64M8 a, VI64M8 b) { return vand_vv_i64m8(a, b); }

template <class V, HWY_IF_FLOAT_V(V)>
HWY_API V And(const V a, const V b) {
  using DF = DFromV<V>;
  using DU = RebindToUnsigned<DF>;
  return BitCast(DF(), And(BitCast(DU(), a), BitCast(DU(), b)));
}

// ------------------------------ Or

HWY_API VU8M1 Or(VU8M1 a, VU8M1 b) { return vor_vv_u8m1(a, b); }
HWY_API VU8M2 Or(VU8M2 a, VU8M2 b) { return vor_vv_u8m2(a, b); }
HWY_API VU8M4 Or(VU8M4 a, VU8M4 b) { return vor_vv_u8m4(a, b); }
HWY_API VU8M8 Or(VU8M8 a, VU8M8 b) { return vor_vv_u8m8(a, b); }

HWY_API VU16M1 Or(VU16M1 a, VU16M1 b) { return vor_vv_u16m1(a, b); }
HWY_API VU16M2 Or(VU16M2 a, VU16M2 b) { return vor_vv_u16m2(a, b); }
HWY_API VU16M4 Or(VU16M4 a, VU16M4 b) { return vor_vv_u16m4(a, b); }
HWY_API VU16M8 Or(VU16M8 a, VU16M8 b) { return vor_vv_u16m8(a, b); }

HWY_API VU32M1 Or(VU32M1 a, VU32M1 b) { return vor_vv_u32m1(a, b); }
HWY_API VU32M2 Or(VU32M2 a, VU32M2 b) { return vor_vv_u32m2(a, b); }
HWY_API VU32M4 Or(VU32M4 a, VU32M4 b) { return vor_vv_u32m4(a, b); }
HWY_API VU32M8 Or(VU32M8 a, VU32M8 b) { return vor_vv_u32m8(a, b); }

HWY_API VU64M1 Or(VU64M1 a, VU64M1 b) { return vor_vv_u64m1(a, b); }
HWY_API VU64M2 Or(VU64M2 a, VU64M2 b) { return vor_vv_u64m2(a, b); }
HWY_API VU64M4 Or(VU64M4 a, VU64M4 b) { return vor_vv_u64m4(a, b); }
HWY_API VU64M8 Or(VU64M8 a, VU64M8 b) { return vor_vv_u64m8(a, b); }

HWY_API VI8M1 Or(VI8M1 a, VI8M1 b) { return vor_vv_i8m1(a, b); }
HWY_API VI8M2 Or(VI8M2 a, VI8M2 b) { return vor_vv_i8m2(a, b); }
HWY_API VI8M4 Or(VI8M4 a, VI8M4 b) { return vor_vv_i8m4(a, b); }
HWY_API VI8M8 Or(VI8M8 a, VI8M8 b) { return vor_vv_i8m8(a, b); }

HWY_API VI16M1 Or(VI16M1 a, VI16M1 b) { return vor_vv_i16m1(a, b); }
HWY_API VI16M2 Or(VI16M2 a, VI16M2 b) { return vor_vv_i16m2(a, b); }
HWY_API VI16M4 Or(VI16M4 a, VI16M4 b) { return vor_vv_i16m4(a, b); }
HWY_API VI16M8 Or(VI16M8 a, VI16M8 b) { return vor_vv_i16m8(a, b); }

HWY_API VI32M1 Or(VI32M1 a, VI32M1 b) { return vor_vv_i32m1(a, b); }
HWY_API VI32M2 Or(VI32M2 a, VI32M2 b) { return vor_vv_i32m2(a, b); }
HWY_API VI32M4 Or(VI32M4 a, VI32M4 b) { return vor_vv_i32m4(a, b); }
HWY_API VI32M8 Or(VI32M8 a, VI32M8 b) { return vor_vv_i32m8(a, b); }

HWY_API VI64M1 Or(VI64M1 a, VI64M1 b) { return vor_vv_i64m1(a, b); }
HWY_API VI64M2 Or(VI64M2 a, VI64M2 b) { return vor_vv_i64m2(a, b); }
HWY_API VI64M4 Or(VI64M4 a, VI64M4 b) { return vor_vv_i64m4(a, b); }
HWY_API VI64M8 Or(VI64M8 a, VI64M8 b) { return vor_vv_i64m8(a, b); }

template <class V, HWY_IF_FLOAT_V(V)>
HWY_API V Or(const V a, const V b) {
  using DF = DFromV<V>;
  using DU = RebindToUnsigned<DF>;
  return BitCast(DF(), Or(BitCast(DU(), a), BitCast(DU(), b)));
}

// ------------------------------ Xor

// Non-vector version (ideally immediate) for use with Iota0
namespace detail {

HWY_API VU8M1 Xor(VU8M1 a, uint8_t b) { return vxor_vx_u8m1(a, b); }
HWY_API VU8M2 Xor(VU8M2 a, uint8_t b) { return vxor_vx_u8m2(a, b); }
HWY_API VU8M4 Xor(VU8M4 a, uint8_t b) { return vxor_vx_u8m4(a, b); }
HWY_API VU8M8 Xor(VU8M8 a, uint8_t b) { return vxor_vx_u8m8(a, b); }

HWY_API VU16M1 Xor(VU16M1 a, uint16_t b) { return vxor_vx_u16m1(a, b); }
HWY_API VU16M2 Xor(VU16M2 a, uint16_t b) { return vxor_vx_u16m2(a, b); }
HWY_API VU16M4 Xor(VU16M4 a, uint16_t b) { return vxor_vx_u16m4(a, b); }
HWY_API VU16M8 Xor(VU16M8 a, uint16_t b) { return vxor_vx_u16m8(a, b); }

HWY_API VU32M1 Xor(VU32M1 a, uint32_t b) { return vxor_vx_u32m1(a, b); }
HWY_API VU32M2 Xor(VU32M2 a, uint32_t b) { return vxor_vx_u32m2(a, b); }
HWY_API VU32M4 Xor(VU32M4 a, uint32_t b) { return vxor_vx_u32m4(a, b); }
HWY_API VU32M8 Xor(VU32M8 a, uint32_t b) { return vxor_vx_u32m8(a, b); }

HWY_API VU64M1 Xor(VU64M1 a, uint64_t b) { return vxor_vx_u64m1(a, b); }
HWY_API VU64M2 Xor(VU64M2 a, uint64_t b) { return vxor_vx_u64m2(a, b); }
HWY_API VU64M4 Xor(VU64M4 a, uint64_t b) { return vxor_vx_u64m4(a, b); }
HWY_API VU64M8 Xor(VU64M8 a, uint64_t b) { return vxor_vx_u64m8(a, b); }

HWY_API VI8M1 Xor(VI8M1 a, int8_t b) { return vxor_vx_i8m1(a, b); }
HWY_API VI8M2 Xor(VI8M2 a, int8_t b) { return vxor_vx_i8m2(a, b); }
HWY_API VI8M4 Xor(VI8M4 a, int8_t b) { return vxor_vx_i8m4(a, b); }
HWY_API VI8M8 Xor(VI8M8 a, int8_t b) { return vxor_vx_i8m8(a, b); }

HWY_API VI16M1 Xor(VI16M1 a, int16_t b) { return vxor_vx_i16m1(a, b); }
HWY_API VI16M2 Xor(VI16M2 a, int16_t b) { return vxor_vx_i16m2(a, b); }
HWY_API VI16M4 Xor(VI16M4 a, int16_t b) { return vxor_vx_i16m4(a, b); }
HWY_API VI16M8 Xor(VI16M8 a, int16_t b) { return vxor_vx_i16m8(a, b); }

HWY_API VI32M1 Xor(VI32M1 a, int32_t b) { return vxor_vx_i32m1(a, b); }
HWY_API VI32M2 Xor(VI32M2 a, int32_t b) { return vxor_vx_i32m2(a, b); }
HWY_API VI32M4 Xor(VI32M4 a, int32_t b) { return vxor_vx_i32m4(a, b); }
HWY_API VI32M8 Xor(VI32M8 a, int32_t b) { return vxor_vx_i32m8(a, b); }

HWY_API VI64M1 Xor(VI64M1 a, int64_t b) { return vxor_vx_i64m1(a, b); }
HWY_API VI64M2 Xor(VI64M2 a, int64_t b) { return vxor_vx_i64m2(a, b); }
HWY_API VI64M4 Xor(VI64M4 a, int64_t b) { return vxor_vx_i64m4(a, b); }
HWY_API VI64M8 Xor(VI64M8 a, int64_t b) { return vxor_vx_i64m8(a, b); }

}  // namespace detail

HWY_API VU8M1 Xor(VU8M1 a, VU8M1 b) { return vxor_vv_u8m1(a, b); }
HWY_API VU8M2 Xor(VU8M2 a, VU8M2 b) { return vxor_vv_u8m2(a, b); }
HWY_API VU8M4 Xor(VU8M4 a, VU8M4 b) { return vxor_vv_u8m4(a, b); }
HWY_API VU8M8 Xor(VU8M8 a, VU8M8 b) { return vxor_vv_u8m8(a, b); }

HWY_API VU16M1 Xor(VU16M1 a, VU16M1 b) { return vxor_vv_u16m1(a, b); }
HWY_API VU16M2 Xor(VU16M2 a, VU16M2 b) { return vxor_vv_u16m2(a, b); }
HWY_API VU16M4 Xor(VU16M4 a, VU16M4 b) { return vxor_vv_u16m4(a, b); }
HWY_API VU16M8 Xor(VU16M8 a, VU16M8 b) { return vxor_vv_u16m8(a, b); }

HWY_API VU32M1 Xor(VU32M1 a, VU32M1 b) { return vxor_vv_u32m1(a, b); }
HWY_API VU32M2 Xor(VU32M2 a, VU32M2 b) { return vxor_vv_u32m2(a, b); }
HWY_API VU32M4 Xor(VU32M4 a, VU32M4 b) { return vxor_vv_u32m4(a, b); }
HWY_API VU32M8 Xor(VU32M8 a, VU32M8 b) { return vxor_vv_u32m8(a, b); }

HWY_API VU64M1 Xor(VU64M1 a, VU64M1 b) { return vxor_vv_u64m1(a, b); }
HWY_API VU64M2 Xor(VU64M2 a, VU64M2 b) { return vxor_vv_u64m2(a, b); }
HWY_API VU64M4 Xor(VU64M4 a, VU64M4 b) { return vxor_vv_u64m4(a, b); }
HWY_API VU64M8 Xor(VU64M8 a, VU64M8 b) { return vxor_vv_u64m8(a, b); }

HWY_API VI8M1 Xor(VI8M1 a, VI8M1 b) { return vxor_vv_i8m1(a, b); }
HWY_API VI8M2 Xor(VI8M2 a, VI8M2 b) { return vxor_vv_i8m2(a, b); }
HWY_API VI8M4 Xor(VI8M4 a, VI8M4 b) { return vxor_vv_i8m4(a, b); }
HWY_API VI8M8 Xor(VI8M8 a, VI8M8 b) { return vxor_vv_i8m8(a, b); }

HWY_API VI16M1 Xor(VI16M1 a, VI16M1 b) { return vxor_vv_i16m1(a, b); }
HWY_API VI16M2 Xor(VI16M2 a, VI16M2 b) { return vxor_vv_i16m2(a, b); }
HWY_API VI16M4 Xor(VI16M4 a, VI16M4 b) { return vxor_vv_i16m4(a, b); }
HWY_API VI16M8 Xor(VI16M8 a, VI16M8 b) { return vxor_vv_i16m8(a, b); }

HWY_API VI32M1 Xor(VI32M1 a, VI32M1 b) { return vxor_vv_i32m1(a, b); }
HWY_API VI32M2 Xor(VI32M2 a, VI32M2 b) { return vxor_vv_i32m2(a, b); }
HWY_API VI32M4 Xor(VI32M4 a, VI32M4 b) { return vxor_vv_i32m4(a, b); }
HWY_API VI32M8 Xor(VI32M8 a, VI32M8 b) { return vxor_vv_i32m8(a, b); }

HWY_API VI64M1 Xor(VI64M1 a, VI64M1 b) { return vxor_vv_i64m1(a, b); }
HWY_API VI64M2 Xor(VI64M2 a, VI64M2 b) { return vxor_vv_i64m2(a, b); }
HWY_API VI64M4 Xor(VI64M4 a, VI64M4 b) { return vxor_vv_i64m4(a, b); }
HWY_API VI64M8 Xor(VI64M8 a, VI64M8 b) { return vxor_vv_i64m8(a, b); }

template <class V, HWY_IF_FLOAT_V(V)>
HWY_API V Xor(const V a, const V b) {
  using DF = DFromV<V>;
  using DU = RebindToUnsigned<DF>;
  return BitCast(DF(), Xor(BitCast(DU(), a), BitCast(DU(), b)));
}

// ------------------------------ AndNot

template <class V>
HWY_API V AndNot(const V not_a, const V b) {
  return And(Not(not_a), b);
}

// ------------------------------ CopySign

HWY_API VF32M1 CopySign(const VF32M1 magn, const VF32M1 sign) {
  return vfsgnj_vv_f32m1(magn, sign);
}
HWY_API VF32M2 CopySign(const VF32M2 magn, const VF32M2 sign) {
  return vfsgnj_vv_f32m2(magn, sign);
}
HWY_API VF32M4 CopySign(const VF32M4 magn, const VF32M4 sign) {
  return vfsgnj_vv_f32m4(magn, sign);
}
HWY_API VF32M8 CopySign(const VF32M8 magn, const VF32M8 sign) {
  return vfsgnj_vv_f32m8(magn, sign);
}

HWY_API VF64M1 CopySign(const VF64M1 magn, const VF64M1 sign) {
  return vfsgnj_vv_f64m1(magn, sign);
}
HWY_API VF64M2 CopySign(const VF64M2 magn, const VF64M2 sign) {
  return vfsgnj_vv_f64m2(magn, sign);
}
HWY_API VF64M4 CopySign(const VF64M4 magn, const VF64M4 sign) {
  return vfsgnj_vv_f64m4(magn, sign);
}
HWY_API VF64M8 CopySign(const VF64M8 magn, const VF64M8 sign) {
  return vfsgnj_vv_f64m8(magn, sign);
}

template <class V>
HWY_API V CopySignToAbs(const V abs, const V sign) {
  // RVV can also handle abs < 0, so no extra action needed.
  return CopySign(abs, sign);
}

// ================================================== ARITHMETIC

// ------------------------------ Add

namespace detail {
HWY_API VU8M1 Add(VU8M1 a, uint8_t b) { return vadd_vx_u8m1(a, b); }
HWY_API VU8M2 Add(VU8M2 a, uint8_t b) { return vadd_vx_u8m2(a, b); }
HWY_API VU8M4 Add(VU8M4 a, uint8_t b) { return vadd_vx_u8m4(a, b); }
HWY_API VU8M8 Add(VU8M8 a, uint8_t b) { return vadd_vx_u8m8(a, b); }

HWY_API VU16M1 Add(VU16M1 a, uint16_t b) { return vadd_vx_u16m1(a, b); }
HWY_API VU16M2 Add(VU16M2 a, uint16_t b) { return vadd_vx_u16m2(a, b); }
HWY_API VU16M4 Add(VU16M4 a, uint16_t b) { return vadd_vx_u16m4(a, b); }
HWY_API VU16M8 Add(VU16M8 a, uint16_t b) { return vadd_vx_u16m8(a, b); }

HWY_API VU32M1 Add(VU32M1 a, uint32_t b) { return vadd_vx_u32m1(a, b); }
HWY_API VU32M2 Add(VU32M2 a, uint32_t b) { return vadd_vx_u32m2(a, b); }
HWY_API VU32M4 Add(VU32M4 a, uint32_t b) { return vadd_vx_u32m4(a, b); }
HWY_API VU32M8 Add(VU32M8 a, uint32_t b) { return vadd_vx_u32m8(a, b); }

HWY_API VU64M1 Add(VU64M1 a, uint64_t b) { return vadd_vx_u64m1(a, b); }
HWY_API VU64M2 Add(VU64M2 a, uint64_t b) { return vadd_vx_u64m2(a, b); }
HWY_API VU64M4 Add(VU64M4 a, uint64_t b) { return vadd_vx_u64m4(a, b); }
HWY_API VU64M8 Add(VU64M8 a, uint64_t b) { return vadd_vx_u64m8(a, b); }

HWY_API VI8M1 Add(VI8M1 a, int8_t b) { return vadd_vx_i8m1(a, b); }
HWY_API VI8M2 Add(VI8M2 a, int8_t b) { return vadd_vx_i8m2(a, b); }
HWY_API VI8M4 Add(VI8M4 a, int8_t b) { return vadd_vx_i8m4(a, b); }
HWY_API VI8M8 Add(VI8M8 a, int8_t b) { return vadd_vx_i8m8(a, b); }

HWY_API VI16M1 Add(VI16M1 a, int16_t b) { return vadd_vx_i16m1(a, b); }
HWY_API VI16M2 Add(VI16M2 a, int16_t b) { return vadd_vx_i16m2(a, b); }
HWY_API VI16M4 Add(VI16M4 a, int16_t b) { return vadd_vx_i16m4(a, b); }
HWY_API VI16M8 Add(VI16M8 a, int16_t b) { return vadd_vx_i16m8(a, b); }

HWY_API VI32M1 Add(VI32M1 a, int32_t b) { return vadd_vx_i32m1(a, b); }
HWY_API VI32M2 Add(VI32M2 a, int32_t b) { return vadd_vx_i32m2(a, b); }
HWY_API VI32M4 Add(VI32M4 a, int32_t b) { return vadd_vx_i32m4(a, b); }
HWY_API VI32M8 Add(VI32M8 a, int32_t b) { return vadd_vx_i32m8(a, b); }

HWY_API VI64M1 Add(VI64M1 a, int64_t b) { return vadd_vx_i64m1(a, b); }
HWY_API VI64M2 Add(VI64M2 a, int64_t b) { return vadd_vx_i64m2(a, b); }
HWY_API VI64M4 Add(VI64M4 a, int64_t b) { return vadd_vx_i64m4(a, b); }
HWY_API VI64M8 Add(VI64M8 a, int64_t b) { return vadd_vx_i64m8(a, b); }

HWY_API VF32M1 Add(VF32M1 a, float32_t b) { return vfadd_vf_f32m1(a, b); }
HWY_API VF32M2 Add(VF32M2 a, float32_t b) { return vfadd_vf_f32m2(a, b); }
HWY_API VF32M4 Add(VF32M4 a, float32_t b) { return vfadd_vf_f32m4(a, b); }
HWY_API VF32M8 Add(VF32M8 a, float32_t b) { return vfadd_vf_f32m8(a, b); }

HWY_API VF64M1 Add(VF64M1 a, float64_t b) { return vfadd_vf_f64m1(a, b); }
HWY_API VF64M2 Add(VF64M2 a, float64_t b) { return vfadd_vf_f64m2(a, b); }
HWY_API VF64M4 Add(VF64M4 a, float64_t b) { return vfadd_vf_f64m4(a, b); }
HWY_API VF64M8 Add(VF64M8 a, float64_t b) { return vfadd_vf_f64m8(a, b); }

}  // namespace detail

HWY_API VU8M1 Add(VU8M1 a, VU8M1 b) { return vadd_vv_u8m1(a, b); }
HWY_API VU8M2 Add(VU8M2 a, VU8M2 b) { return vadd_vv_u8m2(a, b); }
HWY_API VU8M4 Add(VU8M4 a, VU8M4 b) { return vadd_vv_u8m4(a, b); }
HWY_API VU8M8 Add(VU8M8 a, VU8M8 b) { return vadd_vv_u8m8(a, b); }

HWY_API VU16M1 Add(VU16M1 a, VU16M1 b) { return vadd_vv_u16m1(a, b); }
HWY_API VU16M2 Add(VU16M2 a, VU16M2 b) { return vadd_vv_u16m2(a, b); }
HWY_API VU16M4 Add(VU16M4 a, VU16M4 b) { return vadd_vv_u16m4(a, b); }
HWY_API VU16M8 Add(VU16M8 a, VU16M8 b) { return vadd_vv_u16m8(a, b); }

HWY_API VU32M1 Add(VU32M1 a, VU32M1 b) { return vadd_vv_u32m1(a, b); }
HWY_API VU32M2 Add(VU32M2 a, VU32M2 b) { return vadd_vv_u32m2(a, b); }
HWY_API VU32M4 Add(VU32M4 a, VU32M4 b) { return vadd_vv_u32m4(a, b); }
HWY_API VU32M8 Add(VU32M8 a, VU32M8 b) { return vadd_vv_u32m8(a, b); }

HWY_API VU64M1 Add(VU64M1 a, VU64M1 b) { return vadd_vv_u64m1(a, b); }
HWY_API VU64M2 Add(VU64M2 a, VU64M2 b) { return vadd_vv_u64m2(a, b); }
HWY_API VU64M4 Add(VU64M4 a, VU64M4 b) { return vadd_vv_u64m4(a, b); }
HWY_API VU64M8 Add(VU64M8 a, VU64M8 b) { return vadd_vv_u64m8(a, b); }

HWY_API VI8M1 Add(VI8M1 a, VI8M1 b) { return vadd_vv_i8m1(a, b); }
HWY_API VI8M2 Add(VI8M2 a, VI8M2 b) { return vadd_vv_i8m2(a, b); }
HWY_API VI8M4 Add(VI8M4 a, VI8M4 b) { return vadd_vv_i8m4(a, b); }
HWY_API VI8M8 Add(VI8M8 a, VI8M8 b) { return vadd_vv_i8m8(a, b); }

HWY_API VI16M1 Add(VI16M1 a, VI16M1 b) { return vadd_vv_i16m1(a, b); }
HWY_API VI16M2 Add(VI16M2 a, VI16M2 b) { return vadd_vv_i16m2(a, b); }
HWY_API VI16M4 Add(VI16M4 a, VI16M4 b) { return vadd_vv_i16m4(a, b); }
HWY_API VI16M8 Add(VI16M8 a, VI16M8 b) { return vadd_vv_i16m8(a, b); }

HWY_API VI32M1 Add(VI32M1 a, VI32M1 b) { return vadd_vv_i32m1(a, b); }
HWY_API VI32M2 Add(VI32M2 a, VI32M2 b) { return vadd_vv_i32m2(a, b); }
HWY_API VI32M4 Add(VI32M4 a, VI32M4 b) { return vadd_vv_i32m4(a, b); }
HWY_API VI32M8 Add(VI32M8 a, VI32M8 b) { return vadd_vv_i32m8(a, b); }

HWY_API VI64M1 Add(VI64M1 a, VI64M1 b) { return vadd_vv_i64m1(a, b); }
HWY_API VI64M2 Add(VI64M2 a, VI64M2 b) { return vadd_vv_i64m2(a, b); }
HWY_API VI64M4 Add(VI64M4 a, VI64M4 b) { return vadd_vv_i64m4(a, b); }
HWY_API VI64M8 Add(VI64M8 a, VI64M8 b) { return vadd_vv_i64m8(a, b); }

HWY_API VF32M1 Add(VF32M1 a, VF32M1 b) { return vfadd_vv_f32m1(a, b); }
HWY_API VF32M2 Add(VF32M2 a, VF32M2 b) { return vfadd_vv_f32m2(a, b); }
HWY_API VF32M4 Add(VF32M4 a, VF32M4 b) { return vfadd_vv_f32m4(a, b); }
HWY_API VF32M8 Add(VF32M8 a, VF32M8 b) { return vfadd_vv_f32m8(a, b); }

HWY_API VF64M1 Add(VF64M1 a, VF64M1 b) { return vfadd_vv_f64m1(a, b); }
HWY_API VF64M2 Add(VF64M2 a, VF64M2 b) { return vfadd_vv_f64m2(a, b); }
HWY_API VF64M4 Add(VF64M4 a, VF64M4 b) { return vfadd_vv_f64m4(a, b); }
HWY_API VF64M8 Add(VF64M8 a, VF64M8 b) { return vfadd_vv_f64m8(a, b); }

// ------------------------------ Sub

HWY_API VU8M1 Sub(VU8M1 a, VU8M1 b) { return vsub_vv_u8m1(a, b); }
HWY_API VU8M2 Sub(VU8M2 a, VU8M2 b) { return vsub_vv_u8m2(a, b); }
HWY_API VU8M4 Sub(VU8M4 a, VU8M4 b) { return vsub_vv_u8m4(a, b); }
HWY_API VU8M8 Sub(VU8M8 a, VU8M8 b) { return vsub_vv_u8m8(a, b); }

HWY_API VU16M1 Sub(VU16M1 a, VU16M1 b) { return vsub_vv_u16m1(a, b); }
HWY_API VU16M2 Sub(VU16M2 a, VU16M2 b) { return vsub_vv_u16m2(a, b); }
HWY_API VU16M4 Sub(VU16M4 a, VU16M4 b) { return vsub_vv_u16m4(a, b); }
HWY_API VU16M8 Sub(VU16M8 a, VU16M8 b) { return vsub_vv_u16m8(a, b); }

HWY_API VU32M1 Sub(VU32M1 a, VU32M1 b) { return vsub_vv_u32m1(a, b); }
HWY_API VU32M2 Sub(VU32M2 a, VU32M2 b) { return vsub_vv_u32m2(a, b); }
HWY_API VU32M4 Sub(VU32M4 a, VU32M4 b) { return vsub_vv_u32m4(a, b); }
HWY_API VU32M8 Sub(VU32M8 a, VU32M8 b) { return vsub_vv_u32m8(a, b); }

HWY_API VU64M1 Sub(VU64M1 a, VU64M1 b) { return vsub_vv_u64m1(a, b); }
HWY_API VU64M2 Sub(VU64M2 a, VU64M2 b) { return vsub_vv_u64m2(a, b); }
HWY_API VU64M4 Sub(VU64M4 a, VU64M4 b) { return vsub_vv_u64m4(a, b); }
HWY_API VU64M8 Sub(VU64M8 a, VU64M8 b) { return vsub_vv_u64m8(a, b); }

HWY_API VI8M1 Sub(VI8M1 a, VI8M1 b) { return vsub_vv_i8m1(a, b); }
HWY_API VI8M2 Sub(VI8M2 a, VI8M2 b) { return vsub_vv_i8m2(a, b); }
HWY_API VI8M4 Sub(VI8M4 a, VI8M4 b) { return vsub_vv_i8m4(a, b); }
HWY_API VI8M8 Sub(VI8M8 a, VI8M8 b) { return vsub_vv_i8m8(a, b); }

HWY_API VI16M1 Sub(VI16M1 a, VI16M1 b) { return vsub_vv_i16m1(a, b); }
HWY_API VI16M2 Sub(VI16M2 a, VI16M2 b) { return vsub_vv_i16m2(a, b); }
HWY_API VI16M4 Sub(VI16M4 a, VI16M4 b) { return vsub_vv_i16m4(a, b); }
HWY_API VI16M8 Sub(VI16M8 a, VI16M8 b) { return vsub_vv_i16m8(a, b); }

HWY_API VI32M1 Sub(VI32M1 a, VI32M1 b) { return vsub_vv_i32m1(a, b); }
HWY_API VI32M2 Sub(VI32M2 a, VI32M2 b) { return vsub_vv_i32m2(a, b); }
HWY_API VI32M4 Sub(VI32M4 a, VI32M4 b) { return vsub_vv_i32m4(a, b); }
HWY_API VI32M8 Sub(VI32M8 a, VI32M8 b) { return vsub_vv_i32m8(a, b); }

HWY_API VI64M1 Sub(VI64M1 a, VI64M1 b) { return vsub_vv_i64m1(a, b); }
HWY_API VI64M2 Sub(VI64M2 a, VI64M2 b) { return vsub_vv_i64m2(a, b); }
HWY_API VI64M4 Sub(VI64M4 a, VI64M4 b) { return vsub_vv_i64m4(a, b); }
HWY_API VI64M8 Sub(VI64M8 a, VI64M8 b) { return vsub_vv_i64m8(a, b); }

HWY_API VF32M1 Sub(VF32M1 a, VF32M1 b) { return vfsub_vv_f32m1(a, b); }
HWY_API VF32M2 Sub(VF32M2 a, VF32M2 b) { return vfsub_vv_f32m2(a, b); }
HWY_API VF32M4 Sub(VF32M4 a, VF32M4 b) { return vfsub_vv_f32m4(a, b); }
HWY_API VF32M8 Sub(VF32M8 a, VF32M8 b) { return vfsub_vv_f32m8(a, b); }

HWY_API VF64M1 Sub(VF64M1 a, VF64M1 b) { return vfsub_vv_f64m1(a, b); }
HWY_API VF64M2 Sub(VF64M2 a, VF64M2 b) { return vfsub_vv_f64m2(a, b); }
HWY_API VF64M4 Sub(VF64M4 a, VF64M4 b) { return vfsub_vv_f64m4(a, b); }
HWY_API VF64M8 Sub(VF64M8 a, VF64M8 b) { return vfsub_vv_f64m8(a, b); }

// ------------------------------ SaturatedAdd

HWY_API VU8M1 SaturatedAdd(VU8M1 a, VU8M1 b) { return vsaddu_vv_u8m1(a, b); }
HWY_API VU8M2 SaturatedAdd(VU8M2 a, VU8M2 b) { return vsaddu_vv_u8m2(a, b); }
HWY_API VU8M4 SaturatedAdd(VU8M4 a, VU8M4 b) { return vsaddu_vv_u8m4(a, b); }
HWY_API VU8M8 SaturatedAdd(VU8M8 a, VU8M8 b) { return vsaddu_vv_u8m8(a, b); }

HWY_API VU16M1 SaturatedAdd(VU16M1 a, VU16M1 b) {
  return vsaddu_vv_u16m1(a, b);
}
HWY_API VU16M2 SaturatedAdd(VU16M2 a, VU16M2 b) {
  return vsaddu_vv_u16m2(a, b);
}
HWY_API VU16M4 SaturatedAdd(VU16M4 a, VU16M4 b) {
  return vsaddu_vv_u16m4(a, b);
}
HWY_API VU16M8 SaturatedAdd(VU16M8 a, VU16M8 b) {
  return vsaddu_vv_u16m8(a, b);
}

HWY_API VI8M1 SaturatedAdd(VI8M1 a, VI8M1 b) { return vsadd_vv_i8m1(a, b); }
HWY_API VI8M2 SaturatedAdd(VI8M2 a, VI8M2 b) { return vsadd_vv_i8m2(a, b); }
HWY_API VI8M4 SaturatedAdd(VI8M4 a, VI8M4 b) { return vsadd_vv_i8m4(a, b); }
HWY_API VI8M8 SaturatedAdd(VI8M8 a, VI8M8 b) { return vsadd_vv_i8m8(a, b); }

HWY_API VI16M1 SaturatedAdd(VI16M1 a, VI16M1 b) { return vsadd_vv_i16m1(a, b); }
HWY_API VI16M2 SaturatedAdd(VI16M2 a, VI16M2 b) { return vsadd_vv_i16m2(a, b); }
HWY_API VI16M4 SaturatedAdd(VI16M4 a, VI16M4 b) { return vsadd_vv_i16m4(a, b); }
HWY_API VI16M8 SaturatedAdd(VI16M8 a, VI16M8 b) { return vsadd_vv_i16m8(a, b); }

// ------------------------------ SaturatedSub

HWY_API VU8M1 SaturatedSub(VU8M1 a, VU8M1 b) { return vssubu_vv_u8m1(a, b); }
HWY_API VU8M2 SaturatedSub(VU8M2 a, VU8M2 b) { return vssubu_vv_u8m2(a, b); }
HWY_API VU8M4 SaturatedSub(VU8M4 a, VU8M4 b) { return vssubu_vv_u8m4(a, b); }
HWY_API VU8M8 SaturatedSub(VU8M8 a, VU8M8 b) { return vssubu_vv_u8m8(a, b); }

HWY_API VU16M1 SaturatedSub(VU16M1 a, VU16M1 b) {
  return vssubu_vv_u16m1(a, b);
}
HWY_API VU16M2 SaturatedSub(VU16M2 a, VU16M2 b) {
  return vssubu_vv_u16m2(a, b);
}
HWY_API VU16M4 SaturatedSub(VU16M4 a, VU16M4 b) {
  return vssubu_vv_u16m4(a, b);
}
HWY_API VU16M8 SaturatedSub(VU16M8 a, VU16M8 b) {
  return vssubu_vv_u16m8(a, b);
}

HWY_API VI8M1 SaturatedSub(VI8M1 a, VI8M1 b) { return vssub_vv_i8m1(a, b); }
HWY_API VI8M2 SaturatedSub(VI8M2 a, VI8M2 b) { return vssub_vv_i8m2(a, b); }
HWY_API VI8M4 SaturatedSub(VI8M4 a, VI8M4 b) { return vssub_vv_i8m4(a, b); }
HWY_API VI8M8 SaturatedSub(VI8M8 a, VI8M8 b) { return vssub_vv_i8m8(a, b); }

HWY_API VI16M1 SaturatedSub(VI16M1 a, VI16M1 b) { return vssub_vv_i16m1(a, b); }
HWY_API VI16M2 SaturatedSub(VI16M2 a, VI16M2 b) { return vssub_vv_i16m2(a, b); }
HWY_API VI16M4 SaturatedSub(VI16M4 a, VI16M4 b) { return vssub_vv_i16m4(a, b); }
HWY_API VI16M8 SaturatedSub(VI16M8 a, VI16M8 b) { return vssub_vv_i16m8(a, b); }

// ------------------------------ AverageRound

// TODO(janwas): check vxrm rounding mode

HWY_API VU8M1 AverageRound(VU8M1 a, VU8M1 b) { return vaaddu_vv_u8m1(a, b); }
HWY_API VU8M2 AverageRound(VU8M2 a, VU8M2 b) { return vaaddu_vv_u8m2(a, b); }
HWY_API VU8M4 AverageRound(VU8M4 a, VU8M4 b) { return vaaddu_vv_u8m4(a, b); }
HWY_API VU8M8 AverageRound(VU8M8 a, VU8M8 b) { return vaaddu_vv_u8m8(a, b); }

HWY_API VU16M1 AverageRound(VU16M1 a, VU16M1 b) {
  return vaaddu_vv_u16m1(a, b);
}
HWY_API VU16M2 AverageRound(VU16M2 a, VU16M2 b) {
  return vaaddu_vv_u16m2(a, b);
}
HWY_API VU16M4 AverageRound(VU16M4 a, VU16M4 b) {
  return vaaddu_vv_u16m4(a, b);
}
HWY_API VU16M8 AverageRound(VU16M8 a, VU16M8 b) {
  return vaaddu_vv_u16m8(a, b);
}

// ------------------------------ ShiftLeft

// Intrinsics do not define .vi forms, so use .vx instead.

template <int kBits>
HWY_API VU8M1 ShiftLeft(VU8M1 v) {
  return vsll_vx_u8m1(v, kBits);
}
template <int kBits>
HWY_API VU8M2 ShiftLeft(VU8M2 v) {
  return vsll_vx_u8m2(v, kBits);
}
template <int kBits>
HWY_API VU8M4 ShiftLeft(VU8M4 v) {
  return vsll_vx_u8m4(v, kBits);
}
template <int kBits>
HWY_API VU8M8 ShiftLeft(VU8M8 v) {
  return vsll_vx_u8m8(v, kBits);
}

template <int kBits>
HWY_API VU16M1 ShiftLeft(VU16M1 v) {
  return vsll_vx_u16m1(v, kBits);
}
template <int kBits>
HWY_API VU16M2 ShiftLeft(VU16M2 v) {
  return vsll_vx_u16m2(v, kBits);
}
template <int kBits>
HWY_API VU16M4 ShiftLeft(VU16M4 v) {
  return vsll_vx_u16m4(v, kBits);
}
template <int kBits>
HWY_API VU16M8 ShiftLeft(VU16M8 v) {
  return vsll_vx_u16m8(v, kBits);
}

template <int kBits>
HWY_API VU32M1 ShiftLeft(VU32M1 v) {
  return vsll_vx_u32m1(v, kBits);
}
template <int kBits>
HWY_API VU32M2 ShiftLeft(VU32M2 v) {
  return vsll_vx_u32m2(v, kBits);
}
template <int kBits>
HWY_API VU32M4 ShiftLeft(VU32M4 v) {
  return vsll_vx_u32m4(v, kBits);
}
template <int kBits>
HWY_API VU32M8 ShiftLeft(VU32M8 v) {
  return vsll_vx_u32m8(v, kBits);
}

template <int kBits>
HWY_API VU64M1 ShiftLeft(VU64M1 v) {
  return vsll_vx_u64m1(v, kBits);
}
template <int kBits>
HWY_API VU64M2 ShiftLeft(VU64M2 v) {
  return vsll_vx_u64m2(v, kBits);
}
template <int kBits>
HWY_API VU64M4 ShiftLeft(VU64M4 v) {
  return vsll_vx_u64m4(v, kBits);
}
template <int kBits>
HWY_API VU64M8 ShiftLeft(VU64M8 v) {
  return vsll_vx_u64m8(v, kBits);
}

template <int kBits>
HWY_API VI8M1 ShiftLeft(VI8M1 v) {
  return vsll_vx_i8m1(v, kBits);
}
template <int kBits>
HWY_API VI8M2 ShiftLeft(VI8M2 v) {
  return vsll_vx_i8m2(v, kBits);
}
template <int kBits>
HWY_API VI8M4 ShiftLeft(VI8M4 v) {
  return vsll_vx_i8m4(v, kBits);
}
template <int kBits>
HWY_API VI8M8 ShiftLeft(VI8M8 v) {
  return vsll_vx_i8m8(v, kBits);
}

template <int kBits>
HWY_API VI16M1 ShiftLeft(VI16M1 v) {
  return vsll_vx_i16m1(v, kBits);
}
template <int kBits>
HWY_API VI16M2 ShiftLeft(VI16M2 v) {
  return vsll_vx_i16m2(v, kBits);
}
template <int kBits>
HWY_API VI16M4 ShiftLeft(VI16M4 v) {
  return vsll_vx_i16m4(v, kBits);
}
template <int kBits>
HWY_API VI16M8 ShiftLeft(VI16M8 v) {
  return vsll_vx_i16m8(v, kBits);
}

template <int kBits>
HWY_API VI32M1 ShiftLeft(VI32M1 v) {
  return vsll_vx_i32m1(v, kBits);
}
template <int kBits>
HWY_API VI32M2 ShiftLeft(VI32M2 v) {
  return vsll_vx_i32m2(v, kBits);
}
template <int kBits>
HWY_API VI32M4 ShiftLeft(VI32M4 v) {
  return vsll_vx_i32m4(v, kBits);
}
template <int kBits>
HWY_API VI32M8 ShiftLeft(VI32M8 v) {
  return vsll_vx_i32m8(v, kBits);
}

template <int kBits>
HWY_API VI64M1 ShiftLeft(VI64M1 v) {
  return vsll_vx_i64m1(v, kBits);
}
template <int kBits>
HWY_API VI64M2 ShiftLeft(VI64M2 v) {
  return vsll_vx_i64m2(v, kBits);
}
template <int kBits>
HWY_API VI64M4 ShiftLeft(VI64M4 v) {
  return vsll_vx_i64m4(v, kBits);
}
template <int kBits>
HWY_API VI64M8 ShiftLeft(VI64M8 v) {
  return vsll_vx_i64m8(v, kBits);
}

// ------------------------------ ShiftRight

template <int kBits>
HWY_API VU8M1 ShiftRight(VU8M1 v) {
  return vsrl_vx_u8m1(v, kBits);
}
template <int kBits>
HWY_API VU8M2 ShiftRight(VU8M2 v) {
  return vsrl_vx_u8m2(v, kBits);
}
template <int kBits>
HWY_API VU8M4 ShiftRight(VU8M4 v) {
  return vsrl_vx_u8m4(v, kBits);
}
template <int kBits>
HWY_API VU8M8 ShiftRight(VU8M8 v) {
  return vsrl_vx_u8m8(v, kBits);
}

template <int kBits>
HWY_API VU16M1 ShiftRight(VU16M1 v) {
  return vsrl_vx_u16m1(v, kBits);
}
template <int kBits>
HWY_API VU16M2 ShiftRight(VU16M2 v) {
  return vsrl_vx_u16m2(v, kBits);
}
template <int kBits>
HWY_API VU16M4 ShiftRight(VU16M4 v) {
  return vsrl_vx_u16m4(v, kBits);
}
template <int kBits>
HWY_API VU16M8 ShiftRight(VU16M8 v) {
  return vsrl_vx_u16m8(v, kBits);
}

template <int kBits>
HWY_API VU32M1 ShiftRight(VU32M1 v) {
  return vsrl_vx_u32m1(v, kBits);
}
template <int kBits>
HWY_API VU32M2 ShiftRight(VU32M2 v) {
  return vsrl_vx_u32m2(v, kBits);
}
template <int kBits>
HWY_API VU32M4 ShiftRight(VU32M4 v) {
  return vsrl_vx_u32m4(v, kBits);
}
template <int kBits>
HWY_API VU32M8 ShiftRight(VU32M8 v) {
  return vsrl_vx_u32m8(v, kBits);
}

template <int kBits>
HWY_API VU64M1 ShiftRight(VU64M1 v) {
  return vsrl_vx_u64m1(v, kBits);
}
template <int kBits>
HWY_API VU64M2 ShiftRight(VU64M2 v) {
  return vsrl_vx_u64m2(v, kBits);
}
template <int kBits>
HWY_API VU64M4 ShiftRight(VU64M4 v) {
  return vsrl_vx_u64m4(v, kBits);
}
template <int kBits>
HWY_API VU64M8 ShiftRight(VU64M8 v) {
  return vsrl_vx_u64m8(v, kBits);
}

template <int kBits>
HWY_API VI8M1 ShiftRight(VI8M1 v) {
  return vsra_vx_i8m1(v, kBits);
}
template <int kBits>
HWY_API VI8M2 ShiftRight(VI8M2 v) {
  return vsra_vx_i8m2(v, kBits);
}
template <int kBits>
HWY_API VI8M4 ShiftRight(VI8M4 v) {
  return vsra_vx_i8m4(v, kBits);
}
template <int kBits>
HWY_API VI8M8 ShiftRight(VI8M8 v) {
  return vsra_vx_i8m8(v, kBits);
}

template <int kBits>
HWY_API VI16M1 ShiftRight(VI16M1 v) {
  return vsra_vx_i16m1(v, kBits);
}
template <int kBits>
HWY_API VI16M2 ShiftRight(VI16M2 v) {
  return vsra_vx_i16m2(v, kBits);
}
template <int kBits>
HWY_API VI16M4 ShiftRight(VI16M4 v) {
  return vsra_vx_i16m4(v, kBits);
}
template <int kBits>
HWY_API VI16M8 ShiftRight(VI16M8 v) {
  return vsra_vx_i16m8(v, kBits);
}

template <int kBits>
HWY_API VI32M1 ShiftRight(VI32M1 v) {
  return vsra_vx_i32m1(v, kBits);
}
template <int kBits>
HWY_API VI32M2 ShiftRight(VI32M2 v) {
  return vsra_vx_i32m2(v, kBits);
}
template <int kBits>
HWY_API VI32M4 ShiftRight(VI32M4 v) {
  return vsra_vx_i32m4(v, kBits);
}
template <int kBits>
HWY_API VI32M8 ShiftRight(VI32M8 v) {
  return vsra_vx_i32m8(v, kBits);
}

template <int kBits>
HWY_API VI64M1 ShiftRight(VI64M1 v) {
  return vsra_vx_i64m1(v, kBits);
}
template <int kBits>
HWY_API VI64M2 ShiftRight(VI64M2 v) {
  return vsra_vx_i64m2(v, kBits);
}
template <int kBits>
HWY_API VI64M4 ShiftRight(VI64M4 v) {
  return vsra_vx_i64m4(v, kBits);
}
template <int kBits>
HWY_API VI64M8 ShiftRight(VI64M8 v) {
  return vsra_vx_i64m8(v, kBits);
}

// No i64 ShiftRight in Highway.

// ------------------------------ Shl

// Unsigned (no u8,u16)
HWY_API VU32M1 Shl(VU32M1 v, VU32M1 bits) { return vsll_vv_u32m1(v, bits); }
HWY_API VU32M2 Shl(VU32M2 v, VU32M2 bits) { return vsll_vv_u32m2(v, bits); }
HWY_API VU32M4 Shl(VU32M4 v, VU32M4 bits) { return vsll_vv_u32m4(v, bits); }
HWY_API VU32M8 Shl(VU32M8 v, VU32M8 bits) { return vsll_vv_u32m8(v, bits); }

HWY_API VU64M1 Shl(VU64M1 v, VU64M1 bits) { return vsll_vv_u64m1(v, bits); }
HWY_API VU64M2 Shl(VU64M2 v, VU64M2 bits) { return vsll_vv_u64m2(v, bits); }
HWY_API VU64M4 Shl(VU64M4 v, VU64M4 bits) { return vsll_vv_u64m4(v, bits); }
HWY_API VU64M8 Shl(VU64M8 v, VU64M8 bits) { return vsll_vv_u64m8(v, bits); }

// Signed (no i8,i16,i64)
HWY_API VI32M1 Shl(const VI32M1 v, const VI32M1 bits) {
  return vsll_vv_i32m1(v, detail::BitCastToUnsigned(bits));
}
HWY_API VI32M2 Shl(const VI32M2 v, const VI32M2 bits) {
  return vsll_vv_i32m2(v, detail::BitCastToUnsigned(bits));
}
HWY_API VI32M4 Shl(const VI32M4 v, const VI32M4 bits) {
  return vsll_vv_i32m4(v, detail::BitCastToUnsigned(bits));
}
HWY_API VI32M8 Shl(const VI32M8 v, const VI32M8 bits) {
  return vsll_vv_i32m8(v, detail::BitCastToUnsigned(bits));
}

// ------------------------------ Shr

HWY_API VU32M1 Shr(VU32M1 v, VU32M1 bits) { return vsrl_vv_u32m1(v, bits); }
HWY_API VU32M2 Shr(VU32M2 v, VU32M2 bits) { return vsrl_vv_u32m2(v, bits); }
HWY_API VU32M4 Shr(VU32M4 v, VU32M4 bits) { return vsrl_vv_u32m4(v, bits); }
HWY_API VU32M8 Shr(VU32M8 v, VU32M8 bits) { return vsrl_vv_u32m8(v, bits); }

HWY_API VU64M1 Shr(VU64M1 v, VU64M1 bits) { return vsrl_vv_u64m1(v, bits); }
HWY_API VU64M2 Shr(VU64M2 v, VU64M2 bits) { return vsrl_vv_u64m2(v, bits); }
HWY_API VU64M4 Shr(VU64M4 v, VU64M4 bits) { return vsrl_vv_u64m4(v, bits); }
HWY_API VU64M8 Shr(VU64M8 v, VU64M8 bits) { return vsrl_vv_u64m8(v, bits); }

HWY_API VI32M1 Shr(const VI32M1 v, const VI32M1 bits) {
  return vsra_vv_i32m1(v, detail::BitCastToUnsigned(bits));
}
HWY_API VI32M2 Shr(const VI32M2 v, const VI32M2 bits) {
  return vsra_vv_i32m2(v, detail::BitCastToUnsigned(bits));
}
HWY_API VI32M4 Shr(const VI32M4 v, const VI32M4 bits) {
  return vsra_vv_i32m4(v, detail::BitCastToUnsigned(bits));
}
HWY_API VI32M8 Shr(const VI32M8 v, const VI32M8 bits) {
  return vsra_vv_i32m8(v, detail::BitCastToUnsigned(bits));
}

// ------------------------------ Min

HWY_API VU8M1 Min(VU8M1 a, VU8M1 b) { return vminu_vv_u8m1(a, b); }
HWY_API VU8M2 Min(VU8M2 a, VU8M2 b) { return vminu_vv_u8m2(a, b); }
HWY_API VU8M4 Min(VU8M4 a, VU8M4 b) { return vminu_vv_u8m4(a, b); }
HWY_API VU8M8 Min(VU8M8 a, VU8M8 b) { return vminu_vv_u8m8(a, b); }

HWY_API VU16M1 Min(VU16M1 a, VU16M1 b) { return vminu_vv_u16m1(a, b); }
HWY_API VU16M2 Min(VU16M2 a, VU16M2 b) { return vminu_vv_u16m2(a, b); }
HWY_API VU16M4 Min(VU16M4 a, VU16M4 b) { return vminu_vv_u16m4(a, b); }
HWY_API VU16M8 Min(VU16M8 a, VU16M8 b) { return vminu_vv_u16m8(a, b); }

HWY_API VU32M1 Min(VU32M1 a, VU32M1 b) { return vminu_vv_u32m1(a, b); }
HWY_API VU32M2 Min(VU32M2 a, VU32M2 b) { return vminu_vv_u32m2(a, b); }
HWY_API VU32M4 Min(VU32M4 a, VU32M4 b) { return vminu_vv_u32m4(a, b); }
HWY_API VU32M8 Min(VU32M8 a, VU32M8 b) { return vminu_vv_u32m8(a, b); }

HWY_API VU64M1 Min(VU64M1 a, VU64M1 b) { return vminu_vv_u64m1(a, b); }
HWY_API VU64M2 Min(VU64M2 a, VU64M2 b) { return vminu_vv_u64m2(a, b); }
HWY_API VU64M4 Min(VU64M4 a, VU64M4 b) { return vminu_vv_u64m4(a, b); }
HWY_API VU64M8 Min(VU64M8 a, VU64M8 b) { return vminu_vv_u64m8(a, b); }

HWY_API VI8M1 Min(VI8M1 a, VI8M1 b) { return vmin_vv_i8m1(a, b); }
HWY_API VI8M2 Min(VI8M2 a, VI8M2 b) { return vmin_vv_i8m2(a, b); }
HWY_API VI8M4 Min(VI8M4 a, VI8M4 b) { return vmin_vv_i8m4(a, b); }
HWY_API VI8M8 Min(VI8M8 a, VI8M8 b) { return vmin_vv_i8m8(a, b); }

HWY_API VI16M1 Min(VI16M1 a, VI16M1 b) { return vmin_vv_i16m1(a, b); }
HWY_API VI16M2 Min(VI16M2 a, VI16M2 b) { return vmin_vv_i16m2(a, b); }
HWY_API VI16M4 Min(VI16M4 a, VI16M4 b) { return vmin_vv_i16m4(a, b); }
HWY_API VI16M8 Min(VI16M8 a, VI16M8 b) { return vmin_vv_i16m8(a, b); }

HWY_API VI32M1 Min(VI32M1 a, VI32M1 b) { return vmin_vv_i32m1(a, b); }
HWY_API VI32M2 Min(VI32M2 a, VI32M2 b) { return vmin_vv_i32m2(a, b); }
HWY_API VI32M4 Min(VI32M4 a, VI32M4 b) { return vmin_vv_i32m4(a, b); }
HWY_API VI32M8 Min(VI32M8 a, VI32M8 b) { return vmin_vv_i32m8(a, b); }

HWY_API VI64M1 Min(VI64M1 a, VI64M1 b) { return vmin_vv_i64m1(a, b); }
HWY_API VI64M2 Min(VI64M2 a, VI64M2 b) { return vmin_vv_i64m2(a, b); }
HWY_API VI64M4 Min(VI64M4 a, VI64M4 b) { return vmin_vv_i64m4(a, b); }
HWY_API VI64M8 Min(VI64M8 a, VI64M8 b) { return vmin_vv_i64m8(a, b); }

HWY_API VF32M1 Min(VF32M1 a, VF32M1 b) { return vfmin_vv_f32m1(a, b); }
HWY_API VF32M2 Min(VF32M2 a, VF32M2 b) { return vfmin_vv_f32m2(a, b); }
HWY_API VF32M4 Min(VF32M4 a, VF32M4 b) { return vfmin_vv_f32m4(a, b); }
HWY_API VF32M8 Min(VF32M8 a, VF32M8 b) { return vfmin_vv_f32m8(a, b); }

HWY_API VF64M1 Min(VF64M1 a, VF64M1 b) { return vfmin_vv_f64m1(a, b); }
HWY_API VF64M2 Min(VF64M2 a, VF64M2 b) { return vfmin_vv_f64m2(a, b); }
HWY_API VF64M4 Min(VF64M4 a, VF64M4 b) { return vfmin_vv_f64m4(a, b); }
HWY_API VF64M8 Min(VF64M8 a, VF64M8 b) { return vfmin_vv_f64m8(a, b); }

// ------------------------------ Max

namespace detail {

HWY_API VU8M1 Max(VU8M1 a, uint8_t b) { return vmaxu_vx_u8m1(a, b); }
HWY_API VU8M2 Max(VU8M2 a, uint8_t b) { return vmaxu_vx_u8m2(a, b); }
HWY_API VU8M4 Max(VU8M4 a, uint8_t b) { return vmaxu_vx_u8m4(a, b); }
HWY_API VU8M8 Max(VU8M8 a, uint8_t b) { return vmaxu_vx_u8m8(a, b); }

HWY_API VU16M1 Max(VU16M1 a, uint16_t b) { return vmaxu_vx_u16m1(a, b); }
HWY_API VU16M2 Max(VU16M2 a, uint16_t b) { return vmaxu_vx_u16m2(a, b); }
HWY_API VU16M4 Max(VU16M4 a, uint16_t b) { return vmaxu_vx_u16m4(a, b); }
HWY_API VU16M8 Max(VU16M8 a, uint16_t b) { return vmaxu_vx_u16m8(a, b); }

HWY_API VU32M1 Max(VU32M1 a, uint32_t b) { return vmaxu_vx_u32m1(a, b); }
HWY_API VU32M2 Max(VU32M2 a, uint32_t b) { return vmaxu_vx_u32m2(a, b); }
HWY_API VU32M4 Max(VU32M4 a, uint32_t b) { return vmaxu_vx_u32m4(a, b); }
HWY_API VU32M8 Max(VU32M8 a, uint32_t b) { return vmaxu_vx_u32m8(a, b); }

HWY_API VU64M1 Max(VU64M1 a, uint64_t b) { return vmaxu_vx_u64m1(a, b); }
HWY_API VU64M2 Max(VU64M2 a, uint64_t b) { return vmaxu_vx_u64m2(a, b); }
HWY_API VU64M4 Max(VU64M4 a, uint64_t b) { return vmaxu_vx_u64m4(a, b); }
HWY_API VU64M8 Max(VU64M8 a, uint64_t b) { return vmaxu_vx_u64m8(a, b); }

HWY_API VI8M1 Max(VI8M1 a, int8_t b) { return vmax_vx_i8m1(a, b); }
HWY_API VI8M2 Max(VI8M2 a, int8_t b) { return vmax_vx_i8m2(a, b); }
HWY_API VI8M4 Max(VI8M4 a, int8_t b) { return vmax_vx_i8m4(a, b); }
HWY_API VI8M8 Max(VI8M8 a, int8_t b) { return vmax_vx_i8m8(a, b); }

HWY_API VI16M1 Max(VI16M1 a, int16_t b) { return vmax_vx_i16m1(a, b); }
HWY_API VI16M2 Max(VI16M2 a, int16_t b) { return vmax_vx_i16m2(a, b); }
HWY_API VI16M4 Max(VI16M4 a, int16_t b) { return vmax_vx_i16m4(a, b); }
HWY_API VI16M8 Max(VI16M8 a, int16_t b) { return vmax_vx_i16m8(a, b); }

HWY_API VI32M1 Max(VI32M1 a, int32_t b) { return vmax_vx_i32m1(a, b); }
HWY_API VI32M2 Max(VI32M2 a, int32_t b) { return vmax_vx_i32m2(a, b); }
HWY_API VI32M4 Max(VI32M4 a, int32_t b) { return vmax_vx_i32m4(a, b); }
HWY_API VI32M8 Max(VI32M8 a, int32_t b) { return vmax_vx_i32m8(a, b); }

HWY_API VI64M1 Max(VI64M1 a, int64_t b) { return vmax_vx_i64m1(a, b); }
HWY_API VI64M2 Max(VI64M2 a, int64_t b) { return vmax_vx_i64m2(a, b); }
HWY_API VI64M4 Max(VI64M4 a, int64_t b) { return vmax_vx_i64m4(a, b); }
HWY_API VI64M8 Max(VI64M8 a, int64_t b) { return vmax_vx_i64m8(a, b); }

HWY_API VF32M1 Max(VF32M1 a, float b) { return vfmax_vf_f32m1(a, b); }
HWY_API VF32M2 Max(VF32M2 a, float b) { return vfmax_vf_f32m2(a, b); }
HWY_API VF32M4 Max(VF32M4 a, float b) { return vfmax_vf_f32m4(a, b); }
HWY_API VF32M8 Max(VF32M8 a, float b) { return vfmax_vf_f32m8(a, b); }

HWY_API VF64M1 Max(VF64M1 a, double b) { return vfmax_vf_f64m1(a, b); }
HWY_API VF64M2 Max(VF64M2 a, double b) { return vfmax_vf_f64m2(a, b); }
HWY_API VF64M4 Max(VF64M4 a, double b) { return vfmax_vf_f64m4(a, b); }
HWY_API VF64M8 Max(VF64M8 a, double b) { return vfmax_vf_f64m8(a, b); }

}  // namespace detail

HWY_API VU8M1 Max(VU8M1 a, VU8M1 b) { return vmaxu_vv_u8m1(a, b); }
HWY_API VU8M2 Max(VU8M2 a, VU8M2 b) { return vmaxu_vv_u8m2(a, b); }
HWY_API VU8M4 Max(VU8M4 a, VU8M4 b) { return vmaxu_vv_u8m4(a, b); }
HWY_API VU8M8 Max(VU8M8 a, VU8M8 b) { return vmaxu_vv_u8m8(a, b); }

HWY_API VU16M1 Max(VU16M1 a, VU16M1 b) { return vmaxu_vv_u16m1(a, b); }
HWY_API VU16M2 Max(VU16M2 a, VU16M2 b) { return vmaxu_vv_u16m2(a, b); }
HWY_API VU16M4 Max(VU16M4 a, VU16M4 b) { return vmaxu_vv_u16m4(a, b); }
HWY_API VU16M8 Max(VU16M8 a, VU16M8 b) { return vmaxu_vv_u16m8(a, b); }

HWY_API VU32M1 Max(VU32M1 a, VU32M1 b) { return vmaxu_vv_u32m1(a, b); }
HWY_API VU32M2 Max(VU32M2 a, VU32M2 b) { return vmaxu_vv_u32m2(a, b); }
HWY_API VU32M4 Max(VU32M4 a, VU32M4 b) { return vmaxu_vv_u32m4(a, b); }
HWY_API VU32M8 Max(VU32M8 a, VU32M8 b) { return vmaxu_vv_u32m8(a, b); }

HWY_API VU64M1 Max(VU64M1 a, VU64M1 b) { return vmaxu_vv_u64m1(a, b); }
HWY_API VU64M2 Max(VU64M2 a, VU64M2 b) { return vmaxu_vv_u64m2(a, b); }
HWY_API VU64M4 Max(VU64M4 a, VU64M4 b) { return vmaxu_vv_u64m4(a, b); }
HWY_API VU64M8 Max(VU64M8 a, VU64M8 b) { return vmaxu_vv_u64m8(a, b); }

HWY_API VI8M1 Max(VI8M1 a, VI8M1 b) { return vmax_vv_i8m1(a, b); }
HWY_API VI8M2 Max(VI8M2 a, VI8M2 b) { return vmax_vv_i8m2(a, b); }
HWY_API VI8M4 Max(VI8M4 a, VI8M4 b) { return vmax_vv_i8m4(a, b); }
HWY_API VI8M8 Max(VI8M8 a, VI8M8 b) { return vmax_vv_i8m8(a, b); }

HWY_API VI16M1 Max(VI16M1 a, VI16M1 b) { return vmax_vv_i16m1(a, b); }
HWY_API VI16M2 Max(VI16M2 a, VI16M2 b) { return vmax_vv_i16m2(a, b); }
HWY_API VI16M4 Max(VI16M4 a, VI16M4 b) { return vmax_vv_i16m4(a, b); }
HWY_API VI16M8 Max(VI16M8 a, VI16M8 b) { return vmax_vv_i16m8(a, b); }

HWY_API VI32M1 Max(VI32M1 a, VI32M1 b) { return vmax_vv_i32m1(a, b); }
HWY_API VI32M2 Max(VI32M2 a, VI32M2 b) { return vmax_vv_i32m2(a, b); }
HWY_API VI32M4 Max(VI32M4 a, VI32M4 b) { return vmax_vv_i32m4(a, b); }
HWY_API VI32M8 Max(VI32M8 a, VI32M8 b) { return vmax_vv_i32m8(a, b); }

HWY_API VI64M1 Max(VI64M1 a, VI64M1 b) { return vmax_vv_i64m1(a, b); }
HWY_API VI64M2 Max(VI64M2 a, VI64M2 b) { return vmax_vv_i64m2(a, b); }
HWY_API VI64M4 Max(VI64M4 a, VI64M4 b) { return vmax_vv_i64m4(a, b); }
HWY_API VI64M8 Max(VI64M8 a, VI64M8 b) { return vmax_vv_i64m8(a, b); }

HWY_API VF32M1 Max(VF32M1 a, VF32M1 b) { return vfmax_vv_f32m1(a, b); }
HWY_API VF32M2 Max(VF32M2 a, VF32M2 b) { return vfmax_vv_f32m2(a, b); }
HWY_API VF32M4 Max(VF32M4 a, VF32M4 b) { return vfmax_vv_f32m4(a, b); }
HWY_API VF32M8 Max(VF32M8 a, VF32M8 b) { return vfmax_vv_f32m8(a, b); }

HWY_API VF64M1 Max(VF64M1 a, VF64M1 b) { return vfmax_vv_f64m1(a, b); }
HWY_API VF64M2 Max(VF64M2 a, VF64M2 b) { return vfmax_vv_f64m2(a, b); }
HWY_API VF64M4 Max(VF64M4 a, VF64M4 b) { return vfmax_vv_f64m4(a, b); }
HWY_API VF64M8 Max(VF64M8 a, VF64M8 b) { return vfmax_vv_f64m8(a, b); }

// ------------------------------ Mul

HWY_API VU16M1 Mul(VU16M1 a, VU16M1 b) { return vmul_vv_u16m1(a, b); }
HWY_API VU16M2 Mul(VU16M2 a, VU16M2 b) { return vmul_vv_u16m2(a, b); }
HWY_API VU16M4 Mul(VU16M4 a, VU16M4 b) { return vmul_vv_u16m4(a, b); }
HWY_API VU16M8 Mul(VU16M8 a, VU16M8 b) { return vmul_vv_u16m8(a, b); }

HWY_API VU32M1 Mul(VU32M1 a, VU32M1 b) { return vmul_vv_u32m1(a, b); }
HWY_API VU32M2 Mul(VU32M2 a, VU32M2 b) { return vmul_vv_u32m2(a, b); }
HWY_API VU32M4 Mul(VU32M4 a, VU32M4 b) { return vmul_vv_u32m4(a, b); }
HWY_API VU32M8 Mul(VU32M8 a, VU32M8 b) { return vmul_vv_u32m8(a, b); }

HWY_API VI16M1 Mul(VI16M1 a, VI16M1 b) { return vmul_vv_i16m1(a, b); }
HWY_API VI16M2 Mul(VI16M2 a, VI16M2 b) { return vmul_vv_i16m2(a, b); }
HWY_API VI16M4 Mul(VI16M4 a, VI16M4 b) { return vmul_vv_i16m4(a, b); }
HWY_API VI16M8 Mul(VI16M8 a, VI16M8 b) { return vmul_vv_i16m8(a, b); }

HWY_API VI32M1 Mul(VI32M1 a, VI32M1 b) { return vmul_vv_i32m1(a, b); }
HWY_API VI32M2 Mul(VI32M2 a, VI32M2 b) { return vmul_vv_i32m2(a, b); }
HWY_API VI32M4 Mul(VI32M4 a, VI32M4 b) { return vmul_vv_i32m4(a, b); }
HWY_API VI32M8 Mul(VI32M8 a, VI32M8 b) { return vmul_vv_i32m8(a, b); }

// ------------------------------ MulHigh

HWY_API VU16M1 MulHigh(VU16M1 a, VU16M1 b) { return vmulhu_vv_u16m1(a, b); }
HWY_API VU16M2 MulHigh(VU16M2 a, VU16M2 b) { return vmulhu_vv_u16m2(a, b); }
HWY_API VU16M4 MulHigh(VU16M4 a, VU16M4 b) { return vmulhu_vv_u16m4(a, b); }
HWY_API VU16M8 MulHigh(VU16M8 a, VU16M8 b) { return vmulhu_vv_u16m8(a, b); }

HWY_API VI16M1 MulHigh(VI16M1 a, VI16M1 b) { return vmulh_vv_i16m1(a, b); }
HWY_API VI16M2 MulHigh(VI16M2 a, VI16M2 b) { return vmulh_vv_i16m2(a, b); }
HWY_API VI16M4 MulHigh(VI16M4 a, VI16M4 b) { return vmulh_vv_i16m4(a, b); }
HWY_API VI16M8 MulHigh(VI16M8 a, VI16M8 b) { return vmulh_vv_i16m8(a, b); }

// ------------------------------ Mul

HWY_API VF32M1 Mul(VF32M1 a, VF32M1 b) { return vfmul_vv_f32m1(a, b); }
HWY_API VF32M2 Mul(VF32M2 a, VF32M2 b) { return vfmul_vv_f32m2(a, b); }
HWY_API VF32M4 Mul(VF32M4 a, VF32M4 b) { return vfmul_vv_f32m4(a, b); }
HWY_API VF32M8 Mul(VF32M8 a, VF32M8 b) { return vfmul_vv_f32m8(a, b); }

HWY_API VF64M1 Mul(VF64M1 a, VF64M1 b) { return vfmul_vv_f64m1(a, b); }
HWY_API VF64M2 Mul(VF64M2 a, VF64M2 b) { return vfmul_vv_f64m2(a, b); }
HWY_API VF64M4 Mul(VF64M4 a, VF64M4 b) { return vfmul_vv_f64m4(a, b); }
HWY_API VF64M8 Mul(VF64M8 a, VF64M8 b) { return vfmul_vv_f64m8(a, b); }

// ------------------------------ Div

HWY_API VF32M1 Div(VF32M1 a, VF32M1 b) { return vfdiv_vv_f32m1(a, b); }
HWY_API VF32M2 Div(VF32M2 a, VF32M2 b) { return vfdiv_vv_f32m2(a, b); }
HWY_API VF32M4 Div(VF32M4 a, VF32M4 b) { return vfdiv_vv_f32m4(a, b); }
HWY_API VF32M8 Div(VF32M8 a, VF32M8 b) { return vfdiv_vv_f32m8(a, b); }

HWY_API VF64M1 Div(VF64M1 a, VF64M1 b) { return vfdiv_vv_f64m1(a, b); }
HWY_API VF64M2 Div(VF64M2 a, VF64M2 b) { return vfdiv_vv_f64m2(a, b); }
HWY_API VF64M4 Div(VF64M4 a, VF64M4 b) { return vfdiv_vv_f64m4(a, b); }
HWY_API VF64M8 Div(VF64M8 a, VF64M8 b) { return vfdiv_vv_f64m8(a, b); }

// ------------------------------ ApproximateReciprocal

// TODO(janwas): not yet supported in intrinsics
template <class V>
HWY_API V ApproximateReciprocal(const V v) {
  return Set(DFromV<V>(), 1) / v;
}

// HWY_API VF32M1 ApproximateReciprocal(const VF32M1 v) {
//  return vfrece7_v_f32m1(v);
//}
// HWY_API VF32M2 ApproximateReciprocal(const VF32M2 v) {
//  return vfrece7_v_f32m2(v);
//}
// HWY_API VF32M4 ApproximateReciprocal(const VF32M4 v) {
//  return vfrece7_v_f32m4(v);
//}
// HWY_API VF32M8 ApproximateReciprocal(const VF32M8 v) {
//  return vfrece7_v_f32m8(v);
//}

// ------------------------------ Sqrt

HWY_API VF32M1 Sqrt(const VF32M1 v) { return vfsqrt_v_f32m1(v); }
HWY_API VF32M2 Sqrt(const VF32M2 v) { return vfsqrt_v_f32m2(v); }
HWY_API VF32M4 Sqrt(const VF32M4 v) { return vfsqrt_v_f32m4(v); }
HWY_API VF32M8 Sqrt(const VF32M8 v) { return vfsqrt_v_f32m8(v); }

HWY_API VF64M1 Sqrt(const VF64M1 v) { return vfsqrt_v_f64m1(v); }
HWY_API VF64M2 Sqrt(const VF64M2 v) { return vfsqrt_v_f64m2(v); }
HWY_API VF64M4 Sqrt(const VF64M4 v) { return vfsqrt_v_f64m4(v); }
HWY_API VF64M8 Sqrt(const VF64M8 v) { return vfsqrt_v_f64m8(v); }

// ------------------------------ ApproximateReciprocalSqrt

// TODO(janwas): not yet supported in intrinsics
template <class V>
HWY_API V ApproximateReciprocalSqrt(const V v) {
  return ApproximateReciprocal(Sqrt(v));
}

// HWY_API VF32M1 ApproximateReciprocalSqrt(const VF32M1 v) {
//  return vfrsqrte7_v_f32m1(v);
//}
// HWY_API VF32M2 ApproximateReciprocalSqrt(const VF32M2 v) {
//  return vfrsqrte7_v_f32m2(v);
//}
// HWY_API VF32M4 ApproximateReciprocalSqrt(const VF32M4 v) {
//  return vfrsqrte7_v_f32m4(v);
//}
// HWY_API VF32M8 ApproximateReciprocalSqrt(const VF32M8 v) {
//  return vfrsqrte7_v_f32m8(v);
//}

// ------------------------------ MulAdd

HWY_API VF32M1 MulAdd(const VF32M1 mul, const VF32M1 x, const VF32M1 add) {
  return vfmacc_vv_f32m1(add, mul, x);
}
HWY_API VF32M2 MulAdd(const VF32M2 mul, const VF32M2 x, const VF32M2 add) {
  return vfmacc_vv_f32m2(add, mul, x);
}
HWY_API VF32M4 MulAdd(const VF32M4 mul, const VF32M4 x, const VF32M4 add) {
  return vfmacc_vv_f32m4(add, mul, x);
}
HWY_API VF32M8 MulAdd(const VF32M8 mul, const VF32M8 x, const VF32M8 add) {
  return vfmacc_vv_f32m8(add, mul, x);
}

HWY_API VF64M1 MulAdd(const VF64M1 mul, const VF64M1 x, const VF64M1 add) {
  return vfmacc_vv_f64m1(add, mul, x);
}
HWY_API VF64M2 MulAdd(const VF64M2 mul, const VF64M2 x, const VF64M2 add) {
  return vfmacc_vv_f64m2(add, mul, x);
}
HWY_API VF64M4 MulAdd(const VF64M4 mul, const VF64M4 x, const VF64M4 add) {
  return vfmacc_vv_f64m4(add, mul, x);
}
HWY_API VF64M8 MulAdd(const VF64M8 mul, const VF64M8 x, const VF64M8 add) {
  return vfmacc_vv_f64m8(add, mul, x);
}

// ------------------------------ NegMulAdd

HWY_API VF32M1 NegMulAdd(const VF32M1 mul, const VF32M1 x, const VF32M1 add) {
  return vfnmsac_vv_f32m1(add, mul, x);
}
HWY_API VF32M2 NegMulAdd(const VF32M2 mul, const VF32M2 x, const VF32M2 add) {
  return vfnmsac_vv_f32m2(add, mul, x);
}
HWY_API VF32M4 NegMulAdd(const VF32M4 mul, const VF32M4 x, const VF32M4 add) {
  return vfnmsac_vv_f32m4(add, mul, x);
}
HWY_API VF32M8 NegMulAdd(const VF32M8 mul, const VF32M8 x, const VF32M8 add) {
  return vfnmsac_vv_f32m8(add, mul, x);
}

HWY_API VF64M1 NegMulAdd(const VF64M1 mul, const VF64M1 x, const VF64M1 add) {
  return vfnmsac_vv_f64m1(add, mul, x);
}
HWY_API VF64M2 NegMulAdd(const VF64M2 mul, const VF64M2 x, const VF64M2 add) {
  return vfnmsac_vv_f64m2(add, mul, x);
}
HWY_API VF64M4 NegMulAdd(const VF64M4 mul, const VF64M4 x, const VF64M4 add) {
  return vfnmsac_vv_f64m4(add, mul, x);
}
HWY_API VF64M8 NegMulAdd(const VF64M8 mul, const VF64M8 x, const VF64M8 add) {
  return vfnmsac_vv_f64m8(add, mul, x);
}

// ------------------------------ MulSub

HWY_API VF32M1 MulSub(const VF32M1 mul, const VF32M1 x, const VF32M1 sub) {
  return vfmsac_vv_f32m1(sub, mul, x);
}
HWY_API VF32M2 MulSub(const VF32M2 mul, const VF32M2 x, const VF32M2 sub) {
  return vfmsac_vv_f32m2(sub, mul, x);
}
HWY_API VF32M4 MulSub(const VF32M4 mul, const VF32M4 x, const VF32M4 sub) {
  return vfmsac_vv_f32m4(sub, mul, x);
}
HWY_API VF32M8 MulSub(const VF32M8 mul, const VF32M8 x, const VF32M8 sub) {
  return vfmsac_vv_f32m8(sub, mul, x);
}

HWY_API VF64M1 MulSub(const VF64M1 mul, const VF64M1 x, const VF64M1 sub) {
  return vfmsac_vv_f64m1(sub, mul, x);
}
HWY_API VF64M2 MulSub(const VF64M2 mul, const VF64M2 x, const VF64M2 sub) {
  return vfmsac_vv_f64m2(sub, mul, x);
}
HWY_API VF64M4 MulSub(const VF64M4 mul, const VF64M4 x, const VF64M4 sub) {
  return vfmsac_vv_f64m4(sub, mul, x);
}
HWY_API VF64M8 MulSub(const VF64M8 mul, const VF64M8 x, const VF64M8 sub) {
  return vfmsac_vv_f64m8(sub, mul, x);
}

// ------------------------------ NegMulSub

HWY_API VF32M1 NegMulSub(const VF32M1 mul, const VF32M1 x, const VF32M1 sub) {
  return vfnmacc_vv_f32m1(sub, mul, x);
}
HWY_API VF32M2 NegMulSub(const VF32M2 mul, const VF32M2 x, const VF32M2 sub) {
  return vfnmacc_vv_f32m2(sub, mul, x);
}
HWY_API VF32M4 NegMulSub(const VF32M4 mul, const VF32M4 x, const VF32M4 sub) {
  return vfnmacc_vv_f32m4(sub, mul, x);
}
HWY_API VF32M8 NegMulSub(const VF32M8 mul, const VF32M8 x, const VF32M8 sub) {
  return vfnmacc_vv_f32m8(sub, mul, x);
}

HWY_API VF64M1 NegMulSub(const VF64M1 mul, const VF64M1 x, const VF64M1 sub) {
  return vfnmacc_vv_f64m1(sub, mul, x);
}
HWY_API VF64M2 NegMulSub(const VF64M2 mul, const VF64M2 x, const VF64M2 sub) {
  return vfnmacc_vv_f64m2(sub, mul, x);
}
HWY_API VF64M4 NegMulSub(const VF64M4 mul, const VF64M4 x, const VF64M4 sub) {
  return vfnmacc_vv_f64m4(sub, mul, x);
}
HWY_API VF64M8 NegMulSub(const VF64M8 mul, const VF64M8 x, const VF64M8 sub) {
  return vfnmacc_vv_f64m8(sub, mul, x);
}

// ------------------------------ Round

// TODO(janwas): not yet in spec

HWY_API VF32M1 Round(const VF32M1 v) { return v; }

// ------------------------------ Trunc

HWY_API VF32M1 Trunc(const VF32M1 v) { return v; }

// ------------------------------ Ceil

HWY_API VF32M1 Ceil(const VF32M1 v) { return v; }

// ------------------------------ Floor

HWY_API VF32M1 Floor(const VF32M1 v) { return v; }

// ================================================== COMPARE

// Comparisons set a mask bit to 1 if the condition is true, else 0. The XX in
// vboolXX_t is a power of two divisor for vector bits. SLEN 8 / LMUL 1 = 1/8th
// of all bits; SLEN 8 / LMUL 4 = half of all bits.

// ------------------------------ Eq

// Unsigned
HWY_API vbool8_t Eq(VU8M1 a, VU8M1 b) { return vmseq_vv_u8m1_b8(a, b); }
HWY_API vbool4_t Eq(VU8M2 a, VU8M2 b) { return vmseq_vv_u8m2_b4(a, b); }
HWY_API vbool2_t Eq(VU8M4 a, VU8M4 b) { return vmseq_vv_u8m4_b2(a, b); }
HWY_API vbool1_t Eq(VU8M8 a, VU8M8 b) { return vmseq_vv_u8m8_b1(a, b); }

HWY_API vbool16_t Eq(VU16M1 a, VU16M1 b) { return vmseq_vv_u16m1_b16(a, b); }
HWY_API vbool8_t Eq(VU16M2 a, VU16M2 b) { return vmseq_vv_u16m2_b8(a, b); }
HWY_API vbool4_t Eq(VU16M4 a, VU16M4 b) { return vmseq_vv_u16m4_b4(a, b); }
HWY_API vbool2_t Eq(VU16M8 a, VU16M8 b) { return vmseq_vv_u16m8_b2(a, b); }

HWY_API vbool32_t Eq(VU32M1 a, VU32M1 b) { return vmseq_vv_u32m1_b32(a, b); }
HWY_API vbool16_t Eq(VU32M2 a, VU32M2 b) { return vmseq_vv_u32m2_b16(a, b); }
HWY_API vbool8_t Eq(VU32M4 a, VU32M4 b) { return vmseq_vv_u32m4_b8(a, b); }
HWY_API vbool4_t Eq(VU32M8 a, VU32M8 b) { return vmseq_vv_u32m8_b4(a, b); }

HWY_API vbool64_t Eq(VU64M1 a, VU64M1 b) { return vmseq_vv_u64m1_b64(a, b); }
HWY_API vbool32_t Eq(VU64M2 a, VU64M2 b) { return vmseq_vv_u64m2_b32(a, b); }
HWY_API vbool16_t Eq(VU64M4 a, VU64M4 b) { return vmseq_vv_u64m4_b16(a, b); }
HWY_API vbool8_t Eq(VU64M8 a, VU64M8 b) { return vmseq_vv_u64m8_b8(a, b); }

// Signed
HWY_API vbool8_t Eq(VI8M1 a, VI8M1 b) { return vmseq_vv_i8m1_b8(a, b); }
HWY_API vbool4_t Eq(VI8M2 a, VI8M2 b) { return vmseq_vv_i8m2_b4(a, b); }
HWY_API vbool2_t Eq(VI8M4 a, VI8M4 b) { return vmseq_vv_i8m4_b2(a, b); }
HWY_API vbool1_t Eq(VI8M8 a, VI8M8 b) { return vmseq_vv_i8m8_b1(a, b); }

HWY_API vbool16_t Eq(VI16M1 a, VI16M1 b) { return vmseq_vv_i16m1_b16(a, b); }
HWY_API vbool8_t Eq(VI16M2 a, VI16M2 b) { return vmseq_vv_i16m2_b8(a, b); }
HWY_API vbool4_t Eq(VI16M4 a, VI16M4 b) { return vmseq_vv_i16m4_b4(a, b); }
HWY_API vbool2_t Eq(VI16M8 a, VI16M8 b) { return vmseq_vv_i16m8_b2(a, b); }

HWY_API vbool32_t Eq(VI32M1 a, VI32M1 b) { return vmseq_vv_i32m1_b32(a, b); }
HWY_API vbool16_t Eq(VI32M2 a, VI32M2 b) { return vmseq_vv_i32m2_b16(a, b); }
HWY_API vbool8_t Eq(VI32M4 a, VI32M4 b) { return vmseq_vv_i32m4_b8(a, b); }
HWY_API vbool4_t Eq(VI32M8 a, VI32M8 b) { return vmseq_vv_i32m8_b4(a, b); }

HWY_API vbool64_t Eq(VI64M1 a, VI64M1 b) { return vmseq_vv_i64m1_b64(a, b); }
HWY_API vbool32_t Eq(VI64M2 a, VI64M2 b) { return vmseq_vv_i64m2_b32(a, b); }
HWY_API vbool16_t Eq(VI64M4 a, VI64M4 b) { return vmseq_vv_i64m4_b16(a, b); }
HWY_API vbool8_t Eq(VI64M8 a, VI64M8 b) { return vmseq_vv_i64m8_b8(a, b); }

// Float
HWY_API vbool32_t Eq(VF32M1 a, VF32M1 b) { return vmfeq_vv_f32m1_b32(a, b); }
HWY_API vbool16_t Eq(VF32M2 a, VF32M2 b) { return vmfeq_vv_f32m2_b16(a, b); }
HWY_API vbool8_t Eq(VF32M4 a, VF32M4 b) { return vmfeq_vv_f32m4_b8(a, b); }
HWY_API vbool4_t Eq(VF32M8 a, VF32M8 b) { return vmfeq_vv_f32m8_b4(a, b); }

HWY_API vbool64_t Eq(VF64M1 a, VF64M1 b) { return vmfeq_vv_f64m1_b64(a, b); }
HWY_API vbool32_t Eq(VF64M2 a, VF64M2 b) { return vmfeq_vv_f64m2_b32(a, b); }
HWY_API vbool16_t Eq(VF64M4 a, VF64M4 b) { return vmfeq_vv_f64m4_b16(a, b); }
HWY_API vbool8_t Eq(VF64M8 a, VF64M8 b) { return vmfeq_vv_f64m8_b8(a, b); }

// ------------------------------ Ne

HWY_API vbool8_t Ne(VU8M1 a, VU8M1 b) { return vmsne_vv_u8m1_b8(a, b); }
HWY_API vbool4_t Ne(VU8M2 a, VU8M2 b) { return vmsne_vv_u8m2_b4(a, b); }
HWY_API vbool2_t Ne(VU8M4 a, VU8M4 b) { return vmsne_vv_u8m4_b2(a, b); }
HWY_API vbool1_t Ne(VU8M8 a, VU8M8 b) { return vmsne_vv_u8m8_b1(a, b); }

HWY_API vbool16_t Ne(VU16M1 a, VU16M1 b) { return vmsne_vv_u16m1_b16(a, b); }
HWY_API vbool8_t Ne(VU16M2 a, VU16M2 b) { return vmsne_vv_u16m2_b8(a, b); }
HWY_API vbool4_t Ne(VU16M4 a, VU16M4 b) { return vmsne_vv_u16m4_b4(a, b); }
HWY_API vbool2_t Ne(VU16M8 a, VU16M8 b) { return vmsne_vv_u16m8_b2(a, b); }

HWY_API vbool32_t Ne(VU32M1 a, VU32M1 b) { return vmsne_vv_u32m1_b32(a, b); }
HWY_API vbool16_t Ne(VU32M2 a, VU32M2 b) { return vmsne_vv_u32m2_b16(a, b); }
HWY_API vbool8_t Ne(VU32M4 a, VU32M4 b) { return vmsne_vv_u32m4_b8(a, b); }
HWY_API vbool4_t Ne(VU32M8 a, VU32M8 b) { return vmsne_vv_u32m8_b4(a, b); }

HWY_API vbool64_t Ne(VU64M1 a, VU64M1 b) { return vmsne_vv_u64m1_b64(a, b); }
HWY_API vbool32_t Ne(VU64M2 a, VU64M2 b) { return vmsne_vv_u64m2_b32(a, b); }
HWY_API vbool16_t Ne(VU64M4 a, VU64M4 b) { return vmsne_vv_u64m4_b16(a, b); }
HWY_API vbool8_t Ne(VU64M8 a, VU64M8 b) { return vmsne_vv_u64m8_b8(a, b); }

HWY_API vbool8_t Ne(VI8M1 a, VI8M1 b) { return vmsne_vv_i8m1_b8(a, b); }
HWY_API vbool4_t Ne(VI8M2 a, VI8M2 b) { return vmsne_vv_i8m2_b4(a, b); }
HWY_API vbool2_t Ne(VI8M4 a, VI8M4 b) { return vmsne_vv_i8m4_b2(a, b); }
HWY_API vbool1_t Ne(VI8M8 a, VI8M8 b) { return vmsne_vv_i8m8_b1(a, b); }

HWY_API vbool16_t Ne(VI16M1 a, VI16M1 b) { return vmsne_vv_i16m1_b16(a, b); }
HWY_API vbool8_t Ne(VI16M2 a, VI16M2 b) { return vmsne_vv_i16m2_b8(a, b); }
HWY_API vbool4_t Ne(VI16M4 a, VI16M4 b) { return vmsne_vv_i16m4_b4(a, b); }
HWY_API vbool2_t Ne(VI16M8 a, VI16M8 b) { return vmsne_vv_i16m8_b2(a, b); }

HWY_API vbool32_t Ne(VI32M1 a, VI32M1 b) { return vmsne_vv_i32m1_b32(a, b); }
HWY_API vbool16_t Ne(VI32M2 a, VI32M2 b) { return vmsne_vv_i32m2_b16(a, b); }
HWY_API vbool8_t Ne(VI32M4 a, VI32M4 b) { return vmsne_vv_i32m4_b8(a, b); }
HWY_API vbool4_t Ne(VI32M8 a, VI32M8 b) { return vmsne_vv_i32m8_b4(a, b); }

HWY_API vbool64_t Ne(VI64M1 a, VI64M1 b) { return vmsne_vv_i64m1_b64(a, b); }
HWY_API vbool32_t Ne(VI64M2 a, VI64M2 b) { return vmsne_vv_i64m2_b32(a, b); }
HWY_API vbool16_t Ne(VI64M4 a, VI64M4 b) { return vmsne_vv_i64m4_b16(a, b); }
HWY_API vbool8_t Ne(VI64M8 a, VI64M8 b) { return vmsne_vv_i64m8_b8(a, b); }

HWY_API vbool32_t Ne(VF32M1 a, VF32M1 b) { return vmfne_vv_f32m1_b32(a, b); }
HWY_API vbool16_t Ne(VF32M2 a, VF32M2 b) { return vmfne_vv_f32m2_b16(a, b); }
HWY_API vbool8_t Ne(VF32M4 a, VF32M4 b) { return vmfne_vv_f32m4_b8(a, b); }
HWY_API vbool4_t Ne(VF32M8 a, VF32M8 b) { return vmfne_vv_f32m8_b4(a, b); }

HWY_API vbool64_t Ne(VF64M1 a, VF64M1 b) { return vmfne_vv_f64m1_b64(a, b); }
HWY_API vbool32_t Ne(VF64M2 a, VF64M2 b) { return vmfne_vv_f64m2_b32(a, b); }
HWY_API vbool16_t Ne(VF64M4 a, VF64M4 b) { return vmfne_vv_f64m4_b16(a, b); }
HWY_API vbool8_t Ne(VF64M8 a, VF64M8 b) { return vmfne_vv_f64m8_b8(a, b); }

// ------------------------------ Lt

HWY_API vbool8_t Lt(VI8M1 a, VI8M1 b) { return vmslt_vv_i8m1_b8(a, b); }
HWY_API vbool4_t Lt(VI8M2 a, VI8M2 b) { return vmslt_vv_i8m2_b4(a, b); }
HWY_API vbool2_t Lt(VI8M4 a, VI8M4 b) { return vmslt_vv_i8m4_b2(a, b); }
HWY_API vbool1_t Lt(VI8M8 a, VI8M8 b) { return vmslt_vv_i8m8_b1(a, b); }

HWY_API vbool16_t Lt(VI16M1 a, VI16M1 b) { return vmslt_vv_i16m1_b16(a, b); }
HWY_API vbool8_t Lt(VI16M2 a, VI16M2 b) { return vmslt_vv_i16m2_b8(a, b); }
HWY_API vbool4_t Lt(VI16M4 a, VI16M4 b) { return vmslt_vv_i16m4_b4(a, b); }
HWY_API vbool2_t Lt(VI16M8 a, VI16M8 b) { return vmslt_vv_i16m8_b2(a, b); }

HWY_API vbool32_t Lt(VI32M1 a, VI32M1 b) { return vmslt_vv_i32m1_b32(a, b); }
HWY_API vbool16_t Lt(VI32M2 a, VI32M2 b) { return vmslt_vv_i32m2_b16(a, b); }
HWY_API vbool8_t Lt(VI32M4 a, VI32M4 b) { return vmslt_vv_i32m4_b8(a, b); }
HWY_API vbool4_t Lt(VI32M8 a, VI32M8 b) { return vmslt_vv_i32m8_b4(a, b); }

HWY_API vbool64_t Lt(VI64M1 a, VI64M1 b) { return vmslt_vv_i64m1_b64(a, b); }
HWY_API vbool32_t Lt(VI64M2 a, VI64M2 b) { return vmslt_vv_i64m2_b32(a, b); }
HWY_API vbool16_t Lt(VI64M4 a, VI64M4 b) { return vmslt_vv_i64m4_b16(a, b); }
HWY_API vbool8_t Lt(VI64M8 a, VI64M8 b) { return vmslt_vv_i64m8_b8(a, b); }

HWY_API vbool32_t Lt(VF32M1 a, VF32M1 b) { return vmfeq_vv_f32m1_b32(a, b); }
HWY_API vbool16_t Lt(VF32M2 a, VF32M2 b) { return vmfeq_vv_f32m2_b16(a, b); }
HWY_API vbool8_t Lt(VF32M4 a, VF32M4 b) { return vmfeq_vv_f32m4_b8(a, b); }
HWY_API vbool4_t Lt(VF32M8 a, VF32M8 b) { return vmfeq_vv_f32m8_b4(a, b); }

HWY_API vbool64_t Lt(VF64M1 a, VF64M1 b) { return vmfeq_vv_f64m1_b64(a, b); }
HWY_API vbool32_t Lt(VF64M2 a, VF64M2 b) { return vmfeq_vv_f64m2_b32(a, b); }
HWY_API vbool16_t Lt(VF64M4 a, VF64M4 b) { return vmfeq_vv_f64m4_b16(a, b); }
HWY_API vbool8_t Lt(VF64M8 a, VF64M8 b) { return vmfeq_vv_f64m8_b8(a, b); }

// ------------------------------ Gt

template <class V>
HWY_API auto Gt(const V a, const V b) -> decltype(Lt(a, b)) {
  return Lt(b, a);
}

// ------------------------------ Le

HWY_API vbool32_t Le(VF32M1 a, VF32M1 b) { return vmfle_vv_f32m1_b32(a, b); }
HWY_API vbool16_t Le(VF32M2 a, VF32M2 b) { return vmfle_vv_f32m2_b16(a, b); }
HWY_API vbool8_t Le(VF32M4 a, VF32M4 b) { return vmfle_vv_f32m4_b8(a, b); }
HWY_API vbool4_t Le(VF32M8 a, VF32M8 b) { return vmfle_vv_f32m8_b4(a, b); }

HWY_API vbool64_t Le(VF64M1 a, VF64M1 b) { return vmfle_vv_f64m1_b64(a, b); }
HWY_API vbool32_t Le(VF64M2 a, VF64M2 b) { return vmfle_vv_f64m2_b32(a, b); }
HWY_API vbool16_t Le(VF64M4 a, VF64M4 b) { return vmfle_vv_f64m4_b16(a, b); }
HWY_API vbool8_t Le(VF64M8 a, VF64M8 b) { return vmfle_vv_f64m8_b8(a, b); }

// ------------------------------ Ge

template <class V>
HWY_API auto Ge(const V a, const V b) -> decltype(Le(a, b)) {
  return Le(b, a);
}

// ------------------------------ TestBit

template <class V>
HWY_API auto TestBit(const V a, const V bit) -> decltype(Eq(a, bit)) {
  return Ne(And(a, bit), Zero(DFromV<V>()));
}

// ------------------------------ And
HWY_API vbool1_t And(vbool1_t a, vbool1_t b) { return vmand_mm_b1(a, b); }
HWY_API vbool2_t And(vbool2_t a, vbool2_t b) { return vmand_mm_b2(a, b); }
HWY_API vbool4_t And(vbool4_t a, vbool4_t b) { return vmand_mm_b4(a, b); }
HWY_API vbool8_t And(vbool8_t a, vbool8_t b) { return vmand_mm_b8(a, b); }
HWY_API vbool16_t And(vbool16_t a, vbool16_t b) { return vmand_mm_b16(a, b); }
HWY_API vbool32_t And(vbool32_t a, vbool32_t b) { return vmand_mm_b32(a, b); }
HWY_API vbool64_t And(vbool64_t a, vbool64_t b) { return vmand_mm_b64(a, b); }

// ------------------------------ AndNot

HWY_API vbool1_t AndNot(vbool1_t a, vbool1_t b) { return vmandnot_mm_b1(a, b); }
HWY_API vbool2_t AndNot(vbool2_t a, vbool2_t b) { return vmandnot_mm_b2(a, b); }
HWY_API vbool4_t AndNot(vbool4_t a, vbool4_t b) { return vmandnot_mm_b4(a, b); }
HWY_API vbool8_t AndNot(vbool8_t a, vbool8_t b) { return vmandnot_mm_b8(a, b); }
HWY_API vbool16_t AndNot(vbool16_t a, vbool16_t b) {
  return vmandnot_mm_b16(a, b);
}
HWY_API vbool32_t AndNot(vbool32_t a, vbool32_t b) {
  return vmandnot_mm_b32(a, b);
}
HWY_API vbool64_t AndNot(vbool64_t a, vbool64_t b) {
  return vmandnot_mm_b64(a, b);
}

// ------------------------------ Or
HWY_API vbool1_t Or(vbool1_t a, vbool1_t b) { return vmor_mm_b1(a, b); }
HWY_API vbool2_t Or(vbool2_t a, vbool2_t b) { return vmor_mm_b2(a, b); }
HWY_API vbool4_t Or(vbool4_t a, vbool4_t b) { return vmor_mm_b4(a, b); }
HWY_API vbool8_t Or(vbool8_t a, vbool8_t b) { return vmor_mm_b8(a, b); }
HWY_API vbool16_t Or(vbool16_t a, vbool16_t b) { return vmor_mm_b16(a, b); }
HWY_API vbool32_t Or(vbool32_t a, vbool32_t b) { return vmor_mm_b32(a, b); }
HWY_API vbool64_t Or(vbool64_t a, vbool64_t b) { return vmor_mm_b64(a, b); }

// ------------------------------ Xor
HWY_API vbool1_t Xor(vbool1_t a, vbool1_t b) { return vmxor_mm_b1(a, b); }
HWY_API vbool2_t Xor(vbool2_t a, vbool2_t b) { return vmxor_mm_b2(a, b); }
HWY_API vbool4_t Xor(vbool4_t a, vbool4_t b) { return vmxor_mm_b4(a, b); }
HWY_API vbool8_t Xor(vbool8_t a, vbool8_t b) { return vmxor_mm_b8(a, b); }
HWY_API vbool16_t Xor(vbool16_t a, vbool16_t b) { return vmxor_mm_b16(a, b); }
HWY_API vbool32_t Xor(vbool32_t a, vbool32_t b) { return vmxor_mm_b32(a, b); }
HWY_API vbool64_t Xor(vbool64_t a, vbool64_t b) { return vmxor_mm_b64(a, b); }

// ------------------------------ IfThenElse

HWY_API VU8M1 IfThenElse(vbool8_t mask, VU8M1 yes, VU8M1 no) {
  return vmerge_vvm_u8m1(mask, no, yes);
}
HWY_API VU8M2 IfThenElse(vbool4_t mask, VU8M2 yes, VU8M2 no) {
  return vmerge_vvm_u8m2(mask, no, yes);
}
HWY_API VU8M4 IfThenElse(vbool2_t mask, VU8M4 yes, VU8M4 no) {
  return vmerge_vvm_u8m4(mask, no, yes);
}
HWY_API VU8M8 IfThenElse(vbool1_t mask, VU8M8 yes, VU8M8 no) {
  return vmerge_vvm_u8m8(mask, no, yes);
}

HWY_API VU16M1 IfThenElse(vbool16_t mask, VU16M1 yes, VU16M1 no) {
  return vmerge_vvm_u16m1(mask, no, yes);
}
HWY_API VU16M2 IfThenElse(vbool8_t mask, VU16M2 yes, VU16M2 no) {
  return vmerge_vvm_u16m2(mask, no, yes);
}
HWY_API VU16M4 IfThenElse(vbool4_t mask, VU16M4 yes, VU16M4 no) {
  return vmerge_vvm_u16m4(mask, no, yes);
}
HWY_API VU16M8 IfThenElse(vbool2_t mask, VU16M8 yes, VU16M8 no) {
  return vmerge_vvm_u16m8(mask, no, yes);
}

HWY_API VU32M1 IfThenElse(vbool32_t mask, VU32M1 yes, VU32M1 no) {
  return vmerge_vvm_u32m1(mask, no, yes);
}
HWY_API VU32M2 IfThenElse(vbool16_t mask, VU32M2 yes, VU32M2 no) {
  return vmerge_vvm_u32m2(mask, no, yes);
}
HWY_API VU32M4 IfThenElse(vbool8_t mask, VU32M4 yes, VU32M4 no) {
  return vmerge_vvm_u32m4(mask, no, yes);
}
HWY_API VU32M8 IfThenElse(vbool4_t mask, VU32M8 yes, VU32M8 no) {
  return vmerge_vvm_u32m8(mask, no, yes);
}

HWY_API VU64M1 IfThenElse(vbool64_t mask, VU64M1 yes, VU64M1 no) {
  return vmerge_vvm_u64m1(mask, no, yes);
}
HWY_API VU64M2 IfThenElse(vbool32_t mask, VU64M2 yes, VU64M2 no) {
  return vmerge_vvm_u64m2(mask, no, yes);
}
HWY_API VU64M4 IfThenElse(vbool16_t mask, VU64M4 yes, VU64M4 no) {
  return vmerge_vvm_u64m4(mask, no, yes);
}
HWY_API VU64M8 IfThenElse(vbool8_t mask, VU64M8 yes, VU64M8 no) {
  return vmerge_vvm_u64m8(mask, no, yes);
}

HWY_API VI8M1 IfThenElse(vbool8_t mask, VI8M1 yes, VI8M1 no) {
  return vmerge_vvm_i8m1(mask, no, yes);
}
HWY_API VI8M2 IfThenElse(vbool4_t mask, VI8M2 yes, VI8M2 no) {
  return vmerge_vvm_i8m2(mask, no, yes);
}
HWY_API VI8M4 IfThenElse(vbool2_t mask, VI8M4 yes, VI8M4 no) {
  return vmerge_vvm_i8m4(mask, no, yes);
}
HWY_API VI8M8 IfThenElse(vbool1_t mask, VI8M8 yes, VI8M8 no) {
  return vmerge_vvm_i8m8(mask, no, yes);
}

HWY_API VI16M1 IfThenElse(vbool16_t mask, VI16M1 yes, VI16M1 no) {
  return vmerge_vvm_i16m1(mask, no, yes);
}
HWY_API VI16M2 IfThenElse(vbool8_t mask, VI16M2 yes, VI16M2 no) {
  return vmerge_vvm_i16m2(mask, no, yes);
}
HWY_API VI16M4 IfThenElse(vbool4_t mask, VI16M4 yes, VI16M4 no) {
  return vmerge_vvm_i16m4(mask, no, yes);
}
HWY_API VI16M8 IfThenElse(vbool2_t mask, VI16M8 yes, VI16M8 no) {
  return vmerge_vvm_i16m8(mask, no, yes);
}

HWY_API VI32M1 IfThenElse(vbool32_t mask, VI32M1 yes, VI32M1 no) {
  return vmerge_vvm_i32m1(mask, no, yes);
}
HWY_API VI32M2 IfThenElse(vbool16_t mask, VI32M2 yes, VI32M2 no) {
  return vmerge_vvm_i32m2(mask, no, yes);
}
HWY_API VI32M4 IfThenElse(vbool8_t mask, VI32M4 yes, VI32M4 no) {
  return vmerge_vvm_i32m4(mask, no, yes);
}
HWY_API VI32M8 IfThenElse(vbool4_t mask, VI32M8 yes, VI32M8 no) {
  return vmerge_vvm_i32m8(mask, no, yes);
}

HWY_API VI64M1 IfThenElse(vbool64_t mask, VI64M1 yes, VI64M1 no) {
  return vmerge_vvm_i64m1(mask, no, yes);
}
HWY_API VI64M2 IfThenElse(vbool32_t mask, VI64M2 yes, VI64M2 no) {
  return vmerge_vvm_i64m2(mask, no, yes);
}
HWY_API VI64M4 IfThenElse(vbool16_t mask, VI64M4 yes, VI64M4 no) {
  return vmerge_vvm_i64m4(mask, no, yes);
}
HWY_API VI64M8 IfThenElse(vbool8_t mask, VI64M8 yes, VI64M8 no) {
  return vmerge_vvm_i64m8(mask, no, yes);
}

HWY_API VF32M1 IfThenElse(vbool32_t mask, VF32M1 yes, VF32M1 no) {
  return vmerge_vvm_f32m1(mask, no, yes);
}
HWY_API VF32M2 IfThenElse(vbool16_t mask, VF32M2 yes, VF32M2 no) {
  return vmerge_vvm_f32m2(mask, no, yes);
}
HWY_API VF32M4 IfThenElse(vbool8_t mask, VF32M4 yes, VF32M4 no) {
  return vmerge_vvm_f32m4(mask, no, yes);
}
HWY_API VF32M8 IfThenElse(vbool4_t mask, VF32M8 yes, VF32M8 no) {
  return vmerge_vvm_f32m8(mask, no, yes);
}

HWY_API VF64M1 IfThenElse(vbool64_t mask, VF64M1 yes, VF64M1 no) {
  return vmerge_vvm_f64m1(mask, no, yes);
}
HWY_API VF64M2 IfThenElse(vbool32_t mask, VF64M2 yes, VF64M2 no) {
  return vmerge_vvm_f64m2(mask, no, yes);
}
HWY_API VF64M4 IfThenElse(vbool16_t mask, VF64M4 yes, VF64M4 no) {
  return vmerge_vvm_f64m4(mask, no, yes);
}
HWY_API VF64M8 IfThenElse(vbool8_t mask, VF64M8 yes, VF64M8 no) {
  return vmerge_vvm_f64m8(mask, no, yes);
}

// ------------------------------ IfThenElseZero

template <class M, class V>
HWY_API V IfThenElseZero(const M mask, const V yes) {
  return IfThenElse(mask, yes, Zero(DFromV<V>()));
}

// ------------------------------ IfThenZeroElse

template <class M, class V>
HWY_API V IfThenZeroElse(const M mask, const V no) {
  return IfThenElse(mask, Zero(DFromV<V>()), no);
}

// ------------------------------ MaskFromVec

template <class V>
HWY_API auto MaskFromVec(const V v) -> decltype(Eq(v, v)) {
  return Ne(v, Zero(DFromV<V>()));
}

// ------------------------------ VecFromMask

template <class D, class M>
HWY_API VFromD<D> VecFromMask(const D d, const M mask) {
  // TODO(janwas): masked Set with maskedoff = 0 would require one op less.
  return IfThenElseZero(mask, Set(d, -1));
}

// ------------------------------ ZeroIfNegative

template <class V>
HWY_API V ZeroIfNegative(const V v) {
  const auto v0 = Zero(DFromV<V>());
  return IfThenElse(Lt(v, v0), v0, v);
}

// ------------------------------ AllFalse

HWY_API bool AllFalse(const vbool1_t v) { return vfirst_m_b1(v) < 0; }
HWY_API bool AllFalse(const vbool2_t v) { return vfirst_m_b2(v) < 0; }
HWY_API bool AllFalse(const vbool4_t v) { return vfirst_m_b4(v) < 0; }
HWY_API bool AllFalse(const vbool8_t v) { return vfirst_m_b8(v) < 0; }
HWY_API bool AllFalse(const vbool16_t v) { return vfirst_m_b16(v) < 0; }
HWY_API bool AllFalse(const vbool32_t v) { return vfirst_m_b32(v) < 0; }
HWY_API bool AllFalse(const vbool64_t v) { return vfirst_m_b64(v) < 0; }

// ------------------------------ AllTrue

HWY_API bool AllTrue(const vbool1_t v) { return AllFalse(vmnot_m_b1(v)); }
HWY_API bool AllTrue(const vbool2_t v) { return AllFalse(vmnot_m_b2(v)); }
HWY_API bool AllTrue(const vbool4_t v) { return AllFalse(vmnot_m_b4(v)); }
HWY_API bool AllTrue(const vbool8_t v) { return AllFalse(vmnot_m_b8(v)); }
HWY_API bool AllTrue(const vbool16_t v) { return AllFalse(vmnot_m_b16(v)); }
HWY_API bool AllTrue(const vbool32_t v) { return AllFalse(vmnot_m_b32(v)); }
HWY_API bool AllTrue(const vbool64_t v) { return AllFalse(vmnot_m_b64(v)); }

// ------------------------------ CountTrue
HWY_API size_t CountTrue(const vbool1_t mask) { return vpopc_m_b1(mask); }
HWY_API size_t CountTrue(const vbool2_t mask) { return vpopc_m_b2(mask); }
HWY_API size_t CountTrue(const vbool4_t mask) { return vpopc_m_b4(mask); }
HWY_API size_t CountTrue(const vbool8_t mask) { return vpopc_m_b8(mask); }
HWY_API size_t CountTrue(const vbool16_t mask) { return vpopc_m_b16(mask); }
HWY_API size_t CountTrue(const vbool32_t mask) { return vpopc_m_b32(mask); }
HWY_API size_t CountTrue(const vbool64_t mask) { return vpopc_m_b64(mask); }

// ================================================== MEMORY

// ------------------------------ Load

HWY_API VU8M1 Load(DU8M1 /* d */, const uint8_t* HWY_RESTRICT aligned) {
  return vle8_v_u8m1(aligned);
}
HWY_API VU8M2 Load(DU8M2 /* d */, const uint8_t* HWY_RESTRICT aligned) {
  return vle8_v_u8m2(aligned);
}
HWY_API VU8M4 Load(DU8M4 /* d */, const uint8_t* HWY_RESTRICT aligned) {
  return vle8_v_u8m4(aligned);
}
HWY_API VU8M8 Load(DU8M8 /* d */, const uint8_t* HWY_RESTRICT aligned) {
  return vle8_v_u8m8(aligned);
}

HWY_API VU16M1 Load(DU16M1 /* d */, const uint16_t* HWY_RESTRICT aligned) {
  return vle16_v_u16m1(aligned);
}
HWY_API VU16M2 Load(DU16M2 /* d */, const uint16_t* HWY_RESTRICT aligned) {
  return vle16_v_u16m2(aligned);
}
HWY_API VU16M4 Load(DU16M4 /* d */, const uint16_t* HWY_RESTRICT aligned) {
  return vle16_v_u16m4(aligned);
}
HWY_API VU16M8 Load(DU16M8 /* d */, const uint16_t* HWY_RESTRICT aligned) {
  return vle16_v_u16m8(aligned);
}

HWY_API VU32M1 Load(DU32M1 /* d */, const uint32_t* HWY_RESTRICT aligned) {
  return vle32_v_u32m1(aligned);
}
HWY_API VU32M2 Load(DU32M2 /* d */, const uint32_t* HWY_RESTRICT aligned) {
  return vle32_v_u32m2(aligned);
}
HWY_API VU32M4 Load(DU32M4 /* d */, const uint32_t* HWY_RESTRICT aligned) {
  return vle32_v_u32m4(aligned);
}
HWY_API VU32M8 Load(DU32M8 /* d */, const uint32_t* HWY_RESTRICT aligned) {
  return vle32_v_u32m8(aligned);
}

HWY_API VU64M1 Load(DU64M1 /* d */, const uint64_t* HWY_RESTRICT aligned) {
  return vle64_v_u64m1(aligned);
}
HWY_API VU64M2 Load(DU64M2 /* d */, const uint64_t* HWY_RESTRICT aligned) {
  return vle64_v_u64m2(aligned);
}
HWY_API VU64M4 Load(DU64M4 /* d */, const uint64_t* HWY_RESTRICT aligned) {
  return vle64_v_u64m4(aligned);
}
HWY_API VU64M8 Load(DU64M8 /* d */, const uint64_t* HWY_RESTRICT aligned) {
  return vle64_v_u64m8(aligned);
}

HWY_API VI8M1 Load(DI8M1 /* d */, const int8_t* HWY_RESTRICT aligned) {
  return vle8_v_i8m1(aligned);
}
HWY_API VI8M2 Load(DI8M2 /* d */, const int8_t* HWY_RESTRICT aligned) {
  return vle8_v_i8m2(aligned);
}
HWY_API VI8M4 Load(DI8M4 /* d */, const int8_t* HWY_RESTRICT aligned) {
  return vle8_v_i8m4(aligned);
}
HWY_API VI8M8 Load(DI8M8 /* d */, const int8_t* HWY_RESTRICT aligned) {
  return vle8_v_i8m8(aligned);
}

HWY_API VI16M1 Load(DI16M1 /* d */, const int16_t* HWY_RESTRICT aligned) {
  return vle16_v_i16m1(aligned);
}
HWY_API VI16M2 Load(DI16M2 /* d */, const int16_t* HWY_RESTRICT aligned) {
  return vle16_v_i16m2(aligned);
}
HWY_API VI16M4 Load(DI16M4 /* d */, const int16_t* HWY_RESTRICT aligned) {
  return vle16_v_i16m4(aligned);
}
HWY_API VI16M8 Load(DI16M8 /* d */, const int16_t* HWY_RESTRICT aligned) {
  return vle16_v_i16m8(aligned);
}

HWY_API VI32M1 Load(DI32M1 /* d */, const int32_t* HWY_RESTRICT aligned) {
  return vle32_v_i32m1(aligned);
}
HWY_API VI32M2 Load(DI32M2 /* d */, const int32_t* HWY_RESTRICT aligned) {
  return vle32_v_i32m2(aligned);
}
HWY_API VI32M4 Load(DI32M4 /* d */, const int32_t* HWY_RESTRICT aligned) {
  return vle32_v_i32m4(aligned);
}
HWY_API VI32M8 Load(DI32M8 /* d */, const int32_t* HWY_RESTRICT aligned) {
  return vle32_v_i32m8(aligned);
}

HWY_API VI64M1 Load(DI64M1 /* d */, const int64_t* HWY_RESTRICT aligned) {
  return vle64_v_i64m1(aligned);
}
HWY_API VI64M2 Load(DI64M2 /* d */, const int64_t* HWY_RESTRICT aligned) {
  return vle64_v_i64m2(aligned);
}
HWY_API VI64M4 Load(DI64M4 /* d */, const int64_t* HWY_RESTRICT aligned) {
  return vle64_v_i64m4(aligned);
}
HWY_API VI64M8 Load(DI64M8 /* d */, const int64_t* HWY_RESTRICT aligned) {
  return vle64_v_i64m8(aligned);
}

HWY_API VF32M1 Load(DF32M1 /* d */, const float* HWY_RESTRICT aligned) {
  return vle32_v_f32m1(aligned);
}
HWY_API VF32M2 Load(DF32M2 /* d */, const float* HWY_RESTRICT aligned) {
  return vle32_v_f32m2(aligned);
}
HWY_API VF32M4 Load(DF32M4 /* d */, const float* HWY_RESTRICT aligned) {
  return vle32_v_f32m4(aligned);
}
HWY_API VF32M8 Load(DF32M8 /* d */, const float* HWY_RESTRICT aligned) {
  return vle32_v_f32m8(aligned);
}

HWY_API VF64M1 Load(DF64M1 /* d */, const double* HWY_RESTRICT aligned) {
  return vle64_v_f64m1(aligned);
}
HWY_API VF64M2 Load(DF64M2 /* d */, const double* HWY_RESTRICT aligned) {
  return vle64_v_f64m2(aligned);
}
HWY_API VF64M4 Load(DF64M4 /* d */, const double* HWY_RESTRICT aligned) {
  return vle64_v_f64m4(aligned);
}
HWY_API VF64M8 Load(DF64M8 /* d */, const double* HWY_RESTRICT aligned) {
  return vle64_v_f64m8(aligned);
}

// Partial load
template <typename T, size_t N, HWY_IF_LE128(T, N)>
HWY_API VFromD<Simd<T, N>> Load(Simd<T, N> d, const T* HWY_RESTRICT p) {
  // TODO(janwas): set VL
  return Load(d, p);
}

// ------------------------------ LoadU

// RVV only requires lane alignment, not natural alignment of the entire vector.
template <class D>
HWY_API VFromD<D> LoadU(D d, const TFromD<D>* HWY_RESTRICT p) {
  return Load(d, p);
}

// ------------------------------ Store

HWY_API void Store(VU8M1 v, DU8M1 /* d */, uint8_t* HWY_RESTRICT aligned) {
  vse8_v_u8m1(aligned, v);
}
HWY_API void Store(VU8M2 v, DU8M2 /* d */, uint8_t* HWY_RESTRICT aligned) {
  vse8_v_u8m2(aligned, v);
}
HWY_API void Store(VU8M4 v, DU8M4 /* d */, uint8_t* HWY_RESTRICT aligned) {
  vse8_v_u8m4(aligned, v);
}
HWY_API void Store(VU8M8 v, DU8M8 /* d */, uint8_t* HWY_RESTRICT aligned) {
  vse8_v_u8m8(aligned, v);
}

HWY_API void Store(VU16M1 v, DU16M1 /* d */, uint16_t* HWY_RESTRICT aligned) {
  vse16_v_u16m1(aligned, v);
}
HWY_API void Store(VU16M2 v, DU16M2 /* d */, uint16_t* HWY_RESTRICT aligned) {
  vse16_v_u16m2(aligned, v);
}
HWY_API void Store(VU16M4 v, DU16M4 /* d */, uint16_t* HWY_RESTRICT aligned) {
  vse16_v_u16m4(aligned, v);
}
HWY_API void Store(VU16M8 v, DU16M8 /* d */, uint16_t* HWY_RESTRICT aligned) {
  vse16_v_u16m8(aligned, v);
}

HWY_API void Store(VU32M1 v, DU32M1 /* d */, uint32_t* HWY_RESTRICT aligned) {
  vse32_v_u32m1(aligned, v);
}
HWY_API void Store(VU32M2 v, DU32M2 /* d */, uint32_t* HWY_RESTRICT aligned) {
  vse32_v_u32m2(aligned, v);
}
HWY_API void Store(VU32M4 v, DU32M4 /* d */, uint32_t* HWY_RESTRICT aligned) {
  vse32_v_u32m4(aligned, v);
}
HWY_API void Store(VU32M8 v, DU32M8 /* d */, uint32_t* HWY_RESTRICT aligned) {
  vse32_v_u32m8(aligned, v);
}

HWY_API void Store(VU64M1 v, DU64M1 /* d */, uint64_t* HWY_RESTRICT aligned) {
  vse64_v_u64m1(aligned, v);
}
HWY_API void Store(VU64M2 v, DU64M2 /* d */, uint64_t* HWY_RESTRICT aligned) {
  vse64_v_u64m2(aligned, v);
}
HWY_API void Store(VU64M4 v, DU64M4 /* d */, uint64_t* HWY_RESTRICT aligned) {
  vse64_v_u64m4(aligned, v);
}
HWY_API void Store(VU64M8 v, DU64M8 /* d */, uint64_t* HWY_RESTRICT aligned) {
  vse64_v_u64m8(aligned, v);
}

HWY_API void Store(VI8M1 v, DI8M1 /* d */, int8_t* HWY_RESTRICT aligned) {
  vse8_v_i8m1(aligned, v);
}
HWY_API void Store(VI8M2 v, DI8M2 /* d */, int8_t* HWY_RESTRICT aligned) {
  vse8_v_i8m2(aligned, v);
}
HWY_API void Store(VI8M4 v, DI8M4 /* d */, int8_t* HWY_RESTRICT aligned) {
  vse8_v_i8m4(aligned, v);
}
HWY_API void Store(VI8M8 v, DI8M8 /* d */, int8_t* HWY_RESTRICT aligned) {
  vse8_v_i8m8(aligned, v);
}

HWY_API void Store(VI16M1 v, DI16M1 /* d */, int16_t* HWY_RESTRICT aligned) {
  vse16_v_i16m1(aligned, v);
}
HWY_API void Store(VI16M2 v, DI16M2 /* d */, int16_t* HWY_RESTRICT aligned) {
  vse16_v_i16m2(aligned, v);
}
HWY_API void Store(VI16M4 v, DI16M4 /* d */, int16_t* HWY_RESTRICT aligned) {
  vse16_v_i16m4(aligned, v);
}
HWY_API void Store(VI16M8 v, DI16M8 /* d */, int16_t* HWY_RESTRICT aligned) {
  vse16_v_i16m8(aligned, v);
}

HWY_API void Store(VI32M1 v, DI32M1 /* d */, int32_t* HWY_RESTRICT aligned) {
  vse32_v_i32m1(aligned, v);
}
HWY_API void Store(VI32M2 v, DI32M2 /* d */, int32_t* HWY_RESTRICT aligned) {
  vse32_v_i32m2(aligned, v);
}
HWY_API void Store(VI32M4 v, DI32M4 /* d */, int32_t* HWY_RESTRICT aligned) {
  vse32_v_i32m4(aligned, v);
}
HWY_API void Store(VI32M8 v, DI32M8 /* d */, int32_t* HWY_RESTRICT aligned) {
  vse32_v_i32m8(aligned, v);
}

HWY_API void Store(VI64M1 v, DI64M1 /* d */, int64_t* HWY_RESTRICT aligned) {
  vse64_v_i64m1(aligned, v);
}
HWY_API void Store(VI64M2 v, DI64M2 /* d */, int64_t* HWY_RESTRICT aligned) {
  vse64_v_i64m2(aligned, v);
}
HWY_API void Store(VI64M4 v, DI64M4 /* d */, int64_t* HWY_RESTRICT aligned) {
  vse64_v_i64m4(aligned, v);
}
HWY_API void Store(VI64M8 v, DI64M8 /* d */, int64_t* HWY_RESTRICT aligned) {
  vse64_v_i64m8(aligned, v);
}

HWY_API void Store(VF32M1 v, DF32M1 /* d */, float* HWY_RESTRICT aligned) {
  vse32_v_f32m1(aligned, v);
}
HWY_API void Store(VF32M2 v, DF32M2 /* d */, float* HWY_RESTRICT aligned) {
  vse32_v_f32m2(aligned, v);
}
HWY_API void Store(VF32M4 v, DF32M4 /* d */, float* HWY_RESTRICT aligned) {
  vse32_v_f32m4(aligned, v);
}
HWY_API void Store(VF32M8 v, DF32M8 /* d */, float* HWY_RESTRICT aligned) {
  vse32_v_f32m8(aligned, v);
}

HWY_API void Store(VF64M1 v, DF64M1 /* d */, double* HWY_RESTRICT aligned) {
  vse64_v_f64m1(aligned, v);
}
HWY_API void Store(VF64M2 v, DF64M2 /* d */, double* HWY_RESTRICT aligned) {
  vse64_v_f64m2(aligned, v);
}
HWY_API void Store(VF64M4 v, DF64M4 /* d */, double* HWY_RESTRICT aligned) {
  vse64_v_f64m4(aligned, v);
}
HWY_API void Store(VF64M8 v, DF64M8 /* d */, double* HWY_RESTRICT aligned) {
  vse64_v_f64m8(aligned, v);
}

// ------------------------------ StoreU

// RVV only requires lane alignment, not natural alignment of the entire vector.
template <class V, class D>
HWY_API void StoreU(const V v, D d, TFromD<D>* HWY_RESTRICT p) {
  Store(v, d, p);
}

// ------------------------------ Stream

template <class V, class D, typename T>
HWY_API void Stream(const V v, D d, T* HWY_RESTRICT aligned) {
  Store(v, d, aligned);
}

// ------------------------------ GatherOffset

// TODO(janwas): add u (unordered) once supported

HWY_API VU32M1 GatherOffset(DU32M1 /* d */, const uint32_t* HWY_RESTRICT base,
                            const VI32M1 offset) {
  return vlxei32_v_u32m1(base, detail::BitCastToUnsigned(offset));
}
HWY_API VU32M2 GatherOffset(DU32M2 /* d */, const uint32_t* HWY_RESTRICT base,
                            const VI32M2 offset) {
  return vlxei32_v_u32m2(base, detail::BitCastToUnsigned(offset));
}
HWY_API VU32M4 GatherOffset(DU32M4 /* d */, const uint32_t* HWY_RESTRICT base,
                            const VI32M4 offset) {
  return vlxei32_v_u32m4(base, detail::BitCastToUnsigned(offset));
}
HWY_API VU32M8 GatherOffset(DU32M8 /* d */, const uint32_t* HWY_RESTRICT base,
                            const VI32M8 offset) {
  return vlxei32_v_u32m8(base, detail::BitCastToUnsigned(offset));
}

HWY_API VU64M1 GatherOffset(DU64M1 /* d */, const uint64_t* HWY_RESTRICT base,
                            const VI64M1 offset) {
  return vlxei64_v_u64m1(base, detail::BitCastToUnsigned(offset));
}
HWY_API VU64M2 GatherOffset(DU64M2 /* d */, const uint64_t* HWY_RESTRICT base,
                            const VI64M2 offset) {
  return vlxei64_v_u64m2(base, detail::BitCastToUnsigned(offset));
}
HWY_API VU64M4 GatherOffset(DU64M4 /* d */, const uint64_t* HWY_RESTRICT base,
                            const VI64M4 offset) {
  return vlxei64_v_u64m4(base, detail::BitCastToUnsigned(offset));
}
HWY_API VU64M8 GatherOffset(DU64M8 /* d */, const uint64_t* HWY_RESTRICT base,
                            const VI64M8 offset) {
  return vlxei64_v_u64m8(base, detail::BitCastToUnsigned(offset));
}

HWY_API VI32M1 GatherOffset(DI32M1 /* d */, const int32_t* HWY_RESTRICT base,
                            const VI32M1 offset) {
  return vlxei32_v_i32m1(base, detail::BitCastToUnsigned(offset));
}
HWY_API VI32M2 GatherOffset(DI32M2 /* d */, const int32_t* HWY_RESTRICT base,
                            const VI32M2 offset) {
  return vlxei32_v_i32m2(base, detail::BitCastToUnsigned(offset));
}
HWY_API VI32M4 GatherOffset(DI32M4 /* d */, const int32_t* HWY_RESTRICT base,
                            const VI32M4 offset) {
  return vlxei32_v_i32m4(base, detail::BitCastToUnsigned(offset));
}
HWY_API VI32M8 GatherOffset(DI32M8 /* d */, const int32_t* HWY_RESTRICT base,
                            const VI32M8 offset) {
  return vlxei32_v_i32m8(base, detail::BitCastToUnsigned(offset));
}

HWY_API VI64M1 GatherOffset(DI64M1 /* d */, const int64_t* HWY_RESTRICT base,
                            const VI64M1 offset) {
  return vlxei64_v_i64m1(base, detail::BitCastToUnsigned(offset));
}
HWY_API VI64M2 GatherOffset(DI64M2 /* d */, const int64_t* HWY_RESTRICT base,
                            const VI64M2 offset) {
  return vlxei64_v_i64m2(base, detail::BitCastToUnsigned(offset));
}
HWY_API VI64M4 GatherOffset(DI64M4 /* d */, const int64_t* HWY_RESTRICT base,
                            const VI64M4 offset) {
  return vlxei64_v_i64m4(base, detail::BitCastToUnsigned(offset));
}
HWY_API VI64M8 GatherOffset(DI64M8 /* d */, const int64_t* HWY_RESTRICT base,
                            const VI64M8 offset) {
  return vlxei64_v_i64m8(base, detail::BitCastToUnsigned(offset));
}

HWY_API VF32M1 GatherOffset(DF32M1 /* d */, const float* HWY_RESTRICT base,
                            const VI32M1 offset) {
  return vlxei32_v_f32m1(base, detail::BitCastToUnsigned(offset));
}
HWY_API VF32M2 GatherOffset(DF32M2 /* d */, const float* HWY_RESTRICT base,
                            const VI32M2 offset) {
  return vlxei32_v_f32m2(base, detail::BitCastToUnsigned(offset));
}
HWY_API VF32M4 GatherOffset(DF32M4 /* d */, const float* HWY_RESTRICT base,
                            const VI32M4 offset) {
  return vlxei32_v_f32m4(base, detail::BitCastToUnsigned(offset));
}
HWY_API VF32M8 GatherOffset(DF32M8 /* d */, const float* HWY_RESTRICT base,
                            const VI32M8 offset) {
  return vlxei32_v_f32m8(base, detail::BitCastToUnsigned(offset));
}

HWY_API VF64M1 GatherOffset(DF64M1 /* d */, const double* HWY_RESTRICT base,
                            const VI64M1 offset) {
  return vlxei64_v_f64m1(base, detail::BitCastToUnsigned(offset));
}
HWY_API VF64M2 GatherOffset(DF64M2 /* d */, const double* HWY_RESTRICT base,
                            const VI64M2 offset) {
  return vlxei64_v_f64m2(base, detail::BitCastToUnsigned(offset));
}
HWY_API VF64M4 GatherOffset(DF64M4 /* d */, const double* HWY_RESTRICT base,
                            const VI64M4 offset) {
  return vlxei64_v_f64m4(base, detail::BitCastToUnsigned(offset));
}
HWY_API VF64M8 GatherOffset(DF64M8 /* d */, const double* HWY_RESTRICT base,
                            const VI64M8 offset) {
  return vlxei64_v_f64m8(base, detail::BitCastToUnsigned(offset));
}

// ------------------------------ GatherIndex

template <class D, HWY_IF_LANE_SIZE4_D(D)>
HWY_API VFromD<D> GatherIndex(D d, const TFromD<D>* HWY_RESTRICT base,
                              const VFromD<RebindToSigned<D>> index) {
  return GatherOffset(d, base, ShiftLeft<2>(index));
}

template <class D, HWY_IF_LANE_SIZE8_D(D)>
HWY_API VFromD<D> GatherIndex(D d, const TFromD<D>* HWY_RESTRICT base,
                              const VFromD<RebindToSigned<D>> index) {
  return GatherOffset(d, base, ShiftLeft<3>(index));
}

// ================================================== CONVERT

// ------------------------------ PromoteTo U

HWY_API VU16M2 PromoteTo(DU16M2 /* d */, VU8M1 v) { return vzext_vf2_u16m2(v); }
HWY_API VU16M4 PromoteTo(DU16M4 /* d */, VU8M2 v) { return vzext_vf2_u16m4(v); }
HWY_API VU16M8 PromoteTo(DU16M8 /* d */, VU8M4 v) { return vzext_vf2_u16m8(v); }

HWY_API VU32M4 PromoteTo(DU32M4 /* d */, VU8M1 v) { return vzext_vf4_u32m4(v); }
HWY_API VU32M8 PromoteTo(DU32M8 /* d */, VU8M2 v) { return vzext_vf4_u32m8(v); }

HWY_API VU32M2 PromoteTo(DU32M2 /* d */, const VU16M1 v) {
  return vzext_vf2_u32m2(v);
}
HWY_API VU32M4 PromoteTo(DU32M4 /* d */, const VU16M2 v) {
  return vzext_vf2_u32m4(v);
}
HWY_API VU32M8 PromoteTo(DU32M8 /* d */, const VU16M4 v) {
  return vzext_vf2_u32m8(v);
}

HWY_API VU64M2 PromoteTo(DU64M2 /* d */, const VU32M1 v) {
  return vzext_vf2_u64m2(v);
}
HWY_API VU64M4 PromoteTo(DU64M4 /* d */, const VU32M2 v) {
  return vzext_vf2_u64m4(v);
}
HWY_API VU64M8 PromoteTo(DU64M8 /* d */, const VU32M4 v) {
  return vzext_vf2_u64m8(v);
}

template <size_t N>
HWY_API VFromD<Simd<int16_t, N>> PromoteTo(Simd<int16_t, N> d,
                                           VFromD<Simd<uint8_t, N>> v) {
  return BitCast(d, PromoteTo(Simd<uint16_t, N>(), v));
}

template <size_t N>
HWY_API VFromD<Simd<int32_t, N>> PromoteTo(Simd<int32_t, N> d,
                                           VFromD<Simd<uint8_t, N>> v) {
  return BitCast(d, PromoteTo(Simd<uint32_t, N>(), v));
}

template <size_t N>
HWY_API VFromD<Simd<int32_t, N>> PromoteTo(Simd<int32_t, N> d,
                                           VFromD<Simd<uint16_t, N>> v) {
  return BitCast(d, PromoteTo(Simd<uint32_t, N>(), v));
}

template <class V, class D4 = Rebind<MakeWide<MakeWide<TFromV<V>>>, DFromV<V>>>
HWY_API VFromD<D4> U32FromU8(const V v) {
  return PromoteTo(D4(), v);
}

// ------------------------------ PromoteTo I

HWY_API VI16M2 PromoteTo(DI16M2 /* d */, VI8M1 v) { return vsext_vf2_i16m2(v); }
HWY_API VI16M4 PromoteTo(DI16M4 /* d */, VI8M2 v) { return vsext_vf2_i16m4(v); }
HWY_API VI16M8 PromoteTo(DI16M8 /* d */, VI8M4 v) { return vsext_vf2_i16m8(v); }

HWY_API VI32M4 PromoteTo(DI32M4 /* d */, VI8M1 v) { return vsext_vf4_i32m4(v); }
HWY_API VI32M8 PromoteTo(DI32M8 /* d */, VI8M2 v) { return vsext_vf4_i32m8(v); }

HWY_API VI32M2 PromoteTo(DI32M2 /* d */, const VI16M1 v) {
  return vsext_vf2_i32m2(v);
}
HWY_API VI32M4 PromoteTo(DI32M4 /* d */, const VI16M2 v) {
  return vsext_vf2_i32m4(v);
}
HWY_API VI32M8 PromoteTo(DI32M8 /* d */, const VI16M4 v) {
  return vsext_vf2_i32m8(v);
}

HWY_API VI64M2 PromoteTo(DI64M2 /* d */, const VI32M1 v) {
  return vsext_vf2_i64m2(v);
}
HWY_API VI64M4 PromoteTo(DI64M4 /* d */, const VI32M2 v) {
  return vsext_vf2_i64m4(v);
}
HWY_API VI64M8 PromoteTo(DI64M8 /* d */, const VI32M4 v) {
  return vsext_vf2_i64m8(v);
}

// ------------------------------ PromoteTo F

HWY_API VF64M2 PromoteTo(DF64M2 /* d */, const VF32M1 v) {
  return vfwcvt_f_f_v_f64m2(v);
}
HWY_API VF64M4 PromoteTo(DF64M4 /* d */, const VF32M2 v) {
  return vfwcvt_f_f_v_f64m4(v);
}
HWY_API VF64M8 PromoteTo(DF64M8 /* d */, const VF32M4 v) {
  return vfwcvt_f_f_v_f64m8(v);
}

HWY_API VF64M2 PromoteTo(DF64M2 /* d */, const VI32M1 v) {
  return vfwcvt_f_x_v_f64m2(v);
}
HWY_API VF64M4 PromoteTo(DF64M4 /* d */, const VI32M2 v) {
  return vfwcvt_f_x_v_f64m4(v);
}
HWY_API VF64M8 PromoteTo(DF64M8 /* d */, const VI32M4 v) {
  return vfwcvt_f_x_v_f64m8(v);
}

// ------------------------------ DemoteTo U

// First clamp negative numbers to zero to match x86 packus.
HWY_API VU16M1 DemoteTo(DU16M1 /* d */, const VI32M2 v) {
  return vnclipu_wx_u16m1(detail::BitCastToUnsigned(detail::Max(v, 0)), 0);
}
HWY_API VU16M2 DemoteTo(DU16M2 /* d */, const VI32M4 v) {
  return vnclipu_wx_u16m2(detail::BitCastToUnsigned(detail::Max(v, 0)), 0);
}
HWY_API VU16M4 DemoteTo(DU16M4 /* d */, const VI32M8 v) {
  return vnclipu_wx_u16m4(detail::BitCastToUnsigned(detail::Max(v, 0)), 0);
}

HWY_API VU8M1 DemoteTo(DU8M1 /* d */, const VI32M4 v) {
  return vnclipu_wx_u8m1(DemoteTo(DU16M2(), v), 0);
}
HWY_API VU8M2 DemoteTo(DU8M2 /* d */, const VI32M8 v) {
  return vnclipu_wx_u8m2(DemoteTo(DU16M4(), v), 0);
}

HWY_API VU8M1 DemoteTo(DU8M1 /* d */, const VI16M2 v) {
  return vnclipu_wx_u8m1(detail::BitCastToUnsigned(detail::Max(v, 0)), 0);
}
HWY_API VU8M2 DemoteTo(DU8M2 /* d */, const VI16M4 v) {
  return vnclipu_wx_u8m2(detail::BitCastToUnsigned(detail::Max(v, 0)), 0);
}
HWY_API VU8M4 DemoteTo(DU8M4 /* d */, const VI16M8 v) {
  return vnclipu_wx_u8m4(detail::BitCastToUnsigned(detail::Max(v, 0)), 0);
}

HWY_API VU8M1 U8FromU32(const VU32M4 v) {
  return vnclipu_wx_u8m1(vnclipu_wx_u16m2(v, 0), 0);
}
HWY_API VU8M2 U8FromU32(const VU32M8 v) {
  return vnclipu_wx_u8m2(vnclipu_wx_u16m4(v, 0), 0);
}

// ------------------------------ DemoteTo I

HWY_API VI8M1 DemoteTo(DI8M1 /* d */, const VI16M2 v) {
  return vnclip_wx_i8m1(v, 0);
}
HWY_API VI8M2 DemoteTo(DI8M2 /* d */, const VI16M4 v) {
  return vnclip_wx_i8m2(v, 0);
}
HWY_API VI8M4 DemoteTo(DI8M4 /* d */, const VI16M8 v) {
  return vnclip_wx_i8m4(v, 0);
}

HWY_API VI16M1 DemoteTo(DI16M1 /* d */, const VI32M2 v) {
  return vnclip_wx_i16m1(v, 0);
}
HWY_API VI16M2 DemoteTo(DI16M2 /* d */, const VI32M4 v) {
  return vnclip_wx_i16m2(v, 0);
}
HWY_API VI16M4 DemoteTo(DI16M4 /* d */, const VI32M8 v) {
  return vnclip_wx_i16m4(v, 0);
}

HWY_API VI8M1 DemoteTo(DI8M1 d, const VI32M4 v) {
  return DemoteTo(d, DemoteTo(DI16M2(), v));
}
HWY_API VI8M2 DemoteTo(DI8M2 d, const VI32M8 v) {
  return DemoteTo(d, DemoteTo(DI16M4(), v));
}

// ------------------------------ DemoteTo F

HWY_API VF32M1 DemoteTo(DF32M1 /* d */, const VF64M2 v) {
  return vfncvt_rod_f_f_w_f32m1(v);
}
HWY_API VF32M2 DemoteTo(DF32M2 /* d */, const VF64M4 v) {
  return vfncvt_rod_f_f_w_f32m2(v);
}
HWY_API VF32M4 DemoteTo(DF32M4 /* d */, const VF64M8 v) {
  return vfncvt_rod_f_f_w_f32m4(v);
}

HWY_API VI32M1 DemoteTo(DI32M1 /* d */, const VF64M2 v) {
  return vfncvt_rtz_x_f_w_i32m1(v);
}
HWY_API VI32M2 DemoteTo(DI32M2 /* d */, const VF64M4 v) {
  return vfncvt_rtz_x_f_w_i32m2(v);
}
HWY_API VI32M4 DemoteTo(DI32M4 /* d */, const VF64M8 v) {
  return vfncvt_rtz_x_f_w_i32m4(v);
}

// ------------------------------ ConvertTo F

HWY_API VF32M1 ConvertTo(DF32M1 /* d */, const VI32M1 v) {
  return vfcvt_f_x_v_f32m1(v);
}
HWY_API VF32M2 ConvertTo(DF32M2 /* d */, const VI32M2 v) {
  return vfcvt_f_x_v_f32m2(v);
}
HWY_API VF32M4 ConvertTo(DF32M4 /* d */, const VI32M4 v) {
  return vfcvt_f_x_v_f32m4(v);
}
HWY_API VF32M8 ConvertTo(DF32M8 /* d */, const VI32M8 v) {
  return vfcvt_f_x_v_f32m8(v);
}

HWY_API VF64M1 ConvertTo(DF64M1 /* d */, const VI64M1 v) {
  return vfcvt_f_x_v_f64m1(v);
}
HWY_API VF64M2 ConvertTo(DF64M2 /* d */, const VI64M2 v) {
  return vfcvt_f_x_v_f64m2(v);
}
HWY_API VF64M4 ConvertTo(DF64M4 /* d */, const VI64M4 v) {
  return vfcvt_f_x_v_f64m4(v);
}
HWY_API VF64M8 ConvertTo(DF64M8 /* d */, const VI64M8 v) {
  return vfcvt_f_x_v_f64m8(v);
}

// Truncates (rounds toward zero).
HWY_API VI32M1 ConvertTo(DI32M1 /* d */, const VF32M1 v) {
  return vfcvt_rtz_x_f_v_i32m1(v);
}
HWY_API VI32M2 ConvertTo(DI32M2 /* d */, const VF32M2 v) {
  return vfcvt_rtz_x_f_v_i32m2(v);
}
HWY_API VI32M4 ConvertTo(DI32M4 /* d */, const VF32M4 v) {
  return vfcvt_rtz_x_f_v_i32m4(v);
}
HWY_API VI32M8 ConvertTo(DI32M8 /* d */, const VF32M8 v) {
  return vfcvt_rtz_x_f_v_i32m8(v);
}

HWY_API VI64M1 ConvertTo(DI64M1 /* d */, const VF64M1 v) {
  return vfcvt_rtz_x_f_v_i64m1(v);
}
HWY_API VI64M2 ConvertTo(DI64M2 /* d */, const VF64M2 v) {
  return vfcvt_rtz_x_f_v_i64m2(v);
}
HWY_API VI64M4 ConvertTo(DI64M4 /* d */, const VF64M4 v) {
  return vfcvt_rtz_x_f_v_i64m4(v);
}
HWY_API VI64M8 ConvertTo(DI64M8 /* d */, const VF64M8 v) {
  return vfcvt_rtz_x_f_v_i64m8(v);
}

// Uses default rounding mode.
HWY_API VI32M1 NearestInt(VF32M1 v) { return vfcvt_x_f_v_i32m1(v); }
HWY_API VI32M2 NearestInt(VF32M2 v) { return vfcvt_x_f_v_i32m2(v); }
HWY_API VI32M4 NearestInt(VF32M4 v) { return vfcvt_x_f_v_i32m4(v); }
HWY_API VI32M8 NearestInt(VF32M8 v) { return vfcvt_x_f_v_i32m8(v); }

// ================================================== SWIZZLE

// ------------------------------ Compress

HWY_API VU32M1 Compress(const VU32M1 v, const vbool32_t mask) {
  return vcompress_vm_u32m1(mask, v, v);
}
HWY_API VU32M2 Compress(const VU32M2 v, const vbool16_t mask) {
  return vcompress_vm_u32m2(mask, v, v);
}
HWY_API VU32M4 Compress(const VU32M4 v, const vbool8_t mask) {
  return vcompress_vm_u32m4(mask, v, v);
}
HWY_API VU32M8 Compress(const VU32M8 v, const vbool4_t mask) {
  return vcompress_vm_u32m8(mask, v, v);
}

HWY_API VI32M1 Compress(const VI32M1 v, const vbool32_t mask) {
  return vcompress_vm_i32m1(mask, v, v);
}
HWY_API VI32M2 Compress(const VI32M2 v, const vbool16_t mask) {
  return vcompress_vm_i32m2(mask, v, v);
}
HWY_API VI32M4 Compress(const VI32M4 v, const vbool8_t mask) {
  return vcompress_vm_i32m4(mask, v, v);
}
HWY_API VI32M8 Compress(const VI32M8 v, const vbool4_t mask) {
  return vcompress_vm_i32m8(mask, v, v);
}

HWY_API VU64M1 Compress(const VU64M1 v, const vbool64_t mask) {
  return vcompress_vm_u64m1(mask, v, v);
}
HWY_API VU64M2 Compress(const VU64M2 v, const vbool32_t mask) {
  return vcompress_vm_u64m2(mask, v, v);
}
HWY_API VU64M4 Compress(const VU64M4 v, const vbool16_t mask) {
  return vcompress_vm_u64m4(mask, v, v);
}
HWY_API VU64M8 Compress(const VU64M8 v, const vbool8_t mask) {
  return vcompress_vm_u64m8(mask, v, v);
}

HWY_API VI64M1 Compress(const VI64M1 v, const vbool64_t mask) {
  return vcompress_vm_i64m1(mask, v, v);
}
HWY_API VI64M2 Compress(const VI64M2 v, const vbool32_t mask) {
  return vcompress_vm_i64m2(mask, v, v);
}
HWY_API VI64M4 Compress(const VI64M4 v, const vbool16_t mask) {
  return vcompress_vm_i64m4(mask, v, v);
}
HWY_API VI64M8 Compress(const VI64M8 v, const vbool8_t mask) {
  return vcompress_vm_i64m8(mask, v, v);
}

HWY_API VF32M1 Compress(const VF32M1 v, const vbool32_t mask) {
  return vcompress_vm_f32m1(mask, v, v);
}
HWY_API VF32M2 Compress(const VF32M2 v, const vbool16_t mask) {
  return vcompress_vm_f32m2(mask, v, v);
}
HWY_API VF32M4 Compress(const VF32M4 v, const vbool8_t mask) {
  return vcompress_vm_f32m4(mask, v, v);
}
HWY_API VF32M8 Compress(const VF32M8 v, const vbool4_t mask) {
  return vcompress_vm_f32m8(mask, v, v);
}

HWY_API VF64M1 Compress(const VF64M1 v, const vbool64_t mask) {
  return vcompress_vm_f64m1(mask, v, v);
}
HWY_API VF64M2 Compress(const VF64M2 v, const vbool32_t mask) {
  return vcompress_vm_f64m2(mask, v, v);
}
HWY_API VF64M4 Compress(const VF64M4 v, const vbool16_t mask) {
  return vcompress_vm_f64m4(mask, v, v);
}
HWY_API VF64M8 Compress(const VF64M8 v, const vbool8_t mask) {
  return vcompress_vm_f64m8(mask, v, v);
}

// ------------------------------ CompressStore

template <class V, class M, class D>
HWY_API size_t CompressStore(const V v, const M mask, const D d,
                             TFromD<D>* HWY_RESTRICT aligned) {
  Store(Compress(v, mask), d, aligned);
  return CountTrue(mask);
}

// ------------------------------ TableLookupLanes

template <class D, class DU = RebindToUnsigned<D>>
HWY_API VFromD<DU> SetTableIndices(D d, const TFromD<DU>* idx) {
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER)
  const size_t N = Lanes(d);
  for (size_t i = 0; i < N; ++i) {
    HWY_DASSERT(0 <= idx[i] && idx[i] < static_cast<TFromD<D>>(N));
  }
#endif
  return Load(DU(), idx);
}

// <32bit are not part of Highway API, but used in Broadcast. This limits VLMAX
// to 2048! We could instead use vrgatherei16.
HWY_API VU8M1 TableLookupLanes(const VU8M1 v, VU8M1 idx) {
  return vrgather_vv_u8m1(v, idx);
}
HWY_API VU8M2 TableLookupLanes(const VU8M2 v, VU8M2 idx) {
  return vrgather_vv_u8m2(v, idx);
}
HWY_API VU8M4 TableLookupLanes(const VU8M4 v, VU8M4 idx) {
  return vrgather_vv_u8m4(v, idx);
}
HWY_API VU8M8 TableLookupLanes(const VU8M8 v, VU8M8 idx) {
  return vrgather_vv_u8m8(v, idx);
}

HWY_API VU16M1 TableLookupLanes(const VU16M1 v, VU16M1 idx) {
  return vrgather_vv_u16m1(v, idx);
}
HWY_API VU16M2 TableLookupLanes(const VU16M2 v, VU16M2 idx) {
  return vrgather_vv_u16m2(v, idx);
}
HWY_API VU16M4 TableLookupLanes(const VU16M4 v, VU16M4 idx) {
  return vrgather_vv_u16m4(v, idx);
}
HWY_API VU16M8 TableLookupLanes(const VU16M8 v, VU16M8 idx) {
  return vrgather_vv_u16m8(v, idx);
}

HWY_API VU32M1 TableLookupLanes(const VU32M1 v, VU32M1 idx) {
  return vrgather_vv_u32m1(v, idx);
}
HWY_API VU32M2 TableLookupLanes(const VU32M2 v, VU32M2 idx) {
  return vrgather_vv_u32m2(v, idx);
}
HWY_API VU32M4 TableLookupLanes(const VU32M4 v, VU32M4 idx) {
  return vrgather_vv_u32m4(v, idx);
}
HWY_API VU32M8 TableLookupLanes(const VU32M8 v, VU32M8 idx) {
  return vrgather_vv_u32m8(v, idx);
}

HWY_API VU64M1 TableLookupLanes(const VU64M1 v, VU64M1 idx) {
  return vrgather_vv_u64m1(v, idx);
}
HWY_API VU64M2 TableLookupLanes(const VU64M2 v, VU64M2 idx) {
  return vrgather_vv_u64m2(v, idx);
}
HWY_API VU64M4 TableLookupLanes(const VU64M4 v, VU64M4 idx) {
  return vrgather_vv_u64m4(v, idx);
}
HWY_API VU64M8 TableLookupLanes(const VU64M8 v, VU64M8 idx) {
  return vrgather_vv_u64m8(v, idx);
}

// <32bit are not part of Highway API, but used in Broadcast. This limits VLMAX
// to 2048! We could instead use vrgatherei16.
HWY_API VI8M1 TableLookupLanes(const VI8M1 v, VU8M1 idx) {
  return vrgather_vv_i8m1(v, idx);
}
HWY_API VI8M2 TableLookupLanes(const VI8M2 v, VU8M2 idx) {
  return vrgather_vv_i8m2(v, idx);
}
HWY_API VI8M4 TableLookupLanes(const VI8M4 v, VU8M4 idx) {
  return vrgather_vv_i8m4(v, idx);
}
HWY_API VI8M8 TableLookupLanes(const VI8M8 v, VU8M8 idx) {
  return vrgather_vv_i8m8(v, idx);
}

HWY_API VI16M1 TableLookupLanes(const VI16M1 v, VU16M1 idx) {
  return vrgather_vv_i16m1(v, idx);
}
HWY_API VI16M2 TableLookupLanes(const VI16M2 v, VU16M2 idx) {
  return vrgather_vv_i16m2(v, idx);
}
HWY_API VI16M4 TableLookupLanes(const VI16M4 v, VU16M4 idx) {
  return vrgather_vv_i16m4(v, idx);
}
HWY_API VI16M8 TableLookupLanes(const VI16M8 v, VU16M8 idx) {
  return vrgather_vv_i16m8(v, idx);
}

HWY_API VI32M1 TableLookupLanes(const VI32M1 v, VU32M1 idx) {
  return vrgather_vv_i32m1(v, idx);
}
HWY_API VI32M2 TableLookupLanes(const VI32M2 v, VU32M2 idx) {
  return vrgather_vv_i32m2(v, idx);
}
HWY_API VI32M4 TableLookupLanes(const VI32M4 v, VU32M4 idx) {
  return vrgather_vv_i32m4(v, idx);
}
HWY_API VI32M8 TableLookupLanes(const VI32M8 v, VU32M8 idx) {
  return vrgather_vv_i32m8(v, idx);
}

HWY_API VI64M1 TableLookupLanes(const VI64M1 v, VU64M1 idx) {
  return vrgather_vv_i64m1(v, idx);
}
HWY_API VI64M2 TableLookupLanes(const VI64M2 v, VU64M2 idx) {
  return vrgather_vv_i64m2(v, idx);
}
HWY_API VI64M4 TableLookupLanes(const VI64M4 v, VU64M4 idx) {
  return vrgather_vv_i64m4(v, idx);
}
HWY_API VI64M8 TableLookupLanes(const VI64M8 v, VU64M8 idx) {
  return vrgather_vv_i64m8(v, idx);
}

HWY_API VF32M1 TableLookupLanes(const VF32M1 v, VU32M1 idx) {
  return vrgather_vv_f32m1(v, idx);
}
HWY_API VF32M2 TableLookupLanes(const VF32M2 v, VU32M2 idx) {
  return vrgather_vv_f32m2(v, idx);
}
HWY_API VF32M4 TableLookupLanes(const VF32M4 v, VU32M4 idx) {
  return vrgather_vv_f32m4(v, idx);
}
HWY_API VF32M8 TableLookupLanes(const VF32M8 v, VU32M8 idx) {
  return vrgather_vv_f32m8(v, idx);
}

HWY_API VF64M1 TableLookupLanes(const VF64M1 v, VU64M1 idx) {
  return vrgather_vv_f64m1(v, idx);
}
HWY_API VF64M2 TableLookupLanes(const VF64M2 v, VU64M2 idx) {
  return vrgather_vv_f64m2(v, idx);
}
HWY_API VF64M4 TableLookupLanes(const VF64M4 v, VU64M4 idx) {
  return vrgather_vv_f64m4(v, idx);
}
HWY_API VF64M8 TableLookupLanes(const VF64M8 v, VU64M8 idx) {
  return vrgather_vv_f64m8(v, idx);
}

// ------------------------------ Shuffle01

template <class V>
HWY_API V Shuffle01(const V v) {
  using D = DFromV<V>;
  static_assert(sizeof(TFromD<D>) == 8, "Defined for 64-bit types");
  const auto idx = detail::Xor(detail::Iota0(D()), 1);
  return TableLookupLanes(v, idx);
}

// ------------------------------ Shuffle2301

template <class V>
HWY_API V Shuffle2301(const V v) {
  using D = DFromV<V>;
  static_assert(sizeof(TFromD<D>) == 4, "Defined for 32-bit types");
  const auto idx = detail::Xor(detail::Iota0(D()), 1);
  return TableLookupLanes(v, idx);
}

// ------------------------------ Shuffle1032

template <class V>
HWY_API V Shuffle1032(const V v) {
  using D = DFromV<V>;
  static_assert(sizeof(TFromD<D>) == 4, "Defined for 32-bit types");
  const auto idx = detail::Xor(detail::Iota0(D()), 2);
  return TableLookupLanes(v, idx);
}

// ------------------------------ Shuffle0123

template <class V>
HWY_API V Shuffle0123(const V v) {
  using D = DFromV<V>;
  static_assert(sizeof(TFromD<D>) == 4, "Defined for 32-bit types");
  const auto idx = detail::Xor(detail::Iota0(D()), 3);
  return TableLookupLanes(v, idx);
}

// ------------------------------ Shuffle2103

template <class V>
HWY_API V Shuffle2103(const V v) {
  using D = DFromV<V>;
  static_assert(sizeof(TFromD<D>) == 4, "Defined for 32-bit types");
  const auto i = detail::Iota0(D());
  // This shuffle is a rotation. We can compute subtraction modulo 4 (number of
  // lanes per 128-bit block) via bitwise ops.
  const auto lsb = detail::And(i, 1);
  const auto borrow = Add(lsb, lsb);
  const auto idx = detail::Xor(Xor(i, borrow), 1);
  return TableLookupLanes(v, idx);
}

// ------------------------------ Shuffle0321

template <class V>
HWY_API V Shuffle0321(const V v) {
  using D = DFromV<V>;
  static_assert(sizeof(TFromD<D>) == 4, "Defined for 32-bit types");
  const auto i = detail::Iota0(D());
  // This shuffle is a rotation. We can compute subtraction modulo 4 (number of
  // lanes per 128-bit block) via bitwise ops.
  const auto lsb = detail::And(i, 1);
  const auto borrow = Add(lsb, lsb);
  const auto idx = detail::Xor(Xor(i, borrow), 3);
  return TableLookupLanes(v, idx);
}

// ------------------------------ TableLookupBytes

namespace detail {

// For x86-compatible behaviour mandated by Highway API: TableLookupBytes
// offsets are implicitly relative to the start of their 128-bit block.
template <class D>
constexpr size_t LanesPerBlock(D) {
  return 16 / sizeof(TFromD<D>);
}

template <class D, class V>
HWY_API V OffsetsOf128BitBlocks(const D d, const V iota0) {
  using T = MakeUnsigned<TFromD<D>>;
  return detail::And(iota0, static_cast<T>(~(LanesPerBlock(d) - 1)));
}

}  // namespace detail

template <class V>
HWY_API V TableLookupBytes(const V v, const V idx) {
  using D = DFromV<V>;
  const Repartition<uint8_t, D> d8;
  constexpr size_t kLanesPerBlock = detail::LanesPerBlock(d8);
  const auto offsets128 = detail::OffsetsOf128BitBlocks(d8, detail::Iota0(d8));
  const auto idx8 = Add(BitCast(d8, idx), offsets128);
  return BitCast(D(), TableLookupLanes(BitCast(d8, v), idx8));
}

// ------------------------------ Broadcast

template <int kLane, class V>
HWY_API V Broadcast(const V v) {
  const DFromV<V> d;
  constexpr size_t kLanesPerBlock = detail::LanesPerBlock(d);
  static_assert(0 <= kLane && kLane < kLanesPerBlock, "Invalid lane");
  auto idx = detail::OffsetsOf128BitBlocks(d, detail::Iota0(d));
  if (kLane != 0) {
    idx = detail::Add(idx, kLane);
  }
  return TableLookupLanes(v, idx);
}

// ------------------------------ GetLane

HWY_API uint8_t GetLane(const VU8M1 v) { return vmv_x_s_u8m1_u8(v); }
HWY_API uint8_t GetLane(const VU8M2 v) { return vmv_x_s_u8m2_u8(v); }
HWY_API uint8_t GetLane(const VU8M4 v) { return vmv_x_s_u8m4_u8(v); }
HWY_API uint8_t GetLane(const VU8M8 v) { return vmv_x_s_u8m8_u8(v); }

HWY_API uint16_t GetLane(const VU16M1 v) { return vmv_x_s_u16m1_u16(v); }
HWY_API uint16_t GetLane(const VU16M2 v) { return vmv_x_s_u16m2_u16(v); }
HWY_API uint16_t GetLane(const VU16M4 v) { return vmv_x_s_u16m4_u16(v); }
HWY_API uint16_t GetLane(const VU16M8 v) { return vmv_x_s_u16m8_u16(v); }

HWY_API uint32_t GetLane(const VU32M1 v) { return vmv_x_s_u32m1_u32(v); }
HWY_API uint32_t GetLane(const VU32M2 v) { return vmv_x_s_u32m2_u32(v); }
HWY_API uint32_t GetLane(const VU32M4 v) { return vmv_x_s_u32m4_u32(v); }
HWY_API uint32_t GetLane(const VU32M8 v) { return vmv_x_s_u32m8_u32(v); }

HWY_API uint64_t GetLane(const VU64M1 v) { return vmv_x_s_u64m1_u64(v); }
HWY_API uint64_t GetLane(const VU64M2 v) { return vmv_x_s_u64m2_u64(v); }
HWY_API uint64_t GetLane(const VU64M4 v) { return vmv_x_s_u64m4_u64(v); }
HWY_API uint64_t GetLane(const VU64M8 v) { return vmv_x_s_u64m8_u64(v); }

HWY_API int8_t GetLane(const VI8M1 v) { return vmv_x_s_i8m1_i8(v); }
HWY_API int8_t GetLane(const VI8M2 v) { return vmv_x_s_i8m2_i8(v); }
HWY_API int8_t GetLane(const VI8M4 v) { return vmv_x_s_i8m4_i8(v); }
HWY_API int8_t GetLane(const VI8M8 v) { return vmv_x_s_i8m8_i8(v); }

HWY_API int16_t GetLane(const VI16M1 v) { return vmv_x_s_i16m1_i16(v); }
HWY_API int16_t GetLane(const VI16M2 v) { return vmv_x_s_i16m2_i16(v); }
HWY_API int16_t GetLane(const VI16M4 v) { return vmv_x_s_i16m4_i16(v); }
HWY_API int16_t GetLane(const VI16M8 v) { return vmv_x_s_i16m8_i16(v); }

HWY_API int32_t GetLane(const VI32M1 v) { return vmv_x_s_i32m1_i32(v); }
HWY_API int32_t GetLane(const VI32M2 v) { return vmv_x_s_i32m2_i32(v); }
HWY_API int32_t GetLane(const VI32M4 v) { return vmv_x_s_i32m4_i32(v); }
HWY_API int32_t GetLane(const VI32M8 v) { return vmv_x_s_i32m8_i32(v); }

HWY_API int64_t GetLane(const VI64M1 v) { return vmv_x_s_i64m1_i64(v); }
HWY_API int64_t GetLane(const VI64M2 v) { return vmv_x_s_i64m2_i64(v); }
HWY_API int64_t GetLane(const VI64M4 v) { return vmv_x_s_i64m4_i64(v); }
HWY_API int64_t GetLane(const VI64M8 v) { return vmv_x_s_i64m8_i64(v); }

HWY_API float GetLane(const VF32M1 v) { return vfmv_f_s_f32m1_f32(v); }
HWY_API float GetLane(const VF32M2 v) { return vfmv_f_s_f32m2_f32(v); }
HWY_API float GetLane(const VF32M4 v) { return vfmv_f_s_f32m4_f32(v); }
HWY_API float GetLane(const VF32M8 v) { return vfmv_f_s_f32m8_f32(v); }

HWY_API double GetLane(const VF64M1 v) { return vfmv_f_s_f64m1_f64(v); }
HWY_API double GetLane(const VF64M2 v) { return vfmv_f_s_f64m2_f64(v); }
HWY_API double GetLane(const VF64M4 v) { return vfmv_f_s_f64m4_f64(v); }
HWY_API double GetLane(const VF64M8 v) { return vfmv_f_s_f64m8_f64(v); }

// ------------------------------ ShiftLeftLanes

namespace detail {

VU8M1 SlideUp(VU8M1 v, size_t lanes) { return vslideup_vx_u8m1(v, v, lanes); }
VU8M2 SlideUp(VU8M2 v, size_t lanes) { return vslideup_vx_u8m2(v, v, lanes); }
VU8M4 SlideUp(VU8M4 v, size_t lanes) { return vslideup_vx_u8m4(v, v, lanes); }
VU8M8 SlideUp(VU8M8 v, size_t lanes) { return vslideup_vx_u8m8(v, v, lanes); }

VU16M1 SlideUp(VU16M1 v, size_t lanes) {
  return vslideup_vx_u16m1(v, v, lanes);
}
VU16M2 SlideUp(VU16M2 v, size_t lanes) {
  return vslideup_vx_u16m2(v, v, lanes);
}
VU16M4 SlideUp(VU16M4 v, size_t lanes) {
  return vslideup_vx_u16m4(v, v, lanes);
}
VU16M8 SlideUp(VU16M8 v, size_t lanes) {
  return vslideup_vx_u16m8(v, v, lanes);
}

VU32M1 SlideUp(VU32M1 v, size_t lanes) {
  return vslideup_vx_u32m1(v, v, lanes);
}
VU32M2 SlideUp(VU32M2 v, size_t lanes) {
  return vslideup_vx_u32m2(v, v, lanes);
}
VU32M4 SlideUp(VU32M4 v, size_t lanes) {
  return vslideup_vx_u32m4(v, v, lanes);
}
VU32M8 SlideUp(VU32M8 v, size_t lanes) {
  return vslideup_vx_u32m8(v, v, lanes);
}

VU64M1 SlideUp(VU64M1 v, size_t lanes) {
  return vslideup_vx_u64m1(v, v, lanes);
}
VU64M2 SlideUp(VU64M2 v, size_t lanes) {
  return vslideup_vx_u64m2(v, v, lanes);
}
VU64M4 SlideUp(VU64M4 v, size_t lanes) {
  return vslideup_vx_u64m4(v, v, lanes);
}
VU64M8 SlideUp(VU64M8 v, size_t lanes) {
  return vslideup_vx_u64m8(v, v, lanes);
}

VI8M1 SlideUp(VI8M1 v, size_t lanes) { return vslideup_vx_i8m1(v, v, lanes); }
VI8M2 SlideUp(VI8M2 v, size_t lanes) { return vslideup_vx_i8m2(v, v, lanes); }
VI8M4 SlideUp(VI8M4 v, size_t lanes) { return vslideup_vx_i8m4(v, v, lanes); }
VI8M8 SlideUp(VI8M8 v, size_t lanes) { return vslideup_vx_i8m8(v, v, lanes); }

VI16M1 SlideUp(VI16M1 v, size_t lanes) {
  return vslideup_vx_i16m1(v, v, lanes);
}
VI16M2 SlideUp(VI16M2 v, size_t lanes) {
  return vslideup_vx_i16m2(v, v, lanes);
}
VI16M4 SlideUp(VI16M4 v, size_t lanes) {
  return vslideup_vx_i16m4(v, v, lanes);
}
VI16M8 SlideUp(VI16M8 v, size_t lanes) {
  return vslideup_vx_i16m8(v, v, lanes);
}

VI32M1 SlideUp(VI32M1 v, size_t lanes) {
  return vslideup_vx_i32m1(v, v, lanes);
}
VI32M2 SlideUp(VI32M2 v, size_t lanes) {
  return vslideup_vx_i32m2(v, v, lanes);
}
VI32M4 SlideUp(VI32M4 v, size_t lanes) {
  return vslideup_vx_i32m4(v, v, lanes);
}
VI32M8 SlideUp(VI32M8 v, size_t lanes) {
  return vslideup_vx_i32m8(v, v, lanes);
}

VI64M1 SlideUp(VI64M1 v, size_t lanes) {
  return vslideup_vx_i64m1(v, v, lanes);
}
VI64M2 SlideUp(VI64M2 v, size_t lanes) {
  return vslideup_vx_i64m2(v, v, lanes);
}
VI64M4 SlideUp(VI64M4 v, size_t lanes) {
  return vslideup_vx_i64m4(v, v, lanes);
}
VI64M8 SlideUp(VI64M8 v, size_t lanes) {
  return vslideup_vx_i64m8(v, v, lanes);
}

VF32M1 SlideUp(VF32M1 v, size_t lanes) {
  return vslideup_vx_f32m1(v, v, lanes);
}
VF32M2 SlideUp(VF32M2 v, size_t lanes) {
  return vslideup_vx_f32m2(v, v, lanes);
}
VF32M4 SlideUp(VF32M4 v, size_t lanes) {
  return vslideup_vx_f32m4(v, v, lanes);
}
VF32M8 SlideUp(VF32M8 v, size_t lanes) {
  return vslideup_vx_f32m8(v, v, lanes);
}

VF64M1 SlideUp(VF64M1 v, size_t lanes) {
  return vslideup_vx_f64m1(v, v, lanes);
}
VF64M2 SlideUp(VF64M2 v, size_t lanes) {
  return vslideup_vx_f64m2(v, v, lanes);
}
VF64M4 SlideUp(VF64M4 v, size_t lanes) {
  return vslideup_vx_f64m4(v, v, lanes);
}
VF64M8 SlideUp(VF64M8 v, size_t lanes) {
  return vslideup_vx_f64m8(v, v, lanes);
}

}  // namespace detail

template <int kLanes, class V>
HWY_API V ShiftLeftLanes(const V v) {
  using D = DFromV<V>;
  const RebindToSigned<D> di;
  const auto shifted = detail::SlideUp(v, kLanes);
  // Match x86 semantics by zeroing lower lanes in 128-bit blocks
  constexpr size_t kLanesPerBlock = 16 / sizeof(TFromV<V>);
  const auto idx_mod = detail::And(detail::Iota0(D()), kLanesPerBlock - 1);
  const auto clear = Lt(BitCast(di, idx_mod), Set(di, kLanes));
  return IfThenZeroElse(clear, shifted);
}

// ------------------------------ ShiftLeftBytes

template <int kBytes, class V>
HWY_API V ShiftLeftBytes(const V v) {
  using D = DFromV<V>;
  const Repartition<uint8_t, D> d8;
  return BitCast(D(), ShiftLeftLanes<kBytes>(BitCast(d8, v)));
}

// ------------------------------ ShiftRightLanes

namespace detail {

VU8M1 SlideDown(VU8M1 v, size_t lanes) {
  return vslidedown_vx_u8m1(v, v, lanes);
}
VU8M2 SlideDown(VU8M2 v, size_t lanes) {
  return vslidedown_vx_u8m2(v, v, lanes);
}
VU8M4 SlideDown(VU8M4 v, size_t lanes) {
  return vslidedown_vx_u8m4(v, v, lanes);
}
VU8M8 SlideDown(VU8M8 v, size_t lanes) {
  return vslidedown_vx_u8m8(v, v, lanes);
}

VU16M1 SlideDown(VU16M1 v, size_t lanes) {
  return vslidedown_vx_u16m1(v, v, lanes);
}
VU16M2 SlideDown(VU16M2 v, size_t lanes) {
  return vslidedown_vx_u16m2(v, v, lanes);
}
VU16M4 SlideDown(VU16M4 v, size_t lanes) {
  return vslidedown_vx_u16m4(v, v, lanes);
}
VU16M8 SlideDown(VU16M8 v, size_t lanes) {
  return vslidedown_vx_u16m8(v, v, lanes);
}

VU32M1 SlideDown(VU32M1 v, size_t lanes) {
  return vslidedown_vx_u32m1(v, v, lanes);
}
VU32M2 SlideDown(VU32M2 v, size_t lanes) {
  return vslidedown_vx_u32m2(v, v, lanes);
}
VU32M4 SlideDown(VU32M4 v, size_t lanes) {
  return vslidedown_vx_u32m4(v, v, lanes);
}
VU32M8 SlideDown(VU32M8 v, size_t lanes) {
  return vslidedown_vx_u32m8(v, v, lanes);
}

VU64M1 SlideDown(VU64M1 v, size_t lanes) {
  return vslidedown_vx_u64m1(v, v, lanes);
}
VU64M2 SlideDown(VU64M2 v, size_t lanes) {
  return vslidedown_vx_u64m2(v, v, lanes);
}
VU64M4 SlideDown(VU64M4 v, size_t lanes) {
  return vslidedown_vx_u64m4(v, v, lanes);
}
VU64M8 SlideDown(VU64M8 v, size_t lanes) {
  return vslidedown_vx_u64m8(v, v, lanes);
}

VI8M1 SlideDown(VI8M1 v, size_t lanes) {
  return vslidedown_vx_i8m1(v, v, lanes);
}
VI8M2 SlideDown(VI8M2 v, size_t lanes) {
  return vslidedown_vx_i8m2(v, v, lanes);
}
VI8M4 SlideDown(VI8M4 v, size_t lanes) {
  return vslidedown_vx_i8m4(v, v, lanes);
}
VI8M8 SlideDown(VI8M8 v, size_t lanes) {
  return vslidedown_vx_i8m8(v, v, lanes);
}

VI16M1 SlideDown(VI16M1 v, size_t lanes) {
  return vslidedown_vx_i16m1(v, v, lanes);
}
VI16M2 SlideDown(VI16M2 v, size_t lanes) {
  return vslidedown_vx_i16m2(v, v, lanes);
}
VI16M4 SlideDown(VI16M4 v, size_t lanes) {
  return vslidedown_vx_i16m4(v, v, lanes);
}
VI16M8 SlideDown(VI16M8 v, size_t lanes) {
  return vslidedown_vx_i16m8(v, v, lanes);
}

VI32M1 SlideDown(VI32M1 v, size_t lanes) {
  return vslidedown_vx_i32m1(v, v, lanes);
}
VI32M2 SlideDown(VI32M2 v, size_t lanes) {
  return vslidedown_vx_i32m2(v, v, lanes);
}
VI32M4 SlideDown(VI32M4 v, size_t lanes) {
  return vslidedown_vx_i32m4(v, v, lanes);
}
VI32M8 SlideDown(VI32M8 v, size_t lanes) {
  return vslidedown_vx_i32m8(v, v, lanes);
}

VI64M1 SlideDown(VI64M1 v, size_t lanes) {
  return vslidedown_vx_i64m1(v, v, lanes);
}
VI64M2 SlideDown(VI64M2 v, size_t lanes) {
  return vslidedown_vx_i64m2(v, v, lanes);
}
VI64M4 SlideDown(VI64M4 v, size_t lanes) {
  return vslidedown_vx_i64m4(v, v, lanes);
}
VI64M8 SlideDown(VI64M8 v, size_t lanes) {
  return vslidedown_vx_i64m8(v, v, lanes);
}

VF32M1 SlideDown(VF32M1 v, size_t lanes) {
  return vslidedown_vx_f32m1(v, v, lanes);
}
VF32M2 SlideDown(VF32M2 v, size_t lanes) {
  return vslidedown_vx_f32m2(v, v, lanes);
}
VF32M4 SlideDown(VF32M4 v, size_t lanes) {
  return vslidedown_vx_f32m4(v, v, lanes);
}
VF32M8 SlideDown(VF32M8 v, size_t lanes) {
  return vslidedown_vx_f32m8(v, v, lanes);
}

VF64M1 SlideDown(VF64M1 v, size_t lanes) {
  return vslidedown_vx_f64m1(v, v, lanes);
}
VF64M2 SlideDown(VF64M2 v, size_t lanes) {
  return vslidedown_vx_f64m2(v, v, lanes);
}
VF64M4 SlideDown(VF64M4 v, size_t lanes) {
  return vslidedown_vx_f64m4(v, v, lanes);
}
VF64M8 SlideDown(VF64M8 v, size_t lanes) {
  return vslidedown_vx_f64m8(v, v, lanes);
}

}  // namespace detail

template <int kLanes, class V>
HWY_API V ShiftRightLanes(const V v) {
  using D = DFromV<V>;
  const RebindToSigned<D> di;
  const auto shifted = detail::SlideDown(v, kLanes);
  // Match x86 semantics by zeroing upper lanes in 128-bit blocks
  constexpr size_t kLanesPerBlock = 16 / sizeof(TFromV<V>);
  const auto idx_mod = detail::And(detail::Iota0(D()), kLanesPerBlock - 1);
  const auto keep = Lt(BitCast(di, idx_mod), Set(di, kLanes));
  return IfThenElseZero(keep, shifted);
}

// ------------------------------ ShiftRightBytes

template <int kBytes, class V>
HWY_API V ShiftRightBytes(const V v) {
  using D = DFromV<V>;
  const Repartition<uint8_t, D> d8;
  return BitCast(D(), ShiftRightLanes<kBytes>(BitCast(d8, v)));
}

// ------------------------------ OddEven

template <class V>
HWY_API V OddEven(const V a, const V b) {
  const RebindToUnsigned<DFromV<V>> du;  // Iota0 is unsigned only
  const auto is_even = Eq(detail::And(detail::Iota0(du), 1), Zero(du));
  return IfThenElse(is_even, b, a);
}

// ------------------------------ InterleaveLower

template <class V>
HWY_API V InterleaveLower(const V a, const V b) {
  const DFromV<V> d;
  const RebindToUnsigned<decltype(d)> du;
  constexpr size_t kLanesPerBlock = detail::LanesPerBlock(d);
  const auto i = detail::Iota0(d);
  const auto idx_mod = ShiftRight<1>(detail::And(i, kLanesPerBlock - 1));
  const auto idx = Add(idx_mod, detail::OffsetsOf128BitBlocks(d, i));
  const auto is_even = Eq(detail::And(i, 1), Zero(du));
  return IfThenElse(is_even, TableLookupLanes(a, idx),
                    TableLookupLanes(b, idx));
}

// ------------------------------ InterleaveUpper

template <class V>
HWY_API V InterleaveUpper(const V a, const V b) {
  const DFromV<V> d;
  const RebindToUnsigned<decltype(d)> du;
  constexpr size_t kLanesPerBlock = detail::LanesPerBlock(d);
  const auto i = detail::Iota0(d);
  const auto idx_mod = ShiftRight<1>(detail::And(i, kLanesPerBlock - 1));
  const auto idx_lower = Add(idx_mod, detail::OffsetsOf128BitBlocks(d, i));
  const auto idx = detail::Add(idx_lower, kLanesPerBlock / 2);
  const auto is_even = Eq(detail::And(i, 1), Zero(du));
  return IfThenElse(is_even, TableLookupLanes(a, idx),
                    TableLookupLanes(b, idx));
}

// ------------------------------ ZipLower

template <class V>
HWY_API VFromD<RepartitionToWide<DFromV<V>>> ZipLower(const V a, const V b) {
  RepartitionToWide<DFromV<V>> dw;
  return BitCast(dw, InterleaveLower(a, b));
}

// ------------------------------ ZipUpper

template <class V>
HWY_API VFromD<RepartitionToWide<DFromV<V>>> ZipUpper(const V a, const V b) {
  RepartitionToWide<DFromV<V>> dw;
  return BitCast(dw, InterleaveUpper(a, b));
}

// ------------------------------ Combine

// TODO(janwas): implement after LMUL ext/trunc
#if 0

template <class V>
HWY_API V Combine(const V a, const V b) {
  using D = DFromV<V>;
  // double LMUL of inputs, then SlideUp with Lanes().
}

#endif

// ================================================== REDUCE

// ------------------------------ SumOfLanes

namespace detail {

HWY_API VU32M1 RedSum(const VU32M1 v, const VU32M1 v0) {
  return vrgather_vx_u32m1(vredsum_vs_u32m1_u32m1(v0, v, v0), 0);
}
HWY_API VU32M2 RedSum(const VU32M2 v, const VU32M1 v0) {
  return Set(DU32M2(), GetLane(vredsum_vs_u32m2_u32m1(v0, v, v0)));
}
HWY_API VU32M4 RedSum(const VU32M4 v, const VU32M1 v0) {
  return Set(DU32M4(), GetLane(vredsum_vs_u32m4_u32m1(v0, v, v0)));
}
HWY_API VU32M8 RedSum(const VU32M8 v, const VU32M1 v0) {
  return Set(DU32M8(), GetLane(vredsum_vs_u32m8_u32m1(v0, v, v0)));
}

HWY_API VU64M1 RedSum(const VU64M1 v, const VU64M1 v0) {
  return vrgather_vx_u64m1(vredsum_vs_u64m1_u64m1(v0, v, v0), 0);
}
HWY_API VU64M2 RedSum(const VU64M2 v, const VU64M1 v0) {
  return Set(DU64M2(), GetLane(vredsum_vs_u64m2_u64m1(v0, v, v0)));
}
HWY_API VU64M4 RedSum(const VU64M4 v, const VU64M1 v0) {
  return Set(DU64M4(), GetLane(vredsum_vs_u64m4_u64m1(v0, v, v0)));
}
HWY_API VU64M8 RedSum(const VU64M8 v, const VU64M1 v0) {
  return Set(DU64M8(), GetLane(vredsum_vs_u64m8_u64m1(v0, v, v0)));
}

HWY_API VI32M1 RedSum(const VI32M1 v, const VI32M1 v0) {
  return vrgather_vx_i32m1(vredsum_vs_i32m1_i32m1(v0, v, v0), 0);
}
HWY_API VI32M2 RedSum(const VI32M2 v, const VI32M1 v0) {
  return Set(DI32M2(), GetLane(vredsum_vs_i32m2_i32m1(v0, v, v0)));
}
HWY_API VI32M4 RedSum(const VI32M4 v, const VI32M1 v0) {
  return Set(DI32M4(), GetLane(vredsum_vs_i32m4_i32m1(v0, v, v0)));
}
HWY_API VI32M8 RedSum(const VI32M8 v, const VI32M1 v0) {
  return Set(DI32M8(), GetLane(vredsum_vs_i32m8_i32m1(v0, v, v0)));
}

HWY_API VI64M1 RedSum(const VI64M1 v, const VI64M1 v0) {
  return vrgather_vx_i64m1(vredsum_vs_i64m1_i64m1(v0, v, v0), 0);
}
HWY_API VI64M2 RedSum(const VI64M2 v, const VI64M1 v0) {
  return Set(DI64M2(), GetLane(vredsum_vs_i64m2_i64m1(v0, v, v0)));
}
HWY_API VI64M4 RedSum(const VI64M4 v, const VI64M1 v0) {
  return Set(DI64M4(), GetLane(vredsum_vs_i64m4_i64m1(v0, v, v0)));
}
HWY_API VI64M8 RedSum(const VI64M8 v, const VI64M1 v0) {
  return Set(DI64M8(), GetLane(vredsum_vs_i64m8_i64m1(v0, v, v0)));
}

HWY_API VF32M1 RedSum(const VF32M1 v, const VF32M1 v0) {
  return vrgather_vx_f32m1(vfredsum_vs_f32m1_f32m1(v0, v, v0), 0);
}
HWY_API VF32M2 RedSum(const VF32M2 v, const VF32M1 v0) {
  return Set(DF32M2(), GetLane(vfredsum_vs_f32m2_f32m1(v0, v, v0)));
}
HWY_API VF32M4 RedSum(const VF32M4 v, const VF32M1 v0) {
  return Set(DF32M4(), GetLane(vfredsum_vs_f32m4_f32m1(v0, v, v0)));
}
HWY_API VF32M8 RedSum(const VF32M8 v, const VF32M1 v0) {
  return Set(DF32M8(), GetLane(vfredsum_vs_f32m8_f32m1(v0, v, v0)));
}

HWY_API VF64M1 RedSum(const VF64M1 v, const VF64M1 v0) {
  return vrgather_vx_f64m1(vfredsum_vs_f64m1_f64m1(v0, v, v0), 0);
}
HWY_API VF64M2 RedSum(const VF64M2 v, const VF64M1 v0) {
  return Set(DF64M2(), GetLane(vfredsum_vs_f64m2_f64m1(v0, v, v0)));
}
HWY_API VF64M4 RedSum(const VF64M4 v, const VF64M1 v0) {
  return Set(DF64M4(), GetLane(vfredsum_vs_f64m4_f64m1(v0, v, v0)));
}
HWY_API VF64M8 RedSum(const VF64M8 v, const VF64M1 v0) {
  return Set(DF64M8(), GetLane(vfredsum_vs_f64m8_f64m1(v0, v, v0)));
}

}  // namespace detail

template <class V>
HWY_API V SumOfLanes(const V v) {
  using T = TFromV<V>;
  const auto v0 = Zero(Simd<T, HWY_LANES(T)>());  // always m1
  return detail::RedSum(v, v0);
}

// ------------------------------ MinOfLanes
namespace detail {

HWY_API VU32M1 RedMin(const VU32M1 v, const VU32M1 v0) {
  return vrgather_vx_u32m1(vredminu_vs_u32m1_u32m1(v0, v, v0), 0);
}
HWY_API VU32M2 RedMin(const VU32M2 v, const VU32M1 v0) {
  return Set(DU32M2(), GetLane(vredminu_vs_u32m2_u32m1(v0, v, v0)));
}
HWY_API VU32M4 RedMin(const VU32M4 v, const VU32M1 v0) {
  return Set(DU32M4(), GetLane(vredminu_vs_u32m4_u32m1(v0, v, v0)));
}
HWY_API VU32M8 RedMin(const VU32M8 v, const VU32M1 v0) {
  return Set(DU32M8(), GetLane(vredminu_vs_u32m8_u32m1(v0, v, v0)));
}

HWY_API VU64M1 RedMin(const VU64M1 v, const VU64M1 v0) {
  return vrgather_vx_u64m1(vredminu_vs_u64m1_u64m1(v0, v, v0), 0);
}
HWY_API VU64M2 RedMin(const VU64M2 v, const VU64M1 v0) {
  return Set(DU64M2(), GetLane(vredminu_vs_u64m2_u64m1(v0, v, v0)));
}
HWY_API VU64M4 RedMin(const VU64M4 v, const VU64M1 v0) {
  return Set(DU64M4(), GetLane(vredminu_vs_u64m4_u64m1(v0, v, v0)));
}
HWY_API VU64M8 RedMin(const VU64M8 v, const VU64M1 v0) {
  return Set(DU64M8(), GetLane(vredminu_vs_u64m8_u64m1(v0, v, v0)));
}

HWY_API VI32M1 RedMin(const VI32M1 v, const VI32M1 v0) {
  return vrgather_vx_i32m1(vredmin_vs_i32m1_i32m1(v0, v, v0), 0);
}
HWY_API VI32M2 RedMin(const VI32M2 v, const VI32M1 v0) {
  return Set(DI32M2(), GetLane(vredmin_vs_i32m2_i32m1(v0, v, v0)));
}
HWY_API VI32M4 RedMin(const VI32M4 v, const VI32M1 v0) {
  return Set(DI32M4(), GetLane(vredmin_vs_i32m4_i32m1(v0, v, v0)));
}
HWY_API VI32M8 RedMin(const VI32M8 v, const VI32M1 v0) {
  return Set(DI32M8(), GetLane(vredmin_vs_i32m8_i32m1(v0, v, v0)));
}

HWY_API VI64M1 RedMin(const VI64M1 v, const VI64M1 v0) {
  return vrgather_vx_i64m1(vredmin_vs_i64m1_i64m1(v0, v, v0), 0);
}
HWY_API VI64M2 RedMin(const VI64M2 v, const VI64M1 v0) {
  return Set(DI64M2(), GetLane(vredmin_vs_i64m2_i64m1(v0, v, v0)));
}
HWY_API VI64M4 RedMin(const VI64M4 v, const VI64M1 v0) {
  return Set(DI64M4(), GetLane(vredmin_vs_i64m4_i64m1(v0, v, v0)));
}
HWY_API VI64M8 RedMin(const VI64M8 v, const VI64M1 v0) {
  return Set(DI64M8(), GetLane(vredmin_vs_i64m8_i64m1(v0, v, v0)));
}

HWY_API VF32M1 RedMin(const VF32M1 v, const VF32M1 v0) {
  return vrgather_vx_f32m1(vfredmin_vs_f32m1_f32m1(v0, v, v0), 0);
}
HWY_API VF32M2 RedMin(const VF32M2 v, const VF32M1 v0) {
  return Set(DF32M2(), GetLane(vfredmin_vs_f32m2_f32m1(v0, v, v0)));
}
HWY_API VF32M4 RedMin(const VF32M4 v, const VF32M1 v0) {
  return Set(DF32M4(), GetLane(vfredmin_vs_f32m4_f32m1(v0, v, v0)));
}
HWY_API VF32M8 RedMin(const VF32M8 v, const VF32M1 v0) {
  return Set(DF32M8(), GetLane(vfredmin_vs_f32m8_f32m1(v0, v, v0)));
}

HWY_API VF64M1 RedMin(const VF64M1 v, const VF64M1 v0) {
  return vrgather_vx_f64m1(vfredmin_vs_f64m1_f64m1(v0, v, v0), 0);
}
HWY_API VF64M2 RedMin(const VF64M2 v, const VF64M1 v0) {
  return Set(DF64M2(), GetLane(vfredmin_vs_f64m2_f64m1(v0, v, v0)));
}
HWY_API VF64M4 RedMin(const VF64M4 v, const VF64M1 v0) {
  return Set(DF64M4(), GetLane(vfredmin_vs_f64m4_f64m1(v0, v, v0)));
}
HWY_API VF64M8 RedMin(const VF64M8 v, const VF64M1 v0) {
  return Set(DF64M8(), GetLane(vfredmin_vs_f64m8_f64m1(v0, v, v0)));
}

}  // namespace detail

template <class V>
HWY_API V MinOfLanes(const V v) {
  using T = TFromV<V>;
  const auto v0 = Zero(Simd<T, HWY_LANES(T)>());  // always m1
  return detail::RedMin(v, v0);
}

// ------------------------------ MaxOfLanes
namespace detail {

HWY_API VU32M1 RedMax(const VU32M1 v, const VU32M1 v0) {
  return vrgather_vx_u32m1(vredmaxu_vs_u32m1_u32m1(v0, v, v0), 0);
}
HWY_API VU32M2 RedMax(const VU32M2 v, const VU32M1 v0) {
  return Set(DU32M2(), GetLane(vredmaxu_vs_u32m2_u32m1(v0, v, v0)));
}
HWY_API VU32M4 RedMax(const VU32M4 v, const VU32M1 v0) {
  return Set(DU32M4(), GetLane(vredmaxu_vs_u32m4_u32m1(v0, v, v0)));
}
HWY_API VU32M8 RedMax(const VU32M8 v, const VU32M1 v0) {
  return Set(DU32M8(), GetLane(vredmaxu_vs_u32m8_u32m1(v0, v, v0)));
}

HWY_API VU64M1 RedMax(const VU64M1 v, const VU64M1 v0) {
  return vrgather_vx_u64m1(vredmaxu_vs_u64m1_u64m1(v0, v, v0), 0);
}
HWY_API VU64M2 RedMax(const VU64M2 v, const VU64M1 v0) {
  return Set(DU64M2(), GetLane(vredmaxu_vs_u64m2_u64m1(v0, v, v0)));
}
HWY_API VU64M4 RedMax(const VU64M4 v, const VU64M1 v0) {
  return Set(DU64M4(), GetLane(vredmaxu_vs_u64m4_u64m1(v0, v, v0)));
}
HWY_API VU64M8 RedMax(const VU64M8 v, const VU64M1 v0) {
  return Set(DU64M8(), GetLane(vredmaxu_vs_u64m8_u64m1(v0, v, v0)));
}

HWY_API VI32M1 RedMax(const VI32M1 v, const VI32M1 v0) {
  return vrgather_vx_i32m1(vredmax_vs_i32m1_i32m1(v0, v, v0), 0);
}
HWY_API VI32M2 RedMax(const VI32M2 v, const VI32M1 v0) {
  return Set(DI32M2(), GetLane(vredmax_vs_i32m2_i32m1(v0, v, v0)));
}
HWY_API VI32M4 RedMax(const VI32M4 v, const VI32M1 v0) {
  return Set(DI32M4(), GetLane(vredmax_vs_i32m4_i32m1(v0, v, v0)));
}
HWY_API VI32M8 RedMax(const VI32M8 v, const VI32M1 v0) {
  return Set(DI32M8(), GetLane(vredmax_vs_i32m8_i32m1(v0, v, v0)));
}

HWY_API VI64M1 RedMax(const VI64M1 v, const VI64M1 v0) {
  return vrgather_vx_i64m1(vredmax_vs_i64m1_i64m1(v0, v, v0), 0);
}
HWY_API VI64M2 RedMax(const VI64M2 v, const VI64M1 v0) {
  return Set(DI64M2(), GetLane(vredmax_vs_i64m2_i64m1(v0, v, v0)));
}
HWY_API VI64M4 RedMax(const VI64M4 v, const VI64M1 v0) {
  return Set(DI64M4(), GetLane(vredmax_vs_i64m4_i64m1(v0, v, v0)));
}
HWY_API VI64M8 RedMax(const VI64M8 v, const VI64M1 v0) {
  return Set(DI64M8(), GetLane(vredmax_vs_i64m8_i64m1(v0, v, v0)));
}

HWY_API VF32M1 RedMax(const VF32M1 v, const VF32M1 v0) {
  return vrgather_vx_f32m1(vfredmax_vs_f32m1_f32m1(v0, v, v0), 0);
}
HWY_API VF32M2 RedMax(const VF32M2 v, const VF32M1 v0) {
  return Set(DF32M2(), GetLane(vfredmax_vs_f32m2_f32m1(v0, v, v0)));
}
HWY_API VF32M4 RedMax(const VF32M4 v, const VF32M1 v0) {
  return Set(DF32M4(), GetLane(vfredmax_vs_f32m4_f32m1(v0, v, v0)));
}
HWY_API VF32M8 RedMax(const VF32M8 v, const VF32M1 v0) {
  return Set(DF32M8(), GetLane(vfredmax_vs_f32m8_f32m1(v0, v, v0)));
}

HWY_API VF64M1 RedMax(const VF64M1 v, const VF64M1 v0) {
  return vrgather_vx_f64m1(vfredmax_vs_f64m1_f64m1(v0, v, v0), 0);
}
HWY_API VF64M2 RedMax(const VF64M2 v, const VF64M1 v0) {
  return Set(DF64M2(), GetLane(vfredmax_vs_f64m2_f64m1(v0, v, v0)));
}
HWY_API VF64M4 RedMax(const VF64M4 v, const VF64M1 v0) {
  return Set(DF64M4(), GetLane(vfredmax_vs_f64m4_f64m1(v0, v, v0)));
}
HWY_API VF64M8 RedMax(const VF64M8 v, const VF64M1 v0) {
  return Set(DF64M8(), GetLane(vfredmax_vs_f64m8_f64m1(v0, v, v0)));
}

}  // namespace detail

template <class V>
HWY_API V MaxOfLanes(const V v) {
  using T = TFromV<V>;
  const auto v0 = Zero(Simd<T, HWY_LANES(T)>());  // always m1
  return detail::RedMax(v, v0);
}

// ================================================== Ops with dependencies

// ------------------------------ LoadDup128

template <class D>
HWY_API VFromD<D> LoadDup128(D d, const TFromD<D>* const HWY_RESTRICT p) {
  // TODO(janwas): set VL
  const auto loaded = Load(d, p);
  constexpr size_t kLanesPerBlock = detail::LanesPerBlock(d);
  // Broadcast the first block
  const auto idx = detail::And(detail::Iota0(d), kLanesPerBlock - 1);
  return TableLookupLanes(loaded, idx);
}

// ------------------------------ Neg

template <class V, HWY_IF_SIGNED_V(V)>
HWY_API V Neg(const V v) {
  return Sub(Zero(DFromV<V>()), v);
}

HWY_API VF32M1 Neg(VF32M1 v) { return vfsgnjn_vv_f32m1(v, v); }
HWY_API VF32M2 Neg(VF32M2 v) { return vfsgnjn_vv_f32m2(v, v); }
HWY_API VF32M4 Neg(VF32M4 v) { return vfsgnjn_vv_f32m4(v, v); }
HWY_API VF32M8 Neg(VF32M8 v) { return vfsgnjn_vv_f32m8(v, v); }

HWY_API VF64M1 Neg(VF64M1 v) { return vfsgnjn_vv_f64m1(v, v); }
HWY_API VF64M2 Neg(VF64M2 v) { return vfsgnjn_vv_f64m2(v, v); }
HWY_API VF64M4 Neg(VF64M4 v) { return vfsgnjn_vv_f64m4(v, v); }
HWY_API VF64M8 Neg(VF64M8 v) { return vfsgnjn_vv_f64m8(v, v); }

// ------------------------------ Abs

template <class V, HWY_IF_SIGNED_V(V)>
HWY_API V Abs(const V v) {
  return Max(v, Neg(v));
}

HWY_API VF32M1 Abs(VF32M1 v) { return vfsgnjx_vv_f32m1(v, v); }
HWY_API VF32M2 Abs(VF32M2 v) { return vfsgnjx_vv_f32m2(v, v); }
HWY_API VF32M4 Abs(VF32M4 v) { return vfsgnjx_vv_f32m4(v, v); }
HWY_API VF32M8 Abs(VF32M8 v) { return vfsgnjx_vv_f32m8(v, v); }

HWY_API VF64M1 Abs(VF64M1 v) { return vfsgnjx_vv_f64m1(v, v); }
HWY_API VF64M2 Abs(VF64M2 v) { return vfsgnjx_vv_f64m2(v, v); }
HWY_API VF64M4 Abs(VF64M4 v) { return vfsgnjx_vv_f64m4(v, v); }
HWY_API VF64M8 Abs(VF64M8 v) { return vfsgnjx_vv_f64m8(v, v); }

// ------------------------------ AbsDiff

template <class V>
HWY_API V AbsDiff(const V a, const V b) {
  return Abs(Sub(a, b));
}

// ------------------------------ Iota

template <class D, HWY_IF_UNSIGNED_D(D)>
HWY_API auto Iota(const D d, TFromD<D> first) {
  return Add(detail::Iota0(d), Set(d, first));
}

template <class D, HWY_IF_SIGNED_D(D)>
HWY_API auto Iota(const D d, TFromD<D> first) {
  const RebindToUnsigned<D> du;
  return Add(BitCast(d, detail::Iota0(du)), Set(d, first));
}

template <class D, HWY_IF_FLOAT_D(D)>
HWY_API auto Iota(const D d, TFromD<D> first) {
  const RebindToUnsigned<D> du;
  const RebindToSigned<D> di;
  return detail::Add(ConvertTo(d, BitCast(di, detail::Iota0(du))), first);
}

// ------------------------------ MulEven

// Using vwmul does not work for m8, so use mulh instead. Highway only provides
// MulHigh for 16-bit, so use a private wrapper.
namespace detail {

HWY_API VU32M1 MulHigh(VU32M1 a, VU32M1 b) { return vmulhu_vv_u32m1(a, b); }
HWY_API VU32M2 MulHigh(VU32M2 a, VU32M2 b) { return vmulhu_vv_u32m2(a, b); }
HWY_API VU32M4 MulHigh(VU32M4 a, VU32M4 b) { return vmulhu_vv_u32m4(a, b); }
HWY_API VU32M8 MulHigh(VU32M8 a, VU32M8 b) { return vmulhu_vv_u32m8(a, b); }

HWY_API VI32M1 MulHigh(VI32M1 a, VI32M1 b) { return vmulh_vv_i32m1(a, b); }
HWY_API VI32M2 MulHigh(VI32M2 a, VI32M2 b) { return vmulh_vv_i32m2(a, b); }
HWY_API VI32M4 MulHigh(VI32M4 a, VI32M4 b) { return vmulh_vv_i32m4(a, b); }
HWY_API VI32M8 MulHigh(VI32M8 a, VI32M8 b) { return vmulh_vv_i32m8(a, b); }

}  // namespace detail

template <class V>
HWY_API VFromD<RepartitionToWide<DFromV<V>>> MulEven(const V a, const V b) {
  const auto lo = Mul(a, b);
  const auto hi = detail::MulHigh(a, b);
  const RepartitionToWide<DFromV<V>> dw;
  return BitCast(dw, OddEven(detail::SlideUp(hi, 1), lo));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
