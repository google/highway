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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/convert_test.cc"
#include "hwy/foreach_target.h"
// ^ must come before highway.h and any *-inl.h.

#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Cast and ensure bytes are the same. Called directly from TestBitCast or
// via TestBitCastFrom.
template <typename ToT>
struct TestBitCastT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D d) {
    const Simd<ToT, MaxLanes(d) * sizeof(T) / sizeof(ToT)> dto;
    const auto vf = Iota(d, 1);
    const auto vt = BitCast(dto, vf);
    static_assert(sizeof(vf) == sizeof(vt), "Cast must return same size");
    // Must return the same bits
    HWY_ALIGN T from_lanes[MaxLanes(d)];
    HWY_ALIGN ToT to_lanes[MaxLanes(dto)];
    Store(vf, d, from_lanes);
    Store(vt, dto, to_lanes);
    HWY_ASSERT(BytesEqual(from_lanes, to_lanes, Lanes(d) * sizeof(T)));
  }
};

// From D to all types.
struct TestBitCastFrom {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T t, D d) {
    TestBitCastT<uint8_t>()(t, d);
    TestBitCastT<uint16_t>()(t, d);
    TestBitCastT<uint32_t>()(t, d);
#if HWY_CAP_INTEGER64
    TestBitCastT<uint64_t>()(t, d);
#endif
    TestBitCastT<int8_t>()(t, d);
    TestBitCastT<int16_t>()(t, d);
    TestBitCastT<int32_t>()(t, d);
#if HWY_CAP_INTEGER64
    TestBitCastT<int64_t>()(t, d);
#endif
    TestBitCastT<float>()(t, d);
#if HWY_CAP_FLOAT64
    TestBitCastT<double>()(t, d);
#endif
  }
};

HWY_NOINLINE void TestBitCast() {
  // For HWY_SCALAR and partial vectors, we can only cast to same-sized types:
  // the former can't partition its single lane, and the latter can be smaller
  // than a destination type.
  const ForPartialVectors<TestBitCastT<uint8_t>> to_u8;
  to_u8(uint8_t());
  to_u8(int8_t());

  const ForPartialVectors<TestBitCastT<int8_t>> to_i8;
  to_i8(uint8_t());
  to_i8(int8_t());

  const ForPartialVectors<TestBitCastT<uint16_t>> to_u16;
  to_u16(uint16_t());
  to_u16(int16_t());

  const ForPartialVectors<TestBitCastT<int16_t>> to_i16;
  to_i16(uint16_t());
  to_i16(int16_t());

  const ForPartialVectors<TestBitCastT<uint32_t>> to_u32;
  to_u32(uint32_t());
  to_u32(int32_t());
  to_u32(float());

  const ForPartialVectors<TestBitCastT<int32_t>> to_i32;
  to_i32(uint32_t());
  to_i32(int32_t());
  to_i32(float());

#if HWY_CAP_INTEGER64
  const ForPartialVectors<TestBitCastT<uint64_t>> to_u64;
  to_u64(uint64_t());
  to_u64(int64_t());
#if HWY_CAP_FLOAT64
  to_u64(double());
#endif

  const ForPartialVectors<TestBitCastT<int64_t>> to_i64;
  to_i64(uint64_t());
  to_i64(int64_t());
#if HWY_CAP_FLOAT64
  to_i64(double());
#endif
#endif  // HWY_CAP_INTEGER64

  const ForPartialVectors<TestBitCastT<float>> to_float;
  to_float(uint32_t());
  to_float(int32_t());
  to_float(float());

#if HWY_CAP_FLOAT64
  const ForPartialVectors<TestBitCastT<double>> to_double;
  to_double(double());
#if HWY_CAP_INTEGER64
  to_double(uint64_t());
  to_double(int64_t());
#endif  // HWY_CAP_INTEGER64
#endif  // HWY_CAP_FLOAT64

  // For non-scalar vectors, we can cast all types to all.
  ForAllTypes(ForGE128Vectors<TestBitCastFrom>());
}

template <typename ToT>
struct TestPromoteToT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D from_d) {
    const Simd<ToT, MaxLanes(from_d)> to_d;

    const auto from_p1 = Iota(from_d, 1);
    const auto from_n1 = Set(from_d, T(-1));
    const auto from_min = Set(from_d, LimitsMin<T>());
    const auto from_max = Set(from_d, LimitsMax<T>());
    const auto to_p1 = Iota(to_d, ToT(1));
    const auto to_n1 = Set(to_d, ToT(T(-1)));
    const auto to_min = Set(to_d, ToT(LimitsMin<T>()));
    const auto to_max = Set(to_d, ToT(LimitsMax<T>()));
    HWY_ASSERT_VEC_EQ(to_d, to_p1, PromoteTo(to_d, from_p1));
    HWY_ASSERT_VEC_EQ(to_d, to_n1, PromoteTo(to_d, from_n1));
    HWY_ASSERT_VEC_EQ(to_d, to_min, PromoteTo(to_d, from_min));
    HWY_ASSERT_VEC_EQ(to_d, to_max, PromoteTo(to_d, from_max));
  }
};

HWY_NOINLINE void TestPromote() {
  const ForPartialVectors<TestPromoteToT<uint16_t>, 2> to_u16div2;
  to_u16div2(uint8_t());

  const ForPartialVectors<TestPromoteToT<uint32_t>, 4> to_u32div4;
  to_u32div4(uint8_t());

  const ForPartialVectors<TestPromoteToT<uint32_t>, 2> to_u32div2;
  to_u32div2(uint16_t());

  const ForPartialVectors<TestPromoteToT<int16_t>, 2> to_i16div2;
  to_i16div2(uint8_t());
  to_i16div2(int8_t());

  const ForPartialVectors<TestPromoteToT<int32_t>, 2> to_i32div2;
  to_i32div2(uint16_t());
  to_i32div2(int16_t());

  const ForPartialVectors<TestPromoteToT<int32_t>, 4> to_i32div4;
  to_i32div4(uint8_t());
  to_i32div4(int8_t());

#if HWY_CAP_INTEGER64
  const ForPartialVectors<TestPromoteToT<uint64_t>, 2> to_u64div2;
  to_u64div2(uint32_t());

  const ForPartialVectors<TestPromoteToT<int64_t>, 2> to_i64div2;
  to_i64div2(int32_t());
#endif
#if HWY_CAP_FLOAT64
  const ForPartialVectors<TestPromoteToT<double>, 2> to_f64div2;
  to_f64div2(float());
#endif
}

template <typename ToT>
struct TestDemoteToT {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, D from_d) {
    const Simd<ToT, MaxLanes(from_d)> to_d;

    const auto from = Iota(from_d, 1);
    const auto from_n1 = Set(from_d, T(ToT(-1)));
    const auto from_min = Set(from_d, T(LimitsMin<ToT>()));
    const auto from_max = Set(from_d, T(LimitsMax<ToT>()));
    const auto to = Iota(to_d, ToT(1));
    const auto to_n1 = Set(to_d, ToT(-1));
    const auto to_min = Set(to_d, LimitsMin<ToT>());
    const auto to_max = Set(to_d, LimitsMax<ToT>());
    HWY_ASSERT_VEC_EQ(to_d, to, DemoteTo(to_d, from));
    HWY_ASSERT_VEC_EQ(to_d, to_n1, DemoteTo(to_d, from_n1));
    HWY_ASSERT_VEC_EQ(to_d, to_min, DemoteTo(to_d, from_min));
    HWY_ASSERT_VEC_EQ(to_d, to_max, DemoteTo(to_d, from_max));
  }
};

HWY_NOINLINE void TestDemoteTo() {
  const ForPartialVectors<TestDemoteToT<uint8_t>> to_u8;
  to_u8(int16_t());
  to_u8(int32_t());

  const ForPartialVectors<TestDemoteToT<int8_t>> to_i8;
  to_i8(int16_t());
  to_i8(int32_t());

  const ForPartialVectors<TestDemoteToT<int16_t>> to_i16;
  to_i16(int32_t());

  const ForPartialVectors<TestDemoteToT<uint16_t>> to_u16;
  to_u16(int32_t());

#if HWY_CAP_FLOAT64
  const ForPartialVectors<TestDemoteToT<float>> to_float;
  to_float(double());
#endif
}

struct TestConvertU8 {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, const D du32) {
    const Simd<uint8_t, MaxLanes(du32) * sizeof(uint32_t)> du8;
    HWY_ALIGN uint8_t lanes8[MaxLanes(du8)];
    Store(Iota(du8, 0), du8, lanes8);
    HWY_ASSERT_VEC_EQ(du32, Iota(du32, 0), U32FromU8(LoadDup128(du8, lanes8)));
    Store(Iota(du8, 0x7F), du8, lanes8);
    HWY_ASSERT_VEC_EQ(du32, Iota(du32, 0x7F),
                      U32FromU8(LoadDup128(du8, lanes8)));
    const HWY_CAPPED(uint8_t, MaxLanes(du32)) p8;
    HWY_ASSERT_VEC_EQ(p8, Iota(p8, 0), U8FromU32(Iota(du32, 0)));
    HWY_ASSERT_VEC_EQ(p8, Iota(p8, 0x7F), U8FromU32(Iota(du32, 0x7F)));
  }
};

struct TestIntFromFloat {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, const D df) {
    using TI = MakeSigned<T>;
    const Simd<TI, MaxLanes(df)> di;
    // Integer positive
    HWY_ASSERT_VEC_EQ(di, Iota(di, TI(4)), ConvertTo(di, Iota(df, T(4.0))));

    // Integer negative
    HWY_ASSERT_VEC_EQ(di, Iota(di, TI(-32)), ConvertTo(di, Iota(df, T(-32.0))));

    // Above positive
    HWY_ASSERT_VEC_EQ(di, Iota(di, TI(2)), ConvertTo(di, Iota(df, T(2.001))));

    // Below positive
    HWY_ASSERT_VEC_EQ(di, Iota(di, TI(3)), ConvertTo(di, Iota(df, T(3.9999))));

    // Above negative
    HWY_ASSERT_VEC_EQ(di, Iota(di, TI(-23)),
                      ConvertTo(di, Iota(df, T(-23.9999))));

    // Below negative
    HWY_ASSERT_VEC_EQ(di, Iota(di, TI(-24)),
                      ConvertTo(di, Iota(df, T(-24.001))));
  }
};

struct TestFloatFromInt {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, const D di) {
    using TF = MakeFloat<T>;
    const Simd<TF, MaxLanes(di)> df;

    // Integer positive
    HWY_ASSERT_VEC_EQ(df, Iota(df, TF(4.0)), ConvertTo(df, Iota(di, T(4))));

    // Integer negative
    HWY_ASSERT_VEC_EQ(df, Iota(df, TF(-32.0)), ConvertTo(df, Iota(di, T(-32))));

    // Above positive
    HWY_ASSERT_VEC_EQ(df, Iota(df, TF(2.0)), ConvertTo(df, Iota(di, T(2))));

    // Below positive
    HWY_ASSERT_VEC_EQ(df, Iota(df, TF(4.0)), ConvertTo(df, Iota(di, T(4))));

    // Above negative
    HWY_ASSERT_VEC_EQ(df, Iota(df, TF(-4.0)), ConvertTo(df, Iota(di, T(-4))));

    // Below negative
    HWY_ASSERT_VEC_EQ(df, Iota(df, TF(-2.0)), ConvertTo(df, Iota(di, T(-2))));
  }
};

HWY_NOINLINE void TestConvertFloatInt() {
  ForPartialVectors<TestIntFromFloat>()(float());
  ForPartialVectors<TestFloatFromInt>()(int32_t());
#if HWY_CAP_FLOAT64 && HWY_CAP_INTEGER64
  ForPartialVectors<TestIntFromFloat>()(double());
  ForPartialVectors<TestFloatFromInt>()(int64_t());
#endif
}

struct TestNearestInt {
  template <typename T, class D>
  HWY_NOINLINE void operator()(T /*unused*/, const D di) {
    const Simd<float, MaxLanes(di)> df;

    // Integer positive
    HWY_ASSERT_VEC_EQ(di, Iota(di, 4), NearestInt(Iota(df, 4.0f)));

    // Integer negative
    HWY_ASSERT_VEC_EQ(di, Iota(di, -32), NearestInt(Iota(df, -32.0f)));

    // Above positive
    HWY_ASSERT_VEC_EQ(di, Iota(di, 2), NearestInt(Iota(df, 2.001f)));

    // Below positive
    HWY_ASSERT_VEC_EQ(di, Iota(di, 4), NearestInt(Iota(df, 3.9999f)));

    // Above negative
    HWY_ASSERT_VEC_EQ(di, Iota(di, -24), NearestInt(Iota(df, -23.9999f)));

    // Below negative
    HWY_ASSERT_VEC_EQ(di, Iota(di, -24), NearestInt(Iota(df, -24.001f)));
  }
};

HWY_NOINLINE void TestAllConvertU8() {
  ForGE128Vectors<TestConvertU8>()(uint32_t());
}

HWY_NOINLINE void TestAllNearestInt() {
  ForPartialVectors<TestNearestInt>()(int32_t());
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {

class HwyConvertTest : public hwy::TestWithParamTarget {};

HWY_TARGET_INSTANTIATE_TEST_SUITE_P(HwyConvertTest);

HWY_EXPORT_AND_TEST_P(HwyConvertTest, TestBitCast);
HWY_EXPORT_AND_TEST_P(HwyConvertTest, TestPromote);
HWY_EXPORT_AND_TEST_P(HwyConvertTest, TestDemoteTo);
HWY_EXPORT_AND_TEST_P(HwyConvertTest, TestAllConvertU8);
HWY_EXPORT_AND_TEST_P(HwyConvertTest, TestConvertFloatInt);
HWY_EXPORT_AND_TEST_P(HwyConvertTest, TestAllNearestInt);

}  // namespace hwy
#endif
