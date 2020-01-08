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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/convert_test.cc"

#include "hwy/tests/test_util.h"
struct ConvertTest {
  HWY_DECLARE(void, ())
};
TEST(HwyConvertTest, Run) { hwy::RunTests<ConvertTest>(); }

#endif  // HWY_TARGET_INCLUDE
#include "hwy/tests/test_target_util.h"

namespace hwy {
namespace HWY_NAMESPACE {
namespace {

constexpr HWY_FULL(uint8_t) du8;
constexpr HWY_FULL(uint32_t) du32;

#if HWY_BITS != 0 || HWY_IDE
constexpr HWY_FULL(uint16_t) du16;
constexpr HWY_FULL(uint64_t) du64;
constexpr HWY_FULL(int8_t) di8;
constexpr HWY_FULL(int16_t) di16;
constexpr HWY_FULL(int32_t) di32;
constexpr HWY_FULL(int64_t) di64;
constexpr HWY_FULL(float) df;
#endif

template <typename FromT, typename ToT>
HWY_NOINLINE HWY_ATTR void TestCastFromTo() {
  const HWY_FULL(FromT) dfrom;
  const HWY_FULL(ToT) dto;
  const auto vf = Iota(dfrom, 1);
  const auto vt = BitCast(dto, vf);
  static_assert(sizeof(vf) == sizeof(vt), "Cast must return same size");
  // Must return the same bits
  HWY_ALIGN FromT from_lanes[dfrom.N];
  HWY_ALIGN ToT to_lanes[dto.N];
  Store(vf, dfrom, from_lanes);
  Store(vt, dto, to_lanes);
  HWY_ASSERT_EQ(true, BytesEqual(from_lanes, to_lanes, sizeof(vf)));
}

// From D to all types.
template <class D>
HWY_NOINLINE HWY_ATTR void TestCastFrom(D /*d*/) {
  using FromT = typename D::T;
  TestCastFromTo<FromT, uint8_t>();
  TestCastFromTo<FromT, uint16_t>();
  TestCastFromTo<FromT, uint32_t>();
#if HWY_HAS_INT64
  TestCastFromTo<FromT, uint64_t>();
#endif
  TestCastFromTo<FromT, int8_t>();
  TestCastFromTo<FromT, int16_t>();
  TestCastFromTo<FromT, int32_t>();
#if HWY_HAS_INT64
  TestCastFromTo<FromT, int64_t>();
#endif
  TestCastFromTo<FromT, float>();
#if HWY_HAS_DOUBLE
  TestCastFromTo<FromT, double>();
#endif
}

HWY_NOINLINE HWY_ATTR void TestCast() {
#if HWY_BITS == 0 || HWY_IDE
  // Promotion is undefined => only test same-sized types.
  TestCastFromTo<uint8_t, uint8_t>();
  TestCastFromTo<int8_t, int8_t>();
  TestCastFromTo<uint8_t, int8_t>();
  TestCastFromTo<int8_t, uint8_t>();

  TestCastFromTo<uint16_t, uint16_t>();
  TestCastFromTo<int16_t, int16_t>();
  TestCastFromTo<uint16_t, int16_t>();
  TestCastFromTo<int16_t, uint16_t>();

  TestCastFromTo<uint32_t, uint32_t>();
  TestCastFromTo<int32_t, int32_t>();
  TestCastFromTo<uint32_t, int32_t>();
  TestCastFromTo<int32_t, uint32_t>();
  TestCastFromTo<uint32_t, float>();
  TestCastFromTo<int32_t, float>();
  TestCastFromTo<float, float>();
  TestCastFromTo<float, uint32_t>();
  TestCastFromTo<float, int32_t>();

#if HWY_HAS_INT64
  TestCastFromTo<uint64_t, uint64_t>();
  TestCastFromTo<int64_t, int64_t>();
  TestCastFromTo<uint64_t, int64_t>();
  TestCastFromTo<int64_t, uint64_t>();
#endif
#if HWY_HAS_DOUBLE
  TestCastFromTo<uint64_t, double>();
  TestCastFromTo<int64_t, double>();
  TestCastFromTo<double, double>();
  TestCastFromTo<double, uint64_t>();
  TestCastFromTo<double, int64_t>();
#endif
#else
  HWY_FOREACH_U(TestCastFrom);
  HWY_FOREACH_I(TestCastFrom);
#endif

  // Float <-> u/i32
  TestCastFromTo<uint32_t, float>();
  TestCastFromTo<int32_t, float>();
  TestCastFromTo<float, uint32_t>();
  TestCastFromTo<float, int32_t>();
}

template <typename FromT, typename ToT>
HWY_NOINLINE HWY_ATTR void TestPromoteT() {
  constexpr size_t N = HWY_LANES_OR_0(ToT);
  const Desc<FromT, N> from_d;
  const Desc<ToT, N> to_d;

  const auto from = Iota(from_d, 1);
  const auto from_n1 = Set(from_d, FromT(-1));
  const auto from_min = Set(from_d, LimitsMin<FromT>());
  const auto from_max = Set(from_d, LimitsMax<FromT>());
  const auto to = Iota(to_d, 1);
  const auto to_n1 = Set(to_d, ToT(FromT(-1)));
  const auto to_min = Set(to_d, ToT(LimitsMin<FromT>()));
  const auto to_max = Set(to_d, ToT(LimitsMax<FromT>()));
  HWY_ASSERT_VEC_EQ(to_d, to, ConvertTo(to_d, from));
  HWY_ASSERT_VEC_EQ(to_d, to_n1, ConvertTo(to_d, from_n1));
  HWY_ASSERT_VEC_EQ(to_d, to_min, ConvertTo(to_d, from_min));
  HWY_ASSERT_VEC_EQ(to_d, to_max, ConvertTo(to_d, from_max));
}

template <typename FromT, typename ToT>
HWY_NOINLINE HWY_ATTR void TestDemoteT() {
  constexpr size_t N = HWY_LANES_OR_0(FromT);
  const Desc<FromT, N> from_d;
  const Desc<ToT, N> to_d;

  const auto from = Iota(from_d, 1);
  const auto from_n1 = Set(from_d, FromT(ToT(-1)));
  const auto from_min = Set(from_d, FromT(LimitsMin<ToT>()));
  const auto from_max = Set(from_d, FromT(LimitsMax<ToT>()));
  const auto to = Iota(to_d, 1);
  const auto to_n1 = Set(to_d, ToT(-1));
  const auto to_min = Set(to_d, LimitsMin<ToT>());
  const auto to_max = Set(to_d, LimitsMax<ToT>());
  HWY_ASSERT_VEC_EQ(to_d, to, ConvertTo(to_d, from));
  HWY_ASSERT_VEC_EQ(to_d, to_n1, ConvertTo(to_d, from_n1));
  HWY_ASSERT_VEC_EQ(to_d, to_min, ConvertTo(to_d, from_min));
  HWY_ASSERT_VEC_EQ(to_d, to_max, ConvertTo(to_d, from_max));
}

template <typename FromT, typename ToT>
HWY_NOINLINE HWY_ATTR void TestDupPromoteT() {
  constexpr size_t N = HWY_LANES_OR_0(ToT);
  const Desc<FromT, N> from_d;
  const Desc<ToT, N> to_d;

  const auto from = Iota(from_d, 1);
  const auto from_n1 = Set(from_d, FromT(-1));
  const auto from_min = Set(from_d, LimitsMin<FromT>());
  const auto from_max = Set(from_d, LimitsMax<FromT>());
  const auto to = Iota(to_d, 1);
  const auto to_n1 = Set(to_d, ToT(FromT(-1)));
  const auto to_min = Set(to_d, ToT(LimitsMin<FromT>()));
  const auto to_max = Set(to_d, ToT(LimitsMax<FromT>()));
  HWY_ASSERT_VEC_EQ(to_d, to, ConvertTo(to_d, from));
  HWY_ASSERT_VEC_EQ(to_d, to_n1, ConvertTo(to_d, from_n1));
  HWY_ASSERT_VEC_EQ(to_d, to_min, ConvertTo(to_d, from_min));
  HWY_ASSERT_VEC_EQ(to_d, to_max, ConvertTo(to_d, from_max));
}

HWY_NOINLINE HWY_ATTR void TestConvert() {
#if HWY_BITS != 0 || HWY_IDE
  (void)di64;
  (void)du64;
#endif

  TestCast();

  HWY_ALIGN uint8_t lanes8[du8.N];
  Store(Iota(du8, 0), du8, lanes8);
  HWY_ASSERT_VEC_EQ(du32, Iota(du32, 0), U32FromU8(LoadDup128(du8, lanes8)));
  Store(Iota(du8, 0x7F), du8, lanes8);
  HWY_ASSERT_VEC_EQ(du32, Iota(du32, 0x7F), U32FromU8(LoadDup128(du8, lanes8)));
  const HWY_CAPPED(uint8_t, du32.N) p8;
  HWY_ASSERT_VEC_EQ(p8, Iota(p8, 0), U8FromU32(Iota(du32, 0)));
  HWY_ASSERT_VEC_EQ(p8, Iota(p8, 0x7F), U8FromU32(Iota(du32, 0x7F)));

  // Promote: no u64,i64
#if HWY_HAS_DOUBLE
  TestPromoteT<float, double>();
#endif
  TestPromoteT<uint8_t, int16_t>();
  TestPromoteT<uint8_t, int32_t>();
  TestPromoteT<uint16_t, int32_t>();
  TestPromoteT<int8_t, int16_t>();
  TestPromoteT<int8_t, int32_t>();
  TestPromoteT<int16_t, int32_t>();
#if HWY_HAS_INT64
  TestPromoteT<uint32_t, uint64_t>();
  TestPromoteT<int32_t, int64_t>();
#endif

  // Demote
  TestDemoteT<int16_t, int8_t>();
  TestDemoteT<int32_t, int8_t>();
  TestDemoteT<int32_t, int16_t>();
  TestDemoteT<int16_t, uint8_t>();
  TestDemoteT<int32_t, uint8_t>();
  TestDemoteT<int32_t, uint16_t>();

  TestDupPromoteT<uint8_t, uint32_t>();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy

// Instantiate for the current target.
void ConvertTest::HWY_FUNC() { hwy::HWY_NAMESPACE::TestConvert(); }
