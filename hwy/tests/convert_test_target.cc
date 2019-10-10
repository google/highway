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

#include "highway/convert_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

struct TestCastT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    Test<uint8_t, T>();
    Test<uint16_t, T>();
    Test<uint32_t, T>();
    Test<uint64_t, T>();
    Test<int8_t, T>();
    Test<int16_t, T>();
    Test<int32_t, T>();
    Test<int64_t, T>();
    Test<float, T>();
    Test<double, T>();
  }

  template <typename FromT, typename ToT>
  SIMD_ATTR void Test() const {
    const SIMD_FULL(FromT) df;
    const SIMD_FULL(ToT) dt;
    const auto vf = iota(df, 1);
    const auto vt = bit_cast(dt, vf);
    static_assert(sizeof(vf) == sizeof(vt), "Cast must return same size");
    // Must return the same bits
    SIMD_ALIGN FromT from_lanes[df.N];
    SIMD_ALIGN ToT to_lanes[dt.N];
    store(vf, df, from_lanes);
    store(vt, dt, to_lanes);
    SIMD_ASSERT_EQ(true, BytesEqual(from_lanes, to_lanes, sizeof(vf)));
  }
};

SIMD_ATTR void TestCast() {
#if SIMD_BITS == 0
  // Promotion is undefined => only test same-sized types.
  TestCastT().Test<uint8_t, uint8_t>();
  TestCastT().Test<int8_t, int8_t>();
  TestCastT().Test<uint8_t, int8_t>();
  TestCastT().Test<int8_t, uint8_t>();

  TestCastT().Test<uint16_t, uint16_t>();
  TestCastT().Test<int16_t, int16_t>();
  TestCastT().Test<uint16_t, int16_t>();
  TestCastT().Test<int16_t, uint16_t>();

  TestCastT().Test<uint32_t, uint32_t>();
  TestCastT().Test<int32_t, int32_t>();
  TestCastT().Test<uint32_t, int32_t>();
  TestCastT().Test<int32_t, uint32_t>();
  TestCastT().Test<uint32_t, float>();
  TestCastT().Test<int32_t, float>();
  TestCastT().Test<float, float>();
  TestCastT().Test<float, uint32_t>();
  TestCastT().Test<float, int32_t>();

  TestCastT().Test<uint64_t, uint64_t>();
  TestCastT().Test<int64_t, int64_t>();
  TestCastT().Test<uint64_t, int64_t>();
  TestCastT().Test<int64_t, uint64_t>();
  TestCastT().Test<uint64_t, double>();
  TestCastT().Test<int64_t, double>();
  TestCastT().Test<double, double>();
  TestCastT().Test<double, uint64_t>();
  TestCastT().Test<double, int64_t>();
#else
  ForeachUnsignedLaneType<TestCastT>();
  ForeachSignedLaneType<TestCastT>();
#endif

  // Float <-> u/i32
  TestCastT().Test<uint32_t, float>();
  TestCastT().Test<int32_t, float>();
  TestCastT().Test<float, uint32_t>();
  TestCastT().Test<float, int32_t>();
}

template <typename FromT, typename ToT>
SIMD_ATTR void TestPromoteT() {
  constexpr size_t N = SIMD_LANES_OR_0(ToT);
  const Desc<FromT, N> from_d;
  const Desc<ToT, N> to_d;

  const auto from = iota(from_d, 1);
  const auto from_n1 = set1(from_d, FromT(-1));
  const auto from_min = set1(from_d, LimitsMin<FromT>());
  const auto from_max = set1(from_d, LimitsMax<FromT>());
  const auto to = iota(to_d, 1);
  const auto to_n1 = set1(to_d, ToT(FromT(-1)));
  const auto to_min = set1(to_d, ToT(LimitsMin<FromT>()));
  const auto to_max = set1(to_d, ToT(LimitsMax<FromT>()));
  SIMD_ASSERT_VEC_EQ(to_d, to, convert_to(to_d, from));
  SIMD_ASSERT_VEC_EQ(to_d, to_n1, convert_to(to_d, from_n1));
  SIMD_ASSERT_VEC_EQ(to_d, to_min, convert_to(to_d, from_min));
  SIMD_ASSERT_VEC_EQ(to_d, to_max, convert_to(to_d, from_max));
}

template <typename FromT, typename ToT>
SIMD_ATTR void TestDemoteT() {
  constexpr size_t N = SIMD_LANES_OR_0(FromT);
  const Desc<FromT, N> from_d;
  const Desc<ToT, N> to_d;

  const auto from = iota(from_d, 1);
  const auto from_n1 = set1(from_d, FromT(ToT(-1)));
  const auto from_min = set1(from_d, FromT(LimitsMin<ToT>()));
  const auto from_max = set1(from_d, FromT(LimitsMax<ToT>()));
  const auto to = iota(to_d, 1);
  const auto to_n1 = set1(to_d, ToT(-1));
  const auto to_min = set1(to_d, LimitsMin<ToT>());
  const auto to_max = set1(to_d, LimitsMax<ToT>());
  SIMD_ASSERT_VEC_EQ(to_d, to, convert_to(to_d, from));
  SIMD_ASSERT_VEC_EQ(to_d, to_n1, convert_to(to_d, from_n1));
  SIMD_ASSERT_VEC_EQ(to_d, to_min, convert_to(to_d, from_min));
  SIMD_ASSERT_VEC_EQ(to_d, to_max, convert_to(to_d, from_max));
}

template <typename FromT, typename ToT>
SIMD_ATTR void TestDupPromoteT() {
  constexpr size_t N = SIMD_LANES_OR_0(ToT);
  const Desc<FromT, N> from_d;
  const Desc<ToT, N> to_d;

  const auto from = iota(from_d, 1);
  const auto from_n1 = set1(from_d, FromT(-1));
  const auto from_min = set1(from_d, LimitsMin<FromT>());
  const auto from_max = set1(from_d, LimitsMax<FromT>());
  const auto to = iota(to_d, 1);
  const auto to_n1 = set1(to_d, ToT(FromT(-1)));
  const auto to_min = set1(to_d, ToT(LimitsMin<FromT>()));
  const auto to_max = set1(to_d, ToT(LimitsMax<FromT>()));
  SIMD_ASSERT_VEC_EQ(to_d, to, convert_to(to_d, from));
  SIMD_ASSERT_VEC_EQ(to_d, to_n1, convert_to(to_d, from_n1));
  SIMD_ASSERT_VEC_EQ(to_d, to_min, convert_to(to_d, from_min));
  SIMD_ASSERT_VEC_EQ(to_d, to_max, convert_to(to_d, from_max));
}

SIMD_ATTR void TestConvert() {
  TestCast();

  const SIMD_FULL(uint8_t) d8;
  const SIMD_FULL(uint32_t) d32;
  SIMD_ALIGN uint8_t lanes8[d8.N];
  store(iota(d8, 0), d8, lanes8);
  SIMD_ASSERT_VEC_EQ(d32, iota(d32, 0), u32_from_u8(load_dup128(d8, lanes8)));
  store(iota(d8, 0x7F), d8, lanes8);
  SIMD_ASSERT_VEC_EQ(d32, iota(d32, 0x7F),
                     u32_from_u8(load_dup128(d8, lanes8)));
  const SIMD_CAPPED(uint8_t, d32.N) p8;
  SIMD_ASSERT_VEC_EQ(p8, iota(p8, 0), u8_from_u32(iota(d32, 0)));
  SIMD_ASSERT_VEC_EQ(p8, iota(p8, 0x7F), u8_from_u32(iota(d32, 0x7F)));

  // Promote: no u64,i64
  TestPromoteT<float, double>();
  TestPromoteT<uint8_t, int16_t>();
  TestPromoteT<uint8_t, int32_t>();
  TestPromoteT<uint16_t, int32_t>();
  TestPromoteT<int8_t, int16_t>();
  TestPromoteT<int8_t, int32_t>();
  TestPromoteT<int16_t, int32_t>();
  TestPromoteT<uint32_t, uint64_t>();
  TestPromoteT<int32_t, int64_t>();

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
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void ConvertTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestConvert();
}
