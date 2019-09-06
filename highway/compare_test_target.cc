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

#include "highway/compare_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

struct TestSignedCompare {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v2 = iota(d, 2);
    const auto v2b = iota(d, 2);
    const auto vn = iota(d, -T(d.N));
    const auto yes = set1(d, static_cast<T>(-1));
    const auto no = setzero(d);

    SIMD_ASSERT_VEC_EQ(d, no, v2 == vn);
    SIMD_ASSERT_VEC_EQ(d, yes, v2 == v2b);

    SIMD_ASSERT_VEC_EQ(d, yes, v2 > vn);
    SIMD_ASSERT_VEC_EQ(d, yes, vn < v2);
    SIMD_ASSERT_VEC_EQ(d, no, v2 < vn);
    SIMD_ASSERT_VEC_EQ(d, no, vn > v2);
  }
};

struct TestUnsignedCompare {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v2 = iota(d, 2);
    const auto v2b = iota(d, 2);
    const auto v3 = iota(d, 3);
    const auto yes = set1(d, T(~0ull));
    const auto no = setzero(d);

    SIMD_ASSERT_VEC_EQ(d, no, v2 == v3);
    SIMD_ASSERT_VEC_EQ(d, yes, v2 == v2b);
  }
};

struct TestFloatCompare {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    constexpr size_t N8 = SIMD_FULL(uint8_t)::N;
    const auto v2 = iota(d, 2);
    const auto v2b = iota(d, 2);
    const auto vn = iota(d, -T(d.N));
    const auto no = setzero(d);

    SIMD_ASSERT_VEC_EQ(d, no, v2 == vn);
    SIMD_ASSERT_VEC_EQ(d, no, v2 < vn);
    SIMD_ASSERT_VEC_EQ(d, no, vn > v2);

    // Equality is represented as 1-bits, which is a NaN, so compare bytes.
    uint8_t yes[N8];
    SetBytes(0xFF, &yes);

    SIMD_ALIGN T lanes[d.N];
    store(v2 == v2b, d, lanes);
    SIMD_ASSERT_EQ(true, BytesEqual(lanes, yes, N8));
    store(v2 > vn, d, lanes);
    SIMD_ASSERT_EQ(true, BytesEqual(lanes, yes, N8));
    store(vn < v2, d, lanes);
    SIMD_ASSERT_EQ(true, BytesEqual(lanes, yes, N8));
  }
};

// Returns "bits" after zeroing any upper bits that wouldn't be returned by
// movemask for the given vector "D".
template <class D>
uint64_t ValidBits(D d, const uint64_t bits) {
  const size_t shift = 64 - d.N;  // 0..63 - avoids UB for d.N == 64.
  return (bits << shift) >> shift;
}

SIMD_ATTR void TestMovemask() {
  const SIMD_FULL(uint8_t) d;
  SIMD_ALIGN const uint8_t bytes[kTestMaxVectorSize] = {
      0x80, 0xFF, 0x7F, 0x00, 0x01, 0x10, 0x20, 0x40, 0x80, 0x02, 0x04,
      0x08, 0xC0, 0xC1, 0xFE, 0x0F, 0x0F, 0xFE, 0xC1, 0xC0, 0x08, 0x04,
      0x02, 0x80, 0x40, 0x20, 0x10, 0x01, 0x00, 0x7F, 0xFF, 0x80, 0x0F,
      0xFE, 0xC1, 0xC0, 0x08, 0x04, 0x02, 0x80, 0x40, 0x20, 0x10, 0x01,
      0x00, 0x7F, 0xFF, 0x80, 0x80, 0xFF, 0x7F, 0x00, 0x01, 0x10, 0x20,
      0x40, 0x80, 0x02, 0x04, 0x08, 0xC0, 0xC1, 0xFE, 0x0F};
  SIMD_ASSERT_EQ(ValidBits(d, 0x7103C08EC08E7103ull),
                 ext::movemask(load(d, bytes)));

  SIMD_ALIGN const float lanes[kTestMaxVectorSize / sizeof(float)] = {
      -1.0f,  1E30f, -0.0f, 1E-30f, 0.0f,  -0.0f, 1E30f, -1.0f,
      1E-30f, -0.0f, 1E30f, -1.0f,  -1.0f, 1E30f, -0.0f, 1E-30f};
  const SIMD_FULL(float) df;
  SIMD_ASSERT_EQ(ValidBits(df, 0x5aa5), ext::movemask(load(df, lanes)));

  const SIMD_FULL(double) dd;
  SIMD_ALIGN const double lanes2[kTestMaxVectorSize / sizeof(double)] = {
      1E300, -1E-300, -0.0, 1E-10, -0.0, 0.0, 1E300, -1E-10};
  SIMD_ASSERT_EQ(ValidBits(dd, 0x96), ext::movemask(load(dd, lanes2)));
}

struct TestAllZero {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const T max = LimitsMax<T>();
    const T min_nonzero = LimitsMin<T>() + 1;

    // all lanes zero
    auto v = setzero(d);
    SIMD_ALIGN T lanes[d.N] = {};  // Initialized for clang-analyzer.
    store(v, d, lanes);

    // Set each lane to nonzero and ensure !all_zero
    for (size_t i = 0; i < d.N; ++i) {
      lanes[i] = max;
      v = load(d, lanes);
      SIMD_ASSERT_EQ(false, ext::all_zero(v));

      lanes[i] = min_nonzero;
      v = load(d, lanes);
      SIMD_ASSERT_EQ(false, ext::all_zero(v));

      // Reset to all zero
      lanes[i] = T(0);
      v = load(d, lanes);
      SIMD_ASSERT_EQ(true, ext::all_zero(v));
    }
  }
};

SIMD_ATTR void TestCompare() {
  // ForeachSignedLaneType<TestSignedCompare>();
  ForeachUnsignedLaneType<TestUnsignedCompare>();
  ForeachFloatLaneType<TestFloatCompare>();

  TestMovemask();

  ForeachUnsignedLaneType<TestAllZero>();
  ForeachSignedLaneType<TestAllZero>();
  // No float.
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void CompareTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestCompare();
}
