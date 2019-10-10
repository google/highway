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

#include "highway/logical_test.h"
#include "highway/test_target_util.h"

namespace jxl {
namespace SIMD_NAMESPACE {
namespace {

struct TestLogicalT {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    const auto v0 = setzero(d);
    const auto vi = iota(d, 0);

    SIMD_ASSERT_VEC_EQ(d, v0, v0 & vi);
    SIMD_ASSERT_VEC_EQ(d, v0, vi & v0);
    SIMD_ASSERT_VEC_EQ(d, vi, vi & vi);

    SIMD_ASSERT_VEC_EQ(d, vi, v0 | vi);
    SIMD_ASSERT_VEC_EQ(d, vi, vi | v0);
    SIMD_ASSERT_VEC_EQ(d, vi, vi | vi);

    SIMD_ASSERT_VEC_EQ(d, vi, v0 ^ vi);
    SIMD_ASSERT_VEC_EQ(d, vi, vi ^ v0);
    SIMD_ASSERT_VEC_EQ(d, v0, vi ^ vi);

    SIMD_ASSERT_VEC_EQ(d, vi, andnot(v0, vi));
    SIMD_ASSERT_VEC_EQ(d, v0, andnot(vi, v0));
    SIMD_ASSERT_VEC_EQ(d, v0, andnot(vi, vi));

    auto v = vi;
    v &= vi;
    SIMD_ASSERT_VEC_EQ(d, vi, v);
    v &= v0;
    SIMD_ASSERT_VEC_EQ(d, v0, v);

    v |= vi;
    SIMD_ASSERT_VEC_EQ(d, vi, v);
    v |= v0;
    SIMD_ASSERT_VEC_EQ(d, vi, v);

    v ^= vi;
    SIMD_ASSERT_VEC_EQ(d, v0, v);
    v ^= v0;
    SIMD_ASSERT_VEC_EQ(d, v0, v);
  }
};

struct TestIfThenElse {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    RandomState rng{1234};
    const T mask0(0);
    const uint64_t ones = ~0ull;
    T mask1;
    CopyBytes<sizeof(T)>(&ones, &mask1);

    SIMD_ALIGN T lanes1[d.N] = {};  // Initialized for clang-analyzer.
    SIMD_ALIGN T lanes2[d.N] = {};  // Initialized for clang-analyzer.
    SIMD_ALIGN T masks[d.N] = {};   // Initialized for clang-analyzer.
    for (size_t i = 0; i < d.N; ++i) {
      lanes1[i] = int32_t(Random32(&rng));
      lanes2[i] = int32_t(Random32(&rng));
      masks[i] = (Random32(&rng) & 1024) ? mask0 : mask1;
    }

    // out := mask ? lanes2 : lanes1
    SIMD_ALIGN T out_lanes[d.N];
    store(if_then_else(load(d, masks), load(d, lanes2), load(d, lanes1)), d,
          out_lanes);
    for (size_t i = 0; i < d.N; ++i) {
      SIMD_ASSERT_EQ((masks[i] == mask0) ? lanes1[i] : lanes2[i], out_lanes[i]);
    }
  }
};

struct TestIfThenElseSign {
  template <typename T, class D>
  SIMD_ATTR void operator()(T, D d) const {
    RandomState rng{1234};

    SIMD_ALIGN T lanes1[d.N];
    SIMD_ALIGN T lanes2[d.N];
    SIMD_ALIGN T masks[d.N];
    for (size_t i = 0; i < d.N; ++i) {
      lanes1[i] = Random32(&rng);
      lanes2[i] = Random32(&rng);
      masks[i] = (Random32(&rng) & 1024) ? lanes1[i] : -lanes2[i];
    }

    // out := mask < 0 ? lanes2 : lanes1
    SIMD_ALIGN T out_lanes[d.N];
    const auto mask = mask_from_sign(load(d, masks));
    store(if_then_else(mask, load(d, lanes2), load(d, lanes1)), d, out_lanes);
    for (size_t i = 0; i < d.N; ++i) {
      SIMD_ASSERT_EQ((masks[i] < T(0.0)) ? lanes2[i] : lanes1[i], out_lanes[i]);
    }
  }
};

SIMD_ATTR void TestLogical() {
  ForeachLaneType<TestLogicalT>();
  ForeachLaneType<TestIfThenElse>();
  ForeachFloatLaneType<TestIfThenElseSign>();
}

}  // namespace
}  // namespace SIMD_NAMESPACE  // NOLINT(google-readability-namespace-comments)
}  // namespace jxl

// Instantiate for the current target.
template <>
void LogicalTest::operator()<jxl::SIMD_TARGET>() {
  jxl::SIMD_NAMESPACE::TestLogical();
}
