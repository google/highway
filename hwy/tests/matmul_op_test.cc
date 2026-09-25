// Copyright 2026 Google LLC
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

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tests/matmul_op_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

#ifndef HWY_NATIVE_PER_BLOCK_2X2_MATMUL_INT8
#error "Bug in set_macros-inl.h, did not set HWY_NATIVE_PER_BLOCK_2X2_MATMUL_INT8"
#endif

#ifndef HWY_NATIVE_PER_BLOCK_2X2_MATMUL_BF16
#error "Bug in set_macros-inl.h, did not set HWY_NATIVE_PER_BLOCK_2X2_MATMUL_BF16"
#endif

#ifndef HWY_NATIVE_TILE_64B_MATMUL_BF16
#error "Bug in set_macros-inl.h, did not set HWY_NATIVE_TILE_64B_MATMUL_BF16"
#endif

#ifndef HWY_NATIVE_TILE_64B_MATMUL_I8
#error "Bug in set_macros-inl.h, did not set HWY_NATIVE_TILE_64B_MATMUL_I8"
#endif

struct TestPerBlock2x2MatMulInt8 {
  template <typename TN, class DN>
  HWY_NOINLINE void operator()(TN /*unused*/, DN dn) {
#if HWY_NATIVE_PER_BLOCK_2X2_MATMUL_INT8
    static_assert(IsSame<TN, int32_t>(), "TN should be int32_t");
    const Repartition<int8_t, DN> di8;
    using VI8 = Vec<decltype(di8)>;
    using V32 = Vec<decltype(dn)>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(dn);

    // Allocate aligned buffers for scalar verification
    auto in_a = AllocateAligned<int8_t>(N * 4);
    auto in_b = AllocateAligned<int8_t>(N * 4);
    auto in_c = AllocateAligned<int32_t>(N);
    auto expected = AllocateAligned<int32_t>(N);
    HWY_ASSERT(in_a && in_b && in_c && expected);

    // Populate buffers
    for (size_t i = 0; i < N * 4; ++i) {
      in_a[i] = static_cast<int8_t>((i % 25) - 12);
      in_b[i] = static_cast<int8_t>((i % 19) - 9);
    }
    for (size_t i = 0; i < N; ++i) {
      in_c[i] = static_cast<int32_t>((i % 7) + 10);
      expected[i] = in_c[i];
    }

    // Scalar emulation loop (matching hardware svmmla interleaving)
    for (size_t block = 0; block < N; block += 4) {
      const size_t block_i8 = block * 4;
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          int32_t sum = 0;
          for (size_t k = 0; k < 8; ++k) {
            sum += static_cast<int32_t>(in_a[block_i8 + i * 8 + k]) *
                   static_cast<int32_t>(in_b[block_i8 + j * 8 + k]);
          }
          expected[block + i * 2 + j] += sum;
        }
      }
    }

    const VI8 va = Load(di8, in_a.get());
    const VI8 vb = Load(di8, in_b.get());
    const V32 vc = Load(dn, in_c.get());

    const V32 actual = PerBlock2x2MatMul(dn, va, vb, vc);
    HWY_ASSERT_VEC_EQ(dn, expected.get(), actual);
#else
    (void)dn;
#endif
  }
};

HWY_NOINLINE void TestAllPerBlock2x2MatMulInt8() {
  ForGEVectors<128, TestPerBlock2x2MatMulInt8>()(int32_t());
}

struct TestPerBlock2x2MatMulUint8Int8 {
  template <typename TN, class DN>
  HWY_NOINLINE void operator()(TN /*unused*/, DN dn) {
#if HWY_NATIVE_PER_BLOCK_2X2_MATMUL_INT8
    static_assert(IsSame<TN, int32_t>(), "TN should be int32_t");
    const Repartition<uint8_t, DN> du8;
    const Repartition<int8_t, DN> di8;
    using VU8 = Vec<decltype(du8)>;
    using VI8 = Vec<decltype(di8)>;
    using V32 = Vec<decltype(dn)>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(dn);

    // Allocate aligned buffers for scalar verification
    auto in_a = AllocateAligned<uint8_t>(N * 4);
    auto in_b = AllocateAligned<int8_t>(N * 4);
    auto in_c = AllocateAligned<int32_t>(N);
    auto expected = AllocateAligned<int32_t>(N);
    HWY_ASSERT(in_a && in_b && in_c && expected);

    // Populate buffers
    for (size_t i = 0; i < N * 4; ++i) {
      in_a[i] = static_cast<uint8_t>((i % 250) + 1);
      in_b[i] = static_cast<int8_t>((i % 19) - 9);
    }
    for (size_t i = 0; i < N; ++i) {
      in_c[i] = static_cast<int32_t>((i % 7) + 10);
      expected[i] = in_c[i];
    }

    // Scalar emulation loop (matching hardware svusmmla interleaving)
    for (size_t block = 0; block < N; block += 4) {
      const size_t block_u8 = block * 4;
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          int32_t sum = 0;
          for (size_t k = 0; k < 8; ++k) {
            sum += static_cast<int32_t>(in_a[block_u8 + i * 8 + k]) *
                   static_cast<int32_t>(in_b[block_u8 + j * 8 + k]);
          }
          expected[block + i * 2 + j] += sum;
        }
      }
    }

    const VU8 va = Load(du8, in_a.get());
    const VI8 vb = Load(di8, in_b.get());
    const V32 vc = Load(dn, in_c.get());

    const V32 actual = PerBlock2x2MatMul(dn, va, vb, vc);
    HWY_ASSERT_VEC_EQ(dn, expected.get(), actual);
#else
    (void)dn;
#endif
  }
};

HWY_NOINLINE void TestAllPerBlock2x2MatMulUint8Int8() {
  ForGEVectors<128, TestPerBlock2x2MatMulUint8Int8>()(int32_t());
}

struct TestPerBlock2x2MatMulBF16 {
  template <typename TN, class DN>
  HWY_NOINLINE void operator()(TN /*unused*/, DN dn) {
#if HWY_NATIVE_PER_BLOCK_2X2_MATMUL_BF16
    static_assert(IsSame<TN, float>(), "TN should be float");
    const Repartition<hwy::bfloat16_t, DN> dbf;
    using VBF = Vec<decltype(dbf)>;
    using VF = Vec<decltype(dn)>;
    HWY_LANES_CONSTEXPR size_t N = Lanes(dn);

    auto in_a = AllocateAligned<hwy::bfloat16_t>(N * 2);
    auto in_b = AllocateAligned<hwy::bfloat16_t>(N * 2);
    auto in_c = AllocateAligned<float>(N);
    auto expected = AllocateAligned<float>(N);
    HWY_ASSERT(in_a && in_b && in_c && expected);

    for (size_t i = 0; i < N * 2; ++i) {
      in_a[i] = hwy::ConvertScalarTo<hwy::bfloat16_t>(
          static_cast<float>(i % 5) * 0.5f);
      in_b[i] = hwy::ConvertScalarTo<hwy::bfloat16_t>(
          static_cast<float>(i % 7) * 0.25f);
    }
    for (size_t i = 0; i < N; ++i) {
      in_c[i] = static_cast<float>(i % 3) + 1.0f;
      expected[i] = in_c[i];
    }

    for (size_t block = 0; block < N; block += 4) {
      const size_t block_bf = block * 2;
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
          float sum = 0.0f;
          for (size_t k = 0; k < 4; ++k) {
            sum += hwy::ConvertScalarTo<float>(in_a[block_bf + i * 4 + k]) *
                   hwy::ConvertScalarTo<float>(in_b[block_bf + j * 4 + k]);
          }
          expected[block + i * 2 + j] += sum;
        }
      }
    }

    const VBF va = Load(dbf, in_a.get());
    const VBF vb = Load(dbf, in_b.get());
    const VF vc = Load(dn, in_c.get());

    const VF actual = PerBlock2x2MatMul(dn, va, vb, vc);
    HWY_ASSERT_VEC_EQ(dn, expected.get(), actual);
#else
    (void)dn;
#endif
  }
};

HWY_NOINLINE void TestAllPerBlock2x2MatMulBF16() {
  ForGEVectors<128, TestPerBlock2x2MatMulBF16>()(float());
}

struct TestTile64BMatMulBF16 {
  // AMX is only supported on x64; this template is unused on 32-bit builds.
  template <typename TN, class DN>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(TN /*unused*/, DN dn) {
#if HWY_NATIVE_TILE_64B_MATMUL_BF16
    if (!hwy::HaveTile64BMatMulBF16()) {
      return;
    }
    constexpr size_t kRows = 16;
    constexpr size_t kColsF32 = 16;
    constexpr size_t kDimStepBF16 = 32;
    constexpr size_t kRowBytes = 64;

    auto in_a = AllocateAligned<hwy::bfloat16_t>(kRows * kDimStepBF16);
    auto in_b = AllocateAligned<hwy::bfloat16_t>(kRows * kDimStepBF16);
    auto in_c = AllocateAligned<float>(kRows * kColsF32);
    auto actual = AllocateAligned<float>(kRows * kColsF32);
    auto expected = AllocateAligned<float>(kRows * kColsF32);
    HWY_ASSERT(in_a && in_b && in_c && actual && expected);

    for (size_t i = 0; i < kRows * kDimStepBF16; ++i) {
      in_a[i] = hwy::ConvertScalarTo<hwy::bfloat16_t>(
          static_cast<float>((i % 7) - 3) * 0.5f);
      in_b[i] = hwy::ConvertScalarTo<hwy::bfloat16_t>(
          static_cast<float>((i % 5) - 2) * 0.25f);
    }
    for (size_t i = 0; i < kRows * kColsF32; ++i) {
      in_c[i] = static_cast<float>(i % 11) * 0.125f;
    }

    const Repartition<hwy::bfloat16_t, DN> dbf;
    const RebindToUnsigned<DN> du32;
    using VBF = Vec<decltype(dbf)>;
    using VF = Vec<decltype(dn)>;

    for (size_t r = 0; r < kRows; ++r) {
      VF sum0 = Zero(dn);
      VF sum1 = Zero(dn);
      for (size_t k_pair = 0; k_pair < kDimStepBF16 / 2; ++k_pair) {
        uint32_t a_pair;
        CopyBytes<4>(&in_a[r * kDimStepBF16 + 2 * k_pair], &a_pair);
        const VBF va = BitCast(dbf, Set(du32, a_pair));
        const VBF vb = Load(dbf, in_b.get() + k_pair * kDimStepBF16);
        sum0 = ReorderWidenMulAccumulate(dn, va, vb, sum0, sum1);
      }
      const VF vc = Load(dn, in_c.get() + r * kColsF32);
      const VF row_expected = Add(vc, RearrangeToOddPlusEven(sum0, sum1));
      Store(row_expected, dn, expected.get() + r * kColsF32);
    }

    auto tile_c = MakeTile64B(kRows, kRowBytes);
    auto tile_a = MakeTile64B(kRows, kRowBytes);
    auto tile_b = MakeTile64B(kRows, kRowBytes);

    Tile64BLoad(&tile_c, in_c.get(), kRowBytes);
    Tile64BLoad(&tile_a, in_a.get(), kRowBytes);
    Tile64BLoad(&tile_b, in_b.get(), kRowBytes);
    Tile64BMatMul(dn, &tile_c, &tile_a, &tile_b);
    Tile64BStore(&tile_c, actual.get(), kRowBytes);
    Tile64BRelease();

    for (size_t r = 0; r < kRows; ++r) {
      HWY_ASSERT_VEC_EQ(dn, expected.get() + r * kColsF32,
                        Load(dn, actual.get() + r * kColsF32));
    }
#else
    (void)dn;
#endif
  }
};

HWY_NOINLINE void TestAllTile64BMatMulBF16() {
  // AMX is independent of vector length, hence only test 512.
  ForGEVectors<512, TestTile64BMatMulBF16>()(float());
}

struct TestTile64BMatMulI8 {
  // AMX is only supported on x64; this template is unused on 32-bit builds.
  template <typename TN, class DN>
  HWY_NOINLINE HWY_MAYBE_UNUSED void operator()(TN /*unused*/, DN dn) {
#if HWY_NATIVE_TILE_64B_MATMUL_I8
    if (!hwy::HaveTile64BMatMulI8()) {
      return;
    }
    TestOne<int8_t, int8_t>(dn);
    TestOne<int8_t, uint8_t>(dn);
    TestOne<uint8_t, int8_t>(dn);
    TestOne<uint8_t, uint8_t>(dn);
#else
    (void)dn;
#endif
  }

#if HWY_NATIVE_TILE_64B_MATMUL_I8
 private:
  // Reinterprets `bits` as TA, which also yields negative values for int8.
  // Avoids implementation-defined behavior of out-of-range static_cast.
  template <typename T>
  static T FromBits(uint8_t bits) {
    T t;
    CopyBytes<1>(&bits, &t);
    return t;
  }

  // TA/TB are the lane types of the A/B tiles; the C tile is always int32.
  template <typename TA, typename TB, class DN>
  static void TestOne(DN dn) {
    constexpr size_t kRows = 16;
    constexpr size_t kColsI32 = 16;
    constexpr size_t kDimStepI8 = 64;  // K, also the tile row size in bytes.
    constexpr size_t kRowBytes = 64;

    auto in_a = AllocateAligned<TA>(kRows * kDimStepI8);
    auto in_b = AllocateAligned<TB>(kRows * kRowBytes);
    auto in_c = AllocateAligned<int32_t>(kRows * kColsI32);
    auto actual = AllocateAligned<int32_t>(kRows * kColsI32);
    auto expected = AllocateAligned<int32_t>(kRows * kColsI32);
    HWY_ASSERT(in_a && in_b && in_c && actual && expected);

    // Cover the full 8-bit range so that the four instructions differ.
    for (size_t i = 0; i < kRows * kDimStepI8; ++i) {
      in_a[i] = FromBits<TA>(static_cast<uint8_t>(i * 13 + 7));
      in_b[i] = FromBits<TB>(static_cast<uint8_t>(i * 29 + 3));
    }
    for (size_t i = 0; i < kRows * kColsI32; ++i) {
      in_c[i] = static_cast<int32_t>(i) - 8;
    }

    // C[r][c] += sum_k A[r][k] * B[k][c]. B is in 4-wide VNNI layout: its tile
    // row k / 4 holds, at byte offset c * 4 + k % 4, the logical B[k][c].
    // The maximum magnitude is 64 * 128 * 255, hence int32 does not overflow.
    for (size_t r = 0; r < kRows; ++r) {
      for (size_t c = 0; c < kColsI32; ++c) {
        int32_t sum = in_c[r * kColsI32 + c];
        for (size_t k = 0; k < kDimStepI8; ++k) {
          const TB b = in_b[(k / 4) * kRowBytes + c * 4 + (k % 4)];
          sum += static_cast<int32_t>(in_a[r * kDimStepI8 + k]) *
                 static_cast<int32_t>(b);
        }
        expected[r * kColsI32 + c] = sum;
      }
    }

    // Only used to select the instruction, hence the vector size is irrelevant.
    const Rebind<TA, DN> da;
    const Rebind<TB, DN> db;

    auto tile_c = MakeTile64B(kRows, kRowBytes);
    auto tile_a = MakeTile64B(kRows, kRowBytes);
    auto tile_b = MakeTile64B(kRows, kRowBytes);

    Tile64BLoad(&tile_c, in_c.get(), kRowBytes);
    Tile64BLoad(&tile_a, in_a.get(), kRowBytes);
    Tile64BLoad(&tile_b, in_b.get(), kRowBytes);
    Tile64BMatMul(dn, da, db, &tile_c, &tile_a, &tile_b);
    Tile64BStore(&tile_c, actual.get(), kRowBytes);
    Tile64BRelease();

    for (size_t r = 0; r < kRows; ++r) {
      HWY_ASSERT_VEC_EQ(dn, expected.get() + r * kColsI32,
                        Load(dn, actual.get() + r * kColsI32));
    }
  }
#endif  // HWY_NATIVE_TILE_64B_MATMUL_I8
};

HWY_NOINLINE void TestAllTile64BMatMulI8() {
  // AMX is independent of vector length, hence only test 512.
  ForGEVectors<512, TestTile64BMatMulI8>()(int32_t());
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_BEFORE_TEST(HwyMatmulOpTest);
HWY_EXPORT_AND_TEST_P(HwyMatmulOpTest, TestAllPerBlock2x2MatMulInt8);
HWY_EXPORT_AND_TEST_P(HwyMatmulOpTest, TestAllPerBlock2x2MatMulUint8Int8);
HWY_EXPORT_AND_TEST_P(HwyMatmulOpTest, TestAllPerBlock2x2MatMulBF16);
HWY_EXPORT_AND_TEST_P(HwyMatmulOpTest, TestAllTile64BMatMulBF16);
HWY_EXPORT_AND_TEST_P(HwyMatmulOpTest, TestAllTile64BMatMulI8);
HWY_AFTER_TEST();
}  // namespace
}  // namespace hwy
HWY_TEST_MAIN();
#endif  // HWY_ONCE
