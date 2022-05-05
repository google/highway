// Copyright 2021 Google LLC
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

// Target-independent types/functions defined after target-specific ops.

// Relies on the external include guard in highway.h.
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// The lane type of a vector type, e.g. float for Vec<ScalableTag<float>>.
template <class V>
using LaneType = decltype(GetLane(V()));

// Vector type, e.g. Vec128<float> for CappedTag<float, 4>. Useful as the return
// type of functions that do not take a vector argument, or as an argument type
// if the function only has a template argument for D, or for explicit type
// names instead of auto. This may be a built-in type.
template <class D>
using Vec = decltype(Zero(D()));

// Mask type. Useful as the return type of functions that do not take a mask
// argument, or as an argument type if the function only has a template argument
// for D, or for explicit type names instead of auto.
template <class D>
using Mask = decltype(MaskFromVec(Zero(D())));

// Returns the closest value to v within [lo, hi].
template <class V>
HWY_API V Clamp(const V v, const V lo, const V hi) {
  return Min(Max(lo, v), hi);
}

// CombineShiftRightBytes (and -Lanes) are not available for the scalar target,
// and RVV has its own implementation of -Lanes.
#if HWY_TARGET != HWY_SCALAR && HWY_TARGET != HWY_RVV

template <size_t kLanes, class D, class V = VFromD<D>>
HWY_API V CombineShiftRightLanes(D d, const V hi, const V lo) {
  constexpr size_t kBytes = kLanes * sizeof(LaneType<V>);
  static_assert(kBytes < 16, "Shift count is per-block");
  return CombineShiftRightBytes<kBytes>(d, hi, lo);
}

#endif

// Returns lanes with the most significant bit set and all other bits zero.
template <class D>
HWY_API Vec<D> SignBit(D d) {
  const RebindToUnsigned<decltype(d)> du;
  return BitCast(d, Set(du, SignMask<TFromD<D>>()));
}

// Returns quiet NaN.
template <class D>
HWY_API Vec<D> NaN(D d) {
  const RebindToSigned<D> di;
  // LimitsMax sets all exponent and mantissa bits to 1. The exponent plus
  // mantissa MSB (to indicate quiet) would be sufficient.
  return BitCast(d, Set(di, LimitsMax<TFromD<decltype(di)>>()));
}

// Returns positive infinity.
template <class D>
HWY_API Vec<D> Inf(D d) {
  const RebindToUnsigned<D> du;
  using T = TFromD<D>;
  using TU = TFromD<decltype(du)>;
  const TU max_x2 = static_cast<TU>(MaxExponentTimes2<T>());
  return BitCast(d, Set(du, max_x2 >> 1));
}

// ------------------------------ SafeFillN

template <class D, typename T = TFromD<D>>
HWY_API void SafeFillN(const size_t num, const T value, D d,
                       T* HWY_RESTRICT to) {
#if HWY_MEM_OPS_MIGHT_FAULT
  (void)d;
  for (size_t i = 0; i < num; ++i) {
    to[i] = value;
  }
#else
  BlendedStore(Set(d, value), FirstN(d, num), d, to);
#endif
}

// ------------------------------ SafeCopyN

template <class D, typename T = TFromD<D>>
HWY_API void SafeCopyN(const size_t num, D d, const T* HWY_RESTRICT from,
                       T* HWY_RESTRICT to) {
#if HWY_MEM_OPS_MIGHT_FAULT
  (void)d;
  for (size_t i = 0; i < num; ++i) {
    to[i] = from[i];
  }
#else
  const Mask<D> mask = FirstN(d, num);
  BlendedStore(MaskedLoad(mask, d, from), mask, d, to);
#endif
}

// ------------------------------ StoreInterleaved2

// "Include guard": skip if native instructions are available. The generic
// implementation is currently only shared between x86_128 and wasm_*, but it is
// too large to duplicate.

#if (defined(HWY_NATIVE_STORE_INTERLEAVED) == defined(HWY_TARGET_TOGGLE))
#ifdef HWY_NATIVE_STORE_INTERLEAVED
#undef HWY_NATIVE_STORE_INTERLEAVED
#else
#define HWY_NATIVE_STORE_INTERLEAVED
#endif

// 128 bit vector, 8..32 bit lanes
template <typename T, HWY_IF_NOT_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved2(const Vec128<T> v0, const Vec128<T> v1,
                               Full128<T> d, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  const RepartitionToWide<decltype(d)> dw;
  const auto v10L = BitCast(d, ZipLower(dw, v0, v1));  // .. v1[0] v0[0]
  const auto v10H = BitCast(d, ZipUpper(dw, v0, v1));  // .. v1[N/2] v0[N/2]
  StoreU(v10L, d, unaligned + 0 * N);
  StoreU(v10H, d, unaligned + 1 * N);
}

// 128 bit vector, 64 bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved2(const Vec128<T> v0, const Vec128<T> v1,
                               Full128<T> d, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  const auto v10L = InterleaveLower(d, v0, v1);
  const auto v10H = InterleaveUpper(d, v0, v1);
  StoreU(v10L, d, unaligned + 0 * N);
  StoreU(v10H, d, unaligned + 1 * N);
}

// 64 bits
template <typename T>
HWY_API void StoreInterleaved2(const Vec64<T> part0, const Vec64<T> part1,
                               Full64<T> /*tag*/, T* HWY_RESTRICT unaligned) {
  // Use full vectors to reduce the number of stores.
  const Full128<T> d_full;
  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const auto v10 = InterleaveLower(d_full, v0, v1);
  StoreU(v10, d_full, unaligned);
}

// <= 32 bits
template <typename T, size_t N, HWY_IF_LE32(T, N)>
HWY_API void StoreInterleaved2(const Vec128<T, N> part0,
                               const Vec128<T, N> part1, Simd<T, N, 0> /*tag*/,
                               T* HWY_RESTRICT unaligned) {
  // Use full vectors to reduce the number of stores.
  const Full128<T> d_full;
  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const auto v10 = InterleaveLower(d_full, v0, v1);
  alignas(16) T buf[16 / sizeof(T)];
  StoreU(v10, d_full, buf);
  CopyBytes<2 * N * sizeof(T)>(buf, unaligned);
}

// ------------------------------ StoreInterleaved3 (CombineShiftRightBytes,
// TableLookupBytes)

// 128-bit vector, 8-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 1)>
HWY_API void StoreInterleaved3(const Vec128<T> v0, const Vec128<T> v1,
                               const Vec128<T> v2, Full128<T> d,
                               T* HWY_RESTRICT unaligned) {
  const RebindToUnsigned<decltype(d)> du;
  constexpr size_t N = 16 / sizeof(T);
  const auto k5 = Set(du, 5);
  const auto k6 = Set(du, 6);

  // Interleave (v0,v1,v2) to (MSB on left, lane 0 on right):
  // v0[5], v2[4],v1[4],v0[4] .. v2[0],v1[0],v0[0]. We're expanding v0 lanes
  // to their place, with 0x80 so lanes to be filled from other vectors are 0
  // to enable blending by ORing together.
  alignas(16) static constexpr uint8_t tbl_v0[16] = {
      0, 0x80, 0x80, 1, 0x80, 0x80, 2, 0x80, 0x80,  //
      3, 0x80, 0x80, 4, 0x80, 0x80, 5};
  alignas(16) static constexpr uint8_t tbl_v1[16] = {
      0x80, 0, 0x80, 0x80, 1, 0x80,  //
      0x80, 2, 0x80, 0x80, 3, 0x80, 0x80, 4, 0x80, 0x80};
  // The interleaved vectors will be named A, B, C; temporaries with suffix
  // 0..2 indicate which input vector's lanes they hold.
  const auto shuf_A0 = Load(du, tbl_v0);
  const auto shuf_A1 = Load(du, tbl_v1);  // cannot reuse shuf_A0 (5 in MSB)
  const auto shuf_A2 = CombineShiftRightBytes<15>(du, shuf_A1, shuf_A1);
  const auto A0 = TableLookupBytesOr0(v0, shuf_A0);  // 5..4..3..2..1..0
  const auto A1 = TableLookupBytesOr0(v1, shuf_A1);  // ..4..3..2..1..0.
  const auto A2 = TableLookupBytesOr0(v2, shuf_A2);  // .4..3..2..1..0..
  const Vec128<T> A = BitCast(d, A0 | A1 | A2);
  StoreU(A, d, unaligned + 0 * N);

  // B: v1[10],v0[10], v2[9],v1[9],v0[9] .. , v2[6],v1[6],v0[6], v2[5],v1[5]
  const auto shuf_B0 = shuf_A2 + k6;  // .A..9..8..7..6..
  const auto shuf_B1 = shuf_A0 + k5;  // A..9..8..7..6..5
  const auto shuf_B2 = shuf_A1 + k5;  // ..9..8..7..6..5.
  const auto B0 = TableLookupBytesOr0(v0, shuf_B0);
  const auto B1 = TableLookupBytesOr0(v1, shuf_B1);
  const auto B2 = TableLookupBytesOr0(v2, shuf_B2);
  const Vec128<T> B = BitCast(d, B0 | B1 | B2);
  StoreU(B, d, unaligned + 1 * N);

  // C: v2[15],v1[15],v0[15], v2[11],v1[11],v0[11], v2[10]
  const auto shuf_C0 = shuf_B2 + k6;  // ..F..E..D..C..B.
  const auto shuf_C1 = shuf_B0 + k5;  // .F..E..D..C..B..
  const auto shuf_C2 = shuf_B1 + k5;  // F..E..D..C..B..A
  const auto C0 = TableLookupBytesOr0(v0, shuf_C0);
  const auto C1 = TableLookupBytesOr0(v1, shuf_C1);
  const auto C2 = TableLookupBytesOr0(v2, shuf_C2);
  const Vec128<T> C = BitCast(d, C0 | C1 | C2);
  StoreU(C, d, unaligned + 2 * N);
}

// 128-bit vector, 16-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 2)>
HWY_API void StoreInterleaved3(const Vec128<T> v0, const Vec128<T> v1,
                               const Vec128<T> v2, Full128<T> d,
                               T* HWY_RESTRICT unaligned) {
  const Repartition<uint8_t, decltype(d)> du8;
  constexpr size_t N = 16 / sizeof(T);
  const auto k2 = Set(du8, 2 * sizeof(T));
  const auto k3 = Set(du8, 3 * sizeof(T));

  // Interleave (v0,v1,v2) to (MSB on left, lane 0 on right):
  // v1[2],v0[2], v2[1],v1[1],v0[1], v2[0],v1[0],v0[0]. 0x80 so lanes to be
  // filled from other vectors are 0 for blending. Note that these are byte
  // indices for 16-bit lanes.
  alignas(16) static constexpr uint8_t tbl_v1[16] = {
      0x80, 0x80, 0,    1,    0x80, 0x80, 0x80, 0x80,
      2,    3,    0x80, 0x80, 0x80, 0x80, 4,    5};
  alignas(16) static constexpr uint8_t tbl_v2[16] = {
      0x80, 0x80, 0x80, 0x80, 0,    1,    0x80, 0x80,
      0x80, 0x80, 2,    3,    0x80, 0x80, 0x80, 0x80};

  // The interleaved vectors will be named A, B, C; temporaries with suffix
  // 0..2 indicate which input vector's lanes they hold.
  const auto shuf_A1 = Load(du8, tbl_v1);  // 2..1..0.
                                           // .2..1..0
  const auto shuf_A0 = CombineShiftRightBytes<2>(du8, shuf_A1, shuf_A1);
  const auto shuf_A2 = Load(du8, tbl_v2);  // ..1..0..

  const auto A0 = TableLookupBytesOr0(v0, shuf_A0);
  const auto A1 = TableLookupBytesOr0(v1, shuf_A1);
  const auto A2 = TableLookupBytesOr0(v2, shuf_A2);
  const Vec128<T> A = BitCast(d, A0 | A1 | A2);
  StoreU(A, d, unaligned + 0 * N);

  // B: v0[5] v2[4],v1[4],v0[4], v2[3],v1[3],v0[3], v2[2]
  const auto shuf_B0 = shuf_A1 + k3;  // 5..4..3.
  const auto shuf_B1 = shuf_A2 + k3;  // ..4..3..
  const auto shuf_B2 = shuf_A0 + k2;  // .4..3..2
  const auto B0 = TableLookupBytesOr0(v0, shuf_B0);
  const auto B1 = TableLookupBytesOr0(v1, shuf_B1);
  const auto B2 = TableLookupBytesOr0(v2, shuf_B2);
  const Vec128<T> B = BitCast(d, B0 | B1 | B2);
  StoreU(B, d, unaligned + 1 * N);

  // C: v2[7],v1[7],v0[7], v2[6],v1[6],v0[6], v2[5],v1[5]
  const auto shuf_C0 = shuf_B1 + k3;  // ..7..6..
  const auto shuf_C1 = shuf_B2 + k3;  // .7..6..5
  const auto shuf_C2 = shuf_B0 + k2;  // 7..6..5.
  const auto C0 = TableLookupBytesOr0(v0, shuf_C0);
  const auto C1 = TableLookupBytesOr0(v1, shuf_C1);
  const auto C2 = TableLookupBytesOr0(v2, shuf_C2);
  const Vec128<T> C = BitCast(d, C0 | C1 | C2);
  StoreU(C, d, unaligned + 2 * N);
}

// 128-bit vector, 32-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 4)>
HWY_API void StoreInterleaved3(const Vec128<T> v0, const Vec128<T> v1,
                               const Vec128<T> v2, Full128<T> d,
                               T* HWY_RESTRICT unaligned) {
  const RepartitionToWide<decltype(d)> dw;
  constexpr size_t N = 16 / sizeof(T);

  const Vec128<T> v10_v00 = InterleaveLower(d, v0, v1);
  const Vec128<T> v01_v20 = OddEven(v0, v2);
  // A: v0[1], v2[0],v1[0],v0[0] (<- lane 0)
  const Vec128<T> A = BitCast(
      d, InterleaveLower(dw, BitCast(dw, v10_v00), BitCast(dw, v01_v20)));
  StoreU(A, d, unaligned + 0 * N);

  const Vec128<T> v1_321 = ShiftRightLanes<1>(d, v1);
  const Vec128<T> v0_32 = ShiftRightLanes<2>(d, v0);
  const Vec128<T> v21_v11 = OddEven(v2, v1_321);
  const Vec128<T> v12_v02 = OddEven(v1_321, v0_32);
  // B: v1[2],v0[2], v2[1],v1[1]
  const Vec128<T> B = BitCast(
      d, InterleaveLower(dw, BitCast(dw, v21_v11), BitCast(dw, v12_v02)));
  StoreU(B, d, unaligned + 1 * N);

  // Notation refers to the upper 2 lanes of the vector for InterleaveUpper.
  const Vec128<T> v23_v13 = OddEven(v2, v1_321);
  const Vec128<T> v03_v22 = OddEven(v0, v2);
  // C: v2[3],v1[3],v0[3], v2[2]
  const Vec128<T> C = BitCast(
      d, InterleaveUpper(dw, BitCast(dw, v03_v22), BitCast(dw, v23_v13)));
  StoreU(C, d, unaligned + 2 * N);
}

// 128-bit vector, 64-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved3(const Vec128<T> v0, const Vec128<T> v1,
                               const Vec128<T> v2, Full128<T> d,
                               T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 2;
  const Vec128<T> v10_v00 = InterleaveLower(d, v0, v1);
  const Vec128<T> v01_v20 = OddEven(v1, v2);
  const Vec128<T> v21_v11 = InterleaveUpper(d, v1, v2);
  StoreU(v10_v00, d, unaligned + 0 * N);
  StoreU(v01_v20, d, unaligned + 1 * N);
  StoreU(v21_v11, d, unaligned + 2 * N);
}

// 64-bit vector, 8-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 1)>
HWY_API void StoreInterleaved3(const Vec64<T> part0, const Vec64<T> part1,
                               const Vec64<T> part2, Full64<T> d,
                               T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  // Use full vectors for the shuffles and first result.
  const Full128<uint8_t> du;
  const Full128<T> d_full;
  const auto k5 = Set(du, 5);
  const auto k6 = Set(du, 6);

  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};

  // Interleave (v0,v1,v2) to (MSB on left, lane 0 on right):
  // v1[2],v0[2], v2[1],v1[1],v0[1], v2[0],v1[0],v0[0]. 0x80 so lanes to be
  // filled from other vectors are 0 for blending.
  alignas(16) static constexpr uint8_t tbl_v0[16] = {
      0, 0x80, 0x80, 1, 0x80, 0x80, 2, 0x80, 0x80,  //
      3, 0x80, 0x80, 4, 0x80, 0x80, 5};
  alignas(16) static constexpr uint8_t tbl_v1[16] = {
      0x80, 0, 0x80, 0x80, 1, 0x80,  //
      0x80, 2, 0x80, 0x80, 3, 0x80, 0x80, 4, 0x80, 0x80};
  // The interleaved vectors will be named A, B, C; temporaries with suffix
  // 0..2 indicate which input vector's lanes they hold.
  const auto shuf_A0 = Load(du, tbl_v0);
  const auto shuf_A1 = Load(du, tbl_v1);  // cannot reuse shuf_A0 (5 in MSB)
  const auto shuf_A2 = CombineShiftRightBytes<15>(du, shuf_A1, shuf_A1);
  const auto A0 = TableLookupBytesOr0(v0, shuf_A0);  // 5..4..3..2..1..0
  const auto A1 = TableLookupBytesOr0(v1, shuf_A1);  // ..4..3..2..1..0.
  const auto A2 = TableLookupBytesOr0(v2, shuf_A2);  // .4..3..2..1..0..
  const auto A = BitCast(d_full, A0 | A1 | A2);
  StoreU(A, d_full, unaligned + 0 * N);

  // Second (HALF) vector: v2[7],v1[7],v0[7], v2[6],v1[6],v0[6], v2[5],v1[5]
  const auto shuf_B0 = shuf_A2 + k6;  // ..7..6..
  const auto shuf_B1 = shuf_A0 + k5;  // .7..6..5
  const auto shuf_B2 = shuf_A1 + k5;  // 7..6..5.
  const auto B0 = TableLookupBytesOr0(v0, shuf_B0);
  const auto B1 = TableLookupBytesOr0(v1, shuf_B1);
  const auto B2 = TableLookupBytesOr0(v2, shuf_B2);
  const Vec64<T> B{(B0 | B1 | B2).raw};
  StoreU(B, d, unaligned + 1 * N);
}

// 64-bit vector, 16-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 2)>
HWY_API void StoreInterleaved3(const Vec64<T> part0, const Vec64<T> part1,
                               const Vec64<T> part2, Full64<T> dh,
                               T* HWY_RESTRICT unaligned) {
  const Full128<T> d;
  const Full128<uint8_t> du8;
  constexpr size_t N = 16 / sizeof(T);
  const auto k2 = Set(du8, 2 * sizeof(T));
  const auto k3 = Set(du8, 3 * sizeof(T));

  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};

  // Interleave part (v0,v1,v2) to full (MSB on left, lane 0 on right):
  // v1[2],v0[2], v2[1],v1[1],v0[1], v2[0],v1[0],v0[0]. We're expanding v0 lanes
  // to their place, with 0x80 so lanes to be filled from other vectors are 0
  // to enable blending by ORing together.
  alignas(16) static constexpr uint8_t tbl_v1[16] = {
      0x80, 0x80, 0,    1,    0x80, 0x80, 0x80, 0x80,
      2,    3,    0x80, 0x80, 0x80, 0x80, 4,    5};
  alignas(16) static constexpr uint8_t tbl_v2[16] = {
      0x80, 0x80, 0x80, 0x80, 0,    1,    0x80, 0x80,
      0x80, 0x80, 2,    3,    0x80, 0x80, 0x80, 0x80};

  // The interleaved vectors will be named A, B; temporaries with suffix
  // 0..2 indicate which input vector's lanes they hold.
  const auto shuf_A1 = Load(du8, tbl_v1);  // 2..1..0.
                                           // .2..1..0
  const auto shuf_A0 = CombineShiftRightBytes<2>(du8, shuf_A1, shuf_A1);
  const auto shuf_A2 = Load(du8, tbl_v2);  // ..1..0..

  const auto A0 = TableLookupBytesOr0(v0, shuf_A0);
  const auto A1 = TableLookupBytesOr0(v1, shuf_A1);
  const auto A2 = TableLookupBytesOr0(v2, shuf_A2);
  const Vec128<T> A = BitCast(d, A0 | A1 | A2);
  StoreU(A, d, unaligned + 0 * N);

  // Second (HALF) vector: v2[3],v1[3],v0[3], v2[2]
  const auto shuf_B0 = shuf_A1 + k3;  // ..3.
  const auto shuf_B1 = shuf_A2 + k3;  // .3..
  const auto shuf_B2 = shuf_B0 + k2;  // 3..2
  const auto B0 = TableLookupBytesOr0(v0, shuf_B0);
  const auto B1 = TableLookupBytesOr0(v1, shuf_B1);
  const auto B2 = TableLookupBytesOr0(v2, shuf_B2);
  const Vec128<T> B = BitCast(d, B0 | B1 | B2);
  StoreU(Vec64<T>{B.raw}, dh, unaligned + 1 * N);
}

// 64-bit vector, 32-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 4)>
HWY_API void StoreInterleaved3(const Vec64<T> v0, const Vec64<T> v1,
                               const Vec64<T> v2, Full64<T> d,
                               T* HWY_RESTRICT unaligned) {
  // (same code as 128-bit vector, 64-bit lanes)
  constexpr size_t N = 2;
  const Vec64<T> v10_v00 = InterleaveLower(d, v0, v1);
  const Vec64<T> v01_v20 = OddEven(v1, v2);
  const Vec64<T> v21_v11 = InterleaveUpper(d, v1, v2);
  StoreU(v10_v00, d, unaligned + 0 * N);
  StoreU(v01_v20, d, unaligned + 1 * N);
  StoreU(v21_v11, d, unaligned + 2 * N);
}

// 64-bit lanes are handled by the N=1 case below.

// <= 32-bit vector, 8-bit lanes
template <typename T, size_t N, HWY_IF_LANE_SIZE(T, 1), HWY_IF_LE32(T, N)>
HWY_API void StoreInterleaved3(const Vec128<T, N> part0,
                               const Vec128<T, N> part1,
                               const Vec128<T, N> part2, Simd<T, N, 0> /*tag*/,
                               T* HWY_RESTRICT unaligned) {
  // Use full vectors for the shuffles and result.
  const Full128<uint8_t> du;
  const Full128<T> d_full;

  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};

  // Interleave (v0,v1,v2). We're expanding v0 lanes to their place, with 0x80
  // so lanes to be filled from other vectors are 0 to enable blending by ORing
  // together.
  alignas(16) static constexpr uint8_t tbl_v0[16] = {
      0,    0x80, 0x80, 1,    0x80, 0x80, 2,    0x80,
      0x80, 3,    0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
  // The interleaved vector will be named A; temporaries with suffix
  // 0..2 indicate which input vector's lanes they hold.
  const auto shuf_A0 = Load(du, tbl_v0);
  const auto shuf_A1 = CombineShiftRightBytes<15>(du, shuf_A0, shuf_A0);
  const auto shuf_A2 = CombineShiftRightBytes<14>(du, shuf_A0, shuf_A0);
  const auto A0 = TableLookupBytesOr0(v0, shuf_A0);  // ......3..2..1..0
  const auto A1 = TableLookupBytesOr0(v1, shuf_A1);  // .....3..2..1..0.
  const auto A2 = TableLookupBytesOr0(v2, shuf_A2);  // ....3..2..1..0..
  const Vec128<T> A = BitCast(d_full, A0 | A1 | A2);
  alignas(16) T buf[16 / sizeof(T)];
  StoreU(A, d_full, buf);
  CopyBytes<N * 3 * sizeof(T)>(buf, unaligned);
}

// 32-bit vector, 16-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 2)>
HWY_API void StoreInterleaved3(const Vec128<T, 2> part0,
                               const Vec128<T, 2> part1,
                               const Vec128<T, 2> part2, Simd<T, 2, 0> /*tag*/,
                               T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 4 / sizeof(T);
  // Use full vectors for the shuffles and result.
  const Full128<uint8_t> du8;
  const Full128<T> d_full;

  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};

  // Interleave (v0,v1,v2). We're expanding v0 lanes to their place, with 0x80
  // so lanes to be filled from other vectors are 0 to enable blending by ORing
  // together.
  alignas(16) static constexpr uint8_t tbl_v2[16] = {
      0x80, 0x80, 0x80, 0x80, 0,    1,    0x80, 0x80,
      0x80, 0x80, 2,    3,    0x80, 0x80, 0x80, 0x80};
  // The interleaved vector will be named A; temporaries with suffix
  // 0..2 indicate which input vector's lanes they hold.
  const auto shuf_A2 =  // ..1..0..
      Load(du8, tbl_v2);
  const auto shuf_A1 =  // ...1..0.
      CombineShiftRightBytes<2>(du8, shuf_A2, shuf_A2);
  const auto shuf_A0 =  // ....1..0
      CombineShiftRightBytes<4>(du8, shuf_A2, shuf_A2);
  const auto A0 = TableLookupBytesOr0(v0, shuf_A0);  // ..1..0
  const auto A1 = TableLookupBytesOr0(v1, shuf_A1);  // .1..0.
  const auto A2 = TableLookupBytesOr0(v2, shuf_A2);  // 1..0..
  const auto A = BitCast(d_full, A0 | A1 | A2);
  alignas(16) T buf[16 / sizeof(T)];
  StoreU(A, d_full, buf);
  CopyBytes<N * 3 * sizeof(T)>(buf, unaligned);
}

// Single-element vector, any lane size: just store directly
template <typename T>
HWY_API void StoreInterleaved3(const Vec128<T, 1> v0, const Vec128<T, 1> v1,
                               const Vec128<T, 1> v2, Simd<T, 1, 0> d,
                               T* HWY_RESTRICT unaligned) {
  StoreU(v0, d, unaligned + 0);
  StoreU(v1, d, unaligned + 1);
  StoreU(v2, d, unaligned + 2);
}

// ------------------------------ StoreInterleaved4

// 128-bit vector, 8..32-bit lanes
template <typename T, HWY_IF_NOT_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved4(const Vec128<T> v0, const Vec128<T> v1,
                               const Vec128<T> v2, const Vec128<T> v3,
                               Full128<T> d, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  const RepartitionToWide<decltype(d)> dw;
  const auto v10L = ZipLower(dw, v0, v1);  // .. v1[0] v0[0]
  const auto v32L = ZipLower(dw, v2, v3);
  const auto v10H = ZipUpper(dw, v0, v1);
  const auto v32H = ZipUpper(dw, v2, v3);
  // The interleaved vectors are A, B, C, D.
  const auto A = BitCast(d, InterleaveLower(dw, v10L, v32L));  // 3210
  const auto B = BitCast(d, InterleaveUpper(dw, v10L, v32L));
  const auto C = BitCast(d, InterleaveLower(dw, v10H, v32H));
  const auto D = BitCast(d, InterleaveUpper(dw, v10H, v32H));
  StoreU(A, d, unaligned + 0 * N);
  StoreU(B, d, unaligned + 1 * N);
  StoreU(C, d, unaligned + 2 * N);
  StoreU(D, d, unaligned + 3 * N);
}

// 128-bit vector, 64-bit lanes
template <typename T, HWY_IF_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved4(const Vec128<T> v0, const Vec128<T> v1,
                               const Vec128<T> v2, const Vec128<T> v3,
                               Full128<T> d, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  // The interleaved vectors are A, B, C, D.
  const auto A = InterleaveLower(d, v0, v1);  // v1[0] v0[0]
  const auto B = InterleaveLower(d, v2, v3);
  const auto C = InterleaveUpper(d, v0, v1);
  const auto D = InterleaveUpper(d, v2, v3);
  StoreU(A, d, unaligned + 0 * N);
  StoreU(B, d, unaligned + 1 * N);
  StoreU(C, d, unaligned + 2 * N);
  StoreU(D, d, unaligned + 3 * N);
}

// 64-bit vector, 8..32-bit lanes
template <typename T, HWY_IF_NOT_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved4(const Vec64<T> part0, const Vec64<T> part1,
                               const Vec64<T> part2, const Vec64<T> part3,
                               Full64<T> /*tag*/, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  // Use full vectors to reduce the number of stores.
  const Full128<T> d_full;
  const RepartitionToWide<decltype(d_full)> dw;
  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};
  const Vec128<T> v3{part3.raw};
  const auto v10 = ZipLower(dw, v0, v1);  // v1[0] v0[0]
  const auto v32 = ZipLower(dw, v2, v3);
  const auto A = BitCast(d_full, InterleaveLower(dw, v10, v32));
  const auto B = BitCast(d_full, InterleaveUpper(dw, v10, v32));
  StoreU(A, d_full, unaligned + 0 * N);
  StoreU(B, d_full, unaligned + 1 * N);
}

// 64-bit vector, 64-bit lane
template <typename T, HWY_IF_LANE_SIZE(T, 8)>
HWY_API void StoreInterleaved4(const Vec64<T> part0, const Vec64<T> part1,
                               const Vec64<T> part2, const Vec64<T> part3,
                               Full64<T> /*tag*/, T* HWY_RESTRICT unaligned) {
  constexpr size_t N = 16 / sizeof(T);
  // Use full vectors to reduce the number of stores.
  const Full128<T> d_full;
  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};
  const Vec128<T> v3{part3.raw};
  const auto A = InterleaveLower(d_full, v0, v1);  // v1[0] v0[0]
  const auto B = InterleaveLower(d_full, v2, v3);
  StoreU(A, d_full, unaligned + 0 * N);
  StoreU(B, d_full, unaligned + 1 * N);
}

// <= 32-bit vectors
template <typename T, size_t N, HWY_IF_LE32(T, N)>
HWY_API void StoreInterleaved4(const Vec128<T, N> part0,
                               const Vec128<T, N> part1,
                               const Vec128<T, N> part2,
                               const Vec128<T, N> part3, Simd<T, N, 0> /*tag*/,
                               T* HWY_RESTRICT unaligned) {
  // Use full vectors to reduce the number of stores.
  const Full128<T> d_full;
  const RepartitionToWide<decltype(d_full)> dw;
  const Vec128<T> v0{part0.raw};
  const Vec128<T> v1{part1.raw};
  const Vec128<T> v2{part2.raw};
  const Vec128<T> v3{part3.raw};
  const auto v10 = ZipLower(dw, v0, v1);  // .. v1[0] v0[0]
  const auto v32 = ZipLower(dw, v2, v3);
  const auto v3210 = BitCast(d_full, InterleaveLower(dw, v10, v32));
  alignas(16) T buf[16 / sizeof(T)];
  StoreU(v3210, d_full, buf);
  CopyBytes<4 * N * sizeof(T)>(buf, unaligned);
}

#endif  // HWY_NATIVE_STORE_INTERLEAVED

// ------------------------------ AESRound

// Cannot implement on scalar: need at least 16 bytes for TableLookupBytes.
#if HWY_TARGET != HWY_SCALAR

// Define for white-box testing, even if native instructions are available.
namespace detail {

// Constant-time: computes inverse in GF(2^4) based on "Accelerating AES with
// Vector Permute Instructions" and the accompanying assembly language
// implementation: https://crypto.stanford.edu/vpaes/vpaes.tgz. See also Botan:
// https://botan.randombit.net/doxygen/aes__vperm_8cpp_source.html .
//
// A brute-force 256 byte table lookup can also be made constant-time, and
// possibly competitive on NEON, but this is more performance-portable
// especially for x86 and large vectors.
template <class V>  // u8
HWY_INLINE V SubBytes(V state) {
  const DFromV<V> du;
  const auto mask = Set(du, 0xF);

  // Change polynomial basis to GF(2^4)
  {
    alignas(16) static constexpr uint8_t basisL[16] = {
        0x00, 0x70, 0x2A, 0x5A, 0x98, 0xE8, 0xB2, 0xC2,
        0x08, 0x78, 0x22, 0x52, 0x90, 0xE0, 0xBA, 0xCA};
    alignas(16) static constexpr uint8_t basisU[16] = {
        0x00, 0x4D, 0x7C, 0x31, 0x7D, 0x30, 0x01, 0x4C,
        0x81, 0xCC, 0xFD, 0xB0, 0xFC, 0xB1, 0x80, 0xCD};
    const auto sL = And(state, mask);
    const auto sU = ShiftRight<4>(state);  // byte shift => upper bits are zero
    const auto gf4L = TableLookupBytes(LoadDup128(du, basisL), sL);
    const auto gf4U = TableLookupBytes(LoadDup128(du, basisU), sU);
    state = Xor(gf4L, gf4U);
  }

  // Inversion in GF(2^4). Elements 0 represent "infinity" (division by 0) and
  // cause TableLookupBytesOr0 to return 0.
  alignas(16) static constexpr uint8_t kZetaInv[16] = {
      0x80, 7, 11, 15, 6, 10, 4, 1, 9, 8, 5, 2, 12, 14, 13, 3};
  alignas(16) static constexpr uint8_t kInv[16] = {
      0x80, 1, 8, 13, 15, 6, 5, 14, 2, 12, 11, 10, 9, 3, 7, 4};
  const auto tbl = LoadDup128(du, kInv);
  const auto sL = And(state, mask);      // L=low nibble, U=upper
  const auto sU = ShiftRight<4>(state);  // byte shift => upper bits are zero
  const auto sX = Xor(sU, sL);
  const auto invL = TableLookupBytes(LoadDup128(du, kZetaInv), sL);
  const auto invU = TableLookupBytes(tbl, sU);
  const auto invX = TableLookupBytes(tbl, sX);
  const auto outL = Xor(sX, TableLookupBytesOr0(tbl, Xor(invL, invU)));
  const auto outU = Xor(sU, TableLookupBytesOr0(tbl, Xor(invL, invX)));

  // Linear skew (cannot bake 0x63 bias into the table because out* indices
  // may have the infinity flag set).
  alignas(16) static constexpr uint8_t kAffineL[16] = {
      0x00, 0xC7, 0xBD, 0x6F, 0x17, 0x6D, 0xD2, 0xD0,
      0x78, 0xA8, 0x02, 0xC5, 0x7A, 0xBF, 0xAA, 0x15};
  alignas(16) static constexpr uint8_t kAffineU[16] = {
      0x00, 0x6A, 0xBB, 0x5F, 0xA5, 0x74, 0xE4, 0xCF,
      0xFA, 0x35, 0x2B, 0x41, 0xD1, 0x90, 0x1E, 0x8E};
  const auto affL = TableLookupBytesOr0(LoadDup128(du, kAffineL), outL);
  const auto affU = TableLookupBytesOr0(LoadDup128(du, kAffineU), outU);
  return Xor(Xor(affL, affU), Set(du, 0x63));
}

}  // namespace detail

#endif  // HWY_TARGET != HWY_SCALAR

// "Include guard": skip if native AES instructions are available.
#if (defined(HWY_NATIVE_AES) == defined(HWY_TARGET_TOGGLE))
#ifdef HWY_NATIVE_AES
#undef HWY_NATIVE_AES
#else
#define HWY_NATIVE_AES
#endif

// (Must come after HWY_TARGET_TOGGLE, else we don't reset it for scalar)
#if HWY_TARGET != HWY_SCALAR

namespace detail {

template <class V>  // u8
HWY_API V ShiftRows(const V state) {
  const DFromV<V> du;
  alignas(16) static constexpr uint8_t kShiftRow[16] = {
      0,  5,  10, 15,  // transposed: state is column major
      4,  9,  14, 3,   //
      8,  13, 2,  7,   //
      12, 1,  6,  11};
  const auto shift_row = LoadDup128(du, kShiftRow);
  return TableLookupBytes(state, shift_row);
}

template <class V>  // u8
HWY_API V MixColumns(const V state) {
  const DFromV<V> du;
  // For each column, the rows are the sum of GF(2^8) matrix multiplication by:
  // 2 3 1 1  // Let s := state*1, d := state*2, t := state*3.
  // 1 2 3 1  // d are on diagonal, no permutation needed.
  // 1 1 2 3  // t1230 indicates column indices of threes for the 4 rows.
  // 3 1 1 2  // We also need to compute s2301 and s3012 (=1230 o 2301).
  alignas(16) static constexpr uint8_t k2301[16] = {
      2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13};
  alignas(16) static constexpr uint8_t k1230[16] = {
      1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12};
  const RebindToSigned<decltype(du)> di;  // can only do signed comparisons
  const auto msb = Lt(BitCast(di, state), Zero(di));
  const auto overflow = BitCast(du, IfThenElseZero(msb, Set(di, 0x1B)));
  const auto d = Xor(Add(state, state), overflow);  // = state*2 in GF(2^8).
  const auto s2301 = TableLookupBytes(state, LoadDup128(du, k2301));
  const auto d_s2301 = Xor(d, s2301);
  const auto t_s2301 = Xor(state, d_s2301);  // t(s*3) = XOR-sum {s, d(s*2)}
  const auto t1230_s3012 = TableLookupBytes(t_s2301, LoadDup128(du, k1230));
  return Xor(d_s2301, t1230_s3012);  // XOR-sum of 4 terms
}

}  // namespace detail

template <class V>  // u8
HWY_API V AESRound(V state, const V round_key) {
  // Intel docs swap the first two steps, but it does not matter because
  // ShiftRows is a permutation and SubBytes is independent of lane index.
  state = detail::SubBytes(state);
  state = detail::ShiftRows(state);
  state = detail::MixColumns(state);
  state = Xor(state, round_key);  // AddRoundKey
  return state;
}

template <class V>  // u8
HWY_API V AESLastRound(V state, const V round_key) {
  // LIke AESRound, but without MixColumns.
  state = detail::SubBytes(state);
  state = detail::ShiftRows(state);
  state = Xor(state, round_key);  // AddRoundKey
  return state;
}

// Constant-time implementation inspired by
// https://www.bearssl.org/constanttime.html, but about half the cost because we
// use 64x64 multiplies and 128-bit XORs.
template <class V>
HWY_API V CLMulLower(V a, V b) {
  const DFromV<V> d;
  static_assert(IsSame<TFromD<decltype(d)>, uint64_t>(), "V must be u64");
  const auto k1 = Set(d, 0x1111111111111111ULL);
  const auto k2 = Set(d, 0x2222222222222222ULL);
  const auto k4 = Set(d, 0x4444444444444444ULL);
  const auto k8 = Set(d, 0x8888888888888888ULL);
  const auto a0 = And(a, k1);
  const auto a1 = And(a, k2);
  const auto a2 = And(a, k4);
  const auto a3 = And(a, k8);
  const auto b0 = And(b, k1);
  const auto b1 = And(b, k2);
  const auto b2 = And(b, k4);
  const auto b3 = And(b, k8);

  auto m0 = Xor(MulEven(a0, b0), MulEven(a1, b3));
  auto m1 = Xor(MulEven(a0, b1), MulEven(a1, b0));
  auto m2 = Xor(MulEven(a0, b2), MulEven(a1, b1));
  auto m3 = Xor(MulEven(a0, b3), MulEven(a1, b2));
  m0 = Xor(m0, Xor(MulEven(a2, b2), MulEven(a3, b1)));
  m1 = Xor(m1, Xor(MulEven(a2, b3), MulEven(a3, b2)));
  m2 = Xor(m2, Xor(MulEven(a2, b0), MulEven(a3, b3)));
  m3 = Xor(m3, Xor(MulEven(a2, b1), MulEven(a3, b0)));
  return Or(Or(And(m0, k1), And(m1, k2)), Or(And(m2, k4), And(m3, k8)));
}

template <class V>
HWY_API V CLMulUpper(V a, V b) {
  const DFromV<V> d;
  static_assert(IsSame<TFromD<decltype(d)>, uint64_t>(), "V must be u64");
  const auto k1 = Set(d, 0x1111111111111111ULL);
  const auto k2 = Set(d, 0x2222222222222222ULL);
  const auto k4 = Set(d, 0x4444444444444444ULL);
  const auto k8 = Set(d, 0x8888888888888888ULL);
  const auto a0 = And(a, k1);
  const auto a1 = And(a, k2);
  const auto a2 = And(a, k4);
  const auto a3 = And(a, k8);
  const auto b0 = And(b, k1);
  const auto b1 = And(b, k2);
  const auto b2 = And(b, k4);
  const auto b3 = And(b, k8);

  auto m0 = Xor(MulOdd(a0, b0), MulOdd(a1, b3));
  auto m1 = Xor(MulOdd(a0, b1), MulOdd(a1, b0));
  auto m2 = Xor(MulOdd(a0, b2), MulOdd(a1, b1));
  auto m3 = Xor(MulOdd(a0, b3), MulOdd(a1, b2));
  m0 = Xor(m0, Xor(MulOdd(a2, b2), MulOdd(a3, b1)));
  m1 = Xor(m1, Xor(MulOdd(a2, b3), MulOdd(a3, b2)));
  m2 = Xor(m2, Xor(MulOdd(a2, b0), MulOdd(a3, b3)));
  m3 = Xor(m3, Xor(MulOdd(a2, b1), MulOdd(a3, b0)));
  return Or(Or(And(m0, k1), And(m1, k2)), Or(And(m2, k4), And(m3, k8)));
}

#endif  // HWY_NATIVE_AES
#endif  // HWY_TARGET != HWY_SCALAR

// "Include guard": skip if native POPCNT-related instructions are available.
#if (defined(HWY_NATIVE_POPCNT) == defined(HWY_TARGET_TOGGLE))
#ifdef HWY_NATIVE_POPCNT
#undef HWY_NATIVE_POPCNT
#else
#define HWY_NATIVE_POPCNT
#endif

#undef HWY_MIN_POW2_FOR_128
#if HWY_TARGET == HWY_RVV
#define HWY_MIN_POW2_FOR_128 1
#else
// All other targets except HWY_SCALAR (which is excluded by HWY_IF_GE128_D)
// guarantee 128 bits anyway.
#define HWY_MIN_POW2_FOR_128 0
#endif

// This algorithm requires vectors to be at least 16 bytes, which is the case
// for LMUL >= 2. If not, use the fallback below.
template <typename V, HWY_IF_LANES_ARE(uint8_t, V), HWY_IF_GE128_D(DFromV<V>),
          HWY_IF_POW2_GE(DFromV<V>, HWY_MIN_POW2_FOR_128)>
HWY_API V PopulationCount(V v) {
  const DFromV<V> d;
  HWY_ALIGN constexpr uint8_t kLookup[16] = {
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
  };
  const auto lo = And(v, Set(d, 0xF));
  const auto hi = ShiftRight<4>(v);
  const auto lookup = LoadDup128(d, kLookup);
  return Add(TableLookupBytes(lookup, hi), TableLookupBytes(lookup, lo));
}

// RVV has a specialization that avoids the Set().
#if HWY_TARGET != HWY_RVV
// Slower fallback for capped vectors.
template <typename V, HWY_IF_LANES_ARE(uint8_t, V), HWY_IF_LT128_D(DFromV<V>)>
HWY_API V PopulationCount(V v) {
  const DFromV<V> d;
  // See https://arxiv.org/pdf/1611.07612.pdf, Figure 3
  v = Sub(v, And(ShiftRight<1>(v), Set(d, 0x55)));
  v = Add(And(ShiftRight<2>(v), Set(d, 0x33)), And(v, Set(d, 0x33)));
  return And(Add(v, ShiftRight<4>(v)), Set(d, 0x0F));
}
#endif  // HWY_TARGET != HWY_RVV

template <typename V, HWY_IF_LANES_ARE(uint16_t, V)>
HWY_API V PopulationCount(V v) {
  const DFromV<V> d;
  const Repartition<uint8_t, decltype(d)> d8;
  const auto vals = BitCast(d, PopulationCount(BitCast(d8, v)));
  return Add(ShiftRight<8>(vals), And(vals, Set(d, 0xFF)));
}

template <typename V, HWY_IF_LANES_ARE(uint32_t, V)>
HWY_API V PopulationCount(V v) {
  const DFromV<V> d;
  Repartition<uint16_t, decltype(d)> d16;
  auto vals = BitCast(d, PopulationCount(BitCast(d16, v)));
  return Add(ShiftRight<16>(vals), And(vals, Set(d, 0xFF)));
}

#if HWY_HAVE_INTEGER64
template <typename V, HWY_IF_LANES_ARE(uint64_t, V)>
HWY_API V PopulationCount(V v) {
  const DFromV<V> d;
  Repartition<uint32_t, decltype(d)> d32;
  auto vals = BitCast(d, PopulationCount(BitCast(d32, v)));
  return Add(ShiftRight<32>(vals), And(vals, Set(d, 0xFF)));
}
#endif

#endif  // HWY_NATIVE_POPCNT

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
