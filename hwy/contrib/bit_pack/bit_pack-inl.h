// Copyright 2022 Google LLC
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

// Per-target include guard
#if defined(HIGHWAY_HWY_CONTRIB_BIT_PACK_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BIT_PACK_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BIT_PACK_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BIT_PACK_INL_H_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace detail {

// Primary template, specialized below for each number of bits
template <size_t kBits>  // <= 8
struct Pack8 {};

template <>
struct Pack8<1> {
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 1;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // 16-bit shifts avoid masking (bits will not cross 8-bit lanes).
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));
    const VU16 raw2 = BitCast(d16, LoadU(d8, raw + 2 * N8));
    const VU16 raw3 = BitCast(d16, LoadU(d8, raw + 3 * N8));
    const VU16 raw4 = BitCast(d16, LoadU(d8, raw + 4 * N8));
    const VU16 raw5 = BitCast(d16, LoadU(d8, raw + 5 * N8));
    const VU16 raw6 = BitCast(d16, LoadU(d8, raw + 6 * N8));
    const VU16 raw7 = BitCast(d16, LoadU(d8, raw + 7 * N8));

    const VU16 packed76 = Or(ShiftLeft<7>(raw7), ShiftLeft<6>(raw6));
    const VU16 packed54 = Or(ShiftLeft<5>(raw5), ShiftLeft<4>(raw4));
    const VU16 packed32 = Or(ShiftLeft<3>(raw3), ShiftLeft<2>(raw2));
    const VU16 packed10 = Or(ShiftLeft<1>(raw1), raw0);
    const VU16 packed = Or3(Or(packed76, packed54), packed32, packed10);
    StoreU(BitCast(d8, packed), d8, packed_out);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // We extract the lowest bit from each byte, then shift right.
    const VU16 mask = Set(d16, 0x0101u);

    VU16 packed = BitCast(d16, LoadU(d8, packed_in));

    const VU16 raw0 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    const VU16 raw3 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);

    const VU16 raw4 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw4), d8, raw + 4 * N8);

    const VU16 raw5 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw5), d8, raw + 5 * N8);

    const VU16 raw6 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(BitCast(d8, raw6), d8, raw + 6 * N8);

    const VU16 raw7 = And(packed, mask);
    StoreU(BitCast(d8, raw7), d8, raw + 7 * N8);
  }
};  // Pack<1>

template <>
struct Pack8<2> {
  static constexpr size_t kRawVectors = 4;
  static constexpr size_t kPackedVectors = 1;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // 16-bit shifts avoid masking (bits will not cross 8-bit lanes).
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));
    const VU16 raw2 = BitCast(d16, LoadU(d8, raw + 2 * N8));
    const VU16 raw3 = BitCast(d16, LoadU(d8, raw + 3 * N8));

    const VU16 packed32 = Or(ShiftLeft<6>(raw3), ShiftLeft<4>(raw2));
    const VU16 packed10 = Or(ShiftLeft<2>(raw1), raw0);
    const VU16 packed = Or(packed32, packed10);
    StoreU(BitCast(d8, packed), d8, packed_out);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // We extract the lowest two bits from each byte, then shift right.
    const VU16 mask = Set(d16, 0x0303u);

    VU16 packed = BitCast(d16, LoadU(d8, packed_in));

    const VU16 raw0 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    const VU16 raw3 = And(packed, mask);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);
  }
};  // Pack<2>

template <>
struct Pack8<3> {
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 3;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));
    const VU16 raw2 = BitCast(d16, LoadU(d8, raw + 2 * N8));
    const VU16 raw3 = BitCast(d16, LoadU(d8, raw + 3 * N8));
    const VU16 raw4 = BitCast(d16, LoadU(d8, raw + 4 * N8));
    const VU16 raw5 = BitCast(d16, LoadU(d8, raw + 5 * N8));
    const VU16 raw6 = BitCast(d16, LoadU(d8, raw + 6 * N8));
    const VU16 raw7 = BitCast(d16, LoadU(d8, raw + 7 * N8));

    // The upper two bits of these three will be filled with packed3 (6 bits).
    VU16 packed0 = Or(ShiftLeft<3>(raw4), raw0);
    VU16 packed1 = Or(ShiftLeft<3>(raw5), raw1);
    VU16 packed2 = Or(ShiftLeft<3>(raw6), raw2);
    const VU16 packed3 = Or(ShiftLeft<3>(raw7), raw3);

    const VU16 hi2 = Set(d16, 0xC0C0u);
    packed0 = OrAnd(packed0, ShiftLeft<2>(packed3), hi2);
    packed1 = OrAnd(packed1, ShiftLeft<4>(packed3), hi2);
    packed2 = OrAnd(packed2, ShiftLeft<6>(packed3), hi2);
    StoreU(BitCast(d8, packed0), d8, packed_out + 0 * N8);
    StoreU(BitCast(d8, packed1), d8, packed_out + 1 * N8);
    StoreU(BitCast(d8, packed2), d8, packed_out + 2 * N8);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // We extract the lowest three bits from each byte.
    const VU16 mask = Set(d16, 0x0707u);

    VU16 packed0 = BitCast(d16, LoadU(d8, packed_in + 0 * N8));
    VU16 packed1 = BitCast(d16, LoadU(d8, packed_in + 1 * N8));
    VU16 packed2 = BitCast(d16, LoadU(d8, packed_in + 2 * N8));

    const VU16 raw0 = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    const VU16 raw4 = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(BitCast(d8, raw4), d8, raw + 4 * N8);

    const VU16 raw5 = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(BitCast(d8, raw5), d8, raw + 5 * N8);

    const VU16 raw6 = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(BitCast(d8, raw6), d8, raw + 6 * N8);

    // packed3 is the concatenation of the low two bits in packed0..2.
    const VU16 mask2 = Set(d16, 0x0303u);
    VU16 packed3 = And(mask2, packed0);  // high bits in lower two bits
    packed3 = ShiftLeft<2>(packed3);
    packed3 = OrAnd(packed3, mask2, packed1);  // insert mid 2 bits
    packed3 = ShiftLeft<2>(packed3);
    packed3 = OrAnd(packed3, mask2, packed2);  // insert low 2 bits

    const VU16 raw3 = And(mask, packed3);
    packed3 = ShiftRight<3>(packed3);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);

    const VU16 raw7 = And(mask, packed3);
    StoreU(BitCast(d8, raw7), d8, raw + 7 * N8);
  }
};  // Pack<3>

template <>
struct Pack8<4> {
  static constexpr size_t kRawVectors = 2;
  static constexpr size_t kPackedVectors = 1;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // 16-bit shifts avoid masking (bits will not cross 8-bit lanes).
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));

    const VU16 packed = Or(ShiftLeft<4>(raw1), raw0);
    StoreU(BitCast(d8, packed), d8, packed_out);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // We extract the lowest four bits from each byte, then shift right.
    const VU16 mask = Set(d16, 0x0F0Fu);

    VU16 packed = BitCast(d16, LoadU(d8, packed_in));

    const VU16 raw0 = And(packed, mask);
    packed = ShiftRight<4>(packed);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = And(packed, mask);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);
  }
};  // Pack<4>

template <>
struct Pack8<5> {
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 5;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));
    const VU16 raw2 = BitCast(d16, LoadU(d8, raw + 2 * N8));
    const VU16 raw3 = BitCast(d16, LoadU(d8, raw + 3 * N8));
    const VU16 raw4 = BitCast(d16, LoadU(d8, raw + 4 * N8));
    const VU16 raw5 = BitCast(d16, LoadU(d8, raw + 5 * N8));
    const VU16 raw6 = BitCast(d16, LoadU(d8, raw + 6 * N8));
    const VU16 raw7 = BitCast(d16, LoadU(d8, raw + 7 * N8));

    // Fill upper three bits with upper bits from raw4..7.
    const VU16 hi3 = Set(d16, 0xE0E0u);
    const VU16 packed0 = OrAnd(raw0, ShiftLeft<3>(raw4), hi3);
    const VU16 packed1 = OrAnd(raw1, ShiftLeft<3>(raw5), hi3);
    const VU16 packed2 = OrAnd(raw2, ShiftLeft<3>(raw6), hi3);
    const VU16 packed3 = OrAnd(raw3, ShiftLeft<3>(raw7), hi3);
    StoreU(BitCast(d8, packed0), d8, packed_out + 0 * N8);
    StoreU(BitCast(d8, packed1), d8, packed_out + 1 * N8);
    StoreU(BitCast(d8, packed2), d8, packed_out + 2 * N8);
    StoreU(BitCast(d8, packed3), d8, packed_out + 3 * N8);

    // Combine lower two bits of raw4..7 into packed4.
    const VU16 lo2 = Set(d16, 0x0303u);
    VU16 packed4 = And(lo2, raw7);
    packed4 = ShiftLeft<2>(packed4);
    packed4 = OrAnd(packed4, lo2, raw6);
    packed4 = ShiftLeft<2>(packed4);
    packed4 = OrAnd(packed4, lo2, raw5);
    packed4 = ShiftLeft<2>(packed4);
    packed4 = OrAnd(packed4, lo2, raw4);
    StoreU(BitCast(d8, packed4), d8, packed_out + 4 * N8);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);

    const VU16 packed0 = BitCast(d16, LoadU(d8, packed_in + 0 * N8));
    const VU16 packed1 = BitCast(d16, LoadU(d8, packed_in + 1 * N8));
    const VU16 packed2 = BitCast(d16, LoadU(d8, packed_in + 2 * N8));
    const VU16 packed3 = BitCast(d16, LoadU(d8, packed_in + 3 * N8));
    VU16 packed4 = BitCast(d16, LoadU(d8, packed_in + 4 * N8));

    // We extract the lowest five bits from each byte.
    const VU16 hi3 = Set(d16, 0xE0E0u);

    const VU16 raw0 = AndNot(hi3, packed0);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = AndNot(hi3, packed1);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = AndNot(hi3, packed2);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    const VU16 raw3 = AndNot(hi3, packed3);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);

    // The upper bits are the top 3 bits shifted right by three.
    const VU16 top4 = ShiftRight<3>(And(hi3, packed0));
    const VU16 top5 = ShiftRight<3>(And(hi3, packed1));
    const VU16 top6 = ShiftRight<3>(And(hi3, packed2));
    const VU16 top7 = ShiftRight<3>(And(hi3, packed3));

    // Insert the lower 2 bits, which were concatenated into a byte.
    const VU16 lo2 = Set(d16, 0x0303u);
    const VU16 raw4 = OrAnd(top4, lo2, packed4);
    packed4 = ShiftRight<2>(packed4);
    const VU16 raw5 = OrAnd(top5, lo2, packed4);
    packed4 = ShiftRight<2>(packed4);
    const VU16 raw6 = OrAnd(top6, lo2, packed4);
    packed4 = ShiftRight<2>(packed4);
    const VU16 raw7 = OrAnd(top7, lo2, packed4);

    StoreU(BitCast(d8, raw4), d8, raw + 4 * N8);
    StoreU(BitCast(d8, raw5), d8, raw + 5 * N8);
    StoreU(BitCast(d8, raw6), d8, raw + 6 * N8);
    StoreU(BitCast(d8, raw7), d8, raw + 7 * N8);
  }
};  // Pack<5>

template <>
struct Pack8<6> {
  static constexpr size_t kRawVectors = 4;
  static constexpr size_t kPackedVectors = 3;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));
    const VU16 raw2 = BitCast(d16, LoadU(d8, raw + 2 * N8));
    const VU16 raw3 = BitCast(d16, LoadU(d8, raw + 3 * N8));

    // The upper two bits of these three will be filled with raw3 (6 bits).
    VU16 packed0 = raw0;
    VU16 packed1 = raw1;
    VU16 packed2 = raw2;

    const VU16 hi2 = Set(d16, 0xC0C0u);
    packed0 = OrAnd(packed0, ShiftLeft<2>(raw3), hi2);
    packed1 = OrAnd(packed1, ShiftLeft<4>(raw3), hi2);
    packed2 = OrAnd(packed2, ShiftLeft<6>(raw3), hi2);
    StoreU(BitCast(d8, packed0), d8, packed_out + 0 * N8);
    StoreU(BitCast(d8, packed1), d8, packed_out + 1 * N8);
    StoreU(BitCast(d8, packed2), d8, packed_out + 2 * N8);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // We extract the lowest six bits from each byte. Negated mask so we can
    // use OrAnd below.
    const VU16 mask = Set(d16, 0xC0C0u);

    const VU16 packed0 = BitCast(d16, LoadU(d8, packed_in + 0 * N8));
    const VU16 packed1 = BitCast(d16, LoadU(d8, packed_in + 1 * N8));
    const VU16 packed2 = BitCast(d16, LoadU(d8, packed_in + 2 * N8));

    const VU16 raw0 = AndNot(mask, packed0);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = AndNot(mask, packed1);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = AndNot(mask, packed2);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    // raw3 is the concatenation of the upper two bits in packed0..2.
    VU16 raw3 = And(mask, packed2);  // low 2 bits in top of byte
    raw3 = ShiftRight<2>(raw3);
    raw3 = OrAnd(raw3, mask, packed1);  // insert mid 2 bits
    raw3 = ShiftRight<2>(raw3);
    raw3 = OrAnd(raw3, mask, packed0);  // insert high 2 bits
    raw3 = ShiftRight<2>(raw3);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);
  }
};  // Pack<6>

template <>
struct Pack8<7> {
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 7;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    const VU16 raw0 = BitCast(d16, LoadU(d8, raw + 0 * N8));
    const VU16 raw1 = BitCast(d16, LoadU(d8, raw + 1 * N8));
    const VU16 raw2 = BitCast(d16, LoadU(d8, raw + 2 * N8));
    const VU16 raw3 = BitCast(d16, LoadU(d8, raw + 3 * N8));
    const VU16 raw4 = BitCast(d16, LoadU(d8, raw + 4 * N8));
    const VU16 raw5 = BitCast(d16, LoadU(d8, raw + 5 * N8));
    const VU16 raw6 = BitCast(d16, LoadU(d8, raw + 6 * N8));
    // Top bit is inserted into packed0..6 and then shifted left.
    VU16 raw7 = BitCast(d16, LoadU(d8, raw + 7 * N8));

    const VU16 hi1 = Set(d16, 0x8080u);
    raw7 = Add(raw7, raw7);
    const VU16 packed0 = OrAnd(raw0, hi1, raw7);
    raw7 = Add(raw7, raw7);
    const VU16 packed1 = OrAnd(raw1, hi1, raw7);
    raw7 = Add(raw7, raw7);
    const VU16 packed2 = OrAnd(raw2, hi1, raw7);
    raw7 = Add(raw7, raw7);
    const VU16 packed3 = OrAnd(raw3, hi1, raw7);
    raw7 = Add(raw7, raw7);
    const VU16 packed4 = OrAnd(raw4, hi1, raw7);
    raw7 = Add(raw7, raw7);
    const VU16 packed5 = OrAnd(raw5, hi1, raw7);
    raw7 = Add(raw7, raw7);
    const VU16 packed6 = OrAnd(raw6, hi1, raw7);
    StoreU(BitCast(d8, packed0), d8, packed_out + 0 * N8);
    StoreU(BitCast(d8, packed1), d8, packed_out + 1 * N8);
    StoreU(BitCast(d8, packed2), d8, packed_out + 2 * N8);
    StoreU(BitCast(d8, packed3), d8, packed_out + 3 * N8);
    StoreU(BitCast(d8, packed4), d8, packed_out + 4 * N8);
    StoreU(BitCast(d8, packed5), d8, packed_out + 5 * N8);
    StoreU(BitCast(d8, packed6), d8, packed_out + 6 * N8);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);

    const VU16 packed0 = BitCast(d16, LoadU(d8, packed_in + 0 * N8));
    const VU16 packed1 = BitCast(d16, LoadU(d8, packed_in + 1 * N8));
    const VU16 packed2 = BitCast(d16, LoadU(d8, packed_in + 2 * N8));
    const VU16 packed3 = BitCast(d16, LoadU(d8, packed_in + 3 * N8));
    const VU16 packed4 = BitCast(d16, LoadU(d8, packed_in + 4 * N8));
    const VU16 packed5 = BitCast(d16, LoadU(d8, packed_in + 5 * N8));
    const VU16 packed6 = BitCast(d16, LoadU(d8, packed_in + 6 * N8));

    // We extract the lowest seven bits from each byte.
    const VU16 hi1 = Set(d16, 0x8080u);

    const VU16 raw0 = AndNot(hi1, packed0);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = AndNot(hi1, packed1);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = AndNot(hi1, packed2);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    const VU16 raw3 = AndNot(hi1, packed3);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);

    const VU16 raw4 = AndNot(hi1, packed4);
    StoreU(BitCast(d8, raw4), d8, raw + 4 * N8);

    const VU16 raw5 = AndNot(hi1, packed5);
    StoreU(BitCast(d8, raw5), d8, raw + 5 * N8);

    const VU16 raw6 = AndNot(hi1, packed6);
    StoreU(BitCast(d8, raw6), d8, raw + 6 * N8);

    VU16 raw7 = And(hi1, packed6);  // will shift this down into LSB
    raw7 = ShiftRight<1>(raw7);
    raw7 = OrAnd(raw7, hi1, packed5);
    raw7 = ShiftRight<1>(raw7);
    raw7 = OrAnd(raw7, hi1, packed4);
    raw7 = ShiftRight<1>(raw7);
    raw7 = OrAnd(raw7, hi1, packed3);
    raw7 = ShiftRight<1>(raw7);
    raw7 = OrAnd(raw7, hi1, packed2);
    raw7 = ShiftRight<1>(raw7);
    raw7 = OrAnd(raw7, hi1, packed1);
    raw7 = ShiftRight<1>(raw7);
    raw7 = OrAnd(raw7, hi1, packed0);
    raw7 = ShiftRight<1>(raw7);
    StoreU(BitCast(d8, raw7), d8, raw + 7 * N8);
  }
};  // Pack<7>

template <>
struct Pack8<8> {
  static constexpr size_t kRawVectors = 1;
  static constexpr size_t kPackedVectors = 1;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    StoreU(LoadU(d8, raw), d8, packed_out);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    StoreU(LoadU(d8, packed_in), d8, raw);
  }
};  // Pack<8>

}  // namespace detail

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_BIT_PACK_INL_H_
