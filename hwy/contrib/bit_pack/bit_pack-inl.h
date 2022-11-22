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
  static constexpr size_t kBits = 1;
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
};  // Pack8<1>

template <>
struct Pack8<2> {
  static constexpr size_t kBits = 2;
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
};  // Pack8<2>

template <>
struct Pack8<3> {
  static constexpr size_t kBits = 3;
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
};  // Pack8<3>

template <>
struct Pack8<4> {
  static constexpr size_t kBits = 4;
  // 2x unrolled (matches size of 2/6 bit cases) for increased efficiency.
  static constexpr size_t kRawVectors = 4;
  static constexpr size_t kPackedVectors = 2;

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

    const VU16 packed0 = Or(ShiftLeft<4>(raw1), raw0);
    const VU16 packed1 = Or(ShiftLeft<4>(raw3), raw2);
    StoreU(BitCast(d8, packed0), d8, packed_out + 0 * N8);
    StoreU(BitCast(d8, packed1), d8, packed_out + 1 * N8);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    const RepartitionToWide<decltype(d8)> d16;
    using VU16 = Vec<decltype(d16)>;
    const size_t N8 = Lanes(d8);
    // We extract the lowest four bits from each byte, then shift right.
    const VU16 mask = Set(d16, 0x0F0Fu);

    VU16 packed0 = BitCast(d16, LoadU(d8, packed_in + 0 * N8));
    VU16 packed1 = BitCast(d16, LoadU(d8, packed_in + 1 * N8));

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<4>(packed0);
    StoreU(BitCast(d8, raw0), d8, raw + 0 * N8);

    const VU16 raw1 = And(packed0, mask);
    StoreU(BitCast(d8, raw1), d8, raw + 1 * N8);

    const VU16 raw2 = And(packed1, mask);
    packed1 = ShiftRight<4>(packed1);
    StoreU(BitCast(d8, raw2), d8, raw + 2 * N8);

    const VU16 raw3 = And(packed1, mask);
    StoreU(BitCast(d8, raw3), d8, raw + 3 * N8);
  }
};  // Pack8<4>

template <>
struct Pack8<5> {
  static constexpr size_t kBits = 5;
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
};  // Pack8<5>

template <>
struct Pack8<6> {
  static constexpr size_t kBits = 6;
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
};  // Pack8<6>

template <>
struct Pack8<7> {
  static constexpr size_t kBits = 7;
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
};  // Pack8<7>

template <>
struct Pack8<8> {
  static constexpr size_t kBits = 8;
  // 4x unrolled (matches size of 2/6 bit cases) for increased efficiency.
  static constexpr size_t kRawVectors = 4;
  static constexpr size_t kPackedVectors = 4;

  template <class D8>
  HWY_INLINE void Pack(D8 d8, const uint8_t* HWY_RESTRICT raw,
                       uint8_t* HWY_RESTRICT packed_out) {
    using VU8 = Vec<decltype(d8)>;
    const size_t N8 = Lanes(d8);
    const VU8 raw0 = LoadU(d8, raw + 0 * N8);
    const VU8 raw1 = LoadU(d8, raw + 1 * N8);
    const VU8 raw2 = LoadU(d8, raw + 2 * N8);
    const VU8 raw3 = LoadU(d8, raw + 3 * N8);

    StoreU(raw0, d8, packed_out + 0 * N8);
    StoreU(raw1, d8, packed_out + 1 * N8);
    StoreU(raw2, d8, packed_out + 2 * N8);
    StoreU(raw3, d8, packed_out + 3 * N8);
  }

  template <class D8>
  HWY_INLINE void Unpack(D8 d8, const uint8_t* HWY_RESTRICT packed_in,
                         uint8_t* HWY_RESTRICT raw) {
    using VU8 = Vec<decltype(d8)>;
    const size_t N8 = Lanes(d8);
    const VU8 raw0 = LoadU(d8, packed_in + 0 * N8);
    const VU8 raw1 = LoadU(d8, packed_in + 1 * N8);
    const VU8 raw2 = LoadU(d8, packed_in + 2 * N8);
    const VU8 raw3 = LoadU(d8, packed_in + 3 * N8);

    StoreU(raw0, d8, raw + 0 * N8);
    StoreU(raw1, d8, raw + 1 * N8);
    StoreU(raw2, d8, raw + 2 * N8);
    StoreU(raw3, d8, raw + 3 * N8);
  }
};  // Pack8<8>

// Primary template, specialized below for each number of bits
template <size_t kBits>  // <= 16
struct Pack16 {};

template <>
struct Pack16<1> {
  static constexpr size_t kBits = 1;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 1;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    const VU16 p0 = Or3(ShiftLeft<2>(raw2), Add(raw1, raw1), raw0);
    const VU16 p1 =
        Or3(ShiftLeft<5>(raw5), ShiftLeft<4>(raw4), ShiftLeft<3>(raw3));
    const VU16 p2 =
        Or3(ShiftLeft<8>(raw8), ShiftLeft<7>(raw7), ShiftLeft<6>(raw6));
    const VU16 p3 =
        Or3(ShiftLeft<0xB>(rawB), ShiftLeft<0xA>(rawA), ShiftLeft<9>(raw9));
    const VU16 p4 =
        Or3(ShiftLeft<0xE>(rawE), ShiftLeft<0xD>(rawD), ShiftLeft<0xC>(rawC));
    const VU16 p5 = Or3(p1, p0, ShiftLeft<0xF>(rawF));
    const VU16 p6 = Or3(p2, p3, p4);
    const VU16 packed = Or(p5, p6);
    StoreU(packed, d, packed_out);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    // We extract the lowest bit from each u16, then shift right.
    const VU16 mask = Set(d, 1u);

    VU16 packed = LoadU(d, packed_in);

    const VU16 raw0 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 raw8 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw8, d, raw + 8 * N);

    const VU16 raw9 = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(raw9, d, raw + 9 * N);

    const VU16 rawA = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(rawA, d, raw + 0xA * N);

    const VU16 rawB = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(rawB, d, raw + 0xB * N);

    const VU16 rawC = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(rawC, d, raw + 0xC * N);

    const VU16 rawD = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(rawD, d, raw + 0xD * N);

    const VU16 rawE = And(packed, mask);
    packed = ShiftRight<1>(packed);
    StoreU(rawE, d, raw + 0xE * N);

    const VU16 rawF = And(packed, mask);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<1>

template <>
struct Pack16<2> {
  static constexpr size_t kBits = 2;
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 1;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);

    const VU16 p0 = Or3(ShiftLeft<4>(raw2), ShiftLeft<2>(raw1), raw0);
    const VU16 p1 =
        Or3(ShiftLeft<10>(raw5), ShiftLeft<8>(raw4), ShiftLeft<6>(raw3));
    const VU16 p2 = Or3(p0, ShiftLeft<14>(raw7), ShiftLeft<12>(raw6));
    const VU16 packed = Or(p1, p2);
    StoreU(packed, d, packed_out);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    // We extract the lowest two bits, then shift right.
    const VU16 mask = Set(d, 0x3u);

    VU16 packed = LoadU(d, packed_in);

    const VU16 raw0 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(packed, mask);
    packed = ShiftRight<2>(packed);
    StoreU(raw7, d, raw + 7 * N);
  }
};  // Pack16<2>

template <>
struct Pack16<3> {
  static constexpr size_t kBits = 3;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 3;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    // We can fit 15 raw vectors in three packed vectors (five each).
    const VU16 raw630 = Or3(ShiftLeft<6>(raw6), ShiftLeft<3>(raw3), raw0);
    const VU16 raw741 = Or3(ShiftLeft<6>(raw7), ShiftLeft<3>(raw4), raw1);
    const VU16 raw852 = Or3(ShiftLeft<6>(raw8), ShiftLeft<3>(raw5), raw2);

    // rawF will be scattered into the upper bits of these three.
    VU16 packed0 = Or3(raw630, ShiftLeft<12>(rawC), ShiftLeft<9>(raw9));
    VU16 packed1 = Or3(raw741, ShiftLeft<12>(rawD), ShiftLeft<9>(rawA));
    VU16 packed2 = Or3(raw852, ShiftLeft<12>(rawE), ShiftLeft<9>(rawB));

    const VU16 hi1 = Set(d, 0x8000u);
    packed0 = Or(packed0, ShiftLeft<15>(rawF));  // MSB only, no mask
    packed1 = OrAnd(packed1, ShiftLeft<14>(rawF), hi1);
    packed2 = OrAnd(packed2, ShiftLeft<13>(rawF), hi1);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    // We extract the lowest three bits.
    const VU16 mask = Set(d, 0x7u);

    VU16 packed0 = LoadU(d, packed_in + 0 * N);
    VU16 packed1 = LoadU(d, packed_in + 1 * N);
    VU16 packed2 = LoadU(d, packed_in + 2 * N);

    const VU16 raw0 = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 raw8 = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(raw8, d, raw + 8 * N);

    const VU16 raw9 = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(raw9, d, raw + 9 * N);

    const VU16 rawA = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(rawA, d, raw + 0xA * N);

    const VU16 rawB = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(rawB, d, raw + 0xB * N);

    const VU16 rawC = And(mask, packed0);
    packed0 = ShiftRight<3>(packed0);
    StoreU(rawC, d, raw + 0xC * N);

    const VU16 rawD = And(mask, packed1);
    packed1 = ShiftRight<3>(packed1);
    StoreU(rawD, d, raw + 0xD * N);

    const VU16 rawE = And(mask, packed2);
    packed2 = ShiftRight<3>(packed2);
    StoreU(rawE, d, raw + 0xE * N);

    // rawF is the concatenation of the lower bit of packed0..2. No masking is
    // required because we have shifted that bit downward from the MSB.
    const VU16 rawF =
        Or3(ShiftLeft<2>(packed2), Add(packed1, packed1), packed0);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<3>

template <>
struct Pack16<4> {
  static constexpr size_t kBits = 4;
  // 2x unrolled (matches size of 2/6 bit cases) for increased efficiency.
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 2;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = BitCast(d, LoadU(d, raw + 0 * N));
    const VU16 raw1 = BitCast(d, LoadU(d, raw + 1 * N));
    const VU16 raw2 = BitCast(d, LoadU(d, raw + 2 * N));
    const VU16 raw3 = BitCast(d, LoadU(d, raw + 3 * N));
    const VU16 raw4 = BitCast(d, LoadU(d, raw + 4 * N));
    const VU16 raw5 = BitCast(d, LoadU(d, raw + 5 * N));
    const VU16 raw6 = BitCast(d, LoadU(d, raw + 6 * N));
    const VU16 raw7 = BitCast(d, LoadU(d, raw + 7 * N));

    const VU16 raw20 = Or3(ShiftLeft<8>(raw2), ShiftLeft<4>(raw1), raw0);
    const VU16 packed0 = Or(raw20, ShiftLeft<12>(raw3));
    const VU16 raw64 = Or3(ShiftLeft<8>(raw6), ShiftLeft<4>(raw5), raw4);
    const VU16 packed1 = Or(raw64, ShiftLeft<12>(raw7));
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    // We extract the lowest four bits, then shift right.
    const VU16 mask = Set(d, 0xFu);

    VU16 packed0 = LoadU(d, packed_in + 0 * N);
    VU16 packed1 = LoadU(d, packed_in + 1 * N);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<4>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed0, mask);
    packed0 = ShiftRight<4>(packed0);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed0, mask);
    packed0 = ShiftRight<4>(packed0);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = packed0;  // shifted down, no mask required
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed1, mask);
    packed1 = ShiftRight<4>(packed1);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed1, mask);
    packed1 = ShiftRight<4>(packed1);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed1, mask);
    packed1 = ShiftRight<4>(packed1);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = packed1;  // shifted down, no mask required
    StoreU(raw7, d, raw + 7 * N);
  }
};  // Pack16<4>

template <>
struct Pack16<5> {
  static constexpr size_t kBits = 5;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 5;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    // We can fit 15 raw vectors in five packed vectors (three each).
    const VU16 rawA50 = Or3(ShiftLeft<10>(rawA), ShiftLeft<5>(raw5), raw0);
    const VU16 rawB61 = Or3(ShiftLeft<10>(rawB), ShiftLeft<5>(raw6), raw1);
    const VU16 rawC72 = Or3(ShiftLeft<10>(rawC), ShiftLeft<5>(raw7), raw2);
    const VU16 rawD83 = Or3(ShiftLeft<10>(rawD), ShiftLeft<5>(raw8), raw3);
    const VU16 rawE94 = Or3(ShiftLeft<10>(rawE), ShiftLeft<5>(raw9), raw4);

    // rawF will be scattered into the upper bits of these five.
    const VU16 hi1 = Set(d, 0x8000u);
    const VU16 packed0 = Or(rawA50, ShiftLeft<15>(rawF));  // MSB only, no mask
    const VU16 packed1 = OrAnd(rawB61, ShiftLeft<14>(rawF), hi1);
    const VU16 packed2 = OrAnd(rawC72, ShiftLeft<13>(rawF), hi1);
    const VU16 packed3 = OrAnd(rawD83, ShiftLeft<12>(rawF), hi1);
    const VU16 packed4 = OrAnd(rawE94, ShiftLeft<11>(rawF), hi1);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = LoadU(d, packed_in + 0 * N);
    VU16 packed1 = LoadU(d, packed_in + 1 * N);
    VU16 packed2 = LoadU(d, packed_in + 2 * N);
    VU16 packed3 = LoadU(d, packed_in + 3 * N);
    VU16 packed4 = LoadU(d, packed_in + 4 * N);

    // We extract the lowest five bits and shift right.
    const VU16 mask = Set(d, 0x1Fu);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<5>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed1, mask);
    packed1 = ShiftRight<5>(packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed2, mask);
    packed2 = ShiftRight<5>(packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed3, mask);
    packed3 = ShiftRight<5>(packed3);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed4, mask);
    packed4 = ShiftRight<5>(packed4);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed0, mask);
    packed0 = ShiftRight<5>(packed0);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed1, mask);
    packed1 = ShiftRight<5>(packed1);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(packed2, mask);
    packed2 = ShiftRight<5>(packed2);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 raw8 = And(packed3, mask);
    packed3 = ShiftRight<5>(packed3);
    StoreU(raw8, d, raw + 8 * N);

    const VU16 raw9 = And(packed4, mask);
    packed4 = ShiftRight<5>(packed4);
    StoreU(raw9, d, raw + 9 * N);

    const VU16 rawA = And(packed0, mask);
    packed0 = ShiftRight<5>(packed0);
    StoreU(rawA, d, raw + 0xA * N);

    const VU16 rawB = And(packed1, mask);
    packed1 = ShiftRight<5>(packed1);
    StoreU(rawB, d, raw + 0xB * N);

    const VU16 rawC = And(packed2, mask);
    packed2 = ShiftRight<5>(packed2);
    StoreU(rawC, d, raw + 0xC * N);

    const VU16 rawD = And(packed3, mask);
    packed3 = ShiftRight<5>(packed3);
    StoreU(rawD, d, raw + 0xD * N);

    const VU16 rawE = And(packed4, mask);
    packed4 = ShiftRight<5>(packed4);
    StoreU(rawE, d, raw + 0xE * N);

    // rawF is the concatenation of the lower bit of packed0..4. No masking is
    // required because we have shifted that bit downward from the MSB.
    const VU16 p0 = Or3(ShiftLeft<2>(packed2), Add(packed1, packed1), packed0);
    const VU16 rawF = Or3(ShiftLeft<4>(packed4), ShiftLeft<3>(packed3), p0);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<5>

template <>
struct Pack16<6> {
  static constexpr size_t kBits = 6;
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 3;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);

    const VU16 packed3 = Or(ShiftLeft<6>(raw7), raw3);
    // Three vectors, two 6-bit raw each; packed3 (12 bits) is spread over the
    // four remainder bits at the top of each vector.
    const VU16 packed0 = Or3(ShiftLeft<12>(packed3), ShiftLeft<6>(raw4), raw0);
    VU16 packed1 = Or(ShiftLeft<6>(raw5), raw1);
    VU16 packed2 = Or(ShiftLeft<6>(raw6), raw2);

    const VU16 hi4 = Set(d, 0xF000u);
    packed1 = OrAnd(packed1, ShiftLeft<8>(packed3), hi4);
    packed2 = OrAnd(packed2, ShiftLeft<4>(packed3), hi4);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    // We extract the lowest six bits and shift right.
    const VU16 mask = Set(d, 0x3Fu);

    VU16 packed0 = LoadU(d, packed_in + 0 * N);
    VU16 packed1 = LoadU(d, packed_in + 1 * N);
    VU16 packed2 = LoadU(d, packed_in + 2 * N);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<6>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed1, mask);
    packed1 = ShiftRight<6>(packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed2, mask);
    packed2 = ShiftRight<6>(packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw4 = And(packed0, mask);
    packed0 = ShiftRight<6>(packed0);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed1, mask);
    packed1 = ShiftRight<6>(packed1);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed2, mask);
    packed2 = ShiftRight<6>(packed2);
    StoreU(raw6, d, raw + 6 * N);

    // packed3 is the concatenation of the four bits in packed0..2.
    VU16 packed3 = Or3(ShiftLeft<8>(packed2), ShiftLeft<4>(packed1), packed0);
    const VU16 raw3 = And(packed3, mask);
    packed3 = ShiftRight<6>(packed3);
    StoreU(raw3, d, raw + 3 * N);
    const VU16 raw7 = packed3;  // upper bits already zero
    StoreU(raw7, d, raw + 7 * N);
  }
};  // Pack16<6>

template <>
struct Pack16<7> {
  static constexpr size_t kBits = 7;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 7;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    const VU16 packed7 = Or(ShiftLeft<7>(rawF), raw7);
    // Seven vectors, two 7-bit raw each; packed7 (14 bits) is spread over the
    // two remainder bits at the top of each vector.
    const VU16 packed0 = Or3(ShiftLeft<14>(packed7), ShiftLeft<7>(raw8), raw0);
    VU16 packed1 = Or(ShiftLeft<7>(raw9), raw1);
    VU16 packed2 = Or(ShiftLeft<7>(rawA), raw2);
    VU16 packed3 = Or(ShiftLeft<7>(rawB), raw3);
    VU16 packed4 = Or(ShiftLeft<7>(rawC), raw4);
    VU16 packed5 = Or(ShiftLeft<7>(rawD), raw5);
    VU16 packed6 = Or(ShiftLeft<7>(rawE), raw6);

    const VU16 hi2 = Set(d, 0xC000u);
    packed1 = OrAnd(packed1, ShiftLeft<12>(packed7), hi2);
    packed2 = OrAnd(packed2, ShiftLeft<10>(packed7), hi2);
    packed3 = OrAnd(packed3, ShiftLeft<8>(packed7), hi2);
    packed4 = OrAnd(packed4, ShiftLeft<6>(packed7), hi2);
    packed5 = OrAnd(packed5, ShiftLeft<4>(packed7), hi2);
    packed6 = OrAnd(packed6, ShiftLeft<2>(packed7), hi2);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));

    // We extract the lowest seven bits and shift right.
    const VU16 mask = Set(d, 0x7Fu);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<7>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed1, mask);
    packed1 = ShiftRight<7>(packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed2, mask);
    packed2 = ShiftRight<7>(packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed3, mask);
    packed3 = ShiftRight<7>(packed3);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed4, mask);
    packed4 = ShiftRight<7>(packed4);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed5, mask);
    packed5 = ShiftRight<7>(packed5);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed6, mask);
    packed6 = ShiftRight<7>(packed6);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw8 = And(packed0, mask);
    packed0 = ShiftRight<7>(packed0);
    StoreU(raw8, d, raw + 8 * N);

    const VU16 raw9 = And(packed1, mask);
    packed1 = ShiftRight<7>(packed1);
    StoreU(raw9, d, raw + 9 * N);

    const VU16 rawA = And(packed2, mask);
    packed2 = ShiftRight<7>(packed2);
    StoreU(rawA, d, raw + 0xA * N);

    const VU16 rawB = And(packed3, mask);
    packed3 = ShiftRight<7>(packed3);
    StoreU(rawB, d, raw + 0xB * N);

    const VU16 rawC = And(packed4, mask);
    packed4 = ShiftRight<7>(packed4);
    StoreU(rawC, d, raw + 0xC * N);

    const VU16 rawD = And(packed5, mask);
    packed5 = ShiftRight<7>(packed5);
    StoreU(rawD, d, raw + 0xD * N);

    const VU16 rawE = And(packed6, mask);
    packed6 = ShiftRight<7>(packed6);
    StoreU(rawE, d, raw + 0xE * N);

    // packed7 is the concatenation of the two bits in packed0..6.
    const VU16 p0 = Or3(ShiftLeft<4>(packed2), ShiftLeft<2>(packed1), packed0);
    const VU16 p1 = Or3(ShiftLeft<10>(packed5), ShiftLeft<8>(packed4),
                        ShiftLeft<6>(packed3));
    VU16 packed7 = Or3(ShiftLeft<12>(packed6), p1, p0);
    const VU16 raw7 = And(packed7, mask);
    packed7 = ShiftRight<7>(packed7);
    StoreU(raw7, d, raw + 7 * N);
    const VU16 rawF = packed7;  // upper bits already zero
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<7>

template <>
struct Pack16<8> {
  static constexpr size_t kBits = 8;
  // 4x unrolled (matches size of 2/6 bit cases) for increased efficiency.
  static constexpr size_t kRawVectors = 8;
  static constexpr size_t kPackedVectors = 4;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    // This is equivalent to ConcatEven with 8-bit lanes, but much more
    // efficient on RVV and slightly less efficient on SVE2.
    const VU16 packed0 = Or(ShiftLeft<8>(raw1), raw0);
    const VU16 packed1 = Or(ShiftLeft<8>(raw3), raw2);
    const VU16 packed2 = Or(ShiftLeft<8>(raw5), raw4);
    const VU16 packed3 = Or(ShiftLeft<8>(raw7), raw6);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    // We extract the lowest eight bits and shift right.
    const VU16 mask = Set(d, 0xFFu);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<8>(packed0);
    StoreU(raw0, d, raw + 0 * N);
    const VU16 raw1 = packed0;  // upper bits already zero
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed1, mask);
    packed1 = ShiftRight<8>(packed1);
    StoreU(raw2, d, raw + 2 * N);
    const VU16 raw3 = packed1;  // upper bits already zero
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed2, mask);
    packed2 = ShiftRight<8>(packed2);
    StoreU(raw4, d, raw + 4 * N);
    const VU16 raw5 = packed2;  // upper bits already zero
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed3, mask);
    packed3 = ShiftRight<8>(packed3);
    StoreU(raw6, d, raw + 6 * N);
    const VU16 raw7 = packed3;  // upper bits already zero
    StoreU(raw7, d, raw + 7 * N);
  }
};  // Pack16<8>

template <>
struct Pack16<9> {
  static constexpr size_t kBits = 9;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 9;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);
    // 8 vectors, each with 9+7 bits; top 2 bits are concatenated into packed8.
    const VU16 packed0 = Or(ShiftLeft<9>(raw8), raw0);
    const VU16 packed1 = Or(ShiftLeft<9>(raw9), raw1);
    const VU16 packed2 = Or(ShiftLeft<9>(rawA), raw2);
    const VU16 packed3 = Or(ShiftLeft<9>(rawB), raw3);
    const VU16 packed4 = Or(ShiftLeft<9>(rawC), raw4);
    const VU16 packed5 = Or(ShiftLeft<9>(rawD), raw5);
    const VU16 packed6 = Or(ShiftLeft<9>(rawE), raw6);
    const VU16 packed7 = Or(ShiftLeft<9>(rawF), raw7);

    // We could shift down, OR and shift up, but two shifts are typically more
    // expensive than AND, shift into position, and OR (which can be further
    // reduced via Or3).
    const VU16 mid2 = Set(d, 0x180u);  // top 2 in lower 9
    const VU16 part8 = ShiftRight<7>(And(raw8, mid2));
    const VU16 part9 = ShiftRight<5>(And(raw9, mid2));
    const VU16 partA = ShiftRight<3>(And(rawA, mid2));
    const VU16 partB = ShiftRight<1>(And(rawB, mid2));
    const VU16 partC = ShiftLeft<1>(And(rawC, mid2));
    const VU16 partD = ShiftLeft<3>(And(rawD, mid2));
    const VU16 partE = ShiftLeft<5>(And(rawE, mid2));
    const VU16 partF = ShiftLeft<7>(And(rawF, mid2));
    const VU16 partA8 = Or3(part8, part9, partA);
    const VU16 partDB = Or3(partB, partC, partD);
    const VU16 partFE = Or(partE, partF);
    const VU16 packed8 = Or3(partA8, partDB, partFE);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);
    StoreU(packed8, d, packed_out + 8 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));

    // We extract the lowest nine bits and shift right.
    const VU16 mask = Set(d, 0x1FFu);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<9>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed1, mask);
    packed1 = ShiftRight<9>(packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed2, mask);
    packed2 = ShiftRight<9>(packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed3, mask);
    packed3 = ShiftRight<9>(packed3);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed4, mask);
    packed4 = ShiftRight<9>(packed4);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed5, mask);
    packed5 = ShiftRight<9>(packed5);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed6, mask);
    packed6 = ShiftRight<9>(packed6);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(packed7, mask);
    packed7 = ShiftRight<9>(packed7);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 mid2 = Set(d, 0x180u);  // top 2 in lower 9
    const VU16 raw8 = OrAnd(packed0, ShiftLeft<7>(packed8), mid2);
    const VU16 raw9 = OrAnd(packed1, ShiftLeft<5>(packed8), mid2);
    const VU16 rawA = OrAnd(packed2, ShiftLeft<3>(packed8), mid2);
    const VU16 rawB = OrAnd(packed3, ShiftLeft<1>(packed8), mid2);
    const VU16 rawC = OrAnd(packed4, ShiftRight<1>(packed8), mid2);
    const VU16 rawD = OrAnd(packed5, ShiftRight<3>(packed8), mid2);
    const VU16 rawE = OrAnd(packed6, ShiftRight<5>(packed8), mid2);
    const VU16 rawF = OrAnd(packed7, ShiftRight<7>(packed8), mid2);
    StoreU(raw8, d, raw + 8 * N);
    StoreU(raw9, d, raw + 9 * N);
    StoreU(rawA, d, raw + 0xA * N);
    StoreU(rawB, d, raw + 0xB * N);
    StoreU(rawC, d, raw + 0xC * N);
    StoreU(rawD, d, raw + 0xD * N);
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<9>

template <>
struct Pack16<10> {
  static constexpr size_t kBits = 10;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 10;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);
    // 8 vectors, each with 10+6 bits; top 4 bits are concatenated into
    // packed8 and packed9.
    const VU16 packed0 = Or(ShiftLeft<10>(raw8), raw0);
    const VU16 packed1 = Or(ShiftLeft<10>(raw9), raw1);
    const VU16 packed2 = Or(ShiftLeft<10>(rawA), raw2);
    const VU16 packed3 = Or(ShiftLeft<10>(rawB), raw3);
    const VU16 packed4 = Or(ShiftLeft<10>(rawC), raw4);
    const VU16 packed5 = Or(ShiftLeft<10>(rawD), raw5);
    const VU16 packed6 = Or(ShiftLeft<10>(rawE), raw6);
    const VU16 packed7 = Or(ShiftLeft<10>(rawF), raw7);

    // We could shift down, OR and shift up, but two shifts are typically more
    // expensive than AND, shift into position, and OR (which can be further
    // reduced via Or3).
    const VU16 mid4 = Set(d, 0x3C0u);  // top 4 in lower 10
    const VU16 part8 = ShiftRight<6>(And(raw8, mid4));
    const VU16 part9 = ShiftRight<2>(And(raw9, mid4));
    const VU16 partA = ShiftLeft<2>(And(rawA, mid4));
    const VU16 partB = ShiftLeft<6>(And(rawB, mid4));
    const VU16 partC = ShiftRight<6>(And(rawC, mid4));
    const VU16 partD = ShiftRight<2>(And(rawD, mid4));
    const VU16 partE = ShiftLeft<2>(And(rawE, mid4));
    const VU16 partF = ShiftLeft<6>(And(rawF, mid4));
    const VU16 partA8 = Or3(part8, part9, partA);
    const VU16 partEC = Or3(partC, partD, partE);
    const VU16 packed8 = Or(partA8, partB);
    const VU16 packed9 = Or(partEC, partF);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);
    StoreU(packed8, d, packed_out + 8 * N);
    StoreU(packed9, d, packed_out + 9 * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    VU16 packed9 = BitCast(d, LoadU(d, packed_in + 9 * N));

    // We extract the lowest ten bits and shift right.
    const VU16 mask = Set(d, 0x3FFu);

    const VU16 raw0 = And(packed0, mask);
    packed0 = ShiftRight<10>(packed0);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed1, mask);
    packed1 = ShiftRight<10>(packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed2, mask);
    packed2 = ShiftRight<10>(packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed3, mask);
    packed3 = ShiftRight<10>(packed3);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed4, mask);
    packed4 = ShiftRight<10>(packed4);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed5, mask);
    packed5 = ShiftRight<10>(packed5);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed6, mask);
    packed6 = ShiftRight<10>(packed6);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(packed7, mask);
    packed7 = ShiftRight<10>(packed7);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 mid4 = Set(d, 0x3C0u);  // top 4 in lower 10
    const VU16 raw8 = OrAnd(packed0, ShiftLeft<6>(packed8), mid4);
    const VU16 raw9 = OrAnd(packed1, ShiftLeft<2>(packed8), mid4);
    const VU16 rawA = OrAnd(packed2, ShiftRight<2>(packed8), mid4);
    const VU16 rawB = OrAnd(packed3, ShiftRight<6>(packed8), mid4);
    const VU16 rawC = OrAnd(packed4, ShiftLeft<6>(packed9), mid4);
    const VU16 rawD = OrAnd(packed5, ShiftLeft<2>(packed9), mid4);
    const VU16 rawE = OrAnd(packed6, ShiftRight<2>(packed9), mid4);
    const VU16 rawF = OrAnd(packed7, ShiftRight<6>(packed9), mid4);
    StoreU(raw8, d, raw + 8 * N);
    StoreU(raw9, d, raw + 9 * N);
    StoreU(rawA, d, raw + 0xA * N);
    StoreU(rawB, d, raw + 0xB * N);
    StoreU(rawC, d, raw + 0xC * N);
    StoreU(rawD, d, raw + 0xD * N);
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<10>

template <>
struct Pack16<11> {
  static constexpr size_t kBits = 11;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 11;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);
    // It is not obvious what the optimal partitioning looks like. To reduce the
    // number of constants, we want to minimize the number of distinct bit
    // lengths. 11+5 also requires 6-bit remnants with 4-bit leftovers.
    // 8+3 seems better: it is easier to scatter 3 bits into the MSBs.
    const VU16 lo8 = Set(d, 0xFFu);

    // Lower 8 bits of all raw
    const VU16 packed0 = OrAnd(ShiftLeft<8>(raw1), raw0, lo8);
    const VU16 packed1 = OrAnd(ShiftLeft<8>(raw3), raw2, lo8);
    const VU16 packed2 = OrAnd(ShiftLeft<8>(raw5), raw4, lo8);
    const VU16 packed3 = OrAnd(ShiftLeft<8>(raw7), raw6, lo8);
    const VU16 packed4 = OrAnd(ShiftLeft<8>(raw9), raw8, lo8);
    const VU16 packed5 = OrAnd(ShiftLeft<8>(rawB), rawA, lo8);
    const VU16 packed6 = OrAnd(ShiftLeft<8>(rawD), rawC, lo8);
    const VU16 packed7 = OrAnd(ShiftLeft<8>(rawF), rawE, lo8);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);

    // Three vectors, five 3bit remnants each, plus one 3bit in their MSB.
    const VU16 top0 = ShiftRight<8>(raw0);
    const VU16 top1 = ShiftRight<8>(raw1);
    const VU16 top2 = ShiftRight<8>(raw2);
    // Insert top raw bits into 3-bit groups within packed8..A. Moving the
    // mask along avoids masking each of raw0..E and enables OrAnd.
    VU16 next = Set(d, 0x38u);  // 0x7 << 3
    VU16 packed8 = OrAnd(top0, ShiftRight<5>(raw3), next);
    VU16 packed9 = OrAnd(top1, ShiftRight<5>(raw4), next);
    VU16 packedA = OrAnd(top2, ShiftRight<5>(raw5), next);
    next = ShiftLeft<3>(next);
    packed8 = OrAnd(packed8, ShiftRight<2>(raw6), next);
    packed9 = OrAnd(packed9, ShiftRight<2>(raw7), next);
    packedA = OrAnd(packedA, ShiftRight<2>(raw8), next);
    next = ShiftLeft<3>(next);
    packed8 = OrAnd(packed8, Add(raw9, raw9), next);
    packed9 = OrAnd(packed9, Add(rawA, rawA), next);
    packedA = OrAnd(packedA, Add(rawB, rawB), next);
    next = ShiftLeft<3>(next);
    packed8 = OrAnd(packed8, ShiftLeft<4>(rawC), next);
    packed9 = OrAnd(packed9, ShiftLeft<4>(rawD), next);
    packedA = OrAnd(packedA, ShiftLeft<4>(rawE), next);

    // Scatter upper 3 bits of rawF into the upper bits.
    next = ShiftLeft<3>(next);  // = 0x8000u
    packed8 = OrAnd(packed8, ShiftLeft<7>(rawF), next);
    packed9 = OrAnd(packed9, ShiftLeft<6>(rawF), next);
    packedA = OrAnd(packedA, ShiftLeft<5>(rawF), next);

    StoreU(packed8, d, packed_out + 8 * N);
    StoreU(packed9, d, packed_out + 9 * N);
    StoreU(packedA, d, packed_out + 0xA * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    const VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    const VU16 packed9 = BitCast(d, LoadU(d, packed_in + 9 * N));
    const VU16 packedA = BitCast(d, LoadU(d, packed_in + 0xA * N));
    // We extract the lowest eight bits and shift right.
    const VU16 mask8 = Set(d, 0xFFu);
    const VU16 low0 = And(packed0, mask8);
    packed0 = ShiftRight<8>(packed0);
    const VU16 low2 = And(packed1, mask8);
    packed1 = ShiftRight<8>(packed1);
    const VU16 low4 = And(packed2, mask8);
    packed2 = ShiftRight<8>(packed2);
    const VU16 low6 = And(packed3, mask8);
    packed3 = ShiftRight<8>(packed3);
    const VU16 low8 = And(packed4, mask8);
    packed4 = ShiftRight<8>(packed4);
    const VU16 lowA = And(packed5, mask8);
    packed5 = ShiftRight<8>(packed5);
    const VU16 lowC = And(packed6, mask8);
    packed6 = ShiftRight<8>(packed6);
    const VU16 lowE = And(packed7, mask8);
    packed7 = ShiftRight<8>(packed7);

    // Three bits from packed8..A, eight bits alternating from low and packed.
    const VU16 top3 = Set(d, 0x700u);
    const VU16 raw0 = OrAnd(low0, ShiftLeft<8>(packed8), top3);
    const VU16 raw1 = OrAnd(packed0, ShiftLeft<8>(packed9), top3);
    const VU16 raw2 = OrAnd(low2, ShiftLeft<8>(packedA), top3);

    const VU16 raw3 = OrAnd(packed1, ShiftLeft<5>(packed8), top3);
    const VU16 raw4 = OrAnd(low4, ShiftLeft<5>(packed9), top3);
    const VU16 raw5 = OrAnd(packed2, ShiftLeft<5>(packedA), top3);

    const VU16 raw6 = OrAnd(low6, ShiftLeft<2>(packed8), top3);
    const VU16 raw7 = OrAnd(packed3, ShiftLeft<2>(packed9), top3);
    const VU16 raw8 = OrAnd(low8, ShiftLeft<2>(packedA), top3);

    const VU16 raw9 = OrAnd(packed4, ShiftRight<1>(packed8), top3);
    const VU16 rawA = OrAnd(lowA, ShiftRight<1>(packed9), top3);
    const VU16 rawB = OrAnd(packed5, ShiftRight<1>(packedA), top3);

    const VU16 rawC = OrAnd(lowC, ShiftRight<4>(packed8), top3);
    const VU16 rawD = OrAnd(packed6, ShiftRight<4>(packed9), top3);
    const VU16 rawE = OrAnd(lowE, ShiftRight<4>(packedA), top3);

    // Shift MSB into the top 3-of-11 and mask.
    VU16 rawF = OrAnd(packed7, ShiftRight<7>(packed8), top3);
    rawF = OrAnd(rawF, ShiftRight<6>(packed9), top3);
    rawF = OrAnd(rawF, ShiftRight<5>(packedA), top3);

    StoreU(raw0, d, raw + 0 * N);
    StoreU(raw1, d, raw + 1 * N);
    StoreU(raw2, d, raw + 2 * N);
    StoreU(raw3, d, raw + 3 * N);
    StoreU(raw4, d, raw + 4 * N);
    StoreU(raw5, d, raw + 5 * N);
    StoreU(raw6, d, raw + 6 * N);
    StoreU(raw7, d, raw + 7 * N);
    StoreU(raw8, d, raw + 8 * N);
    StoreU(raw9, d, raw + 9 * N);
    StoreU(rawA, d, raw + 0xA * N);
    StoreU(rawB, d, raw + 0xB * N);
    StoreU(rawC, d, raw + 0xC * N);
    StoreU(rawD, d, raw + 0xD * N);
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<11>

template <>
struct Pack16<12> {
  static constexpr size_t kBits = 12;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 12;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);
    // 8 vectors, each with 12+4 bits; top 8 bits are concatenated into
    // packed8 to packedB.
    const VU16 packed0 = Or(ShiftLeft<12>(raw8), raw0);
    const VU16 packed1 = Or(ShiftLeft<12>(raw9), raw1);
    const VU16 packed2 = Or(ShiftLeft<12>(rawA), raw2);
    const VU16 packed3 = Or(ShiftLeft<12>(rawB), raw3);
    const VU16 packed4 = Or(ShiftLeft<12>(rawC), raw4);
    const VU16 packed5 = Or(ShiftLeft<12>(rawD), raw5);
    const VU16 packed6 = Or(ShiftLeft<12>(rawE), raw6);
    const VU16 packed7 = Or(ShiftLeft<12>(rawF), raw7);

    // Masking after shifting left enables OrAnd.
    const VU16 top8 = Set(d, 0xFF00u);
    const VU16 packed8 = OrAnd(ShiftRight<4>(raw8), ShiftLeft<4>(raw9), top8);
    const VU16 packed9 = OrAnd(ShiftRight<4>(rawA), ShiftLeft<4>(rawB), top8);
    const VU16 packedA = OrAnd(ShiftRight<4>(rawC), ShiftLeft<4>(rawD), top8);
    const VU16 packedB = OrAnd(ShiftRight<4>(rawE), ShiftLeft<4>(rawF), top8);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);
    StoreU(packed8, d, packed_out + 8 * N);
    StoreU(packed9, d, packed_out + 9 * N);
    StoreU(packedA, d, packed_out + 0xA * N);
    StoreU(packedB, d, packed_out + 0xB * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    const VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    const VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    const VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    const VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    const VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    const VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    const VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    const VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    const VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    const VU16 packed9 = BitCast(d, LoadU(d, packed_in + 9 * N));
    const VU16 packedA = BitCast(d, LoadU(d, packed_in + 0xA * N));
    const VU16 packedB = BitCast(d, LoadU(d, packed_in + 0xB * N));

    // We extract the lowest 12 bits.
    const VU16 mask = Set(d, 0xFFFu);

    const VU16 raw0 = And(packed0, mask);
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = And(packed1, mask);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = And(packed2, mask);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = And(packed3, mask);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = And(packed4, mask);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = And(packed5, mask);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = And(packed6, mask);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = And(packed7, mask);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 mid8 = Set(d, 0xFF0u);  // upper 8 in lower 12
    const VU16 raw8 =
        OrAnd(ShiftRight<12>(packed0), ShiftLeft<4>(packed8), mid8);
    const VU16 raw9 =
        OrAnd(ShiftRight<12>(packed1), ShiftRight<4>(packed8), mid8);
    const VU16 rawA =
        OrAnd(ShiftRight<12>(packed2), ShiftLeft<4>(packed9), mid8);
    const VU16 rawB =
        OrAnd(ShiftRight<12>(packed3), ShiftRight<4>(packed9), mid8);
    const VU16 rawC =
        OrAnd(ShiftRight<12>(packed4), ShiftLeft<4>(packedA), mid8);
    const VU16 rawD =
        OrAnd(ShiftRight<12>(packed5), ShiftRight<4>(packedA), mid8);
    const VU16 rawE =
        OrAnd(ShiftRight<12>(packed6), ShiftLeft<4>(packedB), mid8);
    const VU16 rawF =
        OrAnd(ShiftRight<12>(packed7), ShiftRight<4>(packedB), mid8);
    StoreU(raw8, d, raw + 8 * N);
    StoreU(raw9, d, raw + 9 * N);
    StoreU(rawA, d, raw + 0xA * N);
    StoreU(rawB, d, raw + 0xB * N);
    StoreU(rawC, d, raw + 0xC * N);
    StoreU(rawD, d, raw + 0xD * N);
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<12>

template <>
struct Pack16<13> {
  static constexpr size_t kBits = 13;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 13;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);
    // As with 11 bits, it is not obvious what the optimal partitioning looks
    // like. We similarly go with an 8+5 split.
    const VU16 lo8 = Set(d, 0xFFu);

    // Lower 8 bits of all raw
    const VU16 packed0 = OrAnd(ShiftLeft<8>(raw1), raw0, lo8);
    const VU16 packed1 = OrAnd(ShiftLeft<8>(raw3), raw2, lo8);
    const VU16 packed2 = OrAnd(ShiftLeft<8>(raw5), raw4, lo8);
    const VU16 packed3 = OrAnd(ShiftLeft<8>(raw7), raw6, lo8);
    const VU16 packed4 = OrAnd(ShiftLeft<8>(raw9), raw8, lo8);
    const VU16 packed5 = OrAnd(ShiftLeft<8>(rawB), rawA, lo8);
    const VU16 packed6 = OrAnd(ShiftLeft<8>(rawD), rawC, lo8);
    const VU16 packed7 = OrAnd(ShiftLeft<8>(rawF), rawE, lo8);
    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);

    // Five vectors, three 5bit remnants each, plus one 5bit in their MSB.
    const VU16 top0 = ShiftRight<8>(raw0);
    const VU16 top1 = ShiftRight<8>(raw1);
    const VU16 top2 = ShiftRight<8>(raw2);
    const VU16 top3 = ShiftRight<8>(raw3);
    const VU16 top4 = ShiftRight<8>(raw4);
    // Insert top raw bits into 5-bit groups within packed8..C. Moving the
    // mask along avoids masking each of raw0..E and enables OrAnd.
    VU16 next = Set(d, 0x3E0u);  // 0x1F << 5
    VU16 packed8 = OrAnd(top0, ShiftRight<3>(raw5), next);
    VU16 packed9 = OrAnd(top1, ShiftRight<3>(raw6), next);
    VU16 packedA = OrAnd(top2, ShiftRight<3>(raw7), next);
    VU16 packedB = OrAnd(top3, ShiftRight<3>(raw8), next);
    VU16 packedC = OrAnd(top4, ShiftRight<3>(raw9), next);
    next = ShiftLeft<5>(next);
    packed8 = OrAnd(packed8, ShiftLeft<2>(rawA), next);
    packed9 = OrAnd(packed9, ShiftLeft<2>(rawB), next);
    packedA = OrAnd(packedA, ShiftLeft<2>(rawC), next);
    packedB = OrAnd(packedB, ShiftLeft<2>(rawD), next);
    packedC = OrAnd(packedC, ShiftLeft<2>(rawE), next);

    // Scatter upper 5 bits of rawF into the upper bits.
    next = ShiftLeft<3>(next);  // = 0x8000u
    packed8 = OrAnd(packed8, ShiftLeft<7>(rawF), next);
    packed9 = OrAnd(packed9, ShiftLeft<6>(rawF), next);
    packedA = OrAnd(packedA, ShiftLeft<5>(rawF), next);
    packedB = OrAnd(packedB, ShiftLeft<4>(rawF), next);
    packedC = OrAnd(packedC, ShiftLeft<3>(rawF), next);

    StoreU(packed8, d, packed_out + 8 * N);
    StoreU(packed9, d, packed_out + 9 * N);
    StoreU(packedA, d, packed_out + 0xA * N);
    StoreU(packedB, d, packed_out + 0xB * N);
    StoreU(packedC, d, packed_out + 0xC * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    const VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    const VU16 packed9 = BitCast(d, LoadU(d, packed_in + 9 * N));
    const VU16 packedA = BitCast(d, LoadU(d, packed_in + 0xA * N));
    const VU16 packedB = BitCast(d, LoadU(d, packed_in + 0xB * N));
    const VU16 packedC = BitCast(d, LoadU(d, packed_in + 0xC * N));
    // We extract the lowest eight bits and shift right.
    const VU16 mask8 = Set(d, 0xFFu);
    const VU16 low0 = And(packed0, mask8);
    packed0 = ShiftRight<8>(packed0);
    const VU16 low2 = And(packed1, mask8);
    packed1 = ShiftRight<8>(packed1);
    const VU16 low4 = And(packed2, mask8);
    packed2 = ShiftRight<8>(packed2);
    const VU16 low6 = And(packed3, mask8);
    packed3 = ShiftRight<8>(packed3);
    const VU16 low8 = And(packed4, mask8);
    packed4 = ShiftRight<8>(packed4);
    const VU16 lowA = And(packed5, mask8);
    packed5 = ShiftRight<8>(packed5);
    const VU16 lowC = And(packed6, mask8);
    packed6 = ShiftRight<8>(packed6);
    const VU16 lowE = And(packed7, mask8);
    packed7 = ShiftRight<8>(packed7);

    // Five bits from packed8..C, eight bits alternating from low and packed.
    const VU16 top5 = Set(d, 0x1F00u);
    const VU16 raw0 = OrAnd(low0, ShiftLeft<8>(packed8), top5);
    const VU16 raw1 = OrAnd(packed0, ShiftLeft<8>(packed9), top5);
    const VU16 raw2 = OrAnd(low2, ShiftLeft<8>(packedA), top5);
    const VU16 raw3 = OrAnd(packed1, ShiftLeft<8>(packedB), top5);
    const VU16 raw4 = OrAnd(low4, ShiftLeft<8>(packedC), top5);

    const VU16 raw5 = OrAnd(packed2, ShiftLeft<3>(packed8), top5);
    const VU16 raw6 = OrAnd(low6, ShiftLeft<3>(packed9), top5);
    const VU16 raw7 = OrAnd(packed3, ShiftLeft<3>(packedA), top5);
    const VU16 raw8 = OrAnd(low8, ShiftLeft<3>(packed9), top5);
    const VU16 raw9 = OrAnd(packed4, ShiftLeft<3>(packedA), top5);

    const VU16 rawA = OrAnd(lowA, ShiftRight<2>(packed8), top5);
    const VU16 rawB = OrAnd(packed5, ShiftRight<2>(packed9), top5);
    const VU16 rawC = OrAnd(lowC, ShiftRight<2>(packedA), top5);
    const VU16 rawD = OrAnd(packed6, ShiftRight<2>(packed9), top5);
    const VU16 rawE = OrAnd(lowE, ShiftRight<2>(packedA), top5);

    // Shift MSB into the top 5-of-11 and mask.
    VU16 rawF = OrAnd(packed7, ShiftRight<7>(packed8), top5);
    rawF = OrAnd(rawF, ShiftRight<6>(packed9), top5);
    rawF = OrAnd(rawF, ShiftRight<5>(packedA), top5);
    rawF = OrAnd(rawF, ShiftRight<4>(packedB), top5);
    rawF = OrAnd(rawF, ShiftRight<3>(packedC), top5);

    StoreU(raw0, d, raw + 0 * N);
    StoreU(raw1, d, raw + 1 * N);
    StoreU(raw2, d, raw + 2 * N);
    StoreU(raw3, d, raw + 3 * N);
    StoreU(raw4, d, raw + 4 * N);
    StoreU(raw5, d, raw + 5 * N);
    StoreU(raw6, d, raw + 6 * N);
    StoreU(raw7, d, raw + 7 * N);
    StoreU(raw8, d, raw + 8 * N);
    StoreU(raw9, d, raw + 9 * N);
    StoreU(rawA, d, raw + 0xA * N);
    StoreU(rawB, d, raw + 0xB * N);
    StoreU(rawC, d, raw + 0xC * N);
    StoreU(rawD, d, raw + 0xD * N);
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<13>

template <>
struct Pack16<14> {
  static constexpr size_t kBits = 14;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 14;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    // 14 vectors, each with 14+2 bits; two raw vectors are scattered
    // across the upper 2 bits.
    const VU16 top2 = Set(d, 0xC000u);
    const VU16 packed0 = Or(raw0, ShiftLeft<14>(rawE));
    const VU16 packed1 = OrAnd(raw1, ShiftLeft<12>(rawE), top2);
    const VU16 packed2 = OrAnd(raw2, ShiftLeft<10>(rawE), top2);
    const VU16 packed3 = OrAnd(raw3, ShiftLeft<8>(rawE), top2);
    const VU16 packed4 = OrAnd(raw4, ShiftLeft<6>(rawE), top2);
    const VU16 packed5 = OrAnd(raw5, ShiftLeft<4>(rawE), top2);
    const VU16 packed6 = OrAnd(raw6, ShiftLeft<2>(rawE), top2);
    const VU16 packed7 = Or(raw7, ShiftLeft<14>(rawF));
    const VU16 packed8 = OrAnd(raw8, ShiftLeft<12>(rawF), top2);
    const VU16 packed9 = OrAnd(raw9, ShiftLeft<10>(rawF), top2);
    const VU16 packedA = OrAnd(rawA, ShiftLeft<8>(rawF), top2);
    const VU16 packedB = OrAnd(rawB, ShiftLeft<6>(rawF), top2);
    const VU16 packedC = OrAnd(rawC, ShiftLeft<4>(rawF), top2);
    const VU16 packedD = OrAnd(rawD, ShiftLeft<2>(rawF), top2);

    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);
    StoreU(packed8, d, packed_out + 8 * N);
    StoreU(packed9, d, packed_out + 9 * N);
    StoreU(packedA, d, packed_out + 0xA * N);
    StoreU(packedB, d, packed_out + 0xB * N);
    StoreU(packedC, d, packed_out + 0xC * N);
    StoreU(packedD, d, packed_out + 0xD * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    VU16 packed9 = BitCast(d, LoadU(d, packed_in + 9 * N));
    VU16 packedA = BitCast(d, LoadU(d, packed_in + 0xA * N));
    VU16 packedB = BitCast(d, LoadU(d, packed_in + 0xB * N));
    VU16 packedC = BitCast(d, LoadU(d, packed_in + 0xC * N));
    VU16 packedD = BitCast(d, LoadU(d, packed_in + 0xD * N));

    // We extract the lowest 14 bits.
    const VU16 top2 = Set(d, 0xC000u);

    const VU16 raw0 = AndNot(top2, packed0);
    // Can skip the And for packed0, will be right-shifted 14 bits anyway.
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = AndNot(top2, packed1);
    packed1 = And(top2, packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = AndNot(top2, packed2);
    packed2 = And(top2, packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = AndNot(top2, packed3);
    packed3 = And(top2, packed3);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = AndNot(top2, packed4);
    packed4 = And(top2, packed4);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = AndNot(top2, packed5);
    packed5 = And(top2, packed5);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = AndNot(top2, packed6);
    packed6 = And(top2, packed6);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = AndNot(top2, packed7);
    // Can skip the And for packed7, will be right-shifted 14 bits anyway.
    StoreU(raw7, d, raw + 7 * N);

    const VU16 raw8 = AndNot(top2, packed8);
    packed8 = And(top2, packed8);
    StoreU(raw8, d, raw + 8 * N);

    const VU16 raw9 = AndNot(top2, packed9);
    packed9 = And(top2, packed9);
    StoreU(raw9, d, raw + 9 * N);

    const VU16 rawA = AndNot(top2, packedA);
    packedA = And(top2, packedA);
    StoreU(rawA, d, raw + 0xA * N);

    const VU16 rawB = AndNot(top2, packedB);
    packedB = And(top2, packedB);
    StoreU(rawB, d, raw + 0xB * N);

    const VU16 rawC = AndNot(top2, packedC);
    packedC = And(top2, packedC);
    StoreU(rawC, d, raw + 0xC * N);

    const VU16 rawD = AndNot(top2, packedD);
    packedD = And(top2, packedD);
    StoreU(rawD, d, raw + 0xD * N);

    // rawE is the concatenation of the top two bits in packed0..6.
    const VU16 E0 = Or3(ShiftRight<14>(packed0), ShiftRight<12>(packed1),
                        ShiftRight<10>(packed2));
    const VU16 E1 = Or3(ShiftRight<8>(packed3), ShiftRight<6>(packed4),
                        ShiftRight<4>(packed5));
    const VU16 rawE = Or3(E0, E1, ShiftRight<2>(packed6));
    const VU16 F0 = Or3(ShiftRight<14>(packed7), ShiftRight<12>(packed8),
                        ShiftRight<10>(packed9));
    const VU16 F1 = Or3(ShiftRight<8>(packedA), ShiftRight<6>(packedB),
                        ShiftRight<4>(packedC));
    const VU16 rawF = Or3(F0, F1, ShiftRight<2>(packedD));
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<14>

template <>
struct Pack16<15> {
  static constexpr size_t kBits = 15;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 15;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    // 15 vectors, each with 15+1 bits; one packed vector is scattered
    // across the upper bit.
    const VU16 top1 = Set(d, 0x8000u);
    const VU16 packed0 = Or(raw0, ShiftLeft<15>(rawF));
    const VU16 packed1 = OrAnd(raw1, ShiftLeft<14>(rawF), top1);
    const VU16 packed2 = OrAnd(raw2, ShiftLeft<13>(rawF), top1);
    const VU16 packed3 = OrAnd(raw3, ShiftLeft<12>(rawF), top1);
    const VU16 packed4 = OrAnd(raw4, ShiftLeft<11>(rawF), top1);
    const VU16 packed5 = OrAnd(raw5, ShiftLeft<10>(rawF), top1);
    const VU16 packed6 = OrAnd(raw6, ShiftLeft<9>(rawF), top1);
    const VU16 packed7 = OrAnd(raw7, ShiftLeft<8>(rawF), top1);
    const VU16 packed8 = OrAnd(raw8, ShiftLeft<7>(rawF), top1);
    const VU16 packed9 = OrAnd(raw9, ShiftLeft<6>(rawF), top1);
    const VU16 packedA = OrAnd(rawA, ShiftLeft<5>(rawF), top1);
    const VU16 packedB = OrAnd(rawB, ShiftLeft<4>(rawF), top1);
    const VU16 packedC = OrAnd(rawC, ShiftLeft<3>(rawF), top1);
    const VU16 packedD = OrAnd(rawD, ShiftLeft<2>(rawF), top1);
    const VU16 packedE = OrAnd(rawE, ShiftLeft<1>(rawF), top1);

    StoreU(packed0, d, packed_out + 0 * N);
    StoreU(packed1, d, packed_out + 1 * N);
    StoreU(packed2, d, packed_out + 2 * N);
    StoreU(packed3, d, packed_out + 3 * N);
    StoreU(packed4, d, packed_out + 4 * N);
    StoreU(packed5, d, packed_out + 5 * N);
    StoreU(packed6, d, packed_out + 6 * N);
    StoreU(packed7, d, packed_out + 7 * N);
    StoreU(packed8, d, packed_out + 8 * N);
    StoreU(packed9, d, packed_out + 9 * N);
    StoreU(packedA, d, packed_out + 0xA * N);
    StoreU(packedB, d, packed_out + 0xB * N);
    StoreU(packedC, d, packed_out + 0xC * N);
    StoreU(packedD, d, packed_out + 0xD * N);
    StoreU(packedE, d, packed_out + 0xE * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    VU16 packed0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    VU16 packed1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    VU16 packed2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    VU16 packed3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    VU16 packed4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    VU16 packed5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    VU16 packed6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    VU16 packed7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    VU16 packed8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    VU16 packed9 = BitCast(d, LoadU(d, packed_in + 9 * N));
    VU16 packedA = BitCast(d, LoadU(d, packed_in + 0xA * N));
    VU16 packedB = BitCast(d, LoadU(d, packed_in + 0xB * N));
    VU16 packedC = BitCast(d, LoadU(d, packed_in + 0xC * N));
    VU16 packedD = BitCast(d, LoadU(d, packed_in + 0xD * N));
    VU16 packedE = BitCast(d, LoadU(d, packed_in + 0xE * N));

    // We extract the lowest 15 bits.
    const VU16 top1 = Set(d, 0x8000u);

    const VU16 raw0 = AndNot(top1, packed0);
    // Can skip the And for packed0, will be right-shifted 15 bits anyway.
    StoreU(raw0, d, raw + 0 * N);

    const VU16 raw1 = AndNot(top1, packed1);
    packed1 = And(top1, packed1);
    StoreU(raw1, d, raw + 1 * N);

    const VU16 raw2 = AndNot(top1, packed2);
    packed2 = And(top1, packed2);
    StoreU(raw2, d, raw + 2 * N);

    const VU16 raw3 = AndNot(top1, packed3);
    packed3 = And(top1, packed3);
    StoreU(raw3, d, raw + 3 * N);

    const VU16 raw4 = AndNot(top1, packed4);
    packed4 = And(top1, packed4);
    StoreU(raw4, d, raw + 4 * N);

    const VU16 raw5 = AndNot(top1, packed5);
    packed5 = And(top1, packed5);
    StoreU(raw5, d, raw + 5 * N);

    const VU16 raw6 = AndNot(top1, packed6);
    packed6 = And(top1, packed6);
    StoreU(raw6, d, raw + 6 * N);

    const VU16 raw7 = AndNot(top1, packed7);
    packed7 = And(top1, packed7);
    StoreU(raw7, d, raw + 7 * N);

    const VU16 raw8 = AndNot(top1, packed8);
    packed8 = And(top1, packed8);
    StoreU(raw8, d, raw + 8 * N);

    const VU16 raw9 = AndNot(top1, packed9);
    packed9 = And(top1, packed9);
    StoreU(raw9, d, raw + 9 * N);

    const VU16 rawA = AndNot(top1, packedA);
    packedA = And(top1, packedA);
    StoreU(rawA, d, raw + 0xA * N);

    const VU16 rawB = AndNot(top1, packedB);
    packedB = And(top1, packedB);
    StoreU(rawB, d, raw + 0xB * N);

    const VU16 rawC = AndNot(top1, packedC);
    packedC = And(top1, packedC);
    StoreU(rawC, d, raw + 0xC * N);

    const VU16 rawD = AndNot(top1, packedD);
    packedD = And(top1, packedD);
    StoreU(rawD, d, raw + 0xD * N);

    const VU16 rawE = AndNot(top1, packedE);
    packedE = And(top1, packedE);
    StoreU(rawE, d, raw + 0xE * N);

    // rawF is the concatenation of the top bit in packed0..E.
    const VU16 F0 = Or3(ShiftRight<15>(packed0), ShiftRight<14>(packed1),
                        ShiftRight<13>(packed2));
    const VU16 F1 = Or3(ShiftRight<12>(packed3), ShiftRight<11>(packed4),
                        ShiftRight<10>(packed5));
    const VU16 F2 = Or3(ShiftRight<9>(packed6), ShiftRight<8>(packed7),
                        ShiftRight<7>(packed8));
    const VU16 F3 = Or3(ShiftRight<6>(packed9), ShiftRight<5>(packedA),
                        ShiftRight<4>(packedB));
    const VU16 F4 = Or3(ShiftRight<3>(packedC), ShiftRight<2>(packedD),
                        ShiftRight<1>(packedE));
    const VU16 rawF = Or3(F0, F1, Or3(F2, F3, F4));
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<15>

template <>
struct Pack16<16> {
  static constexpr size_t kBits = 16;
  static constexpr size_t kRawVectors = 16;
  static constexpr size_t kPackedVectors = 16;

  template <class D>
  HWY_INLINE void Pack(D d, const uint16_t* HWY_RESTRICT raw,
                       uint16_t* HWY_RESTRICT packed_out) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);
    const VU16 raw0 = LoadU(d, raw + 0 * N);
    const VU16 raw1 = LoadU(d, raw + 1 * N);
    const VU16 raw2 = LoadU(d, raw + 2 * N);
    const VU16 raw3 = LoadU(d, raw + 3 * N);
    const VU16 raw4 = LoadU(d, raw + 4 * N);
    const VU16 raw5 = LoadU(d, raw + 5 * N);
    const VU16 raw6 = LoadU(d, raw + 6 * N);
    const VU16 raw7 = LoadU(d, raw + 7 * N);
    const VU16 raw8 = LoadU(d, raw + 8 * N);
    const VU16 raw9 = LoadU(d, raw + 9 * N);
    const VU16 rawA = LoadU(d, raw + 0xA * N);
    const VU16 rawB = LoadU(d, raw + 0xB * N);
    const VU16 rawC = LoadU(d, raw + 0xC * N);
    const VU16 rawD = LoadU(d, raw + 0xD * N);
    const VU16 rawE = LoadU(d, raw + 0xE * N);
    const VU16 rawF = LoadU(d, raw + 0xF * N);

    StoreU(raw0, d, packed_out + 0 * N);
    StoreU(raw1, d, packed_out + 1 * N);
    StoreU(raw2, d, packed_out + 2 * N);
    StoreU(raw3, d, packed_out + 3 * N);
    StoreU(raw4, d, packed_out + 4 * N);
    StoreU(raw5, d, packed_out + 5 * N);
    StoreU(raw6, d, packed_out + 6 * N);
    StoreU(raw7, d, packed_out + 7 * N);
    StoreU(raw8, d, packed_out + 8 * N);
    StoreU(raw9, d, packed_out + 9 * N);
    StoreU(rawA, d, packed_out + 0xA * N);
    StoreU(rawB, d, packed_out + 0xB * N);
    StoreU(rawC, d, packed_out + 0xC * N);
    StoreU(rawD, d, packed_out + 0xD * N);
    StoreU(rawE, d, packed_out + 0xE * N);
    StoreU(rawF, d, packed_out + 0xF * N);
  }

  template <class D>
  HWY_INLINE void Unpack(D d, const uint16_t* HWY_RESTRICT packed_in,
                         uint16_t* HWY_RESTRICT raw) {
    using VU16 = Vec<decltype(d)>;
    const size_t N = Lanes(d);

    const VU16 raw0 = BitCast(d, LoadU(d, packed_in + 0 * N));
    const VU16 raw1 = BitCast(d, LoadU(d, packed_in + 1 * N));
    const VU16 raw2 = BitCast(d, LoadU(d, packed_in + 2 * N));
    const VU16 raw3 = BitCast(d, LoadU(d, packed_in + 3 * N));
    const VU16 raw4 = BitCast(d, LoadU(d, packed_in + 4 * N));
    const VU16 raw5 = BitCast(d, LoadU(d, packed_in + 5 * N));
    const VU16 raw6 = BitCast(d, LoadU(d, packed_in + 6 * N));
    const VU16 raw7 = BitCast(d, LoadU(d, packed_in + 7 * N));
    const VU16 raw8 = BitCast(d, LoadU(d, packed_in + 8 * N));
    const VU16 raw9 = BitCast(d, LoadU(d, packed_in + 9 * N));
    const VU16 rawA = BitCast(d, LoadU(d, packed_in + 0xA * N));
    const VU16 rawB = BitCast(d, LoadU(d, packed_in + 0xB * N));
    const VU16 rawC = BitCast(d, LoadU(d, packed_in + 0xC * N));
    const VU16 rawD = BitCast(d, LoadU(d, packed_in + 0xD * N));
    const VU16 rawE = BitCast(d, LoadU(d, packed_in + 0xE * N));
    const VU16 rawF = BitCast(d, LoadU(d, packed_in + 0xF * N));

    StoreU(raw0, d, raw + 0 * N);
    StoreU(raw1, d, raw + 1 * N);
    StoreU(raw2, d, raw + 2 * N);
    StoreU(raw3, d, raw + 3 * N);
    StoreU(raw4, d, raw + 4 * N);
    StoreU(raw5, d, raw + 5 * N);
    StoreU(raw6, d, raw + 6 * N);
    StoreU(raw7, d, raw + 7 * N);
    StoreU(raw8, d, raw + 8 * N);
    StoreU(raw9, d, raw + 9 * N);
    StoreU(rawA, d, raw + 0xA * N);
    StoreU(rawB, d, raw + 0xB * N);
    StoreU(rawC, d, raw + 0xC * N);
    StoreU(rawD, d, raw + 0xD * N);
    StoreU(rawE, d, raw + 0xE * N);
    StoreU(rawF, d, raw + 0xF * N);
  }
};  // Pack16<12>

}  // namespace detail

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_BIT_PACK_INL_H_
