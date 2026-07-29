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

// Include guard (still compiled once per target)
#if defined(HIGHWAY_HWY_CONTRIB_BASE64_BASE64_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BASE64_BASE64_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BASE64_BASE64_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BASE64_BASE64_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace detail {

HWY_INLINE uint8_t Base64Value(const uint8_t c) {
  if (c >= 'A' && c <= 'Z') return static_cast<uint8_t>(c - 'A');
  if (c >= 'a' && c <= 'z') return static_cast<uint8_t>(c - 'a' + 26);
  if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0' + 52);
  if (c == '+') return 62;
  if (c == '/') return 63;
  return 0xFF;
}

HWY_INLINE void EncodeBase64Tail(const uint8_t* HWY_RESTRICT input,
                                 const size_t input_size,
                                 char* HWY_RESTRICT output) {
  static const char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (input_size == 0) return;

  const uint8_t b0 = input[0];
  output[0] = kAlphabet[b0 >> 2];
  if (input_size == 1) {
    output[1] = kAlphabet[(b0 & 3) << 4];
    output[2] = '=';
    output[3] = '=';
    return;
  }

  const uint8_t b1 = input[1];
  output[1] = kAlphabet[((b0 & 3) << 4) | (b1 >> 4)];
  output[2] = kAlphabet[(b1 & 15) << 2];
  if (input_size == 2) {
    output[3] = '=';
    return;
  }

  const uint8_t b2 = input[2];
  output[2] = kAlphabet[((b1 & 15) << 2) | (b2 >> 6)];
  output[3] = kAlphabet[b2 & 63];
}

#if (HWY_ARCH_ARM_A64 && HWY_TARGET_IS_NEON) || HWY_TARGET <= HWY_AVX3_DL

template <class D>
HWY_INLINE void EncodeBase64BlockInterleaved(D d,
                                             const uint8_t* HWY_RESTRICT input,
                                             char* HWY_RESTRICT output) {
  VFromD<D> b0;
  VFromD<D> b1;
  VFromD<D> b2;
  LoadInterleaved3(d, input, b0, b1, b2);

  const auto mask63 = Set(d, uint8_t{63});
  const auto s0 = ShiftRight<2>(b0);
  const auto s1 = AndXor(mask63, ShiftLeft<4>(b0), ShiftRight<4>(b1));
  const auto s2 = AndXor(mask63, ShiftLeft<2>(b1), ShiftRight<6>(b2));
  const auto s3 = And(b2, mask63);

  HWY_ALIGN static const uint8_t kAlphabet[64] = {
      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
  StoreInterleaved4(Lookup64(d, kAlphabet, s0), Lookup64(d, kAlphabet, s1),
                    Lookup64(d, kAlphabet, s2), Lookup64(d, kAlphabet, s3), d,
                    reinterpret_cast<uint8_t*>(output));
}

// Encodes 3 * Lanes(d) input bytes to 4 * Lanes(d) output characters.
template <class D>
HWY_INLINE void EncodeBase64Block(D d, const uint8_t* HWY_RESTRICT input,
                                  char* HWY_RESTRICT output) {
#if HWY_TARGET <= HWY_AVX3_DL
  const auto input0 = LoadU(d, input + 0 * Lanes(d));
  const auto input1 = LoadU(d, input + 1 * Lanes(d));
  const auto input2 = LoadU(d, input + 2 * Lanes(d));

  HWY_ALIGN static const uint8_t kShuffle[64] = {
      1,  0,  2,  1,  4,  3,  5,  4,  7,  6,  8,  7,  10, 9,  11, 10,
      13, 12, 14, 13, 16, 15, 17, 16, 19, 18, 20, 19, 22, 21, 23, 22,
      25, 24, 26, 25, 28, 27, 29, 28, 31, 30, 32, 31, 34, 33, 35, 34,
      37, 36, 38, 37, 40, 39, 41, 40, 43, 42, 44, 43, 46, 45, 47, 46};
  const auto idx0 = Load(d, kShuffle);
  const auto idx1 = Add(idx0, Set(d, uint8_t{48}));
  const auto idx2 = Add(idx0, Set(d, uint8_t{32}));
  const auto idx3 = Add(idx0, Set(d, uint8_t{16}));
  const auto grouped0 = TableLookupLanes(input0, IndicesFromVec(d, idx0));
  const auto grouped1 =
      TwoTablesLookupLanes(input0, input1, IndicesFromVec(d, idx1));
  const auto grouped2 =
      TwoTablesLookupLanes(input1, input2, IndicesFromVec(d, idx2));
  const auto grouped3 = TableLookupLanes(input2, IndicesFromVec(d, idx3));

  const Repartition<uint64_t, D> du64;
  const auto shifts =
      BitCast(d, Set(du64, static_cast<uint64_t>(0x3036242a1016040aULL)));
  const auto indices0 =
      BitCast(d, MultiRotateRight(BitCast(du64, grouped0), shifts));
  const auto indices1 =
      BitCast(d, MultiRotateRight(BitCast(du64, grouped1), shifts));
  const auto indices2 =
      BitCast(d, MultiRotateRight(BitCast(du64, grouped2), shifts));
  const auto indices3 =
      BitCast(d, MultiRotateRight(BitCast(du64, grouped3), shifts));
  const auto mask63 = Set(d, uint8_t{63});

  HWY_ALIGN static const uint8_t kAlphabet[64] = {
      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
      'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};
  const auto alphabet = Load(d, kAlphabet);
  const auto encoded0 =
      TableLookupLanes(alphabet, IndicesFromVec(d, And(indices0, mask63)));
  const auto encoded1 =
      TableLookupLanes(alphabet, IndicesFromVec(d, And(indices1, mask63)));
  const auto encoded2 =
      TableLookupLanes(alphabet, IndicesFromVec(d, And(indices2, mask63)));
  const auto encoded3 =
      TableLookupLanes(alphabet, IndicesFromVec(d, And(indices3, mask63)));
  StoreU(encoded0, d, reinterpret_cast<uint8_t*>(output) + 0 * Lanes(d));
  StoreU(encoded1, d, reinterpret_cast<uint8_t*>(output) + 1 * Lanes(d));
  StoreU(encoded2, d, reinterpret_cast<uint8_t*>(output) + 2 * Lanes(d));
  StoreU(encoded3, d, reinterpret_cast<uint8_t*>(output) + 3 * Lanes(d));
#else
  EncodeBase64BlockInterleaved(d, input, output);
#endif
}

#endif  // NEON64 || HWY_TARGET <= HWY_AVX3_DL

template <class D>
HWY_INLINE VFromD<D> DecodeBase64Vector(D d, VFromD<D> encoded) {
  // Invalid table entries set bit 7.
  HWY_ALIGN static const uint8_t kLow[64] = {
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 62,
      0x80, 0x80, 0x80, 63,   52,   53,   54,   55,   56,   57,   58,
      59,   60,   61,   0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
  HWY_ALIGN static const uint8_t kHigh[64] = {
      0x80, 0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,  11,
      12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,  24,
      25,   0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 26,   27,   28,   29,   30,  31,
      32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,  44,
      45,   46,   47,   48,   49,   50,   51,   0x80, 0x80, 0x80, 0x80, 0x80};

#if HWY_TARGET <= HWY_AVX3_DL
  const auto index = And(encoded, Set(d, uint8_t{0x7F}));
  return TwoTablesLookupLanes(Load(d, kLow), Load(d, kHigh),
                              IndicesFromVec(d, index));
#elif HWY_ARCH_ARM_A64 && HWY_TARGET_IS_NEON
  // TBL returns zero for indices >= 64, then TBX keeps the low-table result for
  // indices outside [64, 127].
  const uint8x16x4_t low_table = {
      {Load(d, kLow + 0).raw, Load(d, kLow + 16).raw, Load(d, kLow + 32).raw,
       Load(d, kLow + 48).raw}};
  const uint8x16x4_t high_table = {
      {Load(d, kHigh + 0).raw, Load(d, kHigh + 16).raw, Load(d, kHigh + 32).raw,
       Load(d, kHigh + 48).raw}};
  const auto low = Vec128<uint8_t>{vqtbl4q_u8(low_table, encoded.raw)};
  const auto high_index = Sub(encoded, Set(d, uint8_t{64}));
  return Vec128<uint8_t>{vqtbx4q_u8(low.raw, high_table, high_index.raw)};
#else
  const auto index = And(encoded, Set(d, uint8_t{63}));
  const auto low = Lookup64(d, kLow, index);
  const auto high = Lookup64(d, kHigh, index);
  return IfThenElse(TestBit(encoded, Set(d, uint8_t{64})), high, low);
#endif
}

// Decodes Lanes(d) contiguous input characters to 3/4 * Lanes(d) output bytes
// on AVX3_DL. Other targets decode 4 * Lanes(d) interleaved input characters
// to 3 * Lanes(d) output bytes. Returns bytes whose high bit is set if any
// input character was invalid.
template <class D>
HWY_INLINE VFromD<D> DecodeBase64Block(D d, const char* HWY_RESTRICT input,
                                       uint8_t* HWY_RESTRICT output) {
#if HWY_TARGET <= HWY_AVX3_DL
  const auto encoded = LoadU(d, reinterpret_cast<const uint8_t*>(input));
  const auto sextets = DecodeBase64Vector(d, encoded);
  const auto invalid = Or(encoded, sextets);

  const Repartition<int16_t, D> di16;
  const Repartition<int32_t, D> di32;
  const Rebind<int8_t, D> di8;
  const auto mul_ab = BitCast(di8, Set(di32, static_cast<int32_t>(0x01400140)));
  const auto merged16 = SatWidenMulPairwiseAdd(di16, sextets, mul_ab);
  const auto mul_pairs =
      BitCast(di16, Set(di32, static_cast<int32_t>(0x00011000)));
  const auto merged32 = WidenMulPairwiseAdd(di32, merged16, mul_pairs);

  HWY_ALIGN static const uint8_t kPack[64] = {
      2,  1,  0,  6,  5,  4,  10, 9,  8,  14, 13, 12, 18, 17, 16, 22,
      21, 20, 26, 25, 24, 30, 29, 28, 34, 33, 32, 38, 37, 36, 42, 41,
      40, 46, 45, 44, 50, 49, 48, 54, 53, 52, 58, 57, 56, 62, 61, 60,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  const auto packed =
      TableLookupLanes(BitCast(d, merged32), SetTableIndices(d, kPack));
  StoreN(packed, d, output, 3 * Lanes(d) / 4);
  return invalid;
#else
  VFromD<D> encoded0;
  VFromD<D> encoded1;
  VFromD<D> encoded2;
  VFromD<D> encoded3;
  LoadInterleaved4(d, reinterpret_cast<const uint8_t*>(input), encoded0,
                   encoded1, encoded2, encoded3);

  const auto sextets0 = DecodeBase64Vector(d, encoded0);
  const auto sextets1 = DecodeBase64Vector(d, encoded1);
  const auto sextets2 = DecodeBase64Vector(d, encoded2);
  const auto sextets3 = DecodeBase64Vector(d, encoded3);
  const auto invalid =
      Or3(Or3(encoded0, sextets0, encoded1), Or3(sextets1, encoded2, sextets2),
          Or(encoded3, sextets3));

#if HWY_ARCH_ARM_A64 && HWY_TARGET_IS_NEON
  const auto out0 =
      Vec128<uint8_t>{vsliq_n_u8(vshrq_n_u8(sextets1.raw, 4), sextets0.raw, 2)};
  const auto out1 =
      Vec128<uint8_t>{vsliq_n_u8(vshrq_n_u8(sextets2.raw, 2), sextets1.raw, 4)};
  const auto out2 = Vec128<uint8_t>{vsliq_n_u8(sextets3.raw, sextets2.raw, 6)};
#else
  const auto out0 = Or(ShiftLeft<2>(sextets0), ShiftRight<4>(sextets1));
  const auto out1 = Or(ShiftLeft<4>(sextets1), ShiftRight<2>(sextets2));
  const auto out2 = Or(ShiftLeft<6>(sextets2), sextets3);
#endif
  StoreInterleaved3(out0, out1, out2, d, output);
  return invalid;
#endif
}

}  // namespace detail

// Returns the number of output bytes required to encode `input_size` bytes.
HWY_INLINE size_t Base64EncodedSize(const size_t input_size) {
  return ((input_size + 2) / 3) * 4;
}

// Encodes using the RFC 4648 base64 alphabet. `output` must have room for
// Base64EncodedSize(input_size) bytes. Returns the number of bytes written.
HWY_INLINE size_t Base64Encode(const uint8_t* HWY_RESTRICT input,
                               const size_t input_size,
                               char* HWY_RESTRICT output) {
  size_t in = 0;
  size_t out = 0;
#if (HWY_ARCH_ARM_A64 && HWY_TARGET_IS_NEON) || HWY_TARGET <= HWY_AVX3_DL
  const ScalableTag<uint8_t> d;
  const size_t input_block = 3 * Lanes(d);
  const size_t output_block = 4 * Lanes(d);
#if HWY_TARGET <= HWY_AVX3_DL
  // MultiRotateRight minimizes compute cost for cache-resident input, whereas
  // the interleaved path has higher streaming throughput for larger input.
  constexpr size_t kMultiRotateMaxInputSize = 512 * 1024;
  if (input_size > kMultiRotateMaxInputSize) {
    for (; input_size - in >= input_block;
         in += input_block, out += output_block) {
      detail::EncodeBase64BlockInterleaved(d, input + in, output + out);
    }
  } else {
    for (; input_size - in >= input_block;
         in += input_block, out += output_block) {
      detail::EncodeBase64Block(d, input + in, output + out);
    }
  }
#else
  for (; input_size - in >= input_block;
       in += input_block, out += output_block) {
    detail::EncodeBase64Block(d, input + in, output + out);
  }
#endif
#endif
  if (input_size >= 3) {
    for (; in <= input_size - 3; in += 3, out += 4) {
      detail::EncodeBase64Tail(input + in, 3, output + out);
    }
  }
  if (in != input_size) {
    detail::EncodeBase64Tail(input + in, input_size - in, output + out);
    out += 4;
  }
  return out;
}

// Strictly decodes RFC 4648 base64. Whitespace and non-canonical padding bits
// are rejected. `output` must have room for input_size / 4 * 3 bytes. On
// success, stores the number of bytes written in `output_size`.
HWY_INLINE bool Base64Decode(const char* HWY_RESTRICT input,
                             const size_t input_size,
                             uint8_t* HWY_RESTRICT output,
                             size_t* HWY_RESTRICT output_size) {
  *output_size = 0;
  if (input_size % 4 != 0) return false;

  size_t in = 0;
  size_t out = 0;
  const ScalableTag<uint8_t> d;
  if (CanLookup64(d)) {
    const size_t lanes = Lanes(d);
#if HWY_TARGET <= HWY_AVX3_DL
    const size_t input_block = lanes;
    const size_t output_block = 3 * lanes / 4;
#else
    const size_t input_block = 4 * lanes;
    const size_t output_block = 3 * lanes;
#endif
    size_t simd_end = input_size - input_size % input_block;
    // Only the final two characters can contain legal padding. Leave the final
    // SIMD block to the scalar tail when the input ends on a block boundary.
    if (simd_end == input_size && input_size >= input_block &&
        (input[input_size - 2] == '=' || input[input_size - 1] == '=')) {
      simd_end -= input_block;
    }

    const size_t batch_input = 4 * input_block;
    const size_t batch_output = 4 * output_block;
    if (simd_end >= batch_input) {
      for (; in <= simd_end - batch_input;
           in += batch_input, out += batch_output) {
        const auto invalid0 =
            detail::DecodeBase64Block(d, input + in, output + out);
        const auto invalid1 = detail::DecodeBase64Block(
            d, input + in + input_block, output + out + output_block);
        const auto invalid2 = detail::DecodeBase64Block(
            d, input + in + 2 * input_block, output + out + 2 * output_block);
        const auto invalid3 = detail::DecodeBase64Block(
            d, input + in + 3 * input_block, output + out + 3 * output_block);
        const auto invalid = Or3(invalid0, invalid1, Or(invalid2, invalid3));
        if (HWY_UNLIKELY(
                !AllFalse(d, Ge(invalid, Set(d, uint8_t{0x80}))))) {
          return false;
        }
      }
    }
    if (simd_end >= input_block) {
      for (; in <= simd_end - input_block;
           in += input_block, out += output_block) {
        const auto invalid =
            detail::DecodeBase64Block(d, input + in, output + out);
        if (HWY_UNLIKELY(
                !AllFalse(d, Ge(invalid, Set(d, uint8_t{0x80}))))) {
          return false;
        }
      }
    }
  }

  for (; in < input_size; in += 4) {
    const bool is_last = in + 4 == input_size;
    const uint8_t c0 = static_cast<uint8_t>(input[in + 0]);
    const uint8_t c1 = static_cast<uint8_t>(input[in + 1]);
    const uint8_t c2 = static_cast<uint8_t>(input[in + 2]);
    const uint8_t c3 = static_cast<uint8_t>(input[in + 3]);
    const uint8_t v0 = detail::Base64Value(c0);
    const uint8_t v1 = detail::Base64Value(c1);
    if (v0 == 0xFF || v1 == 0xFF) return false;

    output[out++] = static_cast<uint8_t>((v0 << 2) | (v1 >> 4));
    if (c2 == '=') {
      if (!is_last || c3 != '=' || (v1 & 15) != 0) return false;
      continue;
    }

    const uint8_t v2 = detail::Base64Value(c2);
    if (v2 == 0xFF) return false;
    output[out++] = static_cast<uint8_t>((v1 << 4) | (v2 >> 2));
    if (c3 == '=') {
      if (!is_last || (v2 & 3) != 0) return false;
      continue;
    }

    const uint8_t v3 = detail::Base64Value(c3);
    if (v3 == 0xFF) return false;
    output[out++] = static_cast<uint8_t>((v2 << 6) | v3);
  }

  *output_size = out;
  return true;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // toggle guard
