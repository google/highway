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

// Non-SIMD parts of Iguana's ANS32 entropy coder: the frequency model, the
// dense-table (de)serialization, and the scalar encoder / reference decoder.
// The SIMD decoder is in ans-inl.h.
//
// This is a port of the 32-way interleaved rANS entropy stage of Iguana
// (github.com/SnellerInc/sneller, ion/zion/iguana), which is the "10 GB/s"
// core of that compressor. It is based on the pure-Go reference implementation
// (ans32.go, ans_statistics.go); the bitstream is byte-for-byte compatible.
// The rANS scheme itself is Fabian Giesen's ryg_rans.
//
// This covers only the entropy stage. Iguana's LZ77 layer is a possible
// follow-up.

#ifndef HIGHWAY_HWY_CONTRIB_IGUANA_ANS_H_
#define HIGHWAY_HWY_CONTRIB_IGUANA_ANS_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace hwy {
namespace iguana {

// ------------------------------ Constants (bitstream format)

constexpr uint32_t kAnsWordLBits = 16;
constexpr uint32_t kAnsWordL = uint32_t{1} << kAnsWordLBits;  // 65536
constexpr uint32_t kAnsWordMBits = 12;
constexpr uint32_t kAnsWordM = uint32_t{1} << kAnsWordMBits;  // 4096
constexpr uint32_t kAnsFreqMask = kAnsWordM - 1;

// 32 interleaved rANS streams: 16 written/read forwards, 16 backwards.
constexpr int kAnsLanes = 32;

// Longest possible serialized frequency table.
constexpr size_t kAnsCtrlBlockSize = 96;
constexpr size_t kAnsDenseTableMaxLength = kAnsCtrlBlockSize + 384;

// ------------------------------ Frequency model

// Normalized per-symbol frequencies, summing to kAnsWordM. `packed[i]` is
// (cumulative_freq[i] << 12) | freq[i]; `Freq()` / `CumFreq()` unpack it.
struct AnsStatistics {
  uint32_t packed[256] = {};

  uint32_t Freq(size_t sym) const { return packed[sym] & kAnsFreqMask; }
  uint32_t CumFreq(size_t sym) const {
    return (packed[sym] >> kAnsWordMBits) & kAnsFreqMask;
  }

  // Builds the model from `data` (the Iguana "observe" step: histogram +
  // normalization, with the empty-input and single-symbol edge cases).
  static AnsStatistics FromData(const uint8_t* data, size_t size);

  // Appends the serialized table (Iguana "EncodeFull" + a zero level byte).
  void Serialize(std::vector<uint8_t>& out) const;
};

// 4096-entry rANS decoding table: entry = (sym << 24) | (i << 12) | freq.
using AnsDenseTable = std::vector<uint32_t>;

// Parses a table serialized by AnsStatistics::Serialize from the END of
// `src[0, size)`. Returns the length of the data that precedes it (the rANS
// payload), or SIZE_MAX on malformed input.
size_t DeserializeAnsTable(AnsDenseTable& table, const uint8_t* src,
                           size_t size);

// ------------------------------ ANS32 codec

// Encodes `data` into the 32-way interleaved rANS payload followed by the
// serialized frequency table (i.e. a complete ANS32 block).
std::vector<uint8_t> Ans32Encode(const uint8_t* data, size_t size);

// Scalar reference decoder for a block produced by Ans32Encode. `orig_size` is
// the decompressed length. Returns false on malformed input. The SIMD decoder
// (ans-inl.h) produces identical output.
bool Ans32DecodeScalar(const uint8_t* src, size_t src_size, uint8_t* dst,
                       size_t orig_size);

// Same, but the frequency table has already been parsed: `payload` is the rANS
// data only (DeserializeAnsTable's prefix length), `table` its dense table.
bool Ans32DecodePayloadScalar(const uint8_t* payload, size_t payload_size,
                              const AnsDenseTable& table, uint8_t* dst,
                              size_t orig_size);

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_ANS_H_
