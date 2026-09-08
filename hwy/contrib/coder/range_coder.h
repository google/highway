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

// Non-SIMD parts of the interleaved range coder: the scalar encoder/decoder,
// the probability-model helpers, the (scalar) interleaved encoder, and a
// scalar reference decoder. The SIMD interleaved decoder is in
// range_coder-inl.h.
//
// This is a re-implementation of Richard Geldreich's public-domain
// "sserangecoding" (https://github.com/richgel999/sserangecoding), a 24-bit
// interleaved Pavlov/Subbotin range coder over an 8-bit alphabet with 16
// interleaved streams. The bitstream format is byte-for-byte identical to that
// project; only the decoder is expressed via Highway ops so a single source
// runs on every target.

#ifndef HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_H_
#define HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include <vector>

#include "hwy/base.h"  // HWY_DASSERT

namespace hwy {

// ------------------------------ Constants (part of the bitstream format)

// Number of fractional bits in a scaled probability; probabilities are in
// [0, kRangeProbScale].
constexpr uint32_t kRangeProbBits = 12;
constexpr uint32_t kRangeProbScale = 1u << kRangeProbBits;  // 4096

// The coder keeps `length` in [kRangeMinLen, kRangeMaxLen] (24-bit range).
constexpr uint32_t kRangeMinLen = 0x00010000u;
constexpr uint32_t kRangeMaxLen = 0x00FFFFFFu;

constexpr uint32_t kRangeMinSyms = 2;
constexpr uint32_t kRangeMaxSyms = 256;

// Number of interleaved streams used by EncodeInterleaved / DecodeInterleaved.
constexpr uint32_t kRangeLanes = 16;
constexpr uint32_t kRangeLaneMask = kRangeLanes - 1;

// ------------------------------ Scalar encoder

// Direct translation of sserangecoding's range_enc.
class RangeEncoder {
 public:
  RangeEncoder() { Init(); }

  void Init() {
    base_ = 0;
    length_ = kRangeMaxLen;
    buf_.clear();
    buf_.reserve(4096);
  }

  // Encodes the interval [low_prob, high_prob) (both scaled by
  // kRangeProbScale).
  void Encode(uint32_t low_prob, uint32_t high_prob) {
    HWY_DASSERT((low_prob < high_prob) && (high_prob <= kRangeProbScale));
    HWY_DASSERT((high_prob - low_prob) < kRangeProbScale);

    const uint32_t r = length_ >> kRangeProbBits;
    const uint32_t l = low_prob * r;
    const uint32_t h = high_prob * r;

    const uint32_t orig_base = base_;
    base_ = (base_ + l) & kRangeMaxLen;
    length_ = h - l;

    if (orig_base > base_) PropagateCarry();
    if (length_ < kRangeMinLen) RenormEncInterval();
  }

  void Flush() {
    const uint32_t orig_base = base_;

    if (length_ > 2 * kRangeMinLen) {
      base_ = (base_ + kRangeMinLen) & kRangeMaxLen;
      length_ = kRangeMinLen >> 1;
    } else {
      base_ = (base_ + (kRangeMinLen >> 1)) & kRangeMaxLen;
      length_ = kRangeMinLen >> 9;
    }

    if (orig_base > base_) PropagateCarry();
    RenormEncInterval();

    while (buf_.size() < 3) buf_.push_back(0);
    buf_.push_back(0);
    buf_.push_back(0);
  }

  const std::vector<uint8_t>& buf() const { return buf_; }
  std::vector<uint8_t>& buf() { return buf_; }

 private:
  void PropagateCarry() {
    if (buf_.empty()) return;
    size_t index = buf_.size() - 1;
    for (;;) {
      uint8_t& c = buf_[index];
      if (c == 0xFF) {
        c = 0;
      } else {
        ++c;
        break;
      }
      if (index == 0) break;
      --index;
    }
  }

  void RenormEncInterval() {
    HWY_DASSERT((base_ & ~kRangeMaxLen) == 0);
    do {
      buf_.push_back(static_cast<uint8_t>(base_ >> 16));
      base_ = (base_ << 8) & kRangeMaxLen;
      length_ <<= 8;
    } while (length_ < kRangeMinLen);
  }

  uint32_t base_;
  uint32_t length_;
  std::vector<uint8_t> buf_;
};

// ------------------------------ Scalar decoder

// Direct translation of sserangecoding's range_dec. Used for the scalar tail of
// the SIMD decoder and as a reference implementation.
class RangeDecoder {
 public:
  RangeDecoder() : length_(0), value_(0) {}

  // Reads 3 bytes (a big-endian 24-bit value) and advances `p`.
  void Init(const uint8_t*& p) {
    length_ = kRangeMaxLen;
    value_ = static_cast<uint32_t>(p[0]) << 16;
    value_ |= static_cast<uint32_t>(p[1]) << 8;
    value_ |= static_cast<uint32_t>(p[2]);
    p += 3;
  }

  // Decodes one symbol using `table` (see BuildDecodeTable) and consumes 0..2
  // bytes from `cur`.
  uint32_t DecodeSymbol(const uint32_t* table, const uint8_t*& cur) {
    const uint32_t r = length_ >> kRangeProbBits;
    const uint32_t q = value_ / r;

    // The AND is only for safety against corrupted input.
    const uint32_t encoded_val = table[q & (kRangeProbScale - 1)];

    const uint32_t sym = encoded_val & 0xFFu;
    const uint32_t low_prob = (encoded_val >> 8) & (kRangeProbScale - 1);
    const uint32_t prob_range = encoded_val >> (8 + kRangeProbBits);

    HWY_DASSERT(q >= low_prob && q < (low_prob + prob_range));

    value_ -= low_prob * r;
    length_ = prob_range * r;

    while (length_ < kRangeMinLen) {
      value_ = (value_ << 8) | static_cast<uint32_t>(*cur++);
      length_ <<= 8;
    }
    return sym;
  }

  uint32_t length_;
  uint32_t value_;
};

// ------------------------------ Probability model

// Scales `freq` (per-symbol counts, one entry per symbol) into a cumulative
// probability table `scaled_cum_prob` of size freq.size() + 1, summing to
// kRangeProbScale. `freq` may be modified if only one symbol was used. Returns
// false if the model is degenerate.
bool CreateCumulativeProbs(std::vector<uint32_t>& scaled_cum_prob,
                           std::vector<uint32_t>& freq);

// Builds the 4096-entry decode lookup table from `scaled_cum_prob` (as produced
// by CreateCumulativeProbs for `num_syms` symbols). Each entry packs
// sym | (cum_prob << 8) | (prob_range << 20).
void BuildDecodeTable(uint32_t num_syms,
                      const std::vector<uint32_t>& scaled_cum_prob,
                      std::vector<uint32_t>& table);

// ------------------------------ Interleaved codec

// Encodes `data` into 16 interleaved range-coded streams, appending the result
// to `enc_buf` layout expected by DecodeInterleaved (48 header bytes, then the
// renormalization bytes in stream-round-robin order, then 2 zero pad bytes).
void EncodeInterleaved(const std::vector<uint8_t>& data,
                       std::vector<uint8_t>& enc_buf,
                       const std::vector<uint32_t>& scaled_cum_prob);

// Scalar reference decoder for data produced by EncodeInterleaved. `table` is
// from BuildDecodeTable. Returns false if the input is truncated. The SIMD
// version (range_coder-inl.h) produces identical output.
bool DecodeInterleavedScalar(const uint8_t* src, size_t comp_size, uint8_t* dst,
                             size_t orig_size, const uint32_t* table);

// ------------------------------ SIMD decoder shuffle tables

// Per-normalization-mask (8-bit: bits 0..3 = "stream needs >=1 byte",
// bits 4..7 = "stream needs 2 bytes") shuffle constants used by the SIMD
// decoder. `shift`/`dist` are pshufb-style byte indices (0x80 => zero).
struct RangeShuffleTables {
  uint8_t num_bytes[256];
  uint8_t shift[256][16];
  uint8_t dist[256][16];
};

// Returns a lazily-built, immutable instance (thread-safe since C++11).
const RangeShuffleTables& GetRangeShuffleTables();

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_CODER_RANGE_CODER_H_
