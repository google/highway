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

#include "hwy/contrib/coder/range_coder.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include <vector>

#include "hwy/base.h"  // HWY_DASSERT

namespace hwy {

namespace {

uint32_t ClampU32(uint32_t v, uint32_t lo, uint32_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

uint32_t ReadBE24(const uint8_t*& src) {
  const uint32_t res = (static_cast<uint32_t>(src[0]) << 16) |
                       (static_cast<uint32_t>(src[1]) << 8) |
                       static_cast<uint32_t>(src[2]);
  src += 3;
  return res;
}

}  // namespace

bool CreateCumulativeProbs(std::vector<uint32_t>& scaled_cum_prob,
                           std::vector<uint32_t>& freq) {
  const uint32_t num_syms = static_cast<uint32_t>(freq.size());
  if (num_syms < kRangeMinSyms || num_syms > kRangeMaxSyms) return false;

  uint64_t total_freq = 0;
  uint32_t total_used_syms = 0;
  for (uint32_t i = 0; i < num_syms; ++i) {
    total_freq += freq[i];
    if (freq[i]) ++total_used_syms;
  }
  if (total_used_syms == 0) return false;

  // The coder needs at least two live symbols; synthesize a second one.
  if (total_used_syms == 1) {
    for (uint32_t i = 0; i < num_syms; ++i) {
      if (!freq[i]) {
        freq[i]++;
        total_freq++;
        break;
      }
    }
    total_used_syms++;
  }

  scaled_cum_prob.resize(num_syms + 1);

  uint32_t sym_index_to_boost = 0, boost_amount = 0;

  // Lower the effective scale until no live symbol rounds to a frequency of 0
  // (those get bumped to 1, which could overshoot the total).
  uint32_t adjusted_prob_scale = kRangeProbScale;
  for (;;) {
    uint32_t num_truncated_syms = 0;
    for (uint32_t i = 0; i < num_syms; ++i) {
      if (freq[i]) {
        const uint32_t l = static_cast<uint32_t>(
            (static_cast<uint64_t>(freq[i]) * adjusted_prob_scale) /
            total_freq);
        if (!l) ++num_truncated_syms;
      }
    }
    if (!num_truncated_syms) break;

    const uint32_t new_scale = kRangeProbScale - num_truncated_syms;
    if (new_scale == adjusted_prob_scale) break;
    adjusted_prob_scale = new_scale;
  }

  for (uint32_t pass = 0; pass < 2; ++pass) {
    uint32_t most_prob_sym_freq = 0, most_prob_sym_index = 0;

    uint32_t ci = 0;
    for (uint32_t i = 0; i < num_syms; ++i) {
      scaled_cum_prob[i] = ci;
      if (!freq[i]) continue;

      if (freq[i] > most_prob_sym_freq) {
        most_prob_sym_freq = freq[i];
        most_prob_sym_index = i;
      }

      uint32_t l = static_cast<uint32_t>(
          (static_cast<uint64_t>(freq[i]) * adjusted_prob_scale) / total_freq);
      l = ClampU32(l, 1, kRangeProbScale - (total_used_syms - 1));

      if (pass && i == sym_index_to_boost) l += boost_amount;

      ci += l;
      if (ci > kRangeProbScale) return false;
    }
    scaled_cum_prob[num_syms] = kRangeProbScale;

    if (ci == kRangeProbScale) break;
    if (pass) return false;  // should not happen

    sym_index_to_boost = most_prob_sym_index;
    boost_amount = kRangeProbScale - ci;
  }

  return true;
}

void BuildDecodeTable(uint32_t num_syms,
                      const std::vector<uint32_t>& scaled_cum_prob,
                      std::vector<uint32_t>& table) {
  HWY_DASSERT(scaled_cum_prob.size() == num_syms + 1);
  table.assign(kRangeProbScale, 0);

  for (uint32_t sym = 0; sym < num_syms; ++sym) {
    const uint32_t lo = scaled_cum_prob[sym];
    const uint32_t hi = scaled_cum_prob[sym + 1];
    const uint32_t n = hi - lo;
    if (!n) continue;

    HWY_DASSERT(lo < kRangeProbScale && n < kRangeProbScale);
    const uint32_t k = sym | (lo << 8) | (n << 20);
    for (uint32_t j = 0; j < n; ++j) table[lo + j] = k;
  }
}

void EncodeInterleaved(const std::vector<uint8_t>& data,
                       std::vector<uint8_t>& enc_buf,
                       const std::vector<uint32_t>& scaled_cum_prob) {
  const size_t file_size = data.size();
  HWY_DASSERT(file_size != 0);

  std::vector<RangeEncoder> encs(kRangeLanes);
  std::vector<uint8_t> bytes_written(file_size);
  uint64_t total_enc_size = 0;

  for (uint32_t i = 0; i < kRangeLanes; ++i) {
    encs[i].buf().reserve(1 + file_size / kRangeLanes);
  }

  for (size_t i = 0; i < file_size; ++i) {
    const uint32_t sym = data[i];
    const uint32_t lane = static_cast<uint32_t>(i) & kRangeLaneMask;

    const size_t before = encs[lane].buf().size();
    encs[lane].Encode(scaled_cum_prob[sym], scaled_cum_prob[sym + 1]);
    const size_t after = encs[lane].buf().size();

    bytes_written[i] = static_cast<uint8_t>(after - before);
    total_enc_size += after - before;
  }

  for (uint32_t lane = 0; lane < kRangeLanes; ++lane) encs[lane].Flush();

  uint32_t cur_ofs[kRangeLanes] = {0};
  const uint64_t final_size = kRangeLanes * 3 + total_enc_size + 2;
  enc_buf.resize(static_cast<size_t>(final_size));

  uint8_t* dst = enc_buf.data();

  for (uint32_t lane = 0; lane < kRangeLanes; ++lane) {
    for (uint32_t j = 0; j < 3; ++j) {
      *dst++ = encs[lane].buf()[cur_ofs[lane]++];
    }
  }

  for (size_t i = 0; i < file_size; ++i) {
    const uint32_t num_bytes = bytes_written[i];
    if (!num_bytes) continue;
    const uint32_t lane = static_cast<uint32_t>(i) & kRangeLaneMask;
    memcpy(dst, &encs[lane].buf()[cur_ofs[lane]], num_bytes);
    dst += num_bytes;
    cur_ofs[lane] += num_bytes;
  }

  *dst++ = 0;
  *dst++ = 0;
  HWY_DASSERT(static_cast<size_t>(dst - enc_buf.data()) == enc_buf.size());
}

bool DecodeInterleavedScalar(const uint8_t* src, size_t comp_size, uint8_t* dst,
                             size_t orig_size, const uint32_t* table) {
  // Decode from a zero-padded copy so a truncated/corrupt stream can never read
  // out of bounds; a valid stream never reads past `comp_size`.
  std::vector<uint8_t> buf;
  buf.reserve(comp_size + 16);
  buf.assign(src, src + comp_size);
  buf.resize(comp_size + 16, 0);

  const uint8_t* const p_start = buf.data();
  const uint8_t* p = p_start;

  uint32_t value[kRangeLanes];
  uint32_t length[kRangeLanes];
  for (uint32_t s = 0; s < kRangeLanes; ++s) {
    value[s] = ReadBE24(p);
    length[s] = kRangeMaxLen;
  }

  RangeDecoder dec;
  for (size_t i = 0; i < orig_size; ++i) {
    const uint32_t s = static_cast<uint32_t>(i) & kRangeLaneMask;
    dec.length_ = length[s];
    dec.value_ = value[s];
    dst[i] = static_cast<uint8_t>(dec.DecodeSymbol(table, p));
    length[s] = dec.length_;
    value[s] = dec.value_;
  }

  return static_cast<size_t>(p - p_start) <= comp_size;
}

const RangeShuffleTables& GetRangeShuffleTables() {
  static const RangeShuffleTables tables = [] {
    RangeShuffleTables t;
    memset(&t, 0, sizeof(t));

    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t nb = 0;
      for (uint32_t j = 0; j < 4; ++j) {
        if ((i >> j) & 0x10) {
          nb += 2;
        } else if ((i >> j) & 1) {
          nb += 1;
        }
      }
      t.num_bytes[i] = static_cast<uint8_t>(nb);

      // "shift": rotate each 4-byte lane right to make room for new low bytes.
      for (uint32_t j = 0; j < 4; ++j) {
        uint8_t* x = &t.shift[i][j * 4];
        if ((i >> j) & 0x10) {
          x[0] = 0x80;
          x[1] = 0x80;
          x[2] = static_cast<uint8_t>(j * 4 + 0);
          x[3] = static_cast<uint8_t>(j * 4 + 1);
        } else if ((i >> j) & 1) {
          x[0] = 0x80;
          x[1] = static_cast<uint8_t>(j * 4 + 0);
          x[2] = static_cast<uint8_t>(j * 4 + 1);
          x[3] = static_cast<uint8_t>(j * 4 + 2);
        } else {
          x[0] = static_cast<uint8_t>(j * 4 + 0);
          x[1] = static_cast<uint8_t>(j * 4 + 1);
          x[2] = static_cast<uint8_t>(j * 4 + 2);
          x[3] = static_cast<uint8_t>(j * 4 + 3);
        }
      }

      // "dist": scatter the freshly loaded source bytes into the freed slots.
      uint32_t src_ofs = 0;
      for (uint32_t j = 0; j < 4; ++j) {
        uint8_t* x = &t.dist[i][j * 4];
        if ((i >> j) & 0x10) {
          x[0] = static_cast<uint8_t>(src_ofs + 1);
          x[1] = static_cast<uint8_t>(src_ofs);
          x[2] = 0x80;
          x[3] = 0x80;
          src_ofs += 2;
        } else if ((i >> j) & 1) {
          x[0] = static_cast<uint8_t>(src_ofs++);
          x[1] = 0x80;
          x[2] = 0x80;
          x[3] = 0x80;
        } else {
          x[0] = 0x80;
          x[1] = 0x80;
          x[2] = 0x80;
          x[3] = 0x80;
        }
      }
    }
    return t;
  }();
  return tables;
}

}  // namespace hwy
