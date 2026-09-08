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

#include "hwy/contrib/iguana/ans.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "hwy/base.h"  // HWY_DASSERT
#include "hwy/contrib/iguana/ans_detail.h"

namespace hwy {
namespace iguana {

namespace {

uint32_t Read16LE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}
uint32_t Read32LE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
void Append16LE(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back(static_cast<uint8_t>(x));
  v.push_back(static_cast<uint8_t>(x >> 8));
}
void Append16BE(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back(static_cast<uint8_t>(x >> 8));
  v.push_back(static_cast<uint8_t>(x));
}
void Append32LE(std::vector<uint8_t>& v, uint32_t x) {
  Append16LE(v, x & 0xFFFF);
  Append16LE(v, x >> 16);
}
void Append32BE(std::vector<uint8_t>& v, uint32_t x) {
  Append16BE(v, x >> 16);
  Append16BE(v, x & 0xFFFF);
}

// rANS renormalization threshold for a symbol of the given frequency.
uint32_t RenormThreshold(uint32_t freq) {
  return ((kAnsWordL >> kAnsWordMBits) << kAnsWordLBits) * freq;
}

// ------------------------------ Frequency model (Iguana "observe")

struct RawStats {
  uint32_t freqs[256] = {};
  uint32_t cum[257] = {};
};

int Histogram(uint32_t freqs[256], const uint8_t* src, size_t n) {
  uint32_t h[4][256] = {};
  const size_t e = n & ~size_t{3};
  for (size_t i = 0; i < e; i += 4) {
    h[0][src[i + 0]]++;
    h[1][src[i + 1]]++;
    h[2][src[i + 2]]++;
    h[3][src[i + 3]]++;
  }
  for (size_t i = e; i < n; ++i) h[0][src[i]]++;
  for (int i = 0; i < 256; ++i) {
    freqs[i] = h[0][i] + h[1][i] + h[2][i] + h[3][i];
  }
  for (int i = 0; i < 256; ++i) {
    if (freqs[i] != 0) return i;
  }
  return -1;
}

void NormalizeFreqs(RawStats& s) {
  for (int i = 0; i < 256; ++i) s.cum[i + 1] = s.cum[i] + s.freqs[i];

  const uint32_t cur = s.cum[256];
  for (int i = 1; i <= 256; ++i) {
    s.cum[i] = static_cast<uint32_t>(
        (static_cast<uint64_t>(kAnsWordM) * s.cum[i]) / cur);
  }

  // Any symbol that was rounded to zero frequency steals range from the
  // smallest symbol that still has more than one.
  for (int i = 0; i < 256; ++i) {
    if (s.freqs[i] != 0 && s.cum[i + 1] == s.cum[i]) {
      uint32_t best_freq = ~uint32_t{0};
      int best_steal = -1;
      for (int j = 0; j < 256; ++j) {
        const uint32_t f = s.cum[j + 1] - s.cum[j];
        if (f > 1 && f < best_freq) {
          best_freq = f;
          best_steal = j;
        }
      }
      if (best_steal < i) {
        for (int j = best_steal + 1; j <= i; ++j) s.cum[j]--;
      } else {
        for (int j = i + 1; j <= best_steal; ++j) s.cum[j]++;
      }
    }
  }

  for (int i = 0; i < 256; ++i) s.freqs[i] = s.cum[i + 1] - s.cum[i];
}

// ------------------------------ Serialized-table bit stream (LSB-first)

struct BitWriter {
  uint64_t acc = 0;
  int cnt = 0;
  std::vector<uint8_t> buf;
  void Add(uint32_t v, uint32_t k) {
    const uint32_t mask = ~(~uint32_t{0} << k);
    acc |= static_cast<uint64_t>(v & mask) << cnt;
    cnt += static_cast<int>(k);
    while (cnt >= 8) {
      buf.push_back(static_cast<uint8_t>(acc));
      acc >>= 8;
      cnt -= 8;
    }
  }
  void Flush() {
    while (cnt > 0) {
      buf.push_back(static_cast<uint8_t>(acc));
      acc >>= 8;
      cnt -= 8;
    }
  }
};

// Reads nibbles back-to-front; `idx` counts nibbles, `ok` clears on underflow.
uint32_t FetchNibble(const uint8_t* src, int& idx, bool& ok) {
  if (idx < 0) {
    ok = false;
    return 0;
  }
  const uint8_t x = src[static_cast<size_t>(idx) >> 1];
  const uint32_t r = (idx & 1) ? static_cast<uint32_t>(x & 0x0F)
                               : static_cast<uint32_t>(x >> 4);
  --idx;
  return r;
}

void BuildDenseTable(AnsDenseTable& table, const uint32_t freqs[256]) {
  table.assign(kAnsWordM, 0);
  uint32_t start = 0;
  for (uint32_t sym = 0; sym < 256; ++sym) {
    const uint32_t freq = freqs[sym];
    for (uint32_t i = 0; i < freq; ++i) {
      table[start + i] = (sym << 24) | (i << kAnsWordMBits) | freq;
    }
    start += freq;
  }
}

}  // namespace

AnsStatistics AnsStatistics::FromData(const uint8_t* data, size_t size) {
  RawStats s;
  if (size == 0) {
    s.freqs[254] = kAnsWordM / 2;
    s.freqs[255] = kAnsWordM / 2;
    s.cum[255] = kAnsWordM / 2;
    s.cum[256] = kAnsWordM;
  } else {
    const int nz = Histogram(s.freqs, data, size);
    HWY_DASSERT(nz >= 0);
    if (s.freqs[nz] == static_cast<uint32_t>(size)) {
      // Single distinct byte: give it kAnsWordM - 1 so the total is encodable.
      s.freqs[nz] = kAnsWordM - 1;
      for (int i = nz + 1; i < 257; ++i) s.cum[i] = kAnsWordM - 1;
    } else {
      NormalizeFreqs(s);
    }
  }

  AnsStatistics out;
  for (int i = 0; i < 256; ++i) {
    out.packed[i] = (s.cum[i] << kAnsWordMBits) | s.freqs[i];
  }
  return out;
}

void AnsStatistics::Serialize(std::vector<uint8_t>& out) const {
  BitWriter ctrl, data;
  for (int i = 0; i < 256; ++i) {
    const uint32_t f = Freq(static_cast<size_t>(i));
    if (f < 5) {
      ctrl.Add(f, 3);
    } else if (f < 21) {
      ctrl.Add(0b101, 3);
      data.Add(f - 5, 4);
    } else if (f < 277) {
      ctrl.Add(0b110, 3);
      data.Add(f - 21, 8);
    } else {
      ctrl.Add(0b111, 3);
      data.Add(f - 277, 12);
    }
  }
  ctrl.Flush();
  data.Flush();

  const size_t base = out.size();
  out.resize(base + data.buf.size() + ctrl.buf.size());
  for (size_t i = 0; i < data.buf.size(); ++i) {
    out[base + data.buf.size() - i - 1] = data.buf[i];
  }
  if (!ctrl.buf.empty()) {
    memcpy(&out[base + data.buf.size()], ctrl.buf.data(), ctrl.buf.size());
  }
  out.push_back(0);  // "full table" compression level
}

size_t DeserializeAnsTable(AnsDenseTable& table, const uint8_t* src,
                           size_t size) {
  if (size < 1 + kAnsCtrlBlockSize) return SIZE_MAX;
  const uint8_t level = src[size - 1];
  if (level != 0) return SIZE_MAX;  // only the full table is supported here
  size -= 1;

  const uint8_t* ctrl = src + size - kAnsCtrlBlockSize;
  int nibidx = static_cast<int>(size - kAnsCtrlBlockSize - 1) * 2 + 1;
  uint32_t freqs[256] = {};
  bool ok = true;
  int k = 0;
  for (size_t i = 0; i < kAnsCtrlBlockSize; i += 3) {
    uint32_t x = static_cast<uint32_t>(ctrl[i]) |
                 (static_cast<uint32_t>(ctrl[i + 1]) << 8) |
                 (static_cast<uint32_t>(ctrl[i + 2]) << 16);
    for (int j = 0; j < 8; ++j, ++k) {
      const uint32_t v = x & 7;
      x >>= 3;
      if (v == 7) {
        const uint32_t x0 = FetchNibble(src, nibidx, ok);
        const uint32_t x1 = FetchNibble(src, nibidx, ok);
        const uint32_t x2 = FetchNibble(src, nibidx, ok);
        freqs[k] = (x0 | (x1 << 4) | (x2 << 8)) + 277;
      } else if (v == 6) {
        const uint32_t x0 = FetchNibble(src, nibidx, ok);
        const uint32_t x1 = FetchNibble(src, nibidx, ok);
        freqs[k] = (x0 | (x1 << 4)) + 21;
      } else if (v == 5) {
        freqs[k] = FetchNibble(src, nibidx, ok) + 5;
      } else {
        freqs[k] = v;
      }
    }
  }
  if (!ok) return SIZE_MAX;

  // The normalized frequencies sum to kAnsWordM, except for the single-symbol
  // edge case where they sum to kAnsWordM - 1 (see AnsStatistics::FromData).
  uint64_t total = 0;
  for (uint32_t f : freqs) total += f;
  if (total > kAnsWordM) return SIZE_MAX;

  BuildDenseTable(table, freqs);
  return static_cast<size_t>((nibidx + 1) >> 1);
}

// ------------------------------ ANS32 encoder (scalar; unchanged from Iguana)

std::vector<uint8_t> Ans32Encode(const uint8_t* data, size_t size) {
  const AnsStatistics stats = AnsStatistics::FromData(data, size);

  uint32_t state[kAnsLanes];
  for (int i = 0; i < kAnsLanes; ++i) state[i] = kAnsWordL;
  std::vector<uint8_t> fwd, rev;

  const auto put = [&](const uint8_t* chunk, size_t avail) {
    for (int lane = 15; lane >= 0; --lane) {
      if (static_cast<size_t>(lane) >= avail) continue;
      const uint32_t freq = stats.Freq(chunk[lane]);
      const uint32_t start = stats.CumFreq(chunk[lane]);
      uint32_t x = state[lane];
      if (x >= RenormThreshold(freq)) {
        Append16BE(fwd, x & 0xFFFF);
        x >>= kAnsWordLBits;
      }
      state[lane] = ((x / freq) << kAnsWordMBits) + (x % freq) + start;
    }
    for (int lane = 31; lane >= 16; --lane) {
      if (static_cast<size_t>(lane) >= avail) continue;
      const uint32_t freq = stats.Freq(chunk[lane]);
      const uint32_t start = stats.CumFreq(chunk[lane]);
      uint32_t x = state[lane];
      if (x >= RenormThreshold(freq)) {
        Append16LE(rev, x & 0xFFFF);
        x >>= kAnsWordLBits;
      }
      state[lane] = ((x / freq) << kAnsWordMBits) + (x % freq) + start;
    }
  };

  const size_t last = size % 32;
  size_t k = size - last;
  put(data + k, last);
  for (long kk = static_cast<long>(k) - 32; kk >= 0; kk -= 32) {
    put(data + static_cast<size_t>(kk), 32);
  }

  for (int lane = 15; lane >= 0; --lane) Append32BE(fwd, state[lane]);
  for (int lane = 16; lane < 32; ++lane) Append32LE(rev, state[lane]);

  std::vector<uint8_t> out(fwd.rbegin(), fwd.rend());
  out.insert(out.end(), rev.begin(), rev.end());

  AnsStatistics::FromData(data, size).Serialize(out);
  return out;
}

// ------------------------------ ANS32 scalar reference decoder

bool Ans32DecodePayloadScalar(const uint8_t* src, size_t src_size,
                              const AnsDenseTable& table, uint8_t* dst,
                              size_t orig_size) {
  if (src_size < 128) return false;

  uint32_t state[kAnsLanes];
  size_t cursor_fwd = 64;
  size_t cursor_rev = src_size - 64;
  for (int lane = 0; lane < 16; ++lane) {
    state[lane] = Read32LE(src + lane * 4);
    state[lane + 16] =
        Read32LE(src + static_cast<size_t>(lane) * 4 + cursor_rev);
  }

  size_t cursor_dst = 0;
  for (;;) {
    bool stop = false;
    for (int lane = 0; lane < 32; ++lane) {
      const uint32_t x = state[lane];
      const uint32_t t = table[x & kAnsFreqMask];
      const uint32_t freq = t & kAnsFreqMask;
      const uint32_t bias = (t >> kAnsWordMBits) & kAnsFreqMask;
      state[lane] = freq * (x >> kAnsWordMBits) + bias;
      if (cursor_dst < orig_size) {
        dst[cursor_dst++] = static_cast<uint8_t>(t >> 24);
      } else {
        stop = true;
        break;
      }
    }
    if (stop) break;

    for (int lane = 0; lane < 16; ++lane) {
      if (state[lane] < kAnsWordL) {
        if (cursor_fwd + 2 > cursor_rev) return false;
        state[lane] =
            (state[lane] << kAnsWordLBits) | Read16LE(src + cursor_fwd);
        cursor_fwd += 2;
      }
    }
    for (int lane = 16; lane < 32; ++lane) {
      if (state[lane] < kAnsWordL) {
        if (cursor_rev < cursor_fwd + 2) return false;
        state[lane] =
            (state[lane] << kAnsWordLBits) | Read16LE(src + cursor_rev - 2);
        cursor_rev -= 2;
      }
    }
  }
  return true;
}

bool Ans32DecodeScalar(const uint8_t* src, size_t src_size, uint8_t* dst,
                       size_t orig_size) {
  AnsDenseTable table;
  const size_t payload = DeserializeAnsTable(table, src, src_size);
  if (payload == SIZE_MAX) return false;
  return Ans32DecodePayloadScalar(src, payload, table, dst, orig_size);
}

}  // namespace iguana
}  // namespace hwy
