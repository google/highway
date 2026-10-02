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

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/iguana/ans_detail.h"
#include "hwy/highway_export.h"

namespace hwy {
namespace iguana {

namespace {

// rANS renormalization threshold for a symbol of the given frequency.
uint32_t RenormThreshold(uint32_t freq) {
  return ((kAnsWordL >> kAnsWordMBits) << kAnsWordLBits) * freq;
}

// ------------------------------ Division-free rANS encoding (ryg_rans)
//
// The rANS update is `state = (x / freq) * M + (x % freq) + start`, i.e. a
// hardware 32-bit divide (20-40 cycles, not pipelined) for every byte of every
// stream. Rewriting it as `state = x + start + (x / freq) * (M - freq)` leaves
// only the quotient, which a precomputed reciprocal turns into one multiply.
// The table is indexed by every byte of every stream, so its footprint is on
// the encoder's critical path. `bias` and `cmpl_freq` are bounded by the model
// (see BuildEncTable below) and fit in 16 bits, which brings the struct to
// exactly 16 bytes: the 256-entry table is then 4 KiB rather than 6 KiB, and
// scaling the index is a shift instead of a multiply by 24.
struct AnsEncSymbol {
  uint32_t rcp32;  // low 32 bits of ceil(2^(32 + shift) / freq)
  uint32_t x_max;  // renormalize while state >= x_max
  uint16_t bias;
  uint16_t cmpl_freq;  // M - freq
  uint8_t shift;       // ceil(log2(freq)) in [0, 12]
  uint8_t pad[3];
};
static_assert(sizeof(AnsEncSymbol) == 16, "should be a power of two");

// After renormalization, `x < x_max = freq * 2^20`. For freq >= 2 with
// `shift = ceil(log2(freq))` in [1, 12], `rcp = ceil(2^(32 + shift) / freq)`
// lies in `[2^32, 2^33)`, so `rcp = 2^32 + rcp32` where `rcp32` fits in
// uint32_t, and `floor(x / freq) == (x + ((uint64_t(x) * rcp32) >> 32)) >>
// shift` using only a 32x32->64 multiply instead of a 128-bit Mul128.
void BuildEncTable(AnsEncSymbol table[256], const AnsStatistics& stats) {
  for (size_t sym = 0; sym < 256; ++sym) {
    const uint32_t freq = stats.Freq(sym);
    const uint32_t start = stats.CumFreq(sym);
    AnsEncSymbol& s = table[sym];
    s.x_max = RenormThreshold(freq);
    if (freq < 2) {
      s.rcp32 = 0;
      s.bias = static_cast<uint16_t>(start);
      s.cmpl_freq = static_cast<uint16_t>(kAnsWordM - 1);
      s.shift = 0;
    } else {
      const uint32_t shift =
          32u - static_cast<uint32_t>(Num0BitsAboveMS1Bit_Nonzero32(freq - 1));
      const uint64_t num = (uint64_t{1} << (32 + shift)) + freq - 1;
      const uint64_t rcp = num / freq;
      s.rcp32 = static_cast<uint32_t>(rcp);
      s.bias = static_cast<uint16_t>(start);
      s.cmpl_freq = static_cast<uint16_t>(kAnsWordM - freq);
      // When freq == 2^shift, rcp == 2^32 (so rcp32 == 0) and (x + 0) >> shift
      // is already exact.
      s.shift = static_cast<uint8_t>(shift);
    }
  }
}

// Advances `state` past one symbol, emitting the 16 renormalization bits at
// `spec` when they are needed. Whether they are is close to a coin flip, so
// the store is made unconditionally into space the caller has not committed
// yet (and which the next store overwrites); only the cursor advance,
// `2 * renorm`, is conditional, and that is a shift and an add.
HWY_INLINE uint32_t AnsEncPut(uint32_t x, const AnsEncSymbol& s,
                              uint8_t* HWY_RESTRICT spec,
                              uint32_t* HWY_RESTRICT renorm) {
  *renorm = x >= s.x_max ? 1u : 0u;
  ScalarStoreULittleEndian<uint16_t>(static_cast<uint16_t>(x), spec);
  x >>= kAnsWordLBits * *renorm;
  const uint64_t hi = (static_cast<uint64_t>(x) * s.rcp32) >> 32;
  const uint32_t quotient =
      static_cast<uint32_t>((static_cast<uint64_t>(x) + hi) >> s.shift);
  return x + s.bias + quotient * uint32_t{s.cmpl_freq};
}

// ------------------------------ Frequency model (Iguana "observe")

struct RawStats {
  uint32_t freqs[256] = {};
  uint32_t cum[257] = {};
};

int Histogram(uint32_t freqs[256], const uint8_t* src, size_t n) {
  uint32_t h[4][256] = {};
  const size_t e = n & ~size_t{7};
  for (size_t i = 0; i < e; i += 8) {
    const uint64_t w = ScalarLoadULittleEndian<uint64_t>(src + i);
    h[0][static_cast<uint8_t>(w)]++;
    h[1][static_cast<uint8_t>(w >> 8)]++;
    h[2][static_cast<uint8_t>(w >> 16)]++;
    h[3][static_cast<uint8_t>(w >> 24)]++;
    h[0][static_cast<uint8_t>(w >> 32)]++;
    h[1][static_cast<uint8_t>(w >> 40)]++;
    h[2][static_cast<uint8_t>(w >> 48)]++;
    h[3][static_cast<uint8_t>(w >> 56)]++;
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
  uint8_t* pos = nullptr;
  uint8_t* begin = nullptr;

  explicit BitWriter(uint8_t* buf) : pos(buf), begin(buf) {}

  void Add(uint32_t v, uint32_t k) {
    const uint32_t mask = ~(~uint32_t{0} << k);
    acc |= static_cast<uint64_t>(v & mask) << cnt;
    cnt += static_cast<int>(k);
    while (cnt >= 8) {
      *pos++ = static_cast<uint8_t>(acc);
      acc >>= 8;
      cnt -= 8;
    }
  }
  void Flush() {
    while (cnt > 0) {
      *pos++ = static_cast<uint8_t>(acc);
      acc >>= 8;
      cnt -= 8;
    }
  }
  size_t size() const { return static_cast<size_t>(pos - begin); }
};

// Reads nibbles back-to-front; `idx` counts nibbles, `ok` clears on underflow.
uint32_t FetchNibble(const uint8_t* src, int64_t& idx, bool& ok) {
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

// `freqs` sum to kAnsWordM, or to kAnsWordM - 1 in the single-symbol case, in
// which case `single_sym` is that symbol (see AnsStatistics::FromData) and the
// one leftover slot is given to it with frequency 1. Every slot must end up
// with freq >= 1: a zero frequency would collapse the decoder state to zero
// and emit a spurious symbol, and the callers parse untrusted input.
void BuildDenseTable(AnsDenseTable& table, const uint32_t freqs[256],
                     uint32_t single_sym) {
  uint32_t* HWY_RESTRICT dst = table.data();
  uint32_t start = 0;
  for (uint32_t sym = 0; sym < 256; ++sym) {
    const uint32_t freq = freqs[sym];
    const uint32_t base = (sym << 24) | freq;
    uint32_t i = 0;
    for (; i + 4 <= freq; i += 4) {
      const uint32_t b = base + (i << kAnsWordMBits);
      dst[start + i + 0] = b;
      dst[start + i + 1] = b + (1u << kAnsWordMBits);
      dst[start + i + 2] = b + (2u << kAnsWordMBits);
      dst[start + i + 3] = b + (3u << kAnsWordMBits);
    }
    for (; i < freq; ++i) {
      dst[start + i] = base + (i << kAnsWordMBits);
    }
    start += freq;
  }
  HWY_DASSERT(start == kAnsWordM || start == kAnsWordM - 1);
  if (start == kAnsWordM - 1) {
    dst[start] = (single_sym << 24) | 1;
  }
}

}  // namespace

HWY_CONTRIB_DLLEXPORT AnsStatistics AnsStatistics::FromData(
    const uint8_t* data, size_t size) {
  RawStats s;
  // Histogram returns -1 only if every counter is zero, which for size != 0
  // means the uint32_t counters wrapped (size >= 2^32; Ans32Encode rejects
  // that). Treat it like the empty input rather than reading freqs[-1].
  const int nz = size == 0 ? -1 : Histogram(s.freqs, data, size);
  HWY_DASSERT(size == 0 || nz >= 0);
  if (nz < 0) {
    for (int i = 0; i < 256; ++i) s.freqs[i] = 0;
    for (int i = 0; i < 257; ++i) s.cum[i] = 0;
    s.freqs[254] = kAnsWordM / 2;
    s.freqs[255] = kAnsWordM / 2;
    s.cum[255] = kAnsWordM / 2;
    s.cum[256] = kAnsWordM;
  } else if (s.freqs[nz] == static_cast<uint32_t>(size)) {
    // Single distinct byte: give it kAnsWordM - 1 so the total is encodable.
    s.freqs[nz] = kAnsWordM - 1;
    for (int i = nz + 1; i < 257; ++i) s.cum[i] = kAnsWordM - 1;
  } else {
    NormalizeFreqs(s);
  }

  AnsStatistics out;
  for (int i = 0; i < 256; ++i) {
    out.packed[i] = (s.cum[i] << kAnsWordMBits) | s.freqs[i];
  }
  return out;
}

HWY_CONTRIB_DLLEXPORT size_t AnsStatistics::Serialize(uint8_t* out) const {
  // The control block has a fixed length (3 bits x 256 symbols) and is written
  // where it belongs; the data block is stored in reverse, so it is built in a
  // small stack buffer first. Both are bounded, hence no allocation.
  uint8_t data_buf[kAnsMaxSerializedBytes];
  BitWriter ctrl(out + kAnsCtrlBlockSize);  // provisional; moved below
  BitWriter data(data_buf);
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
  const size_t data_size = data.size();
  const size_t ctrl_size = ctrl.size();
  HWY_DASSERT(ctrl_size == kAnsCtrlBlockSize);

  // The control block was written at out + kAnsCtrlBlockSize because
  // `data_size` was not yet known; move it to its real place, which is after
  // the data block.
  memmove(out + data_size, out + kAnsCtrlBlockSize, ctrl_size);
  for (size_t i = 0; i < data_size; ++i) {
    out[data_size - i - 1] = data_buf[i];
  }
  out[data_size + ctrl_size] = 0;  // "full table" compression level
  return data_size + ctrl_size + 1;
}

HWY_CONTRIB_DLLEXPORT size_t DeserializeAnsTable(AnsDenseTable& table,
                                                 const uint8_t* src,
                                                 size_t size) {
  if (size < 1 + kAnsCtrlBlockSize) return SIZE_MAX;
  const uint8_t level = src[size - 1];
  if (level != 0) return SIZE_MAX;  // only the full table is supported here
  size -= 1;

  const uint8_t* ctrl = src + size - kAnsCtrlBlockSize;
  int64_t nibidx =
      (static_cast<int64_t>(size) - int64_t{kAnsCtrlBlockSize} - 1) * 2 + 1;
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
  // Anything else is malformed: a smaller total would leave the dense table
  // with unassigned slots, a larger one would alias symbols.
  uint64_t total = 0;
  for (uint32_t f : freqs) total += f;
  if (total != kAnsWordM && total != kAnsWordM - 1) return SIZE_MAX;
  // A single symbol must not claim the whole table (or more): the cumulative
  // sums below would then overflow their 12 bits and the table would be bogus.
  for (uint32_t f : freqs) {
    if (f >= kAnsWordM) return SIZE_MAX;
  }

  // A total of kAnsWordM - 1 is only valid for the single-symbol table, which
  // AnsStatistics::FromData writes as one symbol with freq kAnsWordM - 1 and
  // the rest zero. Any other split summing to kAnsWordM - 1 would leave a
  // table slot without a symbol.
  uint32_t single_sym = 0;
  if (total == kAnsWordM - 1) {
    bool found = false;
    for (uint32_t sym = 0; sym < 256; ++sym) {
      if (freqs[sym] == kAnsWordM - 1) {
        single_sym = sym;
        found = true;
        break;
      }
    }
    if (!found) return SIZE_MAX;
  }

  BuildDenseTable(table, freqs, single_sym);
  return static_cast<size_t>((nibidx + 1) >> 1);
}

// ------------------------------ ANS32 encoder (scalar)

// Upper bound on the bytes one half of the interleaved payload can occupy.
// Each half consumes 16 of every 32 input symbols, i.e. at most
// `size / 2 + 16` symbols emitting two bytes each, plus 64 bytes of final
// lane state: `size + 96`, rounded up for good measure.
static size_t AnsHalfSize(size_t size) { return size + 128; }

// Bytes reserved at either end of the scratch for AnsEncPut's speculative
// 16-bit stores, which can reach two bytes past the committed region.
constexpr size_t kAnsScratchSlack = 8;

HWY_CONTRIB_DLLEXPORT size_t Ans32EncodeScratchSize(size_t size) {
  // The two halves meet in the middle, and the serialized frequency table
  // follows the upper one.
  return 2 * kAnsScratchSlack + 2 * AnsHalfSize(size) + kAnsMaxSerializedBytes;
}

// The histogram counters and the frequency model are uint32_t, so an input of
// 2^32 bytes or more would wrap them: Histogram could then find no nonzero
// symbol and return -1 (an out-of-bounds freqs[-1] read below), or leave
// NormalizeFreqs dividing by a zero total. Refuse instead; hwy::iguana's
// Compress already caps its input far below this.
HWY_CONTRIB_DLLEXPORT Span<const uint8_t> Ans32Encode(const uint8_t* data,
                                                      size_t size,
                                                      Span<uint8_t> scratch) {
  if (static_cast<uint64_t>(size) >= (uint64_t{1} << 32)) return {};
  if (scratch.size() < Ans32EncodeScratchSize(size)) return {};

  const AnsStatistics stats = AnsStatistics::FromData(data, size);
  AnsEncSymbol enc[256];
  BuildEncTable(enc, stats);

  uint32_t state[kAnsLanes];
  for (int i = 0; i < kAnsLanes; ++i) state[i] = kAnsWordL;

  // rANS encodes back to front, so Iguana's reference implementation builds
  // the lower half of the payload in reverse and flips it at the end. Writing
  // it downwards from the middle of the scratch instead yields the same bytes
  // (a 16-bit big-endian append followed by a whole-buffer byte reversal is a
  // 16-bit little-endian store at a decreasing address), and leaves the two
  // halves already adjacent, so neither a reversal nor a copy is needed.
  uint8_t* const mid = scratch.data() + kAnsScratchSlack + AnsHalfSize(size);
  uint8_t* fwd = mid;  // grows downwards
  uint8_t* rev = mid;  // grows upwards

  const auto put = [&](const uint8_t* chunk, size_t avail) {
    for (int lane = 15; lane >= 0; --lane) {
      if (static_cast<size_t>(lane) >= avail) continue;
      uint32_t renorm;
      state[lane] = AnsEncPut(state[lane], enc[chunk[lane]], fwd - 2, &renorm);
      fwd -= 2 * renorm;
    }
    for (int lane = 31; lane >= 16; --lane) {
      if (static_cast<size_t>(lane) >= avail) continue;
      uint32_t renorm;
      state[lane] = AnsEncPut(state[lane], enc[chunk[lane]], rev, &renorm);
      rev += 2 * renorm;
    }
  };

  const size_t last = size % 32;
  const size_t k = size - last;
  put(data + k, last);
  for (int64_t kk = static_cast<int64_t>(k) - 32; kk >= 0; kk -= 32) {
    put(data + static_cast<size_t>(kk), 32);
  }

  for (int lane = 15; lane >= 0; --lane) {
    fwd -= 4;
    ScalarStoreULittleEndian<uint32_t>(state[lane], fwd);
  }
  for (int lane = 16; lane < 32; ++lane) {
    ScalarStoreULittleEndian<uint32_t>(state[lane], rev);
    rev += 4;
  }

  rev += stats.Serialize(rev);
  return Span<const uint8_t>(fwd, static_cast<size_t>(rev - fwd));
}

// ------------------------------ ANS32 scalar reference decoder

HWY_CONTRIB_DLLEXPORT bool Ans32DecodePayloadScalar(Span<const uint8_t> src,
                                                    const AnsDenseTable& table,
                                                    Span<uint8_t> dst) {
  const size_t payload_size = src.size();
  if (payload_size < 128) return false;
  // Both renormalization loops index relative to the payload base, so unpack
  // it once instead of re-reading the span member on every access.
  const uint8_t* payload = src.data();
  const size_t orig_size = dst.size();

  uint32_t state[kAnsLanes];
  size_t cursor_fwd = 64;
  size_t cursor_rev = payload_size - 64;
  for (int lane = 0; lane < 16; ++lane) {
    state[lane] = ScalarLoadULittleEndian<uint32_t>(payload + lane * 4);
    state[lane + 16] = ScalarLoadULittleEndian<uint32_t>(
        payload + static_cast<size_t>(lane) * 4 + cursor_rev);
  }

  size_t cursor_dst = 0;

  // Bulk loop: runs while a whole chunk of symbols fits in the output and the
  // two payload cursors are far enough apart that all 32 lanes can renormalize
  // (16 x 2 bytes from each end) without meeting. Both conditions only ever
  // become harder, so once one fails the careful loop below takes over and
  // handles the tail plus every malformed input. Hoisting them out lets this
  // loop drop the per-symbol bounds check and, more importantly, replace the
  // near-random `state < kAnsWordL` renormalization branch - one of the worst
  // mispredictors in the decoder - with an unconditional load and a cmov.
  while (orig_size - cursor_dst >= kAnsLanes &&
         cursor_rev - cursor_fwd >= 2 * kAnsLanes) {
    for (int lane = 0; lane < 32; ++lane) {
      const uint32_t x = state[lane];
      const uint32_t t = table[x & kAnsFreqMask];
      const uint32_t freq = t & kAnsFreqMask;
      const uint32_t bias = (t >> kAnsWordMBits) & kAnsFreqMask;
      state[lane] = freq * (x >> kAnsWordMBits) + bias;
      dst[cursor_dst + static_cast<size_t>(lane)] =
          static_cast<uint8_t>(t >> 24);
    }
    cursor_dst += kAnsLanes;

    for (int lane = 0; lane < 16; ++lane) {
      const uint32_t s = state[lane];
      const uint32_t need = s < kAnsWordL ? 1u : 0u;
      const uint32_t w =
          ScalarLoadULittleEndian<uint16_t>(payload + cursor_fwd);
      state[lane] = need ? ((s << kAnsWordLBits) | w) : s;
      cursor_fwd += 2 * need;
    }
    for (int lane = 16; lane < 32; ++lane) {
      const uint32_t s = state[lane];
      const uint32_t need = s < kAnsWordL ? 1u : 0u;
      const uint32_t w =
          ScalarLoadULittleEndian<uint16_t>(payload + cursor_rev - 2);
      state[lane] = need ? ((s << kAnsWordLBits) | w) : s;
      cursor_rev -= 2 * need;
    }
  }

  // Careful loop: tail of the output, and all malformed input.
  for (;;) {
    for (int lane = 0; lane < 32; ++lane) {
      const uint32_t x = state[lane];
      const uint32_t t = table[x & kAnsFreqMask];
      const uint32_t freq = t & kAnsFreqMask;
      const uint32_t bias = (t >> kAnsWordMBits) & kAnsFreqMask;
      state[lane] = freq * (x >> kAnsWordMBits) + bias;
      if (cursor_dst < orig_size) {
        dst[cursor_dst++] = static_cast<uint8_t>(t >> 24);
      } else {
        return true;
      }
    }

    for (int lane = 0; lane < 16; ++lane) {
      if (state[lane] < kAnsWordL) {
        if (cursor_fwd + 2 > cursor_rev) return false;
        state[lane] = (state[lane] << kAnsWordLBits) |
                      ScalarLoadULittleEndian<uint16_t>(payload + cursor_fwd);
        cursor_fwd += 2;
      }
    }
    for (int lane = 16; lane < 32; ++lane) {
      if (state[lane] < kAnsWordL) {
        if (cursor_rev < cursor_fwd + 2) return false;
        state[lane] =
            (state[lane] << kAnsWordLBits) |
            ScalarLoadULittleEndian<uint16_t>(payload + cursor_rev - 2);
        cursor_rev -= 2;
      }
    }
  }
}

HWY_CONTRIB_DLLEXPORT bool Ans32DecodeScalar(Span<const uint8_t> src,
                                             Span<uint8_t> dst) {
  HWY_ALIGN_MAX AnsDenseTable table;
  const size_t payload = DeserializeAnsTable(table, src.data(), src.size());
  if (payload == SIZE_MAX) return false;
  return Ans32DecodePayloadScalar(src.first(payload), table, dst);
}

}  // namespace iguana
}  // namespace hwy
