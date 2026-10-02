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

#include "hwy/contrib/iguana/iguana.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <utility>  // std::move

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/iguana/ans.h"
#include "hwy/contrib/iguana/iguana_detail.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_IGUANA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/iguana/iguana.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/iguana/iguana-inl.h"

#if HWY_ONCE
namespace hwy {
namespace iguana {
namespace {

constexpr uint32_t kMaxU16 = (1u << 16) - 1;
constexpr uint32_t kVarThresh1 = 254;
constexpr uint32_t kVarThresh3 = 254u * 254;
// Largest value the 4-byte stream varint can represent (byte 0 is 255, then
// v %% 254, t %% 254, t / 254 with t = v / 254 < 254).
constexpr uint32_t kMaxStreamVarint = 254u * 254u * 254u - 1;
// A match length is transmitted as (len - kMaxShortMatchLen), so this is the
// longest match the format can encode in one token; longer ones are split.
constexpr int64_t kMaxEncodableMatchLen =
    static_cast<int64_t>(kMaxStreamVarint) + kMaxShortMatchLen;
// Same idea for the literal length, which is sent as (len - kMaxShortLitLen).
constexpr size_t kMaxEncodableLitLen = kMaxStreamVarint + kMaxShortLitLen;

// ------------------------------ byte stream writer (encoder)

// Append-only view of a preallocated buffer. Every stream the encoder writes
// has a size bound that StreamCapacities() derives from the input length, so
// the buffer never has to grow: unlike std::vector this needs neither a
// capacity check per byte nor a reallocation + copy. Overruns are a bug in
// those bounds rather than an input-dependent condition, hence HWY_DASSERT
// plus the Overflowed() check Compress() runs once at the end.
struct ByteWriter {
  uint8_t* pos = nullptr;
  const uint8_t* begin = nullptr;
  const uint8_t* end = nullptr;

  ByteWriter() = default;
  ByteWriter(uint8_t* buf, size_t capacity)
      : pos(buf), begin(buf), end(buf + capacity) {}

  void Push(uint8_t v) {
    HWY_DASSERT(pos < end);
    *pos++ = v;
  }
  void Append(const uint8_t* p, size_t num) {
    HWY_DASSERT(static_cast<size_t>(end - pos) >= num);
    CopyBytes(p, pos, num);
    pos += num;
  }

  const uint8_t* data() const { return begin; }
  size_t size() const { return static_cast<size_t>(pos - begin); }
  bool empty() const { return pos == begin; }
  bool Overflowed() const { return pos > end; }
};

void AppendVarUint(ByteWriter& s, uint32_t v) {
  if (v < kVarThresh1) {
    s.Push(static_cast<uint8_t>(v));
  } else if (v < kVarThresh3) {
    s.Push(254);
    s.Push(static_cast<uint8_t>(v % 254));
    s.Push(static_cast<uint8_t>(v / 254));
  } else {
    HWY_DASSERT(v < 254u * 254 * 254);  // fits the encoder's stream varint
    const uint32_t t = v / 254;
    s.Push(255);
    s.Push(static_cast<uint8_t>(v % 254));
    s.Push(static_cast<uint8_t>(t % 254));
    s.Push(static_cast<uint8_t>(t / 254));
  }
}
void AppendU24(ByteWriter& s, uint32_t v) {
  s.Push(static_cast<uint8_t>(v));
  s.Push(static_cast<uint8_t>(v >> 8));
  s.Push(static_cast<uint8_t>(v >> 16));
}
void AppendU16(ByteWriter& s, uint32_t v) {
  s.Push(static_cast<uint8_t>(v));
  s.Push(static_cast<uint8_t>(v >> 8));
}

// ------------------------------ control-byte writer (encoder)

// One command byte plus a 10-byte length varint, then, for kCmdDecodeIguana,
// the stream-type nibbles and up to two varints per stream. 128 bytes is well
// above that maximum.
constexpr size_t kMaxControlBytes = 128;

struct ControlWriter {
  uint8_t buf[kMaxControlBytes];
  ByteWriter ctrl{buf, kMaxControlBytes};
  int64_t last_command_offset = -1;

  void Command(uint8_t v) {
    if (last_command_offset >= 0) {
      buf[static_cast<size_t>(last_command_offset)] &= kCommandMask;
    }
    last_command_offset = static_cast<int64_t>(ctrl.size());
    ctrl.Push(static_cast<uint8_t>(v | kLastCommandMarker));
  }
  void VarUint(uint64_t v) {
    // Num0BitsAboveMS1Bit_Nonzero64 gives the index of the highest set bit, so
    // bit_len == that index + 1 (and 0 for v == 0).
    const int bit_len =
        v == 0 ? 0 : static_cast<int>(64 - Num0BitsAboveMS1Bit_Nonzero64(v));
    const int count = bit_len / 7 + 1;
    for (int i = count - 1; i >= 0; --i) {
      uint32_t x = static_cast<uint32_t>(v >> (i * 7)) & 0x7Fu;
      if (i == 0) x |= 0x80u;
      ctrl.Push(static_cast<uint8_t>(x));
    }
  }
};

// ------------------------------ match finder (encoder)
//
// Lizard-style hash chain. `chains` is a direct-indexed hash table: one bucket
// per HashSeq() value (kChainBits bits), and each bucket keeps the most recent
// kHistSize positions that hashed to it, newest first. Those positions form a
// short chain of candidate matches for the current position - the same 5-byte
// sequence often repeats with different history, so probing the last few
// occurrences is enough to find a good match without a full search. Insert()
// pushes a position onto its bucket (shifting the previous candidates down);
// BestChainMatch() walks the chain and MatchExtend() extends each candidate
// backwards to see which one makes the longest encodable match.

// The encoder deliberately interprets the input as *little-endian* words
// regardless of the host: Lcp() locates the first differing byte via the
// lowest set bit, and HashSeq() folds the low kHashBytes. Using native byte
// order on a big-endian host would break the former and make the latter
// select different matches, so the same input would compress to a different
// (still valid) bitstream depending on the architecture.
// ScalarLoadULittleEndian is a single unaligned load plus, on big-endian
// hosts, one byte swap.

// Returns both the bucket index (the top kChainBits of the mix, in bits
// [kChainBits-1:0]) and an 8-bit collision tag (the next 8 bits, moved to bits
// [31:24] so it can be OR-ed straight into a chain entry), both derived from
// the same 5-byte prefix `u & 0xFFFFFFFFFF`. The two bit ranges are disjoint,
// so the tag carries information the bucket index does not.
HWY_INLINE uint32_t HashAndTagU64(uint64_t u, uint32_t* HWY_RESTRICT out_tag) {
  const uint64_t mixed = (u << 24) * 889523592379ull;  // kHashBytes == 5
  *out_tag = (static_cast<uint32_t>(mixed >> (64 - kChainBits - 8)) & 0xFFu)
             << 24;
  return static_cast<uint32_t>(mixed >> (64 - kChainBits));
}

// Longest common prefix of src[lo..] and src[hi..] (lo < hi), starting at m.
int64_t Lcp(const uint8_t* src, size_t src_len, int64_t lo, int64_t hi,
            int64_t m = 0) {
  const int64_t n = static_cast<int64_t>(src_len);
  while (n - (hi + m) >= 8) {
    // Little-endian loads: Num0BitsBelowLS1Bit_Nonzero64 below counts from
    // the least significant *bit*, which must be the first byte.
    const uint64_t a = ScalarLoadULittleEndian<uint64_t>(src + lo + m);
    const uint64_t b = ScalarLoadULittleEndian<uint64_t>(src + hi + m);
    const uint64_t d = a ^ b;
    if (d == 0) {
      m += 8;
      continue;
    }
    // First differing byte: Num0BitsBelowLS1Bit_Nonzero64 is the index of the
    // lowest set bit, i.e. 8 * (byte index) + bit index within that byte.
    const size_t first_diff = Num0BitsBelowLS1Bit_Nonzero64(d);
    return m + static_cast<int64_t>(first_diff / 8);
  }
  while (n - (hi + m) > 0 && src[lo + m] == src[hi + m]) ++m;
  return m;
}

// A match is encodable if its offset either fits the 16-bit stream, or is long
// enough to be worth the 24-bit one; in both cases the length has to fit the
// 4-byte stream varint (AddU24 caps the offset, kMaxEncodableMatchLen the len).
bool IsLegal(int64_t offs, int64_t length) {
  if (length <= 0 || length > kMaxEncodableMatchLen) return false;
  if (offs <= static_cast<int64_t>(kMaxU16)) return true;
  return length > static_cast<int64_t>(kMaxShortMatchLen) &&
         offs <= static_cast<int64_t>(kMaxU24);
}

constexpr int64_t kNiceLength = 32;
constexpr int64_t kMaxLazyLength = 12;

// Indices into Encoder::stream, in the order the container writes them.
enum StreamIndex {
  kTokens = 0,
  kOffset16,
  kOffset24,
  kVarLitLen,
  kVarMatchLen,
  kLiterals,
};

// Upper bounds on the six streams for an input of `n` bytes.
//
// Emit() is called at most `m <= n/4` times, because it advances `litpos` past
// a match of at least 4 bytes and the next call's literal run starts there.
// The literal runs [litpos, tp) and the matches [tp, tp + len) are therefore
// two families of disjoint subranges of the input, which bounds not just the
// number of calls but also how often a long value can occur:
//  - literals: the disjoint literal runs, hence at most n bytes.
//  - tokens: at most two per call (the long-offset form emits a separate
//    literal token), plus one per literal run longer than kMaxEncodableLitLen
//    (~16 MB), of which there are at most n / kMaxEncodableLitLen.
//  - offset16: one 2-byte offset per call.
//  - offset24: one 3-byte offset per call, but IsLegal() only admits an
//    offset above kMaxU16 for matches longer than kMaxShortMatchLen == 15, so
//    those calls consume at least 16 input bytes; n/8 is twice that bound.
//  - var_lit_len / var_match_len: one varint per call, which costs 1 byte
//    below 254, 3 below 254^2 and 4 beyond. The 3-byte form needs a run or
//    match of >= 261 resp. 269 bytes and the 4-byte form >= 64523 resp.
//    64531, so the surplus over one byte per call is under n/128; n/64 is
//    twice that.
// The +64 covers the rounding of the divisions.
void StreamCapacities(size_t n, size_t cap[kStreamCount]) {
  cap[kTokens] = n / 2 + n / 1024 + 64;
  cap[kOffset16] = n / 2 + 64;
  cap[kOffset24] = 3 * (n / 8) + 64;
  cap[kVarLitLen] = n / 4 + n / 64 + 64;
  cap[kVarMatchLen] = n / 4 + n / 64 + 64;
  cap[kLiterals] = n + 64;
}

constexpr size_t kPageSize = 4096;

// Unused page between the streams. Should a bound above ever be wrong, the
// overrun stays inside the workspace (and ByteWriter::Overflowed reports it)
// rather than corrupting the next stream or the heap.
constexpr size_t kStreamGuard = kPageSize;

// Hash-chain table: one bucket of kHistSize positions per HashSeq() value.
constexpr size_t kChainSize = size_t{1} << kChainBits;
constexpr size_t kChainBytes = kChainSize * kHistSize * sizeof(uint32_t);

static_assert(kChainBytes <= kHugePage,
              "the chain table should fit in one huge page");

// The six write cursors advance independently, so when their bases are
// congruent modulo the page size they contend for the same L1 sets (a line's
// set is chosen by address % 4096 on the cores we target). Every capacity
// above is a simple fraction of `n` and the guard is exactly one page, which
// makes that congruence the usual case rather than a rare one. Sweeping the
// workspace base address over a page showed Compress ranging from 124 to 148
// ms - 9% - purely from where the arena landed. Advancing each stream to a
// page boundary and then skewing it by a distinct multiple of kSetSkew pins
// the relative placement, so the cost no longer depends on the allocator.
constexpr size_t kSetSkew = kPageSize / kStreamCount / 64 * 64;

// Byte offsets of each region from the start of the workspace. The base is
// huge-page aligned, so these are also the addresses modulo the page size.
struct WorkspaceLayout {
  size_t stream[kStreamCount];
  size_t cap[kStreamCount];
  size_t ans;
  size_t ans_size;
  size_t total;
};

// Workspace layout, in order: hash chains, the six guarded streams, then the
// rANS encoder's scratch (sized for the largest stream, which is the only one
// that can reach it).
WorkspaceLayout ComputeLayout(size_t n) {
  WorkspaceLayout layout;
  StreamCapacities(n, layout.cap);
  size_t off = kChainBytes;  // a whole number of pages
  size_t max_cap = 0;
  for (size_t i = 0; i < kStreamCount; ++i) {
    layout.stream[i] = RoundUpTo(off, kPageSize) + i * kSetSkew;
    off = layout.stream[i] + layout.cap[i] + kStreamGuard;
    max_cap = HWY_MAX(max_cap, layout.cap[i]);
  }
  layout.ans = RoundUpTo(off, kPageSize);
  layout.ans_size = Ans32EncodeScratchSize(max_cap);
  layout.total = layout.ans + layout.ans_size;
  return layout;
}

size_t WorkspaceSize(size_t n) { return ComputeLayout(n).total; }

// Bytes of private scratch one worker needs. Every worker compresses one
// kChunkSize chunk at a time, so this no longer depends on the input length.
// Rounded up to a huge page so that each worker's hash-chain table - the one
// structure here with random access - gets its own, and so that neighbouring
// workers cannot share a line.
size_t WorkerRegionSize() {
  return RoundUpTo(WorkspaceSize(kChunkSize), kHugePage);
}

// Staging arena: each chunk's compressed form is written here before the
// serial pass concatenates them into `dst`. A chunk cannot be written straight
// to its final place because that depends on how well its predecessors
// compressed, and the chunks finish out of order. Slots are fixed-size so a
// worker can address its own without synchronizing.
constexpr size_t kChunkSlotSize = kChunkSize + kMaxControlBytes;

// Per-chunk result of the parallel phase. `ctrl` is this chunk's command plus
// its varints, in ControlWriter order; the serial pass appends them in chunk
// order to form the block's control section.
struct ChunkResult {
  size_t payload_size;
  size_t ctrl_size;
  bool ok;
  uint8_t ctrl[kMaxControlBytes];
};

size_t TotalWorkspaceSize(size_t n, size_t num_workers) {
  const size_t num_chunks = NumChunks(n);
  return num_workers * WorkerRegionSize() + num_chunks * kChunkSlotSize +
         RoundUpTo(num_chunks * sizeof(ChunkResult), kPageSize);
}


struct Encoder {
  const uint8_t* src = nullptr;
  size_t src_len = 0;
  ByteWriter stream[kStreamCount];
  uint32_t last_encoded_offset = 0;
  // kChainSize buckets of kHistSize entries each, newest first. Each uint32_t
  // entry packs an 8-bit hash tag in bits [31:24] and `(pos + 1) & 0xFFFFFF`
  // in bits [23:0] (0 means empty slot).
  uint32_t* chains = nullptr;
  Span<uint8_t> ans_scratch;

  // `ws` must be page aligned and have at least WorkspaceSize(size) bytes.
  Encoder(const uint8_t* data, size_t size, uint8_t* ws)
      : src(data), src_len(size) {
    HWY_DASSERT(reinterpret_cast<size_t>(ws) % kPageSize == 0);
    chains = HWY_RCAST_ALIGNED(uint32_t*, ws);
    ZeroBytes(ws, kChainBytes);

    const WorkspaceLayout layout = ComputeLayout(size);
    for (size_t i = 0; i < kStreamCount; ++i) {
      stream[i] = ByteWriter(ws + layout.stream[i], layout.cap[i]);
    }
    ans_scratch = Span<uint8_t>(ws + layout.ans, layout.ans_size);
  }

  bool Overflowed() const {
    for (const ByteWriter& w : stream) {
      if (w.Overflowed()) return true;
    }
    return false;
  }

  void Insert(int64_t pos) {
    uint32_t tag;
    const uint32_t idx =
        HashAndTagU64(ScalarLoadULittleEndian<uint64_t>(src + pos), &tag);
    uint32_t* HWY_RESTRICT h = chains + idx * kHistSize;
    const uint32_t low24 = static_cast<uint32_t>(pos + 1) & 0x00FFFFFFu;
    h[3] = h[2];
    h[2] = h[1];
    h[1] = h[0];
    h[0] = tag | (low24 ? low24 : 1u);
  }

  // `cur7` holds at least 7 valid little-endian bytes starting at `pos`
  // (allowing `cur8` for `pos` and `cur8 >> 8` for `pos + 1` without a second
  // memory load). When `kInsertPos` is true, inserts `pos` into `h[0..3]`
  // immediately after reading the bucket while its cache line is hot in L1.
  template <bool kInsertPos>
  void BestChainMatch(int64_t litmin, int64_t pos, uint64_t cur7, uint32_t idx,
                      uint32_t tag, int64_t best_so_far,
                      int64_t* HWY_RESTRICT out_match_pos,
                      int64_t* HWY_RESTRICT out_chain_pos,
                      int64_t* HWY_RESTRICT out_len) {
    *out_match_pos = 0;
    *out_chain_pos = 0;
    *out_len = 0;
    uint32_t* HWY_RESTRICT h = chains + idx * kHistSize;
    const uint32_t entries[kHistSize] = {h[0], h[1], h[2], h[3]};
    const uint32_t pos1 = static_cast<uint32_t>(pos + 1);
    if constexpr (kInsertPos) {
      const uint32_t low24 = pos1 & 0x00FFFFFFu;
      h[3] = entries[2];
      h[2] = entries[1];
      h[1] = entries[0];
      h[0] = tag | (low24 ? low24 : 1u);
    }
    const int64_t n = static_cast<int64_t>(src_len);

    for (size_t i = 0; i < static_cast<size_t>(kHistSize); ++i) {
      const uint32_t entry = entries[i];
      if (entry == 0) break;
      const int64_t offs = static_cast<int64_t>((pos1 - entry) & 0x00FFFFFFu);
      const int64_t chain_pos = pos - offs;
      if (offs <= 0 || chain_pos < 0) break;
      // Reject 5-byte hash collisions in L1/L2 cache using the 8-bit tag before
      // touching the sliding window at `src + chain_pos`.
      if ((entry & 0xFF000000u) != tag) continue;

      const int64_t max_back = HWY_MIN(chain_pos, pos - litmin);
      const int64_t min_req = HWY_MAX(
          HWY_MAX(*out_len, best_so_far),
          offs > static_cast<int64_t>(kMaxU16)
              ? static_cast<int64_t>(kMaxShortMatchLen)
              : int64_t{3});

      const uint64_t cand8 = ScalarLoadULittleEndian<uint64_t>(src + chain_pos);
      const uint64_t d = (cand8 ^ cur7) & 0x00FFFFFFFFFFFFFFull;
      if ((d << 24) != 0) continue;

      int64_t fwd_len;
      if (d != 0) {
        fwd_len = static_cast<int64_t>(Num0BitsBelowLS1Bit_Nonzero64(d) >> 3);
        if (fwd_len + max_back <= min_req) continue;
      } else {
        const int64_t need_fwd = min_req + 1 - max_back;
        if (need_fwd > 7 && pos + need_fwd <= n &&
            ScalarLoadULittleEndian<uint32_t>(src + chain_pos + need_fwd - 4) !=
                ScalarLoadULittleEndian<uint32_t>(src + pos + need_fwd - 4)) {
          continue;
        }
        fwd_len = Lcp(src, src_len, chain_pos, pos, 7);
      }

      int64_t cand_chain_pos = chain_pos;
      int64_t cand_match_pos = pos;
      int64_t cand_len = fwd_len;
      while (cand_chain_pos > 0 && cand_match_pos > litmin &&
             src[cand_chain_pos - 1] == src[cand_match_pos - 1]) {
        --cand_chain_pos;
        --cand_match_pos;
        ++cand_len;
      }
      if (cand_len > kMaxEncodableMatchLen) cand_len = kMaxEncodableMatchLen;
      if (cand_len > *out_len && IsLegal(offs, cand_len)) {
        *out_match_pos = cand_match_pos;
        *out_chain_pos = cand_chain_pos;
        *out_len = cand_len;
        if (cand_len >= kNiceLength) break;
      }
    }
  }

  void BestMatchAt(int64_t litpos, int64_t pos, int64_t last,
                   int64_t* HWY_RESTRICT out_match_pos,
                   int64_t* HWY_RESTRICT out_chain_pos,
                   int64_t* HWY_RESTRICT out_len) {
    *out_match_pos = pos;
    *out_chain_pos = 0;
    *out_len = 0;
    const uint64_t cur8 = ScalarLoadULittleEndian<uint64_t>(src + pos);
    uint32_t tag0;
    const uint32_t idx0 = HashAndTagU64(cur8, &tag0);

    const int64_t repeat_pos = pos - static_cast<int64_t>(last_encoded_offset);
    if (repeat_pos >= 0 && repeat_pos < pos) {
      const uint64_t rep8 = ScalarLoadULittleEndian<uint64_t>(src + repeat_pos);
      const uint64_t d = rep8 ^ cur8;
      if (static_cast<uint32_t>(d) == 0) {
        int64_t repeat_len = d == 0
                                 ? Lcp(src, src_len, repeat_pos, pos, 8)
                                 : static_cast<int64_t>(
                                       Num0BitsBelowLS1Bit_Nonzero64(d) >> 3);
        if (repeat_len > kMaxEncodableMatchLen) {
          repeat_len = kMaxEncodableMatchLen;
        }
        if (IsLegal(static_cast<int64_t>(last_encoded_offset), repeat_len)) {
          *out_match_pos = pos;
          *out_chain_pos = repeat_pos;
          *out_len = repeat_len;
        }
      } else if (pos < last && static_cast<uint32_t>(d >> 8) == 0) {
        int64_t repeat_len =
            (d >> 8) == 0
                ? Lcp(src, src_len, repeat_pos + 1, pos + 1, 7)
                : static_cast<int64_t>(Num0BitsBelowLS1Bit_Nonzero64(d >> 8) >>
                                       3);
        if (repeat_len > kMaxEncodableMatchLen) {
          repeat_len = kMaxEncodableMatchLen;
        }
        if (IsLegal(static_cast<int64_t>(last_encoded_offset), repeat_len)) {
          *out_match_pos = pos + 1;
          *out_chain_pos = repeat_pos + 1;
          *out_len = repeat_len;
        }
      }
    }
    if (*out_len < kNiceLength) {
      int64_t chain_match_pos, chain_chain_pos, chain_len;
      // Always insert `pos` into `chains[idx0]` while probing `pos` so literal
      // steps never need a second hash + cache-line touch.
      BestChainMatch<true>(litpos, pos, cur8, idx0, tag0, *out_len + 1,
                           &chain_match_pos, &chain_chain_pos, &chain_len);
      if (chain_len - *out_len > 1) {
        *out_match_pos = chain_match_pos;
        *out_chain_pos = chain_chain_pos;
        *out_len = chain_len;
      }
      if (*out_len < kMaxLazyLength && pos < last) {
        uint32_t tag1;
        const uint32_t idx1 = HashAndTagU64(cur8 >> 8, &tag1);
        BestChainMatch<false>(litpos, pos + 1, cur8 >> 8, idx1, tag1, *out_len,
                              &chain_match_pos, &chain_chain_pos, &chain_len);
        if (chain_len > *out_len) {
          *out_match_pos = chain_match_pos;
          *out_chain_pos = chain_chain_pos;
          *out_len = chain_len;
        }
      }
    } else {
      // `pos` found a >= kNiceLength repeat match; still record `pos` in its
      // bucket using the already-computed `idx0`/`tag0`.
      uint32_t* HWY_RESTRICT h = chains + idx0 * kHistSize;
      const uint32_t low24 = static_cast<uint32_t>(pos + 1) & 0x00FFFFFFu;
      h[3] = h[2];
      h[2] = h[1];
      h[1] = h[0];
      h[0] = tag0 | (low24 ? low24 : 1u);
    }

    // Keep the decoder's final 32-byte match write inside the output buffer.
    if (*out_match_pos + *out_len >
        static_cast<int64_t>(src_len) - static_cast<int64_t>(kMinOffset)) {
      if (*out_match_pos - *out_chain_pos >= static_cast<int64_t>(kMinOffset)) {
        constexpr int64_t lomask = static_cast<int64_t>(kMinOffset) - 1;
        if (*out_match_pos + ((*out_len + lomask) & ~lomask) >
            static_cast<int64_t>(src_len)) {
          *out_len &= ~lomask;
        }
      } else {
        const int64_t movsize = *out_match_pos - *out_chain_pos;
        const int64_t tailpos =
            movsize ? *out_len - (*out_len % movsize) : *out_len;
        const int64_t end = static_cast<int64_t>(src_len);
        if (*out_match_pos + tailpos + static_cast<int64_t>(kMinOffset) > end) {
          const int64_t safedist =
              (end - static_cast<int64_t>(kMinOffset)) - *out_match_pos;
          *out_len = movsize ? (safedist / movsize) * movsize : 0;
        }
      }
    }
  }

  // Emits a token that only carries literals: bit 0x80 means "reuse the
  // previous offset", and a zero match length means the decoder copies none.
  void EmitLiteralsOnly(size_t lit_len) {
    stream[kTokens].Push(static_cast<uint8_t>(0x80 | kMaxShortLitLen));
    AppendVarUint(stream[kVarLitLen],
                  static_cast<uint32_t>(lit_len - kMaxShortLitLen));
  }

  void Emit(const uint8_t* lit, size_t lit_len, uint32_t offs,
            uint32_t match_len) {
    stream[kLiterals].Append(lit, lit_len);
    // A single token transmits at most kMaxEncodableLitLen literals, so long
    // runs of literals (incompressible data) are split into several
    // literal-only tokens. Matches longer than kMaxEncodableMatchLen were
    // already clamped in MatchExtend / BestMatchAt.
    size_t lit_done = 0;
    while (lit_len - lit_done > kMaxEncodableLitLen) {
      EmitLiteralsOnly(kMaxEncodableLitLen);
      lit_done += kMaxEncodableLitLen;
    }
    const size_t lit_rest = lit_len - lit_done;
    lit = lit + lit_done;
    lit_len = lit_rest;
    const uint32_t lit32 = static_cast<uint32_t>(lit_len);
    const uint32_t kShortLit = static_cast<uint32_t>(kMaxShortLitLen);
    const uint32_t kShortMatch = static_cast<uint32_t>(kMaxShortMatchLen);

    if (offs == last_encoded_offset || offs <= kMaxU16) {
      uint32_t token = 0x80;
      if (offs != last_encoded_offset) {
        token = 0x00;
        AppendU16(stream[kOffset16], offs);
      }
      if (lit32 < kShortLit) {
        token |= lit32;
      } else {
        token |= kShortLit;
        AppendVarUint(stream[kVarLitLen], lit32 - kShortLit);
      }
      if (match_len < kShortMatch) {
        token |= match_len << kLiteralLenBits;
      } else {
        token |= kShortMatch << kLiteralLenBits;
        AppendVarUint(stream[kVarMatchLen], match_len - kShortMatch);
      }
      stream[kTokens].Push(static_cast<uint8_t>(token));
    } else {
      if (lit_len > 0) {
        uint32_t token = 0x80;
        if (lit32 < kShortLit) {
          token |= lit32;
        } else {
          token |= kShortLit;
          AppendVarUint(stream[kVarLitLen], lit32 - kShortLit);
        }
        stream[kTokens].Push(static_cast<uint8_t>(token));
      }
      AppendU24(stream[kOffset24], offs);
      const uint32_t kLongBase =
          static_cast<uint32_t>(kLastLongOffset + kMMLongOffsets);
      uint32_t token;
      if (match_len < kLongBase) {
        token = match_len - static_cast<uint32_t>(kMMLongOffsets);
      } else {
        token = 0x1F;
        AppendVarUint(stream[kVarMatchLen], match_len - kLongBase);
      }
      stream[kTokens].Push(static_cast<uint8_t>(token));
    }
    last_encoded_offset = offs;
  }

  void CompressSrc() {
    constexpr int64_t kSkipStep = 2;
    const int64_t last =
        static_cast<int64_t>(src_len) - static_cast<int64_t>(kMinOffset);
    last_encoded_offset = 0;
    int64_t pos = 5;
    int64_t litpos = 0;
    Insert(0);

    while (pos <= last) {
      int64_t tp, mp, len;
      BestMatchAt(litpos, pos, last, &tp, &mp, &len);
      if (len >= 4) {
        Emit(src + litpos, static_cast<size_t>(tp - litpos),
             static_cast<uint32_t>(tp - mp), static_cast<uint32_t>(len));
        const int64_t match_end = tp + len;
        if (len <= 64) {
          for (int64_t i = tp; i < match_end && i < last; i += kSkipStep) {
            if (i != pos) Insert(i);
          }
        } else {
          for (int64_t i = tp; i < tp + 16 && i < last; i += kSkipStep) {
            if (i != pos) Insert(i);
          }
          for (int64_t i = match_end - 8; i < match_end && i < last;
               i += kSkipStep) {
            if (i != pos) Insert(i);
          }
        }
        pos = match_end;
        litpos = pos;
      } else {
        // `pos` was already inserted into `chains` by BestMatchAt.
        pos += kSkipStep;
      }
    }
    const size_t tail = src_len - static_cast<size_t>(litpos);
    stream[kLiterals].Append(src + litpos, tail);
  }
};

}  // namespace

// ------------------------------ container

// The decoder rejects blocks that declare more than kMaxUncompressedSize, so
// producing one would break the round-trip guarantee. Staying below it also
// keeps every stream below the 4 GiB limit of the ANS coder (asserted below)
// and every position below the 2 GiB that the match finder's int32_t needs.
static_assert(kMaxUncompressedSize < (uint64_t{1} << 31),
              "the match finder stores positions as int32_t");
HWY_CONTRIB_DLLEXPORT bool DecompressedSize(Span<const uint8_t> src,
                                            size_t* HWY_RESTRICT out_size) {
  return ParseDecompressedSize(src, out_size);
}

HWY_CONTRIB_DLLEXPORT size_t IguanaWorkspace::SizeFor(size_t max_size,
                                                      size_t num_workers) {
  return TotalWorkspaceSize(max_size, num_workers);
}

HWY_CONTRIB_DLLEXPORT bool IguanaWorkspace::Reserve(size_t max_size,
                                                    size_t num_workers) {
  const size_t want = TotalWorkspaceSize(max_size, num_workers);
  if (capacity_ >= want) return true;
  // Each worker's hash-chain table is indexed by a hash, so its accesses are
  // essentially random over 2 MiB. Starting the region on a 2 MiB boundary
  // lets a single transparent huge page cover all of it, which costs one TLB
  // entry instead of up to 513. WorkerRegionSize() is a whole number of huge
  // pages, so aligning the base aligns every worker's table.
  AlignedFreeUniquePtr<uint8_t[]> mem =
      AllocateAligned<uint8_t>(want + kHugePage - 1);
  if (mem == nullptr) return false;  // keep the previous, still usable buffer
  base_ = reinterpret_cast<uint8_t*>(
      RoundUpTo(reinterpret_cast<size_t>(mem.get()), kHugePage));
  mem_ = std::move(mem);
  capacity_ = want;
  return true;
}

namespace {

// Compresses one chunk into `out` (which has kChunkSlotSize bytes) using
// `work` (a WorkerRegionSize() region), and records its control bytes. This is
// the whole of the former single-threaded Compress body apart from the block
// header: one chunk is exactly what used to be one block's worth of work.
void CompressChunk(Span<const uint8_t> src, uint8_t* HWY_RESTRICT out,
                   uint8_t* HWY_RESTRICT work,
                   ChunkResult* HWY_RESTRICT res) {
  const uint8_t* const data = src.data();
  const size_t size = src.size();
  res->ok = false;
  res->payload_size = 0;

  ControlWriter cw;
  size_t out_pos = 0;
  const auto append = [&](const uint8_t* p, size_t n) -> bool {
    if (n > kChunkSlotSize - out_pos) return false;
    CopyBytes(p, out + out_pos, n);
    out_pos += n;
    return true;
  };

  if (size < static_cast<size_t>(kMinLength + kHashBytes)) {
    cw.Command(kCmdCopyRaw);
    cw.VarUint(size);
    if (!append(data, size)) return;
  } else {
    Encoder enc(data, size, work);
    enc.CompressSrc();
    // The capacities StreamCapacities() derived are upper bounds, so this can
    // only fire if one of them is wrong. Fail rather than emit a truncated
    // stream; the guard region kept the overrun inside the workspace.
    if (HWY_UNLIKELY(enc.Overflowed())) return;

    // Entropy-code each stream and append whichever of the two forms is
    // smaller, in the order the format expects. The control bytes cannot be
    // written yet - they start with the stream-type nibbles, which are only
    // known once every stream has been tried - so the chosen lengths are
    // remembered here and emitted below.
    size_t enc_size[kStreamCount];
    uint64_t hdr = 0;
    int64_t total = 0;
    bool fits = true;

    for (size_t i = 0; i < kStreamCount && fits; ++i) {
      const ByteWriter& u = enc.stream[i];
      // The ANS coder would wrap around (and divide by zero) at 4 GiB.
      HWY_DASSERT(static_cast<uint64_t>(u.size()) < (uint64_t{1} << 32));
      // An ANS32 block carries 128 bytes of final lane states, 96 bytes of
      // 3-bit control codes and 1 level byte (225 bytes minimum), so streams
      // of 225 bytes or fewer can never shrink under ANS32.
      constexpr size_t kMinAns32Bytes = 128 + 96 + 1;
      const Span<const uint8_t> cs =
          u.size() > kMinAns32Bytes
              ? Ans32Encode(u.data(), u.size(), enc.ans_scratch)
              : Span<const uint8_t>();
      const bool worthwhile = !cs.empty() && cs.size() < u.size();
      if (worthwhile) hdr |= uint64_t{1} << (i * 4);  // EntropyANS32
      const uint8_t* bytes = worthwhile ? cs.data() : u.data();
      enc_size[i] = worthwhile ? cs.size() : u.size();
      total += static_cast<int64_t>(enc_size[i]);
      fits = append(bytes, enc_size[i]);
    }

    // Falling back also covers `!fits`: the slot was too small for the
    // streams, but it always has room for the raw form.
    if (!fits || total + static_cast<int64_t>(kStreamCount) + 1 >=
                     static_cast<int64_t>(size)) {
      out_pos = 0;  // discard the streams; no control bytes written yet
      cw.Command(kCmdCopyRaw);
      cw.VarUint(size);
      if (!append(data, size)) return;
    } else {
      cw.Command(kCmdDecodeIguana);
      cw.VarUint(hdr);
      for (size_t i = 0; i < kStreamCount; ++i) {
        cw.VarUint(enc.stream[i].size());
      }
      for (size_t i = 0; i < kStreamCount; ++i) {
        if ((hdr >> (i * 4)) & 0xF) cw.VarUint(enc_size[i]);
      }
    }
  }

  res->payload_size = out_pos;
  res->ctrl_size = cw.ctrl.size();
  CopyBytes(cw.buf, res->ctrl, res->ctrl_size);
  res->ok = true;
}

}  // namespace

HWY_CONTRIB_DLLEXPORT size_t Compress(Span<const uint8_t> src,
                                      Span<uint8_t> dst, IguanaWorkspace& ws,
                                      ThreadPool& pool) {
  const uint8_t* const data = src.data();
  const size_t size = src.size();
  if (HWY_UNLIKELY(size > kMaxUncompressedSize)) return 0;

  ControlWriter header;
  header.VarUint(size);  // total uncompressed length

  if (size == 0) {  // no command
    if (header.ctrl.size() > dst.size()) return 0;
    size_t out_pos = 0;
    for (size_t i = header.ctrl.size(); i-- > 0;) dst[out_pos++] = header.buf[i];
    return out_pos;
  }

  const size_t num_workers = HWY_MAX(pool.NumWorkers(), size_t{1});
  const size_t num_chunks = NumChunks(size);
  if (HWY_UNLIKELY(!ws.Reserve(size, num_workers))) return 0;

  uint8_t* const mem = ws.Memory();
  uint8_t* const arena = mem + num_workers * WorkerRegionSize();
  ChunkResult* const results =
      HWY_RCAST_ALIGNED(ChunkResult*, arena + num_chunks * kChunkSlotSize);

  // Chunks are independent by construction: each Encoder sees only its own
  // slice of `src`, so no match can reach across a boundary and no worker
  // reads another's state.
  pool.Run(0, num_chunks, [&](uint64_t task, size_t worker) {
    const size_t k = static_cast<size_t>(task);
    const size_t begin = k * kChunkSize;
    CompressChunk(Span<const uint8_t>(data + begin, ChunkLen(size, k)),
                  arena + k * kChunkSlotSize,
                  mem + worker * WorkerRegionSize(), &results[k]);
  });

  // Serial concatenation. The payloads go to the front of `dst` in chunk
  // order, the control sections behind them in reverse; the decoder reads the
  // total length from the last byte and then walks the commands forwards.
  size_t payload_total = 0;
  size_t ctrl_total = header.ctrl.size();
  for (size_t k = 0; k < num_chunks; ++k) {
    if (HWY_UNLIKELY(!results[k].ok)) return 0;
    payload_total += results[k].payload_size;
    ctrl_total += results[k].ctrl_size;
  }
  if (payload_total + ctrl_total > dst.size()) return 0;

  size_t out_pos = 0;
  for (size_t k = 0; k < num_chunks; ++k) {
    CopyBytes(arena + k * kChunkSlotSize, dst.data() + out_pos,
              results[k].payload_size);
    out_pos += results[k].payload_size;
  }

  // The control section is the reverse of [header][chunk 0]...[chunk n-1], so
  // byte `i` of that sequence lands at the far end. Only the last command may
  // keep kLastCommandMarker; each chunk set it on its own, so all but the
  // final one are cleared here. A chunk's command byte is the first of its
  // control bytes, because ControlWriter emits the command before its varints.
  uint8_t* const ctrl_out = dst.data() + out_pos;
  size_t i = 0;
  const auto emit_ctrl = [&](uint8_t v) { ctrl_out[ctrl_total - 1 - i++] = v; };
  for (size_t j = 0; j < header.ctrl.size(); ++j) emit_ctrl(header.buf[j]);
  for (size_t k = 0; k < num_chunks; ++k) {
    for (size_t j = 0; j < results[k].ctrl_size; ++j) {
      const uint8_t v = results[k].ctrl[j];
      emit_ctrl(j == 0 && k + 1 != num_chunks ? (v & kCommandMask) : v);
    }
  }
  return out_pos + ctrl_total;
}

// Decompresses a block produced by Compress. The container is shared with the
// SIMD path (detail.h); only the entropy stage differs.
HWY_CONTRIB_DLLEXPORT size_t DecompressScalar(Span<const uint8_t> src,
                                              Span<uint8_t> dst,
                                              IguanaWorkspace& ws,
                                              ThreadPool& pool) {
  HWY_DASSERT(WorkerRegionSize() == kHugePage);
  const auto decode = [](Span<const uint8_t> payload, Span<uint8_t> out) {
    return Ans32DecodeScalar(payload, out);
  };
  return DecompressBlockParallel(src, dst, decode, ws, pool);
}

HWY_EXPORT(DecompressStatic);

// Dispatches to the best target available at run time.
HWY_CONTRIB_DLLEXPORT size_t Decompress(Span<const uint8_t> src,
                                        Span<uint8_t> dst, IguanaWorkspace& ws,
                                        ThreadPool& pool) {
  HWY_DASSERT(WorkerRegionSize() == kHugePage);
  return HWY_DYNAMIC_DISPATCH(DecompressStatic)(src, dst, ws, pool);
}

}  // namespace iguana
}  // namespace hwy
#endif  // HWY_ONCE
