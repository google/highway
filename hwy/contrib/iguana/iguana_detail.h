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

// Internals of the Iguana codec, shared by the scalar path (iguana.cc) and the
// SIMD path (iguana-inl.h). Not part of the public API: include iguana.h for
// Compress / DecompressScalar.
//
// The container parse, the LZ77 stage and the control-varint reader are
// inherently serial and target-independent, so they live here and are used by
// both paths; only the entropy stage differs (Ans32DecodeScalar vs
// Ans32Decode). DecompressBlock is templated over that stage.

#ifndef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_DETAIL_H_
#define HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_DETAIL_H_

#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/contrib/iguana/iguana.h"  // kDecompressFailed
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace hwy {
namespace iguana {

// iguana.cc and iguana_test.cc must agree on this, else we get a null pointer
// for targets that the latter selected but the former disabled.
#define HWY_IGUANA_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3)

// ------------------------------ Format constants
//
// Sizes are unsigned. The few places that compare them against a signed cursor
// cast explicitly, because e.g. src_len - kMinOffset must stay signed.

HWY_INLINE_VAR constexpr size_t kIguanaChunkSize = 32;
HWY_INLINE_VAR constexpr size_t kMinOffset = 32;
HWY_INLINE_VAR constexpr size_t kMinLength = 32;
HWY_INLINE_VAR constexpr size_t kLiteralLenBits = 3;
HWY_INLINE_VAR constexpr size_t kMMLongOffsets = 16;
HWY_INLINE_VAR constexpr size_t kMaxShortLitLen = 7;
HWY_INLINE_VAR constexpr size_t kMaxShortMatchLen = 15;
HWY_INLINE_VAR constexpr size_t kLastLongOffset = 31;
HWY_INLINE_VAR constexpr size_t kHashBytes = 5;
HWY_INLINE_VAR constexpr size_t kHistSize = 4;
HWY_INLINE_VAR constexpr size_t kStreamCount = 6;

enum Command {
  kCmdCopyRaw = 0,
  kCmdDecodeIguana = 1,
  kCmdDecodeANS32 = 2,
  kCmdDecodeANS1 = 3,
  kCmdDecodeANSNibble = 4,
};
HWY_INLINE_VAR constexpr uint8_t kLastCommandMarker = 0x80;
HWY_INLINE_VAR constexpr uint8_t kCommandMask = 0x7F;

// Largest offset representable in a 24-bit stream.
HWY_INLINE_VAR constexpr uint64_t kMaxU24 = (uint64_t{1} << 24) - 1;

// Ceiling for every allocation driven by the (untrusted) header. A small,
// malformed block can claim a huge uncompressed length ("zip bomb"), so the
// declared total and each per-stream/per-command length are checked against
// this before resizing, and the output is capped by it as well.
HWY_INLINE_VAR constexpr size_t kMaxUncompressedSize = size_t{1} << 30;  // 1GiB

// ------------------------------ Chunking
//
// Compress splits its input into independently coded chunks of exactly this
// many uncompressed bytes (the last one holds the remainder), and emits one
// command per chunk. Nothing in the bitstream marks them as chunks: the format
// has always allowed several commands per block and the existing serial
// decoder reads them unchanged. What the fixed size buys is that a decoder can
// derive each command's *output* offset (`k * kChunkSize`) from the block
// header alone, without first decoding its predecessors - which is what lets
// the chunks be decoded in parallel.
//
// 256 KiB follows Oodle's quantum, LZFSE's block payload and LZ4 -B5. It is
// large enough that the six per-chunk ANS tables (~1.1 KiB) stay near 0.4% of
// the compressed size and the whole 64 KiB offset16 window remains usable,
// and small enough that one worker's hash chains plus streams stay in L2/L3
// and that a 1-8 MB input still splits across enough chunks to fill a machine.
HWY_INLINE_VAR constexpr size_t kChunkSize = size_t{256} << 10;

// Number of chunks `n` uncompressed bytes are split into. Zero bytes produce
// no command at all, hence no chunk.
HWY_INLINE constexpr size_t NumChunks(size_t n) {
  return (n + kChunkSize - 1) / kChunkSize;
}

// Uncompressed length of chunk `k`, which every chunk but the last one must
// decode to exactly. Decoders use this as an assertion on untrusted input:
// a chunk that produces a different count is rejected.
HWY_INLINE constexpr size_t ChunkLen(size_t n, size_t k) {
  return HWY_MIN(kChunkSize, n - k * kChunkSize);
}

// ------------------------------ Encoder tuning
//
// Not part of the format: the match finder's hash table only decides which
// matches Compress finds, never how a block is decoded. Changing it re-encodes
// future inputs but leaves existing byte streams decodable.
//
// The table holds `(1 << kChainBits) * kHistSize` positions and is zeroed once
// per chunk. That zeroing used to be amortized over a whole multi-megabyte
// input; now it is charged to every kChunkSize bytes, so the table's size is
// a direct per-byte cost: at kChainBits = 17 it was 2 MiB per 256 KiB chunk,
// i.e. 8 bytes cleared per input byte. 2^15 buckets keep 2^17 positions in
// 512 KiB, which is ample - a chunk inserts far fewer than one position per
// byte, since a match of kMinLength or more covers many bytes with at most a
// few inserts.
HWY_INLINE_VAR constexpr size_t kChainBits = 15;
static_assert((size_t{1} << kChainBits) * kHistSize <= kChunkSize,
              "the hash table holds more positions than a chunk can insert");

// ------------------------------ Reader over one LZ77 stream

struct StreamReader {
  const uint8_t* data = nullptr;
  size_t size = 0;
  size_t cursor = 0;

  bool IsEmpty() const { return cursor >= size; }
  size_t RemainingBytes() const { return size - cursor; }
  // Subtraction, not addition: cursor + n could wrap around.
  bool HaveBytes(size_t n) const { return n <= size - cursor; }

  uint8_t U8(bool* ok) {
    if (!HaveBytes(1)) {
      *ok = false;
      return 0;
    }
    return data[cursor++];
  }
  uint32_t U16(bool* ok) {
    if (!HaveBytes(2)) {
      *ok = false;
      return 0;
    }
    const uint32_t r = static_cast<uint32_t>(data[cursor]) |
                       (static_cast<uint32_t>(data[cursor + 1]) << 8);
    cursor += 2;
    return r;
  }
  uint32_t U24(bool* ok) {
    if (!HaveBytes(3)) {
      *ok = false;
      return 0;
    }
    const uint32_t r = static_cast<uint32_t>(data[cursor]) |
                       (static_cast<uint32_t>(data[cursor + 1]) << 8) |
                       (static_cast<uint32_t>(data[cursor + 2]) << 16);
    cursor += 3;
    return r;
  }
  // Iguana stream varint (base-254, forward).
  int64_t VarUint(bool* ok) {
    const uint32_t a = U8(ok);
    if (!*ok) return 0;
    if (a < 0xFE) return static_cast<int64_t>(a);
    if (a == 0xFE) {
      const uint32_t b = U16(ok);
      if (!*ok) return 0;
      return static_cast<int64_t>((b >> 8) * 254 + (b & 0xFF));
    }
    const uint32_t b = U24(ok);
    if (!*ok) return 0;
    const int64_t x0 = b & 0xFF;
    const int64_t x1 = (b >> 8) & 0xFF;
    const int64_t x2 = b >> 16;
    return ((x2 * 254) + x1) * 254 + x0;
  }
  const uint8_t* Sequence(size_t n, bool* ok) {
    if (!HaveBytes(n)) {
      *ok = false;
      return nullptr;
    }
    const uint8_t* r = data + cursor;
    cursor += n;
    return r;
  }
};

// ------------------------------ Control varint (base-128, read backwards)

// Reads a base-128 varint backwards from src[*cursor], moving *cursor before
// the consumed bytes. The first byte read is the most significant one. Sets
// *ok=false on underflow, or if the value would not fit in 64 bits.
// Nothing in this header is exported; only iguana.h is public API. Kept inline
// because it is tiny and stays target-independent.
HWY_INLINE uint64_t ReadControlVarUint(const uint8_t* src, int64_t* cursor,
                                       bool* ok) {
  uint64_t r = 0;
  int groups = 0;  // number of bytes consumed so far
  while (*cursor >= 0) {
    const uint8_t v = src[*cursor];
    --*cursor;
    if (groups == 9) {
      // 10th byte is the most significant: it may contribute a single bit,
      // otherwise the shift below would silently drop bits.
      if ((v & 0x7F) > 1) {
        *ok = false;
        return 0;
      }
    } else if (groups >= 10) {
      *ok = false;
      return 0;
    }
    r = (r << 7) | (v & 0x7F);
    ++groups;
    if (v & 0x80) return r;
  }
  *ok = false;
  return 0;
}

// Parses the uncompressed size from the block header into `*out_size`.
// Returns false if the header is malformed or exceeds kMaxUncompressedSize.
HWY_INLINE bool ParseDecompressedSize(Span<const uint8_t> src,
                                      size_t* HWY_RESTRICT out_size) {
  if (src.empty()) return false;
  bool ok = true;
  int64_t ctrl = static_cast<int64_t>(src.size()) - 1;
  const uint64_t uncompressed_len = ReadControlVarUint(src.data(), &ctrl, &ok);
  if (!ok || uncompressed_len > kMaxUncompressedSize) return false;
  if (uncompressed_len == 0 && ctrl >= 0) return false;
  *out_size = static_cast<size_t>(uncompressed_len);
  return true;
}

// ------------------------------ LZ77 stage (decoder)

// One of the six token/literal/offset streams handed to the LZ77 stage.
struct IguanaStream {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// Copies `match_len` bytes from `dst[match_pos..)` into `dst[*io_out_pos..)`.
// Overlapping runs (where the source reaches into the bytes being produced)
// are handled by copying in 16- or 8-byte chunks while the chunk still fits
// inside the match distance, and byte by byte otherwise. Never writes past
// `dst[max_out_size)`, so this is also correct for the very last match.
HWY_INLINE bool CopyMatch(uint8_t* const HWY_RESTRICT dst,
                          size_t* HWY_RESTRICT io_out_pos, size_t match_pos,
                          size_t match_len, size_t max_out_size) {
  const size_t out_pos = *io_out_pos;
  if (match_len > max_out_size - out_pos) return false;
  const size_t dist = out_pos - match_pos;
  uint8_t* const out = dst + out_pos;
  const uint8_t* const in = dst + match_pos;
  size_t i = 0;
  // A chunk may only be copied while it fits within the distance, otherwise
  // the source would reach into bytes this loop has not written yet.
  if (HWY_LIKELY(dist >= 16)) {
    for (; i + 16 <= match_len; i += 16) CopyBytes<16>(in + i, out + i);
  } else if (dist >= 8) {
    for (; i + 8 <= match_len; i += 8) CopyBytes<8>(in + i, out + i);
  }
  for (; i < match_len; ++i) out[i] = in[i];
  *io_out_pos = out_pos + match_len;
  return true;
}

// Validates `last_offs` (the negated match distance) against the bytes
// produced so far, then copies the match. A single unsigned compare rejects
// both "no match seen yet" (last_offs == 0) and an offset that would read
// before the start of the output.
HWY_INLINE bool CheckedCopyMatch(uint8_t* const HWY_RESTRICT dst,
                                 size_t* HWY_RESTRICT io_out_pos,
                                 int64_t last_offs, int64_t match_len,
                                 size_t max_out_size) {
  HWY_DASSERT(match_len >= 0);
  if (match_len == 0) return true;
  const size_t out_pos = *io_out_pos;
  const uint64_t dist = static_cast<uint64_t>(-last_offs);
  // dist == 0 wraps to the largest value, so this is `dist == 0 || dist >
  // out_pos` in one compare.
  if (HWY_UNLIKELY(dist - 1 >= out_pos)) return false;
  return CopyMatch(dst, io_out_pos, out_pos - static_cast<size_t>(dist),
                   static_cast<size_t>(match_len), max_out_size);
}

// The LZ77 stage: expands the six streams into `dst[*io_out_pos..)`. The token
// loop is inherently serial, so this stays scalar on the SIMD path too.
// Returns false on malformed input.
//
// Following the LZSSE decoders, this runs two loops. The *fast interior loop*
// executes only while every stream and the output are provably far enough from
// their ends that the per-token bounds checks are redundant, which lets it
// decode a token with fixed-width over-shooting copies and without the
// data-dependent branches that dominate this function's mispredictions. All of
// its guard conditions are monotone (cursors only ever advance), so once one
// fails it stays failed and control drops into the *careful loop*, which is the
// original fully-checked implementation and handles both the tail and every
// malformed or adversarial input.
HWY_INLINE bool DecompressIguanaLZ(uint8_t* const HWY_RESTRICT dst,
                                   size_t* HWY_RESTRICT io_out_pos,
                                   const IguanaStream streams_in[kStreamCount],
                                   size_t max_out_size) {
  StreamReader reader[kStreamCount];
  for (size_t i = 0; i < kStreamCount; ++i) {
    reader[i].data = streams_in[i].data;
    reader[i].size = streams_in[i].size;
  }
  StreamReader& token_stream = reader[0];
  StreamReader& off16_stream = reader[1];
  StreamReader& off24_stream = reader[2];
  StreamReader& var_lit_len_stream = reader[3];
  StreamReader& var_match_len_stream = reader[4];
  StreamReader& literal_stream = reader[5];

  size_t out_pos = *io_out_pos;
  bool ok = true;
  // Offset of the previous match, negated: the streams encode the distance
  // back from the current output position. Zero means "no match seen yet", and
  // a match that refers to it is malformed.
  int64_t last_offs = 0;

  // ------------------------------ Fast interior loop
  // The wild copies write up to 8 literal plus 16 match bytes past `out_pos`;
  // two chunks of slack cover both with room to spare.
  if (max_out_size >= 2 * kIguanaChunkSize) {
    const size_t wild_end = max_out_size - 2 * kIguanaChunkSize;
    const uint8_t* token_cur = token_stream.data;
    const uint8_t* const token_end = token_cur + token_stream.size;
    const uint8_t* off16_cur = off16_stream.data;
    const uint8_t* const off16_end = off16_cur + off16_stream.size;
    const uint8_t* lit_cur = literal_stream.data;
    const uint8_t* const lit_end = lit_cur + literal_stream.size;

    // The three streams read without bounds checks below need 8 literal bytes
    // (the over-shooting store) and 2 offset bytes (the unconditional load).
    while (token_cur != token_end && out_pos <= wild_end &&
           static_cast<size_t>(lit_end - lit_cur) >= 8 &&
           static_cast<size_t>(off16_end - off16_cur) >= 2) {
      const uint8_t token = *token_cur++;
      int64_t match_len;

      if (HWY_UNLIKELY(token < 32)) {
        // Long (24-bit) offset. Left fully checked: the 24-bit and varint
        // streams are untouched by this loop, so their readers stay in sync.
        if (token < kLastLongOffset) {
          match_len = static_cast<int64_t>(token) +
                      static_cast<int64_t>(kMMLongOffsets);
        } else {
          match_len = var_match_len_stream.VarUint(&ok) +
                      static_cast<int64_t>(kLastLongOffset + kMMLongOffsets);
          if (!ok) return false;
        }
        last_offs = -static_cast<int64_t>(off24_stream.U24(&ok));
        if (!ok) return false;
        if (!CheckedCopyMatch(dst, &out_pos, last_offs, match_len,
                              max_out_size)) {
          return false;
        }
        continue;
      }
      // 0x80 is a NOP; see the careful loop below for why it is rejected.
      if (HWY_UNLIKELY(token == 0x80)) return false;

      // Literals: one unconditional 8-byte store covers every short run
      // (0..6 bytes), including the empty one, with no length dispatch.
      size_t lit_len = token & kMaxShortLitLen;
      if (HWY_UNLIKELY(lit_len == kMaxShortLitLen)) {
        const int64_t extra = var_lit_len_stream.VarUint(&ok);
        if (!ok) return false;
        lit_len = static_cast<size_t>(extra) + kMaxShortLitLen;
        if (lit_len > max_out_size - out_pos) return false;
        if (lit_len > static_cast<size_t>(lit_end - lit_cur)) return false;
        if (HWY_LIKELY(lit_len <= 16 && out_pos + 16 <= wild_end &&
                       static_cast<size_t>(lit_end - lit_cur) >= 16)) {
          CopyBytes<16>(lit_cur, dst + out_pos);
        } else {
          CopyBytes(lit_cur, dst + out_pos, lit_len);
        }
      } else {
        CopyBytes<8>(lit_cur, dst + out_pos);
      }
      lit_cur += lit_len;
      out_pos += lit_len;

      // Offset: always load the next 16-bit value, then select between it and
      // the repeated previous offset. Whether a token repeats is close to a
      // coin flip, so branching on it mispredicts; this compiles to a cmov.
      const int64_t new_offs =
          -static_cast<int64_t>(ScalarLoadULittleEndian<uint16_t>(off16_cur));
      const size_t read_off = (~static_cast<size_t>(token) >> 7) & 1;
      off16_cur += 2 * read_off;
      last_offs = read_off ? new_offs : last_offs;

      match_len =
          static_cast<int64_t>((token >> kLiteralLenBits) & kMaxShortMatchLen);
      if (HWY_UNLIKELY(match_len == static_cast<int64_t>(kMaxShortMatchLen))) {
        match_len = var_match_len_stream.VarUint(&ok) +
                    static_cast<int64_t>(kMaxShortMatchLen);
        if (!ok) return false;
      }

      // One 16-byte store covers any short match, and because the encoder
      // never emits a distance below kMinOffset (32), the source does not
      // overlap it. The compare chain is branchless and the branch itself is
      // taken for essentially every token of a well-formed block.
      const uint64_t dist = static_cast<uint64_t>(-last_offs);
      if (HWY_LIKELY(static_cast<uint64_t>(match_len) <= 16 && dist >= 16 &&
                     dist <= out_pos && out_pos <= wild_end)) {
        CopyBytes<16>(dst + out_pos - dist, dst + out_pos);
        out_pos += static_cast<size_t>(match_len);
      } else if (!CheckedCopyMatch(dst, &out_pos, last_offs, match_len,
                                   max_out_size)) {
        return false;
      }
    }

    token_stream.cursor = static_cast<size_t>(token_cur - token_stream.data);
    off16_stream.cursor = static_cast<size_t>(off16_cur - off16_stream.data);
    literal_stream.cursor = static_cast<size_t>(lit_cur - literal_stream.data);
  }

  // ------------------------------ Careful loop (tail and malformed input)
  while (!token_stream.IsEmpty()) {
    int64_t match_len = 0;
    const uint8_t token = token_stream.U8(&ok);
    // 0x80 is a NOP: it carries neither literals nor a match, so a block could
    // pad itself with them and make us walk the token stream for no output.
    if (!ok || token == 0x80) return false;

    if (token >= 32) {
      int64_t lit_len = static_cast<int64_t>(token & kMaxShortLitLen);
      if (lit_len == static_cast<int64_t>(kMaxShortLitLen)) {
        lit_len = var_lit_len_stream.VarUint(&ok) +
                  static_cast<int64_t>(kMaxShortLitLen);
        if (!ok) return false;
      }
      if (lit_len > 0) {
        const size_t n = static_cast<size_t>(lit_len);
        if (n > max_out_size - out_pos) return false;
        const uint8_t* const p = literal_stream.Sequence(n, &ok);
        if (!ok) return false;
        CopyBytes(p, dst + out_pos, n);
        out_pos += n;
      }
      if ((token & 0x80) == 0) {
        last_offs = -static_cast<int64_t>(off16_stream.U16(&ok));
        if (!ok) return false;
      }
      match_len =
          static_cast<int64_t>((token >> kLiteralLenBits) & kMaxShortMatchLen);
      if (match_len == static_cast<int64_t>(kMaxShortMatchLen)) {
        match_len = var_match_len_stream.VarUint(&ok) +
                    static_cast<int64_t>(kMaxShortMatchLen);
        if (!ok) return false;
      }
    } else if (token < kLastLongOffset) {
      match_len =
          static_cast<int64_t>(token) + static_cast<int64_t>(kMMLongOffsets);
      last_offs = -static_cast<int64_t>(off24_stream.U24(&ok));
      if (!ok) return false;
    } else {
      match_len = var_match_len_stream.VarUint(&ok) +
                  static_cast<int64_t>(kLastLongOffset + kMMLongOffsets);
      if (!ok) return false;
      last_offs = -static_cast<int64_t>(off24_stream.U24(&ok));
      if (!ok) return false;
    }

    if (!CheckedCopyMatch(dst, &out_pos, last_offs, match_len, max_out_size)) {
      return false;
    }
  }

  // The offset/length streams have to be fully consumed: leftover bytes mean
  // the block is inconsistent even if the token stream ended cleanly.
  if (!off16_stream.IsEmpty() || !off24_stream.IsEmpty() ||
      !var_lit_len_stream.IsEmpty() || !var_match_len_stream.IsEmpty()) {
    return false;
  }

  const size_t remaining = literal_stream.RemainingBytes();
  if (remaining > 0) {
    if (remaining > max_out_size - out_pos) return false;
    const uint8_t* const p = literal_stream.Sequence(remaining, &ok);
    if (!ok) return false;
    CopyBytes(p, dst + out_pos, remaining);
    out_pos += remaining;
  }
  *io_out_pos = out_pos;
  return true;
}

// ------------------------------ Container (shared by both paths)
//
// This parses untrusted input, so every length is validated before any
// allocation: offsets are checked by subtraction (addition can wrap around),
// each command may only consume payload bytes that are still in front of the
// control section, and no command may produce more than the declared total,
// which (together with kMaxUncompressedSize) bounds "zip bombs".
//
// Reading a command and executing it are separate steps so that both the
// serial and the parallel driver below can share one implementation. The
// parallel driver needs the split anyway: it has to locate every command's
// payload before it can hand the commands to workers.

// Size of each worker's private scratch region in IguanaWorkspace (one 2 MiB
// huge page per worker).
HWY_INLINE_VAR constexpr size_t kHugePage = size_t{2} << 20;
HWY_INLINE_VAR constexpr size_t kStreamScratchPad = 64;

// Everything needed to execute one command without re-reading the control
// section. Filled in completely by ParseOneCommand, so it deliberately has no
// default member initializers: it is bulk-allocated as raw storage.
struct CommandDesc {
  uint64_t hdr;                  // kCmdDecodeIguana: 4-bit mode per stream
  uint64_t ulens[kStreamCount];  // kCmdDecodeIguana: uncompressed stream sizes
  uint64_t clens[kStreamCount];  // kCmdDecodeIguana: ANS32 streams only
  size_t data_begin;             // payload offset in `src`
  uint64_t lu;                   // kCmdCopyRaw / kCmdDecodeANS32: output bytes
  uint64_t lc;                   // kCmdDecodeANS32: payload bytes
  uint8_t cmd;                   // including kLastCommandMarker
};

// Scratch bytes `DecodeCommand` needs to buffer the ANS32-coded streams of
// `desc`. `ParseOneCommand` already verified that `ulens` sum to at most
// `kMaxUncompressedSize`, so this addition cannot overflow `size_t`.
HWY_INLINE size_t CommandScratchBytes(const CommandDesc& desc) {
  if ((desc.cmd & kCommandMask) != kCmdDecodeIguana) return 0;
  size_t needed = 0;
  for (size_t i = 0; i < kStreamCount; ++i) {
    if (((desc.hdr >> (i * 4)) & 0xF) != 0) {
      needed +=
          RoundUpTo(static_cast<size_t>(desc.ulens[i]) + kStreamScratchPad,
                    kStreamScratchPad);
    }
  }
  return needed;
}

// Reads one command's control fields, moving `*ctrl` backwards past them and
// `*data_cursor` forwards past its payload. Validates that the payload stays
// in front of the control section; the output lengths depend on how much of
// `dst` earlier commands consumed and are checked by DecodeCommand.
// Returns false on malformed input.
HWY_INLINE bool ParseOneCommand(const uint8_t* HWY_RESTRICT src,
                                int64_t* HWY_RESTRICT ctrl,
                                size_t* HWY_RESTRICT data_cursor,
                                CommandDesc* HWY_RESTRICT desc) {
  if (*ctrl < 0) return false;
  bool ok = true;
  desc->cmd = src[static_cast<size_t>(*ctrl)];
  --*ctrl;

  // The payload is read forwards from the start and the control bytes
  // backwards from the end: they must not overlap, so a command may only use
  // bytes that are still in front of the control section. `*ctrl` is the last
  // unconsumed control byte, hence the +1; once it goes negative nothing is
  // left. The subtraction has to be guarded (and it is a subtraction, not
  // `*data_cursor + len <= ...`, because that sum could wrap around).
  // ctrl_begin <= src_size, so this also bounds the read.
  const auto have_payload = [&](uint64_t len) {
    const uint64_t ctrl_begin =
        *ctrl < 0 ? 0 : static_cast<uint64_t>(*ctrl) + 1;
    if (ctrl_begin < *data_cursor) return false;
    return len <= ctrl_begin - *data_cursor;
  };

  desc->data_begin = *data_cursor;

  switch (desc->cmd & kCommandMask) {
    case kCmdCopyRaw: {
      const uint64_t n = ReadControlVarUint(src, ctrl, &ok);
      if (!ok || !have_payload(n)) return false;
      desc->lu = n;
      desc->lc = n;
      *data_cursor += static_cast<size_t>(n);
      return true;
    }
    case kCmdDecodeANS32: {
      desc->lu = ReadControlVarUint(src, ctrl, &ok);
      desc->lc = ReadControlVarUint(src, ctrl, &ok);
      if (!ok || !have_payload(desc->lc)) return false;
      *data_cursor += static_cast<size_t>(desc->lc);
      return true;
    }
    case kCmdDecodeIguana: {
      desc->hdr = ReadControlVarUint(src, ctrl, &ok);
      if (!ok) return false;
      // Each stream is buffered in full before the LZ stage runs, so it is
      // the *total* that has to be bounded: checking only each ulens[i]
      // against the declared output would still let a tiny block ask for
      // kStreamCount times that. The streams of a block the encoder
      // produced are together smaller than its output, so the cap costs
      // nothing in practice and the peak stays within kMaxUncompressedSize.
      uint64_t ulens_total = 0;
      for (size_t i = 0; i < kStreamCount; ++i) {
        desc->ulens[i] = ReadControlVarUint(src, ctrl, &ok);
        if (!ok || desc->ulens[i] > kMaxUncompressedSize - ulens_total) {
          return false;
        }
        ulens_total += desc->ulens[i];
      }
      for (size_t i = 0; i < kStreamCount; ++i) {
        const size_t mode = static_cast<size_t>((desc->hdr >> (i * 4)) & 0xF);
        if (mode == 0) {
          if (!have_payload(desc->ulens[i])) return false;
          desc->clens[i] = desc->ulens[i];
          *data_cursor += static_cast<size_t>(desc->ulens[i]);
        } else if (mode == 1) {  // EntropyANS32
          desc->clens[i] = ReadControlVarUint(src, ctrl, &ok);
          if (!ok || !have_payload(desc->clens[i])) return false;
          *data_cursor += static_cast<size_t>(desc->clens[i]);
        } else {
          return false;  // ANS1 / ANS_nibble not implemented
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// Executes a parsed command, appending to `dst[*io_out_pos]`. `max_out_size`
// bounds the output; matches may reach back to `dst[0]`, which for the serial
// driver is the start of the block and for the parallel one the start of the
// chunk (making each chunk self-contained). `scratch` provides the worker's
// pre-allocated buffer for ANS32-coded streams. Returns false on malformed
// input.
template <class DecodeFn>
bool DecodeCommand(const uint8_t* HWY_RESTRICT src, const CommandDesc& desc,
                   uint8_t* HWY_RESTRICT dst, size_t* HWY_RESTRICT io_out_pos,
                   size_t max_out_size, Span<uint8_t> scratch,
                   DecodeFn decode) {
  size_t out_pos = *io_out_pos;

  switch (desc.cmd & kCommandMask) {
    case kCmdCopyRaw: {
      if (desc.lu > max_out_size - out_pos) return false;
      if (desc.lu > 0) {
        CopyBytes(src + desc.data_begin, dst + out_pos,
                  static_cast<size_t>(desc.lu));
        out_pos += static_cast<size_t>(desc.lu);
      }
      break;
    }
    case kCmdDecodeANS32: {
      if (desc.lu > max_out_size - out_pos) return false;
      if (!decode(Span<const uint8_t>(src + desc.data_begin,
                                      static_cast<size_t>(desc.lc)),
                  Span<uint8_t>(dst + out_pos, static_cast<size_t>(desc.lu)))) {
        return false;
      }
      out_pos += static_cast<size_t>(desc.lu);
      break;
    }
    case kCmdDecodeIguana: {
      IguanaStream streams[kStreamCount];
      uint8_t* const scratch_base = scratch.data();
      size_t scratch_off = 0;
      size_t cursor = desc.data_begin;
      for (size_t i = 0; i < kStreamCount; ++i) {
        const size_t ulen = static_cast<size_t>(desc.ulens[i]);
        const size_t clen = static_cast<size_t>(desc.clens[i]);
        if (((desc.hdr >> (i * 4)) & 0xF) == 0) {
          streams[i].data = src + cursor;
          streams[i].size = ulen;
        } else {
          const size_t step =
              RoundUpTo(ulen + kStreamScratchPad, kStreamScratchPad);
          if (HWY_UNLIKELY(step > scratch.size() - scratch_off)) return false;
          uint8_t* const out_buf = scratch_base + scratch_off;
          scratch_off += step;
          if (!decode(Span<const uint8_t>(src + cursor, clen),
                      Span<uint8_t>(out_buf, ulen))) {
            return false;
          }
          streams[i].data = out_buf;
          streams[i].size = ulen;
        }
        cursor += clen;
      }
      if (!DecompressIguanaLZ(dst, &out_pos, streams, max_out_size)) {
        return false;
      }
      break;
    }
    default:
      return false;
  }

  // Every command has to make progress. A token that carries neither a
  // literal nor a match (0x80, or a zero-length command) would otherwise
  // decode nothing while consuming control bytes, letting a small block ask
  // for unbounded work.
  if (out_pos == *io_out_pos) return false;
  *io_out_pos = out_pos;
  return true;
}

// Reads the block header into `*out_total` and positions `*ctrl` on the first
// command. Returns false if the header is malformed, if it declares more than
// kMaxUncompressedSize, or if `dst_size` cannot hold the result. `*out_empty`
// is set for a well-formed block that carries no command at all.
HWY_INLINE bool ParseBlockHeader(Span<const uint8_t> src, size_t dst_size,
                                 size_t* HWY_RESTRICT out_total,
                                 int64_t* HWY_RESTRICT out_ctrl,
                                 bool* HWY_RESTRICT out_empty) {
  if (src.empty()) return false;
  bool ok = true;
  int64_t ctrl = static_cast<int64_t>(src.size()) - 1;
  const uint64_t uncompressed_len = ReadControlVarUint(src.data(), &ctrl, &ok);
  if (!ok) return false;
  // An empty block carries no command, so the length must have been the only
  // thing in it; trailing bytes mean the input is not the block it claims.
  if (uncompressed_len == 0) {
    *out_empty = true;
    *out_total = 0;
    *out_ctrl = ctrl;
    return ctrl < 0;
  }
  if (uncompressed_len > kMaxUncompressedSize) return false;
  if (dst_size < static_cast<size_t>(uncompressed_len)) return false;
  *out_empty = false;
  *out_total = static_cast<size_t>(uncompressed_len);
  *out_ctrl = ctrl;
  return true;
}

// Decodes a complete Iguana block into pre-allocated `dst`. `decode` is the
// entropy stage, i.e. Ans32DecodeScalar (scalar path) or Ans32Decode (SIMD
// path):
//   bool(Span<const uint8_t> src, Span<uint8_t> dst)
//
// Returns the number of bytes written, or kDecompressFailed on malformed input
// or if `dst.size() < uncompressed_len`. Failure is not reported as 0 because a
// valid block may legitimately decode to nothing.
template <class DecodeFn>
size_t DecompressBlock(Span<const uint8_t> src_span, Span<uint8_t> dst_span,
                       DecodeFn decode, IguanaWorkspace& ws) {
  const uint8_t* const src = src_span.data();
  size_t max_out_size;
  int64_t ctrl;
  bool empty;
  if (!ParseBlockHeader(src_span, dst_span.size(), &max_out_size, &ctrl,
                        &empty)) {
    return kDecompressFailed;
  }
  if (empty) return 0;

  uint8_t* const dst = dst_span.data();
  size_t out_pos = 0;
  size_t data_cursor = 0;

  for (;;) {
    CommandDesc desc;
    if (!ParseOneCommand(src, &ctrl, &data_cursor, &desc)) {
      return kDecompressFailed;
    }
    const size_t needed = CommandScratchBytes(desc);
    if (needed > ws.Capacity()) {
      if (!ws.Reserve(needed, 1)) return kDecompressFailed;
    }
    const Span<uint8_t> scratch(ws.Memory(), ws.Capacity());
    if (!DecodeCommand(src, desc, dst, &out_pos, max_out_size, scratch,
                       decode)) {
      return kDecompressFailed;
    }
    if (desc.cmd & kLastCommandMarker) {
      // A complete block must produce exactly the declared number of bytes and
      // consume every payload byte in front of the control section.
      if (out_pos != max_out_size) return kDecompressFailed;
      if (data_cursor != static_cast<size_t>(ctrl + 1)) {
        return kDecompressFailed;
      }
      return out_pos;
    }
  }
}

// Locates every command of a block that follows the fixed-size chunk
// convention, i.e. one command per kChunkSize of output. Returns false - and
// the caller then falls back to DecompressBlock, which handles arbitrary
// blocks - if the block was produced by an encoder that chunks differently, or
// not at all. This is a rejection of the *parallel* path only, never of the
// input: a block that fails here still decodes serially.
HWY_INLINE bool ParseChunkedCommands(Span<const uint8_t> src, size_t total,
                                     int64_t ctrl, size_t num_chunks,
                                     CommandDesc* HWY_RESTRICT descs) {
  size_t data_cursor = 0;
  for (size_t k = 0; k < num_chunks; ++k) {
    if (!ParseOneCommand(src.data(), &ctrl, &data_cursor, &descs[k])) {
      return false;
    }
    const bool last = (descs[k].cmd & kLastCommandMarker) != 0;
    // Exactly one command per chunk: a block with more or fewer is not the
    // layout this path assumes, whoever produced it.
    if (last != (k + 1 == num_chunks)) return false;
    // The two self-describing commands can be checked here; kCmdDecodeIguana
    // only reveals its output length while decoding, so it is checked against
    // ChunkLen after the fact (see DecompressBlockParallel).
    const size_t expected = ChunkLen(total, k);
    if ((descs[k].cmd & kCommandMask) != kCmdDecodeIguana &&
        descs[k].lu != expected) {
      return false;
    }
    // Each worker's scratch slice is kHugePage (2 MiB); any chunk that asks
    // for more (impossible for a 256 KiB chunk produced by Compress, whose
    // streams total < 260 KiB) falls back to DecompressBlock.
    if (CommandScratchBytes(descs[k]) > kHugePage) return false;
  }
  // Same whole-block consistency check as the serial driver.
  return data_cursor == static_cast<size_t>(ctrl + 1);
}

// Parallel counterpart of DecompressBlock: decodes the chunks of a block that
// Compress produced on `pool`, indexing per-worker regions in `ws` with the
// worker ID (`worker`), and falls back to the serial driver for blocks that do
// not follow the chunk convention. Each worker writes only its own `dst` slice
// and its own `ws` slice, so a malformed chunk cannot corrupt another's output;
// matches are confined to the chunk for the same reason.
template <class DecodeFn>
size_t DecompressBlockParallel(Span<const uint8_t> src, Span<uint8_t> dst,
                               DecodeFn decode, IguanaWorkspace& ws,
                               ThreadPool& pool) {
  size_t total;
  int64_t ctrl;
  bool empty;
  if (!ParseBlockHeader(src, dst.size(), &total, &ctrl, &empty)) {
    return kDecompressFailed;
  }
  if (empty) return 0;

  const size_t num_chunks = NumChunks(total);
  const size_t num_workers = HWY_MAX(pool.NumWorkers(), size_t{1});
  if (num_chunks < 2 || num_workers < 2) {
    return DecompressBlock(src, dst, decode, ws);
  }

  if (HWY_UNLIKELY(!ws.Reserve(total, num_workers))) {
    return kDecompressFailed;
  }
  uint8_t* const mem = ws.Memory();
  CommandDesc* const descs =
      HWY_RCAST_ALIGNED(CommandDesc*, mem + num_workers * kHugePage);
  if (!ParseChunkedCommands(src, total, ctrl, num_chunks, descs)) {
    return DecompressBlock(src, dst, decode, ws);
  }

  // `failed` is only ever set, never cleared, so a relaxed store suffices:
  // the joining barrier in Run() provides the ordering.
  std::atomic<bool> failed{false};
  pool.Run(0, num_chunks, [&](uint64_t task, size_t worker) {
    const size_t k = static_cast<size_t>(task);
    const size_t expected = ChunkLen(total, k);
    const Span<uint8_t> scratch(mem + worker * kHugePage, kHugePage);
    size_t out_pos = 0;
    if (!DecodeCommand(src.data(), descs[k], dst.data() + k * kChunkSize,
                       &out_pos, expected, scratch, decode) ||
        out_pos != expected) {
      failed.store(true, std::memory_order_relaxed);
    }
  });

  return failed.load(std::memory_order_relaxed) ? kDecompressFailed : total;
}

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_DETAIL_H_
