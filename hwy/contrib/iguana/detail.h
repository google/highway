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

#ifndef HIGHWAY_HWY_CONTRIB_IGUANA_DETAIL_H_
#define HIGHWAY_HWY_CONTRIB_IGUANA_DETAIL_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/base.h"

namespace hwy {
namespace iguana {

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
HWY_INLINE_VAR constexpr size_t kChainBits = 17;
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

// ------------------------------ LZ77 stage (decoder)

// One of the six token/literal/offset streams handed to the LZ77 stage.
struct IguanaStream {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// Appends match_len bytes copied from dst[match_pos..). Overlapping runs
// (where the source reaches into the bytes being produced) are the common
// case, so we resize first - which keeps the destination pointers valid,
// and is what makes the copy below well-defined - and then copy forward,
// where every byte read has already been written.
HWY_INLINE bool CopyMatch(std::vector<uint8_t>& dst, size_t match_pos,
                          size_t match_len, uint64_t uncompressed_len) {
  if (match_len > uncompressed_len - dst.size()) return false;
  const size_t old_size = dst.size();
  dst.resize(old_size + match_len);
  uint8_t* const HWY_RESTRICT out = dst.data();
  for (size_t i = 0; i < match_len; ++i) {
    out[old_size + i] = out[match_pos + i];
  }
  return true;
}

// The LZ77 stage: expands the six streams into `dst` (appended). The token
// loop is inherently serial, so this stays scalar on the SIMD path too.
// Returns false on malformed input.
HWY_INLINE bool DecompressIguanaLZ(std::vector<uint8_t>& dst,
                                   const IguanaStream streams_in[kStreamCount],
                                   uint64_t uncompressed_len) {
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

  bool ok = true;
  // Offset of the previous match, negated: the streams encode the distance
  // back from the current output position. Zero means "no match seen yet", and
  // a match that refers to it is malformed.
  int64_t last_offs = 0;
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
        if (n > uncompressed_len - dst.size()) return false;
        const uint8_t* const p = literal_stream.Sequence(n, &ok);
        if (!ok) return false;
        dst.insert(dst.end(), p, p + n);
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

    if (match_len > 0) {
      // A match must refer to an offset already emitted, otherwise it would
      // read before the start of the output.
      if (last_offs == 0) return false;
      const int64_t match_pos = static_cast<int64_t>(dst.size()) + last_offs;
      if (match_pos < 0 || match_pos > static_cast<int64_t>(dst.size())) {
        return false;
      }
      if (!CopyMatch(dst, static_cast<size_t>(match_pos),
                     static_cast<size_t>(match_len), uncompressed_len)) {
        return false;
      }
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
    if (remaining > uncompressed_len - dst.size()) return false;
    const uint8_t* const p = literal_stream.Sequence(remaining, &ok);
    if (!ok) return false;
    dst.insert(dst.end(), p, p + remaining);
  }
  return true;
}

// ------------------------------ Container loop (shared by both paths)

// Decodes a complete Iguana block, appending to `out`. `decode` is the entropy
// stage, i.e. Ans32DecodeScalar (scalar path) or Ans32Decode (SIMD path):
//   bool(const uint8_t* src, size_t src_size, uint8_t* dst, size_t orig_size)
//
// This parses untrusted input, so every length is validated before any
// allocation: offsets are checked by subtraction (addition can wrap around),
// each command may only consume payload bytes that are still in front of the
// control section, and no command may produce more than the declared total,
// which (together with kMaxUncompressedSize) bounds "zip bombs".
// Returns false on malformed input.
template <class DecodeFn>
bool DecompressBlock(const uint8_t* src, size_t src_size,
                     std::vector<uint8_t>& out, DecodeFn decode) {
  if (src_size == 0) return false;
  bool ok = true;
  int64_t ctrl = static_cast<int64_t>(src_size) - 1;
  const uint64_t uncompressed_len = ReadControlVarUint(src, &ctrl, &ok);
  if (!ok) return false;
  out.clear();
  if (uncompressed_len == 0) return true;
  if (uncompressed_len > kMaxUncompressedSize) return false;

  size_t data_cursor = 0;
  // The payload is read forwards from the start and the control bytes backwards
  // from the end: they must not overlap, so a command may only use bytes that
  // are still in front of the control section. `ctrl` is the last unconsumed
  // control byte, hence the +1.
  const auto have_payload = [&](uint64_t len) {
    if (len > src_size - data_cursor) return false;
    return data_cursor + len <= static_cast<uint64_t>(ctrl) + 1;
  };
  // One per stream, as a fixed array: a vector of vectors would reallocate as
  // entries are appended, and the streams below point into these buffers.
  std::vector<uint8_t> ent_bufs[kStreamCount];

  for (;;) {
    if (ctrl < 0) return false;
    const uint8_t cmd = src[static_cast<size_t>(ctrl)];
    --ctrl;
    const size_t prev_out_size = out.size();

    switch (cmd & kCommandMask) {
      case kCmdCopyRaw: {
        const uint64_t n = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || !have_payload(n)) return false;
        if (n > uncompressed_len - out.size()) return false;
        out.insert(out.end(), src + data_cursor,
                   src + data_cursor + static_cast<size_t>(n));
        data_cursor += static_cast<size_t>(n);
        break;
      }
      case kCmdDecodeANS32: {
        const uint64_t lu = ReadControlVarUint(src, &ctrl, &ok);
        const uint64_t lc = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || !have_payload(lc)) return false;
        if (lu > uncompressed_len - out.size()) return false;
        const size_t out_pos = out.size();
        out.resize(out_pos + static_cast<size_t>(lu));
        if (!decode(src + data_cursor, static_cast<size_t>(lc),
                    out.data() + out_pos, static_cast<size_t>(lu))) {
          return false;
        }
        data_cursor += static_cast<size_t>(lc);
        break;
      }
      case kCmdDecodeIguana: {
        const uint64_t hdr = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok) return false;
        IguanaStream streams[kStreamCount];
        uint64_t ulens[kStreamCount];
        for (size_t i = 0; i < kStreamCount; ++i) {
          ulens[i] = ReadControlVarUint(src, &ctrl, &ok);
          // Each stream is part of the output, so the declared total bounds
          // it; the cap alone would still permit full-size allocations.
          if (!ok || ulens[i] > uncompressed_len) return false;
        }
        for (size_t i = 0; i < kStreamCount; ++i) {
          const size_t mode = static_cast<size_t>((hdr >> (i * 4)) & 0xF);
          if (mode == 0) {
            if (!have_payload(ulens[i])) return false;
            streams[i].data = src + data_cursor;
            streams[i].size = static_cast<size_t>(ulens[i]);
            data_cursor += static_cast<size_t>(ulens[i]);
          } else if (mode == 1) {  // EntropyANS32
            const uint64_t clen = ReadControlVarUint(src, &ctrl, &ok);
            if (!ok || !have_payload(clen)) return false;
            std::vector<uint8_t>& buf = ent_bufs[i];
            buf.resize(static_cast<size_t>(ulens[i]));
            if (!decode(src + data_cursor, static_cast<size_t>(clen),
                        buf.data(), static_cast<size_t>(ulens[i]))) {
              return false;
            }
            streams[i].data = buf.data();
            streams[i].size = buf.size();
            data_cursor += static_cast<size_t>(clen);
          } else {
            return false;  // ANS1 / ANS_nibble not implemented
          }
        }
        if (!DecompressIguanaLZ(out, streams, uncompressed_len)) return false;
        // The streams pointed into ent_bufs and the LZ stage is done with them;
        // releasing them keeps memory bounded across commands.
        for (auto& buf : ent_bufs) buf.clear();
        break;
      }
      default:
        return false;
    }

    // Every command has to make progress. A token that carries neither a
    // literal nor a match (0x80, or a zero-length command) would otherwise
    // decode nothing while consuming control bytes, letting a small block ask
    // for unbounded work.
    if (out.size() == prev_out_size) return false;

    if (cmd & kLastCommandMarker) {
      // A complete block must produce exactly the declared number of bytes and
      // consume every payload byte in front of the control section.
      if (out.size() != uncompressed_len) return false;
      if (data_cursor != static_cast<size_t>(ctrl + 1)) return false;
      return true;
    }
  }
}

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_DETAIL_H_
