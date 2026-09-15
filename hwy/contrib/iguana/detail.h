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
#include "hwy/contrib/iguana/ans.h"
#include "hwy/highway_export.h"

namespace hwy {
namespace iguana {

// ------------------------------ Format constants
//
// Sizes are size_t. The few places that compare them against a signed cursor
// cast explicitly, because e.g. src_len - kMinOffset must stay signed.

constexpr size_t kIguanaChunkSize = 32;
constexpr size_t kMinOffset = 32;
constexpr size_t kMinLength = 32;
constexpr size_t kLiteralLenBits = 3;
constexpr size_t kMMLongOffsets = 16;
constexpr size_t kMaxShortLitLen = 7;
constexpr size_t kMaxShortMatchLen = 15;
constexpr size_t kLastLongOffset = 31;
constexpr size_t kChainBits = 17;
constexpr size_t kHashBytes = 5;
constexpr size_t kHistSize = 4;
constexpr size_t kStreamCount = 6;

enum Command {
  kCmdCopyRaw = 0,
  kCmdDecodeIguana = 1,
  kCmdDecodeANS32 = 2,
  kCmdDecodeANS1 = 3,
  kCmdDecodeANSNibble = 4,
};
constexpr uint8_t kLastCommandMarker = 0x80;
constexpr uint8_t kCommandMask = 0x7F;

// Largest offset representable in a 24-bit stream.
constexpr uint64_t kMaxU24 = (uint64_t{1} << 24) - 1;

// Ceiling for every allocation driven by the (untrusted) header. A small,
// malformed block can claim a huge uncompressed length ("zip bomb"), so the
// declared total and each per-stream/per-command length are checked against
// this before resizing, and the output is capped by it as well.
constexpr size_t kMaxUncompressedSize = size_t{1} << 30;  // 1 GiB

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
HWY_CONTRIB_DLLEXPORT uint64_t ReadControlVarUint(const uint8_t* src,
                                                  int64_t* cursor, bool* ok);

// ------------------------------ LZ77 stage (decoder)

// One of the six token/literal/offset streams handed to the LZ77 stage.
struct IguanaStream {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// The LZ77 stage: expands the six streams into `dst` (appended). The token
// loop is inherently serial, so this stays scalar on the SIMD path too.
// Returns false on malformed input. Defined in iguana.cc.
HWY_CONTRIB_DLLEXPORT bool DecompressIguanaLZ(
    std::vector<uint8_t>& dst, const IguanaStream streams[kStreamCount]);

// ------------------------------ Container loop (shared by both paths)

// Decodes a complete Iguana block, appending to `out`. `decode` is the entropy
// stage, i.e. Ans32DecodeScalar (scalar path) or Ans32Decode (SIMD path):
//   bool(const uint8_t* src, size_t src_size, uint8_t* dst, size_t orig_size)
//
// This parses untrusted input, so every length is validated before any
// allocation: offsets are checked by subtraction (addition can wrap around),
// and the declared uncompressed length, each stream length and each command
// length are capped at kMaxUncompressedSize, which bounds "zip bombs".
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
  std::vector<std::vector<uint8_t>> ent_bufs;

  for (;;) {
    if (ctrl < 0) return false;
    const uint8_t cmd = src[static_cast<size_t>(ctrl)];
    --ctrl;

    switch (cmd & kCommandMask) {
      case kCmdCopyRaw: {
        const uint64_t n = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || n > src_size - data_cursor) return false;
        if (n > kMaxUncompressedSize - out.size()) return false;
        out.insert(out.end(), src + data_cursor,
                   src + data_cursor + static_cast<size_t>(n));
        data_cursor += static_cast<size_t>(n);
        break;
      }
      case kCmdDecodeANS32: {
        const uint64_t lu = ReadControlVarUint(src, &ctrl, &ok);
        const uint64_t lc = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || lc > src_size - data_cursor) return false;
        if (lu > kMaxUncompressedSize - out.size()) return false;
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
          if (!ok || ulens[i] > kMaxUncompressedSize) return false;
        }
        for (size_t i = 0; i < kStreamCount; ++i) {
          const size_t mode = static_cast<size_t>((hdr >> (i * 4)) & 0xF);
          if (mode == 0) {
            if (ulens[i] > src_size - data_cursor) return false;
            streams[i].data = src + data_cursor;
            streams[i].size = static_cast<size_t>(ulens[i]);
            data_cursor += static_cast<size_t>(ulens[i]);
          } else if (mode == 1) {  // EntropyANS32
            const uint64_t clen = ReadControlVarUint(src, &ctrl, &ok);
            if (!ok || clen > src_size - data_cursor) return false;
            ent_bufs.emplace_back();
            std::vector<uint8_t>& buf = ent_bufs.back();
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
        if (!DecompressIguanaLZ(out, streams)) return false;
        break;
      }
      default:
        return false;
    }

    if (cmd & kLastCommandMarker) return true;
  }
}

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_DETAIL_H_
