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

// Iguana: a Lizard-derived LZ77 + rANS compressor
// (github.com/SnellerInc/sneller, ion/zion/iguana), ported to Highway from the
// pure-Go reference. The bitstream is byte-for-byte compatible with it.
//
// This header exposes the scalar codec. The SIMD decode path (which routes the
// entropy-coded streams through the vectorized ANS32 decoder in ans-inl.h) is
// in iguana-inl.h. Encoding is scalar, as in the reference.
//
// Covers the EncodingIguana / EntropyANS32 pipeline (what Encoder.Compress
// produces): the container, the LZ77 layer, and ANS32-coded streams. ANS1 and
// ANS_nibble stream modes are not implemented.

#ifndef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_
#define HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/contrib/iguana/ans.h"

namespace hwy {
namespace iguana {

// ------------------------------ Format constants

constexpr int kIguanaChunkSize = 32;
constexpr int kMinOffset = 32;
constexpr int kMinLength = 32;
constexpr int kLiteralLenBits = 3;
constexpr int kMMLongOffsets = 16;
constexpr int kMaxShortLitLen = 7;
constexpr int kMaxShortMatchLen = 15;
constexpr int kLastLongOffset = 31;
constexpr int kChainBits = 17;
constexpr int kHashBytes = 5;
constexpr int kHistSize = 4;
constexpr int kStreamCount = 6;

enum Command {
  kCmdCopyRaw = 0,
  kCmdDecodeIguana = 1,
  kCmdDecodeANS32 = 2,
  kCmdDecodeANS1 = 3,
  kCmdDecodeANSNibble = 4,
};
constexpr uint8_t kLastCommandMarker = 0x80;
constexpr uint8_t kCommandMask = 0x7F;

// ------------------------------ Codec

// Compresses `data` into a complete Iguana block (EncodingIguana / ANS32).
std::vector<uint8_t> Compress(const uint8_t* data, size_t size);

// Decompresses a block produced by Compress. Returns false on malformed input.
// The SIMD path in iguana-inl.h produces identical output.
bool DecompressScalar(const uint8_t* src, size_t src_size,
                      std::vector<uint8_t>& out);

// ------------------------------ Internals shared with iguana-inl.h

// One of the six token/literal/offset streams handed to the LZ77 stage.
struct IguanaStream {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// The LZ77 stage: expands the six streams into `dst` (appended). Scalar; the
// token loop is inherently serial. Returns false on malformed input.
bool DecompressIguanaLZ(std::vector<uint8_t>& dst,
                        const IguanaStream streams[kStreamCount]);

// Reads a big-endian base-128 varint backwards from src[*cursor], moving
// *cursor before the consumed bytes. Sets *ok=false on underflow.
uint64_t ReadControlVarUint(const uint8_t* src, int64_t* cursor, bool* ok);

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_
