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
// This header is the public API: the scalar codec implemented in iguana.cc and
// the SIMD decode path in iguana-inl.h (which routes the entropy-coded streams
// through the vectorized ANS32 decoder in ans-inl.h). Encoding is scalar, as in
// the reference. Internals shared by both paths live in detail.h.
//
// Covers the EncodingIguana / EntropyANS32 pipeline (what Encoder.Compress
// produces): the container, the LZ77 layer, and ANS32-coded streams. ANS1 and
// ANS_nibble stream modes are not implemented.

#ifndef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_
#define HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/base.h"            // HWY_INLINE, HWY_RESTRICT
#include "hwy/highway_export.h"  // HWY_CONTRIB_DLLEXPORT

namespace hwy {
namespace iguana {

// Compresses `data` into a complete Iguana block (EncodingIguana / ANS32).
HWY_CONTRIB_DLLEXPORT std::vector<uint8_t> Compress(const uint8_t* data,
                                                    size_t size);

// Decompresses a block produced by Compress. Returns false on malformed input.
// The SIMD path in iguana-inl.h produces identical output.
HWY_CONTRIB_DLLEXPORT bool DecompressScalar(const uint8_t* src, size_t src_size,
                                            std::vector<uint8_t>& out);

// Same, but dispatches at run time to the best available target (the SIMD
// decoders in iguana-inl.h). Prefer this unless you specifically need the
// scalar reference; the two produce identical output. Defined in iguana-inl.h,
// which is what callers include: unlike the two functions above, there is no
// per-target entry point to select by hand.
HWY_INLINE bool Decompress(const uint8_t* src, size_t src_size,
                           std::vector<uint8_t>& out);

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_
