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

// SIMD decode path for Iguana: the container loop routes each entropy-coded
// stream through the vectorized ANS32 decoder (ans-inl.h); the container
// parsing and the LZ77 stage (both inherently serial) are the target-
// independent helpers in iguana.h. Output is identical to
// hwy::iguana::DecompressScalar.

#if defined(HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_
#undef HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_INL_H_
#endif

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/contrib/iguana/ans-inl.h"
#include "hwy/contrib/iguana/ans.h"
#include "hwy/contrib/iguana/iguana.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace iguana_full {

namespace hi = hwy::iguana;

// Decompresses a block produced by hi::Compress, appending to `out`. The
// entropy-coded streams go through the SIMD ANS32 decoder. Returns false on
// malformed input; output matches hi::DecompressScalar.
HWY_INLINE bool Decompress(const uint8_t* HWY_RESTRICT src, size_t src_size,
                           std::vector<uint8_t>& out) {
  if (src_size == 0) return false;
  bool ok = true;
  int64_t ctrl = static_cast<int64_t>(src_size) - 1;
  const uint64_t uncompressed_len = hi::ReadControlVarUint(src, &ctrl, &ok);
  if (!ok) return false;
  out.clear();
  if (uncompressed_len == 0) return true;

  uint64_t data_cursor = 0;
  std::vector<std::vector<uint8_t>> ent_bufs;

  for (;;) {
    if (ctrl < 0) return false;
    const uint8_t cmd = src[ctrl];
    --ctrl;

    switch (cmd & hi::kCommandMask) {
      case hi::kCmdCopyRaw: {
        const uint64_t n = hi::ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || data_cursor + n > src_size) return false;
        out.insert(out.end(), src + data_cursor, src + data_cursor + n);
        data_cursor += n;
        break;
      }
      case hi::kCmdDecodeANS32: {
        const uint64_t lu = hi::ReadControlVarUint(src, &ctrl, &ok);
        const uint64_t lc = hi::ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || data_cursor + lc > src_size) return false;
        const size_t out_pos = out.size();
        out.resize(out_pos + static_cast<size_t>(lu));
        if (!iguana_ans::Ans32Decode(src + data_cursor, static_cast<size_t>(lc),
                                     out.data() + out_pos,
                                     static_cast<size_t>(lu))) {
          return false;
        }
        data_cursor += lc;
        break;
      }
      case hi::kCmdDecodeIguana: {
        const uint64_t hdr = hi::ReadControlVarUint(src, &ctrl, &ok);
        if (!ok) return false;
        hi::IguanaStream streams[hi::kStreamCount];
        uint64_t ulens[hi::kStreamCount];
        for (int i = 0; i < hi::kStreamCount; ++i) {
          ulens[i] = hi::ReadControlVarUint(src, &ctrl, &ok);
          if (!ok) return false;
        }
        for (int i = 0; i < hi::kStreamCount; ++i) {
          const int em = static_cast<int>((hdr >> (i * 4)) & 0xF);
          if (em == 0) {
            if (data_cursor + ulens[i] > src_size) return false;
            streams[i].data = src + data_cursor;
            streams[i].size = static_cast<size_t>(ulens[i]);
            data_cursor += ulens[i];
          } else if (em == 1) {
            const uint64_t clen = hi::ReadControlVarUint(src, &ctrl, &ok);
            if (!ok || data_cursor + clen > src_size) return false;
            ent_bufs.emplace_back();
            std::vector<uint8_t>& b = ent_bufs.back();
            b.resize(static_cast<size_t>(ulens[i]));
            if (!iguana_ans::Ans32Decode(src + data_cursor,
                                         static_cast<size_t>(clen), b.data(),
                                         static_cast<size_t>(ulens[i]))) {
              return false;
            }
            streams[i].data = b.data();
            streams[i].size = b.size();
            data_cursor += clen;
          } else {
            return false;
          }
        }
        if (!hi::DecompressIguanaLZ(out, streams)) return false;
        break;
      }
      default:
        return false;
    }

    if (cmd & hi::kLastCommandMarker) return true;
  }
}

}  // namespace iguana_full
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // include guard
