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

#include <array>
#include <utility>
#include <vector>

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
using Bytes = std::vector<uint8_t>;
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

void AppendVarUint(Bytes& s, uint32_t v) {
  if (v < kVarThresh1) {
    s.push_back(static_cast<uint8_t>(v));
  } else if (v < kVarThresh3) {
    s.push_back(254);
    s.push_back(static_cast<uint8_t>(v % 254));
    s.push_back(static_cast<uint8_t>(v / 254));
  } else {
    HWY_DASSERT(v < 254u * 254 * 254);  // fits the encoder's stream varint
    const uint32_t t = v / 254;
    s.push_back(255);
    s.push_back(static_cast<uint8_t>(v % 254));
    s.push_back(static_cast<uint8_t>(t % 254));
    s.push_back(static_cast<uint8_t>(t / 254));
  }
}
void AppendU24(Bytes& s, uint32_t v) {
  s.push_back(static_cast<uint8_t>(v));
  s.push_back(static_cast<uint8_t>(v >> 8));
  s.push_back(static_cast<uint8_t>(v >> 16));
}
void AppendU16(Bytes& s, uint32_t v) {
  s.push_back(static_cast<uint8_t>(v));
  s.push_back(static_cast<uint8_t>(v >> 8));
}

// ------------------------------ control-byte writer (encoder)

struct ControlWriter {
  Bytes ctrl;
  int64_t last_command_offset = -1;

  void Command(uint8_t v) {
    if (last_command_offset >= 0) {
      ctrl[static_cast<size_t>(last_command_offset)] &= kCommandMask;
    }
    last_command_offset = static_cast<int64_t>(ctrl.size());
    ctrl.push_back(static_cast<uint8_t>(v | kLastCommandMarker));
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
      ctrl.push_back(static_cast<uint8_t>(x));
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

constexpr size_t kChainSize = size_t{1} << kChainBits;

// Loads 8 bytes little-endian, independent of the host byte order.
uint64_t Load64LE(const uint8_t* HWY_RESTRICT p) {
  return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
         (static_cast<uint64_t>(p[2]) << 16) |
         (static_cast<uint64_t>(p[3]) << 24) |
         (static_cast<uint64_t>(p[4]) << 32) |
         (static_cast<uint64_t>(p[5]) << 40) |
         (static_cast<uint64_t>(p[6]) << 48) |
         (static_cast<uint64_t>(p[7]) << 56);
}

uint32_t HashSeq(const uint8_t* HWY_RESTRICT seq) {
  const uint64_t u = Load64LE(seq);
  const uint64_t mixed = (u << 24) * 889523592379ull;  // kHashBytes == 5
  return static_cast<uint32_t>(mixed >> (64 - kChainBits));
}

// Longest common prefix of src[lo..] and src[hi..] (lo < hi).
int64_t Lcp(const uint8_t* src, size_t src_len, int64_t lo, int64_t hi) {
  int64_t m = 0;
  const int64_t n = static_cast<int64_t>(src_len);
  while (n - (hi + m) >= 8) {
    // Little-endian loads: Num0BitsBelowLS1Bit_Nonzero64 below counts from
    // the least significant *bit*, which must be the first byte.
    const uint64_t a = Load64LE(src + lo + m);
    const uint64_t b = Load64LE(src + hi + m);
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

// Extends a candidate match backwards (so the token covers as many bytes as
// possible); reports a zero length if the pair turns out not to be encodable.
void MatchExtend(const uint8_t* HWY_RESTRICT src, size_t src_len,
                 int64_t min_match_pos, int64_t chain_pos, int64_t match_pos,
                 int64_t* HWY_RESTRICT out_match_pos,
                 int64_t* HWY_RESTRICT out_chain_pos,
                 int64_t* HWY_RESTRICT out_len) {
  *out_match_pos = match_pos;
  *out_chain_pos = chain_pos;
  *out_len = Lcp(src, src_len, *out_chain_pos, *out_match_pos);
  while (*out_chain_pos > 0 &&
         src[*out_chain_pos - 1] == src[*out_match_pos - 1] &&
         *out_match_pos > min_match_pos) {
    --*out_chain_pos;
    --*out_match_pos;
    ++*out_len;
  }
  // Clamp to what the format can encode in one token; the decoder is fine
  // with a shorter match, it just copies fewer bytes.
  if (*out_len > kMaxEncodableMatchLen) *out_len = kMaxEncodableMatchLen;
  if (*out_chain_pos >= *out_match_pos ||
      !IsLegal(*out_match_pos - *out_chain_pos, *out_len)) {
    *out_match_pos = *out_chain_pos = *out_len = 0;
  }
}

struct Encoder {
  const uint8_t* src = nullptr;
  size_t src_len = 0;
  Bytes tokens, offset16, offset24, var_lit_len, var_match_len, literals;
  uint32_t last_encoded_offset = 0;
  std::vector<std::array<int32_t, kHistSize>> chains;

  Encoder() : chains(kChainSize) {}

  void Insert(int64_t pos) {
    auto& h = chains[HashSeq(src + pos)];
    h[3] = h[2];
    h[2] = h[1];
    h[1] = h[0];
    h[0] = static_cast<int32_t>(pos);
  }

  void BestChainMatch(int64_t litmin, int64_t pos,
                      int64_t* HWY_RESTRICT out_match_pos,
                      int64_t* HWY_RESTRICT out_chain_pos,
                      int64_t* HWY_RESTRICT out_len) {
    const auto& h = chains[HashSeq(src + pos)];
    MatchExtend(src, src_len, litmin, h[0], pos, out_match_pos, out_chain_pos,
                out_len);
    for (size_t i = 1; i < static_cast<size_t>(kHistSize); ++i) {
      if (h[i] == 0) break;
      int64_t cand_match_pos, cand_chain_pos, cand_len;
      MatchExtend(src, src_len, litmin, h[i], pos, &cand_match_pos,
                  &cand_chain_pos, &cand_len);
      if (cand_len > *out_len) {
        *out_match_pos = cand_match_pos;
        *out_chain_pos = cand_chain_pos;
        *out_len = cand_len;
      }
    }
  }

  void BestMatchAt(int64_t litpos, int64_t pos,
                   int64_t* HWY_RESTRICT out_match_pos,
                   int64_t* HWY_RESTRICT out_chain_pos,
                   int64_t* HWY_RESTRICT out_len) {
    *out_match_pos = pos;
    *out_chain_pos = 0;
    *out_len = 0;
    const int64_t repeat_pos = pos - static_cast<int64_t>(last_encoded_offset);
    if (repeat_pos >= 0 && repeat_pos < pos) {
      // Unlike the chain path this does not go through MatchExtend, so the
      // format limits have to be applied here as well: a single token carries
      // at most kMaxEncodableMatchLen, and the offset must still fit the
      // 24-bit stream.
      int64_t repeat_len = Lcp(src, src_len, repeat_pos, pos);
      if (repeat_len > kMaxEncodableMatchLen) {
        repeat_len = kMaxEncodableMatchLen;
      }
      // IsLegal takes the distance back from `pos`, which is the repeat
      // distance itself, not the source position we copy from.
      if (IsLegal(static_cast<int64_t>(last_encoded_offset), repeat_len)) {
        *out_chain_pos = repeat_pos;
        *out_len = repeat_len;
      }
    }
    int64_t chain_match_pos, chain_chain_pos, chain_len;
    BestChainMatch(litpos, pos, &chain_match_pos, &chain_chain_pos, &chain_len);
    if (chain_len - *out_len > 1) {
      *out_match_pos = chain_match_pos;
      *out_chain_pos = chain_chain_pos;
      *out_len = chain_len;
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
    tokens.push_back(static_cast<uint8_t>(0x80 | kMaxShortLitLen));
    AppendVarUint(var_lit_len,
                  static_cast<uint32_t>(lit_len - kMaxShortLitLen));
  }

  void Emit(const uint8_t* lit, size_t lit_len, uint32_t offs,
            uint32_t match_len) {
    literals.insert(literals.end(), lit, lit + lit_len);
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
        AppendU16(offset16, offs);
      }
      if (lit32 < kShortLit) {
        token |= lit32;
      } else {
        token |= kShortLit;
        AppendVarUint(var_lit_len, lit32 - kShortLit);
      }
      if (match_len < kShortMatch) {
        token |= match_len << kLiteralLenBits;
      } else {
        token |= kShortMatch << kLiteralLenBits;
        AppendVarUint(var_match_len, match_len - kShortMatch);
      }
      tokens.push_back(static_cast<uint8_t>(token));
    } else {
      if (lit_len > 0) {
        uint32_t token = 0x80;
        if (lit32 < kShortLit) {
          token |= lit32;
        } else {
          token |= kShortLit;
          AppendVarUint(var_lit_len, lit32 - kShortLit);
        }
        tokens.push_back(static_cast<uint8_t>(token));
      }
      AppendU24(offset24, offs);
      const uint32_t kLongBase =
          static_cast<uint32_t>(kLastLongOffset + kMMLongOffsets);
      uint32_t token;
      if (match_len < kLongBase) {
        token = match_len - static_cast<uint32_t>(kMMLongOffsets);
      } else {
        token = 0x1F;
        AppendVarUint(var_match_len, match_len - kLongBase);
      }
      tokens.push_back(static_cast<uint8_t>(token));
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
      BestMatchAt(litpos, pos, &tp, &mp, &len);
      if (pos < last) {
        int64_t t1, p1, l1;
        BestMatchAt(litpos, pos + 1, &t1, &p1, &l1);
        if (l1 > len) {
          tp = t1;
          mp = p1;
          len = l1;
        }
      }
      if (len >= 4) {
        Emit(src + litpos, static_cast<size_t>(tp - litpos),
             static_cast<uint32_t>(tp - mp), static_cast<uint32_t>(len));
        for (int64_t i = tp; i < tp + len && i < last; i += kSkipStep)
          Insert(i);
        pos = tp + len;
        litpos = pos;
      } else {
        Insert(pos);
        pos += kSkipStep;
      }
    }
    literals.insert(literals.end(), src + litpos, src + src_len);
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
HWY_CONTRIB_DLLEXPORT std::vector<uint8_t> Compress(const uint8_t* data,
                                                    size_t size) {
  if (size > kMaxUncompressedSize) return {};

  ControlWriter cw;
  cw.VarUint(size);  // total uncompressed length
  Bytes dst;

  if (size == 0) {
    // no command
  } else if (size < static_cast<size_t>(kMinLength + kHashBytes)) {
    cw.Command(kCmdCopyRaw);
    cw.VarUint(size);
    dst.insert(dst.end(), data, data + size);
  } else {
    Encoder enc;
    enc.src = data;
    enc.src_len = size;
    enc.CompressSrc();

    Bytes ustreams[kStreamCount] = {enc.tokens,        enc.offset16,
                                    enc.offset24,      enc.var_lit_len,
                                    enc.var_match_len, enc.literals};
    Bytes cstreams[kStreamCount];
    uint64_t hdr = 0;
    int64_t total = 0;
    for (const auto& u : ustreams) total += static_cast<int64_t>(u.size());

    for (size_t i = 0; i < kStreamCount; ++i) {
      // The ANS coder would wrap around (and divide by zero) at 4 GiB.
      HWY_DASSERT(static_cast<uint64_t>(ustreams[i].size()) <
                  (uint64_t{1} << 32));
      Bytes cs = Ans32Encode(ustreams[i].data(), ustreams[i].size());
      const double ratio = ustreams[i].empty()
                               ? 1e9
                               : static_cast<double>(cs.size()) /
                                     static_cast<double>(ustreams[i].size());
      if (ratio < 1.0) {
        hdr |= uint64_t{1} << (i * 4);  // EntropyANS32
        total -= static_cast<int64_t>(ustreams[i].size());
        total += static_cast<int64_t>(cs.size());
        cstreams[i] = std::move(cs);
      }
    }

    if (total + static_cast<int64_t>(kStreamCount) + 1 >=
        static_cast<int64_t>(size)) {
      cw.Command(kCmdCopyRaw);
      cw.VarUint(size);
      dst.assign(data, data + size);
    } else {
      cw.Command(kCmdDecodeIguana);
      cw.VarUint(hdr);
      for (size_t i = 0; i < kStreamCount; ++i) {
        cw.VarUint(ustreams[i].size());
      }
      for (size_t i = 0; i < kStreamCount; ++i) {
        const int em = static_cast<int>((hdr >> (i * 4)) & 0xF);
        if (em == 0) {
          dst.insert(dst.end(), ustreams[i].begin(), ustreams[i].end());
        } else {
          cw.VarUint(cstreams[i].size());
          dst.insert(dst.end(), cstreams[i].begin(), cstreams[i].end());
        }
      }
    }
  }

  for (size_t i = cw.ctrl.size(); i-- > 0;) dst.push_back(cw.ctrl[i]);
  return dst;
}

// Decompresses a block produced by Compress. The container loop is
// shared with the SIMD path (detail.h); only the entropy stage differs.
HWY_CONTRIB_DLLEXPORT bool DecompressScalar(const uint8_t* HWY_RESTRICT src,
                                            size_t src_size, Bytes& out) {
  const auto decode = [](const uint8_t* payload, size_t payload_size,
                         uint8_t* dst, size_t dst_size) {
    return Ans32DecodeScalar(payload, payload_size, dst, dst_size);
  };
  return DecompressBlock(src, src_size, out, decode);
}

HWY_EXPORT(DecompressStatic);

// Dispatches to the best target available at run time.
HWY_CONTRIB_DLLEXPORT bool Decompress(const uint8_t* HWY_RESTRICT src,
                                      size_t src_size,
                                      std::vector<uint8_t>& out) {
  return HWY_DYNAMIC_DISPATCH(DecompressStatic)(src, src_size, out);
}

}  // namespace iguana
}  // namespace hwy
#endif  // HWY_ONCE
