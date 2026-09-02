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
#include <vector>

#include "hwy/base.h"
#include "hwy/contrib/iguana/ans.h"

namespace hwy {
namespace iguana {

using Bytes = std::vector<uint8_t>;

namespace {

constexpr uint32_t kMaxU16 = (1u << 16) - 1;
constexpr uint32_t kVarThresh1 = 254;
constexpr uint32_t kVarThresh3 = 254u * 254;

// ------------------------------ little-endian helpers

uint32_t Load16LE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}
uint32_t Load24LE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16);
}

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

// ------------------------------ stream reader (LZ77 input)

struct StreamReader {
  const uint8_t* data = nullptr;
  size_t size = 0;
  size_t cursor = 0;

  bool Empty() const { return cursor >= size; }
  size_t Remaining() const { return size - cursor; }
  bool Have(size_t n) const { return cursor + n <= size; }

  uint8_t U8(bool* ok) {
    if (!Have(1)) {
      *ok = false;
      return 0;
    }
    return data[cursor++];
  }
  uint32_t U16(bool* ok) {
    if (!Have(2)) {
      *ok = false;
      return 0;
    }
    const uint32_t r = Load16LE(data + cursor);
    cursor += 2;
    return r;
  }
  uint32_t U24(bool* ok) {
    if (!Have(3)) {
      *ok = false;
      return 0;
    }
    const uint32_t r = Load24LE(data + cursor);
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
    if (!Have(n)) {
      *ok = false;
      return nullptr;
    }
    const uint8_t* r = data + cursor;
    cursor += n;
    return r;
  }
};

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
    int bit_len = 0;
    for (uint64_t t = v; t != 0; t >>= 1) ++bit_len;
    const int count = bit_len / 7 + 1;
    for (int i = count - 1; i >= 0; --i) {
      uint32_t x = static_cast<uint32_t>(v >> (i * 7)) & 0x7Fu;
      if (i == 0) x |= 0x80u;
      ctrl.push_back(static_cast<uint8_t>(x));
    }
  }
};

// ------------------------------ match finder (encoder)

constexpr size_t kChainSize = size_t{1} << kChainBits;

uint32_t HashSeq(const uint8_t* seq) {
  uint64_t u;
  memcpy(&u, seq, 8);
  u = (u << 24) * 889523592379ull;  // kHashBytes == 5
  return static_cast<uint32_t>(u >> (64 - kChainBits));
}

// Longest common prefix of src[lo..] and src[hi..] (lo < hi).
int64_t Lcp(const uint8_t* src, size_t src_len, int64_t lo, int64_t hi) {
  int64_t m = 0;
  const int64_t n = static_cast<int64_t>(src_len);
  while (n - (hi + m) >= 8) {
    uint64_t a, b;
    memcpy(&a, src + lo + m, 8);
    memcpy(&b, src + hi + m, 8);
    const uint64_t d = a ^ b;
    if (d == 0) {
      m += 8;
      continue;
    }
    unsigned tz = 0;
    while (((d >> tz) & 1) == 0) ++tz;
    return m + static_cast<int64_t>(tz / 8);
  }
  while (n - (hi + m) > 0 && src[lo + m] == src[hi + m]) ++m;
  return m;
}

bool IsLegal(int64_t offs, int64_t length) {
  return offs <= static_cast<int64_t>(kMaxU16) || length > kMaxShortMatchLen;
}

// Extends a candidate match backwards; sets len 0 if the pair is not encodable.
void MatchExtend(const uint8_t* src, size_t src_len, int64_t minto,
                 int64_t from, int64_t to, int64_t* tp, int64_t* mp,
                 int64_t* len) {
  *tp = to;
  *mp = from;
  *len = Lcp(src, src_len, *mp, *tp);
  while (*mp > 0 && src[*mp - 1] == src[*tp - 1] && *tp > minto) {
    --*mp;
    --*tp;
    ++*len;
  }
  if (*mp >= *tp || !IsLegal(*tp - *mp, *len)) {
    *tp = *mp = *len = 0;
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

  void BestChainMatch(int64_t litmin, int64_t pos, int64_t* t, int64_t* p,
                      int64_t* len) {
    const auto& h = chains[HashSeq(src + pos)];
    MatchExtend(src, src_len, litmin, h[0], pos, t, p, len);
    for (size_t i = 1; i < static_cast<size_t>(kHistSize); ++i) {
      if (h[i] == 0) break;
      int64_t at, ap, al;
      MatchExtend(src, src_len, litmin, h[i], pos, &at, &ap, &al);
      if (al > *len) {
        *t = at;
        *p = ap;
        *len = al;
      }
    }
  }

  void BestMatchAt(int64_t litpos, int64_t pos, int64_t* tp, int64_t* mp,
                   int64_t* len) {
    *tp = pos;
    *mp = 0;
    *len = 0;
    const int64_t rep = pos - static_cast<int64_t>(last_encoded_offset);
    if (rep >= 0 && rep < pos) {
      *mp = rep;
      *len = Lcp(src, src_len, rep, pos);
    }
    int64_t ht, hp, hl;
    BestChainMatch(litpos, pos, &ht, &hp, &hl);
    if (hl - *len > 1) {
      *tp = ht;
      *mp = hp;
      *len = hl;
    }

    // Keep the decoder's final 32-byte match write inside the output buffer.
    if (*tp + *len > static_cast<int64_t>(src_len) - kMinOffset) {
      if (*tp - *mp >= kMinOffset) {
        constexpr int64_t lomask = kMinOffset - 1;
        if (*tp + ((*len + lomask) & ~lomask) > static_cast<int64_t>(src_len)) {
          *len &= ~lomask;
        }
      } else {
        const int64_t movsize = *tp - *mp;
        const int64_t tailpos = movsize ? *len - (*len % movsize) : *len;
        const int64_t end = static_cast<int64_t>(src_len);
        if (*tp + tailpos + kMinOffset > end) {
          const int64_t safedist = (end - kMinOffset) - *tp;
          *len = movsize ? (safedist / movsize) * movsize : 0;
        }
      }
    }
  }

  void Emit(const uint8_t* lit, size_t lit_len, uint32_t offs,
            uint32_t match_len) {
    literals.insert(literals.end(), lit, lit + lit_len);
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
    for (auto& e : chains) e = {};
    const int64_t last = static_cast<int64_t>(src_len) - kMinOffset;
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

// ------------------------------ LZ77 wild copy (decoder)

void WildCopy(Bytes& dst, size_t pos, size_t match_len) {
  if (pos + match_len <= dst.size()) {
    dst.insert(dst.end(), dst.begin() + static_cast<ptrdiff_t>(pos),
               dst.begin() + static_cast<ptrdiff_t>(pos + match_len));
    return;
  }
  while (match_len > 0) {
    size_t dist = dst.size() - pos;
    if (match_len < dist) dist = match_len;
    const size_t base = pos;
    for (size_t i = 0; i < dist; ++i) dst.push_back(dst[base + i]);
    pos += dist;
    match_len -= dist;
  }
}

}  // namespace

uint64_t ReadControlVarUint(const uint8_t* src, int64_t* cursor, bool* ok) {
  uint64_t r = 0;
  while (*cursor >= 0) {
    const uint8_t v = src[*cursor];
    --*cursor;
    r = (r << 7) | (v & 0x7F);
    if (v & 0x80) return r;
  }
  *ok = false;
  return 0;
}

bool DecompressIguanaLZ(Bytes& dst,
                        const IguanaStream streams_in[kStreamCount]) {
  StreamReader s[kStreamCount];
  for (int i = 0; i < kStreamCount; ++i) {
    s[i].data = streams_in[i].data;
    s[i].size = streams_in[i].size;
  }
  StreamReader& tok = s[0];
  StreamReader& o16 = s[1];
  StreamReader& o24 = s[2];
  StreamReader& vll = s[3];
  StreamReader& vml = s[4];
  StreamReader& lit = s[5];

  bool ok = true;
  int64_t last_offs = 0;
  while (!tok.Empty()) {
    int64_t match_len = 0;
    const uint8_t token = tok.U8(&ok);
    if (!ok) return false;

    if (token >= 32) {
      int64_t lit_len = token & kMaxShortLitLen;
      if (lit_len == kMaxShortLitLen) {
        lit_len = vll.VarUint(&ok) + kMaxShortLitLen;
        if (!ok) return false;
      }
      if (lit_len > 0) {
        const uint8_t* p = lit.Sequence(static_cast<size_t>(lit_len), &ok);
        if (!ok) return false;
        dst.insert(dst.end(), p, p + lit_len);
      }
      if ((token & 0x80) == 0) {
        last_offs = -static_cast<int64_t>(o16.U16(&ok));
        if (!ok) return false;
      }
      match_len = (token >> kLiteralLenBits) & kMaxShortMatchLen;
      if (match_len == kMaxShortMatchLen) {
        match_len = vml.VarUint(&ok) + kMaxShortMatchLen;
        if (!ok) return false;
      }
    } else if (token < kLastLongOffset) {
      match_len = static_cast<int64_t>(token) + kMMLongOffsets;
      last_offs = -static_cast<int64_t>(o24.U24(&ok));
      if (!ok) return false;
    } else {
      match_len = vml.VarUint(&ok) + kLastLongOffset + kMMLongOffsets;
      if (!ok) return false;
      last_offs = -static_cast<int64_t>(o24.U24(&ok));
      if (!ok) return false;
    }

    if (match_len > 0) {
      const int64_t match = static_cast<int64_t>(dst.size()) + last_offs;
      if (match < 0 || match > static_cast<int64_t>(dst.size())) return false;
      WildCopy(dst, static_cast<size_t>(match), static_cast<size_t>(match_len));
    }
  }

  const size_t rem = lit.Remaining();
  if (rem > 0) {
    const uint8_t* p = lit.Sequence(rem, &ok);
    if (!ok) return false;
    dst.insert(dst.end(), p, p + rem);
  }
  return true;
}

// ------------------------------ container

std::vector<uint8_t> Compress(const uint8_t* data, size_t size) {
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

    for (int i = 0; i < kStreamCount; ++i) {
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

    if (total + kStreamCount + 1 >= static_cast<int64_t>(size)) {
      cw.Command(kCmdCopyRaw);
      cw.VarUint(size);
      dst.assign(data, data + size);
    } else {
      cw.Command(kCmdDecodeIguana);
      cw.VarUint(hdr);
      for (int i = 0; i < kStreamCount; ++i) {
        cw.VarUint(ustreams[i].size());
      }
      for (int i = 0; i < kStreamCount; ++i) {
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

bool DecompressScalar(const uint8_t* src, size_t src_size, Bytes& out) {
  if (src_size == 0) return false;
  bool ok = true;
  int64_t ctrl = static_cast<int64_t>(src_size) - 1;
  const uint64_t uncompressed_len = ReadControlVarUint(src, &ctrl, &ok);
  if (!ok) return false;
  out.clear();
  if (uncompressed_len == 0) return true;

  uint64_t data_cursor = 0;
  std::vector<Bytes> ent_bufs;

  for (;;) {
    if (ctrl < 0) return false;
    const uint8_t cmd = src[ctrl];
    --ctrl;

    switch (cmd & kCommandMask) {
      case kCmdCopyRaw: {
        const uint64_t n = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || data_cursor + n > src_size) return false;
        out.insert(out.end(), src + data_cursor, src + data_cursor + n);
        data_cursor += n;
        break;
      }
      case kCmdDecodeANS32: {
        const uint64_t lu = ReadControlVarUint(src, &ctrl, &ok);
        const uint64_t lc = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok || data_cursor + lc > src_size) return false;
        const size_t out_pos = out.size();
        out.resize(out_pos + static_cast<size_t>(lu));
        if (!Ans32DecodeScalar(src + data_cursor, static_cast<size_t>(lc),
                               out.data() + out_pos, static_cast<size_t>(lu))) {
          return false;
        }
        data_cursor += lc;
        break;
      }
      case kCmdDecodeIguana: {
        const uint64_t hdr = ReadControlVarUint(src, &ctrl, &ok);
        if (!ok) return false;
        IguanaStream streams[kStreamCount];
        uint64_t ulens[kStreamCount];
        for (int i = 0; i < kStreamCount; ++i) {
          ulens[i] = ReadControlVarUint(src, &ctrl, &ok);
          if (!ok) return false;
        }
        for (int i = 0; i < kStreamCount; ++i) {
          const int em = static_cast<int>((hdr >> (i * 4)) & 0xF);
          if (em == 0) {
            if (data_cursor + ulens[i] > src_size) return false;
            streams[i].data = src + data_cursor;
            streams[i].size = static_cast<size_t>(ulens[i]);
            data_cursor += ulens[i];
          } else if (em == 1) {  // EntropyANS32
            const uint64_t clen = ReadControlVarUint(src, &ctrl, &ok);
            if (!ok || data_cursor + clen > src_size) return false;
            ent_bufs.emplace_back();
            Bytes& b = ent_bufs.back();
            b.resize(static_cast<size_t>(ulens[i]));
            if (!Ans32DecodeScalar(src + data_cursor, static_cast<size_t>(clen),
                                   b.data(), static_cast<size_t>(ulens[i]))) {
              return false;
            }
            streams[i].data = b.data();
            streams[i].size = b.size();
            data_cursor += clen;
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
