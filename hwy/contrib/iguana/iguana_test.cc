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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "hwy/contrib/iguana/iguana_detail.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_IGUANA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/iguana/iguana_test.cc"  // NOLINT
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/iguana/iguana.h"
#include "hwy/contrib/iguana/iguana-inl.h"
#include "hwy/tests/test_util-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

std::vector<uint8_t> MakeData(size_t n, uint64_t seed, int mode) {
  RandomState rng(seed | 1);
  std::vector<uint8_t> v(n);
  if (mode == 0) {
    for (auto& x : v) x = static_cast<uint8_t>(Random64(&rng));
  } else if (mode == 1) {  // skewed
    for (auto& x : v) {
      uint32_t a = 0;
      for (int k = 0; k < 3; ++k)
        a += static_cast<uint32_t>(Random64(&rng) & 0x3F);
      x = static_cast<uint8_t>(a);
    }
  } else {  // repetitive words (compresses well)
    static const char* const w[] = {"the ",   "quick ", "brown ", "fox ",
                                    "jumps ", "over ",  "lazy ",  "dog "};
    std::string s;
    while (s.size() < n) s += w[Random64(&rng) & 7];
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(s[i]);
  }
  return v;
}

// One workspace shared by every call below, which is also what a real caller
// would do: if Compress() failed to reset its hash chains, the second and
// later blocks would decode to something other than their input.
hwy::iguana::IguanaWorkspace* SharedWorkspace() {
  static hwy::iguana::IguanaWorkspace ws;
  return &ws;
}

// A real pool, so that the tests actually exercise the chunk-parallel paths
// rather than only the single-worker fallbacks. Inputs below are mostly
// smaller than one chunk; TestRoundTripSizes covers the multi-chunk case.
hwy::ThreadPool* SharedPool() {
  static hwy::ThreadPool pool(hwy::ThreadPool::NumThreadsFromCores());
  return &pool;
}

// Stands in for the vector-based Compress() overload the API used to have.
// Most tests below care about the resulting block, not about sizing `dst`.
size_t CompressVec(Span<const uint8_t> src, AlignedVector<uint8_t>& out) {
  const size_t max_len = hwy::iguana::MaxCompressedSize(src.size());
  if (max_len == 0) {
    out.clear();
    return 0;
  }
  out.resize(max_len);
  const size_t written =
      hwy::iguana::Compress(src, Span<uint8_t>(out.data(), out.size()),
                            *SharedWorkspace(), *SharedPool());
  out.resize(written);
  return written;
}

// Stand-ins for the vector-based Decompress() overloads the API used to have:
// size `out` from the block header, then decode into it. Returning bool keeps
// the assertions below readable, and clearing `out` on rejection keeps the
// "a rejected block leaves no partial output" property testable.
template <class DecompressFn>
bool DecompressVec(Span<const uint8_t> src, AlignedVector<uint8_t>& out,
                   const DecompressFn& decompress) {
  size_t len = 0;
  if (!hwy::iguana::DecompressedSize(src, &len)) {
    out.clear();
    return false;
  }
  out.resize(len);
  if (decompress(src, Span<uint8_t>(out.data(), out.size())) ==
      hwy::iguana::kDecompressFailed) {
    out.clear();
    return false;
  }
  return true;
}

bool DecompressScalarVec(Span<const uint8_t> src, AlignedVector<uint8_t>& out) {
  return DecompressVec(src, out, [](Span<const uint8_t> s, Span<uint8_t> d) {
    return hwy::iguana::DecompressScalar(s, d, *SharedWorkspace(),
                                         *SharedPool());
  });
}

bool DecompressStaticVec(Span<const uint8_t> src, AlignedVector<uint8_t>& out) {
  return DecompressVec(src, out, [](Span<const uint8_t> s, Span<uint8_t> d) {
    return DecompressStatic(s, d, *SharedWorkspace(), *SharedPool());
  });
}

void RoundTrip(const std::vector<uint8_t>& data) {
  const Span<const uint8_t> src_span(data.data(), data.size());

  AlignedVector<uint8_t> comp_buf(hwy::iguana::MaxCompressedSize(data.size()));
  const size_t comp_size = hwy::iguana::Compress(
      src_span, Span<uint8_t>(comp_buf.data(), comp_buf.size()),
      *SharedWorkspace(), *SharedPool());
  HWY_ASSERT(comp_size > 0);
  const Span<const uint8_t> comp_span(comp_buf.data(), comp_size);

  size_t expected_dec_size = 0;
  HWY_ASSERT(hwy::iguana::DecompressedSize(comp_span, &expected_dec_size));
  HWY_ASSERT(expected_dec_size == data.size());

  AlignedVector<uint8_t> dec_buf(data.size());
  const Span<uint8_t> dec_span(dec_buf.data(), dec_buf.size());
  HWY_ASSERT(DecompressStatic(comp_span, dec_span, *SharedWorkspace(),
                              *SharedPool()) == data.size());
  HWY_ASSERT(data.empty() ||
             memcmp(dec_buf.data(), data.data(), data.size()) == 0);

  AlignedVector<uint8_t> dec_scalar_buf(data.size());
  HWY_ASSERT(hwy::iguana::DecompressScalar(
                 comp_span,
                 Span<uint8_t>(dec_scalar_buf.data(), dec_scalar_buf.size()),
                 *SharedWorkspace(), *SharedPool()) == data.size());
  HWY_ASSERT(dec_scalar_buf == dec_buf);

  AlignedVector<uint8_t> dec_dyn_buf(data.size());
  HWY_ASSERT(hwy::iguana::Decompress(
                 comp_span,
                 Span<uint8_t>(dec_dyn_buf.data(), dec_dyn_buf.size()),
                 *SharedWorkspace(), *SharedPool()) == data.size());
  HWY_ASSERT(dec_dyn_buf == dec_buf);

  // Decoding the same block without a pool must give the same bytes: the
  // chunk-parallel path and the serial one are two readings of one format.
  hwy::ThreadPool no_pool(0);
  AlignedVector<uint8_t> dec_serial_buf(data.size());
  HWY_ASSERT(hwy::iguana::Decompress(
                 comp_span,
                 Span<uint8_t>(dec_serial_buf.data(), dec_serial_buf.size()),
                 *SharedWorkspace(), no_pool) == data.size());
  HWY_ASSERT(dec_serial_buf == dec_buf);

  // A fresh workspace must produce exactly the same block as the reused one,
  // and so must compressing without a pool: chunking is a property of the
  // format, not of how many workers happened to be available.
  hwy::iguana::IguanaWorkspace fresh;
  AlignedVector<uint8_t> comp_buf2(hwy::iguana::MaxCompressedSize(data.size()));
  const size_t comp_size2 = hwy::iguana::Compress(
      src_span, Span<uint8_t>(comp_buf2.data(), comp_buf2.size()), fresh,
      no_pool);
  HWY_ASSERT(comp_size2 == comp_size);
  HWY_ASSERT(memcmp(comp_buf2.data(), comp_buf.data(), comp_size) == 0);
}

void TestWorkspaceReserve() {
  constexpr size_t kWorkers = 4;
  hwy::iguana::IguanaWorkspace ws;
  HWY_ASSERT(ws.Capacity() == 0);
  HWY_ASSERT(ws.Reserve(1 << 20, kWorkers));
  const size_t cap = ws.Capacity();
  HWY_ASSERT(cap >= hwy::iguana::IguanaWorkspace::SizeFor(1 << 20, kWorkers));
  // Reserving no more than what is already there must not reallocate.
  HWY_ASSERT(ws.Reserve(1 << 19, kWorkers));
  HWY_ASSERT(ws.Capacity() == cap);
  // The per-worker regions are sized for one chunk, so they do not grow with
  // the input; only the staging arena does, at roughly 1x.
  HWY_ASSERT(hwy::iguana::IguanaWorkspace::SizeFor(2 << 20, kWorkers) -
                 hwy::iguana::IguanaWorkspace::SizeFor(1 << 20, kWorkers) <
             2 << 20);
  // More workers cost more, and fewer cost less.
  HWY_ASSERT(hwy::iguana::IguanaWorkspace::SizeFor(1 << 20, 1) <
             hwy::iguana::IguanaWorkspace::SizeFor(1 << 20, kWorkers));
}

void TestRoundTripSizes() {
  // The sizes from kChunkSize on exercise the multi-chunk paths: exactly one
  // chunk, one chunk plus a one-byte remainder, an exact multiple, and a
  // partial last chunk.
  const size_t kSizes[] = {0,      1,      10,     31,     36,
                           37,     64,     100,    1000,   5000,
                           40000,  65536,  200000, 262144, 262145,
                           524288, 700000};
  for (size_t idx = 0; idx < sizeof(kSizes) / sizeof(kSizes[0]); ++idx) {
    const size_t n = kSizes[idx];
    RoundTrip(MakeData(n, n * 7 + 1, 0));
    RoundTrip(MakeData(n, n * 11 + 2, 1));
    RoundTrip(MakeData(n, n * 13 + 3, 2));
  }
}

void TestRoundTripStructure() {
  // A far-back (>64 KiB) repeat forces the 24-bit offset path.
  std::vector<uint8_t> a = MakeData(200000, 4242, 0);
  std::vector<uint8_t> v = a;
  v.insert(v.end(), a.begin(), a.begin() + 6000);
  v.insert(v.end(), a.begin() + 90000, a.begin() + 98000);
  RoundTrip(v);

  // Highly compressible.
  RoundTrip(std::vector<uint8_t>(100000, 0x5A));

  // A short quotable string re-sliced (exercises the raw-copy fallback).
  const std::string s =
      "this is a short string that we will re-slice for small test-cases";
  for (size_t i = 0; i < s.size(); ++i) {
    RoundTrip(
        std::vector<uint8_t>(s.begin() + static_cast<ptrdiff_t>(i), s.end()));
  }
}

// Malformed or truncated inputs must be rejected: no crash, and no allocation
// driven by an attacker-controlled length.
void TestRejectsMalformed() {
  AlignedVector<uint8_t> out;

  // No header at all.
  HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(), out));
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(), out));

  // A hand-built block whose very first LZ token is a match that reuses the
  // "previous offset" (bit 0x80) when there has not been one yet - the offset
  // would be zero, i.e. read from before the start of the output.
  // Layout: [tokens payload][ulen 5..0][hdr][cmd][uncompressed_len], the
  // control bytes read backwards from the end.
  {
    const uint8_t first_match_reuse[] = {
        0xA0,  // token: reuse offs, len 4
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x81,                                  // ulens: 0,0,0,0,0,1 (tokens)
        0x80,                                  // hdr: all streams raw
        0x80 | hwy::iguana::kCmdDecodeIguana,  // command (last)
        0x84,                                  // uncompressed_len = 4
    };
    const Span<const uint8_t> span(first_match_reuse,
                                   sizeof(first_match_reuse));
    HWY_ASSERT(!DecompressScalarVec(span, out));
    HWY_ASSERT(!DecompressStaticVec(span, out));
  }

  // Every truncation of a valid block is rejected or decodes a shorter output.
  const std::vector<uint8_t> data = MakeData(5000, 777, 2);
  AlignedVector<uint8_t> comp;
  HWY_ASSERT(CompressVec(Span<const uint8_t>(data), comp) > 0);
  for (size_t n = 0; n < comp.size(); ++n) {
    AlignedVector<uint8_t> scalar;
    AlignedVector<uint8_t> simd;
    const Span<const uint8_t> trunc(comp.data(), n);
    const bool ok_scalar = DecompressScalarVec(trunc, scalar);
    const bool ok_simd = DecompressStaticVec(trunc, simd);
    HWY_ASSERT(ok_scalar == ok_simd);
    if (ok_scalar) {
      // A truncated block may decode, but never to more than the original.
      HWY_ASSERT(simd == scalar);
      HWY_ASSERT(scalar.size() <= data.size());
    }
  }

  // Single-byte mutations: accepted output is still bounded by the cap.
  for (size_t i = 0; i < comp.size(); i += 5) {
    AlignedVector<uint8_t> mutated = comp;
    mutated[i] ^= 0xFF;
    AlignedVector<uint8_t> dec;
    if (DecompressScalarVec(Span<const uint8_t>(mutated), dec)) {
      HWY_ASSERT(dec.size() <= (size_t{1} << 30));
    }
  }
}

// ------------------------------ Security

// The decoder parses untrusted input: every length, offset and command byte in
// a block comes from the attacker. These tests pin what must hold regardless of
// the bytes: malformed input is rejected (never decoded "best effort"), no
// allocation is driven by an attacker-controlled length, and the scalar and
// SIMD paths agree on what they accept, what they reject, and what they
// produce.

// Appends a control varint the way the encoder does: 7 bits per byte, most
// significant group first, with a stop bit (0x80) on the last byte.
void AppendCtrlVarUint(std::vector<uint8_t>& ctrl, uint64_t v) {
  int groups = 0;
  for (uint64_t t = v; t != 0; t >>= 7) ++groups;
  for (int i = groups; i >= 0; --i) {
    uint32_t x = static_cast<uint32_t>(v >> (i * 7)) & 0x7F;
    if (i == 0) x |= 0x80;
    ctrl.push_back(static_cast<uint8_t>(x));
  }
}

// Builds a DecodeIguana block around the six (raw) streams, so a test can state
// the exact malformed shape it means instead of guessing at encoder output. The
// control section is written last and read backwards by the decoder.
std::vector<uint8_t> MakeIguanaBlock(const std::vector<uint8_t> streams[6],
                                     const uint64_t ulens[6],
                                     uint64_t uncompressed_len,
                                     int ansi_mode_stream = -1) {
  std::vector<uint8_t> out;
  for (int i = 0; i < 6; ++i) {
    out.insert(out.end(), streams[i].begin(), streams[i].end());
  }
  uint64_t hdr = 0;
  if (ansi_mode_stream >= 0) {
    hdr |= uint64_t{2} << (ansi_mode_stream * 4);  // ANS1: not implemented
  }

  std::vector<uint8_t> ctrl;
  // Push in the order the decoder consumes (it reads the block backwards), so
  // that the reversed copy below matches what Compress() writes.
  AppendCtrlVarUint(ctrl, uncompressed_len);
  ctrl.push_back(static_cast<uint8_t>(0x80 | hwy::iguana::kCmdDecodeIguana));
  AppendCtrlVarUint(ctrl, hdr);
  for (int i = 0; i < 6; ++i) AppendCtrlVarUint(ctrl, ulens[i]);
  // The decoder reads the control bytes backwards, so reverse them here.
  for (size_t i = ctrl.size(); i-- > 0;) out.push_back(ctrl[i]);
  return out;
}

// A block that declares more output than kMaxUncompressedSize must be rejected
// before anything is allocated: a ~30-byte input asking for more than 1 GiB.
void TestSecurityZipBomb() {
  std::vector<uint8_t> streams[6];
  uint64_t ulens[6] = {1, 0, 0, 0, 0, 0};
  streams[0].push_back(0xA0);  // one literal-only token
  const uint64_t huge = (uint64_t{1} << 30) + 1;
  const std::vector<uint8_t> bomb = MakeIguanaBlock(streams, ulens, huge);

  AlignedVector<uint8_t> out;
  HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(bomb), out));
  HWY_ASSERT(out.empty());
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(bomb), out));

  // The cap is part of the contract, so keep it visible here.
  HWY_ASSERT(hwy::iguana::kMaxUncompressedSize == (size_t{1} << 30));
}

// Each of the six streams is buffered in full before the LZ stage runs, so the
// bound has to be on their total, not on each one: six streams that are each
// just under the cap would otherwise allocate kStreamCount GiB from a block of
// a few dozen bytes. Mode 1 (ANS32) is what reaches the resize; the decode
// itself never gets to run.
void TestSecurityStreamTotalBomb() {
  std::vector<uint8_t> payload;  // no stream data at all
  uint64_t ulens[6];
  for (int i = 0; i < 6; ++i) ulens[i] = uint64_t{1} << 29;  // 512 MiB each

  uint64_t hdr = 0;
  for (int i = 0; i < 6; ++i) hdr |= uint64_t{1} << (i * 4);  // all ANS32

  std::vector<uint8_t> ctrl;
  AppendCtrlVarUint(ctrl, hwy::iguana::kMaxUncompressedSize);
  ctrl.push_back(static_cast<uint8_t>(0x80 | hwy::iguana::kCmdDecodeIguana));
  AppendCtrlVarUint(ctrl, hdr);
  for (int i = 0; i < 6; ++i) AppendCtrlVarUint(ctrl, ulens[i]);
  std::vector<uint8_t> block(payload);
  for (size_t i = ctrl.size(); i-- > 0;) block.push_back(ctrl[i]);

  AlignedVector<uint8_t> out;
  HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
  HWY_ASSERT(out.empty());
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));
  HWY_ASSERT(out.empty());
}

// A block declaring zero output consists of nothing but that length, so any
// bytes in front of it are unaccounted for and the block must be rejected
// rather than silently reported as "successfully decoded to nothing".
void TestSecurityEmptyBlockIsExact() {
  AlignedVector<uint8_t> out;

  // What Compress() emits for an empty input: a single stop-bit-only varint.
  AlignedVector<uint8_t> empty;
  HWY_ASSERT(CompressVec(Span<const uint8_t>(), empty) == 1);
  HWY_ASSERT(empty.size() == 1);
  HWY_ASSERT(DecompressScalarVec(Span<const uint8_t>(empty), out));
  HWY_ASSERT(out.empty());
  HWY_ASSERT(DecompressStaticVec(Span<const uint8_t>(empty), out));
  HWY_ASSERT(out.empty());

  // The same length byte with unaccounted-for bytes in front of it.
  std::vector<uint8_t> padded = {0xDE, 0xAD, 0xBE, 0xEF, 0x80};
  HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(padded), out));
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(padded), out));
}

// A rejected block must leave `out` empty: callers that forget to check the
// return value must not see half-decoded, attacker-chosen bytes.
void TestSecurityRejectedOutputIsCleared() {
  // A token stream that produces 100 bytes and then hits a NOP, so the block
  // is rejected only after `out` has already been appended to.
  std::vector<uint8_t> streams[6];
  streams[0].push_back(0x87);  // literal-only token: 7 + var_lit_len literals
  streams[0].push_back(0x80);  // NOP -> rejected
  streams[3].push_back(93);    // var_lit_len: 7 + 93 = 100
  streams[5].assign(100, 'A');
  const uint64_t ulens[6] = {2, 0, 0, 1, 0, 100};
  const std::vector<uint8_t> block = MakeIguanaBlock(streams, ulens, 100);

  AlignedVector<uint8_t> out;
  HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
  HWY_ASSERT(out.empty());
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));
  HWY_ASSERT(out.empty());
}

// Malformed containers: unknown commands, unimplemented stream modes, a control
// varint longer than 64 bits, and auxiliary streams left with leftover bytes.
void TestSecurityMalformedContainer() {
  AlignedVector<uint8_t> out;

  // Unknown command byte: only CopyRaw / DecodeIguana / DecodeANS32 exist.
  {
    std::vector<uint8_t> ctrl;
    ctrl.push_back(static_cast<uint8_t>(0x80 | 0x7F));
    AppendCtrlVarUint(ctrl, 1);
    std::vector<uint8_t> block(ctrl.rbegin(), ctrl.rend());
    HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
    HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));
  }

  // Stream mode 2 (ANS1) is declared but not implemented: must be rejected.
  {
    std::vector<uint8_t> streams[6];
    uint64_t ulens[6] = {0, 0, 0, 0, 0, 0};
    ulens[2] = 4;
    streams[2].assign(4, 0);
    const std::vector<uint8_t> block =
        MakeIguanaBlock(streams, ulens, 64, /*ansi_mode_stream=*/2);
    HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
    HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));
  }

  // A control varint of 11 bytes: more bits than fit in 64.
  {
    std::vector<uint8_t> block;
    for (int i = 0; i < 11; ++i) block.push_back(0xFF);
    HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
    HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));
  }

  // Leftover bytes in an auxiliary stream: the token stream is empty, but the
  // 16-bit offset stream still holds two bytes, so the block is inconsistent.
  {
    std::vector<uint8_t> streams[6];
    uint64_t ulens[6] = {0, 2, 0, 0, 0, 0};
    streams[1].assign(2, 0);
    const std::vector<uint8_t> block = MakeIguanaBlock(streams, ulens, 64);
    HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
    HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));
  }
}

// LZ77-level malformed input: a match whose offset points before the output.
void TestSecurityMalformedLZ() {
  AlignedVector<uint8_t> out;

  // tokens = {0x28}: short form, literal length 0, the repeat-offset bit is
  // clear so a new 16-bit offset is read, and it says 65535 while the output is
  // still empty - the match would read before its start.
  std::vector<uint8_t> streams[6];
  streams[0].push_back(0x28);
  streams[1].push_back(0xFF);
  streams[1].push_back(0xFF);
  const uint64_t ulens[6] = {1, 2, 0, 0, 0, 0};
  const std::vector<uint8_t> block = MakeIguanaBlock(streams, ulens, 100);
  HWY_ASSERT(!DecompressScalarVec(Span<const uint8_t>(block), out));
  HWY_ASSERT(out.empty());
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(block), out));

  // A NOP token (0x80) carries neither literals nor a match. A block can be
  // padded with them: it still produces its declared output, so without an
  // explicit rejection we would walk the whole padded token stream for nothing.
  // The control below is the same block with the NOPs removed.
  std::vector<uint8_t> ok_streams[6];
  ok_streams[0].push_back(
      0x87);                    // literal-only token: 7 + var_lit_len literals
  ok_streams[3].push_back(93);  // var_lit_len: 7 + 93 = 100
  ok_streams[5].assign(100, 'A');
  const uint64_t ok_ulens[6] = {1, 0, 0, 1, 0, 100};
  const std::vector<uint8_t> ok_block =
      MakeIguanaBlock(ok_streams, ok_ulens, 100);
  const bool ok_res =
      DecompressScalarVec(Span<const uint8_t>(ok_block), out);
  HWY_ASSERT(ok_res);
  HWY_ASSERT(out.size() == 100 && out[0] == 'A');

  std::vector<uint8_t> nop_streams[6];
  nop_streams[0].push_back(
      0x87);  // literal-only token: 7 + var_lit_len literals
  nop_streams[0].push_back(0x80);  // NOP
  nop_streams[0].push_back(0x80);  // NOP
  nop_streams[3].push_back(93);    // var_lit_len: 7 + 93 = 100
  nop_streams[5].assign(100, 'A');
  const uint64_t nop_ulens[6] = {3, 0, 0, 1, 0, 100};
  const std::vector<uint8_t> nop_block =
      MakeIguanaBlock(nop_streams, nop_ulens, 100);
  // Both paths must reject it, and leave no partial output behind.
  const bool nop_res =
      DecompressScalarVec(Span<const uint8_t>(nop_block), out);
  HWY_ASSERT(!nop_res);
  HWY_ASSERT(out.empty());
  HWY_ASSERT(!DecompressStaticVec(Span<const uint8_t>(nop_block), out));
  HWY_ASSERT(out.empty());
}

// Deterministic mutations of valid blocks: the same seeds every run, so any
// failure reproduces. This is the property that has to survive adversarial
// input - never crash, never allocate beyond the cap, and reject exactly what
// the SIMD path rejects.
void TestSecurityMutationSweep() {
  RandomState rng(0x5EC0DEULL);
  for (int round = 0; round < 64; ++round) {
    const size_t n = 1 + static_cast<size_t>(Random64(&rng) % 4000);
    const std::vector<uint8_t> data =
        MakeData(n, Random64(&rng) & 0xFFFF, round % 3);
    AlignedVector<uint8_t> comp;
    if (CompressVec(Span<const uint8_t>(data), comp) == 0) continue;

    AlignedVector<uint8_t> mutated = comp;
    const int mutations = 1 + static_cast<int>(Random64(&rng) % 4);
    for (int m = 0; m < mutations; ++m) {
      const size_t pos = static_cast<size_t>(Random64(&rng) % mutated.size());
      switch (Random64(&rng) % 3) {
        case 0:
          mutated[pos] =
              static_cast<uint8_t>(mutated[pos] ^ (1u << (Random64(&rng) & 7)));
          break;
        case 1:
          mutated[pos] = static_cast<uint8_t>(Random64(&rng));
          break;
        default:
          mutated.resize(pos);  // truncate
          break;
      }
    }

    AlignedVector<uint8_t> scalar;
    AlignedVector<uint8_t> simd;
    const bool ok_scalar =
        DecompressScalarVec(Span<const uint8_t>(mutated), scalar);
    const bool ok_simd =
        DecompressStaticVec(Span<const uint8_t>(mutated), simd);
    HWY_ASSERT(ok_scalar == ok_simd);
    if (ok_scalar) {
      HWY_ASSERT(scalar == simd);
      HWY_ASSERT(scalar.size() <= hwy::iguana::kMaxUncompressedSize);
    }
  }
}

// Garbage that the encoder would never produce. Random bytes almost always
// declare a header far above the cap, so this also covers "reject cheaply".
void TestSecurityRandomInput() {
  RandomState rng(0xC0FFEEULL);
  std::vector<uint8_t> bytes(1024);
  for (int round = 0; round < 256; ++round) {
    const size_t n = static_cast<size_t>(Random64(&rng) % bytes.size());
    for (size_t i = 0; i < n; ++i) {
      bytes[i] = static_cast<uint8_t>(Random64(&rng));
    }
    AlignedVector<uint8_t> scalar;
    AlignedVector<uint8_t> simd;
    const Span<const uint8_t> span(bytes.data(), n);
    const bool ok_scalar = DecompressScalarVec(span, scalar);
    const bool ok_simd = DecompressStaticVec(span, simd);
    HWY_ASSERT(ok_scalar == ok_simd);
    if (ok_scalar) {
      HWY_ASSERT(scalar == simd);
      HWY_ASSERT(scalar.size() <= hwy::iguana::kMaxUncompressedSize);
    }
  }
}

// One token can carry at most kMaxEncodableLitLen literals, so longer literal
// runs have to be split across several literal-only tokens. The trigger is a
// maximal-length LFSR sequence of degree 24 rendered as bytes over a
// two-symbol alphabet: every 24-bit window is unique, so the matcher cannot
// find any match and the entire input becomes one literal run, while the
// skewed alphabet still lets the entropy stage win - which is what selects
// the LZ path in the first place. Decoding only round-trips if the split is
// correct, so this is the only coverage of that path.
void TestLongLiteralRunSplit() {
  // 254^3 - 1: the most literals a single token can carry. This mirrors
  // kMaxEncodableLitLen in iguana.cc, which is internal to that file.
  constexpr size_t kMaxTokenLiterals = 254 * 254 * 254 - 1;
  // x^24 + x^4 + x^3 + x + 1 is primitive, so the period is 2^24 - 1, which is
  // longer than one token can carry.
  uint32_t lfsr = 1;
  const size_t num_bytes = (size_t{1} << 24) + 4096;
  HWY_ASSERT(num_bytes > kMaxTokenLiterals);
  std::vector<uint8_t> data;
  data.reserve(num_bytes);
  for (size_t i = 0; i < num_bytes; ++i) {
    data.push_back(static_cast<uint8_t>('0' + (lfsr & 1)));
    lfsr = (lfsr >> 1) |
           (((lfsr ^ (lfsr >> 1) ^ (lfsr >> 3) ^ (lfsr >> 4)) & 1) << 23);
  }

  AlignedVector<uint8_t> comp;
  HWY_ASSERT(CompressVec(Span<const uint8_t>(data), comp) > 0);
  // If the raw path had won, the LZ path (and the splitting inside it) would
  // never have run, so the test would prove nothing.
  HWY_ASSERT(comp.size() * 4 < data.size());

  AlignedVector<uint8_t> dec;
  HWY_ASSERT(DecompressStaticVec(Span<const uint8_t>(comp), dec));
  HWY_ASSERT(dec.size() == data.size() &&
             memcmp(dec.data(), data.data(), data.size()) == 0);
  AlignedVector<uint8_t> dec_scalar;
  HWY_ASSERT(
      DecompressScalarVec(Span<const uint8_t>(comp), dec_scalar));
  HWY_ASSERT(dec_scalar == dec);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(IguanaTest);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestWorkspaceReserve);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestRoundTripSizes);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestRoundTripStructure);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestRejectsMalformed);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityZipBomb);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityStreamTotalBomb);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityEmptyBlockIsExact);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityRejectedOutputIsCleared);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityMalformedContainer);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityMalformedLZ);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityMutationSweep);
HWY_EXPORT_AND_TEST_P(IguanaTest, TestSecurityRandomInput);
// Compressing 16 MB is too slow to repeat for every target, and the code it
// covers (the encoder's token splitting) is target-independent, so run it
// on one target only.
HWY_EXPORT_AND_TEST_BEST_P(IguanaTest, TestLongLiteralRunSplit);
HWY_AFTER_TEST();
}  // namespace hwy
#endif
