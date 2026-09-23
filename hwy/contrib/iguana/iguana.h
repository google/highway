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

#include "hwy/aligned_allocator.h"  // AlignedVector, Span
#include "hwy/base.h"            // HWY_INLINE, HWY_RESTRICT
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway_export.h"  // HWY_CONTRIB_DLLEXPORT

namespace hwy {
namespace iguana {

// Returns an upper bound on the compressed size for an input of `size` bytes,
// or 0 if `size` exceeds the maximum supported block size (1 GiB).
HWY_INLINE constexpr size_t MaxCompressedSize(size_t size) {
  return size <= (size_t{1} << 30) ? size + 128 : 0;
}

// Returned by the Decompress* functions in place of a length when the input is
// malformed. Deliberately not 0, because a well-formed block may decode to
// zero bytes and callers have to be able to tell that apart from a rejection.
HWY_INLINE_VAR constexpr size_t kDecompressFailed = ~size_t{0};

// Parses the uncompressed size from the block header into `*out_size`.
// Returns false if the header is malformed or exceeds the 1 GiB cap.
HWY_CONTRIB_DLLEXPORT bool DecompressedSize(Span<const uint8_t> src,
                                            size_t* HWY_RESTRICT out_size);

// All of Compress and Decompress's scratch memory: one private region per
// worker (for Compress: the match-finder hash chains, the six LZ streams and
// the rANS encoder's working buffer; for Decompress: the decoded ANS32 stream
// buffers) plus a shared arena (for Compress: per-chunk outputs; for
// Decompress: parsed CommandDesc table). Reusing one instance across a series
// of blocks makes both Compress and Decompress allocation-free after the first
// call, and keeps the buffers warm in cache.
//
// One instance is shared by all workers of a single Compress or Decompress call
// - each writes only its own region - but an instance must not be used by two
// concurrent calls.
class IguanaWorkspace {
 public:
  IguanaWorkspace() = default;

  // Ensures the workspace can compress or decompress inputs of up to `max_size`
  // bytes with `num_workers` workers (i.e. `pool.NumWorkers()`), reallocating
  // if it is currently smaller. Never shrinks. Returns false if the allocation
  // failed, leaving the previous contents usable. Compress() and Decompress()
  // call this themselves; call it directly only to keep the allocation out of a
  // timed or real-time section.
  HWY_CONTRIB_DLLEXPORT bool Reserve(size_t max_size, size_t num_workers);

  // Usable bytes Reserve(max_size, num_workers) sets up:
  // `num_workers` regions of a few MiB each (dominated by the 512 KiB
  // hash-chain table and stream buffers, and independent of `max_size` because
  // each worker only ever sees one chunk) plus roughly `max_size` for the
  // staging arena. The buffers are sized from worst-case bounds; untouched
  // pages stay uncommitted, so the resident set tracks actual use rather than
  // this number. Reserve() additionally allocates up to one huge page beyond
  // this to align the base.
  static HWY_CONTRIB_DLLEXPORT size_t SizeFor(size_t max_size,
                                              size_t num_workers);

  // Bytes currently usable, or 0 before the first successful Reserve().
  size_t Capacity() const { return capacity_; }

  // Start of the reserved region, or null before the first successful
  // Reserve(). Huge-page aligned, which the internal layout relies on. This is
  // plumbing for Compress() and Decompress(); there is nothing here for
  // callers.
  uint8_t* Memory() const { return base_; }

 private:
  AlignedFreeUniquePtr<uint8_t[]> mem_;
  // Start of the usable region inside `mem_`, rounded up so the first worker's
  // hash-chain table begins on a huge-page boundary.
  uint8_t* base_ = nullptr;
  size_t capacity_ = 0;
};

// Compresses `src` into a complete Iguana block (EncodingIguana / ANS32) in
// pre-allocated `dst`, which should have size >= MaxCompressedSize(src.size()).
// Returns the number of bytes written (always >= 1 on success, even for empty
// `src`), or 0 on failure.
//
// The input is split into independently coded chunks that `pool` compresses
// concurrently; the result is a single ordinary block that any Iguana decoder
// reads, including a serial one. Passing a pool with no extra workers is
// valid and simply compresses the chunks in order.
//
// `ws` holds every buffer the encoder needs; it is grown on demand and may be
// reused across calls (see IguanaWorkspace).
HWY_CONTRIB_DLLEXPORT size_t Compress(Span<const uint8_t> src,
                                      Span<uint8_t> dst, IguanaWorkspace& ws,
                                      ThreadPool& pool);

// Decompresses a block produced by Compress into pre-allocated `dst`, which
// must have size >= the block's DecompressedSize. Returns the number of bytes
// written, or kDecompressFailed on malformed input.
//
// Blocks this Compress produced are decoded chunk-parallel on `pool` using
// per-worker regions in `ws`. Blocks from an encoder that chunks differently
// still decode correctly, just serially.
HWY_CONTRIB_DLLEXPORT size_t DecompressScalar(Span<const uint8_t> src,
                                              Span<uint8_t> dst,
                                              IguanaWorkspace& ws,
                                              ThreadPool& pool);

// Same as DecompressScalar, but dispatches at run time to the best available
// target (the SIMD decoders in iguana-inl.h). Prefer this unless you
// specifically need the scalar reference; the two produce identical output.
HWY_CONTRIB_DLLEXPORT size_t Decompress(Span<const uint8_t> src,
                                        Span<uint8_t> dst, IguanaWorkspace& ws,
                                        ThreadPool& pool);

}  // namespace iguana
}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_IGUANA_IGUANA_H_
