// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIGHWAY_HWY_IO_IO_SPAN_H_
#define HIGHWAY_HWY_IO_IO_SPAN_H_

// Spans of memory for scatter/gather I/O, and per-worker scratch buffers.
// Freestanding: does not require OS file support.

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/status.h"

namespace hwy {

// Compatible with Linux iovec.
struct IOSpan {
  void* mem;
  size_t bytes;
};

class FileView;
class IOBatch;

// Non-owning view of `IOSpan`s that computes and caches the total byte count
// once (avoiding repeated scans across `IOBatch`, `FileView`, and `File`).
class IOSpans {
 public:
  IOSpans() = default;
  IOSpans(Span<const IOSpan> spans)  // NOLINT
      : spans_(spans), total_bytes_(ComputeTotalBytes(spans)) {}

  Span<const IOSpan> Spans() const noexcept { return spans_; }
  const IOSpan* data() const noexcept { return spans_.data(); }
  size_t size() const noexcept { return spans_.size(); }
  bool empty() const noexcept { return spans_.empty(); }
  const IOSpan* begin() const noexcept { return spans_.begin(); }
  const IOSpan* end() const noexcept { return spans_.end(); }
  const IOSpan& operator[](size_t index) const {
    HWY_DASSERT(index < size());
    return spans_[index];
  }

  bool IsValid() const noexcept { return total_bytes_ != kInvalidBytes; }
  uint64_t TotalBytes() const {
    HWY_DASSERT(IsValid());
    return total_bytes_;
  }

  // Returns `Status(total_bytes)` if valid and `offset + total_bytes` fits in
  // `int64_t` (`off_t`) and `Status::kMaxPayload`, else `Status::kArgument`.
  Status ValidateAt(uint64_t offset) const {
    if (!IsValid() || !SumFits<int64_t>(offset, total_bytes_)) {
      return Status::kArgument;
    }
    return Status(total_bytes_);
  }

 private:
  friend class FileView;
  friend class IOBatch;

  // O(1) constructor when the caller already tracked the byte sum. Private so
  // external callers cannot violate `total_bytes_ == ComputeTotalBytes(spans)`
  // in builds without `HWY_DASSERT`.
  IOSpans(Span<const IOSpan> spans, uint64_t total_bytes)
      : spans_(spans),
        total_bytes_(total_bytes <= Status::kMaxPayload ? total_bytes
                                                        : kInvalidBytes) {
    HWY_DASSERT(total_bytes_ == ComputeTotalBytes(spans));
  }

  static constexpr uint64_t kInvalidBytes = ~uint64_t{0};

  static uint64_t ComputeTotalBytes(Span<const IOSpan> spans) {
    uint64_t total = 0;
    for (const IOSpan& span : spans) {
      if (!SumWithin(total, span.bytes, Status::kMaxPayload)) {
        return kInvalidBytes;
      }
      total = SafeAdd<uint64_t>(total, span.bytes);
    }
    return total;
  }

  Span<const IOSpan> spans_;
  uint64_t total_bytes_ = 0;
};

// Per-worker reusable scratch buffer for `File::ReadSpans` and `IOBatch::Read`
// on backends without native scatter reads (e.g. Windows, `MemoryFile`, or
// custom `FileSystem` backends). Move-only and lazily sized on demand
// (retaining the high-water-mark allocation across calls), preventing
// accidental undersized manual allocations. Not thread-safe: pass a separate
// `IOWorkerBuf` on each worker thread.
class IOWorkerBuf {
 public:
  IOWorkerBuf() = default;
  IOWorkerBuf(const IOWorkerBuf&) = delete;
  IOWorkerBuf& operator=(const IOWorkerBuf&) = delete;
  IOWorkerBuf(IOWorkerBuf&&) noexcept = default;
  IOWorkerBuf& operator=(IOWorkerBuf&&) noexcept = default;

  // Current allocated capacity in bytes (0 until `Ensure` allocates).
  size_t Capacity() const noexcept { return buf_.size(); }

  // Frees any retained buffer.
  void Release() noexcept { buf_.reset(); }

  // For `File` implementations (e.g. `File::ReadSpans` overrides): grows the
  // retained allocation if `Capacity() < bytes` and returns a pointer to at
  // least `bytes` aligned bytes (or `nullptr` if `bytes == 0` when empty, or if
  // allocation fails).
  uint8_t* Ensure(size_t bytes) {
    if (buf_.size() < bytes) {
      buf_.reset();
      buf_ = AllocateAligned<uint8_t>(bytes);
    }
    return buf_.get();
  }

 private:
  AlignedPtr<uint8_t> buf_;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_IO_SPAN_H_
