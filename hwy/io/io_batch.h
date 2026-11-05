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

#ifndef HIGHWAY_HWY_IO_IO_BATCH_H_
#define HIGHWAY_HWY_IO_IO_BATCH_H_

// Builder for scatter/gather `File::ReadSpans` / `WriteSpans`. Freestanding:
// does not require OS file support.

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/file.h"
#include "hwy/io/io_span.h"
#include "hwy/status.h"

namespace hwy {

// Builder for `File::ReadSpans` and `File::WriteSpans` (which use
// `preadv`/`pwritev` on Linux/BSD and fall back to `File::Read`/`File::Write`
// elsewhere). For discontiguous memory (e.g. strided rows with padding, or a
// frame header plus payload), issuing one syscall per span is expensive;
// `preadv`/`pwritev` combine up to `IOV_MAX` (1024) spans per syscall.
// The file data must be contiguous starting from `IOBatch::offset_`, because
// `preadv`/`pwritev` do not support per-`IOSpan` offsets.
//
// `IOBatch` also serves as the request record reserved for future asynchronous
// I/O: its starting `Offset()`, contiguous `IOSpan` (`iovec`) array, read vs.
// write-only direction, and 64-bit `UserTag()` map 1:1 onto a Linux `io_uring`
// SQE (`IORING_OP_READV` / `IORING_OP_WRITEV` keyed by `user_data`) or Windows
// `IORING` / IOCP submission, enabling a future `File::Submit(const IOBatch&)`
// + `WaitCompletions` API without changing `IOBatch`.
class IOBatch {
 public:
  // `user_tag` is not interpreted; callers can use it to identify the batch.
  // Does not reserve up front because callers may create many small batches;
  // the vector grows as needed.
  explicit IOBatch(uint64_t offset, uint64_t user_tag = 0)
      : offset_(offset), user_tag_(user_tag) {}

  // Clears spans and sets a new `offset` and `user_tag`, retaining allocated
  // vector capacity for reuse across batches.
  void Reset(uint64_t offset, uint64_t user_tag = 0) {
    offset_ = offset;
    total_bytes_ = 0;
    user_tag_ = user_tag;
    write_only_ = false;
    spans_.clear();
  }

  // Appends `mem` as the next span in the batch. Returns false if the batch is
  // full; if so, call again on a new batch. Always succeeds for the first span,
  // regardless of its size, so this cannot loop forever (if `offset_ +
  // mem.size()` exceeds `off_t`, `Read`/`Write` will return `kArgument`, and
  // `NextOffset` must not be called if it overflows `uint64_t`).
  HWY_NODISCARD HWY_IO_DLLEXPORT bool Add(Span<uint8_t> mem);

  // As `Add`, for write-only batches (`Read` will return `IOCode::kWrongMode`
  // if this succeeded on a non-empty `mem`). Separate name avoids overload
  // ambiguity when passing containers or arrays convertible to both `Span`s.
  HWY_NODISCARD bool AddForWrite(Span<const uint8_t> mem) {
    if (!Add(Span<uint8_t>(const_cast<uint8_t*>(mem.data()), mem.size()))) {
      return false;
    }
    if (!mem.empty()) write_only_ = true;
    return true;
  }

  // Appends up to `rows` rows of `bytes_per_row` bytes each, spaced by
  // `stride_bytes >= bytes_per_row` in memory, starting at `dst`. When
  // `stride_bytes == bytes_per_row`, coalesces the fitting rows in O(1).
  // Returns the number of rows added (`1..rows` on an empty batch, or `0..rows`
  // if the batch already contained spans), so callers can advance by
  // `added * stride_bytes` and start a new `IOBatch` at `NextOffset()` when the
  // return value is less than `rows`.
  HWY_NODISCARD HWY_IO_DLLEXPORT size_t AddStrided(void* dst, size_t rows,
                                                   size_t bytes_per_row,
                                                   size_t stride_bytes);

  // As `AddStrided`, for write-only batches (`Read` will return
  // `IOCode::kWrongMode` if this added any non-empty rows).
  HWY_NODISCARD size_t AddStridedForWrite(const void* src, size_t rows,
                                          size_t bytes_per_row,
                                          size_t stride_bytes) {
    const size_t added =
        AddStrided(const_cast<void*>(src), rows, bytes_per_row, stride_bytes);
    if (added != 0 && bytes_per_row != 0) write_only_ = true;
    return added;
  }

  bool IsEmpty() const { return spans_.empty(); }
  uint64_t Offset() const { return offset_; }
  uint64_t TotalBytes() const { return total_bytes_; }
  uint64_t UserTag() const { return user_tag_; }
  uint64_t NextOffset() const {
    return SafeAdd<uint64_t>(offset_, total_bytes_);
  }
  IOSpans Spans() const {
    return IOSpans(Span<const IOSpan>(spans_.data(), spans_.size()),
                   total_bytes_);
  }

  // Calls `file.ReadSpans(Offset(), Spans(), ...)`. Returns `TotalBytes()` if
  // all spans were completely read (including `Status(0)` when `IsEmpty()`,
  // e.g. after adding 0-byte spans), otherwise an error (`IOCode::kWrongMode`
  // if `AddForWrite` or `AddStridedForWrite` added non-empty bytes,
  // `Status::kTruncated` with the bytes read before EOF as payload, or an OS
  // error status). See `File::ReadSpans` regarding concurrent calls (with
  // distinct `worker_buf` instances per thread) and partial progress. Empty
  // batches are forwarded so that backends still validate open state, mode,
  // and `Offset()`.
  Status Read(const File& file, IOWorkerBuf& worker_buf) const {
    if (write_only_) return IOCode::kWrongMode;
    return file.ReadSpans(offset_, Spans(), worker_buf);
  }

  // Calls `file.WriteSpans(Offset(), Spans())`. Returns `TotalBytes()` if all
  // spans were written (including `Status(0)` when `IsEmpty()`), otherwise an
  // error. Arbitrary offsets and concurrent calls require
  // `file.Capabilities().Has(FileCap::kRandomWrite)`.
  Status Write(File& file) const { return file.WriteSpans(offset_, Spans()); }

 private:
  uint64_t offset_;
  uint64_t total_bytes_ = 0;
  uint64_t user_tag_;
  bool write_only_ = false;
  std::vector<IOSpan> spans_;  // contiguous in the file.
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_IO_BATCH_H_
