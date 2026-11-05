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

#include "hwy/io/file.h"

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/io_internal.h"
#include "hwy/io/io_span.h"
#include "hwy/profiler.h"
#include "hwy/status.h"

#if PROFILER_ENABLED
#include <atomic>
#endif

#if !HWY_IO_HAVE_OS_FILES
#include <memory>
#endif

namespace hwy {

namespace detail {

#if PROFILER_ENABLED
struct alignas(HWY_ALIGNMENT) PaddedAtomic {
  std::atomic<uint64_t> v{0};
  HWY_MAYBE_UNUSED uint8_t
      padding[HWY_ALIGNMENT - sizeof(std::atomic<uint64_t>)] = {};
};
static_assert(sizeof(PaddedAtomic) == HWY_ALIGNMENT, "size");

PaddedAtomic g_bytes_read;
PaddedAtomic g_bytes_written;
PaddedAtomic g_syscalls;
PaddedAtomic g_eintr_retries;
PaddedAtomic g_short_reads;
#endif  // PROFILER_ENABLED

void RecordRead(uint64_t bytes) {
#if PROFILER_ENABLED
  g_bytes_read.v.fetch_add(bytes, std::memory_order_relaxed);
#else
  (void)bytes;
#endif
}
void RecordWrite(uint64_t bytes) {
#if PROFILER_ENABLED
  g_bytes_written.v.fetch_add(bytes, std::memory_order_relaxed);
#else
  (void)bytes;
#endif
}
void RecordSyscall() {
#if PROFILER_ENABLED
  g_syscalls.v.fetch_add(1, std::memory_order_relaxed);
#endif
}
void RecordEintr() {
#if PROFILER_ENABLED
  g_eintr_retries.v.fetch_add(1, std::memory_order_relaxed);
#endif
}
void RecordShortRead() {
#if PROFILER_ENABLED
  g_short_reads.v.fetch_add(1, std::memory_order_relaxed);
#endif
}

}  // namespace detail

IOCounters IOCounters::Get() {
  IOCounters c;
#if PROFILER_ENABLED
  c.bytes_read = detail::g_bytes_read.v.load(std::memory_order_relaxed);
  c.bytes_written = detail::g_bytes_written.v.load(std::memory_order_relaxed);
  c.syscalls = detail::g_syscalls.v.load(std::memory_order_relaxed);
  c.eintr_retries = detail::g_eintr_retries.v.load(std::memory_order_relaxed);
  c.short_reads = detail::g_short_reads.v.load(std::memory_order_relaxed);
#endif
  return c;
}

void IOCounters::Reset() {
#if PROFILER_ENABLED
  detail::g_bytes_read.v.store(0, std::memory_order_relaxed);
  detail::g_bytes_written.v.store(0, std::memory_order_relaxed);
  detail::g_syscalls.v.store(0, std::memory_order_relaxed);
  detail::g_eintr_retries.v.store(0, std::memory_order_relaxed);
  detail::g_short_reads.v.store(0, std::memory_order_relaxed);
#endif
}

constexpr size_t File::kMaxBatchBytes;

Status File::Stat(FileStat& out) const {
  out = FileStat();
  const Status size = Size();
  if (!size.ok()) return size;
  out.size = size.size();
  out.kind = EntryKind::kFile;
  return size;
}

Status File::ReadSpans(uint64_t offset, const IOSpans& spans,
                       IOWorkerBuf& worker_buf) const {
  PROFILER_ZONE("IO.ReadSpans");
  const Status sum_status = spans.ValidateAt(offset);
  if (!sum_status.ok()) return sum_status;
  const uint64_t total_bytes = sum_status.size();
  if (total_bytes == 0) return Read(offset, Span<uint8_t>());

  // 1. Fast Path: single span (or coalesced into one in `IOBatch::Add`).
  if (spans.size() == 1) {
    return Read(offset, WritableByteSpan(spans[0].mem, spans[0].bytes));
  }

  // 2. Strided Padding: Single bulk read + scatter memcpy when `total_bytes`
  // is bounded by `kMaxBatchBytes`.
  if (total_bytes <= kMaxBatchBytes) {
    const size_t total = SafeCast<size_t>(total_bytes);
    uint8_t* buf = worker_buf.Ensure(total);
    if (buf != nullptr) {
      const Status status = Read(offset, Span<uint8_t>(buf, total));
      const uint64_t avail = status.ok()                     ? total_bytes
                             : status.Is(Status::kTruncated) ? status.payload()
                                                             : 0;
      const uint8_t* src = buf;
      uint64_t remaining_avail = HWY_MIN(avail, total_bytes);
      for (const IOSpan& span : spans) {
        if (span.bytes == 0) continue;
        const size_t copy_bytes =
            SafeCast<size_t>(HWY_MIN(remaining_avail, uint64_t{span.bytes}));
        if (copy_bytes == 0) break;
        CopyBytes(src, span.mem, copy_bytes);
        src += copy_bytes;
        remaining_avail -= copy_bytes;
      }
      return status;
    }
  }

  // 3. Fallback if scratch allocation fails or `total_bytes` exceeds
  // `kMaxBatchBytes`: per-span sequential reads.
  uint64_t cur_offset = offset;
  for (const IOSpan& span : spans) {
    if (span.bytes == 0) continue;
    const Status status =
        Read(cur_offset, WritableByteSpan(span.mem, span.bytes));
    if (!status.ok()) {
      if (status.Is(Status::kTruncated)) {
        return Status(Status::kTruncated,
                      SafeAdd<uint64_t>(cur_offset - offset, status.payload()));
      }
      return status;
    }
    cur_offset = SafeAdd<uint64_t>(cur_offset, span.bytes);
  }
  return Status(total_bytes);
}

Status File::WriteSpans(uint64_t offset, const IOSpans& spans) {
  PROFILER_ZONE("IO.WriteSpans");
  const Status sum_status = spans.ValidateAt(offset);
  if (!sum_status.ok()) return sum_status;
  const uint64_t total_bytes = sum_status.size();
  if (total_bytes == 0) return Write(offset, Span<const uint8_t>());
  uint64_t cur_offset = offset;
  for (const IOSpan& span : spans) {
    if (span.bytes == 0) continue;
    const Status status = Write(cur_offset, ByteSpan(span.mem, span.bytes));
    if (!status.ok()) return status;
    cur_offset = SafeAdd<uint64_t>(cur_offset, span.bytes);
  }
  return Status(total_bytes);
}

#if !HWY_IO_HAVE_OS_FILES
// Defined in posix_file.cc or windows_file.cc if OS files are available. Here
// rather than in file_system.cc so that users of `StreamFile` need not link it.
HWY_IO_DLLEXPORT Status OpenStdin(std::unique_ptr<File>& file) {
  file.reset();
  return Status::kUnimplemented;
}

HWY_IO_DLLEXPORT Status OpenStdout(std::unique_ptr<File>& file) {
  file.reset();
  return Status::kUnimplemented;
}
#endif  // !HWY_IO_HAVE_OS_FILES

}  // namespace hwy
