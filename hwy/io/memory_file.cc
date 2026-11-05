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

#include "hwy/io/memory_file.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/status.h"

namespace hwy {

Status MemoryFile::SetSize(uint64_t size) {
  const Status s = CheckWritable();
  if (!s.ok()) return s;
  if (!ScalarFits<int64_t>(size)) return Status::kArgument;
  if (size > capacity_) return Status::kResource;
  const uint64_t prev = size_.load(std::memory_order_relaxed);
  if (size > prev) {
    ZeroBytes(data_ + SafeCast<size_t>(prev), SafeCast<size_t>(size - prev));
    size_.store(size, std::memory_order_release);
  } else if (size < prev) {
    // Does not zero `[size, prev)`: after `MaybeSpanForWrite` reserved more
    // than was used, those bytes may belong to the caller (e.g. a ring buffer)
    // or merely be unwritten. Growing again via `SetSize` zero-fills.
    size_.store(size, std::memory_order_release);
  }
  return Status(size);
}

Status MemoryFile::Read(uint64_t offset, Span<uint8_t> to) const {
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  if (!SumFits<int64_t>(offset, to.size())) return Status::kArgument;
  const uint64_t file_size = size_.load(std::memory_order_acquire);
  const size_t avail =
      offset < file_size
          ? SafeCast<size_t>(HWY_MIN(file_size - offset, uint64_t{to.size()}))
          : 0;
  if (avail != 0) {
    CopyBytes(data_ + SafeCast<size_t>(offset), to.data(), avail);
  }
  if (avail < to.size()) return Status(Status::kTruncated, avail);
  return Status(avail);
}

Span<const uint8_t> MemoryFile::MaybeSpan(uint64_t offset, size_t size) const {
  if (IsClosed() || !SumFits<int64_t>(offset, size)) {
    return Span<const uint8_t>();
  }
  const uint64_t file_size = size_.load(std::memory_order_acquire);
  if (!SumWithin(offset, size, file_size)) return Span<const uint8_t>();
  return Span<const uint8_t>(data_ + SafeCast<size_t>(offset), size);
}

template <typename T>
Status MemoryFile::View(AlignedPtr<T>& map) const {
  map.reset();
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  const uint64_t raw_size = size_.load(std::memory_order_acquire);
  if (!ScalarFits<size_t>(raw_size)) return Status::kRange;
  const size_t file_size = static_cast<size_t>(raw_size);
  if (file_size == 0) return Status(0);
  map = AlignedPtr<T>(data_, file_size, /*deleter=*/nullptr,
                      /*opaque=*/nullptr);
  return Status(file_size);
}

Status MemoryFile::MapImmutable(MapPtr& map) { return View(map); }

Status MemoryFile::MapShared(MapPtr& map) { return View(map); }

Status MemoryFile::MapWritable(MutableMapPtr& map) {
  map.reset();
  const Status s = CheckWritable();
  if (!s.ok()) return s;
  return View(map);
}

Status MemoryFile::Write(uint64_t offset, Span<const uint8_t> from) {
  const Status s = CheckWritable();
  if (!s.ok()) return s;
  if (!SumFits<int64_t>(offset, from.size())) return Status::kArgument;
  // Before the empty check, for consistency with `MaybeSpanForWrite`.
  if (!SumWithin(offset, from.size(), capacity_)) return Status::kResource;
  if (from.empty()) return Status(0);
  // Copy before publishing the extended size so concurrent acquire loads in
  // `Size`/`Read`/`MaybeSpan`/`Map*` never observe unwritten bytes.
  CopyBytes(from.data(), data_ + SafeCast<size_t>(offset), from.size());
  ExtendSizeTo(SafeAdd<uint64_t>(offset, from.size()));
  return Status(from.size());
}

Span<uint8_t> MemoryFile::MaybeSpanForWrite(uint64_t offset, size_t size) {
  if (IsClosed() || read_only_ || !SumFits<int64_t>(offset, size) ||
      !SumWithin(offset, size, capacity_)) {
    return Span<uint8_t>();
  }
  if (size != 0) ExtendSizeTo(SafeAdd<uint64_t>(offset, size));
  return Span<uint8_t>(data_ + SafeCast<size_t>(offset), size);
}

void MemoryFile::ExtendSizeTo(uint64_t end) {
  uint64_t prev = size_.load(std::memory_order_relaxed);
  while (prev < end &&
         !size_.compare_exchange_weak(prev, end, std::memory_order_release,
                                      std::memory_order_relaxed)) {
  }
}

}  // namespace hwy
