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

#include "hwy/io/file_view.h"

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/io_span.h"
#include "hwy/status.h"

namespace hwy {

FileCaps FileView::Capabilities() const {
  FileCaps caps =
      file_.Capabilities() &
      (FileCap::kRandomAccess | FileCap::kAdvise | FileCap::kNativeScatter);
  if (SumFits<int64_t>(begin_, size_) &&
      file_.MaybeSpan(0, 0).data() != nullptr) {
    caps |= FileCap::kMapImmutable | FileCap::kMapShared;
  }
  return caps;
}

Status FileView::Size() const {
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  if (!SumFits<int64_t>(begin_, size_)) return Status::kArgument;
  const Status parent_size = file_.Size();
  if (!parent_size.ok()) return parent_size;
  if (!SumWithin(begin_, size_, parent_size.size())) {
    return Status(Status::kRange, begin_ < parent_size.size()
                                      ? parent_size.size() - begin_
                                      : 0);
  }
  return Status(size_);
}

Status FileView::Advise(uint64_t offset, uint64_t size, AccessHint hint) const {
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  if (!SumFits<int64_t>(begin_, size_) || !SumFits<int64_t>(offset, size)) {
    return Status::kArgument;
  }
  if (offset > size_) return Status::kRange;
  const uint64_t avail = size_ - offset;
  const uint64_t clamped = (size == 0 || size > avail) ? avail : size;
  // `File::Advise` treats `size == 0` as "until EOF from offset"; avoid
  // advising past the end of this view onto the rest of the parent file.
  if (clamped == 0) return Status();
  return file_.Advise(SafeAdd<uint64_t>(begin_, offset), clamped, hint);
}

Status FileView::Read(uint64_t offset, Span<uint8_t> to) const {
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  if (!SumFits<int64_t>(offset, to.size()) ||
      !SumFits<int64_t>(begin_, size_)) {
    return Status::kArgument;
  }
  const size_t avail =
      offset < size_
          ? SafeCast<size_t>(HWY_MIN(size_ - offset, uint64_t{to.size()}))
          : 0;
  if (avail != 0) {
    const Status status = file_.Read(SafeAdd<uint64_t>(begin_, offset),
                                     Span<uint8_t>(to.data(), avail));
    if (!status.ok()) return status;
  }
  if (avail < to.size()) return Status(Status::kTruncated, avail);
  return Status(avail);
}

Span<const uint8_t> FileView::MaybeSpan(uint64_t offset, size_t size) const {
  if (IsClosed() || !SumFits<int64_t>(begin_, size_) ||
      !SumFits<int64_t>(offset, size) || !SumWithin(offset, size, size_)) {
    return Span<const uint8_t>();
  }
  return file_.MaybeSpan(SafeAdd<uint64_t>(begin_, offset), size);
}

Status FileView::MapImmutable(MapPtr& map) {
  map.reset();
  const Status size_status = Size();
  if (!size_status.ok()) return size_status;
  if (!ScalarFits<size_t>(size_)) return Status::kRange;
  const Span<const uint8_t> mem =
      MaybeSpan(/*offset=*/0, static_cast<size_t>(size_));
  if (mem.data() == nullptr) return Status::kUnimplemented;
  if (mem.empty()) return Status(0);
  map = MapPtr(mem.data(), mem.size(), /*deleter=*/nullptr,
               /*opaque=*/nullptr);
  return Status(mem.size());
}

Status FileView::ReadSpans(uint64_t offset, const IOSpans& spans,
                           IOWorkerBuf& worker_buf) const {
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  if (!SumFits<int64_t>(begin_, size_)) return Status::kArgument;
  const Status total = spans.ValidateAt(offset);
  if (!total.ok()) return total;
  const uint64_t total_bytes = total.size();
  if (total_bytes == 0) return Status(0);
  if (SumWithin(offset, total_bytes, size_)) {
    return file_.ReadSpans(SafeAdd<uint64_t>(begin_, offset), spans,
                           worker_buf);
  }
  if (offset >= size_) return Status(Status::kTruncated, 0);

  // Partial read straddling `size_`: forward only the `avail`-byte prefix to
  // `file_.ReadSpans` so OS backends still use zero-allocation `preadv` and
  // fallback backends allocate at most `avail` bytes instead of
  // `total_bytes`.
  const uint64_t avail = size_ - offset;
  std::vector<IOSpan> prefix;
  prefix.reserve(spans.size());
  uint64_t rem = avail;
  for (const IOSpan& span : spans) {
    if (rem == 0) break;
    if (span.bytes == 0) continue;
    const size_t take = SafeCast<size_t>(HWY_MIN(rem, uint64_t{span.bytes}));
    prefix.push_back(IOSpan{span.mem, take});
    rem -= take;
  }
  HWY_DASSERT(rem == 0);
  const IOSpans prefix_spans(Span<const IOSpan>(prefix.data(), prefix.size()),
                             avail);
  const Status status = file_.ReadSpans(SafeAdd<uint64_t>(begin_, offset),
                                        prefix_spans, worker_buf);
  if (!status.ok()) return status;
  return Status(Status::kTruncated, avail);
}

}  // namespace hwy
