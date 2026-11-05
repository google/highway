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

#include "hwy/io/io_batch.h"

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/io_span.h"

namespace hwy {

bool IOBatch::Add(Span<uint8_t> mem) {
  if (mem.empty()) return true;
  // Always accept the first span, otherwise callers would loop forever.
  if (!spans_.empty()) {
    if (!SumWithin(total_bytes_, mem.size(), File::kMaxBatchBytes)) {
      return false;
    }
    const uint64_t new_total = SafeAdd<uint64_t>(total_bytes_, mem.size());
    if (!SumFits<int64_t>(offset_, new_total)) return false;
    IOSpan& back = spans_.back();
    if (static_cast<uint8_t*>(back.mem) + back.bytes == mem.data()) {
      back.bytes = SafeAdd<size_t>(back.bytes, mem.size());
      total_bytes_ = new_total;
      return true;
    }
  }
  spans_.push_back(IOSpan{mem.data(), mem.size()});
  total_bytes_ = SafeAdd<uint64_t>(total_bytes_, mem.size());
  return true;
}

size_t IOBatch::AddStrided(void* dst, size_t rows, size_t bytes_per_row,
                           size_t stride_bytes) {
  if (rows == 0 || bytes_per_row == 0) return rows;
  HWY_ASSERT(stride_bytes >= bytes_per_row);
  uint8_t* row_ptr = static_cast<uint8_t*>(dst);

  // O(1) fast path when rows are packed contiguously in memory.
  if (stride_bytes == bytes_per_row) {
    const uint64_t max_off = static_cast<uint64_t>(LimitsMax<int64_t>());
    const uint64_t max_batch = File::kMaxBatchBytes;
    uint64_t max_add_bytes = 0;
    if (total_bytes_ < max_batch && offset_ <= max_off &&
        total_bytes_ <= max_off - offset_) {
      max_add_bytes =
          HWY_MIN(max_batch - total_bytes_, max_off - offset_ - total_bytes_);
    }
    const uint64_t max_rows =
        HWY_MIN(static_cast<uint64_t>(rows),
                HWY_MIN(max_add_bytes / bytes_per_row,
                        static_cast<uint64_t>(~size_t{0} / bytes_per_row)));
    size_t fit_rows = static_cast<size_t>(max_rows);
    if (fit_rows == 0) {
      if (!spans_.empty()) return 0;
      fit_rows = 1;  // Always accept at least 1 row on an empty batch.
    }
    HWY_ASSERT(Add(Span<uint8_t>(row_ptr, fit_rows * bytes_per_row)));
    return fit_rows;
  }

  for (size_t r = 0; r < rows; ++r) {
    if (!Add(Span<uint8_t>(row_ptr, bytes_per_row))) return r;
    if (r + 1 < rows) row_ptr += stride_bytes;
  }
  return rows;
}

}  // namespace hwy
