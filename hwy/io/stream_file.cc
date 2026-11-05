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

#include "hwy/io/stream_file.h"

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/status.h"

namespace hwy {

Status StreamFile::Read(uint64_t offset, Span<uint8_t> to) const {
  const Status s = Check(Direction::kRead, offset, to.size());
  if (!s.ok()) return s;
  size_t pos = 0;
  while (pos < to.size()) {
    const Status got =
        ReadSome(Span<uint8_t>(to.data() + pos, to.size() - pos));
    if (!got.ok() || got.size() == 0) {
      position_ += pos;
      return got.ok() ? Status(Status::kTruncated, pos) : got;
    }
    HWY_DASSERT(got.size() <= to.size() - pos);
    pos += static_cast<size_t>(got.size());
  }
  position_ += pos;
  return Status(pos);
}

Status StreamFile::Write(uint64_t offset, Span<const uint8_t> from) {
  const Status s = Check(Direction::kWrite, offset, from.size());
  if (!s.ok()) return s;
  size_t pos = 0;
  while (pos < from.size()) {
    const Status put =
        WriteSome(Span<const uint8_t>(from.data() + pos, from.size() - pos));
    // Zero bytes without error should not happen; avoid an infinite loop.
    if (!put.ok() || put.size() == 0) {
      position_ += pos;
      return put.ok() ? Status(IOCode::kNoProgress, pos) : put;
    }
    HWY_DASSERT(put.size() <= from.size() - pos);
    pos += static_cast<size_t>(put.size());
  }
  position_ += pos;
  return Status(pos);
}

Status StreamFile::Check(Direction direction, uint64_t offset,
                         size_t size) const {
  const Status s = CheckOpen();
  if (!s.ok()) return s;
  if (direction != direction_) return IOCode::kWrongMode;
  // `SumWithin` rather than `size > kMaxPayload`, which is always false (and
  // warns) on 32-bit platforms; widening casts would not silence that.
  if (!SumFits<int64_t>(offset, size) ||
      !SumWithin(uint64_t{0}, size, Status::kMaxPayload)) {
    return Status::kArgument;
  }
  if (offset != position_) return Status::kArgument;
  return Status();
}

}  // namespace hwy
