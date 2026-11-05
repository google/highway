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

#include "hwy/io/file_util.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <utility>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/file_system.h"
#include "hwy/io/path.h"
#include "hwy/profiler.h"
#include "hwy/status.h"

namespace hwy {

HWY_IO_DLLEXPORT Status VerifyEmpty(const File& file) {
  uint8_t probe;
  const Status status = file.Read(/*offset=*/0, WritableByteSpan(&probe, 1));
  if (status == Status(Status::kTruncated, 0)) return Status();
  return status.ok() ? Status(Status::kUnimplemented) : status;
}

HWY_IO_DLLEXPORT Status ReadFileToAligned(const Path& pathname,
                                          AlignedPtr<uint8_t>& out) {
  PROFILER_FUNC;
  out.reset();
  std::unique_ptr<File> file;
  const Status open_status =
      MaybeOpenFile(pathname, {FileMode::kRead, AccessHint::kSequential}, file);
  if (!open_status.ok()) return open_status;
  const Status size_status = file->Size();
  if (!size_status.ok()) return size_status;
  const uint64_t file_size = size_status.size();
  if (!ScalarFits<size_t>(file_size)) return Status::kRange;
  if (file_size == 0) {
    const Status empty = VerifyEmpty(*file);
    return empty.ok() ? Status(0) : empty;
  }
  AlignedPtr<uint8_t> buf =
      AllocateAligned<uint8_t>(SafeCast<size_t>(file_size));
  if (!buf) return Status::kResource;
  const Status read_status = file->Read(/*offset=*/0, buf.span());
  if (!read_status.ok()) return read_status;
  out = std::move(buf);
  return Status(file_size);
}

HWY_IO_DLLEXPORT Status WriteBytesToFile(const Path& pathname,
                                         Span<const uint8_t> bytes) {
  PROFILER_FUNC;
  std::unique_ptr<File> file;
  const Status open_status = MaybeOpenFile(
      pathname, {FileMode::kWriteTruncate, AccessHint::kSequential}, file);
  if (!open_status.ok()) return open_status;
  const Status write_status = file->Write(/*offset=*/0, bytes);
  // Always call `Close()` explicitly so deferred flush errors on network or
  // distributed filesystems are not silently discarded by `~File()`.
  const Status close_status = file->Close();
  if (!write_status.ok()) return write_status;
  if (!close_status.ok()) return close_status;
  return write_status;
}

}  // namespace hwy
