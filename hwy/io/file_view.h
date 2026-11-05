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

#ifndef HIGHWAY_HWY_IO_FILE_VIEW_H_
#define HIGHWAY_HWY_IO_FILE_VIEW_H_

// Read-only `File` view of a slice of another `File`. Freestanding: does not
// require OS file support.

#include <stddef.h>
#include <stdint.h>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/file.h"
#include "hwy/io/io_span.h"
#include "hwy/status.h"

namespace hwy {

// Read-only view of the slice `[begin, begin + size)` of another `File` (e.g.
// a single blob or frame within a container file), which must outlive this
// object. Offsets passed to `Read` are relative to `begin`.
class HWY_IO_DLLEXPORT FileView : public detail::ClosableFile {
 public:
  FileView(const File& file, uint64_t begin, uint64_t size)
      : file_(file), begin_(begin), size_(size) {}
  // Prevent binding `file_` to a temporary `File` that dies at the semicolon.
  FileView(const File&&, uint64_t, uint64_t) = delete;

  // The parent's read-related capabilities, plus `kMapImmutable` and
  // `kMapShared` if the parent is resident in memory (OS mappings require
  // page-aligned offsets).
  FileCaps Capabilities() const override;

  // Returns `Status::kRange` (with the available size as the payload) if the
  // parent is smaller than `begin + size`.
  Status Size() const override;

  Status SetSize(uint64_t /*size*/) override { return RejectWrite(); }

  Status Preallocate(uint64_t /*offset*/, uint64_t /*size*/) override {
    return RejectWrite();
  }

  // Clamped to the view, i.e. does not advise beyond `begin + size`.
  Status Advise(uint64_t offset, uint64_t size, AccessHint hint) const override;

  Status Read(uint64_t offset, Span<uint8_t> to) const override;

  Span<const uint8_t> MaybeSpan(uint64_t offset, size_t size) const override;

  // OS mmap requires page/64 KiB alignment, so only in-memory views are mapped;
  // otherwise returns `Status::kUnimplemented`. Views are coherent with writes
  // to the parent `MemoryFile`, hence also suffice for `MapShared`.
  Status MapShared(MapPtr& map) override { return MapImmutable(map); }
  Status MapImmutable(MapPtr& map) override;

  Status Write(uint64_t /*offset*/, Span<const uint8_t> /*from*/) override {
    return RejectWrite();
  }

  Status WriteSpans(uint64_t /*offset*/, const IOSpans& /*spans*/) override {
    return RejectWrite();
  }

  Status Sync() override { return RejectWrite(); }

  Status ReadSpans(uint64_t offset, const IOSpans& spans,
                   IOWorkerBuf& worker_buf) const override;

 private:
  Status RejectWrite() const {
    return IsClosed() ? Status(Status::kPrecondition)
                      : Status(IOCode::kWrongMode);
  }

  const File& file_;
  uint64_t begin_;
  uint64_t size_;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_FILE_VIEW_H_
