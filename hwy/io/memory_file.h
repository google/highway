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

#ifndef HIGHWAY_HWY_IO_MEMORY_FILE_H_
#define HIGHWAY_HWY_IO_MEMORY_FILE_H_

// In-memory `File`. Freestanding: does not require OS file support.

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <utility>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/file.h"
#include "hwy/status.h"

namespace hwy {

// In-memory implementation of File backed by either caller-owned memory, which
// must outlive this object, or memory owned by this object (see the
// constructors taking `MapPtr` / `MutableMapPtr`). In both cases, mappings
// returned by `Map*()` are non-owning views. Unlike POSIX, bytes between
// `Size()` and a later `Write`/`MaybeSpanForWrite` offset are not zeroed (doing
// so would race with concurrent disjoint writers), nor are bytes beyond a
// smaller `SetSize`; call `SetSize` to grow the file if holes must read as
// zero.
class HWY_IO_DLLEXPORT MemoryFile : public detail::ClosableFile {
 public:
  // Read-only file over existing bytes (`Size() == bytes.size()`).
  explicit MemoryFile(Span<const uint8_t> bytes)
      // Never mutated when `read_only_` is true
      // (`Write`/`MaybeSpanForWrite`/`SetSize` reject).
      : data_(NonNullData(const_cast<uint8_t*>(bytes.data()))),
        capacity_(bytes.size()),
        size_(bytes.size()),
        read_only_(true) {}

  // File over mutable `buffer` opened according to `mode`:
  // - kWriteNew / kWriteTruncate: initially empty (`Size() == 0`), grows up to
  //   `buffer.size()` via `Write`, `MaybeSpanForWrite`, or `SetSize`.
  // - kWriteUpdate: initially `Size() == buffer.size()`, writable in place
  //   up to `buffer.size()`.
  // - kRead: read-only (`Size() == buffer.size()`).
  MemoryFile(Span<uint8_t> buffer, FileMode mode)
      : data_(NonNullData(buffer.data())),
        capacity_(buffer.size()),
        size_(mode == FileMode::kWriteTruncate || mode == FileMode::kWriteNew
                  ? 0
                  : buffer.size()),
        read_only_(mode == FileMode::kRead) {}

  // As above, but takes ownership of the memory, which is freed when this
  // object is destroyed (not by `Close`, so that spans returned by `MaybeSpan`
  // and `MaybeSpanForWrite` remain valid as documented). For example, wrap an
  // OS file's `MapWritable` mapping (e.g. of a memfd shared with a sandbox)
  // to obtain a `File` with zero-copy `MaybeSpanForWrite`, or own a buffer
  // from `AllocateAligned`. Also prevents binding a temporary to the `Span`
  // constructors, which would dangle.
  explicit MemoryFile(MapPtr&& bytes) : MemoryFile(bytes.span()) {
    owned_ = std::move(bytes);
  }
  // Read-only, owning a writable buffer. Note that an lvalue `MutableMapPtr`
  // instead binds to the non-owning `Span<const uint8_t>` constructor.
  explicit MemoryFile(MutableMapPtr&& bytes)
      : MemoryFile(MapPtr(std::move(bytes))) {}
  MemoryFile(MutableMapPtr&& buffer, FileMode mode)
      : MemoryFile(buffer.span(), mode) {
    owned_ = std::move(buffer);
  }

  // Same for all modes (read-only instances reject writes with
  // `IOCode::kWrongMode`). `kPreallocate` because the fixed `Capacity()` is
  // already reserved.
  FileCaps Capabilities() const override {
    return FileCap::kRandomAccess | FileCap::kRandomWrite | FileCap::kExtend |
           FileCap::kPreallocate | FileCap::kSync | FileCap::kAdvise |
           FileCap::kMapImmutable | FileCap::kMapShared | FileCap::kMapWritable;
  }

  // High-water mark of the end of `Write`s and `MaybeSpanForWrite`
  // reservations (even if the caller wrote fewer bytes into the span), unless
  // changed by `SetSize`. Initially 0 for `kWriteNew`/`kWriteTruncate`,
  // otherwise the capacity. Thread-safe: concurrent `Write` and
  // `MaybeSpanForWrite` to disjoint ranges are allowed. To reserve a bound and
  // then use only part of it, call `SetSize` with the actual end afterwards.
  Status Size() const override {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    return Status(size_.load(std::memory_order_acquire));
  }
  uint64_t Capacity() const { return capacity_; }

  // Zero-fills when extending, but does not modify bytes when shrinking
  // (they are merely no longer part of the file), hence shrinking after an
  // oversized `MaybeSpanForWrite` reservation is O(1) and does not overwrite
  // the unused part of the span. Returns `Status::kResource` if `size` exceeds
  // `Capacity()`. Unlike `Write` and `MaybeSpanForWrite`, must be externally
  // serialized with writes: extending zero-fills bytes that a concurrent
  // writer may just have stored.
  Status SetSize(uint64_t size) override;

  // Succeeds iff `[offset, offset + size)` lies within `Capacity()`, otherwise
  // returns `Status::kResource`.
  Status Preallocate(uint64_t offset, uint64_t size) override {
    const Status s = CheckWritable();
    if (!s.ok()) return s;
    if (!SumFits<int64_t>(offset, size)) return Status::kArgument;
    if (!SumWithin(offset, size, capacity_)) return Status::kResource;
    return Status();
  }

  Status Advise(uint64_t offset, uint64_t size,
                AccessHint /*hint*/) const override {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    if (!SumFits<int64_t>(offset, size)) return Status::kArgument;
    return Status();
  }

  Status Read(uint64_t offset, Span<uint8_t> to) const override;

  Span<const uint8_t> MaybeSpan(uint64_t offset, size_t size) const override;

  // The following return non-owning views of [0, Size()), or an error if
  // closed. They are all coherent with `Write` because they view the backing
  // store directly. `MapWritable` returns `IOCode::kWrongMode` if read-only.
  Status MapImmutable(MapPtr& map) override;
  Status MapShared(MapPtr& map) override;
  Status MapWritable(MutableMapPtr& map) override;

  // Returns `Status::kResource` (payload 0) if `offset + from.size()` exceeds
  // `Capacity()`, also for zero-length writes (as `MaybeSpanForWrite` would
  // fail). Zero-length writes within the capacity succeed without extending
  // the file.
  Status Write(uint64_t offset, Span<const uint8_t> from) override;

  // Note: when `size != 0`, extends `Size()` immediately before the caller
  // populates the returned span, so concurrent readers must synchronize
  // externally with `MaybeSpanForWrite`.
  Span<uint8_t> MaybeSpanForWrite(uint64_t offset, size_t size) override;

  Status Sync() override { return CheckWritable(); }

 private:
  // Ensures `MaybeSpan(0, 0)` and `MaybeSpanForWrite(0, 0)` return a non-null
  // `data()` pointer even when constructed from a default `{nullptr, 0}` span
  // (never dereferenced because `capacity_ == 0` whenever `p == nullptr`).
  static uint8_t* NonNullData(uint8_t* p) {
    static uint8_t dummy = 0;
    return p != nullptr ? p : &dummy;
  }

  Status CheckWritable() const {
    if (IsClosed()) return Status::kPrecondition;
    if (read_only_) return IOCode::kWrongMode;
    return Status();
  }

  // Non-owning view of [0, Size()) as `MapPtr` or `MutableMapPtr`.
  template <typename T>
  Status View(AlignedPtr<T>& map) const;

  void ExtendSizeTo(uint64_t end);

  uint8_t* data_;
  uint64_t capacity_;
  std::atomic<uint64_t> size_;
  bool read_only_;
  MapPtr owned_;  // Empty unless constructed from `MapPtr`/`MutableMapPtr`.
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_MEMORY_FILE_H_
