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

#ifndef HIGHWAY_HWY_IO_STREAM_FILE_H_
#define HIGHWAY_HWY_IO_STREAM_FILE_H_

// Sequential `File`s such as pipes and standard streams. The `StreamFile` base
// class is freestanding; `OpenStdin` and `OpenStdout` require OS file support.

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/file.h"
#include "hwy/status.h"

namespace hwy {

// Non-seekable `File` for one direction of a byte stream, e.g. a pipe,
// terminal, or the output of a decompressor. Accesses must be in order: the
// `offset` passed to `Read`/`Write` must equal `Position()`, otherwise they
// return `Status::kArgument`. `ReadSpans`/`WriteSpans` (and thus `IOBatch`)
// transfer their spans in order. Because `Capabilities()` lacks
// `FileCap::kRandomAccess` and `FileCap::kRandomWrite`, calls must not be
// concurrent. `Size`, `Stat`, `SetSize`, `Preallocate`, `Advise`, `Sync` and
// `Map*` return `Status::kUnimplemented`. As for other `File`s, operations
// return `Status::kPrecondition` after `Close()`, and those requiring the other
// direction (including `SetSize`, `Preallocate` and `Sync` on read streams)
// return `IOCode::kWrongMode`.
//
// Subclasses implement `ReadSome`/`WriteSome`; this class handles position
// checks and loops until the request is satisfied (short transfers are normal
// for pipes). Unlike seekable files, bytes transferred before an error are
// consumed, hence `Position()` advances by them even if an error is returned.
// A possible future extension is to allow forward offsets on read streams by
// consuming and discarding the gap, e.g. to skip frames; write streams would
// remain strict.
class HWY_IO_DLLEXPORT StreamFile : public detail::ClosableFile {
 public:
  enum class Direction { kRead, kWrite };

  explicit StreamFile(Direction direction) : direction_(direction) {}

  FileCaps Capabilities() const override { return FileCaps(); }

  // Number of bytes transferred so far, which is the `offset` required by the
  // next `Read` or `Write`.
  uint64_t Position() const { return position_; }

  Status Size() const override {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    return Status::kUnimplemented;
  }

  Status MapImmutable(MapPtr& map) override { return RejectMap(map); }

  Status SetSize(uint64_t /*size*/) override { return UnimplementedForWrite(); }

  Status Preallocate(uint64_t /*offset*/, uint64_t /*size*/) override {
    return UnimplementedForWrite();
  }

  Status Advise(uint64_t /*offset*/, uint64_t /*size*/,
                AccessHint /*hint*/) const override {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    return Status::kUnimplemented;
  }

  Status Sync() override { return UnimplementedForWrite(); }

  // Blocks until `to.size()` bytes have been read or the stream ends, in which
  // case it returns `Status::kTruncated` with the number of bytes read.
  Status Read(uint64_t offset, Span<uint8_t> to) const override;

  // Blocks until all of `from` has been written.
  Status Write(uint64_t offset, Span<const uint8_t> from) override;

 protected:
  // Blocks until at least one byte is available, then reads up to `to.size()`
  // (> 0) bytes into `to`. Returns the number of bytes read, 0 at the end of
  // the stream, or an error. Implementations retry `EINTR`.
  virtual Status ReadSome(Span<uint8_t> to) const = 0;

  // Writes between 1 and `from.size()` (> 0) bytes of `from` and returns the
  // number written, or an error. Implementations retry `EINTR`.
  virtual Status WriteSome(Span<const uint8_t> from) = 0;

 private:
  // As for other `File`s, rejects transfers whose end exceeds `int64_t` (and
  // sizes exceeding the `Status` payload).
  Status Check(Direction direction, uint64_t offset, size_t size) const;

  Status UnimplementedForWrite() const {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    if (direction_ != Direction::kWrite) return IOCode::kWrongMode;
    return Status::kUnimplemented;
  }

  Direction direction_;
  // Mutable because `Read` is const; not atomic because calls are sequential.
  mutable uint64_t position_ = 0;
};

// Sets `file` to a `StreamFile` that reads the process's standard input, or to
// nullptr on failure. Raw bytes for pipes and redirected files on all OSes;
// terminals are line-cooked by the OS (POSIX canonical mode, Windows console),
// so callers handling binary data should check `IsTerminal()` and refuse.
// Returns `Status()` on success, or an error (`Status::kUnimplemented` if
// `HWY_IO_HAVE_OS_FILES` is 0, `Status::kNotFound` if there is no standard
// input). `Close()` does not close the underlying stream, and other readers
// (e.g. `std::cin`) must not be used concurrently because their buffering would
// consume bytes out of order. Reads block until data is available; on POSIX, a
// non-blocking descriptor returns `Status::kUnavailable` instead.
HWY_IO_DLLEXPORT Status OpenStdin(std::unique_ptr<File>& file);

// As `OpenStdin`, but for writing the process's standard output. Writes are
// unbuffered, hence interleaving with `printf`/`std::cout` output requires
// flushing those first. Binary output to a terminal garbles the display, hence
// callers should check `IsTerminal()` and refuse unless forced. Writing to a
// pipe whose reader exited raises `SIGPIPE` on POSIX unless ignored, in which
// case `Write` returns `Status::kOSError` (`EPIPE`).
HWY_IO_DLLEXPORT Status OpenStdout(std::unique_ptr<File>& file);

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_STREAM_FILE_H_
