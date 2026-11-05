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

#ifndef HIGHWAY_HWY_IO_FILE_H_
#define HIGHWAY_HWY_IO_FILE_H_

// Low-overhead, parallel-friendly file I/O, more capable and efficient than
// `<fstream>` or `FILE*`:
// - Positional reads/writes (`pread`-style) without a shared file offset or
//   locks, hence one handle can be used from many threads concurrently.
// - Scatter/gather (`preadv`/`pwritev`) of many discontiguous buffers per
//   syscall, e.g. strided rows (`IOBatch`).
// - Zero-copy access via memory mapping (`MapImmutable`, `MapShared`,
//   `MapWritable`) and in-memory backends (`MaybeSpan`), plus access hints
//   (`fadvise`/`madvise`), preallocation and sparse pre-extension for parallel
//   out-of-order writes.
// - No intermediate buffering or copies, no exceptions; errors are `Status`
//   codes with the OS error as payload, and backends report their
//   `Capabilities` up front instead of failing at runtime.
// - Pluggable file systems and in-memory/view files share one interface;
//   64-bit offsets also on 32-bit platforms.
//
// This header defines the `File` interface; see also memory_file.h,
// file_view.h, stream_file.h (pipes, stdin/stdout), io_batch.h, file_system.h
// (OS and other backends), and file_util.h. Freestanding: does not require OS
// file support.
//
// Design notes and current limitations:
// - Synchronous by design; async submission (`File::Submit(const IOBatch&)` +
//   `WaitCompletions`) can be added later without changing `IOBatch`.
// - Mappings cover the whole file; a `MapRange` path can be added if needed.
// - File sizes and offsets in `Status` payloads are capped at 48 bits (256
//   TiB).

#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/io_span.h"
#include "hwy/status.h"

// Whether the OS backend (POSIX or Windows) is available. Users may override
// this (e.g. define to 0 for freestanding or bare-metal builds), in which case
// only `FileSystem`s registered via `AddFileSystem` can open files, and
// `MemoryFile` remains usable.
//
// Determined only from predefined macros. `__has_include` would be tempting,
// but its result can differ between translation units (e.g. when a header is
// compiled as a Clang module), which would cause ODR violations and duplicate
// symbols because the .cc files select their definitions via this macro.
#ifndef HWY_IO_HAVE_OS_FILES
#if (defined(__STDC_HOSTED__) && !__STDC_HOSTED__) || \
    (HWY_ARCH_RISCV && !HWY_OS_LINUX)
#define HWY_IO_HAVE_OS_FILES 0
#elif HWY_OS_WIN || HWY_OS_LINUX || HWY_OS_APPLE || HWY_OS_FREEBSD || \
    HWY_OS_OPENBSD || defined(__unix__) || defined(__EMSCRIPTEN__)
#define HWY_IO_HAVE_OS_FILES 1
#else
#define HWY_IO_HAVE_OS_FILES 0
#endif
#endif  // HWY_IO_HAVE_OS_FILES

namespace hwy {

// -----------------------------------------------------------------------------
// Status codes, modes, hints

// Read-only mapping of an entire file, or non-owning view of a `MemoryFile`.
using MapPtr = AlignedPtr<const uint8_t>;
// Writable mapping returned by `File::MapWritable`. Movable into `MapPtr`.
using MutableMapPtr = AlignedPtr<uint8_t>;

// Status codes specific to this module, in addition to the general codes
// (kNotFound, kPermission, kResource, kTruncated for EOF, kUnimplemented,
// kOSError for otherwise unmapped OS errors, etc.). The payload of errors
// originating in the OS is the OS error (errno or GetLastError); other errors
// (such as kPrecondition, kArgument from overflow checks, or
// IOCode::kWrongMode) carry payload 0 unless documented otherwise.
enum class IOCode : Status::Code {
  // Operation incompatible with how the file or batch was opened/configured
  // (e.g. writing/syncing/resizing/preallocating a read-only file, or reading
  // into a write-only `IOBatch`).
  kWrongMode = Status::ModuleError(Status::Module::kIO),
  // The file or directory already exists (`EEXIST`).
  kExists,
  // A write transferred zero bytes without reporting an error (e.g. `write`
  // returning 0), which would otherwise cause an infinite retry loop. The
  // payload is the number of bytes written before.
  kNoProgress,
};

// Note that all write modes request both read and write access (POSIX
// `O_RDWR`, Windows `GENERIC_READ | GENERIC_WRITE`), e.g. for `Map*` and
// `ReadSpans` of output files, hence opening a pathname whose permissions only
// allow writing fails with `Status::kPermission`; to write such files, open an
// `O_WRONLY` descriptor and pass it to `MaybeOpenHandle`.
enum class FileMode {
  // Opens an existing regular file for reading.
  kRead,
  // Creates a new regular file and opens it for reading and writing; fails with
  // `IOCode::kExists` if the file already exists.
  kWriteNew,
  // Creates the file, or truncates it to zero size, then opens it for reading
  // and writing. The POSIX backend also accepts existing seekable non-terminal
  // character devices such as /dev/null or /dev/full, which are not truncated.
  kWriteTruncate,
  // Opens an existing regular file (or, as above, character device), or
  // creates it, for reading and writing without truncating.
  kWriteUpdate
};

inline const char* FileModeName(FileMode mode) {
  return mode == FileMode::kRead            ? "read"
         : mode == FileMode::kWriteNew      ? "write (new)"
         : mode == FileMode::kWriteTruncate ? "write (truncate)"
                                            : "write (update)";
}

// Hint for expected access pattern (at open time or via `File::Advise` /
// `AdviseMap`).
enum class AccessHint {
  kNormal,      // Default OS behavior.
  kSequential,  // Sequential access: larger readahead window.
  kRandom,      // Random access: disable speculative readahead.
  kWillNeed,    // Prefetch range into page cache (`Advise` / `AdviseMap`).
  kDontNeed     // Drop range from page cache (`Advise` / `AdviseMap`).
};

// Process-wide OS I/O counters (snapshot returned by `IOCounters::Get`).
// Updated only when the `hwy_io` library is compiled with `PROFILER_ENABLED`
// set to a non-zero value (defining it only in caller translation units has no
// effect); otherwise all fields remain 0.
struct IOCounters {
  static HWY_IO_DLLEXPORT IOCounters Get();
  static HWY_IO_DLLEXPORT void Reset();

  uint64_t bytes_read = 0;
  uint64_t bytes_written = 0;
  uint64_t syscalls = 0;       // pread, preadv, pwrite, pwritev, open, etc.
  uint64_t eintr_retries = 0;  // retries after EINTR
  uint64_t short_reads = 0;    // partial reads before EOF or completion
};

enum class EntryKind : uint8_t {
  kFile,     // Regular file.
  kDir,      // Directory.
  kSymlink,  // Symbolic link or Windows junction (only from `ListDir`).
  kOther     // FIFO, socket, device, etc.
};

// Metadata of an entry, returned by `StatFile` without opening a `File`, or by
// `File::Stat`. Capabilities do not depend on the entry and are available via
// `PathCapabilities`.
struct FileStat {
  // Returns whether `a` and `b` are known to refer to the same file, e.g. to
  // detect that input and output are the same, or that a file was replaced
  // between `StatFile` and opening. False if either identity is unknown.
  static bool SameFile(const FileStat& a, const FileStat& b) {
    return a.inode != 0 && a.device == b.device && a.inode == b.inode;
  }

  // Size in bytes if `kind == EntryKind::kFile`, otherwise 0.
  uint64_t size = 0;
  // Of the entry after following symbolic links, hence never `kSymlink`.
  EntryKind kind = EntryKind::kOther;
  // Identity of the file within the OS: POSIX `st_dev` and `st_ino`, or the
  // Windows volume serial number and 64-bit file index (which ReFS may not
  // guarantee to be unique). Both are 0 if unknown, e.g. for non-OS backends
  // such as `MemoryFile` and pipes; see `SameFile`.
  uint64_t device = 0;
  uint64_t inode = 0;
};

// -----------------------------------------------------------------------------
// Capabilities

// Capability bits reported by `File::Capabilities()` and, before opening, by
// `PathCapabilities`, so callers (e.g. a compressor choosing parallel vs.
// sequential writes, pre-extension, and mapping vs. `Read`) can query support
// instead of probing for `Status::kUnimplemented`. Capabilities are static
// properties of the backend
// code (e.g. whether `preadv` is available), the path within a backend (e.g.
// append-only CNS vs. Placer), and the object kind (e.g. `FileView`,
// `MemoryFile`). They are never filtered by `FileMode`: operations disallowed
// by the mode fail with `IOCode::kWrongMode` instead. Individual calls can
// still fail due to runtime limits or object/filesystem constraints (e.g.
// `ENOSPC`, `EOPNOTSUPP` on a mount, or files exceeding `size_t` when
// mapping).
enum class FileCap : uint32_t {
  kNone = 0,
  // `Read` / `ReadSpans` at arbitrary offsets, concurrent reads, and `Size()`
  // are supported (absent on sequential-only streams such as `StreamFile`).
  kRandomAccess = 1u << 0,
  // `Write` / `WriteSpans` at arbitrary offsets and concurrent writes are
  // supported (absent on read-only object kinds such as `FileView`, sequential
  // streams, and append-only backends).
  kRandomWrite = 1u << 1,
  // `SetSize` can extend the file past its current `Size()` (absent on
  // `FileView` and backends whose `Truncate` only shrinks).
  kExtend = 1u << 2,
  // `Preallocate` reserves backing storage (`fallocate` / `F_PREALLOCATE`, or
  // `MemoryFile::Capacity()`), so out-of-space conditions can be detected
  // before writing. Without this bit, `Preallocate` returns
  // `Status::kUnimplemented`.
  kPreallocate = 1u << 3,
  // `Sync()` flushes written data and metadata to persistent storage (absent on
  // `FileView` and backends that do not implement syncing).
  kSync = 1u << 4,
  // Can call `Advise()` for hinting about access patterns / prefetch.
  kAdvise = 1u << 5,
  // `ReadSpans` (and `WriteSpans` when writable) uses native OS vector I/O
  // (`preadv` / `pwritev`) without `IOWorkerBuf` scratch copies.
  kNativeScatter = 1u << 6,
  // Read-only private mapping of an unchanging file.
  kMapImmutable = 1u << 7,
  // Read-only shared mapping which may be modified by other processes.
  kMapShared = 1u << 8,
  // Writable shared mapping, also visible to/modifiable by other processes.
  kMapWritable = 1u << 9,
};

// Type-safe bitset of `FileCap` flags. Note that empty `FileCaps` means "none"
// when returned by `File::Capabilities()`, but "unknown" when returned by
// `PathCapabilities` (only possible for custom `FileSystem`s).
class FileCaps {
 public:
  constexpr FileCaps() noexcept = default;
  constexpr FileCaps(FileCap cap) noexcept  // NOLINT
      : bits_(static_cast<uint32_t>(cap)) {}

  static constexpr FileCaps FromBits(uint32_t bits) noexcept {
    return FileCaps(bits);
  }
  constexpr uint32_t Bits() const noexcept { return bits_; }
  constexpr bool Empty() const noexcept { return bits_ == 0; }
  constexpr bool Has(FileCap cap) const noexcept {
    return (bits_ & static_cast<uint32_t>(cap)) != 0;
  }
  constexpr bool HasAll(FileCaps required) const noexcept {
    return (bits_ & required.bits_) == required.bits_;
  }
  constexpr bool HasAny(FileCaps mask) const noexcept {
    return (bits_ & mask.bits_) != 0;
  }

  constexpr FileCaps operator|(FileCaps other) const noexcept {
    return FileCaps(bits_ | other.bits_);
  }
  constexpr FileCaps operator&(FileCaps other) const noexcept {
    return FileCaps(bits_ & other.bits_);
  }
  constexpr FileCaps operator~() const noexcept { return FileCaps(~bits_); }
  FileCaps& operator|=(FileCaps other) noexcept {
    bits_ |= other.bits_;
    return *this;
  }
  FileCaps& operator&=(FileCaps other) noexcept {
    bits_ &= other.bits_;
    return *this;
  }
  friend constexpr bool operator==(FileCaps a, FileCaps b) noexcept {
    return a.bits_ == b.bits_;
  }
  friend constexpr bool operator!=(FileCaps a, FileCaps b) noexcept {
    return a.bits_ != b.bits_;
  }

 private:
  explicit constexpr FileCaps(uint32_t bits) noexcept : bits_(bits) {}
  uint32_t bits_ = 0;
};

constexpr FileCaps operator|(FileCap a, FileCap b) noexcept {
  return FileCaps(a) | FileCaps(b);
}

// -----------------------------------------------------------------------------
// File

// Abstract base class enables multiple I/O backends in the same binary. Files
// are closed when the object is destroyed (typically by resetting its
// unique_ptr); call `Close()` first to observe errors.
class HWY_IO_DLLEXPORT File {
 public:
  // Maximum byte count for a multi-span `IOBatch` and for the default bulk-read
  // scatter path in `File::ReadSpans` (matches Linux `MAX_RW_COUNT`).
  static constexpr size_t kMaxBatchBytes = 0x7FFFF000;

  File() = default;
  virtual ~File() = default;

  // Noncopyable.
  File(const File& other) = delete;
  const File& operator=(const File& other) = delete;

  // ---------------------------------------------------------------------------
  // Metadata

  // Returns the capabilities of this backend/path and object kind (see
  // `FileCap`). These are constant for the lifetime of the object and
  // independent of `FileMode`; e.g. an OS file opened with `FileMode::kRead`
  // still reports `kRandomWrite`, but `Write` returns `IOCode::kWrongMode`. For
  // files opened via `MaybeOpenFile`, equal to `PathCapabilities(pathname)` (if
  // that is known), which is available before opening. Exception: character
  // devices opened by the POSIX backend (e.g. /dev/null) lack `kExtend`,
  // `kPreallocate`, `kSync`, `kAdvise` and the `kMap*` bits.
  virtual FileCaps Capabilities() const = 0;

  // Returns the size in bytes (`status.size()`), or an error (e.g.
  // `Status::kPrecondition` if closed, or `Status::kUnimplemented` for
  // non-seekable streams without `FileCap::kRandomAccess`).
  virtual Status Size() const = 0;

  // Sets `out` to the metadata of the open file, including its identity
  // (`FileStat::device` and `inode`), which unlike `StatFile` cannot be
  // affected by a concurrent rename of the pathname. Returns `Status(out.size)`
  // on success, or resets `out` and returns an error. The default sets `kind`
  // to `EntryKind::kFile` and `size` to `Size()`, with unknown identity; OS
  // backends use `fstat` / `GetFileInformationByHandle`.
  virtual Status Stat(FileStat& out) const;

  // Returns whether this is an interactive terminal (POSIX `isatty`, or a
  // Windows console). Terminals are line-cooked by the OS, and binary output
  // would garble the display, hence callers handling binary data (e.g.
  // compressed streams) should refuse to read or write them unless forced.
  // Only standard streams (`OpenStdin`/`OpenStdout`) can be terminals; false
  // for other `File`s (OS backends only open regular files and seekable
  // non-terminal character devices) and after `Close()`.
  virtual bool IsTerminal() const { return false; }

  // ---------------------------------------------------------------------------
  // Resizing and preallocation

  // Sets the logical size to `size` bytes, truncating or extending. Extended
  // bytes read as zero; on OS files, they are typically sparse holes whose
  // physical blocks are allocated by later writes unless reserved via
  // `Preallocate`. Pre-extending an output file to its upper-bounded size
  // avoids repeated metadata updates during parallel out-of-order writes; call
  // again with the final size before `Close()`. Returns `Status(size)` on
  // success, or an error (`IOCode::kWrongMode` if read-only,
  // `Status::kResource` if `MemoryFile::Capacity()` is exceeded,
  // `Status::kUnimplemented` if the backend does not support resizing, or
  // extending without `FileCap::kExtend`). Note that on Windows, `SetSize`
  // fails (`ERROR_USER_MAPPED_FILE`) while a mapping is alive, and extending a
  // file marks it sparse (`FSCTL_SET_SPARSE`) to avoid synchronous zero-filling
  // during out-of-order writes (`Close()` clears this attribute if no sparse
  // holes remain). Must be externally serialized with concurrent writes
  // (including `MaybeSpanForWrite` and `MapWritable` stores) to the same file.
  virtual Status SetSize(uint64_t /*size*/) { return Status::kUnimplemented; }

  // Reserves backing storage for `[offset, offset + size)` without changing
  // `Size()`, so that later writes within this range do not fail for lack of
  // space, and parallel out-of-order writes do not fragment extents or contend
  // on block allocation. For an output whose size is bounded by `bound`, call
  // `(void)file.Preallocate(0, bound); file.SetSize(bound);` if reservation is
  // merely advisory, or check the `Preallocate` status to fail early when out
  // of space. Returns `Status()` on success, or an error
  // (`Status::kUnimplemented` without `FileCap::kPreallocate` or if the
  // filesystem does not support reservation, `Status::kResource` if out of
  // space (`ENOSPC` / `EDQUOT`, or exceeding `MemoryFile::Capacity()`),
  // `IOCode::kWrongMode` if read-only). Reserved blocks count towards disk
  // usage and quota until the file is shrunk via `SetSize` (or deleted), hence
  // only preallocate tight bounds: if `bound` may far exceed the actual size
  // (e.g. the worst case of a compressor), reserve the expected size or skip
  // `Preallocate`. Note that macOS `F_PREALLOCATE` cannot
  // target a range: it extends the total allocated capacity to
  // `offset + size` bytes, so blocks already allocated beyond that (e.g. by
  // sparse writes) count towards the reservation instead of filling holes.
  virtual Status Preallocate(uint64_t /*offset*/, uint64_t /*size*/) {
    return Status::kUnimplemented;
  }

  // ---------------------------------------------------------------------------
  // Access hints

  // Advises the OS on expected access to `[offset, offset + size)` (`size == 0`
  // means until EOF from `offset`). Best-effort hint; on Linux writable files,
  // `kDontNeed` flushes dirty pages in the range via `sync_file_range` so the
  // kernel can evict them, which may block on writeback (call `Sync()` if
  // durability is required). Returns `Status()` on success or
  // `Status::kUnimplemented` if unsupported by the backend/OS.
  virtual Status Advise(uint64_t /*offset*/, uint64_t /*size*/,
                        AccessHint /*hint*/) const {
    return Status::kUnimplemented;
  }

  // ---------------------------------------------------------------------------
  // Reading

  // Reads `to.size()` bytes starting at `offset`. Returns `to.size()` on
  // success, or an error. If EOF was reached first, the error is
  // `Status::kTruncated`, its payload is the number of bytes read before EOF,
  // and the first `payload()` bytes of `to` are valid. `EINTR` is retried
  // internally; if an unrecoverable OS error (such as `EIO`) occurs after a
  // partial transfer, the OS error status is returned. Arbitrary offsets,
  // concurrent calls, and discarding partial progress after errors require
  // `Capabilities().Has(FileCap::kRandomAccess)`; streams (`StreamFile`)
  // instead require sequential calls in order and consume partial progress.
  virtual Status Read(uint64_t offset, Span<uint8_t> to) const = 0;

  // Reads contiguous file bytes starting at `offset` into `spans` in order
  // (scatter read, e.g. `preadv` on POSIX). Prefer building `spans` via
  // `IOBatch`, which coalesces adjacent spans and bounds batch byte size for
  // parallel reads. Backends without native scatter reads (see
  // `FileCap::kNativeScatter`) read into `worker_buf`, which retains its
  // high-water-mark allocation, and copy from there; pass the same per-thread
  // instance to all calls to avoid allocations. Returns the total bytes read
  // across all `spans` on success, or an error (`Status::kTruncated` with the
  // bytes read before EOF as payload and the first `payload()` bytes across
  // `spans` valid, or an OS error status). As for `Read`, arbitrary offsets,
  // concurrent calls (with distinct `worker_buf` instances per thread), and
  // discarding partial progress after errors require
  // `Capabilities().Has(FileCap::kRandomAccess)`.
  virtual Status ReadSpans(uint64_t offset, const IOSpans& spans,
                           IOWorkerBuf& worker_buf) const;

  // Optional zero-copy read access: if `[offset, offset + size)` is already
  // resident in memory (e.g. `MemoryFile`, or a `FileView` over one), returns a
  // view of those `size` bytes (`span.data() != nullptr`, even when
  // `size == 0`) that remains valid while the `File` and its backing store are
  // alive and unmodified; otherwise (e.g. for OS files, if closed, or if
  // `[offset, offset + size)` exceeds `Size()`) returns a default `Span`
  // (`span.data() == nullptr`). Unlike `MapImmutable()`, this never maps an OS
  // file or initiates readahead.
  virtual Span<const uint8_t> MaybeSpan(uint64_t /*offset*/,
                                        size_t /*size*/) const {
    return Span<const uint8_t>();
  }

  // Mapping: the three functions below map the entire file (`[0, Size())`)
  // into memory (`map`) and return `Status(map.size())` (for an empty file,
  // `map` is empty and `Status(0)` is returned), or set `map` to nullptr and
  // return an error (`Status::kRange` if the file is larger than `size_t`,
  // `Status::kUnimplemented` if unsupported; see `FileCap::kMap*`). For POSIX
  // files whose `Size()` is 0, `VerifyEmpty` checks that they are actually
  // empty: pseudo-files such as /proc/self/status return
  // `Status::kUnimplemented` (or the probe's error) instead of an empty
  // mapping. Other backends return an empty mapping for size 0. We do not
  // support offsets because Windows requires them to be a multiple of the
  // allocation granularity, which is 64 KiB. Non-const because they may modify
  // internal state. For OS files, mappings remain valid after the `File` is
  // destroyed, do not grow with the file, and accesses past the end of a file
  // that was shrunk (via `SetSize` or another process, which may be untrusted)
  // fault (`SIGBUS` on POSIX; `F_SEAL_SHRINK` prevents this for memfd). For
  // `MemoryFile`, they are non-owning views that dangle once the backing store
  // (or a `MemoryFile` that owns it) is freed. Note that a mapping does not
  // protect against concurrent writers: when parsing data that an untrusted
  // process can modify, copy it first or seal the memfd (`F_SEAL_WRITE`).

  // Read-only mapping for files that the caller expects not to be modified
  // while mapped (e.g. inputs or model weights); otherwise, whether the changes
  // are visible is unspecified. Uses the cheapest and most widely supported
  // mechanism (POSIX `MAP_PRIVATE`, which also works on FUSE mounts and does
  // not block `F_SEAL_WRITE` or `execve` of the file).
  virtual Status MapImmutable(MapPtr& map) = 0;

  // Read-only mapping that is guaranteed to observe later writes by this or
  // other handles and processes (POSIX `MAP_SHARED`), e.g. for a shared-memory
  // buffer reused across sandbox calls. On POSIX, while this is mapped from an
  // `O_RDWR` descriptor (i.e. a write mode), `F_SEAL_WRITE` fails with `EBUSY`
  // and `execve` with `ETXTBSY`. Can fail on some file systems even if
  // `FileCap::kMapShared` is set (e.g. FUSE `direct_io`).
  virtual Status MapShared(MapPtr& map) {
    map.reset();
    return Status::kUnimplemented;
  }

  // Read-write shared mapping (POSIX `MAP_SHARED`): stores are visible to
  // `Read` and other mappings/processes and vice versa, e.g. for a codec in a
  // sandbox to write its output in place. Returns `IOCode::kWrongMode` if
  // opened with `FileMode::kRead`. To write beyond `Size()`, call `SetSize`
  // before mapping. Unlike `Write`, out-of-space conditions during stores fault
  // (`SIGBUS` on POSIX) instead of returning a `Status`; reserve space via
  // `Preallocate` (which suffices for memfd/tmpfs and local file systems
  // supporting `fallocate`). Stores are not guaranteed to be covered by `Sync`,
  // hence use `Write` for data that must be durable.
  virtual Status MapWritable(MutableMapPtr& map) {
    map.reset();
    return Status::kUnimplemented;
  }

  // ---------------------------------------------------------------------------
  // Writing

  // Writes all of `from` at `offset`, extending the file if necessary. Returns
  // `from.size()` on success, or an error (`IOCode::kWrongMode` if opened with
  // `FileMode::kRead`, `Status::kResource` if out of space). If an
  // unrecoverable OS error occurs after a partial write, the OS error status is
  // returned. Arbitrary offsets, concurrent calls, and discarding partial
  // progress after errors require `Capabilities().Has(FileCap::kRandomWrite)`;
  // streams (`StreamFile`) consume partial progress.
  virtual Status Write(uint64_t offset, Span<const uint8_t> from) = 0;

  // Writes `spans` in order to contiguous file bytes starting at `offset`
  // (gather write, e.g. `pwritev` on POSIX), extending the file if necessary.
  // Returns the total bytes written across all `spans` on success, or an error
  // (`IOCode::kWrongMode` if read-only, `Status::kResource` if out of space).
  // As for `Write`, arbitrary offsets, concurrent calls, and discarding partial
  // progress after errors require `Capabilities().Has(FileCap::kRandomWrite)`.
  virtual Status WriteSpans(uint64_t offset, const IOSpans& spans);

  // Optional zero-copy write access, only supported by writable in-memory
  // backends such as `MemoryFile`. If the backing store can accommodate
  // `[offset, offset + size)`, returns a mutable `Span` of those `size` bytes
  // (`span.data() != nullptr`, even when `size == 0`) and, when `size != 0`,
  // extends `Size()` to at least `offset + size`, as if `Write` had been
  // called. Otherwise (if closed, read-only, out of capacity, or on OS files),
  // returns a default `Span` (`span.data() == nullptr`). OS files do not map
  // spans on demand (each call would require a syscall, and growth would
  // require remapping, which invalidates earlier spans); instead, wrap a
  // `MapWritable` mapping in a `MemoryFile`. The span remains valid until the
  // `File` or its backing store is destroyed.
  virtual Span<uint8_t> MaybeSpanForWrite(uint64_t /*offset*/,
                                          size_t /*size*/) {
    return Span<uint8_t>();
  }

  // ---------------------------------------------------------------------------
  // Durability and lifetime

  // Flushes written data and metadata to persistent storage (e.g. `fsync` or
  // `FlushFileBuffers`; `Close()` does not do this). Returns `Status()` on
  // success, or an error (`IOCode::kWrongMode` if read-only,
  // `Status::kUnimplemented` if the backend does not support syncing).
  virtual Status Sync() { return Status::kUnimplemented; }

  // Closes the file and returns any error, e.g. from flushing buffered writes
  // on network file systems (does not `fsync`; call `Sync()` first if
  // durability is required). Subsequent operations fail with
  // `Status::kPrecondition` (`Close()` itself is idempotent). Not thread-safe
  // with respect to concurrent operations on the same `File`: callers must
  // synchronize externally so all other calls have completed before `Close()`.
  // The destructor closes the file if this was not called, ignoring errors.
  virtual Status Close() { return Status(); }
};

namespace detail {
// Shared atomic closed-state mixin for `File` implementations that do not own
// an OS handle (`MemoryFile`, `FileView`, and `StreamFile`, whose standard
// streams are borrowed).
class HWY_IO_DLLEXPORT ClosableFile : public File {
 public:
  // Unless overridden, unsupported, but still report `Status::kPrecondition`
  // after `Close()`, as for other operations.
  Status MapShared(MapPtr& map) override { return RejectMap(map); }
  Status MapWritable(MutableMapPtr& map) override { return RejectMap(map); }

  Status Close() override {
    closed_.store(true, std::memory_order_release);
    return Status();
  }

 protected:
  bool IsClosed() const { return closed_.load(std::memory_order_acquire); }
  Status CheckOpen() const {
    return IsClosed() ? Status(Status::kPrecondition) : Status();
  }

  template <typename T>
  Status RejectMap(AlignedPtr<T>& map) const {
    map.reset();
    const Status s = CheckOpen();
    return s.ok() ? Status(Status::kUnimplemented) : s;
  }

 private:
  std::atomic<bool> closed_{false};
};
}  // namespace detail

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_FILE_H_
