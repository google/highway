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

#ifndef HIGHWAY_HWY_IO_FILE_SYSTEM_H_
#define HIGHWAY_HWY_IO_FILE_SYSTEM_H_

// Pluggable file systems (the OS, virtual/test) and free functions that open,
// query, rename and remove files and directories at a `Path`, plus
// `MaybeOpenHandle` for existing POSIX descriptors (e.g. in sandboxes).
//
// Design notes:
// - Factory functions take `std::unique_ptr<File>&` because Highway has no
//   `StatusOr` template (avoiding exceptions and Abseil dependencies).
// - Not built on `std::filesystem`, which requires C++17, lacks vectored reads
//   and memory mapping, and Google-internal file systems require a wrapper
//   anyway (see `FileSystem`).

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "hwy/base.h"
#include "hwy/highway_export.h"
#include "hwy/io/file.h"
#include "hwy/io/path.h"
#include "hwy/status.h"

namespace hwy {

// Advises on expected access to `[offset, offset + size)` within `map`
// (`size == 0` means until `map.size()`), even after the `File` itself has been
// closed. Safe for all `File::Map*` backends: no-op `Status()` on non-owning
// `MemoryFile`/`FileView` mappings (avoiding `MADV_DONTNEED` on anonymous
// memory); page-aligned `madvise` / `PrefetchVirtualMemory` for OS files; and
// `Status::kUnimplemented` for mappings from other `FileSystem`s.
HWY_IO_DLLEXPORT Status AdviseMap(const MapPtr& map, uint64_t offset,
                                  uint64_t size, AccessHint hint);
// Same, for mappings returned by `File::MapWritable`.
HWY_IO_DLLEXPORT Status AdviseMap(const MutableMapPtr& map, uint64_t offset,
                                  uint64_t size, AccessHint hint);

// Directory entry returned by `ListDir`.
struct DirEntry {
  // UTF-8 name relative to the listed directory, i.e. without separators; use
  // `dir.Join(name)` to obtain its pathname.
  std::string name;
  // Size in bytes if `kind == EntryKind::kFile`, otherwise 0.
  uint64_t size = 0;
  // Allows recursing into subdirectories without additional `StatFile` calls.
  // Symbolic links are reported as `kSymlink` rather than followed, so
  // recursion can skip them (as `gzip -r` does) and cannot cycle; call
  // `StatFile` on the joined pathname to follow them.
  EntryKind kind = EntryKind::kOther;
};

// Options for `MaybeOpenFile` / `OpenFileOrAbort` / `FileSystem::Open` /
// `MaybeOpenHandle`.
// Bundled into a struct so future options (e.g. deadlines or locking) are easy
// to add.
struct OpenOptions {
  // Implicit from `FileMode` so `MaybeOpenFile(pathname, FileMode::kRead,
  // file)` remains concise when defaults suffice.
  constexpr OpenOptions(FileMode file_mode,  // NOLINT
                        AccessHint access_hint = AccessHint::kNormal)
      : mode(file_mode), hint(access_hint) {}

  FileMode mode;
  AccessHint hint;
};

namespace detail {
struct FileSystemList;
}  // namespace detail

// Backend for a set of paths, e.g. the OS (POSIX or Windows), network file
// systems, or virtual/test file systems. Callers typically use the free
// functions below (`MaybeOpenFile`, `StatFile`, `ListDir` etc.), which dispatch
// to the most recently added `FileSystem` whose `Claims` returns true, or the
// OS file system, which is the tail of the list and claims all paths.
//
// The free functions validate arguments and normalize results (e.g. clear
// outputs on error), so implementations need not; they reject empty paths with
// `Status::kNotFound` (or empty `FileCaps`), hence `Claims` and all other
// methods may assume `pathname` (or `dir`) is non-empty. Except for `Claims`,
// `Open` and `Stat`, the default implementations return
// `Status::kUnimplemented` (or report no capabilities). Because `Claims`
// decides ownership, `Status::kUnimplemented` returned by a claiming file
// system is reported to the caller rather than falling through to the OS.
class HWY_IO_DLLEXPORT FileSystem {
 public:
  virtual ~FileSystem() = default;

  FileSystem(const FileSystem&) = delete;
  FileSystem& operator=(const FileSystem&) = delete;

  // Returns whether this file system handles `pathname`. Must be thread-safe
  // and cheap (called for every operation), e.g. a prefix comparison. To
  // disable a registered file system (e.g. between tests), return false.
  virtual bool Claims(const Path& pathname) const = 0;

  // Returns the `File::Capabilities()` that files at `pathname` would have when
  // opened by this file system (see `PathCapabilities`). Must not access the
  // file system. The default returns no capabilities, meaning unknown; the OS
  // file systems always answer.
  virtual FileCaps Capabilities(const Path& pathname) const;

  // Opens `pathname` according to `options` and sets `file` (see
  // `MaybeOpenFile`).
  virtual Status Open(const Path& pathname, OpenOptions options,
                      std::unique_ptr<File>& file) = 0;

  // Sets `out` for the entry at `pathname` (see `StatFile`), including
  // directories and other kinds. There is deliberately no default: emulating it
  // via `Open` would misreport directories. Backends without a metadata query
  // can open the file with `FileMode::kRead` and query `Size()`, but must not
  // treat `Status::kPermission` as existence.
  virtual Status Stat(const Path& pathname, FileStat& out) = 0;

  // See `RenameFile`; only called if this file system claims both paths.
  virtual Status Rename(const Path& from, const Path& to);

  // See `RemoveFile`.
  virtual Status Remove(const Path& pathname);

  // Appends entries of directory `dir` to `out` (see `ListDir`).
  virtual Status ListDir(const Path& dir, std::vector<DirEntry>& out);

  // See `CreateDir`.
  virtual Status CreateDir(const Path& dir);

  // See `RemoveDir`.
  virtual Status RemoveDir(const Path& dir);

 protected:
  // constexpr enables constant initialization of static instances.
  constexpr FileSystem() noexcept = default;

 private:
  friend struct detail::FileSystemList;
  FileSystem* next_ = nullptr;  // Set by `AddFileSystem`.
};

// Prepends `fs` to the process-global lock-free list consulted by the free
// functions below (most recently added first, before the OS file system).
// `fs` must not already be registered, and must outlive the process or all
// subsequent file calls (typically static storage, registered from a static
// initializer of an alwayslink backend library). There is deliberately no
// removal because concurrent calls may be traversing the list; to disable a
// file system, have its `Claims` return false.
HWY_IO_DLLEXPORT void AddFileSystem(FileSystem* fs);

// Opens `pathname` according to `options`, setting `file` on success or to
// nullptr on failure. Not named `OpenFile` to avoid a conflict with the
// Windows.h macro. Returns `Status()` on success, or an error status whose
// payload is the OS error (errno or GetLastError) or backend error code
// (`Status::kNotFound` if `pathname.empty()`).
HWY_IO_DLLEXPORT Status MaybeOpenFile(const Path& pathname, OpenOptions options,
                                      std::unique_ptr<File>& file);

// As `MaybeOpenFile`, but returns the file and aborts on failure.
inline std::unique_ptr<File> OpenFileOrAbort(const Path& pathname,
                                             OpenOptions options) {
  std::unique_ptr<File> file;
  const Status status = MaybeOpenFile(pathname, options, file);
  if (!status.ok()) {
    HWY_ABORT("Failed to open %s for %s: %s", pathname.c_str(),
              FileModeName(options.mode), status.Name());
  }
  return file;
}

// Whether a `File` created from an existing descriptor closes it.
enum class HandleOwnership : uint8_t {
  // The caller retains ownership: `File::Close` and the destructor only detach
  // (subsequent operations still fail with `Status::kPrecondition`). For
  // descriptors managed elsewhere, e.g. by a sandbox (`sapi::v::Fd`) or C API.
  kBorrow,
  // The `File` closes the descriptor, as for `MaybeOpenFile`.
  kTake
};

// Sets `file` to a `File` over the already-open POSIX descriptor `handle`
// (`intptr_t` so that a Windows `HANDLE` can be supported later), or to nullptr
// on failure, e.g. for descriptors passed into a sandbox. Accepts the same
// kinds as `MaybeOpenFile` (regular files including memfd and Linux POSIX
// shared memory, plus character devices such as /dev/null for write modes).
// Note that macOS `shm_open` descriptors are not regular files
// (`pread`/`pwrite` fail with `ESPIPE`) and are therefore rejected; `mmap` them
// directly instead. `options.mode` must be compatible with the descriptor's
// access mode (write modes require `O_WRONLY` or `O_RDWR`; `Read` and mappings
// of the former return `IOCode::kWrongMode`). `FileMode::kWriteTruncate`
// truncates regular files to zero size. `FileMode::kWriteNew` cannot be
// enforced for existing descriptors and is rejected, as are write modes on
// `O_APPEND` descriptors (Linux `pwrite` would ignore the offset). Ownership
// only transfers on success: after a failure, the caller remains responsible
// for closing `handle`. Returns `Status()` on success, or an error
// (`Status::kArgument` if `handle` is not a valid descriptor, of an unsupported
// kind, or as above; `IOCode::kWrongMode` if the access mode is incompatible;
// `Status::kUnimplemented` on non-POSIX systems or if `HWY_IO_HAVE_OS_FILES` is
// 0).
HWY_IO_DLLEXPORT Status MaybeOpenHandle(intptr_t handle, OpenOptions options,
                                        HandleOwnership ownership,
                                        std::unique_ptr<File>& file);

// Returns the `File::Capabilities()` that a file at `pathname` would have
// after opening (with any `FileMode`), without accessing the file system,
// e.g. to choose an output strategy before creating the file. The OS file
// systems always answer, so the result is only empty (meaning unknown) for
// custom `FileSystem`s that do not override `Capabilities`, for empty
// `pathname`, or if `HWY_IO_HAVE_OS_FILES` is 0 (when OS paths cannot be
// opened at all). For POSIX character devices (e.g. /dev/null), this reports
// the capabilities of regular files, which are a superset of the actual
// `File::Capabilities()`.
HWY_IO_DLLEXPORT FileCaps PathCapabilities(const Path& pathname);

// Queries `out` for the entry (of any kind) at `pathname` without opening a
// `File` handle where supported, like `std::filesystem::status`. Follows
// symlinks (dangling ones are `Status::kNotFound`) and succeeds even when the
// entry lacks read data permissions on OS filesystems (like POSIX `stat`).
// Returns `Status(out.size)` on success, or resets `out` and returns an error
// (`Status::kNotFound` if missing or `pathname.empty()`, `Status::kPermission`
// if parent traversal or metadata access is denied, `Status::kRange` if size
// exceeds `Status::kMaxPayload`). Thus `StatFile(pathname, stat).ok()`
// indicates whether an entry existed at `pathname` when called, and `stat.kind`
// whether it is a file, directory, or something else such as a FIFO.
HWY_IO_DLLEXPORT Status StatFile(const Path& pathname, FileStat& out);

// Returns whether an entry of any kind (after following symlinks) existed at
// `pathname` when called, like Python's `os.path.exists`. Also false if that
// cannot be determined, e.g. due to `Status::kPermission`; call `StatFile` to
// distinguish such errors from `Status::kNotFound` or to obtain the kind.
inline bool PathExists(const Path& pathname) {
  FileStat stat;
  return StatFile(pathname, stat).ok();
}

// Atomically renames `from` to `to` (replacing `to` if it already exists, e.g.
// for temp-then-rename output). Renames between different file systems fail
// with `Status::kArgument`, and empty paths with `Status::kNotFound`.
HWY_IO_DLLEXPORT Status RenameFile(const Path& from, const Path& to);

// Removes a file at `pathname` (not directories, which fail with
// `Status::kArgument`; see `RemoveDir`). Empty paths fail with
// `Status::kNotFound`. Not named `DeleteFile` to avoid a
// conflict with the Windows.h macro.
HWY_IO_DLLEXPORT Status RemoveFile(const Path& pathname);

// Sets `out` to the entries of directory `dir` (excluding "." and ".."), in
// unspecified order, as produced by `readdir`, `FindNextFileW`, or Python's
// `os.listdir`. There is no filter parameter; callers can filter by
// `DirEntry::name` or `kind`. `out` is cleared on entry, hence empty after
// errors. Returns `Status(out.size())` on success, or an error
// (`Status::kNotFound` if `dir` does not exist, `Status::kArgument` if it is
// not a directory, `Status::kPermission` if listing is denied).
HWY_IO_DLLEXPORT Status ListDir(const Path& dir, std::vector<DirEntry>& out);

// Creates a single directory `dir`, whose parent must exist. Returns `Status()`
// on success, or an error (`IOCode::kExists` if `dir` already exists, whether
// as a directory or another kind of entry; `Status::kNotFound` if the parent
// does not exist, `Status::kPermission` if denied).
HWY_IO_DLLEXPORT Status CreateDir(const Path& dir);

// Removes the empty directory `dir`. Returns `Status()` on success, or an error
// (`Status::kNotFound` if missing, `Status::kArgument` if not a directory,
// `Status::kPrecondition` if not empty, `Status::kPermission` if denied).
HWY_IO_DLLEXPORT Status RemoveDir(const Path& dir);

// Returns the OS directory for temporary files (`TMPDIR` or "/tmp" on POSIX,
// `GetTempPathW` on Windows) without a trailing separator, or an empty `Path`
// on failure or if `HWY_IO_HAVE_OS_FILES` is 0. Does not check existence.
HWY_IO_DLLEXPORT Path TempDir();

}  // namespace hwy

#endif  // HIGHWAY_HWY_IO_FILE_SYSTEM_H_
