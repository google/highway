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

#include "hwy/base.h"

#if HWY_OS_WIN
#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/io/file.h"
#include "hwy/io/file_system.h"
#include "hwy/io/io_internal.h"
#include "hwy/io/path.h"
#include "hwy/io/stream_file.h"
#include "hwy/profiler.h"
#include "hwy/status.h"

// Without OS files, file_system.cc defines the stub OS file system.
#if HWY_IO_HAVE_OS_FILES

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>  // FSCTL_SET_SPARSE, FSCTL_QUERY_ALLOCATED_RANGES

#ifndef ERROR_FILE_SYSTEM_LIMITATION
#define ERROR_FILE_SYSTEM_LIMITATION 665L
#endif
#ifndef FIND_FIRST_EX_LARGE_FETCH
#define FIND_FIRST_EX_LARGE_FETCH 2
#endif

namespace hwy {
namespace {

// Maps GetLastError to a general status code, with the error as the payload.
Status WinStatus(DWORD err) {
  Status::Code code = Status::kOSError;
  switch (err) {
    case ERROR_OPERATION_ABORTED:
    case ERROR_CANCELLED:
      code = Status::kCancelled;
      break;
    case WAIT_TIMEOUT:
    case ERROR_SEM_TIMEOUT:
      code = Status::kTimeout;
      break;
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      code = Status::kNotFound;
      break;
    case ERROR_FILE_EXISTS:
    case ERROR_ALREADY_EXISTS:
      code = static_cast<Status::Code>(IOCode::kExists);
      break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
      code = Status::kPermission;
      break;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_TOO_MANY_OPEN_FILES:
      code = Status::kResource;
      break;
    case ERROR_DIR_NOT_EMPTY:
      code = Status::kPrecondition;
      break;
    case ERROR_HANDLE_EOF:
      code = Status::kTruncated;
      break;
    case ERROR_INVALID_PARAMETER:
    case ERROR_INVALID_HANDLE:
    case ERROR_DIRECTORY:
    case ERROR_NOT_SAME_DEVICE:
    case ERROR_INVALID_NAME:
    case ERROR_FILENAME_EXCED_RANGE:
    case ERROR_BAD_PATHNAME:
    case ERROR_CANT_RESOLVE_FILENAME:
      code = Status::kArgument;
      break;
    case ERROR_CALL_NOT_IMPLEMENTED:
    case ERROR_NOT_SUPPORTED:
    case ERROR_NEGATIVE_SEEK:
    case ERROR_SEEK_ON_DEVICE:
      code = Status::kUnimplemented;
      break;
    default:
      break;
  }
  return Status::Error(code, static_cast<uint64_t>(err));
}

void UnmapFileWin(void* opaque, void* ptr) {
  detail::RecordSyscall();
  HWY_ASSERT(UnmapViewOfFile(ptr));
  detail::RecordSyscall();
  HWY_ASSERT(CloseHandle(static_cast<HANDLE>(opaque)));
}

// Converts the UTF-8 `pathname` to UTF-16 (including the terminating null),
// or returns an empty vector on failure.
std::vector<wchar_t> Utf8ToWide(const Path& pathname) {
  const std::string& utf8 = pathname.path;
  std::vector<wchar_t> wide;
  if (!SumFits<int>(utf8.size(), size_t{1})) return wide;
  const int len_with_null = SafeAdd<int>(utf8.size(), size_t{1});
  const int num_wide = MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(), len_with_null, nullptr, 0);
  if (num_wide <= 0) return wide;
  wide.resize(static_cast<size_t>(num_wide));
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(),
                          len_with_null, wide.data(), num_wide) != num_wide) {
    wide.clear();
  }
  return wide;
}

// Converts a null-terminated UTF-16 string to UTF-8, replacing unpaired
// surrogates (which NTFS allows in names) with U+FFFD. Returns an empty string
// on failure.
std::string WideToUtf8(const wchar_t* wide) {
  const int len_with_null = WideCharToMultiByte(
      CP_UTF8, /*dwFlags=*/0, wide, -1, nullptr, 0, nullptr, nullptr);
  if (len_with_null <= 1) return std::string();
  std::string utf8(static_cast<size_t>(len_with_null), '\0');
  if (WideCharToMultiByte(CP_UTF8, /*dwFlags=*/0, wide, -1, &utf8[0],
                          len_with_null, nullptr, nullptr) != len_with_null) {
    return std::string();
  }
  utf8.resize(static_cast<size_t>(len_with_null - 1));
  return utf8;
}

// Manual-reset event for waiting on one synchronous overlapped request per
// thread. Cached in thread_local storage to avoid CreateEventW/CloseHandle
// syscalls on every Read/Write; safe because each call waits via
// GetOverlappedResult before returning.
class OverlappedEvent {
 public:
  OverlappedEvent()
      : event_(CreateEventW(/*lpEventAttributes=*/nullptr,
                            /*bManualReset=*/TRUE, /*bInitialState=*/FALSE,
                            /*lpName=*/nullptr)),
        err_(event_ == nullptr ? GetLastError() : ERROR_SUCCESS) {}
  ~OverlappedEvent() {
    if (event_ != nullptr) HWY_ASSERT(CloseHandle(event_));
  }
  OverlappedEvent(const OverlappedEvent&) = delete;
  OverlappedEvent& operator=(const OverlappedEvent&) = delete;

  HANDLE Get() const { return event_; }
  DWORD Error() const { return err_; }

 private:
  HANDLE event_;
  DWORD err_;
};

// Closes the cached event handle when the thread exits (avoiding kernel handle
// leaks across transient threads). Note: if `hwy_io` is built as a DLL, it
// must not be `FreeLibrary`-unloaded while worker threads that called into it
// remain alive, lest their TLS destructor callbacks dangle on non-UCRT
// runtimes.
const OverlappedEvent& GetThreadEvent() {
  thread_local const OverlappedEvent event;
  return event;
}

enum class TransferOp { kRead, kWrite };

// Issues a single ReadFile or WriteFile at `offset` and waits for it to
// complete. Sets `got` to the number of bytes transferred. Returns Ok, or an
// error (kTruncated at EOF).
Status TransferAt(HANDLE file, HANDLE event, TransferOp op, uint64_t offset,
                  void* bytes, DWORD want, DWORD* got) {
  OVERLAPPED overlapped = {};
  overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFu);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  // ReadFile/WriteFile reset the event to non-signaled when they begin.
  overlapped.hEvent = event;
  // For overlapped handles, the byte count must be obtained from
  // GetOverlappedResult, hence pass nullptr.
  detail::RecordSyscall();
  const BOOL ok = op == TransferOp::kWrite
                      ? WriteFile(file, bytes, want, nullptr, &overlapped)
                      : ReadFile(file, bytes, want, nullptr, &overlapped);
  *got = 0;
  if (!ok) {
    const DWORD err = GetLastError();
    if (err != ERROR_IO_PENDING) {
      if (err == ERROR_HANDLE_EOF) {
        // Populate `*got` from `overlapped.InternalHigh` if a synchronous read
        // transferred partial bytes before hitting EOF.
        (void)GetOverlappedResult(file, &overlapped, got, /*bWait=*/FALSE);
      }
      return WinStatus(err);  // incl. EOF
    }
  }
  if (!GetOverlappedResult(file, &overlapped, got, /*bWait=*/TRUE)) {
    return WinStatus(GetLastError());
  }
  return Status::Ok(*got);
}

// Issues a synchronous `DeviceIoControl` on a `FILE_FLAG_OVERLAPPED` handle.
BOOL SyncDeviceIoControl(HANDLE file, HANDLE event, DWORD code, void* in_buf,
                         DWORD in_size, void* out_buf, DWORD out_size,
                         DWORD* bytes_returned) {
  OVERLAPPED overlapped = {};
  overlapped.hEvent = event;
  *bytes_returned = 0;
  detail::RecordSyscall();
  const BOOL ok =
      DeviceIoControl(file, code, in_buf, in_size, out_buf, out_size,
                      /*lpBytesReturned=*/nullptr, &overlapped);
  if (!ok && GetLastError() != ERROR_IO_PENDING) {
    return FALSE;
  }
  return GetOverlappedResult(file, &overlapped, bytes_returned,
                             /*bWait=*/TRUE);
}

// Returns true iff `[0, file_size)` has no unallocated sparse holes. NTFS
// coalesces adjacent allocated extents, so a fully written file returns a
// single `[0, file_size)` range; if more than 8 ranges exist
// (`ERROR_MORE_DATA`), returning false conservatively keeps the sparse
// attribute.
bool HasNoSparseHoles(HANDLE file, HANDLE event, LONGLONG file_size) {
  if (file_size <= 0) return true;
  FILE_ALLOCATED_RANGE_BUFFER query = {};
  query.Length.QuadPart = file_size;
  FILE_ALLOCATED_RANGE_BUFFER ranges[8] = {};
  DWORD bytes_returned = 0;
  if (!SyncDeviceIoControl(file, event, FSCTL_QUERY_ALLOCATED_RANGES, &query,
                           sizeof(query), ranges, sizeof(ranges),
                           &bytes_returned)) {
    return false;
  }
  const size_t count = bytes_returned / sizeof(ranges[0]);
  LONGLONG covered = 0;
  for (size_t i = 0; i < count && covered < file_size; ++i) {
    if (ranges[i].FileOffset.QuadPart != covered ||
        ranges[i].Length.QuadPart <= 0) {
      return false;
    }
    covered += ranges[i].Length.QuadPart;
  }
  return covered >= file_size;
}

// Independent of `FileMode` (see `FileCap`). No `kPreallocate` because sparse
// handling in `SetSize` already avoids synchronous zero-filling.
FileCaps WinFileCaps() {
  return FileCap::kRandomAccess | FileCap::kRandomWrite | FileCap::kExtend |
         FileCap::kSync | FileCap::kMapImmutable | FileCap::kMapShared |
         FileCap::kMapWritable;
}

// Sets `out` from `GetFileInformationByHandle` of a disk file or directory.
Status FileStatFrom(const BY_HANDLE_FILE_INFORMATION& info, FileStat& out) {
  out.device = info.dwVolumeSerialNumber;
  out.inode = (static_cast<uint64_t>(info.nFileIndexHigh) << 32) |
              static_cast<uint64_t>(info.nFileIndexLow);
  if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    out.kind = EntryKind::kDir;
    out.size = 0;
    return Status(0);
  }
  const uint64_t size = (static_cast<uint64_t>(info.nFileSizeHigh) << 32) |
                        static_cast<uint64_t>(info.nFileSizeLow);
  if (size > Status::kMaxPayload) return Status::kRange;
  out.size = size;
  out.kind = EntryKind::kFile;
  return Status(size);
}

// Returns whether `attrs` (from `GetFileAttributesW`) denote a directory.
bool IsDirectoryAttrs(DWORD attrs) {
  return attrs != INVALID_FILE_ATTRIBUTES &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

}  // namespace

// Opened with FILE_FLAG_OVERLAPPED so that concurrent Read/Write calls on the
// same handle can proceed in parallel. Otherwise, the I/O manager would
// serialize them via a per-file-object lock, even with explicit offsets. Each
// call still waits for its own completion, hence the interface is synchronous.
class FileWin : public File {
 public:
  FileWin(HANDLE hFile, FileMode mode)
      : hFile_(hFile), read_only_(mode == FileMode::kRead) {
    HWY_ASSERT(hFile != INVALID_HANDLE_VALUE);
  }
  ~FileWin() override { (void)Close(); }

  Status Close() override {
    if (hFile_ == INVALID_HANDLE_VALUE) return Status();
    const HANDLE hFile = hFile_;
    hFile_ = INVALID_HANDLE_VALUE;
    // Clear the sparse attribute set in `SetSize` if no holes remain, avoiding
    // persistent sparse treatment by backup/copy/AV tools without forcing NTFS
    // to allocate clusters for intentionally sparse files.
    if (made_sparse_.exchange(false, std::memory_order_relaxed)) {
      const OverlappedEvent& event = GetThreadEvent();
      LARGE_INTEGER file_size;
      if (event.Get() != nullptr) {
        detail::RecordSyscall();
        if (GetFileSizeEx(hFile, &file_size) &&
            HasNoSparseHoles(hFile, event.Get(), file_size.QuadPart)) {
          FILE_SET_SPARSE_BUFFER unsparse = {FALSE};
          DWORD bytes_returned = 0;
          (void)SyncDeviceIoControl(
              hFile, event.Get(), FSCTL_SET_SPARSE, &unsparse, sizeof(unsparse),
              /*out_buf=*/nullptr, /*out_size=*/0, &bytes_returned);
        }
      }
    }
    detail::RecordSyscall();
    if (!CloseHandle(hFile)) return WinStatus(GetLastError());
    return Status();
  }

  FileCaps Capabilities() const override { return WinFileCaps(); }

  intptr_t Handle() const override {
    return hFile_ == INVALID_HANDLE_VALUE ? -1
                                          : reinterpret_cast<intptr_t>(hFile_);
  }

  Status Advise(uint64_t offset, uint64_t size,
                AccessHint /*hint*/) const override {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    if (!SumFits<int64_t>(offset, size)) return Status::kArgument;
    return Status::kUnimplemented;
  }

  // Overridden only to report closed/read-only/argument errors consistently.
  Status Preallocate(uint64_t offset, uint64_t size) override {
    const Status s = CheckWritable();
    if (!s.ok()) return s;
    if (!SumFits<int64_t>(offset, size)) return Status::kArgument;
    return Status::kUnimplemented;
  }

  Status Size() const override {
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    LARGE_INTEGER size;
    detail::RecordSyscall();
    if (!GetFileSizeEx(hFile_, &size)) return WinStatus(GetLastError());
    if (size.QuadPart < 0 ||
        static_cast<uint64_t>(size.QuadPart) > Status::kMaxPayload) {
      return Status::kRange;
    }
    return Status(static_cast<uint64_t>(size.QuadPart));
  }

  Status Stat(FileStat& out) const override {
    out = FileStat();
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    BY_HANDLE_FILE_INFORMATION info;
    detail::RecordSyscall();
    if (GetFileInformationByHandle(hFile_, &info) == 0) {
      return WinStatus(GetLastError());
    }
    const Status status = FileStatFrom(info, out);
    if (!status.ok()) out = FileStat();
    return status;
  }

  Status SetSize(uint64_t size) override {
    PROFILER_ZONE("IO.Win.SetSize");
    const Status s = CheckWritable();
    if (!s.ok()) return s;
    if (!ScalarFits<LONGLONG>(size) || size > Status::kMaxPayload) {
      return Status::kArgument;
    }
    const Status cur = Size();
    if (cur.ok()) {
      if (size == cur.size()) return Status(size);
      // On NTFS, extending EOF leaves Valid Data Length (VDL) unchanged, so
      // parallel out-of-order writes past VDL otherwise serialize while NTFS
      // zero-fills `[VDL, offset)`. Marking the file sparse avoids synchronous
      // zero-filling while preserving zero-reads on unwritten holes (unlike
      // `SetFileValidData`, which exposes stale disk clusters).
      if (size > cur.size() &&
          !sparse_checked_.load(std::memory_order_relaxed)) {
        const OverlappedEvent& event = GetThreadEvent();
        if (event.Get() != nullptr) {
          BY_HANDLE_FILE_INFORMATION file_info = {};
          detail::RecordSyscall();
          const BOOL got_info = GetFileInformationByHandle(hFile_, &file_info);
          if (!got_info ||
              (file_info.dwFileAttributes & FILE_ATTRIBUTE_SPARSE_FILE) == 0) {
            DWORD bytes_returned = 0;
            if (SyncDeviceIoControl(hFile_, event.Get(), FSCTL_SET_SPARSE,
                                    /*in_buf=*/nullptr, /*in_size=*/0,
                                    /*out_buf=*/nullptr, /*out_size=*/0,
                                    &bytes_returned)) {
              made_sparse_.store(true, std::memory_order_relaxed);
            }
          }
          sparse_checked_.store(true, std::memory_order_relaxed);
        }
      }
    }
    FILE_END_OF_FILE_INFO info;
    info.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
    detail::RecordSyscall();
    if (!SetFileInformationByHandle(hFile_, FileEndOfFileInfo, &info,
                                    sizeof(info))) {
      return WinStatus(GetLastError());
    }
    return Status(size);
  }

  Status Read(uint64_t offset, Span<uint8_t> to) const override {
    PROFILER_ZONE("IO.Win.Read");
    const Status s = CheckOpen();
    if (!s.ok()) return s;
    return Transfer(TransferOp::kRead, offset, to.size(), to.data());
  }

  Status Write(uint64_t offset, Span<const uint8_t> from) override {
    PROFILER_ZONE("IO.Win.Write");
    const Status s = CheckWritable();
    if (!s.ok()) return s;
    // WriteFile does not modify the buffer.
    return Transfer(TransferOp::kWrite, offset, from.size(),
                    const_cast<uint8_t*>(from.data()));
  }

  Status Sync() override {
    PROFILER_ZONE("IO.Win.Sync");
    const Status s = CheckWritable();
    if (!s.ok()) return s;
    detail::RecordSyscall();
    if (!FlushFileBuffers(hFile_)) return WinStatus(GetLastError());
    return Status();
  }

  // Views of local files are coherent with `ReadFile`/`WriteFile` and other
  // views, hence all three use the same mechanism. Copy-on-write
  // (`FILE_MAP_COPY`) would charge commit for the entire view.
  Status MapImmutable(MapPtr& map) override {
    PROFILER_ZONE("IO.Win.MapImmutable");
    return MapView(PAGE_READONLY, FILE_MAP_READ, map);
  }

  Status MapShared(MapPtr& map) override {
    PROFILER_ZONE("IO.Win.MapShared");
    return MapView(PAGE_READONLY, FILE_MAP_READ, map);
  }

  Status MapWritable(MutableMapPtr& map) override {
    PROFILER_ZONE("IO.Win.MapWritable");
    map.reset();
    const Status s = CheckWritable();
    if (!s.ok()) return s;
    return MapView(PAGE_READWRITE, FILE_MAP_WRITE, map);
  }

 private:
  Status CheckOpen() const {
    return hFile_ == INVALID_HANDLE_VALUE ? Status(Status::kPrecondition)
                                          : Status();
  }

  Status CheckWritable() const {
    if (hFile_ == INVALID_HANDLE_VALUE) return Status::kPrecondition;
    if (read_only_) return IOCode::kWrongMode;
    return Status();
  }

  // Maps `[0, Size())` with `protect` and view `access` into `map` (`MapPtr`
  // or `MutableMapPtr`).
  template <typename T>
  Status MapView(DWORD protect, DWORD access, AlignedPtr<T>& map) {
    map.reset();
    const Status size_status = Size();
    if (!size_status.ok()) return size_status;
    const uint64_t file_size = size_status.size();
    if (!ScalarFits<size_t>(file_size)) return Status::kRange;
    if (file_size == 0) return Status(0);
    detail::RecordSyscall();
    const HANDLE hMapping = CreateFileMappingW(
        hFile_, /*lpFileMappingAttributes=*/nullptr, protect,
        /*dwMaximumSizeHigh=*/0, /*dwMaximumSizeLow=*/0, /*lpName=*/nullptr);
    if (hMapping == nullptr) return WinStatus(GetLastError());

    detail::RecordSyscall();
    void* ptr = MapViewOfFile(hMapping, access, /*dwFileOffsetHigh=*/0,
                              /*dwFileOffsetLow=*/0,
                              /*dwNumberOfBytesToMap=*/0);
    if (ptr == nullptr) {
      const DWORD err = GetLastError();
      HWY_ASSERT(CloseHandle(hMapping));
      return WinStatus(err);
    }
    const size_t mapping_size = SafeCast<size_t>(file_size);
    map = AlignedPtr<T>(static_cast<T*>(ptr), mapping_size, &UnmapFileWin,
                        static_cast<void*>(hMapping));
    return Status(mapping_size);
  }

  Status Transfer(TransferOp op, uint64_t offset, size_t size,
                  void* bytes) const {
    // The returned `Status` payload must be able to represent `size`.
    if (!SumFits<int64_t>(offset, size) ||
        !SumWithin(uint64_t{0}, size, Status::kMaxPayload)) {
      return Status::kArgument;
    }
    const size_t total = size;
    if (size == 0) return Status(0);
    // Thread-local cached event suffices because chunks within a call execute
    // sequentially and wait before returning.
    const OverlappedEvent& event = GetThreadEvent();
    if (event.Get() == nullptr) return WinStatus(event.Error());
    uint8_t* pos = static_cast<uint8_t*>(bytes);
    while (size != 0) {
      // Size argument is 32-bit, so cap to 1 GiB per call.
      const DWORD want = static_cast<DWORD>(HWY_MIN(size, size_t{1} << 30));
      DWORD got = 0;
      const Status status =
          TransferAt(hFile_, event.Get(), op, offset, pos, want, &got);
      if (!status.ok()) {
        // As on POSIX, the payload for EOF is the number of bytes transferred
        // (GetOverlappedResult can return ERROR_HANDLE_EOF with `got > 0` for
        // a partial read crossing EOF).
        if (op == TransferOp::kRead && status.Is(Status::kTruncated)) {
          HWY_ASSERT(got <= size);
          if (got != 0) {
            detail::RecordRead(got);
            if (got < want) detail::RecordShortRead();
          }
          return Status(Status::kTruncated,
                        SafeAdd<uint64_t>(total - size, got));
        }
        return status;
      }
      // Zero bytes without error should not happen; avoid an infinite loop.
      if (got == 0) {
        if (op == TransferOp::kWrite) {
          return Status(IOCode::kNoProgress, total - size);
        }
        return Status(Status::kTruncated, total - size);
      }
      HWY_ASSERT(got <= size);
      if (op == TransferOp::kWrite) {
        detail::RecordWrite(got);
      } else {
        detail::RecordRead(got);
        if (got < want) detail::RecordShortRead();
      }
      pos += got;
      offset = SafeAdd<uint64_t>(offset, got);
      size -= got;
    }
    return Status(total);
  }

  HANDLE hFile_ = INVALID_HANDLE_VALUE;
  bool read_only_ = false;
  std::atomic<bool> sparse_checked_{false};
  std::atomic<bool> made_sparse_{false};
};  // FileWin

namespace {

// Maps file-onto-directory and directory-onto-file rename failures (which
// report `ERROR_ACCESS_DENIED`, `ERROR_ALREADY_EXISTS`, etc. on Windows
// depending on the API and filesystem) to `ERROR_DIRECTORY`
// (`Status::kArgument`), matching POSIX `EISDIR` / `ENOTDIR`.
DWORD MapDirectoryRenameMismatch(DWORD err, const wchar_t* wide_from,
                                 const wchar_t* wide_to) {
  if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND ||
      err == ERROR_NOT_SAME_DEVICE) {
    return err;
  }
  detail::RecordSyscall();
  const DWORD to_attrs = GetFileAttributesW(wide_to);
  if (to_attrs == INVALID_FILE_ATTRIBUTES) return err;
  detail::RecordSyscall();
  const DWORD from_attrs = GetFileAttributesW(wide_from);
  if (from_attrs != INVALID_FILE_ATTRIBUTES &&
      ((to_attrs ^ from_attrs) & FILE_ATTRIBUTE_DIRECTORY) != 0) {
    return ERROR_DIRECTORY;
  }
  return err;
}

// Returns `ERROR_DIRECTORY` instead of `ERROR_ACCESS_DENIED` if `wide` is a
// directory, because Windows reports the latter when opening or deleting a
// directory as a file.
DWORD MapAccessDeniedForDirectory(DWORD err, const wchar_t* wide) {
  if (err == ERROR_ACCESS_DENIED) {
    detail::RecordSyscall();
    if (IsDirectoryAttrs(GetFileAttributesW(wide))) return ERROR_DIRECTORY;
  }
  return err;
}

class WindowsFileSystem final : public FileSystem {
 public:
  bool Claims(const Path& /*pathname*/) const override { return true; }

  FileCaps Capabilities(const Path& /*pathname*/) const override {
    return WinFileCaps();
  }

  // FILE_FLAG_NO_BUFFERING is intentionally omitted for the same OS page-cache
  // and alignment reasons as O_DIRECT on POSIX (see posix_file.cc).
  Status Open(const Path& pathname, OpenOptions options,
              std::unique_ptr<File>& file) override {
    PROFILER_ZONE("IO.Win.Open");
    const std::vector<wchar_t> wide = Utf8ToWide(pathname);
    if (wide.empty()) return Status::kArgument;

    const FileMode mode = options.mode;
    const bool is_read = mode == FileMode::kRead;
    const DWORD access =
        is_read ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    // Allow other handles to read, write and delete/rename, as on POSIX, where
    // there is no mandatory locking. Otherwise, opening a file that another
    // process is writing fails with ERROR_SHARING_VIOLATION.
    const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    const DWORD create = is_read                            ? OPEN_EXISTING
                         : mode == FileMode::kWriteTruncate ? CREATE_ALWAYS
                         : mode == FileMode::kWriteNew      ? CREATE_NEW
                                                            : OPEN_ALWAYS;
    DWORD hint_flag = 0;
    if (options.hint == AccessHint::kSequential ||
        options.hint == AccessHint::kWillNeed) {
      hint_flag = FILE_FLAG_SEQUENTIAL_SCAN;
    } else if (options.hint == AccessHint::kRandom) {
      hint_flag = FILE_FLAG_RANDOM_ACCESS;
    }
    const DWORD flags =
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | hint_flag;
    detail::RecordSyscall();
    const HANDLE hFile = CreateFileW(wide.data(), access, share,
                                     /*lpSecurityAttributes=*/nullptr, create,
                                     flags, /*hTemplateFile=*/nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
      const DWORD err = GetLastError();
      return WinStatus(MapAccessDeniedForDirectory(err, wide.data()));
    }
    // CreateFileW without FILE_FLAG_BACKUP_SEMANTICS already rejects
    // directories, but also reject devices/pipes, which lack random access.
    if (GetFileType(hFile) != FILE_TYPE_DISK) {
      (void)CloseHandle(hFile);
      return WinStatus(ERROR_INVALID_PARAMETER);
    }
    file = std::make_unique<FileWin>(hFile, mode);
    return Status();
  }

  Status Stat(const Path& pathname, FileStat& out) override {
    PROFILER_ZONE("IO.Win.Stat");
    const std::vector<wchar_t> wide = Utf8ToWide(pathname);
    if (wide.empty()) return Status::kArgument;
    // `dwDesiredAccess = 0` queries metadata without requesting data
    // permissions (succeeding on DACL-read-denied or exclusively-opened files,
    // like POSIX `stat`). `FILE_FLAG_BACKUP_SEMANTICS` is required to open
    // directories, and omitting `FILE_FLAG_OPEN_REPARSE_POINT` follows
    // symlinks. Never treat `ERROR_ACCESS_DENIED` as existence: ACL-denied
    // parent directories also return it.
    detail::RecordSyscall();
    const HANDLE hFile =
        CreateFileW(wide.data(), /*dwDesiredAccess=*/0,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    /*lpSecurityAttributes=*/nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_BACKUP_SEMANTICS,
                    /*hTemplateFile=*/nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return WinStatus(GetLastError());
    // Devices such as `NUL`/`CON` and pipes are neither files nor directories.
    detail::RecordSyscall();
    if (GetFileType(hFile) != FILE_TYPE_DISK) {
      detail::RecordSyscall();
      (void)CloseHandle(hFile);
      out.kind = EntryKind::kOther;
      return Status(0);
    }
    BY_HANDLE_FILE_INFORMATION info;
    detail::RecordSyscall();
    if (GetFileInformationByHandle(hFile, &info) == 0) {
      const DWORD err = GetLastError();
      detail::RecordSyscall();
      (void)CloseHandle(hFile);
      return WinStatus(err);
    }
    detail::RecordSyscall();
    (void)CloseHandle(hFile);
    return FileStatFrom(info, out);
  }

  Status Rename(const Path& from, const Path& to) override {
    PROFILER_ZONE("IO.Win.Rename");
    const std::vector<wchar_t> wide_from = Utf8ToWide(from);
    const std::vector<wchar_t> wide_to = Utf8ToWide(to);
    if (wide_from.empty() || wide_to.empty()) return Status::kArgument;

    // Prefer `SetFileInformationByHandle` with `FileRenameInfoEx` (Windows 10
    // 1607+ on NTFS/ReFS) and `FILE_RENAME_FLAG_POSIX_SEMANTICS`: unlike
    // `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`, this atomically unlinks and
    // replaces `to` even while `to` has open `FILE_SHARE_DELETE` handles (note:
    // user-mode `FileRenameInfoEx` still honors `FILE_ATTRIBUTE_READONLY` on
    // `to`). Defined locally so older `_WIN32_WINNT` SDK targets still compile.
    struct FileRenameInfoExLayout {
      DWORD Flags;
      HANDLE RootDirectory;
      DWORD FileNameLength;
      WCHAR FileName[1];
    };
    static_assert(sizeof(FileRenameInfoExLayout) == sizeof(FILE_RENAME_INFO),
                  "FILE_RENAME_INFO layout mismatch");
    constexpr DWORD kReplaceIfExists = 0x00000001u;
    constexpr DWORD kPosixSemantics = 0x00000002u;
    const auto kFileRenameInfoEx = static_cast<FILE_INFO_BY_HANDLE_CLASS>(22);

    const size_t wide_bytes = wide_to.size() * sizeof(wchar_t);
    const size_t name_bytes = (wide_to.size() - 1) * sizeof(wchar_t);
    const size_t header_bytes = offsetof(FileRenameInfoExLayout, FileName);
    if (!SumFits<DWORD>(header_bytes, wide_bytes)) return Status::kArgument;
    const size_t total_bytes = header_bytes + wide_bytes;

    detail::RecordSyscall();
    const HANDLE hFrom =
        CreateFileW(wide_from.data(), DELETE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    /*lpSecurityAttributes=*/nullptr, OPEN_EXISTING,
                    FILE_FLAG_OPEN_REPARSE_POINT, /*hTemplateFile=*/nullptr);
    if (hFrom == INVALID_HANDLE_VALUE) {
      const DWORD create_err = GetLastError();
      if (create_err == ERROR_FILE_NOT_FOUND ||
          create_err == ERROR_PATH_NOT_FOUND) {
        return WinStatus(create_err);
      }
    } else {
      AlignedPtr<uint8_t> buf = AllocateAligned<uint8_t>(total_bytes);
      if (!buf) {
        (void)CloseHandle(hFrom);
        return Status::kResource;
      }
      ZeroBytes(buf.get(), header_bytes);
      auto* info = reinterpret_cast<FileRenameInfoExLayout*>(buf.get());
      info->Flags = kReplaceIfExists | kPosixSemantics;
      info->RootDirectory = nullptr;
      info->FileNameLength = static_cast<DWORD>(name_bytes);
      CopyBytes(wide_to.data(), info->FileName, wide_bytes);

      detail::RecordSyscall();
      const BOOL ok = SetFileInformationByHandle(
          hFrom, kFileRenameInfoEx, info, static_cast<DWORD>(total_bytes));
      const DWORD err = ok ? ERROR_SUCCESS : GetLastError();
      detail::RecordSyscall();
      (void)CloseHandle(hFrom);
      if (ok) return Status();
      // Fall back to `MoveFileExW` only when the OS or filesystem (e.g. FAT32,
      // exFAT returning `ERROR_FILE_SYSTEM_LIMITATION`, or pre-Win10 1607)
      // does not support `FileRenameInfoEx`.
      if (err != ERROR_INVALID_PARAMETER && err != ERROR_INVALID_FUNCTION &&
          err != ERROR_NOT_SUPPORTED && err != ERROR_CALL_NOT_IMPLEMENTED &&
          err != ERROR_FILE_SYSTEM_LIMITATION) {
        return WinStatus(
            MapDirectoryRenameMismatch(err, wide_from.data(), wide_to.data()));
      }
    }

    // `CreateFileW` without `FILE_FLAG_BACKUP_SEMANTICS` rejects directories
    // (`hFrom == INVALID_HANDLE_VALUE`), reaching here when `from` is a
    // directory. Check directory/non-directory mismatch before calling
    // `MoveFileExW(..., MOVEFILE_REPLACE_EXISTING)`: on Windows 10 1709+ /
    // Server 2019+, `MoveFileExW` unlinks an existing regular file `to` and
    // replaces it with directory `from` (returning `TRUE`), and on older
    // Windows it fails with `ERROR_ALREADY_EXISTS` (`IOCode::kExists`) instead
    // of POSIX `ENOTDIR` (`Status::kArgument`).
    detail::RecordSyscall();
    const DWORD to_attrs = GetFileAttributesW(wide_to.data());
    if (to_attrs != INVALID_FILE_ATTRIBUTES) {
      detail::RecordSyscall();
      const DWORD from_attrs = GetFileAttributesW(wide_from.data());
      if (from_attrs != INVALID_FILE_ATTRIBUTES &&
          ((to_attrs ^ from_attrs) & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return WinStatus(ERROR_DIRECTORY);
      }
    }

    detail::RecordSyscall();
    if (!MoveFileExW(wide_from.data(), wide_to.data(),
                     MOVEFILE_REPLACE_EXISTING)) {
      const DWORD err = MapDirectoryRenameMismatch(
          GetLastError(), wide_from.data(), wide_to.data());
      return WinStatus(err);
    }
    return Status();
  }

  Status Remove(const Path& pathname) override {
    PROFILER_ZONE("IO.Win.Remove");
    const std::vector<wchar_t> wide = Utf8ToWide(pathname);
    if (wide.empty()) return Status::kArgument;
    detail::RecordSyscall();
    if (!DeleteFileW(wide.data())) {
      const DWORD err = GetLastError();
      return WinStatus(MapAccessDeniedForDirectory(err, wide.data()));
    }
    return Status();
  }

  Status ListDir(const Path& dir, std::vector<DirEntry>& out) override {
    PROFILER_ZONE("IO.Win.ListDir");
    // Backslash rather than `Join`'s '/' because `\\?\` paths require it.
    Path pattern = dir;
    const char last = pattern.path.back();
    if (last != '/' && last != '\\') pattern.path += '\\';
    pattern.path += '*';
    const std::vector<wchar_t> wide_pattern = Utf8ToWide(pattern);
    if (wide_pattern.empty()) return Status::kArgument;
    WIN32_FIND_DATAW data;
    detail::RecordSyscall();
    // `FindExInfoBasic` skips querying 8.3 short names.
    const HANDLE find = FindFirstFileExW(
        wide_pattern.data(), FindExInfoBasic, &data, FindExSearchNameMatch,
        /*lpSearchFilter=*/nullptr, FIND_FIRST_EX_LARGE_FETCH);
    if (find == INVALID_HANDLE_VALUE) {
      const DWORD err = GetLastError();
      // Distinguish missing paths and non-directories from empty directories
      // (drive roots lack "." entries, hence `ERROR_FILE_NOT_FOUND`).
      const std::vector<wchar_t> wide_dir = Utf8ToWide(dir);
      detail::RecordSyscall();
      const DWORD attrs = GetFileAttributesW(wide_dir.data());
      if (attrs == INVALID_FILE_ATTRIBUTES) return WinStatus(GetLastError());
      if (!IsDirectoryAttrs(attrs)) return WinStatus(ERROR_DIRECTORY);
      if (err == ERROR_FILE_NOT_FOUND) return Status();
      return WinStatus(err);
    }
    DWORD err = ERROR_SUCCESS;
    for (;;) {
      const wchar_t* name = data.cFileName;
      const bool is_dot =
          name[0] == L'.' &&
          (name[1] == L'\0' || (name[1] == L'.' && name[2] == L'\0'));
      if (!is_dot) {
        DirEntry entry;
        entry.name = WideToUtf8(name);
        const DWORD attrs = data.dwFileAttributes;
        // `dwReserved0` holds the reparse tag. Symlinks and junctions are not
        // followed, but other reparse points (e.g. cloud placeholders or
        // deduplicated files) are ordinary files or directories.
        if ((attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0 &&
            (data.dwReserved0 == IO_REPARSE_TAG_SYMLINK ||
             data.dwReserved0 == IO_REPARSE_TAG_MOUNT_POINT)) {
          entry.kind = EntryKind::kSymlink;
        } else if ((attrs & FILE_ATTRIBUTE_DIRECTORY) != 0) {
          entry.kind = EntryKind::kDir;
        } else if ((attrs & FILE_ATTRIBUTE_DEVICE) != 0) {
          entry.kind = EntryKind::kOther;
        } else {
          entry.kind = EntryKind::kFile;
          entry.size = (static_cast<uint64_t>(data.nFileSizeHigh) << 32) |
                       static_cast<uint64_t>(data.nFileSizeLow);
        }
        if (!entry.name.empty()) out.push_back(std::move(entry));
      }
      detail::RecordSyscall();
      if (!FindNextFileW(find, &data)) {
        err = GetLastError();
        break;
      }
    }
    detail::RecordSyscall();
    (void)FindClose(find);
    if (err != ERROR_NO_MORE_FILES) return WinStatus(err);
    return Status();
  }

  Status CreateDir(const Path& dir) override {
    PROFILER_ZONE("IO.Win.CreateDir");
    const std::vector<wchar_t> wide = Utf8ToWide(dir);
    if (wide.empty()) return Status::kArgument;
    detail::RecordSyscall();
    if (CreateDirectoryW(wide.data(), /*lpSecurityAttributes=*/nullptr)) {
      return Status();
    }
    // ERROR_ALREADY_EXISTS (any kind of entry) -> `IOCode::kExists`.
    return WinStatus(GetLastError());
  }

  Status RemoveDir(const Path& dir) override {
    PROFILER_ZONE("IO.Win.RemoveDir");
    const std::vector<wchar_t> wide = Utf8ToWide(dir);
    if (wide.empty()) return Status::kArgument;
    detail::RecordSyscall();
    if (RemoveDirectoryW(wide.data())) return Status();
    // `ERROR_DIR_NOT_EMPTY` -> `Status::kPrecondition`, `ERROR_DIRECTORY` (not
    // a directory) -> `Status::kArgument`.
    return WinStatus(GetLastError());
  }
};

// Constant-initialized (constexpr constructor); see `FileSystemList::Head` in
// file_system.cc for why this makes registration from static initializers safe.
WindowsFileSystem g_windows_file_system;

// Standard input or output via non-overlapped `ReadFile`/`WriteFile`, which
// transfer raw bytes (no text-mode newline translation). The handle is
// borrowed, i.e. not closed.
class StreamFileWin final : public StreamFile {
 public:
  StreamFileWin(HANDLE handle, Direction direction)
      : StreamFile(direction), handle_(handle) {}

  // `GetFileType` alone also reports FILE_TYPE_CHAR for other character
  // devices such as NUL; only consoles support `GetConsoleMode`.
  bool IsTerminal() const override {
    if (IsClosed()) return false;
    detail::RecordSyscall();
    if (GetFileType(handle_) != FILE_TYPE_CHAR) return false;
    DWORD mode;
    detail::RecordSyscall();
    return GetConsoleMode(handle_, &mode) != 0;
  }

  intptr_t Handle() const override {
    return IsClosed() ? -1 : reinterpret_cast<intptr_t>(handle_);
  }

 protected:
  Status ReadSome(Span<uint8_t> to) const override {
    // Size argument is 32-bit, so cap to 1 GiB per call.
    const DWORD want = static_cast<DWORD>(HWY_MIN(to.size(), size_t{1} << 30));
    DWORD got = 0;
    detail::RecordSyscall();
    if (!ReadFile(handle_, to.data(), want, &got, /*lpOverlapped=*/nullptr)) {
      const DWORD err = GetLastError();
      // The writer closed its end of the pipe: end of stream.
      if (err == ERROR_BROKEN_PIPE || err == ERROR_HANDLE_EOF) return Status(0);
      return WinStatus(err);
    }
    if (got != 0) {
      detail::RecordRead(got);
      if (got < want) detail::RecordShortRead();
    }
    return Status(got);
  }

  Status WriteSome(Span<const uint8_t> from) override {
    const DWORD want =
        static_cast<DWORD>(HWY_MIN(from.size(), size_t{1} << 30));
    DWORD put = 0;
    detail::RecordSyscall();
    if (!WriteFile(handle_, from.data(), want, &put,
                   /*lpOverlapped=*/nullptr)) {
      return WinStatus(GetLastError());
    }
    detail::RecordWrite(put);
    return Status(put);
  }

 private:
  HANDLE handle_;
};

Status OpenStdHandle(DWORD which, StreamFile::Direction direction,
                     std::unique_ptr<File>& file) {
  file.reset();
  detail::RecordSyscall();
  const HANDLE handle = GetStdHandle(which);
  // nullptr: the process has no such handle (e.g. GUI applications).
  // INVALID_HANDLE_VALUE: the parent disabled this stdio slot (or
  // `SetStdHandle` stored it); `GetStdHandle` then does not `SetLastError`, so
  // `GetLastError` would be stale.
  if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
    return Status::Error(Status::kNotFound,
                         static_cast<uint64_t>(ERROR_INVALID_HANDLE));
  }
  // Detect inherited handles that were closed, as POSIX `OpenStream` does via
  // `fcntl`. `FILE_TYPE_UNKNOWN` is also valid, hence check the error.
  SetLastError(NO_ERROR);
  detail::RecordSyscall();
  if (GetFileType(handle) == FILE_TYPE_UNKNOWN) {
    const DWORD err = GetLastError();
    if (err != NO_ERROR) {
      return Status::Error(Status::kNotFound, static_cast<uint64_t>(err));
    }
  }
  file = std::make_unique<StreamFileWin>(handle, direction);
  return Status();
}

}  // namespace

HWY_IO_DLLEXPORT Status OpenStdin(std::unique_ptr<File>& file) {
  return OpenStdHandle(STD_INPUT_HANDLE, StreamFile::Direction::kRead, file);
}

HWY_IO_DLLEXPORT Status OpenStdout(std::unique_ptr<File>& file) {
  return OpenStdHandle(STD_OUTPUT_HANDLE, StreamFile::Direction::kWrite, file);
}

HWY_IO_DLLEXPORT Path TempDir() {
  wchar_t buf[MAX_PATH + 2];
  detail::RecordSyscall();
  const DWORD len = GetTempPathW(MAX_PATH + 2, buf);
  if (len == 0 || len > MAX_PATH + 1) return Path();
  std::string dir = WideToUtf8(buf);
  // Strip trailing separators, except for drive roots such as "C:\".
  while (dir.size() > 1 && (dir.back() == '\\' || dir.back() == '/') &&
         !(dir.size() == 3 && dir[1] == ':')) {
    dir.pop_back();
  }
  return Path(std::move(dir));
}

namespace detail {
FileSystem& OSFileSystem() { return g_windows_file_system; }

FreePtr OSMapFreeFunc() { return &UnmapFileWin; }

Status AdviseOSMapRange(Span<const uint8_t> range, AccessHint hint,
                        MapEnd /*map_end*/) {
  if (hint != AccessHint::kWillNeed) return Status::kUnimplemented;
  struct RangeEntry {
    void* VirtualAddress;
    SIZE_T NumberOfBytes;
  };
  using PrefetchFunc = BOOL(WINAPI*)(HANDLE, ULONG_PTR, RangeEntry*, ULONG);
  static const PrefetchFunc prefetch_fn = []() -> PrefetchFunc {
    const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    if (kernel32 == nullptr) return nullptr;
    return reinterpret_cast<PrefetchFunc>(reinterpret_cast<void*>(
        GetProcAddress(kernel32, "PrefetchVirtualMemory")));
  }();
  if (prefetch_fn == nullptr) return Status::kUnimplemented;
  RangeEntry entry = {const_cast<uint8_t*>(range.data()), range.size()};
  detail::RecordSyscall();
  if (!prefetch_fn(GetCurrentProcess(), 1, &entry, 0)) {
    return WinStatus(GetLastError());
  }
  return Status();
}
}  // namespace detail
}  // namespace hwy
#endif  // HWY_IO_HAVE_OS_FILES
#endif  // HWY_OS_WIN
