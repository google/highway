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

// Feature-test macros must be defined before including any system header,
// hence this file must be compiled as its own translation unit (no unity
// builds or precompiled headers).
// Replacement for the _BSD_SOURCE specified by preadv documentation (glibc).
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

// Safe to include here: does not include any system headers.
#include "hwy/detect_compiler_arch.h"

#if HWY_OS_LINUX
// Expose Linux `fallocate()` in `<fcntl.h>` as well as `madvise()`.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
// Request POSIX 2008, including `pread()` and `posix_fadvise()`. This also
// implies `_POSIX_C_SOURCE`. Not done on BSD/Apple, where it would hide
// extensions such as `preadv` and `madvise`.
#if !defined(_XOPEN_SOURCE) || _XOPEN_SOURCE < 700
#undef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700  // SUSv4
#endif
#endif  // HWY_OS_LINUX

// Make `off_t` 64-bit even on 32-bit systems. Works for Android >= r15c.
#undef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/file_system.h"
#include "hwy/io/file_util.h"
#include "hwy/io/io_internal.h"
#include "hwy/io/io_span.h"
#include "hwy/io/path.h"
#include "hwy/io/stream_file.h"
#include "hwy/profiler.h"
#include "hwy/status.h"

#define HWY_HAVE_POSIX_IO (HWY_IO_HAVE_OS_FILES && !HWY_OS_WIN)

#if HWY_HAVE_POSIX_IO && HWY_OS_APPLE
#include <Availability.h>
#endif

#if ((HWY_OS_LINUX || HWY_OS_FREEBSD || HWY_OS_OPENBSD) &&    \
     (!defined(__ANDROID_API__) || __ANDROID_API__ >= 24)) || \
    (HWY_OS_APPLE && defined(__MAC_11_0) &&                   \
     ((defined(__MAC_OS_X_VERSION_MIN_REQUIRED) &&            \
       __MAC_OS_X_VERSION_MIN_REQUIRED >= 110000) ||          \
      (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) &&           \
       __IPHONE_OS_VERSION_MIN_REQUIRED >= 140000) ||         \
      (defined(__TV_OS_VERSION_MIN_REQUIRED) &&               \
       __TV_OS_VERSION_MIN_REQUIRED >= 140000) ||             \
      (defined(__WATCH_OS_VERSION_MIN_REQUIRED) &&            \
       __WATCH_OS_VERSION_MIN_REQUIRED >= 70000) ||           \
      defined(__VISION_OS_VERSION_MIN_REQUIRED)))
#define HWY_IO_PREADV 1
#else
#define HWY_IO_PREADV 0
#endif

#if (HWY_OS_LINUX || HWY_OS_FREEBSD) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
#define HWY_IO_FADVISE 1
#else
#define HWY_IO_FADVISE 0
#endif

#if HWY_OS_LINUX && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 21)
#define HWY_IO_FALLOCATE 1
#else
#define HWY_IO_FALLOCATE 0
#endif

#if HWY_HAVE_POSIX_IO

#if HWY_IO_PREADV
#include <sys/uio.h>  // preadv
#endif

#include <dirent.h>  // opendir
#include <fcntl.h>   // open
#include <limits.h>  // IOV_MAX
#include <stdio.h>   // rename
#include <stdlib.h>  // getenv
#include <sys/types.h>
// Old OSX may require sys/types.h before sys/mman.h.
#include <sys/mman.h>  // mmap
#include <sys/stat.h>  // fstat, O_RDONLY
#include <unistd.h>    // pread, pwrite, close, unlink

#if HWY_IO_FALLOCATE && !defined(FALLOC_FL_KEEP_SIZE)
#define FALLOC_FL_KEEP_SIZE 0x01  // from <linux/falloc.h>
#endif

#endif  // HWY_HAVE_POSIX_IO

#if HWY_HAVE_POSIX_IO

namespace hwy {

static_assert(sizeof(off_t) == sizeof(int64_t) && IsSigned<off_t>(),
              "off_t must be int64_t");

namespace {

#ifdef IOV_MAX
HWY_MAYBE_UNUSED constexpr size_t kMaxSpans = IOV_MAX;
#else
HWY_MAYBE_UNUSED constexpr size_t kMaxSpans = 1024;  // Linux limit
#endif

}  // namespace

namespace {

// Maps errno to a general status code, with errno as the payload.
Status ErrnoStatus(int err) {
  Status::Code code = Status::kOSError;
  switch (err) {
#ifdef ECANCELED
    case ECANCELED:
      code = Status::kCancelled;
      break;
#endif
    case ETIMEDOUT:
      code = Status::kTimeout;
      break;
    case ENOENT:
      code = Status::kNotFound;
      break;
    case EEXIST:
      code = static_cast<Status::Code>(IOCode::kExists);
      break;
    case EACCES:
    case EPERM:
    case EROFS:
      code = Status::kPermission;
      break;
    case ENOMEM:
    case ENOSPC:
    case EFBIG:
    case EMFILE:
    case ENFILE:
#ifdef EDQUOT
    case EDQUOT:
#endif
      code = Status::kResource;
      break;
    case EINVAL:
    case EISDIR:
    case ENOTDIR:
    case ENAMETOOLONG:
    case ELOOP:
    case EBADF:
    case EOVERFLOW:
    case EXDEV:
      code = Status::kArgument;
      break;
#ifdef ENOTEMPTY
    case ENOTEMPTY:
      code = Status::kPrecondition;
      break;
#endif
    case ESPIPE:
    case ENOSYS:
    case ENODEV:  // e.g. `mmap` on a file system without mapping support
#ifdef ENOTSUP
    case ENOTSUP:
#endif
#if defined(EOPNOTSUPP) && (!defined(ENOTSUP) || EOPNOTSUPP != ENOTSUP)
    case EOPNOTSUPP:
#endif
      code = Status::kUnimplemented;
      break;
    case EAGAIN:
#if defined(EWOULDBLOCK) && EWOULDBLOCK != EAGAIN
    case EWOULDBLOCK:
#endif
    case EBUSY:
#ifdef ETXTBSY
    case ETXTBSY:
#endif
      code = Status::kUnavailable;
      break;
    default:
      break;
  }
  return Status::Error(code, static_cast<uint64_t>(err));
}

void UnmapFile(void* opaque, void* ptr) {
  const size_t mapping_size =
      static_cast<size_t>(reinterpret_cast<uintptr_t>(opaque));
  detail::RecordSyscall();
  HWY_ASSERT(munmap(ptr, mapping_size) == 0);
}

#if HWY_IO_PREADV
// Advances `*cur`, `*num_spans`, and `*partial_offset` past `got` transferred
// bytes, plus any trailing zero-length spans in the current batch (down to
// `target_rem` remaining spans).
void AdvanceSpans(uint64_t got, size_t target_rem, const IOSpan** cur,
                  size_t* num_spans, size_t* partial_offset) {
  while (got != 0) {
    HWY_ASSERT(*num_spans != 0);
    if ((*cur)->bytes == 0) {
      HWY_DASSERT(*partial_offset == 0);
      ++(*cur);
      --(*num_spans);
      continue;
    }
    const size_t avail = (*cur)->bytes - *partial_offset;
    if (got < avail) {
      *partial_offset += static_cast<size_t>(got);
      return;
    }
    got -= avail;
    *partial_offset = 0;
    ++(*cur);
    --(*num_spans);
  }
  while (*num_spans > target_rem && (*cur)->bytes == 0) {
    ++(*cur);
    --(*num_spans);
  }
}
#endif  // HWY_IO_PREADV

#if HWY_IO_FADVISE && HWY_OS_LINUX &&                       \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 26) && \
    defined(SYNC_FILE_RANGE_WRITE)
// Writes back and waits on dirty pages in `[offset, offset + size)` (`size ==
// 0` means until EOF) so a subsequent `POSIX_FADV_DONTNEED` can actually evict
// them. Linux `POSIX_FADV_DONTNEED` triggers asynchronous writeback and then
// immediately calls `invalidate_mapping_pages` without waiting, silently
// skipping pages that are still dirty or under writeback. Restricted to
// `sync_file_range` (no whole-file `fdatasync` fallback, which would turn
// per-chunk streaming `Advise` calls into O(N^2) full-file flushes).
void FlushRangeForEvict(int fd, off_t offset, off_t size) {
  for (;;) {
    detail::RecordSyscall();
    if (sync_file_range(fd, offset, size,
                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                            SYNC_FILE_RANGE_WAIT_AFTER) == 0) {
      return;
    }
    if (errno != EINTR) return;
    detail::RecordEintr();
  }
}
#endif

enum class TransferOp { kRead, kWrite };

// Independent of `FileMode` (see `FileCap`).
FileCaps PosixFileCaps() {
  FileCaps caps = FileCap::kRandomAccess | FileCap::kRandomWrite |
                  FileCap::kExtend | FileCap::kSync | FileCap::kMapImmutable |
                  FileCap::kMapShared | FileCap::kMapWritable;
#if HWY_IO_FADVISE || \
    (HWY_OS_APPLE && defined(F_RDAHEAD) && defined(F_RDADVISE))
  caps |= FileCap::kAdvise;
#endif
#if HWY_IO_PREADV
  caps |= FileCap::kNativeScatter;
#endif
#if HWY_IO_FALLOCATE || defined(F_PREALLOCATE)
  caps |= FileCap::kPreallocate;
#endif
  return caps;
}

// Returns the kind of a `stat` or `lstat` result.
EntryKind KindFromMode(mode_t mode) {
  return S_ISREG(mode)   ? EntryKind::kFile
         : S_ISDIR(mode) ? EntryKind::kDir
         : S_ISLNK(mode) ? EntryKind::kSymlink
                         : EntryKind::kOther;
}

// Sets `out` from a `stat` or `fstat` result (which follow symlinks).
Status FileStatFrom(const struct stat& st, FileStat& out) {
  out.kind = KindFromMode(st.st_mode);
  out.size = 0;
  if (out.kind == EntryKind::kFile) {
    if (st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > Status::kMaxPayload) {
      return Status::kRange;
    }
    out.size = static_cast<uint64_t>(st.st_size);
  }
  // `dev_t` is signed on some platforms (e.g. macOS), but sign extension still
  // yields consistent values for comparison.
  out.device = static_cast<uint64_t>(st.st_dev);
  out.inode = static_cast<uint64_t>(st.st_ino);
  return Status(out.size);
}

// Kinds of descriptors supported by `FilePosix`.
enum class FdKind : uint8_t {
  kRegular,
  // Seekable non-terminal character device such as /dev/null or /dev/full,
  // only for write modes. `pwrite` succeeds (or fails with e.g. `ENOSPC`) at
  // any offset, but there is no size to extend, preallocate, sync or map.
  kCharDevice
};

// Which transfers a `FilePosix` permits, from its `FileMode` and, for
// `MaybeOpenHandle`, the descriptor's `O_ACCMODE`.
enum class FdAccess : uint8_t { kReadOnly, kWriteOnly, kReadWrite };

bool IsWriteMode(FileMode mode) { return mode != FileMode::kRead; }

// Sets `kind` and returns `Status()` if `FilePosix` supports `fd` with `mode`.
// Errors have an errno payload, as from `open`.
Status ClassifyFd(int fd, FileMode mode, FdKind& kind) {
  struct stat st;
  detail::RecordSyscall();
  if (fstat(fd, &st) != 0) return ErrnoStatus(errno);
  // Only regular files support pread/pwrite at arbitrary offsets and mmap.
  // Notably, open(O_RDONLY) succeeds for directories.
  if (S_ISREG(st.st_mode)) {
    kind = FdKind::kRegular;
    return Status();
  }
  if (S_ISDIR(st.st_mode)) return ErrnoStatus(EISDIR);
  if (S_ISCHR(st.st_mode) && IsWriteMode(mode)) {
    // Terminals and other non-seekable devices would fail `pwrite` with
    // `ESPIPE` (on Linux), so reject them here.
    detail::RecordSyscall();
    if (isatty(fd) == 1) return ErrnoStatus(EINVAL);
    detail::RecordSyscall();
    if (lseek(fd, 0, SEEK_CUR) == -1) return ErrnoStatus(EINVAL);
    kind = FdKind::kCharDevice;
    return Status();
  }
  return ErrnoStatus(EINVAL);
}

// Converts the `intptr_t` passed to `MaybeOpenHandle` (wide enough for a
// Windows `HANDLE`) to a POSIX descriptor, or returns -1 if out of range.
int FdFromHandle(intptr_t handle) {
  if (handle < 0 || handle > static_cast<intptr_t>(LimitsMax<int>())) return -1;
  return static_cast<int>(handle);
}

// For `FileMode::kWriteTruncate`. Applied after `ClassifyFd` rather than via
// `O_TRUNC`, whose effect on character devices is unspecified.
Status TruncateToZero(int fd) {
  for (;;) {
    detail::RecordSyscall();
    if (ftruncate(fd, 0) == 0) return Status();
    const int err = errno;
    if (err != EINTR) return ErrnoStatus(err);
    detail::RecordEintr();
  }
}

}  // namespace

class FilePosix : public File {
 public:
  FilePosix(int fd, FdAccess access, HandleOwnership ownership, FdKind kind)
      : fd_(fd), access_(access), ownership_(ownership), kind_(kind) {
    HWY_ASSERT(fd >= 0);
    HWY_ASSERT(kind == FdKind::kRegular || access != FdAccess::kReadOnly);
  }
  ~FilePosix() override { (void)Close(); }

  FileCaps Capabilities() const override {
    if (kind_ == FdKind::kRegular) return PosixFileCaps();
    return PosixFileCaps() &
           ~(FileCap::kExtend | FileCap::kPreallocate | FileCap::kSync |
             FileCap::kAdvise | FileCap::kMapImmutable | FileCap::kMapShared |
             FileCap::kMapWritable);
  }

  Status Size() const override {
    const Status open_status = CheckOpen();
    if (!open_status.ok()) return open_status;
    struct stat st;
    detail::RecordSyscall();
    if (fstat(fd_, &st) != 0) return ErrnoStatus(errno);
    if (st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > Status::kMaxPayload) {
      return Status::kRange;
    }
    return Status(static_cast<uint64_t>(st.st_size));
  }

  Status Stat(FileStat& out) const override {
    out = FileStat();
    const Status open_status = CheckOpen();
    if (!open_status.ok()) return open_status;
    struct stat st;
    detail::RecordSyscall();
    if (fstat(fd_, &st) != 0) return ErrnoStatus(errno);
    const Status status = FileStatFrom(st, out);
    if (!status.ok()) out = FileStat();
    return status;
  }

  Status SetSize(uint64_t size) override {
    PROFILER_ZONE("IO.Posix.SetSize");
    const Status open_status = CheckWritable();
    if (!open_status.ok()) return open_status;
    if (kind_ != FdKind::kRegular) return Status::kUnimplemented;
    if (!ScalarFits<off_t>(size) || size > Status::kMaxPayload) {
      return Status::kArgument;
    }
    for (;;) {
      detail::RecordSyscall();
      if (ftruncate(fd_, static_cast<off_t>(size)) == 0) return Status(size);
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
  }

  Status Preallocate(uint64_t offset, uint64_t size) override {
    PROFILER_ZONE("IO.Posix.Preallocate");
    const Status open_status = CheckWritable();
    if (!open_status.ok()) return open_status;
    if (kind_ != FdKind::kRegular) return Status::kUnimplemented;
    if (!SumFits<off_t>(offset, size)) return Status::kArgument;
    if (size == 0) return Status();
#if HWY_IO_FALLOCATE
    // Avoid `posix_fallocate`: glibc emulates unsupported filesystems with a
    // non-atomic userspace zero-write loop that races with concurrent
    // `pwrite`s. `FALLOC_FL_KEEP_SIZE` leaves the logical size to `SetSize`.
    // Unsupported filesystems return `EOPNOTSUPP` (`Status::kUnimplemented`).
    for (;;) {
      detail::RecordSyscall();
      if (fallocate(fd_, FALLOC_FL_KEEP_SIZE, static_cast<off_t>(offset),
                    static_cast<off_t>(size)) == 0) {
        return Status();
      }
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
#elif defined(F_PREALLOCATE)
    // `F_PEOFPOSMODE` allocates the requested number of bytes beyond the
    // currently allocated blocks (physical EOF) without changing `st_size`.
    struct stat st;
    detail::RecordSyscall();
    if (fstat(fd_, &st) != 0) return ErrnoStatus(errno);
    // `st_blocks` rather than `st_size` because `SetSize` creates sparse tails.
    const uint64_t allocated =
        st.st_blocks > 0 ? static_cast<uint64_t>(st.st_blocks) * 512 : 0;
    const uint64_t end = offset + size;
    if (end <= allocated) return Status();
    const off_t len = static_cast<off_t>(end - allocated);
    auto try_preallocate = [&](unsigned int fst_flags) -> int {
      for (;;) {
        fstore_t store = {};
        store.fst_flags = fst_flags;
        store.fst_posmode = F_PEOFPOSMODE;
        store.fst_offset = 0;
        store.fst_length = len;
        detail::RecordSyscall();
        if (fcntl(fd_, F_PREALLOCATE, &store) != -1) return 0;
        const int err = errno;
        if (err != EINTR) return err;
        detail::RecordEintr();
      }
    };
    int err = try_preallocate(F_ALLOCATECONTIG | F_ALLOCATEALL);
    if (err != 0) err = try_preallocate(F_ALLOCATEALL);
    return err == 0 ? Status() : ErrnoStatus(err);
#else
    return Status::kUnimplemented;
#endif
  }

  Status Advise(uint64_t offset, uint64_t size,
                AccessHint hint) const override {
    const Status open_status = CheckOpen();
    if (!open_status.ok()) return open_status;
    if (!SumFits<off_t>(offset, size)) return Status::kArgument;
    if (kind_ != FdKind::kRegular) return Status::kUnimplemented;
#if HWY_IO_FADVISE
#if HWY_OS_LINUX && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 26) && \
    defined(SYNC_FILE_RANGE_WRITE)
    if (access_ != FdAccess::kReadOnly && hint == AccessHint::kDontNeed) {
      FlushRangeForEvict(fd_, static_cast<off_t>(offset),
                         static_cast<off_t>(size));
    }
#endif
    int advice = POSIX_FADV_NORMAL;
    switch (hint) {
      case AccessHint::kNormal:
        advice = POSIX_FADV_NORMAL;
        break;
      case AccessHint::kSequential:
        advice = POSIX_FADV_SEQUENTIAL;
        break;
      case AccessHint::kRandom:
        advice = POSIX_FADV_RANDOM;
        break;
      case AccessHint::kWillNeed:
        advice = POSIX_FADV_WILLNEED;
        break;
      case AccessHint::kDontNeed:
        advice = POSIX_FADV_DONTNEED;
        break;
    }
    detail::RecordSyscall();
    // posix_fadvise returns an error number directly instead of setting errno.
    const int err = posix_fadvise(fd_, static_cast<off_t>(offset),
                                  static_cast<off_t>(size), advice);
    if (err != 0) return ErrnoStatus(err);
    return Status();
#elif HWY_OS_APPLE && defined(F_RDAHEAD) && defined(F_RDADVISE)
    switch (hint) {
      case AccessHint::kNormal:
      case AccessHint::kSequential:
        detail::RecordSyscall();
        if (fcntl(fd_, F_RDAHEAD, 1) == -1) return ErrnoStatus(errno);
        return Status();
      case AccessHint::kRandom:
        detail::RecordSyscall();
        if (fcntl(fd_, F_RDAHEAD, 0) == -1) return ErrnoStatus(errno);
        return Status();
      case AccessHint::kWillNeed: {
        // `F_RDADVISE` reads ahead, which requires read access.
        const Status readable = CheckReadable();
        if (!readable.ok()) return readable;
        uint64_t len = size;
        if (len == 0) {
          const Status cur = Size();
          if (!cur.ok()) return cur;
          if (offset >= cur.size()) return Status();
          len = cur.size() - offset;
        }
        if (len == 0) return Status();
        struct radvisory ra;
        ra.ra_offset = static_cast<off_t>(offset);
        ra.ra_count = static_cast<int>(
            HWY_MIN(len, static_cast<uint64_t>(LimitsMax<int>())));
        detail::RecordSyscall();
        if (fcntl(fd_, F_RDADVISE, &ra) == -1) return ErrnoStatus(errno);
        return Status();
      }
      case AccessHint::kDontNeed:
        return Status::kUnimplemented;
    }
    return Status::kUnimplemented;
#else
    (void)hint;
    return Status::kUnimplemented;
#endif
  }

  Status Read(uint64_t offset, Span<uint8_t> to) const override {
    PROFILER_ZONE("IO.Posix.Read");
    const Status open_status = CheckReadable();
    if (!open_status.ok()) return open_status;
    return Transfer(TransferOp::kRead, offset, to.size(), to.data());
  }

  Status Write(uint64_t offset, Span<const uint8_t> from) override {
    PROFILER_ZONE("IO.Posix.Write");
    const Status open_status = CheckWritable();
    if (!open_status.ok()) return open_status;
    return Transfer(TransferOp::kWrite, offset, from.size(),
                    const_cast<uint8_t*>(from.data()));
  }

  Status WriteSpans(uint64_t offset, const IOSpans& spans) override {
    PROFILER_ZONE("IO.Posix.WriteSpans");
    const Status open_status = CheckWritable();
    if (!open_status.ok()) return open_status;
#if HWY_IO_PREADV
    const Status total = spans.ValidateAt(offset);
    if (!total.ok()) return total;
    const uint64_t total_bytes = total.size();
    if (total_bytes <= kMaxBatchBytes) {
      return TransferSpans(TransferOp::kWrite, offset, total_bytes,
                           spans.Spans());
    }
#endif
    return File::WriteSpans(offset, spans);
  }

  Status MapImmutable(MapPtr& map) override {
    PROFILER_ZONE("IO.Posix.MapImmutable");
    // `MAP_PRIVATE` rather than `MAP_SHARED`: for `PROT_READ` both share the
    // same physical page-cache pages without copying (COW only triggers on
    // write faults), and `MAP_PRIVATE` works on FUSE/special mounts that lack
    // `MAP_SHARED` while avoiding `VM_MAYWRITE` / `inode->i_writecount` on
    // `O_RDWR` fds (which would otherwise block `execve` with `ETXTBSY` and
    // `F_SEAL_WRITE` with `EBUSY` until `MapPtr` is destroyed). POSIX leaves it
    // unspecified whether file writes after `mmap` are visible through
    // `MAP_PRIVATE`, which matches the immutable-file contract.
    map.reset();
    const Status open_status = CheckReadable();
    if (!open_status.ok()) return open_status;
    return MapWhole(PROT_READ, MAP_PRIVATE, map);
  }

  Status MapShared(MapPtr& map) override {
    PROFILER_ZONE("IO.Posix.MapShared");
    map.reset();
    const Status open_status = CheckReadable();
    if (!open_status.ok()) return open_status;
    return MapWhole(PROT_READ, MAP_SHARED, map);
  }

  // `mmap` requires read access even for `PROT_WRITE`.
  Status MapWritable(MutableMapPtr& map) override {
    PROFILER_ZONE("IO.Posix.MapWritable");
    map.reset();
    Status open_status = CheckWritable();
    if (open_status.ok()) open_status = CheckReadable();
    if (!open_status.ok()) return open_status;
    return MapWhole(PROT_READ | PROT_WRITE, MAP_SHARED, map);
  }

  Status Sync() override {
    PROFILER_ZONE("IO.Posix.Sync");
    const Status open_status = CheckWritable();
    if (!open_status.ok()) return open_status;
    // `fsync` fails with `EINVAL` on e.g. /dev/null.
    if (kind_ != FdKind::kRegular) return Status::kUnimplemented;
    for (;;) {
      detail::RecordSyscall();
      if (fsync(fd_) == 0) return Status();
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
  }

  Status Close() override {
    if (fd_ < 0) return Status();
    const int fd = fd_;
    fd_ = -1;  // Subsequent operations fail with `Status::kPrecondition`.
    if (ownership_ == HandleOwnership::kBorrow) return Status();
    detail::RecordSyscall();
    // Do not retry on EINTR: Linux releases the fd regardless, and retrying
    // could close an fd that another thread was just given.
    if (close(fd) != 0 && errno != EINTR) return ErrnoStatus(errno);
    return Status();
  }

  Status ReadSpans(uint64_t offset, const IOSpans& spans,
                   IOWorkerBuf& worker_buf) const override {
    PROFILER_ZONE("IO.Posix.ReadSpans");
    const Status open_status = CheckReadable();
    if (!open_status.ok()) return open_status;
#if HWY_IO_PREADV
    const Status total = spans.ValidateAt(offset);
    if (!total.ok()) return total;
    const uint64_t total_bytes = total.size();
    // Linux `preadv`/`pwritev` clamp transfers to `MAX_RW_COUNT` (0x7FFFF000),
    // whereas macOS XNU, OpenBSD, and FreeBSD sum `iov_len` across the batch
    // and fail with `EINVAL` if the sum exceeds `INT_MAX`. Delegate >2 GiB
    // calls to `File::ReadSpans`/`WriteSpans`, which chunk in
    // `kMaxBatchBytes` pieces.
    if (total_bytes <= kMaxBatchBytes) {
      return TransferSpans(TransferOp::kRead, offset, total_bytes,
                           spans.Spans());
    }
#endif
    return File::ReadSpans(offset, spans, worker_buf);
  }

 private:
  Status CheckOpen() const {
    return fd_ < 0 ? Status(Status::kPrecondition) : Status();
  }
  // `IOCode::kWrongMode` for `O_WRONLY` descriptors from `MaybeOpenHandle`,
  // which would otherwise fail with `EBADF` or `EACCES`.
  Status CheckReadable() const {
    if (fd_ < 0) return Status::kPrecondition;
    if (access_ == FdAccess::kWriteOnly) return IOCode::kWrongMode;
    return Status();
  }
  Status CheckWritable() const {
    if (fd_ < 0) return Status::kPrecondition;
    if (access_ == FdAccess::kReadOnly) return IOCode::kWrongMode;
    return Status();
  }

  // Maps `[0, Size())` with `prot` and `flags` into `map` (`MapPtr` or
  // `MutableMapPtr`). No `MAP_POPULATE` or unconditional `MADV_WILLNEED`
  // because callers may map multi-GB files for random access (use `Advise` for
  // readahead).
  template <typename T>
  Status MapWhole(int prot, int flags, AlignedPtr<T>& map) {
    map.reset();
    const Status size_status = Size();
    if (!size_status.ok()) return size_status;
    if (kind_ != FdKind::kRegular) return Status::kUnimplemented;
    const uint64_t file_size = size_status.size();
    if (!ScalarFits<size_t>(file_size)) return Status::kRange;
    if (file_size == 0) {
      // Pseudo-files (e.g. /proc) report size 0 but have contents, which an
      // empty mapping would silently drop.
      const Status empty = VerifyEmpty(*this);
      return empty.ok() ? Status(0) : empty;
    }
    const size_t mapping_size = SafeCast<size_t>(file_size);
    detail::RecordSyscall();
    void* mapping = mmap(nullptr, mapping_size, prot, flags, fd_, /*offset=*/0);
    if (mapping == MAP_FAILED) return ErrnoStatus(errno);
    map = AlignedPtr<T>(
        static_cast<T*>(mapping), mapping_size, &UnmapFile,
        reinterpret_cast<void*>(static_cast<uintptr_t>(mapping_size)));
    return Status(mapping_size);
  }

  Status Transfer(TransferOp op, uint64_t offset, size_t size,
                  void* bytes) const {
    // The returned `Status` payload must be able to represent `size`.
    if (!SumFits<off_t>(offset, size) ||
        !SumWithin(uint64_t{0}, size, Status::kMaxPayload)) {
      return Status::kArgument;
    }
    uint8_t* ptr = static_cast<uint8_t*>(bytes);
    size_t pos = 0;
    while (pos < size) {
      const size_t want = HWY_MIN(size - pos, kMaxBatchBytes);
      // pread/pwrite are faster than lseek + read/write when parallelized.
      detail::RecordSyscall();
      const off_t call_off = SafeAdd<off_t>(offset, pos);
      const ssize_t ret = op == TransferOp::kWrite
                              ? pwrite(fd_, ptr + pos, want, call_off)
                              : pread(fd_, ptr + pos, want, call_off);
      if (ret < 0) {
        const int err = errno;
        if (err == EINTR) {
          detail::RecordEintr();
          continue;
        }
        return ErrnoStatus(err);
      }
      if (ret == 0) {
        // EOF for reads; unexpected 0-byte write would loop forever. Both
        // carry the number of bytes transferred so far.
        return op == TransferOp::kWrite ? Status(IOCode::kNoProgress, pos)
                                        : Status(Status::kTruncated, pos);
      }
      const size_t got = static_cast<size_t>(ret);
      if (op == TransferOp::kWrite) {
        detail::RecordWrite(got);
      } else {
        detail::RecordRead(got);
        if (got < want) detail::RecordShortRead();
      }
      pos = SafeAdd<size_t>(pos, got);
      HWY_ASSERT(pos <= size);
    }
    return Status(size);
  }

#if HWY_IO_PREADV
  Status TransferSpans(TransferOp op, uint64_t offset, uint64_t total_bytes,
                       Span<const IOSpan> spans) const {
    if (total_bytes == 0) return Status(0);
    if (spans.size() == 1) {
      return Transfer(op, offset, spans[0].bytes, spans[0].mem);
    }
    static_assert(sizeof(IOSpan) == sizeof(iovec), "IOSpan must match iovec");
    const IOSpan* cur = spans.data();
    size_t num_spans = spans.size();
    size_t partial_offset = 0;  // Bytes already transferred in cur[0].
    std::vector<IOSpan>
        adjusted;  // Allocated only if a short transfer splits cur[0].
    uint64_t pos = 0;
    while (pos < total_bytes) {
      // Skip empty spans from direct callers (IOBatch::Add drops them).
      while (num_spans != 0 && cur->bytes == 0) {
        HWY_DASSERT(partial_offset == 0);
        ++cur;
        --num_spans;
      }
      HWY_ASSERT(num_spans != 0);
      const size_t batch_spans = HWY_MIN(num_spans, kMaxSpans);
      const size_t target_rem = num_spans - batch_spans;
      const IOSpan* call_spans = cur;
      if (partial_offset != 0) {
        adjusted.resize(batch_spans);
        adjusted[0] = IOSpan{static_cast<uint8_t*>(cur[0].mem) + partial_offset,
                             cur[0].bytes - partial_offset};
        for (size_t i = 1; i < batch_spans; ++i) {
          adjusted[i] = cur[i];
        }
        call_spans = adjusted.data();
      }
      detail::RecordSyscall();
      const iovec* iov = reinterpret_cast<const iovec*>(call_spans);
      const int iovcnt = static_cast<int>(batch_spans);
      const off_t call_off = SafeAdd<off_t>(offset, pos);
      const ssize_t ret = op == TransferOp::kWrite
                              ? pwritev(fd_, iov, iovcnt, call_off)
                              : preadv(fd_, iov, iovcnt, call_off);
      if (ret < 0) {
        const int err = errno;
        if (err == EINTR) {
          detail::RecordEintr();
          continue;
        }
        return ErrnoStatus(err);
      }
      if (ret == 0) {
        return op == TransferOp::kWrite ? Status(IOCode::kNoProgress, pos)
                                        : Status(Status::kTruncated, pos);
      }
      const uint64_t got = static_cast<uint64_t>(ret);
      if (op == TransferOp::kWrite) {
        detail::RecordWrite(got);
      } else {
        detail::RecordRead(got);
      }
      pos = SafeAdd<uint64_t>(pos, got);
      HWY_ASSERT(pos <= total_bytes);
      if (pos == total_bytes) break;

      AdvanceSpans(got, target_rem, &cur, &num_spans, &partial_offset);
      if (op == TransferOp::kRead && num_spans > target_rem) {
        detail::RecordShortRead();
      }
    }
    return Status(total_bytes);
  }
#endif  // HWY_IO_PREADV

  int fd_ = -1;
  const FdAccess access_;
  const HandleOwnership ownership_;
  const FdKind kind_;
};  // FilePosix

namespace {

// Shared by `PosixFileSystem::Open` and `MaybeOpenHandle` after `ClassifyFd`.
Status FinishOpen(int fd, OpenOptions options, FdAccess access,
                  HandleOwnership ownership, FdKind kind,
                  std::unique_ptr<File>& file) {
  if (options.mode == FileMode::kWriteTruncate && kind == FdKind::kRegular) {
    const Status truncate_status = TruncateToZero(fd);
    if (!truncate_status.ok()) return truncate_status;
  }
  std::unique_ptr<FilePosix> posix_file =
      std::make_unique<FilePosix>(fd, access, ownership, kind);
  if (options.hint != AccessHint::kNormal) {
    (void)posix_file->Advise(/*offset=*/0, /*size=*/0, options.hint);
  }
  file = std::move(posix_file);
  return Status();
}

}  // namespace

namespace {

class PosixFileSystem final : public FileSystem {
 public:
  bool Claims(const Path& /*pathname*/) const override { return true; }

  FileCaps Capabilities(const Path& /*pathname*/) const override {
    return PosixFileCaps();
  }

  // Note: O_DIRECT is intentionally omitted because we want to use the OS page
  // cache between consecutive runs (and avoid alignment constraints on
  // offsets, sizes, and buffers); callers can evict ranges via
  // Advise(..., kDontNeed).
  Status Open(const Path& pathname, OpenOptions options,
              std::unique_ptr<File>& file) override {
    const FileMode mode = options.mode;
    const bool is_read = mode == FileMode::kRead;
    int flags = is_read ? O_RDONLY : (O_RDWR | O_CREAT);
    if (mode == FileMode::kWriteNew) flags |= O_EXCL;
    // No `O_TRUNC`; see `TruncateToZero`.
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    // Avoid blocking on FIFOs or character devices (and prevent a session
    // leader without a controlling terminal from acquiring `/dev/tty*`) before
    // `ClassifyFd` rejects and closes `fd`.
    int open_flags = flags;
#ifdef O_NONBLOCK
    open_flags |= O_NONBLOCK;
#endif
#ifdef O_NOCTTY
    open_flags |= O_NOCTTY;
#endif
    int fd = -1;
    int open_err = 0;
    for (;;) {
      detail::RecordSyscall();
      // Pass 0666 (like `fopen` / `std::ofstream`) so the kernel's `& ~umask`
      // honors group-writable `0002` umasks while still producing `0644` under
      // the default `0022` umask.
      fd = open(pathname.c_str(), open_flags, 0666);
      if (fd >= 0) break;
      open_err = errno;
      if (open_err != EINTR) break;
      detail::RecordEintr();
    }
    if (fd < 0) {
      if (open_err == EACCES || open_err == EPERM) {
        struct stat st;
        detail::RecordSyscall();
        if (stat(pathname.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
          open_err = EISDIR;
        }
      }
      return ErrnoStatus(open_err);
    }

    FdKind kind = FdKind::kRegular;
    Status status = ClassifyFd(fd, mode, kind);
#ifdef O_NONBLOCK
    // Clear only `O_NONBLOCK`: `F_SETFL` with `flags` would pass access-mode
    // and creation flags, which are ignored only by some kernels.
    if (status.ok()) {
      detail::RecordSyscall();
      const int fl = fcntl(fd, F_GETFL);
      if (fl == -1) {
        status = ErrnoStatus(errno);
      } else {
        detail::RecordSyscall();
        if (fcntl(fd, F_SETFL, fl & ~O_NONBLOCK) == -1) {
          status = ErrnoStatus(errno);
        }
      }
    }
#endif
    if (status.ok()) {
      const FdAccess access =
          is_read ? FdAccess::kReadOnly : FdAccess::kReadWrite;
      status =
          FinishOpen(fd, options, access, HandleOwnership::kTake, kind, file);
    }
    if (!status.ok()) {
      detail::RecordSyscall();
      (void)close(fd);
    }
    return status;
  }

  Status Stat(const Path& pathname, FileStat& out) override {
    PROFILER_ZONE("IO.Posix.Stat");
    // `stat` follows symlinks, avoids opening FIFOs/devices, fails with
    // `EACCES` (`Status::kPermission`) on missing paths under ACL-denied
    // directories, and succeeds even when the entry itself lacks read
    // permission (`chmod 000`).
    struct stat st;
    for (;;) {
      detail::RecordSyscall();
      if (stat(pathname.c_str(), &st) == 0) return FileStatFrom(st, out);
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
  }

  Status Rename(const Path& from, const Path& to) override {
    PROFILER_ZONE("IO.Posix.Rename");
    for (;;) {
      detail::RecordSyscall();
      if (rename(from.c_str(), to.c_str()) == 0) return Status();
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
  }

  Status Remove(const Path& pathname) override {
    PROFILER_ZONE("IO.Posix.Remove");
    for (;;) {
      detail::RecordSyscall();
      if (unlink(pathname.c_str()) == 0) return Status();
      int err = errno;
      if (err == EINTR) {
        detail::RecordEintr();
        continue;
      }
      if (err == EPERM || err == EACCES) {
        // `unlink` does not follow symlinks: use `lstat` so a symlink to a
        // directory that failed with `EACCES`/`EPERM` (e.g. non-writable or
        // sticky parent dir) stays `Status::kPermission` instead of `EISDIR`.
        struct stat st;
        detail::RecordSyscall();
        if (lstat(pathname.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
          err = EISDIR;
        }
      }
      return ErrnoStatus(err);
    }
  }

  Status ListDir(const Path& dir, std::vector<DirEntry>& out) override {
    PROFILER_ZONE("IO.Posix.ListDir");
    DIR* stream = nullptr;
    for (;;) {
      detail::RecordSyscall();
      stream = opendir(dir.c_str());
      if (stream != nullptr) break;
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);  // incl. ENOTDIR
      detail::RecordEintr();
    }
    const int dir_fd = dirfd(stream);
    int err = 0;
    for (;;) {
      errno = 0;
      detail::RecordSyscall();
      const struct dirent* ent = readdir(stream);
      if (ent == nullptr) {
        err = errno;  // 0 at the end of the directory.
        break;
      }
      const char* name = ent->d_name;
      if (name[0] == '.' &&
          (name[1] == '\0' || (name[1] == '.' && name[2] == '\0'))) {
        continue;
      }
      DirEntry entry;
      entry.name = name;
      // `fstatat` is required for the size of regular files and when the file
      // system does not report `d_type`. Symlinks are not followed (`lstat`).
      bool need_stat = true;
#ifdef DT_UNKNOWN
      switch (ent->d_type) {
        case DT_REG:
          entry.kind = EntryKind::kFile;
          break;
        case DT_DIR:
          entry.kind = EntryKind::kDir;
          need_stat = false;
          break;
        case DT_LNK:
          entry.kind = EntryKind::kSymlink;
          need_stat = false;
          break;
        case DT_UNKNOWN:
          break;
        default:  // DT_FIFO, DT_SOCK, DT_CHR, DT_BLK etc.
          entry.kind = EntryKind::kOther;
          need_stat = false;
          break;
      }
#endif
      if (need_stat) {
        struct stat st;
        detail::RecordSyscall();
        if (fstatat(dir_fd, name, &st, AT_SYMLINK_NOFOLLOW) == 0) {
          entry.kind = KindFromMode(st.st_mode);
          if (entry.kind == EntryKind::kFile && st.st_size > 0) {
            entry.size = static_cast<uint64_t>(st.st_size);
          }
        } else if (errno == ENOENT) {
          continue;  // Concurrently removed since `readdir`.
        }
        // Otherwise (e.g. `EACCES`), keep the kind from `d_type` and size 0.
      }
      out.push_back(std::move(entry));
    }
    detail::RecordSyscall();
    (void)closedir(stream);
    if (err != 0) return ErrnoStatus(err);
    return Status();
  }

  Status CreateDir(const Path& dir) override {
    PROFILER_ZONE("IO.Posix.CreateDir");
    for (;;) {
      detail::RecordSyscall();
      // 0777 so that the umask determines permissions, as for `mkdir(1)`.
      if (mkdir(dir.c_str(), 0777) == 0) return Status();
      const int err = errno;
      if (err == EINTR) {
        detail::RecordEintr();
        continue;
      }
      return ErrnoStatus(err);  // EEXIST (any kind of entry) -> kExists
    }
  }

  Status RemoveDir(const Path& dir) override {
    PROFILER_ZONE("IO.Posix.RemoveDir");
    for (;;) {
      detail::RecordSyscall();
      if (rmdir(dir.c_str()) == 0) return Status();
      int err = errno;
      if (err == EINTR) {
        detail::RecordEintr();
        continue;
      }
#ifdef ENOTEMPTY
      // POSIX allows `EEXIST` instead of `ENOTEMPTY` for non-empty directories.
      if (err == EEXIST) err = ENOTEMPTY;
#endif
      return ErrnoStatus(err);  // ENOTDIR -> kArgument
    }
  }
};

// Constant-initialized (constexpr constructor); see `FileSystemList::Head` in
// file_system.cc for why this makes registration from static initializers safe.
PosixFileSystem g_posix_file_system;

// Standard input or output. The descriptor is borrowed, i.e. not closed.
class StreamFilePosix final : public StreamFile {
 public:
  StreamFilePosix(int fd, Direction direction)
      : StreamFile(direction), fd_(fd) {}

  bool IsTerminal() const override {
    if (IsClosed()) return false;
    detail::RecordSyscall();
    return isatty(fd_) == 1;
  }

 protected:
  Status ReadSome(Span<uint8_t> to) const override {
    const size_t want = HWY_MIN(to.size(), File::kMaxBatchBytes);
    for (;;) {
      detail::RecordSyscall();
      const ssize_t ret = read(fd_, to.data(), want);
      if (ret >= 0) {
        const size_t got = static_cast<size_t>(ret);
        if (got != 0) {
          detail::RecordRead(got);
          if (got < want) detail::RecordShortRead();
        }
        return Status(got);
      }
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
  }

  Status WriteSome(Span<const uint8_t> from) override {
    const size_t want = HWY_MIN(from.size(), File::kMaxBatchBytes);
    for (;;) {
      detail::RecordSyscall();
      const ssize_t ret = write(fd_, from.data(), want);
      if (ret >= 0) {
        const size_t put = static_cast<size_t>(ret);
        detail::RecordWrite(put);
        return Status(put);
      }
      const int err = errno;
      if (err != EINTR) return ErrnoStatus(err);
      detail::RecordEintr();
    }
  }

 private:
  int fd_;
};

Status OpenStream(int fd, StreamFile::Direction direction,
                  std::unique_ptr<File>& file) {
  file.reset();
  // Fails with `EBADF` if the parent process closed the descriptor.
  detail::RecordSyscall();
  if (fcntl(fd, F_GETFL) == -1) {
    return Status::Error(Status::kNotFound, static_cast<uint64_t>(errno));
  }
  file = std::make_unique<StreamFilePosix>(fd, direction);
  return Status();
}

}  // namespace

HWY_IO_DLLEXPORT Status OpenStdin(std::unique_ptr<File>& file) {
  return OpenStream(STDIN_FILENO, StreamFile::Direction::kRead, file);
}

HWY_IO_DLLEXPORT Status OpenStdout(std::unique_ptr<File>& file) {
  return OpenStream(STDOUT_FILENO, StreamFile::Direction::kWrite, file);
}

HWY_IO_DLLEXPORT Status MaybeOpenHandle(intptr_t handle, OpenOptions options,
                                        HandleOwnership ownership,
                                        std::unique_ptr<File>& file) {
  PROFILER_ZONE("IO.Posix.OpenHandle");
  file.reset();
  const int fd = FdFromHandle(handle);
  if (fd < 0 || options.mode == FileMode::kWriteNew) return Status::kArgument;
  detail::RecordSyscall();
  const int fl = fcntl(fd, F_GETFL);
  if (fl == -1) return ErrnoStatus(errno);
  const int acc_mode = fl & O_ACCMODE;
  FdAccess access = FdAccess::kReadOnly;
  if (IsWriteMode(options.mode)) {
    if (acc_mode == O_WRONLY) {
      access = FdAccess::kWriteOnly;
    } else if (acc_mode == O_RDWR) {
      access = FdAccess::kReadWrite;
    } else {
      return IOCode::kWrongMode;
    }
    if ((fl & O_APPEND) != 0) return Status::kArgument;
  } else if (acc_mode != O_RDONLY && acc_mode != O_RDWR) {
    return IOCode::kWrongMode;
  }
  FdKind kind = FdKind::kRegular;
  const Status status = ClassifyFd(fd, options.mode, kind);
  if (!status.ok()) return status;
  return FinishOpen(fd, options, access, ownership, kind, file);
}

HWY_IO_DLLEXPORT Path TempDir() {
#if defined(__ANDROID__)
  const char* default_dir = "/data/local/tmp";
#else
  const char* default_dir = "/tmp";
#endif
  const char* env = getenv("TMPDIR");
  std::string dir = (env != nullptr && env[0] != '\0') ? env : default_dir;
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  return Path(std::move(dir));
}

namespace detail {
FileSystem& OSFileSystem() { return g_posix_file_system; }

FreePtr OSMapFreeFunc() { return &UnmapFile; }

Status AdviseOSMapRange(Span<const uint8_t> range, AccessHint hint,
                        MapEnd map_end) {
#if defined(MADV_NORMAL) && defined(MADV_SEQUENTIAL) && \
    defined(MADV_RANDOM) && defined(MADV_WILLNEED) && defined(MADV_DONTNEED)
  static const size_t page_size = []() -> size_t {
    const long ps = sysconf(_SC_PAGESIZE);  // NOLINT
    return (ps > 0 &&
            (static_cast<size_t>(ps) & (static_cast<size_t>(ps) - 1)) == 0)
               ? static_cast<size_t>(ps)
               : size_t{4096};
  }();

  const uintptr_t begin = reinterpret_cast<uintptr_t>(range.data());
  if (!SumFits<uintptr_t>(begin, range.size())) return Status::kArgument;
  const uintptr_t end = begin + range.size();
  const uintptr_t mask = page_size - 1;

  uintptr_t aligned_begin = 0;
  size_t aligned_len = 0;
  int advice = MADV_NORMAL;
  switch (hint) {
    case AccessHint::kNormal:
      advice = MADV_NORMAL;
      break;
    case AccessHint::kSequential:
      advice = MADV_SEQUENTIAL;
      break;
    case AccessHint::kRandom:
      advice = MADV_RANDOM;
      break;
    case AccessHint::kWillNeed:
      advice = MADV_WILLNEED;
      break;
    case AccessHint::kDontNeed:
      advice = MADV_DONTNEED;
      break;
  }
  if (hint == AccessHint::kDontNeed) {
    // Only evict whole pages strictly inside `range` so neighboring slices
    // sharing boundary pages are not prematurely evicted, except at the end
    // of an entire `MapPtr` where no slice follows.
    if (!SumFits<uintptr_t>(begin, mask)) return Status();
    aligned_begin = (begin + mask) & ~mask;
    uintptr_t aligned_end = end & ~mask;
    if (map_end == MapEnd::kAtEnd) {
      if (!SumFits<uintptr_t>(end, mask)) return Status();
      aligned_end = (end + mask) & ~mask;
    }
    if (aligned_begin >= aligned_end) return Status();
    aligned_len = static_cast<size_t>(aligned_end - aligned_begin);
  } else {
    aligned_begin = begin & ~mask;
    aligned_len = static_cast<size_t>(end - aligned_begin);
  }

  for (;;) {
    detail::RecordSyscall();
    if (madvise(reinterpret_cast<void*>(aligned_begin), aligned_len, advice) ==
        0) {
      return Status();
    }
    const int err = errno;
    if (err != EINTR) return ErrnoStatus(err);
    detail::RecordEintr();
  }
#else
  (void)range;
  (void)hint;
  (void)map_end;
  return Status::kUnimplemented;
#endif
}
}  // namespace detail

}  // namespace hwy

#else  // !HWY_HAVE_POSIX_IO

namespace hwy {

HWY_IO_DLLEXPORT Status MaybeOpenHandle(intptr_t /*handle*/,
                                        OpenOptions /*options*/,
                                        HandleOwnership /*ownership*/,
                                        std::unique_ptr<File>& file) {
  file.reset();
  return Status::kUnimplemented;
}

}  // namespace hwy

#endif  // HWY_HAVE_POSIX_IO
