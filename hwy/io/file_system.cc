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

#include "hwy/io/file_system.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <memory>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/io_internal.h"
#include "hwy/io/path.h"
#include "hwy/profiler.h"
#include "hwy/status.h"

namespace hwy {

namespace {

const char* IOCodeName(Status::Code code) {
  switch (
      static_cast<IOCode>(Status::MakeCode(Status::Severity::kError, code))) {
    case IOCode::kWrongMode:
      return "kWrongMode";
    case IOCode::kExists:
      return "kExists";
    case IOCode::kNoProgress:
      return "kNoProgress";
  }
  return nullptr;
}

// Registers the `IOCode` names for `Status::Name`. Idempotent and thread-safe.
// Called on first use of any `FileSystem` function, which is where users of OS
// files obtain most `IOCode`s. Binaries that only use freestanding `File`s such
// as `MemoryFile` need not link this TU; their `IOCode::kWrongMode` is then
// named "(module/app code)" by `Status::Name`.
void RegisterIOCodeNames() {
  (void)Status::SetModuleCodeNameFunc(Status::Module::kIO, &IOCodeName);
}

}  // namespace

namespace detail {

// Lock-free singly-linked list of `FileSystem`, most recently added first.
struct FileSystemList {
  static std::atomic<FileSystem*>& Head() {
    // Static initialization is safe in both directions: `head` is dynamically
    // initialized on first use (thread-safe, regardless of translation unit
    // order), and the OS file system it points to is constant-initialized,
    // hence usable (including its vptr) before any dynamic initializer runs.
    // The OS file system is the tail and claims all paths, so `Find` needs no
    // special case. Also registers the `IOCode` names.
    static std::atomic<FileSystem*> head{
        (RegisterIOCodeNames(), &OSFileSystem())};
    return head;
  }

  static void Add(FileSystem* fs) {
    // Re-adding would create a cycle.
    HWY_ASSERT(fs != nullptr && fs->next_ == nullptr && fs != &OSFileSystem());
    std::atomic<FileSystem*>& head = Head();
    FileSystem* old_head = head.load(std::memory_order_relaxed);
    do {
      fs->next_ = old_head;
    } while (!head.compare_exchange_weak(
        old_head, fs, std::memory_order_release, std::memory_order_relaxed));
  }

  static FileSystem& Find(const Path& pathname) {
    FileSystem* fs = Head().load(std::memory_order_acquire);
    while (!fs->Claims(pathname)) {
      fs = fs->next_;
      HWY_DASSERT(fs != nullptr);
    }
    return *fs;
  }
};

}  // namespace detail

HWY_IO_DLLEXPORT void AddFileSystem(FileSystem* fs) {
  detail::FileSystemList::Add(fs);
}

HWY_IO_DLLEXPORT Status MaybeOpenFile(const Path& pathname, OpenOptions options,
                                      std::unique_ptr<File>& file) {
  PROFILER_ZONE("IO.Open");
  file.reset();
  if (pathname.empty()) return Status::kNotFound;
  const Status status =
      detail::FileSystemList::Find(pathname).Open(pathname, options, file);
  if (!status.ok()) {
    file.reset();
  } else if (file == nullptr) {
    return Status::kInternal;
  }
  return status;
}

HWY_IO_DLLEXPORT FileCaps PathCapabilities(const Path& pathname) {
  if (pathname.empty()) return FileCaps();
  return detail::FileSystemList::Find(pathname).Capabilities(pathname);
}

HWY_IO_DLLEXPORT Status StatFile(const Path& pathname, FileStat& out) {
  PROFILER_ZONE("IO.Stat");
  out = FileStat{};
  if (pathname.empty()) return Status::kNotFound;
  const Status status =
      detail::FileSystemList::Find(pathname).Stat(pathname, out);
  if (!status.ok()) {
    out = FileStat{};
    return status;
  }
  if (out.size > Status::kMaxPayload) {
    out = FileStat{};
    return Status::kRange;
  }
  return Status(out.size);
}

HWY_IO_DLLEXPORT Status RenameFile(const Path& from, const Path& to) {
  PROFILER_ZONE("IO.Rename");
  if (from.empty() || to.empty()) return Status::kNotFound;
  FileSystem& fs_from = detail::FileSystemList::Find(from);
  FileSystem& fs_to = detail::FileSystemList::Find(to);
  if (&fs_from != &fs_to) return Status::kArgument;
  return fs_from.Rename(from, to);
}

HWY_IO_DLLEXPORT Status RemoveFile(const Path& pathname) {
  PROFILER_ZONE("IO.Remove");
  if (pathname.empty()) return Status::kNotFound;
  return detail::FileSystemList::Find(pathname).Remove(pathname);
}

HWY_IO_DLLEXPORT Status ListDir(const Path& dir, std::vector<DirEntry>& out) {
  PROFILER_ZONE("IO.ListDir");
  out.clear();
  if (dir.empty()) return Status::kNotFound;
  const Status status = detail::FileSystemList::Find(dir).ListDir(dir, out);
  if (!status.ok()) {
    out.clear();
    return status;
  }
  return Status(out.size());
}

HWY_IO_DLLEXPORT Status CreateDir(const Path& dir) {
  PROFILER_ZONE("IO.CreateDir");
  if (dir.empty()) return Status::kNotFound;
  return detail::FileSystemList::Find(dir).CreateDir(dir);
}

HWY_IO_DLLEXPORT Status RemoveDir(const Path& dir) {
  PROFILER_ZONE("IO.RemoveDir");
  if (dir.empty()) return Status::kNotFound;
  return detail::FileSystemList::Find(dir).RemoveDir(dir);
}

Status FileSystem::Rename(const Path& /*from*/, const Path& /*to*/) {
  return Status::kUnimplemented;
}

Status FileSystem::Remove(const Path& /*pathname*/) {
  return Status::kUnimplemented;
}

Status FileSystem::ListDir(const Path& /*dir*/,
                           std::vector<DirEntry>& /*out*/) {
  return Status::kUnimplemented;
}

Status FileSystem::CreateDir(const Path& /*dir*/) {
  return Status::kUnimplemented;
}

Status FileSystem::RemoveDir(const Path& /*dir*/) {
  return Status::kUnimplemented;
}

FileCaps FileSystem::Capabilities(const Path& /*pathname*/) const {
  return FileCaps();
}

#if !HWY_IO_HAVE_OS_FILES

namespace {

// Claims all paths so that lookups always succeed, but cannot open files: only
// file systems registered via `AddFileSystem` can.
class NoOSFileSystem final : public FileSystem {
 public:
  bool Claims(const Path& /*pathname*/) const override { return true; }
  Status Open(const Path& /*pathname*/, OpenOptions /*options*/,
              std::unique_ptr<File>& /*file*/) override {
    return Status::kUnimplemented;
  }
  Status Stat(const Path& /*pathname*/, FileStat& /*out*/) override {
    return Status::kUnimplemented;
  }
};

// Constant-initialized; see `FileSystemList::Head`.
NoOSFileSystem g_no_os_file_system;

}  // namespace

HWY_IO_DLLEXPORT Path TempDir() { return Path(); }

namespace detail {
FileSystem& OSFileSystem() { return g_no_os_file_system; }
FreePtr OSMapFreeFunc() { return nullptr; }
Status AdviseOSMapRange(Span<const uint8_t> /*range*/, AccessHint /*hint*/,
                        MapEnd /*map_end*/) {
  return Status::kUnimplemented;
}
}  // namespace detail

#endif  // !HWY_IO_HAVE_OS_FILES

namespace {

// Shared by the `MapPtr` and `MutableMapPtr` overloads of `AdviseMap`.
Status AdviseMapping(Span<const uint8_t> mapping, FreePtr free_fn,
                     uint64_t offset, uint64_t size, AccessHint hint) {
  if (!SumFits<int64_t>(offset, size)) return Status::kArgument;
  if (offset > mapping.size()) return Status::kRange;
  const size_t offset_s = SafeCast<size_t>(offset);
  const size_t avail = mapping.size() - offset_s;
  const size_t len =
      (size == 0 || size > avail) ? avail : SafeCast<size_t>(size);
  // Non-owning mappings (`MemoryFile`, `FileView`) are not OS mappings.
  if (len == 0 || free_fn == nullptr) return Status();
  // Note that `OSMapFreeFunc()` is nullptr without OS file support.
  if (free_fn != detail::OSMapFreeFunc()) return Status::kUnimplemented;
  const detail::MapEnd map_end = offset_s + len == mapping.size()
                                     ? detail::MapEnd::kAtEnd
                                     : detail::MapEnd::kInterior;
  return detail::AdviseOSMapRange(
      Span<const uint8_t>(mapping.data() + offset_s, len), hint, map_end);
}

}  // namespace

HWY_IO_DLLEXPORT Status AdviseMap(const MapPtr& map, uint64_t offset,
                                  uint64_t size, AccessHint hint) {
  return AdviseMapping(Span<const uint8_t>(map.get(), map.size()),
                       map.free_fn(), offset, size, hint);
}

HWY_IO_DLLEXPORT Status AdviseMap(const MutableMapPtr& map, uint64_t offset,
                                  uint64_t size, AccessHint hint) {
  return AdviseMapping(Span<const uint8_t>(map.get(), map.size()),
                       map.free_fn(), offset, size, hint);
}

}  // namespace hwy
