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
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/file_util.h"
#include "hwy/io/io_test_util.h"
#include "hwy/io/memory_file.h"
#include "hwy/io/path.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#if HWY_IO_TEST_OS_FILES && !HWY_OS_WIN
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>  // memfd_create
#include <unistd.h>
#define HWY_IO_TEST_POSIX 1
#else
#define HWY_IO_TEST_POSIX 0
#endif

namespace hwy {
namespace {

TEST(FileSystemTest, TestOpenMissingFile) {
  FileStat st;
  st.size = 99;
  HWY_ASSERT(StatFile(Path(), st).Is(Status::kNotFound));
  HWY_ASSERT_EQ(uint64_t{0}, st.size);

  // Empty paths are rejected before dispatching to any `FileSystem`.
  {
    std::unique_ptr<File> empty_file;
    HWY_ASSERT(MaybeOpenFile(Path(), FileMode::kRead, empty_file)
                   .Is(Status::kNotFound));
    HWY_ASSERT(empty_file == nullptr);
    HWY_ASSERT(PathCapabilities(Path()).Empty());
    HWY_ASSERT(RenameFile(Path(), Path("a")).Is(Status::kNotFound));
    HWY_ASSERT(RenameFile(Path("a"), Path()).Is(Status::kNotFound));
    HWY_ASSERT(RemoveFile(Path()).Is(Status::kNotFound));
    HWY_ASSERT(!PathExists(Path()));
  }

  const Path missing = TempPathname("hwy_io_test_does_not_exist.bin");
  HWY_ASSERT(!IsFile(missing));
  HWY_ASSERT(!PathExists(missing));
  st.size = 99;
  const Status missing_stat = StatFile(missing, st);
  HWY_ASSERT(!missing_stat.ok());
  HWY_ASSERT_EQ(uint64_t{0}, st.size);
  std::unique_ptr<File> file;
  const Status status = MaybeOpenFile(missing, FileMode::kRead, file);
  HWY_ASSERT(!status.ok());
  HWY_ASSERT(file == nullptr);
#if HWY_IO_TEST_OS_FILES
  HWY_ASSERT(missing_stat.Is(Status::kNotFound));
  HWY_ASSERT(status.Is(Status::kNotFound));
  HWY_ASSERT(status.payload() != 0);
#endif
  // kWriteUpdate creates missing files; clean up afterwards.
#if HWY_IO_TEST_OS_FILES
  {
    std::unique_ptr<File> created;
    HWY_ASSERT(MaybeOpenFile(missing, FileMode::kWriteUpdate, created).ok());
    HWY_ASSERT(created != nullptr);
    HWY_ASSERT(created->Size() == Status(0));
  }
  HWY_ASSERT(IsFile(missing));
  HWY_ASSERT(PathExists(missing));
  const uint8_t payload[4] = {10, 20, 30, 40};
  AlignedPtr<uint8_t> aligned;
  // Atomic temp-then-rename output and RemoveFile.
  const Path renamed = TempPathname("hwy_io_test_renamed.bin");
  (void)RemoveFile(renamed);
  HWY_ASSERT(RenameFile(missing, renamed).ok());
  HWY_ASSERT(!IsFile(missing));
  HWY_ASSERT(IsFile(renamed));
  // Overwriting an existing destination via RenameFile succeeds atomically,
  // even while an existing reader handle on `renamed` remains open (exercising
  // `FileRenameInfoEx` + `FILE_RENAME_FLAG_POSIX_SEMANTICS` on Windows).
  {
    std::unique_ptr<File> open_dst = OpenFileOrAbort(renamed, FileMode::kRead);
    HWY_ASSERT(WriteBytesToFile(missing, Span<const uint8_t>(payload, 4)) ==
               Status(4));
    HWY_ASSERT(RenameFile(missing, renamed).ok());
    HWY_ASSERT(open_dst->Size() == Status(0));
  }
  HWY_ASSERT(!IsFile(missing));
  HWY_ASSERT(ReadFileToAligned(renamed, aligned) == Status(4));
  HWY_ASSERT_ARRAY_EQ(payload, aligned.get(), 4);
  HWY_ASSERT(RemoveFile(renamed).ok());
  HWY_ASSERT(!IsFile(renamed));
  HWY_ASSERT(RemoveFile(renamed).Is(Status::kNotFound));
  HWY_ASSERT(RenameFile(renamed, missing).Is(Status::kNotFound));
#endif
}

TEST(FileSystemTest, TestOpenDirectoryFails) {
#if HWY_IO_TEST_OS_FILES
  const char* dir = getenv("TEST_TMPDIR");
  const Path path(dir != nullptr && dir[0] != '\0' ? dir : ".");
  std::unique_ptr<File> file;
  const Status status = MaybeOpenFile(path, FileMode::kRead, file);
  HWY_ASSERT(status.Is(Status::kArgument));
  HWY_ASSERT(status.payload() != 0);
  HWY_ASSERT(file == nullptr);
  HWY_ASSERT(!IsFile(path));
  FileStat dir_stat;
  // StatFile succeeds for directories.
  dir_stat.size = 99;
  HWY_ASSERT(StatFile(path, dir_stat) == Status(0));
  HWY_ASSERT(dir_stat.kind == EntryKind::kDir);
  HWY_ASSERT_EQ(uint64_t{0}, dir_stat.size);
  const Status write_status = MaybeOpenFile(path, FileMode::kWriteUpdate, file);
  HWY_ASSERT(write_status.Is(Status::kArgument));
  HWY_ASSERT(RemoveFile(path).Is(Status::kArgument));

  // Renaming a regular file onto an existing directory (or vice versa) fails
  // with Status::kArgument without deleting either entry.
  const Path sub_dir = TempPathname("hwy_io_test_subdir");
  const Path temp_file = TempPathname("hwy_io_test_dir_rename.bin");
  (void)RemoveDir(sub_dir);
  HWY_ASSERT(CreateDir(sub_dir).ok());
#if HWY_OS_WIN
  HWY_ASSERT(!IsFile(Path("NUL")));
  HWY_ASSERT(StatFile(Path("NUL"), dir_stat) == Status(0));
  HWY_ASSERT(dir_stat.kind == EntryKind::kOther);
#else
  HWY_ASSERT(!IsFile(Path("/dev/null")));
  HWY_ASSERT(StatFile(Path("/dev/null"), dir_stat) == Status(0));
  HWY_ASSERT(dir_stat.kind == EntryKind::kOther);
#endif
  const uint8_t kByte = 0x5A;
  HWY_ASSERT(WriteBytesToFile(temp_file, ByteSpan(&kByte, 1)) == Status(1));
  HWY_ASSERT(!IsFile(sub_dir));
  const Status file_to_dir = RenameFile(temp_file, sub_dir);
  if (!file_to_dir.Is(Status::kArgument)) {
    HWY_ABORT("RenameFile(file, dir): %s (payload %llu)", file_to_dir.Name(),
              static_cast<unsigned long long>(file_to_dir.payload()));
  }
  const Status dir_to_file = RenameFile(sub_dir, temp_file);
  if (!dir_to_file.Is(Status::kArgument)) {
    HWY_ABORT("RenameFile(dir, file): %s (payload %llu)", dir_to_file.Name(),
              static_cast<unsigned long long>(dir_to_file.payload()));
  }
  HWY_ASSERT(IsFile(temp_file));
  HWY_ASSERT(RemoveFile(sub_dir).Is(Status::kArgument));
  HWY_ASSERT(RemoveFile(temp_file).ok());
  HWY_ASSERT(RemoveDir(sub_dir).ok());
#if !HWY_OS_WIN
  // POSIX symlinks and permission corner cases for StatFile and RemoveFile.
  const Path parent_dir = TempPathname("hwy_io_test_perm_dir");
  const Path target_dir = TempPathname("hwy_io_test_target_dir");
  const Path target_file = TempPathname("hwy_io_test_target_file.bin");
  (void)RemoveDir(parent_dir);
  (void)RemoveDir(target_dir);
  HWY_ASSERT(CreateDir(parent_dir).ok());
  HWY_ASSERT(CreateDir(target_dir).ok());
  HWY_ASSERT(WriteBytesToFile(target_file, ByteSpan(&kByte, 1)) == Status(1));

  const Path link_to_dir = parent_dir.Join("link_to_dir");
  const Path link_to_file = parent_dir.Join("link_to_file");
  (void)unlink(link_to_dir.path.c_str());
  (void)unlink(link_to_file.path.c_str());
  HWY_ASSERT_EQ(0, symlink(target_dir.path.c_str(), link_to_dir.path.c_str()));
  HWY_ASSERT_EQ(0,
                symlink(target_file.path.c_str(), link_to_file.path.c_str()));

  HWY_ASSERT(IsFile(link_to_file));
  FileStat link_stat;
  HWY_ASSERT(StatFile(link_to_file, link_stat) == Status(1));
  HWY_ASSERT(link_stat.kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{1}, link_stat.size);
  HWY_ASSERT(!IsFile(link_to_dir));
  // StatFile follows symlinks.
  HWY_ASSERT(StatFile(link_to_dir, link_stat) == Status(0));
  HWY_ASSERT(link_stat.kind == EntryKind::kDir);

  // Unprivileged permission checks (skipped if root or CAP_DAC_* bypasses DAC).
  HWY_ASSERT_EQ(0, chmod(parent_dir.path.c_str(), 0500));
  if (access(parent_dir.path.c_str(), W_OK) != 0) {
    // Unlinking a symlink to a directory in a non-writable parent must report
    // kPermission (via lstat), not kArgument (which following stat would).
    HWY_ASSERT(RemoveFile(link_to_dir).Is(Status::kPermission));
  }
  HWY_ASSERT_EQ(0, chmod(parent_dir.path.c_str(), 0000));
  if (access(link_to_file.path.c_str(), F_OK) != 0) {
    HWY_ASSERT(!IsFile(parent_dir.Join("missing")));
    HWY_ASSERT(StatFile(parent_dir.Join("missing"), link_stat)
                   .Is(Status::kPermission));
  }
  HWY_ASSERT_EQ(0, chmod(parent_dir.path.c_str(), 0700));

  // Read-denied regular file still exists via stat metadata.
  HWY_ASSERT_EQ(0, chmod(target_file.path.c_str(), 0000));
  HWY_ASSERT(IsFile(target_file));
  HWY_ASSERT(StatFile(target_file, link_stat) == Status(1));
  HWY_ASSERT_EQ(uint64_t{1}, link_stat.size);
  HWY_ASSERT_EQ(0, chmod(target_file.path.c_str(), 0600));
  // Identity: the symlink and its target are the same file, unlike the
  // directory; `File::Stat` of an open handle agrees with `StatFile`.
  {
    FileStat target_stat;
    HWY_ASSERT(StatFile(target_file, target_stat) == Status(1));
    HWY_ASSERT(target_stat.inode != 0);
    HWY_ASSERT(StatFile(link_to_file, link_stat) == Status(1));
    HWY_ASSERT(FileStat::SameFile(target_stat, link_stat));
    FileStat dir_identity;
    HWY_ASSERT(StatFile(target_dir, dir_identity).ok());
    HWY_ASSERT(!FileStat::SameFile(target_stat, dir_identity));
    std::unique_ptr<File> opened =
        OpenFileOrAbort(link_to_file, FileMode::kRead);
    FileStat open_stat;
    HWY_ASSERT(opened->Stat(open_stat) == Status(1));
    HWY_ASSERT(open_stat.kind == EntryKind::kFile);
    HWY_ASSERT(FileStat::SameFile(open_stat, target_stat));
    HWY_ASSERT(opened->Close().ok());
    HWY_ASSERT(opened->Stat(open_stat).Is(Status::kPrecondition));
    HWY_ASSERT(open_stat.inode == 0);
  }

  // Removing a symlink to a directory unlinks the link without touching target.
  HWY_ASSERT(RemoveFile(link_to_dir).ok());
  struct stat st;
  HWY_ASSERT_EQ(0, stat(target_dir.path.c_str(), &st));
  HWY_ASSERT(S_ISDIR(st.st_mode));

  HWY_ASSERT(RemoveFile(link_to_file).ok());
  HWY_ASSERT(RemoveFile(target_file).ok());
  HWY_ASSERT(RemoveDir(target_dir).ok());
  HWY_ASSERT(RemoveDir(parent_dir).ok());
#endif  // !HWY_OS_WIN
#endif  // HWY_IO_TEST_OS_FILES
}

// File identity via `StatFile` and `File::Stat` on all OS backends.
TEST(FileSystemTest, TestIdentity) {
#if HWY_IO_TEST_OS_FILES
  const Path path_a = TempPathname("hwy_io_test_identity_a.bin");
  const Path path_b = TempPathname("hwy_io_test_identity_b.bin");
  const Path path_c = TempPathname("hwy_io_test_identity_c.bin");
  const uint8_t kBytes[3] = {1, 2, 3};
  HWY_ASSERT(WriteBytesToFile(path_a, ByteSpan(kBytes, 3)) == Status(3));
  HWY_ASSERT(WriteBytesToFile(path_b, ByteSpan(kBytes, 2)) == Status(2));

  FileStat stat_a, stat_b;
  HWY_ASSERT(StatFile(path_a, stat_a) == Status(3));
  HWY_ASSERT(StatFile(path_b, stat_b) == Status(2));
  HWY_ASSERT(stat_a.inode != 0 && stat_b.inode != 0);
  HWY_ASSERT(FileStat::SameFile(stat_a, stat_a));
  HWY_ASSERT(!FileStat::SameFile(stat_a, stat_b));

  // An open handle reports the same identity, also after a rename.
  {
    std::unique_ptr<File> file = OpenFileOrAbort(path_b, FileMode::kRead);
    FileStat open_stat;
    HWY_ASSERT(file->Stat(open_stat) == Status(2));
    HWY_ASSERT(open_stat.kind == EntryKind::kFile);
    HWY_ASSERT(FileStat::SameFile(open_stat, stat_b));
    HWY_ASSERT(!FileStat::SameFile(open_stat, stat_a));
    HWY_ASSERT(file->Close().ok());
    HWY_ASSERT(file->Stat(open_stat).Is(Status::kPrecondition));
    HWY_ASSERT(open_stat.inode == 0);
  }
  HWY_ASSERT(RenameFile(path_b, path_c).ok());
  FileStat stat_c;
  HWY_ASSERT(StatFile(path_c, stat_c) == Status(2));
  HWY_ASSERT(FileStat::SameFile(stat_b, stat_c));

  // Directories also have an identity, distinct from their files.
  FileStat dir_stat;
  HWY_ASSERT(StatFile(path_a.Dirname(), dir_stat).ok());
  HWY_ASSERT(dir_stat.kind == EntryKind::kDir && dir_stat.inode != 0);
  HWY_ASSERT(!FileStat::SameFile(dir_stat, stat_a));

  HWY_ASSERT(RemoveFile(path_a).ok());
  HWY_ASSERT(RemoveFile(path_c).ok());
#endif  // HWY_IO_TEST_OS_FILES
}

// Returns the entry named `name`, or nullptr.
HWY_MAYBE_UNUSED const DirEntry* FindEntry(const std::vector<DirEntry>& entries,
                                           const std::string& name) {
  for (const DirEntry& entry : entries) {
    if (entry.name == name) return &entry;
  }
  return nullptr;
}

TEST(FileSystemTest, TestDirectories) {
  std::vector<DirEntry> entries;
  entries.resize(1);
  HWY_ASSERT(ListDir(Path(), entries).Is(Status::kNotFound));
  HWY_ASSERT(entries.empty());
  HWY_ASSERT(CreateDir(Path()).Is(Status::kNotFound));
  HWY_ASSERT(RemoveDir(Path()).Is(Status::kNotFound));
#if HWY_IO_TEST_OS_FILES
  const Path tmp = TempDir();
  HWY_ASSERT(!tmp.empty());

  const Path dir = TempPathname("hwy_io_test_list_dir");
  const Path sub_dir = dir.Join("sub");
  const Path file_a = dir.Join("a.bin");
  const Path file_b = dir.Join(".hidden");
  // Clean up after a previous failed run.
  (void)RemoveFile(file_a);
  (void)RemoveFile(file_b);
  (void)RemoveDir(sub_dir);
  (void)RemoveDir(dir);

  entries.resize(1);
  HWY_ASSERT(ListDir(dir, entries).Is(Status::kNotFound));
  HWY_ASSERT(entries.empty());
  HWY_ASSERT(RemoveDir(dir).Is(Status::kNotFound));
  // Parent does not exist.
  HWY_ASSERT(CreateDir(sub_dir).Is(Status::kNotFound));

  HWY_ASSERT(CreateDir(dir) == Status());
  HWY_ASSERT(PathExists(dir));  // Also true for directories.
  const Status exists_status = CreateDir(dir);
  HWY_ASSERT(exists_status.Is(IOCode::kExists));
  // `FileSystem` functions register the `IOCode` names.
  HWY_ASSERT(strcmp(exists_status.Name(), "kExists") == 0);
  HWY_ASSERT(ListDir(dir, entries) == Status(0));
  HWY_ASSERT(entries.empty());

  const uint8_t kBytes[3] = {1, 2, 3};
  HWY_ASSERT(WriteBytesToFile(file_a, ByteSpan(kBytes, 3)) == Status(3));
  HWY_ASSERT(WriteBytesToFile(file_b, Span<const uint8_t>()) == Status(0));
  HWY_ASSERT(CreateDir(sub_dir) == Status());

  // Files are not directories.
  HWY_ASSERT(CreateDir(file_a).Is(IOCode::kExists));
  HWY_ASSERT(ListDir(file_a, entries).Is(Status::kArgument));
  HWY_ASSERT(entries.empty());
  HWY_ASSERT(RemoveDir(file_a).Is(Status::kArgument));
  HWY_ASSERT(IsFile(file_a));

  // Order is unspecified; "." and ".." are excluded; hidden files included.
  HWY_ASSERT(ListDir(dir, entries) == Status(3));
  HWY_ASSERT_EQ(size_t{3}, entries.size());
  const DirEntry* entry_a = FindEntry(entries, "a.bin");
  const DirEntry* entry_b = FindEntry(entries, ".hidden");
  const DirEntry* entry_sub = FindEntry(entries, "sub");
  HWY_ASSERT(entry_a && entry_b && entry_sub);
  HWY_ASSERT(entry_a->kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{3}, entry_a->size);
  HWY_ASSERT(entry_b->kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{0}, entry_b->size);
  HWY_ASSERT(entry_sub->kind == EntryKind::kDir);
  HWY_ASSERT_EQ(uint64_t{0}, entry_sub->size);
  // Names rejoin to paths.
  HWY_ASSERT(IsFile(dir.Join(entry_a->name)));
  HWY_ASSERT(ListDir(dir.Join(entry_sub->name), entries) == Status(0));

#if !HWY_OS_WIN
  // Trailing separators are accepted. ListDir reports symlinks (including
  // dangling ones) as kSymlink, whereas StatFile follows them.
  const Path link = dir.Join("link");
  const Path dangling = dir.Join("dangling");
  (void)unlink(link.c_str());
  (void)unlink(dangling.c_str());
  HWY_ASSERT_EQ(0, symlink(sub_dir.c_str(), link.c_str()));
  HWY_ASSERT_EQ(0, symlink(dir.Join("missing").c_str(), dangling.c_str()));
  HWY_ASSERT(ListDir(dir.Join(""), entries) == Status(5));  // Trailing '/'.
  const DirEntry* entry_link = FindEntry(entries, "link");
  HWY_ASSERT(entry_link && entry_link->kind == EntryKind::kSymlink);
  HWY_ASSERT_EQ(uint64_t{0}, entry_link->size);
  const DirEntry* entry_dangling = FindEntry(entries, "dangling");
  HWY_ASSERT(entry_dangling && entry_dangling->kind == EntryKind::kSymlink);
  FileStat link_stat;
  HWY_ASSERT(StatFile(link, link_stat) == Status(0));
  HWY_ASSERT(link_stat.kind == EntryKind::kDir);
  HWY_ASSERT(StatFile(dangling, link_stat).Is(Status::kNotFound));
  HWY_ASSERT(PathExists(link));
  HWY_ASSERT(!PathExists(dangling));
  HWY_ASSERT(RemoveDir(link).Is(Status::kArgument));
  HWY_ASSERT(RemoveFile(link).ok());
  HWY_ASSERT(RemoveFile(dangling).ok());
#endif

  // Non-empty directories cannot be removed.
  HWY_ASSERT(RemoveDir(dir).Is(Status::kPrecondition));
  HWY_ASSERT(RemoveFile(file_a).ok());
  HWY_ASSERT(RemoveFile(file_b).ok());
  HWY_ASSERT(RemoveDir(sub_dir).ok());
  HWY_ASSERT(RemoveDir(dir).ok());
  HWY_ASSERT(ListDir(dir, entries).Is(Status::kNotFound));
#endif  // HWY_IO_TEST_OS_FILES
}

// Mock writable file whose `Close()` fails, verifying `WriteBytesToFile`
// surfaces deferred flush errors that `~File()` would ignore.
class CloseFailingFile : public MemoryFile {
 public:
  explicit CloseFailingFile(Span<uint8_t> buf)
      : MemoryFile(buf, FileMode::kWriteTruncate) {}
  Status Close() override {
    (void)MemoryFile::Close();
    return closed_once_++ == 0 ? Status(Status::kResource, 28) : Status();
  }

 private:
  uint8_t closed_once_ = 0;
};

// Test file system for "hook://" paths. File systems remain registered for the
// lifetime of the process, hence `enabled` toggles whether it claims paths.
// `FreePtr` of mappings not created by the OS file system.
void FreeNothing(void* /*opaque*/, void* /*ptr*/) {}

class HookFileSystem final : public FileSystem {
 public:
  explicit HookFileSystem(const uint8_t* data) : data_(data) {}

  bool Claims(const Path& pathname) const override {
    return enabled && pathname.path.rfind("hook://", 0) == 0;
  }

  Status Open(const Path& pathname, OpenOptions options,
              std::unique_ptr<File>& file) override {
    ++open_calls;
    if (pathname == Path("hook://close_fail")) {
      static uint8_t sink[16];
      file = std::make_unique<CloseFailingFile>(Span<uint8_t>(sink, 16));
      return Status();
    }
    if (pathname != Path("hook://memory")) return Status::kUnimplemented;
    last_hint = options.hint;
    if (options.mode != FileMode::kRead) return IOCode::kWrongMode;
    file = std::make_unique<MemoryFile>(Span<const uint8_t>(data_, 3));
    return Status();
  }

  Status Stat(const Path& pathname, FileStat& out) override {
    ++stat_calls;
    if (pathname == Path("hook://memory")) {
      out.size = 3;
      out.kind = EntryKind::kFile;
      // Deliberately return payload 0 to verify StatFile normalizes to
      // Status(3).
      return Status();
    }
    if (pathname == Path("hook://huge")) {
      out.size = Status::kMaxPayload + 1;
      return Status();
    }
    return Status::kNotFound;
  }

  Status Rename(const Path& from, const Path& to) override {
    last_renamed_from = from.path;
    last_renamed_to = to.path;
    return Status();
  }

  Status Remove(const Path& pathname) override {
    last_removed = pathname.path;
    return Status();
  }

  Status ListDir(const Path& dir, std::vector<DirEntry>& out) override {
    if (dir != Path("hook://dir")) return Status::kNotFound;
    DirEntry entry;
    entry.name = "memory";
    entry.size = 3;
    entry.kind = EntryKind::kFile;
    out.push_back(entry);
    return Status();
  }

  Status CreateDir(const Path& dir) override {
    last_created = dir.path;
    return Status();
  }

  FileCaps Capabilities(const Path& /*pathname*/) const override {
    return FileCap::kRandomAccess | FileCap::kMapImmutable;
  }

  bool enabled = false;
  AccessHint last_hint = AccessHint::kNormal;
  size_t open_calls = 0;
  size_t stat_calls = 0;
  std::string last_renamed_from;
  std::string last_renamed_to;
  std::string last_removed;
  std::string last_created;

 private:
  const uint8_t* data_;
};

// Only implements the mandatory `Claims`, `Open` and `Stat`, so other
// operations return kUnimplemented.
class OtherFileSystem final : public FileSystem {
 public:
  bool Claims(const Path& pathname) const override {
    return enabled && pathname.path.rfind("other://", 0) == 0;
  }

  Status Open(const Path& pathname, OpenOptions /*options*/,
              std::unique_ptr<File>& file) override {
    if (pathname == Path("other://ok")) {
      static const uint8_t kByte = 7;
      file = std::make_unique<MemoryFile>(Span<const uint8_t>(&kByte, 1));
      return Status();
    }
    if (pathname == Path("other://perm_denied")) {
      return Status::kPermission;
    }
    return Status::kUnimplemented;
  }

  // Emulated via `Open`, which only works for files.
  Status Stat(const Path& pathname, FileStat& out) override {
    std::unique_ptr<File> file;
    const Status open_status = Open(pathname, FileMode::kRead, file);
    if (!open_status.ok()) return open_status;
    const Status size_status = file->Size();
    if (!size_status.ok()) return size_status;
    out.size = size_status.size();
    out.kind = EntryKind::kFile;
    return Status(out.size);
  }

  bool enabled = false;
};

TEST(FileSystemTest, TestFileSystem) {
  const Path hook_path("hook://memory");
  std::unique_ptr<File> file;
  static const uint8_t kData[3] = {1, 2, 3};
  static HookFileSystem test_fs(kData);
  static OtherFileSystem other_fs;
  struct RegisterTestFileSystems {
    RegisterTestFileSystems() {
      AddFileSystem(&other_fs);
      AddFileSystem(&test_fs);
    }
  };
  static RegisterTestFileSystems register_fs;
  (void)register_fs;

  test_fs.enabled = false;
  other_fs.enabled = false;
  test_fs.open_calls = 0;
  test_fs.stat_calls = 0;
  HWY_ASSERT(!IsFile(hook_path));
  HWY_ASSERT(!MaybeOpenFile(hook_path, FileMode::kRead, file).ok());

  test_fs.enabled = true;
  other_fs.enabled = true;
  // Overridden `Stat` is used directly without opening a File handle.
  FileStat hook_stat;
  HWY_ASSERT(StatFile(hook_path, hook_stat) == Status(3));
  HWY_ASSERT_EQ(uint64_t{3}, hook_stat.size);
  HWY_ASSERT_EQ(size_t{1}, test_fs.stat_calls);
  HWY_ASSERT_EQ(size_t{0}, test_fs.open_calls);
  HWY_ASSERT(IsFile(hook_path));
  HWY_ASSERT_EQ(size_t{2}, test_fs.stat_calls);
  HWY_ASSERT_EQ(size_t{0}, test_fs.open_calls);
  HWY_ASSERT(StatFile(Path("hook://huge"), hook_stat).Is(Status::kRange));
  HWY_ASSERT_EQ(uint64_t{0}, hook_stat.size);
  HWY_ASSERT(
      StatFile(Path("hook://unhandled"), hook_stat).Is(Status::kNotFound));
  HWY_ASSERT(!IsFile(Path("hook://unhandled")));
  HWY_ASSERT_EQ(size_t{5}, test_fs.stat_calls);
  HWY_ASSERT_EQ(size_t{0}, test_fs.open_calls);
  // `OtherFileSystem::Stat` opens the file and never treats
  // `Status::kPermission` as existence.
  HWY_ASSERT(StatFile(Path("other://ok"), hook_stat) == Status(1));
  HWY_ASSERT(hook_stat.kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{1}, hook_stat.size);
  HWY_ASSERT(IsFile(Path("other://ok")));
  const Status perm_status = StatFile(Path("other://perm_denied"), hook_stat);
  HWY_ASSERT(perm_status.Is(Status::kPermission));
  HWY_ASSERT(strcmp(perm_status.Name(), "kPermission") == 0);
  HWY_ASSERT_EQ(uint64_t{0}, hook_stat.size);
  HWY_ASSERT(!IsFile(Path("other://perm_denied")));
  HWY_ASSERT(
      StatFile(Path("other://missing"), hook_stat).Is(Status::kUnimplemented));
  HWY_ASSERT(!IsFile(Path("other://missing")));

  // Capabilities by path, defaulting to empty (unknown).
  HWY_ASSERT(PathCapabilities(hook_path) ==
             (FileCap::kRandomAccess | FileCap::kMapImmutable));
  HWY_ASSERT(PathCapabilities(Path("other://ok")).Empty());

  // AdviseMap only supports OS mappings: those of custom file systems (here
  // identified by a non-OS `free_fn`) are unimplemented, except empty ranges.
  {
    MapPtr map(kData, 3, &FreeNothing, /*opaque=*/nullptr);
    HWY_ASSERT(
        AdviseMap(map, 1, 0, AccessHint::kWillNeed).Is(Status::kUnimplemented));
    HWY_ASSERT(AdviseMap(map, 3, 0, AccessHint::kWillNeed).ok());
  }

  HWY_ASSERT(
      MaybeOpenFile(hook_path, {FileMode::kRead, AccessHint::kRandom}, file)
          .ok());
  HWY_ASSERT(test_fs.last_hint == AccessHint::kRandom);
  HWY_ASSERT(file != nullptr);
  HWY_ASSERT(file->Size() == Status(3));
  uint8_t bytes[3] = {};
  HWY_ASSERT(file->Read(0, WritableByteSpan(bytes)) == Status(3));
  HWY_ASSERT_EQ(uint8_t{2}, bytes[1]);
  HWY_ASSERT(file->Close().ok());
  // The file system's specific error is returned when it claims the path,
  // including `kUnimplemented` (never falling through to the OS).
  HWY_ASSERT(MaybeOpenFile(hook_path, FileMode::kWriteTruncate, file)
                 .Is(IOCode::kWrongMode));
  HWY_ASSERT(file == nullptr);
  HWY_ASSERT(MaybeOpenFile(Path("hook://unhandled"), FileMode::kRead, file)
                 .Is(Status::kUnimplemented));
  // WriteBytesToFile surfaces deferred Close() flush errors.
  HWY_ASSERT(WriteBytesToFile(Path("hook://close_fail"),
                              Span<const uint8_t>(kData, 3)) ==
             Status(Status::kResource, 28));
  // Rename, rejection of renames between file systems (custom <-> OS and
  // custom <-> custom), default kUnimplemented, and remove.
  HWY_ASSERT(RenameFile(Path("hook://src"), Path("hook://dst")).ok());
  HWY_ASSERT(test_fs.last_renamed_from == "hook://src");
  HWY_ASSERT(test_fs.last_renamed_to == "hook://dst");
  const Path missing = TempPathname("hwy_io_test_hook_missing.bin");
  HWY_ASSERT(RenameFile(Path("hook://src"), missing).Is(Status::kArgument));
  HWY_ASSERT(RenameFile(missing, Path("hook://dst")).Is(Status::kArgument));
  HWY_ASSERT(RenameFile(Path("hook://src"), Path("other://dst"))
                 .Is(Status::kArgument));
  HWY_ASSERT(RenameFile(Path("other://a"), Path("other://b"))
                 .Is(Status::kUnimplemented));
  HWY_ASSERT(RemoveFile(Path("other://a")).Is(Status::kUnimplemented));
  HWY_ASSERT(RemoveFile(Path("hook://dst")).ok());
  HWY_ASSERT(test_fs.last_removed == "hook://dst");

  // Directory operations; `ListDir` clears `entries` on entry and on error.
  std::vector<DirEntry> entries(2);
  HWY_ASSERT(ListDir(Path("hook://dir"), entries) == Status(1));
  HWY_ASSERT_EQ(size_t{1}, entries.size());
  HWY_ASSERT(entries[0].name == "memory");
  HWY_ASSERT(entries[0].kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{3}, entries[0].size);
  HWY_ASSERT(ListDir(Path("hook://missing"), entries).Is(Status::kNotFound));
  HWY_ASSERT(entries.empty());
  HWY_ASSERT(CreateDir(Path("hook://new_dir")).ok());
  HWY_ASSERT(test_fs.last_created == "hook://new_dir");
  HWY_ASSERT(RemoveDir(Path("hook://new_dir")).Is(Status::kUnimplemented));
  HWY_ASSERT(ListDir(Path("other://dir"), entries).Is(Status::kUnimplemented));
  HWY_ASSERT(CreateDir(Path("other://dir")).Is(Status::kUnimplemented));
  HWY_ASSERT(RemoveDir(Path("other://dir")).Is(Status::kUnimplemented));

  // When `Claims` declines a path, the OS file system's status is preserved.
  const Status missing_status = MaybeOpenFile(missing, FileMode::kRead, file);
  HWY_ASSERT(!missing_status.ok());
#if HWY_IO_TEST_OS_FILES
  HWY_ASSERT(missing_status.Is(Status::kNotFound));
#endif
  test_fs.enabled = false;
  other_fs.enabled = false;
  HWY_ASSERT(!IsFile(hook_path));
  HWY_ASSERT(!MaybeOpenFile(hook_path, FileMode::kRead, file).ok());
}

TEST(FileSystemTest, TestAdviseMap) {
  // Empty MapPtr is a safe no-op.
  MapPtr empty_map;
  HWY_ASSERT(AdviseMap(empty_map, 0, 0, AccessHint::kDontNeed).ok());
  HWY_ASSERT(
      AdviseMap(empty_map, 1, 0, AccessHint::kNormal).Is(Status::kRange));

  // Non-owning MemoryFile MapPtr: AdviseMap(map, ..., kDontNeed) must be a
  // no-op and must NOT pass heap memory to MADV_DONTNEED (which zeroes
  // anonymous pages on Linux).
  constexpr size_t kMemBytes = 20 * 1024;
  AlignedPtr<uint8_t> mem_buf = AllocateAligned<uint8_t>(kMemBytes);
  HWY_ASSERT(mem_buf != nullptr);
  for (size_t i = 0; i < kMemBytes; ++i) {
    mem_buf[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
  }
  MemoryFile mem_file(Span<const uint8_t>(mem_buf.get(), kMemBytes));
  MapPtr mem_map;
  HWY_ASSERT(mem_file.MapImmutable(mem_map) == Status(kMemBytes));
  HWY_ASSERT(AdviseMap(mem_map, 0, 0, AccessHint::kDontNeed).ok());
  HWY_ASSERT(AdviseMap(mem_map, 100, 8192, AccessHint::kWillNeed).ok());
  HWY_ASSERT(AdviseMap(mem_map, kMemBytes + 1, 0, AccessHint::kNormal)
                 .Is(Status::kRange));
  const uint64_t kMaxOff = static_cast<uint64_t>(LimitsMax<int64_t>());
  HWY_ASSERT(AdviseMap(mem_map, 1, kMaxOff, AccessHint::kNormal)
                 .Is(Status::kArgument));
  for (size_t i = 0; i < kMemBytes; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>((i * 37 + 11) & 0xFF), mem_map[i]);
  }

#if HWY_IO_TEST_OS_FILES
  const Path path = TempPathname("hwy_io_test_advise_map.bin");
  constexpr size_t kFileBytes = 20 * 1024 + 333;
  std::vector<uint8_t> expected(kFileBytes);
  for (size_t i = 0; i < kFileBytes; ++i) {
    expected[i] = static_cast<uint8_t>((i * 59 + 7) & 0xFF);
  }
  {
    std::unique_ptr<File> out = OpenFileOrAbort(path, FileMode::kWriteTruncate);
    HWY_ASSERT(out->Write(0, ByteSpan(expected)) == Status(kFileBytes));
    HWY_ASSERT(out->Close().ok());
  }

  MapPtr os_map;
  {
    std::unique_ptr<File> in = OpenFileOrAbort(path, FileMode::kRead);
    HWY_ASSERT(in->MapImmutable(os_map) == Status(kFileBytes));
    HWY_ASSERT(in->Close().ok());
  }
  // File is closed and destroyed; AdviseMap works on the whole MapPtr and on
  // subranges (including unaligned offsets and sub-page ranges).
  auto check_hint = [](Status s) {
    HWY_ASSERT(s.ok() || s.Is(Status::kUnimplemented));
  };
  check_hint(AdviseMap(os_map, 0, 0, AccessHint::kSequential));
  check_hint(AdviseMap(os_map, 77, 500, AccessHint::kDontNeed));
  check_hint(AdviseMap(os_map, 123, 9000, AccessHint::kWillNeed));
  check_hint(AdviseMap(os_map, 500, 12000, AccessHint::kDontNeed));
  check_hint(AdviseMap(os_map, 16384, 0, AccessHint::kDontNeed));
  check_hint(AdviseMap(os_map, 0, 0, AccessHint::kRandom));
  check_hint(AdviseMap(os_map, 0, 0, AccessHint::kNormal));

  check_hint(AdviseMap(os_map, 77, 12000, AccessHint::kWillNeed));
  check_hint(AdviseMap(os_map, 77, 12000, AccessHint::kDontNeed));

  // Mapped contents remain intact after page-cache eviction.
  HWY_ASSERT_ARRAY_EQ(expected.data(), os_map.get(), kFileBytes);
  os_map.reset();
  (void)RemoveFile(path);
#endif  // HWY_IO_TEST_OS_FILES
}

#if HWY_IO_TEST_POSIX
// Returns whether `fd` is (still) an open descriptor.
bool IsOpenFd(int fd) { return fcntl(fd, F_GETFD) != -1 || errno != EBADF; }
#endif

TEST(FileSystemTest, TestOpenFd) {
  std::unique_ptr<File> file;
#if HWY_IO_TEST_POSIX
  const Path path = TempPathname("hwy_io_test_open_fd.bin");
  const uint8_t kBytes[4] = {1, 2, 3, 4};
  uint8_t buf[4] = {};
  HWY_ASSERT(WriteBytesToFile(path, ByteSpan(kBytes, 4)) == Status(4));

  HWY_ASSERT(
      MaybeOpenHandle(-1, FileMode::kRead, HandleOwnership::kBorrow, file)
          .Is(Status::kArgument));
  // Handles that do not fit in a POSIX `int` descriptor are rejected.
  const int64_t too_large = int64_t{LimitsMax<int>()} + 1;
  if (too_large <= int64_t{LimitsMax<intptr_t>()}) {
    const intptr_t handle = static_cast<intptr_t>(too_large);
    HWY_ASSERT(
        MaybeOpenHandle(handle, FileMode::kRead, HandleOwnership::kBorrow, file)
            .Is(Status::kArgument));
  }
  const int rd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  HWY_ASSERT(rd >= 0);
  // Exclusive creation cannot be enforced for existing descriptors.
  HWY_ASSERT(
      MaybeOpenHandle(rd, FileMode::kWriteNew, HandleOwnership::kTake, file)
          .Is(Status::kArgument));
  // Read-only descriptor cannot be opened for writing.
  HWY_ASSERT(
      MaybeOpenHandle(rd, FileMode::kWriteUpdate, HandleOwnership::kTake, file)
          .Is(IOCode::kWrongMode));
  HWY_ASSERT(file == nullptr);
  HWY_ASSERT(IsOpenFd(rd));  // Not taken after failure.

  // Borrowed: `Close` and the destructor detach without closing.
  HWY_ASSERT(
      MaybeOpenHandle(rd, FileMode::kRead, HandleOwnership::kBorrow, file)
          .ok());
  HWY_ASSERT(file->Capabilities() == PathCapabilities(path));
  HWY_ASSERT(file->Read(0, WritableByteSpan(buf, 4)) == Status(4));
  HWY_ASSERT_ARRAY_EQ(kBytes, buf, 4);
  HWY_ASSERT(file->Write(0, ByteSpan(kBytes, 1)).Is(IOCode::kWrongMode));
  HWY_ASSERT(file->Close().ok());
  HWY_ASSERT(file->Read(0, WritableByteSpan(buf, 1)).Is(Status::kPrecondition));
  HWY_ASSERT(IsOpenFd(rd));
  HWY_ASSERT(
      MaybeOpenHandle(rd, FileMode::kRead, HandleOwnership::kBorrow, file)
          .ok());
  file.reset();
  HWY_ASSERT(IsOpenFd(rd));

  // Taken: `Close` closes the descriptor.
  HWY_ASSERT(
      MaybeOpenHandle(rd, FileMode::kRead, HandleOwnership::kTake, file).ok());
  HWY_ASSERT(file->Close().ok());
  HWY_ASSERT(!IsOpenFd(rd));

  // Write modes; kWriteTruncate truncates regular files.
  int rw = open(path.c_str(), O_RDWR | O_CLOEXEC);
  HWY_ASSERT(rw >= 0);
  HWY_ASSERT(MaybeOpenHandle(rw, FileMode::kWriteUpdate,
                             HandleOwnership::kBorrow, file)
                 .ok());
  HWY_ASSERT(file->Size() == Status(4));
  file.reset();
  HWY_ASSERT(MaybeOpenHandle(rw, FileMode::kWriteTruncate,
                             HandleOwnership::kTake, file)
                 .ok());
  HWY_ASSERT(file->Size() == Status(0));
  HWY_ASSERT(file->Write(0, ByteSpan(kBytes, 4)) == Status(4));
  file.reset();  // Destructor closes taken descriptors.
  HWY_ASSERT(!IsOpenFd(rw));
  FileStat stat;
  HWY_ASSERT(StatFile(path, stat) == Status(4));

  // `O_APPEND` would ignore `pwrite` offsets, hence only allowed for reading.
  rw = open(path.c_str(), O_RDWR | O_APPEND | O_CLOEXEC);
  HWY_ASSERT(rw >= 0);
  HWY_ASSERT(
      MaybeOpenHandle(rw, FileMode::kWriteUpdate, HandleOwnership::kTake, file)
          .Is(Status::kArgument));
  HWY_ASSERT(
      MaybeOpenHandle(rw, FileMode::kRead, HandleOwnership::kTake, file).ok());
  file.reset();

  // `O_WRONLY` descriptors only permit writes; reads and mappings report
  // `IOCode::kWrongMode` (rather than `EBADF`/`EACCES`), even when empty.
  const int wr = open(path.c_str(), O_WRONLY | O_CLOEXEC);
  HWY_ASSERT(wr >= 0);
  HWY_ASSERT(MaybeOpenHandle(wr, FileMode::kWriteTruncate,
                             HandleOwnership::kTake, file)
                 .ok());
  HWY_ASSERT(file->Size() == Status(0));
  HWY_ASSERT(file->Read(0, WritableByteSpan(buf, 0)).Is(IOCode::kWrongMode));
  MapPtr wr_map;
  HWY_ASSERT(file->MapImmutable(wr_map).Is(IOCode::kWrongMode));
  HWY_ASSERT(file->MapShared(wr_map).Is(IOCode::kWrongMode));
  MutableMapPtr wr_writable;
  HWY_ASSERT(file->MapWritable(wr_writable).Is(IOCode::kWrongMode));
  HWY_ASSERT(file->Write(0, ByteSpan(kBytes, 4)) == Status(4));
  HWY_ASSERT(file->Read(0, WritableByteSpan(buf, 4)).Is(IOCode::kWrongMode));
  HWY_ASSERT(file->MapImmutable(wr_map).Is(IOCode::kWrongMode));
  file.reset();

  // Handle() returns the descriptor, which MaybeOpenHandle accepts (as in a
  // sandboxee receiving it), and -1 after Close or for non-OS files.
  {
    std::unique_ptr<File> owner;
    HWY_ASSERT(MaybeOpenFile(path, FileMode::kRead, owner).ok());
    const intptr_t handle = owner->Handle();
    HWY_ASSERT(handle >= 0);
    std::unique_ptr<File> borrowed;
    HWY_ASSERT(MaybeOpenHandle(handle, FileMode::kRead,
                               HandleOwnership::kBorrow, borrowed)
                   .ok());
    HWY_ASSERT(borrowed->Handle() == handle);
    HWY_ASSERT(borrowed->Read(0, WritableByteSpan(buf, 4)) == Status(4));
    HWY_ASSERT_ARRAY_EQ(kBytes, buf, 4);
    HWY_ASSERT(borrowed->Close().ok());
    HWY_ASSERT(borrowed->Handle() == -1);
    HWY_ASSERT(IsOpenFd(static_cast<int>(handle)));  // borrowed
    HWY_ASSERT(owner->Close().ok());
    HWY_ASSERT(owner->Handle() == -1);
  }

  // Directories are rejected.
  const int dir = open(TempDir().c_str(), O_RDONLY | O_CLOEXEC);
  HWY_ASSERT(dir >= 0);
  HWY_ASSERT(MaybeOpenHandle(dir, FileMode::kRead, HandleOwnership::kTake, file)
                 .Is(Status::kArgument));
  HWY_ASSERT_EQ(0, close(dir));
  HWY_ASSERT(RemoveFile(path).ok());
#elif !HWY_IO_HAVE_OS_FILES || HWY_OS_WIN
  HWY_ASSERT(MaybeOpenHandle(0, FileMode::kRead, HandleOwnership::kBorrow, file)
                 .Is(Status::kUnimplemented));
  HWY_ASSERT(file == nullptr);
#endif
}

TEST(FileSystemTest, TestCharDevice) {
#if HWY_IO_TEST_POSIX
  const Path null_path("/dev/null");
  const uint8_t kBytes[4] = {1, 2, 3, 4};
  std::unique_ptr<File> file;
  // Only write modes accept character devices.
  HWY_ASSERT(
      MaybeOpenFile(null_path, FileMode::kRead, file).Is(Status::kArgument));
  const FileCaps kUnsupported = FileCap::kExtend | FileCap::kPreallocate |
                                FileCap::kSync | FileCap::kAdvise |
                                FileCap::kMapImmutable | FileCap::kMapShared |
                                FileCap::kMapWritable;
  for (FileMode mode : {FileMode::kWriteTruncate, FileMode::kWriteUpdate}) {
    HWY_ASSERT(MaybeOpenFile(null_path, mode, file).ok());
    const FileCaps caps = file->Capabilities();
    HWY_ASSERT(caps.HasAll(FileCap::kRandomAccess | FileCap::kRandomWrite));
    HWY_ASSERT(!caps.HasAny(kUnsupported));
    HWY_ASSERT(!file->IsTerminal());
    HWY_ASSERT(file->Write(123, ByteSpan(kBytes, 4)) == Status(4));
    HWY_ASSERT(file->Size() == Status(0));
    HWY_ASSERT(file->SetSize(1).Is(Status::kUnimplemented));
    HWY_ASSERT(file->Preallocate(0, 1).Is(Status::kUnimplemented));
    HWY_ASSERT(file->Sync().Is(Status::kUnimplemented));
    HWY_ASSERT(
        file->Advise(0, 0, AccessHint::kSequential).Is(Status::kUnimplemented));
    MapPtr map;
    HWY_ASSERT(file->MapImmutable(map).Is(Status::kUnimplemented));
    HWY_ASSERT(file->MapShared(map).Is(Status::kUnimplemented));
    MutableMapPtr writable;
    HWY_ASSERT(file->MapWritable(writable).Is(Status::kUnimplemented));
    HWY_ASSERT(file->Close().ok());
  }
  FileStat stat;
  HWY_ASSERT(StatFile(null_path, stat).ok());
  HWY_ASSERT(stat.kind == EntryKind::kOther);

  // Write-only descriptors are accepted for write modes.
  const int fd = open(null_path.c_str(), O_WRONLY | O_CLOEXEC);
  HWY_ASSERT(fd >= 0);
  HWY_ASSERT(MaybeOpenHandle(fd, FileMode::kRead, HandleOwnership::kTake, file)
                 .Is(IOCode::kWrongMode));
  HWY_ASSERT(MaybeOpenHandle(fd, FileMode::kWriteTruncate,
                             HandleOwnership::kTake, file)
                 .ok());
  HWY_ASSERT(file->Write(0, ByteSpan(kBytes, 4)) == Status(4));
  file.reset();

  // Writes to /dev/full fail with `ENOSPC` (if present, e.g. on Linux).
  const Path full_path("/dev/full");
  if (PathExists(full_path)) {
    HWY_ASSERT(MaybeOpenFile(full_path, FileMode::kWriteTruncate, file).ok());
    HWY_ASSERT(file->Write(0, ByteSpan(kBytes, 4)).Is(Status::kResource));
  }
#endif
}

TEST(FileSystemTest, TestMapSharedWritable) {
#if HWY_IO_TEST_OS_FILES
  const Path path = TempPathname("hwy_io_test_map_shared.bin");
  constexpr size_t kSize = 8192 + 5;
  std::vector<uint8_t> bytes(kSize);
  for (size_t i = 0; i < kSize; ++i) bytes[i] = static_cast<uint8_t>(i * 7);
  HWY_ASSERT(WriteBytesToFile(path, ByteSpan(bytes)) == Status(kSize));

  // Writable mappings require a write mode.
  MapPtr ro;
  MutableMapPtr rw;
  {
    std::unique_ptr<File> in = OpenFileOrAbort(path, FileMode::kRead);
    HWY_ASSERT(in->MapWritable(rw).Is(IOCode::kWrongMode));
    HWY_ASSERT(!rw);
    HWY_ASSERT(in->MapShared(ro) == Status(kSize));
    HWY_ASSERT_ARRAY_EQ(bytes.data(), ro.get(), kSize);
  }

  std::unique_ptr<File> file = OpenFileOrAbort(path, FileMode::kWriteUpdate);
  HWY_ASSERT(file->MapWritable(rw) == Status(kSize));
  // Writes via the `File` are visible through both shared mappings, including
  // the one obtained from a since-closed `File`.
  const uint8_t kNew = 0xAB;
  HWY_ASSERT(file->Write(7, ByteSpan(&kNew, 1)) == Status(1));
  HWY_ASSERT_EQ(kNew, ro[7]);
  HWY_ASSERT_EQ(kNew, rw[7]);
  // Stores via the writable mapping are visible to `Read` and other mappings.
  rw[100] = kNew;
  HWY_ASSERT_EQ(kNew, ro[100]);
  uint8_t got = 0;
  HWY_ASSERT(file->Read(100, WritableByteSpan(&got, 1)) == Status(1));
  HWY_ASSERT_EQ(kNew, got);
  // Same free function as `MapImmutable`, hence `AdviseMap` applies.
  const Status advise = AdviseMap(rw, 0, 0, AccessHint::kWillNeed);
  HWY_ASSERT(advise.ok() || advise.Is(Status::kUnimplemented));

  // Zero-copy `MemoryFile` over the mappings.
  MemoryFile in_mem(ro.span());
  HWY_ASSERT(in_mem.MaybeSpan(0, kSize).data() == ro.get());
  MemoryFile out_mem(rw.span(), FileMode::kWriteUpdate);
  const Span<uint8_t> span = out_mem.MaybeSpanForWrite(10, 4);
  HWY_ASSERT(span.data() == rw.get() + 10);
  span[0] = kNew;
  HWY_ASSERT_EQ(kNew, ro[10]);

  // Sandbox use case: the same `File*` interface over an owned writable
  // mapping, with zero-copy `MaybeSpanForWrite`. Destroying it unmaps.
  {
    MutableMapPtr owned;
    HWY_ASSERT(file->MapWritable(owned) == Status(kSize));
    std::unique_ptr<File> mapped =
        std::make_unique<MemoryFile>(std::move(owned), FileMode::kWriteUpdate);
    HWY_ASSERT(mapped->Size() == Status(kSize));
    const uint8_t kOther = 0xCD;
    HWY_ASSERT(mapped->Write(200, ByteSpan(&kOther, 1)) == Status(1));
    HWY_ASSERT_EQ(kOther, ro[200]);
    Span<uint8_t> direct = mapped->MaybeSpanForWrite(300, 1);
    HWY_ASSERT(direct.data() != nullptr);
    direct[0] = kOther;
    HWY_ASSERT(file->Read(300, WritableByteSpan(&got, 1)) == Status(1));
    HWY_ASSERT_EQ(kOther, got);
  }
  HWY_ASSERT(file->Close().ok());
  HWY_ASSERT(file->MapWritable(rw).Is(Status::kPrecondition));
  HWY_ASSERT(!rw);
  ro.reset();
  HWY_ASSERT(RemoveFile(path).ok());

// Bionic defines `MFD_CLOEXEC` regardless of the API level, but only declares
// `memfd_create` for API 30 and later.
#if HWY_IO_TEST_POSIX && HWY_OS_LINUX && defined(MFD_CLOEXEC) && \
    (!defined(__ANDROID_API__) || __ANDROID_API__ >= 30)
  // Reused memfd: host writes via a borrowed `File` are visible through a
  // shared mapping obtained from a separate read-only `File`.
  const int memfd = memfd_create("hwy_io_test", MFD_CLOEXEC);
  if (memfd >= 0) {
    std::unique_ptr<File> host;
    HWY_ASSERT(MaybeOpenHandle(memfd, FileMode::kWriteUpdate,
                               HandleOwnership::kBorrow, host)
                   .ok());
    HWY_ASSERT(host->SetSize(4096) == Status(4096));
    std::unique_ptr<File> guest;
    HWY_ASSERT(
        MaybeOpenHandle(memfd, FileMode::kRead, HandleOwnership::kBorrow, guest)
            .ok());
    MapPtr view;
    HWY_ASSERT(guest->MapShared(view) == Status(4096));
    for (uint8_t round = 1; round <= 3; ++round) {
      HWY_ASSERT(host->Write(0, ByteSpan(&round, 1)) == Status(1));
      HWY_ASSERT_EQ(round, view[0]);
    }
    view.reset();
    HWY_ASSERT_EQ(0, close(memfd));
  }
#endif  // HWY_IO_TEST_POSIX && HWY_OS_LINUX && memfd_create
#endif  // HWY_IO_TEST_OS_FILES
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
