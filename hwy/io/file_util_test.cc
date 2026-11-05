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

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/file_system.h"
#include "hwy/io/io_test_util.h"
#include "hwy/io/memory_file.h"
#include "hwy/io/path.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#if HWY_IO_TEST_OS_FILES && !HWY_OS_WIN
#include <fcntl.h>  // open
#include <unistd.h>
#endif

namespace hwy {
namespace {

TEST(FileUtilTest, TestVerifyEmpty) {
  MemoryFile empty(Span<const uint8_t>{});
  HWY_ASSERT(VerifyEmpty(empty).ok());
  const uint8_t byte = 1;
  MemoryFile non_empty(Span<const uint8_t>(&byte, 1));
  HWY_ASSERT(VerifyEmpty(non_empty).Is(Status::kUnimplemented));
#if HWY_IO_TEST_OS_FILES
  // Empty regular file.
  const Path path = TempPathname("hwy_io_file_util_verify_empty.bin");
  HWY_ASSERT(WriteBytesToFile(path, Span<const uint8_t>()) == Status(0));
  {
    std::unique_ptr<File> file = OpenFileOrAbort(path, FileMode::kRead);
    HWY_ASSERT(VerifyEmpty(*file).ok());
  }
#if !HWY_OS_WIN
  // Write-only descriptors cannot be read.
  const int fd = open(path.c_str(), O_WRONLY | O_CLOEXEC);
  HWY_ASSERT(fd >= 0);
  std::unique_ptr<File> write_only;
  HWY_ASSERT(MaybeOpenHandle(fd, FileMode::kWriteUpdate, HandleOwnership::kTake,
                             write_only)
                 .ok());
  HWY_ASSERT(VerifyEmpty(*write_only).Is(IOCode::kWrongMode));
  write_only.reset();
#endif
  HWY_ASSERT(RemoveFile(path).ok());
#endif  // HWY_IO_TEST_OS_FILES
#if HWY_IO_TEST_OS_FILES && HWY_OS_LINUX
  // Reports size 0 but has contents.
  std::unique_ptr<File> proc;
  if (MaybeOpenFile(Path("/proc/self/status"), FileMode::kRead, proc).ok()) {
    HWY_ASSERT(proc->Size() == Status(0));
    HWY_ASSERT(VerifyEmpty(*proc).Is(Status::kUnimplemented));
  }
#endif
}

TEST(FileUtilTest, TestReadWriteWholeFile) {
  const Path missing = TempPathname("hwy_io_file_util_missing.bin");
  AlignedPtr<uint8_t> aligned;
  HWY_ASSERT(!ReadFileToAligned(missing, aligned).ok());
  HWY_ASSERT(aligned == nullptr);
#if HWY_IO_TEST_OS_FILES
  const Path path = TempPathname("hwy_io_file_util.bin");
  // Empty file ReadFileToAligned succeeds with 0 bytes.
  HWY_ASSERT(WriteBytesToFile(path, Span<const uint8_t>()) == Status(0));
  HWY_ASSERT(ReadFileToAligned(path, aligned) == Status(0));
  HWY_ASSERT(aligned == nullptr);
  HWY_ASSERT_EQ(size_t{0}, aligned.size());
  // Writable mappings of an empty file are also empty (the probe finds no
  // data).
  {
    std::unique_ptr<File> empty = OpenFileOrAbort(path, FileMode::kWriteUpdate);
    MutableMapPtr empty_map;
    HWY_ASSERT(empty->MapWritable(empty_map) == Status(0));
    HWY_ASSERT(!empty_map);
  }

  // One-shot WriteBytesToFile writes, truncates on overwrite, and handles empty
  // spans.
  const uint8_t payload[6] = {10, 20, 30, 40, 50, 60};
  HWY_ASSERT(WriteBytesToFile(path, Span<const uint8_t>(payload, 6)) ==
             Status(6));
  HWY_ASSERT(ReadFileToAligned(path, aligned) == Status(6));
  HWY_ASSERT_ARRAY_EQ(payload, aligned.get(), 6);
  HWY_ASSERT(WriteBytesToFile(path, Span<const uint8_t>(payload, 2)) ==
             Status(2));
  HWY_ASSERT(ReadFileToAligned(path, aligned) == Status(2));
  HWY_ASSERT_ARRAY_EQ(payload, aligned.get(), 2);
  HWY_ASSERT(WriteBytesToFile(path, Span<const uint8_t>()) == Status(0));
  HWY_ASSERT(ReadFileToAligned(path, aligned) == Status(0));
  HWY_ASSERT(RemoveFile(path).ok());
#endif  // HWY_IO_TEST_OS_FILES
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
