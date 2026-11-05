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

#include "hwy/io/file_view.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>

#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/io_batch.h"
#include "hwy/io/io_test_util.h"
#include "hwy/io/memory_file.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

TEST(FileViewTest, TestFileView) {
  IOWorkerBuf io_buf;
  static_assert(std::is_constructible<FileView, const MemoryFile&, uint64_t,
                                      uint64_t>::value,
                "lvalue File must bind");
  static_assert(
      !std::is_constructible<FileView, MemoryFile&&, uint64_t, uint64_t>::value,
      "rvalue File must be rejected");
  static_assert(!std::is_constructible<FileView, const MemoryFile&&, uint64_t,
                                       uint64_t>::value,
                "const rvalue File must be rejected");

  std::vector<uint8_t> src(100);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
  MemoryFile parent(src);

  FileView view(parent, /*begin=*/20, /*size=*/30);
  HWY_ASSERT(view.Capabilities() ==
             (FileCap::kRandomAccess | FileCap::kAdvise |
              FileCap::kMapImmutable | FileCap::kMapShared));
  HWY_ASSERT(view.Size() == Status(30));
  const Span<const uint8_t> in_mem = view.MaybeSpan(0, 30);
  HWY_ASSERT_EQ(size_t{30}, in_mem.size());
  HWY_ASSERT_EQ(uint8_t{20}, in_mem[0]);
  HWY_ASSERT_EQ(uint8_t{49}, in_mem[29]);
  const Span<const uint8_t> sub_mem = view.MaybeSpan(5, 10);
  HWY_ASSERT_EQ(size_t{10}, sub_mem.size());
  HWY_ASSERT_EQ(uint8_t{25}, sub_mem[0]);
  HWY_ASSERT(view.MaybeSpan(25, 6).data() == nullptr);
  HWY_ASSERT(view.MaybeSpan(30, 0).data() != nullptr);
  HWY_ASSERT(view.MaybeSpan(31, 0).data() == nullptr);
  HWY_ASSERT(view.MaybeSpanForWrite(0, 0).data() == nullptr);
  HWY_ASSERT(view.MaybeSpanForWrite(0, 4).data() == nullptr);

  MapPtr mapped;
  HWY_ASSERT(view.MapImmutable(mapped) == Status(30));
  HWY_ASSERT_EQ(uint8_t{20}, mapped[0]);
  MapPtr shared;
  HWY_ASSERT(view.MapShared(shared) == Status(30));
  HWY_ASSERT(shared.get() == mapped.get());
  // Read-only object kind.
  MutableMapPtr writable;
  HWY_ASSERT(view.MapWritable(writable).Is(Status::kUnimplemented));

  FileView empty_view(parent, /*begin=*/20, /*size=*/0);
  MapPtr empty_view_map;
  HWY_ASSERT(empty_view.MapImmutable(empty_view_map) == Status(0));
  HWY_ASSERT(empty_view_map == nullptr);

  uint8_t buf[10] = {};
  HWY_ASSERT(view.Read(/*offset=*/5, WritableByteSpan(buf, 10)) == Status(10));
  for (size_t i = 0; i < 10; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(25 + i), buf[i]);
  }
  // Reading past the end of the view (even though parent has more bytes)
  // truncates at the view boundary.
  HWY_ASSERT(view.Read(/*offset=*/25, WritableByteSpan(buf, 10)) ==
             Status(Status::kTruncated, 5));
  HWY_ASSERT(view.Read(/*offset=*/30, Span<uint8_t>()) == Status(0));
  HWY_ASSERT(view.Read(/*offset=*/100, Span<uint8_t>()) == Status(0));
  HWY_ASSERT(view.Write(0, ByteSpan(buf, 1)).Is(IOCode::kWrongMode));
  HWY_ASSERT(view.WriteSpans(0, IOSpans()).Is(IOCode::kWrongMode));
  HWY_ASSERT(view.SetSize(10).Is(IOCode::kWrongMode));
  HWY_ASSERT(view.Preallocate(0, 10).Is(IOCode::kWrongMode));
  HWY_ASSERT(view.Sync().Is(IOCode::kWrongMode));
  HWY_ASSERT(view.Advise(0, 10, AccessHint::kSequential).ok());

  // Advising on an empty FileView or at offset == view.Size() must NOT forward
  // size == 0 ("until parent EOF") to the parent File.
  struct AdviseSpyFile : public MemoryFile {
    using MemoryFile::MemoryFile;
    Status Advise(uint64_t offset, uint64_t size,
                  AccessHint hint) const override {
      ++calls;
      last_offset = offset;
      last_size = size;
      return MemoryFile::Advise(offset, size, hint);
    }
    mutable size_t calls = 0;
    mutable uint64_t last_offset = 0;
    mutable uint64_t last_size = 0;
  };
  AdviseSpyFile spy_parent(src);
  FileView spy_empty(spy_parent, /*begin=*/20, /*size=*/0);
  HWY_ASSERT(spy_empty.Advise(0, 0, AccessHint::kDontNeed).ok());
  HWY_ASSERT_EQ(size_t{0}, spy_parent.calls);
  FileView spy_view(spy_parent, /*begin=*/20, /*size=*/30);
  HWY_ASSERT(spy_view.Advise(30, 0, AccessHint::kDontNeed).ok());
  HWY_ASSERT_EQ(size_t{0}, spy_parent.calls);
  HWY_ASSERT(spy_view.Advise(31, 0, AccessHint::kDontNeed).Is(Status::kRange));
  HWY_ASSERT(spy_view.Advise(10, 0, AccessHint::kWillNeed).ok());
  HWY_ASSERT_EQ(size_t{1}, spy_parent.calls);
  HWY_ASSERT_EQ(uint64_t{30}, spy_parent.last_offset);
  HWY_ASSERT_EQ(uint64_t{20}, spy_parent.last_size);

  // IOBatch over FileView (both within bounds and truncated at view end).
  // C arrays and std::vector convert unambiguously to Span<uint8_t> in Add.
  uint8_t part1[4] = {};
  std::vector<uint8_t> part2(6, 0);
  IOBatch batch(/*offset=*/2);
  HWY_ASSERT(batch.Add(part1));
  HWY_ASSERT(batch.Add(part2));
  HWY_ASSERT_EQ(size_t{2}, batch.Spans().size());
  HWY_ASSERT_EQ(uint64_t{10}, batch.Spans().TotalBytes());
  HWY_ASSERT(batch.Read(view, io_buf) == Status(10));
  for (size_t i = 0; i < 4; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(22 + i), part1[i]);
  }
  for (size_t i = 0; i < 6; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(26 + i), part2[i]);
  }
  batch.Reset(/*offset=*/24);
  std::fill(std::begin(part1), std::end(part1), uint8_t{0xAA});
  std::fill(part2.begin(), part2.end(), uint8_t{0xAA});
  HWY_ASSERT(batch.Add(part1));
  HWY_ASSERT(batch.Add(part2));
  HWY_ASSERT(batch.Read(view, io_buf) == Status(Status::kTruncated, 6));
  for (size_t i = 0; i < 4; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(44 + i), part1[i]);
  }
  HWY_ASSERT_EQ(uint8_t{48}, part2[0]);
  HWY_ASSERT_EQ(uint8_t{49}, part2[1]);
  for (size_t i = 2; i < 6; ++i) {
    HWY_ASSERT_EQ(uint8_t{0xAA}, part2[i]);
  }

  // Reading at or past view.Size() returns kTruncated(0) immediately without
  // allocating scratch or dereferencing span pointers.
  IOBatch at_end_batch(/*offset=*/30);
  HWY_ASSERT(at_end_batch.Add(
      Span<uint8_t>(static_cast<uint8_t*>(nullptr), File::kMaxBatchBytes)));
  HWY_ASSERT(at_end_batch.Read(view, io_buf) == Status(Status::kTruncated, 0));

  // Out-of-bounds and overflowing FileView bounds fail safely.
  FileView past_parent(parent, /*begin=*/80, /*size=*/30);
  HWY_ASSERT(past_parent.Size() == Status(Status::kRange, 20));
  HWY_ASSERT_EQ(size_t{20}, past_parent.MaybeSpan(0, 20).size());
  HWY_ASSERT(past_parent.MaybeSpan(0, 21).data() == nullptr);
  HWY_ASSERT(past_parent.MaybeSpan(0, 30).data() == nullptr);
  MapPtr bad_view_map;
  HWY_ASSERT(past_parent.MapImmutable(bad_view_map) ==
             Status(Status::kRange, 20));
  uint8_t tail[25] = {};
  HWY_ASSERT(past_parent.Read(0, WritableByteSpan(tail, 25)) ==
             Status(Status::kTruncated, 20));

  FileView overflow_view(parent, ~uint64_t{0} - 4, 8);
  HWY_ASSERT(!overflow_view.Capabilities().HasAny(FileCap::kMapImmutable |
                                                  FileCap::kMapShared));
  HWY_ASSERT(overflow_view.Size().Is(Status::kArgument));
  HWY_ASSERT(overflow_view.MaybeSpan(0, 0).data() == nullptr);
  HWY_ASSERT(overflow_view.MaybeSpan(0, 4).data() == nullptr);
  HWY_ASSERT(
      overflow_view.Read(0, WritableByteSpan(buf, 4)).Is(Status::kArgument));

  // Stat reports the view's size, with unknown identity.
  FileStat view_stat;
  HWY_ASSERT(view.Stat(view_stat) == Status(30));
  HWY_ASSERT(view_stat.kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{30}, view_stat.size);
  HWY_ASSERT(view_stat.device == 0 && view_stat.inode == 0);

  HWY_ASSERT(view.Close().ok());
  HWY_ASSERT(view.Size().Is(Status::kPrecondition));
  HWY_ASSERT(view.Stat(view_stat).Is(Status::kPrecondition));
  HWY_ASSERT_EQ(uint64_t{0}, view_stat.size);
  HWY_ASSERT(view.SetSize(0).Is(Status::kPrecondition));
  HWY_ASSERT(view.Preallocate(0, 1).Is(Status::kPrecondition));
  HWY_ASSERT(view.Sync().Is(Status::kPrecondition));
  HWY_ASSERT(view.Advise(0, 10, AccessHint::kNormal).Is(Status::kPrecondition));
  HWY_ASSERT(view.MapShared(shared).Is(Status::kPrecondition));
  HWY_ASSERT(view.MapWritable(writable).Is(Status::kPrecondition));
  HWY_ASSERT(view.MaybeSpan(0, 0).data() == nullptr);
  HWY_ASSERT(view.Read(0, WritableByteSpan(buf, 1)).Is(Status::kPrecondition));
  HWY_ASSERT(view.ReadSpans(0, IOSpans(), io_buf).Is(Status::kPrecondition));
  HWY_ASSERT(view.WriteSpans(0, IOSpans()).Is(Status::kPrecondition));
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
