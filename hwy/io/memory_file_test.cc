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

#include "hwy/io/memory_file.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/file_view.h"
#include "hwy/io/io_test_util.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

TEST(MemoryFileTest, TestMemoryFileReadWriteMaybeSpanMap) {
  // Empty MemoryFile: 0-byte read, write, and MaybeSpan succeed without
  // dereferencing null, and `MaybeSpan(0, 0).data() != nullptr` distinguishes
  // an empty in-memory file from unsupported/out-of-bounds spans.
  MemoryFile empty_file((Span<const uint8_t>()));
  HWY_ASSERT(empty_file.Read(0, Span<uint8_t>()) == Status(0));
  const Span<const uint8_t> empty_span = empty_file.MaybeSpan(0, 0);
  HWY_ASSERT(empty_span.data() != nullptr);
  HWY_ASSERT(empty_span.empty());
  HWY_ASSERT(empty_file.MaybeSpan(0, 1).data() == nullptr);
  HWY_ASSERT(empty_file.MaybeSpan(1, 0).data() == nullptr);
  HWY_ASSERT(empty_file.MaybeSpanForWrite(0, 0).data() == nullptr);
  // Reading past the end of an empty file is kTruncated with 0 bytes read.
  uint8_t one_byte = 0;
  HWY_ASSERT(empty_file.Read(0, WritableByteSpan(&one_byte, 1)) ==
             Status(Status::kTruncated, 0));
  MapPtr empty_map;
  HWY_ASSERT(empty_file.MapImmutable(empty_map) == Status(0));
  HWY_ASSERT(empty_map == nullptr);
  HWY_ASSERT_EQ(size_t{0}, empty_map.size());
  FileView empty_file_view(empty_file, 0, 0);
  HWY_ASSERT(empty_file_view.MaybeSpan(0, 0).data() != nullptr);
  HWY_ASSERT(empty_file_view.MapImmutable(empty_map) == Status(0));
  HWY_ASSERT(empty_map == nullptr);

  MemoryFile empty_rw(Span<uint8_t>(), FileMode::kWriteTruncate);
  HWY_ASSERT(empty_rw.MaybeSpan(0, 0).data() != nullptr);
  HWY_ASSERT(empty_rw.MaybeSpanForWrite(0, 0).data() != nullptr);
  HWY_ASSERT(empty_rw.MaybeSpanForWrite(0, 1).data() == nullptr);
  HWY_ASSERT(empty_rw.MaybeSpanForWrite(1, 0).data() == nullptr);
  HWY_ASSERT(empty_rw.Write(0, Span<const uint8_t>()) == Status(0));
  // Zero-length writes beyond the capacity fail, as does MaybeSpanForWrite.
  HWY_ASSERT(empty_rw.Write(1000, Span<const uint8_t>()).Is(Status::kResource));
  HWY_ASSERT(empty_rw.Size() == Status(0));
  HWY_ASSERT(empty_rw.Write(0, ByteSpan(&one_byte, 1)) ==
             Status(Status::kResource));

  std::vector<uint8_t> backing(128, 0);
  MemoryFile rw_file(backing, FileMode::kWriteTruncate);
  // Capabilities are independent of the mode.
  constexpr FileCaps kExpectedMem =
      FileCap::kRandomAccess | FileCap::kRandomWrite | FileCap::kExtend |
      FileCap::kPreallocate | FileCap::kSync | FileCap::kAdvise |
      FileCap::kMapImmutable | FileCap::kMapShared | FileCap::kMapWritable;
  HWY_ASSERT(rw_file.Capabilities() == kExpectedMem);
  HWY_ASSERT(!rw_file.IsTerminal());
  HWY_ASSERT(!rw_file.Capabilities().Has(FileCap::kNativeScatter));
  HWY_ASSERT(rw_file.Size() == Status(0));  // Initially empty.
  HWY_ASSERT_EQ(uint64_t{128}, rw_file.Capacity());
  HWY_ASSERT(rw_file.MapImmutable(empty_map) == Status(0));
  HWY_ASSERT(empty_map == nullptr);
  HWY_ASSERT(rw_file.Sync().ok());

  // Zero-length read past EOF succeeds with 0 bytes.
  HWY_ASSERT(rw_file.Read(/*offset=*/1000, Span<uint8_t>()) == Status(0));

  const char kHello[] = "Hello Highway IO!";
  const size_t kHelloLen = sizeof(kHello);
  char read_buf[sizeof(kHello)] = {};
  // Cannot read beyond what was written.
  HWY_ASSERT(
      rw_file.Read(/*offset=*/16, WritableByteSpan(read_buf, kHelloLen)) ==
      Status(Status::kTruncated, /*payload=*/0));

  Status status = rw_file.Write(/*offset=*/16, ByteSpan(kHello, kHelloLen));
  HWY_ASSERT(status == Status(kHelloLen));
  HWY_ASSERT(rw_file.Size() == Status(16 + kHelloLen));
  HWY_ASSERT_EQ(size_t{16 + kHelloLen},
                rw_file.MaybeSpan(0, 16 + kHelloLen).size());
  const Span<const uint8_t> hello_span = rw_file.MaybeSpan(16, kHelloLen);
  HWY_ASSERT_EQ(kHelloLen, hello_span.size());
  HWY_ASSERT(memcmp(kHello, hello_span.data(), kHelloLen) == 0);
  HWY_ASSERT(rw_file.MaybeSpan(0, 16 + kHelloLen + 1).data() == nullptr);
  // Also test rvalue Span<char> overload of WritableByteSpan.
  status = rw_file.Read(
      /*offset=*/16,
      WritableByteSpan(Span<char>(read_buf, kHelloLen).subspan(0, kHelloLen)));
  HWY_ASSERT(status == Status(kHelloLen));
  HWY_ASSERT(strcmp(kHello, read_buf) == 0);

  // Partial read across EOF copies the available prefix and returns its size.
  memset(read_buf, 0, sizeof(read_buf));
  HWY_ASSERT(
      rw_file.Read(/*offset=*/20, WritableByteSpan(read_buf, kHelloLen)) ==
      Status(Status::kTruncated, kHelloLen - 4));
  HWY_ASSERT(memcmp(kHello + 4, read_buf, kHelloLen - 4) == 0);

  // Writing before the end does not shrink the size.
  HWY_ASSERT(rw_file.Write(/*offset=*/0, ByteSpan(kHello, 4)).ok());
  HWY_ASSERT(rw_file.Size() == Status(16 + kHelloLen));

  // Out-of-bounds Read/Write/MaybeSpan/MaybeSpanForWrite must fail safely.
  HWY_ASSERT(
      !rw_file.Read(/*offset=*/120, WritableByteSpan(read_buf, 16)).ok());
  HWY_ASSERT(!rw_file.Read(/*offset=*/129, WritableByteSpan(read_buf, 1)).ok());
  HWY_ASSERT(rw_file.Write(/*offset=*/120, ByteSpan(kHello, 16)) ==
             Status(Status::kResource));
  HWY_ASSERT(rw_file.MaybeSpanForWrite(/*offset=*/120, /*size=*/16).data() ==
             nullptr);
  HWY_ASSERT(rw_file.MaybeSpanForWrite(/*offset=*/128, /*size=*/0).data() !=
             nullptr);
  HWY_ASSERT(rw_file.MaybeSpanForWrite(/*offset=*/129, /*size=*/0).data() ==
             nullptr);
  // Offsets/sizes exceeding int64_t (off_t) or uint64_t must not wrap around.
  const uint64_t kMaxOff = static_cast<uint64_t>(LimitsMax<int64_t>());
  const uint64_t kMax64 = LimitsMax<uint64_t>();
  const size_t kMaxSize = ~size_t{0};
  HWY_ASSERT(rw_file.MaybeSpanForWrite(kMaxOff - 1, 1).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpanForWrite(kMaxOff, 1).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpanForWrite(kMax64, 1).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpanForWrite(1, kMaxSize).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpanForWrite(kMax64, kMaxSize).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpan(kMaxOff, 1).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpan(kMax64, 1).data() == nullptr);
  HWY_ASSERT(rw_file.MaybeSpan(1, kMaxSize).data() == nullptr);
  HWY_ASSERT(rw_file.Read(kMaxOff - 1, WritableByteSpan(read_buf, 1)) ==
             Status(Status::kTruncated, 0));
  HWY_ASSERT(rw_file.Read(kMaxOff, WritableByteSpan(read_buf, 1))
                 .Is(Status::kArgument));
  HWY_ASSERT(rw_file.Read(kMax64, WritableByteSpan(read_buf, 1))
                 .Is(Status::kArgument));
  HWY_ASSERT(
      rw_file.Write(kMaxOff - 1, ByteSpan(kHello, 1)).Is(Status::kResource));
  HWY_ASSERT(rw_file.Write(kMaxOff, ByteSpan(kHello, 1)).Is(Status::kArgument));
  HWY_ASSERT(rw_file.Write(kMax64, ByteSpan(kHello, 1)).Is(Status::kArgument));
  HWY_ASSERT(
      rw_file.Write(kMaxOff + 1, Span<const uint8_t>()).Is(Status::kArgument));
  HWY_ASSERT(rw_file.Size() == Status(16 + kHelloLen));

  // MaybeSpanForWrite zero-copy write also extends the size when size != 0.
  const Span<uint8_t> direct =
      rw_file.MaybeSpanForWrite(/*offset=*/64, /*size=*/4);
  HWY_ASSERT(direct.data() != nullptr);
  HWY_ASSERT_EQ(size_t{4}, direct.size());
  HWY_ASSERT(rw_file.Size() == Status(68));
  direct[0] = 10;
  direct[1] = 20;
  direct[2] = 30;
  direct[3] = 40;
  HWY_ASSERT_EQ(uint8_t{10}, backing[64]);
  HWY_ASSERT_EQ(uint8_t{40}, backing[67]);

  // SetSize can extend (zero-filling) and shrink within capacity.
  backing[80] = 99;
  HWY_ASSERT(rw_file.SetSize(96) == Status(96));
  HWY_ASSERT(rw_file.Size() == Status(96));
  HWY_ASSERT_EQ(uint8_t{0}, backing[80]);
  // Preallocate checks the capacity without changing the size.
  HWY_ASSERT(rw_file.Preallocate(0, 128) == Status());
  HWY_ASSERT(rw_file.Preallocate(100, 28) == Status());
  HWY_ASSERT(rw_file.Preallocate(0, 129).Is(Status::kResource));
  HWY_ASSERT(
      rw_file.Preallocate(LimitsMax<uint64_t>(), 1).Is(Status::kArgument));
  HWY_ASSERT(rw_file.Size() == Status(96));
  HWY_ASSERT(rw_file.SetSize(112) == Status(112));
  HWY_ASSERT(rw_file.Size() == Status(112));
  // Shrinking does not modify the bytes beyond the new size (e.g. the unused
  // part of a bounded MaybeSpanForWrite reservation); growing zero-fills them.
  backing[60] = 77;
  HWY_ASSERT(rw_file.SetSize(50) == Status(50));
  HWY_ASSERT_EQ(uint8_t{77}, backing[60]);
  HWY_ASSERT(rw_file.SetSize(64) == Status(64));
  HWY_ASSERT_EQ(uint8_t{0}, backing[60]);
  HWY_ASSERT(rw_file.SetSize(50) == Status(50));
  HWY_ASSERT(rw_file.Size() == Status(50));
  const uint8_t tail_vals[4] = {10, 20, 30, 40};
  HWY_ASSERT(rw_file.Write(/*offset=*/64, ByteSpan(tail_vals, 4)) == Status(4));
  HWY_ASSERT(rw_file.Size() == Status(68));
  for (size_t i = 50; i < 64; ++i) {
    HWY_ASSERT_EQ(uint8_t{0}, backing[i]);
  }
  HWY_ASSERT(rw_file.SetSize(129).Is(Status::kResource));

  // Non-owning MapPtr covers Size() and does not free backing memory.
  {
    MapPtr mapped;
    HWY_ASSERT(rw_file.MapImmutable(mapped) == Status(68));
    HWY_ASSERT(mapped != nullptr);
    HWY_ASSERT_EQ(size_t{68}, mapped.size());
    HWY_ASSERT_EQ(uint8_t{10}, mapped[64]);
    HWY_ASSERT_EQ(uint8_t{40}, mapped[67]);
    // All mappings view the backing store, hence are coherent with `Write`.
    MapPtr shared;
    HWY_ASSERT(rw_file.MapShared(shared) == Status(68));
    HWY_ASSERT(shared.get() == mapped.get());
    MutableMapPtr writable;
    HWY_ASSERT(rw_file.MapWritable(writable) == Status(68));
    HWY_ASSERT(writable.get() == backing.data());
    writable[67] = 41;
    HWY_ASSERT_EQ(uint8_t{41}, shared[67]);
    const uint8_t kForty = 40;
    HWY_ASSERT(rw_file.Write(67, ByteSpan(&kForty, 1)) == Status(1));
    HWY_ASSERT_EQ(uint8_t{40}, writable[67]);
  }
  HWY_ASSERT_EQ(uint8_t{10}, backing[64]);

  // Read-only MemoryFile has the full size and rejects Write,
  // MaybeSpanForWrite, SetSize, Preallocate, Sync, but reports the same caps.
  MemoryFile ro_file(backing);
  HWY_ASSERT(ro_file.Capabilities() == kExpectedMem);
  {
    MutableMapPtr writable;
    HWY_ASSERT(ro_file.MapWritable(writable).Is(IOCode::kWrongMode));
    HWY_ASSERT(!writable);
  }
  HWY_ASSERT(ro_file.Size() == Status(128));
  HWY_ASSERT_EQ(size_t{128}, ro_file.MaybeSpan(0, 128).size());
  HWY_ASSERT(ro_file.MaybeSpan(128, 1).data() == nullptr);
  HWY_ASSERT(ro_file.Write(/*offset=*/0, ByteSpan(kHello, kHelloLen))
                 .Is(IOCode::kWrongMode));
  HWY_ASSERT(ro_file.SetSize(64).Is(IOCode::kWrongMode));
  HWY_ASSERT(ro_file.Preallocate(0, 64).Is(IOCode::kWrongMode));
  HWY_ASSERT(ro_file.Sync().Is(IOCode::kWrongMode));
  HWY_ASSERT(ro_file.MaybeSpanForWrite(/*offset=*/0, /*size=*/4).data() ==
             nullptr);
  HWY_ASSERT(ro_file.MaybeSpanForWrite(/*offset=*/0, /*size=*/0).data() ==
             nullptr);
  memset(read_buf, 0, sizeof(read_buf));
  HWY_ASSERT(
      ro_file.Read(/*offset=*/16, WritableByteSpan(read_buf, kHelloLen)).ok());
  HWY_ASSERT(strcmp(kHello, read_buf) == 0);

  // kWriteNew also starts with size 0 on MemoryFile.
  MemoryFile new_file(backing, FileMode::kWriteNew);
  HWY_ASSERT(new_file.Size() == Status(0));
  HWY_ASSERT(new_file.Advise(0, 64, AccessHint::kRandom).ok());

  // kWriteUpdate starts with the full buffer size and allows in-place writes.
  MemoryFile upd_file(backing, FileMode::kWriteUpdate);
  HWY_ASSERT(upd_file.Size() == Status(128));
  HWY_ASSERT(upd_file.Write(/*offset=*/127, ByteSpan("Z", 1)) == Status(1));
  HWY_ASSERT_EQ(uint8_t{'Z'}, backing[127]);

  // Subsequent operations after Close() fail with kPrecondition.
  HWY_ASSERT(upd_file.Close().ok());
  HWY_ASSERT(upd_file.Size().Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.SetSize(0).Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.Preallocate(0, 1).Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.Sync().Is(Status::kPrecondition));
  HWY_ASSERT(
      upd_file.Advise(0, 0, AccessHint::kDontNeed).Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.MaybeSpan(0, 0).data() == nullptr);
  HWY_ASSERT(upd_file.MaybeSpan(0, 1).data() == nullptr);
  MapPtr closed_map;
  HWY_ASSERT(upd_file.MapImmutable(closed_map).Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.MapShared(closed_map).Is(Status::kPrecondition));
  HWY_ASSERT(closed_map == nullptr);
  MutableMapPtr closed_writable;
  HWY_ASSERT(upd_file.MapWritable(closed_writable).Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.MaybeSpanForWrite(0, 0).data() == nullptr);
  HWY_ASSERT(upd_file.MaybeSpanForWrite(0, 1).data() == nullptr);
  HWY_ASSERT(upd_file.Read(0, WritableByteSpan(read_buf, 1))
                 .Is(Status::kPrecondition));
  HWY_ASSERT(upd_file.Write(0, ByteSpan("Z", 1)).Is(Status::kPrecondition));
}

// Concurrent Write/MaybeSpanForWrite to disjoint ranges of a MemoryFile.
TEST(MemoryFileTest, TestParallelMemoryFile) {
  if (!HaveThreadingSupport()) return;
  constexpr size_t kThreads = 8;
  constexpr size_t kBytesPerThread = 4096;
  std::vector<uint8_t> backing(kThreads * kBytesPerThread, 0);
  MemoryFile file(backing, FileMode::kWriteTruncate);
  std::vector<std::thread> threads;
  for (size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      const size_t begin = t * kBytesPerThread;
      // Alternate between Write and MaybeSpanForWrite, in descending order so
      // that the atomic max is exercised.
      for (size_t i = kBytesPerThread; i != 0; i -= 64) {
        const size_t offset = begin + i - 64;
        uint8_t buf[64];
        for (size_t j = 0; j < 64; ++j) {
          buf[j] = static_cast<uint8_t>(offset + j);
        }
        if ((i / 64) & 1) {
          HWY_ASSERT(file.Write(offset, ByteSpan(buf, 64)).ok());
        } else {
          const Span<uint8_t> direct = file.MaybeSpanForWrite(offset, 64);
          HWY_ASSERT_EQ(size_t{64}, direct.size());
          CopyBytes(buf, direct.data(), 64);
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  HWY_ASSERT(file.Size() == Status(kThreads * kBytesPerThread));
  for (size_t i = 0; i < backing.size(); ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(i), backing[i]);
  }

  // Concurrent extending Write + concurrent Size/Read/MaybeSpan must publish
  // copied bytes before updating Size() (no TSan data race or stale reads).
  std::fill(backing.begin(), backing.end(), uint8_t{0});
  MemoryFile seq_file(backing, FileMode::kWriteTruncate);
  std::atomic<bool> writer_done{false};
  std::thread reader([&]() {
    std::vector<uint8_t> read_buf(backing.size());
    while (!writer_done.load(std::memory_order_acquire)) {
      const uint64_t snap_size = seq_file.Size().size();
      const Span<const uint8_t> mem =
          seq_file.MaybeSpan(0, static_cast<size_t>(snap_size));
      HWY_ASSERT(mem.data() != nullptr);
      for (size_t i = 0; i < mem.size(); ++i) {
        HWY_ASSERT_EQ(static_cast<uint8_t>((i * 17 + 3) & 0xFF), mem[i]);
      }
      const uint64_t cur_size = seq_file.Size().size();
      if (cur_size != 0) {
        HWY_ASSERT(
            seq_file.Read(0, Span<uint8_t>(read_buf.data(),
                                           static_cast<size_t>(cur_size))) ==
            Status(cur_size));
        for (size_t i = 0; i < cur_size; ++i) {
          HWY_ASSERT_EQ(static_cast<uint8_t>((i * 17 + 3) & 0xFF), read_buf[i]);
        }
      }
    }
  });
  for (size_t off = 0; off < backing.size(); off += 64) {
    uint8_t chunk[64];
    for (size_t j = 0; j < 64; ++j) {
      chunk[j] = static_cast<uint8_t>(((off + j) * 17 + 3) & 0xFF);
    }
    HWY_ASSERT(seq_file.Write(off, ByteSpan(chunk, 64)) == Status(64));
  }
  writer_done.store(true, std::memory_order_release);
  reader.join();
}

// `FreePtr` that counts calls via `opaque`.
void CountingFree(void* opaque, void* /*ptr*/) { ++*static_cast<int*>(opaque); }

TEST(MemoryFileTest, TestStat) {
  const uint8_t bytes[3] = {1, 2, 3};
  MemoryFile file(Span<const uint8_t>(bytes, 3));
  FileStat stat;
  HWY_ASSERT(file.Stat(stat) == Status(3));
  HWY_ASSERT(stat.kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{3}, stat.size);
  // Unknown identities never compare equal, even to themselves.
  HWY_ASSERT(stat.device == 0 && stat.inode == 0);
  HWY_ASSERT(!FileStat::SameFile(stat, stat));
  HWY_ASSERT(file.Close().ok());
  HWY_ASSERT(file.Stat(stat).Is(Status::kPrecondition));
}

TEST(MemoryFileTest, TestOwnership) {
  // Writable: owns an `AllocateAligned` buffer.
  {
    MutableMapPtr buffer = AllocateAligned<uint8_t>(64);
    HWY_ASSERT(buffer);
    uint8_t* raw = buffer.get();
    std::unique_ptr<File> file = std::make_unique<MemoryFile>(
        std::move(buffer), FileMode::kWriteTruncate);
    HWY_ASSERT(!buffer);  // NOLINT(bugprone-use-after-move)
    HWY_ASSERT(file->Size() == Status(0));
    const uint8_t kBytes[3] = {7, 8, 9};
    HWY_ASSERT(file->Write(0, ByteSpan(kBytes, 3)) == Status(3));
    HWY_ASSERT_EQ(uint8_t{8}, raw[1]);
    HWY_ASSERT(file->MaybeSpanForWrite(60, 4).data() == raw + 60);
    HWY_ASSERT(file->MaybeSpanForWrite(61, 4).data() == nullptr);
    HWY_ASSERT(file->Size() == Status(64));
  }  // Frees the buffer (ASan reports leaks or double frees).

  // Freed on destruction, not by `Close`.
  uint8_t storage[16] = {1, 2, 3};
  int frees = 0;
  {
    MemoryFile file(MutableMapPtr(storage, 16, &CountingFree, &frees),
                    FileMode::kWriteUpdate);
    HWY_ASSERT(file.Size() == Status(16));
    MutableMapPtr view;
    HWY_ASSERT(file.MapWritable(view) == Status(16));
    HWY_ASSERT(view.get() == storage);
    view.reset();  // Non-owning view: does not free.
    HWY_ASSERT_EQ(0, frees);
    HWY_ASSERT(file.Close().ok());
    HWY_ASSERT_EQ(0, frees);
  }
  HWY_ASSERT_EQ(1, frees);

  // Read-only: owns a `MapPtr`.
  frees = 0;
  {
    MemoryFile file(MapPtr(storage, 16, &CountingFree, &frees));
    HWY_ASSERT(file.Size() == Status(16));
    HWY_ASSERT(file.MaybeSpan(0, 3).data() == storage);
    HWY_ASSERT(file.Write(0, ByteSpan(storage, 1)).Is(IOCode::kWrongMode));
    HWY_ASSERT(file.MaybeSpanForWrite(0, 1).data() == nullptr);
  }
  HWY_ASSERT_EQ(1, frees);

  // A writable buffer can also be owned read-only.
  frees = 0;
  {
    MemoryFile file(MutableMapPtr(storage, 16, &CountingFree, &frees));
    HWY_ASSERT(file.Write(0, ByteSpan(storage, 1)).Is(IOCode::kWrongMode));
  }
  HWY_ASSERT_EQ(1, frees);
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
