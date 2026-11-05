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

#include "hwy/io/file.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/io/file_system.h"
#include "hwy/io/file_util.h"
#include "hwy/io/file_view.h"
#include "hwy/io/io_batch.h"
#include "hwy/io/io_span.h"
#include "hwy/io/io_test_util.h"
#include "hwy/io/memory_file.h"
#include "hwy/io/path.h"
#include "hwy/profiler.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

// Application-defined unscoped and scoped enums for testing Status interop.
enum AppStatus : Status::Code {
  kNeedInput = Status::FirstInfo(Status::kFirstAppCode),
  kNeedOutput,

  kAppWarn = Status::FirstWarning(Status::kFirstAppCode),

  kNotMagic = Status::FirstError(Status::kFirstAppCode),
  kCorrupt
};

enum class ScopedAppStatus : Status::Code {
  kCustomError = Status::FirstError(Status::kFirstAppCode + 100)
};

TEST(FileTest, TestStatus) {
  constexpr Status ok;
  static_assert(ok.ok() && !ok.IsInfo() && !ok.IsWarning() && !ok.IsError(),
                "default");
  static_assert(ok.severity() == Status::Severity::kOk, "severity");
  static_assert(ok.size() == 0 && ok.payload() == 0, "default payload");
  static_assert(sizeof(Status) == 8, "size");

  constexpr Status size(12345);
  static_assert(size.ok() && size.size() == 12345 && size.payload() == 12345,
                "size");
  static_assert(size.code() == Status::kOk && size.id() == 0, "code");
  static_assert(size == Status::Ok(12345), "Ok");
  // Payloads of errors are clamped.
  HWY_ASSERT(Status(Status::kDataLoss, Status::kMaxPayload + 1).payload() ==
             Status::kMaxPayload);

  constexpr Status err = Status::kNotFound;
  static_assert(!err.ok() && err.IsError() && !err.IsInfo() && !err.IsWarning(),
                "error");
  static_assert(err.code() == Status::kNotFound, "code");
  static_assert(err.Is(Status::kNotFound) && !err.Is(Status::kDataLoss), "Is");
  HWY_ASSERT(err == Status(Status::kNotFound));
  HWY_ASSERT(err != Status(Status::kDataLoss));
  HWY_ASSERT(strcmp(err.Name(), "kNotFound") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(Status::kCancelled), "kCancelled") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(Status::kArgument), "kArgument") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(Status::kTimeout), "kTimeout") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(Status::kResource), "kResource") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(Status::kTruncated), "kTruncated") == 0);

  constexpr Status err_payload(Status::kDataLoss, 22);
  static_assert(err_payload.IsError() && err_payload.payload() == 22, "errno");
  HWY_ASSERT(err_payload != Status(Status::kDataLoss));  // payload differs
  HWY_ASSERT(err_payload.Is(Status::kDataLoss));         // but Is() ignores it

  // Application enums (unscoped and scoped) implicitly construct Status with
  // the encoded severity, and auto-increment preserves severity.
  constexpr Status info1 = kNeedInput;
  constexpr Status info2 = kNeedOutput;
  static_assert(!info1.ok() && info1.IsInfo() && !info1.IsError(), "info1");
  static_assert(!info2.ok() && info2.IsInfo() && info2.Is(kNeedOutput),
                "info2");
  static_assert(info2.id() == Status::kFirstAppCode + 1, "info2 id");
  HWY_ASSERT(strcmp(info1.Name(), "(module/app code)") == 0);

  constexpr Status warn = Status(kAppWarn, 99);
  static_assert(!warn.ok() && warn.IsWarning() && warn.payload() == 99, "warn");
  static_assert(warn.Is(kAppWarn), "warn code");
#if !HWY_IS_DEBUG_BUILD
  // Release builds mask `size()` so non-ok statuses never leak code/severity
  // bits into size calculations.
  static_assert(warn.size() == 99 && err_payload.size() == 22, "masked size");
#endif

  // Downgrading a GeneralCode to Warning preserves its human-readable name.
  constexpr Status gen_warn = Status::Warning(Status::kDataLoss, 7);
  static_assert(!gen_warn.ok() && gen_warn.IsWarning(), "gen_warn");
  static_assert(!gen_warn.Is(Status::kDataLoss), "severity differs");
  static_assert(gen_warn.id() == Status(Status::kDataLoss).id(), "same id");
  HWY_ASSERT(strcmp(gen_warn.Name(), "kDataLoss") == 0);

  constexpr Status app_err = kCorrupt;
  static_assert(app_err.IsError() && app_err.Is(kCorrupt), "app_err");
  static_assert(app_err.code() != info2.code(), "distinct severity codes");

  constexpr Status scoped_err = ScopedAppStatus::kCustomError;
  static_assert(scoped_err.IsError() &&
                    scoped_err == Status(ScopedAppStatus::kCustomError) &&
                    scoped_err.Is(ScopedAppStatus::kCustomError),
                "scoped");
  constexpr Status scoped_warn =
      Status::Warning(ScopedAppStatus::kCustomError, 5);
  static_assert(scoped_warn.IsWarning() && scoped_warn.payload() == 5,
                "scoped warn");
  HWY_ASSERT(strcmp(Status::CodeName(ScopedAppStatus::kCustomError),
                    "(module/app code)") == 0);

  // Module codes: scoped, with the registered module's range and name lookup.
  constexpr Status io_err = IOCode::kWrongMode;
  static_assert(io_err.IsError() && io_err.Is(IOCode::kWrongMode), "io");
  static_assert(io_err.id() == Status::ModuleBegin(Status::Module::kIO),
                "io id");
  const uint8_t byte = 0;
  MemoryFile read_only(Span<const uint8_t>(&byte, 1));
  const Status write_status = read_only.Write(0, ByteSpan(&byte, 1));
  HWY_ASSERT(write_status == io_err);
  // The names are registered on first use of a `FileSystem` function, hence
  // binaries using only freestanding `File`s need not link file_system.cc.
  FileStat stat;
  (void)StatFile(Path("hwy_io_test_register_names"), stat);
  HWY_ASSERT(strcmp(write_status.Name(), "kWrongMode") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(IOCode::kWrongMode), "kWrongMode") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(IOCode::kExists), "kExists") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(IOCode::kNoProgress), "kNoProgress") == 0);
  // General code: named even without registering `IOCode` names.
  HWY_ASSERT(strcmp(Status::CodeName(Status::kOSError), "kOSError") == 0);
  HWY_ASSERT(strcmp(Status::CodeName(Status::kPermission), "kPermission") == 0);
  static_assert(Status(Status::kOSError).IsError(), "OS errors are errors");
  static_assert(
      Status::ModuleBegin(Status::Module::kIO) == Status::kFirstModuleCode,
      "first module");
  static_assert(Status::kFirstModuleCode +
                        Status::kNumModules * Status::kCodesPerModule ==
                    Status::kFirstAppCode,
                "module ranges fill the space");

  // Highest 14-bit code ID is representable.
  constexpr Status max = Status::Error(Status::kMaxCode, Status::kMaxPayload);
  static_assert(max.id() == Status::kMaxCode, "max id");
  static_assert(max.payload() == Status::kMaxPayload, "max payload");
  static_assert(max.IsError() && !max.ok(), "max is error");

  // Implicit conversion from kOk is plain success.
  constexpr Status from_ok = Status::kOk;
  static_assert(from_ok == Status(), "kOk");
}

TEST(FileTest, TestFileCaps) {
  constexpr FileCaps kEmptyCaps;
  static_assert(kEmptyCaps.Empty(), "default empty");
  static_assert(kEmptyCaps.Bits() == 0, "zero bits");
  static_assert(!kEmptyCaps.Has(FileCap::kNone), "kNone has no bit");
  static_assert(!kEmptyCaps.Has(FileCap::kRandomAccess), "not set");
  constexpr FileCaps kRW = FileCap::kRandomAccess | FileCap::kRandomWrite;
  static_assert(!kRW.Empty(), "non-empty");
  static_assert(kRW.Has(FileCap::kRandomAccess), "has read");
  static_assert(kRW.Has(FileCap::kRandomWrite), "has write");
  static_assert(!kRW.Has(FileCap::kMapImmutable), "no map");
  static_assert(kRW.HasAll(FileCap::kRandomAccess | FileCap::kRandomWrite),
                "has all");
  static_assert(!kRW.HasAll(kRW | FileCap::kMapImmutable), "missing map");
  static_assert(kRW.HasAny(FileCap::kMapImmutable | FileCap::kRandomWrite),
                "has any");
  static_assert(!kRW.HasAny(FileCap::kMapImmutable | FileCap::kExtend),
                "has none");
  static_assert((kRW & ~FileCaps(FileCap::kRandomWrite)) ==
                    FileCaps(FileCap::kRandomAccess),
                "mask out");
  static_assert(FileCaps::FromBits(kRW.Bits()) == kRW, "round-trip bits");
}

TEST(FileTest, TestFileReadWriteMapAndBatch) {
#if HWY_IO_TEST_OS_FILES
  IOWorkerBuf io_buf;
  const Path path = TempPathname("hwy_io_test_file.bin");
  std::vector<uint8_t> expected(4096);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<uint8_t>((i * 73 + 19) & 0xFF);
  }

  FileCaps out_caps;
  {
    std::unique_ptr<File> out = OpenFileOrAbort(path, FileMode::kWriteTruncate);
    out_caps = out->Capabilities();
    // Also available before opening, and independent of the file.
    HWY_ASSERT(PathCapabilities(path) == out_caps);
    HWY_ASSERT(PathCapabilities(TempPathname("hwy_io_missing")) == out_caps);
    HWY_ASSERT(out_caps.HasAll(FileCap::kRandomAccess | FileCap::kRandomWrite |
                               FileCap::kExtend | FileCap::kSync |
                               FileCap::kMapImmutable | FileCap::kMapShared |
                               FileCap::kMapWritable));
#if HWY_OS_LINUX && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 24)
    HWY_ASSERT(out_caps.HasAll(FileCap::kPreallocate | FileCap::kAdvise |
                               FileCap::kNativeScatter));
#elif HWY_OS_WIN
    HWY_ASSERT(!out_caps.HasAny(FileCap::kPreallocate | FileCap::kAdvise |
                                FileCap::kNativeScatter));
#endif
    HWY_ASSERT(out->Size() == Status(0));
    HWY_ASSERT(out->MaybeSpan(0, 0).data() == nullptr);
    HWY_ASSERT(out->MaybeSpanForWrite(0, 0).data() == nullptr);
    HWY_ASSERT(out->MaybeSpanForWrite(0, 16).data() == nullptr);
    // Zero-size files map to an empty MapPtr with Status(0).
    MapPtr empty_map;
    HWY_ASSERT(out->MapImmutable(empty_map) == Status(0));
    HWY_ASSERT(empty_map == nullptr);
    HWY_ASSERT_EQ(size_t{0}, empty_map.size());
    // Preallocate does not change the size. It may be unsupported by the
    // filesystem even with kPreallocate, but never fails with kResource here.
    const Status prealloc = out->Preallocate(0, 12288);
    HWY_ASSERT(prealloc.ok() || prealloc.Is(Status::kUnimplemented));
    if (!out_caps.Has(FileCap::kPreallocate)) {
      HWY_ASSERT(prealloc.Is(Status::kUnimplemented));
    }
    HWY_ASSERT(out->Size() == Status(0));
    if (out_caps.Has(FileCap::kPreallocate)) {
      HWY_ASSERT(out->Preallocate(100, 0) == Status());
    }
    HWY_ASSERT(
        out->Preallocate(LimitsMax<uint64_t>(), 1).Is(Status::kArgument));
    // Pre-extend via SetSize, write in two non-sequential chunks, then shrink.
    HWY_ASSERT(out->SetSize(8192) == Status(8192));
    HWY_ASSERT(out->Size() == Status(8192));
    HWY_ASSERT(out->SetSize(12288) == Status(12288));
    HWY_ASSERT(out->Size() == Status(12288));
    HWY_ASSERT(out->Write(1024, ByteSpan(expected.data() + 1024, 3072)) ==
               Status(3072));
    HWY_ASSERT(out->Write(0, ByteSpan(expected.data(), 1024)).ok());
    HWY_ASSERT(out->SetSize(4096) == Status(4096));
    HWY_ASSERT(out->Size() == Status(4096));
    HWY_ASSERT(out->Sync().ok());
    // Offsets/sizes exceeding off_t (int64_t) fail with kArgument.
    const uint64_t kMaxOff = static_cast<uint64_t>(LimitsMax<int64_t>());
    HWY_ASSERT(out->Write(kMaxOff, ByteSpan(expected.data(), 1))
                   .Is(Status::kArgument));
    // Close reports errors and is idempotent; the destructor is then a no-op.
    HWY_ASSERT(out->Close().ok());
    HWY_ASSERT(out->Close().ok());
    HWY_ASSERT(out->Size().Is(Status::kPrecondition));
    HWY_ASSERT(out->SetSize(0).Is(Status::kPrecondition));
    HWY_ASSERT(out->Preallocate(0, 1).Is(Status::kPrecondition));
    HWY_ASSERT(out->Sync().Is(Status::kPrecondition));
    HWY_ASSERT(out->MapImmutable(empty_map).Is(Status::kPrecondition));
    HWY_ASSERT(
        out->Read(0, WritableByteSpan(expected)).Is(Status::kPrecondition));
    HWY_ASSERT(out->Write(0, ByteSpan(expected)).Is(Status::kPrecondition));
  }

  HWY_ASSERT(IsFile(path));
  FileStat os_stat;
  HWY_ASSERT(StatFile(path, os_stat) == Status(4096));
  HWY_ASSERT(os_stat.kind == EntryKind::kFile);
  HWY_ASSERT_EQ(uint64_t{4096}, os_stat.size);

  // MapPtr outliving File object, and FileView over an OS file.
  MapPtr mapped;
  {
    std::unique_ptr<File> in = OpenFileOrAbort(path, FileMode::kRead);
    // Capabilities are independent of the mode.
    HWY_ASSERT(in->Capabilities() == out_caps);
    HWY_ASSERT(in->Size() == Status(4096));
    HWY_ASSERT(in->MaybeSpan(0, 0).data() == nullptr);
    HWY_ASSERT(in->MaybeSpan(0, 4096).data() == nullptr);
    // Sizes whose `Status` payload would overflow are rejected before access.
    if (sizeof(size_t) == 8) {
      uint8_t dummy;
      const size_t huge = static_cast<size_t>(Status::kMaxPayload) + 1;
      HWY_ASSERT(
          in->Read(0, WritableByteSpan(&dummy, huge)).Is(Status::kArgument));
    }
    // Reading past the end fails with kTruncated and the bytes read so far.
    uint8_t past_end[8];
    HWY_ASSERT(in->Read(/*offset=*/4092, WritableByteSpan(past_end, 8)) ==
               Status(Status::kTruncated, 4));
    // Writing, resizing, preallocating, or syncing a read-only file fails with
    // kWrongMode.
    HWY_ASSERT(in->Write(0, ByteSpan(past_end, 8)).Is(IOCode::kWrongMode));
    HWY_ASSERT(in->SetSize(0).Is(IOCode::kWrongMode));
    HWY_ASSERT(in->Preallocate(0, 8).Is(IOCode::kWrongMode));
    HWY_ASSERT(in->Sync().Is(IOCode::kWrongMode));

    FileView os_view(*in, /*begin=*/512, /*size=*/128);
    HWY_ASSERT(os_view.Capabilities().Has(FileCap::kRandomAccess));
    HWY_ASSERT(!os_view.Capabilities().HasAny(
        FileCap::kRandomWrite | FileCap::kExtend | FileCap::kPreallocate |
        FileCap::kSync | FileCap::kMapImmutable | FileCap::kMapShared |
        FileCap::kMapWritable));
    HWY_ASSERT_EQ(in->Capabilities().Has(FileCap::kNativeScatter),
                  os_view.Capabilities().Has(FileCap::kNativeScatter));
    HWY_ASSERT(os_view.Size() == Status(128));
    HWY_ASSERT(os_view.MaybeSpan(0, 0).data() == nullptr);
    HWY_ASSERT(os_view.MaybeSpan(0, 128).data() == nullptr);
    MapPtr view_map;
    HWY_ASSERT(os_view.MapImmutable(view_map).Is(Status::kUnimplemented));
    HWY_ASSERT(os_view.MapShared(view_map).Is(Status::kUnimplemented));
    MutableMapPtr view_writable;
    HWY_ASSERT(os_view.MapWritable(view_writable).Is(Status::kUnimplemented));
    FileView empty_os_view(*in, /*begin=*/512, /*size=*/0);
    HWY_ASSERT(empty_os_view.MapImmutable(view_map).Is(Status::kUnimplemented));
    uint8_t view_buf[32] = {};
    HWY_ASSERT(os_view.Read(/*offset=*/16, WritableByteSpan(view_buf, 32)) ==
               Status(32));
    HWY_ASSERT_ARRAY_EQ(expected.data() + 512 + 16, view_buf, 32);
    uint8_t view_part1[8] = {};
    uint8_t view_part2[8] = {};
    IOBatch view_batch(/*offset=*/32);
    HWY_ASSERT(view_batch.Add(Span<uint8_t>(view_part1, 8)));
    HWY_ASSERT(view_batch.Add(Span<uint8_t>(view_part2, 8)));
    HWY_ASSERT(view_batch.Read(os_view, io_buf) == Status(16));
    HWY_ASSERT_ARRAY_EQ(expected.data() + 512 + 32, view_part1, 8);
    HWY_ASSERT_ARRAY_EQ(expected.data() + 512 + 40, view_part2, 8);

    // Partial scatter read straddling the end of an OS FileView forwards only
    // the available prefix to preadv without touching trailing span bytes.
    memset(view_part1, 0xCC, sizeof(view_part1));
    memset(view_part2, 0xCC, sizeof(view_part2));
    view_batch.Reset(/*offset=*/116);
    HWY_ASSERT(view_batch.Add(Span<uint8_t>(view_part1, 8)));
    HWY_ASSERT(view_batch.Add(Span<uint8_t>(view_part2, 8)));
    HWY_ASSERT(view_batch.Read(os_view, io_buf) ==
               Status(Status::kTruncated, 12));
    HWY_ASSERT_ARRAY_EQ(expected.data() + 512 + 116, view_part1, 8);
    HWY_ASSERT_ARRAY_EQ(expected.data() + 512 + 124, view_part2, 4);
    for (size_t i = 4; i < 8; ++i) {
      HWY_ASSERT_EQ(uint8_t{0xCC}, view_part2[i]);
    }
    view_batch.Reset(/*offset=*/128);
    HWY_ASSERT(view_batch.Add(Span<uint8_t>(view_part1, 8)));
    HWY_ASSERT(view_batch.Read(os_view, io_buf) ==
               Status(Status::kTruncated, 0));

    HWY_ASSERT(in->MapImmutable(mapped) == Status(4096));
  }
  HWY_ASSERT(mapped != nullptr);
  HWY_ASSERT_EQ(size_t{4096}, mapped.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    HWY_ASSERT_EQ(expected[i], mapped[i]);
  }
  mapped.reset();

  // ReadFileToAligned.
  AlignedPtr<uint8_t> aligned;
  HWY_ASSERT(ReadFileToAligned(path, aligned) == Status(4096));
  HWY_ASSERT(aligned != nullptr);
  HWY_ASSERT_EQ(size_t{4096}, aligned.size());
  HWY_ASSERT_ARRAY_EQ(expected.data(), aligned.get(), expected.size());

  // IOBatch scatter-gather via OS file handle (preadv on POSIX).
  {
    std::unique_ptr<File> in = OpenFileOrAbort(path, FileMode::kRead);
    std::vector<uint8_t> parts(500 + 64 + 700, 0);
    IOBatch batch(/*offset=*/100, /*user_tag=*/1);
    HWY_ASSERT(batch.Add(Span<uint8_t>(parts.data(), 500)));
    HWY_ASSERT(batch.Add(Span<uint8_t>(parts.data() + 500 + 64, 700)));
    IOWorkerBuf os_worker;
    HWY_ASSERT(batch.Read(*in, os_worker) == Status(1200));
#if HWY_OS_LINUX && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 24)
    // Native preadv requires no fallback scratch allocation.
    HWY_ASSERT_EQ(size_t{0}, os_worker.Capacity());
#endif
    HWY_ASSERT_ARRAY_EQ(expected.data() + 100, parts.data(), 500);
    HWY_ASSERT_ARRAY_EQ(expected.data() + 600, parts.data() + 500 + 64, 700);

    // Reading past the end fails rather than returning a partial count.
    std::vector<uint8_t> tail(200);
    IOBatch past_batch(/*offset=*/4000, /*user_tag=*/2);
    HWY_ASSERT(past_batch.Add(Span<uint8_t>(tail.data(), 100)));
    HWY_ASSERT(past_batch.Add(Span<uint8_t>(tail.data() + 100, 100)));
    HWY_ASSERT(past_batch.Read(*in, io_buf) == Status(Status::kTruncated, 96));

    // > IOV_MAX discontiguous spans are chunked across multiple preadv calls.
#ifdef IOV_MAX
    const size_t kManySpans = IOV_MAX + 5;
#else
    const size_t kManySpans = 1029;
#endif
    std::vector<uint8_t> discontiguous(kManySpans * 2, 0);
    IOBatch many_batch(/*offset=*/16);
    for (size_t i = 0; i < kManySpans; ++i) {
      HWY_ASSERT(many_batch.Add(Span<uint8_t>(&discontiguous[i * 2], 1)));
    }
    HWY_ASSERT(many_batch.Read(*in, io_buf) == Status(kManySpans));
    for (size_t i = 0; i < kManySpans; ++i) {
      HWY_ASSERT_EQ(expected[16 + i], discontiguous[i * 2]);
    }

    // Offsets/sizes exceeding off_t (int64_t) fail with kArgument.
    const uint64_t kMaxOff = static_cast<uint64_t>(LimitsMax<int64_t>());
    HWY_ASSERT(in->Read(kMaxOff, WritableByteSpan(tail.data(), 1))
                   .Is(Status::kArgument));
    IOBatch overflow_batch(kMaxOff);
    HWY_ASSERT(overflow_batch.Add(Span<uint8_t>(tail.data(), 1)));
    HWY_ASSERT(overflow_batch.Read(*in, io_buf).Is(Status::kArgument));
  }

#if !HWY_OS_WIN
  // MapImmutable() on a writable (O_RDWR) file succeeds via MAP_PRIVATE and
  // remains readable after Close().
  {
    std::unique_ptr<File> rw = OpenFileOrAbort(path, FileMode::kWriteUpdate);
    MapPtr rw_map;
    HWY_ASSERT(rw->MapImmutable(rw_map) == Status(4096));
    HWY_ASSERT(rw->Close().ok());
    HWY_ASSERT_ARRAY_EQ(expected.data(), rw_map.get(), 4096);
  }

  // FIFOs without a writer (or reader) must not block MaybeOpenFile or
  // StatFile, and are listed as kOther.
  const Path fifo_path = TempPathname("hwy_io_test.fifo");
  (void)unlink(fifo_path.c_str());
  if (mkfifo(fifo_path.c_str(), 0600) == 0) {
    HWY_ASSERT(!IsFile(fifo_path));
    HWY_ASSERT(StatFile(fifo_path, os_stat) == Status(0));
    HWY_ASSERT(os_stat.kind == EntryKind::kOther);
    std::unique_ptr<File> fifo_file;
    HWY_ASSERT(MaybeOpenFile(fifo_path, FileMode::kRead, fifo_file)
                   .Is(Status::kArgument));
    HWY_ASSERT(fifo_file == nullptr);
    HWY_ASSERT(
        !MaybeOpenFile(fifo_path, FileMode::kWriteUpdate, fifo_file).ok());
    HWY_ASSERT(fifo_file == nullptr);
    std::vector<DirEntry> entries;
    HWY_ASSERT(ListDir(fifo_path.Dirname(), entries).ok());
    bool found = false;
    for (const DirEntry& entry : entries) {
      if (entry.name != fifo_path.Filename()) continue;
      found = true;
      HWY_ASSERT(entry.kind == EntryKind::kOther);
    }
    HWY_ASSERT(found);
    (void)unlink(fifo_path.c_str());
  }

#if HWY_OS_LINUX
  // Pseudo-files report size 0 despite having contents; a 1-byte probe turns
  // the would-be silent truncation into an error.
  const Path proc_path("/proc/self/status");
  std::unique_ptr<File> proc_file;
  if (MaybeOpenFile(proc_path, FileMode::kRead, proc_file).ok()) {
    HWY_ASSERT(proc_file->Size() == Status(0));
    MapPtr proc_map;
    HWY_ASSERT(proc_file->MapImmutable(proc_map).Is(Status::kUnimplemented));
    HWY_ASSERT(proc_map == nullptr);
    HWY_ASSERT(proc_file->MapShared(proc_map).Is(Status::kUnimplemented));
    AlignedPtr<uint8_t> proc_bytes;
    HWY_ASSERT(
        ReadFileToAligned(proc_path, proc_bytes).Is(Status::kUnimplemented));
    HWY_ASSERT(proc_bytes == nullptr);
  }
#endif  // HWY_OS_LINUX

  // Unreadable regular files (chmod 000) are still found by StatFile.
  const Path unread_path = TempPathname("hwy_io_test_unreadable.bin");
  {
    std::unique_ptr<File> f =
        OpenFileOrAbort(unread_path, FileMode::kWriteTruncate);
  }
  if (chmod(unread_path.c_str(), 0) == 0) {
    HWY_ASSERT(IsFile(unread_path));
    (void)chmod(unread_path.c_str(), 0600);
  }
  (void)RemoveFile(unread_path);
#endif

  // kWriteUpdate preserves contents and allows reading and overwriting.
  {
    std::unique_ptr<File> upd = OpenFileOrAbort(path, FileMode::kWriteUpdate);
    HWY_ASSERT(upd->Size() == Status(4096));
    uint8_t sample[32] = {};
    HWY_ASSERT(upd->Read(/*offset=*/256, WritableByteSpan(sample, 32)).ok());
    HWY_ASSERT_ARRAY_EQ(expected.data() + 256, sample, sizeof(sample));
    const uint8_t patch[4] = {1, 2, 3, 4};
    HWY_ASSERT(upd->Write(/*offset=*/4096, ByteSpan(patch, 4)).ok());
    HWY_ASSERT(upd->Size() == Status(4100));
  }

  // kWriteTruncate truncates.
  {
    std::unique_ptr<File> out = OpenFileOrAbort(path, FileMode::kWriteTruncate);
    HWY_ASSERT(out->Size() == Status(0));
  }

  // kWriteNew fails with kExists when the file already exists.
  {
    std::unique_ptr<File> excl;
    HWY_ASSERT(
        MaybeOpenFile(path, FileMode::kWriteNew, excl).Is(IOCode::kExists));
    HWY_ASSERT(excl == nullptr);
  }

  (void)RemoveFile(path);

  // kWriteNew succeeds when the file does not exist; test WriteSpans /
  // IOBatch::Write (pwritev on POSIX), OpenOptions, Advise, and IOCounters.
  IOCounters::Reset();
  {
    std::unique_ptr<File> out =
        OpenFileOrAbort(path, {FileMode::kWriteNew, AccessHint::kSequential});
    HWY_ASSERT(out->Size() == Status(0));
    IOBatch write_batch(/*offset=*/0);
    HWY_ASSERT(
        write_batch.AddForWrite(Span<const uint8_t>(expected.data(), 512)));
    HWY_ASSERT(write_batch.AddForWrite(
        Span<const uint8_t>(expected.data() + 1024, 512)));
    HWY_ASSERT(write_batch.Write(*out) == Status(1024));
    HWY_ASSERT(out->Size() == Status(1024));

    // > IOV_MAX discontiguous spans chunked across multiple pwritev calls.
#ifdef IOV_MAX
    const size_t kManySpans = IOV_MAX + 5;
#else
    const size_t kManySpans = 1029;
#endif
    std::vector<uint8_t> discontiguous(kManySpans * 2, 0);
    for (size_t i = 0; i < kManySpans; ++i) {
      discontiguous[i * 2] = expected[i];
    }
    IOBatch many_write(/*offset=*/1024);
    for (size_t i = 0; i < kManySpans; ++i) {
      HWY_ASSERT(many_write.AddForWrite(
          Span<const uint8_t>(&discontiguous[i * 2], 1)));
    }
    HWY_ASSERT(many_write.Write(*out) == Status(kManySpans));
    HWY_ASSERT(out->Size() == Status(1024 + kManySpans));

    // On writable POSIX files, Advise(kDontNeed) flushes dirty pages first so
    // POSIX_FADV_DONTNEED actually evicts them. Also test size == 0 ("to EOF").
    const Status advise_status = out->Advise(0, 1024, AccessHint::kDontNeed);
    HWY_ASSERT(advise_status.ok() || advise_status.Is(Status::kUnimplemented));
    const Status advise_eof = out->Advise(1024, 0, AccessHint::kDontNeed);
    HWY_ASSERT(advise_eof.ok() || advise_eof.Is(Status::kUnimplemented));
    HWY_ASSERT(out->Close().ok());
    HWY_ASSERT(
        out->Advise(0, 0, AccessHint::kNormal).Is(Status::kPrecondition));
    HWY_ASSERT(
        out->WriteSpans(0, Span<const IOSpan>()).Is(Status::kPrecondition));
  }
  {
    std::unique_ptr<File> in =
        OpenFileOrAbort(path, {FileMode::kRead, AccessHint::kRandom});
    HWY_ASSERT(in->WriteSpans(0, Span<const IOSpan>()).Is(IOCode::kWrongMode));
    const Status advise_status = in->Advise(0, 1024, AccessHint::kWillNeed);
    HWY_ASSERT(advise_status.ok() || advise_status.Is(Status::kUnimplemented));
    std::vector<uint8_t> check(1024);
    HWY_ASSERT(in->Read(0, WritableByteSpan(check)) == Status(1024));
    HWY_ASSERT_ARRAY_EQ(expected.data(), check.data(), 512);
    HWY_ASSERT_ARRAY_EQ(expected.data() + 1024, check.data() + 512, 512);
  }
  const IOCounters counters = IOCounters::Get();
#if PROFILER_ENABLED
  HWY_ASSERT(counters.bytes_written >= 1024);
  HWY_ASSERT(counters.bytes_read >= 1024);
  HWY_ASSERT(counters.syscalls >= 4);
#else
  HWY_ASSERT_EQ(uint64_t{0}, counters.bytes_written);
  HWY_ASSERT_EQ(uint64_t{0}, counters.bytes_read);
  HWY_ASSERT_EQ(uint64_t{0}, counters.syscalls);
  HWY_ASSERT_EQ(uint64_t{0}, counters.eintr_retries);
#endif

  (void)RemoveFile(path);
#endif  // HWY_IO_TEST_OS_FILES
}

// Concurrent Read/Write on the same File, each with distinct offsets.
TEST(FileTest, TestParallelReadWrite) {
#if HWY_IO_TEST_OS_FILES
  if (!HaveThreadingSupport()) return;
  const Path path = TempPathname("hwy_io_test_parallel.bin");
  constexpr size_t kThreads = 8;
  constexpr size_t kBytesPerThread = 256 * 1024;
  auto expected = [](size_t i) {
    return static_cast<uint8_t>((i * 131 + (i >> 9)) & 0xFF);
  };

  std::unique_ptr<File> file = OpenFileOrAbort(path, FileMode::kWriteTruncate);
  HWY_ASSERT(file->SetSize(kThreads * kBytesPerThread).ok());
  {
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t]() {
#if PROFILER_ENABLED
        Profiler::Get().ReserveWorker(t + 1);
        Profiler::SetGlobalIdx(t + 1);
#endif
        std::vector<uint8_t> buf(kBytesPerThread);
        const size_t begin = t * kBytesPerThread;
        for (size_t i = 0; i < kBytesPerThread; ++i) {
          buf[i] = expected(begin + i);
        }
        HWY_ASSERT(file->Write(begin, ByteSpan(buf)).ok());
#if PROFILER_ENABLED
        Profiler::Get().FreeWorker(t + 1);
#endif
      });
    }
    for (std::thread& thread : threads) thread.join();
  }
  HWY_ASSERT(file->Size() == Status(kThreads * kBytesPerThread));

  {
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t]() {
#if PROFILER_ENABLED
        Profiler::Get().ReserveWorker(t + 1);
        Profiler::SetGlobalIdx(t + 1);
#endif
        // Read a different thread's region to ensure all writes are visible.
        const size_t begin = ((t + 1) % kThreads) * kBytesPerThread;
        std::vector<uint8_t> buf(kBytesPerThread);
        HWY_ASSERT(file->Read(begin, WritableByteSpan(buf)).ok());
        for (size_t i = 0; i < kBytesPerThread; ++i) {
          HWY_ASSERT_EQ(expected(begin + i), buf[i]);
        }
#if PROFILER_ENABLED
        Profiler::Get().FreeWorker(t + 1);
#endif
      });
    }
    for (std::thread& thread : threads) thread.join();
  }

  file.reset();
  (void)RemoveFile(path);
#endif  // HWY_IO_TEST_OS_FILES
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
