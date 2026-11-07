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

#include "hwy/io/stream_file.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <vector>

#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/file_util.h"
#include "hwy/io/io_batch.h"
#include "hwy/io/io_span.h"
#include "hwy/io/io_test_util.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

#if HWY_OS_WIN && HWY_IO_HAVE_OS_FILES
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// For `posix_openpt` (glibc declares it under `_GNU_SOURCE`, which C++
// compilers define).
#if HWY_OS_LINUX && !defined(__ANDROID__) && HWY_IO_HAVE_OS_FILES
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#define HWY_IO_TEST_PTY 1
#else
#define HWY_IO_TEST_PTY 0
#endif

namespace hwy {
namespace {

// Stream over a vector that transfers at most `max_chunk` bytes per call, as
// pipes may, and can inject an error after `fail_after` bytes.
class VectorStream final : public StreamFile {
 public:
  VectorStream(Direction direction, std::vector<uint8_t>* bytes,
               size_t max_chunk)
      : StreamFile(direction), bytes_(bytes), max_chunk_(max_chunk) {}

  size_t fail_after = ~size_t{0};
  mutable size_t calls = 0;

 protected:
  Status ReadSome(Span<uint8_t> to) const override {
    ++calls;
    HWY_ASSERT(!to.empty());
    if (pos_ >= fail_after) return Status(Status::kUnavailable, 11);
    const size_t n =
        HWY_MIN(HWY_MIN(to.size(), max_chunk_), bytes_->size() - pos_);
    if (n != 0) CopyBytes(bytes_->data() + pos_, to.data(), n);
    pos_ += n;
    return Status(n);
  }

  Status WriteSome(Span<const uint8_t> from) override {
    ++calls;
    HWY_ASSERT(!from.empty());
    if (bytes_->size() >= fail_after) return Status(Status::kOSError, 32);
    const size_t n = HWY_MIN(from.size(), max_chunk_);
    bytes_->insert(bytes_->end(), from.data(), from.data() + n);
    return Status(n);
  }

 private:
  std::vector<uint8_t>* bytes_;
  size_t max_chunk_;
  mutable size_t pos_ = 0;
};

TEST(StreamFileTest, TestRead) {
  std::vector<uint8_t> src(100);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
  VectorStream in(StreamFile::Direction::kRead, &src, /*max_chunk=*/7);
  IOWorkerBuf io_buf;

  // Sequential: no random access, size, or mapping.
  HWY_ASSERT(in.Capabilities().Empty());
  HWY_ASSERT(!in.IsTerminal());
  HWY_ASSERT(in.Size().Is(Status::kUnimplemented));
  // Operations that would modify the stream are wrong for a read stream.
  HWY_ASSERT(in.SetSize(0).Is(IOCode::kWrongMode));
  HWY_ASSERT(in.Preallocate(0, 1).Is(IOCode::kWrongMode));
  HWY_ASSERT(in.Sync().Is(IOCode::kWrongMode));
  HWY_ASSERT(
      in.Advise(0, 0, AccessHint::kSequential).Is(Status::kUnimplemented));
  MapPtr map;
  HWY_ASSERT(in.MapImmutable(map).Is(Status::kUnimplemented));
  FileStat stat;
  HWY_ASSERT(in.Stat(stat).Is(Status::kUnimplemented));
  HWY_ASSERT(in.MapShared(map).Is(Status::kUnimplemented));
  MutableMapPtr writable;
  HWY_ASSERT(in.MapWritable(writable).Is(Status::kUnimplemented));
  HWY_ASSERT(map == nullptr);
  HWY_ASSERT(in.MaybeSpan(0, 0).data() == nullptr);
  HWY_ASSERT_EQ(uint64_t{0}, in.Position());

  // Short transfers are looped until the request is satisfied.
  uint8_t buf[30] = {};
  HWY_ASSERT(in.Read(0, WritableByteSpan(buf, 20)) == Status(20));
  HWY_ASSERT_EQ(size_t{3}, in.calls);
  HWY_ASSERT_ARRAY_EQ(src.data(), buf, 20);
  HWY_ASSERT_EQ(uint64_t{20}, in.Position());
  // Offsets must equal the position.
  HWY_ASSERT(in.Read(0, WritableByteSpan(buf, 1)).Is(Status::kArgument));
  HWY_ASSERT(in.Read(21, WritableByteSpan(buf, 1)).Is(Status::kArgument));
  HWY_ASSERT(in.Read(20, Span<uint8_t>()) == Status(0));
  // Sizes or ends exceeding int64_t are rejected as for other `File`s.
  const uint64_t kMaxOff = static_cast<uint64_t>(LimitsMax<int64_t>());
  HWY_ASSERT(in.Read(kMaxOff, WritableByteSpan(buf, 1)).Is(Status::kArgument));
  HWY_ASSERT_EQ(uint64_t{20}, in.Position());
  HWY_ASSERT(in.Write(20, ByteSpan(buf, 1)).Is(IOCode::kWrongMode));

  // IOBatch reads discontiguous spans in order.
  uint8_t part1[5] = {};
  uint8_t part2[10] = {};
  IOBatch batch(in.Position());
  HWY_ASSERT(batch.Add(Span<uint8_t>(part1, 5)));
  HWY_ASSERT(batch.Add(Span<uint8_t>(part2, 10)));
  HWY_ASSERT(batch.Read(in, io_buf) == Status(15));
  HWY_ASSERT_ARRAY_EQ(src.data() + 20, part1, 5);
  HWY_ASSERT_ARRAY_EQ(src.data() + 25, part2, 10);
  HWY_ASSERT_EQ(uint64_t{35}, in.Position());

  // End of stream: kTruncated with the bytes read; position advances.
  HWY_ASSERT(in.Read(35, WritableByteSpan(buf, 30)) == Status(30));
  HWY_ASSERT(in.Read(65, WritableByteSpan(buf, 30)) == Status(30));
  HWY_ASSERT(in.Read(95, WritableByteSpan(buf, 30)) ==
             Status(Status::kTruncated, 5));
  HWY_ASSERT_ARRAY_EQ(src.data() + 95, buf, 5);
  HWY_ASSERT_EQ(uint64_t{100}, in.Position());
  HWY_ASSERT(in.Read(100, WritableByteSpan(buf, 1)) ==
             Status(Status::kTruncated, 0));

  HWY_ASSERT(in.Close().ok());
  HWY_ASSERT(in.Read(100, WritableByteSpan(buf, 1)).Is(Status::kPrecondition));
  HWY_ASSERT(in.Size().Is(Status::kPrecondition));
  HWY_ASSERT(in.MapImmutable(map).Is(Status::kPrecondition));
  HWY_ASSERT(in.MapWritable(writable).Is(Status::kPrecondition));
  HWY_ASSERT(in.SetSize(0).Is(Status::kPrecondition));
  HWY_ASSERT(in.Preallocate(0, 1).Is(Status::kPrecondition));
  HWY_ASSERT(in.Sync().Is(Status::kPrecondition));
  HWY_ASSERT(in.Advise(0, 0, AccessHint::kNormal).Is(Status::kPrecondition));

  // Errors after a partial transfer: the consumed bytes advance the position.
  VectorStream failing(StreamFile::Direction::kRead, &src, /*max_chunk=*/4);
  failing.fail_after = 8;
  const Status s = failing.Read(0, WritableByteSpan(buf, 10));
  HWY_ASSERT(s == Status(Status::kUnavailable, 11));
  HWY_ASSERT_EQ(uint64_t{8}, failing.Position());
  HWY_ASSERT_ARRAY_EQ(src.data(), buf, 8);
}

TEST(StreamFileTest, TestVerifyEmpty) {
  // Empty stream: success without changing the position.
  std::vector<uint8_t> empty_src;
  VectorStream empty(StreamFile::Direction::kRead, &empty_src,
                     /*max_chunk=*/4);
  HWY_ASSERT(VerifyEmpty(empty).ok());
  HWY_ASSERT_EQ(uint64_t{0}, empty.Position());

  // Non-empty stream: the probe consumes one byte.
  std::vector<uint8_t> src(3, 7);
  VectorStream in(StreamFile::Direction::kRead, &src, /*max_chunk=*/4);
  HWY_ASSERT(VerifyEmpty(in).Is(Status::kUnimplemented));
  HWY_ASSERT_EQ(uint64_t{1}, in.Position());
  // Requires `Position() == 0`.
  HWY_ASSERT(VerifyEmpty(in).Is(Status::kArgument));

  // Write streams cannot be read.
  std::vector<uint8_t> dst;
  VectorStream out(StreamFile::Direction::kWrite, &dst, /*max_chunk=*/4);
  HWY_ASSERT(VerifyEmpty(out).Is(IOCode::kWrongMode));
}

TEST(StreamFileTest, TestWrite) {
  std::vector<uint8_t> src(50);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i * 3);
  }
  std::vector<uint8_t> dst;
  VectorStream out(StreamFile::Direction::kWrite, &dst, /*max_chunk=*/6);
  IOWorkerBuf io_buf;
  uint8_t buf[4];
  HWY_ASSERT(out.Read(0, WritableByteSpan(buf, 1)).Is(IOCode::kWrongMode));
  HWY_ASSERT(out.SetSize(0).Is(Status::kUnimplemented));
  HWY_ASSERT(out.Preallocate(0, 1).Is(Status::kUnimplemented));
  HWY_ASSERT(out.Sync().Is(Status::kUnimplemented));

  HWY_ASSERT(out.Write(0, ByteSpan(src.data(), 20)) == Status(20));
  HWY_ASSERT_EQ(size_t{4}, out.calls);
  HWY_ASSERT_EQ(uint64_t{20}, out.Position());
  HWY_ASSERT(out.Write(0, ByteSpan(src.data(), 1)).Is(Status::kArgument));

  // IOBatch gathers spans in order.
  IOBatch batch(out.Position());
  HWY_ASSERT(batch.AddForWrite(ByteSpan(src.data() + 20, 10)));
  HWY_ASSERT(batch.AddForWrite(ByteSpan(src.data() + 40, 10)));
  HWY_ASSERT(batch.Write(out) == Status(20));
  HWY_ASSERT_EQ(uint64_t{40}, out.Position());
  HWY_ASSERT_EQ(size_t{40}, dst.size());
  HWY_ASSERT_ARRAY_EQ(src.data(), dst.data(), 30);
  HWY_ASSERT_ARRAY_EQ(src.data() + 40, dst.data() + 30, 10);
  // Write-only batches cannot be read.
  HWY_ASSERT(batch.Read(out, io_buf).Is(IOCode::kWrongMode));

  // Errors after a partial transfer advance the position.
  out.fail_after = 45;
  HWY_ASSERT(out.Write(40, ByteSpan(src.data(), 10)) ==
             Status(Status::kOSError, 32));
  HWY_ASSERT_EQ(uint64_t{46}, out.Position());

  // A subclass that transfers zero bytes without an error cannot loop forever.
  {
    std::vector<uint8_t> stuck_dst;
    VectorStream stuck(StreamFile::Direction::kWrite, &stuck_dst,
                       /*max_chunk=*/0);
    HWY_ASSERT(stuck.Write(0, ByteSpan(src.data(), 3)) ==
               Status(IOCode::kNoProgress, 0));
    HWY_ASSERT_EQ(uint64_t{0}, stuck.Position());
  }
  HWY_ASSERT(out.Close().ok());
  HWY_ASSERT(out.Write(46, ByteSpan(src.data(), 1)).Is(Status::kPrecondition));
  HWY_ASSERT(out.SetSize(0).Is(Status::kPrecondition));
  HWY_ASSERT(out.Sync().Is(Status::kPrecondition));
}

TEST(StreamFileTest, TestStdinStdout) {
  std::unique_ptr<File> file;
#if !HWY_IO_HAVE_OS_FILES
  HWY_ASSERT(OpenStdin(file).Is(Status::kUnimplemented));
  HWY_ASSERT(OpenStdout(file).Is(Status::kUnimplemented));
  HWY_ASSERT(file == nullptr);
#elif HWY_IO_TEST_OS_FILES && !HWY_OS_WIN
  const uint8_t kMessage[] = "standard streams";
  const size_t kLen = sizeof(kMessage);

  // Temporarily redirect stdin to a pipe.
  int fds[2];
  HWY_ASSERT_EQ(0, pipe(fds));
  HWY_ASSERT(write(fds[1], kMessage, kLen) == static_cast<ssize_t>(kLen));
  HWY_ASSERT_EQ(0, close(fds[1]));  // Reader sees end of stream.
  const int saved_stdin = dup(STDIN_FILENO);
  HWY_ASSERT(saved_stdin >= 0);
  HWY_ASSERT(dup2(fds[0], STDIN_FILENO) == STDIN_FILENO);
  uint8_t buf[64] = {};
  HWY_ASSERT(OpenStdin(file).ok());
  HWY_ASSERT(file->Capabilities().Empty());
  HWY_ASSERT(!file->IsTerminal());  // Pipe.
  HWY_ASSERT(file->Handle() == STDIN_FILENO);
  const Status read_status = file->Read(0, WritableByteSpan(buf, sizeof(buf)));
  const Status write_status = file->Write(0, ByteSpan(buf, 1));
  HWY_ASSERT(file->Close().ok());  // Does not close STDIN_FILENO.
  HWY_ASSERT(file->Handle() == -1);
  HWY_ASSERT(dup2(saved_stdin, STDIN_FILENO) == STDIN_FILENO);
  HWY_ASSERT_EQ(0, close(saved_stdin));
  HWY_ASSERT_EQ(0, close(fds[0]));
  HWY_ASSERT(read_status == Status(Status::kTruncated, kLen));
  HWY_ASSERT_ARRAY_EQ(kMessage, buf, kLen);
  HWY_ASSERT(write_status.Is(IOCode::kWrongMode));

  // Temporarily redirect stdout to a pipe (small enough not to block).
  HWY_ASSERT_EQ(0, pipe(fds));
  fflush(stdout);
  const int saved_stdout = dup(STDOUT_FILENO);
  HWY_ASSERT(saved_stdout >= 0);
  HWY_ASSERT(dup2(fds[1], STDOUT_FILENO) == STDOUT_FILENO);
  const Status open_status = OpenStdout(file);
  Status put_status, put2_status, get_status;
  bool is_terminal = true;
  if (open_status.ok()) {
    is_terminal = file->IsTerminal();
    put_status = file->Write(0, ByteSpan(kMessage, 5));
    put2_status = file->Write(5, ByteSpan(kMessage + 5, kLen - 5));
    get_status = file->Read(0, WritableByteSpan(buf, 1));
    (void)file->Close();
  }
  HWY_ASSERT(dup2(saved_stdout, STDOUT_FILENO) == STDOUT_FILENO);
  HWY_ASSERT_EQ(0, close(saved_stdout));
  HWY_ASSERT_EQ(0, close(fds[1]));
  HWY_ASSERT(open_status.ok());
  HWY_ASSERT(!is_terminal);
  HWY_ASSERT(put_status == Status(5));
  HWY_ASSERT(put2_status == Status(kLen - 5));
  HWY_ASSERT(get_status.Is(IOCode::kWrongMode));
  memset(buf, 0, sizeof(buf));
  HWY_ASSERT(read(fds[0], buf, sizeof(buf)) == static_cast<ssize_t>(kLen));
  HWY_ASSERT_ARRAY_EQ(kMessage, buf, kLen);
  HWY_ASSERT_EQ(0, close(fds[0]));
#elif HWY_IO_TEST_OS_FILES && HWY_OS_WIN
  const uint8_t kMessage[] = "standard streams";
  const DWORD kLen = static_cast<DWORD>(sizeof(kMessage));

  // Temporarily redirect the standard input handle to a pipe.
  HANDLE read_end = nullptr;
  HANDLE write_end = nullptr;
  HWY_ASSERT(CreatePipe(&read_end, &write_end, nullptr, 0));
  DWORD put = 0;
  HWY_ASSERT(WriteFile(write_end, kMessage, kLen, &put, nullptr));
  HWY_ASSERT_EQ(kLen, put);
  HWY_ASSERT(CloseHandle(write_end));  // Reader sees end of stream.
  const HANDLE saved_stdin = GetStdHandle(STD_INPUT_HANDLE);
  HWY_ASSERT(SetStdHandle(STD_INPUT_HANDLE, read_end));
  uint8_t buf[64] = {};
  const Status open_in = OpenStdin(file);
  Status read_status;
  Status write_status;
  bool in_is_terminal = true;
  if (open_in.ok()) {
    in_is_terminal = file->IsTerminal();
    read_status = file->Read(0, WritableByteSpan(buf, sizeof(buf)));
    write_status = file->Write(0, ByteSpan(buf, 1));
    (void)file->Close();  // Does not close the handle.
  }
  HWY_ASSERT(SetStdHandle(STD_INPUT_HANDLE, saved_stdin));
  HWY_ASSERT(CloseHandle(read_end));
  HWY_ASSERT(open_in.ok());
  HWY_ASSERT(!in_is_terminal);  // Pipe.
  HWY_ASSERT(read_status == Status(Status::kTruncated, kLen));
  HWY_ASSERT_ARRAY_EQ(kMessage, buf, kLen);
  HWY_ASSERT(write_status.Is(IOCode::kWrongMode));

  // Temporarily redirect the standard output handle to a pipe (small enough
  // not to block).
  HWY_ASSERT(CreatePipe(&read_end, &write_end, nullptr, 0));
  fflush(stdout);
  const HANDLE saved_stdout = GetStdHandle(STD_OUTPUT_HANDLE);
  HWY_ASSERT(SetStdHandle(STD_OUTPUT_HANDLE, write_end));
  const Status open_out = OpenStdout(file);
  Status put_status;
  Status get_status;
  bool out_is_terminal = true;
  if (open_out.ok()) {
    out_is_terminal = file->IsTerminal();
    put_status = file->Write(0, ByteSpan(kMessage, kLen));
    get_status = file->Read(0, WritableByteSpan(buf, 1));
    (void)file->Close();
  }
  HWY_ASSERT(SetStdHandle(STD_OUTPUT_HANDLE, saved_stdout));
  HWY_ASSERT(CloseHandle(write_end));
  HWY_ASSERT(open_out.ok());
  HWY_ASSERT(!out_is_terminal);
  HWY_ASSERT(put_status == Status(kLen));
  HWY_ASSERT(get_status.Is(IOCode::kWrongMode));
  memset(buf, 0, sizeof(buf));
  DWORD got = 0;
  HWY_ASSERT(ReadFile(read_end, buf, sizeof(buf), &got, nullptr));
  HWY_ASSERT_EQ(kLen, got);
  HWY_ASSERT_ARRAY_EQ(kMessage, buf, kLen);
  HWY_ASSERT(CloseHandle(read_end));
#else
  (void)file;
#endif
}

// Pseudo-terminals are terminals, as opposed to the pipes above. Skips if the
// environment (e.g. a sandbox) provides no `/dev/ptmx`.
TEST(StreamFileTest, TestTerminal) {
#if HWY_IO_TEST_PTY
  const int master = posix_openpt(O_RDWR | O_NOCTTY);
  if (master < 0) {
    HWY_WARN("Skipping: posix_openpt failed (errno %d).", errno);
    return;
  }
  const char* name = (grantpt(master) == 0 && unlockpt(master) == 0)
                         ? ptsname(master)
                         : nullptr;
  const int slave = name ? open(name, O_RDWR | O_NOCTTY) : -1;
  if (slave < 0) {
    HWY_WARN("Skipping: cannot open pty slave (errno %d).", errno);
    HWY_ASSERT_EQ(0, close(master));
    return;
  }

  std::unique_ptr<File> file;
  fflush(stdout);
  const int saved_stdin = dup(STDIN_FILENO);
  const int saved_stdout = dup(STDOUT_FILENO);
  HWY_ASSERT(saved_stdin >= 0 && saved_stdout >= 0);
  HWY_ASSERT(dup2(slave, STDIN_FILENO) == STDIN_FILENO);
  HWY_ASSERT(dup2(slave, STDOUT_FILENO) == STDOUT_FILENO);
  const Status open_in = OpenStdin(file);
  const bool in_is_terminal = open_in.ok() && file->IsTerminal();
  if (open_in.ok()) HWY_ASSERT(file->Close().ok());
  const bool closed_is_terminal = open_in.ok() && file->IsTerminal();
  const Status open_out = OpenStdout(file);
  const bool out_is_terminal = open_out.ok() && file->IsTerminal();
  file.reset();
  HWY_ASSERT(dup2(saved_stdin, STDIN_FILENO) == STDIN_FILENO);
  HWY_ASSERT(dup2(saved_stdout, STDOUT_FILENO) == STDOUT_FILENO);
  HWY_ASSERT_EQ(0, close(saved_stdin));
  HWY_ASSERT_EQ(0, close(saved_stdout));
  HWY_ASSERT_EQ(0, close(slave));
  HWY_ASSERT_EQ(0, close(master));

  HWY_ASSERT(open_in.ok());
  HWY_ASSERT(in_is_terminal);
  HWY_ASSERT(!closed_is_terminal);
  HWY_ASSERT(open_out.ok());
  HWY_ASSERT(out_is_terminal);
#endif
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
