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

#include "hwy/io/io_batch.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include "hwy/base.h"
#include "hwy/io/file.h"
#include "hwy/io/io_span.h"
#include "hwy/io/io_test_util.h"
#include "hwy/io/memory_file.h"
#include "hwy/status.h"
#include "hwy/tests/hwy_gtest.h"
#include "hwy/tests/test_util-inl.h"

namespace hwy {
namespace {

TEST(IOBatchTest, TestIOBatchMemoryFile) {
  IOWorkerBuf io_buf;
  std::vector<uint8_t> src(256);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i);
  }
  MemoryFile ro_file(src);

  // Contiguous spans coalesce into a single span (so > 256 contiguous additions
  // succeed) and empty spans are ignored without consuming a slot.
  uint8_t contiguous_dst[40] = {};
  IOBatch contiguous_batch(/*offset=*/10, /*user_tag=*/3);
  HWY_ASSERT(contiguous_batch.IsEmpty());
  HWY_ASSERT(contiguous_batch.Add(Span<uint8_t>()));
  HWY_ASSERT(contiguous_batch.IsEmpty());
  for (size_t i = 0; i < sizeof(contiguous_dst); ++i) {
    HWY_ASSERT(contiguous_batch.Add(Span<uint8_t>(contiguous_dst + i, 1)));
  }
  HWY_ASSERT(!contiguous_batch.IsEmpty());
  HWY_ASSERT_EQ(uint64_t{10}, contiguous_batch.Offset());
  HWY_ASSERT_EQ(uint64_t{40}, contiguous_batch.TotalBytes());
  HWY_ASSERT_EQ(uint64_t{3}, contiguous_batch.UserTag());
  HWY_ASSERT_EQ(uint64_t{50}, contiguous_batch.NextOffset());
  HWY_ASSERT(contiguous_batch.Read(ro_file, io_buf) == Status(40));
  for (size_t i = 0; i < sizeof(contiguous_dst); ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(10 + i), contiguous_dst[i]);
  }

  // Discontiguous spans: slow path. Reuse `contiguous_batch` via `Reset`.
  constexpr size_t kLen1 = 16;
  constexpr size_t kGap = 8;
  constexpr size_t kLen2 = 24;
  constexpr uint8_t kSentinel = 0xEE;
  std::vector<uint8_t> dst(kLen1 + kGap + kLen2, kSentinel);
  contiguous_batch.Reset(/*offset=*/32);
  HWY_ASSERT(contiguous_batch.IsEmpty());
  HWY_ASSERT_EQ(uint64_t{0}, contiguous_batch.UserTag());
  HWY_ASSERT(contiguous_batch.Add(Span<uint8_t>(dst.data(), kLen1)));
  HWY_ASSERT(
      contiguous_batch.Add(Span<uint8_t>(dst.data() + kLen1 + kGap, kLen2)));
  HWY_ASSERT(contiguous_batch.Read(ro_file, io_buf) == Status(kLen1 + kLen2));
  for (size_t i = 0; i < kLen1; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + i), dst[i]);
  }
  for (size_t i = 0; i < kGap; ++i) {
    HWY_ASSERT_EQ(kSentinel, dst[kLen1 + i]);  // Gap untouched.
  }
  for (size_t i = 0; i < kLen2; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + kLen1 + i), dst[kLen1 + kGap + i]);
  }

  // Reading past the end of the file fails.
  uint8_t past_end[16];
  IOBatch past_batch(/*offset=*/250);
  HWY_ASSERT(past_batch.Add(Span<uint8_t>(past_end, sizeof(past_end))));
  HWY_ASSERT(past_batch.Read(ro_file, io_buf) == Status(Status::kTruncated, 6));

  // Discontiguous spans past EOF: the payload is the total bytes read, and the
  // available prefix is scattered into the spans; the rest is untouched.
  std::vector<uint8_t> partial(10 + kGap + 10, kSentinel);
  IOBatch partial_batch(/*offset=*/240);
  HWY_ASSERT(partial_batch.Add(Span<uint8_t>(partial.data(), 10)));
  HWY_ASSERT(partial_batch.Add(Span<uint8_t>(partial.data() + 10 + kGap, 10)));
  HWY_ASSERT(partial_batch.Read(ro_file, io_buf) ==
             Status(Status::kTruncated, 16));
  for (size_t i = 0; i < 10; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(240 + i), partial[i]);
  }
  for (size_t i = 0; i < kGap; ++i) {
    HWY_ASSERT_EQ(kSentinel, partial[10 + i]);
  }
  for (size_t i = 0; i < 6; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(250 + i), partial[10 + kGap + i]);
  }
  for (size_t i = 6; i < 10; ++i) {
    HWY_ASSERT_EQ(kSentinel, partial[10 + kGap + i]);
  }

  // Per-worker `IOWorkerBuf` in `ReadSpans` / `IOBatch::Read`: starts empty,
  // stays empty on single-span fast path, and retains high-water-mark capacity
  // across multi-span fallback reads.
  static_assert(std::is_constructible<IOSpans, IOSpan(&)[4]>::value,
                "C array must construct IOSpans");
  static_assert(std::is_constructible<IOSpans, Span<IOSpan>>::value,
                "Span<IOSpan> must construct IOSpans");
  static_assert(!std::is_constructible<IOSpans, std::vector<IOSpan>&&>::value,
                "rvalue container must be rejected");
  static_assert(
      !std::is_constructible<IOSpans, Span<const IOSpan>, uint64_t>::value,
      "unchecked 2-arg IOSpans constructor must be private");
  static_assert(!std::is_constructible<IOSpans, Span<IOSpan>, size_t>::value,
                "unchecked 2-arg IOSpans constructor must be private");
  IOWorkerBuf worker_buf;
  HWY_ASSERT_EQ(size_t{0}, worker_buf.Capacity());
  HWY_ASSERT(worker_buf.Ensure(0) == nullptr);
  HWY_ASSERT_EQ(size_t{0}, worker_buf.Capacity());
  IOBatch single_batch(/*offset=*/0);
  HWY_ASSERT(single_batch.Add(Span<uint8_t>(contiguous_dst, 16)));
  HWY_ASSERT(single_batch.Read(ro_file, worker_buf) == Status(16));
  HWY_ASSERT_EQ(size_t{0}, worker_buf.Capacity());

  std::fill(dst.begin(), dst.end(), kSentinel);
  HWY_ASSERT(contiguous_batch.Read(ro_file, worker_buf) ==
             Status(kLen1 + kLen2));
  HWY_ASSERT_EQ(kLen1 + kLen2, worker_buf.Capacity());
  for (size_t i = 0; i < kLen1; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + i), dst[i]);
  }
  for (size_t i = 0; i < kLen2; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + kLen1 + i), dst[kLen1 + kGap + i]);
  }

  // Growing to a larger multi-span batch reallocates once to the new high-water
  // mark; subsequent smaller batches reuse it without shrinking.
  std::vector<uint8_t> larger_dst(30 + kGap + 30, kSentinel);
  IOBatch larger_batch(/*offset=*/0);
  HWY_ASSERT(larger_batch.Add(Span<uint8_t>(larger_dst.data(), 30)));
  HWY_ASSERT(
      larger_batch.Add(Span<uint8_t>(larger_dst.data() + 30 + kGap, 30)));
  HWY_ASSERT(larger_batch.Read(ro_file, worker_buf) == Status(60));
  HWY_ASSERT_EQ(size_t{60}, worker_buf.Capacity());
  HWY_ASSERT_ARRAY_EQ(src.data(), larger_dst.data(), 30);
  HWY_ASSERT_ARRAY_EQ(src.data() + 30, larger_dst.data() + 30 + kGap, 30);

  // Direct File::ReadSpans call with empty IOSpan{nullptr, 0} preceding and
  // separating non-empty spans must skip the empty spans and scatter the rest.
  // Also exercises IOSpans construction from C arrays and Span<IOSpan>, and
  // custom File backends calling IOWorkerBuf::Ensure directly.
  std::fill(dst.begin(), dst.end(), kSentinel);
  IOSpan mixed_spans[4] = {
      {nullptr, 0},
      {dst.data(), kLen1},
      {nullptr, 0},
      {dst.data() + kLen1 + kGap, kLen2},
  };
  const IOSpans mixed_from_mutable((Span<IOSpan>(mixed_spans)));
  HWY_ASSERT_EQ(kLen1 + kLen2, mixed_from_mutable.TotalBytes());
  HWY_ASSERT(ro_file.ReadSpans(/*offset=*/32, IOSpans(mixed_spans), io_buf) ==
             Status(kLen1 + kLen2));
  for (size_t i = 0; i < kLen1; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + i), dst[i]);
  }
  for (size_t i = 0; i < kGap; ++i) {
    HWY_ASSERT_EQ(kSentinel, dst[kLen1 + i]);
  }
  for (size_t i = 0; i < kLen2; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + kLen1 + i), dst[kLen1 + kGap + i]);
  }

  struct CustomScratchFile : public MemoryFile {
    using MemoryFile::MemoryFile;

    Status ReadSpans(uint64_t offset, const IOSpans& spans,
                     IOWorkerBuf& buf) const override {
      const Status total = spans.ValidateAt(offset);
      if (!total.ok()) return total;
      if (total.size() == 0) return MemoryFile::ReadSpans(offset, spans, buf);
      const size_t want = static_cast<size_t>(total.size());
      uint8_t* scratch = buf.Ensure(want);
      HWY_ASSERT(scratch != nullptr);
      const Status s = Read(offset, Span<uint8_t>(scratch, want));
      if (!s.ok()) return s;
      size_t pos = 0;
      for (const IOSpan& span : spans) {
        if (span.bytes != 0) CopyBytes(scratch + pos, span.mem, span.bytes);
        pos += span.bytes;
      }
      return s;
    }
  };
  std::fill(dst.begin(), dst.end(), kSentinel);
  CustomScratchFile custom_ro(src);
  IOWorkerBuf custom_buf;
  HWY_ASSERT(custom_ro.ReadSpans(/*offset=*/32, IOSpans(mixed_spans),
                                 custom_buf) == Status(kLen1 + kLen2));
  HWY_ASSERT_EQ(kLen1 + kLen2, custom_buf.Capacity());
  HWY_ASSERT(custom_buf.Ensure(0) != nullptr);
  for (size_t i = 0; i < kLen1; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + i), dst[i]);
  }
  for (size_t i = 0; i < kLen2; ++i) {
    HWY_ASSERT_EQ(static_cast<uint8_t>(32 + kLen1 + i), dst[kLen1 + kGap + i]);
  }

  // Gather write via IOBatch::Write and AddForWrite.
  std::vector<uint8_t> out_buf(128, 0);
  MemoryFile rw_file(out_buf, FileMode::kWriteTruncate);
  IOBatch write_batch(/*offset=*/8);
  HWY_ASSERT(write_batch.AddForWrite(Span<const uint8_t>(src.data() + 10, 12)));
  HWY_ASSERT(write_batch.AddForWrite(Span<const uint8_t>(src.data() + 40, 20)));
  // Reading into a write-only batch fails with kWrongMode.
  HWY_ASSERT(write_batch.Read(ro_file, io_buf).Is(IOCode::kWrongMode));
  HWY_ASSERT(write_batch.Read(ro_file, worker_buf).Is(IOCode::kWrongMode));
  HWY_ASSERT(write_batch.Write(ro_file).Is(IOCode::kWrongMode));
  HWY_ASSERT(write_batch.Write(rw_file) == Status(32));
  HWY_ASSERT(rw_file.Size() == Status(40));
  HWY_ASSERT_ARRAY_EQ(src.data() + 10, out_buf.data() + 8, 12);
  HWY_ASSERT_ARRAY_EQ(src.data() + 40, out_buf.data() + 20, 20);

  // IOSpans overflow validation (> Status::kMaxPayload) fails with kArgument.
  if (ScalarFits<size_t>(Status::kMaxPayload)) {
    const size_t kMaxPay = static_cast<size_t>(Status::kMaxPayload);
    const std::vector<IOSpan> bad_vec = {{nullptr, kMaxPay}, {nullptr, 1}};
    const IOSpans bad_spans(bad_vec);
    HWY_ASSERT(!bad_spans.IsValid());
    HWY_ASSERT(ro_file.ReadSpans(0, bad_spans, io_buf).Is(Status::kArgument));
    HWY_ASSERT(rw_file.WriteSpans(0, bad_spans).Is(Status::kArgument));
  }

  // AddStrided: packed fast path (stride_bytes == bytes_per_row) coalesces into
  // a single span; padded path (stride_bytes > bytes_per_row) scatters/gathers
  // row by row leaving padding untouched.
  IOBatch strided_batch(/*offset=*/0);
  HWY_ASSERT_EQ(size_t{0}, strided_batch.AddStrided(dst.data(), /*rows=*/0,
                                                    /*bytes_per_row=*/4,
                                                    /*stride_bytes=*/8));
  HWY_ASSERT_EQ(size_t{3}, strided_batch.AddStrided(dst.data(), /*rows=*/3,
                                                    /*bytes_per_row=*/0,
                                                    /*stride_bytes=*/8));
  HWY_ASSERT(strided_batch.IsEmpty());
  // Empty read/write batches after 0-byte Add/AddStrided succeed with Status(0)
  // rather than aborting.
  HWY_ASSERT(strided_batch.Read(ro_file, io_buf) == Status(0));
  HWY_ASSERT(strided_batch.Write(rw_file) == Status(0));
  HWY_ASSERT(strided_batch.Write(ro_file).Is(IOCode::kWrongMode));

  std::vector<uint8_t> packed(4 * 6, 0);
  HWY_ASSERT_EQ(size_t{4}, strided_batch.AddStrided(packed.data(), /*rows=*/4,
                                                    /*bytes_per_row=*/6,
                                                    /*stride_bytes=*/6));
  HWY_ASSERT_EQ(size_t{1}, strided_batch.Spans().size());
  HWY_ASSERT(strided_batch.Read(ro_file, io_buf) == Status(24));
  HWY_ASSERT_ARRAY_EQ(src.data(), packed.data(), 24);

  constexpr size_t kRows = 5;
  constexpr size_t kRowBytes = 7;
  constexpr size_t kStride = 16;
  std::vector<uint8_t> padded(kRows * kStride, kSentinel);
  strided_batch.Reset(/*offset=*/20);
  HWY_ASSERT_EQ(kRows, strided_batch.AddStrided(padded.data(), kRows, kRowBytes,
                                                kStride));
  HWY_ASSERT_EQ(kRows, strided_batch.Spans().size());
  HWY_ASSERT(strided_batch.Read(ro_file, worker_buf) ==
             Status(kRows * kRowBytes));
  // High-water-mark capacity (60) is retained across smaller (35-byte) reads.
  HWY_ASSERT_EQ(size_t{60}, worker_buf.Capacity());
  IOWorkerBuf moved_buf = std::move(worker_buf);
  HWY_ASSERT_EQ(size_t{60}, moved_buf.Capacity());
  // NOLINTNEXTLINE(bugprone-use-after-move)
  HWY_ASSERT_EQ(size_t{0}, worker_buf.Capacity());
  moved_buf.Release();
  HWY_ASSERT_EQ(size_t{0}, moved_buf.Capacity());
  for (size_t r = 0; r < kRows; ++r) {
    HWY_ASSERT_ARRAY_EQ(src.data() + 20 + r * kRowBytes,
                        padded.data() + r * kStride, kRowBytes);
    for (size_t p = kRowBytes; p < kStride; ++p) {
      HWY_ASSERT_EQ(kSentinel, padded[r * kStride + p]);
    }
  }

  // Gather write from padded rows into a packed output buffer.
  std::vector<uint8_t> gathered(kRows * kRowBytes, 0);
  MemoryFile gather_file(gathered, FileMode::kWriteTruncate);
  IOBatch strided_write(/*offset=*/0);
  // Zero-byte or zero-row write additions keep the batch empty (`Read`/`Write`
  // return `Status(0)`) without marking the batch write-only until bytes are
  // actually added (matching `AddForWrite(empty)`).
  HWY_ASSERT(strided_write.AddForWrite(Span<const uint8_t>()));
  HWY_ASSERT_EQ(size_t{0}, strided_write.AddStridedForWrite(
                               padded.data(), /*rows=*/0, kRowBytes, kStride));
  HWY_ASSERT_EQ(kRows,
                strided_write.AddStridedForWrite(padded.data(), kRows,
                                                 /*bytes_per_row=*/0, kStride));
  HWY_ASSERT(strided_write.IsEmpty());
  HWY_ASSERT(strided_write.Read(ro_file, io_buf) == Status(0));
  HWY_ASSERT(strided_write.Write(gather_file) == Status(0));

  HWY_ASSERT_EQ(kRows, strided_write.AddStridedForWrite(padded.data(), kRows,
                                                        kRowBytes, kStride));
  HWY_ASSERT(strided_write.Read(ro_file, io_buf).Is(IOCode::kWrongMode));
  HWY_ASSERT(strided_write.Write(gather_file) == Status(kRows * kRowBytes));
  HWY_ASSERT_ARRAY_EQ(src.data() + 20, gathered.data(), kRows * kRowBytes);
}

TEST(IOBatchTest, TestIOBatchFirstSpanAlwaysAccepted) {
  IOWorkerBuf io_buf;
  IOBatch batch(/*offset=*/0);
  // Larger than the batch limit, but must still be accepted as the first span.
  // Not read, so the pointer is never dereferenced.
  const size_t kHuge = static_cast<size_t>(
      HWY_MIN(uint64_t{0x80000000}, static_cast<uint64_t>(~size_t{0})));
  HWY_ASSERT(batch.Add(Span<uint8_t>(static_cast<uint8_t*>(nullptr), kHuge)));
  // Subsequent spans exceeding the limit are rejected.
  uint8_t one;
  HWY_ASSERT(!batch.Add(Span<uint8_t>(&one, 1)));

  // More than IOV_MAX discontiguous spans are accepted by Add and chunked by
  // ReadSpans.
#ifdef IOV_MAX
  const size_t kMaxSpans = IOV_MAX;
#else
  const size_t kMaxSpans = 1024;
#endif
  std::vector<uint8_t> src(kMaxSpans + 5);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
  std::vector<uint8_t> discontiguous((kMaxSpans + 5) * 2, 0);
  IOBatch max_spans_batch(/*offset=*/0);
  for (size_t i = 0; i < kMaxSpans + 5; ++i) {
    HWY_ASSERT(max_spans_batch.Add(Span<uint8_t>(&discontiguous[i * 2], 1)));
  }
  MemoryFile src_file(src);
  HWY_ASSERT(max_spans_batch.Read(src_file, io_buf) == Status(kMaxSpans + 5));
  for (size_t i = 0; i < kMaxSpans + 5; ++i) {
    HWY_ASSERT_EQ(src[i], discontiguous[i * 2]);
  }

  // Subsequent spans whose offset + total_bytes would exceed int64_t (off_t)
  // are rejected by Add, and a single span exceeding off_t is rejected by Read.
  const uint64_t kMaxOff = static_cast<uint64_t>(LimitsMax<int64_t>());
  IOBatch overflow_batch(kMaxOff - 1);
  HWY_ASSERT(overflow_batch.Add(Span<uint8_t>(&one, 1)));
  HWY_ASSERT(!overflow_batch.Add(Span<uint8_t>(&one, 1)));
  // A failed AddForWrite does not mark the batch read-only.
  HWY_ASSERT(!overflow_batch.AddForWrite(Span<const uint8_t>(&one, 1)));
  MemoryFile empty_file((Span<const uint8_t>()));
  HWY_ASSERT(overflow_batch.Read(empty_file, io_buf) ==
             Status(Status::kTruncated, 0));
  overflow_batch.Reset(kMaxOff);
  HWY_ASSERT(overflow_batch.Add(Span<uint8_t>(&one, 1)));
  HWY_ASSERT(overflow_batch.Read(empty_file, io_buf).Is(Status::kArgument));
}

}  // namespace
}  // namespace hwy

HWY_TEST_MAIN();
