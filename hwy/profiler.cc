// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hwy/profiler.h"

#include "hwy/detect_compiler_arch.h"

#if PROFILER_ENABLED || HWY_IDE

#include <stdio.h>
#include <string.h>  // strcmp

#include <algorithm>  // std::sort
#include <atomic>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
#include "hwy/cache_control.h"  // FlushStream
#include "hwy/robust_statistics.h"
#include "hwy/timer.h"

namespace hwy {

// Upper bounds for fixed-size data structures (guarded via HWY_DASSERT):
static constexpr size_t kMaxDepth = 32;   // Maximum nesting of zones.
static constexpr size_t kMaxZones = 256;  // Total number of zones.

// How many threads can actually enter a zone (those that don't do not count).
// WARNING: a fiber library can spawn hundreds of threads.
static constexpr size_t kMaxThreads = 256;  // enough for Turin cores

// How many mebibytes to allocate (if PROFILER_ENABLED) per thread that
// enters at least one zone. After every root `ThreadPool::Run`, or in the
// unlikely event the buffer fills before then, the thread will analyze and
// discard packets, thus temporarily adding some observer overhead.
#ifndef PROFILER_THREAD_STORAGE
#define PROFILER_THREAD_STORAGE size_t{4}
#endif

static constexpr bool kPrintOverhead = false;

template <typename T>
inline T ClampedSubtract(const T minuend, const T subtrahend) {
  if (subtrahend > minuend) {
    return 0;
  }
  return minuend - subtrahend;
}

// Overwrites "to" while attempting to bypass the cache (read-for-ownership).
// Both pointers must be aligned.
static HWY_INLINE void StreamCacheLine(const uint64_t* HWY_RESTRICT from,
                                       uint64_t* HWY_RESTRICT to) {
#if HWY_COMPILER_CLANG
  for (size_t i = 0; i < HWY_ALIGNMENT / sizeof(uint64_t); ++i) {
    __builtin_nontemporal_store(from[i], to + i);
  }
#else
  hwy::CopyBytes(from, to, HWY_ALIGNMENT);
#endif
}

#pragma pack(push, 1)

// Special zone_id indicating we are exiting the current zone instead of
// entering a new one.
static constexpr uint32_t kZoneExit = 0;

// Represents zone entry/exit events. Stores a full-resolution timestamp and
// integers identifying the calling thread and zone.
class Packet {
 public:
  static constexpr size_t kThreadBits = CeilLog2(kMaxThreads);
  static constexpr uint64_t kThreadMask = (1ULL << kThreadBits) - 1;

  static constexpr size_t kZoneBits = CeilLog2(kMaxZones);
  static constexpr uint64_t kZoneMask = (1ULL << kZoneBits) - 1;

  // We need full-resolution timestamps; at an effective rate of 5 GHz,
  // this permits ~15 hour zone durations. Wraparound is handled by masking.
  static constexpr size_t kTimestampBits = 64 - kThreadBits - kZoneBits;
  static_assert(kTimestampBits >= 40, "Want at least 3min @ 5 GHz");
  static constexpr uint64_t kTimestampMask = (1ULL << kTimestampBits) - 1;

  Packet() : bits_(0) {}
  Packet(size_t thread_id, uint32_t zone_id, uint64_t timestamp) {
    HWY_DASSERT(thread_id <= kThreadMask);
    HWY_DASSERT(zone_id <= kZoneMask);

    bits_ = (thread_id << kZoneBits) + zone_id;
    bits_ <<= kTimestampBits;
    bits_ |= (timestamp & kTimestampMask);

    HWY_DASSERT(ThreadId() == thread_id);
    HWY_DASSERT(ZoneId() == zone_id);
    HWY_DASSERT(Timestamp() == (timestamp & kTimestampMask));
  }

  size_t ThreadId() const { return bits_ >> (kZoneBits + kTimestampBits); }
  uint32_t ZoneId() const {
    return static_cast<uint32_t>((bits_ >> kTimestampBits) & kZoneMask);
  }
  uint64_t Timestamp() const { return bits_ & kTimestampMask; }

 private:
  uint64_t bits_;
};
static_assert(sizeof(Packet) == 8, "Wrong Packet size");

// Representation of an active zone, stored in a stack. Used to deduct
// child duration from the parent's self time.
struct Node {
  Packet packet;
  uint64_t child_total = 0;
};
static_assert(sizeof(Node) == 16, "Wrong Node size");

#pragma pack(pop)

static const char* s_ptrs[kMaxZones];
static std::atomic<uint32_t> s_next_ptr{kZoneExit + 1};  // will be zone_id
static char s_chars[kMaxZones * 64];
static std::atomic<uint32_t> s_next_char{0};

// Returns a copy of the `name` passed to `ProfilerAddZone` that returned the
// given `zone_id`.
static const char* ZoneName(uint32_t zone_id) {
  HWY_ASSERT(zone_id < kMaxZones);
  return s_ptrs[zone_id];
}

HWY_DLLEXPORT uint32_t ProfilerAddZone(const char* name) {
  // Linear search whether it already exists.
  const uint32_t num_zones = s_next_ptr.load(std::memory_order_relaxed);
  HWY_ASSERT(num_zones < kMaxZones);
  for (uint32_t i = kZoneExit + 1; i < num_zones; ++i) {
    if (!strcmp(s_ptrs[i], name)) return i;
  }

  // Reserve the next `zone_id` (index in `s_ptrs`). `AnalyzePackets` relies on
  // these being contiguous.
  const uint32_t zone_id = s_next_ptr.fetch_add(1, std::memory_order_relaxed);
  HWY_ASSERT(zone_id != kZoneExit);

  // Copy into `name` into `s_chars`.
  const size_t len = strlen(name) + 1;
  const uint64_t pos = s_next_char.fetch_add(len, std::memory_order_relaxed);
  HWY_ASSERT(pos + len <= sizeof(s_chars));
  strcpy(s_chars + pos, name);  // NOLINT

  s_ptrs[zone_id] = s_chars + pos;
  HWY_DASSERT(!strcmp(ZoneName(zone_id), name));
  return zone_id;
}

// Holds number of calls and duration for a zone, plus which threads entered it.
// Stored in an array indexed by `zone_id`.
struct Accumulator {
  Accumulator() : zone_id(kZoneExit), num_calls(0), duration(0) {}

  void Add(size_t thread_id, uint32_t new_zone_id, uint64_t self_duration) {
    HWY_DASSERT(thread_id < kMaxThreads);
    HWY_DASSERT(zone_id == kZoneExit || zone_id == new_zone_id);
    zone_id = new_zone_id;
    num_calls += 1;
    duration += self_duration;
    threads[thread_id / 64].Set(thread_id % 64);  // index into BitSet64
  }

  void Assimilate(Accumulator& other) {
    // Not called if `other` was not visited, but `*this` may not have been
    // visited yet, hence set its zone_id to the other's.
    HWY_DASSERT(zone_id == other.zone_id || zone_id == kZoneExit);
    zone_id = other.zone_id;

    num_calls += other.num_calls;
    other.num_calls = 0;

    duration += other.duration;
    other.duration = 0;

    for (size_t i = 0; i < DivCeil(kMaxThreads, 64); ++i) {
      threads[i].SetNonzeroBitsFrom64(other.threads[i].Get64());
      other.threads[i] = BitSet64();
    }
  }

  uint32_t zone_id;
  uint32_t num_calls;
  uint64_t duration;
  BitSet64 threads[DivCeil(kMaxThreads, 64)];
};

// Reduced version of `hwy::Stats`: we are not interested in the geomean, nor
// the variance/kurtosis/skewness. Because concurrency is a small integer, we
// can simply compute sums rather than online moments.
class ConcurrencyStats {
 public:
  ConcurrencyStats() { Reset(); }

  void Notify(const size_t x) {
    ++n_;
    min_ = HWY_MIN(min_, x);
    max_ = HWY_MAX(max_, x);
    sum_ += x;
  }

  void Assimilate(const ConcurrencyStats& other) {
    n_ += other.n_;
    min_ = HWY_MIN(min_, other.min_);
    max_ = HWY_MAX(max_, other.max_);
    sum_ += other.sum_;
  }

  size_t Count() const { return n_; }
  size_t Min() const { return min_; }
  size_t Max() const { return max_; }
  double Mean() const { return static_cast<double>(sum_) / n_; }

  void Reset() {
    n_ = 0;
    min_ = hwy::HighestValue<size_t>();
    max_ = hwy::LowestValue<size_t>();
    sum_ = 0;
  }

 private:
  size_t n_;
  size_t min_;
  size_t max_;
  uint64_t sum_;
};

// Global, only updated by the main thread after the root `ThreadPool::Run` and
// during `ProfilerPrintResults`.
static ConcurrencyStats s_concurrency[kMaxZones];

using ZoneSet = BitSet4096<kMaxZones>;

// Per-thread call graph (stack) and Accumulator for each zone.
class Results {
 public:
  // Used for computing overhead when this thread encounters its first Zone.
  // This has no observable effect apart from increasing "analyze_elapsed_".
  uint64_t ZoneDuration(const Packet* packets) {
    HWY_DASSERT(depth_ == 0);
    AnalyzePackets(packets, 2);
    HWY_DASSERT(depth_ == 0);

    // Extract duration and reset the zone. ComputeOverhead created the second,
    // though the first user Zone from which we are called has not yet exited.
    HWY_DASSERT(visited_zones_.Count() == 1);
    const uint32_t zone_id = visited_zones_.First();
    HWY_DASSERT(zone_id == 2);
    HWY_DASSERT(visited_zones_.Get(zone_id));
    const uint64_t duration = zones_[zone_id].duration;
    zones_[zone_id] = Accumulator();
    visited_zones_.Clear(zone_id);

    return duration;
  }

  void SetOverheads(uint64_t self_overhead, uint64_t child_overhead) {
    self_overhead_ = self_overhead;
    child_overhead_ = child_overhead;
  }

  // Draw all required information from the packets, which can be discarded
  // afterwards. Called whenever this thread's storage is full.
  void AnalyzePackets(const Packet* packets, const size_t num_packets) {
    const uint64_t t0 = timer::Start();

    for (size_t i = 0; i < num_packets; ++i) {
      const Packet p = packets[i];
      // Entering a zone
      if (p.ZoneId() != kZoneExit) {
        HWY_DASSERT(depth_ < kMaxDepth);
        nodes_[depth_].packet = p;
        nodes_[depth_].child_total = 0;
        ++depth_;
        continue;
      }

      HWY_DASSERT(depth_ != 0);
      const Node& node = nodes_[depth_ - 1];
      // Masking correctly handles unsigned wraparound.
      const uint64_t duration =
          (p.Timestamp() - node.packet.Timestamp()) & Packet::kTimestampMask;
      const uint64_t self_duration = ClampedSubtract(
          duration, self_overhead_ + child_overhead_ + node.child_total);

      const uint32_t zone_id = node.packet.ZoneId();
      zones_[zone_id].Add(node.packet.ThreadId(), zone_id, self_duration);
      // For faster Assimilate() - this is usually sparse.
      visited_zones_.Set(zone_id);
      --depth_;

      // Deduct this nested node's time from its parent's self_duration.
      if (HWY_LIKELY(depth_ != 0)) {
        nodes_[depth_ - 1].child_total += duration + child_overhead_;
      }
    }

    const uint64_t t1 = timer::Stop();
    analyze_elapsed_ += t1 - t0;
  }

  // Incorporates results from another thread and resets its accumulators.
  // Must be called from the main thread.
  void Assimilate(Results& other) {
    const uint64_t t0 = timer::Start();

    other.visited_zones_.Foreach([&](size_t zone_id) {
      zones_[zone_id].Assimilate(other.zones_[zone_id]);
      visited_zones_.Set(zone_id);
      other.zones_[zone_id] = Accumulator();
    });
    // OK to reset even if `other` still has active zones, because we set
    // `visited_zones_` when exiting the zone.
    other.visited_zones_ = ZoneSet();

    const uint64_t t1 = timer::Stop();
    analyze_elapsed_ += t1 - t0 + other.analyze_elapsed_;
    other.analyze_elapsed_ = 0;
  }

  // Adds the number of unique `thread_id` as a data point to `s_concurrency`,
  // then resets the `thread_id` bitset.
  void UpdateConcurrency(uint32_t zone_id) {
    Accumulator& z = zones_[zone_id];
    HWY_DASSERT(z.zone_id == zone_id);

    size_t total_threads = 0;
    for (size_t i = 0; i < DivCeil(kMaxThreads, 64); ++i) {
      total_threads += z.threads[i].Count();
      z.threads[i] = BitSet64();
    }
    if (HWY_LIKELY(total_threads != 0)) {
      s_concurrency[z.zone_id].Notify(total_threads);
    }
  }

  void UpdateAllConcurrency() {
    visited_zones_.Foreach([&](size_t zone_id) {
      UpdateConcurrency(static_cast<uint32_t>(zone_id));
    });
  }

  // Single-threaded.
  void Print() {
    const uint64_t t0 = timer::Start();
    const double inv_freq = 1.0 / platform::InvariantTicksPerSecond();

    // Sort by decreasing total (self) cost. `zones_` are sparse, so sort an
    // index vector instead.
    std::vector<uint32_t> indices;
    indices.reserve(visited_zones_.Count());
    visited_zones_.Foreach([&](size_t zone_id) {
      indices.push_back(static_cast<uint32_t>(zone_id));
      // In case the zone exited after ProfilerEndRootRun.
      UpdateConcurrency(static_cast<uint32_t>(zone_id));
    });
    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
      return zones_[a].duration > zones_[b].duration;
    });

    for (uint32_t zone_id : indices) {
      Accumulator& z = zones_[zone_id];  // cleared after printing
      HWY_DASSERT(z.zone_id == zone_id);
      HWY_DASSERT(z.num_calls != 0);  // otherwise visited_zones_ is wrong

      ConcurrencyStats& concurrency = s_concurrency[zone_id];

      const double duration = static_cast<double>(z.duration);
      const double per_call = z.duration / z.num_calls;
      // Durations are per-CPU, but end to end performance is defined by wall
      // time. Assuming fork-join parallelism, zones are entered by multiple
      // threads concurrently, which means the total number of unique threads is
      // also the degree of concurrency, so we can estimate wall time as CPU
      // time divided by the number of unique threads seen. Varying thread
      // counts per call site are supported because `ProfilerEndRootRun`
      // calls UpdateAllConcurrency() after each top-level `ThreadPool::Run`.
      const double num_threads = HWY_MAX(1.0, concurrency.Mean());
      printf("%-40s: %10.0f x %15.0f / %5.1f (%5zu %3zu-%3zu) = %9.6f\n",
             ZoneName(zone_id), static_cast<double>(z.num_calls), per_call,
             num_threads, concurrency.Count(), concurrency.Min(),
             concurrency.Max(), duration * inv_freq / num_threads);

      z = Accumulator();
      concurrency.Reset();
    }
    visited_zones_ = ZoneSet();

    const uint64_t t1 = timer::Stop();
    analyze_elapsed_ += t1 - t0;
    printf("Total analysis [s]: %f\n",
           static_cast<double>(analyze_elapsed_) * inv_freq);
    analyze_elapsed_ = 0;
  }

 private:
  uint64_t analyze_elapsed_ = 0;
  uint64_t self_overhead_ = 0;
  uint64_t child_overhead_ = 0;

  size_t depth_ = 0;       // Number of currently active zones in `nodes_`.
  ZoneSet visited_zones_;  // Which `zones_` have been active on this thread.

  alignas(HWY_ALIGNMENT) Node nodes_[kMaxDepth];         // Stack
  alignas(HWY_ALIGNMENT) Accumulator zones_[kMaxZones];
};

class PerThread {
  static constexpr size_t kBufferCapacity = HWY_ALIGNMENT / sizeof(Packet);

 public:
  PerThread()
      : max_packets_((PROFILER_THREAD_STORAGE << 20) / sizeof(Packet)),
        packets_(AllocateAligned<Packet>(max_packets_)),
        num_packets_(0) {}

  void ComputeOverhead() {
    // Delay after capturing timestamps before/after the actual zone runs. Even
    // with frequency throttling disabled, this has a multimodal distribution,
    // including 32, 34, 48, 52, 59, 62.
    uint64_t self_overhead;
    {
      const size_t kNumSamples = 32;
      uint32_t samples[kNumSamples];
      for (size_t idx_sample = 0; idx_sample < kNumSamples; ++idx_sample) {
        const size_t kNumDurations = 1024;
        uint32_t durations[kNumDurations];

        for (size_t idx_duration = 0; idx_duration < kNumDurations;
             ++idx_duration) {
          {
            PROFILER_ZONE("internal Zone (never shown)");
          }
          const uint64_t duration = results_.ZoneDuration(buffer_);
          buffer_size_ = 0;
          durations[idx_duration] = static_cast<uint32_t>(duration);
          HWY_DASSERT(num_packets_ == 0);
        }
        robust_statistics::CountingSort(durations, kNumDurations);
        samples[idx_sample] = robust_statistics::Mode(durations, kNumDurations);
      }
      // Median.
      robust_statistics::CountingSort(samples, kNumSamples);
      self_overhead = samples[kNumSamples / 2];
      HWY_IF_CONSTEXPR(kPrintOverhead) {
        printf("Overhead: %.0f\n", static_cast<double>(self_overhead));
      }
    }

    // Delay before capturing start timestamp / after end timestamp.
    const size_t kNumSamples = 32;
    uint32_t samples[kNumSamples];
    for (size_t idx_sample = 0; idx_sample < kNumSamples; ++idx_sample) {
      const size_t kNumDurations = 16;
      uint32_t durations[kNumDurations];
      for (size_t idx_duration = 0; idx_duration < kNumDurations;
           ++idx_duration) {
        const size_t kReps = 10000;
        // Analysis time should not be included => must fit within buffer.
        HWY_DASSERT(kReps * 2 < max_packets_);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const uint64_t t0 = timer::Start();
        for (size_t i = 0; i < kReps; ++i) {
          PROFILER_ZONE("internal Zone2 (never shown)");
        }
        FlushStream();
        const uint64_t t1 = timer::Stop();
        HWY_DASSERT(num_packets_ + buffer_size_ == kReps * 2);
        buffer_size_ = 0;
        num_packets_ = 0;
        const uint64_t avg_duration = (t1 - t0 + kReps / 2) / kReps;
        durations[idx_duration] =
            static_cast<uint32_t>(ClampedSubtract(avg_duration, self_overhead));
      }
      robust_statistics::CountingSort(durations, kNumDurations);
      samples[idx_sample] = robust_statistics::Mode(durations, kNumDurations);
    }
    robust_statistics::CountingSort(samples, kNumSamples);
    const uint64_t child_overhead = samples[9 * kNumSamples / 10];
    HWY_IF_CONSTEXPR(kPrintOverhead) {
      printf("Child overhead: %.0f\n", static_cast<double>(child_overhead));
    }
    results_.SetOverheads(self_overhead, child_overhead);
  }

  HWY_INLINE void WriteEntry(const size_t thread_id, const uint32_t zone_id,
                             const uint64_t timestamp) {
    Write(Packet(thread_id, zone_id, timestamp));
  }

  HWY_INLINE void WriteExit(const uint64_t timestamp) {
    Write(Packet(0, kZoneExit, timestamp));
  }

  void AnalyzeRemainingPackets() {
    // Ensures prior weakly-ordered streaming stores are globally visible.
    FlushStream();

    // Storage full => empty it.
    if (HWY_UNLIKELY(num_packets_ + buffer_size_ > max_packets_)) {
      results_.AnalyzePackets(packets_.get(), num_packets_);
      num_packets_ = 0;
    }
    CopyBytes(buffer_, packets_.get() + num_packets_,
              buffer_size_ * sizeof(Packet));
    num_packets_ += buffer_size_;
    buffer_size_ = 0;

    results_.AnalyzePackets(packets_.get(), num_packets_);
    num_packets_ = 0;
  }

  Results& GetResults() { return results_; }

 private:
  HWY_NOINLINE void FlushBuffer() {
    // Storage full => empty it.
    if (HWY_UNLIKELY(num_packets_ + kBufferCapacity > max_packets_)) {
      results_.AnalyzePackets(packets_.get(), num_packets_);
      num_packets_ = 0;
    }
    // Copy buffer to storage without polluting caches.
    // This buffering halves observer overhead and decreases the overall
    // runtime by about 3%. Casting is safe because the first member is u64.
    StreamCacheLine(reinterpret_cast<const uint64_t*>(buffer_),
                    reinterpret_cast<uint64_t*>(packets_.get() + num_packets_));
    num_packets_ += kBufferCapacity;
    buffer_size_ = 0;
  }

  // Write packet to buffer/storage, emptying them as needed.
  HWY_INLINE void Write(const Packet packet) {
    if (HWY_UNLIKELY(buffer_size_ == kBufferCapacity)) FlushBuffer();
    buffer_[buffer_size_] = packet;
    ++buffer_size_;
  }

  // Write-combining buffer to avoid cache pollution. Must be the first
  // non-static member to ensure cache-line alignment.
  Packet buffer_[kBufferCapacity];
  size_t buffer_size_ = 0;

  const size_t max_packets_;
  // Contiguous storage for zone enter/exit packets.
  AlignedFreeUniquePtr<Packet[]> packets_;
  size_t num_packets_;
  Results results_;
  HWY_MAYBE_UNUSED uint8_t padding_[HWY_ALIGNMENT];
};

alignas(HWY_ALIGNMENT) static PerThread s_threads[kMaxThreads];
std::atomic<size_t> s_num_threads{0};
static thread_local PerThread* HWY_RESTRICT s_thread;

// Returns false in the unlikely event that no zone has been entered yet,
// which means `PerThread` has not yet been initialized.
static bool AssimilateResultsIntoMainThread() {
  const size_t num_threads = s_num_threads.load(std::memory_order_acquire);
  if (HWY_UNLIKELY(num_threads == 0)) return false;

  PerThread* main = &s_threads[0];
  main->AnalyzeRemainingPackets();

  for (size_t i = 1; i < num_threads; ++i) {
    PerThread* thread = &s_threads[i];
    thread->AnalyzeRemainingPackets();
    main->GetResults().Assimilate(thread->GetResults());
  }
  return true;
}

// We want to report the concurrency of each separate 'invocation' of a zone.
// A unique per-call identifier (could be approximated with the line number and
// return address) is not sufficient because the caller may in turn be called
// from differing parallel sections. A per-`ThreadPool::Run` counter also
// under-reports concurrency because each pool in nested parallelism (over
// packages and CCXes) would be considered separate invocations.
//
// The alternative of detecting overlapping zones via timestamps is not 100%
// reliable because timers may not be synchronized across sockets or perhaps
// even cores. "Invariant" x86 TSCs are indeed synchronized across cores, but
// not across sockets unless the RESET# signal reaches each at the same time.
// Linux seems to make an effort to correct this, and Arm's "generic timer"
// broadcasts to "all cores", but there is no universal guarantee.
//
// Under the assumption that all concurrency is via our `ThreadPool`, we can
// record all `thread_id` for each outermost (root) `ThreadPool::Run`. This
// collapses all nested pools into one 'invocation'. We then compute per-zone
// concurrency as the number of unique `thread_id` seen.
static std::atomic_flag s_run_active = ATOMIC_FLAG_INIT;

HWY_DLLEXPORT bool ProfilerIsRootRun() {
  // We are not the root if a Run was already active.
  return !s_run_active.test_and_set(std::memory_order_acquire);
}

HWY_DLLEXPORT void ProfilerEndRootRun() {
  // We know that only the main thread is running, so this is thread-safe, but
  // but some zones may still be active. Their concurrency will be updated when
  // Print() is called.

  if (HWY_LIKELY(AssimilateResultsIntoMainThread())) {
    s_thread[0].GetResults().UpdateAllConcurrency();
  }

  s_run_active.clear(std::memory_order_release);
}

static HWY_NOINLINE void InitThread() {
  // Ensure the CPU supports our timer.
  char cpu[100];
  if (HWY_UNLIKELY(!platform::HaveTimerStop(cpu))) {
    HWY_ABORT("CPU %s is too old for PROFILER_ENABLED=1, exiting", cpu);
  }

  const size_t idx = s_num_threads.fetch_add(1, std::memory_order_relaxed);
  HWY_DASSERT(idx < kMaxThreads);
  s_thread = &s_threads[idx];

  // After setting s_thread, because ComputeOverhead re-enters Zone().
  s_thread->ComputeOverhead();
}

HWY_DLLEXPORT Zone::Zone(size_t thread_id, uint32_t zone_id) {
  if (HWY_UNLIKELY(s_thread == nullptr)) InitThread();

  // (Capture timestamp ASAP, not inside WriteEntry.)
  HWY_FENCE;
  const uint64_t timestamp = timer::Start();
  s_thread->WriteEntry(thread_id, zone_id, timestamp);
  HWY_FENCE;
}

HWY_DLLEXPORT Zone::~Zone() {
  HWY_FENCE;
  const uint64_t timestamp = timer::Stop();
  s_thread->WriteExit(timestamp);
  HWY_FENCE;
}

HWY_DLLEXPORT void ProfilerPrintResults() {
  if (AssimilateResultsIntoMainThread()) {
    s_threads[0].GetResults().Print();
  }
}

}  // namespace hwy

#endif  // PROFILER_ENABLED
