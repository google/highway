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

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/bit_set.h"
#include "hwy/cache_control.h"  // FlushStream
#include "hwy/robust_statistics.h"
#include "hwy/timer.h"

namespace hwy {

// Upper bounds for fixed-size data structures (guarded via HWY_DASSERT):
static constexpr size_t kMaxDepth = 64;   // Maximum nesting of zones.
static constexpr size_t kMaxZones = 256;  // Total number of zones.

// How many threads can actually enter a zone (those that don't do not count).
// Memory use is about kMaxThreads * PROFILER_THREAD_STORAGE MiB.
// WARNING: a fiber library can spawn hundreds of threads.
static constexpr size_t kMaxThreads = 256;

// How many mebibytes to allocate (if PROFILER_ENABLED) per thread that
// enters at least one zone. Once this buffer is full, the thread will analyze
// and discard packets, thus temporarily adding some observer overhead.
// Each zone occupies 16 bytes.
#ifndef PROFILER_THREAD_STORAGE
#define PROFILER_THREAD_STORAGE 200ULL
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
static void StreamCacheLine(const uint64_t* HWY_RESTRICT from,
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
  static constexpr size_t kZoneBits = 16;
  static constexpr uint64_t kZoneMask = (1ULL << kZoneBits) - 1;

  // We need full-resolution timestamps; at an effective rate of 5 GHz,
  // this permits 3 minute zone durations. If zone durations are longer than
  // that, split them into multiple zones. Wraparound is handled by masking.
  static constexpr size_t kTimestampBits = 40;
  static constexpr uint64_t kTimestampMask = (1ULL << kTimestampBits) - 1;

  Packet() : bits_(0) {}
  // `zone_id` 0 means exiting the zone.
  Packet(const size_t thread_id, const uint32_t zone_id,
         const uint64_t timestamp) {
    HWY_DASSERT(zone_id <= kZoneMask);

    bits_ = ((thread_id & 0xFF) << kZoneBits) + zone_id;
    bits_ <<= kTimestampBits;
    bits_ |= (timestamp & kTimestampMask);

    HWY_DASSERT(ThreadId() == thread_id);
    HWY_DASSERT(ZoneId() == zone_id);
    HWY_DASSERT(Timestamp() == (timestamp & kTimestampMask));
  }

  size_t ThreadId() const { return bits_ >> (kZoneBits + kTimestampBits); }
  size_t ZoneId() const { return (bits_ >> kTimestampBits) & kZoneMask; }
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

// Holds statistics for all zones with the same `zone_id`.
struct Accumulator {
  Accumulator() : zone_id(kZoneExit), num_calls(0), duration(0) {}
  Accumulator(size_t thread_id, uint32_t zone_id, uint64_t num_calls,
              uint64_t duration)
      : zone_id(zone_id),
        num_calls(static_cast<uint32_t>(num_calls)),
        duration(duration) {
    HWY_DASSERT(zone_id != kZoneExit);  // must be a valid name
    HWY_DASSERT(zone_id <= Packet::kZoneMask);
    HWY_DASSERT(num_calls <= 0xFFFFFFFFu);

    // Capping at 128 enables 32-byte size which is convenient for indexing and
    // copying. This means we overestimate the real time for massively parallel
    // sections, which is fine because we are mainly interested in finding
    // serial sections. Turin CPUs actually have 256 cores, and we will divide
    // the total duration by 128 instead of 256, which is fine because serial
    // code will still be relatively more prominent (only divided by 1).
    if (HWY_LIKELY(thread_id < 128)) {
      threads[thread_id / 64].Set(thread_id % 64);
    }
  }

  void Assimilate(const Accumulator& other) {
    HWY_DASSERT(zone_id == other.zone_id);
    num_calls += other.num_calls;
    duration += other.duration;
    threads[0].SetNonzeroBitsFrom64(other.threads[0].Get64());
    threads[1].SetNonzeroBitsFrom64(other.threads[1].Get64());
  }

  uint32_t zone_id;
  uint32_t num_calls;
  uint64_t duration;
  BitSet64 threads[2];
};
static_assert(sizeof(Accumulator) == 32, "Wrong Accumulator size");

#pragma pack(pop)

static char s_chars[16384];
static std::atomic<uint32_t> s_next{kZoneExit + 1};

static_assert(sizeof(s_chars) <= (1ULL << Packet::kZoneBits),
              "s_chars too large");

HWY_DLLEXPORT uint32_t ProfilerAddZone(const char* name) {
  const size_t len = strlen(name) + 1;
  HWY_ASSERT(s_next + len <= sizeof(s_chars));
  const uint32_t pos = s_next.fetch_add(len, std::memory_order_relaxed);
  HWY_ASSERT(pos != kZoneExit);
  strcpy(s_chars + pos, name);  // NOLINT
  return pos;
}

// Returns a copy of the `name` passed to `ProfilerAddZone` that returned the
// given `zone_id`.
static const char* ZoneName(uint32_t zone_id) {
  HWY_ASSERT(zone_id < sizeof(s_chars));
  return s_chars + zone_id;
}

// Per-thread call graph (stack) and Accumulator for each zone.
class Results {
 public:
  // Used for computing overhead when this thread encounters its first Zone.
  // This has no observable effect apart from increasing "analyze_elapsed_".
  uint64_t ZoneDuration(const Packet* packets) {
    HWY_DASSERT(depth_ == 0);
    HWY_DASSERT(num_zones_ == 0);
    AnalyzePackets(packets, 2);
    const uint64_t duration = zones_[0].duration;
    const uint32_t zone_id = kZoneExit + 1;
    zones_[0] = Accumulator(0, zone_id, 0, 0);
    HWY_DASSERT(depth_ == 0);
    num_zones_ = 0;
    return duration;
  }

  void SetSelfOverhead(const uint64_t self_overhead) {
    self_overhead_ = self_overhead;
  }

  void SetChildOverhead(const uint64_t child_overhead) {
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

      const Accumulator acc(node.packet.ThreadId(), node.packet.ZoneId(),
                            /*num_calls=*/1, self_duration);
      UpdateOrAdd(acc);
      --depth_;

      // Deduct this nested node's time from its parent's self_duration.
      if (depth_ != 0) {
        nodes_[depth_ - 1].child_total += duration + child_overhead_;
      }
    }

    const uint64_t t1 = timer::Stop();
    analyze_elapsed_ += t1 - t0;
  }

  // Incorporates results from another thread. Call after all threads have
  // exited any zones.
  void Assimilate(Results& other) {
    const uint64_t t0 = timer::Start();
    HWY_DASSERT(depth_ == 0);
    HWY_DASSERT(other.depth_ == 0);

    for (size_t i = 0; i < other.num_zones_; ++i) {
      UpdateOrAdd(other.zones_[i]);
    }
    other.num_zones_ = 0;
    const uint64_t t1 = timer::Stop();
    analyze_elapsed_ += t1 - t0 + other.analyze_elapsed_;
  }

  // Single-threaded.
  void Print() {
    const uint64_t t0 = timer::Start();
    MergeDuplicates();

    // Sort by decreasing total (self) cost.
    std::sort(zones_, zones_ + num_zones_, [](Accumulator& a, Accumulator& b) {
      return a.duration > b.duration;
    });

    const double inv_freq = 1.0 / platform::InvariantTicksPerSecond();

    for (size_t i = 0; i < num_zones_; ++i) {
      const Accumulator& z = zones_[i];
      const double duration = static_cast<double>(z.duration);
      const double per_call = z.duration / z.num_calls;
      // Durations are per-CPU, but end to end performance is defined by wall
      // time. Assuming fork-join parallelism, zones are entered by multiple
      // threads concurrently, which means the total number of unique threads is
      // also the degree of concurrency, so we can estimate wall time as CPU
      // time divided by the number of unique threads seen. Note that this can
      // be inaccurate if the same zone is entered by varying number of threads.
      // However, that would be visible as a lower average number of threads
      // for that zone.
      const size_t num_threads = z.threads[0].Count() + z.threads[1].Count();
      printf("%-40s: %10u x %15.0f / %3zu = %9.6f\n", ZoneName(z.zone_id),
             z.num_calls, per_call, num_threads,
             duration * inv_freq / num_threads);
    }
    num_zones_ = 0;

    const uint64_t t1 = timer::Stop();
    analyze_elapsed_ += t1 - t0;
    printf("Total analysis [s]: %f\n",
           static_cast<double>(analyze_elapsed_) * inv_freq);
  }

 private:
  // Updates an existing Accumulator (uniquely identified by zone_id) or
  // adds one if this is the first time this thread analyzed that zone.
  // Uses a self-organizing list data structure, which avoids dynamic memory
  // allocations and is far faster than unordered_map.
  void UpdateOrAdd(const Accumulator& other) {
    // Special case for first zone: (maybe) update, without swapping.
    if (num_zones_ != 0 && zones_[0].zone_id == other.zone_id) {
      zones_[0].Assimilate(other);
      return;
    }

    // Look for a zone with the same offset.
    for (size_t i = 1; i < num_zones_; ++i) {
      if (zones_[i].zone_id == other.zone_id) {
        zones_[i].Assimilate(other);
        // Swap with predecessor (more conservative than move to front,
        // but at least as successful).
        const Accumulator prev = zones_[i - 1];
        zones_[i - 1] = zones_[i];
        zones_[i] = prev;
        return;
      }
    }

    // Not found; create a new Accumulator.
    HWY_DASSERT(num_zones_ < kMaxZones);
    zones_[num_zones_] = other;
    ++num_zones_;
  }

  // Each instantiation of a function template gets its own zone_id. Merge all
  // Accumulator with the same name. An N^2 search for duplicates is fine
  // because we only expect a few dozen zones.
  void MergeDuplicates() {
    for (size_t i = 0; i < num_zones_; ++i) {
      const char* name = ZoneName(zones_[i].zone_id);

      // Add any subsequent duplicates to num_calls and total_duration.
      for (size_t j = i + 1; j < num_zones_;) {
        Accumulator& other = zones_[j];
        if (!strcmp(name, ZoneName(other.zone_id))) {
          zones_[i].Assimilate(other);
          // j was the last zone, so we are done.
          if (j == num_zones_ - 1) break;
          // Replace current zone with the last one, and check it next.
          other = zones_[--num_zones_];
        } else {  // Name differed, try next Accumulator.
          ++j;
        }
      }
    }
  }

  uint64_t analyze_elapsed_ = 0;
  uint64_t self_overhead_ = 0;
  uint64_t child_overhead_ = 0;

  size_t depth_ = 0;      // Number of active zones.
  size_t num_zones_ = 0;  // Number of retired zones.

  alignas(HWY_ALIGNMENT) Node nodes_[kMaxDepth];         // Stack
  alignas(HWY_ALIGNMENT) Accumulator zones_[kMaxZones];  // Self-organizing list
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
            PROFILER_ZONE("Dummy Zone (never shown)");
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
      results_.SetSelfOverhead(self_overhead);
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
          PROFILER_ZONE("Dummy");
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
    results_.SetChildOverhead(child_overhead);
  }

  void WriteEntry(size_t thread_id, uint32_t zone_id,
                  const uint64_t timestamp) {
    Write(Packet(thread_id, zone_id, timestamp));
  }

  void WriteExit(const uint64_t timestamp) {
    Write(Packet(0, kZoneExit, timestamp));
  }

  void AnalyzeRemainingPackets() {
    // Ensures prior weakly-ordered streaming stores are globally visible.
    FlushStream();

    // Storage full => empty it.
    if (num_packets_ + buffer_size_ > max_packets_) {
      results_.AnalyzePackets(packets_.get(), num_packets_);
      num_packets_ = 0;
    }
    CopyBytes(buffer_, packets_.get() + num_packets_,
              buffer_size_ * sizeof(Packet));
    num_packets_ += buffer_size_;

    results_.AnalyzePackets(packets_.get(), num_packets_);
    num_packets_ = 0;
  }

  Results& GetResults() { return results_; }

 private:
  // Write packet to buffer/storage, emptying them as needed.
  void Write(const Packet packet) {
    // Buffer full => copy to storage.
    if (buffer_size_ == kBufferCapacity) {
      // Storage full => empty it.
      if (num_packets_ + kBufferCapacity > max_packets_) {
        results_.AnalyzePackets(packets_.get(), num_packets_);
        num_packets_ = 0;
      }
      // This buffering halves observer overhead and decreases the overall
      // runtime by about 3%. Casting is safe because the first member is u64.
      StreamCacheLine(
          reinterpret_cast<const uint64_t*>(buffer_),
          reinterpret_cast<uint64_t*>(packets_.get() + num_packets_));
      num_packets_ += kBufferCapacity;
      buffer_size_ = 0;
    }
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

HWY_DLLEXPORT Zone::Zone(size_t thread_id, uint32_t zone_id) {
  HWY_FENCE;
  if (HWY_UNLIKELY(s_thread == nullptr)) {
    // Ensure the CPU supports our timer.
    char cpu[100];
    if (!platform::HaveTimerStop(cpu)) {
      HWY_ABORT("CPU %s is too old for PROFILER_ENABLED=1, exiting", cpu);
    }

    const size_t idx = s_num_threads.fetch_add(1, std::memory_order_relaxed);
    HWY_DASSERT(idx < kMaxThreads);
    s_thread = &s_threads[idx];

    // After setting s_thread, because ComputeOverhead re-enters Zone().
    s_thread->ComputeOverhead();
  }

  // (Capture timestamp ASAP, not inside WriteEntry.)
  HWY_FENCE;
  const uint64_t timestamp = timer::Start();
  s_thread->WriteEntry(thread_id, zone_id, timestamp);
}

HWY_DLLEXPORT Zone::~Zone() {
  HWY_FENCE;
  const uint64_t timestamp = timer::Stop();
  s_thread->WriteExit(timestamp);
  HWY_FENCE;
}

HWY_DLLEXPORT void ProfilerPrintResults() {
  const size_t num_threads = s_num_threads.load(std::memory_order_acquire);

  PerThread* main = &s_threads[0];
  main->AnalyzeRemainingPackets();

  for (size_t i = 1; i < num_threads; ++i) {
    PerThread* ts = &s_threads[i];
    ts->AnalyzeRemainingPackets();
    main->GetResults().Assimilate(ts->GetResults());
  }

  if (num_threads != 0) {
    main->GetResults().Print();
  }
}

}  // namespace hwy

#endif  // PROFILER_ENABLED
