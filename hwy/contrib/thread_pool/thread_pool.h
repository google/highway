// Copyright 2023 Google LLC
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

// Modified from BSD-licensed code
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// See https://github.com/libjxl/libjxl/blob/main/LICENSE.

#ifndef HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
#define HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>  // snprintf

#include <array>
#include <atomic>
#include <thread>  // NOLINT
#include <vector>

#if HWY_OS_FREEBSD
#include <pthread_np.h>
#endif

#include "hwy/aligned_allocator.h"  // HWY_ALIGNMENT
#include "hwy/base.h"
#include "hwy/cache_control.h"  // Pause
#include "hwy/contrib/thread_pool/futex.h"
#include "hwy/contrib/thread_pool/spin.h"
#include "hwy/contrib/thread_pool/topology.h"

// Define to HWY_NOINLINE to see profiles of `WorkerRun*` and waits.
#define HWY_POOL_PROFILE

namespace hwy {

// Sets the name of the current thread to the format string `format`, which must
// include %d for `thread`. Currently only implemented for pthreads (*nix and
// OSX); Windows involves throwing an exception.
static inline void SetThreadName(const char* format, int thread) {
  char buf[16] = {};  // Linux limit, including \0
  const int chars_written = snprintf(buf, sizeof(buf), format, thread);
  HWY_ASSERT(0 < chars_written &&
             chars_written <= static_cast<int>(sizeof(buf) - 1));

#if HWY_OS_LINUX && (!defined(__ANDROID__) || __ANDROID_API__ >= 19)
  HWY_ASSERT(0 == pthread_setname_np(pthread_self(), buf));
#elif HWY_OS_FREEBSD
  HWY_ASSERT(0 == pthread_set_name_np(pthread_self(), buf));
#elif HWY_OS_APPLE
  // Different interface: single argument, current thread only.
  HWY_ASSERT(0 == pthread_setname_np(buf));
#endif
}

// Generates a random permutation of [0, size). O(1) storage.
class ShuffledIota {
 public:
  ShuffledIota() : coprime_(1) {}  // for PoolWorker
  explicit ShuffledIota(uint32_t coprime) : coprime_(coprime) {}

  // Returns the next after `current`, using an LCG-like generator.
  uint32_t Next(uint32_t current, const Divisor64& divisor) const {
    HWY_DASSERT(current < divisor.GetDivisor());
    // (coprime * i + current) % size, see https://lemire.me/blog/2017/09/18/.
    return divisor.Remainder(current + coprime_);
  }

  // Returns true if a and b have no common denominator except 1. Based on
  // binary GCD. Assumes a and b are nonzero. Also used in tests.
  static bool CoprimeNonzero(uint32_t a, uint32_t b) {
    const size_t trailing_a = Num0BitsBelowLS1Bit_Nonzero32(a);
    const size_t trailing_b = Num0BitsBelowLS1Bit_Nonzero32(b);
    // If both have at least one trailing zero, they are both divisible by 2.
    if (HWY_MIN(trailing_a, trailing_b) != 0) return false;

    // If one of them has a trailing zero, shift it out.
    a >>= trailing_a;
    b >>= trailing_b;

    for (;;) {
      // Swap such that a >= b.
      const uint32_t tmp_a = a;
      a = HWY_MAX(tmp_a, b);
      b = HWY_MIN(tmp_a, b);

      // When the smaller number is 1, they were coprime.
      if (b == 1) return true;

      a -= b;
      // a == b means there was a common factor, so not coprime.
      if (a == 0) return false;
      a >>= Num0BitsBelowLS1Bit_Nonzero32(a);
    }
  }

  // Returns another coprime >= `start`, or 1 for small `size`.
  // Used to seed independent ShuffledIota instances.
  static uint32_t FindAnotherCoprime(uint32_t size, uint32_t start) {
    if (size <= 2) {
      return 1;
    }

    // Avoids even x for even sizes, which are sure to be rejected.
    const uint32_t inc = (size & 1) ? 1 : 2;

    for (uint32_t x = start | 1; x < start + size * 16; x += inc) {
      if (CoprimeNonzero(x, static_cast<uint32_t>(size))) {
        return x;
      }
    }

    HWY_ABORT("unreachable");
  }

  uint32_t coprime_;
};

// Signature of the (internal) function called from workers(s) for each
// `task` in the [`begin`, `end`) passed to Run(). Closures (lambdas) do not
// receive the first argument, which points to the lambda object.
typedef void (*PoolRunFunc)(const void* opaque, uint64_t task, size_t worker);

// We want predictable struct/class sizes so we can reason about cache lines.
#pragma pack(push, 1)

// Per-worker storage for work stealing and barrier.
class alignas(HWY_ALIGNMENT) PoolWorker {  // HWY_ALIGNMENT bytes
  static constexpr size_t kMaxVictims = 4;

  static constexpr auto kAcq = std::memory_order_acquire;
  static constexpr auto kRel = std::memory_order_release;

 public:
  PoolWorker(size_t worker, const Divisor64& div_workers) {
    HWY_DASSERT(IsAligned(this, HWY_ALIGNMENT));
    const size_t num_workers = div_workers.GetDivisor();
    HWY_DASSERT(worker < num_workers);
    num_victims_ = static_cast<uint32_t>(HWY_MIN(kMaxVictims, num_workers));

    // Increase gap between coprimes to reduce collisions.
    const uint32_t coprime = ShuffledIota::FindAnotherCoprime(
        static_cast<uint32_t>(num_workers),
        static_cast<uint32_t>((worker + 1) * 257 + worker * 13));
    const ShuffledIota shuffled_iota(coprime);

    // To simplify `WorkerRun`, this worker is the first to 'steal' from.
    victims_[0] = static_cast<uint32_t>(worker);
    for (uint32_t i = 1; i < num_victims_; ++i) {
      victims_[i] = shuffled_iota.Next(victims_[i - 1], div_workers);
      HWY_DASSERT(victims_[i] != worker);
    }
  }

  // Assigns workers their share of `[begin, end)`. Called from the main
  // thread; workers are initializing or spinning for a command.
  static void DivideRangeAmongWorkers(const uint64_t begin, const uint64_t end,
                                      const Divisor64& div_workers,
                                      PoolWorker* workers) {
    const size_t num_workers = div_workers.GetDivisor();
    HWY_DASSERT(num_workers > 1);  // Else Run() runs on the main thread.
    HWY_DASSERT(begin <= end);
    const size_t num_tasks = static_cast<size_t>(end - begin);

    // Assigning all remainders to the last worker causes imbalance. We instead
    // give one more to each worker whose index is less. This may be zero when
    // called from `TestTasks`.
    const size_t min_tasks = div_workers.Divide(num_tasks);
    const size_t remainder = div_workers.Remainder(num_tasks);

    uint64_t my_begin = begin;
    for (size_t worker = 0; worker < num_workers; ++worker) {
      const uint64_t my_end = my_begin + min_tasks + (worker < remainder);
      workers[worker].my_begin_.store(my_begin, kRel);
      workers[worker].my_end_.store(my_end, kRel);
      my_begin = my_end;
    }
    HWY_DASSERT(my_begin == end);
  }

  // Must be called for each `worker` in [0, num_workers).
  //
  // A prior version of this code attempted to assign only as much work as a
  // worker will actually use. As with OpenMP's 'guided' strategy, we assigned
  // remaining/(k*num_threads) in each iteration. Although the worst-case
  // imbalance is bounded, this required several rounds of work allocation, and
  // the atomic counter did not scale to > 30 threads.
  //
  // We now use work stealing instead, where already-finished workers look for
  // and perform work from others, as if they were that worker. This deals with
  // imbalances as they arise, but care is required to reduce contention. We
  // randomize the order in which threads choose victims to steal from.
  static HWY_POOL_PROFILE void WorkerRunWithStealing(const size_t worker,
                                                     PoolWorker* workers,
                                                     PoolRunFunc func,
                                                     const void* opaque) {
    // For each worker in random order, attempt to do all their work.
    for (uint32_t victim : workers[worker].Victims()) {
      PoolWorker* other_worker = workers + victim;

      // Until all of other_worker's work is done:
      const uint64_t other_end = other_worker->my_end_.load(kAcq);
      for (;;) {
        // The worker that first sets `task` to `other_end` exits this loop.
        // After that, `task` can be incremented up to `num_workers - 1` times,
        // once per other worker.
        const uint64_t task = other_worker->WorkerReserveTask();
        if (HWY_UNLIKELY(task >= other_end)) {
          hwy::Pause();  // Reduce coherency traffic while stealing.
          break;
        }
        // `worker` is the one we are actually running on; this is important
        // because it is the TLS index for user code.
        func(opaque, task, worker);
      }
    }
  }

  // Barrier storage: an epoch counter can be used as a flag which does not
  // require resetting afterwards. Note that the counter increment is relaxed,
  // and here we only use store-release.
  std::atomic<uint32_t>& Epoch() { return epoch_; }
  void StoreEpoch(uint32_t epoch) { epoch_.store(epoch, kRel); }

 private:
  hwy::Span<const uint32_t> Victims() const {
    return hwy::Span<const uint32_t>(victims_.data(),
                                     static_cast<size_t>(num_victims_));
  }

  // Returns the next task to execute. If >= my_end_, it must be skipped.
  uint64_t WorkerReserveTask() {
    // TODO(janwas): replace with cooperative work-stealing.
    return my_begin_.fetch_add(1, std::memory_order_relaxed);
  }

  // Set by DivideRangeAmongWorkers:
  std::atomic<uint64_t> my_begin_;
  std::atomic<uint64_t> my_end_;

  std::atomic<uint32_t> epoch_{0};

  uint32_t num_victims_;  // <= kPoolMaxVictims
  std::array<uint32_t, kMaxVictims> victims_;

  HWY_MAYBE_UNUSED uint8_t padding_[HWY_ALIGNMENT - 24 - sizeof(victims_)];
};
static_assert(sizeof(PoolWorker) == HWY_ALIGNMENT, "");

// Creates/destroys `PoolWorker` using preallocated storage. See comment at
// `ThreadPool::worker_bytes_` for why we do not dynamically allocate.
class PoolWorkerLifecycle {  // 0 bytes
 public:
  // Placement new for `PoolWorker` into `storage` because its ctor requires
  // the worker index. Returns array of all workers.
  static PoolWorker* Init(uint8_t* storage, const Divisor64& div_workers) {
    PoolWorker* workers = new (storage) PoolWorker(0, div_workers);
    for (size_t worker = 1; worker < div_workers.GetDivisor(); ++worker) {
      new (Worker(storage, worker)) PoolWorker(worker, div_workers);
      // Ensure pointer arithmetic is the same (will be used in Destroy).
      HWY_DASSERT(reinterpret_cast<uintptr_t>(workers + worker) ==
                  reinterpret_cast<uintptr_t>(Worker(storage, worker)));
    }

    // Publish non-atomic stores in `workers` for use by `ThreadFunc`.
    std::atomic_thread_fence(std::memory_order_release);

    return workers;
  }

  static void Destroy(PoolWorker* workers, size_t num_workers) {
    for (size_t worker = 0; worker < num_workers; ++worker) {
      workers[worker].~PoolWorker();
    }
  }

 private:
  static uint8_t* Worker(uint8_t* storage, size_t worker) {
    return storage + worker * sizeof(PoolWorker);
  }
};

// Stores arguments to `Run`: the function and range of task indices. Set by
// the main thread, read by workers including the main thread.
class alignas(8) PoolRange {
  static constexpr auto kAcq = std::memory_order_acquire;

 public:
  PoolRange() { HWY_DASSERT(IsAligned(this, 8)); }

  template <class Closure>
  void Set(uint64_t begin, uint64_t end, const Closure& closure) {
    constexpr auto kRel = std::memory_order_release;
    // More than one task, otherwise `Run` would have run the closure directly.
    HWY_DASSERT(begin < end + 1);
    begin_.store(begin, kRel);
    end_.store(end, kRel);
    func_.store(static_cast<PoolRunFunc>(&CallClosure<Closure>), kRel);
    opaque_.store(reinterpret_cast<const void*>(&closure), kRel);
  }

  size_t NumTasks() const {
    return static_cast<size_t>(end_.load(kAcq) - begin_.load(kAcq));
  }

  // For passing to `PoolWorker::WorkerRunWithStealing`.
  const void* Opaque() const { return opaque_.load(kAcq); }
  PoolRunFunc Func() const { return func_.load(kAcq); }

  // Special case for <= 1 task per worker, where stealing is unnecessary.
  HWY_POOL_PROFILE void WorkerRunSingle(size_t worker) const {
    const uint64_t begin = begin_.load(kAcq);
    const uint64_t end = end_.load(kAcq);
    // More than one task, otherwise `Run` would have run the closure directly.
    HWY_DASSERT(begin < end + 1);

    const uint64_t task = begin + worker;
    // We might still have more workers than tasks, so check first.
    if (HWY_LIKELY(task < end)) {
      const void* opaque = Opaque();
      const PoolRunFunc func = Func();
      func(opaque, task, worker);
    }
  }

 private:
  // Calls closure(task, worker). Signature must match `PoolRunFunc`.
  template <class Closure>
  static void CallClosure(const void* opaque, uint64_t task, size_t worker) {
    (*reinterpret_cast<const Closure*>(opaque))(task, worker);
  }

  std::atomic<uint64_t> begin_;
  std::atomic<uint64_t> end_;
  std::atomic<PoolRunFunc> func_;
  std::atomic<const void*> opaque_;
};
static_assert(sizeof(PoolRange) == 16 + 2 * sizeof(void*), "");

// Differing barrier types: performance varies with worker count and locality.
enum class BarrierType : uint8_t { kOrdered, kCounter1, kGroup2, kGroup4 };

// All methods are const because they only use storage in `PoolWorker` and
// `CallWithBarrier` passes temporaries and we prefer to pass empty classes as
// arguments to enable type deduction.

// Set flag, spin on each flag.
class BarrierOrdered {
 public:
  BarrierType Type() const { return BarrierType::kOrdered; }

  void Reset(PoolWorker* /*workers*/) const {}

  template <class Spin>
  void WorkerNotify(size_t thread, size_t /*num_threads*/, PoolWorker* workers,
                    uint32_t seq, const Spin&) const {
    workers[thread].StoreEpoch(seq);
  }

  template <class Spin>
  void Wait(size_t num_threads, PoolWorker* workers, uint32_t seq,
            const Spin& spin) const {
    for (size_t i = 0; i < num_threads; ++i) {
      spin.UntilEqual(seq, workers[i].Epoch());
    }
  }
};

// Single atomic counter. TODO: remove if not competitive?
class BarrierCounter1 {
 public:
  BarrierType Type() const { return BarrierType::kCounter1; }

  void Reset(PoolWorker* workers) const {
    workers[0].Epoch().store(0, std::memory_order_release);
  }

  template <class Spin>
  void WorkerNotify(size_t thread, size_t /*num_threads*/, PoolWorker* workers,
                    uint32_t seq, const Spin& /*spin*/) const {
    workers[0].Epoch().fetch_add(1, std::memory_order_acq_rel);
  }

  template <class Spin>
  void Wait(size_t num_threads, PoolWorker* workers, uint32_t seq,
            const Spin& spin) const {
    (void)spin.UntilEqual(static_cast<uint32_t>(num_threads),
                          workers[0].Epoch());
  }
};

// Leader threads wait for others in the group, loop over leaders.
template <size_t kGroupSize>
class BarrierGroup {
 public:
  BarrierType Type() const {
    return kGroupSize == 2 ? BarrierType::kGroup2 : BarrierType::kGroup4;
  }

  void Reset(PoolWorker* /*workers*/) const {}

  template <class Spin>
  void WorkerNotify(size_t thread, size_t num_threads, PoolWorker* workers,
                    uint32_t seq, const Spin& spin) const {
    // Leaders wait for all others in their group before marking themselves.
    if (thread % kGroupSize == 0) {
      for (size_t i = thread + 1; i < HWY_MIN(thread + kGroupSize, num_threads);
           ++i) {
        (void)spin.UntilEqual(seq, workers[i].Epoch());
      }
    }
    workers[thread].StoreEpoch(seq);
  }

  template <class Spin>
  void Wait(size_t num_threads, PoolWorker* workers, uint32_t seq,
            const Spin& spin) const {
    for (size_t i = 0; i < num_threads; i += kGroupSize) {
      (void)spin.UntilEqual(seq, workers[i].Epoch());
    }
  }
};

// Whether `PoolWaiter` should block or spin. Encoded into two bits. Always
// nonzero so the first `Wake` changes `current_` from the initial 0, and thus
// triggers `WaitUntilDifferent`.
enum class PoolWaitMode : uint8_t { kBlock = 1, kSpin = 2 };

// Encodes parameters sent from the main thread to workers into 32 bits so they
// fit in `PoolWaiter`, and can be quickly swapped in when autotuning.
struct alignas(8) PoolConfig {  // 8 bytes
  PoolConfig(BarrierType barrier_type, SpinType spin_type)
      : barrier_type(barrier_type), spin_type(spin_type) {}

  explicit PoolConfig(uint32_t bits) {
    seq = bits >> 8;
    spin_type = static_cast<SpinType>((bits >> 5) & 7);
    barrier_type = static_cast<BarrierType>((bits >> 2) & 7);
    wait_mode = static_cast<PoolWaitMode>(bits & 3);
  }

  uint32_t Bits() const {
    const uint32_t spin_bits = static_cast<uint32_t>(spin_type);
    const uint32_t barrier_bits = static_cast<uint32_t>(barrier_type);
    const uint32_t wait_bits = static_cast<uint32_t>(wait_mode);
    HWY_DASSERT(spin_bits < 8);
    HWY_DASSERT(barrier_bits < 8);
    HWY_DASSERT(wait_bits < 4);
    uint32_t bits = seq << 8;
    bits |= spin_bits << 5;
    bits |= barrier_bits << 2;
    bits |= wait_bits;
    return bits;
  }

  uint32_t seq = 0;
  // Default is to block, because spinning only makes sense when threads are
  // pinned and wake latency is important.
  PoolWaitMode wait_mode = PoolWaitMode::kBlock;
  BarrierType barrier_type;
  SpinType spin_type;
  HWY_MAYBE_UNUSED uint8_t padding_;
};
static_assert(sizeof(PoolConfig) == 8, "");

// Single-slot mailbox for `PoolConfig`, updated whenever we want the worker
// threads to wake.
class alignas(8) PoolWaiter {  // 8 bytes
 public:
  PoolWaiter() { HWY_DASSERT(IsAligned(&current_, 8)); }

  // Exit flag. Kept separate from `PoolConfig` so that autotuning does not
  // accidentally overwrite it.
  void RequestStop() { stop_.store(1, std::memory_order_release); }
  bool WantStop() const { return stop_.load(std::memory_order_acquire) != 0; }

  // Updates `main_config.seq`, which the main thread passes to `barrier.Wait`,
  // and wakes all workers by sending them the *new* config, which they also
  // pass to their `barrier.WorkerNotify`. `prev_wait_mode` was the value of
  // `main_config.wait_mode` before any update, which is the way workers are
  // currently waiting. It is the same as `main_config.wait_mode` except during
  // a call to `ThreadPool::SetWaitMode`.
  void Wake(const PoolWaitMode prev_wait_mode, PoolConfig& main_config) {
    ++main_config.seq;
    const uint32_t next = main_config.Bits();
    // Never returns to initial 0 state because `PoolWaitMode` is never 0.
    HWY_DASSERT(next != 0);
    // Store-release is much more efficient than read-modify-write fetch_add on
    // many-core x86 because the latter (even with memory_order_relaxed)
    // involves a LOCK prefix, which interferes with other cores'
    // cache-coherency transactions and drains our core's store buffer.
    current_.store(next, std::memory_order_release);

    // Only call `WakeAll`, which involves an expensive syscall, if workers are
    // *currently* blocking due to the initial or previously sent config.
    if (HWY_UNLIKELY(prev_wait_mode == PoolWaitMode::kBlock)) {
      WakeAll(current_);
    }

    // Workers are either starting up, or waiting for a command. Either way,
    // they will not miss this command, so no need to wait for them here.
  }

  // Waits using `worker_config.wait_mode` until `Wake()` has been called again,
  // then updates `worker_config` and `prev`.
  template <class Spin>
  HWY_POOL_PROFILE void WaitUntilDifferent(PoolConfig& worker_config,
                                           const Spin& spin, uint32_t& prev) {
    if (HWY_LIKELY(worker_config.wait_mode == PoolWaitMode::kSpin)) {
      const SpinResult result = spin.UntilDifferent(prev, current_);
      // TODO: store result.reps in stats.
      prev = result.current;
    } else {
      prev = BlockUntilDifferent(prev, current_);
    }
    worker_config = PoolConfig(prev);
  }

 private:
  // Use u32 to match futex.h. Initializer is the same as in `ThreadFunc`.
  std::atomic<uint32_t> current_{0};
  std::atomic<uint32_t> stop_{0};
};
static_assert(sizeof(PoolWaiter) == 8, "");

#pragma pack(pop)

// State shared by main and worker threads.
class PoolShared {
 public:
  explicit PoolShared(size_t num_threads) : num_threads(num_threads) {}

  // The two functions below inline the barrier and spin policy classes to
  // minimize branching:

  // Called from worker threads' `ThreadFunc`.
  template <class Spin, class Barrier>
  void WorkNotifyWait(const size_t thread, PoolConfig& worker_config,
                      uint32_t& prev, const Spin& spin,
                      const Barrier& barrier) {
    DoWork(thread);

    barrier.WorkerNotify(thread, num_threads, workers, worker_config.seq, spin);

    waiter.WaitUntilDifferent(worker_config, spin, prev);
  }

  // Called on main thread during `ThreadPool::Run()`.
  template <class Spin, class Barrier>
  void WakeWorkWait(PoolWaitMode prev_wait_mode, const Spin& spin,
                    const Barrier& barrier) {
    barrier.Reset(workers);

    waiter.Wake(prev_wait_mode, main_config);

    // Also perform work on the main thread before waiting.
    DoWork(num_threads);

    // Independent of wait mode, spins until all threads called `WorkerNotify`.
    // Note that workers threads then `WaitUntilDifferent` until we later call
    // `Wake()` again, so there is no 'release' phase of the barrier.
    barrier.Wait(num_threads, workers, main_config.seq, spin);
  }

  PoolWorker* workers;  // points inside `ThreadPool::worker_storage_`.
  PoolWaiter waiter;
  // TODO: autotune this.
  PoolConfig main_config = PoolConfig(BarrierType::kGroup4, DetectSpin());
  PoolRange range;

 private:
  void DoWork(size_t worker) {
    if (range.NumTasks() > num_threads + 1) {
      PoolWorker::WorkerRunWithStealing(worker, workers, range.Func(),
                                        range.Opaque());
    } else {
      range.WorkerRunSingle(worker);
    }
  }

  // Passed to `Barrier*::WorkerNotify`. Only threads participate in the
  // barrier; the main thread only calls `Wait`, not `WorkerNotify`.
  const size_t num_threads;
};

// Highly efficient parallel-for, intended for workloads with thousands of
// fork-join regions which consist of calling tasks[t](i) for a few hundred i,
// using dozens of threads.
//
// To reduce scheduling overhead, we assume that tasks are statically known and
// that threads do not schedule new work themselves. This allows us to avoid
// queues and only store a counter plus the current task. The latter is a
// pointer to a lambda function, without the allocation/indirection required for
// std::function.
//
// To reduce fork/join latency, we choose an efficient barrier, optionally
// enable spin-waits via SetWaitMode, and avoid any mutex/lock.
//
// To eliminate false sharing and enable reasoning about cache line traffic, the
// class is aligned and holds all worker state.
//
// For load-balancing, we use work stealing in random order.
class alignas(HWY_ALIGNMENT) ThreadPool {
 public:
  // This typically includes hyperthreads, hence it is a loose upper bound.
  // -1 because these are in addition to the main thread.
  static size_t MaxThreads() {
    LogicalProcessorSet lps;
    // This is OS dependent, but more accurate if available because it takes
    // into account restrictions set by cgroups or numactl/taskset.
    if (GetThreadAffinity(lps)) {
      return lps.Count() - 1;
    }
    return static_cast<size_t>(std::thread::hardware_concurrency() - 1);
  }

  // `num_threads` is the number of *additional* threads to spawn, which should
  // not exceed `MaxThreads()`. Note that the main thread also performs work.
  explicit ThreadPool(size_t num_threads)
      : shared_(num_threads), div_workers_(ClampedNumWorkers(num_threads)) {
    shared_.workers = PoolWorkerLifecycle::Init(worker_bytes_, div_workers_);

    threads_.reserve(num_threads);
    for (size_t thread = 0; thread < num_threads; ++thread) {
      threads_.emplace_back(ThreadFunc(thread, &shared_));
    }

    // No barrier required because `WaitUntilDifferent` works even if it is
    // called after `Wake`, because it checks for a new/different value, and
    // only one value is sent before the next barrier.
  }

  // Waits for all threads to exit.
  ~ThreadPool() {
    // Requests threads exit. No barrier required, we wait for them below.
    shared_.waiter.RequestStop();
    shared_.waiter.Wake(shared_.main_config.wait_mode, shared_.main_config);

    for (std::thread& thread : threads_) {
      HWY_ASSERT(thread.joinable());
      thread.join();
    }

    PoolWorkerLifecycle::Destroy(shared_.workers, NumWorkers());
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Returns number of PoolWorker, i.e., one more than the largest `worker`
  // argument. Useful for callers that want to allocate thread-local storage.
  size_t NumWorkers() const {
    return static_cast<size_t>(div_workers_.GetDivisor());
  }

  // `mode` defaults to `kBlock`, which means futex. Switching to `kSpin`
  // reduces fork-join overhead especially when there are many calls to `Run`,
  // but wastes power when waiting over long intervals. Must not be called
  // concurrently with any `Run`, because this uses the same waiter/barrier.
  void SetWaitMode(PoolWaitMode mode) {
    if (NumWorkers() == 1) {
      shared_.main_config.wait_mode = mode;
      return;
    }

    SetBusy();

    const PoolWaitMode prev_wait_mode = shared_.main_config.wait_mode;
    shared_.main_config.wait_mode = mode;
    // We just want a barrier without duplicating `WakeWorkWait`. Set up a no-op
    // function for it to call. Must be no more tasks than workers because we
    // do not call `DivideRangeAmongWorkers`.
    shared_.range.Set(0, NumWorkers(), [](size_t, size_t) {});
    CallWithSpinAndBarrier(shared_.main_config,
                           WakeWorkWait(prev_wait_mode, shared_));
    // After waking, workers are using the new config for their next wait.

    ClearBusy();
  }

  // For printing which are in use.
  SpinType Spin() const { return shared_.main_config.spin_type; }
  BarrierType Barrier() const { return shared_.main_config.barrier_type; }

  // parallel-for: Runs `closure(task, worker)` on workers for every `task` in
  // `[begin, end)`. Note that the unit of work should be large enough to
  // amortize the function call overhead, but small enough that each worker
  // processes a few tasks. Thus each `task` is usually a loop.
  //
  // Not thread-safe - concurrent parallel-for in the same ThreadPool are
  // forbidden unless `NumWorkers() == 1` or `end <= begin + 1`.
  template <class Closure>
  void Run(uint64_t begin, uint64_t end, const Closure& closure) {
    const size_t num_tasks = static_cast<size_t>(end - begin);
    const size_t num_workers = NumWorkers();

    // If zero or one task, or no extra threads, run on the main thread without
    // setting any member variables, because we may be re-entering Run.
    if (HWY_UNLIKELY(num_tasks <= 1 || num_workers == 1)) {
      for (uint64_t task = begin; task < end; ++task) {
        closure(task, /*worker=*/0);
      }
      return;
    }

    SetBusy();
    shared_.range.Set(begin, end, closure);

    // More than one task per worker: use work stealing.
    if (HWY_LIKELY(num_tasks > num_workers)) {
      PoolWorker::DivideRangeAmongWorkers(begin, end, div_workers_,
                                          shared_.workers);
    }

    CallWithSpinAndBarrier(
        shared_.main_config,
        WakeWorkWait(shared_.main_config.wait_mode, shared_));
    ClearBusy();
  }

  // Can pass this as init_closure when no initialization is needed.
  // DEPRECATED, better to call the Run() overload without the init_closure arg.
  static bool NoInit(size_t /*num_threads*/) { return true; }  // DEPRECATED

  // DEPRECATED equivalent of NumWorkers. Note that this is not the same as the
  // ctor argument because num_threads = 0 has the same effect as 1.
  size_t NumThreads() const { return NumWorkers(); }  // DEPRECATED

  // DEPRECATED prior interface with 32-bit tasks and first calling
  // `init_closure(num_threads)`. Instead, perform any init before this, calling
  // NumWorkers() for an upper bound on the worker index, then call the other
  // overload of Run().
  template <class InitClosure, class RunClosure>
  bool Run(uint64_t begin, uint64_t end, const InitClosure& init_closure,
           const RunClosure& run_closure) {
    if (!init_closure(NumThreads())) return false;
    Run(begin, end, run_closure);
    return true;
  }

 private:
  // Some CPUs already have more than this many threads, but rather than one
  // large pool, we assume applications create multiple pools, ideally per
  // cluster (cores sharing a cache), because this improves locality and barrier
  // latency. In that case, this is a generous upper bound.
  static constexpr size_t kMaxWorkers = 64;

  // Used to initialize ThreadPool::div_workers_ from its ctor argument.
  static size_t ClampedNumWorkers(size_t num_threads) {
    size_t num_workers = num_threads + 1;  // includes main thread
    HWY_ASSERT(num_workers != 0);          // did not overflow

    // Upper bound is required for `worker_bytes_`.
    if (HWY_UNLIKELY(num_workers > kMaxWorkers)) {
      HWY_WARN("ThreadPool: clamping num_workers %zu to %zu.", num_workers,
               kMaxWorkers);
      num_workers = kMaxWorkers;
    }
    return num_workers;
  }

  // Called by `CallWithSpin`; calls `func(spin, barrier)` given `barrier_type`.
  template <class Func>
  class CallWithBarrier {
   public:
    CallWithBarrier(BarrierType barrier_type, Func&& func)
        : func_(std::forward<Func>(func)), barrier_type_(barrier_type) {}

    template <class Spin>
    void operator()(const Spin& spin) const {
      switch (barrier_type_) {
        case BarrierType::kOrdered:
          func_(spin, BarrierOrdered());
          break;
        case hwy::BarrierType::kCounter1:
          func_(spin, BarrierCounter1());
          break;
        case hwy::BarrierType::kGroup2:
          func_(spin, BarrierGroup<2>());
          break;
        case hwy::BarrierType::kGroup4:
          func_(spin, BarrierGroup<4>());
          break;
      }
    }

   private:
    Func func_;
    BarrierType barrier_type_;
  };

  // Double-dispatch: calls `Func::operator()(spin, barrier)`.
  template <class Func>
  static void CallWithSpinAndBarrier(const PoolConfig& config, Func&& func) {
    CallWithSpin(
        config.spin_type,
        CallWithBarrier<Func>(config.barrier_type, std::forward<Func>(func)));
  }

  class alignas(8) ThreadFunc {
   public:
    ThreadFunc(size_t thread, PoolShared* shared)
        : thread_(thread), shared_(shared) {
      HWY_DASSERT(IsAligned(this, 8));
    }

    // Called by std::thread.
    void operator()() {
      SetThreadName("worker%03zu", static_cast<int>(thread_));

      // Ensure main thread's writes are visible (synchronizes with fence in
      // `WorkerLifecycle::Init`).
      std::atomic_thread_fence(std::memory_order_acquire);

      // Wait for the main thread to call `Wake()`. `SpinPause()` is ignored
      // because `worker_config_.wait_mode` is initially `kBlock`. By pulling
      // this out of the loop, we avoid having to skip the work/barrier during
      // the first loop iteration.
      shared_->waiter.WaitUntilDifferent(worker_config_, SpinPause(), prev_);

      // `~ThreadPool` sets the stop flag that exits this loop and thus thread.
      while (!shared_->waiter.WantStop()) {
        // We just woke up. Dispatch each time so that we switch to the barrier
        // and spin types written to `worker_config_` by the other `operator()`.
        CallWithSpinAndBarrier(worker_config_, *this);
      }
    }

    // Called from the other `operator()` via `CallWithSpinAndBarrier`.
    template <class Spin, class Barrier>
    void operator()(const Spin& spin, const Barrier& barrier) {
      shared_->WorkNotifyWait(thread_, worker_config_, prev_, spin, barrier);
    }

   private:
    const size_t thread_;
    PoolShared* const shared_;
    // Safe but ignored defaults: overwritten by `WaitUntilDifferent`.
    PoolConfig worker_config_ =
        PoolConfig(BarrierType::kGroup4, SpinType::kPause);
    // Initial value matters, see `PoolWaitMode`.
    uint32_t prev_ = 0;
  };

  // Adapter template instead of generic lambda so this works in C++11.
  class alignas(8) WakeWorkWait {
   public:
    WakeWorkWait(PoolWaitMode prev_wait_mode, PoolShared& shared)
        : shared_(shared), prev_wait_mode_(prev_wait_mode) {
      HWY_DASSERT(IsAligned(this, 8));
    }

    template <class Spin, class Barrier>
    void operator()(const Spin& spin, const Barrier& barrier) const {
      shared_.WakeWorkWait(prev_wait_mode_, spin, barrier);
    }

   private:
    PoolShared& shared_;
    const PoolWaitMode prev_wait_mode_;
    HWY_MAYBE_UNUSED uint8_t padding_[7];
  };
  static_assert(sizeof(WakeWorkWait) == 16, "");

  // Debug-only re-entrancy detection.
  void SetBusy() { HWY_DASSERT(!busy_.test_and_set()); }
  void ClearBusy() {
    if constexpr (HWY_IS_DEBUG_BUILD) busy_.clear();
  }

  PoolShared shared_;

  Divisor64 div_workers_;

  // In debug builds, detects if functions are re-entered.
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  // Last because it is large. Store inside ThreadPool so that callers can bind
  // it to the NUMA node's memory. Not stored inside WorkerLifecycle because
  // that class would be initialized after workers_.
  alignas(HWY_ALIGNMENT) uint8_t
      worker_bytes_[sizeof(PoolWorker) * kMaxWorkers];
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
