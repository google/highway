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

// IWYU pragma: begin_exports
#include <stddef.h>
#include <stdint.h>

#include <thread>  //NOLINT
// IWYU pragma: end_exports

#include <atomic>
#include <vector>

#include "hwy/aligned_allocator.h"  // HWY_ALIGNMENT
#include "hwy/base.h"
#include "hwy/cache_control.h"  // Pause

namespace hwy {

// We want predictable struct/class sizes so we can reason about cache lines.
#pragma pack(push, 1)

// Precomputation for fast n / divisor and n % divisor, where n is a variable
// and divisor is unchanging but unknown at compile-time.
class Divisor {  // 16 bytes
 public:
  Divisor() = default;  // for PoolWorker
  explicit Divisor(uint32_t divisor) : divisor_(divisor) {
    if (divisor <= 1) return;

    const uint32_t len =
        static_cast<uint32_t>(31 - Num0BitsAboveMS1Bit_Nonzero32(divisor - 1));
    const uint64_t u_hi = (2ULL << len) - divisor;
    const uint32_t q = Truncate((u_hi << 32) / divisor);

    mul_ = q + 1;
    shift1_ = 1;
    shift2_ = len;
  }

  uint32_t GetDivisor() const { return divisor_; }

  // Returns n / divisor_.
  uint32_t Divide(uint32_t n) const {
    const uint64_t mul = mul_;
    const uint32_t t = Truncate((mul * n) >> 32);
    return (t + ((n - t) >> shift1_)) >> shift2_;
  }

  // Returns n % divisor_.
  uint32_t Remainder(uint32_t n) const { return n - (Divide(n) * divisor_); }

 private:
  static uint32_t Truncate(uint64_t x) {
    return static_cast<uint32_t>(x & 0xFFFFFFFFu);
  }

  uint32_t divisor_;
  uint32_t mul_ = 1;
  uint32_t shift1_ = 0;
  uint32_t shift2_ = 0;
};

// Generates a random permutation of [0, size). O(1) storage.
class ShuffledIota {  // 4 bytes
 public:
  ShuffledIota() : coprime_(1) {}  // for PoolWorker
  explicit ShuffledIota(uint32_t coprime) : coprime_(coprime) {}

  // Returns the next after `current`, using an LCG-like generator.
  HWY_INLINE uint32_t Next(uint32_t current, const Divisor& divisor) const {
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

// Worker's private working set.
class PoolWorker {  // HWY_ALIGNMENT bytes
 public:
  // This resides in a variable-length array and only the first instance's
  // ctor/dtor are called. Instead, ThreadFunc/ThreadPool ctor call WorkerInit.
  PoolWorker() = default;
  ~PoolWorker() = default;

  void WorkerInit(size_t thread, size_t num_workers) {
    div_workers_ = Divisor(static_cast<uint32_t>(num_workers));

    // Increase gap between coprimes to reduce collisions.
    const uint32_t coprime = ShuffledIota::FindAnotherCoprime(
        static_cast<uint32_t>(num_workers),
        static_cast<uint32_t>((thread + 1) * 257 + thread * 13));
    shuffled_iota_ = ShuffledIota(coprime);

    (void)align_;  // suppress unused-field warning
    (void)padding_;
  }

  const Divisor& WorkerDivWorkers() const { return div_workers_; }
  const ShuffledIota& WorkerShuffledIota() const { return shuffled_iota_; }

  // Works around invalid codegen on Arm, where begin_ is larger than expected.
#if HWY_ARCH_ARM
  HWY_NOINLINE
#endif
  // Called from main thread in Plan().
  void SetRange(uint64_t begin, uint64_t end) {
    const auto rel = std::memory_order_release;
    begin_.store(begin, rel);
    end_.store(end, rel);
  }

  // Returns the STL-style end of this worker's assigned range.
  uint64_t WorkerGetEnd() const { return end_.load(std::memory_order_acquire); }

  // Returns the next task to execute. If >= WorkerGetEnd(), it must be skipped.
  uint64_t WorkerReserveTask() {
    return begin_.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  Divisor div_workers_;
  ShuffledIota shuffled_iota_;
  uint32_t align_;

  std::atomic<uint64_t> begin_;
  std::atomic<uint64_t> end_;  // only changes during SetRange

  uint8_t padding_[HWY_ALIGNMENT - sizeof(div_workers_) - 8 - 16];
};

// Modified by main thread, shared with all workers.
class PoolTasks {  // 32 bytes
  // Signature of the (internal) function called from workers(s) for each
  // `task` in the [`begin`, `end`) passed to Run(). Closures (lambdas) do not
  // receive the first argument, which points to the lambda object.
  typedef void (*RunFunc)(const void* opaque, uint64_t task, size_t thread_id);

  // Calls closure(task, thread). Signature must match RunFunc.
  template <class Closure>
  static void CallClosure(const void* opaque, uint64_t task, size_t thread) {
    (*reinterpret_cast<const Closure*>(opaque))(task, thread);
  }

 public:
  // Called from main thread in Plan().
  template <class Closure>
  void Store(const Closure& closure, uint64_t begin, uint64_t end) {
    const auto rel = std::memory_order_release;
    func_.store(static_cast<RunFunc>(&CallClosure<Closure>), rel);
    opaque_.store(reinterpret_cast<const void*>(&closure), rel);
    begin_.store(begin, rel);
    end_.store(end, rel);
  }

  RunFunc WorkerGet(uint64_t& begin, uint64_t& end, const void*& opaque) const {
    const auto acq = std::memory_order_acquire;
    begin = begin_.load(acq);
    end = end_.load(acq);
    opaque = opaque_.load(acq);
    return func_.load(acq);
  }

 private:
  std::atomic<RunFunc> func_;
  std::atomic<const void*> opaque_;
  std::atomic<uint64_t> begin_;
  std::atomic<uint64_t> end_;
};

// Modified by main thread, shared with all workers.
class PoolCommands {  // 16 bytes
  static constexpr uint64_t kInitial = 0;
  static constexpr uint64_t kMask = 0xF;  // for command, rest is ABA counter.
  static constexpr size_t kShift = hwy::CeilLog2(kMask);

 public:
  static constexpr uint64_t kStop = 1;
  static constexpr uint64_t kWork = 2;
  // static constexpr uint64_t kBlock = 3;
  // static constexpr uint64_t kUnblock = 4;

  // Workers must initialize their copy to this so that they wait for the first
  // command as intended.
  static uint64_t WorkerInitialSeqCmd() { return kInitial; }

  // Sends `cmd` to all workers.
  void Broadcast(uint64_t cmd) {
    HWY_DASSERT(cmd <= kMask);
    const uint64_t epoch = epoch_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t seq_cmd = (epoch << kShift) | cmd;
    seq_cmd_.store(seq_cmd, std::memory_order_release);
    // Workers are either starting up, or waiting for a command. Either way,
    // they will not miss this command, so no need to wait for them here.
  }

  // Returns the command, i.e., one of the public constants, e.g., kStop.
  uint64_t WorkerSpinUntilNewCommand(uint64_t& prev_seq_cmd) const {
    for (;;) {
      hwy::Pause();
      const uint64_t seq_cmd = seq_cmd_.load(std::memory_order_acquire);
      if (seq_cmd != prev_seq_cmd) {
        prev_seq_cmd = seq_cmd;
        return seq_cmd & kMask;
      }
    }
  }

 private:
  // Counter for ABA-proofing WorkerSpinUntilNewCommand. Stored next to
  // seq_cmd_ because both are written at the same time by the main thread.
  std::atomic<uint64_t> epoch_;
  std::atomic<uint64_t> seq_cmd_{kInitial};
};

// Modified by main thread AND workers.
// TODO(janwas): more scalable tree
class alignas(HWY_ALIGNMENT) PoolBarrier {  // 4 * HWY_ALIGNMENT bytes
  static constexpr size_t kU64PerCacheLine = HWY_ALIGNMENT / sizeof(uint64_t);

 public:
  void Reset() {
    for (size_t i = 0; i < 4; ++i) {
      num_finished_[i * kU64PerCacheLine].store(0, std::memory_order_release);
    }
  }

  void WorkerArrive(size_t thread) {
    const size_t i = (thread & 3);
    num_finished_[i * kU64PerCacheLine].fetch_add(1, std::memory_order_release);
  }

  // Spin until all have called Arrive(). Note that workers spin for a new
  // command, not the barrier itself.
  void WaitAll(size_t num_workers) {
    const auto acq = std::memory_order_acquire;
    for (;;) {
      hwy::Pause();
      const uint64_t sum = num_finished_[0 * kU64PerCacheLine].load(acq) +
                           num_finished_[1 * kU64PerCacheLine].load(acq) +
                           num_finished_[2 * kU64PerCacheLine].load(acq) +
                           num_finished_[3 * kU64PerCacheLine].load(acq);
      if (sum == num_workers) break;
    }
  }

 private:
  // Sharded to reduce contention. Four counters, each in their own cache line.
  std::atomic<uint64_t> num_finished_[4 * kU64PerCacheLine];
};

// All mutable pool and worker state.
struct alignas(HWY_ALIGNMENT) PoolMem {
  PoolTasks tasks;
  PoolCommands commands;
  // barrier is more write-heavy, hence keep in another cache line.
  uint8_t padding[HWY_ALIGNMENT - sizeof(tasks) - sizeof(commands)];

  PoolBarrier barrier;
  static_assert(sizeof(barrier) % HWY_ALIGNMENT == 0, "");

  PoolWorker workers[1];  // variable-length (num_workers)
  static_assert(sizeof(PoolWorker) == HWY_ALIGNMENT, "");
};

// Aligned allocation and initialization of variable-length PoolMem.
class PoolMemOwner {
 public:
  explicit PoolMemOwner(size_t num_workers) {
    const size_t extra = num_workers <= 1 ? 0 : num_workers - 1;
    const size_t size = sizeof(PoolMem) + sizeof(PoolWorker) * extra;
    bytes_ = hwy::AllocateAligned<uint8_t>(size);
    HWY_ASSERT(bytes_);
    alloc_ = new (bytes_.get()) PoolMem();
  }

  ~PoolMemOwner() { alloc_->~PoolMem(); }

  PoolMem* get() const { return alloc_; }

 private:
  // Aligned allocation ensures we do not straddle cache lines.
  hwy::AlignedFreeUniquePtr<uint8_t[]> bytes_;
  PoolMem* alloc_;
};

// Plans and executes parallel-for loops with work-stealing. No synchronization
// because there is no mutable shared state.
class ParallelFor {  // 0 bytes
  // A prior version of this code attempted to assign only as much work as a
  // thread will actually use. As with OpenMP's 'guided' strategy, we assigned
  // remaining/(k*num_threads) in each iteration. Although the worst-case
  // imbalance is bounded, this required several rounds of work allocation, and
  // the atomic counter did not scale to > 30 threads.
  //
  // We now use work stealing instead, where already-finished threads look for
  // and perform work from others, as if they were that thread. This deals with
  // imbalances as they arise, but care is required to reduce contention. We
  // randomize the order in which threads choose victims to steal from.
  //
  // Results: across 10K calls Run(), we observe a mean of 5.1 tasks per
  // thread, and standard deviation 0.67, indicating good load-balance.

 public:
  // Make preparations for workers to later run `closure(i)` for all `i` in
  // `[begin, end)`. Called from the main thread; workers are initializing or
  // spinning for a command. Returns false if there are no tasks or workers.
  template <class Closure>
  static bool Plan(uint64_t begin, uint64_t end, size_t num_workers,
                   const Closure& closure, PoolMem& mem) {
    // If there are no tasks, we are done.
    HWY_DASSERT(begin <= end);
    const size_t num_tasks = static_cast<size_t>(end - begin);
    if (HWY_UNLIKELY(num_tasks == 0)) return false;

    // Store for later retrieval by all workers in WorkerRun. Must happen before
    // the loop below because tests load this.
    mem.tasks.Store(closure, begin, end);

    // If there are no workers, run all tasks already on the main thread without
    // the overhead of planning.
    if (HWY_UNLIKELY(num_workers <= 1)) {
      for (uint64_t task = begin; task < end; ++task) {
        closure(task, /*thread=*/0);
      }
      return false;
    }

    // Assigning all remainders to the last thread causes imbalance. We instead
    // give one more to each thread whose index is less.
    const size_t remainder = num_tasks % num_workers;
    const size_t min_tasks = num_tasks / num_workers;

    uint64_t task = begin;
    for (size_t thread = 0; thread < num_workers; ++thread) {
      const uint64_t my_end = task + min_tasks + (thread < remainder);
      mem.workers[thread].SetRange(task, my_end);
      task = my_end;
    }
    HWY_DASSERT(task == end);
    return true;
  }

  // Must be called for each `thread` in [0, num_workers), but only if
  // Plan returned true.
  static void WorkerRun(size_t thread, size_t num_workers, PoolMem& mem) {
    // Nonzero, otherwise Plan returned false and this should not be called.
    HWY_DASSERT(num_workers != 0);
    HWY_DASSERT(thread < num_workers);

    const PoolTasks& tasks = mem.tasks;
    const Divisor& div_workers = mem.workers[thread].WorkerDivWorkers();
    const ShuffledIota& shuffled_iota =
        mem.workers[thread].WorkerShuffledIota();

    uint64_t begin, end;
    const void* opaque;
    const auto func = tasks.WorkerGet(begin, end, opaque);

    // Special case for <= 1 task per worker - avoid any shared state.
    if (end <= begin + num_workers) {
      const uint64_t task = begin + thread;
      if (task < end) {
        func(opaque, task, thread);
      }
      return;
    }

    // What thread we are actually running on, important for passing to user
    // so they can index into their TLS.
    const size_t hw_thread = thread;

    // For each worker in random order, attempt to do all their work.
    for (size_t i = 0; i < num_workers; ++i) {
      const uint64_t end = mem.workers[thread].WorkerGetEnd();

      // Until all of thread's work is done:
      for (;;) {
        // On x86 this generates a LOCK prefix, but that is only expensive if
        // there is actually contention, which is unlikely because we shard the
        // counters, threads do not quite proceed in lockstep due to memory
        // traffic, and stealing happens in semi-random order.
        uint64_t task = mem.workers[thread].WorkerReserveTask();

        // The worker that first sets `task` to `end` exits this loop. After
        // that, `task` can be incremented up to `num_workers - 1` times, once
        // per other worker.
        HWY_DASSERT(task < end + num_workers);

        if (HWY_LIKELY(task >= end)) {
          hwy::Pause();  // Reduce coherency traffic while stealing.
          break;
        }
        func(opaque, task, hw_thread);
      }

      thread = static_cast<size_t>(
          shuffled_iota.Next(static_cast<uint32_t>(thread), div_workers));
      HWY_DASSERT(thread < num_workers);
    }
  }
};

#pragma pack(pop)

class ThreadPool {
  static void ThreadFunc(size_t thread, size_t num_workers, PoolMem* mem) {
    HWY_DASSERT(thread < num_workers);

    mem->workers[thread].WorkerInit(thread, num_workers);
    const PoolCommands& commands = mem->commands;

    uint64_t prev_seq_cmd = PoolCommands::WorkerInitialSeqCmd();

    // Wait for a run/exit command.
    for (;;) {
      const uint64_t command = commands.WorkerSpinUntilNewCommand(prev_seq_cmd);
      switch (command) {
        case PoolCommands::kStop:
          return;  // exits thread
        case PoolCommands::kWork:
          ParallelFor::WorkerRun(thread, num_workers, *mem);
          mem->barrier.WorkerArrive(thread);
          break;
        default:
          HWY_ABORT("Unknown command %zu\n", static_cast<size_t>(command));
      }
    }
  }

 public:
  // This typically includes hyperthreads, hence it is a loose upper bound.
  static size_t MaxThreads() {
    return static_cast<size_t>(std::thread::hardware_concurrency());
  }

  // `num_threads` should not exceed `MaxThreads()`. If `num_threads` <= 1,
  // Run() runs only on the main thread. Otherwise, we launch `num_threads - 1`
  // threads (see below) without waiting for them.
  explicit ThreadPool(size_t num_threads)
      // num_workers_ >= 1 because the main thread also performs work.
      : num_workers_(HWY_MAX(num_threads, size_t{1})), owner_(num_workers_) {
    PoolMem* mem = owner_.get();

    const size_t main_thread = num_workers_ - 1;
    threads_.reserve(main_thread);
    for (size_t thread = 0; thread < main_thread; ++thread) {
      threads_.emplace_back(ThreadFunc, thread, num_workers_, mem);
    }
    // No need to wait because PoolCommands will be received whenever threads
    // become ready.

    // Main thread is also a worker; initialize its working set.
    mem->workers[main_thread].WorkerInit(main_thread, num_workers_);
  }

  // Waits for all threads to exit.
  ~ThreadPool() {
    PoolMem& mem = *owner_.get();
    mem.commands.Broadcast(PoolCommands::kStop);  // requests threads exit

    for (std::thread& thread : threads_) {
      HWY_ASSERT(thread.joinable());
      thread.join();
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Returns number of PoolWorker, i.e., one more than the largest `thread`
  // argument. Useful for callers that want to allocate thread-local storage.
  size_t NumWorkers() const { return num_workers_; }

  // parallel-for: Runs `closure(task, thread)` on worker thread(s) for every
  // `task` in `[begin, end)`. Note that the unit of work should be large
  // enough to amortize the function call overhead, but small enough that each
  // worker processes a few tasks. Thus each `task` is usually a loop.
  //
  // Not thread-safe - concurrent calls to `Run` in the same ThreadPool are
  // forbidden. We check for that in debug builds.
  template <class Closure>
  void Run(uint64_t begin, uint64_t end, const Closure& closure) {
    const size_t num_workers = NumWorkers();
    PoolMem& mem = *owner_.get();

    HWY_DASSERT(depth_.fetch_add(1) == 0);

    if (ParallelFor::Plan(begin, end, num_workers, closure, mem)) {
      mem.barrier.Reset();
      mem.commands.Broadcast(PoolCommands::kWork);

      // Also perform work on main thread instead of busy-waiting.
      const size_t thread = num_workers - 1;
      ParallelFor::WorkerRun(thread, num_workers, mem);
      mem.barrier.WorkerArrive(thread);

      mem.barrier.WaitAll(num_workers);
    }

    HWY_DASSERT(depth_.fetch_add(-1) == 1);
  }

  // Can pass this as init_closure when no initialization is needed.
  // DEPRECATED, better to call the Run() overload without the init_closure arg.
  static bool NoInit(size_t /*num_threads*/) { return true; }  // DEPRECATED

  // DEPRECATED equivalent of NumWorkers. Note that this is not the same as the
  // ctor argument, num_threads = 0 has the same effect as 1.
  size_t NumThreads() const { return num_workers_; }  // DEPRECATED

  // DEPRECATED prior interface with 32-bit tasks and first calling
  // `init_closure(num_threads)`. Instead, perform any init before this, calling
  // NumWorkers() for an upper bound on the thread indices, then call the
  // other overload.
  template <class InitClosure, class RunClosure>
  bool Run(uint64_t begin, uint64_t end, const InitClosure& init_closure,
           const RunClosure& run_closure) {
    if (!init_closure(NumThreads())) return false;
    Run(begin, end, run_closure);
    return true;
  }

  // Only for use in tests.
  PoolMem& InternalMem() const { return *owner_.get(); }

 private:
  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  const size_t num_workers_;  // != threads_.size()

  PoolMemOwner owner_;

#if HWY_IS_DEBUG_BUILD
  std::atomic<int> depth_{0};  // detects if Run is re-entered.
#endif
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
