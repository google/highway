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

#include "hwy/detect_compiler_arch.h"
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

// Whether workers should block or spin.
enum class PoolWaitMode : uint8_t { kBlock = 1, kSpin };

namespace pool {

// Generates a random permutation of [0, size). O(1) storage.
class ShuffledIota {
 public:
  ShuffledIota() : coprime_(1) {}  // for Worker
  explicit ShuffledIota(uint32_t coprime) : coprime_(coprime) {}

  // Returns the next after `current`, using an LCG-like generator.
  uint32_t Next(uint32_t current, const Divisor64& divisor) const {
    HWY_DASSERT(current < divisor.GetDivisor());
    // (coprime * i + current) % size, see https://lemire.me/blog/2017/09/18/.
    return static_cast<uint32_t>(divisor.Remainder(current + coprime_));
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

// 'Policies' suitable for various worker counts and locality.
enum class WaitType : uint8_t { kBlock, kSpin1, kSpinSeparate };
// Never 0, so that `Config::Encode` is always nonzero and thus different from
// the initial value of prev_, so that `UntilWoken` will trigger.
enum class BarrierType : uint8_t { kOrdered = 1, kCounter1, kGroup2, kGroup4 };

// We want predictable struct/class sizes so we can reason about cache lines.
#pragma pack(push, 1)

// Encoding of parameters and sequence number, including exit flag, sent from
// the main thread to workers within a u32 (matches futex.h).
struct alignas(4) Config {  // 4 bytes
  static constexpr size_t kBits = 6;
  // Keeps uppermost bit (exit flag) clear.
  static constexpr uint32_t kSeqMask = (1U << (32 - kBits - 1)) - 1;
  static constexpr uint32_t kExitFlag = kSeqMask + 1;

  Config(SpinType spin_type, BarrierType barrier_type)
      : spin_type(spin_type), barrier_type(barrier_type) {}

  static Config Decode(uint32_t bits) {
    const SpinType spin_type = static_cast<SpinType>((bits >> 3) & 7);
    const BarrierType barrier_type = static_cast<BarrierType>(bits & 7);
    return Config(spin_type, barrier_type);
  }

  uint32_t Encode() const {
    const uint32_t spin_bits = static_cast<uint32_t>(spin_type);
    const uint32_t barrier_bits = static_cast<uint32_t>(barrier_type);
    HWY_DASSERT(spin_bits < 8);
    HWY_DASSERT(barrier_bits < 8);
    const uint32_t bits = (spin_bits << 3) | barrier_bits;
    HWY_DASSERT(0 != bits && bits < (1u << kBits));
    return bits;
  }

  SpinType spin_type;
  // No WaitType because it must be sent to workers *after* `CallWithWait`.
  BarrierType barrier_type;
  HWY_MAYBE_UNUSED uint16_t padding_;
};
static_assert(sizeof(Config) == 4, "");

// All mutable state for workers (threads and main), including the barrier.
class alignas(HWY_ALIGNMENT) Worker {  // HWY_ALIGNMENT bytes
  static constexpr size_t kMaxVictims = 4;

  static constexpr auto kAcq = std::memory_order_acquire;
  static constexpr auto kRel = std::memory_order_release;

 public:
  Worker(size_t worker, const Divisor64& div_workers)
      : worker_(worker),
        num_threads_(static_cast<size_t>(div_workers.GetDivisor() - 1)) {
    HWY_DASSERT(IsAligned(this, HWY_ALIGNMENT));
    const size_t num_workers = num_threads_ + 1;
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

  size_t Index() const { return worker_; }
  Worker* AllWorkers() { return this - worker_; }
  size_t NumThreads() const { return num_threads_; }

  // ------------------------ Task assignment.

  // Called from the main thread.
  void SetRange(const uint64_t begin, const uint64_t end) {
    my_begin_.store(begin, kRel);
    my_end_.store(end, kRel);
  }

  uint64_t MyEnd() const { return my_end_.load(kAcq); }

  hwy::Span<const uint32_t> Victims() const {
    return hwy::Span<const uint32_t>(victims_.data(),
                                     static_cast<size_t>(num_victims_));
  }

  // Returns the next task to execute. If >= MyEnd(), it must be skipped.
  uint64_t WorkerReserveTask() {
    // TODO(janwas): replace with cooperative work-stealing.
    return my_begin_.fetch_add(1, std::memory_order_relaxed);
  }

  // ------------------------ Wait for main thread.

  // Main thread: how the workers are waiting, for purposes of waking them.
  // Workers: how the worker should wait for the main thread.
  WaitType wait_type() const { return wait_type_; }
  void SetWaitType(WaitType wait_type) { wait_type_ = wait_type; }

  // Main thread: ascending (with wrap-around) sequence number for sending to
  // workers, to ensure that the `Waiter` value changes. Workers: passed to
  // their `barrier.WorkerReached`.
  uint32_t seq() const { return seq_; }
  void SetSeq(uint32_t seq) { seq_ = seq; }

  // Single-slot mailbox for `Config`, updated whenever we want the worker
  // thread to wake. For `PoolWaitMode::kBlock`, only the first worker's
  // `Waiter` is used because one futex can wake multiple waiters.
  std::atomic<uint32_t>& Waiter() { return waiter_; }
  void StoreWaiter(uint32_t next) { waiter_.store(next, kRel); }

  // "wait" and "barrier" functions with inlined policies, called by worker
  // threads via `CallWithSpinWait` and `CallWithSpinBarrier`, respectively.
  template <class Spin, class Wait,
            HWY_IF_SAME(decltype(Wait().Type()), WaitType)>
  void operator()(const Spin& spin, const Wait& wait) {
    prev_ = wait.UntilWoken(worker_, num_threads_, AllWorkers(), spin, prev_);
  }

  // ------------------------ Barrier: notify main thread.

  // Holds the spin and barrier types; wait_type is separate.
  Config config() const { return config_; }
  bool DecodeConfig() {
    seq_ = prev_ >> Config::kBits;
    if (seq_ == Config::kExitFlag) return false;
    config_ = Config::Decode(prev_);
    return true;
  }

  // Barrier storage: writing seq() via store-release serves as a "have arrived"
  // flag that does not require resetting.
  std::atomic<uint32_t>& Barrier() { return barrier_; }
  void StoreBarrier(uint32_t epoch) { barrier_.store(epoch, kRel); }

  template <class Spin, class Barrier,
            HWY_IF_SAME(decltype(Barrier().Type()), BarrierType)>
  void operator()(const Spin& spin, const Barrier& barrier) {
    barrier.WorkerReached(worker_, num_threads_, AllWorkers(), seq_, spin);
  }

 private:
  const size_t worker_;
  const size_t num_threads_;

  // Set by SetRange:
  std::atomic<uint64_t> my_begin_;
  std::atomic<uint64_t> my_end_;

  // Use u32 to match futex.h.
  std::atomic<uint32_t> waiter_{0};
  std::atomic<uint32_t> barrier_{0};

  uint32_t num_victims_;  // <= kPoolMaxVictims
  std::array<uint32_t, kMaxVictims> victims_;

  // Initial value guaranteed to be different from any valid `Config`.
  uint32_t prev_ = 0;
  uint32_t seq_ = 0;
  // For workers, this is not used before being overwritten after `UntilWoken`.
  // For the main thread, this is what chooses the barrier type.
  // TODO: connect to autotuner.
  Config config_ = Config(DetectSpin(), BarrierType::kGroup4);
  // Default is to block, because spinning only makes sense when threads are
  // pinned and wake latency is important, so it must explicitly be requested
  // by calling `pool.SetWaitMode`.
  WaitType wait_type_ = WaitType::kBlock;
  HWY_MAYBE_UNUSED uint8_t padding_[HWY_ALIGNMENT - 73 - sizeof(victims_)];
};
static_assert(sizeof(Worker) == HWY_ALIGNMENT, "");

#pragma pack(pop)

// All methods are const because they only use storage in `Worker`, and we
// prefer to pass const-references to empty classes to enable type deduction.

struct WaitBlock {
  WaitType Type() const { return WaitType::kBlock; }

  // Wakes all workers and sends them `next` (from Config).
  void WakeWorkers(size_t /*num_threads*/, Worker* workers,
                   const uint32_t next) const {
    workers[0].StoreWaiter(next);
    WakeAll(workers[0].Waiter());  // futex: expensive syscall
  }

  // Waits until `WakeWorkers(_, _, next)` has been called and returns `next`.
  template <class Spin>
  HWY_POOL_PROFILE uint32_t UntilWoken(size_t /*thread*/,
                                       size_t /*num_threads*/, Worker* workers,
                                       const Spin& /*spin*/,
                                       const uint32_t prev) const {
    return BlockUntilDifferent(prev, workers[0].Waiter());
  }
};

// Single u32.
struct WaitSpin1 {
  WaitType Type() const { return WaitType::kSpin1; }

  void WakeWorkers(size_t /*num_threads*/, Worker* workers,
                   const uint32_t next) const {
    workers[0].StoreWaiter(next);
  }

  template <class Spin>
  HWY_POOL_PROFILE uint32_t UntilWoken(size_t /*thread*/,
                                       size_t /*num_threads*/, Worker* workers,
                                       const Spin& spin,
                                       const uint32_t prev) const {
    const SpinResult result = spin.UntilDifferent(prev, workers[0].Waiter());
    // TODO: store result.reps in stats.
    return result.current;
  }
};

// Separate u32 per thread: more work for the main thread, but fewer reads from
// a shared cache line.
struct WaitSpinSeparate {
  WaitType Type() const { return WaitType::kSpinSeparate; }

  void WakeWorkers(size_t num_threads, Worker* workers,
                   const uint32_t next) const {
    for (size_t thread = 0; thread < num_threads; ++thread) {
      workers[thread].StoreWaiter(next);
    }
  }

  template <class Spin>
  HWY_POOL_PROFILE uint32_t UntilWoken(size_t thread, size_t /*num_threads*/,
                                       Worker* workers, const Spin& spin,
                                       const uint32_t prev) const {
    const SpinResult result =
        spin.UntilDifferent(prev, workers[thread].Waiter());
    // TODO: store result.reps in stats.
    return result.current;
  }
};

// Set flag, spin on each flag.
class BarrierOrdered {
 public:
  BarrierType Type() const { return BarrierType::kOrdered; }

  void Reset(Worker* /*workers*/) const {}

  template <class Spin>
  void WorkerReached(size_t thread, size_t /*num_threads*/, Worker* workers,
                     uint32_t seq, const Spin&) const {
    workers[thread].StoreBarrier(seq);
  }

  template <class Spin>
  void UntilReached(size_t num_threads, Worker* workers, uint32_t seq,
                    const Spin& spin) const {
    for (size_t i = 0; i < num_threads; ++i) {
      (void)spin.UntilEqual(seq, workers[i].Barrier());
    }
  }
};

// Single atomic counter. TODO: remove if not competitive?
class BarrierCounter1 {
 public:
  BarrierType Type() const { return BarrierType::kCounter1; }

  void Reset(Worker* workers) const { workers[0].StoreBarrier(0); }

  template <class Spin>
  void WorkerReached(size_t /*thread*/, size_t /*num_threads*/, Worker* workers,
                     uint32_t /*seq*/, const Spin& /*spin*/) const {
    workers[0].Barrier().fetch_add(1, std::memory_order_acq_rel);
  }

  template <class Spin>
  void UntilReached(size_t num_threads, Worker* workers, uint32_t /*seq*/,
                    const Spin& spin) const {
    (void)spin.UntilEqual(static_cast<uint32_t>(num_threads),
                          workers[0].Barrier());
  }
};

// Leader threads wait for others in the group, loop over leaders.
template <size_t kGroupSize>
class BarrierGroup {
 public:
  BarrierType Type() const {
    return kGroupSize == 2 ? BarrierType::kGroup2 : BarrierType::kGroup4;
  }

  void Reset(Worker* /*workers*/) const {}

  template <class Spin>
  void WorkerReached(size_t thread, size_t num_threads, Worker* workers,
                     uint32_t seq, const Spin& spin) const {
    // Leaders wait for all others in their group before marking themselves.
    if (thread % kGroupSize == 0) {
      for (size_t i = thread + 1; i < HWY_MIN(thread + kGroupSize, num_threads);
           ++i) {
        (void)spin.UntilEqual(seq, workers[i].Barrier());
      }
    }
    workers[thread].StoreBarrier(seq);
  }

  template <class Spin>
  void UntilReached(size_t num_threads, Worker* workers, uint32_t seq,
                    const Spin& spin) const {
    for (size_t i = 0; i < num_threads; i += kGroupSize) {
      (void)spin.UntilEqual(seq, workers[i].Barrier());
    }
  }
};

// C++11 lacks generic lambdas, hence implement composable adapters manually.
// `CallSpin` is provided by spin.h, and common to the Spin+Wait and
// Spin+Barrier paths, hence it is the outermost.
template <class Func>
class FunctorAddBarrier {
 public:
  FunctorAddBarrier(BarrierType barrier_type, Func&& func)
      : func_(std::forward<Func>(func)), barrier_type_(barrier_type) {}

  template <class Spin>
  void operator()(const Spin& spin) {
    switch (barrier_type_) {
      case BarrierType::kOrdered:
        return func_(spin, BarrierOrdered());
      case BarrierType::kCounter1:
        return func_(spin, BarrierCounter1());
      case BarrierType::kGroup2:
        return func_(spin, BarrierGroup<2>());
      case BarrierType::kGroup4:
        return func_(spin, BarrierGroup<4>());
      default:
        HWY_UNREACHABLE;
    }
  }

  template <class Spin, class Wait>
  void operator()(const Spin& spin, const Wait& wait) {
    switch (barrier_type_) {
      case BarrierType::kOrdered:
        return func_(spin, wait, BarrierOrdered());
      case BarrierType::kCounter1:
        return func_(spin, wait, BarrierCounter1());
      case BarrierType::kGroup2:
        return func_(spin, wait, BarrierGroup<2>());
      case BarrierType::kGroup4:
        return func_(spin, wait, BarrierGroup<4>());
      default:
        HWY_UNREACHABLE;
    }
  }

 private:
  Func&& func_;
  BarrierType barrier_type_;
};

template <class Func>
class FunctorAddWait {
 public:
  FunctorAddWait(WaitType wait_type, Func&& func)
      : func_(std::forward<Func>(func)), wait_type_(wait_type) {}

  template <class Spin>
  void operator()(const Spin& spin) {
    switch (wait_type_) {
      case WaitType::kBlock:
        return func_(spin, WaitBlock());
      case WaitType::kSpin1:
        return func_(spin, WaitSpin1());
      case WaitType::kSpinSeparate:
        return func_(spin, WaitSpinSeparate());
      default:
        HWY_UNREACHABLE;
    }
  }

 private:
  Func&& func_;
  WaitType wait_type_;
};

// Called directly for `RequestExit`.
template <class Func>
void CallWithWait(WaitType wait_type, Func&& func) {
  switch (wait_type) {
    case WaitType::kBlock:
      return func(WaitBlock());
    case WaitType::kSpin1:
      return func(WaitSpin1());
    case WaitType::kSpinSeparate:
      return func(WaitSpinSeparate());
    default:
      HWY_UNREACHABLE;
  }
}

// Calls unrolled code based on all 3 enums. Inlining policy classes into the
// main thread logic reduces branching.
template <class Func>
void CallWithSpinWaitBarrier(SpinType spin_type, WaitType wait_type,
                             const BarrierType barrier_type, Func&& func) {
  CallWithSpin(spin_type,
               FunctorAddWait<FunctorAddBarrier<Func>>(
                   wait_type, FunctorAddBarrier<Func>(
                                  barrier_type, std::forward<Func>(func))));
}

// Same but for Spin+Wait and Spin+Barrier, required by worker threads because
// the types may change in between, and we want to dispatch to the new types.
template <class Func>
void CallWithSpinWait(SpinType spin_type, WaitType wait_type, Func&& func) {
  CallWithSpin(spin_type,
               FunctorAddWait<Func>(wait_type, std::forward<Func>(func)));
}

template <class Func>
void CallWithSpinBarrier(SpinType spin_type, BarrierType barrier_type,
                         Func&& func) {
  CallWithSpin(spin_type,
               FunctorAddBarrier<Func>(barrier_type, std::forward<Func>(func)));
}

#pragma pack(push, 1)
// Stores arguments to `Run`: the function and range of task indices. Set by
// the main thread, read by workers including the main thread.
class alignas(8) Tasks {
  static constexpr auto kAcq = std::memory_order_acquire;

  // Signature of the (internal) function called from workers(s) for each
  // `task` in the [`begin`, `end`) passed to Run(). Closures (lambdas) do not
  // receive the first argument, which points to the lambda object.
  typedef void (*RunFunc)(const void* opaque, uint64_t task, size_t worker);

 public:
  Tasks() { HWY_DASSERT(IsAligned(this, 8)); }

  template <class Closure>
  void Set(uint64_t begin, uint64_t end, const Closure& closure) {
    constexpr auto kRel = std::memory_order_release;
    // More than one task, otherwise `Run` would have run the closure directly.
    HWY_DASSERT(begin < end + 1);
    begin_.store(begin, kRel);
    end_.store(end, kRel);
    func_.store(static_cast<RunFunc>(&CallClosure<Closure>), kRel);
    opaque_.store(reinterpret_cast<const void*>(&closure), kRel);
  }

  // Assigns workers their share of `[begin, end)`. Called from the main
  // thread; workers are initializing or spinning for a command.
  static void DivideRangeAmongWorkers(const uint64_t begin, const uint64_t end,
                                      const Divisor64& div_workers,
                                      Worker* workers) {
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
      workers[worker].SetRange(my_begin, my_end);
      my_begin = my_end;
    }
    HWY_DASSERT(my_begin == end);
  }

  // Runs the worker's assigned range of tasks, plus work stealing if needed.
  HWY_POOL_PROFILE void WorkerRun(Worker* worker) {
    if (NumTasks() > worker->NumThreads() + 1) {
      WorkerRunWithStealing(worker);
    } else {
      WorkerRunSingle(worker->Index());
    }
  }

 private:
  // Special case for <= 1 task per worker, where stealing is unnecessary.
  void WorkerRunSingle(size_t worker) const {
    const uint64_t begin = begin_.load(kAcq);
    const uint64_t end = end_.load(kAcq);
    // More than one task, otherwise `Run` would have run the closure directly.
    HWY_DASSERT(begin < end + 1);

    const uint64_t task = begin + worker;
    // We might still have more workers than tasks, so check first.
    if (HWY_LIKELY(task < end)) {
      const void* opaque = Opaque();
      const RunFunc func = Func();
      func(opaque, task, worker);
    }
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
  HWY_POOL_PROFILE void WorkerRunWithStealing(Worker* worker) {
    Worker* workers = worker->AllWorkers();
    const size_t index = worker->Index();
    const RunFunc func = Func();
    const void* opaque = Opaque();

    // For each worker in random order, starting with our own, attempt to do
    // all their work.
    for (uint32_t victim : worker->Victims()) {
      Worker* other_worker = workers + victim;

      // Until all of other_worker's work is done:
      const uint64_t other_end = other_worker->MyEnd();
      for (;;) {
        // The worker that first sets `task` to `other_end` exits this loop.
        // After that, `task` can be incremented up to `num_workers - 1` times,
        // once per other worker.
        const uint64_t task = other_worker->WorkerReserveTask();
        if (HWY_UNLIKELY(task >= other_end)) {
          hwy::Pause();  // Reduce coherency traffic while stealing.
          break;
        }
        // Pass the index we are actually running on; this is important
        // because it is the TLS index for user code.
        func(opaque, task, index);
      }
    }
  }

  size_t NumTasks() const {
    return static_cast<size_t>(end_.load(kAcq) - begin_.load(kAcq));
  }

  const void* Opaque() const { return opaque_.load(kAcq); }
  RunFunc Func() const { return func_.load(kAcq); }

  // Calls closure(task, worker). Signature must match `RunFunc`.
  template <class Closure>
  static void CallClosure(const void* opaque, uint64_t task, size_t worker) {
    (*reinterpret_cast<const Closure*>(opaque))(task, worker);
  }

  std::atomic<uint64_t> begin_;
  std::atomic<uint64_t> end_;
  std::atomic<RunFunc> func_;
  std::atomic<const void*> opaque_;
};
static_assert(sizeof(Tasks) == 16 + 2 * sizeof(void*), "");
#pragma pack(pop)

// Functor called via `CallWithWait`.
class RequestExit {
 public:
  RequestExit(size_t num_threads, Worker* workers)
      : num_threads_(num_threads), workers_(workers) {}

  template <class Wait>
  void operator()(const Wait& wait) {
    const uint32_t next = Config::kExitFlag << Config::kBits;
    wait.WakeWorkers(num_threads_, workers_, next);
  }

 private:
  const size_t num_threads_;
  Worker* const workers_;
};

// Functor called on main thread via `CallWithSpinWaitBarrier`.
class WakeWorkBarrier {
 public:
  WakeWorkBarrier(Worker* main, Tasks* tasks) : main_(main), tasks_(tasks) {}

  template <class Spin, class Wait, class Barrier>
  void operator()(const Spin& spin, const Wait& wait, const Barrier& barrier) {
    barrier.Reset(main_->AllWorkers());

    const uint32_t seq = (main_->seq() + 1) & Config::kSeqMask;
    main_->SetSeq(seq);
    const uint32_t next = main_->config().Encode() | (seq << Config::kBits);
    // Never returns to initial 0 state because `BarrierType` is never 0.
    HWY_DASSERT(next != 0);

    wait.WakeWorkers(main_->NumThreads(), main_->AllWorkers(), next);
    // Workers are either starting up, or waiting for a command. Either way,
    // they will not miss this command, so no need to wait for them here.

    // Also perform work on the main thread before the barrier.
    tasks_->WorkerRun(main_);

    // Wait until all *threads* (not the main thread, because it already knows
    // it is here) called `WorkerReached`. All policies use some form of
    // spinning. Workers threads then wait `UntilWoken`, which serves as the
    // 'release' phase of the barrier.
    barrier.UntilReached(main_->NumThreads(), main_->AllWorkers(), seq, spin);
  }

  Worker* const main_;
  Tasks* const tasks_;
};

// Creates/destroys `Worker` using preallocated storage. See comment at
// `ThreadPool::worker_bytes_` for why we do not dynamically allocate.
class WorkerLifecycle {  // 0 bytes
 public:
  // Placement new for `Worker` into `storage` because its ctor requires
  // the worker index. Returns array of all workers.
  static Worker* Init(uint8_t* storage, const Divisor64& div_workers) {
    Worker* workers = new (storage) Worker(0, div_workers);
    for (size_t worker = 1; worker < div_workers.GetDivisor(); ++worker) {
      new (Addr(storage, worker)) Worker(worker, div_workers);
      // Ensure pointer arithmetic is the same (will be used in Destroy).
      HWY_DASSERT(reinterpret_cast<uintptr_t>(workers + worker) ==
                  reinterpret_cast<uintptr_t>(Addr(storage, worker)));
    }

    // Publish non-atomic stores in `workers`.
    std::atomic_thread_fence(std::memory_order_release);

    return workers;
  }

  static void Destroy(Worker* workers, size_t num_workers) {
    for (size_t worker = 0; worker < num_workers; ++worker) {
      workers[worker].~Worker();
    }
  }

 private:
  static uint8_t* Addr(uint8_t* storage, size_t worker) {
    return storage + worker * sizeof(Worker);
  }
};

}  // namespace pool

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
      : div_workers_(ClampedNumWorkers(num_threads)),
        workers_(pool::WorkerLifecycle::Init(worker_bytes_, div_workers_)),
        main_(workers_ + num_threads) {
    threads_.reserve(num_threads);
    pool::Tasks* tasks = &tasks_;
    for (size_t thread = 0; thread < num_threads; ++thread) {
      pool::Worker* worker = workers_ + thread;
      threads_.emplace_back([worker, tasks]() {
        SetThreadName("worker%03zu", static_cast<int>(worker->Index()));

        // Ensure main thread's writes are visible (synchronizes with fence in
        // `WorkerLifecycle::Init`).
        std::atomic_thread_fence(std::memory_order_acquire);

        // Loop termination is triggered by `~ThreadPool`.
        for (;;) {
          // Choose wait codepath based on the current config, which may change
          // during `WorkerRun`.
          CallWithSpinWait(worker->config().spin_type, worker->wait_type(),
                           *worker);
          if (!worker->DecodeConfig()) break;

          tasks->WorkerRun(worker);

          // Config may now have changed. Reload and call that codepath.
          const pool::Config config = worker->config();
          CallWithSpinBarrier(config.spin_type, config.barrier_type, *worker);
        }
      });
    }

    // No barrier required because `UntilWoken` works even if it is
    // called after `WakeWorkers`, because it checks for a new/different value,
    // and only one value is sent before the next barrier.
  }

  // Waits for all threads to exit.
  ~ThreadPool() {
    // Exit. No barrier required, we wait for them below.
    pool::CallWithWait(wait_type(), pool::RequestExit(NumWorkers(), workers_));

    for (std::thread& thread : threads_) {
      HWY_ASSERT(thread.joinable());
      thread.join();
    }

    pool::WorkerLifecycle::Destroy(workers_, NumWorkers());
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Returns number of Worker, i.e., one more than the largest `worker`
  // argument. Useful for callers that want to allocate thread-local storage.
  size_t NumWorkers() const {
    return static_cast<size_t>(div_workers_.GetDivisor());
  }

  // `mode` defaults to `kBlock`, which means futex. Switching to `kSpin`
  // reduces fork-join overhead especially when there are many calls to `Run`,
  // but wastes power when waiting over long intervals. Must not be called
  // concurrently with any `Run`, because this uses the same waiter/barrier.
  void SetWaitMode(PoolWaitMode mode) {
    // TODO: autotune this.
    pool::WaitType new_wait_type = mode == PoolWaitMode::kBlock
                                       ? pool::WaitType::kBlock
                                       : pool::WaitType::kSpin1;

    if (NumWorkers() == 1) {
      main_->SetWaitType(new_wait_type);
      return;
    }

    SetBusy();

    pool::Worker* workers = workers_;
    // Must be an lvalue so this stays alive until after the barrier.
    const auto closure = [workers, new_wait_type](uint64_t, size_t worker) {
      workers[worker].SetWaitType(new_wait_type);
    };
    // One task per worker to ensure we take the `WorkerRunSingle` path,
    // required because we do not call `DivideRangeAmongWorkers` here.
    tasks_.Set(0, NumWorkers(), closure);
    // Wake workers using the old wait type, before they change it.
    CallWithSpinWaitBarrier(spin_type(), wait_type(), barrier_type(),
                            pool::WakeWorkBarrier(main_, &tasks_));
    // Workers including main_ have set and are using `new_wait_type`.

    ClearBusy();
  }

  // For printing which are in use.
  SpinType spin_type() const { return main_->config().spin_type; }
  pool::WaitType wait_type() const { return main_->wait_type(); }
  pool::BarrierType barrier_type() const {
    return main_->config().barrier_type;
  }

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
    tasks_.Set(begin, end, closure);

    // More than one task per worker: use work stealing.
    if (HWY_LIKELY(num_tasks > num_workers)) {
      pool::Tasks::DivideRangeAmongWorkers(begin, end, div_workers_, workers_);
    }

    CallWithSpinWaitBarrier(spin_type(), wait_type(), barrier_type(),
                            pool::WakeWorkBarrier(main_, &tasks_));
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

  // Debug-only re-entrancy detection.
  void SetBusy() { HWY_DASSERT(!busy_.test_and_set()); }
  void ClearBusy() { HWY_IF_CONSTEXPR(HWY_IS_DEBUG_BUILD) busy_.clear(); }

  const Divisor64 div_workers_;
  pool::Worker* const workers_;  // points into `worker_bytes_`
  pool::Worker* const main_;     // the last entry in `workers_[]`

  // The only mutable state, written by `Run` and read by workers.
  pool::Tasks tasks_;

  // In debug builds, detects if functions are re-entered.
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  // Last because it is large. Store inside ThreadPool so that callers can bind
  // it to the NUMA node's memory. Not stored inside WorkerLifecycle because
  // that class would be initialized after workers_.
  alignas(HWY_ALIGNMENT) uint8_t
      worker_bytes_[sizeof(pool::Worker) * kMaxWorkers];
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_CONTRIB_THREAD_POOL_THREAD_POOL_H_
