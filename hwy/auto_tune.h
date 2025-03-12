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

#ifndef HIGHWAY_HWY_AUTO_TUNE_H_
#define HIGHWAY_HWY_AUTO_TUNE_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>  // memmove

#include <algorithm>  // std::sort
#include <cmath>
#include <vector>

#include "hwy/aligned_allocator.h"  // Span
#include "hwy/base.h"               // HWY_MIN

// Currently avoid the dependency.
// #include "hwy/contrib/sort/vqsort.h"

// Infrastructure for auto-tuning (choosing optimal parameters at runtime).

namespace hwy {

// O(1) storage to estimate the central tendency of hundreds of independent
// distributions (one per configuration). The number of samples per distribution
// (`kMinSamples`) varies from few to dozens. We support both by first storing
// values in a buffer, and when full, switching to online variance estimation.
// Modified from `hwy/stats.h`.
class CostDistribution {
 public:
#if !HWY_COMPILER_GCC_ACTUAL
  static constexpr size_t kMaxValues = 10;  // for total size of 128 bytes
#else
  // GCC 13 does `__insertion_sort(__first, __first + int(_S_threshold)`, which
  // raises an array-bounds warning/error.
  // TODO: remove if using VQSort, also the other HWY_COMPILER_GCC_ACTUAL below.
  static constexpr size_t kMaxValues = 16;
#endif

  void Notify(const double x) {
#if SIZE_MAX == 0xFFFFFFFFu
    (void)padding_;
#endif

    if (HWY_UNLIKELY(x < 0.0)) {
      HWY_WARN("Ignoring negative cost %f.", x);
      return;
    }

    // Online phase after filling+computing thresholds.
    if (HWY_LIKELY(IsOnline())) return NotifyOnline(x);

    // Fill phase: store up to `kMaxValues` values.
    values_[n_++] = x;
    HWY_DASSERT(n_ <= kMaxValues);

    // Once, after full: compute thresholds via trimmed sample variance.
    if (HWY_UNLIKELY(n_ == kMaxValues)) {
      RemoveOutliers();
      const double mean = SampleMean();
      const double stddev = std::sqrt(SampleVariance(mean));
      // High tolerance because the distribution is not actually Gaussian, and
      // we trimmed up to *half*, and do not want to reject too many values in
      // the online phase.
      lower_threshold_ = mean - 4.0 * stddev;
      upper_threshold_ = mean + 4.0 * stddev;
      HWY_DASSERT(IsOnline());
      // Feed trimmed buffer into online estimator.
      const size_t trimmed = n_;
      n_ = 0;  // now used for online phase.
      for (size_t i = 0; i < trimmed; ++i) {
        NotifyOnline(values_[i]);
      }
    }
  }

  // Returns an estimate of the true cost, minimizing the impact of noise.
  //
  // Background and observations from time measurements in `thread_pool.h`:
  // - Noise is mostly additive (delays and interruptions), which biases the
  //   mean upwards.
  // - Occasional "lucky shots" (1.2-1.6x lower values) occur, making the
  //   absolute minimum a non-robust estimator.
  // - We aim for O(1) storage because there may be hundreds of instances.
  //
  // Approach:
  // - We trim both high and low outliers. Unlike Winsorization, this entirely
  //   removes their influence, under the assumption that they are just noise.
  //   We ignore values outside 4 * sample stddev (a wide range because the
  //   distribution is not actually Gaussian). The stddev is itself trimmed
  //   using MAD or estimated skewness.
  // - We estimate the central tendency as an (equally) weighted average of the
  //   *trimmed* min (lower bound), mean (central tendency), and mean minus
  //   stddev (similar to a lower confidence bound, to further correct for the
  //   upward bias from additive noise).
  //
  // Empirically, this is observed to correspond to a shoulder point on the
  // sorted distribution, separating a region of rapid increase from the tail
  // likely due to additive noise.
  double EstimateCost() {
    double min = HighestValue<double>();
    double mean = 0.0;
    double stddev = 0.0;
    if (IsOnline()) {
      min = min_;
      mean = m1_;
      HWY_DASSERT(mean >= lower_threshold_);  // Else would have skipped.
      stddev = std::sqrt(OnlineVariance());
    } else {  // Buffer not yet filled, use 2-pass algorithm.
      HWY_DASSERT(n_ < kMaxValues);
      RemoveOutliers();
      for (size_t i = 0; i < n_; ++i) {
        min = HWY_MIN(min, values_[i]);
      }
      mean = SampleMean();
      stddev = std::sqrt(SampleVariance(mean));
    }

    return (min + mean + mean - stddev) * (1.0 / 3);
  }

 private:
  static double Median(const double* sorted, size_t n) {
    HWY_DASSERT(n >= 2);
    if (n & 1) return sorted[n / 2];
    // Even length: average of two middle elements.
    return (sorted[n / 2] + sorted[n / 2 - 1]) * 0.5;
  }

  // If `n_` is large enough, sorts and discards outliers: either via MAD, or if
  // too many values are equal, by trimming according to skewness.
  void RemoveOutliers() {
    if (n_ < 3) return;  // Not enough to discard two.
    HWY_DASSERT(n_ <= kMaxValues);

    // With salt and pepper noise, it can happen that 1/4 of the sample *on
    // either side* is an outlier. Use median absolute deviation to trim, which
    // is robust to almost half of the sample being outliers.
    // VQSort(values_, n_, SortAscending());
    std::sort(values_, values_ + n_);
    const double median = Median(values_, n_);
    double abs_dev[kMaxValues];
    for (size_t i = 0; i < n_; ++i) {
      abs_dev[i] = ScalarAbs(values_[i] - median);
    }
    std::sort(abs_dev, abs_dev + n_);
    const double mad = Median(abs_dev, n_);
    // At least half the sample is equal.
    if (mad == 0.0) {
      // Estimate skewness to decide which side to trim more.
      const double skewness =
          (values_[n_ - 1] - median) - (median - values_[0]);

      const size_t trim = HWY_MAX(n_ / 2, size_t{2});
      const size_t left =
          HWY_MAX(skewness < 0.0 ? trim * 3 / 4 : trim / 4, size_t{1});
      n_ -= trim;
      HWY_DASSERT(n_ >= 1);
      memmove(values_, values_ + left, n_ * sizeof(values_[0]));
      return;
    }

    const double right_threshold = median + 5.0 * mad;
    const double left_threshold = median - 5.0 * mad;
    size_t right = n_ - 1;
    while (values_[right] > right_threshold) --right;
    // Nonzero MAD implies no more than half are equal, so we did not advance
    // beyond the median.
    HWY_DASSERT(right >= n_ / 2);

    size_t left = 0;
    while (left < right && values_[left] < left_threshold) ++left;
    HWY_DASSERT(left <= n_ / 2);
    n_ = right - left + 1;
    memmove(values_, values_ + left, n_ * sizeof(values_[0]));
  }

  double SampleMean() const {
    // Only called in non-online phase, but buffer might not be full.
    HWY_DASSERT(!IsOnline() && 0 != n_ && n_ <= kMaxValues);
    double sum = 0.0;
    for (size_t i = 0; i < n_; ++i) {
      sum += values_[i];
    }
    return sum / static_cast<double>(n_);
  }

  // Unbiased estimators for population variance even for smaller n_.
  double SampleVariance(double sample_mean) const {
    HWY_DASSERT(sample_mean >= 0.0);  // we checked costs are non-negative.
    // Only called in non-online phase, but buffer might not be full.
    HWY_DASSERT(!IsOnline() && 0 != n_ && n_ <= kMaxValues);
    if (HWY_UNLIKELY(n_ == 1)) return 0.0;  // prevent divide-by-zero.
    double sum2 = 0.0;
    for (size_t i = 0; i < n_; ++i) {
      const double d = values_[i] - sample_mean;
      sum2 += d * d;
    }
    return sum2 / static_cast<double>(n_ - 1);
  }

  bool IsOnline() const { return upper_threshold_ >= 0.0; }

  void NotifyOnline(const double x) {
    // Ignore outliers.
    if (HWY_UNLIKELY(x < lower_threshold_ || x > upper_threshold_)) {
      return;
    }

    min_ = HWY_MIN(min_, x);

    // Welford's online variance estimator.
    // https://media.thinkbrg.com/wp-content/uploads/2020/06/19094655/720_720_McCrary_ImplementingAlgorithms_Whitepaper_20151119_WEB.pdf#page=7.09
    ++n_;
    const double n = static_cast<double>(n_);
    const double d = x - m1_;
    const double d_div_n = d / n;
    const double d2n1_div_n = d * (n - 1.0) * d_div_n;
    m1_ += d_div_n;
    m2_ += d2n1_div_n;
  }

  double OnlineVariance() const {
    if (n_ == 1) return m2_;
    return m2_ / static_cast<double>(n_ - 1);
  }

  // If IsOnline(), number of non-skipped values absorbed, otherwise size of
  // `values_` (<= `kMaxValues`).
  size_t n_ = 0;
#if SIZE_MAX == 0xFFFFFFFFu
  uint32_t padding_ = 0;
#endif

  // Online phase: outlier rejection, min, variance
  double lower_threshold_ = -1.0;
  double upper_threshold_ = -1.0;
  double min_ = HighestValue<double>();
  double m1_ = 0.0;
  double m2_ = 0.0;

  double values_[kMaxValues];
};
#if !HWY_COMPILER_GCC_ACTUAL
static_assert(sizeof(CostDistribution) == 128, "");
#endif

// Implements a counter with wrap-around, plus the ability to skip values.
// O(1) time, O(N) space via doubly-linked list of indices.
class NextWithSkip {
 public:
  NextWithSkip() {}
  explicit NextWithSkip(size_t num) {
    links_.reserve(num);
    for (size_t i = 0; i < num; ++i) {
      links_.emplace_back(i, num);
    }
  }

  size_t Next(size_t pos) {
    HWY_DASSERT(pos < links_.size());
    HWY_DASSERT(!links_[pos].IsRemoved());
    return links_[pos].Next();
  }

  // Must not be called for an already skipped position. Ignores an attempt to
  // skip the last remaining position.
  void Skip(size_t pos) {
    HWY_DASSERT(!links_[pos].IsRemoved());  // not already skipped.
    const size_t prev = links_[pos].Prev();
    const size_t next = links_[pos].Next();
    if (prev == pos || next == pos) return;  // last remaining position.
    links_[next].SetPrev(prev);
    links_[prev].SetNext(next);
    links_[pos].Remove();
  }

 private:
  // Combine prev/next into one array to improve locality/reduce allocations.
  class Link {
    // Bit-shifts avoid potentially expensive 16-bit loads. Store `next` at the
    // top and `prev` at the bottom for extraction with a single shift/AND.
    // There may be hundreds of configurations, so 8 bits are not enough.
    static constexpr size_t kBits = 14;
    static constexpr size_t kShift = 32 - kBits;
    static constexpr uint32_t kMaxNum = 1u << kBits;

   public:
    Link(size_t pos, size_t num) {
      HWY_DASSERT(num < kMaxNum);
      const size_t prev = pos == 0 ? num - 1 : pos - 1;
      const size_t next = pos == num - 1 ? 0 : pos + 1;
      bits_ =
          (static_cast<uint32_t>(next) << kShift) | static_cast<uint32_t>(prev);
      HWY_DASSERT(Next() == next && Prev() == prev);
      HWY_DASSERT(!IsRemoved());
    }

    bool IsRemoved() const { return (bits_ & kMaxNum) != 0; }
    void Remove() { bits_ |= kMaxNum; }

    size_t Next() const { return bits_ >> kShift; }
    size_t Prev() const { return bits_ & (kMaxNum - 1); }

    void SetNext(size_t next) {
      HWY_DASSERT(next < kMaxNum);
      bits_ &= (~0u >> kBits);  // clear old next
      bits_ |= static_cast<uint32_t>(next) << kShift;
      HWY_DASSERT(Next() == next);
      HWY_DASSERT(!IsRemoved());
    }
    void SetPrev(size_t prev) {
      HWY_DASSERT(prev < kMaxNum);
      bits_ &= ~(kMaxNum - 1);  // clear old prev
      bits_ |= static_cast<uint32_t>(prev);
      HWY_DASSERT(Prev() == prev);
      HWY_DASSERT(!IsRemoved());
    }

   private:
    uint32_t bits_;
  };
  std::vector<Link> links_;
};

// State machine for choosing at runtime the lowest-cost `Config`, which is
// typically a struct containing multiple parameters. For an introduction, see
// "Auto-Tuning and Performance Portability on Heterogeneous Hardware".
//
// **Which parameters**
// Note that simple parameters such as the L2 cache size can be directly queried
// via `hwy/contrib/thread_pool/topology.h`. Difficult to predict parameters
// such as task granularity are more appropriate for auto-tuning. We also
// suggest that at least some parameters should also be 'algorithm variants'
// such as parallel vs. serial, or 2D tiling vs. 1D striping.
//
// **Search strategy**
// To guarantee the optimal result, we use exhaustive search, which is suitable
// for around 10 parameters and a few hundred combinations of 'candidate'
// configurations.
//
// **How to generate candidates**
// To keep this framework simple and generic, applications enumerate the search
// space and pass the list of all feasible candidates to `SetCandidates` before
// the first call to `NextConfig`. Applications should prune the space as much
// as possible, e.g. by upper-bounding parameters based on the known cache
// sizes, and applying constraints such as one being a multiple of another.
//
// **Usage**
// Applications typically conditionally branch to the code implementing the
// configuration returned by `NextConfig`. They measure the cost of running it
// and pass that to `NotifyCost`. Branching avoids the complexity and
// opaqueness of a JIT. The number of branches can be reduced (at the cost of
// code size) by inlining low-level decisions into larger code regions, e.g. by
// hoisting them outside hot loops.
//
// **What is cost**
// Cost is an arbitrary `uint64_t`, with lower values being better. Most
// applications will use the elapsed time. If the tasks being tuned are short,
// it is important to use a high-resolution timer such as `hwy/timer.h`. Energy
// may also be useful [https://www.osti.gov/servlets/purl/1361296].
//
// **Online vs. offline**
// Although applications can auto-tune once, offline, it may be difficult to
// ensure the stored configuration still applies to the current circumstances.
// Thus we recommend online auto-tuning, re-discovering the configuration on
// each run. We assume the overhead of bookkeeping and measuring cost is
// negligible relative to the actual work. The cost of auto-tuning is then that
// of running sub-optimal configurations. Assuming the best configuration is
// better than baseline, and the work is performed many thousands of times, the
// cost is outweighed by the benefits. To further reduce overhead, we exclude
// costly configurations from further measurements after a few samples.
template <typename Config, size_t kMinSamples = 2>
class AutoTune {
 public:
  // Returns non-null best configuration if auto-tuning has already finished.
  // Otherwise, callers continue calling `NextConfig` and `NotifyCost`.
  const Config* Best() const { return best_; }
  double BestCost() {
    HWY_DASSERT(Best());
    return costs_[config_idx_].EstimateCost();
  }

  // If false, caller must call `SetCandidates` before `NextConfig`.
  bool HasCandidates() const {
    HWY_DASSERT(!Best());
    return !candidates_.empty();
  }
  // WARNING: invalidates `Best()`, do not call if that is non-null.
  void SetCandidates(std::vector<Config> candidates) {
    HWY_DASSERT(!HasCandidates());
    candidates_.swap(candidates);
    HWY_DASSERT(HasCandidates());
    costs_.resize(candidates_.size());
    list_ = NextWithSkip(candidates_.size());
  }
  Config FirstCandidate() const {
    HWY_DASSERT(HasCandidates());
    return candidates_[0];
  }

  // Returns the current `Config` to measure.
  const Config& NextConfig() const {
    HWY_DASSERT(!Best() && HasCandidates());
    return candidates_[config_idx_];
  }

  // O(1) except at the end of each round, which is O(N).
  void NotifyCost(uint64_t cost) {
    HWY_DASSERT(HasCandidates());

    costs_[config_idx_].Notify(static_cast<double>(cost));
    // Save now before we update `config_idx_`.
    const size_t my_idx = config_idx_;
    // Only compute if enough samples.
    const double my_cost = rounds_complete_ >= kMinSamples
                               ? costs_[config_idx_].EstimateCost()
                               : 0.0;

    // Advance to next non-skipped config with wrap-around. This decorrelates
    // measurements by not immediately re-measuring the same config.
    config_idx_ = list_.Next(config_idx_);
    // Might still equal `my_idx` if this is the only non-skipped config.

    // Disqualify from future `NextConfig` if cost was too far beyond the
    // current best. This reduces the number of measurements, while tolerating
    // noise in the first few measurements. Must happen after advancing.
    if (my_cost > skip_if_above_) {
      list_.Skip(my_idx);
    }

    // Wrap-around indicates the round is complete.
    if (HWY_UNLIKELY(config_idx_ < my_idx)) {
      ++rounds_complete_;

      // Enough samples for stable estimates: update the thresholds.
      if (rounds_complete_ >= kMinSamples) {
        double best_cost = HighestValue<double>();
        size_t idx_min = 0;
        for (size_t i = 0; i < candidates_.size(); ++i) {
          const double estimate = costs_[i].EstimateCost();
          if (estimate < best_cost) {
            best_cost = estimate;
            idx_min = i;
          }
        }
        skip_if_above_ = best_cost * 1.4;

        // After sufficient rounds, declare the winner.
        if (HWY_UNLIKELY(rounds_complete_ == 2 * kMinSamples)) {
          // Causes `Best()` to be non-null.
          best_ = &candidates_[idx_min];
          config_idx_ = idx_min;
        }
      }
    }
  }

  // Avoid printing during the first few rounds, because those might be noisy
  // and not yet skipped.
  bool ShouldPrint() { return rounds_complete_ > kMinSamples; }

  Span<const CostDistribution> Costs() const {
    return Span<const CostDistribution>(costs_.data(), costs_.size());
  }

 private:
  const Config* best_ = nullptr;
  std::vector<Config> candidates_;
  std::vector<CostDistribution> costs_;  // one per candidate
  size_t config_idx_ = 0;                // [0, candidates_.size())
  NextWithSkip list_;
  size_t rounds_complete_ = 0;

  double skip_if_above_ = 0.0;
};

}  // namespace hwy

#endif  // HIGHWAY_HWY_AUTO_TUNE_H_
