// Copyright 2026 Google LLC
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

// 16-bit hash prospector, inspired by hp16.c by Chris Wellons (public
// domain, https://github.com/skeeto/hash-prospector). Differs in expanded and
// pruned op set, improved max-bias scoring metric, parallelization via
// hwy::ThreadPool and vectorization.
//
// First uses a known-good sequence (MakeXMXMX) to bootstrap multipliers.
// Then enumerates all possible sequences of up to kMaxOps ops and kMaxMulOps
// multipliers and kMaxShiftOps shifts. Parameterizes each with all possible
// shift amounts, and a few known-good multipliers, and random tables.
// Finally, tries a moderate random subset of possible multiplier values and
// prints the best 8.

#include <stdint.h>
#include <stdio.h>

#include <cmath>

#include "hwy/base.h"

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SSE2 | HWY_SSSE3 | HWY_SSE4)
#endif  // HWY_DISABLED_TARGETS

#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/contrib/thread_pool/topology.h"
#include "hwy/timer.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/hash/hash_prospector16.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// After foreach_target
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {
#if HWY_TARGET != HWY_SCALAR

size_t NumThreads(const Topology& topology) {
  if (topology.packages.empty()) return ThreadPool::MaxThreads();
  return topology.packages[0].cores.size() - 1;
}

// Returns [0, range). Somewhat biased, but this is mainly used in tests.
uint32_t LemireMod(uint32_t in, uint32_t range) {
  const uint32_t mod =
      static_cast<uint32_t>((uint64_t{in} * uint64_t{range}) >> 32);
  HWY_DASSERT(mod < range);
  return mod;
}

// Returns random value in [0, range).
uint32_t Choose(uint32_t range, RngStream& rng) {
  // First truncate - requires a widening mul.
  const uint32_t r = static_cast<uint32_t>(rng() & 0xFFFFFFFF);
  return LemireMod(r, range);
}

constexpr uint32_t IPow(uint32_t base, uint32_t exp) {
  uint32_t result = 1;
  for (uint32_t i = 0; i < exp; ++i) {
    result *= base;
  }
  return result;
}
static_assert(IPow(2, 10) == 1024);
static_assert(IPow(3, 5) == 243);

// ---------------------------------------------------------------------------
// Op types

// Note that bswap is not part of the top 15 sequences, and Mul ^ MulHigh is
// not a bijection. We omit XOR_L (x ^= x << val) because MUL is better at
// mixing bits upward. MUL has the strongest mixing per instructions, but MUL2
// is also useful. PSHUFB is similar to a Lai-Massey orthomorphism; it has weak
// mixing for the cost, but is part of some of the best sequences.
//
// WARNING: if changing the set of ops, also remove the s_best_sequences
// initializer to force it to be re-evaluated.
enum class Op : uint8_t {
  XOR_R,    // x ^= x >> val
  MUL,      // x *= val, val odd
  MUL2,     // x += val * x * x, val even
  PSHUFB,   // x ^= PSHUFB[x >> 12]
  SENTINEL  // must be last, determines kNumOpTypes.
};
HWY_INLINE_VAR constexpr size_t kNumOpTypes = static_cast<size_t>(Op::SENTINEL);

// Otherwise, we have too many multiplier values to fit in u32.
HWY_INLINE_VAR constexpr size_t kMaxMulOps = 2;
// This can be exceeded by MakeXMXMX, but holds for random and enumerated
// sequences. More would slow down FindBestVariant.
HWY_INLINE_VAR constexpr size_t kMaxShiftOps = 2;

// LSB is constrained (even or odd depending on op), hence do not include it.
HWY_INLINE_VAR constexpr size_t kBitsPerMul = 8 * sizeof(uint16_t) - 1;
// u16, shift 0 not allowed because x ^= x >> 0 is not a bijection.
HWY_INLINE_VAR constexpr size_t kShiftAmounts = 15;

class OpAndVal {
 public:
  OpAndVal() = default;
  explicit OpAndVal(Op op) : val_(0), op_(op) {}

  // Applies the operation to independent values (iota of possible u16 inputs).
  // table is a pre-loaded PSHUFB table vector.
  template <class DU16, class VU16 = Vec<DU16>,
            class VU8 = Vec<Repartition<uint8_t, DU16>>>
  HWY_INLINE VU16 Apply(DU16 du16, VU16 x, VU8 table) const {
    const VU16 v = Set(du16, val_);

    switch (op_) {
      case Op::XOR_R:
        return Xor(x, ShiftRightSame(x, val_));
      case Op::MUL:
        return Mul(x, v);
      case Op::MUL2:
        return Add(x, Mul(v, Mul(x, x)));
      case Op::PSHUFB:
        return Xor(x, TableLookupBytes(table, ShiftRight<12>(x)));
      default:
        HWY_ABORT("Invalid Op %u", static_cast<unsigned>(op_));
    }
  }

  void Print(uint128_t table) const {
    switch (op_) {
      case Op::XOR_R:
        printf("x ^= x >> %u;\n", val_);
        break;
      case Op::MUL:
        printf("x *= 0x%04x;\n", val_);
        break;
      case Op::MUL2:
        printf("x += 0x%04x * x * x;\n", val_);
        break;
      case Op::PSHUFB:
        printf("x ^= PSHUFB[x >> 12]; table = %016lx %016lx\n", table.lo,
               table.hi);
        break;
      default:
        HWY_ABORT("Invalid Op %u", static_cast<unsigned>(op_));
    }
  }

  void SetRandomVal(RngStream& rng) {
    switch (op_) {
      case Op::XOR_R:
        val_ = static_cast<uint16_t>(1 + Choose(kShiftAmounts, rng));
        break;
      case Op::MUL:
        val_ = static_cast<uint16_t>(rng() >> 48) | 1;
        break;
      case Op::MUL2:
        val_ = static_cast<uint16_t>((rng() >> 48) & ~1);  // even
        break;
      case Op::PSHUFB:
        val_ = 0;
        break;
      default:
        HWY_ABORT("Invalid Op %u", static_cast<unsigned>(op_));
    }
  }

  bool IsMul() const { return op_ == Op::MUL || op_ == Op::MUL2; }
  bool IsShift() const { return op_ == Op::XOR_R; }

  // Copies from bits into val.
  void SetMulBits(uint32_t bits) {
    HWY_DASSERT(IsMul());
    HWY_DASSERT((bits >> kBitsPerMul) == 0);
    bits <<= 1;
    if (op_ == Op::MUL) {
      val_ = static_cast<uint16_t>(bits | 1);  // odd
    } else if (op_ == Op::MUL2) {
      val_ = static_cast<uint16_t>(bits);  // even
    }
  }

  // Returns encoding of val bits without the known LSB.
  uint32_t GetMulBits() const {
    HWY_DASSERT(IsMul());
    return val_ >> 1;
  }

  // Copies from mod into val.
  void SetShift(uint32_t mod) {
    HWY_DASSERT(IsShift());
    HWY_DASSERT(mod < kShiftAmounts);
    val_ = 1 + mod;
  }

  // Returns encoding of val without 0-amount shift.
  uint32_t GetShift() const {
    HWY_DASSERT(IsShift());
    const uint32_t mod = val_ - 1;
    HWY_DASSERT(mod < kShiftAmounts);
    return mod;
  }

 private:
  uint16_t val_;  // odd multiplier or shift amount
  Op op_;
  HWY_MEMBER_VAR_MAYBE_UNUSED uint8_t padding_;
};

// Holds the sequence of ops plus PSHUFB table.
class OpSequence {
  static constexpr size_t kMaxOps = 5;  // limit so this fits in 48 bytes
  static constexpr size_t kNumOps = 4;  // sufficient for quality goal

 public:
  OpSequence() = default;
  OpSequence(const OpSequence& other) = default;
  OpSequence& operator=(const OpSequence& other) = default;

  // Single-op sequence with random table, for bijection verification.
  OpSequence(Op op, RngStream& rng) : ops_{OpAndVal(op)}, num_ops_(1) {
    num_mul_ops_ = ops_[0].IsMul() ? 1 : 0;
    num_shift_ops_ = ops_[0].IsShift() ? 1 : 0;
    ops_[0].SetRandomVal(rng);

    // Regardless of table, Op::PSHUFB is a bijection because we can undo the
    // XOR by constant in the upper byte, recover the index, then undo the
    // table XOR.
    table_.lo = rng();
    table_.hi = rng();
  }

  // Total number of op sequences for pool.Run.
  // Some are invalid (too many muls/shifts); those decode to IsEmpty().
  // Shifts, muls, and tables are looped over separately.
  static constexpr size_t NumOpCombos() { return IPow(kNumOpTypes, kNumOps); }

  // Decode an encoded u32 into an OpSequence with only the ops set.
  // Shifts, multipliers, and table must be set separately via
  // SetVariant. Sets IsEmpty() if the encoding violates constraints.
  explicit OpSequence(uint32_t op_product) : num_ops_(0), table_{} {
    HWY_DASSERT(op_product < NumOpCombos());

    num_ops_ = static_cast<uint32_t>(kNumOps);
    num_mul_ops_ = 0;
    num_shift_ops_ = 0;
    for (size_t i = 0; i < kNumOps; i++) {
      ops_[i] = OpAndVal(static_cast<Op>(op_product % kNumOpTypes));
      op_product /= static_cast<uint32_t>(kNumOpTypes);
      num_mul_ops_ += ops_[i].IsMul();
      num_shift_ops_ += ops_[i].IsShift();
    }
    // Constraint check.
    if (num_mul_ops_ > kMaxMulOps || num_shift_ops_ > kMaxShiftOps) {
      num_ops_ = 0;
      return;
    }
  }

  // Returns XOR_R(8), MUL, XOR_R(7), MUL, XOR_R(9): a known-good sequence
  // used to bootstrap the initial set of multipliers.
  static OpSequence MakeXMXMX() {
    OpSequence seq;
    seq.num_ops_ = 5;
    seq.num_mul_ops_ = 2;
    seq.num_shift_ops_ = 3;
    seq.ops_[0] = OpAndVal(Op::XOR_R);
    seq.ops_[1] = OpAndVal(Op::MUL);
    seq.ops_[2] = OpAndVal(Op::XOR_R);
    seq.ops_[3] = OpAndVal(Op::MUL);
    seq.ops_[4] = OpAndVal(Op::XOR_R);
    // shift_product encodes (shift_amount - 1), first op in the lowest.
    // The actual shift amounts are 8, 7, 9.
    seq.SetShifts(((9 - 1) * kShiftAmounts * kShiftAmounts) +
                  ((7 - 1) * kShiftAmounts) + (8 - 1));
    seq.table_ = {};  // No PSHUFB table needed for this sequence.
    return seq;
  }

  // If true, the sequence violates constraints and must not be used.
  bool IsEmpty() const { return num_ops_ == 0; }

  template <class DU16, class VU16 = Vec<DU16>>
  HWY_INLINE VU16 Apply(DU16 du16, VU16 x) const {
    const Repartition<uint8_t, DU16> du8;
    const Vec<decltype(du8)> table =
        LoadDup128(du8, reinterpret_cast<const uint8_t*>(&table_));
    for (uint32_t i = 0; i < num_ops_; i++) {
      x = ops_[i].Apply(du16, x, table);
    }
    return x;
  }

  void Print() const {
    for (uint32_t i = 0; i < num_ops_; i++) {
      printf("    ");
      ops_[i].Print(table_);
    }
    fflush(stdout);
  }

  // Number of possible shift combinations for this sequence of ops.
  uint32_t NumShiftCombos() const {
    return IPow(kShiftAmounts, num_shift_ops_);
  }

  void RandomizeValuesForTest(RngStream& rng) {
    for (uint32_t i = 0; i < num_ops_; i++) {
      ops_[i].SetRandomVal(rng);
    }
  }

  // For finding random multipliers, without overwriting the shift counts set by
  // MakeXMXMX.
  void RandomizeMultipliers(RngStream& rng) {
    for (uint32_t i = 0; i < num_ops_; i++) {
      if (!ops_[i].IsMul()) continue;
      ops_[i].SetRandomVal(rng);
    }
  }

  void SetMultipliers(uint32_t bits) {
    HWY_DASSERT(bits < (1u << (kBitsPerMul * kMaxMulOps)));
    for (uint32_t i = 0; i < num_ops_; i++) {
      if (ops_[i].IsMul()) {
        ops_[i].SetMulBits(bits & ((1u << kBitsPerMul) - 1));
        bits >>= kBitsPerMul;
      }
    }
    // might not use all bits if we have fewer than kMaxMulOps.
  }

  uint32_t GetMultipliers() const {
    uint32_t bits = 0;
    // Reverse order so that GetMultipliers is the inverse of SetMultipliers.
    for (uint32_t i = num_ops_ - 1; i < num_ops_; --i) {
      if (ops_[i].IsMul()) {
        bits <<= kBitsPerMul;
        bits |= ops_[i].GetMulBits();
      }
    }
    HWY_DASSERT((bits >> (kBitsPerMul * kMaxMulOps)) == 0);
    return bits;
  }

  void SetShifts(uint32_t product) {
    HWY_DASSERT(product < NumShiftCombos());
    for (uint32_t i = 0; i < num_ops_; i++) {
      if (ops_[i].IsShift()) {
        ops_[i].SetShift(product % kShiftAmounts);
        product /= kShiftAmounts;
      }
    }
  }

  uint32_t GetShifts() const {
    uint32_t product = 0;
    // Reverse order so that GetShifts is the inverse of SetShifts.
    for (uint32_t i = num_ops_ - 1; i < num_ops_; --i) {
      if (ops_[i].IsShift()) {
        product *= kShiftAmounts;
        product += ops_[i].GetShift();
      }
    }
    HWY_DASSERT(product < NumShiftCombos());
    return product;
  }

  void SetTable(uint128_t table) { table_ = table; }
  uint128_t GetTable() const { return table_; }

 private:
  OpAndVal ops_[kMaxOps];
  uint32_t num_ops_ = 0;
  uint32_t num_mul_ops_ = 0;
  uint32_t num_shift_ops_ = 0;
  uint128_t table_;
};
static_assert(sizeof(OpSequence) == 48);

// Variations (shifts, multipliers, PSHUFB table) to apply each possible
// OpSequence. Can be packed into a single u32 for EncodedAndCost.
class Variant {
  // Two just in case one table is unlucky. Random tables are probably equally
  // good because Op::PSHUFB is a bijection independent of table.
  static constexpr size_t kNumTableSets = 2;
  static uint128_t s_known_tables[kNumTableSets];  // Set by InitTables.

  Variant(size_t shift_product, size_t mul_idx, size_t table_idx)
      : shift_product_(shift_product),
        mul_idx_(mul_idx),
        table_idx_(table_idx) {}

 public:
  static HWY_NOINLINE void InitTables() {
    AesCtrEngine engine(/*deterministic=*/true);
    for (size_t i = 0; i < kNumTableSets; ++i) {
      RngStream rng(engine, i);
      OpSequence seq(Op::PSHUFB, rng);
      s_known_tables[i] = seq.GetTable();
    }
  }

  // Known-good multiplier bit-packs (encoded via Get/SetMultipliers).
  // Found by random search over MakeXMXMX. Recomputed if first is 0.
  static constexpr size_t kNumMulSets = 8;
  static constexpr uint32_t kKnownMuls[kNumMulSets] = {
      0x0ccb0a29, /* cost=0.027039 */
      0x1934c269, /* cost=0.026442 */
      0x1cb4d1e9, /* cost=0.025179 */
      0x1ecb7625, /* cost=0.025895 */
      0x2732d066, /* cost=0.025971 */
      0x28cb51e9, /* cost=0.023191 */
      0x3734c625, /* cost=0.027042 */
      0x3d32da15, /* cost=0.027021 */
  };

  // Called by FindBestVariant, which also calls ToU32.
  template <class Func>
  static void Foreach(size_t num_shifts, const Func& func) {
    for (uint32_t shift_product = 0; shift_product < num_shifts;
         shift_product++) {
      for (size_t mul_idx = 0; mul_idx < kNumMulSets; mul_idx++) {
        for (size_t table_idx = 0; table_idx < kNumTableSets; table_idx++) {
          func(Variant{shift_product, mul_idx, table_idx});
        }
      }
    }
  }

  // Unpacks from a prior ToU32() result.
  explicit Variant(uint32_t variant) {
    table_idx_ = variant % kNumTableSets;
    variant /= kNumTableSets;
    mul_idx_ = variant % kNumMulSets;
    shift_product_ = variant / kNumMulSets;
  }

  uint32_t ToU32() const {
    return shift_product_ * kNumMulSets * kNumTableSets +
           mul_idx_ * kNumTableSets + table_idx_;
  }

  void ApplyTo(OpSequence& seq) const {
    seq.SetShifts(shift_product_);
    seq.SetMultipliers(kKnownMuls[mul_idx_]);
    seq.SetTable(s_known_tables[table_idx_]);
  }

  bool operator==(const Variant& other) const {
    return shift_product_ == other.shift_product_ &&
           mul_idx_ == other.mul_idx_ && table_idx_ == other.table_idx_;
  }
  bool operator!=(const Variant& other) const { return !(*this == other); }

  void Print() const {
    printf("shift_product=0x%x mul_idx=%u table_idx=%u\n", shift_product_,
           mul_idx_, table_idx_);
  }

 private:
  uint32_t shift_product_;
  uint32_t mul_idx_;
  uint32_t table_idx_;
};
uint128_t Variant::s_known_tables[Variant::kNumTableSets];

// ---------------------------------------------------------------------------
// Consistency checks.

HWY_NOINLINE void VerifyBijections() {
  const ScalableTag<uint16_t> du16;
  using VU16 = Vec<decltype(du16)>;
  const size_t N = Lanes(du16);
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 0);

  // Test each op individually with a known multiplier.
  for (size_t op_val = 0; op_val < kNumOpTypes; ++op_val) {
    OpSequence seq(static_cast<Op>(op_val), rng);

    // Bit array: 65536 bits = 8192 bytes.
    HWY_ALIGN uint8_t seen[8192] = {};

    for (uint32_t x = 0; x < (1u << 16); x += N) {
      const VU16 vx = Iota(du16, x);
      const VU16 vy = seq.Apply(du16, vx);
      HWY_ALIGN uint16_t results[MaxLanes(du16)];
      Store(vy, du16, results);

      for (size_t i = 0; i < N; i++) {
        const uint16_t val = results[i];
        const size_t byte_idx = val / 8;
        const uint8_t bit_mask = 1u << (val % 8);
        if (seen[byte_idx] & bit_mask) {
          HWY_ABORT("  Op %zu: NOT a bijection (value %u seen twice)\n", op_val,
                    val);
        }
        seen[byte_idx] |= bit_mask;
      }
    }
  }

  fprintf(stderr, "All bijections verified.\n");
}

HWY_NOINLINE void VerifyGetMultipliersRoundtrip() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 42);
  OpSequence seq = OpSequence::MakeXMXMX();
  for (int trial = 0; trial < 100; trial++) {
    seq.RandomizeValuesForTest(rng);
    const uint32_t bits = seq.GetMultipliers();
    OpSequence seq2 = seq;
    seq2.RandomizeValuesForTest(rng);
    seq2.SetMultipliers(bits);
    const uint32_t bits2 = seq2.GetMultipliers();
    if (bits != bits2) {
      HWY_ABORT("GetMultipliers roundtrip failed: 0x%08x != 0x%08x\n", bits,
                bits2);
    }
  }
  fprintf(stderr, "GetMultipliers roundtrip verified.\n");
}

HWY_NOINLINE void VerifyGetShiftsRoundtrip() {
  AesCtrEngine engine(/*deterministic=*/true);
  RngStream rng(engine, 42);
  OpSequence seq = OpSequence::MakeXMXMX();
  for (int trial = 0; trial < 100; trial++) {
    seq.RandomizeValuesForTest(rng);
    const uint32_t product = seq.GetShifts();
    OpSequence seq2 = seq;
    seq2.RandomizeValuesForTest(rng);
    seq2.SetShifts(product);
    const uint32_t product2 = seq2.GetShifts();
    if (product != product2) {
      HWY_ABORT("GetShifts roundtrip failed: 0x%08x != 0x%08x\n", product,
                product2);
    }
  }
  fprintf(stderr, "GetShifts roundtrip verified.\n");
}

// For each valid op structure, verify that Variant::ToU32 round-trips.
HWY_NOINLINE void VerifyVariantRoundtrip() {
  constexpr size_t kNumOpCombos = OpSequence::NumOpCombos();
  for (uint32_t op_product = 0; op_product < kNumOpCombos; ++op_product) {
    const OpSequence seq(op_product);
    if (seq.IsEmpty()) continue;
    Variant::Foreach(seq.NumShiftCombos(), [&](const Variant& v) {
      const Variant v2(v.ToU32());
      if (v != v2) {
        v.Print();
        v2.Print();
        HWY_ABORT("Variant roundtrip failed: encoding=0x%x", v.ToU32());
      }
    });
  }
  fprintf(stderr, "Variant roundtrip verified.\n");
}

// ---------------------------------------------------------------------------
// Avalanche score with instruction cost penalty.

// WARNING: if changing the cost function, also remove the s_best_sequences
// initializer to force it to be re-evaluated.
struct Score {
  static constexpr float kL2Cost = 0.4f;

  void Print() const {
    printf("cost = %.6g (bias=%.6g, l2=%.6g)\n", Cost(), max_bias, l2);
  }

  // Cost: lower is better.
  float Cost() const { return max_bias + kL2Cost * l2; }

  float max_bias;  // max |bias| over all 16x16 (input_bit, output_bit)
  float l2;        // RMS of all biases
};

Score Evaluate(const OpSequence& ops) {
  // bins[j][k] counts how many of the 65536 inputs have output bit k flipped
  // when input bit j is flipped.
  int32_t bins[16][16] = {};

  // Exhaustive over all 2^16 inputs.
  {
    const ScalableTag<uint16_t> du16;
    using VU16 = Vec<decltype(du16)>;
    const size_t N = Lanes(du16);

    // Pre-compute h[x] = Apply(x) for all 2^16 inputs (128 KB, fits in L2).
    // This avoids redundantly recomputing h[x] for each of the 16 bit flips.
    AlignedVector<uint16_t> hashes(1u << 16);
    uint16_t* HWY_RESTRICT hashes_ptr = hashes.data();
    for (uint32_t x = 0; x < (1u << 16); x += N) {
      Store(ops.Apply(du16, Iota(du16, x)), du16, hashes_ptr + x);
    }

    const VU16 k1 = Set(du16, 1);
    for (int j = 0; j < 16; j++) {
      const uint32_t flip = 1u << j;

      // Per-output-bit accumulators. Max value per lane is 65536/N, fits u16.
      HWY_ALIGN uint16_t counts[16 * MaxLanes(du16)];
      ZeroBytes(counts, 16 * N * sizeof(uint16_t));

      for (uint32_t x = 0; x < (1u << 16); x += N) {
        const VU16 h_x = Load(du16, hashes_ptr + x);
        VU16 h_xf;
        if (flip >= N) {
          // (x+k) ^ flip = (x ^ flip) + k when flip >= N (both are multiples
          // of N, so the XOR doesn't affect the low bits), hence contiguous.
          h_xf = Load(du16, hashes_ptr + (x ^ flip));
        } else {
          // Flipped indices are a permutation within the same N-element block.
          // Scalar gather; all accesses hit L1 since they're within 2N bytes.
          HWY_ALIGN uint16_t tmp[MaxLanes(du16)];
          for (size_t k = 0; k < N; k++) {
            tmp[k] = hashes_ptr[(x + k) ^ flip];
          }
          h_xf = Load(du16, tmp);
        }
        const VU16 diff = Xor(h_x, h_xf);
        for (int k = 0; k < 16; k++) {
          const VU16 bits = And(ShiftRightSame(diff, k), k1);
          Store(Add(Load(du16, counts + k * N), bits), du16, counts + k * N);
        }
      }

      // Reduce once per (j, k).
      for (int k = 0; k < 16; k++) {
        bins[j][k] =
            static_cast<int32_t>(ReduceSum(du16, Load(du16, counts + k * N)));
      }
    }
  }

  // Compute max and L2 bias.
  const ScalableTag<float> df;
  using VF = Vec<decltype(df)>;
  const size_t NF = Lanes(df);

  const float expected = 32768.0f;  // 1 << 15
  const VF inv_expected = Set(df, 1.0f / expected);

  HWY_ALIGN float flat_bins[256];
  for (int j = 0; j < 16; j++) {
    for (int k = 0; k < 16; k++) {
      flat_bins[j * 16 + k] = static_cast<float>(bins[j][k]);
    }
  }

  VF v_max = Zero(df);
  VF v_sum_sq = Zero(df);

  size_t i = 0;
  for (; i + NF <= 256; i += NF) {
    VF v_bin = Load(df, flat_bins + i);
    VF v_diff = AbsDiff(v_bin, Set(df, expected));
    VF v_bias = Mul(v_diff, inv_expected);
    v_max = Max(v_max, v_bias);
    v_sum_sq = MulAdd(v_bias, v_bias, v_sum_sq);
  }

  if (i < 256) {
    VF v_bin = LoadN(df, flat_bins + i, 256 - i);
    VF v_diff = AbsDiff(v_bin, Set(df, expected));
    VF v_bias = Mul(v_diff, inv_expected);
    v_max = Max(v_max, v_bias);
    v_sum_sq = MulAdd(v_bias, v_bias, v_sum_sq);
  }

  const float max_bias = ReduceMax(df, v_max);
  const float sum_sq = ReduceSum(df, v_sum_sq);
  return Score{max_bias, std::sqrt(sum_sq / 256.0f)};
}

// ---------------------------------------------------------------------------
// Multiplier search: find top-N multiplier bit-packs for a given OpSequence.

struct EncodedAndCost {
  // Either from GetMultipliers, or Combine(op_product, Variant.ToU32()).
  uint32_t encoded;
  float cost;
};
static_assert(sizeof(EncodedAndCost) == 8);

// Per-worker top-K set of EncodedAndCost.
struct TopResults {
  static constexpr size_t kMaxBest = 15;
  EncodedAndCost best[kMaxBest];
  uint32_t count = 0;
  HWY_MEMBER_VAR_MAYBE_UNUSED uint32_t padding = 0;

  // O(K) is fine, this is rarely updated.
  void MaybeInsert(uint32_t encoded, float cost) {
    // Deduplicate: if same encoded exists, update to min cost.
    for (size_t i = 0; i < count; i++) {
      if (best[i].encoded == encoded) {
        best[i].cost = HWY_MIN(best[i].cost, cost);
        return;
      }
    }
    if (count < kMaxBest) {
      best[count++] = {encoded, cost};
      return;
    }
    // Find worst (highest cost) in list.
    size_t worst = 0;
    for (size_t i = 1; i < kMaxBest; i++) {
      if (best[i].cost > best[worst].cost) worst = i;
    }
    if (cost < best[worst].cost) {
      best[worst] = {encoded, cost};
    }
  }
};

// Prints the top-K random multiplier sets.
HWY_MAYBE_UNUSED void PrintBestMultipliers(const OpSequence& seq,
                                           double budget_seconds,
                                           size_t max_muls, ThreadPool& pool) {
  // 20s has 5% better 16th than 2s. 60s is definitely at diminishing returns.
  AesCtrEngine engine(/*deterministic=*/false);
  const size_t num_workers = pool.NumWorkers();
  AlignedVector<TopResults> worker_results(num_workers);

  const double t_start = platform::Now();
  static constexpr size_t kItersPerTask = 1024;
  size_t round = 0;
  for (; platform::Now() - t_start < budget_seconds; ++round) {
    pool.Run(0, num_workers, [&](uint64_t task, size_t worker) {
      RngStream rng(engine, round * num_workers + task);
      OpSequence my_seq = seq;

      for (size_t i = 0; i < kItersPerTask; i++) {
        my_seq.RandomizeMultipliers(rng);
        const uint32_t bits = my_seq.GetMultipliers();
        const Score s = Evaluate(my_seq);
        worker_results[worker].MaybeInsert(bits, s.Cost());
      }
    });
  }

  const size_t total_evals = round * num_workers * kItersPerTask;
  const double elapsed = platform::Now() - t_start;
  printf(
      "Searched %zu random multiplier combos in %.1f seconds "
      "(%.0f evals/sec)\n",
      total_evals, elapsed, static_cast<double>(total_evals) / elapsed);

  // Merge worker top-K into global.
  TopResults merged;
  for (const auto& result : worker_results) {
    for (size_t i = 0; i < result.count; i++) {
      merged.MaybeInsert(result.best[i].encoded, result.best[i].cost);
    }
  }

  const size_t num_muls = HWY_MIN(max_muls, merged.count);
  // Sort by cost for display.
  for (size_t i = 0; i < num_muls; i++) {
    for (size_t j = i + 1; j < merged.count; j++) {
      if (merged.best[j].cost < merged.best[i].cost) {
        const EncodedAndCost tmp = merged.best[i];
        merged.best[i] = merged.best[j];
        merged.best[j] = tmp;
      }
    }
  }
  printf("\nBest %zu multiplier sets:\n", num_muls);
  for (size_t i = 0; i < num_muls; i++) {
    printf("  0x%08x, /* cost=%.6f */\n", merged.best[i].encoded,
           merged.best[i].cost);
    OpSequence print_seq = seq;
    print_seq.SetMultipliers(merged.best[i].encoded);
    print_seq.Print();
  }
}

// ---------------------------------------------------------------------------
// Search for best sequence of ops, then its variants.

// Find the best variant for a given op-structure encoding.
EncodedAndCost FindBestVariant(OpSequence& seq) {
  HWY_DASSERT(!seq.IsEmpty());
  float best_cost = 1e9f;
  uint32_t best_variant = 0;
  Variant::Foreach(seq.NumShiftCombos(), [&](const Variant& variant) {
    variant.ApplyTo(seq);
    const float cost = Evaluate(seq).Cost();
    if (cost < best_cost) {
      best_cost = cost;
      best_variant = variant.ToU32();
    }
  });
  return EncodedAndCost{best_variant, best_cost};
}

// Packs op_product and variant into a single u32.
uint32_t Combine(uint32_t op_product, uint32_t variant) {
  constexpr size_t kNumOpCombos = OpSequence::NumOpCombos();
  return variant * kNumOpCombos + op_product;
}

// Inverse of Combine: decodes op structure and variant into an OpSequence.
OpSequence SequenceFromCombined(uint32_t combined) {
  constexpr size_t kNumOpCombos = OpSequence::NumOpCombos();
  const uint32_t op_product = combined % kNumOpCombos;
  const uint32_t variant = combined / kNumOpCombos;
  OpSequence seq(op_product);
  Variant(variant).ApplyTo(seq);
  return seq;
}

// Output of EnumerateAllSequences, sorted by cost. Encoded as
// `Combine(op_product, variant.ToU32())`. Recomputing takes only 1.3s on Turin,
// but we provide a pre-baked version so all time can be spent on multipliers.
// Recomputed if the first entry is 0.
static uint32_t s_best_sequences[15] = {
    /*cost=0.0269*/ 0x0ac248,
    /*cost=0.0400*/ 0x000eed,
    /*cost=0.0407*/ 0x007e2d,
    /*cost=0.0428*/ 0x09bc88,
    /*cost=0.0437*/ 0x083221,
    /*cost=0.0517*/ 0x072422,
    /*cost=0.0658*/ 0x00bce1,
    /*cost=0.0704*/ 0x0434c8,
    /*cost=0.0706*/ 0x070224,
    /*cost=0.0710*/ 0x030138,
    /*cost=0.0735*/ 0x00a2e2,
    /*cost=0.0738*/ 0x06e12c,
    /*cost=0.0740*/ 0x06e023,
    /*cost=0.0755*/ 0x00621d,
    /*cost=0.0801*/ 0x0004ee,
};

HWY_MAYBE_UNUSED void EnumerateAllSequences(ThreadPool& pool) {
  constexpr size_t kNumOpCombos = OpSequence::NumOpCombos();

  const size_t num_workers = pool.NumWorkers();
  AlignedVector<TopResults> worker_results(num_workers);

  const double t_start = platform::Now();

  pool.Run(0, kNumOpCombos, [&](uint64_t op_product, size_t worker) {
    OpSequence seq(static_cast<uint32_t>(op_product));
    if (HWY_UNLIKELY(seq.IsEmpty())) return;
    const EncodedAndCost ec = FindBestVariant(seq);
    const uint32_t combined = Combine(op_product, ec.encoded);
    worker_results[worker].MaybeInsert(combined, ec.cost);
  });

  const double elapsed = platform::Now() - t_start;

  // Merge worker results.
  TopResults merged;
  for (const auto& wr : worker_results) {
    for (size_t i = 0; i < wr.count; i++) {
      merged.MaybeInsert(wr.best[i].encoded, wr.best[i].cost);
    }
  }

  printf("\nExhaustive search of %zu op combos completed in %.1f seconds\n",
         kNumOpCombos, elapsed);
  printf("\nBest %u results:\n", merged.count);
  // Sort merged results for display.
  for (size_t i = 0; i < merged.count; i++) {
    for (size_t j = i + 1; j < merged.count; j++) {
      if (merged.best[j].cost < merged.best[i].cost) {
        EncodedAndCost tmp = merged.best[i];
        merged.best[i] = merged.best[j];
        merged.best[j] = tmp;
      }
    }
  }
  for (size_t i = 0; i < merged.count; i++) {
    printf("/*cost=%0.4f*/ 0x%06x,\n", merged.best[i].cost,
           merged.best[i].encoded);
    SequenceFromCombined(merged.best[i].encoded).Print();
    s_best_sequences[i] = merged.best[i].encoded;
  }
}

// ---------------------------------------------------------------------------

HWY_NOINLINE void TestHashProspector16() {
  VerifyBijections();
  VerifyGetMultipliersRoundtrip();
  VerifyGetShiftsRoundtrip();
  VerifyVariantRoundtrip();
}

HWY_NOINLINE void RunHashProspector16() {
  Topology topology;
  ThreadPool pool(NumThreads(topology));
  // Evaluating 23m multipliers (40s on Turin) has top 8 costs 0.023-0.027, vs.
  // 0.029-0.034 for 15s.
  const double budget_seconds = HWY_IS_DEBUG_BUILD ? 1.0 : 40.0;

  Variant::InitTables();

  // First get a table of 'strong' multipliers. We bake the result into the
  // source code to avoid waiting in each run. It is not clear why any encoded
  // MUL value should also be useful for MUL2, but it does work in practice
  // because almost all the top 15 sequences use MUL2.
  if constexpr (Variant::kKnownMuls[0] == 0) {
    const OpSequence seq = OpSequence::MakeXMXMX();
    PrintBestMultipliers(seq, budget_seconds, Variant::kNumMulSets, pool);
    return;  // do not continue because kKnownMuls is still empty.
  }

  if (s_best_sequences[0] == 0) {
    // Use those multipliers to enumerate all possible sequences.
    EnumerateAllSequences(pool);
  }
  // Now that we have s_best_sequences, experiment with multipliers again.
  // It is unlikely that any after the first 5 are useful - their higher costs
  // per the prior variant search are not recovered by better multipliers.
  for (size_t i = 0; i < 5; ++i) {
    const OpSequence seq = SequenceFromCombined(s_best_sequences[i]);
    // 8 are enough for a 128-bit table for use by phast-inl.h.
    PrintBestMultipliers(seq, budget_seconds, 8, pool);
  }
}

#else   // HWY_TARGET == HWY_SCALAR
void TestHashProspector16() {}
void RunHashProspector16() {}
#endif  // HWY_TARGET != HWY_SCALAR

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_BEFORE_TEST(HashProspector16);
HWY_EXPORT_AND_TEST_BEST_P(HashProspector16, TestHashProspector16);
HWY_EXPORT_AND_TEST_BEST_P(HashProspector16, RunHashProspector16);
HWY_AFTER_TEST();
}  // namespace hwy
#endif  // HWY_ONCE
