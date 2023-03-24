# Potential new ops for Highway

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'janwas' reviewed: '2023-03-24' }
*-->

[TOC]

## Wishlist

### `LeadingZeroCount`
LZCNT is included in avx512cd. For bit packing/array.
janwas will likely implement this, please raise an issue if you'd like to.

### Shl for 8-bit
janwas will likely implement this, please raise an issue if you'd like to.

### RVV codegen
* `OddEven` for <64bit lanes: use Set of wider constant 0_1, compare that to 1
* `rgather_vx` for broadcasting redsum result?

### SVE codegen
* SVE2: use XAR for `RotateRight`
* `CombineShiftRightBytes` use `TableLookupLanes` instead?
* `Shuffle*`: use `TableLookupLanes` instead?
* Use SME once available: DUP predicate, REVD (rotate 128-bit elements by 64),
  SCLAMP/UCLAMP, 128-bit TRN/UZP/ZIP (also in F64MM)

### emu128 codegen
* `#pragma unroll(1)` in all loops to enable autovectorization

### Add emu256 target
Reuse same wasm256 file, `#if` for wasm-specific parts. Use reserved avx slot.

### Extend contrib/algo
* Reduce
* Min/Max/IdxMin
* CountIf (https://en.algorithmica.org/hpc/simd/masking/)

### Faster `Reverse2` 16-bit
Avoid `RotateRight` - slow for non-avx3; if we had that, could already permute.

### Add `Reverse2` for 8-bit
Use `TableLookupBytes`.

### Add `TableLookupLanes` for 16-bit
`#751`

### `SumOfLanes` returning scalar
Avoids extra broadcast.

### Reductions for 8-bit
For orthogonality; already done for x86+NEON.

### Clear lowest mask bit
For iterating in hash table.

### Conflict detection
For hash tables. Use VPCONFLICT on ZEN4.

### `PromoteToEven`
For `WidenMul`, `MinOfLanes`.

### `TwoTablesLookupLanes`
Cross-lane semantics. Mainly target avx3, also sve/rvv (2 rgather). #982.

### `PromoteTo` for all types
For orthogonality.

### Add `DupEven` for 16-bit
Use in `MinOfLanes` (helps NEON).

### Masked add/sub
For tolower (subtract if in range) or hash table probing.

### `FindLastTrue`
Similar to `FindFirstTrue`.

### `AddSub`
Interval arithmetic?

### `Dup128TableLookupBytes`
Avoids having to add offset on RVV. Table must come from `LoadDup128`.

### `LoadPromoteTo`
For SVE (svld1sb_u32)+WASM? Compiler can probably already fuse.

## Done

* ~~Signbit~~
* ~~ConvertF64<->I32~~ (math-inl)
* ~~Copysign~~ (math)
* ~~CopySignToAbs~~ (math)
* ~~Neg~~
* ~~Compress~~
* ~~Mask ops~~ (math)
* ~~RebindMask~~
* ~~Not~~
* ~~FP16 conversions~~
* ~~Scatter~~
* ~~Gather~~
* ~~Pause~~
* ~~Abs i64~~
* ~~FirstN~~
* ~~Compare i64~~
* ~~AESRound~~
* ~~CLMul~~ (GCM)
* ~~TableLookupBytesOr0~~ (AES)
* ~~FindFirstTrue~~ (strlen)
* ~~NE~~
* ~~Combine partial~~
* ~~LoadMaskBits~~ (FirstN)
* ~~MaskedLoad~~
* ~~Bf16 promote2~~
* ~~ConcatOdd/Even~~
* ~~SwapAdjacentBlocks~~
* ~~OddEvenBlocks~~
* ~~CompressBlendedStore~~
* ~~RotateRight~~ (Reverse2 i16)
* ~~Compare128~~
* ~~OrAnd~~
* ~~IfNegativeThenElse~~
* ~~MulFixedPoint15~~ (codec)
* ~~Insert/ExtractLane~~
* ~~IsNan~~
* ~~IsFinite~~
* ~~StoreInterleaved~~
* ~~LoadInterleaved~~ (codec)
* ~~Or3/Xor3~~
* ~~NotXor~~ (sort)
* ~~FindKnownFirstTrue~~ (sort)
* ~~CompressStore~~ 8-bit
* ~~ExpandLoad~~ (hash)
* ~~Zen4 target~~ (sort)
* ~~SSE2 target~~ - by johnplatts
* ~~AbsDiff int~~ - by johnplatts
* ~~Le integer~~ - by johnplatts
