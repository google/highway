# Potential new ops for Highway

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'janwas' reviewed: '2024-04-09' }
*-->

## Wishlist

### F32RoundToNearestEven

### NEON dot product

### F16 WidenMulAccumulate on non-NEON

### numpy

NeUnordered

Loadn: Gather*, but for stride 2..4 use ld2..4.

LoadnPair: Gather with optimizations in particular for 2x64-bit, which use
128-bit loads plus Combine.
Also StorePair

Lookup128 for 32x 32-bit and 16x 64-bit. permutex2var on AVX-512, else Gather.
Also Lookup64 and Lookup32.

ReduceMin/MaxOrNaN

Document Reduce/Min NaN behavior

_mm512_getmant, _mm512_scalef, _mm512_getexp (f32/f64)

### Clear lowest mask bit

### Remaining math functions for hwy/contrib/math

High-precision! Consider copying from SLEEF. See #1650.

cbrt, cosh, erf, fmod, ilogb, lgamma, logb, modf, nextafter, nexttoward, pow,
scalbn, tan, tgamma

### Remaining STL functions for hwy/contrib/algo

*   Min/MaxValue
*   IndexOfMin/Max
*   AllOf / AnyOf / NoneOf
*   Count(If) (https://en.algorithmica.org/hpc/simd/masking/)
*   EqualSpan
*   ReverseSpan
*   ShuffleSpan
*   IsSorted
*   Reduce

### Range coder

Port https://github.com/richgel999/sserangecoding to Highway (~50 instructions).

### Iguana (fast LZ + ANS)

Port https://github.com/SnellerInc/sneller/tree/master/ion/zion/iguana
(Go+assembly) to Highway.

### AfterN

= Not(FirstN()), replaces several instances. WHILEGE on SVE.

### 52x52=104-bit multiply

For crypto. Native on Icelake+.

### RVV codegen

*   Use new mask<->vec cast instruction, possibly for OddEven, ExpandLoad
*   `rgather_vx` for broadcasting redsum result?

### SVE codegen

*   SVE2.1: TBLQ for `TableLookupBytes`
*   SVE2: use XAR for `RotateRight`
*   `CombineShiftRightBytes` use `TableLookupLanes` instead?
*   `Shuffle*`: use `TableLookupLanes` instead?
*   Use SME once available: DUP predicate, REVD (rotate 128-bit elements by 64),
    SCLAMP/UCLAMP, 128-bit TRN/UZP/ZIP (also in F64MM)

### emu128 codegen

* `#pragma unroll(1)` in all loops to enable autovectorization

### Add emu256 target
Reuse same wasm256 file, `#if` for wasm-specific parts. Use reserved avx slot.

### Conflict detection
For hash tables. Use VPCONFLICT on ZEN4.

### `Dup128TableLookupBytes`
Avoids having to add offset on RVV. Table must come from `LoadDup128`.

### `LoadPromoteTo`
For SVE (svld1sb_u32)+WASM? Compiler can probably already fuse.

## Done

*   ~~Signbit~~
*   ~~ConvertF64<->I32~~ (math-inl)
*   ~~Copysign~~ (math)
*   ~~CopySignToAbs~~ (math)
*   ~~Neg~~
*   ~~Compress~~
*   ~~Mask ops~~ (math)
*   ~~RebindMask~~
*   ~~Not~~
*   ~~FP16 conversions~~
*   ~~Scatter~~
*   ~~Gather~~
*   ~~Pause~~
*   ~~Abs i64~~
*   ~~FirstN~~
*   ~~Compare i64~~
*   ~~AESRound~~
*   ~~CLMul~~ (GCM)
*   ~~TableLookupBytesOr0~~ (AES)
*   ~~FindFirstTrue~~ (strlen)
*   ~~NE~~
*   ~~Combine partial~~
*   ~~LoadMaskBits~~ (FirstN)
*   ~~MaskedLoad~~
*   ~~Bf16 promote2~~
*   ~~ConcatOdd/Even~~
*   ~~SwapAdjacentBlocks~~
*   ~~OddEvenBlocks~~
*   ~~CompressBlendedStore~~
*   ~~RotateRight~~ (Reverse2 i16)
*   ~~Compare128~~
*   ~~OrAnd~~
*   ~~IfNegativeThenElse~~
*   ~~MulFixedPoint15~~ (codec)
*   ~~Insert/ExtractLane~~
*   ~~IsNan~~
*   ~~IsFinite~~
*   ~~StoreInterleaved~~
*   ~~LoadInterleaved~~ (codec)
*   ~~Or3/Xor3~~
*   ~~NotXor~~ (sort)
*   ~~FindKnownFirstTrue~~ (sort)
*   ~~CompressStore~~ 8-bit
*   ~~ExpandLoad~~ (hash)
*   ~~Zen4 target~~ (sort)
*   ~~SSE2 target~~ - by johnplatts
*   ~~AbsDiff int~~ - by johnplatts
*   ~~Le integer~~ - by johnplatts
*   ~~LeadingZeroCount~~ - by johnplatts in #1276
*   ~~8-bit Mul~~
*   ~~(Neg)MulAdd for integer~~
*   ~~AESRoundInv etc~~ - by johnplatts in #1286
*   ~~`OddEven` for <64bit lanes: use Set of wider constant 0_1~~
*   ~~Shl for 8-bit~~
*   ~~Shr for 8-bit~~
*   ~~Faster `Reverse2` 16-bit~~
*   ~~Add `Reverse2` for 8-bit~~
*   ~~`TwoTablesLookupLanes`~~ - by johnplatts in #1303
*   ~~Add 8/16-bit `TableLookupLanes`~~ - by johnplatts in #1303
*   ~~`FindLastTrue`~~ - by johnplatts in #1308
*   ~~Vec2, Create/Get functions~~
*   ~~`PromoteTo` for all types (#915)~~ - by johnplatts in #1387
*   ~~atan2~~
*   ~~Slide1Up/Down~~ - by johnplatts in #1496
*   ~~`MaxOfLanes, MinOfLanes` returning scalar~~
*   ~~Add `DupEven` for 16-bit~~ - by johnplatts in #1431
*   ~~AVX3_SPR target~~
*   ~~MaskedGather returns zero for mask=false.~~
*   ~~GatherIndexN/ScatterIndexN~~
*   ~~MaskedScatter~~
*   ~~float64 support for WASM~~
*   ~~LoadNOr~~
*   ~~PromoteEvenTo~~ - by johnplatts
*   ~~Masked add/sub/div~~
*   ~~ReduceMin/Max like ReduceSum, in addition to Min/MaxOfLanes~~
*   ~~Reductions for 8-bit~~
*   ~~RVV: Fix remaining 8-bit table lookups for large vectors~~
*   ~~QuickSelect algo~~ - by enum-class
*   ~~New tuple interface for segment load/store~~
*   ~~Div (integer division) and Mod~~ - by johnplatts
*   ~~AddSub and MulAddSub~~ - by johnplatts
*   ~~hypot~~ - by johnplatts
*   ~~exp2~~ - by johnplatts
