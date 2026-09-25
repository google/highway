# Potential new ops for Highway

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'janwas' reviewed: '2026-06-08' }
*-->

## Wishlist

### Widening add: u16 + half u8 = u16

### F32RoundToNearestEven

### F16 WidenMulAccumulate on non-NEON

### numpy

Loadn: Gather*, but for stride 2..4 use ld2..4.

LoadnPair: Gather with optimizations in particular for 2x64-bit, which use
128-bit loads plus Combine.
Also StorePair

_mm512_getmant (f32/f64)

### Clear lowest mask bit

### Remaining math functions for hwy/contrib/math

High-precision! Consider copying from SLEEF. See #1650.

fmod, nexttoward

### Remaining STL functions for hwy/contrib/algo

*   ShuffleSpan
*   Reduce (see #3374)

*   In-place Remove / RemoveIf, like CopyIf, but in-place.

*   MinMaxValue, IndexOfMinMax (in minmax-inl.h) - straightforward fuse of the
    existing functions which just compute Min or Max.

*   FindLast / FindLastIf (in find-inl.h) - can use FindLastTrue.

*   index-returning Mismatch(d, a, b, count) (in find-inl.h) - like EqualSpan,
    but returns the first mismatched index.

### AfterN

= Not(FirstN()), replaces several instances. WHILEGE on SVE.

### RVV codegen

*   Use new mask<->vec cast instruction, possibly for OddEven, ExpandLoad
*   `rgather_vx` for broadcasting redsum result?
*   use new vcreate intrinsics
*   Use clipu for ConcatEven, ConcatOdd

### x86 codegen

*   SumOfLanes 8-bit also use SumsOf8+Broadcast

### SVE codegen

*   SVE2.1: TBLQ for `TableLookupBytes`
*   `CombineShiftRightBytes` use `TableLookupLanes` instead?
*   `Shuffle*`: use `TableLookupLanes` instead?
*   Use SME once available: DUP predicate, REVD (rotate 128-bit elements by 64),
    SCLAMP/UCLAMP, 128-bit TRN/UZP/ZIP (also in F64MM)

### emu128 codegen

* `#pragma unroll(1)` in all loops to enable autovectorization

### Guaranteed 256-bit support

For non-scalable, can be similar to wasm256. Unclear how best to support
scalable vectors: would require a pair of vectors because not allowed to wrap
vectors in a struct.

### Conflict detection
For hash tables. Use VPCONFLICT on ZEN4.

### `Dup128TableLookupBytes`
Avoids having to add offset on RVV. Table must come from `LoadDup128`.

### `LoadPromoteTo`
For SVE (svld1sb_u32)+WASM? Compiler can probably already fuse.

## Done

*   ~~IsSorted~~ (algo)
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
*   ~~Min/MaxValue~~
*   ~~Count(If) (https://en.algorithmica.org/hpc/simd/masking/)~~
*   ~~cbrt~~
*   ~~erf~~
*   ~~cosh~~
*   ~~tan~~
*   ~~pow~~
*   ~~Lookup32~~
*   ~~tgamma~~
*   ~~lgamma~~
*   ~~ReduceMin/MaxOrNaN~~
*   ~~Document Reduce/Min NaN behavior~~
*   ~~IndexOfMin/Max~~ (algo)
*   ~~AllOf / AnyOf / NoneOf~~ (algo)
*   ~~EqualSpan~~ (algo)
*   ~~ReverseSpan~~ (algo)
*   ~~NEON dot product~~
*   ~~ilogb, logb, modf, nextafter~~
*   ~~Iguana~~
*   ~~Mul52~~
*   ~~NeUnordered~~
