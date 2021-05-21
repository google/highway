# API synopsis / quick reference

[[_TOC_]]

## Usage modes

Highway can compile for multiple CPU targets, choosing the best available at
runtime (dynamic dispatch), or compile for a single CPU target without runtime
overhead (static dispatch). Examples of both are provided in examples/.

Dynamic dispatch uses the same source code as static, plus `#define
HWY_TARGET_INCLUDE`, `#include "hwy/foreach_target.h"` and
`HWY_DYNAMIC_DISPATCH`.

## Headers

The public headers are:

*   hwy/highway.h: main header, included from source AND/OR header files that
    use vector types. Note that including in headers may increase compile time,
    but allows declaring functions implemented out of line.

*   hwy/base.h: included from headers that only need compiler/platform-dependent
    definitions (e.g. `HWY_ALIGN_MAX` or `PopCount`) without the full highway.h.

*   hwy/foreach_target.h: re-includes the translation unit (specified by
    `HWY_TARGET_INCLUDE`) once per enabled target to generate code from the same
    source code. highway.h must still be included, either before or after.

*   hwy/aligned_allocator.h: defines functions for allocating memory with
    alignment suitable for `Load`/`Store`.

*   hwy/cache_control.h: defines stand-alone functions to control caching (e.g.
    prefetching) and memory barriers, independent of actual SIMD.

*   hwy/nanobenchmark.h: library for precisely measuring elapsed time (under
    varying inputs) for benchmarking small/medium regions of code.

*   hwy/tests/test_util-inl.h: defines macros for invoking tests on all
    available targets, plus per-target functions useful in tests (e.g. Print).

SIMD implementations must be preceded and followed by the following:

```
#include "hwy/highway.h"
HWY_BEFORE_NAMESPACE();  // at file scope
namespace project {  // optional
namespace HWY_NAMESPACE {

// implementation

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace project - optional
HWY_AFTER_NAMESPACE();
```

## Preprocessor macros

*   `HWY_ALIGN`: Ensures an array is aligned and suitable for Load()/Store()
    functions. Example: `HWY_ALIGN float lanes[4];` Note that arrays are mainly
    useful for 128-bit SIMD, or `LoadDup128`; otherwise, use dynamic allocation.

*   `HWY_ALIGN_MAX`: As `HWY_ALIGN`, but aligns to an upper bound suitable for
    all targets on this platform. Use this for caller of SIMD modules, e.g. for
    arrays used as arguments.

## Vector and descriptor types

Highway vectors consist of one or more 'lanes' of the same built-in type
`uint##_t, int##_t` for `## = 8, 16, 32, 64`, plus `float##_t` for `## = 16, 32,
64`.

In Highway, `float16_t` (an IEEE binary16 half-float) only supports load, store,
and conversion to/from `float32_t`; the behavior of `float16_t` infinity and NaN
are implementation-defined due to ARMv7.

On RVV, vectors are sizeless and cannot be wrapped inside a class. The Highway
API allows using built-in types as vectors because operations are expressed as
overloaded functions. Instead of constructors, overloaded initialization
functions such as `Set` take a zero-sized tag argument called `d` of type `D =
Simd<T, N>` and return an actual vector of unspecified type.

`T` is one of the lane types above, and may be retrieved via `TFromD<D>`.

`N` is target-dependent and not directly user-specified. The actual lane count
may not be known at compile time, but can be obtained via `Lanes(d)`. Use this
value, which is potentially different from `N`, to increment loop counters etc.
It is typically a power of two, but that is not guaranteed e.g. on SVE.

`d` lvalues (a tag, NOT actual vector) are typically obtained using two aliases:

*   Most common: pass `HWY_FULL(T[, LMUL=1]) d;` as an argument to return a
    native vector. This is preferred because it fully utilizes vector lanes.

    For targets (e.g. RVV) that support register groups, the optional `LMUL` (1,
    2, 4, 8) specifies the number of registers in the group. This effectively
    multiplies the lane count in each operation by `LMUL`. For mixed-precision
    code, `LMUL` must be at least the ratio of the sizes of the largest and
    smallest type. `LMUL > 1` is more efficient on single-issue machines, but
    larger values reduce the effective number of registers, which may cause the
    compiler to spill them to memory.

*   Less common: pass `HWY_CAPPED(T, N) d;` as an argument to return a vector
    which may be native width, but no more than `N` lanes have observable
    effects such as loading/storing to memory. This is less performance-portable
    because it may not use all available lanes. Note that the resulting lane
    count may also be less than `N`.

    For targets (e.g. RVV) that have compile-time-unknown lane counts, such
    vectors incur additional runtime cost in `Load` etc.

User-specified lane counts or tuples of vectors could cause spills on targets
with fewer or smaller vectors. By contrast, Highway encourages vector-length
agnostic code, which is more performance-portable.

Given that lane counts are potentially compile-time-unknown, storage for vectors
should be dynamically allocated, e.g. via `AllocateAligned(Lanes(d))`. For
applications that require a compile-time estimate, `MaxLanes(d)` returns the `N`
from `Simd<T, N>`, which is NOT necessarily the actual lane count. This is
DISCOURAGED because it is not guaranteed to be an upper bound (RVV vectors may
be very large) and some compilers are not able to interpret it as constexpr.

For mixed-precision code (e.g. `uint8_t` lanes promoted to `float`), tags for
the smaller types must be obtained from those of the larger type (e.g. via
`Rebind<uint8_t, HWY_FULL(float)>`).

## Using unspecified vector types

Because vector types are unspecified, local vector variables are typically
defined using `auto` for type deduction. A template argument `V` suffices for
simple generic functions: `template<class V> V Squared(V v) { return v * v; }`.

Many functions will need a `D` template argument in order to initialize any
constants. They can use a separate `V` template argument for vectors, or use
`Vec<D>`, or where an lvalue `d` is available, `decltype(Zero(d))`. Using such
aliases instead of auto may improve readability of mixed-type code. They can
also be used for member variables, which are discouraged because compilers often
have difficulty mapping them to registers.

## Operations

In the following, the argument or return type `V` denotes a vector with `N`
lanes, and `M` a mask. Operations limited to certain vector types begin with a
constraint of the form `V`: `{prefixes}[{bits}]`. The prefixes `u,i,f` denote
unsigned, signed, and floating-point types, and bits indicates the number of
bits per lane: 8, 16, 32, or 64. Any combination of the specified prefixes and
bits are allowed. Abbreviations of the form `u32 = {u}{32}` may also be used.

Note that Highway functions reside in `hwy::HWY_NAMESPACE`, whereas user-defined
functions reside in `project::[nested]::HWY_NAMESPACE`. Highway functions
generally take either a `Simd` or vector/mask argument. For targets where
vectors and masks are defined in namespace `hwy`, the functions will be found
via Argument-Dependent Lookup. However, this does not work for function
templates, and RVV and SVE both use builtin vectors. Thus we recommend a `using
hwy::HWY_NAMESPACE;` directive inside `project::[nested]::HWY_NAMESPACE`.

### Initialization

*   <code>V **Zero**(D)</code>: returns N-lane vector with all bits set to 0.
*   <code>V **Set**(D, T)</code>: returns N-lane vector with all lanes equal to
    the given value of type `T`.
*   <code>V **Undefined**(D)</code>: returns uninitialized N-lane vector, e.g.
    for use as an output parameter.
*   <code>V **Iota**(D, T)</code>: returns N-lane vector where the lane with
    index `i` has the given value of type `T` plus `i`. The least significant
    lane has index 0. This is useful in tests for detecting lane-crossing bugs.
*   <code>V **SignBit**(D, T)</code>: returns N-lane vector with all lanes set
    to a value whose representation has only the most-significant bit set.

### Arithmetic

*   <code>V **operator+**(V a, V b)</code>: returns `a[i] + b[i]` (mod 2^bits).
*   <code>V **operator-**(V a, V b)</code>: returns `a[i] - b[i]` (mod 2^bits).

*   `V`: `{i,f}` \
    <code>V **Neg**(V a)</code>: returns `-a[i]`.

*   `V`: `{i,f}` \
    <code>V **Abs**(V a)</code> returns the absolute value of `a[i]`; for
    integers, `LimitsMin()` maps to `LimitsMax() + 1`.

*   `V`: `f32` \
    <code>V **AbsDiff**(V a, V b)</code>: returns `|a[i] - b[i]|` in each lane.

*   `V`: `{u,i}{8,16}` \
    <code>V **SaturatedAdd**(V a, V b)</code> returns `a[i] + b[i]` saturated to
    the minimum/maximum representable value.

*   `V`: `{u,i}{8,16}` \
    <code>V **SaturatedSub**(V a, V b)</code> returns `a[i] - b[i]` saturated to
    the minimum/maximum representable value.

*   `V`: `{u}{8,16}` \
    <code>V **AverageRound**(V a, V b)</code> returns `(a[i] + b[i] + 1) / 2`.

*   <code>V **Clamp**(V a, V lo, V hi)</code>: returns `a[i]` clamped to
    `[lo[i], hi[i]]`.

*   `V`: `{f}` \
    <code>V **operator/**(V a, V b)</code>: returns `a[i] / b[i]` in each lane.

*   `V`: `{f}` \
    <code>V **Sqrt**(V a)</code>: returns `sqrt(a[i])`.

*   `V`: `f32` \
    <code>V **ApproximateReciprocalSqrt**(V a)</code>: returns an approximation
    of `1.0 / sqrt(a[i])`. `sqrt(a) ~= ApproximateReciprocalSqrt(a) * a`. x86
    and PPC provide 12-bit approximations but the error on ARM is closer to 1%.

*   `V`: `f32` \
    <code>V **ApproximateReciprocal**(V a)</code>: returns an approximation of
    `1.0 / a[i]`.

**Note**: Min/Max corner cases are target-specific and may change. If either
argument is qNaN, x86 SIMD returns the second argument, ARMv7 Neon returns NaN,
Wasm is supposed to return NaN but does not always, but other targets actually
uphold IEEE 754-2019 minimumNumber: returning the other argument if exactly one
is qNaN, and NaN if both are.

*   <code>V **Min**(V a, V b)</code>: returns `min(a[i], b[i])`.

*   <code>V **Max**(V a, V b)</code>: returns `max(a[i], b[i])`.

#### Multiply

*   `V`: `{u,i}{16,32}` \
    <code>V <b>operator*</b>(V a, V b)</code>: returns the lower half of `a[i] *
    b[i]` in each lane.

*   `V`: `{f}` \
    <code>V <b>operator*</b>(V a, V b)</code>: returns `a[i] * b[i]` in each
    lane.

*   `V`: `i16` \
    <code>V **MulHigh**(V a, V b)</code>: returns the upper half of `a[i] *
    b[i]` in each lane.

*   `V`: `{u,i}{32}` \
    <code>V **MulEven**(V a, V b)</code>: returns double-wide result of `a[i] *
    b[i]` for every even `i`, in lanes `i` (lower) and `i + 1` (upper).

#### Fused multiply-add

When implemented using special instructions, these functions are more precise
and faster than separate multiplication followed by addition. The `*Sub`
variants are somewhat slower on ARM; it is preferable to replace them with
`MulAdd` using a negated constant.

*   `V`: `{f}` \
    <code>V **MulAdd**(V a, V b, V c)</code>: returns `a[i] * b[i] + c[i]`.

*   `V`: `{f}` \
    <code>V **NegMulAdd**(V a, V b, V c)</code>: returns `-a[i] * b[i] + c[i]`.

*   `V`: `{f}` \
    <code>V **MulSub**(V a, V b, V c)</code>: returns `a[i] * b[i] - c[i]`.

*   `V`: `{f}` \
    <code>V **NegMulSub**(V a, V b, V c)</code>: returns `-a[i] * b[i] - c[i]`.

#### Shifts

**Note**: Counts not in `[0, sizeof(T)*8)` yield implementation-defined results.
Left-shifting signed `T` and right-shifting positive signed `T` is the same as
shifting `MakeUnsigned<T>` and casting to `T`. Right-shifting negative signed
`T` is the same as an unsigned shift, except that 1-bits are shifted in.

Compile-time constant shifts, generally the most efficient variant (though 8-bit
shifts are potentially slower than other lane sizes):

*   `V`: `{u,i}` \
    <code>V **ShiftLeft**&lt;int&gt;(V a)</code> returns `a[i] << int`.

*   `V`: `{u,i}` \
    <code>V **ShiftRight**&lt;int&gt;(V a)</code> returns `a[i] >> int`.

Shift all lanes by the same (not necessarily compile-time constant) amount:

*   `V`: `{u,i}` \
    <code>V **ShiftLeftSame**(V a, int bits)</code> returns `a[i] << bits`.

*   `V`: `{u,i}` \
    <code>V **ShiftRightSame**(V a, int bits)</code> returns `a[i] >> bits`.

Per-lane variable shifts (slow if SSE4, or 16-bit, or Shr i64 on AVX2):

*   `V`: `{u,i}{16,32,64}` \
    <code>V **operator<<**(V a, V b)</code> returns `a[i] << b[i]`.

*   `V`: `{u,i}{16,32,64}` \
    <code>V **operator>>**(V a, V b)</code> returns `a[i] >> b[i]`.

#### Floating-point rounding

*   `V`: `{f}` \
    <code>V **Round**(V a)</code>: returns `a[i]` rounded towards the nearest
    integer, with ties to even.

*   `V`: `{f}` \
    <code>V **Trunc**(V a)</code>: returns `a[i]` rounded towards zero
    (truncate).

*   `V`: `{f}` \
    <code>V **Ceil**(V a)</code>: returns `a[i]` rounded towards positive
    infinity (ceiling).

*   `V`: `{f}` \
    <code>V **Floor**(V a)</code>: returns `a[i]` rounded towards negative
    infinity.

### Logical

These operate on individual bits within each lane.

*   `V`: `{u,i}` \
    <code>V **operator&**(V a, V b)</code>: returns `a[i] & b[i]`.

*   `V`: `{u,i}` \
    <code>V **operator|**(V a, V b)</code>: returns `a[i] | b[i]`.

*   `V`: `{u,i}` \
    <code>V **operator^**(V a, V b)</code>: returns `a[i] ^ b[i]`.

*   `V`: `{u,i}` \
    <code>V **Not**(V v)</code>: returns `~v[i]`.

For floating-point types, builtin operators are not always available, so
non-operator functions (also available for integers) must be used:

*   <code>V **And**(V a, V b)</code>: returns `a[i] & b[i]`.

*   <code>V **Or**(V a, V b)</code>: returns `a[i] | b[i]`.

*   <code>V **Xor**(V a, V b)</code>: returns `a[i] ^ b[i]`.

*   <code>V **AndNot**(V a, V b)</code>: returns `~a[i] & b[i]`.

Special functions for signed types:

*   `V`: `{f}` \
    <code>V **CopySign**(V a, V b)</code>: returns the number with the magnitude
    of `a` and sign of `b`.

*   `V`: `{f}` \
    <code>V **CopySignToAbs**(V a, V b)</code>: as above, but potentially
    slightly more efficient; requires the first argument to be non-negative.

*   `V`: `i32/64` \
    <code>V **BroadcastSignBit**(V a)</code> returns `a[i] < 0 ? -1 : 0`.

### Masks

Let `M` denote a mask capable of storing true/false for each lane.

*   <code>M **FirstN**(D, size_t N)</code>: returns mask with the first `N`
    lanes (those with index `< N`) true. `N` larger than `Lanes(D())` result in
    an all-true mask. Useful for implementing "masked" stores by loading `prev`
    followed by `IfThenElse(FirstN(d, N), what_to_store, prev)`.

*   <code>M1 **RebindMask**(D, M2 m)</code>: returns same mask bits as `m`, but
    reinterpreted as a mask for lanes of type `TFromD<D>`. `M1` and `M2` must
    have the same number of lanes.

*   <code>M **Not**(M m)</code>: returns mask of elements indicating whether the
    input mask element was not set.

*   <code>M **And**(M a, M b)</code>: returns mask of elements indicating
    whether both input mask elements were set.

*   <code>M **AndNot**(M not_a, M b)</code>: returns mask of elements indicating
    whether not_a is not set and b is set.

*   <code>M **Or**(M a, M b)</code>: returns mask of elements indicating whether
    either input mask element was set.

*   <code>M **Xor**(M a, M b)</code>: returns mask of elements indicating
    whether exactly one input mask element was set.

*   <code>M **MaskFromVec**(V v)</code>: returns false in lane `i` if `v[i] ==
    0`, or true if `v[i]` has all bits set.

*   <code>V **VecFromMask**(D, M m)</code>: returns 0 in lane `i` if `m[i] ==
    false`, otherwise all bits set.

*   <code>V **VecFromMask**(M m)</code>: returns 0 in lane `i` if `m[i] ==
    false`, otherwise all bits set. DEPRECATED and will be removed before 1.0.

*   <code>V **IfThenElse**(M mask, V yes, V no)</code>: returns `mask[i] ?
    yes[i] : no[i]`.

*   <code>V **IfThenElseZero**(M mask, V yes)</code>: returns `mask[i] ?
    yes[i] : 0`.

*   <code>V **IfThenZeroElse**(M mask, V no)</code>: returns `mask[i] ? 0 :
    no[i]`.

*   <code>V **ZeroIfNegative**(V v)</code>: returns `v[i] < 0 ? 0 : v[i]`.

*   <code>bool **AllTrue**(M m)</code>: returns whether all `m[i]` are true.

*   <code>bool **AllFalse**(M m)</code>: returns whether all `m[i]` are false.

*   <code>size_t **StoreMaskBits**(M m, uint8_t* p)</code>: stores a bit array
    indicating whether `m[i]` is true, in ascending order of `i`, filling the
    bits of each byte from least to most significant, then proceeding to the
    next byte. Returns the number of (partial) bytes written.

*   <code>size_t **CountTrue**(M m)</code>: returns how many of `m[i]` are true
    [0, N]. This is typically more expensive than AllTrue/False.

*   `V`: `{u,i,f}{16,32,64}` \
    <code>V **Compress**(V v, M m)</code>: returns `r` such that `r[n]` is
    `v[i]`, with `i` the n-th lane index (starting from 0) where `m[i]` is true.
    Compacts lanes whose mask is set into the lower lanes; upper lanes are
    implementation-defined. Slow with 16-bit lanes.

*   `V`: `{u,i,f}{16,32,64}` \
    <code>size_t **CompressStore**(V v, M m, D, T* aligned)</code>: writes lanes
    whose mask is set into `aligned`, starting from lane 0. Returns
    `CountTrue(m)`, the number of valid lanes. All subsequent lanes may be
    overwritten! Alignment ensures inactive lanes will not cause faults. Slow
    with 16-bit lanes.

### Comparisons

These return a mask (see above) indicating whether the condition is true.

*   <code>M **operator==**(V a, V b)</code>: returns `a[i] == b[i]`.

*   `V`: `{i,f}` \
    <code>M **operator&lt;**(V a, V b)</code>: returns `a[i] < b[i]`.

*   `V`: `{i,f}` \
    <code>M **operator&gt;**(V a, V b)</code>: returns `a[i] > b[i]`.

*   `V`: `{f}` \
    <code>M **operator&lt;=**(V a, V b)</code>: returns `a[i] <= b[i]`.

*   `V`: `{f}` \
    <code>M **operator&gt;=**(V a, V b)</code>: returns `a[i] >= b[i]`.

*   `V`: `{u,i}` \
    <code>M **TestBit**(V v, V bit)</code>: returns `(v[i] & bit[i]) == bit[i]`.
    `bit[i]` must have exactly one bit set.

### Memory

Memory operands are little-endian, otherwise their order would depend on the
lane configuration. Pointers are the addresses of `N` consecutive `T` values,
either naturally-aligned (`aligned`) or possibly unaligned (`p`).

**Note**: computations with low arithmetic intensity (FLOP/s per memory traffic
bytes), e.g. dot product, can be *1.5 times as fast* when the memory operands
are naturally aligned. An unaligned access may require two load ports.

#### Load

*   <code>Vec&lt;D&gt; **Load**(D, const T* aligned)</code>: returns
    `aligned[i]`. May fault if the pointer is not aligned to the vector size.
    Using this whenever possible improves codegen on SSE4: unlike `LoadU`,
    `Load` can be fused into a memory operand, which reduces register pressure.
*   <code>Vec&lt;D&gt; **LoadU**(D, const T* p)</code>: returns `p[i]`.

*   <code>Vec&lt;D&gt; **LoadDup128**(D, const T* p)</code>: returns one 128-bit
    block loaded from `p` and broadcasted into all 128-bit block\[s\]. This may
    be faster than broadcasting single values, and is more convenient than
    preparing constants for the actual vector length.

#### Scatter/Gather

**Note**: Offsets/indices are of type `VI = Vec<RebindToSigned<D>>` and need not
be unique. The results are implementation-defined if any are negative.

**Note**: Where possible, applications should `Load/Store/TableLookup*` entire
vectors, which is much faster than `Scatter/Gather`. Otherwise, code of the form
`dst[tbl[i]] = F(src[i])` should when possible be transformed to `dst[i] =
F(src[tbl[i]])` because `Scatter` is more expensive than `Gather`.

*   `D`: `{u,i,f}{32,64}` \
    <code>void **ScatterOffset**(Vec&lt;D&gt; v, D, const T* base, VI
    offsets)</code>: stores `v[i]` to the base address plus *byte* `offsets[i]`.

*   `D`: `{u,i,f}{32,64}` \
    <code>void **ScatterIndex**(Vec&lt;D&gt; v, D, const T* base, VI
    indices)</code>: stores `v[i]` to `base[indices[i]]`.

*   `D`: `{u,i,f}{32,64}` \
    <code>Vec&lt;D&gt; **GatherOffset**(D, const T* base, VI offsets)</code>:
    returns elements of base selected by *byte* `offsets[i]`.

*   `D`: `{u,i,f}{32,64}` \
    <code>Vec&lt;D&gt; **GatherIndex**(D, const T* base, VI indices)</code>:
    returns vector of `base[indices[i]]`.

#### Store

*   <code>void **Store**(Vec&lt;D&gt; a, D, T* aligned)</code>: copies `a[i]`
    into `aligned[i]`, which must be naturally aligned. Writes exactly N *
    sizeof(T) bytes.
*   <code>void **StoreU**(Vec&lt;D&gt; a, D, T* p)</code>: as Store, but without
    the alignment requirement.

*   `D`: `u8` \
    <code>void **StoreInterleaved3**(Vec&lt;D&gt; v0, Vec&lt;D&gt; v1,
    Vec&lt;D&gt; v2, D, T* p)</code>: equivalent to shuffling `v0, v1, v2`
    followed by three `StoreU()`, such that `p[0] == v0[0], p[1] == v1[0],
    p[2] == v1[0]`. Useful for RGB samples.

*   `D`: `u8` \
    <code>void **StoreInterleaved4**(Vec&lt;D&gt; v0, Vec&lt;D&gt; v1,
    Vec&lt;D&gt; v2, Vec&lt;D&gt; v3, D, T* p)</code>: as above, but for four
    vectors (e.g. RGBA samples).

### Cache control

All functions except Stream are defined in cache_control.h.

*   <code>void **Stream**(Vec&lt;D&gt; a, D, const T* aligned)</code>: copies
    `a[i]` into `aligned[i]` with non-temporal hint on x86 (for good
    performance, call for all consecutive vectors within the same cache line).
    (Over)writes a multiple of HWY_STREAM_MULTIPLE bytes.

*   <code>void **LoadFence**()</code>: delays subsequent loads until prior loads
    are visible. Also a full fence on Intel CPUs. No effect on non-x86.

*   <code>void **StoreFence**()</code>: ensures previous non-temporal stores are
    visible. No effect on non-x86.

*   <code>void **FlushCacheline**(const void* p)</code>: invalidates and flushes
    the cache line containing "p". No effect on non-x86.

*   <code>void **Prefetch**(const T* p)</code>: begins loading the cache line
    containing "p".

*   <code>void **Pause**()</code>: when called inside a spin-loop, may reduce
    power consumption.

### Type conversion

*   <code>Vec&lt;D&gt; **BitCast**(D, V)</code>: returns the bits of `V`
    reinterpreted as type `Vec<D>`.

*   `V`,`D`: (`u8,i16`), (`u8,i32`), (`u16,i32`), (`i8,i16`), (`i8,i32`),
    (`i16,i32`), (`f16,f32`), (`f32,f64`) \
    <code>Vec&lt;D&gt; **PromoteTo**(D, V part)</code>: returns `part[i]` zero-
    or sign-extended to `MakeWide<T>`.

*   `V`,`D`: `i32,f64` \
    <code>Vec&lt;D&gt; **PromoteTo**(D, V part)</code>: returns `part[i]`
    converted to 64-bit floating point.

*   `V`,`D`: (`u32,u8`) \
    <code>Vec&lt;D&gt; **U8FromU32**(V)</code>: special-case `u32` to `u8`
    conversion when all lanes of `V` are already clamped to `[0, 256)`.

`DemoteTo` and float-to-int `ConvertTo` return the closest representable value
if the input exceeds the destination range.

*   `V`,`D`: (`i16,i8`), (`i32,i8`), (`i32,i16`), (`i16,u8`), (`i32,u8`),
    (`i32,u16`), (`f64,f32`) \
    <code>Vec&lt;D&gt; **DemoteTo**(D, V a)</code>: returns `a[i]` after packing
    with signed/unsigned saturation to `MakeNarrow<T>`.

*   `V`,`D`: `f64,i32` \
    <code>Vec&lt;D&gt; **DemoteTo**(D, V a)</code>: rounds floating point
    towards zero and converts the value to 32-bit integers.

*   `V`,`D`: `f32,f16` \
    <code>Vec&lt;D&gt; **DemoteTo**(D, V a)</code>: narrows float to half.

*   `V`,`D`: (`i32`,`f32`), (`i64`,`f64`) \
    <code>Vec&lt;D&gt; **ConvertTo**(D, V)</code>: converts an integer value to
    same-sized floating point.

*   `V`,`D`: (`f32`,`i32`), (`f64`,`i64`) \
    <code>Vec&lt;D&gt; **ConvertTo**(D, V)</code>: rounds floating point towards
    zero and converts the value to same-sized integer.

*   `V`: `f32`; `Ret`: `i32` \
    <code>Ret **NearestInt**(V a)</code>: returns the integer nearest to `a[i]`;
    results are undefined for NaN.

### Swizzle

*   <code>T **GetLane**(V)</code>: returns lane 0 within `V`. This is useful for
    extracting `SumOfLanes` results.

*   <code>V2 **Upper/LowerHalf**(V)</code>: returns upper or lower half of the
    vector `V`.

*   <code>V **ZeroExtendVector**(V2)</code>: returns vector whose `UpperHalf` is
    zero and whose `LowerHalf` is the argument.

*   <code>V **Combine**(V2, V2)</code>: returns vector whose `UpperHalf` is the
    first argument and whose `LowerHalf` is the second argument. This is
    currently only implemented for RVV, AVX2, AVX3. If you need to assemble
    <128 bit parts, please raise an issue to discuss.

*   <code>V **OddEven**(V a, V b)</code>: returns a vector whose odd lanes are
    taken from `a` and the even lanes from `b`.

**Note**: if vectors are larger than 128 bits, the following operations split
their operands into independently processed 128-bit *blocks*.

*   `V`: `{u,i}{16,32,64}, {f}` \
    <code>V **Broadcast**&lt;int i&gt;(V)</code>: returns individual *blocks*,
    each with lanes set to `input_block[i]`, `i = [0, 16/sizeof(T))`.

*   `Ret`: `MakeWide<T>`; `V`: `{u,i}{8,16,32}` \
    <code>Ret **ZipLower**(V a, V b)</code>: returns the same bits as
    `InterleaveLower`, but repartitioned into double-width lanes (required in
    order to use this operation with scalars).

*   `V`: `{u,i}` \
    <code>V **TableLookupBytes**(V bytes, V from)</code>: returns
    `bytes[from[i]]`. Uses byte lanes regardless of the actual vector types.
    Results are implementation-defined if `from[i] >= HWY_MIN(vector size, 16)`.

**Note**: the following are only available for full vectors (`N` > 1), and split
their operands into independently processed 128-bit *blocks*:

*   `Ret`: `MakeWide<T>`; `V`: `{u,i}{8,16,32}` \
    <code>Ret **ZipUpper**(V a, V b)</code>: returns the same bits as
    `InterleaveUpper`, but repartitioned into double-width lanes (required in
    order to use this operation with scalars)

*   `V`: `{u,i}` \
    <code>V **ShiftLeftBytes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* left by `int` bytes \[1, 15\].

*   <code>V **ShiftLeftLanes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* left by `int` lanes.

*   `V`: `{u,i}` \
    <code>V **ShiftRightBytes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* right by `int` bytes \[1, 15\].

*   <code>V **ShiftRightLanes**&lt;int&gt;(V)</code>: returns the result of
    shifting independent *blocks* right by `int` lanes.

*   `V`: `{u,i}` \
    <code>V **CombineShiftRightBytes**&lt;int&gt;(V hi, V lo)</code>: returns a
    vector of *blocks* each the result of shifting two concatenated *blocks*
    `hi[i] || lo[i]` right by `int` bytes \[1, 16).

*   <code>V **CombineShiftRightLanes**&lt;int&gt;(V hi, V lo)</code>: returns a
    vector of *blocks* each the result of shifting two concatenated *blocks*
    `hi[i] || lo[i]` right by `int` lanes \[1, 16/sizeof(T)).

*   `V`: `{u,i,f}{32}` \
    <code>V **Shuffle2301**(V)</code>: returns *blocks* with 32-bit halves
    swapped inside 64-bit halves.

*   `V`: `{u,i,f}{32}` \
    <code>V **Shuffle1032**(V)</code>: returns *blocks* with 64-bit halves
    swapped.

*   `V`: `{u,i,f}{64}` \
    <code>V **Shuffle01**(V)</code>: returns *blocks* with 64-bit halves
    swapped.

*   `V`: `{u,i,f}{32}` \
    <code>V **Shuffle0321**(V)</code>: returns *blocks* rotated right (toward
    the lower end) by 32 bits.

*   `V`: `{u,i,f}{32}` \
    <code>V **Shuffle2103**(V)</code>: returns *blocks* rotated left (toward the
    upper end) by 32 bits.

*   `V`: `{u,i,f}{32}` \
    <code>V **Shuffle0123**(V)</code>: returns *blocks* with lanes in reverse
    order.

*   <code>V **InterleaveLower**(V a, V b)</code>: returns *blocks* with
    alternating lanes from the lower halves of `a` and `b` (`a[0]` in the
    least-significant lane).

*   <code>V **InterleaveUpper**(V a, V b)</code>: returns *blocks* with
    alternating lanes from the upper halves of `a` and `b` (`a[N/2]` in the
    least-significant lane).

**Note**: the following operations cross block boundaries, which is typically
more expensive on AVX2/AVX-512 than within-block operations.

*   <code>V **ConcatLowerLower**(V hi, V lo)</code>: returns the concatenation
    of the lower halves of `hi` and `lo` without splitting into blocks.

*   <code>V **ConcatUpperUpper**(V hi, V lo)</code>: returns the concatenation
    of the upper halves of `hi` and `lo` without splitting into blocks.

*   <code>V **ConcatLowerUpper**(V hi, V lo)</code>: returns the inner half of
    the concatenation of `hi` and `lo` without splitting into blocks. Useful for
    swapping the two blocks in 256-bit vectors.

*   <code>V **ConcatUpperLower**(V hi, V lo)</code>: returns the outer quarters
    of the concatenation of `hi` and `lo` without splitting into blocks. Unlike
    the other variants, this does not incur a block-crossing penalty on AVX2.

*   `V`: `{u,i,f}{32}` \
    <code>V **TableLookupLanes**(V a, VI)</code> returns a vector of
    `a[indices[i]]`, where `VI` is from `SetTableIndices(D, &indices[0])`.

*   <code>VI **SetTableIndices**(D, int* idx)</code> prepares for
    `TableLookupLanes` with lane indices `idx = [0, N)` (need not be unique).

### Reductions

**Note**: these 'reduce' all lanes to a single result (e.g. sum), which is
broadcasted to all lanes at no extra cost. To obtain a scalar, you can call
`GetLane`.

Being a horizontal operation (across lanes of the same vector), these are slower
than normal SIMD operations and are typically used outside critical loops.

*   `V`: `{u,i,f}{32,64}` \
    <code>V **SumOfLanes**(V v)</code>: returns the sum of all lanes in each
    lane.

*   `V`: `{u,i,f}{32,64}` \
    <code>V **MinOfLanes**(V v)</code>: returns the minimum-valued lane in each
    lane.

*   `V`: `{u,i,f}{32,64}` \
    <code>V **MaxOfLanes**(V v)</code>: returns the maximum-valued lane in each
    lane.

## Advanced macros

Let `Target` denote an instruction set:
`SCALAR/SSE4/AVX2/AVX3/PPC8/NEON/WASM/RVV`. Targets are only used if enabled
(i.e. not broken nor disabled). Baseline means the compiler is allowed to
generate such instructions (implying the target CPU would have to support them).

*   `HWY_Target=##` are powers of two uniquely identifying `Target`.

*   `HWY_STATIC_TARGET` is the best enabled baseline `HWY_Target`, and matches
    `HWY_TARGET` in static dispatch mode. This is useful even in dynamic
    dispatch mode for deducing and printing the compiler flags.

*   `HWY_TARGETS` indicates which targets to generate for dynamic dispatch, and
    which headers to include. It is determined by configuration macros and
    always includes `HWY_STATIC_TARGET`.

*   `HWY_SUPPORTED_TARGETS` is the set of targets available at runtime. Expands
    to a literal if only a single target is enabled, or SupportedTargets().

*   `HWY_TARGET`: which `HWY_Target` is currently being compiled. This is
    initially identical to `HWY_STATIC_TARGET` and remains so in static dispatch
    mode. For dynamic dispatch, this changes before each re-inclusion and
    finally reverts to `HWY_STATIC_TARGET`. Can be used in `#if` expressions to
    provide an alternative to functions which are not supported by HWY_SCALAR.

*   `HWY_IDE` is 0 except when parsed by IDEs; adding it to conditions such as
    `#if HWY_TARGET != HWY_SCALAR || HWY_IDE` avoids code appearing greyed out.

The following signal capabilities and expand to 1 or 0.

*   `HWY_CAP_INTEGER64`: support for 64-bit signed/unsigned integer lanes.
*   `HWY_CAP_FLOAT64`: support for double-precision floating-point lanes.

The following were used to signal the maximum number of lanes for certain
operations, but this is no longer necessary (nor possible on SVE/RVV), so they
are DEPRECATED:

*   `HWY_GATHER_LANES(T)`.
*   `HWY_CAP_GE256`: the current target supports vectors of >= 256 bits.
*   `HWY_CAP_GE512`: the current target supports vectors of >= 512 bits.

## Detecting supported targets

`SupportedTargets()` returns a cached (initialized on-demand) bitfield of the
targets supported on the current CPU, detected using CPUID on x86 or equivalent.
This may include targets that are not in `HWY_TARGETS`, and vice versa. If
there is no overlap the binary will likely crash. This can only happen if:

*   the specified baseline is not supported by the current CPU, which
    contradicts the definition of baseline, so the configuration is invalid; or
*   the baseline does not include the enabled/attainable target(s), which are
    also not supported by the current CPU, and baseline targets (in particular
    `HWY_SCALAR`) were explicitly disabled.

## Advanced configuration macros

The following macros govern which targets to generate. Unless specified
otherwise, they may be defined per translation unit, e.g. to disable >128 bit
vectors in modules that do not benefit from them (if bandwidth-limited or only
called occasionally). This is safe because `HWY_TARGETS` always includes at
least one baseline target which `HWY_EXPORT` can use.

*   `HWY_DISABLE_CACHE_CONTROL` makes the cache-control functions no-ops.
*   `HWY_DISABLE_BMI2_FMA` prevents emitting BMI/BMI2/FMA instructions. This
    allows using AVX2 in VMs that do not support the other instructions, but
    only if defined for all translation units.

The following `*_TARGETS` are zero or more `HWY_Target` bits and can be defined
as an expression, e.g. `-DHWY_DISABLED_TARGETS=(HWY_SSE4|HWY_AVX3)`.

*   `HWY_BROKEN_TARGETS` defaults to a blocklist of known compiler bugs.
    Defining to 0 disables the blocklist.

*   `HWY_DISABLED_TARGETS` defaults to zero. This allows explicitly disabling
    targets without interfering with the blocklist.

*   `HWY_BASELINE_TARGETS` defaults to the set whose predefined macros are
    defined (i.e. those for which the corresponding flag, e.g. -mavx2, was
    passed to the compiler). If specified, this should be the same for all
    translation units, otherwise the safety check in SupportedTargets (that all
    enabled baseline targets are supported) may be inaccurate.

Zero or one of the following macros may be defined to replace the default
policy for selecting `HWY_TARGETS`:

*   `HWY_COMPILE_ONLY_SCALAR` selects only `HWY_SCALAR`, which disables SIMD.
*   `HWY_COMPILE_ONLY_STATIC` selects only `HWY_STATIC_TARGET`, which
    effectively disables dynamic dispatch.
*   `HWY_COMPILE_ALL_ATTAINABLE` selects all attainable targets (i.e. enabled
    and permitted by the compiler, independently of autovectorization), which
    maximizes coverage in tests.

If none are defined, but `HWY_IS_TEST` is defined, the default is
`HWY_COMPILE_ALL_ATTAINABLE`. Otherwise, the default is to select all attainable
targets except any non-best baseline (typically `HWY_SCALAR`), which reduces
code size.

## Compiler support

Clang and GCC require e.g. -mavx2 flags in order to use SIMD intrinsics.
However, this enables AVX2 instructions in the entire translation unit, which
may violate the one-definition rule and cause crashes. Instead, we use
target-specific attributes introduced via #pragma. Function using SIMD must
reside between `HWY_BEFORE_NAMESPACE` and `HWY_AFTER_NAMESPACE`. Alternatively,
individual functions or lambdas may be prefixed with `HWY_ATTR`.

Immediates (compile-time constants) are specified as template arguments to avoid
constant-propagation issues with Clang on ARM.

## Type traits

*   `IsFloat<T>()` returns true if the `T` is a floating-point type.
*   `IsSigned<T>()` returns true if the `T` is a signed or floating-point type.
*   `LimitsMin/Max<T>()` return the smallest/largest value representable in
    integer `T`.
*   `SizeTag<N>` is an empty struct, used to select overloaded functions
    appropriate for `N` bytes.

## Memory allocation

`AllocateAligned<T>(items)` returns a unique pointer to newly allocated memory
for `items` elements of POD type `T`. The start address is aligned as required
by `Load/Store`. Furthermore, successive allocations are not congruent modulo a
platform-specific alignment. This helps prevent false dependencies or cache
conflicts. The memory allocation is analogous to using `malloc()` and `free()`
with a `std::unique_ptr` since the returned items are *not* initialized or
default constructed and it is released using `FreeAlignedBytes()` without
calling `~T()`.

`MakeUniqueAligned<T>(Args&&... args)` creates a single object in newly
allocated aligned memory as above but constructed passing the `args` argument to
`T`'s constructor and returning a unique pointer to it. This is analogous to
using `std::make_unique` with `new` but for aligned memory since the object is
constructed and later destructed when the unique pointer is deleted. Typically
this type `T` is a struct containing multiple members with `HWY_ALIGN` or
`HWY_ALIGN_MAX`, or arrays whose lengths are known to be a multiple of the
vector size.

`MakeUniqueAlignedArray<T>(size_t items, Args&&... args)` creates an array of
objects in newly allocated aligned memory as above and constructs every element
of the new array using the passed constructor parameters, returning a unique
pointer to the array. Note that only the first element is guaranteed to be
aligned to the vector size; because there is no padding between elements,
the alignment of the remaining elements depends on the size of `T`.
