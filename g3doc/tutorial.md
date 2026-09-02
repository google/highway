# Highway SIMD Tutorial

[TOC]

## Introduction

### What is SIMD, and why

**SIMD is available in all major CPUs and does N things at a time**

What is SIMD: e.g. 16 operations per instruction \
What for? More power efficient, amortizes per-instruction cost \
Why Highway? Portable and reliable.

Typical example of where to use it: element-wise sum of two arrays.

### Traditional methods

Some commonly used SIMD methods are auto-vectorization and intrinsics. The
former is a compiler technique that attempts to convert normal C++ into SIMD,
but this often fails in nontrivial cases. The latter are built-in C++ functions
that map closely to SIMD instructions, but are not portable and often verbose or
non type-safe.

### What is Highway

**C++ template library with 'portable intrinsics'**

Highway is a C++ template library which allows for portable, type-safe, and
readable SIMD code. It also includes a toolbox of components which are useful
for high-performance code: aligned memory allocation, bit sets, dynamic
dispatch, prefetching, benchmarking, performance counters, profiling, statistics
for benchmarking, and timing.

The contrib/ directory also includes related libraries: a standard library
subset, base64 encoding/decoding, bit packing, B-tree, dot products, hashing, an
image class, integer division, math functions, matrix-vector products, random
generation, sorting, OpenMP-like thread pool, and loop unroller.

## List of patterns

**Before going into each concept, we here list which ones are coming.**

This tutorial is grouped into eight sections. The first two are prerequisites,
everything else can be read in any order, depending on what operations your use
case requires.

1\) The most [fundamental SIMD operations](#map-pattern) are element-wise
operations on arrays (map pattern) and reductions (fold pattern), plus how to
write loops and handle remainders. \
2\) To be able to use Highway, we must first understand some
[incidental complexity](#prerequisites) required by C++: tag-based dispatch for
function overloading, plus namespaces and attributes, and either dynamic or
static dispatch. \
3\) Writing SIMD implies we care about performance, hence we discuss
[important optimizations](#important-optimizations) such as loop unrolling,
restrict pointers, alignment and inlining. \
4\) More complex SIMD code may involve [conditions](#conditions). When porting
if() expressions to SIMD, we use per-element ‘masks’ to skip operations on
certain lanes. Masks also have specialized reductions. \
5\) The fixed length of vectors necessitates some changes when
[type-casting](#type-conversions): we typically only halve or double the size at
a time, using two or half an input vector, respectively. \
6\) SIMD offers [special instructions](#special-arithmetic) for accelerating
bit-shifts, dot products, integer multiplication, and saturating arithmetic. All
of these differ from typical C semantics. \
7\) Some applications require [moving data between lanes](#cross-lane-movement). \
8\) We conclude with [less-common](#less-common-simd) patterns including
Gather/Scatter, Iota, sign manipulation and cryptography.

After working through these sections, you will be well on the way toward SIMD
expertise\!

## SIMD basics

Note that the sections reference source files which can be found in
[hwy/examples](https://github.com/google/highway/tree/master/hwy/examples).

### Map pattern

**SIMD usually applies the same operation to all lanes simultaneously.**

Apply a pure function (such as addition) to the same lane of two vectors:
`c[i] = f(a[i], b[i])`. Lanes do not interact.

Code: `sum_array_simple.cc`

### Portable vector loops

**Use compile-time-unknown vector length as loop step to support RISC-V/SVE.**

One of the key benefits of Highway is its ability to adapt the loop step size to
the CPU's available vector width, using portable code. Use `N = Lanes(d)` as the
loop increment. It is always a power of two, and `constexpr` on older ISAs but
not on RISC-V V and Arm SVE.

Code: `sum_array_simple.cc`

### Remainder handling

**Handle any iteration count with separate remainder implementation.**

To prevent static analysis warning, use if+for structure:

```c
size_t i = 0;
if (num >= N) {
    for (; i <= num - N; i += N) ..
}
Remainder = num - i;
if (HWY_UNLIKELY(remainder != 0)) { // branch prediction hint
    // Same loop body, but use LoadN/StoreN
}
```

Prefer `LoadN`/`StoreN`: always safe, fast on modern ISA, slow on x86 AVX2 or
earlier \- but OK if used after a long loop. Later we will introduce masking
(MaskedLoad).

The branch prediction hint may improve code layout by telling the compiler that
there usually are no leftovers.

If you know your `num >= N`, and the operation being performed is idempotent
(safe to repeat, e.g. ORing with 0x80), you can also replace the last iteration
with a vector load starting at `num - N`. Note that this potentially overlaps
previously processed elements.

Code: `sum_array_advanced.cc`

### Reductions

**Summarize vectors into a single result (fold pattern).**

Reduction applies an operation to each lane and updates the result, for example:
the sum. Often requires a tree-shaped permute pattern plus the reduction
operation (add).

This is usually expensive. Prefer to reduce once after a loop, not inside the
loop. \
`SumOfLanes`/`MaxOfLanes`/`MinOfLanes` broadcast to each element. \
`ReduceSum`/`ReduceMin`/`ReduceMax` return a single result. \
`GetLane` also returns lane 0; this can be useful if for some reason you want
both the broadcasting behavior of `SumOfLanes`, but also a scalar result.

Code: sum\_array\_advanced.cc.

### Sum of array

**Return scalar sum of array of float.**

This is mainly the [map pattern](#map-pattern) with
[loop unrolling](#loop-unrolling-pattern) plus [reduction](#reductions).

Code: sum\_array\_advanced.cc.

### (no sample) Lane access for debugging

**Insert/Extract is slow; you can also Print().**

If we want to check the value of a particular lane, `GetLane()` returns lane 0
and `ExtractLane(v, idx)` returns the lane with the given index. The latter is
fairly slow. If you are going to access multiple elements, it is probably better
to `Store` to a temporary aligned array. But remember that this should not be
done in time-critical loops, which would preferably only involve true SIMD
operations.

Similarly, you can overwrite a lane using `InsertLane`, but if doing that for
multiple lanes, it may be better to do so on an array, then `Load` it. However,
this is still likely to be slow, because the individual stores may not forward
to a larger load, leading to stalls. We recommend this only for debugging or
testing.

It is common to want to see the values of a vector for debugging. print-inl.h
provides a `Print(d, "caption", v)` function for this. There are two extra
optional arguments specifying the first lane to print, and the maximum number.
The latter is limited to 7, hence if the lane type is small and you want to see
all of them, you can pass `[0, Lanes(d))`.

## Prerequisites

Code: sum\_array\_simple.cc.

### Required namespaces

**Highway requires your code to reside in a namespace called `HWY_NAMESPACE`.**

To enable [dynamic dispatch](#dynamic-dispatch) (section below), your code must
reside inside a namespace called `HWY_NAMESPACE`. Note that this is a macro, but
it should just be used as "the namespace your code lives in", without caring
about what the macro evaluates to. Users typically also wrap their code in an
additional project-specific namespace, e.g. `project`, to avoid clashes between
their definitions and other projects, but this is optional.

Thus your .cc file will include, after the headers:

```c
namespace project { // optional
namespace HWY_NAMESPACE {
// …
}  // namespace HWY_NAMESPACE
}  // namespace project

```

### Required attributes

**Compilers require extra annotations (pragmas) in your source code for
functions using SIMD.**

`clang` and `GCC` used to require compiler flags in order to use SIMD, but this
no longer works if we want dynamic dispatch. Instead, compilers have started
supporting `#pragma`. Highway offers two mechanisms for hiding both these
requirements from users.

1\) You can prepend a `HWY_ATTR` macro to each function. This is brittle and not
recommended, especially when there are many functions. \
2\) Recommended: insert `HWY_BEFORE_NAMESPACE` and `HWY_AFTER_NAMESPACE`
before/after the above namespaces. The effect of this is as if `HWY_ATTR` had
been written before any member or free functions residing between them.

(Due to a prior compiler bug, it was necessary to place `HWY_BEFORE_NAMESPACE`
at the root (top-level) namespace scope, hence the name. The abstraction leaks
in one way: lambda functions are unfortunately not covered due to compiler
limitations; they require a `HWY_ATTR` annotation before their opening brace.)

### Static dispatch

For users that know exactly which instructions are available on their CPU (such
as personal desktops or supercomputers), Highway allows static dispatch. An
`#include "hwy/highway.h"` is sufficient before your actual
SIMD code. You will have an entry point (`CallMyFunction`), which then uses
`HWY_STATIC_DISPATCH` to call your SIMD functions.

You can tell the compiler which instructions to generate using \-m compiler
flags such as `-march=skx`. Highway will use the best available ‘target’
(combination of instruction sets) enabled by the `-m` flags, which it detects
using predefined macros. Some targets require multiple flags. For example,
`HWY_AVX2` requires both `-march=haswell` and `-maes` because not all Haswell
CPUs support AES, but Highway guarantees AES is available if AVX2 is. You can
use the `list_targets.cc` program to print which target is supported/enabled.

See sum\_array\_simple.cc for an example.

### Dynamic Dispatch

**Highway strength: choosing at runtime the instruction set to use.**

Runtime dispatch, aka. dynamic dispatch, means compiling multiple codepaths, one
per instruction set, and choosing the best available one at runtime. This is
important when deploying to unknown hardware such as cloud or user devices.
Highway supports this without having to write your code multiple times. It
builds a table of function pointers to each version of your code, and calls them
via `HWY_DYNAMIC_DISPATCH`. Some boilerplate is required in addition to the
static dispatch.

First, we tell foreach\_target.h to re-include this file multiple times, near
the top of the source file. We define a `HWY_TARGET_INCLUDE` macro to the name
or path to the current file. This is effectively an ‘argument’ passed to and
used by foreach\_target.h. The source path is relative to the compiler's include
path(s). \
System headers can reside before this, at the top of the file. Any includes of
highway.h, typically via headers we call \*-inl.h, must come after
foreach\_target.h.

```c
// 1. Define HWY_TARGET_INCLUDE pointing to this file
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "path/to/this_file.cc"
#include "hwy/foreach_target.h"
// 2. Include highway.h AFTER foreach_target.h
#include "hwy/highway.h"
```

Then comes the actual ‘business logic’, the SIMD portions of your code, wrapped
in namespaces and attributes per the prior two sections. Let's say it defines a
function called `MyFunction`, referenced below.

Finally, at the end of the source file, we generate the entry point into the
SIMD code for use by callers outside this .cc file. Everything that should only
be compiled once must reside in the `#if HWY_ONCE` block. This includes
`HWY_EXPORT`, which generates a table of function pointers, one per SIMD target,
and `CallMyFunction`, which dispatches to the version of `MyFunction` compiled
for the best-available instruction set. This is accomplished via
`HWY_DYNAMIC_DISPATCH`, which expands to code that queries the current CPU and
calls the corresponding function pointer in the table. `CallMyFunction` is
typically also declared in a header for calling by other .cc files.

```c
#if HWY_ONCE
namespace project {
HWY_EXPORT(MyFunction);
void CallMyFunction(...) {
  return HWY_DYNAMIC_DISPATCH(MyFunction)(...);
}
}
#endif
```

### Calling Highway ops

**Highway ops require namespace-using, or hn:: prefix in front of the function
calls.**

Highway ops (functions) reside in namespace `hwy::HWY_NAMESPACE`. If your
`project` namespace is chosen to be `hwy`, then you can call Highway ops
directly like normal functions.

Otherwise, thanks to C++ lookup rules (Argument Dependent Lookup), functions
with arguments residing in a namespace are automatically found. This means
functions with a tag argument (see next section) such as `Load` can also be
called directly. On some platforms, this also works for vectors if they are
wrapper classes defined in the same namespace. However, on others (SVE and
RISC-V) vectors are built-in types. Thus functions such as `Add` which have only
vector arguments would not be found. Users must instead call
`hwy::HWY_NAMESPACE::Add`, which is typically abbreviated with namespace aliases
to `hn::Add` after a `namespace hn = hwy::HWY_NAMESPACE;`. We recommend this
prefix for all ops, even `Load`, for consistency.

One possible alternative is to add `using hwy::HWY_NAMESPACE::Add;` directives
for every function that will be called, but missing directives are hard to spot
and will cause build failures on other platforms, making this a more brittle
approach.

### Tag-based dispatch

**Overloaded functions instead of classes.**

For technical reasons, the Highway API is based on overloaded functions, which
have the same name, but differing argument types. Highway uses an empty 'tag'
struct to select the correct function. \
(Most other libraries are instead built around vector class wrappers, but this
does not work for scalable RISC-V V and Arm SVE vectors because compilers do not
allow such vectors to be class members.)

Let `D` denote the ‘tag type’, most commonly `using D = hn::ScalableTag<T>;`,
where `T` is the desired lane type. Your SIMD code typically starts by declaring
an lvalue `d` of type \`D\`. Note that `hn::Vec<D>` is the actual type of a
vector. In specific situations, e.g. when using AES (which often uses 128-bit
vectors), we instead want `using D = hn::FixedTag<uint8_t, 16>;`.

Using this `d`, we can now construct vectors. The function `hn::Set(d, val)`
returns a vector with all lanes set to val. `hn::Zero(d)` is shorthand for
`hn::Set(d, T{0})`.

Compilers also do not allow arrays of vectors. If you want that, use arrays of
the lane type instead. What is allowed: local vector variables, returning
vectors, and pointers/references to vectors.

You are now ready to write your first Highway code\! We strongly recommend you
copy an existing example such as hwy/examples/skeleton\*, because the order of
these components is important.

### Optional: Headers

The simplest approach is to write SIMD code entirely contained within a single
.cc file. This section is only relevant if you want to reuse SIMD code between
multiple .cc files.

Due to the multiple-compilation approach (see
[Dynamic Dispatch](#dynamic-dispatch)), special header include guards are
required. Highway toggles the `HWY_TARGET_TOGGLE` macro between defined and
undefined between every compilation. As a result, the include guard will be
re-enabled for that target, but only once because the user code also toggles its
include guard.

For more information and guidance, see "When to use \-inl.h" in
impl\_details.md. See skeleton-inl.h for an example.

```c
#if defined(HWY_PATH_NAME_INL_H_) == defined(HWY_TARGET_TOGGLE)

// Flip the is-defined state of the ‘include guard’
#ifdef HWY_PATH_NAME_INL_H_
#undef HWY_PATH_NAME_INL_H_
#else
#define HWY_PATH_NAME_INL_H_
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Header contents

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard

```

## Conditions

### Mask type

**Data type used for comparison results, with potentially different
representation than vectors.**

Highway provides a `Mask` type for the result of comparison ops such as
`Gt`/`Eq`. This is because CPUs differ in how they represent these results:

-   AVX-512 has a special register with one bit per element;
-   RVV uses vector registers, but also with one bit per element starting from
    the lower bits;
-   SVE uses one bit per byte in the element, of which only the lowest is
    meaningful;
-   Most other CPUs use vector registers with all element bits either 1 or 0\.

To bridge these differences, Highway provides a rich set of operations operating
on masks:

We can convert between vectors and masks using `VecFromMask`/`MaskFromVec`,
assuming vector is all-1-bits or all-0. \
`MaskFalse(d)`, `SetMask(d, bool)`: initializes all elements to same bool \
`FirstN(d, num)`: true for first `Lanes(d)` elements, false otherwise \
`RebindMask`: type-cast e.g. int32 to float (same number of lanes) \
Can also convert to/from bit array: `LoadMaskBits`/`StoreMaskBits`. For \<=
512-bit vectors `BitsFromMask` is also available.

Code: masks\_and\_logic.cc.

### Predication/Masking

**Per-lane "maybe do something" decisions.**

Masks are most typically used to "select/blend" between two vectors, akin to the
ternary operator in C++. This is expressed as `IfThenElse`, or its variants
`IfThenElseZero`/`IfThenZeroElse` which have an implicit zero operand.

Masks can also be used to skip certain lanes during ops such as `MaskedAddOr`;
this is especially relevant for division. \
The convention is that an "Or" suffix means 'use the first argument if mask is
false', and otherwise, the default value is zero. A common pattern for
conditional addition is `val = MaskedAddOr(val, condition, val, increment)`,
corresponding to `if (condition) val += increment`.

Masked comparisons can also be useful for fusing an AND (e.g. with `FirstN` so
that only valid lanes are considered) into the comparison result, for example
`MaskedEq`.

`MaskedLoad` conditionally loads, or at least zeros out the result. Note that
this is unsafe for unaligned vectors that may cross a page boundary into an
unmapped page, if `HWY_MEM_OPS_MIGHT_FAULT` is true. AMD AVX2 CPUs indeed
reserve the right to page-fault in this case. We can avoid this problem by
ensuring vectors are aligned, or using `LoadN` instead for the common case where
we want to load no more than the given number of elements (e.g. in the last loop
iteration).

Code: masks\_and\_logic.cc.

### Whole-vector decisions

**No per-element branching, can only decide on the entire vector.**

Highway provides special [Reduction](#reductions)\-like operators for masks:

`AllTrue`, `AllFalse`: decide if no matching lane found \
Can also 'reduce' masks: `CountTrue`, `FindFirstTrue`/`FindLastTrue`.

As with reductions, these are slower than elementwise operations and should
preferably be executed only rarely, e.g. after a loop.

Code: mandelbrot.cc.

## Type conversions

### Tag type manipulation

**Highway has a mini-language for compile-time type manipulation.**

Simple programs use only a single type. However, libraries may accept a tag
argument which determines the vector length, and then want to derive from that
new tags for different types or vector lengths.

Typically we use `RebindToSigned<D>` to generate an additional type tag for
signed ints alongside e.g. a float tag, and
`RepartitionToWide<D>`/`RepartitionToNarrow<D>` for wider/narrower types. You
can also `Rebind<T2, D>` or `Repartition<T2, D>` for any new type T2. These
templates change the type of the tag to `T2`; `Rebind` keeps the same number of
lanes (hence `T2` must be the same size as the previous `T`), and Repartition
keeps the overall vector size the same, which means the number of lanes halves
when `T2` is twice as large.

`Half<decltype(d_full)>` and `Twice<D>` respectively halve and double the number
of lanes in a vector.

Sometimes we do not even have a tag argument. Tags can be deduced from vector
arguments: `DFromV<decltype(v)> d;` . For symmetry, we also provide `VFromD<D>`
which is equivalent to `Vec<D>`. You can also obtain the lane type from a tag
with `TFromD<D>`.

Code: see float\_distribution.cc.

### Half vectors

We can split/combine vector halves, for example when type-casting (see below).

`LowerHalf(v)` returns a half-vector. \
`UpperHalf(d, v)` requires a tag argument (which is optional for `LowerHalf`). \
`Combine(d, hi, lo)` rejoins these halves. \
`ZeroExtendVector(d, lo)` is equivalent to `Combine(d,
Zero(Half<decltype(d)>()), lo)`

Code: baker\_mix.cc

### Funnel pattern for casts

**Half/twice as many lanes for promotion/demotion.**

'Promote' means a widening cast and 'demote' means narrowing. \
Most ISAs have a native `PromoteTo`, which promote the lower half of an input
(e.g. int16) into a full output vector (e.g. int32). `DemoteTo` similarly takes
one input vector and return only a half-vector. \
Similarly, `PromoteMaskTo`/`DemoteMaskTo` are provided for masks.

Code: skeleton.cc.

### (no sample) Reinterpreting bit representation

**BitCast treats the underlying representation as a different type.**

Sometimes we want to operate on the bit representation of floats, or Gather
pairs of smaller types. This requires treating the bits in a vector as if they
were a different type. Unlike `PromoteTo`/`DemoteTo`/`ConvertTo`, `BitCast` does
not execute any instruction. It only returns a copy of the same vector bits, but
reinterpreted as another type.

## Special arithmetic

### Shifts

**There are several forms of bit-shifts.**

Historically, CPUs preferred shifts of the form `x << 13`, where the shift
amount is hard-coded into the instruction. This is expressed as
`ShiftLeft<13>(x)`. More recently, support was added for `x << y`, where the
shift amount is runtime variable and also per-lane. This is expressed as `Shl(x,
y)`. There is also an intermediate form which accepts a runtime variable,
applied to all lanes: `ShiftLeft(x, scalar)`. \
Highway also supports bit rotation: `RotateLeft`, `Rol`, `RotateLeftSame`. All
of the above also have variants for rightward shifts/rotations (e.g.
`ShiftRight`).

Code: float\_distribution.cc.

### Reorder pattern

**We can avoid slowdowns caused by ISA differences for use cases that tolerate
lane reordering.**

For a dot product, we only want the sum and are willing to reorder the lanes
(even though this can slightly change the results, see
[Loop unrolling](#loop-unrolling-pattern)). This helps deal with differences in
the ISA. For example, NEON typically processes the upper or lower half of a
vector, whereas SVE prefers to process odd or even-indexed lanes.

Demoting two vectors (e.g. int16) into one full output vector (e.g. int8) can be
more efficient than `DemoteTo`, but is not supported for all types. There are
two variants: `OrderedDemote2To` preserves the lane order, whereas
`ReorderDemote2To` is allowed to change order (useful we we know all lanes are
equal, or we will later reduce).

Code: float\_distribution.cc.

### Mixed precision dot

**CPUs provide a fused promote/widen/multiply/add operation crucial for ML
workloads.**

As of 2026, this is the most important operation for ML, and has one of the
highest compute densities of any instruction. `sum0 =
ReorderWidenMulAccumulate(d, a, b, sum0, sum1)` multiplies `a[i] * b[i]` after
promoting them to the next wider type, and then adds the result to *some lane
of* (see [Reorder pattern](#reorder-pattern)) sum0 *or* sum1. It returns the
updated sum0 and updates sum1 in-place. This reordering is fine for dot
products. If you want to guarantee a certain order, namely that each wider lane
of the result contains the result of multiplying and then pairwise-adding the
two narrow lanes that cover the wider lane, then this will be the return value
of `RearrangeToOddPlusEven(sum0, sum1)`, which should be called outside the
loop. We currently support this operation for signed/unsigned int16 and BF16;
FP16 is anticipated to be added. Due to the high practical relevance of
int8\*int8 \+ int32, CPUs also support a related operation that multiplies and
then adds *four* adjacent 8-bit lanes. This is `SumOfMulQuadAccumulate`.

Code: dot\_product\_mixed\_precision.cc

### Widening integer multiplication

**CPUs provide special instructions for integer multiplication.** \
With integer arithmetic, multiplying 8-bit numbers produces a 16-bit result. How
does this co-exist with the fixed size of a vector? One possibility is returning
the lower or upper half of the result using `Mul` and `MulHigh`, respectively.
Another is that `MulEven` returns a double-width result for every even lane.
Similarly, but less likely to be supported natively, `MulOdd` multiplies every
odd lane and places the result into the lane pair whose upper half it is.

Code: mandelbrot.cc

### Saturating arithmetic

**Integer overflow is well-defined in SIMD, but sometimes we want to
detect/prevent it.**

Addition and subtraction always wrap around for both signed and unsigned types.
It can be useful to instead have sums 'stuck' to the largest or smallest value
\- either to check whether overflow happened, as a debug aid, or as part of a
calculation. For example, broadcasting the sign bit of a byte can be
accomplished via signed min, which replaces positive values with zero, then
unsigned saturating addition with itself. If the sign bit was set, then the sum
would overflow and is instead replaced with the largest unsigned value, 0xFF;
otherwise, it remains zero. Saturating add/subtract is only supported for 8 or
16-bit elements, and is expressed as `SaturatedAdd`/`SaturatedSub`.

Code: sum\_hex.cc.

## Cross-lane movement

Elementwise operations are not always sufficient. Note that the CUDA programming
model attempts to pretend that there is only a simple element at a time, but
‘warp shuffles’ were added to allow passing data between adjacent elements.
Highway explicitly acknowledges the desirability of such movement, which is
often called shuffling or swizzling, and guarantees there are at least 128 bits
of elements in a full vector. Note that they tend to be slower (higher latency)
than simple logic or integer arithmetic, but they also tend to run on different
functional units, hence might be able to run in parallel with other
computations.

### Table lookups

**CPUs efficiently support gather operations within one or two registers.**

(This section is the most general case of cross-lane movement; the following
sections offer specializations which can be faster if they match the desired
pattern.)

All CPUs have good support for looking up one of 16 bytes, within 128-bit
blocks. This is expressed as `TableLookupBytes`: the result for a lane i is
`table[indices[i]]`, which requires `indices[i] < 16`. In other words, each
index lane says which input lane to take to produce the resulting lane. You can
also request that the output should be zero if the index is \>= 128 using
`TableLookupBytesOr0`.

Modern ISAs also support other element types, expressed as the
`TableLookupLanes` op. Except for RISC-V, lookups are usually constant-time, and
much faster than [Gather/Scatter](#gather-scatter). `TwoTablesLookupLanes`
concatenates two table registers. When not directly supported by the CPU,
Highway emulates these ops via `TableLookupBytes`. This may require some
preprocessing, which is why indices must be obtained from registers or pointers
using `IndicesFromVec` or `SetTableIndices`, respectively.

It can be complicated to reconcile large tables with unknown vector lengths. It
is generally safe to assume 128-bit vectors (except for the deprecated
`HWY_SCALAR` target), which when concatenated yields 256-bit tables. Highway
provides a `Lookup8` (plus variants for 16/32/64 elements) which can be more
efficient (may be able to use a single register) as well as more powerful (may
support larger tables) than using `TwoTablesLookupLanes` directly. The
prerequisite is that `HWY_MIN_BYTES / sizeof(T)` should be at least 4\. This is
always the case for 32-bit `T`.

Code: sum\_hex.cc.

### Compress/Expand

**Partition: special form of Swizzling driven by predicates.**

QuickSort/Select involve a partition step that moves to the front of an array
all elements that match a predicate. Special `Compress` instructions compact all
lanes whose predicate is true. You can `StoreU` those to memory and increment by
`CountTrue(d, mask)`, or better yet use the fused `CompressStore(v, mask, d,
ptr)` which returns the number of elements written.

`Expand` is the opposite: depositing consecutive elements at the lanes whose
predicate is true. We also provide a fused `LoadExpand`. This is useful for
unpacking data that was compressed by omitting certain elements (e.g.
zero-valued or negative etc.).

Code: float\_distribution.cc.

### 128-bit blocking

**For historical reasons, many swizzles split into separate 128-bit blocks.**

To avoid overhead on x86, many Highway operations involving cross-lane movement
are defined separately for each 128-bit 'block'. The whole-vector variants may
be more convenient, but often slower.

For example, `Broadcast<0>` copies into all other lanes \- but per 128-bit
block. With AVX2 and int64, lane 0 is copied to lanes 0..1, but lane 2 is copied
to lanes 2..3. This behavior is somewhat surprising, and the quick\_reference.md
calls it "blockwise". Conversely, we also provide ops defined across the entire
vector, without being split into blocks. We now list similar ops and their
whole-vector equivalents:

`Broadcast` | `BroadcastLane` \
`TableLookupBytes` | `TableLookupLanes`, `TwoTablesLookupLanes` \
`InterleaveLower` | `InterleaveWholeLower` \
`InterleaveUpper` | `InterleaveWholeUpper` \
`ShiftLeftBytes`/`ShiftLeftLanes` | `SlideUpLanes` \
`ShiftRightBytes`/`ShiftRightLanes` | `SlideDownLanes`

Some blockwise functions do not yet have whole-vector equivalents, including
`InterleaveEven`/`InterleaveOdd` and
`CombineShiftRightBytes`/`CombineShiftRightLanes`.

Code: benchmark.cc

### Stencil pattern (sliding window)

**Supporting ‘stencils’ (computations involving neighboring elements) via loads
or swizzles.**

In image processing (filters) or scientific computations (fluid dynamics),
computations typically involve neighboring elements. The simplest approach is to
`LoadU` again with an index offset of \+1 or \-1. This can actually be
performant on most CPUs for small stencils (just two neighbors), assuming the
array is padded such that it is safe to load an entire vector. \
For larger stencils or if there are already many loads, it can be better to
synthesize the vectors of left/right neighbors from aligned loads.
`CombineShiftRightBytes` does exactly this, but [blockwise](#128-bit-blocking),
hence it is only practical to use with vectors capped to 128 bits. The more
recently added `Slide1Up`/`Slide1Down` ops could also be helpful because they
operate on the whole register, not blocks, but they insert a zero into the
bottom/top lane, respectively, which is likely not what the application
requires, and may require extra blending/insertion to fix. `SlideUpLanesOr(low,
d, v, 1)` solves this by inserting a value from another vector `low` into the
lower lane of v.

Code: benchmark.cc, game\_of\_life.cc see also Tim Mattson's
[heat.c](https://github.com/tgmattso/ParProgForPhys/blob/main/OMP_GPU_Exercises/heat.c).

## Important optimizations

### (no sample) Inlining code

**Inlining small functions is extremely important for avoiding stack frame setup
overhead.**

Compilers decide whether to ‘inline’ functions (insert their code into the
caller), or instead generate a call to the subroutine. The latter can lead to
more compact code if the subroutine would be called multiple times. However, the
compiler might not have accurate knowledge of whether this happens. Hence
Highway allows specifying `HWY_INLINE` or `HWY_NOINLINE` in front of functions
to forcibly enable or disable inlining. We recommend `HWY_INLINE` for functions
only called once, or small functions (a few instructions). Conversely,
`HWY_NOINLINE` should be used for large functions, or for the top-level function
in a .cc file. The latter prevents warnings/errors about mismatches between
targets during inlining, relating to`HWY_ATTR` (see
[Required Attributes](#required-attributes)).

### Loop unrolling pattern

**Manually unrolling loops is a very important optimization.**

CPUs can often issue 2, 4, or even 6 vector operations per cycle, but it usually
takes 2-4 cycles until the result is complete. If we have a loop updating a
single `sum = Add(sum, Load(d, p)` per iteration, then the next iteration must
wait these 2-4 cycles until the result is ready. This underutilizes the CPU.
Compilers are usually unable to fix this because adding several sums can change
the results due to numerics (floats are not associative, and adding separate
sums changes the order of additions). Thus it is up to the programmer to
manually introduce independent sum variables, update them in an unrolled loop,
and then combine them into a single result after the loop. Moreover, because
compilers may disallow arrays of vectors, this really does require individual
variables. On the plus side, autocomplete is quite good at updating indices
sum0, sum1 etc. after copy-pasting. This optimization is also very worthwhile,
with potential for 2-5x speedups. Choose the number of accumulators as the
throughput (instructions per cycle) times the latency (number of cycles until a
result is ready), typically 4 or 8\. We also want to ensure that all these sums,
plus any active variables from each unrolling of the loop, fit within the
typically 32 registers provided by the architecture.

Code: dot\_product\_unroll.cc.

### (no sample) Restrict pointers

**Annotate pointers which are the only way to access an object/array.**

The presence of pointers in C++ makes it difficult for compilers to reason about
object lifetimes and references. For example, if a pointer is used to change an
object, any local copies made from that object must be re-loaded. As a result,
compilers may pessimistically always reload the object rather than keeping it in
a register, leading to larger and slower code. It is good practice to annotate
pointers using `HWY_RESTRICT` whenever possible. This tells the compiler that
the pointer is the only way to access the object to which it points. Hence the
compiler can keep the object in a register as long as that pointer is not
modified. `HWY_RESTRICT` should not be used when multiple pointers point to the
same object. Thus the presence of this modifier is helpful documentation for
readers, in addition to its benefits for the compiler.

### Alignment

**Vectors should usually be aligned for performance.**

Vector-aligned means their address is a multiple of their size. Sizes are powers
of two, so aligning to large pow2 is enough. Use aligned\_allocator.h for
allocating dynamically, and `HWY_ALIGN` for stack arrays. \
`Load` may crash if not aligned, to ensure this is noticed. `LoadU` does not
crash. Other memory-access functions including `LoadN`/`CompressStore` do not
require alignment.

The main allocation API is `auto arr = hwy::AllocateAligned<float>(count); //
returns unique_ptr`. This aligns to a multiple of 128 bytes, with some effort to
avoid cache conflicts caused by pointers always being congruent modulo some
power of two (i.e. having common factors of 2048 or 4096, which look the same to
parts of the CPU, which must thus assume they conflict).

Separately, even 'unaligned' vectors should still be aligned to their element
size, otherwise C++ considers this UB. `HWY_RCAST_ALIGNED` tells the compiler
that a byte pointer is actually aligned to the element size, and should be used
instead of C++ `reinterpret_cast` to avoid warnings.

Note that you might not see a speedup from aligning for some code on some CPUs.
For example, x86 does a good job of hiding the extra cost in many cases.
However, if your code is limited by accesses to the L1 cache, pointers that
happen to be unaligned may halve your performance.

Code: benchmark and ctf\_aes

### Data layout

**For best results, choose planar (Structure of Array) data layouts.**

Elementwise operations in SIMD tend to require the same type of data in each
element. Hence pixel or point data should have red channel or x coordinate in
all elements of one register. Unfortunately, some image formats are defined as
red, green, blue channels, and some input data may be arranged as Point2D {x,
y}. These are known as Array of Structure. If at all possible, these should be
replaced with planar, Structure of Array layouts.

When this is infeasible, Highway's
`LoadInterleaved2`/`LoadInterleaved3`/`LoadInterleaved4` ops can deinterleave
(for example) XY, RGB, or RGBA data into 2, 3 or 4 registers, each containing
the same type of component. Conversely,
`StoreInterleaved2`/`StoreInterleaved3`/`StoreInterleaved4` interleaves planar
vectors and stores to memory in that format. These are likely much faster than
[Gather/Scatter](#gather-scatter).

Code: baker\_mix.cc and mandelbrot.cc

### Fusion pattern

**CPUs can often perform multiple operations per instruction.**

For good reason, SIMD instruction sets are more complex than a pure "RISC"
(reduced instruction set). Fusing two or even three operations into a single
instruction reduces the number of instructions. Unfortunately, CPUs differ in
what exactly they can fuse. Highway includes a carefully chosen set of operators
which benefit from commonly available fusions without penalizing other
processors that lack them. The most important are fused-multiple add/subtract,
which are also more accurate in addition to faster. Note that some compilers may
automatically contract `Mul`+`Add` into `MulAdd`, but we recommend using the
pre-fused form for safety and pattern-matching for readers.

```c
MulAdd(f, m, a) = Add(Mul(f, m), a)
MulSub(f, m, s) = Sub(Mul(f, m), s)

AbsDiff(a, b) = Abs(Sub(a, b)

AndNot(a1, a2) = And(Not(a1), a2)
OrAnd(o1, a1, a2) = Or(o1, And(a1, a2)
Xor3(x1, x2, x3) = Xor(x1, Xor(x2, x3)
Or3(o1, o2, o3) = Or(o1, Or(o2, o3) - Xor3 should be preferred if applicable.
XorAndNot(x1, a1, a2) = Xor(x1, AndNot(a1, a2))
AndXor(a, x1, x2) = And(a, Xor(x1, x2))

IsNegative(v) = Lt(v, Zero(DFromV<decltype(v)>()))
IsEitherNaN(a, b) = Or(IsNaN(a), IsNaN(b))

CompressStore(v, m, d, p) = StoreU(Compress(v, m), d, p)
LoadExpand(m, d, p) = Expand(LoadU(d, p), m)
ConcatLowerLower(d, hi, lo) = Combine(d, LowerHalf(hi), LowerHalf(lo))
ConcatUpperUpper(d, hi, lo) = Combine(d, UpperHalf(dh, hi), UpperHalf(dh, lo))
ConcatUpperLower(d, hi, lo) = Combine(d, UpperHalf(dh, hi), LowerHalf(dh, lo))
ConcatLowerUpper(d, hi, lo) = Combine(d, LowerHalf(dh, hi), UpperHalf(dh, lo))
LoadDup128(d, p) = BroadcastBlock(ResizeBitCast(d, LoadU(FixedTag<T, 16/sizeof(T)>(), p)))
```

Also, all the `Masked\*` functions are arguably fused, and typically fold an AND
into the operation.

Code: dot\_product\_unroll.cc.

## Less-common SIMD

These are less common, but sometimes necessary.

### Partial vectors

**You can also request ‘partial vectors’.**

Highway encourages scalable vectors: whenever possible, user code should conform
to the number of lanes provided by the CPU. However, there are use cases where
we may want to use fewer lanes. One option is to cap the number of lanes to a
given upper bound. Highway only guarantees ‘as-if’ behavior: loads/stores do not
exceed the given amount, but vectors may still be larger and perform operations
on all elements. One common example is when ‘reducing’ (see
[Reduction](#reductions)) four separate vectors. We can first
`StoreInterleaved4` to a buffer, then load groups of four values at a time and
reduce them e.g. into a global sum. This has the advantage of still partially
vectorizing (four lanes at a time) without vector-length-specific handling.

This pattern requires caution because capped only guarantees `Lanes(d)` does not
exceed the cap. However, if you test on a CPU that actually has that many lanes,
your code may rely on this and then break when running on CPUs with shorter
vectors. It is thus safer to use `FixedTag<T, 16/sizeof(T)>`. Such 128-bit
vectors are guaranteed to be supported on all Highway targets (except the
deprecated `HWY_SCALAR`). One common use case is a prefix sum of 32-bit lanes.
This involves a tree of additions which relies on the fact that there are
exactly four lanes. Another use case is AES-128, which requires vectors of
exactly 16 bytes.

Code: ctf\_aes.cc.

### (no sample) Iota pattern

**Generate ascending integer sequences using Iota.**

Sometimes we want to generate consecutive integers 0..N-1, or more generally
j..j+N-1. This can be done with `Iota(d, j)`. It also works for floating-point
types, for which the integer `j` is converted to that type.

This is most commonly used as inputs to tests, or when vectorizing a loop which
uses/stores the value of the loop counter, for example argmax. Another
interesting usage is to turn an index into a ‘one-hot’ representation, where a
single predicate is true. For example, `Eq(Set(d, val), Iota(d, 0))` is true for
the lane whose index is equal to `val`.

Code: baker\_mix.cc

### Gather Scatter

**Elementwise loads/stores: slow, but sometimes required for vectorizing a
loop.**

For SIMD, we strongly prefer when lanes are loaded/stored to consecutive memory
addresses. If not, e.g. addresses depend on a different value in each lane, then
loading/storing is called a Gather/Scatter. This is much slower than a regular
`Load`/`Store` and should be avoided whenever possible. However, it can still
make sense if the values being produced or consumed require considerable (SIMD)
computation before/after the Gather/Scatter. Conversely, it is not worthwhile to
use a Gather/Scatter instruction simply to perform the memory loads/stores. CPUs
can do about as well just using normal single-element instructions. \
Two variants exist: `GatherIndex` is when the vector represents array indices:
`base + indices[i]`. `GatherOffset` is for byte offsets.

Code: matrix\_transpose\_scatter\_gather.cc.

### Matrix transpose

**Transposing matrices is a common operation which is difficult to make
vector-agnostic.**

One tempting approach is to use `GatherIndex` with a vector equal to
`Mul(Iota(di), Set(di, stride))`. This works, but is not especially fast, though
likely slightly better than `ScatterIndex`. For width-4 or 8 transpose, it is
usually faster to instead use `StoreInterleaved4`, which permutes registers and
then performs normal stores.

Code: matrix\_transpose\_scatter\_gather.cc.

### Floating-point sign manipulation

Highway provides various ops to read/modify floating-point signs.

`CopySign` and its more efficient variant `CopySignToAbs` (when the first
operand is non-negative) set the sign to that of the second argument. These are
useful for implementing math functions.

`Neg` and `Abs` (also available for integers) negate and return the absolute
value.

Code: float\_distribution.cc

### AES (cryptography)

**AESRound is the fastest way to thoroughly scramble data for
hashing/encryption/RNG.**

Highway provides ops to accelerate the AES symmetric block cipher, used in Wifi
and TLS. `AESRound` and `AESRoundInv` each implement one ‘round’ of
encryption/decryption. They are also useful for implementing a fast random
generator (`AesCtrEngine`) or long-string hash.

Code: ctf\_aes.cc.

### Cache control

**Highway also provides ops to exercise limited control over the otherwise
transparent cache.**

The cache hierarchy is critical for performance, but it is mostly managed
transparently. Highway provides a few means of influencing it directly. `Stream`
writes to aligned memory, bypassing the cache. This is sometimes useful if we
know the data will be write-only and not accessed again, because it avoids
evicting useful data in favor of the write-only data. After all such writes and
before loading, the `FlushStream` op must be called to ensure the writes are
visible on the current core (this has nothing to do with cross-core
synchronization, and provides no guarantees in that regard).

`Prefetch` requests that the memory system bring the data into cache(s). The CPU
typically already does this automatically, especially when the access pattern is
predictable, for example an ascending or descending sequence. However, your
program might know before the actual load what the next addresses will be. In
particular, unpredictable sequences as seen in sparse data structures might
benefit from prefetching already while processing the previous data. However,
one vexing question is how far in advance to prefetch, which depends on the CPU.
A good answer might be to auto-tune the best distance at runtime (measure what
works best).

You can also `FlushCacheline`, forcing the cache line to be written back to
memory. This is rarely done in application software, but useful for systems
software such as device drivers.

## Multithreading

We advise first making sure that your code is efficient on a single thread/core.
As a second step, utilizing other cores is necessary to maximize throughput.
Modern server CPUs have a surprising number of cores: 128 for AMD Turin, twice
that for Venice (coming in 2026). This leads to two concerns: first, each core's
share of the per-socket memory bandwidth is small and shrinking. Second, keeping
a coherent view of shared memory entails overhead. Ideally, each core will work
on its own, cache-sized data. This solves both problems by entirely minimizing
traffic on the narrow links to memory and other chiplets. This ideal is not
always feasible in practice, but staying as close as possible to a
‘shared-nothing’ model of independent cores and data is helpful.

How to parallelize? This is mostly orthogonal to the SIMD code, except that the
distribution of work should still be a multiple of the vector (and preferably
cache line/page size). Although out of scope of this tutorial, we briefly
mention some options: \
**OpenMP** introduced a successful approach where you add minimal source code
annotations right before your loop. This is very often sufficient, including
where reduction is required. Where available, we suggest using OpenMP because it
is standardized, simple and well-known. Some environments do not support OpenMP
and instead provide a `ParallelFor` function that loops over work items \[0, N).
For example, Highway provides a `ThreadPool` class which is sufficient for many
use cases.

Code: stream.cc

Other frameworks for specific patterns or use cases include:

-   [https://taskflow.github.io/](https://taskflow.github.io/)
-   [https://www.khronos.org/sycl/](https://www.khronos.org/sycl/)
-   [https://github.com/uxlfoundation/oneTBB](https://github.com/uxlfoundation/oneTBB)

This tutorial focuses on SIMD, not all the topics required for performance.
However, there are two interactions between threading and SIMD:

1\) The start/end of ranges processed by each thread should be vector-aligned,
and better yet cache-aligned to prevent conflicts, unalignment or remainder
handling. \
2\) SIMD benefits from aligned allocations, but these are not always aware of
NUMA effects (e.g. multiple CPU sockets). A simple and mostly sufficient
workaround is to have each thread allocate the memory it will use, e.g. via
`hwy::AllocateAligned`. Note that allocation can be slow and should preferably
be done only once, during initialization.

## Tips

### (no sample) Code alignment

It can be beneficial to align very hot loops to 32 or 64 bytes, but this is
highly CPU-specific. When using clang, you can insert
`[[clang::code_align(64)]]` before the loop to see if it helps.
