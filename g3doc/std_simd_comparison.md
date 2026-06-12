# Why we recommend Highway over std::simd

## Introduction

The author is passionate about efficient software and has been an enthusiastic
SIMD user since 2002. Our Highway SIMD library was created to meet requirements
that `std::experimental::simd` did and could not. I feel that it is time to
discuss these gaps and explain why you may prefer Highway over the recently
standardized `std::simd`.

First, I'd like to express respect and appreciation for the pioneering SIMD work
of Matthias Kretz, whose Vc library dates back to 2009-2012. Unfortunately, the
circumstances have changed and invalidated this design. It is regrettable that
the C++ committee has spent 10+ years standardizing this approach, which retains
several fundamental flaws:

-   A class wrapper is incompatible with the scalable vectors introduced by Arm
    SVE and standardized by RISC-V V.
-   It lacks runtime dispatch, i.e. utilizing what the current CPU supports.
-   The standardization process resulted in relatively few ops, such that many
    real-world SIMD use cases cannot be implemented. See the appendix.
-   Finally, standardization is meant to enable universal deployment, but the
    reality is that many projects are limited to older compilers. It will take
    several years until C++26 is dependably available.

During this decade, projects did not wait for `std::simd` and have instead
adopted libraries such as `xsimd`, or Highway, which is now battle-tested and
extensively deployed in production at Google scale.

Quoting
[Bucher](https://lucisqr.substack.com/p/c26-shipped-a-simd-library-nobody):
"std::simd in 2026 is the 2012 solution arriving after the world moved on".

Note that most other libraries, including `xsimd`, also have at least one of the
above issues, in particular missing support for scalable vectors. `xsimd` does
offer limited support for runtime dispatch, but it requires the set of targets
to be explicitly specified, and reside in separate source files, each compiled
with different flags, which is a maintenance burden.

## Technical issues with `std::simd`

### Scalable vectors vs. constexpr-sized vector class

Arm's SVE made the interesting decision to allow the CPU, rather than software,
to decide the vector length (128 to 2048 bits). RISC-V V ('RVV') followed, with
the twist that software can scale this length by 1/8 to 8x, with vector sizes of
up to 64k bits.

Both are fundamentally incompatible with the constexpr-sized `std::simd`.
Moreover, compilers currently disallow wrapping sizeless vector types in a
class, which breaks the `std::simd` class template design. The odds of this
changing are slim, because it would require ABI changes that the committee
explicitly disavowed.

There is a workaround: `-msve-vector-bits=N` tells the compiler to assume a
length, and then `std::simd` could theoretically use SVE. However, this defeats
the purpose of SVE and breaks when attempting to run on a different CPU. It is
also difficult to imagine combining this with runtime dispatch, because there
are many possible lengths for RVV.

Instead, Highway is designed to be length-agnostic. `ScalableTag<T>` adapts to
whatever the hardware vector length is. On SVE/RVV, Highway uses built-in types
directly, avoiding the class wrapper issue.

### No runtime dispatch

In the world of high-performance computing (supercomputers), the CPU is
typically known and changes infrequently. By contrast, there is a huge variety
of CPUs in phones, laptops and workstations. For x86 alone, there are at least
four major families in use with relevant performance differences that benefit
from specialized codepaths (e.g. to make use of new AVX-512 instructions). The
lowest-common denominator today is perhaps SSE4 from ~2007. Targeting it leaves
19 years of CPU advancements unused.

Unfortunately, and inexplicably, `std::simd` provides zero support for this. It
compiles only one codepath, determined by compiler flags. These are often chosen
conservatively, meaning the binary is unable to use anything newer. If users
choose flags for their CPU, the binary will likely crash if run on older CPUs.
The same is true for the auto-vectorizer. Thus both that and `std::simd` are
problematic when running on user devices or cloud servers.

By contrast, Highway was designed for runtime dispatch: writing only one
implementation, automatically compiling it for a variety of targets, and
choosing at runtime the best one for the current CPU. A single binary targets
most relevant variations of one architecture, with typically only modest binary
size impact because in our experience, SIMD kernels are not huge.

### Lack of ops

`std::experimental::simd` was extremely limited in terms of supported operations
(ops). Thanks to work by Intel, C++26 expanded this with some important ops:

`permute, compress/expand, select, reduce*, gather_from/scatter_to, chunk/cat`.

However, large classes of ops required for many real-world usages are missing:

-   Byte-level table lookups (PSHUFB) — IMO the single most important op
-   SAD (sum of absolute differences) for video codecs
-   Saturating arithmetic for image processing
-   Integer/BF16 multiply-accumulate for machine learning
-   Predicated/masked arithmetic and memory ops for correctness and performance
-   Conversions between masks and bits for data structures
-   Cryptographic primitives (AES, carryless multiply) for browsers and hashing
-   Interleave/deinterleave operations for SoA

Highway supports all of these, >400 ops in total, covering the spectrum of
real-world usage. Crucially, we are able to add ops near-immediately whenever a
new use case arises, without multiple committee meetings. The author has
experience with ISO standardization and believes it is a poor fit for SIMD,
where usage and patterns continue to evolve.

### Universal availability

The seeming advantage of standardization, mentioned by its proponents, is that
it will *eventually* be available on all conforming C++ implementations without
any third-party dependency.

However, our experience is that numerous users are obliged to support older
environments and compilers. It has taken until 2026 for C++17 (at least the
language portion) to look like a viable baseline. Thus the main advantage of
`std::simd` turns out to be a weakness: years will go by before it is actually
supported on all platforms, and OS distributions have updated.

Even then, the problem of QoI (Quality of Implementation) remains. Standard
library implementers are reportedly understaffed and already struggling with the
ever-increasing scope of C++. Worse, SIMD requires domain expertise. Forcing
libraries to include and develop SIMD will likely worsen the historical
observation that implementation quality varies across platforms, which erodes
the argument of portability.

Consider also the "dependency-free" argument. Supply-chain vulnerabilities and
dependency/version management are real concerns. However, the reality is that
Highway is considered critical infrastructure by Google's security team, and
thus benefits from security tooling, code reviews, and rapid patches given its
importance to internal Google software. Few libraries can claim this level of
organizational backing.

Finally, the dependency argument is weaker than it sounds. Highway is quite
universally available, including all major package managers. Given that it
builds with CMake, Bazel and Meson, the integration cost is trivial for many
projects.

## Addressing Criticisms of Highway

Bucher's article raises points about Highway to which we would like to respond.

### "The API is verbose and idiosyncratic"

Bucher mentions `hn::Mul(d, a, b)`. The reason for `Mul` instead of `operator*`
was originally compiler limitations on SVE/RVV. On recent compilers, `operator*`
works and is indeed supported by Highway.

Note that our `Mul` doesn't take a `d` argument. This tag is only passed to
operations such as `Load(d, ptr)` and `UpperHalf(d, v)` that require type or
length information. Differentiating between the actual vector data, a (sometimes
weakly typed) built-in type, and the sizeless tag encoding element type and
length, is the unique design characteristic of Highway that allows using
scalable vectors natively.

### "HWY_DYNAMIC_DISPATCH macros fragment your source"

Partially true: you need about twenty lines of boilerplate, including
`HWY_BEFORE_NAMESPACE` and `HWY_EXPORT`. These are also finicky and best copied
from an existing source. However, this is a one-time cost per module, and I
believe we've converged on the best possible approach in C++. The current
alternative is maintaining one source file or implementation per target, or not
supporting runtime dispatch at all, like `std::simd`.

### "Bus factor — small team, Google priorities"

The concern is understandable. However, there are 30 contributors with five or
more pull requests. Highway is considered "critical infrastructure" within
Google and also heavily used externally. The founder/TL is highly engaged, which
reduces the need for a larger team. However, if incapacitated, I am confident
someone would step up — the project is too important to too many organizations
to be abandoned.

### "Can't easily express fixed-width algorithms"

Highway actually does offer `FixedTag<T, N>` for algorithms that require exactly
N lanes, e.g. `FixedTag<uint8_t, 16>` for AES. To ensure portability, only
128-bit vectors (`N = 16 / sizeof(T)`) are guaranteed to be supported, because
all targets have them.

## Summary and recommendations

From my personal viewpoint, it is difficult to understand why C++
standardization of SIMD has proceeded as it has. Let us consider where it might
be used in future:

-   Large compute-intensive projects have likely already adopted some form of
    SIMD. Why should they switch to a more limited and problematic library that
    lacks support for commonly-used ops and runtime dispatch?

-   Application developers that don't care as much about performance are
    unlikely to impose on themselves the additional learning curve and
    complexity. For simple loops, they might be better served by
    auto-vectorization, though this is still not bulletproof.

`std::simd` thus has a missing-middle problem. It is too limited for serious
usage, but also too complex or different to be casually adopted.

In fairness, one advantage it does have is support for constexpr. This ensures
(table) initializers can run at compile time, and allows `static_assert`. By
contrast, Highway inherits the same lack of constexpr support from intrinsics.
However, this is a non-issue in our experience because initializers are
typically hoisted out of time-critical code, and tests can run separately.

Moreover, `std::simd` comes at least five years after the widespread
availability of battle-tested libraries. Highway is, to the best of our
knowledge, the most widely used, also with the widest range of supported ops and
platforms.

An [independent evaluation](https://www.mnm-team.org/pub/Fopras/rock23/)
concluded: "Highway excelled with a strong performance across multiple SIMD
extensions [...]. Thus, Highway may currently be the most suitable SIMD library
for many software projects."

It was designed with an eye towards easily porting from existing x86 and Arm
intrinsics, which to a large degree have 1:1 equivalents. There is an immediate
benefit to doing so: demonstrated portability to numerous platforms, as opposed
to the as-yet unproven hope of `std::simd`.

Of course a founder/TL might be biased toward their work, but I can without
hesitation recommend adopting Highway for most projects that care about
performance. But what about the most time-critical kernels, where you might want
to use specific non-portable instructions? Highway helps there, too: you can
specialize your implementation for certain platforms, sometimes even using
type-safe wrappers such as `PerBlock2x2MatMul` over Arm's MMLA instructions. At
the very least, a portable implementation of the less-critical parts of your
kernel saves development time that can be reinvested in other optimizations.

One final thought: might coding agents change the landscape? They might greatly
reduce the toil of porting intrinsics or assembly to other platforms. However,
even if quick to write, all the resulting code still wants to be maintained.
First, the resulting code is less readable/understandable than Highway's clean
intrinsics. Second, all that code has to be validated and benchmarked after each
agent-driven change. By contrast, an implementation using Highway has fairly
strong implications of correctness and reasonable performance once written for
any one platform.

--------------------------------------------------------------------------------

## Appendix: std::simd vs Highway feature comparison

| Feature               | std::simd (C++26)       | Google Highway            |
| --------------------- | :---------------------: | :-----------------------: |
| Runtime dispatch      | ❌                       | ✅ Built-in                |
| Scalable vectors      | ❌ Fixed-width only      | ✅ Length-agnostic, uses   |
: (SVE/RVV)             : (class can't wrap       : built-in vector types     :
:                       : sizeless types)         :                           :
| Permutations          | ✅ `permute` (static +   | ✅ `TableLookupLanes`,     |
:                       : dynamic)                : many others               :
| Compress/Expand       | ✅ `compress` / `expand` | ✅ `Compress`,             |
:                       :                         : `CompressStore`,          :
:                       :                         : `LoadExpand`, `Expand`    :
| Byte table lookups    | ❌                       | ✅ `TableLookupBytes`      |
| Gather/Scatter        | ✅ `gather_from` /       | ✅ `GatherIndex` /         |
:                       : `scatter_to`            : `ScatterIndex`            :
| Saturating arithmetic | ❌                       | ✅ `SaturatedAdd` /        |
:                       :                         : `SaturatedSub`            :
| SAD (sum of absolute  | ❌                       | ✅ `SumsOf8AbsDiff`,       |
: differences)          :                         : `SumsOfAdjQuadAbsDiff`    :
| Multiply-accumulate   | ❌                       | ✅ `MulAdd`,               |
:                       :                         : `SumOfMulQuadAccumulate`, :
:                       :                         : etc.                      :
| Conversion            | ✅                       | ✅ Also two-input demote   |
| Interleave            | ❌                       | ✅ `InterleaveLower`,      |
:                       :                         : `InterleaveUpper`, etc.   :
| Pairwise add/sub      | ❌                       | ✅ `PairwiseAdd` /         |
:                       :                         : `PairwiseSub`             :
| Cryptographic ops     | ❌                       | ✅ `AESRound`, `CLMul`,    |
:                       :                         : etc.                      :
| Conditional selection | ✅ `select`              | ✅ `IfThenElse`,           |
:                       :                         : `IfVecThenElse`,          :
:                       :                         : `IfThenZeroElse`          :
| Masked operations     | ❌ Only `where`: no      | ✅ `MaskedGatherIndexOr`,  |
:                       : guarantee of masked     : `MaskedDivOr`, etc.       :
:                       : *compute*               :                           :
| Partial load/store    | ✅ `partial_load` /      | ✅ `LoadN` / `StoreN`      |
:                       : `partial_store`         :                           :
| Horizontal reductions | ✅ `reduce`,             | ✅ `ReduceSum`,            |
:                       : `reduce_min`,           : `ReduceMin`, `ReduceMax`  :
:                       : `reduce_max`            :                           :
| Mask reductions       | ✅ `all_of`, `any_of`,   | ✅ `AllTrue`, `AllFalse`,  |
:                       : `none_of`,              : `CountTrue`,              :
:                       : `reduce_count`          : `FindFirstTrue`, etc.     :
| Math overloads        | ✅ All `<cmath>`         | ✅ Most `<cmath>` in       |
:                       :                         : `hwy/contrib/math`        :
| Bit manipulation      | ✅ All `<bit>`           | ✅ `PopCount`, `Clz`,      |
:                       :                         : `Ctz`, etc.               :
| Compiler support      | GCC only, others in     | GCC, Clang, MSVC, ICC     |
:                       : progress                :                           :
| Production users      | None known              | Chromium, Firefox, JPEG   |
:                       :                         : XL, libaom, TensorFlow,   :
:                       :                         : NumPy, V8, ...            :
| Arm NEON              | ✅                       | ✅                         |
| Arm SVE/SVE2          | ❌                       | ✅                         |
| IBM Z                 | ❌                       | ✅ (Z14, Z15)              |
| LoongArch             | ❌                       | ✅ (LSX, LASX)             |
| POWER                 | ❌                       | ✅ (PPC8, PPC9, PPC10)     |
| RISC-V RVV            | ❌                       | ✅                         |
| WebAssembly SIMD      | ❌                       | ✅ (WASM, WASM_EMU256)     |
| x86                   | ✅                       | ✅ SSE2 through AVX-512,   |
:                       :                         : AVX10                     :
