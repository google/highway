## Efficient and performance-portable SIMD wrapper

This library provides type-safe and source-code portable wrappers over existing
platform-specific intrinsics. Its design aims for simplicity, reliable
efficiency across platforms, and immediate usability with current compilers.

## Current status

Implemented for scalar/SSE4/AVX2/AVX-512/ARMv8 targets, each with unit tests.

A [quick-reference page](quick_reference.md) briefly lists all operations
and their parameters.

## Design philosophy

*   Performance is important but not the sole consideration. Anyone who goes to
    the trouble of using SIMD clearly cares about speed. However, portability,
    maintainability and readability also matter, otherwise we would write in
    assembly. We aim for performance within 10-20% of a hand-written assembly
    implementation on the development platform.

*   The guiding principles of C++ are "pay only for what you use" and "leave no
    room for a lower-level language below C++". We apply these by defining a
    SIMD API that ensures operation costs are visible, predictable and minimal.

*   Performance portability is important, i.e. the API should be efficient on
    all target platforms. Unfortunately, common idioms for one platform can be
    inefficient on others. For example: summing lanes horizontally versus
    shuffling. Documenting which operations are expensive does not prevent their
    use, as evidenced by widespread use of `HADDPS`. Performance acceptance
    tests may detect large regressions, but do not help choose the approach
    during initial development. Analysis tools can warn about some potential
    inefficiencies, but likely not all. We instead provide [a carefully chosen
    set of vector types and operations that are efficient on all target
    platforms][instmtx] (PPC8, SSE4/AVX2+, ARMv8), plus some useful but less
    performance-portable operations in an `ext` namespace to make their cost
    visible.

*   Future SIMD hardware features are difficult to predict. For example, AVX2
    came with surprising semantics (almost no interaction between 128-bit
    blocks) and AVX-512 added two kinds of predicates (writemask and zeromask).
    To ensure the API reflects hardware realities, we suggest a flexible
    approach that adds new operations as they become commonly available.

*   Masking is not yet widely supported on current CPUs. It is difficult to
    define an interface that provides access to all platform features while
    retaining performance portability. The P0214R5 proposal lacks support for
    AVX-512/ARM SVE zeromasks. We suggest standardizing masked operations only
    after the community has gained more experience with them.

*   "Width-agnostic" SIMD is more future-proof than user-specified fixed sizes.
    For example, valarray-like code can iterate over a 1D array with a
    library-specified vector width. This will result in better code when vector
    sizes increase, and matches the direction taken by
    [ARM SVE](https://alastairreid.github.io/papers/sve-ieee-micro-2017.pdf) and
    RiscV hardware as well as Agner Fog's
    [ForwardCom instruction set proposal](https://goo.gl/CFizWu). However, some
    applications may require fixed sizes, so we also guarantee support for
    128-bit vectors in each instruction set.

*   The API and its implementation should be usable and efficient with commonly
    used compilers. Some of our open-source users cannot upgrade, so we need to
    support ~4 year old compilers. For example, we write `shift_left<3>(v)`
    instead of `v << 3` because MSVC 2017 (ARM64) does not propagate the literal
    (https://godbolt.org/g/rKx5Ga). However, we do require function-specific
    target attributes, supported by GCC 4.9 / Clang 3.9 / MSVC 2015.

*   Efficient and safe runtime dispatch is important. Modules such as image or
    video codecs are typically embedded into larger applications such as
    browsers, so they cannot require separate binaries for each CPU. Libraries
    also cannot predict whether the application already uses AVX2 (and pays the
    frequency throttling cost), so this decision must be left to the
    application. Using only the lowest-common denominator instructions
    sacrifices too much performance.
    Therefore, we provide code paths for multiple instruction sets and choose
    the most suitable at runtime. To reduce overhead, dispatch should be hoisted
    to higher layers instead of checking inside every low-level function.
    Generating each code path from the same source reduces implementation- and
    debugging cost.

*   Not every CPU need be supported. For example, pre-SSE4.1 CPUs are
    increasingly rare and the AVX instruction set is limited to floating-point
    operations. To reduce code size and compile time, we provide specializations
    for SSE4, AVX2 and AVX-512 instruction sets on x86.

*   Access to platform-specific intrinsics is necessary for acceptance in
    performance-critical projects. We provide conversions to and from intrinsics
    to allow utilizing specialized platform-specific functionality such as
    `MPSADBW`, and simplify incremental porting of existing code.

*   The core API should be compact and easy to learn. We provide only the few
    dozen operations which are necessary and sufficient for most of the 150+
    SIMD applications we examined.

## Differences versus [P0214R5 proposal](https://goo.gl/zKW4SA)

1.  Adding widely used and portable operations such as `andnot`, `average`,
    bit-shift by immediates and `if_then_else`.

1.  Designing the API to avoid or minimize overhead on AVX2/AVX-512 caused by
    crossing 128-bit 'block' boundaries.

1.  Avoiding the need for non-native vectors. By contrast, P0214R5's `simd_cast`
    returns `fixed_size<>` vectors which are more expensive to access because
    they reside on the stack. We can avoid this plus additional overhead on
    ARM/AVX2 by defining width-expanding operations as functions of a vector
    part, e.g. promoting half a vector of `uint8_t` lanes to one full vector of
    `uint16_t`, or demoting full vectors to half vectors with half-width lanes.

1.  Guaranteeing access to the underlying intrinsic vector type. This ensures
    all platform-specific capabilities can be used. P0214R5 instead only
    'encourages' implementations to provide access.

1.  Enabling safe runtime dispatch and inlining in the same binary. P0214R5 is
    based on the Vc library, which does not provide assistance for linking
    multiple instruction sets into the same binary. The Vc documentation
    suggests compiling separate executables for each instruction set or using
    GCC's ifunc (indirect functions). The latter is compiler-specific and risks
    crashes due to ODR violations when compiling the same function with
    different compiler flags. We solve this problem via target-specific
    attributes (see HOWTO section below). We also permit a mix of static
    target selection and runtime dispatch for hotspots that may benefit from
    newer instruction sets if available.

1.  Using built-in PPC vector types without a wrapper class. This leads to much
    better code generation with GCC 6.3: https://godbolt.org/z/pd2PNP.
    By contrast, P0214R5 requires a wrapper. We avoid this by using only the
    member operators provided by the PPC vectors; all other functions and
    typedefs are non-members. 2019-04 update: Clang power64le does not have
    this issue, so we simplified get_part(d, v) to get_lane(v).

*   Omitting inefficient or non-performance-portable operations such as `hmax`,
    `operator[]`, and unsupported integer comparisons. Applications can often
    replace these operations at lower cost than emulating that exact behavior.

*   Omitting `long double` types: these are not commonly available in hardware.

*   Ensuring signed integer overflow has well-defined semantics (wraparound).

*   Simple header-only implementation and less than a tenth of the size of the
    Vc library from which P0214 was derived (98,000 lines in
    https://github.com/VcDevel/Vc according to the gloc Chrome extension).

*   Avoiding hidden performance costs. P0214R5 allows implicit conversions from
    integer to float, which costs 3-4 cycles on x86. We make these conversions
    explicit to ensure their cost is visible.

## Prior API designs

The author has been writing SIMD code since 2002: first via assembly language,
then intrinsics, later Intel's `F32vec4` wrapper, followed by three generations
of custom vector classes. The first used macros to generate the classes, which
reduces duplication but also readability. The second used templates instead.
The third (used in highwayhash and PIK) added support for AVX2 and runtime
dispatch. The current design enables code generation for multiple platforms
and/or instruction sets from the same source, and improves runtime dispatch.

## Other related work

*   [Neat SIMD](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7568423)
    adopts a similar approach with interchangeable vector/scalar types and
    a compact interface. It allows access to the underlying intrinsics, but
    does not appear to be designed for other platforms than x86.

*   UME::SIMD ([code](https://goo.gl/yPeVZx), [paper](https://goo.gl/2xpZrk))
    also adopts an explicit vectorization model with vector classes.
    However, it exposes the union of all platform capabilities, which makes the
    API harder to learn (209-page spec) and implement (the estimated LOC count
    is [500K](https://goo.gl/1THFRi)). The API is less performance-portable
    because it allows applications to use operations that are inefficient on
    other platforms.

*   Inastemp ([code](https://goo.gl/hg3USM), [paper](https://goo.gl/YcTU7S))
    is a vector library for scientific computing with some innovative features:
    automatic FLOPS counting, and "if/else branches" using lambda functions.
    It supports IBM Power8, but only provides float and double types.

### Overloaded function API

Most C++ vector APIs rely on class templates. However, two PPC compilers
including GCC 6.3 generate inefficient code for classes with a SIMD vector
member: an [extra load/store for every function argument/return
value](https://godbolt.org/z/pd2PNP). To avoid this overhead, we use built-in
vector types on PPC. These provide overloaded arithmetic operators but do not
allow member functions/typedefs such as `size()` or `value_type`. We instead
rely on overloaded functions.

Because full vectors and parts are synonyms on PPC, we need an additional tag
argument for disambiguation. Any function template with multiple return types
uses a descriptor argument to specify the return type. For example, the return
type of `setzero(Desc<T, N>)` is `VT<Desc<T, N>>`. For brevity,
`Desc` is abbreviated to `D` for template arguments and `d` in lvalues.

It may seem preferable to write `setzero<D>()` rather than `setzero(D())`, but
there are technical difficulties. We prefer generic implementations where
possible rather than overloading for every single `T`. Because C++ does not
allow partial specialization of function templates, we need multiple overloads:
one primary template per target. Thus, functions cannot be invoked using
template syntax. Can we instead add a wrapper function template that calls the
appropriate overload? Unfortunately, the compiler mechanism for avoiding
dangerous per-file `-mavx2` requires per-function annotations, and these
attributes are not generic. Thus, a wrapper into which SIMD functions are
inlined cannot be a function, because it would also need a target-specific
attribute. A macro `SETZERO(D)` could work, but this is hardly more clear than a
normal function with arguments. Note that descriptors occur often, so user code
can define a `const SIMD_FULL(float) d;` and then write `setzero(d)`.

## Use cases and HOWTO

Whenever possible, use full-width descriptors for maximum performance
portability: `SIMD_FULL(float)`. When necessary, applications may also rely on
128-bit vectors: `Desc<float, 4>`. There is also the option of a vector of up
to N lanes: `SIMD_CAPPED(float, N)` (the length is always a power of two).

*   Single instruction set per platform: use normal C++ functions with
    `SIMD_ATTR` annotation and `#include "static_targets.h"`.

*   Runtime dispatch (see example code in `runtime_dispatch.h`): move
    target-specific code into a separate *_target.cc file, which includes
    `in_target.h`. Specialize an `operator()<SIMD_TARGET>` in this file; any
    internal implementation must reside in `namespace SIMD_NAMESPACE` to avoid
    ODR violations. All functions must still be prefixed with `SIMD_ATTR`.
    Include the file for each target via `foreach_target.h`. To call,
    `#include "runtime_dispatch"` and choose the 'best' available target for
    the current CPU via `Dispatch(TargetBitfield.Best(), ...)`.

*   Mixing static and runtime dispatch: `static_targets.h` and
    `runtime_dispatch.h` may be included into the same translation unit.
    `static_targets.h` may also be included *before* (typically in headers that
    unconditionally use SIMD) `in_target.h`.

## Demos

`bin/simd_test` prints a bitfield of instruction sets that were
tested, e.g. `6` for SSE4=`4` and AVX2=`2`. The demo compiles the same source
file once per enabled instruction set.

## Example source code

```c++
void FloorLog2(const uint8_t* SIMD_RESTRICT values,
               uint8_t* SIMD_RESTRICT log2) {
  // Descriptors for all required data types:
  const SIMD_FULL(int32_t) d32;
  const SIMD_FULL(float) df;
  const SIMD_CAPPED(uint8_t, d32.N) d8;

  const auto u8 = load(d8, values);
  const auto bits = bit_cast(d32, convert_to(df, convert_to(d32, u8)));
  const auto exponent = shift_right<23>(bits) - set1(d32, 127);
  store(convert_to(d8, exponent), d8, log2);
}
```

This generates the following SSE4 and AVX2 code, as shown by IACA:

```
 p0  p1  p5
|   |   | 1 | CP | pmovzxbd xmm1, dword [rsp+0x25c]
|   | 1 |   |    | cvtdq2ps xmm1, xmm1
| 1 |   |   |    | psrad xmm1, 0x17
|   | 1 |   |    | paddd xmm1, xmm0
|   |   | 1 | CP | packusdw xmm1, xmm0
|   |   | 1 | CP | packuswb xmm1, xmm0
|   |   |   |    | movd [rsp+0x45c], xmm1

|   |   | 1 | CP | vpmovzxbd ymm1, qword [rsp+0x228]
|   | 1 |   |    | vcvtdq2ps ymm1, ymm1
| 1 |   |   |    | vpsrad ymm1, ymm1, 0x17
|   | 1 |   |    | vpaddd ymm1, ymm1, ymm0
|   |   | 1 | CP | vpackusdw ymm1, ymm1, ymm0
|   |   | 1 | CP | vpermq ymm1, ymm1, 0xe8
|   |   | 1 | CP | vpackuswb xmm1, xmm1, xmm0
|   |   |   |    | vmovq [rsp+0x448], xmm1
```

```c++
void Copy(const uint8_t* SIMD_RESTRICT from, const size_t size,
          uint8_t* SIMD_RESTRICT to) {
  // Width-agnostic (library-specified N)
  const SIMD_FULL(uint8_t) d;
  const Scalar<uint8_t> ds;
  size_t i = 0;
  for (; i + d.N <= size; i += d.N) {
    const auto bytes = load(d, from + i);
    store(bytes, d, to + i);
  }

  for (; i < size; ++i) {
    // (Same loop body as above, could factor into a shared template)
    const auto bytes = load(ds, from + i);
    store(bytes, ds, to + i);
  }
}
```

```c++
void MulAdd(const T* SIMD_RESTRICT mul_array, const T* SIMD_RESTRICT add_array,
            const size_t size, T* SIMD_RESTRICT x_array) {
  // Type-agnostic (caller-specified lane type) and width-agnostic (uses
  // best available instruction set).
  const SIMD_FULL(T) d;
  for (size_t i = 0; i < size; i += d.N) {
    const auto mul = load(d, mul_array + i);
    const auto add = load(d, add_array + i);
    auto x = load(d, x_array + i);
    x = mul_add(mul, x, add);
    store(x, d, x_array + i);
  }
}
```

## Additional resources

*   [Overview of instructions per operation on different architectures][instmtx]

[instmtx]: instruction_matrix.pdf

This is not an officially supported Google product.
Contact: janwas@google.com
