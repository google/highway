# Highway Examples

This directory contains examples demonstrating how to use the Highway SIMD
library.

## Introductory examples

### `sum_array_simple.cc`

Minimal code demonstrating how to sum an array of float, with a simple scalar
fallback for remainders.

### `sum_array_advanced.cc`

Higher-performance array sum, including:

-   Loop unrolling (factor of 4) for better performance
-   Use of `LoadN` with `FirstN` mask for remainder handling without scalar
    fallbacks
-   Runtime checks and validation against a scalar implementation.

### `dot_product_unroll.cc`

Similar to sum_array_advanced, but for dot products. Adds:

-   Fused multiply-add `MulAdd`
-   Duff's device for vector remainders
-   Horizontal sum (reduction/fold) using `ReduceSum`.

### `dot_product_mixed_precision.cc`

Similar to array simple but uses type promotion to allow for
a greater range in resulting integer sum values before an
overflow occurs.

### `matrix_transpose_scatter_gather.cc`

Matrix transposition via Gather and Scatter, showing:

-   Loading/storing elements from/to non-contiguous memory via `GatherIndex` and
    `ScatterIndex`
-   Remainder processing using `LoadN`/`StoreN` and
    `GatherIndexN`/`ScatterIndexN`
-   Precomputing strided offsets in registers using `Iota` and `Mul`.

### `stream_triad.cc`

Addition of a vector to a scaled copy of another vector, showing:

-  Multithreading.

## Infrastructure

### `benchmark.cc`

Dot product and delta coding, with benchmarking infrastructure and aligned
allocation.

### `profiler_example.cc`

Shows how to use the our built-in profiler for measuring the elapsed time in
annotated zones, deducting the time spent in nested zones.

### `skeleton*`

A complete example of a module with support for runtime dispatch, and a
'per-target header' for inlining SIMD into multiple .cc files.

## Challenge examples

### `masks_and_logic.cc`

ASCII art renderer, demonstrating:

-   Branching and masking within register
-   Boolean operations on masks (`And`, `AndNot`)
-   Chaining `IfThenElse` for nested conditions
-   Using a lambda function with `HWY_ATTR` for SIMD operations.

### `ctf_aes.cc`

Brute-force password guessing, with:

-   `FixedTag` for fixed-length vectors
-   Portable hardware-accelerated AES round operations (`AESRound`)
-   Mask generation via `FirstN` and masked comparisons via `MaskedEq`.

## How to Run

### Using Bazel
To run the examples using Bazel:

```bash
bazel run //::sum_array_simple
bazel run //::sum_array_advanced
# etc; see BUILD file for the other build targets
```

### Using CMake
If you are building Highway with CMake (from the root of the highway directory):

```bash
mkdir build && cd build
cmake .. -DHWY_ENABLE_EXAMPLES=ON
make
./examples/sum_array_simple
./examples/sum_array_advanced
# etc; see CMakeLists.txt for the other build targets
```

### Using Clang directly
To compile and run using `clang++` (from the root of the highway directory):

```bash
clang++ -std=c++17 -O3 -I. hwy/examples/sum_array_simple.cc hwy/targets.cc hwy/per_target.cc hwy/print.cc hwy/abort.cc hwy/aligned_allocator.cc -o sum_array_simple

./sum_array_simple
```

### Using GCC directly
To compile and run using `g++` (from the root of the highway directory):

```bash
g++ -std=c++17 -O3 -I. hwy/examples/sum_array_simple.cc hwy/targets.cc hwy/per_target.cc hwy/print.cc hwy/abort.cc hwy/aligned_allocator.cc -o sum_array_simple

./sum_array_simple
```

*Note: `g++` might emit some assembler warnings like `no SFrame FDE emitted`,
which are benign and can be ignored.*
