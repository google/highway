Efficient and performance-portable vector software
==================================================

Highway is a C++ library that provides portable SIMD/vector intrinsics.

`Documentation <https://google.github.io/highway/en/master/>`__

Previously licensed under Apache 2, now dual-licensed as Apache 2 /
BSD-3.

Why
---

We are passionate about high-performance software. We see major untapped
potential in CPUs (servers, mobile, desktops). Highway is for engineers
who want to reliably and economically push the boundaries of what is
possible in software.

How
---

CPUs provide SIMD/vector instructions that apply the same operation to
multiple data items. This can reduce energy usage e.g. *fivefold*
because fewer instructions are executed. We also often see *5-10x*
speedups.

Highway makes SIMD/vector programming practical and workable according
to these guiding principles:

**Does what you expect**: Highway is a C++ library with carefully-chosen
functions that map well to CPU instructions without extensive compiler
transformations. The resulting code is more predictable and robust to
code changes/compiler updates than autovectorization.

**Works on widely-used platforms**: Highway supports seven
architectures; the same application code can target various instruction
sets, including those with ‘scalable’ vectors (size unknown at compile
time). Highway only requires C++11 and supports four families of
compilers. If you would like to use Highway on other platforms, please
raise an issue.

**Flexible to deploy**: Applications using Highway can run on
heterogeneous clouds or client devices, choosing the best available
instruction set at runtime. Alternatively, developers may choose to
target a single instruction set without any runtime overhead. In both
cases, the application code is the same except for swapping
``HWY_STATIC_DISPATCH`` with ``HWY_DYNAMIC_DISPATCH`` plus one line of
code. See also @kfjahnke’s `introduction to
dispatching <https://github.com/kfjahnke/zimt/blob/main/examples/multi_isa_example/multi_simd_isa.md>`__.

**Suitable for a variety of domains**: Highway provides an extensive set
of operations, used for image processing (floating-point), compression,
video analysis, linear algebra, cryptography, sorting and random
generation. We recognise that new use-cases may require additional ops
and are happy to add them where it makes sense (e.g. no performance
cliffs on some architectures). If you would like to discuss, please file
an issue.

**Rewards data-parallel design**: Highway provides tools such as Gather,
MaskedLoad, and FixedTag to enable speedups for legacy data structures.
However, the biggest gains are unlocked by designing algorithms and data
structures for scalable vectors. Helpful techniques include batching,
structure-of-array layouts, and aligned/padded allocations.

We recommend these resources for getting started:

-  `SIMD programming with Highway
   talk <https://www.youtube.com/watch?v=R57biOOhnJM>`__
-  `SIMD for C++ Developers <http://const.me/articles/simd/simd.pdf>`__
-  `Algorithms for Modern Hardware <https://en.algorithmica.org/hpc/>`__
-  `Optimizing software in
   C++ <https://agner.org/optimize/optimizing_cpp.pdf>`__
-  `Improving performance with SIMD intrinsics in three use
   cases <https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/>`__

Examples
--------

Online demos using Compiler Explorer:

-  `multiple targets with dynamic
   dispatch <https://gcc.godbolt.org/z/KM3ben7ET>`__ (more complicated,
   but flexible and uses best available SIMD)
-  `single target using -m
   flags <https://gcc.godbolt.org/z/rGnjMevKG>`__ (simpler, but
   requires/only uses the instruction set enabled by compiler flags)

We observe that Highway is referenced in the following open source
projects, found via sourcegraph.com. Most are GitHub repositories. If
you would like to add your project or link to it directly, feel free to
raise an issue or contact us via the below email.

-  Audio: `Zimtohrli perceptual
   metric <https://github.com/google/zimtohrli>`__
-  Browsers: Chromium (+Vivaldi), Firefox (+floorp / foxhound /
   librewolf / Waterfox)
-  Computational biology: `RNA
   analysis <https://github.com/bnprks/BPCells>`__, `long-sequence
   preprocessing <https://github.com/OpenGene/fastplong>`__
-  Computer graphics: ghostty-org/ghostty, `Sparse voxel
   renderer <https://github.com/rools/voxl>`__, `tgfx 2D Graphics
   library <https://github.com/Tencent/tgfx>`__
-  Cryptography: google/distributed_point_functions,
   google/shell-encryption
-  Data structures: bkille/BitLib
-  Image codecs: eustas/2im, `Grok JPEG
   2000 <https://github.com/GrokImageCompression/grok>`__, `JPEG
   XL <https://github.com/libjxl/libjxl>`__,
   `JPEGenc <https://github.com/osamu620/JPEGenc>`__,
   `Jpegli <https://github.com/google/jpegli>`__,
   `libaom <https://aomedia.googlesource.com/aom/>`__,
   `OpenHTJ2K <https://github.com/osamu620/OpenHTJ2K>`__
-  Image processing: awxkee/aire, cloudinary/ssimulacra2,
   `libvips <https://github.com/libvips/libvips>`__,
   m-ab-s/media-autobuild_suite,
-  Image viewers: AlienCowEatCake/ImageViewer, diffractor/diffractor,
   `Lux panorama/image viewer <https://bitbucket.org/kfj/pv/>`__,
   mirillis/jpegxl-wic
-  Information retrieval: `iresearch database
   index <https://github.com/iresearch-toolkit/iresearch>`__,
   michaeljclark/zvec, `nebula interactive analytics /
   OLAP <https://github.com/varchar-io/nebula>`__, ```ScaNN`` Scalable
   Nearest
   Neighbors <https://github.com/google-research/google-research/tree/7a269cb2ce0ae1db591fe11b62cbc0be7d72532a/scann>`__,
-  Machine learning: array2d/deepx,
   `gemma.cpp <https://github.com/google/gemma.cpp>`__, Tensorflow,
   Numpy, zpye/SimpleInfer
-  Programming languages: `AOT-compiled
   python <https://github.com/exaloop/codon>`__, oven-sh/bun, V8/V8,
   yinqiwen/rapidudf
-  Robotics: `MIT Model-Based Design and
   Verification <https://github.com/RobotLocomotion/drake>`__
-  Vector search: 1yefuwang1/vectorlite, vespa-engine/vespa

Other

-  `Evaluation of C++ SIMD
   Libraries <https://www.mnm-team.org/pub/Fopras/rock23/>`__: “Highway
   excelled with a strong performance across multiple SIMD extensions
   [..]. Thus, Highway may currently be the most suitable SIMD library
   for many software projects.”
-  `zimt <https://github.com/kfjahnke/zimt>`__: C++11 template library
   to process n-dimensional arrays with multi-threaded SIMD code
-  `vectorized
   Quicksort <https://github.com/google/highway/tree/master/hwy/contrib/sort>`__
   (`paper <https://arxiv.org/abs/2205.05982>`__)

If you’d like to get Highway, in addition to cloning from this GitHub
repository or using it as a Git submodule, you can also find it in the
following package managers or repositories:

-  alpinelinux
-  conan-io
-  conda-forge
-  DragonFlyBSD,
-  fd00/yacp
-  freebsd
-  getsolus/packages
-  ghostbsd
-  microsoft/vcpkg
-  MidnightBSD
-  MSYS2
-  NetBSD
-  openSUSE
-  opnsense
-  Xilinx/Vitis_Libraries
-  xmake-io/xmake-repo

See also the list at
https://repology.org/project/highway-simd-library/versions .

Current status
--------------

Targets
~~~~~~~

Highway supports 27 targets, listed in alphabetical order of platform:

-  Any: ``EMU128``, ``SCALAR``;
-  Armv7+: ``NEON_WITHOUT_AES``, ``NEON``, ``NEON_BF16``, ``SVE``,
   ``SVE2``, ``SVE_256``, ``SVE2_128``;
-  IBM Z: ``Z14``, ``Z15``;
-  LoongArch: ``LSX``, ``LASX``;
-  POWER: ``PPC8`` (v2.07), ``PPC9`` (v3.0), ``PPC10`` (v3.1B, not yet
   supported due to compiler bugs, see #1207; also requires QEMU 7.2);
-  RISC-V: ``RVV`` (1.0);
-  WebAssembly: ``WASM``, ``WASM_EMU256`` (a 2x unrolled version of
   wasm128, enabled if ``HWY_WANT_WASM2`` is defined. This will remain
   supported until it is potentially superseded by a future version of
   WASM.);
-  x86:

   -  ``SSE2``
   -  ``SSSE3`` (~Intel Core)
   -  ``SSE4`` (~Nehalem, also includes AES + CLMUL).
   -  ``AVX2`` (~Haswell, also includes BMI2 + F16 + FMA)
   -  ``AVX3`` (~Skylake, AVX-512F/BW/CD/DQ/VL)
   -  ``AVX3_DL`` (~Icelake, includes ``BitAlg`` + ``CLMUL`` + ``GFNI``
      + ``VAES`` + ``VBMI`` + ``VBMI2`` + ``VNNI`` + ``VPOPCNT``),
   -  ``AVX3_ZEN4`` (AVX3_DL plus BF16, optimized for AMD Zen4; requires
      opt-in by defining ``HWY_WANT_AVX3_ZEN4`` if compiling for static
      dispatch, but enabled by default for runtime dispatch),
   -  ``AVX3_SPR`` (~Sapphire Rapids, includes AVX-512FP16)
   -  ``AVX10_2`` (~Diamond Rapids)

Our policy is that unless otherwise specified, targets will remain
supported as long as they can be (cross-)compiled with currently
supported Clang or GCC, and tested using QEMU. If the target can be
compiled with LLVM trunk and tested using our version of QEMU without
extra flags, then it is eligible for inclusion in our continuous testing
infrastructure. Otherwise, the target will be manually tested before
releases with selected versions/configurations of Clang and GCC.

SVE was initially tested using farm_sve (see acknowledgments).

Versioning
~~~~~~~~~~

Highway releases aim to follow the semver.org system
(MAJOR.MINOR.PATCH), incrementing MINOR after backward-compatible
additions and PATCH after backward-compatible fixes. We recommend using
releases (rather than the Git tip) because they are tested more
extensively, see below.

The current version 1.0 signals an increased focus on backwards
compatibility. Applications using documented functionality will remain
compatible with future updates that have the same major version number.

Testing
~~~~~~~

Continuous integration tests build with a recent version of Clang
(running on native x86, or QEMU for RISC-V and Arm) and MSVC 2019
(v19.28, running on native x86).

Before releases, we also test on x86 with Clang and GCC, and Armv7/8 via
GCC cross-compile. See the `testing process <release_testing_process.html>`__ for details.

Related modules
~~~~~~~~~~~~~~~

The ``contrib`` directory contains SIMD-related utilities: an image
class with aligned rows, a math library (16 functions already
implemented, mostly trigonometry), and functions for computing dot
products and sorting.

Other libraries
~~~~~~~~~~~~~~~

If you only require x86 support, you may also use Agner Fog’s `VCL
vector class library <https://github.com/vectorclass>`__. It includes
many functions including a complete math library.

If you have existing code using x86/NEON intrinsics, you may be
interested in `SIMDe <https://github.com/simd-everywhere/simde>`__,
which emulates those intrinsics using other platforms’ intrinsics or
autovectorization.

Installation
------------

This project uses CMake to generate and build. In a Debian-based system
you can install it via:

.. code:: bash

   sudo apt install cmake

Highway’s unit tests use
`googletest <https://github.com/google/googletest>`__. By default,
Highway’s CMake downloads this dependency at configuration time. You can
avoid this by setting the ``HWY_SYSTEM_GTEST`` CMake variable to ON and
installing gtest separately:

.. code:: bash

   sudo apt install libgtest-dev

Alternatively, you can define ``HWY_TEST_STANDALONE=1`` and remove all
occurrences of ``gtest_main`` in each BUILD file, then tests avoid the
dependency on GUnit.

Running cross-compiled tests requires support from the OS, which on
Debian is provided by the ``qemu-user-binfmt`` package.

To build Highway as a shared or static library (depending on
BUILD_SHARED_LIBS), the standard CMake workflow can be used:

.. code:: bash

   mkdir -p build && cd build
   cmake ..
   make -j && make test

Or you can run ``run_tests.sh`` (``run_tests.bat`` on Windows).

Bazel is also supported for building, but it is not as widely
used/tested.

When building for Armv7, a limitation of current compilers requires you
to add ``-DHWY_CMAKE_ARM7:BOOL=ON`` to the CMake command line; see #834
and #1032. We understand that work is underway to remove this
limitation.

Building on 32-bit x86 is not officially supported, and AVX2/3 are
disabled by default there. Note that johnplatts has successfully built
and run the Highway tests on 32-bit x86, including AVX2/3, on GCC 7/8
and Clang 8/11/12. On Ubuntu 22.04, Clang 11 and 12, but not later
versions, require extra compiler flags
``-m32 -isystem /usr/i686-linux-gnu/include``. Clang 10 and earlier
require the above plus
``-isystem /usr/i686-linux-gnu/include/c++/12/i686-linux-gnu``. See
#1279.

Building highway - Using vcpkg
------------------------------

highway is now available in
`vcpkg <https://github.com/Microsoft/vcpkg>`__

.. code:: bash

   vcpkg install highway

The highway port in vcpkg is kept up to date by Microsoft team members
and community contributors. If the version is out of date, please
`create an issue or pull request <https://github.com/Microsoft/vcpkg>`__
on the vcpkg repository.

Quick start
-----------

You can use the ``benchmark`` inside examples/ as a starting point.

A `quick-reference page <quick_reference.html>`__ briefly lists all
operations and their parameters, and the
:download:`instruction_matrix <g3doc/instruction_matrix.pdf>` indicates the
number of instructions per operation.

The `FAQ <faq.html>`__ answers questions about portability, API
design and where to find more information.

We recommend using full SIMD vectors whenever possible for maximum
performance portability. To obtain them, pass a ``ScalableTag<float>``
(or equivalently ``HWY_FULL(float)``) tag to functions such as
``Zero/Set/Load``. There are two alternatives for use-cases requiring an
upper bound on the lanes:

-  For up to ``N`` lanes, specify ``CappedTag<T, N>`` or the equivalent
   ``HWY_CAPPED(T, N)``. The actual number of lanes will be ``N``
   rounded down to the nearest power of two, such as 4 if ``N`` is 5, or
   8 if ``N`` is 8. This is useful for data structures such as a narrow
   matrix. A loop is still required because vectors may actually have
   fewer than ``N`` lanes.

-  For exactly a power of two ``N`` lanes, specify ``FixedTag<T, N>``.
   The largest supported ``N`` depends on the target, but is guaranteed
   to be at least ``16/sizeof(T)``.

Due to ADL restrictions, user code calling Highway ops must either:

-  Reside inside ``namespace hwy { namespace HWY_NAMESPACE {``; or
-  prefix each op with an alias such as
   ``namespace hn = hwy::HWY_NAMESPACE;     hn::Add()``; or
-  add using-declarations for each op used:
   ``using hwy::HWY_NAMESPACE::Add;``.

Additionally, each function that calls Highway ops (such as ``Load``)
must either be prefixed with ``HWY_ATTR``, OR reside between
``HWY_BEFORE_NAMESPACE()`` and ``HWY_AFTER_NAMESPACE()``. Lambda
functions currently require ``HWY_ATTR`` before their opening brace.

Do not use namespace-scope nor ``static`` initializers for SIMD vectors
because this can cause SIGILL when using runtime dispatch and the
compiler chooses an initializer compiled for a target not supported by
the current CPU. Instead, constants initialized via ``Set`` should
generally be local (const) variables.

The entry points into code using Highway differ slightly depending on
whether they use static or dynamic dispatch. In both cases, we recommend
that the top-level function receives one or more pointers to arrays,
rather than target-specific vector types.

-  For static dispatch, ``HWY_TARGET`` will be the best available target
   among ``HWY_BASELINE_TARGETS``, i.e. those allowed for use by the
   compiler (see `quick-reference <quick_reference.html>`__).
   Functions inside ``HWY_NAMESPACE`` can be called using
   ``HWY_STATIC_DISPATCH(func)(args)`` within the same module they are
   defined in. You can call the function from other modules by wrapping
   it in a regular function and declaring the regular function in a
   header.

-  For dynamic dispatch, a table of function pointers is generated via
   the ``HWY_EXPORT`` macro that is used by
   ``HWY_DYNAMIC_DISPATCH(func)(args)`` to call the best function
   pointer for the current CPU’s supported targets. A module is
   automatically compiled for each target in ``HWY_TARGETS`` (see
   `quick-reference <quick_reference.html>`__) if
   ``HWY_TARGET_INCLUDE`` is defined and ``foreach_target.h`` is
   included. Note that the first invocation of ``HWY_DYNAMIC_DISPATCH``,
   or each call to the pointer returned by the first invocation of
   ``HWY_DYNAMIC_POINTER``, involves some CPU detection overhead. You
   can prevent this by calling the following before any invocation of
   ``HWY_DYNAMIC_*``:
   ``hwy::GetChosenTarget().Update(hwy::SupportedTargets());``.

See also a separate `introduction to dynamic
dispatch <https://github.com/kfjahnke/zimt/blob/multi_isa/examples/multi_isa_example/multi_simd_isa.md>`__
by @kfjahnke.

When using dynamic dispatch, ``foreach_target.h`` is included from
translation units (.cc files), not headers. Headers containing vector
code shared between several translation units require a special include
guard, for example the following taken from ``examples/skeleton-inl.h``:

::

   #if defined(HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_) == defined(HWY_TARGET_TOGGLE)
   #ifdef HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_
   #undef HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_
   #else
   #define HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_
   #endif

   #include "hwy/highway.h"
   // Your vector code
   #endif

By convention, we name such headers ``-inl.h`` because their contents
(often function templates) are usually inlined.

Compiler flags
--------------

Applications should be compiled with optimizations enabled. Without
inlining SIMD code may slow down by factors of 10 to 100. For clang and
GCC, ``-O2`` is generally sufficient.

For MSVC, we recommend compiling with ``/Gv`` to allow non-inlined
functions to pass vector arguments in registers. If intending to use the
AVX2 target together with half-width vectors (e.g. for ``PromoteTo``),
it is also important to compile with ``/arch:AVX2``. This seems to be
the only way to reliably generate VEX-encoded SSE instructions on MSVC.
Sometimes MSVC generates VEX-encoded SSE instructions, if they are mixed
with AVX, but not always, see
`DevCom-10618264 <https://developercommunity.visualstudio.com/t/10618264>`__.
Otherwise, mixing VEX-encoded AVX2 instructions and non-VEX SSE may
cause severe performance degradation. Unfortunately, with ``/arch:AVX2``
option, the resulting binary will then require AVX2. Note that no such
flag is needed for clang and GCC because they support target-specific
attributes, which we use to ensure proper VEX code generation for AVX2
targets.

Strip-mining loops
------------------

When vectorizing a loop, an important question is whether and how to
deal with a number of iterations (‘trip count’, denoted ``count``) that
does not evenly divide the vector size ``N = Lanes(d)``. For example, it
may be necessary to avoid writing past the end of an array.

In this section, let ``T`` denote the element type and
``d = ScalableTag<T>``. Assume the loop body is given as a function
``template<bool partial, class D> void LoopBody(D d, size_t index, size_t max_n)``.

“Strip-mining” is a technique for vectorizing a loop by transforming it
into an outer loop and inner loop, such that the number of iterations in
the inner loop matches the vector width. Then, the inner loop is
replaced with vector operations.

Highway offers several strategies for loop vectorization:

-  Ensure all inputs/outputs are padded. Then the (outer) loop is simply

   ::

      for (size_t i = 0; i < count; i += N) LoopBody<false>(d, i, 0);

   Here, the template parameter and second function argument are not
   needed.

   This is the preferred option, unless ``N`` is in the thousands and
   vector operations are pipelined with long latencies. This was the
   case for supercomputers in the 90s, but nowadays ALUs are cheap and
   we see most implementations split vectors into 1, 2 or 4 parts, so
   there is little cost to processing entire vectors even if we do not
   need all their lanes. Indeed this avoids the (potentially large) cost
   of predication or partial loads/stores on older targets, and does not
   duplicate code.

-  Process whole vectors and include previously processed elements in
   the last vector:
   ``for (size_t i = 0; i < count; i += N) LoopBody<false>(d, HWY_MIN(i, count - N), 0);``

   This is the second preferred option provided that ``count >= N`` and
   ``LoopBody`` is idempotent. Some elements might be processed twice,
   but a single code path and full vectorization is usually worth it.
   Even if ``count < N``, it usually makes sense to pad inputs/outputs
   up to ``N``.

-  Use the ``Transform*`` functions in hwy/contrib/algo/transform-inl.h.
   This takes care of the loop and remainder handling and you simply
   define a generic lambda function (C++14) or functor which receives
   the current vector from the input/output array, plus optionally
   vectors from up to two extra input arrays, and returns the value to
   write to the input/output array.

   Here is an example implementing the BLAS function SAXPY
   (``alpha * x + y``):

   ::

      Transform1(d, x, n, y, [](auto d, const auto v, const auto v1) HWY_ATTR {
        return MulAdd(Set(d, alpha), v, v1);
      });

-  Process whole vectors as above, followed by a scalar loop:

   ::

      size_t i = 0;
      for (; i + N <= count; i += N) LoopBody<false>(d, i, 0);
      for (; i < count; ++i) LoopBody<false>(CappedTag<T, 1>(), i, 0);

   The template parameter and second function arguments are again not
   needed.

   This avoids duplicating code, and is reasonable if ``count`` is
   large. If ``count`` is small, the second loop may be slower than the
   next option.

-  Process whole vectors as above, followed by a single call to a
   modified ``LoopBody`` with masking:

   ::

      size_t i = 0;
      for (; i + N <= count; i += N) {
        LoopBody<false>(d, i, 0);
      }
      if (i < count) {
        LoopBody<true>(d, i, count - i);
      }

   Now the template parameter and third function argument can be used
   inside ``LoopBody`` to non-atomically ‘blend’ the first
   ``num_remaining`` lanes of ``v`` with the previous contents of memory
   at subsequent locations:
   ``BlendedStore(v, FirstN(d, num_remaining), d, pointer);``.
   Similarly, ``MaskedLoad(FirstN(d, num_remaining), d, pointer)`` loads
   the first ``num_remaining`` elements and returns zero in other lanes.

   This is a good default when it is infeasible to ensure vectors are
   padded, but is only safe ``#if !HWY_MEM_OPS_MIGHT_FAULT``! In
   contrast to the scalar loop, only a single final iteration is needed.
   The increased code size from two loop bodies is expected to be
   worthwhile because it avoids the cost of masking in all but the final
   iteration.

Additional resources
--------------------

-  :download:`Highway introduction (slides) <g3doc/highway_intro.pdf>`
-  :download:`Overview of instructions per operation on different architectures <g3doc/instruction_matrix.pdf>`
-  `Design philosophy and comparison <design_philosophy.html>`__
-  `Implementation details <impl_details.html>`__

Acknowledgments
---------------

We have used `farm-sve <https://gitlab.inria.fr/bramas/farm-sve>`__ by
Berenger Bramas; it has proved useful for checking the SVE port on an
x86 development machine.

This is not an officially supported Google product. Contact:
janwas@google.com
le.com
