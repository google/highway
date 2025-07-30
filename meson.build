project(
  'hwy',
  'cpp',
  version: '1.2.0',
  meson_version: '>= 1.3.0',
  license: 'Apache-2.0 OR BSD-3-Clause',
  default_options: [
    'cpp_std=c++17,c++14,c++11',
    'b_pie=true',
    'b_staticpic=true',
    'buildtype=debugoptimized',
    'default_library=static',
    'warning_level=2',  # -Wall -Wextra on gcc/clang, /W3 on msvc
    'cpp_eh=none',
    'cpp_rtti=false',
    'werror=false',  # meson has builtin warnings_are_errors option
  ]
)

contrib_enabled = get_option('enable_contrib')
tests_enabled = get_option('enable_tests')
examples_enabled = get_option('enable_examples')

cpp = meson.get_compiler('cpp')

pkg = import('pkgconfig')

have_emscripten = cpp.has_define('__EMSCRIPTEN__')
have_riscv = cpp.has_define('__riscv')

support_lsx_flag = false
support_lasx_flag = false
if 'loongarch' in host_machine.cpu_family()
    if get_option('lsx') and cpp.has_define('__loongarch_sx', args: '-mlsx')
        add_project_arguments('-mlsx', language:'cpp')
    endif
    if get_option('lasx') and cpp.has_define('__loongarch_asx', args: '-mlasx')
        add_project_arguments('-mlasx', language:'cpp')
    endif
endif

is_windows = host_machine.system() == 'windows'

pie = get_option('b_pie') and not is_windows
static_pic = get_option('b_staticpic') and not is_windows


atomic_dep = dependency('', required: false)
atomic_code = '''
    #include <atomic>
    #include <cstdint>
    std::atomic<uint8_t> n8 (0); // riscv64
    std::atomic<uint64_t> n64 (0); // armel, mipsel, powerpc
    int main() {
        ++n8;
        ++n64;
        return 0;
    }
    '''
if not cpp.compiles(atomic_code)
    atomic_dep = cpp.find_library('atomic', required: true)
    if not cpp.compiles(atomic_code, dependencies: atomic_dep)
        error('Unable to compile with atomic')
    endif
endif


riscv_xlen = 0
if have_riscv or 'riscv' in host_machine.cpu_family()
    riscv_xlen = cpp.get_define('__riscv_xlen')
    if riscv_xlen == ''
        message('Unable to determine RISC-V XLEN')
    endif
endif

hwy_headers = files(
    'hwy/abort.h',
    'hwy/aligned_allocator.h',
    'hwy/auto_tune.h',
    'hwy/base.h',
    'hwy/cache_control.h',
    'hwy/detect_compiler_arch.h',  # private
    'hwy/detect_targets.h',  # private
    'hwy/foreach_target.h',
    'hwy/highway_export.h',
    'hwy/highway.h',
    'hwy/nanobenchmark.h',
    'hwy/ops/arm_neon-inl.h',
    'hwy/ops/arm_sve-inl.h',
    'hwy/ops/emu128-inl.h',
    'hwy/ops/generic_ops-inl.h',
    'hwy/ops/inside-inl.h',
    'hwy/ops/loongarch_lsx-inl.h',
    'hwy/ops/loongarch_lasx-inl.h',
    'hwy/ops/ppc_vsx-inl.h',
    'hwy/ops/rvv-inl.h',
    'hwy/ops/scalar-inl.h',
    'hwy/ops/set_macros-inl.h',
    'hwy/ops/shared-inl.h',
    'hwy/ops/wasm_128-inl.h',
    'hwy/ops/x86_128-inl.h',
    'hwy/ops/x86_256-inl.h',
    'hwy/ops/x86_512-inl.h',
    'hwy/ops/x86_avx3-inl.h',
    'hwy/per_target.h',
    'hwy/print-inl.h',
    'hwy/print.h',
    'hwy/profiler.h',
    'hwy/robust_statistics.h',
    'hwy/targets.h',
    'hwy/timer-inl.h',
    'hwy/timer.h',
    'hwy/x86_cpuid.h',
)

hwy_sources = files(
    'hwy/abort.cc',
    'hwy/aligned_allocator.cc',
    'hwy/nanobenchmark.cc',
    'hwy/per_target.cc',
    'hwy/perf_counters.cc',
    'hwy/print.cc',
    'hwy/profiler.cc',
    'hwy/targets.cc',
    'hwy/timer.cc',
)

hwy_contrib_headers = files(
    'hwy/contrib/bit_pack/bit_pack-inl.h',
    'hwy/contrib/dot/dot-inl.h',
    'hwy/contrib/image/image.h',
    'hwy/contrib/math/math-inl.h',
    'hwy/contrib/matvec/matvec-inl.h',
    'hwy/contrib/random/random-inl.h',
    'hwy/contrib/sort/order.h',
    'hwy/contrib/sort/shared-inl.h',
    'hwy/contrib/sort/sorting_networks-inl.h',
    'hwy/contrib/sort/traits-inl.h',
    'hwy/contrib/sort/traits128-inl.h',
    'hwy/contrib/sort/vqsort-inl.h',
    'hwy/contrib/sort/vqsort.h',
    'hwy/contrib/thread_pool/futex.h',
    'hwy/contrib/thread_pool/spin.h',
    'hwy/contrib/thread_pool/thread_pool.h',
    'hwy/contrib/thread_pool/topology.h',
    'hwy/contrib/algo/copy-inl.h',
    'hwy/contrib/algo/find-inl.h',
    'hwy/contrib/algo/transform-inl.h',
    'hwy/contrib/unroller/unroller-inl.h',
)

hwy_contrib_sources = files(
    'hwy/contrib/image/image.cc',
    'hwy/contrib/sort/vqsort.cc',
    'hwy/contrib/thread_pool/topology.cc',
    # plus all of the vqsort_*.cc....
    # note meson doesn't directly support glob (by design).
    'hwy/contrib/sort/vqsort_128a.cc',
    'hwy/contrib/sort/vqsort_128d.cc',
    'hwy/contrib/sort/vqsort_f16a.cc',
    'hwy/contrib/sort/vqsort_f16d.cc',
    'hwy/contrib/sort/vqsort_f32a.cc',
    'hwy/contrib/sort/vqsort_f32d.cc',
    'hwy/contrib/sort/vqsort_f64a.cc',
    'hwy/contrib/sort/vqsort_f64d.cc',
    'hwy/contrib/sort/vqsort_i16a.cc',
    'hwy/contrib/sort/vqsort_i16d.cc',
    'hwy/contrib/sort/vqsort_i32a.cc',
    'hwy/contrib/sort/vqsort_i32d.cc',
    'hwy/contrib/sort/vqsort_i64a.cc',
    'hwy/contrib/sort/vqsort_i64d.cc',
    'hwy/contrib/sort/vqsort_kv64a.cc',
    'hwy/contrib/sort/vqsort_kv64d.cc',
    'hwy/contrib/sort/vqsort_kv128a.cc',
    'hwy/contrib/sort/vqsort_kv128d.cc',
    'hwy/contrib/sort/vqsort_u16a.cc',
    'hwy/contrib/sort/vqsort_u16d.cc',
    'hwy/contrib/sort/vqsort_u32a.cc',
    'hwy/contrib/sort/vqsort_u32d.cc',
    'hwy/contrib/sort/vqsort_u64a.cc',
    'hwy/contrib/sort/vqsort_u64d.cc',
)

hwy_test_headers = files(
    'hwy/tests/hwy_gtest.h',
    'hwy/tests/test_util-inl.h',
    'hwy/tests/test_util.h',
)

hwy_test_sources = files(
    'hwy/tests/test_util.cc'
)

cpp_link_args = []
is_msvc_syntax = cpp.get_argument_syntax() == 'msvc'
if is_msvc_syntax
    add_project_arguments(
        '/bigobj',

        # '/GR-',  # handled by cpp_rtti=false default option
        # '/EHs-c-',  # handled by cpp_eh=none default option
        # Disable exceptions in STL code.
        '-D_HAS_EXCEPTIONS=0',
        language:'cpp'
    )
    if get_option('warning_level') == '4'
        add_project_arguments(
            # Disable some W4 warnings.  Enable them individually after they are cleaned up.
            '/wd4100',
            '/wd4127',
            '/wd4324',
            '/wd4456',
            '/wd4701',
            '/wd4702',
            '/wd4723',
            language: 'cpp',
        )
    endif

    if cpp.get_id() == 'clang-cl'
        add_project_arguments(
            '-Wconversion',
            '-Wsign-conversion',
            '-Wvla',
            '-Wnon-virtual-dtor',

            '-Wfloat-overflow-conversion',
            '-Wfloat-zero-conversion',
            '-Wfor-loop-analysis',
            '-Wgnu-redeclared-enum',
            '-Winfinite-recursion',
            '-Wself-assign',
            '-Wstring-conversion',
            '-Wtautological-overlap-compare',
            '-Wthread-safety-analysis',
            '-Wundefined-func-template',
            language:'cpp'
        )
    endif
else
    add_project_arguments(
        # Avoid changing binaries based on the current time and date.
        '-Wno-builtin-macro-redefined',
        '-D__DATE__="redacted"',
        '-D__TIMESTAMP__="redacted"',
        '-D__TIME__="redacted"',

        # Optimizations
        '-fmerge-all-constants',
    
        # Warnings
        # '-Wall', handled by warning_level
        # '-Wextra', handled by warning_level
        # These are not included in Wall nor Wextra:
        '-Wconversion',
        '-Wsign-conversion',
        '-Wvla',
        '-Wnon-virtual-dtor',
        '-Wcast-align', # see -Wcast-align=strict on x86
        language:'cpp'
    )

    if cpp.get_id() == 'clang'
        add_project_arguments(
            '-Wfloat-overflow-conversion',
            '-Wfloat-zero-conversion',
            '-Wfor-loop-analysis',
            '-Wgnu-redeclared-enum',
            '-Winfinite-recursion',
            '-Wself-assign',
            '-Wstring-conversion',
            '-Wtautological-overlap-compare',
            '-Wthread-safety-analysis',
            '-Wundefined-func-template',

            # '-fno-cxx-exceptions' # handled by default build option cpp_eh=none
            '-fno-slp-vectorize',
            '-fno-vectorize',

            # Use color in messages
            '-fdiagnostics-show-option',
            '-fcolor-diagnostics',
            language:'cpp'
        )
        if cpp.version().version_compare('>=6.0')
            add_project_arguments(
                '-Wc++2a-extensions',
                language:'cpp'
            )
        endif
    endif
    if is_windows
        if cpp.get_id() == 'clang'
            add_project_arguments(
                '-Wno-global-constructors',
                '-Wno-language-extension-token',
                '-Wno-used-but-marked-unused',
                '-Wno-shadow-field-in-constructor',
                '-Wno-unused-member-function',
                '-Wno-unused-template',
                '-Wno-c++98-compat-pedantic',
                '-Wno-used-but-marked-unused',
                '-Wno-zero-as-null-pointer-constant',
                language:'cpp',
            )
        endif
        add_project_arguments(
            '-Wno-cast-align',
            '-Wno-double-promotion',
            '-Wno-float-equal',
            '-Wno-format-nonliteral',
            '-Wno-shadow',
            '-Wno-sign-conversion',
            language:'cpp',
        )
    else

        add_project_arguments(
            '-fmath-errno',
            # '-fno-exceptions', # handled by cpp_eh=none default option
            language:'cpp'
        )
    endif

    if get_option('sse2')
        add_project_arguments(
            '-msse2',
            '-mfpmath=sse',
            language:'cpp',
        )
    endif

    if cpp.get_id() == 'gcc' or (cpp.get_id() == 'clang' and cpp.version().version_compare('>=11.0'))
        add_project_arguments(
            '-Wno-psabi',
            language: 'cpp',
        )
    endif

    if get_option('arm7')
        add_project_arguments(
            '-march=armv7-a',
            '-mfpu=neon-vfpv4',
            '-mfloat-abi=hard',
            '-DHWY_HAVE_SCALAR_F16_TYPE=0',  # See #2625
            '-DHWY_NEON_HAVE_F16C=0',
            language: 'cpp',
        )
        if cpp.get_id() == 'gcc'
            add_project_arguments(
                '-mfp16-format=ieee',
                language: 'cpp',
            )
        endif
    endif

    if have_riscv and get_option('rvv')
        riscv_arg = ''
        if riscv_xlen == '64'
            riscv_arg = '-march=rv64gcv1p0'
        elif riscv_xlen == '32'
            riscv_arg = '-march=rv32gcv1p0'
        endif

        if riscv_arg != ''
            add_project_arguments(riscv_arg, language: 'cpp')
            add_project_link_arguments(riscv_arg, language: 'cpp')
        endif

        if cpp.get_id() == 'clang'
            add_project_arguments('-menable-experimental-extensions', language: 'cpp')
        endif
    endif

    if have_emscripten
        add_project_arguments('-matomics', language:'cpp')
    endif

endif

if get_option('header_only')
    add_project_arguments('-DHWY_HEADER_ONLY', language:'cpp')
endif

have_sys_auxv_h = cpp.has_header('sys/auxv.h')
have_asm_hwcap_h = cpp.has_header('asm/hwcap.h')
toolchain_args = []
if not have_sys_auxv_h
    toolchain_args += '-DTOOLCHAIN_MISS_SYS_AUXV_H'
endif
if  not have_asm_hwcap_h
    toolchain_args += '-DTOOLCHAIN_MISS_ASM_HWCAP_H'
endif

hwy_include_dir = include_directories('.')

hwy_version_file = 'hwy'/'hwy.version'
lib_link_args = cpp.first_supported_link_argument('-Wl,--version-script=' + meson.current_source_dir() / hwy_version_file)

hwy_link_depends = []
if lib_link_args.length() > 0
    hwy_link_depends += files(hwy_version_file)
endif

hwy_shared_args = ['-DHWY_SHARED_DEFINE', '-Dhwy_EXPORTS']
hwy_static_args = ['-DHWY_STATIC_DEFINE']

hwy_lib = library(
    'hwy',
    hwy_sources,
    dependencies: atomic_dep,
    cpp_args: toolchain_args,
    cpp_shared_args: hwy_shared_args,
    cpp_static_args: hwy_static_args,
    link_args: lib_link_args,
    install: true,
    pic: static_pic,
    version: meson.project_version(),
    link_depends: hwy_link_depends,
    gnu_symbol_visibility: 'inlineshidden',
)

hwy_dep_args = []
if get_option('default_library') != 'static'
    hwy_dep_args += '-DHWY_SHARED_DEFINE'
endif

hwy_dep = declare_dependency(
    compile_args: hwy_dep_args,
    include_directories: hwy_include_dir,
    dependencies: atomic_dep,
    link_with: hwy_lib,
)

install_headers(hwy_headers, preserve_path:true)

pkg.generate(
    hwy_lib,
    description: 'Efficient and performance-portable SIMD wrapper',
    name: 'libhwy',
    extra_cflags: hwy_dep_args,
)

if contrib_enabled
    # these dependencies look to be only required if enable_contrib
    contrib_deps = [hwy_dep]

    #threading is only required if contrib is enabled
    thread_dep = dependency('threads')
    contrib_deps += thread_dep

    contrib_args = []

    disable_futex = get_option('disable_futex')
    if is_windows
        synch_dep = cpp.find_library('synchronization', required:false)
        synch_run = cpp.run(
                '''
                #ifndef NOMINMAX
                #define NOMINMAX
                #endif

                #include <windows.h>

                int main() {
                    unsigned val1 = 0u;
                    unsigned val2 = 1u;
                    WaitOnAddress(&val1, &val2, sizeof(unsigned), 1);
                    WakeByAddressAll(&val1);
                    WakeByAddressSingle(&val1);
                    return 0;
                }
                ''',
                dependencies: synch_dep  # if synch_dep wasn't found, and this compiles anyway great!
        )
        if synch_run.returncode() != 0
            disable_futex = true
            message('Disabled Futex')
        else
            contrib_deps += synch_dep
        endif
    endif

    if disable_futex
        contrib_args += '-DHWY_DISABLE_FUTEX'
    endif

    hwy_contrib_lib = library(
        'hwy_contrib',
        hwy_contrib_sources,
        dependencies: contrib_deps,
        cpp_args: contrib_args,
        cpp_shared_args: hwy_shared_args,
        cpp_static_args: hwy_static_args,
        link_args: lib_link_args,
        build_by_default: false, # only built if requested...
        install: true,
        pic: static_pic,
        version: meson.project_version(),
        link_depends: hwy_link_depends,
        gnu_symbol_visibility: 'inlineshidden',
    )

    hwy_contrib_dep = declare_dependency(
        compile_args: contrib_args,
        link_with: hwy_contrib_lib,
        dependencies: contrib_deps
    )

    pkg.generate(
        hwy_contrib_lib,
        description: 'Additions to Highway: dot product, image, math, sort',
        name: 'libhwy-contrib',
    )

    install_headers(hwy_contrib_headers, preserve_path:true)
endif

hwy_test_lib = library(
    'hwy_test',
    hwy_test_sources,
    cpp_shared_args: hwy_shared_args,
    cpp_static_args: hwy_static_args,
    link_args: lib_link_args,
    dependencies: hwy_dep,
    pic: static_pic,
    version: meson.project_version(),
    link_depends: hwy_link_depends,
    build_by_default: not meson.is_subproject(),
    install: tests_enabled,
    gnu_symbol_visibility: 'inlineshidden',
)

hwy_test_args = []
if get_option('test_standalone')
    hwy_test_args += '-DHWY_TEST_STANDALONE=1'
else
    hwy_test_args += '-DHWY_TEST_STANDALONE=0'
endif

hwy_test_dep = declare_dependency(
    link_with: hwy_test_lib,
    compile_args: hwy_test_args,
    dependencies: hwy_dep,
)

hwy_list_targets = executable(
    'hwy_list_targets',
    'hwy/tests/list_targets.cc',
    dependencies: hwy_dep,
    pie: pie,
)

if tests_enabled
    install_headers(hwy_test_headers, preserve_path:true)

    pkg.generate(
        hwy_test_lib,
        name: 'libhwy-test',
        description: 'Efficient and performance-portable SIMD wrapper, test helpers.',
    )
endif

if examples_enabled

    executable(
        'hwy_benchmark',
        'hwy/examples/benchmark.cc',
        dependencies: hwy_dep,
        pie: pie,
    )

    if contrib_enabled
        executable(
            'hwy_profiler_example',
            'hwy/examples/profiler_example.cc',
            dependencies: [hwy_dep, hwy_contrib_dep],
            pie: pie,
        )
    endif

endif

if tests_enabled
    
    fs = import('fs')

    if get_option('test_standalone')
        gtest_main_dep = dependency('', required:false) # empty dependency
    else
        # defaults to system gtest with a custom fallback to a local gtest subproject
        gtest_main_dep = dependency(
            'gtest',
            main:true,
            required:false,
        )
        if not gtest_main_dep.found()
            gtest_build_overrides = [
                'cpp_eh=default',  # do not disable cpp exceptions for gtest
                'werror=false',  # do not error on warnings building gtest
            ]
            # find gtest as a subproject (The wrap included here matches 1.12.1)
            # note that 1.12.1 was the last to support C++11.
            gtest_proj = subproject('gtest', default_options:gtest_build_overrides + ['default_library=shared'])
            gtest_main_dep = gtest_proj.get_variable('gtest_main_dep')
            gtest_version = gtest_main_dep.version()
            if gtest_version.version_compare('<1.15.2')

                gtest_incdir = gtest_proj.get_variable('gtest_incdir')
                gtest_libsources = gtest_proj.get_variable('gtest_libsources')
                gtest_mainsources = gtest_proj.get_variable('gtest_mainsources')
                
                # this bit is awkward because the 1.12.1 wrap defines the dependency objects as files
                # causing them to get recompiled with every test executable. The gtest wrap for version 1.15.2-2 was first to build
                # gtest as a library, it's logic is mostly duplicated below.

                thread_dep = dependency('threads')

                if is_windows
                    gtest_dllexp = '-DGTEST_CREATE_SHARED_LIBRARY=1'
                    gtest_dllimp = '-DGTEST_LINKED_AS_SHARED_LIBRARY=1'
                else
                    gtest_dllexp = []
                    gtest_dllimp = []
                endif

                # always build as a shared library (to avoid crt conflicts)
                # (note that these are not installable from this project)
                gtest_lib = shared_library(
                    'gtest-all',
                    gtest_libsources,
                    gnu_symbol_visibility: 'hidden',
                    override_options:gtest_build_overrides,
                    cpp_args: gtest_dllexp,
                    dependencies: thread_dep,
                    include_directories: gtest_incdir,
                )

                gtest_main_lib = shared_library(
                    'gtest_main',
                    gtest_mainsources,
                    gnu_symbol_visibility: 'hidden',
                    override_options:gtest_build_overrides,
                    cpp_args: gtest_dllexp,
                    include_directories: gtest_incdir,
                    link_with: gtest_lib,
                )

                gtest_dep = declare_dependency(
                    compile_args: gtest_dllimp,
                    include_directories: gtest_incdir,
                    link_with: gtest_lib,
                    dependencies: thread_dep,
                    version: gtest_version,
                )

                gtest_main_dep = declare_dependency(
                    link_with: gtest_main_lib,
                    dependencies: gtest_dep,
                    version: gtest_version,
                )
            endif
        endif
    endif

    hwy_test_files = files(
        'hwy/abort_test.cc',
        'hwy/aligned_allocator_test.cc',
        'hwy/base_test.cc',
        'hwy/bit_set_test.cc',
        'hwy/highway_test.cc',
        'hwy/nanobenchmark_test.cc',
        'hwy/perf_counters_test.cc',
        'hwy/targets_test.cc',
        'hwy/examples/skeleton_test.cc',
        'hwy/tests/arithmetic_test.cc',
        'hwy/tests/bit_permute_test.cc',
        'hwy/tests/blockwise_combine_test.cc',
        'hwy/tests/blockwise_shift_test.cc',
        'hwy/tests/blockwise_test.cc',
        'hwy/tests/cast_test.cc',
        'hwy/tests/combine_test.cc',
        'hwy/tests/compare_test.cc',
        'hwy/tests/complex_arithmetic_test.cc',
        'hwy/tests/compress_test.cc',
        'hwy/tests/concat_test.cc',
        'hwy/tests/convert_test.cc',
        'hwy/tests/count_test.cc',
        'hwy/tests/crypto_test.cc',
        'hwy/tests/demote_test.cc',
        'hwy/tests/div_test.cc',
        'hwy/tests/dup128_vec_test.cc',
        'hwy/tests/expand_test.cc',
        'hwy/tests/float_test.cc',
        'hwy/tests/fma_test.cc',
        'hwy/tests/foreach_vec_test.cc',
        'hwy/tests/if_test.cc',
        'hwy/tests/in_range_float_to_int_conv_test.cc',
        'hwy/tests/interleaved_test.cc',
        'hwy/tests/logical_test.cc',
        'hwy/tests/mask_combine_test.cc',
        'hwy/tests/mask_convert_test.cc',
        'hwy/tests/mask_mem_test.cc',
        'hwy/tests/mask_set_test.cc',
        'hwy/tests/mask_slide_test.cc',
        'hwy/tests/mask_test.cc',
        'hwy/tests/masked_arithmetic_test.cc',
        'hwy/tests/masked_minmax_test.cc',
        'hwy/tests/memory_test.cc',
        'hwy/tests/minmax_magnitude_test.cc',
        'hwy/tests/minmax_number_test.cc',
        'hwy/tests/minmax_test.cc',
        'hwy/tests/minmax128_test.cc',
        'hwy/tests/mul_by_pow2_test.cc',
        'hwy/tests/mul_pairwise_test.cc',
        'hwy/tests/mul_test.cc',
        'hwy/tests/reduction_test.cc',
        'hwy/tests/resize_test.cc',
        'hwy/tests/reverse_test.cc',
        'hwy/tests/rotate_test.cc',
        'hwy/tests/saturated_test.cc',
        'hwy/tests/shift_test.cc',
        'hwy/tests/shuffle4_test.cc',
        'hwy/tests/sign_test.cc',
        'hwy/tests/slide_up_down_test.cc',
        'hwy/tests/sums_abs_diff_test.cc',
        'hwy/tests/swizzle_block_test.cc',
        'hwy/tests/swizzle_test.cc',
        'hwy/tests/table_test.cc',
        'hwy/tests/test_util_test.cc',
        'hwy/tests/truncate_test.cc',
        'hwy/tests/tuple_test.cc',
        'hwy/tests/widen_mul_test.cc',
    )

    test_exe_args = ['-DHWY_IS_TEST=1']
    
    test_exe_link_args = []
    if have_emscripten
        test_exe_link_args += ['-s', 'SINGLE_FILE=1']
    endif

    foreach test_src : hwy_test_files
        exe_name = fs.stem(test_src)

        test_srcs = [test_src]
        if exe_name == 'skeleton_test'
            test_srcs += files('hwy/examples/skeleton.cc')
        endif

        test_exe = executable(
            exe_name,
            test_srcs,
            cpp_args: test_exe_args,
            dependencies: [hwy_test_dep, gtest_main_dep],
            link_args: test_exe_link_args,
            pie: pie,
        )
        
        test(exe_name, test_exe, protocol:'gtest')
    endforeach

    if contrib_enabled
        hwy_contrib_test_files = files(
            'hwy/auto_tune_test.cc',
            'hwy/contrib/algo/copy_test.cc',
            'hwy/contrib/algo/find_test.cc',
            'hwy/contrib/algo/transform_test.cc',
            'hwy/contrib/bit_pack/bit_pack_test.cc',
            'hwy/contrib/dot/dot_test.cc',
            'hwy/contrib/matvec/matvec_test.cc',
            'hwy/contrib/image/image_test.cc',
            # Disabled due to SIGILL in clang7 debug build during gtest discovery phase,
            # not reproducible locally. Still tested via bazel build.
            'hwy/contrib/math/math_test.cc',
            'hwy/contrib/random/random_test.cc',
            'hwy/contrib/sort/bench_sort.cc',
            'hwy/contrib/sort/sort_test.cc',
            'hwy/contrib/sort/sort_unit_test.cc',
            'hwy/contrib/thread_pool/spin_test.cc',
            'hwy/contrib/thread_pool/thread_pool_test.cc',
            'hwy/contrib/thread_pool/topology_test.cc',
            'hwy/contrib/unroller/unroller_test.cc',
        )

        foreach test_src : hwy_contrib_test_files
            exe_name = fs.stem(test_src)

            test_exe = executable(
                exe_name,
                test_src,
                cpp_args: test_exe_args,
                dependencies: [hwy_test_dep, hwy_contrib_dep, gtest_main_dep],
                link_args: test_exe_link_args,
                pie: pie,
            )

            test(exe_name, test_exe, protocol:'gtest', suite:'contrib', timeout: (exe_name == 'sort_test')? 90 : 30)
        endforeach
    endif
endif

# if cmake
cmake_bin = find_program('cmake', required: false)
if cmake_bin.found()
    cmake = import('cmake')
    cmake.write_basic_package_version_file(
        name:'hwy-config-version.cmake', compatibility:'SameMajorVersion', version:meson.project_version()
    )
endif
