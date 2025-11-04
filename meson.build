project(
  'hwy',
  'cpp',
  version: '1.3.0',
  meson_version: '>= 1.3.0',
  license: 'Apache-2.0 OR BSD-3-Clause',
  default_options: [
    'cpp_std=c++17,c++14,c++11',
    'b_pie=true',
    'b_staticpic=true',
    'default_library=static',
    'cpp_eh=none',
    'cpp_rtti=false',
  ]
)

contrib_enabled = get_option('contrib').allowed()

# logic here for both examples and tests is to enable them by default when this
# is a project, but disable them by default if this is included as a subproject in another
# project.
is_subproj = meson.is_subproject()
tests_enabled = get_option('tests').disable_auto_if(is_subproj).allowed()
examples_enabled = get_option('examples').disable_auto_if(is_subproj).allowed()

cpp = meson.get_compiler('cpp')

hwy_nowarn_args = cpp.get_supported_arguments([
  '-Wno-gnu-zero-variadic-macro-arguments', # Disable gnu variadic macro warning.
])

hwy_cpp_args = []
hwy_link_args = []

pkg = import('pkgconfig')

have_emscripten = cpp.has_define('__EMSCRIPTEN__')
have_riscv = cpp.has_define('__riscv')

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


riscv_xlen = ''
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
    'hwy/contrib/thread_pool/thread_pool.cc',
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

is_msvc_syntax = cpp.get_argument_syntax() == 'msvc'
if is_msvc_syntax
    hwy_cpp_args += [
        '/bigobj',
        # Disable exceptions in STL code.
        '-D_HAS_EXCEPTIONS=0'
    ]
    if get_option('warning_level') == '3'  # enables /W4
        hwy_nowarn_args += [
            # Disable some W4 warnings.  Enable them individually after they are cleaned up.
            '/wd4100',
            '/wd4127',
            '/wd4324',
            '/wd4456',
            '/wd4701',
            '/wd4702',
            '/wd4723',
        ]
    endif
else
    add_project_arguments(
        # Avoid changing these binaries based on the current time and date.
        '-Wno-builtin-macro-redefined',
        '-D__DATE__="redacted"',
        '-D__TIMESTAMP__="redacted"',
        '-D__TIME__="redacted"',
        language: 'cpp'
    )

    hwy_cpp_args += [
        # Optimizations
        '-fmerge-all-constants',
    ]

    if cpp.get_id() == 'clang'
        hwy_cpp_args += [
            '-fno-slp-vectorize',
            '-fno-vectorize',
        ]
    endif
    if is_windows
        if cpp.get_id() == 'clang'
            hwy_nowarn_args += [
                '-Wno-global-constructors',
                '-Wno-language-extension-token',
                '-Wno-used-but-marked-unused',
                '-Wno-shadow-field-in-constructor',
                '-Wno-unused-member-function',
                '-Wno-unused-template',
                '-Wno-c++98-compat-pedantic',
                '-Wno-used-but-marked-unused',
                '-Wno-zero-as-null-pointer-constant',
            ]
        endif
        hwy_nowarn_args += [
            '-Wno-cast-align',
            '-Wno-double-promotion',
            '-Wno-float-equal',
            '-Wno-format-nonliteral',
            '-Wno-shadow',
            '-Wno-sign-conversion',
        ]
    else
        hwy_cpp_args += '-fmath-errno'
    endif

    if get_option('sse2')
        hwy_cpp_args += [
            '-msse2',
            '-mfpmath=sse',
        ]
    endif

    if cpp.get_id() == 'gcc' or (cpp.get_id() == 'clang' and cpp.version().version_compare('>=11.0'))
        hwy_nowarn_args += '-Wno-psabi'
    endif

    if get_option('arm7')
        hwy_cpp_args += [
            '-march=armv7-a',
            '-mfpu=neon-vfpv4',
            '-mfloat-abi=hard',
            '-DHWY_HAVE_SCALAR_F16_TYPE=0',  # See #2625
            '-DHWY_NEON_HAVE_F16C=0',
        ]
        if cpp.get_id() == 'gcc'
            hwy_cpp_args += '-mfp16-format=ieee'
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
            hwy_cpp_args += riscv_arg
            hwy_link_args += riscv_arg
        endif

        if cpp.get_id() == 'clang'
            hwy_cpp_args += '-menable-experimental-extensions'
        endif
    endif

    if have_emscripten
        hwy_cpp_args += '-matomics'
    endif

endif

hwy_hdr_only = get_option('header_only')
if hwy_hdr_only
    hwy_cpp_args += '-DHWY_HEADER_ONLY'
endif

have_sys_auxv_h = cpp.has_header('sys/auxv.h')
have_asm_hwcap_h = cpp.has_header('asm/hwcap.h')
if not have_sys_auxv_h
    hwy_cpp_args += '-DTOOLCHAIN_MISS_SYS_AUXV_H'
endif
if  not have_asm_hwcap_h
    hwy_cpp_args += '-DTOOLCHAIN_MISS_ASM_HWCAP_H'
endif


add_project_arguments(hwy_cpp_args, language: 'cpp')
add_project_arguments(hwy_nowarn_args, language: 'cpp')
add_project_link_arguments(hwy_link_args, language: 'cpp')

hwy_include_dir = include_directories('.')

hwy_version_file = 'hwy'/'hwy.version'
lib_link_args = cpp.first_supported_link_argument('-Wl,--version-script=' + meson.current_source_dir() / hwy_version_file)

hwy_link_depends = []
if lib_link_args.length() > 0
    hwy_link_depends += files(hwy_version_file)
endif

hwy_shared_import_args = ['-DHWY_SHARED_DEFINE']
hwy_shared_export_args = hwy_shared_import_args + ['-Dhwy_EXPORTS']
hwy_static_args = ['-DHWY_STATIC_DEFINE']

hwy_lib = library(
    'hwy',
    hwy_sources,
    dependencies: atomic_dep,
    cpp_shared_args: hwy_shared_export_args,
    cpp_static_args: hwy_static_args,
    link_args: lib_link_args,
    install: true,
    pic: static_pic,
    build_by_default: not hwy_hdr_only,
    version: meson.project_version(),
    link_depends: hwy_link_depends,
    gnu_symbol_visibility: 'inlineshidden',
)


hwy_dep_args = []
if hwy_hdr_only
    hwy_dep_args += '-DHWY_HEADER_ONLY'
elif get_option('default_library') != 'static'
    hwy_dep_args += hwy_shared_import_args
endif

hwy_dep = declare_dependency(
    compile_args: hwy_dep_args,
    include_directories: hwy_include_dir,
    dependencies: atomic_dep,
    link_with: hwy_hdr_only? [] : hwy_lib,
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
        cpp_shared_args: hwy_shared_export_args,
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
    cpp_shared_args: hwy_shared_export_args,
    cpp_static_args: hwy_static_args,
    link_args: lib_link_args,
    dependencies: hwy_dep,
    pic: static_pic,
    version: meson.project_version(),
    link_depends: hwy_link_depends,
    build_by_default: not is_subproj,
    install: tests_enabled,
    gnu_symbol_visibility: 'inlineshidden',
)

if get_option('test_standalone')
    gtest_dep = dependency('', required: false)
else
    cpp_std = get_option('cpp_std')
    if '11' in cpp_std
        gtest_req_ver = '<=1.12.1'
    elif '14' in cpp_std
        gtest_req_ver = '<=1.16.0'
    else
        gtest_req_ver = []
    endif

    gtest_dep = dependency(
        'gtest',
        version:gtest_req_ver,
        fallback: ['gtest', 'gtest_dep'],
    )
endif

if gtest_dep.found()
    hwy_test_args = '-DHWY_TEST_STANDALONE=0'
else
    hwy_test_args = '-DHWY_TEST_STANDALONE=1'
endif

hwy_test_dep = declare_dependency(
    link_with: hwy_test_lib,
    compile_args: hwy_test_args,
    dependencies: [hwy_dep, gtest_dep],
)

hwy_list_targets = executable(
    'hwy_list_targets',
    'hwy/tests/list_targets.cc',
    dependencies: hwy_dep,
    pie: pie,
    build_by_default: not is_subproj,
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

    if gtest_dep.found()
        gtest_main_dep = dependency(
            'gtest',
            main:true,
            version:gtest_req_ver,
            fallback: ['gtest', 'gtest_main_dep'],
        )
    else
        gtest_main_dep = dependency('', required: false)
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

    if not contrib_enabled
        hwy_test_files += files(           
            'hwy/auto_tune_test.cc',
        )
    endif

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

        extra_defines = []
        if exe_name == 'auto_tune_test'
            extra_defines += '-DHWY_AUTOTUNE_STDSORT'
        endif

        test_exe = executable(
            exe_name,
            test_srcs,
            cpp_args: test_exe_args + extra_defines,
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
            'hwy/contrib/math/math_hyper_test.cc',
            'hwy/contrib/math/math_tan_test.cc',
            'hwy/contrib/math/math_trig_test.cc',
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
                dependencies: [hwy_test_dep, gtest_main_dep, hwy_contrib_dep],
                link_args: test_exe_link_args,
                pie: pie,
            )

            test(exe_name, test_exe, protocol:'gtest', suite:'contrib', timeout: (exe_name == 'sort_test')? 90 : 30)
        endforeach
    endif
endif
