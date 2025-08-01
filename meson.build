project(
    'hwy',
    'cpp',
    version: '1.3.0',
    meson_version: '>= 1.3.0',
    license: 'Apache-2.0 OR BSD-3-Clause',
    default_options: ['cpp_std=c++17,c++14,c++11', 'warning_level=2'],
)

if host_machine.system() == 'windows'
    add_project_arguments('-DNOMINMAX', language: 'cpp')
endif

install_headers(
    [
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
        'hwy/per_target.h',
        'hwy/print-inl.h',
        'hwy/print.h',
        'hwy/profiler.h',
        'hwy/robust_statistics.h',
        'hwy/targets.h',
        'hwy/timer-inl.h',
        'hwy/timer.h',
        'hwy/x86_cpuid.h',
    ],
    subdir: 'hwy',
)

install_headers(
    [
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
    ],
    subdir: 'hwy/ops',
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

hwy_lib = library(
    'hwy',
    sources: hwy_sources,
    cpp_shared_args: ['-DHWY_SHARED_DEFINE'],
    cpp_static_args: ['-DHWY_STATIC_DEFINE'],
    gnu_symbol_visibility: 'hidden',
    install: true,
)

hwy_compile_args = []
if get_option('default_library') != 'static'
    hwy_compile_args += ['-DHWY_SHARED_DEFINE']
endif

hwy_dep = declare_dependency(
    link_with: hwy_lib,
    include_directories: include_directories('.'),
    compile_args: hwy_compile_args,
)

if get_option('contrib').enabled()
    contrib_headers = {
        'hwy/contrib/bit_pack': files('hwy/contrib/bit_pack/bit_pack-inl.h'),
        'hwy/contrib/dot': files('hwy/contrib/dot/dot-inl.h'),
        'hwy/contrib/image': files('hwy/contrib/image/image.h'),
        'hwy/contrib/math': files('hwy/contrib/math/math-inl.h'),
        'hwy/contrib/matvec': files('hwy/contrib/matvec/matvec-inl.h'),
        'hwy/contrib/random': files('hwy/contrib/random/random-inl.h'),
        'hwy/contrib/sort': files(
            'hwy/contrib/sort/order.h',
            'hwy/contrib/sort/shared-inl.h',
            'hwy/contrib/sort/sorting_networks-inl.h',
            'hwy/contrib/sort/traits-inl.h',
            'hwy/contrib/sort/traits128-inl.h',
            'hwy/contrib/sort/vqsort-inl.h',
        ),
        'hwy/contrib/thread_pool': files(
            'hwy/contrib/thread_pool/futex.h',
            'hwy/contrib/thread_pool/spin.h',
            'hwy/contrib/thread_pool/thread_pool.h',
            'hwy/contrib/thread_pool/topology.h',
        ),
        'hwy/contrib/algo': files(
            'hwy/contrib/algo/copy-inl.h',
            'hwy/contrib/algo/find-inl.h',
            'hwy/contrib/algo/transform-inl.h',
        ),
        'hwy/contrib/unroller': files('hwy/contrib/unroller/unroller-inl.h'),
    }

    foreach subdir, files : contrib_headers
        install_headers(files, subdir: subdir)
    endforeach

    hwy_contrib_sources = files(
        'hwy/contrib/image/image.cc',
        'hwy/contrib/sort/vqsort.cc',
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
        'hwy/contrib/sort/vqsort_kv128a.cc',
        'hwy/contrib/sort/vqsort_kv128d.cc',
        'hwy/contrib/sort/vqsort_kv64a.cc',
        'hwy/contrib/sort/vqsort_kv64d.cc',
        'hwy/contrib/sort/vqsort_u16a.cc',
        'hwy/contrib/sort/vqsort_u16d.cc',
        'hwy/contrib/sort/vqsort_u32a.cc',
        'hwy/contrib/sort/vqsort_u32d.cc',
        'hwy/contrib/sort/vqsort_u64a.cc',
        'hwy/contrib/sort/vqsort_u64d.cc',
        'hwy/contrib/thread_pool/topology.cc',
    )

    hwy_contrib_lib = library(
        'hwy_contrib',
        sources: hwy_contrib_sources,
        cpp_shared_args: ['-DHWY_SHARED_DEFINE'],
        cpp_static_args: ['-DHWY_STATIC_DEFINE'],
        dependencies: hwy_dep,
        gnu_symbol_visibility: 'hidden',
        install: true,
    )

    hwy_contrib_dep = declare_dependency(
        link_with: hwy_contrib_lib,
        dependencies: hwy_dep,
    )
endif

if get_option('examples').enabled()
    hwy_benchmark_exe = executable(
        'hwy_benchmark',
        sources: 'hwy/examples/benchmark.cc',
        dependencies: hwy_dep,
    )

    if get_option('contrib').enabled()
        hwy_profiler_example_exe = executable(
            'hwy_profiler_example',
            sources: 'hwy/examples/profiler_example.cc',
            dependencies: hwy_contrib_dep,
        )
    endif
endif

if get_option('tests').enabled()
    gtest_main_dep = dependency('gtest_main')

    install_headers(
        [
            'hwy/tests/hwy_gtest.h',
            'hwy/tests/test_util-inl.h',
            'hwy/tests/test_util.h',
        ],
        subdir: 'hwy/tests',
    )
    hwy_test_sources = files('hwy/tests/test_util.cc')

    hwy_test_lib = library(
        'hwy_test',
        sources: hwy_test_sources,
        cpp_shared_args: ['-DHWY_SHARED_DEFINE'],
        cpp_static_args: ['-DHWY_STATIC_DEFINE'],
        dependencies: hwy_dep,
        gnu_symbol_visibility: 'hidden',
        install: true,
    )

    hwy_test_dep = declare_dependency(
        link_with: hwy_test_lib,
        dependencies: hwy_dep,
    )
    hwy_test_deps = [hwy_test_dep, gtest_main_dep]

    hwy_test_files = files(
        'hwy/abort_test.cc',
        'hwy/aligned_allocator_test.cc',
        'hwy/base_test.cc',
        'hwy/bit_set_test.cc',
        'hwy/examples/skeleton_test.cc',
        'hwy/highway_test.cc',
        'hwy/nanobenchmark_test.cc',
        'hwy/perf_counters_test.cc',
        'hwy/targets_test.cc',
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
        'hwy/tests/minmax128_test.cc',
        'hwy/tests/minmax_magnitude_test.cc',
        'hwy/tests/minmax_number_test.cc',
        'hwy/tests/minmax_test.cc',
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

    if get_option('contrib').enabled()
        hwy_test_deps += hwy_contrib_dep

        hwy_test_files += files(
            'hwy/auto_tune_test.cc',
            'hwy/contrib/algo/copy_test.cc',
            'hwy/contrib/algo/find_test.cc',
            'hwy/contrib/algo/transform_test.cc',
            'hwy/contrib/bit_pack/bit_pack_test.cc',
            'hwy/contrib/dot/dot_test.cc',
            'hwy/contrib/image/image_test.cc',
            'hwy/contrib/math/math_test.cc',
            'hwy/contrib/matvec/matvec_test.cc',
            'hwy/contrib/random/random_test.cc',
            'hwy/contrib/sort/bench_sort.cc',
            'hwy/contrib/sort/sort_test.cc',
            'hwy/contrib/sort/sort_unit_test.cc',
            'hwy/contrib/thread_pool/spin_test.cc',
            'hwy/contrib/thread_pool/thread_pool_test.cc',
            'hwy/contrib/thread_pool/topology_test.cc',
            'hwy/contrib/unroller/unroller_test.cc',
        )
    endif

    fs = import('fs')
    foreach hwy_test_file : hwy_test_files
        test_name = fs.stem(hwy_test_file)

        sources = [hwy_test_file]
        if test_name == 'skeleton_test'
            sources += files('hwy/examples/skeleton.cc')
        endif

        exc = executable(
            test_name,
            sources: sources,
            cpp_args: ['-DHWY_IS_TEST=1'],
            dependencies: hwy_test_deps,
        )
        test(test_name, exc, timeout: 60)
    endforeach
endif
