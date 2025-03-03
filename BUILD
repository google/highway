# Placeholder#1 for Guitar, do not remove
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_license//rules:license.bzl", "license")

load("@rules_cc//cc:defs.bzl", "cc_test")
# Placeholder#2 for Guitar, do not remove

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "highway",
    license_kinds = ["@rules_license//licenses/generic:notice"],
)

# Dual-licensed Apache 2 and 3-clause BSD.
licenses(["notice"])

exports_files(["LICENSE"])

# Detect compiler:
config_setting(
    name = "compiler_clang",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "clang"},
)

config_setting(
    name = "compiler_clangcl",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "lexan"},
)

config_setting(
    name = "compiler_msvc",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "msvc-cl"},
)

config_setting(
    name = "compiler_emscripten",
    constraint_values = [
        "@platforms//cpu:wasm32",
    ],
)

# See https://github.com/bazelbuild/bazel/issues/12707
config_setting(
    name = "compiler_gcc_bug",
    flag_values = {
        "@bazel_tools//tools/cpp:compiler": "compiler",
    },
)

config_setting(
    name = "compiler_gcc_actual",
    flag_values = {
        "@bazel_tools//tools/cpp:compiler": "gcc",
    },
)

selects.config_setting_group(
    name = "compiler_gcc",
    match_any = [
        ":compiler_gcc_bug",
        ":compiler_gcc_actual",
    ],
)

# Additional warnings for Clang OR GCC (skip for MSVC)
CLANG_GCC_COPTS = [
    "-Wunused",
    "-Wextra-semi",
    "-Wunreachable-code",
    "-Wshadow",
]

# Warnings supported by Clang and Clang-cl
CLANG_OR_CLANGCL_OPTS = CLANG_GCC_COPTS + [
    "-Wfloat-overflow-conversion",
    "-Wfloat-zero-conversion",
    "-Wfor-loop-analysis",
    "-Wgnu-redeclared-enum",
    "-Winfinite-recursion",
    "-Wliteral-conversion",
    "-Wno-c++98-compat",
    "-Wno-unused-command-line-argument",
    "-Wprivate-header",
    "-Wself-assign",
    "-Wstring-conversion",
    "-Wtautological-overlap-compare",
    "-Wthread-safety-analysis",
    "-Wundefined-func-template",
    "-Wunreachable-code-aggressive",
    "-Wunused-comparison",
]

# Warnings only supported by Clang, but not Clang-cl
CLANG_ONLY_COPTS = CLANG_OR_CLANGCL_OPTS + [
    # Do not treat the third_party headers as system headers when building
    # highway - the errors are pertinent.
    "--no-system-header-prefix=third_party/highway",
]

COPTS = select({
    ":compiler_msvc": [],
    ":compiler_gcc": CLANG_GCC_COPTS,
    ":compiler_clangcl": CLANG_OR_CLANGCL_OPTS,
    # Default to clang because compiler detection only works in Bazel
    "//conditions:default": CLANG_ONLY_COPTS,
}) + select({
    "@platforms//cpu:riscv64": [
        "-march=rv64gcv1p0",
        "-menable-experimental-extensions",
    ],
    "//conditions:default": [
    ],
})

DEFINES = select({
    ":compiler_msvc": ["HWY_SHARED_DEFINE"],
    ":compiler_clangcl": ["HWY_SHARED_DEFINE"],
    "//conditions:default": [],
})

# Unused on Bazel builds, where this is not defined/known; Copybara replaces
# usages with an empty list.
COMPAT = [
    "//buildenv/target:non_prod",  # includes mobile/vendor.
]

# WARNING: changing flags such as HWY_DISABLED_TARGETS may break users without
# failing integration tests, if the machine running tests does not support the
# newly enabled instruction set, or the failure is only caught by sanitizers
# which do not run in CI.

# NOTE: when adding a new dependency on the Highway library, please add your
# test to the highway.users list in highway.blueprint.
cc_library(
    name = "hwy",
    srcs = [
        "hwy/abort.cc",
        "hwy/aligned_allocator.cc",
        "hwy/per_target.cc",
        "hwy/print.cc",
        "hwy/targets.cc",
    ],
    # Normal headers with include guards
    hdrs = [
        "hwy/abort.h",
        "hwy/aligned_allocator.h",
        "hwy/base.h",
        "hwy/cache_control.h",
        "hwy/detect_compiler_arch.h",  # private
        "hwy/print.h",
        "hwy/x86_cpuid.h",
    ],
    compatible_with = [],
    copts = COPTS,
    defines = DEFINES,
    local_defines = ["hwy_EXPORTS"],
    textual_hdrs = [
        # These are textual because config macros influence them:
        "hwy/detect_targets.h",  # private
        "hwy/targets.h",
        # This .cc file #includes itself through foreach_target.h
        "hwy/per_target.cc",
        # End of list
        "hwy/highway.h",  # public
        "hwy/foreach_target.h",  # public
        "hwy/per_target.h",  # public
        "hwy/print-inl.h",  # public
        "hwy/highway_export.h",  # public
        "hwy/ops/arm_neon-inl.h",
        "hwy/ops/arm_sve-inl.h",
        "hwy/ops/emu128-inl.h",
        "hwy/ops/generic_ops-inl.h",
        "hwy/ops/inside-inl.h",
        "hwy/ops/scalar-inl.h",
        "hwy/ops/set_macros-inl.h",
        "hwy/ops/shared-inl.h",
        "hwy/ops/x86_128-inl.h",
        "hwy/ops/x86_256-inl.h",
        "hwy/ops/x86_512-inl.h",
        "hwy/ops/x86_avx3-inl.h",
        # Select avoids recompiling native arch if only non-native changed
    ] + select({
        ":compiler_emscripten": [
            "hwy/ops/wasm_128-inl.h",
            "hwy/ops/wasm_256-inl.h",
        ],
        "//conditions:default": [],
    }) + select({
        "@platforms//cpu:riscv64": ["hwy/ops/rvv-inl.h"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "stats",
    srcs = ["hwy/stats.cc"],
    hdrs = ["hwy/stats.h"],
    compatible_with = [],
    copts = COPTS,
    deps = [":hwy"],
)

cc_library(
    name = "robust_statistics",
    hdrs = ["hwy/robust_statistics.h"],
    compatible_with = [],
    copts = COPTS,
    deps = [":hwy"],
)

cc_library(
    name = "timer",
    srcs = ["hwy/timer.cc"],
    hdrs = ["hwy/timer.h"],
    compatible_with = [],
    copts = COPTS,
    # Deprecated.
    textual_hdrs = [
        "hwy/timer-inl.h",
    ],
    deps = [
        ":hwy",
        ":robust_statistics",
    ],
)

# Previously provided timer.*, use :timer instead.
cc_library(
    name = "nanobenchmark",
    srcs = ["hwy/nanobenchmark.cc"],
    hdrs = [
        "hwy/nanobenchmark.h",
        # TODO(janwas): remove after users depend on :timer.
        "hwy/timer.h",
    ],
    compatible_with = [],
    copts = COPTS,
    local_defines = ["hwy_EXPORTS"],
    deps = [
        ":hwy",
        ":robust_statistics",
        ":timer",
    ],
)

cc_library(
    name = "bit_set",
    hdrs = ["hwy/bit_set.h"],
    compatible_with = [],
    copts = COPTS,
    deps = [
        ":hwy",  # HWY_ASSERT
    ],
)

cc_library(
    name = "perf_counters",
    srcs = ["hwy/perf_counters.cc"],
    hdrs = ["hwy/perf_counters.h"],
    compatible_with = [],
    copts = COPTS,
    deps = [
        ":bit_set",
        ":hwy",
        ":timer",
    ],
)

cc_library(
    name = "profiler",
    hdrs = [
        "hwy/profiler.h",
    ],
    compatible_with = [],
    copts = COPTS,
    deps = [
        ":hwy",
        ":timer",
        # "//hwy/contrib/sort:vqsort",
    ],
)

cc_binary(
    name = "profiler_example",
    srcs = ["hwy/examples/profiler_example.cc"],
    copts = COPTS,
    deps = [
        ":hwy",
        ":profiler",
        ":timer",
    ],
)

cc_library(
    name = "algo",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/algo/copy-inl.h",
        "hwy/contrib/algo/find-inl.h",
        "hwy/contrib/algo/transform-inl.h",
    ],
    deps = [
        ":hwy",
    ],
)

cc_library(
    name = "bit_pack",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/bit_pack/bit_pack-inl.h",
    ],
    deps = [
        ":hwy",
    ],
)

cc_library(
    name = "dot",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/dot/dot-inl.h",
    ],
    deps = [
        ":hwy",
    ],
)

cc_library(
    name = "topology",
    srcs = ["hwy/contrib/thread_pool/topology.cc"],
    hdrs = ["hwy/contrib/thread_pool/topology.h"],
    compatible_with = [],
    copts = COPTS,
    deps = [
        ":bit_set",
        ":hwy",  # HWY_ASSERT
    ],
)

cc_library(
    name = "thread_pool",
    hdrs = [
        "hwy/contrib/thread_pool/futex.h",
        "hwy/contrib/thread_pool/spin.h",
        "hwy/contrib/thread_pool/thread_pool.h",
    ],
    compatible_with = [],
    copts = COPTS,
    deps = [
        ":bit_set",
        ":hwy",  # HWY_ASSERT
        ":profiler",
        ":stats",
        ":timer",
        ":topology",
    ],
)

cc_library(
    name = "matvec",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/matvec/matvec-inl.h",
    ],
    deps = [
        ":hwy",
        ":thread_pool",
    ],
)

cc_library(
    name = "image",
    srcs = [
        "hwy/contrib/image/image.cc",
    ],
    hdrs = [
        "hwy/contrib/image/image.h",
    ],
    compatible_with = [],
    copts = COPTS,
    local_defines = ["hwy_contrib_EXPORTS"],
    deps = [
        ":hwy",
    ],
)

cc_library(
    name = "math",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/math/math-inl.h",
    ],
    deps = [
        ":hwy",
    ],
)

cc_library(
    name = "random",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/random/random-inl.h",
    ],
    deps = [
        ":hwy",
    ],
)

cc_library(
    name = "unroller",
    compatible_with = [],
    copts = COPTS,
    textual_hdrs = [
        "hwy/contrib/unroller/unroller-inl.h",
    ],
    deps = [
        ":hwy",
    ],
)

# Everything required for tests that use Highway.
cc_library(
    name = "hwy_test_util",
    srcs = ["hwy/tests/test_util.cc"],
    hdrs = ["hwy/tests/test_util.h"],
    compatible_with = [],
    copts = COPTS,
    local_defines = ["hwy_test_EXPORTS"],
    textual_hdrs = [
        "hwy/tests/test_util-inl.h",
        "hwy/tests/hwy_gtest.h",
    ],
    # Must not depend on a gtest variant, which can conflict with the
    # GUNIT_INTERNAL_BUILD_MODE defined by the test.
    deps = [
        ":hwy",
        ":nanobenchmark",
    ],
)

cc_binary(
    name = "benchmark",
    srcs = ["hwy/examples/benchmark.cc"],
    copts = COPTS,
    deps = [
        ":hwy",
        ":nanobenchmark",
    ],
)

cc_library(
    name = "skeleton",
    srcs = ["hwy/examples/skeleton.cc"],
    hdrs = ["hwy/examples/skeleton.h"],
    copts = COPTS,
    local_defines = ["hwy_EXPORTS"],
    textual_hdrs = ["hwy/examples/skeleton-inl.h"],
    deps = [
        ":hwy",
    ],
)

cc_test(
    name = "list_targets",
    size = "small",
    srcs = ["hwy/tests/list_targets.cc"],
    deps = [":hwy"],
)

# path, name
HWY_TESTS = [
    ("hwy/contrib/algo/", "copy_test"),
    ("hwy/contrib/algo/", "find_test"),
    ("hwy/contrib/algo/", "transform_test"),
    ("hwy/contrib/bit_pack/", "bit_pack_test"),
    ("hwy/contrib/dot/", "dot_test"),
    ("hwy/contrib/image/", "image_test"),
    ("hwy/contrib/math/", "math_test"),
    ("hwy/contrib/random/", "random_test"),
    ("hwy/contrib/matvec/", "matvec_test"),
    ("hwy/contrib/thread_pool/", "spin_test"),
    ("hwy/contrib/thread_pool/", "thread_pool_test"),
    ("hwy/contrib/thread_pool/", "topology_test"),
    ("hwy/contrib/unroller/", "unroller_test"),
    # contrib/sort has its own BUILD, we also add sort_test to GUITAR_TESTS.
    # To run bench_sort, specify --test=hwy/contrib/sort:bench_sort.
    ("hwy/examples/", "skeleton_test"),
    ("hwy/", "abort_test"),
    ("hwy/", "aligned_allocator_test"),
    ("hwy/", "base_test"),
    ("hwy/", "bit_set_test"),
    ("hwy/", "highway_test"),
    ("hwy/", "nanobenchmark_test"),
    ("hwy/", "perf_counters_test"),
    ("hwy/", "targets_test"),
    ("hwy/tests/", "arithmetic_test"),
    ("hwy/tests/", "bit_permute_test"),
    ("hwy/tests/", "blockwise_combine_test"),
    ("hwy/tests/", "blockwise_shift_test"),
    ("hwy/tests/", "blockwise_test"),
    ("hwy/tests/", "cast_test"),
    ("hwy/tests/", "combine_test"),
    ("hwy/tests/", "compare_test"),
    ("hwy/tests/", "compress_test"),
    ("hwy/tests/", "complex_arithmetic_test"),
    ("hwy/tests/", "concat_test"),
    ("hwy/tests/", "convert_test"),
    ("hwy/tests/", "count_test"),
    ("hwy/tests/", "crypto_test"),
    ("hwy/tests/", "demote_test"),
    ("hwy/tests/", "div_test"),
    ("hwy/tests/", "dup128_vec_test"),
    ("hwy/tests/", "expand_test"),
    ("hwy/tests/", "float_test"),
    ("hwy/tests/", "fma_test"),
    ("hwy/tests/", "foreach_vec_test"),
    ("hwy/tests/", "if_test"),
    ("hwy/tests/", "in_range_float_to_int_conv_test"),
    ("hwy/tests/", "interleaved_test"),
    ("hwy/tests/", "logical_test"),
    ("hwy/tests/", "mask_combine_test"),
    ("hwy/tests/", "mask_convert_test"),
    ("hwy/tests/", "mask_mem_test"),
    ("hwy/tests/", "mask_set_test"),
    ("hwy/tests/", "mask_slide_test"),
    ("hwy/tests/", "mask_test"),
    ("hwy/tests/", "masked_arithmetic_test"),
    ("hwy/tests/", "masked_minmax_test"),
    ("hwy/tests/", "memory_test"),
    ("hwy/tests/", "minmax_test"),
    ("hwy/tests/", "mul_by_pow2_test"),
    ("hwy/tests/", "mul_pairwise_test"),
    ("hwy/tests/", "mul_test"),
    ("hwy/tests/", "reduction_test"),
    ("hwy/tests/", "resize_test"),
    ("hwy/tests/", "reverse_test"),
    ("hwy/tests/", "rotate_test"),
    ("hwy/tests/", "saturated_test"),
    ("hwy/tests/", "shift_test"),
    ("hwy/tests/", "shuffle4_test"),
    ("hwy/tests/", "sign_test"),
    ("hwy/tests/", "slide_up_down_test"),
    ("hwy/tests/", "sums_abs_diff_test"),
    ("hwy/tests/", "swizzle_block_test"),
    ("hwy/tests/", "swizzle_test"),
    ("hwy/tests/", "table_test"),
    ("hwy/tests/", "test_util_test"),
    ("hwy/tests/", "truncate_test"),
    ("hwy/tests/", "tuple_test"),
    ("hwy/tests/", "widen_mul_test"),
]

HWY_TEST_COPTS = select({
    ":compiler_msvc": [],
    "//conditions:default": [
        # gTest triggers this warning (which is enabled by the
        # extra-semi in COPTS), so we need to disable it here,
        # but it's still enabled for :hwy.
        "-Wno-c++98-compat-extra-semi",
    ],
})

HWY_TEST_DEPS = [
    ":algo",
    ":bit_pack",
    ":bit_set",
    ":dot",
    ":hwy_test_util",
    ":hwy",
    ":image",
    ":math",
    ":matvec",
    ":nanobenchmark",
    ":perf_counters",
    ":random",
    ":skeleton",
    ":thread_pool",
    ":topology",
    ":timer",
    ":unroller",
    "//hwy/contrib/sort:vqsort",
] + select({
    ":compiler_msvc": [],
    "//conditions:default": ["@com_google_googletest//:gtest_main"],
})

[
    [
        cc_test(
            name = test,
            size = "medium",
            timeout = "long",  # default moderate is not enough for math_test
            srcs = [
                subdir + test + ".cc",
            ],
            copts = COPTS + HWY_TEST_COPTS,
            # Fixes OOM for matvec_test on RVV.
            exec_properties = select({
                "@platforms//cpu:riscv64": {"mem": "16g"},
                "//conditions:default": None,
            }),
            features = select({
                "@platforms//cpu:riscv64": ["fully_static_link"],
                "//conditions:default": [],
            }),
            linkopts = select({
                ":compiler_emscripten": [
                    "-s ASSERTIONS=2",
                    "-s ENVIRONMENT=node,shell,web",
                    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
                    "-s EXIT_RUNTIME=1",
                    "-s ALLOW_MEMORY_GROWTH=1",
                    "--pre-js $(location :preamble.js.lds)",
                ],
                "//conditions:default": [],
            }),
            linkstatic = select({
                "@platforms//cpu:riscv64": True,
                "//conditions:default": False,
            }),
            local_defines = ["HWY_IS_TEST"],
            # Placeholder for malloc, do not remove
            # for test_suite.
            tags = ["hwy_ops_test"],
            deps = HWY_TEST_DEPS + select({
                ":compiler_emscripten": [":preamble.js.lds"],
                "//conditions:default": [],
            }),
        ),
    ]
    for subdir, test in HWY_TESTS
]

# For manually building the tests we define here (:all does not work in --config=msvc)
test_suite(
    name = "hwy_ops_tests",
    tags = ["hwy_ops_test"],
)

# Placeholder for integration test, do not remove
