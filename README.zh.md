# 高效且具备高性能可移植性的向量化计算库

[//]: # (placeholder, do not remove)

**简体中文** | [**English**](README.md)

Highway 是一个提供可移植 SIMD/向量原语（intrinsics）的 C++ 核心库。

[官方在线文档](https://google.github.io/highway/en/master/)

此前采用 Apache 2 许可证发布，现已采用 Apache 2 / BSD-3 双重许可证授权。

## 研发初衷 (Why)

CPU（涵盖服务器、移动端与桌面设备）中蕴藏着巨大的未释放算力潜能。Highway 专为希望在软件层面稳健、经济地拓展性能极限的工程师而设计。

## 实现途径 (How)

现代 CPU 提供了 SIMD/向量指令集，可将同一操作并行施加于多个数据项上。由于执行的指令总数大幅减少，通常可将能耗降低至五分之一（*5 倍能效提升*），同时常常带来 *5 到 10 倍* 的实际执行加速。

Highway 遵循以下核心设计准则，使 SIMD/向量化编程变得切实可行与高效：

**符合直觉预期 (Does what you expect)**：Highway 作为一个经过严谨设计的 C++ 库，其函数接口能够高效精准地一对一映射至底层 CPU 机器指令，无需依赖编译器进行繁复的代码变换。与自动向量化（autovectorization）相比，Highway 生成的代码具备更高的可预测性，且在代码重构或编译器版本迭代时表现得更加稳健。

**跨主流平台通用 (Works on widely-used platforms)**：Highway 支持多达 7 种主流 CPU 架构；同一套应用层业务代码能够无缝适配多种指令集，包括变长/可伸缩向量指令集（向量长度在编译期未知，如 ARM SVE / RISC-V RVV）。Highway 仅要求 C++17（仅依赖语言特性，无需完整依赖现代标准库），并全面支持四大主流编译器家族。如需在其他架构平台上使用 Highway，欢迎提交 Issue 反馈。

**部署方式高度灵活 (Flexible to deploy)**：基于 Highway 构建的应用程序可在异构云环境或异构客户端设备上无缝运行，并在运行时动态选取当前 CPU 支持的最佳指令集（动态分发）。此外，开发者亦可选择针对单一目标指令集进行静态编译分发，完全免除任何运行时代销。在两种模式下，业务代码完全一致，仅需将 `HWY_STATIC_DISPATCH` 替换为 `HWY_DYNAMIC_DISPATCH` 并追加单行代码即可。详情可参阅 @kfjahnke 编写的[分发机制指南](https://github.com/kfjahnke/zimt/blob/main/examples/multi_isa_example/multi_simd_isa.md)。

**通用适配多元应用场景 (Suitable for a variety of domains)**：Highway 提供了极为丰富的向量算子集，已广泛应用于图像处理（浮点密集）、数据压缩、视频分析、线性代数、密码学、高性能排序以及随机数生成等领域。当全新业务场景需要扩充新算子时，在技术合理（例如在特定架构上不产生性能断崖）的前提下非常欢迎扩展新指令；如有需求可在 Issue 中共同讨论。

**充分释放数据并行架构潜力 (Rewards data-parallel design)**：Highway 提供了诸如 Gather（聚合读取）、MaskedLoad（掩码加载）以及 FixedTag 等工具，可直接为传统遗留数据结构带来可观加速。然而，最大的性能飞跃来自于针对可伸缩向量专门重新设计算法与数据结构。行之有效的关键技巧包括：批处理、结构体数组转数组结构体（SoA, Structure-of-Arrays）内存布局，以及对齐/填充式内存分配。

推荐参考以下入门资料：

-   [Highway SIMD 编程技术演讲](https://www.youtube.com/watch?v=R57biOOhnJM)
-   [面向 C++ 开发者的 SIMD 实战](http://const.me/articles/simd/simd.pdf)
-   [现代硬件算法与体系结构设计](https://en.algorithmica.org/hpc/)
-   [C++ 软件优化全景指南](https://agner.org/optimize/optimizing_cpp.pdf)
-   [在三大实战场景中通过 SIMD 原语实现性能跃升](https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/)

## 示例程序 (Examples)

基于 Compiler Explorer 的在线交互演示：

-   [多目标指令集运行时动态分发](https://gcc.godbolt.org/z/KM3ben7ET)
    （架构更完备、部署最灵活，能够自动利用当前 CPU 支持的最强 SIMD）
-   [通过编译选项 -m 针对单一目标静态构建](https://gcc.godbolt.org/z/rGnjMevKG)
    （形式更简单，但要求且仅使用编译器标志所启用的固定指令集）

`hwy/examples` 目录下提供了多个教学级别的示例代码；详细信息可参阅该目录下的 README.md：

基础示例：

-   `sum_array_advanced.cc`：4 路循环展开与尾部剩余元素处理。
-   `dot_product_unroll.cc`：与 sum_array_advanced 类似，外加向量规约计算。
-   `matrix_transpose_scatter_gather`：演示并对比 Scatter（离散写入）与 Gather（聚合读取）。

基础设施：

-   `benchmark.cc`：无尾量残留的高效点积循环与性能基准评测。
-   `profiler_example`：演示如何使用内置性能分析器（Profiler）测定带有注释标记区域的代码耗时。
-   `skeleton*`：支持运行时动态分发的完整模块骨架示例。

进阶挑战：

-   `masks_and_logic.cc`：使用 ASCII 字符艺术可视化展示掩码与逻辑比较操作。
-   `ctf_aes.cc`：夺旗赛（CTF）实战，通过向量化暴力破解 AES 密码。

通过 sourcegraph.com 检索，以下知名开源项目均已集成并使用 Highway（多数为 GitHub 仓库；如需收录您的项目，欢迎提交 Issue 或通过下方邮箱联系）：

*   音频处理：[Zimtohrli 感官知觉评估指标](https://github.com/google/zimtohrli)
*   浏览器核心：Chromium（+Vivaldi）、Firefox（+floorp / foxhound / librewolf / Waterfox）
*   计算生物学：[RNA 序列分析 (BPCells)](https://github.com/bnprks/BPCells)、[超长序列预处理 (fastplong)](https://github.com/OpenGene/fastplong)
*   计算机图形学：ghostty-org/ghostty、[稀疏体素渲染器 (voxl)](https://github.com/rools/voxl)、[腾讯 tgfx 2D 图形库](https://github.com/Tencent/tgfx)
*   密码学：google/distributed_point_functions、google/shell-encryption
*   数据结构：bkille/BitLib
*   图像编解码：eustas/2im、[Grok JPEG 2000](https://github.com/GrokImageCompression/grok)、[JPEG XL 官方参考实现](https://github.com/libjxl/libjxl)、[JPEGenc](https://github.com/osamu620/JPEGenc)、[Jpegli 高性能 JPEG 编解码库](https://github.com/google/jpegli)、[libaom](https://aomedia.googlesource.com/aom/)、[OpenHTJ2K](https://github.com/osamu620/OpenHTJ2K)
*   图像处理：awxkee/aire、cloudinary/ssimulacra2、[libvips 高性能图像处理库](https://github.com/libvips/libvips)、m-ab-s/media-autobuild_suite
*   图像查看器：AlienCowEatCake/ImageViewer、diffractor/diffractor、[Lux 全景与图像查看器](https://bitbucket.org/kfj/pv/)、mirillis/jpegxl-wic
*   信息检索与数据库：[iresearch 数据库索引库](https://github.com/iresearch-toolkit/iresearch)、michaeljclark/zvec、[nebula 交互式实时分析 / OLAP](https://github.com/varchar-io/nebula)、[`ScaNN` Google 大规模高维向量相似度检索库](https://github.com/google-research/google-research/tree/7a269cb2ce0ae1db591fe11b62cbc0be7d72532a/scann)
*   机器学习：array2d/deepx、[Google gemma.cpp 端侧轻量大模型推理引擎](https://github.com/google/gemma.cpp)、TensorFlow、NumPy、zpye/SimpleInfer
*   编程语言与运行时：[AOT 静态编译 Python (Codon)](https://github.com/exaloop/codon)、oven-sh/bun 高性能 JavaScript 运行时、V8/V8 引擎、yinqiwen/rapidudf
*   机器人与控制：[MIT 基于模型的系统设计与验证平台 (Drake)](https://github.com/RobotLocomotion/drake)
*   向量数据库与检索：1yefuwang1/vectorlite、vespa-engine/vespa

其他参考：

*   [C++ SIMD 库基准评估报告](https://www.mnm-team.org/pub/Fopras/rock23/)：“Highway 在涵盖多种 SIMD 扩展架构的测试中表现极为出众 […]。因此，对于各类软件工程项目而言，Highway 目前可能是最理想的 SIMD 库选型。”
*   [zimt](https://github.com/kfjahnke/zimt)：用于通过多线程 SIMD 代码处理 N 维数组的 C++11 模板库。
*   [向量化快速排序 (vectorized Quicksort)](https://github.com/google/highway/tree/master/hwy/contrib/sort)（[研究论文](https://arxiv.org/abs/2205.05982)）

Highway 亦可通过各大主流包管理器或分发软件源直接安装：

<details>
<summary>
点击展开 BSD 软件源
</summary>

*   [DragonFlyBSD](https://github.com/DragonFlyBSD/DPorts/tree/master/devel/highway)
*   [FreeBSD](https://cgit.freebsd.org/ports/tree/devel/highway)
*   [GhostBSD](https://github.com/ghostbsd/ghostbsd-ports/tree/main/devel/highway)
*   [MidnightBSD](https://www.midnightbsd.org/mports/devel/highway/README.html)

</details>

<details>
<summary>
点击展开跨平台包管理器
</summary>

*   [conan-io](https://conan.io/center/recipes/highway)
*   [conda-forge](https://anaconda.org/channels/conda-forge/packages/libhwy/overview)
*   [EasyBuild](https://docs.easybuild.io/version-specific/supported-software/h/Highway/)
*   [Homebrew](https://formulae.brew.sh/formula/highway)
*   [microsoft/vcpkg](https://vcpkg.io/en/package/highway)
*   [spack](https://packages.spack.io/package.html?name=highway)
*   [xmake-io/xmake-repo](https://github.com/xmake-io/xmake-repo/tree/dev/packages/h/highway)

</details>

<details>
<summary>
点击展开 Linux 发行版包管理源
</summary>

数据源自 [repology](https://repology.org/project/highway-simd-library/versions)

[![Packaging status](https://repology.org/badge/vertical-allrepos/highway-simd-library.svg)](https://repology.org/project/highway-simd-library/versions)
</details>

<details>
<summary>
点击展开 Windows 软件包源
</summary>

*   [fd00/yacp](https://github.com/fd00/yacp/tree/master/highway)
*   [MSYS2](https://packages.msys2.org/search?t=binpkg&q=highway)

</details>

<details>
<summary>
点击展开 Highway 适配的其他操作系统与运行环境
</summary>

*   [OPNsense](https://opnsense.org/)
*   [Xilinx/Vitis_Libraries](https://github.com/Xilinx/Vitis_Libraries)

</details>

## 当前支持状态 (Current status)

### 支持目标架构 (Targets)

Highway 支持多达 27 种硬件指令目标（按平台字母顺序列出）：

-   通用架构 (Any)：`EMU128`、`SCALAR`；
-   Armv7+：`NEON_WITHOUT_AES`、`NEON`、`NEON_BF16`、`SVE`、`SVE2`、`SVE_256`、`SVE2_128`；
-   IBM Z：`Z14`、`Z15`；
-   龙芯 LoongArch：`LSX`、`LASX`；
-   POWER：`PPC8` (v2.07)、`PPC9` (v3.0)、`PPC10` (v3.1B，由于编译器缺陷目前暂未正式启用，参见 #1207；且需要 QEMU 7.2)；
-   RISC-V：`RVV` (1.0)；
-   WebAssembly：`WASM`、`WASM_EMU256`（wasm128 的 2 倍展开版本，定义 `HWY_WANT_WASM2` 即可启用；该支持将持续保留，直至未来 WebAssembly 规范推出替代方案）；
-   x86 架构：
    -   `SSE2`
    -   `SSSE3`（约等同于 Intel Core 微架构）
    -   `SSE4`（约等同于 Nehalem 微架构，包含 AES + CLMUL）
    -   `AVX2`（约等同于 Haswell 微架构，包含 BMI2 + F16 + FMA）
    -   `AVX3`（约等同于 Skylake 微架构，AVX-512F/BW/CD/DQ/VL）
    -   `AVX3_DL`（约等同于 Icelake 微架构，包含 `BitAlg` + `CLMUL` + `GFNI` + `AVX512IFMA` + `VAES` + `VBMI` + `VBMI2` + `AVX512VNNI` + `VPOPCNT`）
    -   `AVX3_ZEN4`（在 AVX3_DL 基础上增加 BF16，专为 AMD Zen4 架构优化；静态分发编译时需定义 `HWY_WANT_AVX3_ZEN4` 显式开启，运行时动态分发默认启用）
    -   `AVX3_SPR`（约等同于 Sapphire Rapids 微架构，包含 AVX-512FP16 + AMX-TILE + AMX-BF16 + AMX-INT8）
    -   `AVX10_2`（约等同于 Diamond Rapids 微架构）

项目维护策略：除另有说明外，只要目标架构能够使用当前主流支持的 Clang 或 GCC 进行（交叉）编译并通过 QEMU 模拟测试，Highway 就会持续保持支持。若目标能够在 LLVM 主分支源码下直接编译，且无需额外标志即可在项目配套的 QEMU 版本上通过测试，则会被纳入持续集成（CI）自动化测试基建。其余架构则会在正式发版前，采用精选的 Clang 和 GCC 版本与配置进行人工全量验证。

SVE 架构早期测试曾借助于 farm_sve（详见致谢部分）。

### 版本管理规范 (Versioning)

Highway 发版遵循语义化版本规范 semver.org（主版本号.次版本号.修订号，即 MAJOR.MINOR.PATCH）：引入向后兼容的新增特性时增加 MINOR，引入向后兼容的问题修复时增加 PATCH。建议在生产环境中优先使用正式发布版本（Release）而非 Git 主干最新提交，因为正式版本经过了更广泛的测试矩阵验证。

当前 1.0+ 版本标志着对向后兼容性（Backwards Compatibility）的严格承诺。调用已记录文档接口的应用程序，将保持与同一主版本号下后续更新的无缝兼容。

### 测试保障体系 (Testing)

持续集成测试（CI）在最新版本的 Clang（原生运行于 x86，或通过 QEMU 模拟运行 RISC-V 与 Arm）以及 MSVC 2019（v19.28，原生运行于 x86）环境下自动化执行构建与回归。

在版本发布前，还会通过 Clang 和 GCC 在 x86 上进行全套构建测试，并通过 GCC 交叉编译针对 Armv7/8 进行验证。详细流程请参阅 `release_testing_process.md`。

### 扩展模块 (Related modules)

`contrib` 目录包含一系列 SIMD 扩展实用工具库：支持行对齐的图像容器类、数学函数库（已实现包括三角函数在内的 16 种常用算子），以及用于计算向量点积和向量化排序的高性能工具。

### 其他 SIMD 库对比 (Other libraries)

若仅需要支持 x86 架构，可考虑 Agner Fog 开发的 [VCL (Vector Class Library)](https://github.com/vectorclass)，该库功能全面且包含一套完整的数学库。

若已有基于 x86/NEON 原生 intrinsics 编写的代码，可参考 [SIMDe](https://github.com/simd-everywhere/simde)，它能够利用其他目标平台的原语或自动向量化机制来模拟这些 intrinsics。

[xSIMD](https://github.com/xtensor-stack/xsimd) 是一个仅包含头文件的 C++ 库，支持 Arm、Power、RISC-V、WebAssembly 和 x86 平台；具备高级抽象接口，但支持的操作算子相对较少。

[NumKong](https://github.com/ashvardanian/NumKong) 是一个聚焦于点积和混合精度矩阵乘法等操作的 SIMD 加速 C 数学库，支持在 C++、Go、Python、Rust、Swift 和 WebAssembly 中调用，并在 Arm、LoongArch、Power、RISC-V 和 x86 上提供了硬件加速实现。

## 安装说明 (Installation)

本项目使用 CMake 生成并构建。在 Debian / Ubuntu 等衍生系统上可通过以下命令安装依赖：

```bash
sudo apt install cmake
```

Highway 的单元测试依赖 [googletest](https://github.com/google/googletest)。默认情况下，Highway 的 CMake 配置会在构建配置阶段自动拉取此依赖。若希望避免联网拉取，可将 CMake 变量 `HWY_SYSTEM_GTEST` 设为 ON，并单独在系统层面安装 gtest：

```bash
sudo apt install libgtest-dev
```

此外，亦可定义 `HWY_TEST_STANDALONE=1` 并移除各 BUILD 文件中所有出现的 `gtest_main`，使测试脱离对 GUnit 的依赖。

运行交叉编译测试需要操作系统底层的二进制格式支持，在 Debian 系统上由 `qemu-user-binfmt` 软件包提供。

可根据 `BUILD_SHARED_LIBS` 的配置将 Highway 构建为动态库或静态库，标准 CMake 构建流程如下：

```bash
mkdir -p build && cd build
cmake ..
make -j && make test
```

亦可直接运行 `run_tests.sh` 脚本（Windows 环境下运行 `run_tests.bat`）。

Highway 同样支持使用 Bazel 构建，但在社区中的使用与测试频度低于 CMake。

当面向 Armv7 目标构建时，由于当前编译器的局限性，需在 CMake 命令行中追加 `-DHWY_CMAKE_ARM7:BOOL=ON`（参见 #834 与 #1032）；相关消除此限制的底层工作正在推进中。

若希望利用 Armv8/v9 的 `vusdot` 和 `vusdotq` 点积扩展指令，在目标 CPU 具备对应硬件支持的前提下，可在编译器标志 `-march` 中追加 `+i8mm`。

32 位 x86 并非官方正式支持的平台，该架构下默认禁用 AVX2/3。需要说明的是，社区贡献者 johnplatts 已在 GCC 7/8 和 Clang 8/11/12 环境下成功在 32 位 x86 上编译并运行了包含 AVX2/3 在内的 Highway 测试。在 Ubuntu 22.04 上，Clang 11 和 12（更高版本无此要求）需要额外的编译器选项 `-m32 -isystem /usr/i686-linux-gnu/include`；Clang 10 及更早版本则在此基础上还需追加 `-isystem /usr/i686-linux-gnu/include/c++/12/i686-linux-gnu`。详情参见 [#1279](https://github.com/google/highway/issues/1279)。

## 使用 vcpkg 构建 Highway (Building highway - Using vcpkg)

Highway 目前已收录于 [vcpkg](https://github.com/Microsoft/vcpkg) 官方包管理器：

```bash
vcpkg install highway
```

vcpkg 中的 Highway 移植包由微软团队成员与开源社区共同维护。若版本滞后，欢迎前往 [vcpkg 代码仓库创建 Issue 或提交 Pull Request](https://github.com/Microsoft/vcpkg)。

## 快速上手 (Quick start)

推荐以 `examples/` 目录下的 `benchmark` 程序作为实践起点。

-   `quick_reference.md`：简要列举了全部操作算子及其参数签名。
-   `instruction_matrix.md`：详细标明了每个操作算子在各硬件指令集下的实际指令开销。
-   `faq.md`：集中解答关于跨平台可移植性、API 设计原则以及进阶指引的常见问题。

为了获取最极致的跨平台性能，建议尽可能使用完整的原生 SIMD 向量。可通过向 `Zero/Set/Load` 等函数传递 `ScalableTag<float>`（或等效宏 `HWY_FULL(float)`）类型标记来获取。对于需要限制向量通道数量上限的特殊场景，Highway 提供了以下两种备选方案：

-   限制至多 `N` 个通道：指定 `CappedTag<T, N>` 或等效宏 `HWY_CAPPED(T, N)`。实际通道数将为 `N` 向下取整至最接近的 2 的幂（例如 `N` 为 5 时取 4，`N` 为 8 时取 8）。此特性适用于狭长矩阵等特殊数据结构。由于底层向量实际通道数可能小于 `N`，代码中仍需保留循环结构。
-   精确指定为 2 的幂次 `N` 个通道：指定 `FixedTag<T, N>`。支持的最大 `N` 取决于目标架构，但保证至少不低于 `16/sizeof(T)`。

受 C++ 参数依赖查找（ADL）规则限制，调用 Highway 算子的用户代码必须遵循以下三者之一：

*   编写在命名空间 `namespace hwy { namespace HWY_NAMESPACE {` 之内；或
*   为命名空间定义别名并为每个算子添加前缀，例如：`namespace hn = hwy::HWY_NAMESPACE; hn::Add();`；或
*   针对所调用的算子添加 using 声明，例如：`using hwy::HWY_NAMESPACE::Add;`。

此外，每个调用 Highway 算子（如 `Load`）的函数必须：使用 `HWY_ATTR` 前缀修饰，或者置于 `HWY_BEFORE_NAMESPACE()` 与 `HWY_AFTER_NAMESPACE()` 宏之间。Lambda 表达式目前需要在其函数体左大括号 `{` 之前添加 `HWY_ATTR`。

切勿为 SIMD 向量使用命名空间作用域变量或 `static` 静态初始化器，因为在运行时动态分发模式下，若编译器选取了当前 CPU 不支持的目标架构初始化器，将引发非法指令异常（SIGILL）。相反，通过 `Set` 进行初始化的常量通常应作为局部（const）变量声明。

根据采用的是静态分发还是动态分发，使用 Highway 代码的入口点定义略有不同。在两种模式下，均建议顶层函数接收一个或多个底层数组指针，而非具体目标架构专有的向量类型：

*   **静态分发 (Static dispatch)**：`HWY_TARGET` 将自动设为 `HWY_BASELINE_TARGETS`（即当前编译器允许使用的指令集，参见 `quick_reference.md`）中的最优目标。在同一模块内，可通过 `HWY_STATIC_DISPATCH(func)(args)` 直接调用定义在 `HWY_NAMESPACE` 内的函数。如需从其他模块跨文件调用，可将其封装在常规函数中并在头文件中声明该常规函数。
*   **动态分发 (Dynamic dispatch)**：通过 `HWY_EXPORT` 宏生成函数指针查找表，进而通过 `HWY_DYNAMIC_DISPATCH(func)(args)` 自动调用匹配当前 CPU 支持特性的最佳目标函数指针。如果定义了 `HWY_TARGET_INCLUDE` 并包含了 `foreach_target.h`，该模块将自动针对 `HWY_TARGETS` 中的每个目标分别编译（参见 `quick_reference.md`）。需要注意的是，首次调用 `HWY_DYNAMIC_DISPATCH` 或首次调用 `HWY_DYNAMIC_POINTER` 返回的函数指针时，会产生微小的 CPU 硬件特性探测开销。若希望消除这一开销，可在任何 `HWY_DYNAMIC_*` 调用前显式执行：`hwy::GetChosenTarget().Update(hwy::SupportedTargets());`。

另可参阅 @kfjahnke 编写的[动态分发机制深入解析](https://github.com/kfjahnke/zimt/blob/multi_isa/examples/multi_isa_example/multi_simd_isa.md)。

使用动态分发时，`foreach_target.h` 必须在编译单元（.cc 文件）中包含，而非头文件中。在多个编译单元间共享的向量计算头文件需要使用特殊的 include guard 头文件保护，例如以下来自 `examples/skeleton-inl.h` 的标准模板：

```cpp
#if defined(HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_
#undef HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_
#else
#define HIGHWAY_HWY_EXAMPLES_SKELETON_INL_H_
#endif

#include "hwy/highway.h"
// 向量化业务代码
#endif
```

按照通用惯例，此类头文件通常以 `-inl.h` 结尾命名，因为其内容（通常为函数模板）大部分会被内联展开。

## 编译器选项配置 (Compiler flags)

应用程序在编译时必须开启优化选项。若未启用函数内联（Inlining），SIMD 向量代码的运行速度可能会骤降 10 到 100 倍。对于 Clang 和 GCC，通常指定 `-O2` 即可达到预期优化效果。

在使用 Clang 和 GCC 时，推荐采用运行时动态分发（详见“快速上手”章节），这既能避免配置可能与其他第三方库产生冲突的额外编译器选项，又能始终自适应利用当前 CPU 的最强指令集。若明确知晓程序运行的目标 CPU 架构，可将其指定为基准目标（Baseline target），从而避免针对陈旧指令集生成多余冗余代码。编译器选项必须与 `detect_targets.h` 中针对 `HWY_BASELINE_*` 的预定义宏检查保持匹配，亦可从 `set_macros-inl.h` 中的 `HWY_TARGET_STR` 推导得出。对于 x86 架构，当前推荐的编译器标志如下：

-   `HWY_AVX2`：`-march=x86-64-v3 -maes -mpclmul`，或 `-march=haswell -maes`
-   `HWY_AVX3`：`-march=x86-64-v4`，或 `-march=skx`
-   `HWY_AVX3_DL`：`-march=icelake-server`
-   `HWY_ZEN4`：`-march=znver4`
-   `HWY_AVX3_SPR`：`-march=icelake-server -mavx512fp16 -mavx512bf16 -mamx-tile -mamx-bf16 -mamx-int8`，或 `-march=sapphirerapids`
-   `HWY_AVX10_2`：`-march=novalake`（需配合 GCC 16 或 Clang 22）

同时可参阅上述 Godbolt 在线示例中关于 `-m` 目标选项的用法。

对于 MSVC，建议在编译时开启 `/Gv` 选项，以允许非内联函数直接通过硬件寄存器传递向量参数。若计划将 AVX2 目标与半宽向量（例如配合 `PromoteTo`）组合使用，务必同时开启 `/arch:AVX2`。这是在 MSVC 上可靠生成 VEX 编码 SSE 指令的唯一途径。虽然在与 AVX 混合使用时 MSVC 有时会自动生成 VEX 编码的 SSE 指令，但这并非绝对保证，参见 [DevCom-10618264](https://developercommunity.visualstudio.com/t/10618264)。若混合使用 VEX 编码的 AVX2 指令与非 VEX 编码的传统 SSE 指令，可能导致严重的 CPU 状态切换惩罚（AVX-SSE 性能断崖）。遗憾的是，使用 `/arch:AVX2` 选项后编译出的二进制文件将硬性要求运行平台支持 AVX2。需要注意的是，Clang 和 GCC 无需此类全局选项，因为它们原生支持基于特定目标函数的特性属性（Attributes），Highway 已利用该机制确保为 AVX2 目标正确生成 VEX 代码。

## 循环条带化处理 (Strip-mining loops)

在对循环进行向量化重构时，核心挑战之一在于如何妥善处理无法被向量通道数 `N = Lanes(d)` 整除的迭代总次数（称为 "trip count"，记作 `count`）。例如，必须严防发生超出数组边界的越界写入。

本节中，设 `T` 代表元素数据类型，`d = ScalableTag<T>`。假设循环体定义为函数 `template<bool partial, class D> void LoopBody(D d, size_t index, size_t max_n)`。

“条带化”（Strip-mining）是一种经典的循环向量化技术：通过将原循环变换为外层与内层双重循环，使内层循环的迭代步长精确匹配向量宽度，随后以内层单次向量操作整体替代内层循环。

Highway 为循环向量化提供了多种实用策略：

*   **策略一：确保所有输入/输出数据缓冲区均已填充对齐 (Padded)**。此时（外层）循环极其精简：

    ```cpp
    for (size_t i = 0; i < count; i += N) LoopBody<false>(d, i, 0);
    ```

    在此模式下，甚至无需模板参数与第二个函数实参。

    这是首选推荐方案，除非 `N` 达到数千规模且向量操作存在较长流水线延迟。这在 90 年代的向量超级计算机中较为普遍，但现代 CPU 的 ALU 计算资源充沛，主流硬件实现通常将向量拆分为 1、2 或 4 部分执行，因此即使未利用全部通道，处理完整向量的开销也极其微小。该策略不仅能完全规避陈旧目标平台上谓词掩码或部分加载/存储的高昂性能代销，而且杜绝了代码膨胀。

*   **策略二：始终处理完整向量，最后一个向量重叠覆盖先前已处理元素**：

    ```cpp
    for (size_t i = 0; i < count; i += N) LoopBody<false>(d, HWY_MIN(i, count - N), 0);
    ```

    在满足 `count >= N` 且 `LoopBody` 具有幂等性（Idempotent，重复计算相同数据不影响最终结果）的前提下，这是次优推荐策略。虽然末尾少量元素可能被计算了两次，但能够保持单一执行路径且全程满载向量化，通常极具性能收益。即便 `count < N`，将输入/输出预先对齐填充至 `N` 通常也是合理的。

*   **策略三：使用 `hwy/contrib/algo/transform-inl.h` 中的 `Transform*` 系列函数**。该工具自动接管整个循环与尾部余量处理，只需定义一个泛型 Lambda 表达式（C++14）或仿函数，接收输入/输出数组的当前向量，并可额外接收最多两个额外输入数组的向量，最后返回写入输出数组的值。

    以下为实现 BLAS 标准函数 SAXPY（即 `alpha * x + y`）的简洁示例：

    ```cpp
    Transform1(d, x, n, y, [](auto d, const auto v, const auto v1) HWY_ATTR {
      return MulAdd(Set(d, alpha), v, v1);
    });
    ```

*   **策略四：先处理全宽向量，尾部剩余元素由标量循环处理**：

    ```cpp
    size_t i = 0;
    for (; i + N <= count; i += N) LoopBody<false>(d, i, 0);
    for (; i < count; ++i) LoopBody<false>(CappedTag<T, 1>(), i, 0);
    ```

    在此方案中同样无需模板参数与第二个函数实参。

    该策略避免了代码重复；当 `count` 规模较大时非常实用。但如果 `count` 较小，第二阶段标量循环的处理速度可能落后于下一备选方案。

*   **策略五：先处理全宽向量，尾部单次调用带有掩码逻辑的特化 `LoopBody`**：

    ```cpp
    size_t i = 0;
    for (; i + N <= count; i += N) {
      LoopBody<false>(d, i, 0);
    }
    if (i < count) {
      LoopBody<true>(d, i, count - i);
    }
    ```

    在此结构中，模板参数与第三个实参可在 `LoopBody` 内部用于非原子式地将向量 `v` 的前 `num_remaining` 个通道与目标内存后续位置的原内容进行混合（Blend）：`BlendedStore(v, FirstN(d, num_remaining), d, pointer);`。类似地，`MaskedLoad(FirstN(d, num_remaining), d, pointer)` 可加载前 `num_remaining` 个元素，并将其余通道清零。

    当无法确保向量缓冲区已对齐填充时，此策略是一个优秀的默认选择，但请注意仅在 `#if !HWY_MEM_OPS_MIGHT_FAULT` 时安全生效！相比标量循环，该策略只需一次收尾迭代。虽然由两个循环体构成的代码体积略微增加，但由于避免了在主循环迭代中施加掩码计算的开销，整体性能收益通常十分可观。

## 补充资源 (Additional resources)

*   Highway 入门介绍幻灯片：`highway_intro.pdf`
*   各硬件架构下各操作算子的指令数全景对照表：`instruction_matrix.md`
*   设计哲学与技术选型对比：`design_philosophy.md`
*   底层实现细节解析：`impl_details.md`

## 致谢 (Acknowledgments)

在开发与适配过程中使用了 Berenger Bramas 开发的 [farm-sve](https://gitlab.inria.fr/bramas/farm-sve) 工具；该工具在 x86 开发机上对 SVE 端口的移植与验证发挥了巨大作用。

本项目并非 Google 官方正式支持的产品。
联系方式：janwas@google.com
