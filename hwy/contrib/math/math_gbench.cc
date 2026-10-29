// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>

#include <vector>

#include "third_party/benchmark/include/benchmark/benchmark.h"

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/math/math_gbench.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
// clang-format on

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

void EvalExpF32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Exp(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalExpF64(const double* HWY_RESTRICT in, double* HWY_RESTRICT out,
                size_t n) {
  const hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Exp(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalExp2F32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                 size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Exp2(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalExpm1F32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                  size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Expm1(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalExpm1F64(const double* HWY_RESTRICT in, double* HWY_RESTRICT out,
                  size_t n) {
  const hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Expm1(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalSinhF32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                 size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Sinh(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalCoshF32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                 size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Cosh(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalTanhF32(const float* HWY_RESTRICT in, float* HWY_RESTRICT out,
                 size_t n) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Tanh(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

void EvalTanhF64(const double* HWY_RESTRICT in, double* HWY_RESTRICT out,
                 size_t n) {
  const hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);
  for (size_t i = 0; i < n; i += lanes) {
    hn::StoreU(hn::Tanh(d, hn::LoadU(d, in + i)), d, out + i);
  }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {

HWY_EXPORT(EvalExpF32);
HWY_EXPORT(EvalExpF64);
HWY_EXPORT(EvalExp2F32);
HWY_EXPORT(EvalExpm1F32);
HWY_EXPORT(EvalExpm1F64);
HWY_EXPORT(EvalSinhF32);
HWY_EXPORT(EvalCoshF32);
HWY_EXPORT(EvalTanhF32);
HWY_EXPORT(EvalTanhF64);

void BM_Exp_F32(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0f + 16.0f * static_cast<float>(i % 256) / 255.0f;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalExpF32)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Exp_F32)->Arg(256)->Arg(4096);

void BM_Exp_F64(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<double> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0 + 16.0 * static_cast<double>(i % 256) / 255.0;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalExpF64)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Exp_F64)->Arg(256)->Arg(4096);

void BM_Exp2_F32(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0f + 16.0f * static_cast<float>(i % 256) / 255.0f;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalExp2F32)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Exp2_F32)->Arg(256)->Arg(4096);

void BM_Expm1_F32(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0f + 16.0f * static_cast<float>(i % 256) / 255.0f;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalExpm1F32)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Expm1_F32)->Arg(256)->Arg(4096);

void BM_Expm1_F64(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<double> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0 + 16.0 * static_cast<double>(i % 256) / 255.0;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalExpm1F64)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Expm1_F64)->Arg(256)->Arg(4096);

void BM_Sinh_F32(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0f + 16.0f * static_cast<float>(i % 256) / 255.0f;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalSinhF32)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Sinh_F32)->Arg(256)->Arg(4096);

void BM_Cosh_F32(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -8.0f + 16.0f * static_cast<float>(i % 256) / 255.0f;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalCoshF32)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Cosh_F32)->Arg(256)->Arg(4096);

void BM_Tanh_F32(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -5.0f + 10.0f * static_cast<float>(i % 256) / 255.0f;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalTanhF32)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Tanh_F32)->Arg(256)->Arg(4096);

void BM_Tanh_F64(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  std::vector<double> in(n), out(n);
  for (size_t i = 0; i < n; ++i) {
    in[i] = -5.0 + 10.0 * static_cast<double>(i % 256) / 255.0;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(in.data());
    HWY_DYNAMIC_DISPATCH(EvalTanhF64)(in.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_Tanh_F64)->Arg(256)->Arg(4096);

}  // namespace
}  // namespace hwy
#endif  // HWY_ONCE
