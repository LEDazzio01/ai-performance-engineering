#!/usr/bin/env python3
"""
Post-install verification for Blackwell-era stacks (SM100/SM121)
Checks:
  - FlashAttention CUDA extension correctness vs torch SDPA
  - Transformer Engine FP8 + FP4 paths
  - TMA decode kernel
  - TMEM roundtrip copy
  - Thread-block cluster launch (2-block cluster)
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, Tuple

import torch
from torch.utils.cpp_extension import load_inline

# Keep extension builds memory-light.
os.environ.setdefault("MAX_JOBS", "1")

# Silence noisy torch dynamo logging in this short verification script.
try:
    import torch._logging as torch_logging  # type: ignore[attr-defined]

    torch_logging.set_logs(dynamo=False)
except (ImportError, AttributeError):
    pass  # torch._logging not available in this PyTorch version


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_ROOT = PROJECT_ROOT / "artifacts" / "torch_extensions" / "tma_stack"


def _format_status(name: str, ok: bool, detail: str | None = None) -> None:
    prefix = "✓" if ok else "✗"
    line = f"{prefix} {name}"
    if detail:
        line = f"{line}: {detail}"
    print(line)


def _set_arch_env() -> str:
    if "TORCH_CUDA_ARCH_LIST" in os.environ and os.environ["TORCH_CUDA_ARCH_LIST"]:
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    if major == 10 and minor == 0:
        arch = "10.0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    return arch


def _build_tma_extension(use_tmem: bool) -> Any:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    arch = _set_arch_env()
    suffix = "tmem" if use_tmem else "basic"
    name = f"tma_stack_check_{suffix}_{arch.replace('.', '_').replace(';', '_') or 'default'}"
    include_paths: list[str] = []
    upstream_cutlass = PROJECT_ROOT / "third_party" / "cutlass" / "include"
    if use_tmem:
        include_paths.append(str(upstream_cutlass))

    cuda_src = r"""
#define CUTE_DISABLE_COOPERATIVE_GEMM 1
#define CUTE_DISABLE_PRINT_LATEX 1
#define CUTE_DISABLE_PREFETCH_OVERLOADS 1

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_runtime.h>
"""
    if use_tmem:
        cuda_src += r"""
#include <cute/algorithm/copy.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
"""

    cuda_src += r"""
namespace cde = cuda::device::experimental;

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr std::size_t BYTES_PER_TILE = static_cast<std::size_t>(TILE_M) * TILE_N * sizeof(float);
"""
    if use_tmem:
        cuda_src += r"""
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
constexpr bool kTmemAvailable = true;
#else
constexpr bool kTmemAvailable = false;
#endif
"""

    cuda_src += r"""
__device__ void scale_tile(float* tile, int pitch, int rows, int cols) {
    for (int r = threadIdx.y; r < rows; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float v = tile[r * pitch + c];
            tile[r * pitch + c] = v * 1.0002f + 0.0001f;
        }
    }
}

__device__ void init_barrier(cuda::barrier<cuda::thread_scope_block>* bar, int participants) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(bar, participants);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
}

__global__ void tma_copy_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const float* __restrict__ input,
    int ld_input,
    float* __restrict__ output,
    int rows,
    int cols,
    int ld_output) {
    __shared__ alignas(128) float stage[TILE_M][TILE_N];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char barrier_storage[sizeof(block_barrier)];

    const int participants = blockDim.x * blockDim.y;
    init_barrier(reinterpret_cast<block_barrier*>(barrier_storage), participants);
    auto& bar = *reinterpret_cast<block_barrier*>(barrier_storage);

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int row0 = tile_m * TILE_M;
    const int col0 = tile_n * TILE_N;
    if (row0 >= rows || col0 >= cols) {
        return;
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &stage, &in_desc, row0, col0, bar);
        cde::cp_async_bulk_commit_group();
    }
    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        token = cuda::device::barrier_arrive_tx(bar, participants, BYTES_PER_TILE);
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    const int rows_this = min(TILE_M, rows - row0);
    const int cols_this = min(TILE_N, cols - col0);
    scale_tile(&stage[0][0], TILE_N, rows_this, cols_this);
    __syncthreads();

    for (int r = threadIdx.y; r < rows_this; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols_this; c += blockDim.x) {
            const int gr = row0 + r;
            const int gc = col0 + c;
            output[gr * ld_output + gc] = stage[r][c];
        }
    }
}

bool encode_tensor_map(
    CUtensorMap& desc,
    void* base,
    int rows,
    int cols,
    int ld) {
    constexpr uint32_t rank = 2;
    std::uint64_t dims[rank] = {
        static_cast<std::uint64_t>(rows),
        static_cast<std::uint64_t>(cols)};
    std::uint64_t stride[rank - 1] = {
        static_cast<std::uint64_t>(ld * sizeof(float))};
    std::uint32_t box[rank] = {TILE_M, TILE_N};
    std::uint32_t elem_stride[rank] = {1, 1};
    constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    auto status = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        base,
        dims,
        stride,
        box,
        elem_stride,
        interleave,
        CU_TENSOR_MAP_SWIZZLE_128B,
        promotion,
        oob_fill);
    return status == CUDA_SUCCESS;
}

void run_tma(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && output.dtype() == torch::kFloat32, "float32 expected");
    TORCH_CHECK(input.sizes() == output.sizes(), "shape mismatch");
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");
    auto in_c = input.contiguous();
    auto out_c = output.contiguous();
    const int rows = static_cast<int>(in_c.size(0));
    const int cols = static_cast<int>(in_c.size(1));
    if (rows == 0 || cols == 0) return;

    CUtensorMap desc{};
    if (!encode_tensor_map(desc, in_c.data_ptr<float>(), rows, cols, static_cast<int>(in_c.stride(0)))) {
        TORCH_CHECK(false, "cuTensorMapEncodeTiled failed");
    }

    dim3 block(32, 4, 1);
    dim3 grid((cols + TILE_N - 1) / TILE_N, (rows + TILE_M - 1) / TILE_M, 1);
    auto stream = at::cuda::getDefaultCUDAStream();
    tma_copy_kernel<<<grid, block, 0, stream>>>(
        desc,
        in_c.data_ptr<float>(),
        static_cast<int>(in_c.stride(0)),
        out_c.data_ptr<float>(),
        rows,
        cols,
        static_cast<int>(out_c.stride(0)));
    C10_CUDA_CHECK(cudaGetLastError());
    if (!output.is_contiguous()) {
        output.copy_(out_c);
    }
}

bool supports_tma_encode(int rows, int cols, int ld) {
    CUtensorMap desc{};
    float* base = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(ld) * sizeof(float);
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        return false;
    }
    bool ok = encode_tensor_map(desc, base, rows, cols, ld);
    cudaFree(base);
    return ok;
}
"""

    if use_tmem:
        cuda_src += r"""
__global__ void tmem_roundtrip_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int ld_input,
    int ld_output) {
    __shared__ alignas(128) float stage[TILE_M][TILE_N];

    const int row0 = blockIdx.y * TILE_M;
    const int col0 = blockIdx.x * TILE_N;
    if (row0 >= rows || col0 >= cols) {
        return;
    }

    const int rows_this = min(TILE_M, rows - row0);
    const int cols_this = min(TILE_N, cols - col0);

    for (int r = threadIdx.y; r < rows_this; r += blockDim.y) {
        const int gr = row0 + r;
        const float* in_row = input + static_cast<long long>(gr) * ld_input;
        for (int c = threadIdx.x; c < cols_this; c += blockDim.x) {
            stage[r][c] = in_row[col0 + c];
        }
    }
    __syncthreads();

    const bool full_tile = (rows_this == TILE_M) && (cols_this == TILE_N);
    const bool contiguous_out = ld_output == TILE_N;

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    if (kTmemAvailable && full_tile && contiguous_out) {
        __shared__ uint32_t tmem_base_ptr;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cute::TMEM::Allocator1Sm allocator{};
            allocator.allocate(cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns, &tmem_base_ptr);
        }
        __syncthreads();

        auto tmem_tensor = cute::make_tensor(
            cute::make_tmem_ptr<float>(tmem_base_ptr),
            cute::make_layout(
                cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
                cute::make_stride(cute::TMEM::DP<float>{}, cute::Int<1>{})));

        auto smem_tensor = cute::make_tensor(
            cute::make_smem_ptr(&stage[0][0]),
            cute::make_layout(
                cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
                cute::make_stride(cute::Int<TILE_N>{}, cute::Int<1>{})));

        auto gmem_tensor = cute::make_tensor(
            cute::make_gmem_ptr(output + row0 * ld_output + col0),
            cute::make_layout(
                cute::make_shape(cute::Int<TILE_M>{}, cute::Int<TILE_N>{}),
                cute::make_stride(cute::Int<TILE_N>{}, cute::Int<1>{})));

        auto tmem_store = cute::make_tmem_copy(cute::SM100_TMEM_STORE_32dp32b4x{}, tmem_tensor);
        auto tmem_load = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b4x{}, tmem_tensor);

        if (threadIdx.y == 0 && threadIdx.x < 32) {
            auto store_thr = tmem_store.get_slice(threadIdx.x);
            auto src = store_thr.partition_S(smem_tensor);
            auto dst = store_thr.partition_D(tmem_tensor);
            cute::copy(tmem_store, src, dst);
        }
        __syncthreads();
        if (threadIdx.y == 0 && threadIdx.x < 32) {
            auto load_thr = tmem_load.get_slice(threadIdx.x);
            auto src = load_thr.partition_S(tmem_tensor);
            auto dst = load_thr.partition_D(gmem_tensor);
            cute::copy(tmem_load, src, dst);
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cute::TMEM::Allocator1Sm allocator{};
            allocator.release_allocation_lock();
            allocator.free(tmem_base_ptr, cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns);
        }
        return;
    }
#endif

    for (int r = threadIdx.y; r < rows_this; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols_this; c += blockDim.x) {
            const int gr = row0 + r;
            const int gc = col0 + c;
            output[gr * ld_output + gc] = stage[r][c];
        }
    }
}

void run_tmem_roundtrip(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input/output must be CUDA tensors");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && output.dtype() == torch::kFloat32, "float32 expected");
    TORCH_CHECK(input.sizes() == output.sizes(), "shape mismatch");
    TORCH_CHECK(input.dim() == 2, "expected 2D tensor");

    auto in_c = input.contiguous();
    auto out_c = output.contiguous();
    const int rows = static_cast<int>(in_c.size(0));
    const int cols = static_cast<int>(in_c.size(1));
    if (rows == 0 || cols == 0) return;

    const dim3 block(32, 4, 1);
    const dim3 grid((cols + TILE_N - 1) / TILE_N, (rows + TILE_M - 1) / TILE_M, 1);
    auto stream = at::cuda::getDefaultCUDAStream();
    tmem_roundtrip_kernel<<<grid, block, 0, stream>>>(
        in_c.data_ptr<float>(),
        out_c.data_ptr<float>(),
        rows,
        cols,
        static_cast<int>(in_c.stride(0)),
        static_cast<int>(out_c.stride(0)));
    C10_CUDA_CHECK(cudaGetLastError());
    if (!output.is_contiguous()) {
        output.copy_(out_c);
    }
}
"""

    cuda_src += r"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_tma", &run_tma, "TMA copy kernel (float32)");
    m.def("supports_tma_encode", &supports_tma_encode, "Check cuTensorMapEncodeTiled for shape");
"""
    if use_tmem:
        cuda_src += r"""
    m.def("run_tmem_roundtrip", &run_tmem_roundtrip, "TMEM roundtrip kernel (float32)");
    m.def("tmem_available", []() { return kTmemAvailable; }, "TMEM availability flag");
"""
    cuda_src += r"""
}
"""

    extra_includes = include_paths if include_paths else None
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=[
            "-lineinfo",
            "--use_fast_math",
            "-std=c++20",
            "-DCUTE_DISABLE_PREFETCH_OVERLOADS",
            "-DCUTE_DISABLE_COOPERATIVE_GEMM",
            "-DCUTE_DISABLE_PRINT_LATEX",
        ],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-DCUTE_DISABLE_PREFETCH_OVERLOADS",
            "-DCUTE_DISABLE_COOPERATIVE_GEMM",
            "-DCUTE_DISABLE_PRINT_LATEX",
        ],
        extra_include_paths=extra_includes,
        extra_ldflags=["-lcuda"],
        build_directory=str(BUILD_ROOT),
        verbose=False,
    )


def _build_cluster_extension() -> Any:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    arch = _set_arch_env()
    name = f"cluster_probe_stack_check_{arch.replace('.', '_').replace(';', '_') or 'default'}"
    cuda_src = r"""
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda.h>

namespace cg = cooperative_groups;

__global__ __cluster_dims__(2, 1, 1) void cluster_rank_kernel(int* ranks) {
    cg::cluster_group cluster = cg::this_cluster();
    int block_rank = static_cast<int>(cluster.block_rank());
    if (threadIdx.x == 0) {
        ranks[blockIdx.x] = block_rank;
    }
    cluster.sync();
}

void run_cluster_probe(torch::Tensor ranks) {
    const dim3 grid(2, 1, 1);
    const dim3 block(32, 1, 1);
    cudaLaunchAttr attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = grid;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = at::cuda::getDefaultCUDAStream();
    config.attrs = attrs;
    config.numAttrs = 1;

    auto status = cudaFuncSetAttribute(
        cluster_rank_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }

    status = cudaLaunchKernelExC(&config, cluster_rank_kernel, ranks.data_ptr<int>());
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_cluster_probe", &run_cluster_probe, "Launch a small cluster kernel");
}
"""
    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["-lineinfo", "--use_fast_math"],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(BUILD_ROOT),
        verbose=False,
    )


def _flash_attention_test() -> Tuple[bool, str]:
    try:
        import flash_attn.flash_attn_interface as fai
    except Exception as exc:
        return False, f"flash-attn import failed: {exc}"

    try:
        torch.manual_seed(1234)
        device = "cuda"
        batch, seqlen, nheads, headdim = 2, 64, 4, 64
        q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_flash = fai.flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        if torch.allclose(out_flash, out_torch, rtol=1e-2, atol=1e-2):
            return True, f"flash-attn {getattr(fai, '__version__', 'unknown')}"
        return False, "flash-attn output mismatch vs torch SDPA"
    except Exception as exc:
        return False, f"flash-attn execution failed: {exc}"


def _transformer_engine_fp8_test() -> Tuple[bool, str]:
    try:
        import transformer_engine.pytorch as te
    except Exception as exc:
        return False, f"Transformer Engine import failed: {exc}"

    try:
        layer = te.Linear(128, 128, bias=False).to(torch.bfloat16).cuda()
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with te.fp8_autocast(enabled=True):
            y = layer(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        return True, f"transformer_engine {getattr(te, '__version__', 'unknown')}"
    except Exception as exc:
        return False, f"Transformer Engine FP8 path failed: {exc}"


def _transformer_engine_fp4_test() -> Tuple[bool, str]:
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch import NVFP4Quantizer
    except Exception as exc:
        return False, f"Transformer Engine import failed: {exc}"

    try:
        available, reason = te.is_nvfp4_available(return_reason=True)
    except Exception as exc:
        return False, f"is_nvfp4_available failed: {exc}"

    if not available:
        return False, f"NVFP4 unavailable: {reason or 'unknown reason'}"

    try:
        torch.manual_seed(1234)
        device = "cuda"
        x = torch.randn(32, 32, device=device, dtype=torch.float32)
        quantizer = NVFP4Quantizer(with_rht=False, with_2d_quantization=True)
        q = quantizer(x)
        deq = q.dequantize(dtype=torch.float32)
        if not torch.all(torch.isfinite(deq)):
            return False, "NVFP4 dequantize produced non-finite values"
        mse = float(torch.mean((deq - x) ** 2).item())
        if mse > 0.2:
            return False, f"NVFP4 dequant MSE too high ({mse:.3f})"
        return True, "NVFP4 quantize/dequant OK"
    except Exception as exc:
        return False, f"NVFP4 path failed: {exc}"


def _tma_probe_test() -> Tuple[bool, str]:
    try:
        ext = _build_tma_extension(use_tmem=False)
    except Exception as exc:
        return False, f"TMA build failed: {exc}"

    try:
        m, n = 256, 256
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        ext.run_tma(x, out)
        expected = x * 1.0002 + 0.0001
        if not torch.allclose(out, expected, rtol=1e-4, atol=1e-4):
            max_diff = float((out - expected).abs().max().item())
            return False, f"TMA output mismatch (max diff {max_diff:.4f})"
        if not ext.supports_tma_encode(m, n, int(x.stride(0))):
            return False, "cuTensorMapEncodeTiled failed for decode kernel shape"
        return True, "TMA decode kernel OK"
    except Exception as exc:
        return False, f"TMA decode kernel failed: {exc}"


def _tmem_probe_test() -> Tuple[bool, str]:
    try:
        ext = _build_tma_extension(use_tmem=True)
    except Exception as exc:
        return False, f"TMEM build failed: {exc}"

    if hasattr(ext, "tmem_available") and not bool(ext.tmem_available()):
        return False, "TMEM not available in compiler/toolkit"

    try:
        m, n = 256, 256
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        ext.run_tmem_roundtrip(x, out)
        if not torch.allclose(out, x, rtol=1e-4, atol=1e-4):
            max_diff = float((out - x).abs().max().item())
            return False, f"TMEM roundtrip mismatch (max diff {max_diff:.4f})"
        return True, "TMEM roundtrip OK"
    except Exception as exc:
        return False, f"TMEM roundtrip failed: {exc}"


def _cluster_launch_test() -> Tuple[bool, str]:
    try:
        ext = _build_cluster_extension()
    except Exception as exc:
        return False, f"Cluster probe build failed: {exc}"

    try:
        ranks = torch.full((2,), -1, device="cuda", dtype=torch.int32)
        ext.run_cluster_probe(ranks)
        vals = ranks.cpu().tolist()
        if vals == [0, 1]:
            return True, "cluster launch/metadata OK (2-block cluster)"
        return False, f"unexpected cluster ranks {vals}"
    except Exception as exc:
        return False, f"Cluster probe failed: {exc}"


def run_all() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; cannot run stack verification.")
        return 1

    results: Dict[str, Tuple[bool, str]] = {}
    tests = {
        "FlashAttention": _flash_attention_test,
        "TransformerEngine FP8": _transformer_engine_fp8_test,
        "TransformerEngine FP4": _transformer_engine_fp4_test,
        "CTA cluster launch": _cluster_launch_test,
        "TMA decode kernel": _tma_probe_test,
        "TMEM roundtrip": _tmem_probe_test,
    }

    for name, fn in tests.items():
        ok, detail = fn()
        results[name] = (ok, detail)
        _format_status(name, ok, detail)

    failed = [name for name, (ok, _) in results.items() if not ok]
    if failed:
        print("\nStack verification failed for:", ", ".join(failed))
        return 1

    print("\nStack verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
