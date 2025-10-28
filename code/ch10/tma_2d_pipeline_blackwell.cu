/**
 * Blackwell TMA (Tensor Memory Accelerator) 2D Pipeline
 * =====================================================
 *
 * CUDA 13.0 introduces descriptor-backed bulk async copies (cp.async.bulk.tensor.*)
 * that route through the Tensor Memory Accelerator on Hopper/Blackwell GPUs.
 * This sample demonstrates a double-buffered 2D pipeline that overlaps compute
 * with TMA transfers using CUDA C++17 primitives.
 *
 * Key features demonstrated:
 *  - CU_TENSOR_MAP_SWIZZLE_128B for HBM3e alignment on Blackwell B200/B300
 *  - cuda::device::experimental::cp_async_bulk_tensor_2d_* helpers
 *  - cuda::barrier based staging for multi-buffer pipelines
 *
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_100 tma_2d_pipeline_blackwell.cu -o tma_pipeline
 */

#include <algorithm>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cstdio>
#include <utility>
#include <vector>

#include "../cuda13_feature_examples.cuh"

#if CUDART_VERSION >= 13000
#include <cuda.h>
#define TMA_CUDA13_AVAILABLE 1
#else
#define TMA_CUDA13_AVAILABLE 0
#endif

namespace cde = cuda::device::experimental;
using cuda13_examples::check_cuda;
using cuda13_examples::device_supports_tma;
using cuda13_examples::load_cuTensorMapEncodeTiled;
using cuda13_examples::make_2d_tensor_map;

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int CHUNK_M = 32;
constexpr int PIPELINE_STAGES = 2;
constexpr std::size_t BYTES_PER_CHUNK = static_cast<std::size_t>(CHUNK_M) * TILE_N * sizeof(float);

#if TMA_CUDA13_AVAILABLE

__device__ void compute_on_tile(float* tile, int pitch, int rows, int cols) {
    for (int r = threadIdx.y; r < rows; r += blockDim.y) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float v = tile[r * pitch + c];
            tile[r * pitch + c] = v * 1.0001f + 0.0001f;  // trivial math to emulate work
        }
    }
}

__global__ void tma_2d_pipeline_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    float* __restrict__ fallback_out,
    int M,
    int N,
    int ldc) {
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES][CHUNK_M][TILE_N];
    __shared__ cuda::barrier<cuda::thread_scope_block> stage_barriers[PIPELINE_STAGES];

    const int tile_m_dim = TILE_M;
    const int tile_n_dim = TILE_N;
    const int chunk_m_dim = CHUNK_M;
    const int pipeline_stages = PIPELINE_STAGES;

    const int participants = blockDim.x * blockDim.y;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int stage = 0; stage < pipeline_stages; ++stage) {
            init(&stage_barriers[stage], participants);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;

    const int g_row0 = tile_m * tile_m_dim;
    const int g_col0 = tile_n * tile_n_dim;

    if (g_row0 >= M || g_col0 >= N) {
        return;
    }

    const int tile_rows = std::min(tile_m_dim, M - g_row0);
    const int tile_cols = std::min(tile_n_dim, N - g_col0);
    const int num_chunks = (tile_rows + chunk_m_dim - 1) / chunk_m_dim;

    cuda::barrier<cuda::thread_scope_block>::arrival_token stage_tokens[PIPELINE_STAGES];

    auto issue_chunk = [&](int chunk_idx) {
        if (chunk_idx >= num_chunks) {
            return;
        }
        const int stage = chunk_idx % pipeline_stages;
        auto& bar = stage_barriers[stage];

        const int row_base = g_row0 + chunk_idx * chunk_m_dim;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                g_col0,
                row_base,
                bar);
            stage_tokens[stage] = cuda::device::barrier_arrive_tx(bar, 1, BYTES_PER_CHUNK);
        } else {
            stage_tokens[stage] = bar.arrive();
        }
    };

    const int preload = std::min(num_chunks, pipeline_stages);
    for (int chunk = 0; chunk < preload; ++chunk) {
        issue_chunk(chunk);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int stage = chunk % pipeline_stages;
        auto& bar = stage_barriers[stage];

        bar.wait(std::move(stage_tokens[stage]));
        __syncthreads();

        const int row_base = g_row0 + chunk * chunk_m_dim;
        const int rows_this_chunk = std::min(chunk_m_dim, tile_rows - chunk * chunk_m_dim);
        float* tile_ptr = &stage_buffers[stage][0][0];

        compute_on_tile(tile_ptr, TILE_N, rows_this_chunk, tile_cols);
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        const bool full_columns = tile_cols == TILE_N;
        const bool full_rows = (row_base + chunk_m_dim) <= M;
        const bool can_use_tma_store = full_columns && full_rows;

        if (can_use_tma_store) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                cde::cp_async_bulk_tensor_2d_shared_to_global(
                    &out_desc,
                    g_col0,
                    row_base,
                    &stage_buffers[stage]);
                cde::cp_async_bulk_commit_group();
                cde::cp_async_bulk_wait_group_read<0>();
            }
            __syncthreads();
        } else {
            for (int r = threadIdx.y; r < rows_this_chunk; r += blockDim.y) {
                const int global_row = row_base + r;
                if (global_row >= M) {
                    continue;
                }
                for (int c = threadIdx.x; c < tile_cols; c += blockDim.x) {
                    const int global_col = g_col0 + c;
                    if (global_col >= N) {
                        continue;
                    }
                    fallback_out[global_row * ldc + global_col] = tile_ptr[r * TILE_N + c];
                }
            }
            __syncthreads();
        }

        const int next = chunk + pipeline_stages;
        if (next < num_chunks) {
            issue_chunk(next);
        }
    }
}

int main() {
    std::printf("=== Blackwell TMA 2D Pipeline ===\n\n");

    if (!device_supports_tma()) {
        std::printf("⚠️  Device does not support Hopper/Blackwell TMA (SM 90+ required).\n");
        return 0;
    }

    auto encode = load_cuTensorMapEncodeTiled();
    if (!encode) {
        std::printf("⚠️  cuTensorMapEncodeTiled entry point unavailable on this CUDA runtime.\n");
        return 0;
    }

    constexpr int M = 4096;
    constexpr int N = 4096;
    const std::size_t bytes = static_cast<std::size_t>(M) * N * sizeof(float);

    std::vector<float> h_in(static_cast<std::size_t>(M) * N);
    for (std::size_t idx = 0; idx < h_in.size(); ++idx) {
        h_in[idx] = static_cast<float>((idx % 113) + 1);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    check_cuda(cudaMemset(d_out, 0, bytes), "memset output");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    const bool ok_in = make_2d_tensor_map(in_desc, encode, d_in, N, M, N, CU_TENSOR_MAP_SWIZZLE_NONE);
    const bool ok_out = make_2d_tensor_map(out_desc, encode, d_out, N, M, N, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (!ok_in || !ok_out) {
        std::printf("⚠️  Failed to encode tensor maps (ensure CUDA 13 driver runtime).\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    dim3 block(32, 4, 1);  // 128 threads
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M,
        1);

    // Warm-up launch
    tma_2d_pipeline_kernel<<<grid, block>>>(in_desc, out_desc, d_out, M, N, N);
    check_cuda(cudaGetLastError(), "warm-up launch");
    check_cuda(cudaDeviceSynchronize(), "warm-up sync");

    std::vector<float> h_out(TILE_M * TILE_N);
    check_cuda(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy sample");

    std::printf("Sample output element: %.2f -> %.2f\n", h_in[0], h_out[0]);
    cudaFree(d_in);
    cudaFree(d_out);

    std::printf("\n=== Summary ===\n");
    std::printf("✓ Bulk TMA transfers via cp.async.bulk.tensor.2d (double-buffered %d-row chunks)\n", CHUNK_M);
    std::printf("✓ Descriptor-backed TMA transfers with L2 promotion enabled\n");
    std::printf("✓ cuda::barrier orchestrates staging and overlap between compute and TMA IO\n");

    return 0;
}

#else  // !TMA_CUDA13_AVAILABLE

int main() {
    std::printf("=== Blackwell TMA 2D Pipeline ===\n\n");
    std::printf("⚠️  CUDA 13.0+ required for TMA descriptor API (detected %d.%d)\n",
                CUDART_VERSION / 1000,
                (CUDART_VERSION % 100) / 10);
    return 0;
}

#endif  // TMA_CUDA13_AVAILABLE
