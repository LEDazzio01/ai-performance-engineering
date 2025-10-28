// async_prefetch_tma.cu -- double-buffered 1D streaming with CUDA 13 TMA

#include <algorithm>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cstdio>

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
using cuda13_examples::make_1d_tensor_map;

constexpr int TILE_SIZE = 1024;
constexpr int PIPELINE_STAGES = 2;
constexpr std::size_t BYTES_PER_TILE = static_cast<std::size_t>(TILE_SIZE) * sizeof(float);

#if TMA_CUDA13_AVAILABLE

__global__ void async_prefetch_tma_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    int total_tiles) {
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES][TILE_SIZE];
    __shared__ cuda::barrier<cuda::thread_scope_block> stage_barriers[PIPELINE_STAGES];

    const int pipeline_stages = PIPELINE_STAGES;

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < pipeline_stages; ++stage) {
            init(&stage_barriers[stage], blockDim.x);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token tokens[PIPELINE_STAGES];

    auto issue_tile = [&](int tile_idx) {
        if (tile_idx >= total_tiles) {
            return;
        }
        const int stage = tile_idx % pipeline_stages;
        auto& bar = stage_barriers[stage];

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                tile_idx * TILE_SIZE,
                bar);
            tokens[stage] = cuda::device::barrier_arrive_tx(bar, 1, BYTES_PER_TILE);
        } else {
            tokens[stage] = bar.arrive();
        }
    };

    const int preload = std::min(total_tiles, pipeline_stages);
    for (int t = 0; t < preload; ++t) {
        issue_tile(t);
    }

    for (int tile = 0; tile < total_tiles; ++tile) {
        const int stage = tile % pipeline_stages;
        auto& bar = stage_barriers[stage];

        bar.wait(std::move(tokens[stage]));
        __syncthreads();

        float* tile_ptr = stage_buffers[stage];
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            tile_ptr[i] *= 2.0f;
        }
        cde::fence_proxy_async_shared_cta();
        __syncthreads();

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_shared_to_global(
                &out_desc,
                tile * TILE_SIZE,
                &stage_buffers[stage]);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int next_tile = tile + pipeline_stages;
        if (next_tile < total_tiles) {
            issue_tile(next_tile);
        }
    }

}

int main() {
    std::printf("=== CUDA 13 TMA 1D Prefetch ===\n\n");

    if (!device_supports_tma()) {
        std::printf("⚠️  Device does not support Hopper/Blackwell TMA (SM 90+ required).\n");
        return 0;
    }

    auto encode = load_cuTensorMapEncodeTiled();
    if (!encode) {
        std::printf("⚠️  cuTensorMapEncodeTiled entry point unavailable on this CUDA runtime.\n");
        return 0;
    }

    constexpr int tiles = 64;
    constexpr int total = tiles * TILE_SIZE;
    const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(float);

    float* h_in = nullptr;
    float* h_out = nullptr;
    check_cuda(cudaMallocHost(&h_in, bytes), "cudaMallocHost h_in");
    check_cuda(cudaMallocHost(&h_out, bytes), "cudaMallocHost h_out");
    for (int i = 0; i < total; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "copy input");

    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    const bool ok_in = make_1d_tensor_map(in_desc, encode, d_in, total, TILE_SIZE);
    const bool ok_out = make_1d_tensor_map(out_desc, encode, d_out, total, TILE_SIZE);
    if (!ok_in || !ok_out) {
        std::printf("⚠️  Failed to encode tensor maps; ensure CUDA 13 driver runtime.\n");
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
        return 1;
    }

    async_prefetch_tma_kernel<<<1, 256>>>(in_desc, out_desc, tiles);
    check_cuda(cudaGetLastError(), "async_prefetch_tma_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");

    check_cuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "copy output");
    std::printf("out[0]=%.1f (expected %.1f)\n", h_out[0], h_in[0] * 2.0f);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}

#else  // !TMA_CUDA13_AVAILABLE

int main() {
    std::printf("=== CUDA 13 TMA 1D Prefetch ===\n\n");
    std::printf("⚠️  CUDA 13.0+ required for TMA descriptor API (detected %d.%d)\n",
                CUDART_VERSION / 1000,
                (CUDART_VERSION % 100) / 10);
    return 0;
}

#endif  // TMA_CUDA13_AVAILABLE
