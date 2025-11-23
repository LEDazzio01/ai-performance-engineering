// optimized_async_prefetch.cu -- TMA-optimized async prefetch kernel (optimized).
// Note: This requires CUDA 13+ and Blackwell TMA support

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "../common/headers/tma_helpers.cuh"

#if CUDART_VERSION >= 13000
#include <cuda/barrier>
#include <cuda.h>
#define TMA_CUDA13_AVAILABLE 1
#else
#define TMA_CUDA13_AVAILABLE 0
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#if TMA_CUDA13_AVAILABLE
namespace cde = cuda::device::experimental;

constexpr int PIPELINE_STAGES = 2;
constexpr std::size_t BYTES_PER_TILE(int tile_elems) {
    return static_cast<std::size_t>(tile_elems) * sizeof(float);
}

template <int TILE_SIZE>
__global__ void async_prefetch_tma_kernel(
    const __grid_constant__ CUtensorMap in_desc,
    const __grid_constant__ CUtensorMap out_desc,
    int total_tiles) {
    __shared__ alignas(128) float stage_buffers[PIPELINE_STAGES][TILE_SIZE];
    using block_barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(block_barrier) unsigned char stage_barrier_storage[PIPELINE_STAGES][sizeof(block_barrier)];

    const int pipeline_stages = PIPELINE_STAGES;

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < pipeline_stages; ++stage) {
            auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
            init(bar_ptr, blockDim.x);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token tokens[PIPELINE_STAGES];

    const int base_tile = static_cast<int>(blockIdx.x);
    const int stride_tiles = static_cast<int>(gridDim.x);
    const int tiles_this_block = (total_tiles <= base_tile)
                                     ? 0
                                     : (total_tiles - base_tile + stride_tiles - 1) / stride_tiles;

    auto issue_tile = [&](int local_seq) {
        if (local_seq >= tiles_this_block) {
            return;
        }
        const int tile_idx = base_tile + local_seq * stride_tiles;
        if (tile_idx >= total_tiles) {
            return;
        }
        const int stage = local_seq % pipeline_stages;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_global_to_shared(
                &stage_buffers[stage],
                &in_desc,
                tile_idx * TILE_SIZE,
                bar);
            tokens[stage] = cuda::device::barrier_arrive_tx(
                bar,
                1,
                BYTES_PER_TILE(TILE_SIZE));
        } else {
            tokens[stage] = bar.arrive();
        }
    };

    const int preload = std::min(total_tiles, pipeline_stages);
    for (int t = 0; t < preload; ++t) {
        issue_tile(t);
    }

    for (int local_seq = 0; local_seq < tiles_this_block; ++local_seq) {
        const int stage = local_seq % pipeline_stages;
        auto* bar_ptr = reinterpret_cast<block_barrier*>(stage_barrier_storage[stage]);
        auto& bar = *bar_ptr;

        bar.wait(std::move(tokens[stage]));
        __syncthreads();

        const int global_tile = base_tile + local_seq * stride_tiles;
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_1d_shared_to_global(
                &out_desc,
                global_tile * TILE_SIZE,
                &stage_buffers[stage]);
            cde::cp_async_bulk_commit_group();
            cde::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        const int next_seq = local_seq + pipeline_stages;
        if (next_seq < tiles_this_block) {
            issue_tile(next_seq);
        }
    }
}
#endif

int main() {
#if !TMA_CUDA13_AVAILABLE
    printf("SKIPPED: TMA requires CUDA 13.0+.\n");
    return 3;
#else
    if (!cuda_tma::device_supports_tma()) {
        printf("SKIPPED: TMA hardware/runtime support not detected.\n");
        return 3;
    }

    constexpr int TILE_SIZE = 256;
    constexpr int ELEMENTS = TILE_SIZE * 1000;  // match baseline workload size
    constexpr int THREADS = 256;
    const auto limits = cuda_arch::get_tma_limits();
    if (TILE_SIZE > static_cast<int>(limits.max_1d_box_size)) {
        printf("SKIPPED: TILE_SIZE=%d exceeds 1D TMA box limit (%u)\n",
               TILE_SIZE,
               static_cast<unsigned int>(limits.max_1d_box_size));
        return 3;
    }

    // Host data
    std::vector<float> h_in(ELEMENTS);
    for (int i = 0; i < ELEMENTS; ++i) {
        h_in[i] = static_cast<float>(i % 1024) * 0.5f;
    }
    std::vector<float> h_out(ELEMENTS, 0.0f);

    // Device buffers
    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, ELEMENTS * sizeof(float)));

    // Encode tensor maps for 1D copies
    CUtensorMap in_desc{};
    CUtensorMap out_desc{};
    auto encode = cuda_tma::load_cuTensorMapEncodeTiled();
    if (!cuda_tma::make_1d_tensor_map(in_desc, encode, d_in, ELEMENTS, TILE_SIZE) ||
        !cuda_tma::make_1d_tensor_map(out_desc, encode, d_out, ELEMENTS, TILE_SIZE)) {
        printf("SKIPPED: cuTensorMapEncodeTiled unavailable on this runtime.\n");
        cudaFree(d_in);
        cudaFree(d_out);
        return 3;
    }

    const int total_tiles = (ELEMENTS + TILE_SIZE - 1) / TILE_SIZE;

    // Warmup + benchmark
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    const int max_blocks = std::max(1, prop.multiProcessorCount * 16);
    const int blocks_x = std::min(total_tiles, max_blocks);

    dim3 grid(blocks_x);
    dim3 block(THREADS);

    async_prefetch_tma_kernel<TILE_SIZE><<<grid, block>>>(in_desc, out_desc, total_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int kIters = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIters; ++i) {
        async_prefetch_tma_kernel<TILE_SIZE><<<grid, block>>>(in_desc, out_desc, total_tiles);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kIters);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < ELEMENTS; ++i) {
        const float expected = h_in[i];
        if (std::abs(h_out[i] - expected) > 1e-3f) {
            printf("Mismatch at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false;
            break;
        }
    }

    printf("TMA async prefetch: %.4f ms (avg over %d iters) [%s]\n",
           avg_ms, kIters, ok ? "OK" : "FAIL");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
#endif
}
