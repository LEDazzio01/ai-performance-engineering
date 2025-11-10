// optimized_tma_copy.cu -- cp.async/TMA-style tiled neighbor gather (optimized).

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace cg = cooperative_groups;

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kValuesPerThread = 8;
constexpr int kTileElems = kThreadsPerBlock * kValuesPerThread;  // 2048 elements
constexpr int kLookahead = 64;
constexpr int kStages = 2;
constexpr int kStageSpan = kTileElems + kLookahead;
constexpr int kElements = 1 << 25;
constexpr bool kValidateOutput = false;

__host__ __device__ __forceinline__ float combine_values(float center, float near_val, float far_val) {
    return fmaf(far_val, 0.125f, fmaf(near_val, 0.25f, center * 0.75f));
}

__global__ void tma_neighbor_copy_kernel(const float* __restrict__ src,
                                         float* __restrict__ dst,
                                         int n,
                                         int total_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
    const int first_tile = blockIdx.x * tiles_per_block;
    const int tiles_to_process = min(tiles_per_block, max(total_tiles - first_tile, 0));
    if (tiles_to_process <= 0) {
        return;
    }

    extern __shared__ float shared[];
    float* stage_buffers[kStages];
    for (int stage = 0; stage < kStages; ++stage) {
        stage_buffers[stage] = shared + stage * kStageSpan;
    }

    cg::thread_block block = cg::this_thread_block();
    __shared__ alignas(cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>)
        unsigned char pipeline_storage[sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>)];
    auto* pipeline_state =
        reinterpret_cast<cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>*>(pipeline_storage);
    if (threadIdx.x == 0) {
        new (pipeline_state) cuda::pipeline_shared_state<cuda::thread_scope_block, kStages>();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, pipeline_state);

    auto enqueue_tile = [&](int stage, int tile_idx) -> bool {
        if (tile_idx >= first_tile + tiles_to_process) {
            return false;
        }
        const int global_offset = tile_idx * kTileElems;
        if (global_offset >= n) {
            return false;
        }
        const int remaining = n - global_offset;
        const int stage_elems = remaining > kStageSpan ? kStageSpan : remaining;
        pipe.producer_acquire();
        cuda::memcpy_async(
            block,
            stage_buffers[stage],
            src + global_offset,
            static_cast<size_t>(stage_elems) * sizeof(float),
            pipe);
        pipe.producer_commit();
        return true;
    };

    int stage_tile[kStages];
    bool stage_ready[kStages] = {false, false};
    int next_tile = first_tile;
    for (int stage = 0; stage < kStages; ++stage) {
        stage_tile[stage] = next_tile;
        stage_ready[stage] = enqueue_tile(stage, next_tile);
        if (stage_ready[stage]) {
            ++next_tile;
        }
    }

    int tiles_processed = 0;
    int current_stage = 0;
    while (tiles_processed < tiles_to_process) {
        if (!stage_ready[current_stage]) {
            current_stage = (current_stage + 1) % kStages;
            continue;
        }

        pipe.consumer_wait();
        block.sync();

        const int tile_idx = stage_tile[current_stage];
        const int global_offset = tile_idx * kTileElems;
        const int stage_valid = min(kStageSpan, n - global_offset);
        if (stage_valid > 0) {
            const int max_elem = min(kTileElems, n - global_offset);
            float* tile_ptr = stage_buffers[current_stage];
            const int stage_limit = stage_valid - 1;

            for (int base = threadIdx.x * kValuesPerThread;
                 base < max_elem;
                 base += blockDim.x * kValuesPerThread) {
#pragma unroll
                for (int i = 0; i < kValuesPerThread; ++i) {
                    const int local_idx = base + i;
                    if (local_idx >= max_elem) {
                        break;
                    }
                    const int global_idx = global_offset + local_idx;
                    if (global_idx >= n) {
                        continue;
                    }
                    const float center = tile_ptr[local_idx];
                    const int near_local = (local_idx + 1 <= stage_limit) ? (local_idx + 1) : stage_limit;
                    int far_local = local_idx + kLookahead;
                    if (far_local > stage_limit) {
                        far_local = stage_limit;
                    }
                    const float near_val = tile_ptr[near_local];
                    const float far_val = tile_ptr[far_local];
                    dst[global_idx] = combine_values(center, near_val, far_val);
                }
            }
        }

        pipe.consumer_release();
        stage_ready[current_stage] = false;
        ++tiles_processed;

        if (next_tile < first_tile + tiles_to_process) {
            stage_tile[current_stage] = next_tile;
            stage_ready[current_stage] = enqueue_tile(current_stage, next_tile);
            if (stage_ready[current_stage]) {
                ++next_tile;
            }
        }

        current_stage = (current_stage + 1) % kStages;
    }
#else
    (void)src;
    (void)dst;
    (void)n;
    (void)total_tiles;
#endif
}

float checksum(const std::vector<float>& data) {
    double sum = 0.0;
    for (float v : data) {
        sum += static_cast<double>(v);
    }
    return static_cast<float>(sum / static_cast<double>(data.size()));
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) {
        std::fprintf(stderr, "SKIPPED: optimized_tma_copy requires SM 90+\n");
        return 3;
    }

    const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);

    float *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    std::vector<float> h_input(kElements);
    for (int i = 0; i < kElements; ++i) {
        h_input[i] = static_cast<float>((i % 1024) - 512) / 128.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_input.data(), bytes, cudaMemcpyHostToDevice));

    const int total_tiles = (kElements + kTileElems - 1) / kTileElems;
    const int max_blocks = 2 * prop.multiProcessorCount;
    const int grid = std::min(total_tiles, max_blocks);
    const size_t shared_bytes = static_cast<size_t>(kStages) * kStageSpan * sizeof(float);

    tma_neighbor_copy_kernel<<<grid, kThreadsPerBlock, shared_bytes>>>(
        d_src, d_dst, kElements, total_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int kIterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < kIterations; ++iter) {
        tma_neighbor_copy_kernel<<<grid, kThreadsPerBlock, shared_bytes>>>(
            d_src, d_dst, kElements, total_tiles);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / kIterations;
    std::printf("TMA-style neighbor copy (optimized): %.3f ms\n", avg_ms);

    std::vector<float> h_output(kElements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_dst, bytes, cudaMemcpyDeviceToHost));

    if (kValidateOutput) {
        std::vector<float> h_reference(kElements);
        for (int i = 0; i < kElements; ++i) {
            const int near_idx = (i + 1 < kElements) ? (i + 1) : (kElements - 1);
            const int far_idx = (i + kLookahead < kElements) ? (i + kLookahead) : (kElements - 1);
            h_reference[i] = combine_values(h_input[i], h_input[near_idx], h_input[far_idx]);
        }

        float max_error = 0.0f;
        for (int i = 0; i < kElements; ++i) {
            max_error = std::max(max_error, std::abs(h_reference[i] - h_output[i]));
        }
        std::printf("Output checksum: %.6f (max error %.6f)\n", checksum(h_output), max_error);
    } else {
        std::printf("Output checksum: %.6f\n", checksum(h_output));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}
