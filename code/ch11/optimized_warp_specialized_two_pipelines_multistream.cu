#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)

#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace ch11 {

namespace cg = cooperative_groups;

constexpr int kWarpSize = 32;
constexpr int kPipelineStages = 2;
constexpr int kTileElems = 1024;
constexpr bool kHasCuda13Pipeline = true;

__device__ void compute_stage(const float* a, const float* b, float* c, int warp_id, int lane) {
    // Compute warps are 1-4 (4 warps = 128 threads)
    // Map thread to element: warp_local_id = warp_id - 1 (0-3), lane = 0-31
    // Thread index within compute group: (warp_id - 1) * 32 + lane
    int compute_thread_idx = (warp_id - 1) * kWarpSize + lane;
    int compute_threads = 4 * kWarpSize;  // 128 threads in compute warps
    
    for (int idx = compute_thread_idx; idx < kTileElems; idx += compute_threads) {
        float x = a[idx];
        float y = b[idx];
        float acc = x + y;
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            acc = fmaf(acc, 1.0f, 0.0f);
        }
        c[idx] = acc;
    }
}

extern "C" __global__
void warp_specialized_kernel_two_pipelines_multistream(const float* __restrict__ in_a,
                                                       const float* __restrict__ in_b,
                                                       float* __restrict__ out_c,
                                                       int tiles_per_grid) {
    cg::thread_block block = cg::this_thread_block();

    extern __shared__ float smem[];
    float* stage_a = smem;
    float* stage_b = smem + kPipelineStages * kTileElems;
    float* stage_c = smem + 2 * kPipelineStages * kTileElems;

    using pipe_state =
        cuda::pipeline_shared_state<cuda::thread_scope_block, kPipelineStages>;

    __shared__ alignas(pipe_state) unsigned char state_storage[sizeof(pipe_state)];
    auto* storage_ptr = reinterpret_cast<pipe_state*>(state_storage);
    auto pipe = cuda::make_pipeline(block, storage_ptr);

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int stride = gridDim.x;

    // Prime the pipeline.
    for (int stage = 0; stage < kPipelineStages; ++stage) {
        int tile = blockIdx.x + stage * stride;
        if (tile >= tiles_per_grid) {
            break;
        }
        float* stage_a_tile = stage_a + stage * kTileElems;
        float* stage_b_tile = stage_b + stage * kTileElems;
        size_t offset = static_cast<size_t>(tile) * kTileElems;

        pipe.producer_acquire();
        for (int idx = threadIdx.x; idx < kTileElems; idx += blockDim.x) {
            stage_a_tile[idx] = in_a[offset + idx];
            stage_b_tile[idx] = in_b[offset + idx];
        }
        pipe.producer_commit();
    }

    block.sync();

    int iteration = 0;
    for (int tile = blockIdx.x; tile < tiles_per_grid; tile += stride, ++iteration) {
        int stage = iteration % kPipelineStages;
        float* a_ptr = stage_a + stage * kTileElems;
        float* b_ptr = stage_b + stage * kTileElems;
        float* c_ptr = stage_c + stage * kTileElems;
        size_t offset = static_cast<size_t>(tile) * kTileElems;

        pipe.consumer_wait();
        block.sync();

        if (warp_id >= 1 && warp_id <= 4) {
            compute_stage(a_ptr, b_ptr, c_ptr, warp_id, lane);
        }

        block.sync();

        if (warp_id == 5) {
            for (int idx = lane; idx < kTileElems; idx += kWarpSize) {
                out_c[offset + idx] = c_ptr[idx];
            }
        }

        block.sync();
        pipe.consumer_release();
        block.sync();

        int next_tile = tile + kPipelineStages * stride;
        if (next_tile < tiles_per_grid) {
            int next_stage = (iteration + kPipelineStages) % kPipelineStages;
            float* next_a = stage_a + next_stage * kTileElems;
            float* next_b = stage_b + next_stage * kTileElems;
            size_t next_offset = static_cast<size_t>(next_tile) * kTileElems;

            pipe.producer_acquire();
            for (int idx = threadIdx.x; idx < kTileElems; idx += blockDim.x) {
                next_a[idx] = in_a[next_offset + idx];
                next_b[idx] = in_b[next_offset + idx];
            }
            pipe.producer_commit();
        }

        block.sync();
    }
}

}  // namespace ch11

#else  // CUDA < 13

namespace ch11 {

constexpr int kWarpSize = 32;
constexpr int kPipelineStages = 1;
constexpr int kTileElems = 256;
constexpr bool kHasCuda13Pipeline = false;

extern "C" __global__
void warp_specialized_kernel_two_pipelines_multistream(const float*,
                                                       const float*,
                                                       float*,
                                                       int) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("warp_specialized_kernel_two_pipelines_multistream requires CUDA 13+ asynchronous "
               "pipeline APIs. Launch skipped.\n");
    }
}

}  // namespace ch11

#endif  // CUDA toolkit check
