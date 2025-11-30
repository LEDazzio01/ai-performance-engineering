#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ch08 {

constexpr int kElementsPerRow = 512;
constexpr int kWeightPeriod = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int kRowsPerThread = 2;
constexpr int kVectorWidth = 4;
constexpr int kRedundantAccums = 16;
constexpr int kThreadsPerGroup = 32;
constexpr int kRowPairsPerIter = kThreadsPerBlock / kThreadsPerGroup;
static_assert(kThreadsPerBlock % kThreadsPerGroup == 0, "Block size must be a multiple of warp size");
static_assert(kWeightPeriod <= kThreadsPerGroup, "Weight period must fit within a warp");

__global__ void loop_unrolling_naive_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }

    const int weight_mask = kWeightPeriod - 1;
    const float* row_ptr = inputs + idx * kElementsPerRow;
    float sum = 0.0f;

#pragma unroll 1
    for (int k = 0; k < kElementsPerRow; ++k) {
        const float mul = row_ptr[k] * weights[k & weight_mask];
#pragma unroll
        for (int repeat = 0; repeat < kRedundantAccums; ++repeat) {
            sum += mul * (1.0f / static_cast<float>(kRedundantAccums));
        }
    }

    output[idx] = sum;
}

__global__ void loop_unrolling_optimized_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int rows) {
    const int weight_mask = kWeightPeriod - 1;
    extern __shared__ float smem[];
    float* tile_row0 = smem;
    float* tile_row1 = tile_row0 + kRowPairsPerIter * kElementsPerRow;

    const int group_id = threadIdx.x / kThreadsPerGroup;
    const int group_lane = threadIdx.x % kThreadsPerGroup;
    const unsigned weight_warp_mask = __activemask();

    float weight_cache[kWeightPeriod];
    float lane_weight = 0.0f;
    if (group_lane < kWeightPeriod) {
        lane_weight = weights[group_lane];
    }
#pragma unroll
    for (int w = 0; w < kWeightPeriod; ++w) {
        weight_cache[w] = __shfl_sync(weight_warp_mask, lane_weight, w, kThreadsPerGroup);
    }

    for (int block_row_base = blockIdx.x * kRowPairsPerIter * kRowsPerThread;
         block_row_base < rows;
         block_row_base += gridDim.x * kRowPairsPerIter * kRowsPerThread) {
        const int row0 = block_row_base + group_id * kRowsPerThread;
        if (row0 >= rows) {
            continue;
        }
        const int row1 = row0 + 1;
        const bool has_row1 = row1 < rows;

        const float* row_ptr0 = inputs + row0 * kElementsPerRow;
        const float* row_ptr1 = has_row1 ? row_ptr0 + kElementsPerRow : nullptr;

        float* group_tile0 = tile_row0 + group_id * kElementsPerRow;
        float* group_tile1 = tile_row1 + group_id * kElementsPerRow;

        for (int idx = group_lane; idx < kElementsPerRow; idx += kThreadsPerGroup) {
            group_tile0[idx] = row_ptr0[idx];
            if (has_row1) {
                group_tile1[idx] = row_ptr1[idx];
            }
        }
        unsigned warp_mask = __activemask();
        __syncwarp(warp_mask);

        float thread_accum0 = 0.0f;
        float thread_accum1 = 0.0f;
        for (int idx = group_lane; idx < kElementsPerRow; idx += kThreadsPerGroup) {
            const float weight = weight_cache[idx & weight_mask];
            const float mul0 = group_tile0[idx] * weight;
#pragma unroll
            for (int repeat = 0; repeat < kRedundantAccums; ++repeat) {
                thread_accum0 = fmaf(mul0, 1.0f / static_cast<float>(kRedundantAccums), thread_accum0);
            }
            if (has_row1) {
                const float mul1 = group_tile1[idx] * weight;
#pragma unroll
                for (int repeat = 0; repeat < kRedundantAccums; ++repeat) {
                    thread_accum1 = fmaf(mul1, 1.0f / static_cast<float>(kRedundantAccums), thread_accum1);
                }
            }
        }

        for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
            thread_accum0 += __shfl_down_sync(warp_mask, thread_accum0, offset, kThreadsPerGroup);
            if (has_row1) {
                thread_accum1 += __shfl_down_sync(warp_mask, thread_accum1, offset, kThreadsPerGroup);
            }
        }

        if (group_lane == 0) {
            float* row0_ptr = output + row0;
            if (has_row1) {
                float2 store_vals{thread_accum0, thread_accum1};
                constexpr uintptr_t vec_align = alignof(float2) - 1;
                if ((reinterpret_cast<uintptr_t>(row0_ptr) & vec_align) == 0) {
                    reinterpret_cast<float2*>(row0_ptr)[0] = store_vals;
                } else {
                    row0_ptr[0] = store_vals.x;
                    row0_ptr[1] = store_vals.y;
                }
            } else {
                row0_ptr[0] = thread_accum0;
            }
        }
        __syncwarp(warp_mask);
    }
}

inline dim3 loop_unrolling_grid(int rows) {
    const int row_pairs = (rows + kRowsPerThread - 1) / kRowsPerThread;
    int blocks = (row_pairs + kRowPairsPerIter - 1) / kRowPairsPerIter;
    blocks = max(blocks, 1);
    const int max_blocks = 4096;
    return dim3(min(blocks, max_blocks));
}

inline void launch_loop_unrolling_baseline(
    const float* inputs,
    const float* weights,
    float* output,
    int rows,
    cudaStream_t stream) {
    loop_unrolling_naive_kernel<<<loop_unrolling_grid(rows), kThreadsPerBlock, 0, stream>>>(
        inputs,
        weights,
        output,
        rows);
}

inline void launch_loop_unrolling_optimized(
    const float* inputs,
    const float* weights,
    float* output,
    int rows,
    cudaStream_t stream) {
    const size_t shared_bytes =
        sizeof(float) *
        (2 * kRowPairsPerIter * kElementsPerRow);
    loop_unrolling_optimized_kernel<<<loop_unrolling_grid(rows), kThreadsPerBlock, shared_bytes, stream>>>(
        inputs,
        weights,
        output,
        rows);
}

}  // namespace ch08
