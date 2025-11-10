#pragma once

#include <cuda_runtime.h>

namespace ch8 {

constexpr int kDoubleBufferBlock = 256;
constexpr int kDoubleBufferTile = 4;
constexpr int kDoubleBufferInnerLoops = 16;

__device__ __forceinline__ float pipeline_transform(float value) {
    return value * 1.0002f + value * value * 0.00001f;
}

__global__ void double_buffer_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int elements) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int block_span = blockDim.x * kDoubleBufferTile;
    int tile_base = blockIdx.x * block_span;

    for (int tile = 0; tile < kDoubleBufferTile; ++tile) {
        const int idx = tile_base + tile * blockDim.x + tid;
        if (idx < elements) {
            smem[tid] = input[idx];
        } else {
            smem[tid] = 0.0f;
        }
        __syncthreads();

        float value = smem[tid];
#pragma unroll
        for (int loop = 0; loop < kDoubleBufferInnerLoops; ++loop) {
            value = pipeline_transform(value);
        }

        if (idx < elements) {
            output[idx] = value;
        }
        __syncthreads();
    }
}

__global__ void double_buffer_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int elements) {
    extern __shared__ float smem[];
    float* buffer0 = smem;
    float* buffer1 = smem + blockDim.x;
    const int tid = threadIdx.x;
    const int block_span = blockDim.x * kDoubleBufferTile;
    int tile_base = blockIdx.x * block_span;

    int current_tile = 0;
    int current_idx = tile_base + tid;
    if (current_idx < elements) {
        buffer0[tid] = input[current_idx];
    } else {
        buffer0[tid] = 0.0f;
    }
    __syncthreads();

    float* current = buffer0;
    float* next = buffer1;

    for (; current_tile < kDoubleBufferTile; ++current_tile) {
        const int next_idx = tile_base + (current_tile + 1) * blockDim.x + tid;
        if (current_tile + 1 < kDoubleBufferTile) {
            if (next_idx < elements) {
                next[tid] = input[next_idx];
            } else {
                next[tid] = 0.0f;
            }
        }

        float value = current[tid];
#pragma unroll
        for (int loop = 0; loop < kDoubleBufferInnerLoops; ++loop) {
            value = pipeline_transform(value);
        }

        const int write_idx = tile_base + current_tile * blockDim.x + tid;
        if (write_idx < elements) {
            output[write_idx] = value;
        }
        __syncthreads();

        float* tmp = current;
        current = next;
        next = tmp;
    }
}

inline dim3 double_buffer_grid(int elements) {
    const int block_span = kDoubleBufferBlock * kDoubleBufferTile;
    return dim3((elements + block_span - 1) / block_span);
}

inline void launch_double_buffer_baseline(
    const float* input,
    float* output,
    int elements,
    cudaStream_t stream) {
    const size_t shared_bytes = kDoubleBufferBlock * sizeof(float);
    double_buffer_baseline_kernel<<<double_buffer_grid(elements), kDoubleBufferBlock, shared_bytes, stream>>>(
        input,
        output,
        elements);
}

inline void launch_double_buffer_optimized(
    const float* input,
    float* output,
    int elements,
    cudaStream_t stream) {
    const size_t shared_bytes = kDoubleBufferBlock * 2 * sizeof(float);
    double_buffer_optimized_kernel<<<double_buffer_grid(elements), kDoubleBufferBlock, shared_bytes, stream>>>(
        input,
        output,
        elements);
}

}  // namespace ch8
