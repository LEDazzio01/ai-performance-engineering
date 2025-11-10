#pragma once

#include <cuda_runtime.h>

namespace ch8 {

constexpr int kElementsPerRow = 512;
constexpr int kWeightPeriod = 8;
constexpr int kThreadsPerBlock = 256;
constexpr int kRedundantAccums = 16;

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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }

    const int weight_mask = kWeightPeriod - 1;
    const float* row_ptr = inputs + idx * kElementsPerRow;
    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;

    const auto* row_vec = reinterpret_cast<const float4*>(row_ptr);
#pragma unroll
    for (int k = 0; k < kElementsPerRow; k += 4) {
        const float4 values = row_vec[k / 4];
        partial0 = fmaf(values.x, weights[(k + 0) & weight_mask], partial0);
        partial1 = fmaf(values.y, weights[(k + 1) & weight_mask], partial1);
        partial2 = fmaf(values.z, weights[(k + 2) & weight_mask], partial2);
        partial3 = fmaf(values.w, weights[(k + 3) & weight_mask], partial3);
    }

    const float sum = (partial0 + partial1) + (partial2 + partial3);
    output[idx] = sum;
}

inline dim3 loop_unrolling_grid(int rows) {
    return dim3((rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
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
    loop_unrolling_optimized_kernel<<<loop_unrolling_grid(rows), kThreadsPerBlock, 0, stream>>>(
        inputs,
        weights,
        output,
        rows);
}

}  // namespace ch8
