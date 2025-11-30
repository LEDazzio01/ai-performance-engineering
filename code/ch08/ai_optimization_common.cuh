#pragma once

#include <cuda_runtime.h>

namespace ch08 {

constexpr int kAiThreads = 256;

__device__ __forceinline__ float nonlinear_activation(float x) {
    return tanhf(x);
}

__global__ void ai_baseline_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int rows,
    int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = inputs + static_cast<size_t>(row) * cols;
    float acc = 0.0f;
    for (int col = 0; col < cols; ++col) {
        acc = fmaf(row_ptr[col], weights[col], acc);
    }
    output[row] = nonlinear_activation(acc);
}

__global__ void ai_optimized_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int rows,
    int cols) {
    extern __shared__ float shared_weights[];
    for (int idx = threadIdx.x; idx < cols; idx += blockDim.x) {
        shared_weights[idx] = weights[idx];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = inputs + static_cast<size_t>(row) * cols;
    const int vec_cols = cols / 4;
    const float4* row_vec = reinterpret_cast<const float4*>(row_ptr);
    const float4* weight_vec = reinterpret_cast<const float4*>(shared_weights);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma unroll 4
    for (int v = 0; v < vec_cols; ++v) {
        const float4 values = row_vec[v];
        const float4 w = weight_vec[v];
        acc0 = fmaf(values.x, w.x, acc0);
        acc1 = fmaf(values.y, w.y, acc1);
        acc2 = fmaf(values.z, w.z, acc2);
        acc3 = fmaf(values.w, w.w, acc3);
    }

    float acc = acc0 + acc1 + acc2 + acc3;
    const int remaining = cols & 3;
    if (remaining) {
        const int offset = vec_cols * 4;
        for (int col = 0; col < remaining; ++col) {
            acc = fmaf(row_ptr[offset + col], shared_weights[offset + col], acc);
        }
    }

    output[row] = nonlinear_activation(acc);
}

inline dim3 ai_launch_grid(int rows) {
    return dim3((rows + kAiThreads - 1) / kAiThreads);
}

inline void launch_ai_baseline(
    const float* inputs,
    const float* weights,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream) {
    ai_baseline_kernel<<<ai_launch_grid(rows), kAiThreads, 0, stream>>>(
        inputs,
        weights,
        output,
        rows,
        cols);
}

inline void launch_ai_optimized(
    const float* inputs,
    const float* weights,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream) {
    const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
    ai_optimized_kernel<<<ai_launch_grid(rows), kAiThreads, shared_bytes, stream>>>(
        inputs,
        weights,
        output,
        rows,
        cols);
}

}  // namespace ch08

