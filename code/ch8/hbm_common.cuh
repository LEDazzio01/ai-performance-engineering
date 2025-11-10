#pragma once

#include <cuda_runtime.h>

namespace ch8 {

constexpr int kHbmBlockDim = 256;
constexpr int kVectorWidth = 4;

__device__ __forceinline__ float hbm_mix(float value) {
    const float sine = __sinf(value);
    const float cosine = __cosf(value);
    return value * 1.00005f + sine * 0.00095f + cosine * 0.00073f;
}

__global__ void hbm_naive_kernel(
    const float* __restrict__ col_major,
    float* __restrict__ output,
    int rows,
    int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    // Intentional column-major stride to simulate poor HBM utilization.
    for (int col = 0; col < cols; ++col) {
        const int idx = col * rows + row;
        const float value = col_major[idx];
        sum += hbm_mix(value);
    }
    output[row] = sum;
}

__global__ void hbm_vectorized_kernel(
    const float* __restrict__ row_major,
    float* __restrict__ output,
    int rows,
    int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = row_major + static_cast<size_t>(row) * cols;
    const int vec_cols = cols / kVectorWidth;
    const float4* vec_ptr = reinterpret_cast<const float4*>(row_ptr);
    float sum = 0.0f;

#pragma unroll 4
    for (int vec = 0; vec < vec_cols; ++vec) {
        const float4 values = vec_ptr[vec];
        sum += hbm_mix(values.x);
        sum += hbm_mix(values.y);
        sum += hbm_mix(values.z);
        sum += hbm_mix(values.w);
    }

    output[row] = sum;
}

inline dim3 hbm_launch_grid(int rows) {
    return dim3((rows + kHbmBlockDim - 1) / kHbmBlockDim);
}

inline void launch_hbm_naive(
    const float* col_major,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream) {
    hbm_naive_kernel<<<hbm_launch_grid(rows), kHbmBlockDim, 0, stream>>>(
        col_major,
        output,
        rows,
        cols);
}

inline void launch_hbm_vectorized(
    const float* row_major,
    float* output,
    int rows,
    int cols,
    cudaStream_t stream) {
    hbm_vectorized_kernel<<<hbm_launch_grid(rows), kHbmBlockDim, 0, stream>>>(
        row_major,
        output,
        rows,
        cols);
}

}  // namespace ch8
