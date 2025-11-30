// baseline_hbm_copy.cu -- Scalar copy kernel (baseline).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "../core/common/headers/cuda_helpers.cuh"

// Baseline: Scalar copy (very inefficient)
__global__ void scalar_copy_kernel(float* dst, const float* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = tid; i < n; i += stride) {
        dst[i] = src[i];  // 4-byte transactions
    }
}

int main() {
    const size_t size_bytes = 256 * 1024 * 1024;  // 256 MB
    const size_t n_floats = size_bytes / sizeof(float);
    
    float* d_src = nullptr;
    float* d_dst = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_src, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, size_bytes));
    
    // Initialize
    CUDA_CHECK(cudaMemset(d_src, 1, size_bytes));
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        scalar_copy_kernel<<<64, 64>>>(d_dst, d_src, n_floats);
        CUDA_CHECK_LAST_ERROR();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    double bw = (size_bytes * 2 / (avg_ms / 1000.0)) / 1e9;
    
    printf("Scalar copy (baseline): %.2f ms, %.2f GB/s\n", avg_ms, bw);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}





