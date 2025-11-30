// baseline_hbm_peak.cu -- Baseline copy kernel.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Baseline copy kernel
__global__ void baseline_copy(const float* __restrict__ src, 
                               float* __restrict__ dst, 
                               size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    const size_t target_bytes = 512ULL * 1024 * 1024;  // 512 MB
    const size_t n = target_bytes / sizeof(float);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_src, 1, n * sizeof(float)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        baseline_copy<<<8, 64>>>(d_src, d_dst, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double bytes_transferred = 2.0 * n * sizeof(float) * iterations;
    double bandwidth_gbs = (bytes_transferred / elapsed_ms) / 1e6;
    double bandwidth_tbs = bandwidth_gbs / 1024.0;
    
    printf("Baseline copy: %.2f ms, %.2f TB/s\n", elapsed_ms / iterations, bandwidth_tbs);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}





