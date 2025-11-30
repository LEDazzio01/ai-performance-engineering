// optimized_hbm_peak.cu -- HBM peak bandwidth kernel for Blackwell.
// CUDA 13 + Blackwell: Uses Float8 (32-byte aligned) for 256-bit loads

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

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

// HBM peak bandwidth kernel - Blackwell B200/B300 with 256-bit loads
__global__ void hbm_peak_copy(const Float8* __restrict__ src,
                               Float8* __restrict__ dst,
                               size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t base = tid * 4; base < n; base += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            size_t idx = base + j;
            if (idx >= n) break;
            dst[idx] = src[idx];  // 256-bit load/store
        }
    }
}

int main() {
    const size_t target_bytes = 1024ULL * 1024 * 1024;  // 1 GB
    const size_t n_floats = target_bytes / sizeof(float);
    const size_t n_vec8 = n_floats / 8;
    
    Float8 *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, target_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, target_bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, target_bytes));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        hbm_peak_copy<<<2048, 512>>>(d_src, d_dst, n_vec8);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double bytes_transferred = 2.0 * target_bytes * iterations;
    double bandwidth_tbs = (bytes_transferred / elapsed_ms) / 1e9;
    
    printf("HBM peak (Float8): %.2f ms, %.2f TB/s\n", elapsed_ms / iterations, bandwidth_tbs);
    printf("Expected: 7-8 TB/s (B200), 8-9 TB/s (B300)\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}





