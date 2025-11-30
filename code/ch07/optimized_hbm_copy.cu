// optimized_hbm_copy.cu -- 256-byte bursts for Blackwell HBM.
// CUDA 13 + Blackwell: Uses Float8 (32-byte aligned) for 256-bit loads

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "../core/common/headers/cuda_helpers.cuh"

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

// HBM optimized copy - 256-bit loads, 256-byte bursts
__global__ void hbm_copy(Float8* dst, const Float8* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    constexpr int VECTORS_PER_LOOP = 4;  // 128 bytes per iteration
    for (size_t base = tid * VECTORS_PER_LOOP; base < n; base += stride * VECTORS_PER_LOOP) {
        #pragma unroll
        for (int i = 0; i < VECTORS_PER_LOOP; ++i) {
            size_t idx = base + i;
            if (idx >= n) break;
            dst[idx] = src[idx];  // 256-bit load/store
        }
    }
}

int main() {
    const size_t size_bytes = 256 * 1024 * 1024;  // 256 MB
    const size_t n_vec8 = size_bytes / sizeof(Float8);
    
    Float8 *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, size_bytes));
    CUDA_CHECK(cudaMemset(d_src, 1, size_bytes));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        hbm_copy<<<256, 256>>>(d_dst, d_src, n_vec8);
        CUDA_CHECK_LAST_ERROR();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    double bw_tbs = (size_bytes * 2 / (avg_ms / 1000.0)) / 1e12;
    
    printf("HBM (Float8, 256-byte bursts): %.2f ms, %.2f TB/s\n", avg_ms, bw_tbs);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}





