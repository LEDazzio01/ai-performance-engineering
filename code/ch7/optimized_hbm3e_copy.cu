// optimized_hbm3e_copy.cu -- 256-byte bursts with cache streaming (optimized).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "../common/headers/cuda_helpers.cuh"

// Optimized: 256-byte bursts with cache streaming (Blackwell HBM3e optimal)
__global__ void hbm3e_optimized_copy_kernel(float4* dst, const float4* src, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    constexpr int VECTORS_PER_LOOP = 8;
    for (size_t base = tid * VECTORS_PER_LOOP; base < n; base += stride * VECTORS_PER_LOOP) {
        #pragma unroll
        for (int i = 0; i < VECTORS_PER_LOOP; ++i) {
            size_t idx = base + i;
            if (idx >= n) {
                break;
            }
            float4 value = src[idx];
#if __CUDA_ARCH__ >= 900
            asm volatile(
                "st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};\n"
                :
                : "l"(dst + idx),
                  "f"(value.x),
                  "f"(value.y),
                  "f"(value.z),
                  "f"(value.w));
#else
            dst[idx] = value;
#endif
        }
    }
}

int main() {
    const size_t size_bytes = 256 * 1024 * 1024;  // 256 MB
    const size_t n_floats = size_bytes / sizeof(float);
    const size_t n_float4 = n_floats / 4;
    
    float4* d_src = nullptr;
    float4* d_dst = nullptr;
    
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
        hbm3e_optimized_copy_kernel<<<256, 256>>>(d_dst, d_src, n_float4);
        CUDA_CHECK_LAST_ERROR();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    double bw = (size_bytes * 2 / (avg_ms / 1000.0)) / 1e9;
    
    printf("HBM3e optimized (256-byte bursts): %.2f ms, %.2f GB/s\n", avg_ms, bw);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}
