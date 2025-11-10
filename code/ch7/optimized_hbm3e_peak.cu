// optimized_hbm3e_peak.cu -- HBM3e peak bandwidth kernel.

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

// HBM3e peak bandwidth kernel - SIMPLE IS BEST!
// Just use massive parallelism with float4 - no fancy cache hints
__global__ void hbm3e_peak_copy(const float4* __restrict__ src,
                                 float4* __restrict__ dst,
                                 size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t base = tid * 4; base < n; base += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            size_t idx = base + j;
            if (idx >= n) {
                break;
            }
#if __CUDA_ARCH__ >= 900
            float x, y, z, w;
            asm volatile(
                "ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(src + idx));
            asm volatile(
                "st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};\n"
                :
                : "l"(dst + idx), "f"(x), "f"(y), "f"(z), "f"(w));
#else
            dst[idx] = src[idx];
#endif
        }
    }
}

int main() {
    const size_t target_bytes = 512ULL * 1024 * 1024;  // 512 MB
    const size_t n = target_bytes / sizeof(float);
    const size_t n4 = n / 4;
    
    float4 *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
    
    CUDA_CHECK(cudaMemset(d_src, 1, n * sizeof(float)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        hbm3e_peak_copy<<<2048, 512>>>((const float4*)d_src, (float4*)d_dst, n4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double bytes_transferred = 2.0 * n * sizeof(float) * iterations;
    double bandwidth_gbs = (bytes_transferred / elapsed_ms) / 1e6;
    double bandwidth_tbs = bandwidth_gbs / 1024.0;
    
    printf("HBM3e peak copy: %.2f ms, %.2f TB/s\n", elapsed_ms / iterations, bandwidth_tbs);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    
    return 0;
}
