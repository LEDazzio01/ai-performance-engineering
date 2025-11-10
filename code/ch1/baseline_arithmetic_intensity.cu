/*
 * baseline_arithmetic_intensity.cu - Baseline kernel (low arithmetic intensity)
 * 
 * Simple multiply operation: AI ~0.125 FLOP/Byte (memory-bound)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N (64 * 1024 * 1024)  // 64M elements (power of two for mask)
#define BLOCK_SIZE 128
#define STRIDE 17

#define CUDA_CHECK(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Baseline kernel: Simple multiply (low AI)
__global__ void baseline_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    const int mask = n - 1;
    for (int linear = idx; linear < n; linear += grid_stride) {
        int scatter = (linear * STRIDE) & mask;  // strided, non-coalesced reads
        float aval = a[scatter];
        float bval = b[scatter];
        out[linear] = aval * bval;
    }
}

// AI = 1 FLOP / (2 reads + 1 write) × 4 bytes = 1 / 12 = 0.083 FLOP/Byte
// Actually: 1 FLOP / 8 bytes (2 loads) = 0.125 FLOP/Byte (ignoring store)

int main() {
    printf("=== Baseline: Arithmetic Intensity Demo ===\n");
    printf("Array size: %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("AI: ~0.125 FLOP/Byte (memory-bound)\n\n");
    
    // Allocate host memory
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 100) / 100.0f + 0.01f;
        h_b[i] = (float)(rand() % 100) / 100.0f + 0.01f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    int iterations = 200;
    int n = N;
    
    // Warmup
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (int i = 0; i < 5; i++) {
        baseline_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        baseline_kernel<<<grid, block>>>(d_a, d_b, d_out, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    time_ms /= iterations;
    
    double total_flops = 1.0 * N;
    double gflops = (total_flops / (time_ms / 1000.0)) / 1e9;
    double bytes_loaded = 8.0 * N;
    double ai = total_flops / bytes_loaded;
    
    printf("Time: %.3f ms\n", time_ms);
    printf("Performance: %.1f GFLOPS\n", gflops);
    printf("Estimated AI: %.3f FLOP/Byte\n", ai);
    printf("Status: Memory-bound (AI < 250)\n");
    
    // Cleanup
    free(h_a); free(h_b); free(h_out);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
