// baseline_float8_vector.cu - Scalar Loads (Ch7)
//
// WHAT: Naive scalar loads - 1 float (4 bytes) per memory operation.
//
// WHY THIS IS SLOWER:
//   - Each thread issues individual 4-byte loads/stores
//   - Memory controller must handle many small requests
//   - Instruction-limited: too many load/store instructions
//   - Does NOT saturate HBM bandwidth on modern GPUs
//
// BOOK CONTEXT (Ch7 - Memory Coalescing & Vectorization):
//   The book discusses memory coalescing and vectorized loads as key optimizations.
//   Scalar loads waste memory bandwidth because:
//   1. Each warp issues 32 separate 4-byte requests
//   2. Memory controller must coalesce them (overhead)
//   3. Cache lines are 128 bytes, but we only use 4 bytes per request initially
//
// EXPECTED PERFORMANCE:
//   - ~2.5-3 TB/s effective bandwidth (well below HBM peak of 8 TB/s)
//   - Limited by instruction throughput, not memory bandwidth
//
// COMPARE WITH: optimized_float8_vector.cu
//   - Uses float4 (16-byte) vectorized loads
//   - 4 floats per load instruction → fewer instructions
//   - Saturates HBM bandwidth at ~6 TB/s
//   - Expected speedup: ~2-2.5x

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int BLOCK_SIZE = 256;

//============================================================================
// Baseline: Scalar Loads (4 bytes per operation)
// - Each thread loads/stores individual floats
// - Memory controller must coalesce 32 × 4-byte requests per warp
// - Instruction-limited: many load/store instructions
//============================================================================

__global__ void baseline_vector_add_scalar(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Scalar loads: 2 × 4-byte loads, 1 × 4-byte store
        // Per warp: 32 × 4B = 128 bytes, but issued as 32 separate requests
        c[idx] = a[idx] + b[idx];
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline: Scalar (4-byte) Loads\n");
    printf("================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("\n");
    
    // Large array to measure bandwidth accurately
    const int N = 128 * 1024 * 1024;  // 128M floats = 512 MB per array
    const size_t bytes = N * sizeof(float);
    
    printf("Array size: %zu MB per array\n", bytes / (1024 * 1024));
    printf("Total data movement: %zu MB (2 reads + 1 write)\n", 
           3 * bytes / (1024 * 1024));
    printf("Load width: 4 bytes (1 float per instruction)\n\n");
    
    // Allocate
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Initialize
    std::vector<float> h_data(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_a, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_data.data(), bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Clear L2 cache
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        baseline_vector_add_scalar<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        baseline_vector_add_scalar<<<grid, block>>>(d_a, d_b, d_c, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;
    
    // Bandwidth: 2 reads + 1 write
    float bandwidth_gb = (3.0f * bytes) / (ms / 1000.0f) / 1e9f;
    
    printf("Results:\n");
    printf("  Time per iteration: %.3f ms\n", ms);
    printf("  Effective bandwidth: %.1f GB/s\n", bandwidth_gb);
    printf("  HBM utilization: %.1f%% (peak ~8000 GB/s)\n", 
           100.0f * bandwidth_gb / 8000.0f);
    
    printf("\nKey insight:\n");
    printf("  Scalar loads are instruction-limited, not bandwidth-limited.\n");
    printf("  Many small loads → high instruction overhead.\n");
    printf("  See optimized_float8_vector.cu for vectorized loads.\n");

#ifdef VERIFY
    std::vector<float> h_verify(N);
    CUDA_CHECK(cudaMemcpy(h_verify.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    float checksum = 0.0f;
    VERIFY_CHECKSUM(h_verify.data(), N, &checksum);
    VERIFY_PRINT_CHECKSUM(checksum);
#endif
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return 0;
}
