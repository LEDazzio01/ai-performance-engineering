// optimized_float8_vector.cu - Vectorized Loads (Ch7)
//
// WHAT: float4 vectorized loads - 4 floats (16 bytes) per memory operation.
//
// WHY THIS IS FASTER:
//   - Each thread issues single 16-byte loads/stores
//   - 4x fewer memory instructions than scalar
//   - Saturates HBM bandwidth on modern GPUs
//   - Memory controller handles fewer, wider requests
//
// BOOK CONTEXT (Ch7 - Memory Coalescing & Vectorization):
//   The book emphasizes vectorized loads as a key optimization:
//   1. float4 loads issue 128-byte requests per warp (32 threads × 4B = 128B coalesced)
//      vs scalar which issues 32 separate 4B requests that need coalescing
//   2. Fewer instructions = higher throughput
//   3. Better cache line utilization (128B cache lines match float4 × 32 threads)
//
// EXPECTED PERFORMANCE:
//   - ~6 TB/s effective bandwidth (near HBM peak)
//   - Bandwidth-limited: HBM is the bottleneck, not instructions
//   - ~2-2.5x faster than scalar baseline
//
// EDUCATIONAL NOTE ON float8 (32-byte) LOADS:
//   Going from float4 (16B) to float8 (32B) does NOT improve performance for
//   simple kernels because HBM bandwidth is already saturated with float4.
//   32-byte loads help when:
//   - Processing FP8 data (32 FP8 values per load vs 8)
//   - Instruction throughput is the bottleneck (complex kernels)
//   - See optimized_fp8_32byte_loads.cu for FP8 use case

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
// Optimized: float4 Vectorized Loads (16 bytes per operation)
// - Each thread loads/stores 4 floats at once
// - 4x fewer instructions than scalar
// - Saturates HBM memory bandwidth
//============================================================================

__global__ void optimized_vector_add_float4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Vectorized loads: 2 × 16-byte loads, 1 × 16-byte store
        // Per warp: 32 × 16B = 512 bytes per load instruction
        float4 va = a[idx];
        float4 vb = b[idx];
        
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        
        c[idx] = vc;
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Optimized: float4 (16-byte) Vectorized Loads\n");
    printf("=============================================\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("\n");
    
    // Large array to measure bandwidth accurately
    const int N = 128 * 1024 * 1024;  // 128M floats = 512 MB per array
    const size_t bytes = N * sizeof(float);
    const int num_float4 = N / 4;
    
    printf("Array size: %zu MB per array\n", bytes / (1024 * 1024));
    printf("Total data movement: %zu MB (2 reads + 1 write)\n", 
           3 * bytes / (1024 * 1024));
    printf("Load width: 16 bytes (4 floats per instruction)\n\n");
    
    // Allocate (cudaMalloc guarantees 256-byte alignment, sufficient for float4)
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
    dim3 grid((num_float4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Clear L2 cache
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        optimized_vector_add_float4<<<grid, block>>>(
            reinterpret_cast<float4*>(d_a),
            reinterpret_cast<float4*>(d_b),
            reinterpret_cast<float4*>(d_c),
            num_float4
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        optimized_vector_add_float4<<<grid, block>>>(
            reinterpret_cast<float4*>(d_a),
            reinterpret_cast<float4*>(d_b),
            reinterpret_cast<float4*>(d_c),
            num_float4
        );
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
    printf("  float4 loads saturate HBM bandwidth.\n");
    printf("  4x fewer instructions than scalar → higher throughput.\n");
    printf("  Wider loads (float8) don't help further - HBM is the bottleneck.\n");

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
