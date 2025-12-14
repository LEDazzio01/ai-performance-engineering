// optimized_copy_vectorized.cu - Vectorized memory copy (Ch7)
//
// WHAT: float4 vectorized loads - 4 floats (16 bytes) per memory operation.
// Simple memory copy benchmark comparing to scalar baseline.
//
// WHY THIS IS FASTER:
//   - Each thread issues single 16-byte loads/stores
//   - 4x fewer instructions than scalar
//   - Saturates HBM bandwidth

#include <cuda_runtime.h>
#include <cstdio>

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

constexpr int NUM_FLOATS = 64 * 1024 * 1024;  // 64M floats = 256 MB (same as baseline!)
constexpr int NUM_VEC4 = NUM_FLOATS / 4;
constexpr int BLOCK_SIZE = 256;

// float4 vectorized copy: 4 floats per thread
__global__ void copyVectorized(const float4* __restrict__ in,
                               float4* __restrict__ out,
                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // 16-byte load, 16-byte store
    }
}

int main() {
    printf("Optimized: float4 (16-byte) Vectorized Copy\n");
    printf("============================================\n");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    size_t bytes = NUM_FLOATS * sizeof(float);
    printf("Data size: %zu MB\n", bytes / (1024 * 1024));
    printf("Total transfer: %zu MB (1 read + 1 write)\n\n", 2 * bytes / (1024 * 1024));
    
    // Allocate
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    
    // Initialize
    float* h_data = new float[NUM_FLOATS];
    for (int i = 0; i < NUM_FLOATS; ++i) h_data[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice));
    
    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_VEC4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        copyVectorized<<<grid, block>>>(
            reinterpret_cast<float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            NUM_VEC4
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        copyVectorized<<<grid, block>>>(
            reinterpret_cast<float4*>(d_in),
            reinterpret_cast<float4*>(d_out),
            NUM_VEC4
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iterations;
    
    // Calculate bandwidth (read + write)
    float bandwidth_gb = (2.0f * bytes) / (ms / 1000.0f) / 1e9f;
    
    printf("Results:\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
    printf("TIME_MS: %.6f\n", ms);

#ifdef VERIFY
    float* h_verify = new float[NUM_FLOATS];
    CUDA_CHECK(cudaMemcpy(h_verify, d_out, bytes, cudaMemcpyDeviceToHost));
    float checksum = 0.0f;
    VERIFY_CHECKSUM(h_verify, NUM_FLOATS, &checksum);
    VERIFY_PRINT_CHECKSUM(checksum);
    delete[] h_verify;
#endif
    
    // Cleanup
    delete[] h_data;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    
    return 0;
}
