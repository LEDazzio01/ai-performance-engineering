// baseline_async_prefetch.cu -- Naive tiled processing without async prefetch (baseline).
// Demonstrates synchronous load-compute pattern with no overlap.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int TILE_SIZE = 4096;  // Larger tiles = more latency to hide

// Light compute function - memory latency becomes visible
// Async prefetch shines when memory latency > compute time
__device__ __forceinline__ float light_compute(float x) {
    // Simple arithmetic that completes quickly, exposing memory stalls
    float y = x * 2.0f + 1.0f;
    y = y * y - x;
    y = y * 0.5f + x * 0.25f;
    return y;
}

// Baseline: Load tile synchronously, then compute
// No overlap between memory and compute phases
__global__ void naive_tiled_kernel(const float* __restrict__ data,
                                   float* __restrict__ out,
                                   int n,
                                   int total_tiles) {
    extern __shared__ float smem[];
    
    // Process tiles assigned to this block
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int tile_offset = tile * TILE_SIZE;
        int tile_elems = min(TILE_SIZE, n - tile_offset);
        
        // PHASE 1: Load tile into shared memory (synchronous)
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            smem[i] = data[tile_offset + i];
        }
        __syncthreads();  // Must wait for all loads before compute
        
        // PHASE 2: Compute (cannot overlap with next tile's load)
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            float v = smem[i];
            v = light_compute(v);
            out[tile_offset + i] = v;
        }
        __syncthreads();  // Must wait for compute before loading next tile
    }
}

int main() {
    constexpr int N = 64 * 1024 * 1024;  // 64M elements - larger workload
    constexpr int THREADS = 256;
    const size_t bytes = N * sizeof(float);
    
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    
    const int total_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int grid = min(total_tiles, prop.multiProcessorCount * 4);
    const size_t shared_bytes = TILE_SIZE * sizeof(float);
    
    // Warmup
    naive_tiled_kernel<<<grid, THREADS, shared_bytes>>>(d_in, d_out, N, total_tiles);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    constexpr int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        naive_tiled_kernel<<<grid, THREADS, shared_bytes>>>(d_in, d_out, N, total_tiles);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (int i = 0; i < N; i += 10000) {
        checksum += h_out[i];
    }
    
    printf("Naive tiled processing (baseline): %.3f ms\n", avg_ms);
    printf("TIME_MS: %.6f\n", avg_ms);
    printf("Checksum: %.6f\n", checksum);
#ifdef VERIFY
    float verify_checksum = 0.0f;
    VERIFY_CHECKSUM(h_out.data(), N, &verify_checksum);
    VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    
    return 0;
}
