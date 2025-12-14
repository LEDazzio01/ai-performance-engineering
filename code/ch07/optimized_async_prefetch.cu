// optimized_async_prefetch.cu -- Double-buffered async prefetch (optimized).
// Demonstrates overlapping memory loads with computation using software pipelining.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "../core/common/headers/cuda_verify.cuh"

namespace cg = cooperative_groups;

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
constexpr int PIPELINE_STAGES = 2;

// Same light compute as baseline - memory latency becomes visible
__device__ __forceinline__ float light_compute(float x) {
    float y = x * 2.0f + 1.0f;
    y = y * y - x;
    y = y * 0.5f + x * 0.25f;
    return y;
}

// Optimized: Double-buffered pipeline overlaps load(tile N+1) with compute(tile N)
__global__ void pipelined_kernel(const float* __restrict__ data,
                                  float* __restrict__ out,
                                  int n,
                                  int total_tiles) {
    extern __shared__ float shared_mem[];
    float* stage_buf[PIPELINE_STAGES];
    for (int s = 0; s < PIPELINE_STAGES; ++s) {
        stage_buf[s] = shared_mem + s * TILE_SIZE;
    }
    
    cg::thread_block block = cg::this_thread_block();
    
    __shared__ alignas(cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>)
        unsigned char pipe_storage[sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>)];
    auto* pipe_state = reinterpret_cast<cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>*>(pipe_storage);
    if (threadIdx.x == 0) {
        new (pipe_state) cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>();
    }
    block.sync();
    auto pipe = cuda::make_pipeline(block, pipe_state);
    
    // Calculate tiles for this block
    const int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
    const int first_tile = blockIdx.x * tiles_per_block;
    const int last_tile = min(first_tile + tiles_per_block, total_tiles);
    
    auto issue_load = [&](int tile, int stage) {
        int tile_offset = tile * TILE_SIZE;
        int tile_elems = min(TILE_SIZE, n - tile_offset);
        pipe.producer_acquire();
        if (tile_elems > 0) {
            cuda::memcpy_async(block, stage_buf[stage],
                              data + tile_offset,
                              tile_elems * sizeof(float), pipe);
        }
        pipe.producer_commit();
    };
    
    // Prime the pipeline: load first PIPELINE_STAGES tiles
    for (int i = 0; i < PIPELINE_STAGES && (first_tile + i) < last_tile; ++i) {
        issue_load(first_tile + i, i);
    }
    
    // Main loop: compute current tile while loading next
    for (int tile = first_tile; tile < last_tile; ++tile) {
        int stage = (tile - first_tile) % PIPELINE_STAGES;
        int tile_offset = tile * TILE_SIZE;
        int tile_elems = min(TILE_SIZE, n - tile_offset);
        
        // Wait for this tile's data
        pipe.consumer_wait();
        block.sync();
        
        // Compute while next tile is loading (overlap!)
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            float v = stage_buf[stage][i];
            v = light_compute(v);
            out[tile_offset + i] = v;
        }
        
        pipe.consumer_release();
        
        // Issue load for tile + PIPELINE_STAGES (non-blocking)
        int next = tile + PIPELINE_STAGES;
        if (next < last_tile) {
            issue_load(next, stage);
        }
        
        block.sync();
    }
}

int main() {
    constexpr int N = 64 * 1024 * 1024;  // Same as baseline - larger workload
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
    const size_t shared_bytes = PIPELINE_STAGES * TILE_SIZE * sizeof(float);
    
    // Warmup
    pipelined_kernel<<<grid, THREADS, shared_bytes>>>(d_in, d_out, N, total_tiles);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    constexpr int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        pipelined_kernel<<<grid, THREADS, shared_bytes>>>(d_in, d_out, N, total_tiles);
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
    
    printf("Pipelined async prefetch (optimized): %.3f ms\n", avg_ms);
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
