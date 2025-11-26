// baseline_pipeline_3stage.cu - 2-Stage Pipeline Baseline (Ch10)
//
// WHAT: Standard 2-stage (double-buffered) pipeline for comparison.
// Only one load can be in flight while computing.
//
// WHY THIS IS SLOWER:
//   - 2-stage can only hide ~1 load latency
//   - For high-latency HBM, compute may still stall waiting for data
//   - 3-stage provides deeper latency hiding
//
// COMPARE WITH: optimized_pipeline_3stage.cu
//   - Optimized uses 3 buffers to hide 2 load latencies
//   - Better for memory-bound kernels with significant load latency

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Configuration (same as optimized for fair comparison)
constexpr int TILE_SIZE = 64;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_STAGES = 2;  // 2-stage baseline

//============================================================================
// 2-Stage Pipelined GEMV Kernel (Baseline)
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void baseline_gemv_2stage_pipeline(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ x,   // [K]
    float* __restrict__ y,         // [M]
    int M, int K
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * TILE_SIZE;
    
    // Double buffer (2 stages only)
    __shared__ alignas(128) float x_smem[NUM_STAGES][TILE_SIZE];
    
    auto pipeline = cuda::make_pipeline();
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    constexpr int ROWS_PER_THREAD = (TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float acc[ROWS_PER_THREAD] = {0.0f};
    
    //========================================================================
    // PROLOGUE: Load first tile only (2-stage = 1 prefetch)
    //========================================================================
    if (num_tiles > 0) {
        pipeline.producer_acquire();
        if (tid < TILE_SIZE) {
            int k_idx = tid;
            x_smem[0][tid] = (k_idx < K) ? x[k_idx] : 0.0f;
        }
        pipeline.producer_commit();
    }
    
    //========================================================================
    // MAIN LOOP: Can only overlap 1 load with compute
    //========================================================================
    for (int tile = 0; tile < num_tiles; ++tile) {
        int compute_stage = tile % NUM_STAGES;
        int load_stage = (tile + 1) % NUM_STAGES;
        
        // Load next tile (only 1 ahead)
        if (tile + 1 < num_tiles) {
            pipeline.producer_acquire();
            if (tid < TILE_SIZE) {
                int k_idx = (tile + 1) * TILE_SIZE + tid;
                x_smem[load_stage][tid] = (k_idx < K) ? x[k_idx] : 0.0f;
            }
            pipeline.producer_commit();
        }
        
        pipeline.consumer_wait();
        __syncthreads();
        
        int k_base = tile * TILE_SIZE;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            int row_offset = r * BLOCK_SIZE + tid;
            if (row_offset < TILE_SIZE) {
                int global_row = row_start + row_offset;
                if (global_row < M) {
                    float sum = 0.0f;
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE && k_base + k < K; ++k) {
                        sum += A[global_row * K + k_base + k] * x_smem[compute_stage][k];
                    }
                    acc[r] += sum;
                }
            }
        }
        
        __syncthreads();
        pipeline.consumer_release();
    }
    
    //========================================================================
    // EPILOGUE: Write results
    //========================================================================
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int row_offset = r * BLOCK_SIZE + tid;
        if (row_offset < TILE_SIZE) {
            int global_row = row_start + row_offset;
            if (global_row < M) {
                y[global_row] = acc[r];
            }
        }
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline 2-Stage Pipeline GEMV\n");
    printf("==============================\n");
    printf("Device: %s\n\n", prop.name);
    
    // Matrix dimensions (tall matrix for GEMV)
    const int M = 16384;
    const int K = 8192;
    
    printf("GEMV: [%d, %d] x [%d] = [%d]\n", M, K, K, M);
    printf("Tile: %d, Stages: %d (baseline)\n\n", TILE_SIZE, NUM_STAGES);
    
    // Allocate
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_x = K * sizeof(float);
    size_t bytes_y = M * sizeof(float);
    
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_y));
    
    // Initialize
    std::vector<float> h_A(M * K), h_x(K);
    for (size_t i = 0; i < M * K; ++i) h_A[i] = 0.001f;
    for (int i = 0; i < K; ++i) h_x[i] = 0.001f;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes_x, cudaMemcpyHostToDevice));
    
    // Launch config
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 50;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        baseline_gemv_2stage_pipeline<<<grid, block>>>(d_A, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        baseline_gemv_2stage_pipeline<<<grid, block>>>(d_A, d_x, d_y, M, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    double bytes_accessed = (double)M * K * sizeof(float) + K * sizeof(float) + M * sizeof(float);
    double bandwidth = bytes_accessed / (avg_ms / 1000.0) / 1e9;
    
    printf("Results:\n");
    printf("  2-Stage: %.3f ms (%.1f GB/s)\n", avg_ms, bandwidth);
    printf("\nNote: 2-stage can hide ~1 load latency.\n");
    printf("Compare with optimized_pipeline_3stage for deeper hiding.\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    return 0;
}

