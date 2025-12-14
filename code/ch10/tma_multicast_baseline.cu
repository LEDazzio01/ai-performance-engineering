// tma_multicast_baseline.cu - Standard GEMM without TMA Multicast (Ch10)
//
// WHAT: Standard tiled GEMM where each CTA independently loads its tiles.
// No cluster coordination or multicast.
//
// WHY THIS IS SLOWER:
//   - Each CTA loads the same K-tiles redundantly
//   - N× memory bandwidth for N CTAs that need same data
//   - No benefit from cluster-level data sharing
//
// COMPARE WITH: tma_multicast_cluster.cu
//   - Optimized uses TMA multicast to broadcast shared tiles
//   - Single load serves all cluster CTAs
//   - Significant bandwidth savings

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

// Tile dimensions (same as optimized for fair comparison)
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int BLOCK_SIZE = 256;

//============================================================================
// Baseline: Standard Tiled GEMM (No Multicast)
//============================================================================
// Each block independently loads its own A and B tiles.
// No coordination between blocks - redundant loads for shared data.
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void baseline_tiled_gemm_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (tile_m * TILE_M >= M || tile_n * TILE_N >= N) return;
    
    // Shared memory for tiles - each block has its own copy
    __shared__ float A_smem[TILE_M][TILE_K];
    __shared__ float B_smem[TILE_K][TILE_N];
    
    // Accumulator
    float acc[4][4] = {0.0f};
    
    // K-loop: process tiles along K dimension
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        
        //====================================================================
        // BASELINE: Every block loads its own tiles (redundant for shared data)
        //====================================================================
        
        // Load A tile
        for (int i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int mm = i / TILE_K;
            int kk = i % TILE_K;
            int global_m = tile_m * TILE_M + mm;
            int global_k = k_base + kk;
            
            A_smem[mm][kk] = (global_m < M && global_k < K) 
                ? A[global_m * K + global_k] : 0.0f;
        }
        
        // Load B tile - THIS IS LOADED REDUNDANTLY BY ALL BLOCKS
        // In a cluster of 4 CTAs, this same B tile is loaded 4 times!
        for (int i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int kk = i / TILE_N;
            int nn = i % TILE_N;
            int global_k = k_base + kk;
            int global_n = tile_n * TILE_N + nn;
            
            B_smem[kk][nn] = (global_k < K && global_n < N)
                ? B[global_k * N + global_n] : 0.0f;
        }
        
        __syncthreads();
        
        //====================================================================
        // COMPUTE: Standard GEMM tile computation
        //====================================================================
        const int thread_m = (tid / 16) * 4;
        const int thread_n = (tid % 16) * 4;
        
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_vals[4], b_vals[4];
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_vals[i] = A_smem[thread_m + i][k];
                b_vals[i] = B_smem[k][thread_n + i];
            }
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    //========================================================================
    // STORE: Write output tile to global memory
    //========================================================================
    const int thread_m = (tid / 16) * 4;
    const int thread_n = (tid % 16) * 4;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int global_m = tile_m * TILE_M + thread_m + i;
            int global_n = tile_n * TILE_N + thread_n + j;
            
            if (global_m < M && global_n < N) {
                C[global_m * N + global_n] = acc[i][j];
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
    
    printf("Baseline Tiled GEMM (No Multicast)\n");
    printf("==================================\n");
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Matrix dimensions
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    printf("Matrix: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Tile: %dx%dx%d\n\n", TILE_M, TILE_N, TILE_K);
    
    // Allocate
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    // Initialize with random values
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(rand() % 100) / 100.0f;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    
    // Warmup
    baseline_tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        baseline_tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    printf("Results:\n");
    printf("  Avg time: %.3f ms\n", avg_ms);
    printf("  TFLOPS: %.2f\n", tflops);
    printf("\nNote: Each block loads B tiles independently.\n");
    printf("Compare with tma_multicast_cluster.cu for cluster-based sharing.\n");

#ifdef VERIFY
    std::vector<float> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (float v : h_C) {
        checksum += static_cast<double>(v);
    }
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}
