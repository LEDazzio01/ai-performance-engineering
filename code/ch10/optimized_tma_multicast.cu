// optimized_tma_multicast.cu - TMA Multicast for CTA Clusters (Ch10)
//
// WHAT: TMA multicast broadcasts data from global memory to ALL CTAs
// in a thread block cluster simultaneously.
//
// WHY: For algorithms where multiple CTAs need the same data (e.g., GEMM K tiles),
// multicast eliminates redundant memory traffic:
//   - Without multicast: Each CTA loads same tile → N× bandwidth
//   - With multicast: One load broadcasts to all → 1× bandwidth
//
// WHEN TO USE:
//   - GEMM: K-dimension tiles shared across M-split or N-split CTAs
//   - Attention: K/V tiles shared across query-split CTAs
//   - Any algorithm with data reuse across cluster CTAs
//
// REQUIREMENTS:
//   - SM 90+ (Hopper/Blackwell)
//   - Thread block clusters enabled
//   - TMA tensor descriptors

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
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

// Tile dimensions
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

// Cluster configuration: 2x2 = 4 CTAs per cluster
constexpr int CLUSTER_M = 2;
constexpr int CLUSTER_N = 2;
constexpr int CLUSTER_SIZE = CLUSTER_M * CLUSTER_N;

//============================================================================
// TMA Multicast Kernel
//============================================================================
// This kernel demonstrates the concept of TMA multicast where:
// - CTA (0,0) in each cluster loads the K-tile
// - The tile is multicast to all 4 CTAs in the cluster via DSMEM
//============================================================================

__global__ __launch_bounds__(256, 1)
void tma_multicast_gemm_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Get cluster and block info
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();
    
    const int cluster_rank = cluster.block_rank();
    const int cluster_m = cluster_rank / CLUSTER_N;  // 0 or 1
    const int cluster_n = cluster_rank % CLUSTER_N;  // 0 or 1
    
    // Global tile indices
    const int tile_m = blockIdx.x * CLUSTER_M + cluster_m;
    const int tile_n = blockIdx.y * CLUSTER_N + cluster_n;
    
    if (tile_m * TILE_M >= M || tile_n * TILE_N >= N) return;
    
    // Shared memory for tiles
    // A_tile: Each CTA has its own M-slice
    // B_tile: SHARED via multicast - only loaded once per cluster
    __shared__ alignas(128) float A_smem[TILE_M][TILE_K + 4];  // +4 for bank conflicts
    __shared__ alignas(128) float B_smem[TILE_K][TILE_N + 4];
    
    // Accumulator
    float acc[4][4] = {0.0f};  // Each thread computes 4x4 output elements
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // K-loop: process tiles along K dimension
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        
        //====================================================================
        // MULTICAST PATTERN: CTA (0,0) loads B_tile for entire cluster
        //====================================================================
        if (cluster_rank == 0) {
            // Only the "leader" CTA loads B_tile
            // In real TMA, this would use cp.async.bulk.tensor with mcast
            for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
                int kk = i / TILE_N;
                int nn = i % TILE_N;
                int global_k = k_base + kk;
                int global_n = tile_n * TILE_N + nn;
                
                if (global_k < K && global_n < N) {
                    B_smem[kk][nn] = B[global_k * N + global_n];
                } else {
                    B_smem[kk][nn] = 0.0f;
                }
            }
        }
        
        // Each CTA loads its own A_tile slice
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int mm = i / TILE_K;
            int kk = i % TILE_K;
            int global_m = tile_m * TILE_M + mm;
            int global_k = k_base + kk;
            
            if (global_m < M && global_k < K) {
                A_smem[mm][kk] = A[global_m * K + global_k];
            } else {
                A_smem[mm][kk] = 0.0f;
            }
        }
        
        //====================================================================
        // CLUSTER SYNC + DSMEM BROADCAST
        //====================================================================
        // After leader loads B_tile, sync cluster and broadcast via DSMEM
        cluster.sync();
        
        // Non-leader CTAs read B_tile from leader's shared memory via DSMEM
        // map_shared_rank returns a pointer to the same array type as B_smem
        auto leader_B_smem = cluster.map_shared_rank(B_smem, 0);
        
        // Copy from leader to local (simulating TMA multicast result)
        if (cluster_rank != 0) {
            for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
                int kk = i / TILE_N;
                int nn = i % TILE_N;
                B_smem[kk][nn] = leader_B_smem[kk][nn];
            }
        }
        
        block.sync();
        
        //====================================================================
        // COMPUTE: Standard GEMM tile computation
        //====================================================================
        // Each thread computes a 4x4 output tile
        const int thread_m = (tid / 16) * 4;  // 0, 4, 8, ... 60
        const int thread_n = (tid % 16) * 4;  // 0, 4, 8, ... 60
        
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
        
        cluster.sync();  // Ensure all CTAs done before next iteration
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
    
#else
    // Fallback for older architectures - simple tiled GEMM without clusters
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int tid = threadIdx.x;
    
    __shared__ float A_smem[TILE_M][TILE_K];
    __shared__ float B_smem[TILE_K][TILE_N];
    
    float acc[4][4] = {0.0f};
    
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; ++k_tile) {
        // Load tiles
        for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            int mm = i / TILE_K, kk = i % TILE_K;
            int gm = tile_m * TILE_M + mm, gk = k_tile * TILE_K + kk;
            A_smem[mm][kk] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            int kk = i / TILE_N, nn = i % TILE_N;
            int gk = k_tile * TILE_K + kk, gn = tile_n * TILE_N + nn;
            B_smem[kk][nn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        __syncthreads();
        
        // Compute
        int tm = (tid / 16) * 4, tn = (tid % 16) * 4;
        for (int k = 0; k < TILE_K; ++k) {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += A_smem[tm + i][k] * B_smem[k][tn + j];
        }
        __syncthreads();
    }
    
    // Store
    int tm = (tid / 16) * 4, tn = (tid % 16) * 4;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int gm = tile_m * TILE_M + tm + i;
            int gn = tile_n * TILE_N + tn + j;
            if (gm < M && gn < N) C[gm * N + gn] = acc[i][j];
        }
    }
#endif
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("TMA Multicast GEMM Example\n");
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    if (prop.major < 9) {
        printf("SKIPPED: TMA multicast requires SM 90+ (Hopper/Blackwell)\n");
        printf("Running fallback kernel without clusters...\n\n");
    }
    
    // Matrix dimensions
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    printf("Matrix: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Tile: %dx%dx%d, Cluster: %dx%d\n\n", TILE_M, TILE_N, TILE_K, CLUSTER_M, CLUSTER_N);
    
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
    dim3 block(256);
    dim3 grid((M + TILE_M * CLUSTER_M - 1) / (TILE_M * CLUSTER_M),
              (N + TILE_N * CLUSTER_N - 1) / (TILE_N * CLUSTER_N));
    
    // For SM90+, we would set cluster size via launch attributes
    // cudaLaunchAttribute attrs[1];
    // attrs[0].id = cudaLaunchAttributeClusterDimension;
    // attrs[0].val.clusterDim = {CLUSTER_M, CLUSTER_N, 1};
    
    // Warmup
    tma_multicast_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        tma_multicast_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\nNote: Full TMA multicast requires cluster launch attributes.\n");
    printf("This example demonstrates the DSMEM-based multicast pattern.\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}




