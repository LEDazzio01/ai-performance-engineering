// baseline_warp_spec_pingpong.cu - Standard Warp Specialization (No Ping-Pong) (Ch10)
//
// WHAT: Standard warp specialization where all compute warps do the same thing.
// Epilogue runs sequentially after all K-tiles are processed.
//
// WHY THIS IS SLOWER:
//   - Epilogue (bias + activation) runs after compute is done
//   - No overlap between compute and epilogue
//   - Gap in utilization while epilogue runs
//
// COMPARE WITH: optimized_warp_spec_pingpong.cu
//   - Optimized splits compute warps into two groups
//   - One group computes while other runs epilogue
//   - Better utilization through overlap

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

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
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

constexpr int LOADER_WARPS = 2;
constexpr int COMPUTE_WARPS = 6;
constexpr int TOTAL_WARPS = LOADER_WARPS + COMPUTE_WARPS;
constexpr int BLOCK_SIZE = TOTAL_WARPS * 32;

//============================================================================
// Baseline: Standard Warp Specialized GEMM (No Ping-Pong)
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void baseline_warp_specialized_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    
    if (tile_m * TILE_M >= M || tile_n * TILE_N >= N) return;
    
    __shared__ alignas(128) float A_smem[TILE_M][TILE_K + 4];
    __shared__ alignas(128) float B_smem[TILE_K][TILE_N + 4];
    
    const bool is_loader = warp_id < LOADER_WARPS;
    const int compute_warp = warp_id - LOADER_WARPS;
    
    float acc[4][4] = {{0.0f}};
    
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    //========================================================================
    // MAIN LOOP: All compute warps do the same thing
    //========================================================================
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        
        // LOAD: Loader warps fetch tiles
        if (is_loader) {
            const int load_tid = warp_id * 32 + lane_id;
            
            for (int i = load_tid; i < TILE_M * TILE_K; i += LOADER_WARPS * 32) {
                int mm = i / TILE_K, kk = i % TILE_K;
                int gm = tile_m * TILE_M + mm, gk = k_base + kk;
                A_smem[mm][kk] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
            }
            
            for (int i = load_tid; i < TILE_K * TILE_N; i += LOADER_WARPS * 32) {
                int kk = i / TILE_N, nn = i % TILE_N;
                int gk = k_base + kk, gn = tile_n * TILE_N + nn;
                B_smem[kk][nn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
            }
        }
        
        __syncthreads();
        
        // COMPUTE: All compute warps do the same computation
        // (No ping-pong - all warps work on same tile)
        if (!is_loader && compute_warp >= 0 && compute_warp < COMPUTE_WARPS) {
            const int rows_per_warp = TILE_M / COMPUTE_WARPS;
            const int row_start = compute_warp * rows_per_warp;
            const int thread_row = row_start + (lane_id / 8) * 4;
            const int thread_col = (lane_id % 8) * 8;
            
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float a_val = A_smem[thread_row + i][k];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] += a_val * B_smem[k][thread_col + j];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    //========================================================================
    // EPILOGUE: Sequential - runs AFTER all compute is done
    // (This is the bottleneck - no overlap with compute)
    //========================================================================
    if (!is_loader && compute_warp >= 0 && compute_warp < COMPUTE_WARPS) {
        const int rows_per_warp = TILE_M / COMPUTE_WARPS;
        const int row_start = compute_warp * rows_per_warp;
        const int thread_row = row_start + (lane_id / 8) * 4;
        const int thread_col = (lane_id % 8) * 8;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gm = tile_m * TILE_M + thread_row + i;
                int gn = tile_n * TILE_N + thread_col + j;
                if (gm < M && gn < N) {
                    // Bias + ReLU (same as optimized)
                    float val = acc[i][j] + bias[gn];
                    C[gm * N + gn] = fmaxf(val, 0.0f);
                }
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
    
    printf("Baseline Warp Specialization (No Ping-Pong)\n");
    printf("===========================================\n");
    printf("Device: %s\n\n", prop.name);
    
    // Matrix dimensions
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    printf("GEMM: [%d, %d] x [%d, %d] + bias + ReLU\n", M, K, K, N);
    printf("Tile: %dx%dx%d\n", TILE_M, TILE_N, TILE_K);
    printf("Warps: %d loader, %d compute (all same role)\n\n", 
           LOADER_WARPS, COMPUTE_WARPS);
    
    // Allocate
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    size_t bytes_bias = N * sizeof(float);
    
    float *d_A, *d_B, *d_C, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_bias, bytes_bias));
    
    // Initialize
    std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
    for (int i = 0; i < M * K; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < N; ++i) h_bias[i] = (rand() % 10) / 10.0f;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bytes_bias, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 50;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        baseline_warp_specialized_gemm<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        baseline_warp_specialized_gemm<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (avg_ms / 1000.0);
    
    printf("Results:\n");
    printf("  Time: %.3f ms (%.2f TFLOPS)\n", avg_ms, tflops);
    printf("\nNote: Epilogue runs sequentially after compute.\n");
    printf("Compare with optimized_warp_spec_pingpong for overlap.\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_bias));
    
    return 0;
}

