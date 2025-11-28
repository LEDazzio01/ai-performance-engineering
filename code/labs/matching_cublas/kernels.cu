// kernels.cu - Progressive GEMM optimization stages for the Matching cuBLAS lab
//
// Stage 1 only: Naive GEMM without tensor cores (educational baseline)
// All tensor core stages use the tcgen05 module through Python.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// =============================================================================
// Stage 1: Naive GEMM (no tensor cores, basic shared memory tiling)
// =============================================================================
// This demonstrates basic GEMM without tensor cores - intentionally slow
// to show the massive speedup from tensor cores.

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int THREAD_TILE_M = 4;
constexpr int THREAD_TILE_N = 4;

__global__ void gemm_naive_smem(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ half sA[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    __shared__ half sB[TILE_K][TILE_N + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    // Thread tile position within block
    int thread_row = ty * THREAD_TILE_M;
    int thread_col = tx * THREAD_TILE_N;
    
    // Global tile offset
    int tile_row = by * TILE_M;
    int tile_col = bx * TILE_N;
    
    // Accumulator registers
    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    
    // Loop over K tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative loading of A tile (TILE_M x TILE_K)
        int threads_per_block = blockDim.x * blockDim.y;
        for (int idx = tid; idx < TILE_M * TILE_K; idx += threads_per_block) {
            int local_row = idx / TILE_K;
            int local_col = idx % TILE_K;
            int g_row = tile_row + local_row;
            int g_col = k_tile + local_col;
            sA[local_row][local_col] = (g_row < M && g_col < K) 
                ? A[g_row * K + g_col] 
                : __float2half(0.0f);
        }
        
        // Cooperative loading of B tile (TILE_K x TILE_N)
        // B is stored as N x K (row = output column, col = K dimension)
        for (int idx = tid; idx < TILE_K * TILE_N; idx += threads_per_block) {
            int local_row = idx / TILE_N;
            int local_col = idx % TILE_N;
            int g_n = tile_col + local_col;  // Which output column
            int g_k = k_tile + local_row;    // Which K position
            sB[local_row][local_col] = (g_n < N && g_k < K) 
                ? B[g_n * K + g_k]  // B[n, k] in NxK layout
                : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute thread's tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    accum[i][j] += __half2float(sA[thread_row + i][k]) * 
                                   __half2float(sB[k][thread_col + j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int g_row = tile_row + thread_row + i;
            int g_col = tile_col + thread_col + j;
            if (g_row < M && g_col < N) {
                C[g_row * N + g_col] = accum[i][j];
            }
        }
    }
}

// =============================================================================
// Host-side launcher functions
// =============================================================================

extern "C" {

void launch_gemm_naive_smem(
    const half* A, const half* B, float* C,
    int M, int N, int K, cudaStream_t stream
) {
    // 16x16 threads, each handles 4x4 elements = 64x64 tile per block
    dim3 block(16, 16);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    gemm_naive_smem<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

}  // extern "C"
