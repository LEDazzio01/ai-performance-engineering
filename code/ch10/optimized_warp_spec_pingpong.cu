// optimized_warp_spec_pingpong.cu - Warp Specialization with Ping-Pong Pattern (Ch10)
//
// WHAT: CUTLASS-style ping-pong pattern where compute warps alternate between:
//   - Group A: Compute tile N while Group B runs epilogue for tile N-1
//   - Group B: Compute tile N+1 while Group A runs epilogue for tile N
//
// WHY: Standard warp specialization has a gap between compute and epilogue.
// Ping-pong overlaps epilogue of tile N-1 with compute of tile N:
//
//   Without ping-pong:
//     [Load N] [Compute N] [Epilogue N] [Load N+1] [Compute N+1] [Epilogue N+1]
//
//   With ping-pong:
//     [Load N] [Compute N     ] [Load N+1] [Compute N+1   ]
//             [Epilogue N-1  ]           [Epilogue N    ]
//
// WHEN TO USE:
//   - GEMM with expensive epilogues (bias, activation, quantization)
//   - FlashAttention-3 style compute/softmax overlap
//   - Any kernel where epilogue is significant fraction of time
//
// REQUIREMENTS:
//   - Enough warps to split into two consumer groups
//   - Double-buffered accumulator registers
//   - Careful synchronization between groups

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

// Configuration
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

// Warp roles
constexpr int LOADER_WARPS = 2;      // Warps dedicated to loading
constexpr int COMPUTE_WARPS = 6;     // Warps for compute (split into 2 groups)
constexpr int TOTAL_WARPS = LOADER_WARPS + COMPUTE_WARPS;
constexpr int BLOCK_SIZE = TOTAL_WARPS * 32;

// Ping-pong groups
constexpr int COMPUTE_GROUP_SIZE = COMPUTE_WARPS / 2;  // 3 warps per group

//============================================================================
// Ping-Pong Warp Specialized GEMM Kernel
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void pingpong_warp_specialized_gemm(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [K, N]
    const float* __restrict__ bias, // [N]
    float* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    
    if (tile_m * TILE_M >= M || tile_n * TILE_N >= N) return;
    
    // Double-buffered shared memory for A and B tiles
    __shared__ alignas(128) float A_smem[2][TILE_M][TILE_K + 4];
    __shared__ alignas(128) float B_smem[2][TILE_K][TILE_N + 4];
    
    // Determine warp role
    const bool is_loader = warp_id < LOADER_WARPS;
    const int compute_warp = warp_id - LOADER_WARPS;  // -2 to 3
    const int compute_group = (compute_warp >= 0) ? (compute_warp / COMPUTE_GROUP_SIZE) : -1;  // 0 or 1
    
    // Double-buffered accumulators for ping-pong
    // Each compute group has its own accumulator
    float acc[2][4][4] = {{{0.0f}}};  // [buffer][row][col]
    
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    //========================================================================
    // MAIN LOOP with Ping-Pong Pattern
    //========================================================================
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        const int buffer = k_tile % 2;
        const int prev_buffer = (k_tile + 1) % 2;
        
        //====================================================================
        // LOADER WARPS: Load tiles for current iteration
        //====================================================================
        if (is_loader) {
            const int load_tid = warp_id * 32 + lane_id;
            
            // Load A tile
            for (int i = load_tid; i < TILE_M * TILE_K; i += LOADER_WARPS * 32) {
                int mm = i / TILE_K;
                int kk = i % TILE_K;
                int global_m = tile_m * TILE_M + mm;
                int global_k = k_base + kk;
                
                A_smem[buffer][mm][kk] = (global_m < M && global_k < K) 
                    ? A[global_m * K + global_k] : 0.0f;
            }
            
            // Load B tile
            for (int i = load_tid; i < TILE_K * TILE_N; i += LOADER_WARPS * 32) {
                int kk = i / TILE_N;
                int nn = i % TILE_N;
                int global_k = k_base + kk;
                int global_n = tile_n * TILE_N + nn;
                
                B_smem[buffer][kk][nn] = (global_k < K && global_n < N)
                    ? B[global_k * N + global_n] : 0.0f;
            }
        }
        
        __syncthreads();  // Ensure tiles are loaded
        
        //====================================================================
        // COMPUTE WARPS: Ping-Pong Pattern
        //====================================================================
        if (!is_loader && compute_warp >= 0) {
            // Determine which group does what this iteration
            const bool my_turn_compute = (k_tile % 2) == compute_group;
            
            if (my_turn_compute) {
                //============================================================
                // This group: COMPUTE on current tile
                //============================================================
                const int local_warp = compute_warp % COMPUTE_GROUP_SIZE;
                const int rows_per_warp = TILE_M / COMPUTE_GROUP_SIZE;
                const int row_start = local_warp * rows_per_warp;
                
                // Each thread computes 4x4 output elements
                const int thread_row = row_start + (lane_id / 8) * 4;
                const int thread_col = (lane_id % 8) * 8;
                
                #pragma unroll
                for (int k = 0; k < TILE_K; ++k) {
                    float a_vals[4], b_vals[4];
                    
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        a_vals[i] = A_smem[buffer][thread_row + i][k];
                    }
                    
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        b_vals[j] = B_smem[buffer][k][thread_col + j];
                    }
                    
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc[buffer][i][j] += a_vals[i] * b_vals[j];
                        }
                    }
                }
            } else if (k_tile > 0) {
                //============================================================
                // This group: EPILOGUE on previous tile (bias + ReLU)
                //============================================================
                const int local_warp = compute_warp % COMPUTE_GROUP_SIZE;
                const int rows_per_warp = TILE_M / COMPUTE_GROUP_SIZE;
                const int row_start = local_warp * rows_per_warp;
                
                const int thread_row = row_start + (lane_id / 8) * 4;
                const int thread_col = (lane_id % 8) * 8;
                
                // Apply bias and ReLU, then store
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int global_m = tile_m * TILE_M + thread_row + i;
                        int global_n = tile_n * TILE_N + thread_col + j;
                        
                        if (global_m < M && global_n < N) {
                            float val = acc[prev_buffer][i][j];
                            
                            // Add bias
                            val += bias[global_n];
                            
                            // ReLU
                            val = fmaxf(val, 0.0f);
                            
                            // Store
                            C[global_m * N + global_n] = val;
                            
                            // Clear accumulator for reuse
                            acc[prev_buffer][i][j] = 0.0f;
                        }
                    }
                }
            }
        }
        
        __syncthreads();  // Ensure compute/epilogue done before next load
    }
    
    //========================================================================
    // FINAL EPILOGUE: Process last tile's results
    //========================================================================
    if (!is_loader && compute_warp >= 0) {
        const int final_buffer = (num_k_tiles - 1) % 2;
        const int local_warp = compute_warp % COMPUTE_GROUP_SIZE;
        const int rows_per_warp = TILE_M / COMPUTE_GROUP_SIZE;
        const int row_start = local_warp * rows_per_warp;
        
        const int thread_row = row_start + (lane_id / 8) * 4;
        const int thread_col = (lane_id % 8) * 8;
        
        // Only the group that computed the last tile writes final results
        const bool computed_last = ((num_k_tiles - 1) % 2) == compute_group;
        
        if (computed_last) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int global_m = tile_m * TILE_M + thread_row + i;
                    int global_n = tile_n * TILE_N + thread_col + j;
                    
                    if (global_m < M && global_n < N) {
                        float val = acc[final_buffer][i][j];
                        val += bias[global_n];
                        val = fmaxf(val, 0.0f);
                        C[global_m * N + global_n] = val;
                    }
                }
            }
        }
    }
}

//============================================================================
// Standard Warp Specialized GEMM (No Ping-Pong) for Comparison
//============================================================================

__global__ __launch_bounds__(BLOCK_SIZE)
void standard_warp_specialized_gemm(
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
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        
        // Load
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
        
        // Compute (all compute warps do the same thing)
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
    
    // Epilogue (sequential, not overlapped)
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
    
    printf("Warp Specialization Ping-Pong Pattern\n");
    printf("=====================================\n");
    printf("Device: %s\n\n", prop.name);
    
    // Matrix dimensions
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    printf("GEMM: [%d, %d] x [%d, %d] + bias + ReLU\n", M, K, K, N);
    printf("Tile: %dx%dx%d\n", TILE_M, TILE_N, TILE_K);
    printf("Warps: %d loader, %d compute (2 groups of %d)\n\n", 
           LOADER_WARPS, COMPUTE_WARPS, COMPUTE_GROUP_SIZE);
    
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
    
    //========================================================================
    // Benchmark Standard Warp Specialization
    //========================================================================
    for (int i = 0; i < warmup; ++i) {
        standard_warp_specialized_gemm<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        standard_warp_specialized_gemm<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_standard;
    CUDA_CHECK(cudaEventElapsedTime(&ms_standard, start, stop));
    float avg_standard = ms_standard / iterations;
    
    //========================================================================
    // Benchmark Ping-Pong Warp Specialization
    //========================================================================
    for (int i = 0; i < warmup; ++i) {
        pingpong_warp_specialized_gemm<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        pingpong_warp_specialized_gemm<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms_pingpong;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pingpong, start, stop));
    float avg_pingpong = ms_pingpong / iterations;
    
    //========================================================================
    // Results
    //========================================================================
    double flops = 2.0 * M * N * K;
    double tflops_standard = (flops / 1e12) / (avg_standard / 1000.0);
    double tflops_pingpong = (flops / 1e12) / (avg_pingpong / 1000.0);
    
    printf("Results:\n");
    printf("  Standard Warp Spec: %.3f ms (%.2f TFLOPS)\n", avg_standard, tflops_standard);
    printf("  Ping-Pong Warp Spec: %.3f ms (%.2f TFLOPS)\n", avg_pingpong, tflops_pingpong);
    printf("\n");
    
    float speedup = avg_standard / avg_pingpong;
    if (speedup > 1.0f) {
        printf("  Ping-Pong is %.2fx FASTER\n", speedup);
        printf("  (Epilogue overlapped with compute)\n");
    } else {
        printf("  Standard is %.2fx faster\n", 1.0f / speedup);
        printf("  (Ping-pong overhead exceeds epilogue overlap benefit)\n");
    }
    
    printf("\nNote: Ping-pong shines when epilogue is expensive (e.g., softmax, quantization)\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_bias));
    
    return 0;
}

