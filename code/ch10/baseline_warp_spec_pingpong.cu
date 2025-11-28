// baseline_warp_spec_pingpong.cu - Naive GEMM with Global Memory Round-Trip (Ch10)
//
// WHAT: Naive implementation that stores partial results to global memory
// after GEMM, then reads them back for epilogue. This extra round-trip
// is what optimized ping-pong avoids.
//
// WHY THIS IS SLOWER:
//   - Extra global memory write after GEMM compute
//   - Extra global memory read before epilogue
//   - Memory bandwidth becomes bottleneck
//
// COMPARE WITH: optimized_warp_spec_pingpong.cu
//   - Optimized keeps data in registers/shared memory
//   - No intermediate global memory round-trip
//   - Epilogue applied directly to accumulator values

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

// Configuration
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int BLOCK_SIZE = 256;

//============================================================================
// Kernel 1: GEMM compute - stores raw results to global memory
//============================================================================

__global__ void baseline_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C_temp,  // Intermediate storage
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    
    if (tile_m * TILE_M >= M || tile_n * TILE_N >= N) return;
    
    __shared__ alignas(128) float A_smem[TILE_M][TILE_K + 4];
    __shared__ alignas(128) float B_smem[TILE_K][TILE_N + 4];
    
    // Each thread computes 4x4 output elements
    const int thread_row = (tid / 16) * 4;
    const int thread_col = (tid % 16) * 4;
    
    float acc[4][4] = {{0.0f}};
    
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_base = k_tile * TILE_K;
        
        // Cooperative load of A tile
        for (int i = tid; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int mm = i / TILE_K, kk = i % TILE_K;
            int gm = tile_m * TILE_M + mm, gk = k_base + kk;
            A_smem[mm][kk] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
        }
        
        // Cooperative load of B tile
        for (int i = tid; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int kk = i / TILE_N, nn = i % TILE_N;
            int gk = k_base + kk, gn = tile_n * TILE_N + nn;
            B_smem[kk][nn] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute 4x4 tile per thread
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_vals[4], b_vals[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a_vals[i] = A_smem[thread_row + i][k];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b_vals[j] = B_smem[k][thread_col + j];
            
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
    
    // Store raw GEMM results to global memory (wasteful round-trip)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int gm = tile_m * TILE_M + thread_row + i;
            int gn = tile_n * TILE_N + thread_col + j;
            if (gm < M && gn < N) {
                C_temp[gm * N + gn] = acc[i][j];
            }
        }
    }
}

//============================================================================
// Kernel 2: Apply bias - reads from global, writes to global
//============================================================================

__global__ void baseline_bias_kernel(
    const float* __restrict__ C_temp,
    const float* __restrict__ bias,
    float* __restrict__ C_biased,
    int M, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int col = i % N;
        C_biased[i] = C_temp[i] + bias[col];
    }
}

//============================================================================
// Kernel 3: Apply ReLU - reads from global, writes to global
//============================================================================

__global__ void baseline_relu_kernel(
    const float* __restrict__ C_biased,
    float* __restrict__ C,
    int M, int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        C[i] = fmaxf(C_biased[i], 0.0f);
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline GEMM (Global Memory Round-Trip)\n");
    printf("========================================\n");
    printf("Device: %s\n\n", prop.name);
    
    // Matrix dimensions - larger for more memory traffic
    const int M = 4096;
    const int N = 4096;
    const int K = 1024;
    
    printf("GEMM: [%d, %d] x [%d, %d] + bias + ReLU\n", M, K, K, N);
    printf("Approach: GEMM -> Global -> Bias -> Global -> ReLU (3 kernels)\n\n");
    
    // Allocate
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    size_t bytes_bias = N * sizeof(float);
    
    float *d_A, *d_B, *d_C, *d_C_temp, *d_C_biased, *d_bias;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_temp, bytes_C));    // Buffer 1: GEMM output
    CUDA_CHECK(cudaMalloc(&d_C_biased, bytes_C));  // Buffer 2: After bias
    CUDA_CHECK(cudaMalloc(&d_bias, bytes_bias));
    
    // Initialize
    std::vector<float> h_A(M * K), h_B(K * N), h_bias(N);
    for (int i = 0; i < M * K; ++i) h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < N; ++i) h_bias[i] = (rand() % 10) / 10.0f;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bytes_bias, cudaMemcpyHostToDevice));
    
    dim3 gemm_block(BLOCK_SIZE);
    dim3 gemm_grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    
    dim3 elem_block(256);
    dim3 elem_grid(min(4096, (M * N + 255) / 256));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 50;
    
    // Warmup - 3 separate kernels with global memory round-trips
    for (int i = 0; i < warmup; ++i) {
        baseline_gemm_kernel<<<gemm_grid, gemm_block>>>(d_A, d_B, d_C_temp, M, N, K);
        baseline_bias_kernel<<<elem_grid, elem_block>>>(d_C_temp, d_bias, d_C_biased, M, N);
        baseline_relu_kernel<<<elem_grid, elem_block>>>(d_C_biased, d_C, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        baseline_gemm_kernel<<<gemm_grid, gemm_block>>>(d_A, d_B, d_C_temp, M, N, K);
        baseline_bias_kernel<<<elem_grid, elem_block>>>(d_C_temp, d_bias, d_C_biased, M, N);
        baseline_relu_kernel<<<elem_grid, elem_block>>>(d_C_biased, d_C, M, N);
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
    printf("\nNote: Wasteful global memory round-trip between GEMM and epilogue.\n");
    printf("Compare with optimized_warp_spec_pingpong for fused approach.\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_temp));
    CUDA_CHECK(cudaFree(d_C_biased));
    CUDA_CHECK(cudaFree(d_bias));
    
    return 0;
}
