// baseline_warp_specialized_cluster_pipeline.cu - Naive GEMM (No Tiling) (Ch10)
//
// WHAT: Simple GEMM without shared memory tiling.
// Each thread reads directly from global memory.
//
// WHY THIS IS SLOWER:
//   - No data reuse - global memory read for every multiply
//   - Poor cache utilization
//   - High memory bandwidth requirements
//
// COMPARE WITH: optimized_warp_specialized_cluster_pipeline.cu
//   - Uses shared memory tiling for data reuse
//   - Much lower global memory traffic

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr int BLOCK_SIZE = 16;  // 16x16 thread block

//============================================================================
// Baseline: Naive GEMM without shared memory
// Every thread reads directly from global memory
//============================================================================

__global__ void naive_gemm_no_tiling(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [K, N]
    float* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        // Each thread reads K values from A and B from global memory
        // No data reuse between threads
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Baseline Naive GEMM (No Shared Memory Tiling)\n");
    printf("=============================================\n");
    printf("Device: %s\n\n", prop.name);
    
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    printf("GEMM: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Approach: Direct global memory access (no tiling)\n\n");
    
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    
    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = 0.01f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 0.01f;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    
    for (int i = 0; i < warmup; ++i) {
        naive_gemm_no_tiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        naive_gemm_no_tiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\nNote: No shared memory = poor data reuse.\n");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}
