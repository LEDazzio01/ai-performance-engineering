// optimized_warp_specialized_cluster_pipeline.cu - Tiled GEMM (Shared Memory) (Ch10)
//
// WHAT: GEMM with shared memory tiling for data reuse.
// Tiles loaded to shared memory and reused by thread block.
//
// WHY THIS IS FASTER:
//   - Data loaded to shared memory once, used many times
//   - Reduced global memory traffic by TILE_SIZE factor
//   - Better cache utilization
//
// COMPARE WITH: baseline_warp_specialized_cluster_pipeline.cu
//   - Baseline reads from global memory for every operation
//   - No data reuse = wasteful bandwidth

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

constexpr int TILE_SIZE = 16;

//============================================================================
// Optimized: Tiled GEMM with shared memory
// Data reused TILE_SIZE times per load
//============================================================================

__global__ void tiled_gemm_shared_memory(
    const float* __restrict__ A,   // [M, K]
    const float* __restrict__ B,   // [K, N]
    float* __restrict__ C,         // [M, N]
    int M, int N, int K
) {
    __shared__ float A_smem[TILE_SIZE][TILE_SIZE];
    __shared__ float B_smem[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        // Collaborative load of A tile
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        A_smem[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K) ? 
            A[a_row * K + a_col] : 0.0f;
        
        // Collaborative load of B tile
        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;
        B_smem[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N) ? 
            B[b_row * N + b_col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial product using shared memory (reused TILE_SIZE times)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_smem[threadIdx.y][k] * B_smem[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

//============================================================================
// Benchmark
//============================================================================

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Optimized Tiled GEMM (Shared Memory)\n");
    printf("====================================\n");
    printf("Device: %s\n\n", prop.name);
    
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    printf("GEMM: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);
    printf("Tile: %dx%d, Approach: Shared memory tiling\n\n", TILE_SIZE, TILE_SIZE);
    
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
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    const int warmup = 5;
    const int iterations = 20;
    
    for (int i = 0; i < warmup; ++i) {
        tiled_gemm_shared_memory<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        tiled_gemm_shared_memory<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("\nNote: Shared memory tiling = %dx data reuse.\n", TILE_SIZE);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}
