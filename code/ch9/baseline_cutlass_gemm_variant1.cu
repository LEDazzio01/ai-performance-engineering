// baseline_cutlass_gemm_variant1.cu -- Fair comparison baseline: Pre-loaded data
//
// VARIANT 1: Pre-load data before timing to match optimized version
// This ensures we're comparing kernel performance, not data transfer
//
// BOOK REFERENCE (Ch9): CUTLASS and cuBLASLt optimizations focus on tensor
// core utilization, tiling strategies, and memory access patterns. A fair
// comparison should isolate kernel execution time from data movement.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)
#define CUDA_CHECK_LAST_ERROR()                                                  \
  do {                                                                           \
    cudaError_t status = cudaGetLastError();                                     \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " "           \
                << cudaGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

// Naive GEMM kernel with shared memory tiling for fairer comparison
// Still naive but uses tiling to show the benefit of cuBLASLt optimizations
template<int TILE_SIZE = 32>
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;
    constexpr int kIterations = 5;
    constexpr int kBatchCount = 32;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A = elements_A * sizeof(float) * kBatchCount;
    const size_t size_B = elements_B * sizeof(float) * kBatchCount;
    const size_t size_C = elements_C * sizeof(float) * kBatchCount;

    // Host allocation with pinned memory
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    // Initialize
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < elements_A * kBatchCount; ++i) {
        h_A[i] = dis(gen);
    }
    for (size_t i = 0; i < elements_B * kBatchCount; ++i) {
        h_B[i] = dis(gen);
    }
    std::fill(h_C, h_C + elements_C * kBatchCount, 0.0f);

    // Device allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // PRE-LOAD ALL DATA BEFORE TIMING (matches optimized version)
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    constexpr int TILE_SIZE = 32;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    for (int batch = 0; batch < kBatchCount; ++batch) {
        const size_t offset_A = batch * elements_A;
        const size_t offset_B = batch * elements_B;
        const size_t offset_C = batch * elements_C;
        tiled_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_A + offset_A, d_B + offset_B, d_C + offset_C, M, N, K, 1.0f, 0.0f);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // TIMED SECTION: Kernel execution only (no data transfer)
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A;
            const size_t offset_B = batch * elements_B;
            const size_t offset_C = batch * elements_C;
            tiled_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
                d_A + offset_A, d_B + offset_B, d_C + offset_C, M, N, K, 1.0f, 0.0f);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_total, start, stop));
    const float time_avg = time_total / static_cast<float>(kIterations * kBatchCount);
    std::cout << "Tiled GEMM (variant1 - fair baseline): " << time_avg << " ms" << std::endl;
    
    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "Checksum sample: " << h_C[0] << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}


