// baseline_cutlass_gemm_fp16.cu -- Naive FP16 GEMM baseline for tensor core comparison
//
// This baseline uses FP16 data types to provide an apples-to-apples comparison
// with the optimized FP16 cuBLASLt version that uses tensor cores.
//
// BOOK REFERENCE (Ch9): Tensor cores provide massive throughput improvements
// for matrix operations with FP16/BF16 precision. This baseline shows the
// performance of a naive tiled FP16 GEMM without tensor core acceleration.

#include <cuda_fp16.h>
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

// Naive tiled FP16 GEMM kernel (no tensor cores)
// Uses shared memory tiling but standard FP16 arithmetic
template<int TILE_SIZE = 32>
__global__ void tiled_fp16_gemm_kernel(const __half* __restrict__ A,
                                        const __half* __restrict__ B,
                                        __half* __restrict__ C,
                                        int M,
                                        int N,
                                        int K,
                                        __half alpha,
                                        __half beta) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;  // Accumulate in FP32 for numerical stability
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        // Load tile of B into shared memory
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        float alpha_f = __half2float(alpha);
        float beta_f = __half2float(beta);
        float c_val = __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(alpha_f * sum + beta_f * c_val);
    }
}

int main() {
    // Match optimized version's matrix sizes
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    constexpr int kIterations = 10;
    constexpr int kBatchCount = 64;

    const size_t elements_A = static_cast<size_t>(M) * K;
    const size_t elements_B = static_cast<size_t>(K) * N;
    const size_t elements_C = static_cast<size_t>(M) * N;
    const size_t size_A = elements_A * sizeof(__half) * kBatchCount;
    const size_t size_B = elements_B * sizeof(__half) * kBatchCount;
    const size_t size_C = elements_C * sizeof(__half) * kBatchCount;

    // Host allocation with pinned memory
    __half* h_A = nullptr;
    __half* h_B = nullptr;
    __half* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, size_C));

    // Initialize with random FP16 values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (size_t i = 0; i < elements_A * kBatchCount; ++i) {
        h_A[i] = __float2half(dis(gen));
    }
    for (size_t i = 0; i < elements_B * kBatchCount; ++i) {
        h_B[i] = __float2half(dis(gen));
    }
    for (size_t i = 0; i < elements_C * kBatchCount; ++i) {
        h_C[i] = __float2half(0.0f);
    }

    // Device allocation
    __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Pre-load all data before timing
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

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // Warmup
    for (int batch = 0; batch < kBatchCount; ++batch) {
        const size_t offset_A = batch * elements_A;
        const size_t offset_B = batch * elements_B;
        const size_t offset_C = batch * elements_C;
        tiled_fp16_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
            d_A + offset_A, d_B + offset_B, d_C + offset_C, M, N, K, alpha, beta);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Timed section: Kernel execution only
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < kIterations; ++iter) {
        for (int batch = 0; batch < kBatchCount; ++batch) {
            const size_t offset_A = batch * elements_A;
            const size_t offset_B = batch * elements_B;
            const size_t offset_C = batch * elements_C;
            tiled_fp16_gemm_kernel<TILE_SIZE><<<grid_size, block_size, 0, stream>>>(
                d_A + offset_A, d_B + offset_B, d_C + offset_C, M, N, K, alpha, beta);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / static_cast<float>(kIterations * kBatchCount);
    
    // Calculate TFLOPS
    const double flops = 2.0 * M * N * K * kBatchCount * kIterations;
    const double tflops = flops / (total_ms * 1e9);
    
    std::cout << "Naive Tiled FP16 GEMM (baseline): " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    std::cout << "Checksum sample: " << __half2float(h_C[0]) << std::endl;

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






