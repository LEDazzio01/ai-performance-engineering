// baseline_double_buffered_pipeline.cu -- Naive GEMM without shared-memory tiling.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_verify.cuh"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status = (call);                                         \
        if (status != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
                        cudaGetErrorString(status));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    // Standard global memory GEMM - read A and B rows/columns from global memory
    for (int kk = 0; kk < K; ++kk) {
        sum += A[row * K + kk] * B[kk * N + col];
    }
    C[row * N + col] = sum;
}

int main() {
    // Larger matrices to show double-buffering benefit
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 5;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(iterations);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float v : h_C) checksum += v;
    checksum /= static_cast<double>(M * N);

    std::printf("Baseline GEMM (global memory): %.3f ms (avg over %d iters)\n", avg_ms, iterations);
    std::printf("TIME_MS: %.6f\n", avg_ms);
    std::printf("Checksum: %.6f\n", checksum);
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
