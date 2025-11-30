// naive_matmul.cu -- naive matrix multiplication (Chapter 7 baseline).

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;
constexpr int kIterations = 6;
constexpr int kMicroBatches = 16;

__global__ void matmul_naive(const float* A,
                             const float* B,
                             float* C,
                             int m,
                             int n,
                             int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    const volatile float* A_volatile = reinterpret_cast<const volatile float*>(A);
    const volatile float* B_volatile = reinterpret_cast<const volatile float*>(B);
    float sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < k; ++i) {
      // Volatile loads force replay of global memory traffic for each multiply.
      float a = A_volatile[row * k + i];
      float b = B_volatile[i * n + col];
      sum = fmaf(a, b, sum);
    }
    C[row * n + col] = sum;
  }
}

int main() {
  const size_t elementsA = static_cast<size_t>(M) * K;
  const size_t elementsB = static_cast<size_t>(K) * N;
  const size_t elementsC = static_cast<size_t>(M) * N;
  const size_t bytesA = elementsA * sizeof(float);
  const size_t bytesB = elementsB * sizeof(float);
  const size_t bytesC = elementsC * sizeof(float);

  std::vector<float*> host_batches_A(kMicroBatches);
  std::vector<float*> host_batches_B(kMicroBatches);
  float* host_result = nullptr;
  for (int batch = 0; batch < kMicroBatches; ++batch) {
    CUDA_CHECK(cudaMallocHost(&host_batches_A[batch], bytesA));
    CUDA_CHECK(cudaMallocHost(&host_batches_B[batch], bytesB));
    for (size_t i = 0; i < elementsA; ++i) {
      host_batches_A[batch][i] = static_cast<float>((batch + i) % 37) * 0.001f;
    }
    for (size_t i = 0; i < elementsB; ++i) {
      host_batches_B[batch][i] = static_cast<float>((batch * 3 + i) % 53) * 0.002f;
    }
  }
  CUDA_CHECK(cudaMallocHost(&host_result, bytesC));

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytesA));
  CUDA_CHECK(cudaMalloc(&d_B, bytesB));
  CUDA_CHECK(cudaMalloc(&d_C, bytesC));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  dim3 block(16, 8);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  // Warmup one micro-batch to populate caches.
  CUDA_CHECK(cudaMemcpyAsync(d_A, host_batches_A[0], bytesA, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, host_batches_B[0], bytesB, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(d_C, 0, bytesC, stream));
  matmul_naive<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaMemcpyAsync(host_result, d_C, bytesC, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iter = 0; iter < kIterations; ++iter) {
    for (int micro = 0; micro < kMicroBatches; ++micro) {
      const int batch_idx = (iter + micro) % kMicroBatches;
      CUDA_CHECK(cudaMemcpyAsync(d_A, host_batches_A[batch_idx], bytesA, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_B, host_batches_B[(batch_idx * 7 + micro) % kMicroBatches],
                                 bytesB,
                                 cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemsetAsync(d_C, 0, bytesC, stream));
      matmul_naive<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
      CUDA_CHECK_LAST_ERROR();
      CUDA_CHECK(cudaMemcpyAsync(host_result, d_C, bytesC, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / (kIterations * kMicroBatches);
  std::printf("Naive batched matmul (baseline): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);
  std::printf("Sample checksum: %.6f\n", host_result[0]);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  for (int batch = 0; batch < kMicroBatches; ++batch) {
    CUDA_CHECK(cudaFreeHost(host_batches_A[batch]));
    CUDA_CHECK(cudaFreeHost(host_batches_B[batch]));
  }
  CUDA_CHECK(cudaFreeHost(host_result));
  return 0;
}
