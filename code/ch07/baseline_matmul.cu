// baseline_matmul.cu -- naive matrix multiplication (Chapter 7 baseline).

#include <cuda_runtime.h>

#include <cstdio>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

constexpr int N = 1024;
constexpr int kIterations = 20;

__global__ void naive_matmul(const float* A, const float* B, float* C, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float sum = 0.0f;
#pragma unroll 1
    for (int k = 0; k < n; ++k) {
      sum = fmaf(A[row * n + k], B[k * n + col], sum);
    }
    C[row * n + col] = sum;
  }
}

int main() {
  const size_t elements = static_cast<size_t>(N) * N;
  const size_t bytes = elements * sizeof(float);

  float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_A, bytes));
  CUDA_CHECK(cudaMallocHost(&h_B, bytes));
  CUDA_CHECK(cudaMallocHost(&h_C, bytes));

  for (size_t i = 0; i < elements; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 1.0f;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  naive_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < kIterations; ++iter) {
    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_LAST_ERROR();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / kIterations;
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  std::printf("Naive matmul (baseline): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);
  std::printf("C[0]=%.1f\n", h_C[0]);
#ifdef VERIFY
  float checksum = 0.0f;
  VERIFY_CHECKSUM(h_C, static_cast<int>(elements), &checksum);
  VERIFY_PRINT_CHECKSUM(checksum);
#endif

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  return 0;
}
