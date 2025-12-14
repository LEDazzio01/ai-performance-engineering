// optimized_matmul_tiled.cu -- tiled matmul example (Chapter 7 optimized version).

#include <cuda_runtime.h>
#include <cstdio>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

constexpr int N = 1024;
constexpr int TILE = 32;
constexpr int kIterations = 20;

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;
  for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
    int tiled_col = t * TILE + threadIdx.x;
    int tiled_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < n && tiled_col < n) ? A[row * n + tiled_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (tiled_row < n && col < n) ? B[tiled_row * n + col] : 0.0f;
    __syncthreads();

    for (int i = 0; i < TILE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < n && col < n) {
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

  for (size_t i = 0; i < elements; ++i) h_A[i] = 1.0f;
  for (size_t i = 0; i < elements; ++i) h_B[i] = 1.0f;

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  // Warmup
  matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < kIterations; ++iter) {
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_LAST_ERROR();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / kIterations;

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
  printf("Tiled resident matmul (optimized): %.3f ms\n", avg_ms);
  printf("TIME_MS: %.6f\n", avg_ms);
  printf("C[0]=%.1f\n", h_C[0]);
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
