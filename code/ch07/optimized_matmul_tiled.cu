// tiled_matmul.cu -- simple tiled matmul example (Chapter 7 optimized version).

#include <cuda_runtime.h>
#include <cstdio>

#include "../core/common/headers/cuda_helpers.cuh"

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;
constexpr int TILE = 32;
constexpr int kIterations = 80;

__global__ void matmul_tiled(const float* A, const float* B, float* C, int m, int n, int k) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;
  for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
    int tiled_col = t * TILE + threadIdx.x;
    int tiled_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (tiled_row < k && col < n) ? B[tiled_row * n + col] : 0.0f;
    __syncthreads();

    for (int i = 0; i < TILE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * n + col] = sum;
  }
}

int main() {
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  float *h_A, *h_B, *h_C;
  CUDA_CHECK(cudaMallocHost(&h_A, bytesA));
  CUDA_CHECK(cudaMallocHost(&h_B, bytesB));
  CUDA_CHECK(cudaMallocHost(&h_C, bytesC));

  for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
  for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytesA));
  CUDA_CHECK(cudaMalloc(&d_B, bytesB));
  CUDA_CHECK(cudaMalloc(&d_C, bytesC));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  // Warmup
  matmul_tiled<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iter = 0; iter < kIterations; ++iter) {
    matmul_tiled<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST_ERROR();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / kIterations;

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
  printf("Tiled resident matmul (optimized): %.3f ms\n", avg_ms);
  printf("TIME_MS: %.6f\n", avg_ms);
  printf("C[0]=%.1f\n", h_C[0]);

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
