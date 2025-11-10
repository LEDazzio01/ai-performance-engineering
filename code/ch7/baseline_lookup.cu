// naive_lookup.cu -- naive scattered memory access example.

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 20;

constexpr int ITERATIONS = 200;
constexpr int RANDOM_STEPS = 64;

__device__ __forceinline__ int advance_lcg(int idx) {
  return (idx * 1664525 + 1013904223) & (N - 1);
}

__global__ void lookupNaive(const float* table, const int* indices, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = 0.0f;
    int index = indices[idx];
    #pragma unroll 8
    for (int iter = 0; iter < RANDOM_STEPS; ++iter) {
      val += __ldg(table + index);
      index = advance_lcg(index);
    }
    out[idx] = val;
  }
}

int main() {
  float *h_table, *h_out;
  int *h_indices;
  CUDA_CHECK(cudaMallocHost(&h_table, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_indices, N * sizeof(int)));

  for (int i = 0; i < N; ++i) {
    h_table[i] = static_cast<float>(i);
    h_indices[i] = (i * 3) % N;
  }

  float *d_table, *d_out;
  int *d_indices;
  CUDA_CHECK(cudaMalloc(&d_table, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_table, h_table, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    lookupNaive<<<grid, block>>>(d_table, d_indices, d_out, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / ITERATIONS;

  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("out[0]=%.1f\n", h_out[0]);
  printf("TIME_MS: %.4f\n", avg_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_table));
  CUDA_CHECK(cudaFreeHost(h_indices));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
