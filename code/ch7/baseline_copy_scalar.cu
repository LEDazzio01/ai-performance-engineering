// scalar_copy.cu -- naive global load example for Chapter 7.

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 20;
constexpr int RANDOM_PASSES = 64;
constexpr int STRIDE = 97;

__device__ __forceinline__ int advance_index(int idx) {
  return (idx * STRIDE + 0x1f123bb) & (N - 1);
}

__global__ void copyScalarSlow(const float* in, float* out, int n) {
  extern __shared__ float staging[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  float accumulator = in[tid];
  int gather_idx = tid;
  #pragma unroll 8
  for (int pass = 0; pass < RANDOM_PASSES; ++pass) {
    gather_idx = advance_index(gather_idx);
    const float sample = __ldg(in + gather_idx);
    staging[threadIdx.x] = sample;
    __syncthreads();
    const int neighbor = (threadIdx.x + pass) & (blockDim.x - 1);
    accumulator = staging[neighbor] * 0.999f + accumulator * 0.001f;
    __syncthreads();
  }

  out[tid] = accumulator;
}

int main() {
  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(64);
  dim3 grid((N + block.x - 1) / block.x / 8);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < 50; ++iter) {
    copyScalarSlow<<<grid, block, block.x * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const float avg_ms = ms / 50.0f;

  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("out[0]=%.1f\n", h_out[0]);
  printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
