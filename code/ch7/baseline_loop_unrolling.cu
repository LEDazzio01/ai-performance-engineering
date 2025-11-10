// baseline_loop_unrolling.cu -- Loop without unrolling (baseline).

#include <cuda_runtime.h>
#include <cstdio>

#include "../common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 20;
constexpr int INNER_ITERS = 512;
constexpr int STRIDE = 73;

__device__ __forceinline__ float compute_contribution(float sample, int iteration) {
  const float weight = 1.0001f + 0.0002f * static_cast<float>(iteration & 7);
  const float phase = static_cast<float>(iteration) * 0.0005f;
  const float s = __sinf(sample * 0.05f + phase);
  const float c = __cosf(sample * 0.025f - phase);
  return fmaf(sample * weight, s * 0.1f + 0.9f, c * 0.05f);
}

__global__ void kernel_no_unroll(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float accumulator = 0.0f;
    const int mask = n - 1;
    int offset = idx;
    #pragma unroll 1
    for (int i = 0; i < INNER_ITERS; ++i) {
      float sample = in[offset];
      accumulator += compute_contribution(sample, i);
      offset = (offset + STRIDE) & mask;
    }
    out[idx] = accumulator;
  }
}

int main() {
  float *h_in, *h_out;
  CUDA_CHECK(cudaMallocHost(&h_in, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));
  for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(64);
  dim3 grid((N + block.x - 1) / block.x);
  
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  const int iterations = 100;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; i++) {
    kernel_no_unroll<<<grid, block>>>(d_in, d_out, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  float avg_ms = ms / iterations;
  printf("Loop without unroll (baseline): %.3f ms\n", avg_ms);
  printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
