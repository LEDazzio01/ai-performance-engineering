// baseline_work_queue.cu -- static work distribution (baseline).

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "work_queue_common.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void compute_static(const float* input,
                               const int* workloads,
                               float* output,
                               int n,
                               int chunk_size) {
  const int block_start = blockIdx.x * chunk_size;
  const int block_end = min(block_start + chunk_size, n);

  for (int idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
    const int work = workloads[idx];
    float s, c;
    __sincosf(input[idx], &s, &c);
    float sum = 0.0f;
#pragma unroll 1
    for (int iter = 0; iter < work; ++iter) {
      sum = fmaf(s, c, sum + 1e-4f);
      s = s * 0.99991f + 0.00010f * c;
      c = c * 0.99973f - 0.00021f * s;
    }
    output[idx] = sum;
  }
}

int main() {
  constexpr int N = 1 << 20;
  constexpr int kWarmup = 1;
  constexpr int kIters = 10;
  constexpr int blocks = 32;
  constexpr int threads = 256;
  std::vector<float> h_in(N);
  for (int i = 0; i < N; ++i) h_in[i] = float(i) / N;
  std::vector<int> h_work = build_workloads(N);
  float *d_in = nullptr, *d_out = nullptr;
  int* d_work = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_work, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_work, h_work.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  const int chunk_size = (N + blocks - 1) / blocks;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < kWarmup; ++i) {
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    compute_static<<<blocks, threads>>>(d_in, d_work, d_out, N, chunk_size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < kIters; ++iter) {
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    compute_static<<<blocks, threads>>>(d_in, d_work, d_out, N, chunk_size);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float static_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&static_ms, start, stop));

  std::printf("Work queue baseline (static): %.3f ms\n", static_ms / kIters);

  CUDA_CHECK(cudaMemcpy(h_in.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  double checksum = 0.0;
  for (float v : h_in) checksum += static_cast<double>(v);
  std::printf("Baseline checksum: %.6e\n", checksum);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}
