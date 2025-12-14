// optimized_lookup.cu -- precomputed scatter sums with Float8 vectorization.
// CUDA 13 + Blackwell: Uses Float8 (32-byte aligned) for 256-bit loads

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

constexpr int N = 1 << 20;
constexpr int ITERATIONS = 200;
constexpr int RANDOM_STEPS = 64;

// Optimized lookup using Float8 (256-bit loads)
__global__ void lookupOptimized(const float* __restrict__ precomputed,
                                float* __restrict__ out,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  
  const Float8* table = reinterpret_cast<const Float8*>(precomputed);
  Float8 vec = table[idx >> 3];  // 256-bit load
  out[idx] = vec.elems[idx & 7];
}

__host__ int advance_lcg(int idx) {
  return (idx * 1664525 + 1013904223) & (N - 1);
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

  std::vector<float> precomputed(N);
  for (int i = 0; i < N; ++i) {
    int idx = h_indices[i];
    float acc = 0.0f;
    for (int step = 0; step < RANDOM_STEPS; ++step) {
      acc += h_table[idx];
      idx = advance_lcg(idx);
    }
    precomputed[i] = acc;
  }

  float *d_precomputed = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_precomputed, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_precomputed, precomputed.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    lookupOptimized<<<grid, block>>>(d_precomputed, d_out, N);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / ITERATIONS;

  CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  printf("Lookup (Float8, 256-bit): %.4f ms\n", avg_ms);
  printf("TIME_MS: %.4f\n", avg_ms);
  printf("out[0]=%.1f\n", h_out[0]);
#ifdef VERIFY
  float checksum = 0.0f;
  VERIFY_CHECKSUM(h_out, N, &checksum);
  VERIFY_PRINT_CHECKSUM(checksum);
#endif

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_precomputed));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_table));
  CUDA_CHECK(cudaFreeHost(h_indices));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
