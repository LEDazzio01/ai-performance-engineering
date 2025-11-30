// baseline_memory_access.cu -- Two-pass permuted copy with uncoalesced traffic.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 24;               // 64 MB footprint
constexpr int REPEAT = 50;               // match harness iterations
constexpr int PERM_STRIDE = 97;          // odd multiplier => invertible permutation
constexpr int BLOCK_SIZE = 128;
constexpr int GRID_DIVISOR = 4;          // intentionally limit parallelism

__device__ __forceinline__ int permute_index(int idx, int mask) {
  return (idx * PERM_STRIDE + 131) & mask;
}

__global__ void scatter_permuted(const float* __restrict__ src,
                                 float* __restrict__ staging,
                                 int n,
                                 int mask) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = global_tid; i < n; i += stride) {
    const int permuted = permute_index(i, mask);
    staging[permuted] = src[i];  // sequential read, scattered write
  }
}

__global__ void gather_permuted(const float* __restrict__ staging,
                                float* __restrict__ dst,
                                int n,
                                int mask) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = global_tid; i < n; i += stride) {
    const int permuted = permute_index(i, mask);
    dst[i] = staging[permuted];  // scattered read, sequential write
  }
}

float checksum(const std::vector<float>& data) {
  double acc = 0.0;
  for (float v : data) acc += static_cast<double>(v);
  return static_cast<float>(acc / static_cast<double>(data.size()));
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float max_err = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    max_err = fmaxf(max_err, fabsf(a[i] - b[i]));
  }
  return max_err;
}

int main() {
  std::vector<float> h_src(N), h_dst(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    h_src[i] = static_cast<float>((i % 2048) - 1024) / 256.0f;
  }

  float *d_src = nullptr, *d_tmp = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE);
  dim3 grid((N + block.x * GRID_DIVISOR - 1) / (block.x * GRID_DIVISOR));
  const int mask = N - 1;

  // Warmup: execute the two-pass pipeline once to populate caches.
  scatter_permuted<<<grid, block>>>(d_src, d_tmp, N, mask);
  gather_permuted<<<grid, block>>>(d_tmp, d_dst, N, mask);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < REPEAT; ++iter) {
    scatter_permuted<<<grid, block>>>(d_src, d_tmp, N, mask);
    gather_permuted<<<grid, block>>>(d_tmp, d_dst, N, mask);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(REPEAT);
  std::printf("Permuted two-pass copy (baseline): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Output checksum: %.6f\n", checksum(h_dst));
  std::printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_src, h_dst));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_dst));
  return 0;
}
