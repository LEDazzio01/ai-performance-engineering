// baseline_copy_uncoalesced.cu -- multi-pass scattered copy baseline for Chapter 7.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 23;           // 32 MB footprint
constexpr int RANDOM_PASSES = 64;
constexpr int REPEAT = 40;
constexpr int BLOCK_SIZE = 128;
constexpr int GRID_DIVISOR = 16;     // throttle occupancy to exaggerate stalls

__device__ __forceinline__ int permute_index(int idx, int mask) {
  return (idx * 1315423911 + 2654435761) & mask;
}

__global__ void scattered_copy(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n,
                               int mask) {
  __shared__ float staging[BLOCK_SIZE];
  const int lane = threadIdx.x & (BLOCK_SIZE - 1);
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int idx = global_tid; idx < n; idx += stride) {
    int gather_idx = permute_index(idx, mask);
    float accum = 0.0f;
#pragma unroll 4
    for (int pass = 0; pass < RANDOM_PASSES; ++pass) {
      gather_idx = permute_index(gather_idx + pass * 17, mask);
      const float sample = __ldg(in + gather_idx);
      staging[(lane + pass) & (BLOCK_SIZE - 1)] = sample;
      __syncthreads();
      const int neighbor = (lane + pass * 3) & (BLOCK_SIZE - 1);
      accum = __fmaf_rn(staging[neighbor], 0.9985f, accum * 0.0015f + 1e-6f);
      __syncthreads();
    }
    out[idx] = accum;
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
    h_src[i] = static_cast<float>((i % 4096) - 2048) / 512.0f;
  }

  float *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE);
  dim3 grid((N + block.x * GRID_DIVISOR - 1) / (block.x * GRID_DIVISOR));
  const int mask = N - 1;

  // Warmup to stabilize caches and residency.
  scattered_copy<<<grid, block>>>(d_src, d_dst, N, mask);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < REPEAT; ++iter) {
    scattered_copy<<<grid, block>>>(d_src, d_dst, N, mask);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(REPEAT);
  std::printf("Uncoalesced scatter-gather (baseline): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Output checksum: %.6f\n", checksum(h_dst));
  std::printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_src, h_dst));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  return 0;
}
