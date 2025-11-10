// optimized_copy_coalesced.cu -- vectorized streaming copy baseline for Chapter 7.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../common/headers/cuda_helpers.cuh"

constexpr int N = 1 << 23;
constexpr int REPEAT = 40;
constexpr int BLOCK_THREADS = 256;
constexpr int VECTORS_PER_THREAD = 4;  // register blocking

__global__ void coalesced_stream(const float4* __restrict__ src,
                                 float4* __restrict__ dst,
                                 int n_vec4) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (int base = tid; base < n_vec4; base += stride * VECTORS_PER_THREAD) {
    float4 lane_buffer[VECTORS_PER_THREAD];
#pragma unroll
    for (int item = 0; item < VECTORS_PER_THREAD; ++item) {
      const int idx = base + item * stride;
      if (idx < n_vec4) {
        lane_buffer[item] = __ldg(src + idx);
      }
    }
#pragma unroll
    for (int item = 0; item < VECTORS_PER_THREAD; ++item) {
      const int idx = base + item * stride;
      if (idx < n_vec4) {
        dst[idx] = lane_buffer[item];
      }
    }
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
  static_assert(N % 4 == 0, "Vectorized copy requires divisible-by-4 length");
  const int n_vec4 = N / 4;

  std::vector<float> h_src(N), h_dst(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    h_src[i] = static_cast<float>((i % 4096) - 2048) / 512.0f;
  }

  float *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(BLOCK_THREADS);
  dim3 grid((n_vec4 + block.x * VECTORS_PER_THREAD - 1) / (block.x * VECTORS_PER_THREAD));

  coalesced_stream<<<grid, block>>>(
      reinterpret_cast<const float4*>(d_src),
      reinterpret_cast<float4*>(d_dst),
      n_vec4);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < REPEAT; ++iter) {
    coalesced_stream<<<grid, block>>>(
        reinterpret_cast<const float4*>(d_src),
        reinterpret_cast<float4*>(d_dst),
        n_vec4);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(REPEAT);
  std::printf("Vectorized coalesced copy (optimized): %.3f ms\n", avg_ms);
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
