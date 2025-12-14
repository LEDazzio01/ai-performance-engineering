// baseline_memory_access.cu -- scalar coalesced copy baseline (Chapter 7).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

constexpr int N = 1 << 24;  // 64 MB footprint (float32)
constexpr int REPEAT = 50;
constexpr int BLOCK_THREADS = 256;

__global__ void scalar_copy(const float* __restrict__ src, float* __restrict__ dst, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    dst[i] = src[i];
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

  float *d_src = nullptr, *d_dst = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(BLOCK_THREADS);
  dim3 grid((N + block.x - 1) / block.x);

  // Warmup.
  scalar_copy<<<grid, block>>>(d_src, d_dst, N);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < REPEAT; ++iter) {
    scalar_copy<<<grid, block>>>(d_src, d_dst, N);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(REPEAT);
  std::printf("Scalar coalesced copy (baseline): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("Output checksum: %.6f\n", checksum(h_dst));
  std::printf("Max abs diff vs src: %.6e\n", max_abs_diff(h_src, h_dst));
#ifdef VERIFY
  float verify_checksum = 0.0f;
  VERIFY_CHECKSUM(h_dst.data(), N, &verify_checksum);
  VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_dst));
  return 0;
}
