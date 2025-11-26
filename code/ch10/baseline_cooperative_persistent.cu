// baseline_cooperative_persistent.cu
// Multi-launch pipeline that processes large batches with repeated global-memory passes.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../common/headers/cuda_helpers.cuh"

constexpr int ELEMENTS = 1 << 24;          // 16M elements (~64 MB)
constexpr int ITERATIONS = 40;
constexpr int THREADS_PER_BLOCK = 256;

__global__ void scale_kernel(float* data, int n, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = __fmaf_rn(data[idx], scale, 0.0f);
  }
}

__global__ void bias_kernel(float* data, int n, float bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += bias;
  }
}

__global__ void activation_kernel(float* data, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float x = data[idx];
    data[idx] = tanhf(x);
  }
}

__global__ void residual_kernel(float* data, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float x = data[idx];
    data[idx] = x + 0.01f * __sinf(x);
  }
}

__global__ void exp_kernel(float* data, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = __expf(data[idx]) - 1.0f;
  }
}

double checksum(const std::vector<float>& data) {
  double acc = 0.0;
  for (float v : data) acc += static_cast<double>(v);
  return acc / static_cast<double>(data.size());
}

int main() {
  std::vector<float> h_data(ELEMENTS);
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_data) v = dist(rng);

  float* d_data = nullptr;
  const size_t bytes = static_cast<size_t>(ELEMENTS) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid((ELEMENTS + block.x - 1) / block.x);

  // Warmup one full pipeline to prime caches.
  scale_kernel<<<grid, block>>>(d_data, ELEMENTS, 1.001f);
  bias_kernel<<<grid, block>>>(d_data, ELEMENTS, 0.05f);
  activation_kernel<<<grid, block>>>(d_data, ELEMENTS);
  residual_kernel<<<grid, block>>>(d_data, ELEMENTS);
  exp_kernel<<<grid, block>>>(d_data, ELEMENTS);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    scale_kernel<<<grid, block>>>(d_data, ELEMENTS, 1.001f);
    bias_kernel<<<grid, block>>>(d_data, ELEMENTS, 0.05f);
    activation_kernel<<<grid, block>>>(d_data, ELEMENTS);
    residual_kernel<<<grid, block>>>(d_data, ELEMENTS);
    exp_kernel<<<grid, block>>>(d_data, ELEMENTS);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(ITERATIONS);

  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
  const double chk = checksum(h_data);

  std::printf("Baseline cooperative pipeline: %.3f ms (%d iterations)\n", avg_ms, ITERATIONS);
  std::printf("TIME_MS: %.6f\n", avg_ms);
  std::printf("Checksum: %.6f\n", chk);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
