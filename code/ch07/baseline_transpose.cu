// baseline_transpose.cu -- shared-memory transpose with bank conflicts (Chapter 7 baseline).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

constexpr int WIDTH = 4096;
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int REPEAT = 200;

__global__ void transpose_naive(const float* __restrict__ idata, float* __restrict__ odata, int width) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && (y + j) < width) {
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && (y + j) < width) {
      odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

float checksum(const std::vector<float>& data) {
  double acc = 0.0;
  for (float v : data) acc += static_cast<double>(v);
  return static_cast<float>(acc / static_cast<double>(data.size()));
}

int main() {
  const size_t bytes = static_cast<size_t>(WIDTH) * WIDTH * sizeof(float);

  std::vector<float> h_in(static_cast<size_t>(WIDTH) * WIDTH);
  std::vector<float> h_out(static_cast<size_t>(WIDTH) * WIDTH, 0.0f);

  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = static_cast<float>((i % 1024) - 512) / 128.0f;
  }

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_DIM, BLOCK_ROWS);
  dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM, (WIDTH + TILE_DIM - 1) / TILE_DIM);

  // Warmup.
  transpose_naive<<<grid, block>>>(d_in, d_out, WIDTH);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < REPEAT; ++i) {
    transpose_naive<<<grid, block>>>(d_in, d_out, WIDTH);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(REPEAT);
  std::printf("Transpose (bank conflicts, baseline): %.3f ms\n", avg_ms);
  std::printf("TIME_MS: %.6f\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  std::printf("Output checksum: %.6f\n", checksum(h_out));
#ifdef VERIFY
  float verify_checksum = 0.0f;
  VERIFY_CHECKSUM(h_out.data(), static_cast<int>(h_out.size()), &verify_checksum);
  VERIFY_PRINT_CHECKSUM(verify_checksum);
#endif

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
