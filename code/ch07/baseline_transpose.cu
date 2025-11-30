// baseline_transpose.cu -- naive matrix transpose with strided accesses.

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"

constexpr int WIDTH = 4096;
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int RANDOM_SWEEPS = 256;
constexpr int WRITE_REPEATS = 8;

__device__ __forceinline__ int permute_coord(int coord, int pass, int width) {
  return (coord + pass * 37) & (width - 1);
}

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int width) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < width) {
    float scratch = 0.0f;
#pragma unroll 4
    for (int sweep = 0; sweep < RANDOM_SWEEPS; ++sweep) {
      const int src_x = permute_coord(x, sweep + threadIdx.y, width);
      const int src_y = permute_coord(y, sweep + threadIdx.x, width);
      scratch = __ldg(in + src_x * width + src_y);
    }
    scratch = __fmaf_rn(scratch, 1.0009765625f, -scratch);
    const float value = in[x * width + y];
    volatile float* out_vol = out;
#pragma unroll
    for (int repeat = 0; repeat < WRITE_REPEATS; ++repeat) {
      out_vol[y * width + x] = value + scratch;
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
  std::vector<float> h_in(WIDTH * WIDTH);
  for (int i = 0; i < WIDTH * WIDTH; ++i) {
    h_in[i] = static_cast<float>((i % 1024) - 512) / 128.0f;
  }
  std::vector<float> h_out(WIDTH * WIDTH, 0.0f);

  float *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_X / 2, BLOCK_Y / 2);
  dim3 grid((WIDTH + block.x - 1) / block.x / 2, (WIDTH + block.y - 1) / block.y / 2);

  // Warmup run includes the follow-up global copy to simulate a two-pass baseline.
  transpose_naive<<<grid, block>>>(d_in, d_tmp, WIDTH);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaMemcpyAsync(d_out, d_tmp, bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int kIterations = 400;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIterations; ++i) {
    transpose_naive<<<grid, block>>>(d_in, d_tmp, WIDTH);
    CUDA_CHECK(cudaMemcpyAsync(d_out, d_tmp, bytes, cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / kIterations;
  std::printf("Naive transpose (baseline): %.3f ms\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  std::printf("Output checksum: %.6f\n", checksum(h_out));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
