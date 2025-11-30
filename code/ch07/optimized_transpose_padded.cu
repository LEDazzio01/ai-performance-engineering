// optimized_transpose_padded.cu -- tiled transpose with shared-memory padding.
// CUDA 13 Update: Note on vector types - transpose operations benefit from float4
// Float8 could be used but requires careful alignment handling for transpose patterns

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"

// CUDA 13 + Blackwell: 32-byte aligned type for 256-bit loads (available but not used in transpose)
// Transpose operations use float4 for optimal shared memory bank conflict avoidance
struct alignas(32) Float8 {
    float elems[8];
};
static_assert(sizeof(Float8) == 32, "Float8 must be 32 bytes");
static_assert(alignof(Float8) == 32, "Float8 must be 32-byte aligned");

constexpr int WIDTH = 4096;
constexpr int TILE_DIM = 64;
constexpr int ELEMENTS_PER_THREAD = 4;  // Using float4 for optimal transpose pattern
constexpr int BLOCK_ROWS = 16;
constexpr int BLOCK_COLS = TILE_DIM / ELEMENTS_PER_THREAD;

static_assert(TILE_DIM % BLOCK_ROWS == 0, "Tile rows must be divisible by block rows");
static_assert(TILE_DIM % ELEMENTS_PER_THREAD == 0, "Tile dim must be divisible by vector length");

__global__ void transpose_padded(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int width) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  const int thread_col = threadIdx.x * ELEMENTS_PER_THREAD;
  const int thread_row = threadIdx.y;

  const int block_col = blockIdx.x * TILE_DIM;
  const int block_row = blockIdx.y * TILE_DIM;

#pragma unroll
  for (int row_offset = 0; row_offset < TILE_DIM; row_offset += BLOCK_ROWS) {
    const int global_row = block_row + thread_row + row_offset;
    if (global_row >= width) {
      continue;
    }
    const int global_col = block_col + thread_col;
    float* shared_dst = &tile[thread_row + row_offset][thread_col];
    if (global_col + ELEMENTS_PER_THREAD <= width) {
      const float4 loaded = *reinterpret_cast<const float4*>(
          in + static_cast<size_t>(global_row) * width + global_col);
      shared_dst[0] = loaded.x;
      shared_dst[1] = loaded.y;
      shared_dst[2] = loaded.z;
      shared_dst[3] = loaded.w;
    } else {
#pragma unroll
      for (int elem = 0; elem < ELEMENTS_PER_THREAD; ++elem) {
        const int col = global_col + elem;
        shared_dst[elem] =
            (col < width) ? in[static_cast<size_t>(global_row) * width + col] : 0.0f;
      }
    }
  }
  __syncthreads();

  const int trans_block_col = blockIdx.y * TILE_DIM;
  const int trans_block_row = blockIdx.x * TILE_DIM;

#pragma unroll
  for (int row_offset = 0; row_offset < TILE_DIM; row_offset += BLOCK_ROWS) {
    const int global_row = trans_block_row + thread_row + row_offset;
    if (global_row >= width) {
      continue;
    }
    const int global_col = trans_block_col + thread_col;
#pragma unroll
    for (int elem = 0; elem < ELEMENTS_PER_THREAD; ++elem) {
      const int col = global_col + elem;
      if (col < width) {
        out[static_cast<size_t>(global_row) * width + col] =
            tile[thread_col + elem][thread_row + row_offset];
      }
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
    h_in[i] = static_cast<float>((i % 2048) - 1024) / 256.0f;
  }
  std::vector<float> h_out(WIDTH * WIDTH, 0.0f);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_COLS, BLOCK_ROWS);
  dim3 grid((WIDTH + TILE_DIM - 1) / TILE_DIM, (WIDTH + TILE_DIM - 1) / TILE_DIM);

  transpose_padded<<<grid, block>>>(d_in, d_out, WIDTH);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int kIterations = 400;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kIterations; ++i) {
    transpose_padded<<<grid, block>>>(d_in, d_out, WIDTH);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / kIterations;
  std::printf("Shared-memory transpose (optimized): %.3f ms\n", avg_ms);

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  std::printf("Output checksum: %.6f\n", checksum(h_out));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
