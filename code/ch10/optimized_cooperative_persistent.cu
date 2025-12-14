// optimized_cooperative_persistent.cu
// Cooperative persistent kernel that double-buffers tiles in shared memory using cuda::pipeline.

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../core/common/headers/cuda_helpers.cuh"
#include "../core/common/headers/cuda_verify.cuh"

namespace cg = cooperative_groups;

constexpr int ELEMENTS = 1 << 24;
constexpr int ITERATIONS = 40;
constexpr int PIPELINE_STAGES = 2;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_ELEMS = THREADS_PER_BLOCK * ITEMS_PER_THREAD;

__device__ __forceinline__ float fused_transform(float x, float scale, float bias) {
  x = __fmaf_rn(x, scale, bias);
  x = tanhf(x);
  x = x + 0.01f * __sinf(x);
  return __expf(x) - 1.0f;
}

__device__ __forceinline__ int tile_remaining(int elements, int base) {
  int remaining = elements - base;
  remaining = remaining > TILE_ELEMS ? TILE_ELEMS : remaining;
  return remaining > 0 ? remaining : 0;
}

__global__ void persistent_pipeline(float* data, int elements, float scale, float bias) {
  cg::thread_block block = cg::this_thread_block();
  extern __shared__ float shared[];

  float* stage_buffers[PIPELINE_STAGES];
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    stage_buffers[stage] = shared + stage * TILE_ELEMS;
  }

  using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>;
  __shared__ alignas(pipeline_state_t) unsigned char pipe_state_bytes[sizeof(pipeline_state_t)];
  auto* pipe_state = reinterpret_cast<pipeline_state_t*>(pipe_state_bytes);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    new (pipe_state) pipeline_state_t();
  }
  block.sync();
  auto pipe = cuda::make_pipeline(block, pipe_state);

  const int total_tiles = (elements + TILE_ELEMS - 1) / TILE_ELEMS;

  auto prefetch_tile = [&](int tile_id, int stage) {
    const int base = tile_id * TILE_ELEMS;
    const int remaining = tile_remaining(elements, base);
    pipe.producer_acquire();
    if (remaining > 0) {
      cuda::memcpy_async(block,
                         stage_buffers[stage],
                         data + base,
                         static_cast<size_t>(remaining) * sizeof(float),
                         pipe);
    }
    pipe.producer_commit();
  };

  auto process_tile = [&](int tile_id, int stage) {
    const int base = tile_id * TILE_ELEMS;
    const int remaining = tile_remaining(elements, base);
    const int lane_base = threadIdx.x * ITEMS_PER_THREAD;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
      const int local_idx = lane_base + item;
      if (local_idx < remaining) {
        float val = stage_buffers[stage][local_idx];
        val = fused_transform(val, scale, bias);
        data[base + local_idx] = val;
      }
    }
  };

  int next_tile = blockIdx.x;
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    if (next_tile < total_tiles) {
      prefetch_tile(next_tile, stage);
      next_tile += gridDim.x;
    }
  }

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int stage = tile % PIPELINE_STAGES;
    pipe.consumer_wait();
    block.sync();
    process_tile(tile, stage);
    pipe.consumer_release();

    if (next_tile < total_tiles) {
      prefetch_tile(next_tile, stage);
      next_tile += gridDim.x;
    }
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

  int max_blocks = 1;
  {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    max_blocks = prop.multiProcessorCount * 4;
  }
  const int total_tiles = (ELEMENTS + TILE_ELEMS - 1) / TILE_ELEMS;
  int grid_blocks = max_blocks;
  if (grid_blocks > total_tiles) grid_blocks = total_tiles;
  if (grid_blocks < 1) grid_blocks = 1;
  const dim3 grid(grid_blocks);
  const dim3 block(THREADS_PER_BLOCK);
  const size_t shared_bytes = PIPELINE_STAGES * TILE_ELEMS * sizeof(float);

  // Warmup one launch.
  persistent_pipeline<<<grid, block, shared_bytes>>>(
      d_data, ELEMENTS, 1.001f, 0.05f);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    persistent_pipeline<<<grid, block, shared_bytes>>>(
        d_data, ELEMENTS, 1.001f, 0.05f);
  }
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const float avg_ms = total_ms / static_cast<float>(ITERATIONS);

  CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
  const double chk = checksum(h_data);

  std::printf("Optimized cooperative pipeline: %.3f ms (%d iterations)\n", avg_ms, ITERATIONS);
  std::printf("TIME_MS: %.6f\n", avg_ms);
  std::printf("Checksum: %.6f\n", chk);
  VERIFY_PRINT_CHECKSUM(static_cast<float>(chk));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}
