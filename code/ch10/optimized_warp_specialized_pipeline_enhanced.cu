// Adaptive warp-specialized pipeline example.
// Chooses tile size at runtime (32, 16, or 8) based on shared-memory limits.

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "../core/common/headers/arch_detection.cuh"

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

template <int TILE>
__device__ void compute_tile(const float* a, const float* b, float* c, int lane) {
  constexpr int TILE_ELEMS = TILE * TILE;
  for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
    float x = a[idx];
    float y = b[idx];
    c[idx] = sqrtf(x * x + y * y);
  }
}

template <int TILE>
__global__ void warp_specialized_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int total_tiles) {
  constexpr int TILE_ELEMS = TILE * TILE;
  constexpr size_t TILE_BYTES = static_cast<size_t>(TILE_ELEMS) * sizeof(float);
  constexpr int PIPELINE_STAGES = 2;

  cg::thread_block block = cg::this_thread_block();

  extern __shared__ float smem[];
  float* stage_a_base = smem;
  float* stage_b_base = stage_a_base + PIPELINE_STAGES * TILE_ELEMS;
  float* stage_c_base = stage_b_base + PIPELINE_STAGES * TILE_ELEMS;

  using pipeline_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, PIPELINE_STAGES>;
  __shared__ alignas(pipeline_state_t) unsigned char state_bytes[sizeof(pipeline_state_t)];
  auto* state = reinterpret_cast<pipeline_state_t*>(state_bytes);
  if (threadIdx.x == 0) {
    new (state) pipeline_state_t();
  }
  __syncthreads();
  auto pipe = cuda::make_pipeline(block, state);

  int warp_id = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;

  int stride = gridDim.x;
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    int tile_index = blockIdx.x + stage * stride;
    if (tile_index >= total_tiles) {
      break;
    }
    float* stage_a = stage_a_base + stage * TILE_ELEMS;
    float* stage_b = stage_b_base + stage * TILE_ELEMS;
    size_t offset = static_cast<size_t>(tile_index) * TILE_ELEMS;

    pipe.producer_acquire();
    cuda::memcpy_async(block,
                       stage_a,
                       A + offset,
                       cuda::aligned_size_t<16>{TILE_BYTES},
                       pipe);
    cuda::memcpy_async(block,
                       stage_b,
                       B + offset,
                       cuda::aligned_size_t<16>{TILE_BYTES},
                       pipe);
    pipe.producer_commit();
  }

  block.sync();

  int tile_iter = 0;
  for (int tile = blockIdx.x; tile < total_tiles; tile += stride, ++tile_iter) {
    int stage = tile_iter % PIPELINE_STAGES;
    float* stage_a = stage_a_base + stage * TILE_ELEMS;
    float* stage_b = stage_b_base + stage * TILE_ELEMS;
    float* stage_c = stage_c_base + stage * TILE_ELEMS;
    size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

    pipe.consumer_wait();

    block.sync();

    if (warp_id == 1) {
      compute_tile<TILE>(stage_a, stage_b, stage_c, lane);
    }

    block.sync();

    if (warp_id == 2) {
      for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        C[offset + idx] = stage_c[idx];
      }
    }

    block.sync();

    pipe.consumer_release();

    block.sync();

    int next_tile = tile + PIPELINE_STAGES * stride;
    if (next_tile < total_tiles) {
      int next_stage = (tile_iter + PIPELINE_STAGES) % PIPELINE_STAGES;
      float* next_a = stage_a_base + next_stage * TILE_ELEMS;
      float* next_b = stage_b_base + next_stage * TILE_ELEMS;
      size_t next_offset = static_cast<size_t>(next_tile) * TILE_ELEMS;

      pipe.producer_acquire();
      cuda::memcpy_async(block,
                         next_a,
                         A + next_offset,
                         cuda::aligned_size_t<16>{TILE_BYTES},
                         pipe);
      cuda::memcpy_async(block,
                         next_b,
                         B + next_offset,
                         cuda::aligned_size_t<16>{TILE_BYTES},
                         pipe);
      pipe.producer_commit();
    }

    block.sync();
  }
}

template <int TILE>
void run_warp_specialized(int tiles,
                          const std::vector<float>& h_A,
                          const std::vector<float>& h_B,
                          std::vector<float>& h_C) {
  constexpr int TILE_ELEMS = TILE * TILE;
  constexpr int PIPELINE_STAGES = 2;
  size_t elems = static_cast<size_t>(tiles) * TILE_ELEMS;
  size_t bytes = elems * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(96);
  dim3 grid(std::min(tiles, 256));
  size_t shared_bytes = 3 * PIPELINE_STAGES * TILE_ELEMS * sizeof(float);

  // Warmup
  warp_specialized_kernel<TILE><<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed run
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  constexpr int iterations = 10;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    warp_specialized_kernel<TILE><<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / iterations;
  std::printf("Kernel time: %.4f ms\n", avg_ms);
  
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
}

int main() {
  const auto& limits = cuda_arch::get_architecture_limits();
  if (!limits.supports_clusters) {
    std::printf("⚠️  Skipping warp-specialized pipeline: device lacks cluster/pipeline support.\n");
    return 0;
  }

  int tile = cuda_arch::select_square_tile_size<float>(
      /*shared_tiles=*/3, {32, 16, 8});

  int tiles = std::min(128, std::max(32, limits.max_cluster_size * 16));
  size_t elems = static_cast<size_t>(tiles) * tile * tile;

  std::vector<float> h_A(elems), h_B(elems), h_C(elems), h_ref(elems);
  std::iota(h_A.begin(), h_A.end(), 0.0f);
  std::iota(h_B.begin(), h_B.end(), 1.0f);

  switch (tile) {
    case 32:
      run_warp_specialized<32>(tiles, h_A, h_B, h_C);
      break;
    case 16:
      run_warp_specialized<16>(tiles, h_A, h_B, h_C);
      break;
    default:
      run_warp_specialized<8>(tiles, h_A, h_B, h_C);
      break;
  }

  for (size_t i = 0; i < elems; ++i) {
    h_ref[i] = std::sqrt(h_A[i] * h_A[i] + h_B[i] * h_B[i]);
  }

  double max_err = 0.0;
  for (size_t i = 0; i < elems; ++i) {
    max_err = std::max(max_err, static_cast<double>(std::abs(h_C[i] - h_ref[i])));
  }
  std::printf("Selected tile size: %d (shared-memory budget %.1f KB)\n",
              tile, limits.max_shared_mem_per_block / 1024.0);
  std::printf("Max error: %.6e\n", max_err);

  return 0;
}
