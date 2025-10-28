// warp_specialized_pipeline.cu -- warp roles with correct pipeline handoff.

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

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

constexpr int TILE = 128;
constexpr int TILE_ELEMS = TILE * TILE;

__device__ void compute_tile(const float* a, const float* b, float* c, int lane) {
  for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
    float x = a[idx];
    float y = b[idx];
    c[idx] = sqrtf(x * x + y * y);
  }
}

__global__ void warp_specialized_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int total_tiles) {
  cg::thread_block block = cg::this_thread_block();
  extern __shared__ float smem[];
  float* A_tile = smem;
  float* B_tile = smem + TILE_ELEMS;
  float* C_tile = smem + 2 * TILE_ELEMS;

  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> state;
  auto pipe = cuda::make_pipeline(block, &state);

  int warp_id = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int warps_per_block = blockDim.x / warpSize;
  int global_warp = warp_id + warps_per_block * blockIdx.x;
  int total_warps = gridDim.x * warps_per_block;

  for (int tile = global_warp; tile < total_tiles; tile += total_warps) {
    size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

    if (warp_id == 0) {
      pipe.producer_acquire();
      for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        A_tile[idx] = A[offset + idx];
        B_tile[idx] = B[offset + idx];
      }
      pipe.producer_commit();
    }

    if (warp_id == 1) {
      pipe.consumer_wait();
      compute_tile(A_tile, B_tile, C_tile, lane);
      pipe.consumer_release();
    }

    if (warp_id == 2) {
      pipe.consumer_wait();
      for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        C[offset + idx] = C_tile[idx];
      }
      pipe.consumer_release();
    }

    block.sync();
  }
}

int main() {
  int tiles = 256;
  size_t elems = static_cast<size_t>(tiles) * TILE_ELEMS;
  size_t bytes = elems * sizeof(float);

  std::vector<float> h_A(elems), h_B(elems), h_C(elems), h_ref(elems);
  std::iota(h_A.begin(), h_A.end(), 0.0f);
  std::iota(h_B.begin(), h_B.end(), 1.0f);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(96);
  dim3 grid(std::min(tiles, 256));
  size_t shared_bytes = 3 * TILE_ELEMS * sizeof(float);

  warp_specialized_kernel<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < elems; ++i) {
    h_ref[i] = std::sqrt(h_A[i] * h_A[i] + h_B[i] * h_B[i]);
  }

  double max_err = 0.0;
  for (size_t i = 0; i < elems; ++i) {
    max_err = std::max(max_err, (double)std::abs(h_C[i] - h_ref[i]));
  }
  std::printf("Max error: %.6e\n", max_err);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
