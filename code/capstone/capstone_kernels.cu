#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <array>

namespace {

__device__ inline half load_half(const half* ptr, int idx, int bound) {
  if (idx < bound) {
    return ptr[idx];
  }
  return __float2half(0.0f);
}

__global__ void baseline_matmul_kernel(const half* __restrict__ A,
                                       const half* __restrict__ B,
                                       half* __restrict__ C,
                                       int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float a = __half2float(A[row * K + k]);
    float b = __half2float(B[k * N + col]);
    acc += a * b;
  }
  C[row * N + col] = __float2half(acc);
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void optimized_matmul_kernel(const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        half* __restrict__ C,
                                        int M, int N, int K) {
  __shared__ half As[TILE_M][TILE_K];
  __shared__ half Bs[TILE_K][TILE_N];

  int block_row = blockIdx.y * TILE_M;
  int block_col = blockIdx.x * TILE_N;

  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;

  float accum = 0.0f;

  for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
    for (int i = thread_row; i < TILE_M; i += blockDim.y) {
      for (int j = thread_col; j < TILE_K; j += blockDim.x) {
        int global_row = block_row + i;
        int global_col = tile_k + j;
        half value = (global_row < M && global_col < K)
                         ? A[global_row * K + global_col]
                         : __float2half(0.0f);
        As[i][j] = value;
      }
    }

    for (int i = thread_row; i < TILE_K; i += blockDim.y) {
      for (int j = thread_col; j < TILE_N; j += blockDim.x) {
        int global_row = tile_k + i;
        int global_col = block_col + j;
        half value = (global_row < K && global_col < N)
                         ? B[global_row * N + global_col]
                         : __float2half(0.0f);
        Bs[i][j] = value;
      }
    }
    __syncthreads();

    int row = block_row + thread_row;
    int col = block_col + thread_col;
    if (row < M && col < N) {
      float acc = 0.0f;
      for (int k_it = 0; k_it < TILE_K; ++k_it) {
        acc += __half2float(As[thread_row][k_it]) *
               __half2float(Bs[k_it][thread_col]);
      }
      accum += acc;
    }
    __syncthreads();
  }

  int row = block_row + thread_row;
  int col = block_col + thread_col;
  if (row < M && col < N) {
    C[row * N + col] = __float2half(accum);
  }
}

void launch_baseline(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  dim3 block(16, 16);
  dim3 grid((b.size(1) + block.x - 1) / block.x,
            (a.size(0) + block.y - 1) / block.y);

  baseline_matmul_kernel<<<grid, block, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0),
      b.size(1), a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

void launch_optimized(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  constexpr int TILE_M = 64;
  constexpr int TILE_N = 64;
  constexpr int TILE_K = 32;

  dim3 block(16, 16);
  dim3 grid((b.size(1) + TILE_N - 1) / TILE_N,
            (a.size(0) + TILE_M - 1) / TILE_M);

  optimized_matmul_kernel<TILE_M, TILE_N, TILE_K>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
          reinterpret_cast<half*>(c.data_ptr<at::Half>()), a.size(0),
          b.size(1), a.size(1));
  AT_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor baseline_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "incompatible shapes");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must live on CUDA");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "use float16 tensors");

  auto c = torch::empty({a.size(0), b.size(1)}, a.options());
  launch_baseline(a.contiguous(), b.contiguous(), c);
  return c;
}

torch::Tensor optimized_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "expected 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "incompatible shapes");
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must live on CUDA");
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16,
              "use float16 tensors");

  auto c = torch::empty({a.size(0), b.size(1)}, a.options());
  launch_optimized(a.contiguous(), b.contiguous(), c);
  return c;
}

}  // namespace

TORCH_LIBRARY(blackwell_capstone, m) {
  m.def("baseline_matmul(Tensor a, Tensor b) -> Tensor");
  m.def("optimized_matmul(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(blackwell_capstone, CUDA, m) {
  m.impl("baseline_matmul", baseline_matmul);
  m.impl("optimized_matmul", optimized_matmul);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("baseline_matmul", &baseline_matmul);
  m.def("optimized_matmul", &optimized_matmul);
}
