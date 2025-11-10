#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

constexpr int kNaiveBlockDim = 16;
constexpr int kTiledBlockDim = 32;

__global__ void matmul_naive_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M,
    int N,
    int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = acc;
}

template <int TILE>
__global__ void matmul_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M,
    int N,
    int K) {
    __shared__ float shared_a[TILE][TILE];
    __shared__ float shared_b[TILE][TILE];

    const int global_row = blockIdx.y * TILE + threadIdx.y;
    const int global_col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    const int tiles = (K + TILE - 1) / TILE;

    for (int tile_idx = 0; tile_idx < tiles; ++tile_idx) {
        const int tiled_col = tile_idx * TILE + threadIdx.x;
        const int tiled_row = tile_idx * TILE + threadIdx.y;

        if (global_row < M && tiled_col < K) {
            shared_a[threadIdx.y][threadIdx.x] =
                a[global_row * K + tiled_col];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiled_row < K && global_col < N) {
            shared_b[threadIdx.y][threadIdx.x] =
                b[tiled_row * N + global_col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        c[global_row * N + global_col] = acc;
    }
}

void validate_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& out) {
    TORCH_CHECK(a.is_cuda(), "Matrix A must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "Matrix B must be on CUDA");
    TORCH_CHECK(out.is_cuda(), "Output matrix must be on CUDA");

    TORCH_CHECK(
        a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32 &&
            out.dtype() == torch::kFloat32,
        "Matrices must be float32");

    TORCH_CHECK(
        a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
        "All matrices must be 2D");

    TORCH_CHECK(
        a.size(1) == b.size(0),
        "Inner dimensions must match for matmul");
    TORCH_CHECK(
        a.size(0) == out.size(0) && b.size(1) == out.size(1),
        "Output shape mismatch");

    TORCH_CHECK(
        a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
        "All matrices must be contiguous");
}

} // namespace

void matmul_naive(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    validate_inputs(a, b, out);
    at::cuda::CUDAGuard device_guard(a.device());

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const dim3 block(kNaiveBlockDim, kNaiveBlockDim);
    const dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    const auto stream = at::cuda::getCurrentCUDAStream();
    matmul_naive_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        N,
        K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void matmul_tiled(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    validate_inputs(a, b, out);
    at::cuda::CUDAGuard device_guard(a.device());

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const dim3 block(kTiledBlockDim, kTiledBlockDim);
    const dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    const auto stream = at::cuda::getCurrentCUDAStream();
    matmul_tiled_kernel<kTiledBlockDim><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        N,
        K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_naive",
        &matmul_naive,
        "Naive matmul without tiling (global memory accesses only)");
    m.def(
        "matmul_tiled",
        &matmul_tiled,
        "Tiled matmul using shared memory tiles for reuse");
}
